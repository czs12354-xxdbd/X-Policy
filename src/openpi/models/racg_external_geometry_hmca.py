# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Role-preserving external-geometry bridge into HMCA layer adapters."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp


EXTERNAL_ROLE_COUNT = 5
EXTERNAL_ROLE_DIM = 256
HMCA_SEMANTIC_DIM = 1280


@jax.custom_jvp
def exact_zero_geometry_add(parent: jax.Array, residual: jax.Array) -> jax.Array:
    """Preserve the EGP parent primal exactly while retaining bridge gradients."""

    return jnp.where(residual == 0, parent, parent + residual)


@exact_zero_geometry_add.defjvp
def _exact_zero_geometry_add_jvp(primals, tangents):
    parent, residual = primals
    parent_tangent, residual_tangent = tangents
    return (
        exact_zero_geometry_add(parent, residual),
        parent_tangent + residual_tangent,
    )


class ExternalGeometryHMCABridge(nnx.Module):
    """Map all five ordered geometry roles to the HMCA semantic condition.

    Flattening rather than averaging preserves target/source/destination/path/
    safety identity.  The final projection is byte-zero, so adding this bridge
    to a trained EGP checkpoint preserves its policy exactly at transplant.
    """

    def __init__(self, *, hidden_dim: int = 64, rngs: nnx.Rngs):
        if hidden_dim <= 0:
            raise ValueError("geometry-HMCA hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.geometry_down = nnx.Linear(
            EXTERNAL_ROLE_COUNT * EXTERNAL_ROLE_DIM,
            hidden_dim,
            rngs=rngs,
        )
        self.geometry_up = nnx.Linear(
            hidden_dim,
            HMCA_SEMANTIC_DIM,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        role_features: jax.Array,
        role_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if role_features.ndim != 3 or role_features.shape[1:] != (
            EXTERNAL_ROLE_COUNT,
            EXTERNAL_ROLE_DIM,
        ):
            raise ValueError("external role features must be [batch,5,256]")
        if role_mask.shape != role_features.shape[:2]:
            raise ValueError("external role mask must be [batch,5]")
        enabled = jnp.any(role_mask.astype(jnp.bool_), axis=-1)
        safe = jnp.where(role_mask[..., None], role_features, 0.0)
        variance = jnp.mean(
            jnp.square(safe.astype(jnp.float32)), axis=-1, keepdims=True
        )
        normalized = safe * jax.lax.rsqrt(
            variance + jnp.asarray(1.0e-6, jnp.float32)
        )
        flattened = normalized.reshape(normalized.shape[0], -1)
        latent = nnx.swish(self.geometry_down(flattened))
        residual = self.geometry_up(latent)
        residual = jnp.where(enabled[:, None], residual, 0.0)
        return residual, enabled


def parameter_count(hidden_dim: int = 64) -> int:
    if hidden_dim <= 0:
        raise ValueError("geometry-HMCA hidden_dim must be positive")
    return (
        HMCA_SEMANTIC_DIM * hidden_dim
        + hidden_dim
        + hidden_dim * HMCA_SEMANTIC_DIM
        + HMCA_SEMANTIC_DIM
    )
