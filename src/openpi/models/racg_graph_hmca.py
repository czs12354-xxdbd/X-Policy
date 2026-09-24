# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Role/edge graph bridge into hierarchical memory-conditioned adapters."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp


RACG_TOKEN_COUNT = 10  # six ordered roles followed by four typed edges
RACG_TOKEN_DIM = 256
HMCA_SEMANTIC_DIM = 1280


@jax.custom_jvp
def exact_zero_graph_add(parent: jax.Array, residual: jax.Array) -> jax.Array:
    """Keep the GHMA parent bit-exact while preserving bridge gradients."""

    return jnp.where(residual == 0, parent, parent + residual)


@exact_zero_graph_add.defjvp
def _exact_zero_graph_add_jvp(primals, tangents):
    parent, residual = primals
    parent_tangent, residual_tangent = tangents
    return exact_zero_graph_add(parent, residual), parent_tangent + residual_tangent


class RACGGraphHMCABridge(nnx.Module):
    """Preserve all role/edge identities while conditioning every HMCA layer.

    The graph already contains HETM frontier state, proprioception, previous
    actions, geometry, relation type, hazard evidence, and learned contact
    phase.  Flattening its ten ordered tokens retains those typed channels;
    the byte-zero output projection makes this a strict GHMA successor.
    """

    def __init__(self, *, hidden_dim: int = 128, rngs: nnx.Rngs):
        if hidden_dim <= 0:
            raise ValueError("graph-HMCA hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.graph_down = nnx.Linear(
            RACG_TOKEN_COUNT * RACG_TOKEN_DIM,
            hidden_dim,
            rngs=rngs,
        )
        self.graph_up = nnx.Linear(
            hidden_dim,
            HMCA_SEMANTIC_DIM,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        graph_tokens: jax.Array,
        graph_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if graph_tokens.ndim != 3 or graph_tokens.shape[1:] != (
            RACG_TOKEN_COUNT,
            RACG_TOKEN_DIM,
        ):
            raise ValueError("RACG graph tokens must be [batch,10,256]")
        if graph_mask.shape != graph_tokens.shape[:2]:
            raise ValueError("RACG graph mask must be [batch,10]")
        mask = graph_mask.astype(jnp.bool_)
        enabled = jnp.any(mask, axis=-1)
        safe = jnp.where(mask[..., None], graph_tokens, 0.0)
        variance = jnp.mean(
            jnp.square(safe.astype(jnp.float32)), axis=-1, keepdims=True
        )
        normalized = safe * jax.lax.rsqrt(
            variance + jnp.asarray(1.0e-6, jnp.float32)
        )
        latent = nnx.swish(self.graph_down(normalized.reshape(normalized.shape[0], -1)))
        residual = self.graph_up(latent)
        return jnp.where(enabled[:, None], residual, 0.0), enabled


def parameter_count(hidden_dim: int = 128) -> int:
    if hidden_dim <= 0:
        raise ValueError("graph-HMCA hidden_dim must be positive")
    return (
        RACG_TOKEN_COUNT * RACG_TOKEN_DIM * hidden_dim
        + hidden_dim
        + hidden_dim * HMCA_SEMANTIC_DIM
        + HMCA_SEMANTIC_DIM
    )
