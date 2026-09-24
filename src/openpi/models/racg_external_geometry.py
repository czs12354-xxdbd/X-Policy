# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Zero-gated Molmo2-ER geometry prior for a future RACG successor.

The module consumes contextualized visual-prefix states, so its five external
queries remain instruction conditioned without reading evaluator metadata.
The two large leaves are initialized from the already trained Molmo2-ER
artifact; five fresh scalar gates are exact positive zero.  Consequently its
six-role residual is exactly zero at transplant while the first optimization
step can open the gates.

External roles are routed without a learned selector:

* route source -> agent
* target -> target
* destination -> destination
* safety -> hazard
* path -> free-space

RACG's reference node is intentionally untouched because the external
geometry corpus has no independently supervised reference-object role.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import racg


EXTERNAL_ROLE_NAMES = ("target", "source", "destination", "path", "safety")
EXTERNAL_ROLE_COUNT = len(EXTERNAL_ROLE_NAMES)
EXTERNAL_TO_RACG_ROLE = (
    racg.TARGET_ROLE,
    racg.AGENT_ROLE,
    racg.DESTINATION_ROLE,
    racg.FREE_SPACE_ROLE,
    racg.HAZARD_ROLE,
)


@dataclasses.dataclass(frozen=True)
class ExternalGeometryConfig:
    prefix_dim: int = 2048
    hidden_dim: int = 256
    camera_views: int = 2

    def validate(self) -> None:
        if self.prefix_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("external geometry dimensions must be positive")
        if self.camera_views != 2:
            raise ValueError("external geometry prior requires exactly two views")


class ExternalGeometryOutput(NamedTuple):
    role_residual: jax.Array
    external_role_features: jax.Array
    external_role_mask: jax.Array
    attention: jax.Array
    gates: jax.Array


def _rms_normalize(value: jax.Array) -> jax.Array:
    variance = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
    return value * jax.lax.rsqrt(variance + jnp.asarray(1.0e-6, jnp.float32))


def _masked_softmax(logits: jax.Array, mask: jax.Array) -> jax.Array:
    mask = jnp.broadcast_to(mask.astype(jnp.bool_), logits.shape)
    weights = jax.nn.softmax(jnp.where(mask, logits, -1.0e30), axis=-1)
    weights = weights * mask.astype(weights.dtype)
    return weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8)


def apply_external_role_residual(
    inherited_role_nodes: jax.Array, role_residual: jax.Array
) -> jax.Array:
    """Preserve both values and dtype when the residual is exactly zero."""

    if inherited_role_nodes.ndim != 3:
        raise ValueError("inherited role nodes must be [batch, role, hidden]")
    if role_residual.shape != inherited_role_nodes.shape:
        raise ValueError("external geometry residual must match inherited role nodes")
    return inherited_role_nodes + role_residual.astype(inherited_role_nodes.dtype)


class ExternalGeometryRolePrior(nnx.Module):
    """Five pretrained spatial queries with an exact-zero six-role boundary."""

    def __init__(
        self,
        config: ExternalGeometryConfig = ExternalGeometryConfig(),
        *,
        rngs: nnx.Rngs,
    ):
        config.validate()
        self.config = config
        self.external_role_query = nnx.Param(
            nnx.initializers.normal(config.prefix_dim**-0.5)(
                rngs.params(),
                (EXTERNAL_ROLE_COUNT, config.prefix_dim),
                jnp.float32,
            )
        )
        self.external_prefix_out = nnx.Param(
            nnx.initializers.normal(config.prefix_dim**-0.5)(
                rngs.params(),
                (config.prefix_dim, config.hidden_dim),
                jnp.float32,
            )
        )
        self.external_blend_gate = nnx.Param(
            jnp.zeros((EXTERNAL_ROLE_COUNT,), dtype=jnp.float32)
        )

    def __call__(
        self,
        visual_states: jax.Array,
        visual_mask: jax.Array,
        view_valid: jax.Array,
        racg_role_present: jax.Array,
    ) -> ExternalGeometryOutput:
        c = self.config
        if visual_states.ndim != 4 or visual_states.shape[-1] != c.prefix_dim:
            raise ValueError("visual states must be [batch, view, patch, prefix_dim]")
        batch, views, patches = visual_states.shape[:3]
        if views != c.camera_views:
            raise ValueError("visual states must contain exactly two camera views")
        if visual_mask.shape != (batch, views, patches):
            raise ValueError("visual mask must be [batch, view, patch]")
        if view_valid.shape != (batch, views):
            raise ValueError("view validity must be [batch, view]")
        if racg_role_present.shape != (batch, racg.ROLE_COUNT):
            raise ValueError("RACG role presence must be [batch, 6]")

        valid = visual_mask.astype(jnp.bool_) & view_valid[:, :, None].astype(jnp.bool_)
        effective_view = view_valid.astype(jnp.bool_) & jnp.any(valid, axis=-1)
        safe_visual = jnp.where(valid[..., None], visual_states, 0.0)
        query = _rms_normalize(self.external_role_query.value).astype(jnp.float32)
        key = _rms_normalize(safe_visual).astype(jnp.float32)
        logits = jnp.einsum("rd,bvpd->bvrp", query, key) / jnp.sqrt(
            jnp.asarray(c.prefix_dim, jnp.float32)
        )
        attention = _masked_softmax(logits, valid[:, :, None])
        pooled = jnp.einsum("bvrp,bvpd->bvrd", attention, safe_visual)
        per_view_features = jnp.tanh(
            jnp.einsum(
                "bvrd,dh->bvrh",
                pooled.astype(jnp.float32),
                self.external_prefix_out.value,
            )
        )
        view_weight = effective_view.astype(jnp.float32)
        features = jnp.einsum("bv,bvrh->brh", view_weight, per_view_features)
        features = features / jnp.maximum(
            jnp.sum(view_weight, axis=-1, keepdims=True)[..., None], 1.0
        )

        mapped_presence = jnp.stack(
            [racg_role_present[:, role] for role in EXTERNAL_TO_RACG_ROLE], axis=1
        ).astype(jnp.bool_)
        any_view = jnp.any(effective_view, axis=-1, keepdims=True)
        external_mask = mapped_presence & any_view
        features = jnp.where(external_mask[..., None], features, 0.0)
        gated = features * jnp.tanh(self.external_blend_gate.value)[None, :, None]
        role_residual = jnp.zeros(
            (batch, racg.ROLE_COUNT, c.hidden_dim), dtype=jnp.float32
        )
        for external_index, racg_index in enumerate(EXTERNAL_TO_RACG_ROLE):
            role_residual = role_residual.at[:, racg_index].add(
                gated[:, external_index]
            )
        role_residual = jnp.where(
            racg_role_present[..., None].astype(jnp.bool_), role_residual, 0.0
        )
        return ExternalGeometryOutput(
            role_residual=role_residual,
            external_role_features=features,
            external_role_mask=external_mask,
            attention=attention,
            gates=jnp.tanh(self.external_blend_gate.value),
        )


def parameter_count(config: ExternalGeometryConfig = ExternalGeometryConfig()) -> int:
    config.validate()
    return (
        EXTERNAL_ROLE_COUNT * config.prefix_dim
        + config.prefix_dim * config.hidden_dim
        + EXTERNAL_ROLE_COUNT
    )
