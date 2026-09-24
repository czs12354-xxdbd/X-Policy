# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Role-bound affordance causal graph (RACG-v1) core.

This is deliberately a pure JAX/Flax sidecar.  It does not know about Pi0,
checkpoint loading, evaluator metadata, or a benchmark task id.  The only
parent-facing operation is :func:`apply_graph_action_residual`; its projection
is exactly zero at initialization, preserving the inherited action tokens.

Role order is fixed and public: agent, target, reference, destination, hazard,
free-space.  The ten graph tokens are the six role nodes followed by four
directed edges: agent->target, target->reference/destination, target->hazard,
and current-target->previous-target.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

AGENT_ROLE = 0
TARGET_ROLE = 1
REFERENCE_ROLE = 2
DESTINATION_ROLE = 3
HAZARD_ROLE = 4
FREE_SPACE_ROLE = 5
ROLE_COUNT = 6
EDGE_COUNT = 4
CONTACT_STATES = 4
RELATION_CLASSES = 16
RELATION_MARGINAL_CLASSES = 4
TWO_REFERENCE_RELATION_FLAG = 1 << 4
# Ratios are locked to full-inventory counts [341446, 5926, 355858, 5152] from
# ``racg_v1_supervision_audit.json``.  Their common scale is normalized against
# the exact 30k sampler exposure [22841924, 708552, 33770370, 178034] audited in
# ``racg_production_contact_exposure_audit_v1.json``.  Thus the production
# expected target weight is exactly one while rare close/release transitions
# retain the stronger full-inventory inverse-sqrt emphasis.
CONTACT_CLASS_WEIGHTS = (
    0.91632947572262935,
    6.9555559424475941,
    0.8975823513334984,
    7.4597580936702501,
)
# Locked to the exact 30k x batch-32, 4/8-replan production schedule in
# ``racg_production_relation_exposure_audit_v1.json``.  The factorized head
# retains compositional logits for all 16 source/destination pairs; these
# unit-expectation inverse-sqrt target weights prevent the common ``none->on``
# and ``none->in`` pairs from washing out the rare observed directed pairs.
RELATION_CLASS_WEIGHTS = (
    3.1094110830591277,
    0.6529088636786178,
    0.9035682968246415,
    1.5602582899670587,
    0.42186487321791805,
    0.9896345235663968,
    0.42186487321791805,
    0.42186487321791805,
    0.42186487321791805,
    2.2126529020283754,
    0.42186487321791805,
    0.42186487321791805,
    0.42186487321791805,
    1.7996912356829398,
    0.42186487321791805,
    0.42186487321791805,
)
# Exact 30k sampler exposure is [5749888, 5749888, 2879592, 5644048,
# 523120, 5749888] for graph-valid roles and [0, 5749888, 2879592,
# 5644048, 523120, 0] for text spans.  Inverse-sqrt weights are normalized
# against the actual per-example masked mean, so adding role balance does not
# silently change the declared auxiliary-loss scale.  Agent/free-space never
# receive fabricated text spans; their span entries are inert placeholders.
ROLE_VALID_CLASS_WEIGHTS = (
    0.9212194260892393,
    0.9212194260892393,
    1.3017491446806455,
    0.9298168901331807,
    3.0541623421219564,
    0.9212194260892393,
)
ROLE_SPAN_CLASS_WEIGHTS = (
    0.8750187339254827,
    0.8750187339254827,
    1.2364642518477427,
    0.8831850207944383,
    2.9009910018414637,
    0.8750187339254827,
)


@dataclasses.dataclass(frozen=True)
class RACGConfig:
    """Static production topology; dimensions may be reduced in CPU tests."""

    patch_dim: int = 2048
    language_dim: int = 2048
    hetm_role_dim: int = 256
    state_dim: int = 32
    action_hidden_dim: int = 1024
    hidden_dim: int = 256
    object_slots: int = 8
    role_count: int = ROLE_COUNT
    predicate_slots: int = 8
    predicate_states: int = 4
    frontier_slots: int = 8
    previous_action_steps: int = 5
    active_action_dim: int = 7
    action_positions: int = 10
    camera_views: int = 2
    action_read_heads: int = 4
    slot_iterations: int = 2
    graph_iterations: int = 2

    def validate(self) -> None:
        positive = dataclasses.asdict(self)
        if any(not isinstance(value, int) or value <= 0 for value in positive.values()):
            raise ValueError("all RACG dimensions and iteration counts must be positive")
        if self.role_count != ROLE_COUNT:
            raise ValueError("RACG-v1 requires exactly six ordered roles")
        if self.object_slots != 8:
            raise ValueError("RACG-v1 requires K=8 competitive object slots")
        if self.action_positions != 10:
            raise ValueError("RACG-v1 requires ten action-position graph reads")
        if self.camera_views != 2:
            raise ValueError("RACG-v1 requires exactly two camera views")
        if self.action_read_heads != 4:
            raise ValueError("RACG-v1 requires exactly four action-read heads")
        if self.hidden_dim % self.action_read_heads:
            raise ValueError("RACG hidden width must be divisible by action-read heads")
        if self.slot_iterations != 2 or self.graph_iterations != 2:
            raise ValueError("RACG-v1 requires two shared slot and graph updates")


DEFAULT_CONFIG = RACGConfig()


class RACGOutput(NamedTuple):
    action_residual: jax.Array
    graph_read: jax.Array
    graph_tokens: jax.Array
    graph_token_mask: jax.Array
    object_slots: jax.Array
    object_geometry: jax.Array
    slot_attention: jax.Array
    projected_patch_targets: jax.Array
    reconstructed_projected_patches: jax.Array
    cross_view_role_embeddings: jax.Array
    relation_reference_nodes: jax.Array
    relation_reference_geometry: jax.Array
    relation_reference_mask: jax.Array
    relation_reference_bindings: jax.Array
    role_nodes: jax.Array
    role_geometry: jax.Array
    role_bindings: jax.Array
    unexcluded_role_bindings: jax.Array
    directed_edges: jax.Array
    edge_mask: jax.Array
    edge_contact_logits: jax.Array
    contact_logits: jax.Array
    relation_logits: jax.Array
    projected_language_tokens: jax.Array


class RACGScene(NamedTuple):
    """Action-independent scene encoding, computed once per observation."""

    graph_tokens: jax.Array
    graph_token_mask: jax.Array
    object_slots: jax.Array
    object_geometry: jax.Array
    slot_attention: jax.Array
    projected_patch_targets: jax.Array
    reconstructed_projected_patches: jax.Array
    cross_view_role_embeddings: jax.Array
    relation_reference_nodes: jax.Array
    relation_reference_geometry: jax.Array
    relation_reference_mask: jax.Array
    relation_reference_bindings: jax.Array
    role_nodes: jax.Array
    role_geometry: jax.Array
    role_bindings: jax.Array
    unexcluded_role_bindings: jax.Array
    directed_edges: jax.Array
    edge_mask: jax.Array
    edge_contact_logits: jax.Array
    relation_logits: jax.Array
    projected_language_tokens: jax.Array
    depth_lift_output: object | None
    depth_motion_forecast_output: object | None


class _Linear(nnx.Module):
    """Linear with nonzero bias unless this is the unique zero boundary."""

    def __init__(self, in_features: int, out_features: int, *, zero: bool = False, rngs: nnx.Rngs):
        kernel_init = (
            nnx.initializers.zeros_init()
            if zero
            else nnx.initializers.normal(in_features**-0.5)
        )
        bias_init = (
            nnx.initializers.zeros_init()
            if zero
            else nnx.initializers.normal(0.01)
        )
        self.kernel = nnx.Param(
            kernel_init(rngs.params(), (in_features, out_features), jnp.float32)
        )
        self.bias = nnx.Param(
            bias_init(rngs.params(), (out_features,), jnp.float32)
        )

    def __call__(self, value: jax.Array) -> jax.Array:
        return jnp.einsum("...d,dh->...h", value, self.kernel.value) + self.bias.value


class _GatedUpdateLinear(nnx.Module):
    """Joint candidate/gate projection with a conservative initial gate."""

    def __init__(self, in_features: int, width: int, *, rngs: nnx.Rngs):
        candidate_kernel = nnx.initializers.normal(in_features**-0.5)(
            rngs.params(), (in_features, width), jnp.float32
        )
        candidate_bias = nnx.initializers.normal(0.01)(
            rngs.params(), (width,), jnp.float32
        )
        self.kernel = nnx.Param(
            jnp.concatenate(
                [candidate_kernel, jnp.zeros((in_features, width), jnp.float32)],
                axis=-1,
            )
        )
        self.bias = nnx.Param(
            jnp.concatenate(
                [candidate_bias, jnp.full((width,), -2.0, jnp.float32)], axis=-1
            )
        )

    def __call__(self, value: jax.Array) -> jax.Array:
        return jnp.einsum("...d,dh->...h", value, self.kernel.value) + self.bias.value


class _Slots(nnx.Module):
    def __init__(self, count: int, width: int, *, rngs: nnx.Rngs):
        self.value = nnx.Param(
            nnx.initializers.normal(width**-0.5)(
                rngs.params(), (count, width), jnp.float32
            )
        )


def _masked_softmax(logits: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    mask = mask.astype(jnp.bool_)
    safe_logits = jnp.where(mask, logits, -1.0e30)
    weights = jax.nn.softmax(safe_logits, axis=axis) * mask.astype(logits.dtype)
    return weights / jnp.maximum(jnp.sum(weights, axis=axis, keepdims=True), 1.0e-6)


def reference_is_between(relation_kind: jax.Array) -> jax.Array:
    """Decode explicit two-reference cardinality with legacy compatibility."""

    if relation_kind.ndim != 1:
        raise ValueError("RACG relation kind must be [batch]")
    if jnp.issubdtype(relation_kind.dtype, jnp.bool_):
        return relation_kind
    return (
        ((relation_kind & TWO_REFERENCE_RELATION_FLAG) != 0)
        # Direct-between prompts used low bits ``destination=3`` before the
        # explicit cardinality flag existed.  Retain that safe interpretation
        # for old synthetic/inference callers.
        | (relation_kind % RELATION_MARGINAL_CLASSES == 3)
    )


def destination_relation_is_between(relation_kind: jax.Array) -> jax.Array:
    """Decode a direct between-region destination from prompt relation bits."""

    if relation_kind.ndim != 1:
        raise ValueError("RACG relation kind must be [batch]")
    if jnp.issubdtype(relation_kind.dtype, jnp.bool_):
        # Backward compatibility for old synthetic callers that supplied only
        # the direct-between flag instead of the inference relation bitfield.
        return relation_kind
    return relation_kind % RELATION_MARGINAL_CLASSES == 3


def structural_role_type_basis(hidden_dim: int, dtype=jnp.float32) -> jax.Array:
    """Return six distinct unit-norm, parameter-free graph node type tags."""

    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("role type basis width must be positive")
    roles = jnp.arange(1, ROLE_COUNT + 1, dtype=jnp.float32)[:, None]
    channels = jnp.arange(1, hidden_dim + 1, dtype=jnp.float32)[None, :]
    phase = jnp.pi * roles * channels / float(hidden_dim + 1)
    basis = jnp.sin(phase) + jnp.cos(phase * (1.0 + roles / ROLE_COUNT))
    basis = basis / jnp.maximum(
        jnp.linalg.norm(basis, axis=-1, keepdims=True), 1.0e-6
    )
    return basis.astype(dtype)


def role_conditioned_language_context(
    language_states: jax.Array,
    language_mask: jax.Array,
    role_span_mask: jax.Array,
) -> jax.Array:
    """Pool exact contextual prompt spans, with a whole-prompt fallback.

    The span masks are deterministic prompt features available during both
    training and inference.  Structural roles such as agent/free-space do not
    have a literal span, so they retain the previous whole-prompt context.
    """

    if language_states.ndim != 3:
        raise ValueError("language states must be [batch, token, width]")
    if language_mask.shape != language_states.shape[:2]:
        raise ValueError("language mask must be [batch, token]")
    expected = (language_states.shape[0], ROLE_COUNT, language_states.shape[1])
    if role_span_mask.shape != expected:
        raise ValueError(f"role span mask must have shape {expected}")
    language_valid = language_mask.astype(jnp.bool_)
    safe_language = jnp.where(language_valid[..., None], language_states, 0.0)
    whole_prompt = jnp.sum(safe_language, axis=1) / jnp.maximum(
        jnp.sum(language_valid, axis=1, keepdims=True), 1
    )
    span_valid = role_span_mask.astype(jnp.bool_) & language_valid[:, None]
    span_context = jnp.einsum(
        "brt,bth->brh", span_valid.astype(language_states.dtype), safe_language
    ) / jnp.maximum(jnp.sum(span_valid, axis=-1, keepdims=True), 1)
    return jnp.where(
        jnp.any(span_valid, axis=-1, keepdims=True),
        span_context,
        whole_prompt[:, None],
    )


def harmonic_action_coordinates(
    flow_time: jax.Array,
    action_positions: int,
    dtype=jnp.float32,
) -> jax.Array:
    """Encode denoising time, action position, and their interaction."""

    if flow_time.ndim not in (1, 2) or (
        flow_time.ndim == 2 and flow_time.shape[1] != 1
    ):
        raise ValueError("flow time must be [batch] or [batch, 1]")
    if not isinstance(action_positions, int) or action_positions <= 0:
        raise ValueError("action positions must be positive")
    batch = flow_time.shape[0]
    time = jnp.broadcast_to(
        flow_time.reshape(batch, 1, 1).astype(jnp.float32),
        (batch, action_positions, 1),
    )
    position = jnp.linspace(-1.0, 1.0, action_positions, dtype=jnp.float32)
    position = jnp.broadcast_to(
        position[None, :, None], (batch, action_positions, 1)
    )
    return jnp.concatenate(
        [
            time,
            jnp.sin(jnp.pi * time),
            jnp.cos(jnp.pi * time),
            position,
            jnp.sin(jnp.pi * position),
            jnp.cos(jnp.pi * position),
            time * position,
        ],
        axis=-1,
    ).astype(dtype)


def typed_incident_messages(edges: jax.Array) -> jax.Array:
    """Keep all four directed edge types in separate node-update channels."""

    if edges.ndim != 3 or edges.shape[1] != EDGE_COUNT:
        raise ValueError("edges must be [batch,4,width]")
    incidence = jnp.zeros((EDGE_COUNT, ROLE_COUNT), jnp.float32)
    incidence = incidence.at[0, AGENT_ROLE].set(-1.0).at[0, TARGET_ROLE].set(1.0)
    incidence = incidence.at[1, TARGET_ROLE].set(-1.0)
    incidence = incidence.at[1, REFERENCE_ROLE].set(0.5).at[1, DESTINATION_ROLE].set(0.5)
    incidence = incidence.at[2, TARGET_ROLE].set(-1.0).at[2, HAZARD_ROLE].set(1.0)
    incidence = incidence.at[3, TARGET_ROLE].set(1.0)
    typed = jnp.einsum("beh,er->breh", edges, incidence)
    return typed.reshape(edges.shape[0], ROLE_COUNT, EDGE_COUNT * edges.shape[-1])


def structural_action_read_bias(dtype=jnp.float32) -> jax.Array:
    """Softly specialize four action-read heads without hard token routing."""

    bias = jnp.zeros((EDGE_COUNT, ROLE_COUNT + EDGE_COUNT), dtype=dtype)
    strength = jnp.log(jnp.asarray(2.0, dtype=dtype))
    routes = (
        (AGENT_ROLE, TARGET_ROLE, ROLE_COUNT + 0),
        (TARGET_ROLE, REFERENCE_ROLE, DESTINATION_ROLE, ROLE_COUNT + 1),
        (TARGET_ROLE, HAZARD_ROLE, ROLE_COUNT + 2),
        (TARGET_ROLE, ROLE_COUNT + 3),
    )
    for head, tokens in enumerate(routes):
        bias = bias.at[head, jnp.asarray(tokens)].set(strength)
    return bias


def multihead_graph_read(
    action_queries: jax.Array,
    graph_keys: jax.Array,
    graph_values: jax.Array,
    graph_mask: jax.Array,
    heads: int,
    attention_bias: jax.Array | None = None,
) -> jax.Array:
    """Read several graph relations per action instead of one convex mixture."""

    if action_queries.ndim != 3:
        raise ValueError("action queries must be [batch,position,width]")
    if graph_keys.ndim != 3 or graph_values.shape != graph_keys.shape:
        raise ValueError("graph keys and values must share [batch,token,width]")
    if graph_mask.shape != graph_keys.shape[:2]:
        raise ValueError("graph mask must be [batch,token]")
    if action_queries.shape[0] != graph_keys.shape[0]:
        raise ValueError("action and graph batch dimensions must match")
    width = action_queries.shape[-1]
    if graph_keys.shape[-1] != width:
        raise ValueError("action and graph widths must match")
    if not isinstance(heads, int) or heads <= 0 or width % heads:
        raise ValueError("heads must be positive and divide hidden width")
    head_dim = width // heads
    queries = action_queries.reshape(*action_queries.shape[:-1], heads, head_dim)
    keys = graph_keys.reshape(*graph_keys.shape[:-1], heads, head_dim)
    values = graph_values.reshape(*graph_values.shape[:-1], heads, head_dim)
    logits = jnp.einsum("bphd,bghd->bphg", queries, keys) / jnp.sqrt(
        jnp.asarray(head_dim, jnp.float32)
    )
    if attention_bias is not None:
        expected_bias = (heads, graph_keys.shape[1])
        if attention_bias.shape != expected_bias:
            raise ValueError(f"attention bias must have shape {expected_bias}")
        logits = logits + attention_bias[None, None].astype(logits.dtype)
    mask = jnp.broadcast_to(
        graph_mask[:, None, None].astype(jnp.bool_), logits.shape
    )
    attention = _masked_softmax(logits, mask, axis=-1)
    read = jnp.einsum("bphg,bghd->bphd", attention, values)
    return read.reshape(action_queries.shape)


def gated_graph_residual(update_logits: jax.Array) -> jax.Array:
    """Turn joint candidate/gate logits into a bounded node residual."""

    if update_logits.ndim < 1 or update_logits.shape[-1] % 2:
        raise ValueError("gated graph logits must have an even final width")
    candidate, gate = jnp.split(update_logits, 2, axis=-1)
    return jnp.tanh(candidate) * jax.nn.sigmoid(gate)


def apply_graph_action_residual(
    inherited_action_tokens: jax.Array, action_residual: jax.Array
) -> jax.Array:
    """Add the graph residual without changing dtype or the exact zero case."""

    if inherited_action_tokens.ndim != 3:
        raise ValueError("inherited action tokens must be [batch, position, width]")
    if action_residual.shape != inherited_action_tokens.shape:
        raise ValueError("RACG action residual shape must match inherited tokens")
    return inherited_action_tokens + action_residual.astype(inherited_action_tokens.dtype)


class RoleAffordanceCausalGraph(nnx.Module):
    """Shared two-view slots, six-role binding, directed graph, and action read."""

    def __init__(
        self,
        config: RACGConfig = DEFAULT_CONFIG,
        *,
        depth_lift: nnx.Module | None = None,
        depth_forecaster: nnx.Module | None = None,
        rngs: nnx.Rngs,
    ):
        config.validate()
        self.config = config
        h = config.hidden_dim
        self.slot_seed = _Slots(config.object_slots, h, rngs=rngs)
        # Camera identity is an explicit two-way one-hot; xy coordinates and
        # radial xy features account for the other five non-patch inputs.
        self.patch_in = _Linear(
            config.patch_dim + 5 + config.camera_views, h, rngs=rngs
        )
        self.slot_query = _Linear(h, h, rngs=rngs)
        self.patch_key_value = _Linear(h, 2 * h, rngs=rngs)
        self.slot_update = _Linear(2 * h, h, rngs=rngs)
        self.role_query = _Linear(
            config.hetm_role_dim
            + config.state_dim
            + config.frontier_slots
            + config.language_dim,
            h,
            rngs=rngs,
        )
        self.role_key_value = _Linear(h, 2 * h, rngs=rngs)
        self.language_align = _Linear(config.language_dim, h, rngs=rngs)
        self.cross_view_out = _Linear(h, h, rngs=rngs)
        edge_width = (
            # Ordered relation endpoints are encoded separately: the primary
            # endpoint carries reference and the secondary endpoint carries
            # destination.  The previous mean was exactly invariant to
            # swapping these causal roles.
            5 * h
            + 2 * 5
            + 2
            # Two destination-reference endpoints remain separate from the
            # source reference and destination entity.  This is essential for
            # nested commands such as "target next to cereal ... bowl between
            # cabinet and board".
            + 2 * h
            + 2 * 5
            + 2
            + config.predicate_slots * config.predicate_states
            + config.frontier_slots
            + config.state_dim
            + config.previous_action_steps * config.active_action_dim
            + EDGE_COUNT
        )
        self.edge_in = _Linear(edge_width, h, rngs=rngs)
        self.graph_update = _GatedUpdateLinear((1 + EDGE_COUNT) * h, h, rngs=rngs)
        self.contact_head = _Linear(h, CONTACT_STATES, rngs=rngs)
        self.relation_head = _Linear(
            h, 2 * RELATION_MARGINAL_CLASSES, rngs=rngs
        )
        self.action_query = _Linear(config.action_hidden_dim + 7, h, rngs=rngs)
        self.graph_key_value = _Linear(h, 2 * h, rngs=rngs)
        # This is intentionally the only zero-initialized RACG component.
        self.graph_action_out = _Linear(h, config.action_hidden_dim, zero=True, rngs=rngs)
        if depth_lift is not None:
            self.depth_lift = depth_lift
        if depth_forecaster is not None:
            if depth_lift is None:
                raise ValueError("depth forecaster requires depth lift")
            self.depth_forecaster = depth_forecaster

    def _validate_inputs(
        self,
        patches,
        patch_mask,
        patch_xy,
        view_valid,
        language_states,
        language_mask,
        hetm_role_states,
        role_present,
        predicate_probabilities,
        frontier,
        proprioception,
        previous_actions,
        previous_target_anchor,
        previous_target_geometry,
        previous_target_valid,
        episode_start,
        relation_kind,
        role_span_mask,
        source_reference_span_mask,
        destination_reference_span_mask,
        noisy_action_tokens=None,
        flow_time=None,
    ) -> int:
        c = self.config
        if patches.ndim != 4 or patches.shape[-1] != c.patch_dim:
            raise ValueError("patches must be [batch, view, patch, patch_dim]")
        batch, views, patch_count = patches.shape[:3]
        if views != c.camera_views:
            raise ValueError(f"camera view count must be exactly {c.camera_views}")
        checks = (
            (patch_mask, (batch, views, patch_count), "patch mask"),
            (patch_xy, (batch, views, patch_count, 2), "patch xy"),
            (view_valid, (batch, views), "view validity"),
            (language_mask, language_states.shape[:2], "language mask"),
            (hetm_role_states, (batch, ROLE_COUNT, c.hetm_role_dim), "HETM roles"),
            (role_present, (batch, ROLE_COUNT), "role presence"),
            (predicate_probabilities, (batch, c.predicate_slots, c.predicate_states), "predicates"),
            (frontier, (batch, c.frontier_slots), "frontier"),
            (proprioception, (batch, c.state_dim), "proprioception"),
            (
                previous_actions,
                (batch, c.previous_action_steps, c.active_action_dim),
                "previous actions",
            ),
            (previous_target_anchor, (batch, c.hidden_dim), "previous target anchor"),
            (previous_target_geometry, (batch, 5), "previous target geometry"),
            (previous_target_valid, (batch,), "previous target validity"),
            (episode_start, (batch,), "episode start"),
            (relation_kind, (batch,), "relation kind"),
            (
                role_span_mask,
                (batch, ROLE_COUNT, language_states.shape[1]),
                "role span mask",
            ),
            (
                source_reference_span_mask,
                (batch, language_states.shape[1]),
                "source reference span mask",
            ),
            (
                destination_reference_span_mask,
                (batch, 2, language_states.shape[1]),
                "destination reference span mask",
            ),
        )
        if language_states.ndim != 3 or language_states.shape != (
            batch, language_states.shape[1], c.language_dim
        ):
            raise ValueError("language states must be [batch, token, language_dim]")
        for value, expected, name in checks:
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if noisy_action_tokens is not None:
            expected = (batch, c.action_positions, c.action_hidden_dim)
            if noisy_action_tokens.shape != expected:
                raise ValueError(f"noisy action tokens must have shape {expected}")
            if flow_time is None or flow_time.shape not in ((batch,), (batch, 1)):
                raise ValueError("flow time must be [batch] or [batch, 1]")
        elif flow_time is not None:
            raise ValueError("flow time requires noisy action tokens")
        return batch

    def _slots_for_view(self, patches, patch_mask, xy, camera_id, view_valid):
        c = self.config
        batch, patch_count = patches.shape[:2]
        valid = patch_mask.astype(jnp.bool_) & view_valid[:, None].astype(jnp.bool_)
        # Invalid patches are zeroed before any affine map so NaN padding is unobservable.
        patches = jnp.where(valid[..., None], patches, 0.0)
        xy = jnp.where(valid[..., None], xy, 0.0)
        geometry_features = jnp.concatenate(
            [xy, xy * xy, jnp.linalg.norm(xy, axis=-1, keepdims=True)], axis=-1
        )
        camera = jax.nn.one_hot(
            jnp.full((batch, patch_count), camera_id, dtype=jnp.int32),
            c.camera_views,
            dtype=patches.dtype,
        )
        patch_hidden = jnp.tanh(
            self.patch_in(jnp.concatenate([patches, geometry_features, camera], axis=-1))
        )
        keys, values = jnp.split(self.patch_key_value(patch_hidden), 2, axis=-1)
        slots = jnp.broadcast_to(
            self.slot_seed.value.value[None], (batch, c.object_slots, c.hidden_dim)
        )
        attention = jnp.zeros((batch, c.object_slots, patch_count), jnp.float32)
        competition = attention
        for _ in range(c.slot_iterations):
            queries = self.slot_query(slots)
            logits = jnp.einsum("bkh,bph->bkp", queries, keys) / jnp.sqrt(
                jnp.asarray(c.hidden_dim, jnp.float32)
            )
            # Competition is over slots for every patch, followed by a
            # per-slot patch normalization for the update.
            competition = jax.nn.softmax(logits, axis=1) * valid[:, None]
            attention = competition / jnp.maximum(
                jnp.sum(competition, axis=-1, keepdims=True), 1.0e-6
            )
            read = jnp.einsum("bkp,bph->bkh", attention, values)
            slots = jnp.tanh(self.slot_update(jnp.concatenate([slots, read], axis=-1)))
        slots = slots * view_valid[:, None, None].astype(slots.dtype)
        attention = attention * view_valid[:, None, None].astype(attention.dtype)
        competition = competition * view_valid[:, None, None].astype(competition.dtype)
        # Mass must come from the pre-patch-normalization competitive
        # assignment.  Consequently masses vary by slot and sum to the number
        # of valid patches; the update attention itself sums to one per slot.
        mass = jnp.sum(competition, axis=-1, keepdims=True)
        mean = jnp.einsum("bkp,bpd->bkd", attention, xy)
        second = jnp.einsum("bkp,bpd->bkd", attention, xy * xy)
        sigma = jnp.sqrt(jnp.maximum(second - mean * mean, 0.0) + 1.0e-6)
        geometry = jnp.concatenate([mean, sigma, mass], axis=-1)
        geometry = geometry * view_valid[:, None, None].astype(geometry.dtype)
        projected_targets = jax.lax.stop_gradient(patch_hidden)
        reconstructed = jnp.einsum("bkp,bkh->bph", competition, slots)
        projected_targets = jnp.where(
            valid[..., None], projected_targets, 0.0
        )
        reconstructed = jnp.where(valid[..., None], reconstructed, 0.0)
        return slots, geometry, competition, projected_targets, reconstructed

    def _bind_roles(
        self,
        object_slots,
        object_geometry,
        view_valid,
        language_states,
        language_mask,
        hetm_role_states,
        role_present,
        proprioception,
        frontier,
        relation_kind,
        role_span_mask,
        source_reference_span_mask,
        destination_reference_span_mask,
    ):
        c = self.config
        two_reference_relation = reference_is_between(relation_kind)
        direct_between_destination = destination_relation_is_between(relation_kind)
        language_valid = language_mask.astype(jnp.bool_)
        safe_language = jnp.where(language_valid[..., None], language_states, 0.0)
        projected_language = self.language_align(safe_language)
        projected_language = jnp.where(
            language_valid[..., None], projected_language, 0.0
        )
        role_language = role_conditioned_language_context(
            safe_language, language_valid, role_span_mask
        )
        source_span = (
            source_reference_span_mask.astype(jnp.bool_)
            & language_valid
        )
        destination_spans = (
            destination_reference_span_mask.astype(jnp.bool_)
            & language_valid[:, None]
        )
        source_span_valid = jnp.any(source_span, axis=-1)
        destination_span_union = jnp.any(destination_spans, axis=1)
        source_language = jnp.einsum(
            "bt,bth->bh", source_span.astype(language_states.dtype), safe_language
        ) / jnp.maximum(jnp.sum(source_span, axis=-1, keepdims=True), 1)
        destination_language = jnp.einsum(
            "bt,bth->bh",
            destination_span_union.astype(language_states.dtype),
            safe_language,
        ) / jnp.maximum(
            jnp.sum(destination_span_union, axis=-1, keepdims=True), 1
        )
        whole_prompt = jnp.sum(safe_language, axis=1) / jnp.maximum(
            jnp.sum(language_valid, axis=-1, keepdims=True), 1
        )
        reference_language = jnp.where(
            source_span_valid[:, None],
            source_language,
            jnp.where(
                jnp.any(destination_span_union, axis=-1, keepdims=True),
                destination_language,
                whole_prompt,
            ),
        )
        role_language = role_language.at[:, REFERENCE_ROLE].set(
            reference_language
        )
        shared_state = jnp.concatenate([proprioception, frontier], axis=-1)
        shared_state = jnp.broadcast_to(
            shared_state[:, None],
            (shared_state.shape[0], ROLE_COUNT, shared_state.shape[-1]),
        )
        shared = jnp.concatenate([shared_state, role_language], axis=-1)
        queries = self.role_query(jnp.concatenate([hetm_role_states, shared], axis=-1))
        keys, values = jnp.split(self.role_key_value(object_slots), 2, axis=-1)
        logits = jnp.einsum("brh,bvkh->bvrk", queries, keys) / jnp.sqrt(
            jnp.asarray(c.hidden_dim, jnp.float32)
        )
        slot_mask = jnp.broadcast_to(
            view_valid[:, :, None, None].astype(jnp.bool_), logits.shape
        )
        raw = _masked_softmax(logits, slot_mask, axis=-1)
        target = raw[:, :, TARGET_ROLE]
        exclusion = jnp.log(jnp.maximum(1.0 - jax.lax.stop_gradient(target), 1.0e-6))
        adjusted_logits = logits.at[:, :, REFERENCE_ROLE].add(exclusion)
        adjusted_logits = adjusted_logits.at[:, :, DESTINATION_ROLE].add(exclusion)
        # Preserve the ordered entity hierarchy when the instruction names an
        # independent source reference: destination is selected only after
        # target and source-reference.  Do not apply this to the fallback
        # reference extracted from a destination qualifier (for example,
        # "cabinet" in "top layer of the cabinet"); that qualifier and the
        # destination can legitimately share one visual object slot.
        # Stop-gradient prevents the deployed constraint from becoming an
        # asymmetric training shortcut.
        reference_binding = _masked_softmax(
            adjusted_logits[:, :, REFERENCE_ROLE],
            slot_mask[:, :, REFERENCE_ROLE],
            axis=-1,
        )
        reference_exclusion = jnp.log(
            jnp.maximum(
                1.0 - jax.lax.stop_gradient(reference_binding), 1.0e-6
            )
        )
        ordered_roles_present = (
            source_span_valid
            & role_present[:, REFERENCE_ROLE]
            & role_present[:, DESTINATION_ROLE]
        )
        adjusted_logits = adjusted_logits.at[:, :, DESTINATION_ROLE].add(
            jnp.where(
                ordered_roles_present[:, None, None],
                reference_exclusion,
                0.0,
            )
        )
        bindings = _masked_softmax(adjusted_logits, slot_mask, axis=-1)

        # "between" has two references.  A second distribution excludes the
        # primary winner; averaging the two gives exactly two competitive modes.
        ref_logits = adjusted_logits[:, :, REFERENCE_ROLE]
        winner = jnp.argmax(ref_logits, axis=-1)
        second_mask = view_valid[:, :, None].astype(jnp.bool_) & ~jax.nn.one_hot(
            winner, c.object_slots, dtype=jnp.bool_
        )
        second_ref = _masked_softmax(ref_logits, second_mask, axis=-1)
        two_ref = 0.5 * (bindings[:, :, REFERENCE_ROLE] + second_ref)
        reference_weights = jnp.where(
            (two_reference_relation & ~source_span_valid)[:, None, None],
            two_ref,
            bindings[:, :, REFERENCE_ROLE],
        )
        bindings = bindings.at[:, :, REFERENCE_ROLE].set(reference_weights)

        full_mask = (
            view_valid[:, :, None, None].astype(jnp.bool_)
            & role_present[:, None, :, None].astype(jnp.bool_)
        )
        bindings = jnp.where(full_mask, bindings, 0.0)
        raw = jnp.where(full_mask, raw, 0.0)
        per_view_nodes = jnp.einsum("bvrk,bvkh->bvrh", bindings, values)

        # Parameter-shared cross-view attention, independently for each role.
        cross = self.cross_view_out(per_view_nodes)
        cross_logits = jnp.einsum("bvrh,bwrh->brvw", per_view_nodes, cross) / jnp.sqrt(
            jnp.asarray(c.hidden_dim, jnp.float32)
        )
        cross_mask = jnp.broadcast_to(
            view_valid[:, None, None, :].astype(jnp.bool_), cross_logits.shape
        )
        cross_weights = _masked_softmax(cross_logits, cross_mask, axis=-1)
        fused_per_view = jnp.einsum("brvw,bwrh->bvrh", cross_weights, per_view_nodes)
        view_weights = view_valid.astype(jnp.float32)
        fused = jnp.einsum("bv,bvrh->brh", view_weights, fused_per_view) / jnp.maximum(
            jnp.sum(view_weights, axis=-1, keepdims=True)[..., None], 1.0
        )
        fused = jnp.where(role_present[..., None], fused, 0.0)

        role_geometry_view = jnp.einsum("bvrk,bvkd->bvrd", bindings, object_geometry)
        role_geometry = jnp.einsum("bv,bvrd->brd", view_weights, role_geometry_view) / jnp.maximum(
            jnp.sum(view_weights, axis=-1, keepdims=True)[..., None], 1.0
        )
        role_geometry = jnp.where(role_present[..., None], role_geometry, 0.0)
        # Keep the shared-projection per-view role embeddings for the
        # cross-view semantic objective.  Slot indices are local latent
        # variables and have no guaranteed front/wrist correspondence, so a
        # loss must not compare the two binding vectors coordinate-wise.
        cross_view_role_embeddings = jnp.where(
            (
                view_valid[:, :, None]
                & role_present[:, None, :]
            )[..., None],
            cross,
            0.0,
        )

        # Bind the two destination-side relation references independently.
        # Their parser spans are available at inference; plural direct-between
        # tasks deliberately provide the same span twice, while the exclusion
        # below forces the second endpoint onto another visual instance.
        destination_span_valid = jnp.any(destination_spans, axis=-1)
        destination_reference_language = jnp.einsum(
            "brt,bth->brh",
            destination_spans.astype(language_states.dtype),
            safe_language,
        ) / jnp.maximum(
            jnp.sum(destination_spans, axis=-1, keepdims=True), 1
        )
        reference_state = jnp.broadcast_to(
            hetm_role_states[:, REFERENCE_ROLE : REFERENCE_ROLE + 1],
            (hetm_role_states.shape[0], 2, c.hetm_role_dim),
        )
        reference_shared_state = jnp.broadcast_to(
            shared_state[:, REFERENCE_ROLE : REFERENCE_ROLE + 1],
            (shared_state.shape[0], 2, shared_state.shape[-1]),
        )
        destination_reference_queries = self.role_query(
            jnp.concatenate(
                [
                    reference_state,
                    reference_shared_state,
                    destination_reference_language,
                ],
                axis=-1,
            )
        )
        destination_reference_logits = jnp.einsum(
            "brh,bvkh->bvrk", destination_reference_queries, keys
        ) / jnp.sqrt(jnp.asarray(c.hidden_dim, jnp.float32))
        destination_reference_logits = destination_reference_logits + exclusion[:, :, None]
        first_destination_binding = _masked_softmax(
            destination_reference_logits[:, :, :1],
            slot_mask[:, :, :1],
            axis=-1,
        )
        second_exclusion = jnp.log(
            jnp.maximum(
                1.0 - jax.lax.stop_gradient(first_destination_binding[:, :, 0]),
                1.0e-6,
            )
        )
        destination_reference_logits = destination_reference_logits.at[:, :, 1].add(
            jnp.where(
                two_reference_relation[:, None, None], second_exclusion, 0.0
            )
        )
        destination_reference_bindings = _masked_softmax(
            destination_reference_logits, slot_mask[:, :, :2], axis=-1
        )
        destination_reference_full_mask = (
            view_valid[:, :, None, None].astype(jnp.bool_)
            & destination_span_valid[:, None, :, None]
        )
        destination_reference_bindings = jnp.where(
            destination_reference_full_mask,
            destination_reference_bindings,
            0.0,
        )
        destination_reference_per_view = jnp.einsum(
            "bvrk,bvkh->bvrh", destination_reference_bindings, values
        )
        destination_reference_nodes = jnp.einsum(
            "bv,bvrh->brh", view_weights, destination_reference_per_view
        ) / jnp.maximum(jnp.sum(view_weights, axis=-1, keepdims=True)[..., None], 1.0)
        destination_reference_geometry_view = jnp.einsum(
            "bvrk,bvkd->bvrd", destination_reference_bindings, object_geometry
        )
        destination_reference_geometry = jnp.einsum(
            "bv,bvrd->brd", view_weights, destination_reference_geometry_view
        ) / jnp.maximum(jnp.sum(view_weights, axis=-1, keepdims=True)[..., None], 1.0)
        destination_reference_nodes = jnp.where(
            destination_span_valid[..., None], destination_reference_nodes, 0.0
        )
        destination_reference_geometry = jnp.where(
            destination_span_valid[..., None], destination_reference_geometry, 0.0
        )

        # A direct ``push ... to the region between A and B`` destination has
        # no visible object of its own. Represent that free-space region by
        # the midpoint of the two independently bound endpoint objects. A
        # nested qualifier such as ``place X on the bowl between A and B`` has
        # a source span in the production corpus and keeps the learned bowl
        # binding instead. Keep binding, node, geometry and cross-view
        # supervision mutually consistent; shared projections add no arrays.
        direct_between_region = (
            direct_between_destination
            & jnp.all(destination_span_valid, axis=-1)
            & role_present[:, DESTINATION_ROLE]
        )
        endpoint_count = jnp.maximum(
            jnp.sum(destination_span_valid.astype(jnp.float32), axis=-1), 1.0
        )
        region_bindings = jnp.sum(destination_reference_bindings, axis=2) / (
            endpoint_count[:, None, None]
        )
        bindings = bindings.at[:, :, DESTINATION_ROLE].set(
            jnp.where(
                direct_between_region[:, None, None],
                region_bindings,
                bindings[:, :, DESTINATION_ROLE],
            )
        )
        region_per_view = jnp.sum(destination_reference_per_view, axis=2) / (
            endpoint_count[:, None, None]
        )
        region_cross = self.cross_view_out(region_per_view)
        region_cross_logits = jnp.einsum(
            "bvh,bwh->bvw", region_per_view, region_cross
        ) / jnp.sqrt(jnp.asarray(c.hidden_dim, jnp.float32))
        region_cross_mask = jnp.broadcast_to(
            view_valid[:, None, :].astype(jnp.bool_), region_cross_logits.shape
        )
        region_cross_weights = _masked_softmax(
            region_cross_logits, region_cross_mask, axis=-1
        )
        region_fused_per_view = jnp.einsum(
            "bvw,bwh->bvh", region_cross_weights, region_per_view
        )
        region_fused = jnp.einsum(
            "bv,bvh->bh", view_weights, region_fused_per_view
        ) / jnp.maximum(jnp.sum(view_weights, axis=-1, keepdims=True), 1.0)
        fused = fused.at[:, DESTINATION_ROLE].set(
            jnp.where(
                direct_between_region[:, None],
                region_fused,
                fused[:, DESTINATION_ROLE],
            )
        )
        region_geometry_view = jnp.sum(
            destination_reference_geometry_view, axis=2
        ) / endpoint_count[:, None, None]
        region_geometry = jnp.einsum(
            "bv,bvd->bd", view_weights, region_geometry_view
        ) / jnp.maximum(jnp.sum(view_weights, axis=-1, keepdims=True), 1.0)
        role_geometry = role_geometry.at[:, DESTINATION_ROLE].set(
            jnp.where(
                direct_between_region[:, None],
                region_geometry,
                role_geometry[:, DESTINATION_ROLE],
            )
        )
        cross_view_role_embeddings = cross_view_role_embeddings.at[
            :, :, DESTINATION_ROLE
        ].set(
            jnp.where(
                direct_between_region[:, None, None]
                & view_valid[:, :, None],
                region_cross,
                cross_view_role_embeddings[:, :, DESTINATION_ROLE],
            )
        )
        return (
            fused,
            role_geometry,
            bindings,
            raw,
            projected_language,
            cross_view_role_embeddings,
            destination_reference_nodes,
            destination_reference_geometry,
            destination_span_valid,
            destination_reference_bindings,
        )

    def _edge_features(
        self,
        nodes,
        role_geometry,
        role_present,
        predicate_probabilities,
        frontier,
        proprioception,
        previous_actions,
        previous_target_anchor,
        previous_target_geometry,
        previous_target_valid,
        episode_start,
        relation_reference_nodes,
        relation_reference_geometry,
        relation_reference_mask,
    ):
        batch = nodes.shape[0]
        relation_present = role_present[:, REFERENCE_ROLE] | role_present[:, DESTINATION_ROLE]
        previous_ok = previous_target_valid.astype(
            jnp.bool_
        ) & ~episode_start.astype(jnp.bool_)
        sources = jnp.stack(
            [
                nodes[:, AGENT_ROLE],
                nodes[:, TARGET_ROLE],
                nodes[:, TARGET_ROLE],
                nodes[:, TARGET_ROLE],
            ],
            axis=1,
        )
        destinations = jnp.stack(
            [
                nodes[:, TARGET_ROLE],
                nodes[:, REFERENCE_ROLE],
                nodes[:, HAZARD_ROLE],
                previous_target_anchor,
            ],
            axis=1,
        )
        zero_node = jnp.zeros_like(nodes[:, TARGET_ROLE])
        secondary_destinations = jnp.stack(
            [
                zero_node,
                nodes[:, DESTINATION_ROLE],
                zero_node,
                zero_node,
            ],
            axis=1,
        )
        source_geometry = jnp.stack(
            [
                role_geometry[:, AGENT_ROLE],
                role_geometry[:, TARGET_ROLE],
                role_geometry[:, TARGET_ROLE],
                role_geometry[:, TARGET_ROLE],
            ],
            axis=1,
        )
        destination_geometry = jnp.stack(
            [
                role_geometry[:, TARGET_ROLE],
                role_geometry[:, REFERENCE_ROLE],
                role_geometry[:, HAZARD_ROLE],
                previous_target_geometry,
            ],
            axis=1,
        )
        # Zero relative geometry for non-relation edges; on the relation edge
        # this is the ordered target->destination displacement.
        secondary_destination_geometry = source_geometry.at[:, 1].set(
            role_geometry[:, DESTINATION_ROLE]
        )
        edge_mask = jnp.stack(
            [
                role_present[:, AGENT_ROLE] & role_present[:, TARGET_ROLE],
                role_present[:, TARGET_ROLE] & relation_present,
                role_present[:, TARGET_ROLE] & role_present[:, HAZARD_ROLE],
                role_present[:, TARGET_ROLE] & previous_ok,
            ],
            axis=-1,
        )
        global_context = jnp.concatenate(
            [
                predicate_probabilities.reshape(batch, -1),
                frontier,
                proprioception,
                previous_actions.reshape(batch, -1),
            ],
            axis=-1,
        )
        global_context = jnp.broadcast_to(
            global_context[:, None], (batch, EDGE_COUNT, global_context.shape[-1])
        )
        directions = jnp.broadcast_to(
            jnp.eye(EDGE_COUNT, dtype=nodes.dtype)[None], (batch, EDGE_COUNT, EDGE_COUNT)
        )
        relative_geometry = destination_geometry - source_geometry
        secondary_relative_geometry = (
            secondary_destination_geometry - source_geometry
        )
        relation_role_flags = jnp.zeros(
            (batch, EDGE_COUNT, 2), dtype=nodes.dtype
        )
        relation_role_flags = relation_role_flags.at[:, 1, 0].set(
            role_present[:, REFERENCE_ROLE].astype(nodes.dtype)
        )
        relation_role_flags = relation_role_flags.at[:, 1, 1].set(
            role_present[:, DESTINATION_ROLE].astype(nodes.dtype)
        )
        relation_reference_node_features = jnp.zeros(
            (batch, EDGE_COUNT, 2, self.config.hidden_dim), dtype=nodes.dtype
        ).at[:, 1].set(relation_reference_nodes)
        relation_reference_relative_geometry = jnp.zeros(
            (batch, EDGE_COUNT, 2, 5), dtype=role_geometry.dtype
        ).at[:, 1].set(
            relation_reference_geometry - source_geometry[:, 1:2]
        )
        relation_reference_flags = jnp.zeros(
            (batch, EDGE_COUNT, 2), dtype=nodes.dtype
        ).at[:, 1].set(relation_reference_mask.astype(nodes.dtype))
        edge_input = jnp.concatenate(
            [
                sources,
                destinations,
                secondary_destinations,
                sources - destinations,
                sources * destinations,
                relative_geometry,
                secondary_relative_geometry,
                relation_role_flags,
                relation_reference_node_features.reshape(batch, EDGE_COUNT, -1),
                relation_reference_relative_geometry.reshape(batch, EDGE_COUNT, -1),
                relation_reference_flags,
                global_context,
                directions,
            ],
            axis=-1,
        )
        edges = jnp.tanh(self.edge_in(edge_input))
        edges = jnp.where(edge_mask[..., None], edges, 0.0)
        return edges, edge_mask

    def encode_scene(
        self,
        patches: jax.Array,
        patch_mask: jax.Array,
        patch_xy: jax.Array,
        view_valid: jax.Array,
        language_states: jax.Array,
        language_mask: jax.Array,
        hetm_role_states: jax.Array,
        role_present: jax.Array,
        predicate_probabilities: jax.Array,
        frontier: jax.Array,
        proprioception: jax.Array,
        previous_actions: jax.Array,
        previous_target_anchor: jax.Array,
        previous_target_geometry: jax.Array,
        previous_target_valid: jax.Array,
        episode_start: jax.Array,
        relation_kind: jax.Array,
        role_span_mask: jax.Array,
        source_reference_span_mask: jax.Array,
        destination_reference_span_mask: jax.Array,
        external_role_residual: jax.Array | None = None,
        depth_probabilities: jax.Array | None = None,
        depth_mask: jax.Array | None = None,
        previous_target_depth_features: jax.Array | None = None,
        previous_target_depth_valid: jax.Array | None = None,
        previous_depth_grid: jax.Array | None = None,
        previous_depth_confidence: jax.Array | None = None,
        previous_depth_valid: jax.Array | None = None,
        previous_role_depth_features: jax.Array | None = None,
        previous_role_depth_valid: jax.Array | None = None,
        previous_role_depth_forecast: jax.Array | None = None,
        previous_role_depth_forecast_valid: jax.Array | None = None,
    ) -> RACGScene:
        self._validate_inputs(
            patches, patch_mask, patch_xy, view_valid, language_states, language_mask,
            hetm_role_states, role_present, predicate_probabilities, frontier,
            proprioception, previous_actions, previous_target_anchor,
            previous_target_geometry, previous_target_valid, episode_start,
            relation_kind, role_span_mask, source_reference_span_mask,
            destination_reference_span_mask,
        )
        effective_view_valid = view_valid.astype(jnp.bool_) & jnp.any(
            patch_mask.astype(jnp.bool_), axis=-1
        )
        effective_role_present = role_present.astype(jnp.bool_) & jnp.any(
            effective_view_valid, axis=-1, keepdims=True
        )
        per_view = [
            self._slots_for_view(
                patches[:, view],
                patch_mask[:, view],
                patch_xy[:, view],
                view,
                effective_view_valid[:, view],
            )
            for view in range(patches.shape[1])
        ]
        object_slots = jnp.stack([encoded[0] for encoded in per_view], axis=1)
        object_geometry = jnp.stack([encoded[1] for encoded in per_view], axis=1)
        slot_attention = jnp.stack([encoded[2] for encoded in per_view], axis=1)
        projected_patch_targets = jnp.stack(
            [encoded[3] for encoded in per_view], axis=1
        )
        reconstructed_projected_patches = jnp.stack(
            [encoded[4] for encoded in per_view], axis=1
        )
        (
            nodes,
            role_geometry,
            bindings,
            raw_bindings,
            projected_language_tokens,
            cross_view_role_embeddings,
            relation_reference_nodes,
            relation_reference_geometry,
            relation_reference_mask,
            relation_reference_bindings,
        ) = self._bind_roles(
            object_slots, object_geometry, effective_view_valid, language_states,
            language_mask, hetm_role_states, effective_role_present, proprioception,
            frontier, relation_kind, role_span_mask,
            source_reference_span_mask, destination_reference_span_mask,
        )
        # Binding queries choose visual content but are not themselves part of
        # the bound values.  Add explicit type tags so two roles that attend to
        # the same object slot do not become indistinguishable to the
        # permutation-invariant action graph reader.
        role_type = structural_role_type_basis(
            self.config.hidden_dim, nodes.dtype
        )[None]
        nodes = jnp.where(
            effective_role_present[..., None], nodes + role_type, 0.0
        )
        if external_role_residual is not None:
            expected = (nodes.shape[0], ROLE_COUNT, self.config.hidden_dim)
            if external_role_residual.shape != expected:
                raise ValueError(
                    f"external role residual must have shape {expected}"
                )
            nodes = nodes + external_role_residual.astype(nodes.dtype)
            nodes = jnp.where(
                effective_role_present[..., None], nodes, 0.0
            )

        depth_inputs = (
            depth_probabilities,
            depth_mask,
            previous_target_depth_features,
            previous_target_depth_valid,
            previous_depth_grid,
            previous_depth_confidence,
            previous_depth_valid,
        )
        depth_lift_output = None
        depth_motion_forecast_output = None
        if hasattr(self, "depth_lift"):
            if any(value is None for value in depth_inputs):
                raise ValueError("depth-lift RACG requires all depth inputs")
            base_role_geometry = jnp.einsum(
                "brk,bkd->brd", bindings[:, 0], object_geometry[:, 0]
            )
            # Depth is predicted in the base-camera grid, so endpoint xy must
            # be pooled in that same camera frame.  The graph-facing endpoint
            # geometry above is cross-view fused and cannot be mixed with a
            # base-view depth map without a calibrated camera transform.
            base_relation_reference_geometry = jnp.einsum(
                "bek,bkd->bed",
                relation_reference_bindings[:, 0],
                object_geometry[:, 0],
            )
            depth_lift_output = self.depth_lift(
                depth_probabilities=depth_probabilities,
                depth_mask=depth_mask,
                slot_attention=slot_attention[:, 0],
                role_bindings=bindings[:, 0],
                role_xy=base_role_geometry[..., :2],
                role_present=effective_role_present,
                relation_reference_bindings=relation_reference_bindings[:, 0],
                relation_reference_xy=base_relation_reference_geometry[..., :2],
                relation_reference_present=relation_reference_mask,
                base_view_valid=effective_view_valid[:, 0],
                previous_target_depth_features=previous_target_depth_features,
                previous_target_depth_valid=previous_target_depth_valid,
                previous_depth_grid=previous_depth_grid,
                previous_depth_confidence=previous_depth_confidence,
                previous_depth_valid=previous_depth_valid,
                episode_start=episode_start,
            )
            nodes = nodes + depth_lift_output.role_residual.astype(nodes.dtype)
            nodes = jnp.where(effective_role_present[..., None], nodes, 0.0)
            if hasattr(self, "depth_forecaster"):
                if (
                    previous_role_depth_features is None
                    or previous_role_depth_valid is None
                    or previous_role_depth_forecast is None
                    or previous_role_depth_forecast_valid is None
                ):
                    raise ValueError(
                        "depth motion forecasting requires private forecast state"
                    )
                depth_motion_forecast_output = self.depth_forecaster(
                    current_role_features=depth_lift_output.role_depth_features,
                    current_role_valid=depth_lift_output.role_valid,
                    previous_role_features=previous_role_depth_features,
                    previous_role_valid=previous_role_depth_valid,
                    previous_forecast=previous_role_depth_forecast,
                    previous_forecast_valid=previous_role_depth_forecast_valid,
                    proprioception=proprioception,
                    episode_start=episode_start,
                    previous_actions=previous_actions,
                )
                nodes = nodes + depth_motion_forecast_output.role_policy_residual.astype(
                    nodes.dtype
                )
                nodes = jnp.where(effective_role_present[..., None], nodes, 0.0)
        elif any(value is not None for value in depth_inputs):
            raise ValueError("depth inputs require a configured depth-lift module")

        # Two graph iterations share both edge and node-update weights.
        for _ in range(self.config.graph_iterations):
            edges, edge_mask = self._edge_features(
                nodes, role_geometry, effective_role_present, predicate_probabilities,
                frontier, proprioception, previous_actions,
                previous_target_anchor, previous_target_geometry,
                previous_target_valid, episode_start,
                relation_reference_nodes, relation_reference_geometry,
                relation_reference_mask,
            )
            if depth_lift_output is not None:
                edges = edges + depth_lift_output.edge_residual.astype(edges.dtype)
                edges = jnp.where(edge_mask[..., None], edges, 0.0)
            if depth_motion_forecast_output is not None:
                edges = edges + (
                    depth_motion_forecast_output.edge_policy_residual.astype(
                        edges.dtype
                    )
                )
                edges = jnp.where(edge_mask[..., None], edges, 0.0)
            current_contact_logits = self.contact_head(edges[:, 0])
            current_contact_logits = jnp.where(
                edge_mask[:, :1], current_contact_logits, 0.0
            )
            contact_probabilities = jax.nn.softmax(
                current_contact_logits, axis=-1
            )
            phase_basis = jnp.sin(
                (jnp.arange(CONTACT_STATES, dtype=jnp.float32)[:, None] + 1.0)
                * (jnp.arange(self.config.hidden_dim, dtype=jnp.float32)[None] + 1.0)
                / self.config.hidden_dim
            )
            contact_state = contact_probabilities @ phase_basis
            edges = edges.at[:, 0].add(contact_state * edge_mask[:, :1])

            # Preserve each heterogeneous edge in its own signed incidence
            # channel.  Summing first can exactly cancel simultaneous contact,
            # relation, hazard, and temporal evidence at the target node.
            messages = typed_incident_messages(edges)
            updated = gated_graph_residual(
                self.graph_update(jnp.concatenate([nodes, messages], axis=-1))
            )
            nodes = jnp.where(
                effective_role_present[..., None], nodes + updated, 0.0
            )

        graph_tokens = jnp.concatenate([nodes, edges], axis=1)
        graph_mask = jnp.concatenate(
            [effective_role_present, edge_mask], axis=1
        )
        graph_tokens = jnp.where(graph_mask[..., None], graph_tokens, 0.0)
        relation_logits = factorized_relation_logits(
            self.relation_head(edges[:, 1])
        )
        relation_logits = jnp.where(
            edge_mask[:, 1:2], relation_logits, 0.0
        )
        return RACGScene(
            graph_tokens=graph_tokens,
            graph_token_mask=graph_mask,
            object_slots=object_slots,
            object_geometry=object_geometry,
            slot_attention=slot_attention,
            projected_patch_targets=projected_patch_targets,
            reconstructed_projected_patches=reconstructed_projected_patches,
            cross_view_role_embeddings=cross_view_role_embeddings,
            relation_reference_nodes=relation_reference_nodes,
            relation_reference_geometry=relation_reference_geometry,
            relation_reference_mask=relation_reference_mask,
            relation_reference_bindings=relation_reference_bindings,
            role_nodes=nodes,
            role_geometry=role_geometry,
            role_bindings=bindings,
            unexcluded_role_bindings=raw_bindings,
            directed_edges=edges,
            edge_mask=edge_mask,
            edge_contact_logits=current_contact_logits,
            relation_logits=relation_logits,
            projected_language_tokens=projected_language_tokens,
            depth_lift_output=depth_lift_output,
            depth_motion_forecast_output=depth_motion_forecast_output,
        )

    def read_actions(
        self,
        scene: RACGScene,
        noisy_action_tokens: jax.Array,
        flow_time: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Read an encoded scene for one flow sample or ODE evaluation."""

        if noisy_action_tokens.ndim != 3:
            raise ValueError("noisy action tokens must be [batch, position, width]")
        batch = noisy_action_tokens.shape[0]
        expected_action = (
            batch,
            self.config.action_positions,
            self.config.action_hidden_dim,
        )
        if noisy_action_tokens.shape != expected_action:
            raise ValueError(f"noisy action tokens must have shape {expected_action}")
        if scene.graph_tokens.shape != (batch, ROLE_COUNT + EDGE_COUNT, self.config.hidden_dim):
            raise ValueError("RACG scene graph token shape drifted")
        if scene.graph_token_mask.shape != (batch, ROLE_COUNT + EDGE_COUNT):
            raise ValueError("RACG scene graph mask shape drifted")
        if flow_time.shape not in ((batch,), (batch, 1)):
            raise ValueError("flow time must be [batch] or [batch, 1]")
        coordinates = harmonic_action_coordinates(
            flow_time, self.config.action_positions, noisy_action_tokens.dtype
        )
        action_queries = self.action_query(
            jnp.concatenate([noisy_action_tokens, coordinates], axis=-1)
        )
        graph_keys, graph_values = jnp.split(
            self.graph_key_value(scene.graph_tokens), 2, axis=-1
        )
        graph_read = multihead_graph_read(
            action_queries,
            graph_keys,
            graph_values,
            scene.graph_token_mask,
            self.config.action_read_heads,
            structural_action_read_bias(action_queries.dtype),
        )
        graph_valid = jnp.any(scene.graph_token_mask, axis=-1)
        graph_read = jnp.where(graph_valid[:, None, None], graph_read, 0.0)
        action_residual = self.graph_action_out(graph_read)
        action_residual = jnp.where(
            graph_valid[:, None, None], action_residual, 0.0
        )
        # The same contact head that forms the current agent->target edge state
        # is shared over all ten action-position graph reads.
        contact_logits = self.contact_head(graph_read)
        contact_logits = jnp.where(
            scene.edge_mask[:, :1, None], contact_logits, 0.0
        )
        # Position zero is the current anchor transition and directly
        # supervises the same edge-phase logits that formed the latent graph.
        # Later positions retain action-conditioned graph-read predictions.
        contact_logits = contact_logits.at[:, 0].set(
            scene.edge_contact_logits.astype(contact_logits.dtype)
        )
        return action_residual, graph_read, contact_logits

    def __call__(
        self,
        patches: jax.Array,
        patch_mask: jax.Array,
        patch_xy: jax.Array,
        view_valid: jax.Array,
        language_states: jax.Array,
        language_mask: jax.Array,
        hetm_role_states: jax.Array,
        role_present: jax.Array,
        predicate_probabilities: jax.Array,
        frontier: jax.Array,
        proprioception: jax.Array,
        previous_actions: jax.Array,
        previous_target_anchor: jax.Array,
        previous_target_geometry: jax.Array,
        previous_target_valid: jax.Array,
        episode_start: jax.Array,
        relation_kind: jax.Array,
        role_span_mask: jax.Array,
        source_reference_span_mask: jax.Array,
        destination_reference_span_mask: jax.Array,
        noisy_action_tokens: jax.Array,
        flow_time: jax.Array,
        depth_probabilities: jax.Array | None = None,
        depth_mask: jax.Array | None = None,
        previous_target_depth_features: jax.Array | None = None,
        previous_target_depth_valid: jax.Array | None = None,
        previous_depth_grid: jax.Array | None = None,
        previous_depth_confidence: jax.Array | None = None,
        previous_depth_valid: jax.Array | None = None,
        previous_role_depth_features: jax.Array | None = None,
        previous_role_depth_valid: jax.Array | None = None,
    ) -> RACGOutput:
        scene = self.encode_scene(
            patches,
            patch_mask,
            patch_xy,
            view_valid,
            language_states,
            language_mask,
            hetm_role_states,
            role_present,
            predicate_probabilities,
            frontier,
            proprioception,
            previous_actions,
            previous_target_anchor,
            previous_target_geometry,
            previous_target_valid,
            episode_start,
            relation_kind,
            role_span_mask,
            source_reference_span_mask,
            destination_reference_span_mask,
            depth_probabilities=depth_probabilities,
            depth_mask=depth_mask,
            previous_target_depth_features=previous_target_depth_features,
            previous_target_depth_valid=previous_target_depth_valid,
            previous_depth_grid=previous_depth_grid,
            previous_depth_confidence=previous_depth_confidence,
            previous_depth_valid=previous_depth_valid,
            previous_role_depth_features=previous_role_depth_features,
            previous_role_depth_valid=previous_role_depth_valid,
        )
        action_residual, graph_read, contact_logits = self.read_actions(
            scene, noisy_action_tokens, flow_time
        )
        return RACGOutput(
            action_residual=action_residual,
            graph_read=graph_read,
            graph_tokens=scene.graph_tokens,
            graph_token_mask=scene.graph_token_mask,
            object_slots=scene.object_slots,
            object_geometry=scene.object_geometry,
            slot_attention=scene.slot_attention,
            projected_patch_targets=scene.projected_patch_targets,
            reconstructed_projected_patches=scene.reconstructed_projected_patches,
            cross_view_role_embeddings=scene.cross_view_role_embeddings,
            relation_reference_nodes=scene.relation_reference_nodes,
            relation_reference_geometry=scene.relation_reference_geometry,
            relation_reference_mask=scene.relation_reference_mask,
            relation_reference_bindings=scene.relation_reference_bindings,
            role_nodes=scene.role_nodes,
            role_geometry=scene.role_geometry,
            role_bindings=scene.role_bindings,
            unexcluded_role_bindings=scene.unexcluded_role_bindings,
            directed_edges=scene.directed_edges,
            edge_mask=scene.edge_mask,
            edge_contact_logits=scene.edge_contact_logits,
            contact_logits=contact_logits,
            relation_logits=scene.relation_logits,
            projected_language_tokens=scene.projected_language_tokens,
        )


def sample_local_contact_loss(
    output: RACGOutput, targets: jax.Array, valid: jax.Array
) -> jax.Array:
    """Convenience wrapper for the deployable logits-only loss interface."""

    return sample_local_contact_logits_loss(
        output.contact_logits,
        targets,
        valid,
        graph_valid=output.edge_mask[:, 0],
    )


def sample_local_contact_logits_loss(
    contact_logits: jax.Array,
    targets: jax.Array,
    valid: jax.Array,
    *,
    graph_valid: jax.Array | None = None,
    class_weights: jax.Array | None = None,
) -> jax.Array:
    """Class-balanced per-sample contact CE with exact empty-mask zero.

    This lower-level interface is used by Pi0 integration, where action reads
    are already available and constructing a synthetic :class:`RACGOutput`
    would be incorrect.
    """

    if contact_logits.ndim != 3 or contact_logits.shape[1:] != (
        10,
        CONTACT_STATES,
    ):
        raise ValueError("contact logits must be [batch, 10, 4]")
    batch, positions = contact_logits.shape[:2]
    if targets.shape != (batch, positions) or valid.shape != (batch, positions):
        raise ValueError("contact targets and validity must be [batch, position]")
    if graph_valid is None:
        graph_valid = jnp.ones((batch,), dtype=jnp.bool_)
    elif graph_valid.shape != (batch,):
        raise ValueError("contact graph validity must be [batch]")
    if class_weights is None:
        class_weights = jnp.asarray(CONTACT_CLASS_WEIGHTS, dtype=jnp.float32)
    elif class_weights.shape != (CONTACT_STATES,):
        raise ValueError("contact class weights must have shape [4]")
    target_indices = jnp.clip(
        targets.astype(jnp.int32), 0, CONTACT_STATES - 1
    )
    labels = jax.nn.one_hot(
        target_indices,
        CONTACT_STATES,
        dtype=jnp.float32,
    )
    loss = -jnp.sum(labels * jax.nn.log_softmax(contact_logits), axis=-1)
    target_weights = jnp.take(class_weights, target_indices, axis=0)
    loss = loss * target_weights
    mask = valid.astype(jnp.bool_) & graph_valid[:, None].astype(jnp.bool_)
    numerator = jnp.sum(jnp.where(mask, loss, 0.0), axis=-1)
    denominator = jnp.maximum(jnp.sum(mask, axis=-1), 1)
    return numerator / denominator


def sample_local_relation_logits_loss(
    relation_logits: jax.Array,
    targets: jax.Array,
    valid: jax.Array,
    *,
    class_weights: jax.Array | None = None,
) -> jax.Array:
    """Balanced 16-class directed-relation CE with exact empty-mask zero."""

    if relation_logits.ndim != 2 or relation_logits.shape[-1] != RELATION_CLASSES:
        raise ValueError("relation logits must be [batch, 16]")
    batch = relation_logits.shape[0]
    if targets.shape != (batch,) or valid.shape != (batch,):
        raise ValueError("relation targets and validity must be [batch]")
    if class_weights is None:
        class_weights = jnp.asarray(RELATION_CLASS_WEIGHTS, dtype=jnp.float32)
    elif class_weights.shape != (RELATION_CLASSES,):
        raise ValueError("relation class weights must have shape [16]")
    indices = jnp.clip(targets.astype(jnp.int32), 0, RELATION_CLASSES - 1)
    loss = -jnp.take_along_axis(
        jax.nn.log_softmax(relation_logits, axis=-1), indices[:, None], axis=-1
    )[:, 0]
    loss = loss * jnp.take(class_weights, indices, axis=0)
    return jnp.where(valid.astype(jnp.bool_), loss, 0.0)


def factorized_relation_logits(marginal_logits: jax.Array) -> jax.Array:
    """Compose source/destination four-way marginals into 16 joint logits."""

    if marginal_logits.ndim < 1 or marginal_logits.shape[-1] != (
        2 * RELATION_MARGINAL_CLASSES
    ):
        raise ValueError("relation marginal logits must end in width 8")
    source, destination = jnp.split(marginal_logits, 2, axis=-1)
    joint = source[..., :, None] + destination[..., None, :]
    return joint.reshape(*marginal_logits.shape[:-1], RELATION_CLASSES)


def sample_local_slot_reconstruction_loss(
    projected_patch_targets: jax.Array,
    reconstructed_projected_patches: jax.Array,
    patch_valid: jax.Array,
) -> jax.Array:
    """Per-sample masked projected-patch MSE; target is caller stop-gradient safe."""

    if projected_patch_targets.ndim != 4:
        raise ValueError("projected patch targets must be [batch, view, patch, hidden]")
    if reconstructed_projected_patches.shape != projected_patch_targets.shape:
        raise ValueError("reconstructed projected patches must match targets")
    if patch_valid.shape != projected_patch_targets.shape[:3]:
        raise ValueError("projected patch validity must be [batch, view, patch]")
    squared = jnp.mean(
        jnp.square(
            reconstructed_projected_patches
            - jax.lax.stop_gradient(projected_patch_targets)
        ),
        axis=-1,
    )
    mask = patch_valid.astype(jnp.bool_)
    numerator = jnp.sum(jnp.where(mask, squared, 0.0), axis=(1, 2))
    denominator = jnp.maximum(jnp.sum(mask, axis=(1, 2)), 1)
    return numerator / denominator


def sample_local_role_alignment_loss(
    role_nodes: jax.Array,
    projected_language_tokens: jax.Array,
    role_span_mask: jax.Array,
    *,
    role_weights: jax.Array | None = None,
) -> jax.Array:
    """Align each valid RACG role node with its projected language span."""

    if role_nodes.ndim != 3 or role_nodes.shape[1] != ROLE_COUNT:
        raise ValueError("role nodes must be [batch, 6, hidden]")
    batch, token_count, hidden = projected_language_tokens.shape
    if role_nodes.shape != (batch, ROLE_COUNT, hidden):
        raise ValueError("role and projected-language widths must match")
    if role_span_mask.shape != (batch, ROLE_COUNT, token_count):
        raise ValueError("role span mask must be [batch, 6, token]")
    if role_weights is None:
        role_weights = jnp.asarray(ROLE_SPAN_CLASS_WEIGHTS, dtype=jnp.float32)
    elif role_weights.shape != (ROLE_COUNT,):
        raise ValueError("role weights must have shape [6]")
    mask = role_span_mask.astype(jnp.bool_)
    span = jnp.einsum(
        "brt,bth->brh", mask.astype(jnp.float32), projected_language_tokens
    ) / jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
    role_norm = jnp.sqrt(jnp.sum(jnp.square(role_nodes), axis=-1) + 1.0e-6)
    span_norm = jnp.sqrt(jnp.sum(jnp.square(span), axis=-1) + 1.0e-6)
    cosine = jnp.sum(role_nodes * span, axis=-1) / (role_norm * span_norm)
    role_valid = jnp.any(mask, axis=-1)
    numerator = jnp.sum(
        jnp.where(role_valid, (1.0 - cosine) * role_weights[None, :], 0.0),
        axis=-1,
    )
    denominator = jnp.maximum(jnp.sum(role_valid, axis=-1), 1)
    return numerator / denominator


def sample_local_cross_view_role_contrastive_loss(
    cross_view_role_embeddings: jax.Array,
    role_valid: jax.Array,
    *,
    temperature: float = 0.1,
    role_weights: jax.Array | None = None,
) -> jax.Array:
    """Symmetric front/wrist InfoNCE over semantic role nodes.

    The positive for a role is the same role in the other camera; other valid
    roles from the same example are negatives.  This deliberately operates on
    role embeddings instead of slot-binding coordinates because independently
    inferred slot indices are not camera correspondences.
    """

    if (
        cross_view_role_embeddings.ndim != 4
        or cross_view_role_embeddings.shape[1:3] != (2, ROLE_COUNT)
    ):
        raise ValueError(
            "cross-view role embeddings must be [batch, 2, 6, hidden]"
        )
    batch = cross_view_role_embeddings.shape[0]
    if role_valid.shape != (batch, ROLE_COUNT):
        raise ValueError("cross-view role validity must be [batch, 6]")
    if not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError("cross-view temperature must be positive")
    if role_weights is None:
        role_weights = jnp.asarray(ROLE_VALID_CLASS_WEIGHTS, dtype=jnp.float32)
    elif role_weights.shape != (ROLE_COUNT,):
        raise ValueError("role weights must have shape [6]")

    embeddings = cross_view_role_embeddings.astype(jnp.float32)
    embeddings = embeddings / jnp.maximum(
        jnp.linalg.norm(embeddings, axis=-1, keepdims=True), 1.0e-6
    )
    left, right = embeddings[:, 0], embeddings[:, 1]
    logits = jnp.einsum("brh,bsh->brs", left, right) / float(temperature)
    valid = role_valid.astype(jnp.bool_)
    candidate_mask = valid[:, None, :]
    forward_log_prob = jax.nn.log_softmax(
        jnp.where(candidate_mask, logits, -1.0e30), axis=-1
    )
    reverse_log_prob = jax.nn.log_softmax(
        jnp.where(candidate_mask, jnp.swapaxes(logits, 1, 2), -1.0e30),
        axis=-1,
    )
    role_index = jnp.arange(ROLE_COUNT)
    symmetric_loss = -0.5 * (
        forward_log_prob[:, role_index, role_index]
        + reverse_log_prob[:, role_index, role_index]
    )
    weighted = symmetric_loss * role_weights[None, :]
    numerator = jnp.sum(jnp.where(valid, weighted, 0.0), axis=-1)
    denominator = jnp.maximum(jnp.sum(valid, axis=-1), 1)
    return numerator / denominator
