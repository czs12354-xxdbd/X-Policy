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

import functools
import logging
import math
import pathlib
from collections.abc import Mapping
from typing import NamedTuple
from typing_extensions import override

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models import model as _model
from openpi.models import hetm as _hetm
from openpi.models import racg as _racg
from openpi.models import racg_external_geometry as _racg_external
from openpi.models import racg_external_geometry_hmca as _racg_external_hmca
from openpi.models import racg_graph_hmca as _racg_graph_hmca
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from experiments.pi05 import l0s_geometry_aux_model_plumbing_v1 as _geometry_plumbing
from experiments.pi05 import l0s_geometry_aux_sdla_v3_nnx_v1 as _geometry_nnx
from experiments.pi05 import psm_hmca_v4_nnx as _hmca_nnx
from experiments.pi05 import psm_layerwise_memory_attention_v1_nnx as _memory_attention_nnx
from experiments.pi05 import clause_plan_adapter_nnx_v1 as _clause_plan_nnx
from experiments.pi05 import action_conditioned_temporal_role_memory_nnx_v1 as _temporal_role_nnx
from experiments.pi05 import cross_view_role_consensus_nnx_v1 as _cross_view_role_nnx
from experiments.pi05 import contact_risk_calibrated_role_residual_nnx_v1 as _contact_risk_nnx
from experiments.pi05 import relational_role_composer_residual_nnx_v1 as _relational_role_nnx
from experiments.pi05 import clause_role_binding_verifier_nnx_v1 as _clause_role_binding_nnx
from experiments.pi05 import semantic_frontier_completion_verifier_nnx_v1 as _semantic_frontier_nnx
from experiments.pi05 import hierarchical_clause_event_alignment_nnx_v1 as _hcea_nnx
from experiments.pi05 import hcea_causal_recovery_action_experts_candidate_v1 as _hcea_recovery
from experiments.pi05 import hcea_causal_recovery_action_experts_nnx_v1 as _hcea_recovery_nnx
from experiments.pi05 import hcea_causal_role_identity_transport_expert_candidate_v1 as _hcea_role_transport
from experiments.pi05 import hcea_causal_role_identity_transport_expert_nnx_v1 as _hcea_role_transport_nnx


logger = logging.getLogger('openpi')


class RACGPrefixView(NamedTuple):
    visual_states: jax.Array
    visual_mask: jax.Array
    language_states: jax.Array
    language_mask: jax.Array
    camera_names: tuple[str, ...]
    grid_size: int


def hetm_sequence_flow_replan_indices(replan_count: int) -> tuple[int, ...]:
    """Return the fixed 50%-density production flow schedule."""
    if replan_count == 4:
        return (0, 3)
    if replan_count == 8:
        return (0, 2, 4, 7)
    raise ValueError('HETM production sequences must contain 4 or 8 replans')


def _psm_hetm_frontier_consistency_loss(psm_frontier, hetm_frontier):
    """Align HETM to Direct's frontier without updating the PSM teacher."""
    if psm_frontier.shape != hetm_frontier.shape or psm_frontier.ndim != 2:
        raise ValueError('PSM/HETM frontier consistency shape drifted')
    teacher = jax.lax.stop_gradient(psm_frontier.astype(jnp.float32))
    student = hetm_frontier.astype(jnp.float32)
    return jnp.sum(jnp.square(student - teacher), axis=-1)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, ' b'],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, 'b {embedding_dim}']:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f'embedding_dim ({embedding_dim}) must be divisible by 2')

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        'i,j->ij',
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _rms_normalize(x):
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return jnp.asarray(x * jax.lax.rsqrt(variance + 1.0e-6), dtype=x.dtype)


def _action_moe_batch_global_balance_loss(router_logits, top_indices):
    """Balance soft importance and actual top-k load across the minibatch.

    The returned scalar is broadcast over the batch because the surrounding
    training loss is represented per sample. Averaging that loss therefore
    recovers this minibatch-level regularizer without changing its scale.
    """

    if router_logits.ndim != 3:
        raise ValueError('action MoE router logits must be [batch, tokens, experts]')
    if top_indices.ndim != 3 or top_indices.shape[:2] != router_logits.shape[:2]:
        raise ValueError('action MoE top-k indices must be [batch, tokens, k]')
    num_experts = router_logits.shape[-1]
    if num_experts < 2:
        raise ValueError('action MoE routing requires at least two experts')
    soft_routing = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    importance = jnp.mean(soft_routing, axis=(0, 1))
    uniform = 1.0 / num_experts
    soft_dispersion = num_experts * jnp.sum(
        jnp.square(importance - uniform)
    )
    hard_load = jnp.mean(
        jax.nn.one_hot(top_indices, num_experts, dtype=jnp.float32),
        axis=(0, 1, 2),
    )
    # The hard assignment supplies a stop-gradient load target while the soft
    # importance supplies router gradients.  This catches the subtle case in
    # which probabilities are nearly uniform but deterministic top-k tie
    # breaking still sends every token to the same experts. Subtracting the
    # balanced baseline and averaging with dispersion preserves the original
    # zero-to-(E-1) loss scale.
    switch_balance = num_experts * jnp.sum(
        jax.lax.stop_gradient(hard_load) * importance
    )
    balance = 0.5 * (soft_dispersion + switch_balance - 1.0)
    return jnp.broadcast_to(balance, (router_logits.shape[0],))


def _action_moe_router_features(action_hidden, task_context, enabled):
    """Build task-stable router features without action-noise leakage."""
    if not enabled:
        return action_hidden
    if action_hidden.ndim != 3 or task_context.ndim != 3:
        raise ValueError('action MoE router inputs must be rank-three tensors')
    if action_hidden.shape[0] != task_context.shape[0]:
        raise ValueError('action MoE router inputs must share the batch axis')
    if action_hidden.shape[-1] != task_context.shape[-1]:
        raise ValueError('action MoE router inputs must share the feature width')
    task_summary = jnp.mean(task_context, axis=-2, keepdims=True)
    return jnp.broadcast_to(task_summary, action_hidden.shape)


def _persistent_memory_policy_gain(value, *, bounded: bool):
    """Keep new recurrent policy injection stable without changing old models."""

    value = value.astype(jnp.float32)
    return jnp.tanh(value) if bounded else value


def _dense_embedding_lookup(embedding: nnx.Embed, indices: jax.Array) -> jax.Array:
    """Lookup with a dense, replica-deterministic embedding gradient.

    ``nnx.Embed`` delegates to ``jnp.take``.  Its transpose is a sparse scatter,
    which can leave bitwise-divergent optimizer moments on replicated devices
    when an index is repeated many times.  Orbax's replica-parallel writer may
    then serialize slices from different replicas.  A one-hot matrix product is
    mathematically identical for valid integer indices, but its transpose is a
    dense matrix product and therefore produces one coherent replicated state.
    These positional tables are only 16 x 256, so the dense form is bounded.
    """

    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError('embedding indices must be integers')
    table = embedding.embedding.value
    selector = jax.nn.one_hot(
        indices,
        embedding.num_embeddings,
        dtype=table.dtype,
    )
    return jnp.einsum(
        '...n,nf->...f',
        selector,
        table,
        precision=jax.lax.Precision.HIGHEST,
    )


def _l2_normalize(x):
    """Normalize embeddings to unit length for temperature-scaled similarity."""

    squared_norm = jnp.sum(
        jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True
    )
    return jnp.asarray(
        x * jax.lax.rsqrt(squared_norm + 1.0e-6), dtype=x.dtype
    )


def _ordered_role_pair_summary(role_tokens):
    """Summarize target/destination roles without becoming swap invariant."""

    if role_tokens.ndim != 3 or role_tokens.shape[1] != 2:
        raise ValueError('ordered role summary requires [batch, 2, hidden]')
    return (
        jnp.mean(role_tokens, axis=1)
        + 0.25 * (role_tokens[:, 0] - role_tokens[:, 1])
    )


@functools.cache
def _normal_initializer(stddev: float):
    """Return a graph-stable normal initializer for repeated NNX model builds."""
    return nnx.initializers.normal(stddev=stddev)


def _inverse_sqrt_class_weights(counts: np.ndarray, clip: float) -> np.ndarray:
    """Return capped weights whose empirical sample-weight mean is one."""
    counts = np.asarray(counts, dtype=np.float32)
    if counts.ndim != 1 or np.any(counts <= 0):
        raise ValueError('class counts must be a positive rank-one array')
    weights = 1.0 / np.sqrt(counts)
    weights = np.minimum(weights, np.min(weights) * clip)
    sample_mean = np.sum(counts * weights) / np.sum(counts)
    return weights / sample_mean


def _sample_beta_1p5_1(key, shape):
    """Sample Beta(1.5, 1) without Gamma rejection-loop collectives.

    For ``X ~ Beta(a, 1)``, the CDF is ``x**a``.  Therefore
    ``U**(1/a)`` for uniform ``U`` is exactly distributed as ``X``.  JAX's
    generic beta sampler lowers through two Gamma rejection loops whose
    sharded loop predicates introduce dozens of scalar boolean all-reduces.
    The inverse CDF is elementwise and keeps every data shard independent.
    """
    uniform = jax.random.uniform(key, shape)
    return jnp.power(uniform, 2.0 / 3.0)


def _grouped_role_identity_contrastive_loss(
    normalized_roles,
    identity_valid,
    identity_labels,
    *,
    max_group_size: int,
):
    """Contrast adjacent positive pairs without a cross-device all-gather.

    The production sampler emits consecutive same-task pairs.  A group of
    four therefore contains one positive companion and a second task pair as
    negatives.  Grouping before the similarity einsum lets the leading group
    axis shard over data-parallel devices, whereas a full-batch similarity
    matrix requires every device to all-gather the batch.

    Tiny CPU/preflight batches may contain only one pair; those retain the
    well-defined zero-loss behavior of the previous global objective because
    no negative identity is present.
    """
    if normalized_roles.ndim != 3:
        raise ValueError('normalized roles must be [batch, role, hidden]')
    batch_size, role_count, hidden_dim = normalized_roles.shape
    if identity_valid.shape != (batch_size, role_count):
        raise ValueError('identity validity must be [batch, role]')
    if identity_labels.shape != (batch_size, role_count):
        raise ValueError('identity labels must be [batch, role]')
    if batch_size < 2 or batch_size % 2:
        raise ValueError('role contrastive batches must contain complete pairs')
    if max_group_size < 4 or max_group_size % 2:
        raise ValueError('role contrastive max group size must be even and >= 4')

    group_size = min(max_group_size, batch_size)
    while group_size > 2 and batch_size % group_size:
        group_size -= 2
    group_count = batch_size // group_size
    grouped_roles = normalized_roles.reshape(
        group_count, group_size, role_count, hidden_dim
    )
    grouped_valid = identity_valid.reshape(group_count, group_size, role_count)
    grouped_labels = identity_labels.reshape(group_count, group_size, role_count)
    similarities = jnp.einsum(
        'bqrh,bcrh->bqrc',
        grouped_roles,
        grouped_roles,
        preferred_element_type=jnp.float32,
    ) / (float(hidden_dim) * 0.1)
    candidate_valid = jnp.transpose(grouped_valid, (0, 2, 1))[:, None, :, :]
    denominator_mask = grouped_valid[:, :, :, None] & candidate_valid
    denominator_mask = denominator_mask & ~jnp.eye(
        group_size, dtype=jnp.bool_
    )[None, :, None, :]
    candidate_labels = jnp.transpose(grouped_labels, (0, 2, 1))[:, None, :, :]
    positive_mask = denominator_mask & (
        grouped_labels[:, :, :, None] == candidate_labels
    )
    negative_large = jnp.asarray(-1.0e9, dtype=jnp.float32)
    denominator = jax.nn.logsumexp(
        jnp.where(denominator_mask, similarities, negative_large), axis=-1
    )
    positives = jax.nn.logsumexp(
        jnp.where(positive_mask, similarities, negative_large), axis=-1
    )
    eligible = jnp.any(positive_mask, axis=-1)
    per_role = jnp.where(eligible, denominator - positives, 0.0)
    per_example = jnp.sum(per_role, axis=-1) / jnp.maximum(
        jnp.sum(eligible, axis=-1), 1
    )
    return per_example.reshape(batch_size)


def _persistent_role_identity_contrastive_loss(
    normalized_roles,
    normalized_identity_anchors,
    identity_valid,
    identity_labels,
    *,
    max_group_size: int,
):
    """Supervise both current grounding and the slow identity-memory write.

    The current visual binding alone cannot prevent a distractor from
    overwriting the two reserved role-memory tokens at the next replan.  An
    equal-weight objective preserves the original total loss scale while
    making the persistent anchors carry the same supervised object identity.
    """
    if normalized_identity_anchors.shape != normalized_roles.shape:
        raise ValueError('identity anchors must match normalized roles')
    current_loss = _grouped_role_identity_contrastive_loss(
        normalized_roles,
        identity_valid,
        identity_labels,
        max_group_size=max_group_size,
    )
    persistent_loss = _grouped_role_identity_contrastive_loss(
        normalized_identity_anchors,
        identity_valid,
        identity_labels,
        max_group_size=max_group_size,
    )
    return 0.5 * (current_loss + persistent_loss)


def _masked_demo_router_weights(logits, mask, *, temperature: float):
    """Mask padded candidates and normalize active subgoal probabilities."""
    masked_logits = jnp.where(
        mask,
        logits,
        jnp.asarray(-1.0e30, dtype=logits.dtype),
    )
    weights = jax.nn.softmax(
        masked_logits.astype(jnp.float32) / temperature, axis=-1
    ).astype(logits.dtype)
    return masked_logits, weights


def _hard_demo_router_weights(masked_logits, soft_weights, *, train: bool):
    """Select one active subgoal while retaining soft-router gradients."""
    hard_weights = jax.nn.one_hot(
        jnp.argmax(masked_logits, axis=-1),
        masked_logits.shape[-1],
        dtype=soft_weights.dtype,
    )
    if train:
        return soft_weights + jax.lax.stop_gradient(hard_weights - soft_weights)
    return hard_weights


def _soft_progress_targets(progress, bins: int):
    """Linearly interpolate continuous progress over ordered 0..1 anchors."""
    if bins < 2:
        raise ValueError('progress supervision requires at least two anchors')
    progress = jnp.clip(progress.astype(jnp.float32), 0.0, 1.0)
    scaled = progress * (bins - 1)
    lower = jnp.floor(scaled).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, bins - 1)
    upper_weight = scaled - lower.astype(scaled.dtype)
    lower_weight = 1.0 - upper_weight
    return (
        jax.nn.one_hot(lower, bins, dtype=jnp.float32)
        * lower_weight[..., None]
        + jax.nn.one_hot(upper, bins, dtype=jnp.float32)
        * upper_weight[..., None]
    )


class _ExplicitActionReasonerBlock(nnx.Module):
    """Small self/cross-attention block for coarse action-space reasoning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.self_q = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.self_k = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.self_v = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.self_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_q = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_k = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_v = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mlp_in = nnx.Linear(hidden_dim, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_dim, hidden_dim, rngs=rngs)

    def _attention(self, query, key, value):
        head_dim = self.hidden_dim // self.num_heads
        query = query.reshape(*query.shape[:-1], self.num_heads, head_dim)
        key = key.reshape(*key.shape[:-1], self.num_heads, head_dim)
        value = value.reshape(*value.shape[:-1], self.num_heads, head_dim)
        logits = jnp.einsum(
            'bqhd,bkhd->bhqk',
            query,
            key,
            preferred_element_type=jnp.float32,
        )
        logits = logits / jnp.sqrt(float(head_dim))
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        attended = jnp.einsum('bhqk,bkhd->bqhd', weights, value)
        return attended.reshape(*attended.shape[:-2], self.hidden_dim)

    def __call__(self, action_tokens, context_tokens):
        normalized = _rms_normalize(action_tokens)
        self_attended = self._attention(
            self.self_q(normalized),
            self.self_k(normalized),
            self.self_v(normalized),
        )
        action_tokens = action_tokens + self.self_out(self_attended)

        normalized = _rms_normalize(action_tokens)
        normalized_context = _rms_normalize(context_tokens)
        cross_attended = self._attention(
            self.cross_q(normalized),
            self.cross_k(normalized_context),
            self.cross_v(normalized_context),
        )
        action_tokens = action_tokens + self.cross_out(cross_attended)

        normalized = _rms_normalize(action_tokens)
        return action_tokens + self.mlp_out(nnx.swish(self.mlp_in(normalized)))


class _SpatialRelationReasonerBlock(nnx.Module):
    """Self/cross-attention block whose visual-language context is mask aware."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.self_q = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.self_k = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.self_v = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.self_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_q = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_k = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_v = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.cross_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mlp_in = nnx.Linear(hidden_dim, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_dim, hidden_dim, rngs=rngs)

    def _attention(self, query, key, value, mask=None):
        head_dim = self.hidden_dim // self.num_heads
        query = query.reshape(*query.shape[:-1], self.num_heads, head_dim)
        key = key.reshape(*key.shape[:-1], self.num_heads, head_dim)
        value = value.reshape(*value.shape[:-1], self.num_heads, head_dim)
        logits = jnp.einsum(
            'bqhd,bkhd->bhqk',
            query,
            key,
            preferred_element_type=jnp.float32,
        )
        logits = logits / jnp.sqrt(float(head_dim))
        if mask is not None:
            logits = jnp.where(mask[:, None, None, :], logits, -1.0e30)
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        attended = jnp.einsum('bhqk,bkhd->bqhd', weights, value)
        return attended.reshape(*attended.shape[:-2], self.hidden_dim)

    def __call__(self, relation_tokens, context_tokens, context_mask):
        normalized = _rms_normalize(relation_tokens)
        attended = self._attention(
            self.self_q(normalized),
            self.self_k(normalized),
            self.self_v(normalized),
        )
        relation_tokens = relation_tokens + self.self_out(attended)

        normalized = _rms_normalize(relation_tokens)
        normalized_context = _rms_normalize(context_tokens)
        attended = self._attention(
            self.cross_q(normalized),
            self.cross_k(normalized_context),
            self.cross_v(normalized_context),
            context_mask,
        )
        relation_tokens = relation_tokens + self.cross_out(attended)
        normalized = _rms_normalize(relation_tokens)
        return relation_tokens + self.mlp_out(nnx.swish(self.mlp_in(normalized)))


class _BiasFreeLinear(nnx.Module):
    """Minimal NNX linear with one kernel leaf and no ``bias=None`` state."""

    def __init__(self, in_features, out_features, *, kernel_init, rngs):
        self.kernel = nnx.Param(
            kernel_init(rngs.params(), (in_features, out_features))
        )

    def __call__(self, inputs):
        return jnp.matmul(inputs, self.kernel.value)


class _ConditionalMemoryPolicyBridge(nnx.Module):
    """Direct causal memory/program residual for each action token/channel."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        policy_dim: int,
        memory_tokens: int,
        fast_tokens: int,
        subgoal_slots: int,
        action_tokens: int,
        bottleneck_dim: int,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.policy_dim = policy_dim
        self.memory_tokens = memory_tokens
        self.fast_tokens = fast_tokens
        self.subgoal_slots = subgoal_slots
        self.action_tokens = action_tokens
        self.bottleneck_dim = bottleneck_dim
        self.context_in = _BiasFreeLinear(
            5 * hidden_dim,
            bottleneck_dim,
            kernel_init=_normal_initializer((5 * hidden_dim) ** -0.5),
            rngs=rngs,
        )
        self.token_in = _BiasFreeLinear(
            hidden_dim,
            bottleneck_dim,
            kernel_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )
        # This is the sole function-preserving v3 boundary.  Unlike a gate
        # multiplied by the inherited read_out residual, it directly learns
        # new policy-channel directions.
        self.gate_out = _BiasFreeLinear(
            bottleneck_dim,
            policy_dim,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def _structured_memory_summary(self, memory):
        target_anchor = memory[:, self.fast_tokens]
        reference_anchor = memory[:, self.fast_tokens + 1]
        verification = memory[:, -1]
        return (
            jnp.mean(memory, axis=1)
            + jnp.asarray(0.25, memory.dtype)
            * (target_anchor - reference_anchor)
            + jnp.asarray(0.25, memory.dtype) * verification
        )

    def _next_frontier(self, frontier):
        result = jnp.zeros_like(frontier)
        result = result.at[:, 1:].set(frontier[:, :-1])
        return result.at[:, -1].add(frontier[:, -1])

    def _remaining_weights(self, frontier):
        phase_ids = jnp.arange(self.subgoal_slots, dtype=jnp.float32)
        distance = phase_ids[None, :] - phase_ids[:, None]
        kernel = jnp.where(
            distance >= 0.0,
            jnp.power(jnp.asarray(0.75, jnp.float32), distance),
            jnp.asarray(0.0, jnp.float32),
        )
        weights = jnp.einsum(
            'bs,sr->br', frontier.astype(jnp.float32), kernel
        )
        return weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True),
            jnp.asarray(1.0e-8, jnp.float32),
        )

    def _frontier_fourier_table(self):
        positions = jnp.arange(
            self.subgoal_slots, dtype=jnp.float32
        )[:, None]
        channels = jnp.arange(self.hidden_dim, dtype=jnp.int32)
        pairs = (channels // 2).astype(jnp.float32)
        exponent = -jnp.log(jnp.asarray(10_000.0, jnp.float32)) * (
            2.0 * pairs / float(max(self.hidden_dim, 2))
        )
        angles = positions * jnp.exp(exponent)[None]
        return jnp.where(
            (channels % 2)[None] == 0,
            jnp.sin(angles),
            jnp.cos(angles),
        )

    def _semantic_features(self, memory, ordered_program, frontier):
        batch_size = memory.shape[0]
        if memory.shape != (
            batch_size, self.memory_tokens, self.hidden_dim
        ):
            raise ValueError('conditional bridge memory shape drifted')
        if ordered_program.shape != (
            batch_size, self.subgoal_slots, self.hidden_dim
        ):
            raise ValueError('conditional bridge program shape drifted')
        if frontier.shape != (batch_size, self.subgoal_slots):
            raise ValueError('conditional bridge frontier shape drifted')
        finite = (
            jnp.all(jnp.isfinite(memory), axis=(1, 2))
            & jnp.all(jnp.isfinite(ordered_program), axis=(1, 2))
            & jnp.all(jnp.isfinite(frontier), axis=1)
        )
        enabled = (
            finite
            & jnp.all(frontier >= 0.0, axis=1)
            & (jnp.sum(frontier, axis=1) > 1.0e-8)
        )
        safe_memory = jnp.where(
            enabled[:, None, None], memory, jnp.zeros_like(memory)
        )
        safe_program = jnp.where(
            enabled[:, None, None],
            ordered_program,
            jnp.zeros_like(ordered_program),
        )
        safe_frontier = jnp.where(
            enabled[:, None], frontier, jnp.zeros_like(frontier)
        ).astype(jnp.float32)
        safe_frontier = safe_frontier / jnp.maximum(
            jnp.sum(safe_frontier, axis=-1, keepdims=True),
            jnp.asarray(1.0e-8, jnp.float32),
        )
        memory_feature = self._structured_memory_summary(safe_memory)
        current_program = jnp.einsum(
            'bs,bsh->bh',
            safe_frontier.astype(safe_program.dtype),
            safe_program,
        )
        following = self._next_frontier(safe_frontier)
        following_program = jnp.einsum(
            'bs,bsh->bh',
            following.astype(safe_program.dtype),
            safe_program,
        )
        remaining = self._remaining_weights(safe_frontier)
        remaining_program = jnp.einsum(
            'bs,bsh->bh',
            remaining.astype(safe_program.dtype),
            safe_program,
        )
        frontier_feature = jnp.einsum(
            'bs,sh->bh', safe_frontier, self._frontier_fourier_table()
        )
        features = jnp.concatenate(
            [
                _rms_normalize(value).astype(jnp.float32)
                for value in (
                    memory_feature,
                    current_program,
                    following_program,
                    remaining_program,
                    frontier_feature,
                )
            ],
            axis=-1,
        )
        return (
            jnp.where(enabled[:, None], features, jnp.zeros_like(features)),
            enabled,
        )

    def conditional_residual(
        self, parent_read_query, memory, ordered_program, frontier
    ):
        features, enabled = self._semantic_features(
            memory, ordered_program, frontier
        )
        expected_query = (
            features.shape[0], self.action_tokens, self.hidden_dim
        )
        if parent_read_query.shape != expected_query:
            raise ValueError('conditional bridge action-query shape drifted')
        enabled = enabled & jnp.all(
            jnp.isfinite(parent_read_query), axis=(1, 2)
        )
        safe_query = jnp.where(
            enabled[:, None, None],
            parent_read_query,
            jnp.zeros_like(parent_read_query),
        )
        context_latent = jnp.tanh(self.context_in(features))
        token_latent = jnp.tanh(
            self.token_in(_rms_normalize(safe_query).astype(jnp.float32))
        )
        joint = _rms_normalize(context_latent[:, None] + token_latent)
        residual = jnp.tanh(self.gate_out(joint))
        return (
            jnp.where(
                enabled[:, None, None], residual, jnp.zeros_like(residual)
            ),
            enabled,
        )

    def __call__(
        self,
        parent_output,
        parent_read_query,
        memory,
        *,
        ordered_program=None,
        frontier=None,
        policy_scale=1.0,
    ):
        if ordered_program is None or frontier is None:
            return parent_output
        if parent_output.shape != (
            memory.shape[0], self.action_tokens, self.policy_dim
        ):
            raise ValueError('conditional bridge parent-output shape drifted')
        residual, enabled = self.conditional_residual(
            parent_read_query, memory, ordered_program, frontier
        )
        candidate = parent_output + (
            jnp.asarray(policy_scale, dtype=jnp.float32).astype(
                parent_output.dtype
            )
            * residual.astype(parent_output.dtype)
        )
        return jnp.where(
            enabled[:, None, None], candidate, parent_output
        )


class _StructuredDemoModule(nnx.Module):
    """Shared 8+10+2 demonstration encoder and 8-to-20 PSM fusion."""

    def __init__(
        self,
        *,
        semantic_dim: int,
        hidden_dim: int,
        policy_dim: int,
        semantic_slots: int,
        plan_steps: int,
        plan_dim: int,
        action_steps: int,
        action_dim: int,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.policy_dim = policy_dim
        self.semantic_slots = semantic_slots
        self.plan_steps = plan_steps
        self.action_steps = action_steps
        self.semantic_in = _BiasFreeLinear(
            semantic_dim,
            hidden_dim,
            kernel_init=_normal_initializer(semantic_dim**-0.5),
            rngs=rngs,
        )
        self.semantic_type = nnx.Embed(
            semantic_slots,
            hidden_dim,
            embedding_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )
        self.plan_in = _BiasFreeLinear(
            plan_dim,
            hidden_dim,
            kernel_init=_normal_initializer(plan_dim**-0.5),
            rngs=rngs,
        )
        self.action_in = _BiasFreeLinear(
            action_dim,
            hidden_dim,
            kernel_init=_normal_initializer(action_dim**-0.5),
            rngs=rngs,
        )
        self.plan_position = nnx.Embed(
            plan_steps,
            hidden_dim,
            embedding_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )
        self.action_position = nnx.Embed(
            2,
            hidden_dim,
            embedding_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )
        vector_init = _normal_initializer(hidden_dim**-0.5)
        self.plan_type = nnx.Param(vector_init(rngs.params(), (hidden_dim,)))
        self.action_type = nnx.Param(vector_init(rngs.params(), (hidden_dim,)))
        linear_kwargs = {
            'kernel_init': _normal_initializer(hidden_dim**-0.5),
            'rngs': rngs,
        }
        self.fusion_q = _BiasFreeLinear(hidden_dim, hidden_dim, **linear_kwargs)
        self.fusion_k = _BiasFreeLinear(hidden_dim, hidden_dim, **linear_kwargs)
        self.fusion_v = _BiasFreeLinear(hidden_dim, hidden_dim, **linear_kwargs)
        # W_demo_out is intentionally open.  Exact inheritance is controlled
        # by the sole newly-zero conditional projection below.  A scalar gate
        # forces every task, plan phase, and policy channel to agree on one
        # update direction; the live PSM trajectory shows that such a scalar
        # remains effectively closed.  One bias-free matrix retains a single
        # zero leaf while learning independent plan/action channel gates from
        # each row's masked demonstration context.
        self.W_demo_out = _BiasFreeLinear(
            hidden_dim, hidden_dim, **linear_kwargs
        )
        self.g_demo = nnx.Param(
            jnp.zeros((hidden_dim, hidden_dim + policy_dim), dtype=jnp.float32)
        )

    def conditional_gates(self, demo_tokens, demo_token_mask):
        """Return separate per-row plan and action-channel gates.

        The masked mean is invariant to padding and contains no evaluator or
        benchmark metadata.  Both outputs are exact zero at initialization,
        so the parent plan and action functions remain bitwise unchanged.  A
        single matrix leaf avoids the cross-task and plan/action sign
        cancellation imposed by one global scalar while retaining one audited
        function-preserving boundary.
        """
        if demo_tokens.ndim != 3 or demo_tokens.shape[-1] != self.hidden_dim:
            raise ValueError('demo tokens must be [batch, token, hidden]')
        if demo_token_mask.shape != demo_tokens.shape[:2]:
            raise ValueError('demo token mask differs from demo tokens')
        valid = demo_token_mask.astype(jnp.bool_)
        safe_tokens = jnp.where(
            valid[..., None], demo_tokens, jnp.zeros_like(demo_tokens)
        )
        counts = jnp.sum(valid, axis=-1, keepdims=True)
        pooled = jnp.sum(safe_tokens, axis=1) / jnp.maximum(counts, 1)
        pooled = _rms_normalize(pooled).astype(self.g_demo.value.dtype)
        logits = jnp.matmul(pooled, self.g_demo.value)
        plan_logits = logits[:, : self.hidden_dim]
        action_logits = logits[:, self.hidden_dim :]
        context_enabled = jnp.any(valid, axis=-1)
        plan_gate = jnp.where(
            context_enabled[:, None],
            jnp.tanh(plan_logits),
            jnp.zeros_like(plan_logits),
        )
        action_gate = jnp.where(
            context_enabled[:, None],
            jnp.tanh(action_logits),
            jnp.zeros_like(action_logits),
        )
        return plan_gate, action_gate

    def pool_semantics(
        self,
        frozen_prompt_embeddings,
        prompt_mask,
        semantic_span_mask,
        semantic_valid_mask,
    ):
        if frozen_prompt_embeddings.ndim != 3:
            raise ValueError('demo prompt embeddings must be [batch, token, 2048]')
        expected = (
            frozen_prompt_embeddings.shape[0],
            self.semantic_slots,
            frozen_prompt_embeddings.shape[1],
        )
        if semantic_span_mask.shape != expected:
            raise ValueError('demo semantic spans must be [batch, 8, token]')
        if semantic_valid_mask.shape != expected[:2]:
            raise ValueError('demo semantic validity must be [batch, 8]')
        if prompt_mask.shape != frozen_prompt_embeddings.shape[:2]:
            raise ValueError('demo prompt mask differs from embedded prompt')
        effective = (
            semantic_span_mask.astype(jnp.bool_)
            & semantic_valid_mask[..., None].astype(jnp.bool_)
            & prompt_mask[:, None].astype(jnp.bool_)
        )
        values = jnp.where(
            effective[..., None],
            jax.lax.stop_gradient(frozen_prompt_embeddings)[:, None],
            jnp.zeros((), dtype=frozen_prompt_embeddings.dtype),
        )
        counts = jnp.sum(effective, axis=-1, keepdims=True)
        pooled = jnp.sum(values, axis=2) / jnp.maximum(counts, 1)
        return jnp.where(
            semantic_valid_mask[..., None], pooled, jnp.zeros_like(pooled)
        )

    def encode(
        self,
        semantics,
        semantic_mask,
        plan,
        actions,
        context_mask,
        trajectory_mask,
    ):
        batch_size = semantics.shape[0]
        if semantics.shape[:2] != (batch_size, self.semantic_slots):
            raise ValueError('structured demo requires eight semantic slots')
        if semantic_mask.shape != semantics.shape[:2]:
            raise ValueError('structured demo semantic mask differs')
        if plan.shape[:2] != (batch_size, self.plan_steps):
            raise ValueError('structured demo requires ten plan keyframes')
        if actions.shape[:2] != (batch_size, self.action_steps):
            raise ValueError('structured demo requires ten demonstrated actions')
        if context_mask.shape != (batch_size,) or trajectory_mask.shape != (
            batch_size,
        ):
            raise ValueError('structured demo masks must be [batch]')
        semantic_tokens = jnp.tanh(
            self.semantic_in(semantics)
            + self.semantic_type(jnp.arange(self.semantic_slots))[None]
        )
        plan_tokens = jnp.tanh(
            self.plan_in(plan)
            + self.plan_position(jnp.arange(self.plan_steps))[None]
            + self.plan_type.value[None, None]
        )
        midpoint = (self.action_steps + 1) // 2
        summaries = jnp.stack(
            [
                jnp.mean(actions[:, :midpoint], axis=1),
                jnp.mean(actions[:, midpoint:], axis=1),
            ],
            axis=1,
        )
        action_tokens = jnp.tanh(
            self.action_in(summaries)
            + self.action_position(jnp.arange(2))[None]
            + self.action_type.value[None, None]
        )
        semantic_enabled = semantic_mask.astype(jnp.bool_) & context_mask[:, None]
        trajectory_enabled = (context_mask & trajectory_mask)[:, None]
        token_mask = jnp.concatenate(
            [
                semantic_enabled,
                jnp.broadcast_to(trajectory_enabled, plan_tokens.shape[:2]),
                jnp.broadcast_to(trajectory_enabled, action_tokens.shape[:2]),
            ],
            axis=1,
        )
        tokens = jnp.concatenate(
            [semantic_tokens, plan_tokens, action_tokens], axis=1
        )
        return (
            jnp.where(token_mask[..., None], tokens, jnp.zeros_like(tokens)),
            token_mask,
        )

    def fuse(self, parent_subgoals, demo_tokens, demo_token_mask):
        query = self.fusion_q(parent_subgoals)
        key = self.fusion_k(demo_tokens)
        value = self.fusion_v(demo_tokens)
        logits = jnp.einsum(
            'bqh,bkh->bqk', query, key, preferred_element_type=jnp.float32
        ) / math.sqrt(float(self.hidden_dim))
        logits = jnp.where(demo_token_mask[:, None], logits, -1.0e30)
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        weights = weights * demo_token_mask[:, None].astype(weights.dtype)
        weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8
        )
        demo_read = jnp.einsum(
            'bqk,bkh->bqh', weights.astype(value.dtype), value
        )
        context_enabled = jnp.any(demo_token_mask, axis=-1)
        demo_read = jnp.where(
            context_enabled[:, None, None], demo_read, jnp.zeros_like(demo_read)
        )
        shared_context = jnp.tanh(parent_subgoals + demo_read)
        plan_gate, _ = self.conditional_gates(demo_tokens, demo_token_mask)
        plan_residual = self.W_demo_out(demo_read)
        candidate = parent_subgoals + plan_gate[:, None].astype(
            plan_residual.dtype
        ) * plan_residual
        # Explicit select, rather than an arithmetic zero, guarantees exact
        # parent bytes for dropped rows after g_demo has opened.
        deployed = jnp.where(
            context_enabled[:, None, None], candidate, parent_subgoals
        )
        return deployed, shared_context

    def inject_actions(
        self,
        parent_action_tokens,
        demo_tokens,
        demo_token_mask,
        persistent_memory,
    ):
        """Direct demo-to-action residual under the same sole ``g_demo``.

        The inherited PSM read projections provide the policy/hidden adapters,
        while the structured module supplies its already-open q/k/v/output
        projections.  This path intentionally does not read
        the parent's small learned memory gate and therefore cannot be
        attenuated by it.
        """
        action_query = self.fusion_q(
            persistent_memory.read_query(
                _rms_normalize(parent_action_tokens)
            )
        )
        key = self.fusion_k(demo_tokens)
        value = self.fusion_v(demo_tokens)
        logits = jnp.einsum(
            'bqh,bkh->bqk',
            action_query,
            key,
            preferred_element_type=jnp.float32,
        ) / math.sqrt(float(self.hidden_dim))
        logits = jnp.where(demo_token_mask[:, None], logits, -1.0e30)
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        weights = weights * demo_token_mask[:, None].astype(weights.dtype)
        weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8
        )
        demo_read = jnp.einsum(
            'bqk,bkh->bqh', weights.astype(value.dtype), value
        )
        context_enabled = jnp.any(demo_token_mask, axis=-1)
        demo_read = jnp.where(
            context_enabled[:, None, None],
            demo_read,
            jnp.zeros_like(demo_read),
        )
        residual = persistent_memory.read_out(
            _rms_normalize(self.W_demo_out(demo_read))
        )
        _, action_gate = self.conditional_gates(demo_tokens, demo_token_mask)
        candidate = parent_action_tokens + action_gate[:, None].astype(
            residual.dtype
        ) * residual
        return jnp.where(
            context_enabled[:, None, None],
            candidate,
            parent_action_tokens,
        )


class _SpatialLanguageDecoderBlock(nnx.Module):
    """One bias-free causal self/cross-attention decoder block."""

    def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        kwargs = {
            'kernel_init': _normal_initializer(hidden_dim**-0.5),
            'rngs': rngs,
        }
        self.self_q = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.self_k = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.self_v = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.self_out = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.cross_q = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.cross_k = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.cross_v = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.cross_out = _BiasFreeLinear(hidden_dim, hidden_dim, **kwargs)
        self.mlp_in = _BiasFreeLinear(hidden_dim, 4 * hidden_dim, **kwargs)
        self.mlp_out = _BiasFreeLinear(
            4 * hidden_dim,
            hidden_dim,
            kernel_init=_normal_initializer((4 * hidden_dim) ** -0.5),
            rngs=rngs,
        )

    def _attention(self, query, key, value, allowed):
        logits = jnp.einsum(
            'bqh,bkh->bqk', query, key, preferred_element_type=jnp.float32
        ) / math.sqrt(float(self.hidden_dim))
        logits = jnp.where(allowed, logits, -1.0e30)
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        weights = weights * allowed.astype(weights.dtype)
        weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8
        )
        return jnp.einsum('bqk,bkh->bqh', weights.astype(value.dtype), value)

    def __call__(self, hidden, input_mask, context, context_mask):
        length = hidden.shape[1]
        causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None]
        self_allowed = causal[None] & input_mask[:, None]
        self_read = self._attention(
            self.self_q(hidden),
            self.self_k(hidden),
            self.self_v(hidden),
            self_allowed,
        )
        hidden = hidden + self.self_out(self_read)
        cross_read = self._attention(
            self.cross_q(hidden),
            self.cross_k(context),
            self.cross_v(context),
            context_mask[:, None],
        )
        hidden = hidden + self.cross_out(cross_read)
        hidden = hidden + self.mlp_out(nnx.swish(self.mlp_in(hidden)))
        return jnp.where(input_mask[..., None], hidden, jnp.zeros_like(hidden))


class _SpatialLanguageAux(nnx.Module):
    """Exactly two 256-wide training-only decoder blocks."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        vocabulary_size: int,
        language_steps: int,
        bos_token_id: int,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size
        self.language_steps = language_steps
        self.bos_token_id = bos_token_id
        self.token_embedding = nnx.Embed(
            vocabulary_size,
            hidden_dim,
            embedding_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )
        self.position_embedding = nnx.Embed(
            language_steps,
            hidden_dim,
            embedding_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )
        self.blocks = [
            _SpatialLanguageDecoderBlock(hidden_dim, rngs=rngs)
            for _ in range(2)
        ]
        self.logits = _BiasFreeLinear(
            hidden_dim,
            vocabulary_size,
            kernel_init=_normal_initializer(hidden_dim**-0.5),
            rngs=rngs,
        )

    def loss(
        self,
        target_ids,
        target_mask,
        shared_context,
        phase_evidence,
        demo_tokens,
        demo_token_mask,
    ):
        if target_ids.shape[1:] != (self.language_steps,):
            raise ValueError('spatial-language targets must be [batch, 32]')
        if target_mask.shape != target_ids.shape:
            raise ValueError('spatial-language target mask differs from ids')
        safe_ids = jnp.clip(target_ids, 0, self.vocabulary_size - 1)
        bos = jnp.full(
            (target_ids.shape[0], 1), self.bos_token_id, dtype=target_ids.dtype
        )
        decoder_ids = jnp.concatenate([bos, safe_ids[:, :-1]], axis=1)
        input_mask = jnp.concatenate(
            [
                jnp.ones((target_ids.shape[0], 1), dtype=jnp.bool_),
                target_mask[:, :-1].astype(jnp.bool_),
            ],
            axis=1,
        )
        hidden = self.token_embedding(decoder_ids) + self.position_embedding(
            jnp.arange(self.language_steps)
        )[None]
        hidden = jnp.where(input_mask[..., None], hidden, jnp.zeros_like(hidden))
        context = jnp.concatenate(
            [shared_context, phase_evidence[:, None], demo_tokens], axis=1
        )
        context_mask = jnp.concatenate(
            [
                jnp.ones(shared_context.shape[:2], dtype=jnp.bool_),
                jnp.ones((shared_context.shape[0], 1), dtype=jnp.bool_),
                demo_token_mask,
            ],
            axis=1,
        )
        for block in self.blocks:
            hidden = block(hidden, input_mask, context, context_mask)
        logits = self.logits(hidden)
        selected = jnp.take_along_axis(
            jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
            safe_ids[..., None],
            axis=-1,
        )[..., 0]
        weights = target_mask.astype(jnp.float32)
        return -jnp.sum(selected * weights, axis=-1) / jnp.maximum(
            jnp.sum(weights, axis=-1), 1.0
        )


class PersistentSubgoalMemory(nnx.Module):
    """Causal fast/slow memory with ordered subgoal and role binding."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        policy_dim: int,
        state_dim: int,
        action_dim: int,
        previous_action_dim: int,
        memory_tokens: int,
        hidden_dim: int,
        subgoal_slots: int,
        fast_tokens: int,
        fast_update_rate: float,
        slow_update_rate: float,
        bounded_policy_gain: bool = False,
        compositional_phase_init_scale: float = 0.01,
        rngs: nnx.Rngs,
    ):
        self.memory_tokens = memory_tokens
        self.hidden_dim = hidden_dim
        self.subgoal_slots = subgoal_slots
        self.fast_tokens = fast_tokens
        self.bounded_policy_gain = bounded_policy_gain
        self.prefix_in = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.state_in = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.subgoal_queries = nnx.Embed(subgoal_slots, hidden_dim, rngs=rngs)
        # Give every routed slot a shared physical phase identity, then allow
        # the four language-predicted operations to modulate that phase.  This
        # factorization supports unseen operation/workflow combinations instead
        # of asking eight free attention queries to rediscover phase semantics.
        self.semantic_phase_embeddings = nnx.Embed(
            subgoal_slots, hidden_dim, rngs=rngs
        )
        self.operation_phase_embeddings = nnx.Embed(
            4 * subgoal_slots, hidden_dim, rngs=rngs
        )
        # Compose the independently predicted source/destination/condition
        # semantics with physical phases through a low-rank adapter.  This is
        # deliberately separate from the operation bank above: an unseen
        # preposition/workflow can reuse the same approach/contact/transport
        # phases without allocating a full relation-by-phase tensor.
        relation_phase_rank = min(64, hidden_dim)
        self.relation_phase_down = nnx.Linear(
            hidden_dim,
            relation_phase_rank,
            rngs=rngs,
        )
        self.relation_phase_slots = nnx.Embed(
            subgoal_slots, relation_phase_rank, rngs=rngs
        )
        self.relation_phase_up = nnx.Linear(
            relation_phase_rank,
            hidden_dim,
            rngs=rngs,
        )
        # The factor heads below are independently supervised, but an average
        # of their prototype states cannot represent interactions such as
        # ``push X between Y and Z while closed``.  Keep five typed factor
        # tokens (operation, source relation, destination relation, condition,
        # destination qualifier), let them exchange evidence, and let every
        # physical phase query the resulting program graph.  The final
        # projection starts small but nonzero.  Exact inherited-policy
        # preservation is supplied once, at memory_to_policy_gain; opening this
        # internal projection avoids a redundant second zero gate that would
        # initially block all route/flow gradients to the upstream factor
        # graph.
        compositional_heads = min(4, hidden_dim)
        while hidden_dim % compositional_heads:
            compositional_heads -= 1
        self.compositional_factor_identity = nnx.Embed(
            5, hidden_dim, rngs=rngs
        )
        self.compositional_factor_blocks = [
            _ExplicitActionReasonerBlock(
                hidden_dim,
                compositional_heads,
                4 * hidden_dim,
                rngs=rngs,
            )
            for _ in range(2)
        ]
        self.compositional_phase_queries = nnx.Embed(
            subgoal_slots, hidden_dim, rngs=rngs
        )
        self.compositional_phase_blocks = [
            _ExplicitActionReasonerBlock(
                hidden_dim,
                compositional_heads,
                4 * hidden_dim,
                rngs=rngs,
            )
            for _ in range(2)
        ]
        self.compositional_phase_out = nnx.Linear(
            hidden_dim,
            hidden_dim,
            kernel_init=_normal_initializer(compositional_phase_init_scale),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.subgoal_key = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.subgoal_value = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.role_queries = nnx.Embed(2, hidden_dim, rngs=rngs)
        self.source_reference_query = nnx.Embed(1, hidden_dim, rngs=rngs)
        self.destination_reference_queries = nnx.Embed(2, hidden_dim, rngs=rngs)
        self.condition_state_query = nnx.Embed(1, hidden_dim, rngs=rngs)
        self.destination_qualifier_query = nnx.Embed(1, hidden_dim, rngs=rngs)
        self.role_key = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.role_value = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        # Build open-vocabulary object slots independently in each camera,
        # then bind the two prompt roles to those slots before memory writes.
        self.object_count = memory_tokens
        self.visual_key = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.visual_value = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.object_queries = nnx.Embed(self.object_count, hidden_dim, rngs=rngs)
        self.object_query = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.object_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.grounded_role_query = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.object_key = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.bound_role_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.role_geometry_in = nnx.Linear(4, hidden_dim, rngs=rngs)
        self.source_context_geometry_in = nnx.Linear(4, hidden_dim, rngs=rngs)
        self.destination_reference_geometry_in = nnx.Linear(
            6, hidden_dim, rngs=rngs
        )
        self.condition_state_geometry_in = nnx.Linear(
            2, hidden_dim, rngs=rngs
        )
        self.operation_head = nnx.Linear(hidden_dim, 4, rngs=rngs)
        self.source_relation_head = nnx.Linear(hidden_dim, 4, rngs=rngs)
        self.destination_relation_head = nnx.Linear(hidden_dim, 4, rngs=rngs)
        self.condition_head = nnx.Linear(hidden_dim, 3, rngs=rngs)
        self.destination_qualifier_head = nnx.Linear(
            hidden_dim, 8, rngs=rngs
        )
        self.subgoal_in = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.role_in = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.previous_action_dim = previous_action_dim
        self.executed_action_in = nnx.Linear(
            previous_action_dim, hidden_dim, rngs=rngs
        )
        # A mean action erases the ordered evidence that separates approach,
        # contact, lift, transport, and release.  Preserve four causal
        # statistics of the already executed replan chunk for phase routing.
        self.executed_action_phase_in = nnx.Linear(
            4 * previous_action_dim, hidden_dim, rngs=rngs
        )
        self.memory_position = nnx.Embed(memory_tokens, hidden_dim, rngs=rngs)
        rates = jnp.concatenate(
            [
                jnp.full((fast_tokens,), fast_update_rate, dtype=jnp.float32),
                jnp.full(
                    (memory_tokens - fast_tokens,),
                    slow_update_rate,
                    dtype=jnp.float32,
                ),
            ]
        )
        self.memory_update_rate_logits = nnx.Param(
            jnp.log(rates / (1.0 - rates))
        )
        self.write_gate = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.write_candidate = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.route_head = nnx.Linear(hidden_dim, subgoal_slots, rngs=rngs)
        self.transition_head = nnx.Linear(hidden_dim, subgoal_slots, rngs=rngs)
        self.progress_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.next_action_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        # Close the deployed plan-verification loop.  These predictions are
        # available causally at the current replan, but previously only served
        # auxiliary losses and were discarded before the next replan.  Fold
        # them into a dedicated slow memory token so both the current policy
        # read and the following frontier decision can use verified plan state.
        self.verification_in = nnx.Linear(
            subgoal_slots + 1 + action_dim, hidden_dim, rngs=rngs
        )
        self.read_query = nnx.Linear(policy_dim, hidden_dim, rngs=rngs)
        self.read_key = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.read_value = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.read_out = nnx.Linear(hidden_dim, policy_dim, rngs=rngs)
        self.memory_to_policy_gain = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def initial_state(self, batch_size: int, *, dtype=jnp.float32):
        return jnp.zeros(
            (batch_size, self.memory_tokens, self.hidden_dim), dtype=dtype
        )

    def initial_frontier(self, batch_size: int):
        return jax.nn.one_hot(
            jnp.zeros((batch_size,), dtype=jnp.int32),
            self.subgoal_slots,
            dtype=jnp.float32,
        )

    def _attention_weights(self, queries, keys, mask):
        logits = jnp.einsum(
            'bqh,bkh->bqk', queries, keys, preferred_element_type=jnp.float32
        ) / jnp.sqrt(float(self.hidden_dim))
        if mask.ndim == 2:
            mask = mask[:, None, :]
        elif mask.ndim != 3:
            raise ValueError('attention mask must be [batch, key] or [batch, query, key]')
        logits = jnp.where(mask, logits, -1.0e30)
        return jax.nn.softmax(logits, axis=-1)

    def _attend(self, queries, keys, values, mask):
        weights = self._attention_weights(queries, keys, mask).astype(values.dtype)
        return jnp.einsum('bqk,bkh->bqh', weights, values)

    def factor_specific_language_attention(
        self,
        factor_queries,
        keys,
        values,
        plan_token_mask,
        factor_span_masks,
        *,
        span_radius: int = 2,
    ):
        """Keep full context while grounding each factor in its local phrase.

        The five factor queries still read the complete instruction, which is
        necessary for compositional relations.  A bounded span residual makes
        the relevant phrase harder to ignore, and the returned loss directly
        trains the full-context attention to retain probability mass near that
        phrase.  Missing optional factors fall back to full-context attention
        and contribute exact zero auxiliary loss.
        """
        if factor_queries.ndim != 3 or factor_queries.shape[1] != 5:
            raise ValueError('factor queries must be [batch, 5, hidden]')
        if plan_token_mask.ndim != 2:
            raise ValueError('plan token mask must be [batch, key]')
        if factor_span_masks.shape != (
            factor_queries.shape[0],
            5,
            plan_token_mask.shape[1],
        ):
            raise ValueError('factor span masks must be [batch, 5, key]')
        if span_radius < 0:
            raise ValueError('factor span radius must be nonnegative')

        factor_span_masks = factor_span_masks.astype(jnp.bool_)
        expanded_masks = factor_span_masks
        for offset in range(1, span_radius + 1):
            expanded_masks = expanded_masks | jnp.pad(
                factor_span_masks[..., :-offset],
                ((0, 0), (0, 0), (offset, 0)),
            )
            expanded_masks = expanded_masks | jnp.pad(
                factor_span_masks[..., offset:],
                ((0, 0), (0, 0), (0, offset)),
            )
        expanded_masks = expanded_masks & plan_token_mask[:, None, :]
        factor_valid = jnp.any(expanded_masks, axis=-1)
        safe_span_masks = jnp.where(
            factor_valid[..., None],
            expanded_masks,
            plan_token_mask[:, None, :],
        )

        full_weights = self._attention_weights(
            factor_queries, keys, plan_token_mask
        )
        full_states = jnp.einsum(
            'bqk,bkh->bqh', full_weights.astype(values.dtype), values
        )
        span_states = self._attend(
            factor_queries, keys, values, safe_span_masks
        )
        factor_states = 0.75 * full_states + 0.25 * span_states
        relevant_mass = jnp.sum(
            full_weights * expanded_masks.astype(full_weights.dtype), axis=-1
        )
        alignment = -jnp.log(jnp.maximum(relevant_mass, 1.0e-6))
        alignment = jnp.where(
            factor_valid, alignment, jnp.zeros_like(alignment)
        )
        alignment = jnp.sum(alignment, axis=-1) / jnp.maximum(
            jnp.sum(factor_valid, axis=-1), 1
        )
        return factor_states, alignment, factor_valid

    def _language_position_encoding(self, plan_token_mask, *, dtype):
        """Encode language-relative order without depending on image-token count."""

        language_positions = jnp.maximum(
            jnp.cumsum(plan_token_mask.astype(jnp.int32), axis=-1) - 1,
            0,
        )
        dimensions = jnp.arange(self.hidden_dim, dtype=jnp.float32)
        frequencies = jnp.exp(
            -jnp.log(10_000.0)
            * (2.0 * jnp.floor(dimensions / 2.0) / float(self.hidden_dim))
        )
        angles = language_positions[..., None].astype(jnp.float32) * frequencies
        encoding = jnp.where(
            (jnp.arange(self.hidden_dim) % 2) == 0,
            jnp.sin(angles),
            jnp.cos(angles),
        ).astype(dtype)
        return encoding * plan_token_mask[..., None].astype(dtype)

    def competitive_object_patch_weights(self, object_logits):
        """Allocate each visual patch across slots before slot pooling.

        Independent per-slot softmax lets every slot collapse onto the same
        salient object.  Competition across slots makes each patch carry unit
        assignment mass, followed by per-slot normalization for stable visual
        pooling.
        """

        assignments = jax.nn.softmax(
            object_logits.astype(jnp.float32), axis=-2
        ).astype(object_logits.dtype)
        patch_weights = assignments / jnp.maximum(
            jnp.sum(assignments, axis=-1, keepdims=True),
            jnp.asarray(1.0e-6, dtype=assignments.dtype),
        )
        return patch_weights, assignments

    def ordered_role_object_probabilities(self, role_logits, role_valid_mask):
        """Bind target first, then exclude its mass from the reference role.

        The manipulated target and destination/reference are distinct whenever
        both production role spans are valid.  Independent softmaxes can still
        collapse both roles onto the same salient object slot.  Preserve the
        target's language-grounded distribution, then condition the reference
        distribution on the target *not* occupying each slot.  A stop-gradient
        target claim prevents reference losses from making the manipulated
        object diffuse, while the returned independent overlap directly trains
        both role queries to become separable before this deployed refinement.
        """

        if role_logits.ndim != 4 or role_logits.shape[-2] != 2:
            raise ValueError('role-object logits must be [batch, camera, 2, object]')
        if role_valid_mask.shape != (role_logits.shape[0], 2):
            raise ValueError('role validity mask differs from role-object logits')
        independent = jax.nn.softmax(role_logits.astype(jnp.float32), axis=-1)
        target_claim = jax.lax.stop_gradient(independent[:, :, 0])
        reference_logits = role_logits[:, :, 1].astype(jnp.float32) + jnp.log(
            jnp.maximum(1.0 - target_claim, 1.0e-6)
        )
        conditioned_reference = jax.nn.softmax(reference_logits, axis=-1)
        ordered = jnp.stack(
            [independent[:, :, 0], conditioned_reference], axis=-2
        )
        both_roles_valid = jnp.all(role_valid_mask, axis=-1)
        probabilities = jnp.where(
            both_roles_valid[:, None, None, None], ordered, independent
        )
        independent_overlap = jnp.sum(
            independent[:, :, 0] * independent[:, :, 1], axis=-1
        )
        independent_overlap = independent_overlap * both_roles_valid[:, None].astype(
            independent_overlap.dtype
        )
        return probabilities.astype(role_logits.dtype), independent_overlap

    def ordered_destination_reference_probabilities(
        self, target_logits, reference_logits, reference_valid_mask
    ):
        """Bind up to two spatial references without target/peer collapse.

        Destination phrases such as ``between the vase and the teapot`` need
        two independently grounded open-vocabulary objects.  Claim the
        manipulated target first, then bind reference 0 outside that claim and
        reference 1 outside both earlier claims.  Independent distributions
        remain the auxiliary exclusivity teachers so the deployed conditioning
        cannot hide a collapsed query.
        """

        if target_logits.ndim != 3:
            raise ValueError(
                'target object logits must be [batch, camera, object]'
            )
        if (
            reference_logits.ndim != 4
            or reference_logits.shape[:2] != target_logits.shape[:2]
            or reference_logits.shape[-2] != 2
            or reference_logits.shape[-1] != target_logits.shape[-1]
        ):
            raise ValueError(
                'destination-reference logits must be [batch, camera, 2, object]'
            )
        if reference_valid_mask.shape != (target_logits.shape[0], 2):
            raise ValueError(
                'destination-reference validity mask differs from logits'
            )

        target_probability = jax.nn.softmax(
            target_logits.astype(jnp.float32), axis=-1
        )
        independent_references = jax.nn.softmax(
            reference_logits.astype(jnp.float32), axis=-1
        )
        target_exclusion = jnp.maximum(
            1.0 - jax.lax.stop_gradient(target_probability), 1.0e-6
        )
        first_logits = reference_logits[:, :, 0].astype(jnp.float32) + jnp.log(
            target_exclusion
        )
        first_probability = jax.nn.softmax(first_logits, axis=-1)
        first_exclusion = jnp.maximum(
            1.0 - jax.lax.stop_gradient(first_probability), 1.0e-6
        )
        first_exclusion = jnp.where(
            reference_valid_mask[:, None, 0, None], first_exclusion, 1.0
        )
        second_logits = (
            reference_logits[:, :, 1].astype(jnp.float32)
            + jnp.log(target_exclusion)
            + jnp.log(first_exclusion)
        )
        second_probability = jax.nn.softmax(second_logits, axis=-1)
        ordered_references = jnp.stack(
            [first_probability, second_probability], axis=-2
        )
        ordered_references = jnp.where(
            reference_valid_mask[:, None, :, None],
            ordered_references,
            jnp.zeros_like(ordered_references),
        )

        target_reference_overlap = jnp.sum(
            target_probability[:, :, None] * independent_references, axis=-1
        )
        target_reference_overlap = target_reference_overlap * (
            reference_valid_mask[:, None].astype(
                target_reference_overlap.dtype
            )
        )
        both_references_valid = jnp.all(reference_valid_mask, axis=-1)
        reference_pair_overlap = jnp.sum(
            independent_references[:, :, 0]
            * independent_references[:, :, 1],
            axis=-1,
        ) * both_references_valid[:, None].astype(jnp.float32)
        overlap_count = (
            jnp.sum(reference_valid_mask, axis=-1).astype(jnp.float32)
            + both_references_valid.astype(jnp.float32)
        )
        independent_overlap = (
            jnp.sum(target_reference_overlap, axis=-1)
            + reference_pair_overlap
        ) / jnp.maximum(overlap_count[:, None], 1.0)
        return (
            ordered_references.astype(reference_logits.dtype),
            independent_overlap,
        )

    def relation_aware_camera_scores(
        self, assignment_entropy, destination_relation_logits
    ):
        """Prefer two-object reference evidence for the ``between`` relation."""

        if assignment_entropy.ndim != 3 or assignment_entropy.shape[-1] != 2:
            raise ValueError('assignment entropy must be [batch, camera, 2]')
        if (
            destination_relation_logits.ndim != 2
            or destination_relation_logits.shape[0] != assignment_entropy.shape[0]
            or destination_relation_logits.shape[-1] != 4
        ):
            raise ValueError('destination-relation logits must be [batch, 4]')
        between_probability = jax.nn.softmax(
            destination_relation_logits.astype(jnp.float32), axis=-1
        )[:, 3]
        two_slot_entropy = math.log(2.0) / math.log(float(self.object_count))
        target_score = -assignment_entropy[:, :, 0]
        reference_entropy = assignment_entropy[:, :, 1]
        single_reference_score = -reference_entropy
        pair_reference_score = -jnp.abs(reference_entropy - two_slot_entropy)
        reference_score = (
            (1.0 - between_probability[:, None]) * single_reference_score
            + between_probability[:, None] * pair_reference_score
        )
        return jnp.stack([target_score, reference_score], axis=-1), between_probability

    def relation_aware_role_write_confidence(
        self,
        assignment_entropy,
        camera_role_weights,
        between_probability,
        role_valid_mask,
        anchor_similarity,
        anchor_valid,
    ):
        """Retain absolute visual quality after camera-score normalization.

        A camera softmax only ranks views and therefore sums to one even when
        every view is ambiguous.  The recurrent identity writer additionally
        needs an absolute confidence.  The reference role is allowed the
        expected two-slot entropy for ``between`` relations; rows with no
        valid camera evidence receive exact-zero confidence.
        """
        if assignment_entropy.ndim != 3 or assignment_entropy.shape[-1] != 2:
            raise ValueError('assignment entropy must be [batch, camera, 2]')
        if camera_role_weights.shape != assignment_entropy.shape:
            raise ValueError('camera role weights must match assignment entropy')
        batch_size = assignment_entropy.shape[0]
        if between_probability.shape != (batch_size,):
            raise ValueError('between probability must be [batch]')
        if role_valid_mask.shape != (batch_size, 2):
            raise ValueError('role validity must be [batch, 2]')
        if anchor_similarity.shape != assignment_entropy.shape:
            raise ValueError('anchor similarity must match assignment entropy')
        if anchor_valid.shape != (batch_size, 2):
            raise ValueError('anchor validity must be [batch, 2]')
        two_slot_entropy = math.log(2.0) / math.log(float(self.object_count))
        expected_reference_entropy = (
            between_probability[:, None] * two_slot_entropy
        )
        expected_role_entropy = jnp.stack(
            [
                jnp.zeros_like(expected_reference_entropy),
                expected_reference_entropy,
            ],
            axis=-1,
        )
        entropy_deviation = jnp.abs(
            assignment_entropy.astype(jnp.float32) - expected_role_entropy
        )
        weights = camera_role_weights.astype(jnp.float32)
        evidence_mass = jnp.sum(weights, axis=1)
        selected_entropy_deviation = jnp.sum(
            entropy_deviation * weights, axis=1
        )
        confidence = jnp.clip(
            1.0 - selected_entropy_deviation, 0.0, 1.0
        )
        selected_anchor_similarity = jnp.sum(
            anchor_similarity.astype(jnp.float32) * weights, axis=1
        )
        anchor_agreement = jnp.clip(
            0.5 * (selected_anchor_similarity + 1.0), 0.0, 1.0
        )
        # Reset-state anchors are exact zero and intentionally impose no
        # first-frame penalty.  Once an identity exists, a confident but
        # contradictory object assignment cannot overwrite it at full rate.
        confidence = confidence * jnp.where(
            anchor_valid, anchor_agreement, jnp.ones_like(anchor_agreement)
        )
        valid_evidence = evidence_mass > 0.0
        return jnp.where(
            role_valid_mask & valid_evidence,
            confidence,
            jnp.zeros_like(confidence),
        )

    def camera_consistent_relative_geometry(
        self,
        left_centroids,
        right_centroids,
        left_camera_weights,
        right_camera_weights,
        pair_valid,
    ):
        """Aggregate relative geometry without mixing camera coordinate frames.

        Target and reference objects can prefer different views.  Subtracting
        two independently pooled image centroids then introduces the arbitrary
        origin of each camera into the relation.  Compute the displacement in
        each camera first and pool it with the geometric mean of both binding
        confidences, so a camera contributes only when it supports the pair.
        """

        if (
            left_centroids.ndim != 3
            or right_centroids.shape != left_centroids.shape
            or left_centroids.shape[-1] != 2
        ):
            raise ValueError(
                'paired centroids must both be [batch, camera, 2]'
            )
        if (
            left_camera_weights.shape != left_centroids.shape[:2]
            or right_camera_weights.shape != left_centroids.shape[:2]
        ):
            raise ValueError(
                'paired camera weights must be [batch, camera]'
            )
        if pair_valid.shape != (left_centroids.shape[0],):
            raise ValueError('paired geometry validity must be [batch]')
        pair_product = (
            jnp.maximum(left_camera_weights.astype(jnp.float32), 0.0)
            * jnp.maximum(right_camera_weights.astype(jnp.float32), 0.0)
        )
        # Invalid cameras and absent reference slots carry exact-zero
        # confidence.  ``sqrt(0)`` has an infinite derivative, which can turn
        # the subsequently masked gradient into NaN.  Bound the evaluated
        # branch away from zero, then restore exact zeros with a boolean mask.
        pair_weights = jnp.where(
            pair_valid[:, None] & (pair_product > 0.0),
            jnp.sqrt(jnp.maximum(pair_product, 1.0e-12)),
            0.0,
        )
        pair_weights = pair_weights / jnp.maximum(
            jnp.sum(pair_weights, axis=1, keepdims=True), 1.0e-8
        )
        relative = jnp.sum(
            (left_centroids.astype(jnp.float32) - right_centroids.astype(jnp.float32))
            * pair_weights[..., None],
            axis=1,
        )
        return jnp.where(
            pair_valid[:, None], relative, jnp.zeros_like(relative)
        )

    def visual_language_role_contrastive_loss(
        self,
        language_roles,
        camera_bound_roles,
        camera_role_weights,
        role_valid_mask,
    ):
        """Bind target/reference visual roles to their own language spans.

        The two roles in each instruction are mutual negatives.  Language
        roles are stop-gradient semantic teachers so this objective trains the
        visual object assignment path without collapsing the text anchors or
        requiring a closed-set entity classifier.
        """

        visual_roles = jnp.sum(
            camera_bound_roles * camera_role_weights[..., None], axis=1
        )
        normalized_visual = _rms_normalize(visual_roles).astype(jnp.float32)
        normalized_language = jax.lax.stop_gradient(
            _rms_normalize(language_roles).astype(jnp.float32)
        )
        logits = jnp.einsum(
            'brh,bsh->brs',
            normalized_visual,
            normalized_language,
            preferred_element_type=jnp.float32,
        ) / (float(self.hidden_dim) * 0.1)
        matching_log_probs = jax.nn.log_softmax(logits, axis=-1)
        matching_loss = -jnp.mean(
            jnp.diagonal(matching_log_probs, axis1=-2, axis2=-1), axis=-1
        )
        role_evidence_valid = jnp.sum(camera_role_weights, axis=1) > 0
        pair_valid = jnp.all(role_valid_mask & role_evidence_valid, axis=-1)
        return matching_loss * pair_valid.astype(jnp.float32)

    def balanced_factorized_class_weights(self, class_counts, *, count_floor=1.0):
        """Return smoothed inverse-sqrt weights with unit empirical expectation."""

        counts = jnp.asarray(class_counts, dtype=jnp.float32)
        if counts.ndim != 1:
            raise ValueError('factorized class counts must be one-dimensional')
        raw_weights = jax.lax.rsqrt(jnp.maximum(counts, count_floor))
        expected_weight = jnp.sum(raw_weights * counts) / jnp.maximum(
            jnp.sum(counts), 1.0
        )
        return raw_weights / jnp.maximum(expected_weight, 1.0e-8)

    def role_separated_factorized_head_inputs(
        self,
        factor_language_states,
        language_roles,
        *,
        source_reference_language,
        source_reference_valid,
        destination_reference_language,
        destination_reference_valid,
        condition_state_language,
        condition_state_valid,
        destination_qualifier_language,
        destination_qualifier_valid,
    ):
        """Route only the relevant linguistic roles into each factor head.

        Every head receives its own factor-identity query over the complete
        language sequence, while signed role differences expose the directional
        relation it must model.  No pooled whole-instruction vector is shared
        across factor heads.
        """
        if factor_language_states.ndim != 3:
            raise ValueError(
                'factor language states must be [batch, factor, hidden]'
            )
        batch_size, factor_count, hidden_dim = factor_language_states.shape
        if factor_count != 5:
            raise ValueError('factor language states must contain five factors')
        if language_roles.shape != (batch_size, 2, hidden_dim):
            raise ValueError('factorized language roles have an invalid shape')
        if source_reference_language.shape != (batch_size, hidden_dim):
            raise ValueError('source-reference language has an invalid shape')
        if source_reference_valid.shape != (batch_size,):
            raise ValueError('source-reference validity has an invalid shape')
        if destination_reference_language.shape != (batch_size, 2, hidden_dim):
            raise ValueError('destination-reference language has an invalid shape')
        if destination_reference_valid.shape != (batch_size, 2):
            raise ValueError('destination-reference validity has an invalid shape')
        if condition_state_language.shape != (batch_size, hidden_dim):
            raise ValueError('condition-state language has an invalid shape')
        if condition_state_valid.shape != (batch_size,):
            raise ValueError('condition-state validity has an invalid shape')
        if destination_qualifier_language.shape != (batch_size, hidden_dim):
            raise ValueError('destination-qualifier language has an invalid shape')
        if destination_qualifier_valid.shape != (batch_size,):
            raise ValueError('destination-qualifier validity has an invalid shape')

        target_role = language_roles[:, 0]
        destination_role = language_roles[:, 1]
        source_reference = source_reference_language * (
            source_reference_valid[:, None].astype(language_roles.dtype)
        )
        destination_reference_weights = destination_reference_valid.astype(
            language_roles.dtype
        )
        destination_reference = jnp.sum(
            destination_reference_language
            * destination_reference_weights[..., None],
            axis=1,
        ) / jnp.maximum(
            jnp.sum(destination_reference_weights, axis=1, keepdims=True),
            1.0,
        )
        condition_state = condition_state_language * (
            condition_state_valid[:, None].astype(language_roles.dtype)
        )
        destination_qualifier = destination_qualifier_language * (
            destination_qualifier_valid[:, None].astype(language_roles.dtype)
        )
        return {
            'operation': factor_language_states[:, 0] + target_role,
            'source_relation': (
                factor_language_states[:, 1]
                + 0.5 * (target_role - source_reference)
            ),
            'destination_relation': (
                factor_language_states[:, 2]
                + 0.5 * (destination_role - destination_reference)
            ),
            'condition': factor_language_states[:, 3] + condition_state,
            'destination_qualifier': (
                factor_language_states[:, 4]
                + destination_role
                + destination_qualifier
            ),
        }

    def bind_plan(
        self,
        prefix_tokens,
        prefix_mask,
        state,
        *,
        plan_token_mask=None,
        role_span_mask=None,
        source_reference_span_mask=None,
        destination_reference_span_mask=None,
        condition_state_span_mask=None,
        destination_qualifier_span_mask=None,
        role_valid_mask=None,
        clause_span_mask=None,
        clause_valid_mask=None,
        role_memory_anchors=None,
        previous_actions=None,
        previous_actions_valid=None,
        episode_start=None,
        visual_tokens=None,
        camera_mask=None,
        compute_object_reconstruction=False,
        structured_demo=None,
        structured_demo_tokens=None,
        structured_demo_token_mask=None,
    ):
        batch_size = prefix_tokens.shape[0]
        if plan_token_mask is None:
            plan_token_mask = prefix_mask
        if plan_token_mask.shape != prefix_mask.shape:
            raise ValueError('plan token mask must match the prefix mask')
        # Ordered plan slots represent linguistic subgoals.  In production the
        # prefix also contains hundreds of observation-dependent image patches;
        # letting those patches compete for plan attention makes slot identity
        # drift between replans.  Keep the explicit language-only mask separate
        # from the full causal prefix used by current_context and object binding.
        plan_token_mask = plan_token_mask & prefix_mask
        plan_token_mask = jnp.where(
            jnp.any(plan_token_mask, axis=-1, keepdims=True),
            plan_token_mask,
            prefix_mask,
        )
        subgoal_queries = self.subgoal_queries(
            jnp.arange(self.subgoal_slots)
        )[None]
        subgoal_queries = jnp.broadcast_to(
            subgoal_queries, (batch_size, self.subgoal_slots, self.hidden_dim)
        )
        language_position_encoding = self._language_position_encoding(
            plan_token_mask, dtype=prefix_tokens.dtype
        )
        ordered_subgoals = self._attend(
            subgoal_queries,
            self.subgoal_key(prefix_tokens) + language_position_encoding,
            self.subgoal_value(prefix_tokens) + language_position_encoding,
            plan_token_mask,
        )
        clause_plan_residual = jnp.zeros_like(ordered_subgoals)
        clause_plan_attention = None
        clause_plan_adapter = getattr(self, 'clause_plan_adapter', None)
        if clause_plan_adapter is not None:
            if clause_span_mask is None or clause_valid_mask is None:
                raise ValueError('ClausePlan-v1 requires clause masks')
            (
                ordered_subgoals,
                clause_plan_residual,
                clause_plan_attention,
            ) = clause_plan_adapter(
                ordered_subgoals,
                prefix_tokens,
                prefix_mask,
                clause_span_mask,
                clause_valid_mask,
            )
        role_queries = self.role_queries(jnp.arange(2))[None]
        role_queries = jnp.broadcast_to(
            role_queries, (batch_size, 2, self.hidden_dim)
        )
        if role_span_mask is None:
            role_span_mask = jnp.broadcast_to(
                plan_token_mask[:, None, :],
                (batch_size, 2, prefix_mask.shape[1]),
            )
        if role_span_mask.shape != (batch_size, 2, prefix_tokens.shape[1]):
            raise ValueError('factorized role span mask has an invalid shape')
        if role_valid_mask is None:
            role_valid_mask = jnp.ones((batch_size, 2), dtype=jnp.bool_)
        if role_valid_mask.shape != (batch_size, 2):
            raise ValueError('factorized role validity mask has an invalid shape')
        if role_memory_anchors is None:
            role_memory_anchors = jnp.zeros(
                (batch_size, 2, self.hidden_dim), dtype=prefix_tokens.dtype
            )
        if role_memory_anchors.shape != (batch_size, 2, self.hidden_dim):
            raise ValueError('persistent role-memory anchors have an invalid shape')
        effective_role_mask = role_span_mask & prefix_mask[:, None, :]
        role_keys = self.role_key(prefix_tokens)
        role_values = self.role_value(prefix_tokens)
        language_roles = self._attend(
            role_queries,
            role_keys,
            role_values,
            effective_role_mask,
        )
        language_roles = jnp.where(
            role_valid_mask[..., None], language_roles, jnp.zeros_like(language_roles)
        )
        span_weights = effective_role_mask.astype(role_values.dtype)
        role_span_teacher = jax.lax.stop_gradient(
            jnp.einsum('brl,blh->brh', span_weights, role_values)
            / jnp.maximum(jnp.sum(span_weights, axis=-1, keepdims=True), 1.0)
        )
        if source_reference_span_mask is None:
            source_reference_span_mask = jnp.zeros_like(prefix_mask)
        if source_reference_span_mask.shape != prefix_mask.shape:
            raise ValueError('source-reference span mask must match the prefix mask')
        effective_source_reference_mask = (
            source_reference_span_mask & prefix_mask
        )
        source_reference_valid = jnp.any(
            effective_source_reference_mask, axis=-1
        )
        source_reference_query = self.source_reference_query(
            jnp.arange(1)
        )[None]
        source_reference_query = jnp.broadcast_to(
            source_reference_query, (batch_size, 1, self.hidden_dim)
        )
        source_reference_language = self._attend(
            source_reference_query,
            role_keys,
            role_values,
            effective_source_reference_mask,
        )[:, 0]
        source_reference_language = jnp.where(
            source_reference_valid[:, None],
            source_reference_language,
            jnp.zeros_like(source_reference_language),
        )
        source_reference_weights = effective_source_reference_mask.astype(
            role_values.dtype
        )
        source_reference_teacher = jax.lax.stop_gradient(
            jnp.einsum(
                'bl,blh->bh', source_reference_weights, role_values
            )
            / jnp.maximum(
                jnp.sum(source_reference_weights, axis=-1, keepdims=True), 1.0
            )
        )
        if destination_reference_span_mask is None:
            destination_reference_span_mask = jnp.zeros(
                (batch_size, 2, prefix_tokens.shape[1]), dtype=jnp.bool_
            )
        if destination_reference_span_mask.shape != (
            batch_size,
            2,
            prefix_tokens.shape[1],
        ):
            raise ValueError(
                'destination-reference span mask must be [batch, 2, prefix]'
            )
        effective_destination_reference_mask = (
            destination_reference_span_mask & prefix_mask[:, None, :]
        )
        destination_reference_valid = jnp.any(
            effective_destination_reference_mask, axis=-1
        )
        destination_reference_queries = self.destination_reference_queries(
            jnp.arange(2)
        )[None]
        destination_reference_queries = jnp.broadcast_to(
            destination_reference_queries,
            (batch_size, 2, self.hidden_dim),
        )
        destination_reference_language = self._attend(
            destination_reference_queries,
            role_keys,
            role_values,
            effective_destination_reference_mask,
        )
        destination_reference_language = jnp.where(
            destination_reference_valid[..., None],
            destination_reference_language,
            jnp.zeros_like(destination_reference_language),
        )
        destination_reference_weights = (
            effective_destination_reference_mask.astype(role_values.dtype)
        )
        destination_reference_teacher = jax.lax.stop_gradient(
            jnp.einsum(
                'brl,blh->brh', destination_reference_weights, role_values
            )
            / jnp.maximum(
                jnp.sum(
                    destination_reference_weights, axis=-1, keepdims=True
                ),
                1.0,
            )
        )
        if condition_state_span_mask is None:
            condition_state_span_mask = jnp.zeros_like(prefix_mask)
        if condition_state_span_mask.shape != prefix_mask.shape:
            raise ValueError('condition-state span mask must match the prefix mask')
        effective_condition_state_mask = condition_state_span_mask & prefix_mask
        condition_state_valid = jnp.any(
            effective_condition_state_mask, axis=-1
        )
        condition_state_query = self.condition_state_query(jnp.arange(1))[None]
        condition_state_query = jnp.broadcast_to(
            condition_state_query, (batch_size, 1, self.hidden_dim)
        )
        condition_state_language = self._attend(
            condition_state_query,
            role_keys,
            role_values,
            effective_condition_state_mask,
        )[:, 0]
        condition_state_language = jnp.where(
            condition_state_valid[:, None],
            condition_state_language,
            jnp.zeros_like(condition_state_language),
        )
        condition_state_weights = effective_condition_state_mask.astype(
            role_values.dtype
        )
        condition_state_teacher = jax.lax.stop_gradient(
            jnp.einsum(
                'bl,blh->bh', condition_state_weights, role_values
            )
            / jnp.maximum(
                jnp.sum(condition_state_weights, axis=-1, keepdims=True), 1.0
            )
        )
        if destination_qualifier_span_mask is None:
            destination_qualifier_span_mask = jnp.zeros_like(prefix_mask)
        if destination_qualifier_span_mask.shape != prefix_mask.shape:
            raise ValueError(
                'destination-qualifier span mask must match the prefix mask'
            )
        effective_destination_qualifier_mask = (
            destination_qualifier_span_mask & prefix_mask
        )
        destination_qualifier_valid = jnp.any(
            effective_destination_qualifier_mask, axis=-1
        )
        destination_qualifier_query = self.destination_qualifier_query(
            jnp.arange(1)
        )[None]
        destination_qualifier_query = jnp.broadcast_to(
            destination_qualifier_query,
            (batch_size, 1, self.hidden_dim),
        )
        destination_qualifier_language = self._attend(
            destination_qualifier_query,
            role_keys,
            role_values,
            effective_destination_qualifier_mask,
        )[:, 0]
        destination_qualifier_language = jnp.where(
            destination_qualifier_valid[:, None],
            destination_qualifier_language,
            jnp.zeros_like(destination_qualifier_language),
        )
        destination_qualifier_weights = (
            effective_destination_qualifier_mask.astype(role_values.dtype)
        )
        destination_qualifier_teacher = jax.lax.stop_gradient(
            jnp.einsum(
                'bl,blh->bh', destination_qualifier_weights, role_values
            )
            / jnp.maximum(
                jnp.sum(
                    destination_qualifier_weights, axis=-1, keepdims=True
                ),
                1.0,
            )
        )

        factor_language_queries = self.compositional_factor_identity(
            jnp.arange(5)
        )[None]
        factor_language_queries = jnp.broadcast_to(
            factor_language_queries,
            (batch_size, 5, self.hidden_dim),
        )
        factor_span_masks = jnp.stack(
            [
                effective_role_mask[:, 0],
                (
                    effective_role_mask[:, 0]
                    | effective_source_reference_mask
                ),
                (
                    effective_role_mask[:, 1]
                    | jnp.any(
                        effective_destination_reference_mask, axis=1
                    )
                ),
                effective_condition_state_mask,
                (
                    effective_role_mask[:, 1]
                    | effective_destination_qualifier_mask
                ),
            ],
            axis=1,
        )
        (
            factor_language_states,
            factor_attention_alignment_loss,
            factor_attention_valid,
        ) = self.factor_specific_language_attention(
            factor_language_queries,
            role_keys,
            role_values,
            plan_token_mask,
            factor_span_masks,
        )
        factorized_head_inputs = self.role_separated_factorized_head_inputs(
            factor_language_states,
            language_roles,
            source_reference_language=source_reference_language,
            source_reference_valid=source_reference_valid,
            destination_reference_language=destination_reference_language,
            destination_reference_valid=destination_reference_valid,
            condition_state_language=condition_state_language,
            condition_state_valid=condition_state_valid,
            destination_qualifier_language=destination_qualifier_language,
            destination_qualifier_valid=destination_qualifier_valid,
        )
        operation_logits = self.operation_head(factorized_head_inputs['operation'])
        source_relation_logits = self.source_relation_head(
            factorized_head_inputs['source_relation']
        )
        destination_relation_logits = self.destination_relation_head(
            factorized_head_inputs['destination_relation']
        )
        condition_logits = self.condition_head(factorized_head_inputs['condition'])
        destination_qualifier_logits = self.destination_qualifier_head(
            factorized_head_inputs['destination_qualifier']
        )
        (
            ordered_subgoals,
            semantic_phase_state,
            operation_phase_state,
        ) = self.compose_semantic_phase_subgoals(
            ordered_subgoals, operation_logits
        )
        factorized_plan_relation_state = (
            self._factorized_semantic_state_from_logits(
                (
                    (source_relation_logits, self.source_relation_head),
                    (
                        destination_relation_logits,
                        self.destination_relation_head,
                    ),
                    (condition_logits, self.condition_head),
                    (
                        destination_qualifier_logits,
                        self.destination_qualifier_head,
                    ),
                ),
                dtype=ordered_subgoals.dtype,
            )
        )
        ordered_subgoals, relation_phase_state = (
            self.compose_factorized_relation_phase_subgoals(
                ordered_subgoals, factorized_plan_relation_state
            )
        )
        (
            ordered_subgoals,
            compositional_program_state,
            compositional_factor_tokens,
        ) = self.compose_compositional_program_subgoals(
            ordered_subgoals,
            (
                (operation_logits, self.operation_head),
                (source_relation_logits, self.source_relation_head),
                (destination_relation_logits, self.destination_relation_head),
                (condition_logits, self.condition_head),
                (
                    destination_qualifier_logits,
                    self.destination_qualifier_head,
                ),
            ),
        )
        # Preserve the language-defined plan identity separately from the
        # camera-grounded execution program below.  Routing consumes the
        # grounded version, while audits/downstream diagnostics can still
        # distinguish a changed observation from a changed instruction.
        language_ordered_subgoals = ordered_subgoals
        structured_demo_shared_context = ordered_subgoals
        if structured_demo is not None:
            if (
                structured_demo_tokens is None
                or structured_demo_token_mask is None
            ):
                raise ValueError('structured-demo fusion inputs are incomplete')
            ordered_subgoals, structured_demo_shared_context = (
                structured_demo.fuse(
                    ordered_subgoals,
                    structured_demo_tokens,
                    structured_demo_token_mask,
                )
            )
        language_compositional_program_state = compositional_program_state
        grounded_relation_phase_state = jnp.zeros_like(relation_phase_state)

        camera_bound_roles = None
        temporal_role_memory_state = None
        cross_view_role_consensus_state = None
        contact_risk_calibrated_role_residual_state = None
        relational_role_composer_residual_state = None
        source_reference_camera_bound = None
        source_reference_camera_weights = None
        source_reference_bound = source_reference_language
        source_reference_centroid = None
        source_reference_overlap = None
        destination_reference_camera_bound = None
        destination_reference_camera_weights = None
        destination_reference_bound = destination_reference_language
        destination_reference_centroids = None
        destination_reference_overlap = None
        condition_state_camera_bound = None
        condition_state_camera_weights = None
        condition_state_bound = condition_state_language
        condition_state_centroid = None
        condition_state_overlap = None
        condition_state_context = condition_state_language
        object_slot_reconstruction_loss = None
        role_identity_write_confidence = None
        if visual_tokens is not None:
            if visual_tokens.ndim != 4 or visual_tokens.shape[0] != batch_size:
                raise ValueError('visual tokens must be [batch, camera, patch, prefix]')
            if camera_mask is None or camera_mask.shape != visual_tokens.shape[:2]:
                raise ValueError('camera mask differs from visual tokens')
            cameras = visual_tokens.shape[1]
            visual_keys = _rms_normalize(self.visual_key(visual_tokens))
            visual_values = self.visual_value(visual_tokens)
            object_queries = self.object_queries(jnp.arange(self.object_count))
            object_queries = jnp.broadcast_to(
                object_queries[None, None],
                (batch_size, cameras, self.object_count, self.hidden_dim),
            )
            object_logits = jnp.einsum(
                'bcod,bcpd->bcop',
                _rms_normalize(self.object_query(object_queries)),
                visual_keys,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(self.hidden_dim))
            patch_weights, patch_assignments = (
                self.competitive_object_patch_weights(object_logits)
            )
            patch_weights = patch_weights.astype(visual_values.dtype)
            patch_count = visual_tokens.shape[2]
            grid_side = math.isqrt(patch_count)
            if grid_side * grid_side == patch_count:
                row_ids = jnp.arange(patch_count) // grid_side
                column_ids = jnp.arange(patch_count) % grid_side
                denominator = float(max(grid_side - 1, 1))
                patch_coordinates = jnp.stack(
                    [
                        2.0 * column_ids.astype(jnp.float32) / denominator - 1.0,
                        2.0 * row_ids.astype(jnp.float32) / denominator - 1.0,
                    ],
                    axis=-1,
                )
            else:
                # Small contract probes need not use a square patch grid.
                patch_coordinates = jnp.stack(
                    [
                        jnp.linspace(-1.0, 1.0, patch_count),
                        jnp.zeros((patch_count,), dtype=jnp.float32),
                    ],
                    axis=-1,
                )
            object_updates = jnp.einsum(
                'bcop,bcpd->bcod', patch_weights, visual_values
            )
            object_centroids = jnp.einsum(
                'bcop,pd->bcod',
                patch_weights.astype(jnp.float32),
                patch_coordinates,
            )
            object_residuals = self.object_out(object_updates)
            objects = object_queries + object_residuals
            objects = objects * camera_mask[:, :, None, None].astype(objects.dtype)
            if compute_object_reconstruction:
                normalized_slot_contents = _rms_normalize(
                    object_residuals
                ).astype(jnp.float32)
                patch_reconstruction = jnp.einsum(
                    'bcop,bcod->bcpd',
                    patch_assignments.astype(jnp.float32),
                    normalized_slot_contents,
                )
                patch_targets = jax.lax.stop_gradient(
                    _rms_normalize(visual_values).astype(jnp.float32)
                )
                reconstruction_error = jnp.mean(
                    jnp.square(patch_reconstruction - patch_targets), axis=-1
                )
                valid_patches = jnp.broadcast_to(
                    camera_mask[..., None], reconstruction_error.shape
                ).astype(jnp.float32)
                object_slot_reconstruction_loss = jnp.sum(
                    reconstruction_error * valid_patches, axis=(1, 2)
                ) / jnp.maximum(jnp.sum(valid_patches, axis=(1, 2)), 1.0)

            # Query current object slots with both the open-vocabulary prompt
            # role and its persistent identity anchor from the prior replan.
            # Reset memory is exact zero, so the first-replan behavior remains
            # purely language grounded without a learned-bias leakage path.
            grounded_queries = _rms_normalize(
                self.grounded_role_query(language_roles)
                + 0.25 * _rms_normalize(role_memory_anchors)
            )
            object_keys = _rms_normalize(self.object_key(objects))
            role_logits = jnp.einsum(
                'brd,bcod->bcro',
                grounded_queries,
                object_keys,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(self.hidden_dim))
            role_probabilities, role_object_overlap = (
                self.ordered_role_object_probabilities(
                    role_logits, role_valid_mask
                )
            )
            source_reference_grounded_query = _rms_normalize(
                self.grounded_role_query(source_reference_language)
            )
            source_reference_logits = jnp.einsum(
                'bd,bcod->bco',
                source_reference_grounded_query,
                object_keys,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(self.hidden_dim))
            source_pair_logits = jnp.stack(
                [role_logits[:, :, 0], source_reference_logits], axis=-2
            )
            source_pair_valid = jnp.stack(
                [role_valid_mask[:, 0], source_reference_valid], axis=-1
            )
            source_pair_probabilities, source_reference_overlap = (
                self.ordered_role_object_probabilities(
                    source_pair_logits, source_pair_valid
                )
            )
            source_reference_probabilities = source_pair_probabilities[:, :, 1]
            destination_reference_grounded_queries = _rms_normalize(
                self.grounded_role_query(destination_reference_language)
            )
            destination_reference_logits = jnp.einsum(
                'brd,bcod->bcro',
                destination_reference_grounded_queries,
                object_keys,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(self.hidden_dim))
            (
            destination_reference_probabilities,
                destination_reference_overlap,
            ) = self.ordered_destination_reference_probabilities(
                role_logits[:, :, 0],
                destination_reference_logits,
                destination_reference_valid,
            )
            condition_state_grounded_query = _rms_normalize(
                self.grounded_role_query(condition_state_language)
            )
            condition_state_logits = jnp.einsum(
                'bd,bcod->bco',
                condition_state_grounded_query,
                object_keys,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(self.hidden_dim))
            condition_pair_logits = jnp.stack(
                [role_logits[:, :, 0], condition_state_logits], axis=-2
            )
            condition_pair_valid = jnp.stack(
                [role_valid_mask[:, 0], condition_state_valid], axis=-1
            )
            condition_pair_probabilities, condition_state_overlap = (
                self.ordered_role_object_probabilities(
                    condition_pair_logits, condition_pair_valid
                )
            )
            condition_state_probabilities = condition_pair_probabilities[:, :, 1]
            camera_role_centroids = jnp.einsum(
                'bcro,bcod->bcrd',
                role_probabilities.astype(jnp.float32),
                object_centroids,
            )
            source_reference_camera_centroids = jnp.einsum(
                'bco,bcod->bcd',
                source_reference_probabilities.astype(jnp.float32),
                object_centroids,
            )
            destination_reference_camera_centroids = jnp.einsum(
                'bcro,bcod->bcrd',
                destination_reference_probabilities.astype(jnp.float32),
                object_centroids,
            )
            condition_state_camera_centroids = jnp.einsum(
                'bco,bcod->bcd',
                condition_state_probabilities.astype(jnp.float32),
                object_centroids,
            )
            camera_bound_roles = self.bound_role_out(
                jnp.einsum('bcro,bcod->bcrd', role_probabilities, objects)
            )
            camera_bound_roles = camera_bound_roles * camera_mask[
                :, :, None, None
            ].astype(camera_bound_roles.dtype)
            temporal_role_memory = getattr(
                self, 'action_conditioned_temporal_object_residual_v1', None
            )
            if temporal_role_memory is not None:
                if (
                    previous_actions is None
                    or previous_actions_valid is None
                    or episode_start is None
                ):
                    raise ValueError(
                        'temporal role memory requires causal previous actions and reset'
                    )
                camera_bound_roles, temporal_role_memory_state = (
                    temporal_role_memory(
                        camera_bound_roles,
                        role_memory_anchors,
                        previous_actions,
                        previous_actions_valid,
                        camera_mask,
                        role_valid_mask,
                        episode_start,
                    )
                )
            else:
                temporal_role_memory_state = None
            cross_view_role_consensus = getattr(
                self, 'cross_view_role_consensus_v1', None
            )
            if cross_view_role_consensus is not None:
                cross_view_input_roles = camera_bound_roles
                camera_bound_roles, cross_view_role_consensus_state = (
                    cross_view_role_consensus(
                        camera_bound_roles,
                        camera_mask,
                        role_valid_mask,
                    )
                )
                cross_view_role_consensus_state = {
                    **cross_view_role_consensus_state,
                    'input_roles': cross_view_input_roles,
                }
            else:
                cross_view_role_consensus_state = None
            source_reference_camera_bound = self.bound_role_out(
                jnp.einsum(
                    'bco,bcod->bcd', source_reference_probabilities, objects
                )
            )
            destination_reference_camera_bound = self.bound_role_out(
                jnp.einsum(
                    'bcro,bcod->bcrd',
                    destination_reference_probabilities,
                    objects,
                )
            )
            condition_state_camera_bound = self.bound_role_out(
                jnp.einsum(
                    'bco,bcod->bcd', condition_state_probabilities, objects
                )
            )
            source_reference_camera_valid = (
                camera_mask.astype(jnp.bool_)
                & source_reference_valid[:, None]
            )
            source_reference_camera_bound = source_reference_camera_bound * (
                source_reference_camera_valid[..., None].astype(
                    source_reference_camera_bound.dtype
                )
            )
            destination_reference_camera_valid = (
                camera_mask[:, :, None].astype(jnp.bool_)
                & destination_reference_valid[:, None, :]
            )
            destination_reference_camera_bound = (
                destination_reference_camera_bound
                * destination_reference_camera_valid[..., None].astype(
                    destination_reference_camera_bound.dtype
                )
            )
            condition_state_camera_valid = (
                camera_mask.astype(jnp.bool_)
                & condition_state_valid[:, None]
            )
            condition_state_camera_bound = condition_state_camera_bound * (
                condition_state_camera_valid[..., None].astype(
                    condition_state_camera_bound.dtype
                )
            )
            # Prefer views with confident object assignment and, after the
            # first replan, agreement with the causal identity anchor.
            assignment_entropy = -jnp.sum(
                role_probabilities.astype(jnp.float32)
                * jnp.log(
                    jnp.maximum(
                        role_probabilities.astype(jnp.float32), 1.0e-8
                    )
                ),
                axis=-1,
            ) / jnp.log(float(self.object_count))
            anchor_valid = jnp.mean(
                jnp.square(role_memory_anchors.astype(jnp.float32)), axis=-1
            ) > 1.0e-8
            anchor_similarity = jnp.sum(
                _rms_normalize(camera_bound_roles).astype(jnp.float32)
                * _rms_normalize(role_memory_anchors)[:, None].astype(
                    jnp.float32
                ),
                axis=-1,
            ) / float(self.hidden_dim)
            camera_role_valid = (
                camera_mask[:, :, None].astype(jnp.bool_)
                & role_valid_mask[:, None, :]
            )
            camera_scores, between_probability = self.relation_aware_camera_scores(
                assignment_entropy, destination_relation_logits
            )
            camera_scores = camera_scores + 0.5 * jnp.where(
                anchor_valid[:, None, :], anchor_similarity, 0.0
            )
            camera_scores = jnp.where(
                camera_role_valid, camera_scores, -1.0e30
            )
            camera_role_weights = jax.nn.softmax(
                camera_scores, axis=1
            ).astype(camera_bound_roles.dtype)
            camera_role_weights = jnp.where(
                camera_role_valid,
                camera_role_weights,
                jnp.zeros_like(camera_role_weights),
            )
            role_identity_write_confidence = (
                self.relation_aware_role_write_confidence(
                    assignment_entropy,
                    camera_role_weights,
                    between_probability,
                    role_valid_mask,
                    anchor_similarity,
                    anchor_valid,
                )
            )
            source_reference_entropy = -jnp.sum(
                source_reference_probabilities.astype(jnp.float32)
                * jnp.log(
                    jnp.maximum(
                        source_reference_probabilities.astype(jnp.float32),
                        1.0e-8,
                    )
                ),
                axis=-1,
            ) / jnp.log(float(self.object_count))
            source_reference_camera_scores = jnp.where(
                source_reference_camera_valid,
                -source_reference_entropy,
                -1.0e30,
            )
            source_reference_camera_weights = jax.nn.softmax(
                source_reference_camera_scores, axis=1
            ).astype(source_reference_camera_bound.dtype)
            source_reference_camera_weights = jnp.where(
                source_reference_camera_valid,
                source_reference_camera_weights,
                jnp.zeros_like(source_reference_camera_weights),
            )
            source_reference_bound = jnp.sum(
                source_reference_camera_bound
                * source_reference_camera_weights[..., None],
                axis=1,
            )
            source_reference_centroid = jnp.sum(
                source_reference_camera_centroids
                * source_reference_camera_weights[..., None].astype(jnp.float32),
                axis=1,
            )
            destination_reference_entropy = -jnp.sum(
                destination_reference_probabilities.astype(jnp.float32)
                * jnp.log(
                    jnp.maximum(
                        destination_reference_probabilities.astype(jnp.float32),
                        1.0e-8,
                    )
                ),
                axis=-1,
            ) / jnp.log(float(self.object_count))
            destination_reference_camera_scores = jnp.where(
                destination_reference_camera_valid,
                -destination_reference_entropy,
                -1.0e30,
            )
            destination_reference_camera_weights = jax.nn.softmax(
                destination_reference_camera_scores, axis=1
            ).astype(destination_reference_camera_bound.dtype)
            destination_reference_camera_weights = jnp.where(
                destination_reference_camera_valid,
                destination_reference_camera_weights,
                jnp.zeros_like(destination_reference_camera_weights),
            )
            destination_reference_bound = jnp.sum(
                destination_reference_camera_bound
                * destination_reference_camera_weights[..., None],
                axis=1,
            )
            destination_reference_centroids = jnp.sum(
                destination_reference_camera_centroids
                * destination_reference_camera_weights[..., None].astype(
                    jnp.float32
                ),
                axis=1,
            )
            condition_state_entropy = -jnp.sum(
                condition_state_probabilities.astype(jnp.float32)
                * jnp.log(
                    jnp.maximum(
                        condition_state_probabilities.astype(jnp.float32),
                        1.0e-8,
                    )
                ),
                axis=-1,
            ) / jnp.log(float(self.object_count))
            condition_state_camera_scores = jnp.where(
                condition_state_camera_valid,
                -condition_state_entropy,
                -1.0e30,
            )
            condition_state_camera_weights = jax.nn.softmax(
                condition_state_camera_scores, axis=1
            ).astype(condition_state_camera_bound.dtype)
            condition_state_camera_weights = jnp.where(
                condition_state_camera_valid,
                condition_state_camera_weights,
                jnp.zeros_like(condition_state_camera_weights),
            )
            condition_state_bound = jnp.sum(
                condition_state_camera_bound
                * condition_state_camera_weights[..., None],
                axis=1,
            )
            condition_state_centroid = jnp.sum(
                condition_state_camera_centroids
                * condition_state_camera_weights[..., None].astype(jnp.float32),
                axis=1,
            )
            condition_state_context = (
                0.25 * condition_state_bound
                + self.condition_state_geometry_in(
                    condition_state_centroid.astype(condition_state_bound.dtype)
                )
            )
            condition_state_context = jnp.where(
                condition_state_valid[:, None],
                condition_state_context,
                jnp.zeros_like(condition_state_context),
            )
            bound_roles = jnp.sum(
                camera_bound_roles * camera_role_weights[..., None], axis=1
            )
            role_centroids = jnp.sum(
                camera_role_centroids
                * camera_role_weights[..., None].astype(jnp.float32),
                axis=1,
            )
            both_roles_valid = role_valid_mask[:, 0] & role_valid_mask[:, 1]
            relative_geometry = self.camera_consistent_relative_geometry(
                camera_role_centroids[:, :, 0],
                camera_role_centroids[:, :, 1],
                camera_role_weights[:, :, 0],
                camera_role_weights[:, :, 1],
                both_roles_valid,
            )
            geometry_features = jnp.stack(
                [
                    jnp.concatenate(
                        [role_centroids[:, 0], relative_geometry], axis=-1
                    ),
                    jnp.concatenate(
                        [role_centroids[:, 1], -relative_geometry], axis=-1
                    ),
                ],
                axis=1,
            )
            bound_roles = bound_roles + self.role_geometry_in(
                geometry_features.astype(bound_roles.dtype)
            )
            source_relative_geometry = self.camera_consistent_relative_geometry(
                camera_role_centroids[:, :, 0],
                source_reference_camera_centroids,
                camera_role_weights[:, :, 0],
                source_reference_camera_weights,
                source_reference_valid,
            )
            source_geometry_features = jnp.concatenate(
                [source_reference_centroid, source_relative_geometry], axis=-1
            )
            source_context_residual = (
                0.25 * source_reference_bound
                + self.source_context_geometry_in(
                    source_geometry_features.astype(bound_roles.dtype)
                )
            )
            source_context_residual = jnp.where(
                source_reference_valid[:, None],
                source_context_residual,
                jnp.zeros_like(source_context_residual),
            )
            bound_roles = bound_roles.at[:, 0].add(source_context_residual)
            destination_reference_count = jnp.sum(
                destination_reference_valid, axis=-1, keepdims=True
            ).astype(jnp.float32)
            destination_reference_midpoint = jnp.sum(
                destination_reference_centroids
                * destination_reference_valid[..., None].astype(jnp.float32),
                axis=1,
            ) / jnp.maximum(destination_reference_count, 1.0)
            destination_reference_camera_midpoint = jnp.sum(
                destination_reference_camera_centroids
                * destination_reference_valid[:, None, :, None].astype(
                    jnp.float32
                ),
                axis=2,
            ) / jnp.maximum(
                destination_reference_count[:, None, :], 1.0
            )
            destination_reference_camera_weight = jnp.sum(
                destination_reference_camera_weights.astype(jnp.float32)
                * destination_reference_valid[:, None, :].astype(jnp.float32),
                axis=-1,
            ) / jnp.maximum(destination_reference_count, 1.0)
            both_destination_references_valid = jnp.all(
                destination_reference_valid, axis=-1
            )
            destination_reference_axis = (
                self.camera_consistent_relative_geometry(
                    destination_reference_camera_centroids[:, :, 0],
                    destination_reference_camera_centroids[:, :, 1],
                    destination_reference_camera_weights[:, :, 0],
                    destination_reference_camera_weights[:, :, 1],
                    both_destination_references_valid,
                )
            )
            any_destination_reference_valid = jnp.any(
                destination_reference_valid, axis=-1
            )
            destination_role_relative_geometry = (
                self.camera_consistent_relative_geometry(
                    camera_role_centroids[:, :, 1],
                    destination_reference_camera_midpoint,
                    camera_role_weights[:, :, 1],
                    destination_reference_camera_weight,
                    any_destination_reference_valid,
                )
            )
            destination_reference_geometry = jnp.concatenate(
                [
                    destination_reference_midpoint,
                    destination_role_relative_geometry,
                    destination_reference_axis,
                ],
                axis=-1,
            )
            destination_reference_visual = jnp.sum(
                destination_reference_bound
                * destination_reference_valid[..., None].astype(
                    destination_reference_bound.dtype
                ),
                axis=1,
            ) / jnp.maximum(
                destination_reference_count.astype(
                    destination_reference_bound.dtype
                ),
                1.0,
            )
            destination_context_residual = (
                0.25 * destination_reference_visual
                + self.destination_reference_geometry_in(
                    destination_reference_geometry.astype(bound_roles.dtype)
                )
            )
            destination_context_residual = jnp.where(
                any_destination_reference_valid[:, None],
                destination_context_residual,
                jnp.zeros_like(destination_context_residual),
            )
            bound_roles = bound_roles.at[:, 1].add(destination_context_residual)
            # The qualifier tells the policy how to interpret destination
            # geometry.  Keep it as an ordered destination-role residual even
            # when the phrase does not name a separate reference object.
            bound_roles = bound_roles.at[:, 1].add(
                0.25 * destination_qualifier_language
            )
            bound_roles = jnp.where(
                role_valid_mask[..., None], bound_roles, jnp.zeros_like(bound_roles)
            )
            contact_risk_module = getattr(
                self, 'contact_risk_calibrated_role_residual_v1', None
            )
            if contact_risk_module is not None:
                if (
                    temporal_role_memory_state is None
                    or previous_actions is None
                    or previous_actions_valid is None
                    or episode_start is None
                    or state.shape[-1] < 8
                ):
                    raise ValueError(
                        'contact-risk role residual requires temporal memory, '
                        'causal previous actions/reset, and physical state'
                    )
                contact_temporal_memory = jnp.sum(
                    temporal_role_memory_state
                    * camera_role_weights[..., None],
                    axis=1,
                )
                previous_action_valid = (
                    previous_actions_valid.astype(jnp.bool_)
                    & ~episode_start.astype(jnp.bool_)
                )
                physical_previous_actions = previous_actions[..., :7]
                physical_previous_actions = jnp.where(
                    previous_action_valid[:, None, None],
                    physical_previous_actions,
                    jnp.zeros_like(physical_previous_actions),
                )
                contact_risk_input_roles = bound_roles
                (
                    bound_roles,
                    contact_risk_calibrated_role_residual_state,
                ) = contact_risk_module(
                    bound_roles,
                    contact_temporal_memory,
                    physical_previous_actions,
                    state[..., :8],
                    role_valid_mask,
                )
                contact_risk_calibrated_role_residual_state = {
                    **contact_risk_calibrated_role_residual_state,
                    'input_roles': contact_risk_input_roles,
                    'temporal_memory': contact_temporal_memory,
                    'previous_actions': physical_previous_actions,
                    'robot_state': state[..., :8],
                }
            relational_role_module = getattr(
                self, 'relational_role_composer_residual_v1', None
            )
            if relational_role_module is not None:
                if temporal_role_memory_state is None:
                    raise ValueError(
                        'relational role composer requires temporal role memory'
                    )
                composer_temporal_memory = jnp.sum(
                    temporal_role_memory_state
                    * camera_role_weights[..., None],
                    axis=1,
                )
                relational_input_roles = bound_roles
                (
                    bound_roles,
                    relational_role_composer_residual_state,
                ) = relational_role_module(
                    bound_roles,
                    composer_temporal_memory,
                    factorized_plan_relation_state,
                    role_valid_mask,
                )
                relational_role_composer_residual_state = {
                    **relational_role_composer_residual_state,
                    'input_roles': relational_input_roles,
                    'temporal_memory': composer_temporal_memory,
                    'relation_token': factorized_plan_relation_state,
                }
            (
                ordered_subgoals,
                compositional_program_state,
                grounded_relation_phase_state,
            ) = self.ground_compositional_program_subgoals(
                ordered_subgoals,
                compositional_program_state,
                bound_roles=bound_roles,
                source_reference_bound=source_reference_bound,
                destination_reference_bound=destination_reference_bound,
                condition_state_context=condition_state_context,
                role_valid_mask=role_valid_mask,
                source_reference_valid=source_reference_valid,
                destination_reference_valid=destination_reference_valid,
                condition_state_valid=condition_state_valid,
                visual_evidence_valid=jnp.any(camera_mask, axis=1),
            )
            relation_phase_state = (
                relation_phase_state + grounded_relation_phase_state
            )
        else:
            bound_roles = language_roles.at[:, 1].add(
                0.25
                * destination_qualifier_language
                * role_valid_mask[:, 1, None].astype(language_roles.dtype)
            )
            camera_role_weights = None
            role_centroids = None
            role_object_overlap = None
            assignment_entropy = None
            between_probability = jax.nn.softmax(
                destination_relation_logits.astype(jnp.float32), axis=-1
            )[:, 3]
        return {
            'ordered_subgoals': ordered_subgoals,
            'clause_plan_residual': clause_plan_residual,
            'clause_plan_attention': clause_plan_attention,
            'language_ordered_subgoals': language_ordered_subgoals,
            'structured_demo_tokens': structured_demo_tokens,
            'structured_demo_token_mask': structured_demo_token_mask,
            'structured_demo_shared_context': (
                structured_demo_shared_context
            ),
            'semantic_phase_state': semantic_phase_state,
            'operation_phase_state': operation_phase_state,
            'relation_phase_state': relation_phase_state,
            'grounded_relation_phase_state': (
                grounded_relation_phase_state
            ),
            'compositional_program_state': compositional_program_state,
            'language_compositional_program_state': (
                language_compositional_program_state
            ),
            'compositional_factor_tokens': compositional_factor_tokens,
            'factorized_plan_relation_state': (
                factorized_plan_relation_state
            ),
            'language_roles': language_roles,
            'role_span_teacher': role_span_teacher,
            'factor_attention_alignment_loss': (
                factor_attention_alignment_loss
            ),
            'factor_attention_valid': factor_attention_valid,
            'object_slot_reconstruction_loss': (
                object_slot_reconstruction_loss
            ),
            'source_reference_language': source_reference_language,
            'source_reference_teacher': source_reference_teacher,
            'source_reference_valid': source_reference_valid,
            'source_reference_camera_bound': source_reference_camera_bound,
            'source_reference_camera_weights': source_reference_camera_weights,
            'source_reference_bound': source_reference_bound,
            'source_reference_centroid': source_reference_centroid,
            'source_reference_overlap': source_reference_overlap,
            'destination_reference_language': destination_reference_language,
            'destination_reference_teacher': destination_reference_teacher,
            'destination_reference_valid': destination_reference_valid,
            'destination_reference_camera_bound': (
                destination_reference_camera_bound
            ),
            'destination_reference_camera_weights': (
                destination_reference_camera_weights
            ),
            'destination_reference_bound': destination_reference_bound,
            'destination_reference_centroids': destination_reference_centroids,
            'destination_reference_overlap': destination_reference_overlap,
            'condition_state_language': condition_state_language,
            'condition_state_teacher': condition_state_teacher,
            'condition_state_valid': condition_state_valid,
            'condition_state_camera_bound': condition_state_camera_bound,
            'condition_state_camera_weights': condition_state_camera_weights,
            'condition_state_bound': condition_state_bound,
            'condition_state_centroid': condition_state_centroid,
            'condition_state_overlap': condition_state_overlap,
            'condition_state_context': condition_state_context,
            'destination_qualifier_language': destination_qualifier_language,
            'destination_qualifier_teacher': destination_qualifier_teacher,
            'destination_qualifier_valid': destination_qualifier_valid,
            'bound_roles': bound_roles,
            'camera_bound_roles': camera_bound_roles,
            'temporal_role_memory_state': temporal_role_memory_state,
            'cross_view_role_consensus_state': cross_view_role_consensus_state,
            'contact_risk_calibrated_role_residual_state': (
                contact_risk_calibrated_role_residual_state
            ),
            'relational_role_composer_residual_state': (
                relational_role_composer_residual_state
            ),
            'camera_role_weights': camera_role_weights,
            'role_object_overlap': role_object_overlap,
            'role_assignment_entropy': assignment_entropy,
            'role_identity_write_confidence': (
                role_identity_write_confidence
            ),
            'between_relation_probability': between_probability,
            'role_centroids': role_centroids,
            'camera_mask': camera_mask,
            'role_valid_mask': role_valid_mask,
            'prior_role_identity_anchors': role_memory_anchors,
            'operation_logits': operation_logits,
            'source_relation_logits': source_relation_logits,
            'destination_relation_logits': destination_relation_logits,
            'condition_logits': condition_logits,
            'destination_qualifier_logits': destination_qualifier_logits,
        }

    def compose_semantic_phase_subgoals(
        self, ordered_subgoals, operation_logits
    ):
        """Factor plan slots into shared phases and predicted operations.

        Uniform operation evidence contributes exact zero, so an uncertain
        classifier cannot inject a class-average shortcut.  Entropy gates the
        centered operation mixture, matching the causal relation-state design.
        """

        if ordered_subgoals.ndim != 3:
            raise ValueError('ordered subgoals must be [batch, slot, hidden]')
        if ordered_subgoals.shape[1:] != (
            self.subgoal_slots,
            self.hidden_dim,
        ):
            raise ValueError('ordered subgoal shape differs from phase contract')
        if operation_logits.shape != (ordered_subgoals.shape[0], 4):
            raise ValueError('operation logits must have four classes')
        semantic_phase_state = self.semantic_phase_embeddings(
            jnp.arange(self.subgoal_slots)
        )[None]
        semantic_phase_state = jnp.broadcast_to(
            semantic_phase_state, ordered_subgoals.shape
        )
        operation_probabilities = jax.nn.softmax(
            operation_logits.astype(jnp.float32), axis=-1
        )
        centered_operation = operation_probabilities - 0.25
        operation_entropy = -jnp.sum(
            operation_probabilities
            * jnp.log(jnp.maximum(operation_probabilities, 1.0e-8)),
            axis=-1,
        ) / math.log(4.0)
        operation_confidence = (1.0 - operation_entropy)[:, None, None]
        operation_phase_bank = self.operation_phase_embeddings(
            jnp.arange(4 * self.subgoal_slots)
        ).reshape(4, self.subgoal_slots, self.hidden_dim)
        operation_phase_state = jnp.einsum(
            'bo,osh->bsh', centered_operation, operation_phase_bank
        )
        operation_phase_state = (
            operation_confidence
            * operation_phase_state.astype(ordered_subgoals.dtype)
        )
        composed = (
            ordered_subgoals
            + 0.25 * semantic_phase_state
            + 0.25 * operation_phase_state
        )
        return composed, semantic_phase_state, operation_phase_state

    def compose_factorized_relation_phase_subgoals(
        self, ordered_subgoals, factorized_relation_state
    ):
        """Inject a low-rank relation residual into every physical phase.

        The input is the confidence-gated, centered mixture of independently
        supervised source relation, destination relation, condition, and
        destination qualifier prototypes.  Uniform predictions therefore
        produce an exact-zero input.  Bias-free projections preserve that
        zero through an explicit evidence gate while learned slot gates let the same relation affect approach,
        interaction, transport, alignment, and completion differently.
        """

        if ordered_subgoals.ndim != 3:
            raise ValueError('ordered subgoals must be [batch, slot, hidden]')
        if ordered_subgoals.shape[1:] != (
            self.subgoal_slots,
            self.hidden_dim,
        ):
            raise ValueError('ordered subgoal shape differs from phase contract')
        if factorized_relation_state.shape != (
            ordered_subgoals.shape[0],
            self.hidden_dim,
        ):
            raise ValueError(
                'factorized relation state must be [batch, hidden]'
            )
        relation_latent = nnx.swish(
            self.relation_phase_down(factorized_relation_state)
        )
        phase_slots = self.relation_phase_slots(
            jnp.arange(self.subgoal_slots)
        )
        phase_scale = 1.0 + jnp.tanh(phase_slots)
        phase_latent = relation_latent[:, None, :] * phase_scale[None]
        relation_phase_state = self.relation_phase_up(phase_latent).astype(
            ordered_subgoals.dtype
        )
        relation_evidence_present = jnp.any(
            factorized_relation_state != 0, axis=-1
        )
        relation_phase_state = jnp.where(
            relation_evidence_present[:, None, None],
            relation_phase_state,
            jnp.zeros_like(relation_phase_state),
        )
        return ordered_subgoals + 0.25 * relation_phase_state, relation_phase_state

    def _factorized_semantic_state_from_logits(self, entries, *, dtype):
        """Map independent classifier distributions to one semantic state."""

        states = []
        for logits, head in entries:
            logits = logits.astype(jnp.float32)
            probabilities = jax.nn.softmax(logits, axis=-1)
            class_count = probabilities.shape[-1]
            centered = probabilities - (1.0 / float(class_count))
            entropy = -jnp.sum(
                probabilities
                * jnp.log(jnp.maximum(probabilities, 1.0e-8)),
                axis=-1,
            ) / jnp.log(float(class_count))
            confidence = (1.0 - entropy)[..., None]
            prototypes = head.kernel.value.T.astype(jnp.float32)
            semantic = jnp.einsum('bc,ch->bh', centered, prototypes)
            states.append(confidence * semantic)
        if not states:
            raise ValueError('factorized semantic state needs at least one head')
        return (sum(states) / float(len(states))).astype(dtype)

    def compose_compositional_program_subgoals(
        self, ordered_subgoals, entries
    ):
        """Compose typed semantic factors before injecting them into phases.

        Each classifier distribution is centered around uniform and gated by
        confidence.  Consequently, absent evidence produces exact-zero factor
        tokens and an exact-zero phase residual even after training.  This
        prevents the graph from learning a task-frequency shortcut while still
        allowing operation/relation/condition interactions when evidence is
        present.
        """

        if ordered_subgoals.ndim != 3 or ordered_subgoals.shape[1:] != (
            self.subgoal_slots,
            self.hidden_dim,
        ):
            raise ValueError(
                'ordered subgoals differ from compositional program contract'
            )
        entries = tuple(entries)
        if len(entries) != 5:
            raise ValueError('compositional program requires five factors')

        factor_states = []
        factor_confidences = []
        identities = self.compositional_factor_identity(jnp.arange(5))
        for factor_index, (logits, head) in enumerate(entries):
            probabilities = jax.nn.softmax(
                logits.astype(jnp.float32), axis=-1
            )
            class_count = probabilities.shape[-1]
            centered = probabilities - (1.0 / float(class_count))
            entropy = -jnp.sum(
                probabilities
                * jnp.log(jnp.maximum(probabilities, 1.0e-8)),
                axis=-1,
            ) / jnp.log(float(class_count))
            confidence = jnp.where(
                jnp.any(centered != 0.0, axis=-1),
                jnp.clip(1.0 - entropy, 0.0, 1.0),
                0.0,
            )
            prototypes = head.kernel.value.T.astype(jnp.float32)
            semantic = jnp.einsum('bc,ch->bh', centered, prototypes)
            typed = semantic + identities[factor_index].astype(jnp.float32)
            factor_states.append(confidence[:, None] * typed)
            factor_confidences.append(confidence)

        factor_tokens = jnp.stack(factor_states, axis=1).astype(
            ordered_subgoals.dtype
        )
        factor_confidence = jnp.stack(factor_confidences, axis=1)
        for block in self.compositional_factor_blocks:
            factor_tokens = block(factor_tokens, factor_tokens)
        evidence = jnp.max(factor_confidence, axis=-1)
        factor_tokens = factor_tokens * evidence[:, None, None].astype(
            factor_tokens.dtype
        )

        phase_queries = self.compositional_phase_queries(
            jnp.arange(self.subgoal_slots)
        )[None]
        phase_tokens = ordered_subgoals + phase_queries.astype(
            ordered_subgoals.dtype
        )
        for block in self.compositional_phase_blocks:
            phase_tokens = block(phase_tokens, factor_tokens)
        program_state = self.compositional_phase_out(
            _rms_normalize(phase_tokens)
        )
        program_state = program_state * evidence[:, None, None].astype(
            program_state.dtype
        )
        return (
            ordered_subgoals + 0.25 * program_state,
            program_state,
            factor_tokens,
        )

    def ground_compositional_program_subgoals(
        self,
        ordered_subgoals,
        compositional_program_state,
        *,
        bound_roles,
        source_reference_bound,
        destination_reference_bound,
        condition_state_context,
        role_valid_mask,
        source_reference_valid,
        destination_reference_valid,
        condition_state_valid,
        visual_evidence_valid,
    ):
        """Close the visual binding loop into the physical phase program.

        The language program is intentionally composed before visual binding,
        but leaving it there makes grounded prepositions reach the policy only
        through a generic recurrent write.  Build signed target/destination,
        pickup-reference, destination-reference, and condition relations from
        the current grounded roles, then reuse the factorized phase adapter to
        update both the routed subgoals and the program consumed by downstream
        predictive/contact modules.  No teacher target or future observation
        enters this deployed path.

        Missing visual evidence produces an exact-zero residual.  This also
        preserves the inherited controller exactly because the existing PSM
        policy gain and downstream predictive gates initialize at zero.
        """

        batch_size = ordered_subgoals.shape[0]
        if ordered_subgoals.shape[1:] != (
            self.subgoal_slots,
            self.hidden_dim,
        ):
            raise ValueError(
                'ordered subgoals differ from grounded program contract'
            )
        if compositional_program_state.shape != ordered_subgoals.shape:
            raise ValueError(
                'compositional program differs from grounded program contract'
            )
        if bound_roles.shape != (batch_size, 2, self.hidden_dim):
            raise ValueError('bound roles must be [batch, 2, hidden]')
        if source_reference_bound.shape != (batch_size, self.hidden_dim):
            raise ValueError(
                'source reference must be [batch, hidden]'
            )
        if destination_reference_bound.ndim != 3 or (
            destination_reference_bound.shape[0] != batch_size
            or destination_reference_bound.shape[-1] != self.hidden_dim
        ):
            raise ValueError(
                'destination references must be [batch, reference, hidden]'
            )
        if condition_state_context.shape != (batch_size, self.hidden_dim):
            raise ValueError('condition state must be [batch, hidden]')
        if role_valid_mask.shape != (batch_size, 2):
            raise ValueError('role validity must be [batch, 2]')
        if source_reference_valid.shape != (batch_size,):
            raise ValueError('source-reference validity must be [batch]')
        if destination_reference_valid.shape != (
            batch_size,
            destination_reference_bound.shape[1],
        ):
            raise ValueError(
                'destination-reference validity differs from references'
            )
        if condition_state_valid.shape != (batch_size,):
            raise ValueError('condition-state validity must be [batch]')
        if visual_evidence_valid.shape != (batch_size,):
            raise ValueError('visual evidence validity must be [batch]')

        target = bound_roles[:, 0]
        destination = bound_roles[:, 1]
        target_destination_valid = jnp.all(role_valid_mask, axis=-1)
        source_valid = role_valid_mask[:, 0] & source_reference_valid
        destination_reference_any = jnp.any(
            destination_reference_valid, axis=-1
        )
        destination_valid = (
            role_valid_mask[:, 1] & destination_reference_any
        )
        destination_reference_count = jnp.sum(
            destination_reference_valid, axis=-1, keepdims=True
        ).astype(destination_reference_bound.dtype)
        destination_reference_mean = jnp.sum(
            destination_reference_bound
            * destination_reference_valid[..., None].astype(
                destination_reference_bound.dtype
            ),
            axis=1,
        ) / jnp.maximum(destination_reference_count, 1.0)

        relation_terms = jnp.stack(
            [
                jnp.where(
                    target_destination_valid[:, None],
                    target - destination,
                    jnp.zeros_like(target),
                ),
                jnp.where(
                    source_valid[:, None],
                    target - source_reference_bound,
                    jnp.zeros_like(target),
                ),
                jnp.where(
                    destination_valid[:, None],
                    destination - destination_reference_mean,
                    jnp.zeros_like(destination),
                ),
                jnp.where(
                    condition_state_valid[:, None],
                    condition_state_context,
                    jnp.zeros_like(condition_state_context),
                ),
            ],
            axis=1,
        )
        relation_valid = jnp.stack(
            [
                target_destination_valid,
                source_valid,
                destination_valid,
                condition_state_valid,
            ],
            axis=1,
        )
        evidence_count = jnp.sum(
            relation_valid, axis=-1, keepdims=True
        ).astype(relation_terms.dtype)
        grounded_relation = jnp.sum(relation_terms, axis=1) / jnp.maximum(
            evidence_count, 1.0
        )
        grounded_evidence_valid = visual_evidence_valid & (
            evidence_count[:, 0] > 0
        )
        grounded_relation = jnp.where(
            grounded_evidence_valid[:, None],
            self.role_in(_rms_normalize(grounded_relation)),
            jnp.zeros_like(grounded_relation),
        )
        grounded_subgoals, grounded_phase_state = (
            self.compose_factorized_relation_phase_subgoals(
                ordered_subgoals, grounded_relation
            )
        )
        grounded_phase_state = jnp.where(
            grounded_evidence_valid[:, None, None],
            grounded_phase_state,
            jnp.zeros_like(grounded_phase_state),
        )
        grounded_subgoals = jnp.where(
            grounded_evidence_valid[:, None, None],
            grounded_subgoals,
            ordered_subgoals,
        )
        return (
            grounded_subgoals,
            compositional_program_state + grounded_phase_state,
            grounded_phase_state,
        )

    def route(
        self,
        memory,
        ordered_subgoals,
        previous_frontier,
        episode_start,
        *,
        current_context,
        slot_valid_mask=None,
        clause_attention=None,
    ):
        previous_frontier = jax.lax.stop_gradient(previous_frontier)
        frontier = jnp.argmax(previous_frontier, axis=-1)
        following = jnp.minimum(frontier + 1, self.subgoal_slots - 1)
        indices = jnp.arange(self.subgoal_slots)[None]
        allowed = (indices == frontier[:, None]) | (indices == following[:, None])
        # Route at the current replan boundary, not one boundary late.  The
        # context contains only the current observation and already executed
        # previous action chunk, so this remains causal while allowing a newly
        # visible subgoal completion to advance the frontier immediately.
        routing_state = _rms_normalize(
            self.structured_memory_summary(memory) + current_context
        )
        logits = self.route_head(routing_state)
        causal_frontier_transition_in = getattr(
            self, 'causal_frontier_transition_in_v1', None
        )
        causal_frontier_transition_score = getattr(
            self, 'causal_frontier_transition_score_v1', None
        )
        if (causal_frontier_transition_in is None) != (
            causal_frontier_transition_score is None
        ):
            raise ValueError('causal frontier transition gate is incomplete')
        if causal_frontier_transition_in is not None:
            # Compare the currently active ordered subgoal with its only legal
            # successor using strictly causal current-memory/context evidence.
            # Adding the two scores before the reachability mask makes this a
            # direct learned stay/advance decision, not a post-hoc heuristic.
            current_subgoal = jnp.einsum(
                'bs,bsh->bh', previous_frontier, ordered_subgoals
            )
            next_frontier = jax.nn.one_hot(
                following,
                self.subgoal_slots,
                dtype=ordered_subgoals.dtype,
            )
            next_subgoal = jnp.einsum(
                'bs,bsh->bh', next_frontier, ordered_subgoals
            )
            transition_features = jnp.concatenate(
                (
                    routing_state,
                    current_subgoal,
                    next_subgoal,
                    next_subgoal - current_subgoal,
                ),
                axis=-1,
            )
            transition_scores = causal_frontier_transition_score(
                nnx.swish(causal_frontier_transition_in(transition_features))
            )
            logits = logits + jax.nn.one_hot(
                frontier, self.subgoal_slots, dtype=logits.dtype
            ) * transition_scores[:, :1]
            logits = logits + jax.nn.one_hot(
                following, self.subgoal_slots, dtype=logits.dtype
            ) * transition_scores[:, 1:]
        semantic_frontier_completion_state = None
        semantic_frontier_completion_verifier = getattr(
            self, 'semantic_frontier_completion_verifier_v1', None
        )
        if semantic_frontier_completion_verifier is not None:
            if slot_valid_mask is None or slot_valid_mask.shape != logits.shape:
                raise ValueError(
                    'semantic frontier completion requires a valid S8 slot mask'
                )
            logits, semantic_frontier_completion_state = (
                semantic_frontier_completion_verifier(
                    logits,
                    ordered_subgoals,
                    current_context,
                    self.structured_memory_summary(memory),
                    previous_frontier,
                    slot_valid_mask,
                )
            )
        hierarchical_clause_event_alignment_state = None
        hierarchical_clause_event_alignment = getattr(
            self, 'hierarchical_clause_event_alignment_v1', None
        )
        if hierarchical_clause_event_alignment is not None:
            if (
                clause_attention is None
                or slot_valid_mask is None
                or clause_attention.shape
                != (logits.shape[0], self.subgoal_slots, self.subgoal_slots)
                or slot_valid_mask.shape != logits.shape
            ):
                raise ValueError(
                    'HCEA requires ClausePlan S8 attention and a valid S8 slot mask'
                )
            logits, hierarchical_clause_event_alignment_state = (
                hierarchical_clause_event_alignment(
                    logits,
                    ordered_subgoals,
                    clause_attention,
                    current_context,
                    self.structured_memory_summary(memory),
                    previous_frontier,
                    slot_valid_mask,
                )
            )
        logits = jnp.where(allowed, logits, -1.0e30)
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        reset = self.initial_frontier(memory.shape[0])
        probabilities = jnp.where(episode_start[:, None], reset, probabilities)
        active = jnp.einsum(
            'bs,bsh->bh', probabilities.astype(ordered_subgoals.dtype), ordered_subgoals
        )
        return (
            active,
            probabilities,
            frontier,
            semantic_frontier_completion_state,
            hierarchical_clause_event_alignment_state,
        )

    def structured_memory_summary(self, memory):
        """Read causal memory without erasing reserved slot semantics."""

        target_anchor = memory[:, self.fast_tokens]
        reference_anchor = memory[:, self.fast_tokens + 1]
        verification = memory[:, -1]
        return (
            jnp.mean(memory, axis=1)
            + 0.25 * (target_anchor - reference_anchor)
            + 0.25 * verification
        )

    def current_context(
        self,
        *,
        prefix_tokens,
        prefix_mask,
        state,
        bound_roles,
        factorized_relation_state,
        condition_state_context=None,
        previous_actions,
        previous_actions_valid,
    ):
        """Summarize only information available at the current replan."""

        weights = prefix_mask.astype(prefix_tokens.dtype)[..., None]
        prefix_summary = jnp.sum(prefix_tokens * weights, axis=1) / jnp.maximum(
            jnp.sum(weights, axis=1), 1.0
        )
        previous_summary = self.executed_action_in(jnp.mean(previous_actions, axis=1))
        previous_summary = jnp.where(
            previous_actions_valid[:, None],
            previous_summary,
            jnp.zeros_like(previous_summary),
        )
        previous_phase_features = jnp.concatenate(
            [
                jnp.mean(previous_actions, axis=1),
                previous_actions[:, -1],
                previous_actions[:, -1] - previous_actions[:, 0],
                jnp.mean(jnp.abs(previous_actions), axis=1),
            ],
            axis=-1,
        )
        previous_phase_summary = nnx.swish(
            self.executed_action_phase_in(previous_phase_features)
        )
        previous_phase_summary = jnp.where(
            previous_actions_valid[:, None],
            previous_phase_summary,
            jnp.zeros_like(previous_phase_summary),
        )
        # Preserve the order of manipulated target and destination/reference.
        # A plain mean is invariant to swapping those roles and therefore loses
        # exactly the relational direction needed by compositional tasks.
        ordered_role_summary = (
            jnp.mean(bound_roles, axis=1)
            + 0.25 * (bound_roles[:, 0] - bound_roles[:, 1])
        )
        if condition_state_context is None:
            condition_state_context = jnp.zeros_like(factorized_relation_state)
        return (
            self.prefix_in(prefix_summary)
            + self.state_in(state)
            + self.role_in(ordered_role_summary)
            + factorized_relation_state
            + 0.25 * condition_state_context
            + previous_summary
            + previous_phase_summary
        )

    def factorized_relation_state(self, plan):
        """Close the loop from supervised relation heads into causal memory.

        Each classifier kernel doubles as a learned semantic prototype bank.
        Centered probabilities prevent an uninformative uniform prediction from
        injecting a class-average bias, and normalized entropy gates uncertain
        predictions.  No new parameters are required.
        """

        entries = []
        for logits_name, head in (
            ('operation_logits', self.operation_head),
            ('source_relation_logits', self.source_relation_head),
            ('destination_relation_logits', self.destination_relation_head),
            ('condition_logits', self.condition_head),
            ('destination_qualifier_logits', self.destination_qualifier_head),
        ):
            if logits_name not in plan:
                continue
            entries.append((plan[logits_name], head))
        return self._factorized_semantic_state_from_logits(
            tuple(entries), dtype=plan['bound_roles'].dtype
        )

    def update(
        self,
        memory,
        *,
        current_context,
        active_subgoal,
    ):
        context = current_context + self.subgoal_in(active_subgoal)
        positions = self.memory_position(jnp.arange(self.memory_tokens))[None]
        proposed_context = context[:, None] + positions
        write_input = jnp.concatenate(
            [_rms_normalize(memory), _rms_normalize(proposed_context)], axis=-1
        )
        gate = jax.nn.sigmoid(self.write_gate(write_input))
        candidate = jnp.tanh(
            self.write_candidate(write_input) + proposed_context
        )
        rates = jax.nn.sigmoid(self.memory_update_rate_logits.value)[None, :, None]
        updated = memory + rates * gate * (candidate - memory)
        # Reserve the first two slow tokens for target/reference identity and
        # the final slow token for causal plan verification.  Dedicated writers
        # below update them without generic-context identity dilution.
        reserved = jnp.zeros((self.memory_tokens,), dtype=jnp.bool_)
        reserved = reserved.at[self.fast_tokens : self.fast_tokens + 2].set(True)
        reserved = reserved.at[-1].set(True)
        return jnp.where(reserved[None, :, None], memory, updated)

    def role_identity_feedback(
        self,
        memory,
        *,
        bound_roles,
        role_valid_mask,
        role_write_confidence=None,
    ):
        """Persist target/reference identity in two dedicated slow tokens."""

        anchor_indices = jnp.arange(self.fast_tokens, self.fast_tokens + 2)
        previous_anchors = memory[:, anchor_indices]
        role_state = _rms_normalize(bound_roles)
        write_input = jnp.concatenate(
            [_rms_normalize(previous_anchors), role_state], axis=-1
        )
        gate = jax.nn.sigmoid(self.write_gate(write_input))
        candidate = jnp.tanh(self.write_candidate(write_input) + role_state)
        rates = jax.nn.sigmoid(
            self.memory_update_rate_logits.value[anchor_indices]
        )[None, :, None]
        write_scale = jnp.ones(
            previous_anchors.shape[:2], dtype=previous_anchors.dtype
        )
        confidence_gate = getattr(
            self, 'role_identity_confidence_gate', None
        )
        if confidence_gate is not None:
            if role_write_confidence is None or role_write_confidence.shape != (
                memory.shape[0], 2
            ):
                raise ValueError(
                    'confidence-gated identity writes require [batch, 2] confidence'
                )
            confidence = jnp.clip(
                role_write_confidence.astype(previous_anchors.dtype), 0.0, 1.0
            )
            # The new leaf is exactly zero at initialization, so the child is
            # function-identical to its Joint51 parent.  Training can open a
            # role-specific path that attenuates ambiguous later writes.
            # Magnitude parameterization is exact zero with a nonzero JAX
            # subgradient at initialization.  Either optimizer direction can
            # therefore open the gate, while neither direction can amplify a
            # low-confidence write above the inherited writer strength.
            learned_mix = jnp.abs(
                jnp.tanh(confidence_gate.value)
            ).astype(previous_anchors.dtype)
            write_scale = 1.0 - learned_mix[None, :] * (1.0 - confidence)
        updated = previous_anchors + write_scale[..., None] * rates * gate * (
            candidate - previous_anchors
        )
        updated = jnp.where(
            role_valid_mask[..., None], updated, previous_anchors
        )
        return memory.at[:, anchor_indices].set(updated), updated

    def verification_feedback(
        self,
        memory,
        *,
        frontier,
        transition_logits,
        progress,
        next_action_summary,
    ):
        """Write causal plan-verification predictions into one slow token.

        The transition distribution is restricted to the same stay/advance-one
        support used by the deployed route.  No future target enters this path:
        targets supervise the three prediction heads only in compute_loss_sequence.
        """

        current_slot = jnp.argmax(frontier, axis=-1)
        following = jnp.minimum(current_slot + 1, self.subgoal_slots - 1)
        indices = jnp.arange(self.subgoal_slots)[None]
        allowed = (indices == current_slot[:, None]) | (
            indices == following[:, None]
        )
        masked_logits = jnp.where(
            allowed, transition_logits.astype(jnp.float32), -1.0e30
        )
        transition_probabilities = jax.nn.softmax(masked_logits, axis=-1)
        verification_features = jnp.concatenate(
            [
                transition_probabilities.astype(memory.dtype),
                progress[:, None].astype(memory.dtype),
                next_action_summary.astype(memory.dtype),
            ],
            axis=-1,
        )
        verification_state = nnx.swish(
            self.verification_in(verification_features)
        )

        # The last token belongs to the audited slow bank (four fast + four
        # slow in production), making verification persist beyond local motion.
        previous_token = memory[:, -1]
        write_input = jnp.concatenate(
            [_rms_normalize(previous_token), _rms_normalize(verification_state)],
            axis=-1,
        )
        # Reuse the audited memory writer instead of introducing a second
        # 2H->H gate/candidate pair.  This keeps the verifier parameter-light
        # and trains one consistent update geometry for ordinary and verified
        # state writes.
        gate = jax.nn.sigmoid(self.write_gate(write_input))
        candidate = jnp.tanh(
            self.write_candidate(write_input) + verification_state
        )
        slow_rate = jax.nn.sigmoid(self.memory_update_rate_logits.value[-1])
        verified_token = previous_token + slow_rate * gate * (
            candidate - previous_token
        )
        return (
            memory.at[:, -1].set(verified_token),
            verification_state,
            transition_probabilities,
        )

    def infer_state(
        self,
        *,
        prefix_tokens,
        prefix_mask,
        state,
        memory,
        frontier,
        previous_actions,
        previous_actions_valid,
        episode_start,
        plan_token_mask=None,
        role_span_mask=None,
        source_reference_span_mask=None,
        destination_reference_span_mask=None,
        condition_state_span_mask=None,
        destination_qualifier_span_mask=None,
        role_valid_mask=None,
        clause_span_mask=None,
        clause_valid_mask=None,
        visual_tokens=None,
        camera_mask=None,
        compute_object_reconstruction=False,
        structured_demo=None,
        structured_demo_tokens=None,
        structured_demo_token_mask=None,
    ):
        reset_memory = self.initial_state(memory.shape[0], dtype=memory.dtype)
        memory = jnp.where(episode_start[:, None, None], reset_memory, memory)
        role_anchor_indices = jnp.arange(self.fast_tokens, self.fast_tokens + 2)
        role_memory_anchors = memory[:, role_anchor_indices]
        plan = self.bind_plan(
            prefix_tokens,
            prefix_mask,
            state,
            plan_token_mask=plan_token_mask,
            role_span_mask=role_span_mask,
            source_reference_span_mask=source_reference_span_mask,
            destination_reference_span_mask=destination_reference_span_mask,
            condition_state_span_mask=condition_state_span_mask,
            destination_qualifier_span_mask=(
                destination_qualifier_span_mask
            ),
            role_valid_mask=role_valid_mask,
            clause_span_mask=clause_span_mask,
            clause_valid_mask=clause_valid_mask,
            role_memory_anchors=role_memory_anchors,
            previous_actions=previous_actions,
            previous_actions_valid=previous_actions_valid,
            episode_start=episode_start,
            visual_tokens=visual_tokens,
            camera_mask=camera_mask,
            compute_object_reconstruction=compute_object_reconstruction,
            structured_demo=structured_demo,
            structured_demo_tokens=structured_demo_tokens,
            structured_demo_token_mask=structured_demo_token_mask,
        )
        ordered_subgoals = plan['ordered_subgoals']
        bound_roles = plan['bound_roles']
        factorized_relation_state = self.factorized_relation_state(plan)
        clause_role_binding_verifier_state = None
        clause_role_binding_verifier = getattr(
            self, 'clause_role_binding_verifier_v1', None
        )
        if clause_role_binding_verifier is not None:
            if plan['clause_plan_attention'] is None or clause_valid_mask is None:
                raise ValueError(
                    'clause-role binding verifier requires ClausePlan attention '
                    'and current-clause validity'
                )
            reset_frontier = jax.nn.one_hot(
                jnp.zeros((frontier.shape[0],), dtype=jnp.int32),
                self.subgoal_slots,
                dtype=frontier.dtype,
            )
            binding_frontier = jnp.where(
                episode_start[:, None], reset_frontier, frontier
            )
            (
                ordered_subgoals,
                clause_role_binding_verifier_state,
            ) = clause_role_binding_verifier(
                ordered_subgoals,
                plan['clause_plan_attention'],
                binding_frontier,
                bound_roles,
                factorized_relation_state,
                clause_valid_mask,
                plan['role_valid_mask'],
            )
            clause_role_binding_verifier_state = {
                **clause_role_binding_verifier_state,
                'input_plan_slots': plan['ordered_subgoals'],
                'frontier': binding_frontier,
                'role_tokens': bound_roles,
                'relation_token': factorized_relation_state,
            }
        current_context = self.current_context(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            state=state,
            bound_roles=bound_roles,
            factorized_relation_state=factorized_relation_state,
            condition_state_context=plan['condition_state_context'],
            previous_actions=previous_actions,
            previous_actions_valid=previous_actions_valid,
        )
        (
            active,
            next_frontier,
            prior_frontier,
            semantic_frontier_completion_state,
            hierarchical_clause_event_alignment_state,
        ) = self.route(
            memory,
            ordered_subgoals,
            frontier,
            episode_start,
            current_context=current_context,
            slot_valid_mask=clause_valid_mask,
            clause_attention=plan['clause_plan_attention'],
        )
        provisional_memory = self.update(
            memory,
            current_context=current_context,
            active_subgoal=active,
        )
        role_anchored_memory, role_identity_anchors = self.role_identity_feedback(
            provisional_memory,
            bound_roles=bound_roles,
            role_valid_mask=plan['role_valid_mask'],
            role_write_confidence=plan['role_identity_write_confidence'],
        )
        pooled = self.structured_memory_summary(role_anchored_memory)
        # Slow memory supplies persistence, while the direct causal context
        # makes interaction boundaries visible without waiting one replan for
        # the memory writer to integrate them.
        phase_evidence_state = _rms_normalize(pooled + 0.5 * current_context)
        transition_logits = self.transition_head(phase_evidence_state)
        progress = jax.nn.sigmoid(self.progress_head(phase_evidence_state)[..., 0])
        next_action_summary = jnp.tanh(self.next_action_head(phase_evidence_state))
        (
            next_memory,
            verification_state,
            verification_transition_probabilities,
        ) = self.verification_feedback(
            role_anchored_memory,
            frontier=next_frontier,
            transition_logits=transition_logits,
            progress=progress,
            next_action_summary=next_action_summary,
        )
        return {
            'memory': next_memory,
            'frontier': next_frontier,
            'prior_frontier': prior_frontier,
            'ordered_subgoals': ordered_subgoals,
            'slot_valid_mask': (
                jnp.ones(ordered_subgoals.shape[:2], dtype=jnp.bool_)
                if clause_valid_mask is None
                else clause_valid_mask.astype(jnp.bool_)
            ),
            'clause_plan_residual': plan['clause_plan_residual'],
            'clause_plan_attention': plan['clause_plan_attention'],
            'structured_demo_tokens': plan['structured_demo_tokens'],
            'structured_demo_token_mask': plan[
                'structured_demo_token_mask'
            ],
            'structured_demo_shared_context': plan[
                'structured_demo_shared_context'
            ],
            'language_ordered_subgoals': plan[
                'language_ordered_subgoals'
            ],
            'semantic_phase_state': plan['semantic_phase_state'],
            'operation_phase_state': plan['operation_phase_state'],
            'relation_phase_state': plan['relation_phase_state'],
            'grounded_relation_phase_state': plan[
                'grounded_relation_phase_state'
            ],
            'compositional_program_state': plan[
                'compositional_program_state'
            ],
            'language_compositional_program_state': plan[
                'language_compositional_program_state'
            ],
            'compositional_factor_tokens': plan[
                'compositional_factor_tokens'
            ],
            'bound_roles': bound_roles,
            'language_roles': plan['language_roles'],
            'role_span_teacher': plan['role_span_teacher'],
            'factor_attention_alignment_loss': plan[
                'factor_attention_alignment_loss'
            ],
            'factor_attention_valid': plan['factor_attention_valid'],
            'object_slot_reconstruction_loss': plan[
                'object_slot_reconstruction_loss'
            ],
            'source_reference_language': plan['source_reference_language'],
            'source_reference_teacher': plan['source_reference_teacher'],
            'source_reference_valid': plan['source_reference_valid'],
            'source_reference_camera_bound': plan[
                'source_reference_camera_bound'
            ],
            'source_reference_camera_weights': plan[
                'source_reference_camera_weights'
            ],
            'source_reference_bound': plan['source_reference_bound'],
            'source_reference_centroid': plan['source_reference_centroid'],
            'source_reference_overlap': plan['source_reference_overlap'],
            'destination_reference_language': plan[
                'destination_reference_language'
            ],
            'destination_reference_teacher': plan[
                'destination_reference_teacher'
            ],
            'destination_reference_valid': plan['destination_reference_valid'],
            'destination_reference_camera_bound': plan[
                'destination_reference_camera_bound'
            ],
            'destination_reference_camera_weights': plan[
                'destination_reference_camera_weights'
            ],
            'destination_reference_bound': plan['destination_reference_bound'],
            'destination_reference_centroids': plan[
                'destination_reference_centroids'
            ],
            'destination_reference_overlap': plan[
                'destination_reference_overlap'
            ],
            'condition_state_language': plan['condition_state_language'],
            'condition_state_teacher': plan['condition_state_teacher'],
            'condition_state_valid': plan['condition_state_valid'],
            'condition_state_camera_bound': plan[
                'condition_state_camera_bound'
            ],
            'condition_state_camera_weights': plan[
                'condition_state_camera_weights'
            ],
            'condition_state_bound': plan['condition_state_bound'],
            'condition_state_centroid': plan['condition_state_centroid'],
            'condition_state_overlap': plan['condition_state_overlap'],
            'condition_state_context': plan['condition_state_context'],
            'destination_qualifier_language': plan[
                'destination_qualifier_language'
            ],
            'destination_qualifier_teacher': plan[
                'destination_qualifier_teacher'
            ],
            'destination_qualifier_valid': plan[
                'destination_qualifier_valid'
            ],
            'camera_bound_roles': plan['camera_bound_roles'],
            'temporal_role_memory_state': plan[
                'temporal_role_memory_state'
            ],
            'cross_view_role_consensus_state': plan[
                'cross_view_role_consensus_state'
            ],
            'contact_risk_calibrated_role_residual_state': plan[
                'contact_risk_calibrated_role_residual_state'
            ],
            'relational_role_composer_residual_state': plan[
                'relational_role_composer_residual_state'
            ],
            'clause_role_binding_verifier_state': (
                clause_role_binding_verifier_state
            ),
            'semantic_frontier_completion_state': (
                semantic_frontier_completion_state
            ),
            'hierarchical_clause_event_alignment_state': (
                hierarchical_clause_event_alignment_state
            ),
            # Preserve both sides of the causal replan transition for an
            # optional successor action expert. These are current/past model
            # states, never future labels or evaluator state.
            'previous_frontier_distribution': frontier,
            'current_context': current_context,
            'previous_actions': previous_actions[..., :7],
            'previous_actions_valid': (
                previous_actions_valid.astype(jnp.bool_)
                & ~episode_start.astype(jnp.bool_)
            ),
            'camera_role_weights': plan['camera_role_weights'],
            'role_object_overlap': plan['role_object_overlap'],
            'role_assignment_entropy': plan['role_assignment_entropy'],
            'between_relation_probability': plan['between_relation_probability'],
            'role_centroids': plan['role_centroids'],
            'camera_mask': plan['camera_mask'],
            'role_valid_mask': plan['role_valid_mask'],
            'factorized_relation_state': factorized_relation_state,
            'role_identity_anchors': role_identity_anchors,
            'role_identity_write_confidence': plan[
                'role_identity_write_confidence'
            ],
            'operation_logits': plan['operation_logits'],
            'source_relation_logits': plan['source_relation_logits'],
            'destination_relation_logits': plan['destination_relation_logits'],
            'condition_logits': plan['condition_logits'],
            'destination_qualifier_logits': plan[
                'destination_qualifier_logits'
            ],
            # This auxiliary future-transition predictor is deliberately
            # separate from route_head.  route_head is reserved for the exact
            # deployed current-context stay/advance decision in route().
            'transition_logits': transition_logits,
            'progress': progress,
            'next_action_summary': next_action_summary,
            'phase_evidence_state': phase_evidence_state,
            'verification_state': verification_state,
            'verification_transition_probabilities': (
                verification_transition_probabilities
            ),
        }

    def inject(
        self,
        action_tokens,
        memory,
        *,
        ordered_program=None,
        frontier=None,
        policy_scale=1.0,
    ):
        query = self.read_query(_rms_normalize(action_tokens))
        # Preserve stable fast/target/reference/verification slot roles in the
        # deployed read.  The exact-zero scalar gain still preserves the
        # inherited action function before this path is trained open.
        structured_memory = memory + self.memory_position(
            jnp.arange(self.memory_tokens)
        )[None, :, :]
        if (ordered_program is None) != (frontier is None):
            raise ValueError(
                'ordered program and frontier must be provided together'
            )
        if ordered_program is not None:
            if (
                ordered_program.ndim != 3
                or ordered_program.shape[0] != action_tokens.shape[0]
                or ordered_program.shape[1] != self.subgoal_slots
                or ordered_program.shape[2] != self.hidden_dim
            ):
                raise ValueError(
                    'ordered program must be [batch, subgoal, hidden]'
                )
            if frontier.shape != ordered_program.shape[:2]:
                raise ValueError('frontier differs from ordered program')
            # Give the action expert a direct causal view of the execution
            # program instead of forcing all phase information through one
            # recursive memory write.  The frontier identifies the phase to
            # execute now; a small one-step lookahead exposes the transition
            # target without allowing a skip or a wrap at the terminal phase.
            next_frontier = jnp.concatenate(
                [jnp.zeros_like(frontier[:, :1]), frontier[:, :-1]], axis=-1
            )
            next_frontier = next_frontier.at[:, -1].add(frontier[:, -1])
            phase_weights = 0.75 * frontier + 0.25 * next_frontier
            phase_program = jnp.einsum(
                'bs,bsh->bh',
                phase_weights.astype(ordered_program.dtype),
                ordered_program,
            )
            # Local execution alone is insufficient for the benchmark's
            # multi-stage long-horizon workflows.  Add a second, parameter-
            # free token that summarizes only the current and remaining plan,
            # with monotonically decaying weight by phase distance.  A lower-
            # triangular (past-looking) contribution is exactly zero, while
            # the terminal goal remains visible from the first phase.
            phase_ids = jnp.arange(self.subgoal_slots, dtype=jnp.float32)
            phase_distance = phase_ids[None, :] - phase_ids[:, None]
            remaining_kernel = jnp.where(
                phase_distance >= 0.0,
                jnp.power(0.75, phase_distance),
                0.0,
            )
            remaining_weights = jnp.einsum(
                'bs,sr->br', frontier.astype(jnp.float32), remaining_kernel
            )
            remaining_weights = remaining_weights / jnp.maximum(
                jnp.sum(remaining_weights, axis=-1, keepdims=True), 1.0e-8
            )
            remaining_program = jnp.einsum(
                'bs,bsh->bh',
                remaining_weights.astype(ordered_program.dtype),
                ordered_program,
            )
            structured_memory = jnp.concatenate(
                [
                    structured_memory,
                    phase_program[:, None, :],
                    remaining_program[:, None, :],
                ],
                axis=1,
            )
        key = self.read_key(_rms_normalize(structured_memory))
        value = self.read_value(structured_memory)
        logits = jnp.einsum(
            'bth,bmh->btm', query, key, preferred_element_type=jnp.float32
        ) / jnp.sqrt(float(self.hidden_dim))
        attention = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        read = jnp.einsum('btm,bmh->bth', attention, value)
        residual = self.read_out(_rms_normalize(read))
        parent_output = (
            action_tokens
            + jnp.asarray(policy_scale, dtype=jnp.float32)
            * _persistent_memory_policy_gain(
                self.memory_to_policy_gain.value,
                bounded=self.bounded_policy_gain,
            )
            * residual
        )
        bridge = getattr(self, 'conditional_memory_policy_bridge', None)
        if bridge is None:
            return parent_output
        return bridge(
            parent_output,
            query,
            memory,
            ordered_program=ordered_program,
            frontier=frontier,
            policy_scale=policy_scale,
        )


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.grounded_demonstration_camera_context_only = (
            config.grounded_demonstration_camera_context_only
        )
        self.active_action_dim = config.active_action_dim or config.action_dim
        self.main_flow_samples = config.main_flow_samples
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method='init',
            use_adarms=[False, True] if config.pi05 else [False, False],
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant='So400m/14',
                pool_type='none',
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(
            next(iter(config.fake_obs().images.values())),
            train=False,
            rngs=rngs,
        )
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(
            config.action_dim, action_expert_config.width, rngs=rngs
        )
        if config.hierarchical_event_transition_memory:
            self.hetm = _hetm.HierarchicalEventTransitionMemory(
                prefix_dim=paligemma_config.width,
                state_dim=config.action_dim,
                action_hidden_dim=action_expert_config.width,
                rngs=rngs,
            )
            self.hetm_loss_weights = {
                'event': config.hetm_event_loss_weight,
                'predicate': config.hetm_predicate_loss_weight,
                'frontier': config.hetm_frontier_loss_weight,
                'transition': config.hetm_transition_loss_weight,
                'monotonic': config.hetm_monotonic_loss_weight,
            }
            self.hetm_psm_frontier_consistency_loss_weight = (
                config.hetm_psm_frontier_consistency_loss_weight
            )
        if config.role_affordance_causal_graph:
            self.racg = _racg.RoleAffordanceCausalGraph(
                _racg.RACGConfig(
                    patch_dim=paligemma_config.width,
                    language_dim=paligemma_config.width,
                    state_dim=config.action_dim,
                    action_hidden_dim=action_expert_config.width,
                    active_action_dim=self.active_action_dim,
                    action_positions=config.action_horizon,
                ),
                rngs=rngs,
            )
            self.racg_loss_weights = {
                'role_align': config.racg_role_align_loss_weight,
                'relation': config.racg_relation_loss_weight,
                'crossview': config.racg_crossview_loss_weight,
                'slot_reconstruction': config.racg_slot_reconstruction_loss_weight,
                'contact': config.racg_contact_loss_weight,
                'identity_transition': config.racg_identity_transition_loss_weight,
                'slot_diversity': config.racg_slot_diversity_loss_weight,
            }
            if config.racg_external_geometry_prior:
                self.racg_external_geometry = _racg_external.ExternalGeometryRolePrior(
                    _racg_external.ExternalGeometryConfig(
                        prefix_dim=paligemma_config.width,
                        hidden_dim=_hetm.HIDDEN_DIM,
                    ),
                    rngs=rngs,
                )
                if config.racg_external_geometry_hmca:
                    self.racg_external_geometry_hmca = (
                        _racg_external_hmca.ExternalGeometryHMCABridge(
                            hidden_dim=config.racg_external_geometry_hmca_hidden_dim,
                            rngs=rngs,
                        )
                    )
            if config.racg_graph_hmca:
                self.racg_graph_hmca = _racg_graph_hmca.RACGGraphHMCABridge(
                    hidden_dim=config.racg_graph_hmca_hidden_dim,
                    rngs=rngs,
                )
        if config.pi05:
            self.time_mlp_in = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            self.time_mlp_out = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            if config.state_adarms:
                self.state_adarms_in = nnx.Linear(
                    config.action_dim,
                    config.state_adarms_hidden_dim,
                    rngs=rngs,
                )
                self.state_adarms_out = nnx.Linear(
                    config.state_adarms_hidden_dim,
                    action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            if config.context_adarms:
                self.context_adarms_in = nnx.Linear(
                    config.action_prior_hidden_dim,
                    config.context_adarms_hidden_dim,
                    rngs=rngs,
                )
                self.context_adarms_out = nnx.Linear(
                    config.context_adarms_hidden_dim,
                    action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            # The best-anchor successor uses the contextual action-prior states
            # already present in its parent, without importing the much larger
            # dual-action-reasoner branch.  Keep the legacy dual configuration
            # below intact while allowing this refiner to stand alone.
            if config.velocity_refiner and not config.dual_action_reasoner:
                refiner_dim = config.velocity_refiner_hidden_dim
                self.velocity_refiner_loss_weight = (
                    config.velocity_refiner_loss_weight
                )
                self.velocity_refiner_action_in = nnx.Linear(
                    3 * config.action_dim, refiner_dim, rngs=rngs
                )
                self.velocity_refiner_hidden_in = nnx.Linear(
                    action_expert_config.width, refiner_dim, rngs=rngs
                )
                self.velocity_refiner_context_in = nnx.Linear(
                    config.action_prior_hidden_dim,
                    refiner_dim,
                    rngs=rngs,
                )
                self.velocity_refiner_state_in = nnx.Linear(
                    config.action_dim, refiner_dim, rngs=rngs
                )
                self.velocity_refiner_time_in = nnx.Linear(
                    refiner_dim, refiner_dim, rngs=rngs
                )
                self.velocity_refiner_position = nnx.Embed(
                    config.action_horizon, refiner_dim, rngs=rngs
                )
                self.velocity_refiner_blocks = [
                    _ExplicitActionReasonerBlock(
                        refiner_dim,
                        config.velocity_refiner_num_heads,
                        config.velocity_refiner_mlp_dim,
                        rngs=rngs,
                    )
                    for _ in range(config.velocity_refiner_layers)
                ]
                self.velocity_refiner_predict = nnx.Linear(
                    refiner_dim, config.action_dim, rngs=rngs
                )
                self.velocity_refiner_gain = nnx.Param(
                    jnp.zeros((config.action_dim,), dtype=jnp.float32)
                )
            if (
                config.language_subgoal_reasoner
                and not config.dual_action_reasoner
            ):
                subgoal_dim = config.language_subgoal_hidden_dim
                self.language_subgoal_slot_count = config.language_subgoal_slots
                self.language_subgoal_temperature = (
                    config.language_subgoal_temperature
                )
                self.language_subgoal_progress_loss_weight = (
                    config.language_subgoal_progress_loss_weight
                )
                self.language_subgoal_action_loss_weight = (
                    config.language_subgoal_action_loss_weight
                )
                self.language_subgoal_phase_class_weights = (
                    config.language_subgoal_phase_class_weights
                )
                self.language_subgoal_context_in = nnx.Linear(
                    config.action_prior_hidden_dim,
                    subgoal_dim,
                    rngs=rngs,
                )
                self.language_subgoal_prefix_in = nnx.Linear(
                    paligemma_config.width,
                    subgoal_dim,
                    rngs=rngs,
                )
                self.language_subgoal_prefix_query = nnx.Linear(
                    subgoal_dim,
                    subgoal_dim,
                    rngs=rngs,
                )
                self.language_subgoal_state_in = nnx.Linear(
                    config.action_dim, subgoal_dim, rngs=rngs
                )
                self.language_subgoal_slot_queries = nnx.Embed(
                    config.language_subgoal_slots,
                    subgoal_dim,
                    rngs=rngs,
                )
                self.language_subgoal_slot_positions = nnx.Embed(
                    config.language_subgoal_slots,
                    subgoal_dim,
                    rngs=rngs,
                )
                self.language_subgoal_blocks = [
                    _ExplicitActionReasonerBlock(
                        subgoal_dim,
                        config.language_subgoal_num_heads,
                        config.language_subgoal_mlp_dim,
                        rngs=rngs,
                    )
                    for _ in range(config.language_subgoal_layers)
                ]
                self.language_subgoal_score = nnx.Linear(
                    subgoal_dim, 1, rngs=rngs
                )
                self.language_subgoal_action_queries = nnx.Embed(
                    config.action_horizon,
                    subgoal_dim,
                    rngs=rngs,
                )
                self.language_subgoal_action_block = (
                    _ExplicitActionReasonerBlock(
                        subgoal_dim,
                        config.language_subgoal_num_heads,
                        config.language_subgoal_mlp_dim,
                        rngs=rngs,
                    )
                )
                self.language_subgoal_action_predict = nnx.Linear(
                    subgoal_dim, config.action_dim, rngs=rngs
                )
                self.language_subgoal_token_out = nnx.Linear(
                    subgoal_dim,
                    action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            if config.specialist_module_router:
                router_dim = config.specialist_module_router_hidden_dim
                self.specialist_module_router_temperature = (
                    config.specialist_module_router_temperature
                )
                self.specialist_module_router_balance_loss_weight = (
                    config.specialist_module_router_balance_loss_weight
                )
                self.specialist_module_router_context_in = nnx.Linear(
                    config.action_prior_hidden_dim,
                    router_dim,
                    rngs=rngs,
                )
                self.specialist_module_router_state_in = nnx.Linear(
                    config.action_dim,
                    router_dim,
                    rngs=rngs,
                )
                self.specialist_module_router_score = nnx.Linear(
                    router_dim,
                    3,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            if config.persistent_memory_adarms:
                self.persistent_memory_adarms_in = nnx.Linear(
                    config.persistent_memory_hidden_dim,
                    config.persistent_memory_adarms_hidden_dim,
                    rngs=rngs,
                )
                self.persistent_memory_adarms_out = nnx.Linear(
                    config.persistent_memory_adarms_hidden_dim,
                    action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            if config.phase_contact_action_film:
                self.phase_contact_film_memory_in = nnx.Linear(
                    config.persistent_memory_hidden_dim,
                    config.phase_contact_action_film_hidden_dim,
                    rngs=rngs,
                )
                self.phase_contact_film_contact_in = nnx.Linear(
                    action_expert_config.width,
                    config.phase_contact_action_film_hidden_dim,
                    rngs=rngs,
                )
                self.phase_contact_film_out = nnx.Linear(
                    config.phase_contact_action_film_hidden_dim,
                    2 * action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            if config.layerwise_persistent_memory_attention:
                self.layerwise_persistent_memory_attention = (
                    _memory_attention_nnx.LayerwisePersistentMemoryAttention(
                        model_dim=action_expert_config.width,
                        memory_dim=config.persistent_memory_hidden_dim,
                        rank=config.layerwise_persistent_memory_attention_rank,
                        alpha=config.layerwise_persistent_memory_attention_alpha,
                        rngs=rngs,
                    )
                )
            if config.state_action_film:
                self.state_film_in = nnx.Linear(
                    config.action_dim,
                    config.state_action_film_hidden_dim,
                    rngs=rngs,
                )
                self.state_film_out = nnx.Linear(
                    config.state_action_film_hidden_dim,
                    2 * action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            if config.action_prior:
                self.action_prior_horizon = config.action_prior_horizon
                self.action_prior_loss_weight = config.action_prior_loss_weight
                self.action_prior_contextual = config.action_prior_contextual
                self.action_prior_target = config.action_prior_target
                self.action_prior_queries = nnx.Embed(
                    config.action_prior_horizon,
                    config.action_prior_hidden_dim,
                    rngs=rngs,
                )
                self.action_prior_key = nnx.Linear(
                    paligemma_config.width,
                    config.action_prior_hidden_dim,
                    rngs=rngs,
                )
                self.action_prior_value = nnx.Linear(
                    paligemma_config.width,
                    config.action_prior_hidden_dim,
                    rngs=rngs,
                )
                if config.action_prior_state_conditioning:
                    self.action_prior_state = nnx.Linear(
                        config.action_dim,
                        config.action_prior_hidden_dim,
                        rngs=rngs,
                    )
                if config.persistent_action_prior_conditioning:
                    self.persistent_action_prior = nnx.Linear(
                        config.persistent_memory_hidden_dim,
                        config.action_prior_hidden_dim,
                        kernel_init=nnx.initializers.zeros_init(),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                self.action_prior_token_out = nnx.Linear(
                    config.action_prior_hidden_dim,
                    action_expert_config.width,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
                self.action_prior_action_out = nnx.Linear(
                    config.action_prior_hidden_dim,
                    config.action_dim,
                    kernel_init=nnx.initializers.zeros_init(),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
                if config.multimodal_prefix_moe:
                    self.prefix_moe_expert_count = (
                        config.multimodal_prefix_moe_num_experts
                    )
                    self.prefix_moe_top_k = config.multimodal_prefix_moe_top_k
                    self.prefix_moe_temperature = (
                        config.multimodal_prefix_moe_temperature
                    )
                    self.prefix_moe_balance_loss_weight = (
                        config.multimodal_prefix_moe_balance_loss_weight
                    )
                    self.prefix_moe_router_in = nnx.Linear(
                        paligemma_config.width,
                        config.multimodal_prefix_moe_hidden_dim,
                        rngs=rngs,
                    )
                    self.prefix_moe_state_in = nnx.Linear(
                        config.action_dim,
                        config.multimodal_prefix_moe_hidden_dim,
                        rngs=rngs,
                    )
                    self.prefix_moe_router_out = nnx.Linear(
                        config.multimodal_prefix_moe_hidden_dim,
                        config.multimodal_prefix_moe_num_experts,
                        rngs=rngs,
                    )
                    self.prefix_moe_expert_in = [
                        nnx.Linear(
                            paligemma_config.width,
                            config.multimodal_prefix_moe_expert_dim,
                            rngs=rngs,
                        )
                        for _ in range(config.multimodal_prefix_moe_num_experts)
                    ]
                    self.prefix_moe_expert_out = [
                        nnx.Linear(
                            config.multimodal_prefix_moe_expert_dim,
                            4 * paligemma_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        for _ in range(config.multimodal_prefix_moe_num_experts)
                    ]
                if config.layerwise_kv_moe:
                    self.kv_moe_layer_count = paligemma_config.depth
                    self.kv_moe_head_count = paligemma_config.num_kv_heads
                    self.kv_moe_head_dim = paligemma_config.head_dim
                    self.kv_moe_expert_count = (
                        config.layerwise_kv_moe_num_experts
                    )
                    self.kv_moe_top_k = config.layerwise_kv_moe_top_k
                    self.kv_moe_temperature = (
                        config.layerwise_kv_moe_temperature
                    )
                    self.kv_moe_balance_loss_weight = (
                        config.layerwise_kv_moe_balance_loss_weight
                    )
                    self.kv_moe_router_in = nnx.Linear(
                        paligemma_config.width,
                        config.layerwise_kv_moe_hidden_dim,
                        rngs=rngs,
                    )
                    self.kv_moe_state_in = nnx.Linear(
                        config.action_dim,
                        config.layerwise_kv_moe_hidden_dim,
                        rngs=rngs,
                    )
                    self.kv_moe_router_out = nnx.Linear(
                        config.layerwise_kv_moe_hidden_dim,
                        config.layerwise_kv_moe_num_experts,
                        rngs=rngs,
                    )
                    self.kv_moe_expert_in = [
                        nnx.Linear(
                            paligemma_config.width,
                            config.layerwise_kv_moe_expert_dim,
                            rngs=rngs,
                        )
                        for _ in range(config.layerwise_kv_moe_num_experts)
                    ]
                    kv_dim = (
                        paligemma_config.num_kv_heads
                        * paligemma_config.head_dim
                    )
                    self.kv_moe_expert_out = [
                        nnx.Linear(
                            config.layerwise_kv_moe_expert_dim,
                            4 * paligemma_config.depth * kv_dim,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        for _ in range(config.layerwise_kv_moe_num_experts)
                    ]
                if config.dual_action_reasoner:
                    self.action_prior_implicit_layers = (
                        config.implicit_action_reasoner_layers
                    )
                    self.action_prior_implicit_pool_stride = (
                        config.implicit_action_reasoner_pool_stride
                    )
                    kv_dim = paligemma_config.num_kv_heads * paligemma_config.head_dim
                    self.action_prior_implicit_key = nnx.Linear(
                        kv_dim,
                        config.action_prior_hidden_dim,
                        rngs=rngs,
                    )
                    self.action_prior_implicit_value = nnx.Linear(
                        kv_dim,
                        config.action_prior_hidden_dim,
                        rngs=rngs,
                    )
                    self.action_prior_implicit_layer_mix = nnx.Embed(
                        config.action_prior_horizon,
                        len(config.implicit_action_reasoner_layers),
                        rngs=rngs,
                    )
                    self.action_prior_implicit_out = nnx.Linear(
                        config.action_prior_hidden_dim,
                        config.action_prior_hidden_dim,
                        kernel_init=nnx.initializers.zeros_init(),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                    if config.implicit_action_reasoner_layerwise_guidance:
                        layer_count = len(config.implicit_action_reasoner_layers)
                        group_count = (
                            layer_count
                            // config.implicit_action_reasoner_group_size
                        )
                        downsample_dim = (
                            config.implicit_action_reasoner_downsample_dim
                        )
                        self.action_prior_implicit_group_size = (
                            config.implicit_action_reasoner_group_size
                        )
                        self.action_prior_implicit_num_heads = (
                            config.implicit_action_reasoner_num_heads
                        )
                        self.action_prior_implicit_layer_queries = nnx.Embed(
                            layer_count, kv_dim, rngs=rngs
                        )
                        self.action_prior_implicit_group_query = [
                            nnx.Linear(kv_dim, downsample_dim, rngs=rngs)
                            for _ in range(group_count)
                        ]
                        self.action_prior_implicit_group_key = [
                            nnx.Linear(kv_dim, downsample_dim, rngs=rngs)
                            for _ in range(group_count)
                        ]
                        self.action_prior_implicit_group_value = [
                            nnx.Linear(kv_dim, downsample_dim, rngs=rngs)
                            for _ in range(group_count)
                        ]
                        self.action_prior_implicit_group_out = [
                            nnx.Linear(
                                downsample_dim,
                                config.action_prior_hidden_dim,
                                rngs=rngs,
                            )
                            for _ in range(group_count)
                        ]
                        self.action_prior_implicit_layer_to_action = nnx.Linear(
                            config.action_prior_hidden_dim,
                            action_expert_config.width,
                            rngs=rngs,
                        )

                    explicit_dim = config.explicit_action_reasoner_hidden_dim
                    self.action_prior_explicit_loss_weight = (
                        config.explicit_action_reasoner_loss_weight
                    )
                    self.action_prior_explicit_flow_samples = (
                        config.explicit_action_reasoner_flow_samples
                    )
                    self.action_prior_explicit_teacher_forcing = (
                        config.explicit_action_reasoner_teacher_forcing
                    )
                    self.action_prior_explicit_inference_steps = (
                        config.explicit_action_reasoner_inference_steps
                    )
                    self.action_prior_explicit_action_in = nnx.Linear(
                        config.action_dim, explicit_dim, rngs=rngs
                    )
                    self.action_prior_explicit_context_in = nnx.Linear(
                        config.action_prior_hidden_dim,
                        explicit_dim,
                        rngs=rngs,
                    )
                    self.action_prior_explicit_waypoints = nnx.Embed(
                        config.action_prior_horizon,
                        explicit_dim,
                        rngs=rngs,
                    )
                    self.action_prior_explicit_time_in = nnx.Linear(
                        explicit_dim, explicit_dim, rngs=rngs
                    )
                    self.action_prior_explicit_time_out = nnx.Linear(
                        explicit_dim, explicit_dim, rngs=rngs
                    )
                    self.action_prior_explicit_blocks = [
                        _ExplicitActionReasonerBlock(
                            explicit_dim,
                            config.explicit_action_reasoner_num_heads,
                            config.explicit_action_reasoner_mlp_dim,
                            rngs=rngs,
                        )
                        for _ in range(config.explicit_action_reasoner_layers)
                    ]
                    self.action_prior_explicit_velocity_out = nnx.Linear(
                        explicit_dim,
                        config.action_dim,
                        kernel_init=nnx.initializers.zeros_init(),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                    self.action_prior_explicit_token_out = nnx.Linear(
                        explicit_dim,
                        action_expert_config.width,
                        kernel_init=nnx.initializers.zeros_init(),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                    # The additive prior above preserves the inherited policy.
                    # This compact residual implements ACoT's stronger fusion:
                    # current noisy action tokens query the distinct EAR/IAR
                    # guidance sequences before entering the action expert.
                    self.action_prior_guidance_action_in = nnx.Linear(
                        action_expert_config.width, explicit_dim, rngs=rngs
                    )
                    self.action_prior_guidance_prior_in = nnx.Linear(
                        action_expert_config.width, explicit_dim, rngs=rngs
                    )
                    self.action_prior_guidance_block = _ExplicitActionReasonerBlock(
                        explicit_dim,
                        config.explicit_action_reasoner_num_heads,
                        config.explicit_action_reasoner_mlp_dim,
                        rngs=rngs,
                    )
                    self.action_prior_guidance_out = nnx.Linear(
                        explicit_dim,
                        action_expert_config.width,
                        kernel_init=nnx.initializers.zeros_init(),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                    if config.reasoning_pathway_router:
                        router_dim = config.reasoning_pathway_router_hidden_dim
                        self.action_prior_pathway_count = 4
                        self.action_prior_pathway_temperature = (
                            config.reasoning_pathway_router_temperature
                        )
                        self.action_prior_pathway_token_in = nnx.Linear(
                            action_expert_config.width,
                            router_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            router_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_state_in = nnx.Linear(
                            config.action_dim,
                            router_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_embedding = nnx.Embed(
                            self.action_prior_pathway_count,
                            router_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_score = nnx.Param(
                            jnp.zeros((router_dim,), dtype=jnp.float32)
                        )
                    if config.reasoning_pathway_interaction:
                        interaction_dim = (
                            config.reasoning_pathway_interaction_hidden_dim
                        )
                        self.action_prior_pathway_interaction_count = 4
                        self.action_prior_pathway_interaction_token_in = nnx.Linear(
                            action_expert_config.width,
                            interaction_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_interaction_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            interaction_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_interaction_state_in = nnx.Linear(
                            config.action_dim,
                            interaction_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_interaction_path_embedding = nnx.Embed(
                            self.action_prior_pathway_interaction_count,
                            interaction_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_interaction_time_embedding = nnx.Embed(
                            config.action_horizon,
                            interaction_dim,
                            rngs=rngs,
                        )
                        self.action_prior_pathway_interaction_blocks = [
                            _ExplicitActionReasonerBlock(
                                interaction_dim,
                                config.reasoning_pathway_interaction_num_heads,
                                config.reasoning_pathway_interaction_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(
                                config.reasoning_pathway_interaction_layers
                            )
                        ]
                        self.action_prior_pathway_interaction_out = nnx.Linear(
                            interaction_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.action_chunk_verifier:
                        verifier_dim = config.action_chunk_verifier_hidden_dim
                        self.action_chunk_verifier_loss_weight = (
                            config.action_chunk_verifier_loss_weight
                        )
                        self.action_chunk_verifier_temperature = (
                            config.action_chunk_verifier_temperature
                        )
                        self.action_chunk_verifier_action_in = nnx.Linear(
                            config.action_dim, verifier_dim, rngs=rngs
                        )
                        self.action_chunk_verifier_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            verifier_dim,
                            rngs=rngs,
                        )
                        self.action_chunk_verifier_state_in = nnx.Linear(
                            config.action_dim, verifier_dim, rngs=rngs
                        )
                        self.action_chunk_verifier_position = nnx.Embed(
                            config.action_horizon, verifier_dim, rngs=rngs
                        )
                        self.action_chunk_verifier_blocks = [
                            _ExplicitActionReasonerBlock(
                                verifier_dim,
                                config.action_chunk_verifier_num_heads,
                                config.action_chunk_verifier_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.action_chunk_verifier_layers)
                        ]
                        self.action_chunk_verifier_score = nnx.Linear(
                            verifier_dim,
                            1,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.action_chunk_verifier_token_out = nnx.Linear(
                            verifier_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.latent_future_reasoner:
                        future_dim = config.latent_future_hidden_dim
                        self.latent_future_loss_weight = (
                            config.latent_future_loss_weight
                        )
                        self.latent_future_grid_size = (
                            config.latent_future_grid_size
                        )
                        self.latent_future_camera_names = (
                            'base_0_rgb',
                            'left_wrist_0_rgb',
                        )
                        self.latent_future_current_in = nnx.Linear(
                            paligemma_config.width, future_dim, rngs=rngs
                        )
                        self.latent_future_action_in = nnx.Linear(
                            config.action_dim, future_dim, rngs=rngs
                        )
                        self.latent_future_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            future_dim,
                            rngs=rngs,
                        )
                        self.latent_future_state_in = nnx.Linear(
                            config.action_dim, future_dim, rngs=rngs
                        )
                        self.latent_future_camera_position = nnx.Embed(
                            len(self.latent_future_camera_names),
                            future_dim,
                            rngs=rngs,
                        )
                        self.latent_future_patch_position = nnx.Embed(
                            config.latent_future_grid_size**2,
                            future_dim,
                            rngs=rngs,
                        )
                        self.latent_future_action_position = nnx.Embed(
                            config.action_horizon, future_dim, rngs=rngs
                        )
                        self.latent_future_blocks = [
                            _ExplicitActionReasonerBlock(
                                future_dim,
                                config.latent_future_num_heads,
                                config.latent_future_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.latent_future_layers)
                        ]
                        self.latent_future_predict_out = nnx.Linear(
                            future_dim, paligemma_config.width, rngs=rngs
                        )
                        self.latent_future_action_queries = nnx.Embed(
                            config.action_horizon, future_dim, rngs=rngs
                        )
                        self.latent_future_action_block = (
                            _ExplicitActionReasonerBlock(
                                future_dim,
                                config.latent_future_num_heads,
                                config.latent_future_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        # The future predictor can learn immediately from its
                        # feature loss, while this zero head exactly preserves
                        # the inherited Stage-6 policy at initialization.
                        self.latent_future_token_out = nnx.Linear(
                            future_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.state_rollout_reasoner:
                        rollout_dim = config.state_rollout_hidden_dim
                        self.state_rollout_loss_weight = (
                            config.state_rollout_loss_weight
                        )
                        self.state_rollout_target_dim = (
                            config.state_rollout_target_dim
                        )
                        self.state_rollout_action_in = nnx.Linear(
                            config.action_dim, rollout_dim, rngs=rngs
                        )
                        self.state_rollout_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            rollout_dim,
                            rngs=rngs,
                        )
                        self.state_rollout_state_in = nnx.Linear(
                            config.action_dim, rollout_dim, rngs=rngs
                        )
                        self.state_rollout_position = nnx.Embed(
                            config.action_horizon, rollout_dim, rngs=rngs
                        )
                        self.state_rollout_blocks = [
                            _ExplicitActionReasonerBlock(
                                rollout_dim,
                                config.state_rollout_num_heads,
                                config.state_rollout_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.state_rollout_layers)
                        ]
                        self.state_rollout_predict_out = nnx.Linear(
                            rollout_dim,
                            config.state_rollout_target_dim,
                            rngs=rngs,
                        )
                        # The rollout predictor learns immediately through its
                        # dense state loss.  This zero action-facing head keeps
                        # the inherited Stage-6 action function exact at init.
                        self.state_rollout_token_out = nnx.Linear(
                            rollout_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.action_moe_reasoner:
                        moe_dim = config.action_moe_hidden_dim
                        self.action_moe_num_experts = (
                            config.action_moe_num_experts
                        )
                        self.action_moe_top_k = config.action_moe_top_k
                        self.action_moe_temperature = (
                            config.action_moe_temperature
                        )
                        self.action_moe_prediction_loss_weight = (
                            config.action_moe_prediction_loss_weight
                        )
                        self.action_moe_balance_loss_weight = (
                            config.action_moe_balance_loss_weight
                        )
                        self.action_moe_task_consistent_routing = (
                            config.action_moe_task_consistent_routing
                        )
                        self.action_moe_action_in = nnx.Linear(
                            config.action_dim, moe_dim, rngs=rngs
                        )
                        self.action_moe_context_in = nnx.Linear(
                            config.action_prior_hidden_dim, moe_dim, rngs=rngs
                        )
                        self.action_moe_state_in = nnx.Linear(
                            config.action_dim, moe_dim, rngs=rngs
                        )
                        self.action_moe_position = nnx.Embed(
                            config.action_horizon, moe_dim, rngs=rngs
                        )
                        self.action_moe_blocks = [
                            _ExplicitActionReasonerBlock(
                                moe_dim,
                                config.action_moe_num_heads,
                                config.action_moe_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.action_moe_layers)
                        ]
                        self.action_moe_router = nnx.Linear(
                            moe_dim,
                            config.action_moe_num_experts,
                            rngs=rngs,
                        )
                        self.action_moe_expert_in = [
                            nnx.Linear(
                                moe_dim,
                                config.action_moe_expert_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.action_moe_num_experts)
                        ]
                        self.action_moe_expert_out = [
                            nnx.Linear(
                                config.action_moe_expert_dim,
                                moe_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.action_moe_num_experts)
                        ]
                        # Dense action prediction supplies immediate gradients
                        # to the experts and router. Only the zero-initialized
                        # token head touches the inherited flow policy.
                        self.action_moe_predict_out = nnx.Linear(
                            moe_dim, config.action_dim, rngs=rngs
                        )
                        self.action_moe_token_out = nnx.Linear(
                            moe_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.task_progress_reasoner:
                        progress_dim = config.task_progress_hidden_dim
                        self.task_progress_bins = config.task_progress_bins
                        self.task_progress_loss_weight = (
                            config.task_progress_loss_weight
                        )
                        self.task_progress_query = nnx.Embed(
                            1, progress_dim, rngs=rngs
                        )
                        self.task_progress_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            progress_dim,
                            rngs=rngs,
                        )
                        self.task_progress_state_in = nnx.Linear(
                            config.action_dim, progress_dim, rngs=rngs
                        )
                        self.task_progress_blocks = [
                            _ExplicitActionReasonerBlock(
                                progress_dim,
                                config.task_progress_num_heads,
                                config.task_progress_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.task_progress_layers)
                        ]
                        self.task_progress_logits = nnx.Linear(
                            progress_dim,
                            config.task_progress_bins,
                            rngs=rngs,
                        )
                        self.task_progress_bin_embeddings = nnx.Embed(
                            config.task_progress_bins,
                            progress_dim,
                            rngs=rngs,
                        )
                        self.task_progress_action_queries = nnx.Embed(
                            config.action_horizon,
                            progress_dim,
                            rngs=rngs,
                        )
                        self.task_progress_action_block = (
                            _ExplicitActionReasonerBlock(
                                progress_dim,
                                config.task_progress_num_heads,
                                config.task_progress_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        self.task_progress_token_out = nnx.Linear(
                            progress_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.language_subgoal_reasoner:
                        subgoal_dim = config.language_subgoal_hidden_dim
                        self.language_subgoal_slot_count = (
                            config.language_subgoal_slots
                        )
                        self.language_subgoal_temperature = (
                            config.language_subgoal_temperature
                        )
                        self.language_subgoal_progress_loss_weight = (
                            config.language_subgoal_progress_loss_weight
                        )
                        self.language_subgoal_action_loss_weight = (
                            config.language_subgoal_action_loss_weight
                        )
                        self.language_subgoal_phase_class_weights = (
                            config.language_subgoal_phase_class_weights
                        )
                        self.language_subgoal_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            subgoal_dim,
                            rngs=rngs,
                        )
                        # The five inherited action-prior contexts are useful
                        # summaries, but they are too compressed to reliably
                        # decompose long, ordered instructions.  Give every
                        # learned subgoal slot a separate masked read over the
                        # complete contextualized multimodal prefix.  The
                        # action-facing projection below remains exactly zero,
                        # so this richer input does not change the inherited
                        # policy before optimization.
                        self.language_subgoal_prefix_in = nnx.Linear(
                            paligemma_config.width,
                            subgoal_dim,
                            rngs=rngs,
                        )
                        self.language_subgoal_prefix_query = nnx.Linear(
                            subgoal_dim,
                            subgoal_dim,
                            rngs=rngs,
                        )
                        self.language_subgoal_state_in = nnx.Linear(
                            config.action_dim, subgoal_dim, rngs=rngs
                        )
                        self.language_subgoal_slot_queries = nnx.Embed(
                            config.language_subgoal_slots,
                            subgoal_dim,
                            rngs=rngs,
                        )
                        self.language_subgoal_slot_positions = nnx.Embed(
                            config.language_subgoal_slots,
                            subgoal_dim,
                            rngs=rngs,
                        )
                        self.language_subgoal_blocks = [
                            _ExplicitActionReasonerBlock(
                                subgoal_dim,
                                config.language_subgoal_num_heads,
                                config.language_subgoal_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.language_subgoal_layers)
                        ]
                        self.language_subgoal_score = nnx.Linear(
                            subgoal_dim, 1, rngs=rngs
                        )
                        self.language_subgoal_action_queries = nnx.Embed(
                            config.action_horizon,
                            subgoal_dim,
                            rngs=rngs,
                        )
                        self.language_subgoal_action_block = (
                            _ExplicitActionReasonerBlock(
                                subgoal_dim,
                                config.language_subgoal_num_heads,
                                config.language_subgoal_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        self.language_subgoal_action_predict = nnx.Linear(
                            subgoal_dim, config.action_dim, rngs=rngs
                        )
                        self.language_subgoal_token_out = nnx.Linear(
                            subgoal_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.object_subgoal_binding:
                        binding_dim = config.object_subgoal_binding_hidden_dim
                        self.object_subgoal_binding_temperature = (
                            config.object_subgoal_binding_temperature
                        )
                        self.object_subgoal_binding_action_loss_weight = (
                            config.object_subgoal_binding_action_loss_weight
                        )
                        self.object_subgoal_binding_distinct_roles = (
                            config.object_subgoal_binding_distinct_roles
                        )
                        self.object_subgoal_binding_object_in = nnx.Linear(
                            config.object_affordance_hidden_dim,
                            binding_dim,
                            rngs=rngs,
                        )
                        self.object_subgoal_binding_subgoal_in = nnx.Linear(
                            config.language_subgoal_hidden_dim,
                            binding_dim,
                            rngs=rngs,
                        )
                        self.object_subgoal_binding_object_query = nnx.Linear(
                            binding_dim, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_subgoal_key = nnx.Linear(
                            binding_dim, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_reference_key = nnx.Linear(
                            binding_dim, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_pair_fuse = nnx.Linear(
                            2 * binding_dim, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_relation_fuse = nnx.Linear(
                            5 * binding_dim, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_relation_gate = nnx.Linear(
                            binding_dim, 1, rngs=rngs
                        )
                        self.object_subgoal_binding_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            binding_dim,
                            rngs=rngs,
                        )
                        self.object_subgoal_binding_state_in = nnx.Linear(
                            config.action_dim, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_blocks = [
                            _ExplicitActionReasonerBlock(
                                binding_dim,
                                config.object_subgoal_binding_num_heads,
                                config.object_subgoal_binding_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.object_subgoal_binding_layers)
                        ]
                        self.object_subgoal_binding_action_queries = nnx.Embed(
                            config.action_horizon, binding_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_action_block = (
                            _ExplicitActionReasonerBlock(
                                binding_dim,
                                config.object_subgoal_binding_num_heads,
                                config.object_subgoal_binding_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        self.object_subgoal_binding_action_predict = nnx.Linear(
                            binding_dim, config.action_dim, rngs=rngs
                        )
                        self.object_subgoal_binding_token_out = nnx.Linear(
                            binding_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.kinematic_action_reasoner:
                        kinematic_dim = config.kinematic_action_hidden_dim
                        self.kinematic_action_prediction_loss_weight = (
                            config.kinematic_action_prediction_loss_weight
                        )
                        self.kinematic_action_translation_in = nnx.Linear(
                            3, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_rotation_in = nnx.Linear(
                            3, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_gripper_in = nnx.Linear(
                            1, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            kinematic_dim,
                            rngs=rngs,
                        )
                        self.kinematic_action_state_in = nnx.Linear(
                            config.action_dim, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_time_position = nnx.Embed(
                            config.action_horizon, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_group_position = nnx.Embed(
                            3, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_blocks = [
                            _ExplicitActionReasonerBlock(
                                kinematic_dim,
                                config.kinematic_action_num_heads,
                                config.kinematic_action_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.kinematic_action_layers)
                        ]
                        self.kinematic_action_translation_predict = nnx.Linear(
                            kinematic_dim, 3, rngs=rngs
                        )
                        self.kinematic_action_rotation_predict = nnx.Linear(
                            kinematic_dim, 3, rngs=rngs
                        )
                        self.kinematic_action_gripper_predict = nnx.Linear(
                            kinematic_dim, 1, rngs=rngs
                        )
                        self.kinematic_action_group_fusion = nnx.Linear(
                            3 * kinematic_dim, kinematic_dim, rngs=rngs
                        )
                        self.kinematic_action_token_out = nnx.Linear(
                            kinematic_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.spectral_action_reasoner:
                        spectral_dim = config.spectral_action_hidden_dim
                        self.spectral_action_band_count = (
                            config.spectral_action_bands
                        )
                        self.spectral_action_prediction_loss_weight = (
                            config.spectral_action_prediction_loss_weight
                        )
                        self.spectral_action_in = nnx.Linear(
                            config.action_dim, spectral_dim, rngs=rngs
                        )
                        self.spectral_action_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            spectral_dim,
                            rngs=rngs,
                        )
                        self.spectral_action_state_in = nnx.Linear(
                            config.action_dim, spectral_dim, rngs=rngs
                        )
                        self.spectral_action_frequency_position = nnx.Embed(
                            config.action_horizon, spectral_dim, rngs=rngs
                        )
                        self.spectral_action_band_position = nnx.Embed(
                            config.spectral_action_bands,
                            spectral_dim,
                            rngs=rngs,
                        )
                        self.spectral_action_blocks = [
                            _ExplicitActionReasonerBlock(
                                spectral_dim,
                                config.spectral_action_num_heads,
                                config.spectral_action_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.spectral_action_layers)
                        ]
                        self.spectral_action_predict = nnx.Linear(
                            spectral_dim, config.action_dim, rngs=rngs
                        )
                        self.spectral_action_token_out = nnx.Linear(
                            spectral_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.velocity_refiner:
                        refiner_dim = config.velocity_refiner_hidden_dim
                        self.velocity_refiner_loss_weight = (
                            config.velocity_refiner_loss_weight
                        )
                        self.velocity_refiner_action_in = nnx.Linear(
                            3 * config.action_dim, refiner_dim, rngs=rngs
                        )
                        self.velocity_refiner_hidden_in = nnx.Linear(
                            action_expert_config.width, refiner_dim, rngs=rngs
                        )
                        self.velocity_refiner_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            refiner_dim,
                            rngs=rngs,
                        )
                        self.velocity_refiner_state_in = nnx.Linear(
                            config.action_dim, refiner_dim, rngs=rngs
                        )
                        self.velocity_refiner_time_in = nnx.Linear(
                            refiner_dim, refiner_dim, rngs=rngs
                        )
                        self.velocity_refiner_position = nnx.Embed(
                            config.action_horizon, refiner_dim, rngs=rngs
                        )
                        self.velocity_refiner_blocks = [
                            _ExplicitActionReasonerBlock(
                                refiner_dim,
                                config.velocity_refiner_num_heads,
                                config.velocity_refiner_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.velocity_refiner_layers)
                        ]
                        self.velocity_refiner_predict = nnx.Linear(
                            refiner_dim, config.action_dim, rngs=rngs
                        )
                        self.velocity_refiner_gain = nnx.Param(
                            jnp.zeros((config.action_dim,), dtype=jnp.float32)
                        )
                    if config.action_visual_refiner:
                        visual_dim = config.action_visual_refiner_hidden_dim
                        self.action_visual_refiner_loss_weight = (
                            config.action_visual_refiner_loss_weight
                        )
                        self.action_visual_refiner_action_in = nnx.Linear(
                            3 * config.action_dim, visual_dim, rngs=rngs
                        )
                        self.action_visual_refiner_hidden_in = nnx.Linear(
                            action_expert_config.width, visual_dim, rngs=rngs
                        )
                        self.action_visual_refiner_prefix_in = nnx.Linear(
                            paligemma_config.width, visual_dim, rngs=rngs
                        )
                        self.action_visual_refiner_state_in = nnx.Linear(
                            config.action_dim, visual_dim, rngs=rngs
                        )
                        self.action_visual_refiner_time_in = nnx.Linear(
                            visual_dim, visual_dim, rngs=rngs
                        )
                        self.action_visual_refiner_position = nnx.Embed(
                            config.action_horizon, visual_dim, rngs=rngs
                        )
                        self.action_visual_refiner_blocks = [
                            _SpatialRelationReasonerBlock(
                                visual_dim,
                                config.action_visual_refiner_num_heads,
                                config.action_visual_refiner_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.action_visual_refiner_layers)
                        ]
                        self.action_visual_refiner_predict = nnx.Linear(
                            visual_dim, config.action_dim, rngs=rngs
                        )
                        self.action_visual_refiner_gain = nnx.Param(
                            jnp.zeros((config.action_dim,), dtype=jnp.float32)
                        )
                    if config.masked_spatial_reasoner:
                        masked_dim = config.masked_spatial_hidden_dim
                        self.masked_spatial_query_count = (
                            config.masked_spatial_queries
                        )
                        self.masked_spatial_max_cameras = (
                            config.masked_spatial_max_cameras
                        )
                        self.masked_spatial_max_grid_size = (
                            config.masked_spatial_max_grid_size
                        )
                        self.masked_spatial_mask_ratio = (
                            config.masked_spatial_mask_ratio
                        )
                        self.masked_spatial_reconstruction_loss_weight = (
                            config.masked_spatial_reconstruction_loss_weight
                        )
                        self.masked_spatial_action_loss_weight = (
                            config.masked_spatial_action_loss_weight
                        )
                        self.masked_spatial_num_heads = (
                            config.masked_spatial_num_heads
                        )
                        self.masked_spatial_image_in = nnx.Linear(
                            paligemma_config.width, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_contextual_in = nnx.Linear(
                            paligemma_config.width, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_language_in = nnx.Linear(
                            paligemma_config.width, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_state_in = nnx.Linear(
                            config.action_dim, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_camera_position = nnx.Embed(
                            config.masked_spatial_max_cameras,
                            masked_dim,
                            rngs=rngs,
                        )
                        self.masked_spatial_row_position = nnx.Embed(
                            config.masked_spatial_max_grid_size,
                            masked_dim,
                            rngs=rngs,
                        )
                        self.masked_spatial_column_position = nnx.Embed(
                            config.masked_spatial_max_grid_size,
                            masked_dim,
                            rngs=rngs,
                        )
                        self.masked_spatial_mask_token = nnx.Embed(
                            1, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_scene_queries = nnx.Embed(
                            config.masked_spatial_queries,
                            masked_dim,
                            rngs=rngs,
                        )
                        self.masked_spatial_scene_blocks = [
                            _SpatialRelationReasonerBlock(
                                masked_dim,
                                config.masked_spatial_num_heads,
                                config.masked_spatial_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.masked_spatial_layers)
                        ]
                        # Patch reconstruction is cross-attention only: the
                        # O(patches * scene_queries) decoder avoids an
                        # unnecessary quadratic patch self-attention graph.
                        self.masked_spatial_decoder_q = nnx.Linear(
                            masked_dim, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_decoder_k = nnx.Linear(
                            masked_dim, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_decoder_v = nnx.Linear(
                            masked_dim, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_decoder_out = nnx.Linear(
                            masked_dim, masked_dim, rngs=rngs
                        )
                        self.masked_spatial_reconstruct = nnx.Linear(
                            masked_dim, paligemma_config.width, rngs=rngs
                        )
                        self.masked_spatial_waypoint_queries = nnx.Embed(
                            config.action_prior_horizon,
                            masked_dim,
                            rngs=rngs,
                        )
                        self.masked_spatial_waypoint_block = (
                            _ExplicitActionReasonerBlock(
                                masked_dim,
                                config.masked_spatial_num_heads,
                                config.masked_spatial_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        self.masked_spatial_action_out = nnx.Linear(
                            masked_dim,
                            config.action_dim,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.masked_spatial_token_out = nnx.Linear(
                            masked_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.object_future_reasoner:
                        future_dim = config.object_future_hidden_dim
                        self.object_future_query_count = (
                            config.object_future_queries
                        )
                        self.object_future_max_grid_size = (
                            config.object_future_max_grid_size
                        )
                        self.object_future_num_heads = (
                            config.object_future_num_heads
                        )
                        self.object_future_reconstruction_loss_weight = (
                            config.object_future_reconstruction_loss_weight
                        )
                        self.object_future_action_loss_weight = (
                            config.object_future_action_loss_weight
                        )
                        self.object_future_camera_names = (
                            'base_0_rgb',
                            'left_wrist_0_rgb',
                        )
                        self.object_future_image_in = nnx.Linear(
                            paligemma_config.width, future_dim, rngs=rngs
                        )
                        self.object_future_contextual_in = nnx.Linear(
                            paligemma_config.width, future_dim, rngs=rngs
                        )
                        self.object_future_language_in = nnx.Linear(
                            paligemma_config.width, future_dim, rngs=rngs
                        )
                        self.object_future_state_in = nnx.Linear(
                            config.action_dim, future_dim, rngs=rngs
                        )
                        self.object_future_prior_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            future_dim,
                            rngs=rngs,
                        )
                        if config.object_future_affordance_bridge:
                            self.object_future_affordance_in = nnx.Linear(
                                config.object_affordance_hidden_dim,
                                future_dim,
                                rngs=rngs,
                            )
                        self.object_future_camera_position = nnx.Embed(
                            len(self.object_future_camera_names),
                            future_dim,
                            rngs=rngs,
                        )
                        self.object_future_row_position = nnx.Embed(
                            config.object_future_max_grid_size,
                            future_dim,
                            rngs=rngs,
                        )
                        self.object_future_column_position = nnx.Embed(
                            config.object_future_max_grid_size,
                            future_dim,
                            rngs=rngs,
                        )
                        self.object_future_object_queries = nnx.Embed(
                            config.object_future_queries,
                            future_dim,
                            rngs=rngs,
                        )
                        self.object_future_object_blocks = [
                            _SpatialRelationReasonerBlock(
                                future_dim,
                                config.object_future_num_heads,
                                config.object_future_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.object_future_layers)
                        ]
                        self.object_future_forecast_queries = nnx.Embed(
                            config.object_future_queries,
                            future_dim,
                            rngs=rngs,
                        )
                        self.object_future_forecast_blocks = [
                            _ExplicitActionReasonerBlock(
                                future_dim,
                                config.object_future_num_heads,
                                config.object_future_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.object_future_layers)
                        ]
                        # Linear patch-to-slot decoding retains a dense target
                        # without quadratic patch self-attention.
                        self.object_future_decoder_q = nnx.Linear(
                            future_dim, future_dim, rngs=rngs
                        )
                        self.object_future_decoder_k = nnx.Linear(
                            future_dim, future_dim, rngs=rngs
                        )
                        self.object_future_decoder_v = nnx.Linear(
                            future_dim, future_dim, rngs=rngs
                        )
                        self.object_future_decoder_out = nnx.Linear(
                            future_dim, future_dim, rngs=rngs
                        )
                        self.object_future_reconstruct = nnx.Linear(
                            future_dim, paligemma_config.width, rngs=rngs
                        )
                        self.object_future_waypoint_queries = nnx.Embed(
                            config.action_prior_horizon,
                            future_dim,
                            rngs=rngs,
                        )
                        self.object_future_waypoint_block = (
                            _ExplicitActionReasonerBlock(
                                future_dim,
                                config.object_future_num_heads,
                                config.object_future_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        self.object_future_action_out = nnx.Linear(
                            future_dim,
                            config.action_dim,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.object_future_token_out = nnx.Linear(
                            future_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.predicate_binding_reasoner:
                        predicate_dim = config.predicate_binding_hidden_dim
                        self.predicate_binding_object_count = (
                            config.predicate_binding_object_slots
                        )
                        self.predicate_binding_role_count = (
                            config.predicate_binding_role_slots
                        )
                        self.predicate_binding_max_cameras = (
                            config.predicate_binding_max_cameras
                        )
                        self.predicate_binding_max_grid_size = (
                            config.predicate_binding_max_grid_size
                        )
                        self.predicate_binding_temperature = (
                            config.predicate_binding_temperature
                        )
                        self.predicate_binding_contrastive_loss_weight = (
                            config.predicate_binding_contrastive_loss_weight
                        )
                        self.predicate_binding_action_loss_weight = (
                            config.predicate_binding_action_loss_weight
                        )
                        self.predicate_binding_image_in = nnx.Linear(
                            paligemma_config.width, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_contextual_in = nnx.Linear(
                            paligemma_config.width, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_language_in = nnx.Linear(
                            paligemma_config.width, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_state_in = nnx.Linear(
                            config.action_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_prior_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            predicate_dim,
                            rngs=rngs,
                        )
                        self.predicate_binding_camera_position = nnx.Embed(
                            config.predicate_binding_max_cameras,
                            predicate_dim,
                            rngs=rngs,
                        )
                        self.predicate_binding_row_position = nnx.Embed(
                            config.predicate_binding_max_grid_size,
                            predicate_dim,
                            rngs=rngs,
                        )
                        self.predicate_binding_column_position = nnx.Embed(
                            config.predicate_binding_max_grid_size,
                            predicate_dim,
                            rngs=rngs,
                        )
                        # Object slots are deliberately language independent.
                        # Language is introduced only by the three role tokens,
                        # preventing the visual/text matching loss from taking
                        # a direct language shortcut through the object bank.
                        self.predicate_binding_object_queries = nnx.Embed(
                            config.predicate_binding_object_slots,
                            predicate_dim,
                            rngs=rngs,
                        )
                        self.predicate_binding_object_query_in = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_patch_key_in = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_patch_value_in = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_object_update = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_object_blocks = [
                            _SpatialRelationReasonerBlock(
                                predicate_dim,
                                config.predicate_binding_num_heads,
                                config.predicate_binding_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.predicate_binding_layers)
                        ]
                        self.predicate_binding_role_queries = nnx.Embed(
                            config.predicate_binding_role_slots,
                            predicate_dim,
                            rngs=rngs,
                        )
                        self.predicate_binding_role_blocks = [
                            _SpatialRelationReasonerBlock(
                                predicate_dim,
                                config.predicate_binding_num_heads,
                                config.predicate_binding_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.predicate_binding_layers)
                        ]
                        self.predicate_binding_role_query_in = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_object_key_in = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_role_object_out = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_graph_blocks = [
                            _ExplicitActionReasonerBlock(
                                predicate_dim,
                                config.predicate_binding_num_heads,
                                config.predicate_binding_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.predicate_binding_layers)
                        ]
                        self.predicate_binding_text_match_out = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_visual_match_out = nnx.Linear(
                            predicate_dim, predicate_dim, rngs=rngs
                        )
                        self.predicate_binding_waypoint_queries = nnx.Embed(
                            config.action_prior_horizon,
                            predicate_dim,
                            rngs=rngs,
                        )
                        self.predicate_binding_waypoint_block = (
                            _ExplicitActionReasonerBlock(
                                predicate_dim,
                                config.predicate_binding_num_heads,
                                config.predicate_binding_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        self.predicate_binding_action_out = nnx.Linear(
                            predicate_dim,
                            config.action_dim,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.predicate_binding_token_out = nnx.Linear(
                            predicate_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.predictive_world_model_fusion:
                        fusion_dim = config.predictive_world_model_hidden_dim
                        self.predictive_world_model_auxiliary_scale = (
                            config.predictive_world_model_auxiliary_scale
                        )
                        self.predictive_world_model_reliability_loss_weight = (
                            config.predictive_world_model_reliability_loss_weight
                        )
                        self.predictive_world_model_include_action_moe = (
                            config.predictive_world_model_include_action_moe
                        )
                        self.predictive_world_model_router_init_scale = (
                            config.predictive_world_model_router_init_scale
                        )
                        self.predictive_world_model_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.predictive_world_model_state_in = nnx.Linear(
                            config.action_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        if config.persistent_subgoal_memory:
                            # A compound predictive-memory policy routes its
                            # short-horizon dynamics branches using the private
                            # cross-replan memory, including grounded role and
                            # ordered-subgoal evidence.  The downstream gate
                            # starts at exact zero, so this added condition is
                            # function preserving at inheritance.
                            self.predictive_world_model_memory_in = nnx.Linear(
                                config.persistent_memory_hidden_dim,
                                fusion_dim,
                                rngs=rngs,
                            )
                            self.predictive_world_model_memory_position = nnx.Embed(
                                config.persistent_memory_tokens,
                                fusion_dim,
                                rngs=rngs,
                            )
                            # Read the composed physical-phase program before
                            # it is compressed through recurrent memory.  A
                            # differentiable current/next frontier mixture
                            # gives future routing explicit phase semantics
                            # while the zero-initialized downstream fusion
                            # gate keeps PSM inheritance function preserving.
                            self.predictive_world_model_program_in = nnx.Linear(
                                config.persistent_memory_hidden_dim,
                                fusion_dim,
                                rngs=rngs,
                            )
                            self.predictive_world_model_program_position = nnx.Embed(
                                config.persistent_memory_subgoal_slots,
                                fusion_dim,
                                rngs=rngs,
                            )
                        self.predictive_world_model_action_position = nnx.Embed(
                            config.action_horizon,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.predictive_world_model_future_in = nnx.Linear(
                            action_expert_config.width,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.predictive_world_model_rollout_in = nnx.Linear(
                            action_expert_config.width,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.predictive_world_model_progress_in = nnx.Linear(
                            action_expert_config.width,
                            fusion_dim,
                            rngs=rngs,
                        )
                        if self.predictive_world_model_include_action_moe:
                            self.predictive_world_model_expert_in = nnx.Linear(
                                action_expert_config.width,
                                fusion_dim,
                                rngs=rngs,
                            )
                        # Score each projected branch independently before the
                        # global four-way gate.  A shared, bias-free scorer
                        # makes the score depend on branch content rather than
                        # branch identity; zero initialization preserves the
                        # inherited uniform fusion exactly.
                        self.predictive_world_model_content_score = nnx.Param(
                            jnp.zeros((fusion_dim,), dtype=jnp.float32)
                        )
                        reliability_init = nnx.initializers.zeros_init()
                        self.predictive_world_model_future_reliability = nnx.Linear(
                            action_expert_config.width,
                            1,
                            kernel_init=reliability_init,
                            bias_init=reliability_init,
                            rngs=rngs,
                        )
                        self.predictive_world_model_rollout_reliability = nnx.Linear(
                            action_expert_config.width,
                            1,
                            kernel_init=reliability_init,
                            bias_init=reliability_init,
                            rngs=rngs,
                        )
                        self.predictive_world_model_progress_reliability = nnx.Linear(
                            action_expert_config.width,
                            1,
                            kernel_init=reliability_init,
                            bias_init=reliability_init,
                            rngs=rngs,
                        )
                        if self.predictive_world_model_include_action_moe:
                            self.predictive_world_model_expert_reliability = (
                                nnx.Linear(
                                    action_expert_config.width,
                                    1,
                                    kernel_init=reliability_init,
                                    bias_init=reliability_init,
                                    rngs=rngs,
                                )
                            )
                        router_kernel_init = (
                            _normal_initializer(
                                self.predictive_world_model_router_init_scale
                            )
                            if self.predictive_world_model_router_init_scale > 0.0
                            else nnx.initializers.zeros_init()
                        )
                        self.predictive_world_model_gate = nnx.Linear(
                            fusion_dim,
                            4
                            if self.predictive_world_model_include_action_moe
                            else 3,
                            # Persistent descendants use a small-open router;
                            # their four action-facing branch projections stay
                            # exactly zero and preserve the parent policy while
                            # memory/program routing becomes trainable as soon
                            # as those sole outer gates begin to open.
                            kernel_init=router_kernel_init,
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.evidence_combination_router:
                        router_dim = (
                            config.evidence_combination_router_hidden_dim
                        )
                        self.evidence_combination_component_count = (
                            4
                            + 2
                            * int(config.evidence_combination_hierarchical_components)
                            + int(
                                config.evidence_combination_action_verifier_component
                            )
                            + int(
                                config.evidence_combination_object_subgoal_binding_component
                            )
                        )
                        self.evidence_combination_action_verifier_component = (
                            config.evidence_combination_action_verifier_component
                        )
                        self.evidence_combination_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            router_dim,
                            rngs=rngs,
                        )
                        self.evidence_combination_state_in = nnx.Linear(
                            config.action_dim,
                            router_dim,
                            rngs=rngs,
                        )
                        self.evidence_combination_action_position = nnx.Embed(
                            config.action_horizon,
                            router_dim,
                            rngs=rngs,
                        )
                        self.evidence_combination_content_in = nnx.Linear(
                            action_expert_config.width,
                            router_dim,
                            rngs=rngs,
                        )
                        self.evidence_combination_component_embedding = nnx.Embed(
                            self.evidence_combination_component_count,
                            router_dim,
                            rngs=rngs,
                        )
                        self.evidence_combination_content_score = nnx.Param(
                            jnp.zeros(
                                (
                                    self.evidence_combination_component_count,
                                    router_dim,
                                ),
                                dtype=jnp.float32,
                            )
                        )
                        self.evidence_combination_gate = nnx.Linear(
                            router_dim,
                            self.evidence_combination_component_count,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.spatial_relation_reasoner:
                        spatial_dim = config.spatial_relation_hidden_dim
                        self.spatial_relation_max_cameras = (
                            config.spatial_relation_max_cameras
                        )
                        self.spatial_relation_max_grid_size = (
                            config.spatial_relation_max_grid_size
                        )
                        self.spatial_relation_loss_weight = (
                            config.spatial_relation_loss_weight
                        )
                        self.spatial_relation_query_count = (
                            config.spatial_relation_queries
                        )
                        self.spatial_relation_image_in = nnx.Linear(
                            paligemma_config.width, spatial_dim, rngs=rngs
                        )
                        self.spatial_relation_contextual_in = nnx.Linear(
                            paligemma_config.width, spatial_dim, rngs=rngs
                        )
                        self.spatial_relation_language_in = nnx.Linear(
                            paligemma_config.width, spatial_dim, rngs=rngs
                        )
                        self.spatial_relation_camera_position = nnx.Embed(
                            config.spatial_relation_max_cameras,
                            spatial_dim,
                            rngs=rngs,
                        )
                        self.spatial_relation_row_position = nnx.Embed(
                            config.spatial_relation_max_grid_size,
                            spatial_dim,
                            rngs=rngs,
                        )
                        self.spatial_relation_column_position = nnx.Embed(
                            config.spatial_relation_max_grid_size,
                            spatial_dim,
                            rngs=rngs,
                        )
                        self.spatial_relation_token_type = nnx.Embed(
                            2, spatial_dim, rngs=rngs
                        )
                        self.spatial_relation_queries = nnx.Embed(
                            config.spatial_relation_queries,
                            spatial_dim,
                            rngs=rngs,
                        )
                        self.spatial_relation_state_in = nnx.Linear(
                            config.action_dim, spatial_dim, rngs=rngs
                        )
                        self.spatial_relation_blocks = [
                            _SpatialRelationReasonerBlock(
                                spatial_dim,
                                config.spatial_relation_num_heads,
                                config.spatial_relation_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.spatial_relation_layers)
                        ]
                        self.spatial_relation_waypoint_queries = nnx.Embed(
                            config.action_prior_horizon,
                            spatial_dim,
                            rngs=rngs,
                        )
                        self.spatial_relation_waypoint_block = (
                            _ExplicitActionReasonerBlock(
                                spatial_dim,
                                config.spatial_relation_num_heads,
                                config.spatial_relation_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        # Both heads are zero initialized, so a selected
                        # Stage-6 checkpoint has identical policy outputs
                        # before this branch receives training updates.
                        self.spatial_relation_token_out = nnx.Linear(
                            spatial_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.spatial_relation_action_out = nnx.Linear(
                            spatial_dim,
                            config.action_dim,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.object_affordance_graph_reasoner:
                        affordance_dim = config.object_affordance_hidden_dim
                        self.object_affordance_slot_count = (
                            config.object_affordance_slots
                        )
                        self.object_affordance_max_cameras = (
                            config.object_affordance_max_cameras
                        )
                        self.object_affordance_max_grid_size = (
                            config.object_affordance_max_grid_size
                        )
                        self.object_affordance_temperature = (
                            config.object_affordance_temperature
                        )
                        self.object_affordance_loss_weight = (
                            config.object_affordance_loss_weight
                        )
                        self.object_affordance_reconstruction_loss_weight = (
                            config.object_affordance_reconstruction_loss_weight
                        )
                        self.object_affordance_image_in = nnx.Linear(
                            paligemma_config.width, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_contextual_in = nnx.Linear(
                            paligemma_config.width, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_language_in = nnx.Linear(
                            paligemma_config.width, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_camera_position = nnx.Embed(
                            config.object_affordance_max_cameras,
                            affordance_dim,
                            rngs=rngs,
                        )
                        self.object_affordance_row_position = nnx.Embed(
                            config.object_affordance_max_grid_size,
                            affordance_dim,
                            rngs=rngs,
                        )
                        self.object_affordance_column_position = nnx.Embed(
                            config.object_affordance_max_grid_size,
                            affordance_dim,
                            rngs=rngs,
                        )
                        self.object_affordance_slot_queries = nnx.Embed(
                            config.object_affordance_slots,
                            affordance_dim,
                            rngs=rngs,
                        )
                        self.object_affordance_state_in = nnx.Linear(
                            config.action_dim, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_slot_query_in = nnx.Linear(
                            affordance_dim, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_patch_key_in = nnx.Linear(
                            affordance_dim, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_patch_value_in = nnx.Linear(
                            affordance_dim, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_slot_update = nnx.Linear(
                            affordance_dim, affordance_dim, rngs=rngs
                        )
                        self.object_affordance_graph_blocks = [
                            _SpatialRelationReasonerBlock(
                                affordance_dim,
                                config.object_affordance_num_heads,
                                config.object_affordance_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.object_affordance_layers)
                        ]
                        self.object_affordance_waypoint_queries = nnx.Embed(
                            config.action_prior_horizon,
                            affordance_dim,
                            rngs=rngs,
                        )
                        self.object_affordance_waypoint_block = (
                            _ExplicitActionReasonerBlock(
                                affordance_dim,
                                config.object_affordance_num_heads,
                                config.object_affordance_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        # Stage-21 is exactly Stage-6 at inheritance.  Direct
                        # coarse supervision wakes both zero heads without
                        # injecting an untrained residual into the policy.
                        self.object_affordance_token_out = nnx.Linear(
                            affordance_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.object_affordance_action_out = nnx.Linear(
                            affordance_dim,
                            config.action_dim,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.contact_phase_reasoner:
                        phase_dim = config.contact_phase_hidden_dim
                        self.contact_phase_temperature = (
                            config.contact_phase_temperature
                        )
                        self.contact_phase_loss_weight = (
                            config.contact_phase_loss_weight
                        )
                        self.contact_phase_focal_gamma = (
                            config.contact_phase_focal_gamma
                        )
                        self.contact_phase_loss_temperature = (
                            config.contact_phase_loss_temperature
                        )
                        self.contact_phase_gripper_index = (
                            config.contact_phase_gripper_index
                        )
                        self.contact_phase_gripper_indices = tuple(
                            config.contact_phase_gripper_indices
                        )
                        self.contact_phase_state_scalar_indices = tuple(
                            config.contact_phase_state_scalar_indices
                        )
                        self.contact_phase_open_when_positive = (
                            config.contact_phase_open_when_positive
                        )
                        self.contact_phase_state_gripper_indices = tuple(
                            config.contact_phase_state_gripper_indices
                        )
                        self.contact_phase_state_open_threshold = (
                            config.contact_phase_state_open_threshold
                        )
                        contact_phase_counts = np.asarray(
                            config.contact_phase_class_counts,
                            dtype=np.float32,
                        )
                        contact_phase_weights = _inverse_sqrt_class_weights(
                            contact_phase_counts, clip=16.0
                        )
                        contact_phase_weights *= np.asarray(
                            config.contact_phase_transition_boosts,
                            dtype=np.float32,
                        )
                        # Preserve mean sample weight after applying boosts so
                        # the loss weight remains comparable across configs.
                        contact_phase_weights /= (
                            np.sum(contact_phase_counts * contact_phase_weights)
                            / np.sum(contact_phase_counts)
                        )
                        self.contact_phase_class_weights = tuple(
                            float(weight) for weight in contact_phase_weights
                        )
                        self.contact_phase_queries = nnx.Embed(
                            config.action_horizon, phase_dim, rngs=rngs
                        )
                        self.contact_phase_state_in = nnx.Linear(
                            config.action_dim, phase_dim, rngs=rngs
                        )
                        self.contact_phase_context_in = nnx.Linear(
                            config.action_prior_hidden_dim, phase_dim, rngs=rngs
                        )
                        self.contact_phase_blocks = [
                            _ExplicitActionReasonerBlock(
                                phase_dim,
                                config.contact_phase_num_heads,
                                config.contact_phase_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.contact_phase_layers)
                        ]
                        self.contact_phase_logits = nnx.Linear(
                            phase_dim, 4, rngs=rngs
                        )
                        self.contact_phase_embeddings = nnx.Embed(
                            4, phase_dim, rngs=rngs
                        )
                        self.contact_phase_fuse = nnx.Linear(
                            2 * phase_dim, phase_dim, rngs=rngs
                        )
                        self.contact_phase_token_out = nnx.Linear(
                            phase_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.contact_affordance_predictive_fusion:
                        fusion_dim = (
                            config.contact_affordance_fusion_hidden_dim
                        )
                        self.contact_affordance_risk_loss_weight = (
                            config.contact_affordance_risk_loss_weight
                        )
                        self.contact_affordance_relation_contrastive_loss_weight = (
                            config.contact_affordance_relation_contrastive_loss_weight
                        )
                        self.contact_affordance_relation_contrastive_temperature = (
                            config.contact_affordance_relation_contrastive_temperature
                        )
                        self.contact_affordance_transition_verification = (
                            config.contact_affordance_transition_verification
                        )
                        self.contact_affordance_object_in = nnx.Linear(
                            config.object_affordance_hidden_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_phase_in = nnx.Linear(
                            action_expert_config.width,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_phase_logits_in = nnx.Linear(
                            4,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_state_in = nnx.Linear(
                            config.action_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_memory_in = nnx.Linear(
                            config.persistent_memory_hidden_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        # PSM assigns stable semantics to memory slots (fast
                        # state, target identity, reference identity, and plan
                        # verification).  Cross-attention is otherwise
                        # permutation invariant over its context, so preserve
                        # those slot roles explicitly in the contact pathway.
                        self.contact_affordance_memory_position = nnx.Embed(
                            config.persistent_memory_tokens,
                            fusion_dim,
                            rngs=rngs,
                        )
                        # Contact decisions need the uncompressed physical
                        # program as well as recurrent memory.  In particular,
                        # close/release risk depends on which ordered subgoal
                        # is active and which one follows it.  The final action
                        # residual remains zero initialized, preserving the
                        # inherited PPWM policy exactly at initialization.
                        self.contact_affordance_program_in = nnx.Linear(
                            config.persistent_memory_hidden_dim,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_program_position = nnx.Embed(
                            config.persistent_memory_subgoal_slots,
                            fusion_dim,
                            rngs=rngs,
                        )
                        if config.contact_affordance_clause_plan_verification:
                            # Preserve the full slot-to-clause distribution.
                            # Its entropy and competing alignments distinguish
                            # an uncertain linguistic phase from a confident
                            # physical program with otherwise similar content.
                            self.contact_affordance_clause_attention_in = nnx.Linear(
                                config.persistent_clause_plan_slots,
                                fusion_dim,
                                rngs=rngs,
                            )
                        if config.contact_affordance_future_verification:
                            self.contact_affordance_future_in = nnx.Linear(
                                config.object_future_hidden_dim,
                                fusion_dim,
                                rngs=rngs,
                            )
                        if config.contact_affordance_relation_verification:
                            # PSM already grounds manipulated/destination roles
                            # and composes source/destination predicates.  Keep
                            # those four semantic roles distinct rather than
                            # asking the final verifier to reconstruct them from
                            # compressed memory and program tokens.
                            self.contact_affordance_relation_state_in = nnx.Linear(
                                config.persistent_memory_hidden_dim,
                                fusion_dim,
                                rngs=rngs,
                            )
                            self.contact_affordance_grounded_relation_in = nnx.Linear(
                                config.persistent_memory_hidden_dim,
                                fusion_dim,
                                rngs=rngs,
                            )
                            self.contact_affordance_bound_role_in = nnx.Linear(
                                config.persistent_memory_hidden_dim,
                                fusion_dim,
                                rngs=rngs,
                            )
                            self.contact_affordance_relation_position = nnx.Embed(
                                4,
                                fusion_dim,
                                rngs=rngs,
                            )
                        self.contact_affordance_action_position = nnx.Embed(
                            config.action_horizon,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_blocks = [
                            _ExplicitActionReasonerBlock(
                                fusion_dim,
                                config.contact_affordance_fusion_num_heads,
                                config.contact_affordance_fusion_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(
                                config.contact_affordance_fusion_layers
                            )
                        ]
                        self.contact_affordance_risk_logit = nnx.Linear(
                            fusion_dim,
                            1,
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.contact_affordance_risk_in = nnx.Linear(
                            1,
                            fusion_dim,
                            rngs=rngs,
                        )
                        self.contact_affordance_token_out = nnx.Linear(
                            fusion_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.retrieved_demo_conditioning:
                        demo_dim = config.retrieved_demo_hidden_dim
                        self.action_prior_demo_action_in = nnx.Linear(
                            config.action_dim, demo_dim, rngs=rngs
                        )
                        self.action_prior_demo_position = nnx.Embed(
                            config.action_horizon, demo_dim, rngs=rngs
                        )
                        self.action_prior_demo_plan_in = nnx.Linear(
                            config.retrieved_demo_plan_dim, demo_dim, rngs=rngs
                        )
                        self.action_prior_demo_plan_steps = (
                            config.retrieved_demo_plan_steps
                        )
                        self.action_prior_demo_reliability_loss_weight = (
                            config.retrieved_demo_reliability_loss_weight
                        )
                        self.action_prior_demo_plan_position = nnx.Embed(
                            config.retrieved_demo_plan_steps, demo_dim, rngs=rngs
                        )
                        self.action_prior_demo_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            demo_dim,
                            rngs=rngs,
                        )
                        self.action_prior_demo_blocks = [
                            _ExplicitActionReasonerBlock(
                                demo_dim,
                                config.retrieved_demo_num_heads,
                                config.retrieved_demo_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.retrieved_demo_layers)
                        ]
                        self.action_prior_demo_token_out = nnx.Linear(
                            demo_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        # A bounded learned reliability scalar lets the policy
                        # suppress misleading retrieved context. Zero init
                        # starts the multiplier at exactly one.
                        self.action_prior_demo_gate_in = nnx.Linear(
                            2 * demo_dim, demo_dim, rngs=rngs
                        )
                        self.action_prior_demo_gate_out = nnx.Linear(
                            demo_dim,
                            1,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        if config.compositional_demo_routing:
                            self.action_prior_demo_slots = (
                                config.compositional_demo_slots
                            )
                            self.action_prior_demo_router_temperature = (
                                config.compositional_demo_router_temperature
                            )
                            self.action_prior_demo_router_loss_weight = (
                                config.compositional_demo_router_loss_weight
                            )
                            self.action_prior_demo_slot_position = nnx.Embed(
                                config.compositional_demo_slots,
                                demo_dim,
                                rngs=rngs,
                            )
                            self.action_prior_demo_progress_in = nnx.Linear(
                                1, demo_dim, rngs=rngs
                            )
                            self.action_prior_demo_router_blocks = [
                                _ExplicitActionReasonerBlock(
                                    demo_dim,
                                    config.retrieved_demo_num_heads,
                                    config.retrieved_demo_mlp_dim,
                                    rngs=rngs,
                                )
                                for _ in range(
                                    config.compositional_demo_router_layers
                                )
                            ]
                            self.action_prior_demo_router_out = nnx.Linear(
                                demo_dim,
                                1,
                                kernel_init=nnx.initializers.zeros_init(),
                                bias_init=nnx.initializers.zeros_init(),
                                rngs=rngs,
                            )
                            # Prefer the first ordered subgoal before the
                            # learned router has received supervision. Exact
                            # atomic retrieval masks every later slot anyway.
                            self.action_prior_demo_router_order_bias = nnx.Param(
                                jnp.linspace(
                                    1.0,
                                    -1.0,
                                    config.compositional_demo_slots,
                                    dtype=jnp.float32,
                                )
                            )
                    if config.structured_rationale_reasoner:
                        rationale_dim = config.structured_rationale_hidden_dim
                        self.action_prior_rationale_temperature = (
                            config.structured_rationale_temperature
                        )
                        self.action_prior_rationale_loss_weight = (
                            config.structured_rationale_loss_weight
                        )
                        self.action_prior_rationale_neutral_eps = tuple(
                            config.structured_rationale_neutral_eps
                        )
                        self.action_prior_rationale_action_q01 = tuple(
                            config.structured_rationale_action_q01
                        )
                        self.action_prior_rationale_action_q99 = tuple(
                            config.structured_rationale_action_q99
                        )
                        self.action_prior_rationale_class_weights = tuple(
                            tuple(weights)
                            for weights in config.structured_rationale_class_weights
                        )
                        self.action_prior_rationale_context_in = nnx.Linear(
                            paligemma_config.width, rationale_dim, rngs=rngs
                        )
                        self.action_prior_rationale_axis_queries = nnx.Embed(
                            7, rationale_dim, rngs=rngs
                        )
                        self.action_prior_rationale_state_in = nnx.Linear(
                            config.action_dim, rationale_dim, rngs=rngs
                        )
                        self.action_prior_rationale_blocks = [
                            _SpatialRelationReasonerBlock(
                                rationale_dim,
                                config.structured_rationale_num_heads,
                                config.structured_rationale_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(config.structured_rationale_layers)
                        ]
                        self.action_prior_rationale_logits = nnx.Linear(
                            rationale_dim, 3, rngs=rngs
                        )
                        self.action_prior_rationale_class_embeddings = nnx.Embed(
                            3, rationale_dim, rngs=rngs
                        )
                        self.action_prior_rationale_waypoint_queries = nnx.Embed(
                            config.action_horizon, rationale_dim, rngs=rngs
                        )
                        self.action_prior_rationale_fusion = (
                            _ExplicitActionReasonerBlock(
                                rationale_dim,
                                config.structured_rationale_num_heads,
                                config.structured_rationale_mlp_dim,
                                rngs=rngs,
                            )
                        )
                        # Exact function preservation at Stage-6 inheritance:
                        # the auxiliary classifier can learn immediately, but
                        # its action path starts as an all-zero residual.
                        self.action_prior_rationale_token_out = nnx.Linear(
                            rationale_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    if config.discrete_action_codebook_path is not None:
                        codebook_path = pathlib.Path(
                            config.discrete_action_codebook_path
                        ).resolve()
                        with np.load(codebook_path, allow_pickle=False) as archive:
                            if int(archive['format_version']) != 2:
                                raise ValueError(
                                    f'unsupported action codebook: {codebook_path}'
                                )
                            codebook = np.asarray(
                                archive['codebook'], dtype=np.float32
                            )
                            code_counts = np.asarray(
                                archive['counts'], dtype=np.float32
                            )
                            step_codebook = np.asarray(
                                archive['step_codebook'], dtype=np.float32
                            )
                            step_code_counts = np.asarray(
                                archive['step_counts'], dtype=np.float32
                            )
                        expected_shape = (
                            codebook.shape[0],
                            config.action_horizon,
                            config.action_dim,
                        )
                        if codebook.shape != expected_shape:
                            raise ValueError(
                                'action codebook shape must be '
                                f'[codes, {config.action_horizon}, '
                                f'{config.action_dim}], got {codebook.shape}'
                            )
                        if (
                            step_codebook.ndim != 2
                            or step_codebook.shape[1] != config.action_dim
                        ):
                            raise ValueError(
                                'step action codebook shape must be '
                                f'[codes, {config.action_dim}], got '
                                f'{step_codebook.shape}'
                            )
                        if code_counts.shape != (codebook.shape[0],):
                            raise ValueError('global code counts do not match codebook')
                        if step_code_counts.shape != (step_codebook.shape[0],):
                            raise ValueError('step code counts do not match codebook')
                        self.action_prior_discrete_loss_weight = (
                            config.discrete_action_codebook_loss_weight
                        )
                        self.action_prior_discrete_step_loss_weight = (
                            config.discrete_action_step_codebook_loss_weight
                        )
                        self.action_prior_discrete_auxiliary_loss = (
                            config.discrete_action_auxiliary_loss
                        )
                        self.action_prior_discrete_temperature = (
                            config.discrete_action_codebook_temperature
                        )
                        self.action_prior_discrete_robot_dim = (
                            config.discrete_action_codebook_robot_dim
                        )
                        self.action_prior_discrete_codebook = nnx.Param(
                            jnp.asarray(codebook)
                        )
                        self.action_prior_discrete_codebook_steps = nnx.Param(
                            jnp.asarray(step_codebook)
                        )
                        self.action_prior_discrete_codebook_weights = nnx.Param(
                            jnp.asarray(
                                _inverse_sqrt_class_weights(
                                    code_counts,
                                    config.discrete_action_class_weight_clip,
                                )
                            )
                        )
                        self.action_prior_discrete_codebook_step_weights = nnx.Param(
                            jnp.asarray(
                                _inverse_sqrt_class_weights(
                                    step_code_counts,
                                    config.discrete_action_class_weight_clip,
                                )
                            )
                        )
                        discrete_dim = config.discrete_action_codebook_hidden_dim
                        self.action_prior_discrete_context_in = nnx.Linear(
                            config.action_prior_horizon
                            * config.action_prior_hidden_dim,
                            discrete_dim,
                            rngs=rngs,
                        )
                        self.action_prior_discrete_context_out = nnx.Linear(
                            discrete_dim, discrete_dim, rngs=rngs
                        )
                        self.action_prior_discrete_logits = nnx.Linear(
                            discrete_dim,
                            codebook.shape[0],
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.action_prior_discrete_action_in = nnx.Linear(
                            config.action_dim, discrete_dim, rngs=rngs
                        )
                        self.action_prior_discrete_token_out = nnx.Linear(
                            discrete_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.action_prior_discrete_step_queries = nnx.Embed(
                            config.action_horizon, discrete_dim, rngs=rngs
                        )
                        self.action_prior_discrete_step_context_in = nnx.Linear(
                            config.action_prior_hidden_dim,
                            discrete_dim,
                            rngs=rngs,
                        )
                        self.action_prior_discrete_step_blocks = [
                            _ExplicitActionReasonerBlock(
                                discrete_dim,
                                config.discrete_action_step_reasoner_num_heads,
                                config.discrete_action_step_reasoner_mlp_dim,
                                rngs=rngs,
                            )
                            for _ in range(
                                config.discrete_action_step_reasoner_layers
                            )
                        ]
                        self.action_prior_discrete_step_logits = nnx.Linear(
                            discrete_dim,
                            step_codebook.shape[0],
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                        self.action_prior_discrete_step_action_in = nnx.Linear(
                            config.action_dim, discrete_dim, rngs=rngs
                        )
                        self.action_prior_discrete_step_token_out = nnx.Linear(
                            discrete_dim,
                            action_expert_config.width,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
            if config.persistent_subgoal_memory:
                persistent_memory_action_dim = (
                    config.active_action_dim or config.action_dim
                    if config.persistent_memory_action_dim is None
                    else config.persistent_memory_action_dim
                )
                self.persistent_memory = PersistentSubgoalMemory(
                    prefix_dim=paligemma_config.width,
                    policy_dim=action_expert_config.width,
                    state_dim=config.action_dim,
                    # Verification predicts only physical robot dimensions.
                    # The remaining model action columns are static padding
                    # and would otherwise dilute this auxiliary objective.
                    action_dim=persistent_memory_action_dim,
                    previous_action_dim=persistent_memory_action_dim,
                    memory_tokens=config.persistent_memory_tokens,
                    hidden_dim=config.persistent_memory_hidden_dim,
                    subgoal_slots=config.persistent_memory_subgoal_slots,
                    fast_tokens=config.persistent_memory_fast_tokens,
                    fast_update_rate=config.persistent_memory_fast_update_rate,
                    slow_update_rate=config.persistent_memory_slow_update_rate,
                    bounded_policy_gain=(
                        config.persistent_memory_bounded_policy_gain
                    ),
                    compositional_phase_init_scale=(
                        config.persistent_memory_compositional_phase_init_scale
                    ),
                    rngs=rngs,
                )
                if config.persistent_causal_frontier_transition_gate_v1:
                    self.persistent_memory.causal_frontier_transition_in_v1 = (
                        nnx.Linear(
                            4 * config.persistent_memory_hidden_dim,
                            config.persistent_memory_hidden_dim,
                            rngs=rngs,
                        )
                    )
                    self.persistent_memory.causal_frontier_transition_score_v1 = (
                        nnx.Linear(
                            config.persistent_memory_hidden_dim,
                            2,
                            kernel_init=nnx.initializers.zeros_init(),
                            bias_init=nnx.initializers.zeros_init(),
                            rngs=rngs,
                        )
                    )
                if config.persistent_structured_demo_language:
                    self.persistent_memory.structured_demo = (
                        _StructuredDemoModule(
                            semantic_dim=config.structured_demo_semantic_dim,
                            hidden_dim=config.structured_demo_hidden_dim,
                            policy_dim=action_expert_config.width,
                            semantic_slots=config.structured_demo_semantic_slots,
                            plan_steps=config.structured_demo_plan_steps,
                            plan_dim=config.structured_demo_plan_dim,
                            action_steps=config.structured_demo_action_steps,
                            action_dim=config.action_dim,
                            rngs=rngs,
                        )
                    )
                    self.spatial_language_aux = _SpatialLanguageAux(
                        hidden_dim=config.structured_demo_hidden_dim,
                        vocabulary_size=(
                            config.spatial_language_vocabulary_size
                        ),
                        language_steps=config.spatial_language_steps,
                        bos_token_id=config.spatial_language_bos_token_id,
                        rngs=rngs,
                    )
                    self.spatial_language_auxiliary_schedule = (
                        config.spatial_language_auxiliary_initial_weight,
                        config.spatial_language_auxiliary_peak_weight,
                        config.spatial_language_auxiliary_final_weight,
                        config.spatial_language_auxiliary_warmup_steps,
                        config.spatial_language_auxiliary_decay_start_step,
                        config.spatial_language_auxiliary_total_steps,
                    )
                if config.persistent_conditional_memory_policy_bridge:
                    self.persistent_memory.conditional_memory_policy_bridge = (
                        _ConditionalMemoryPolicyBridge(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            policy_dim=action_expert_config.width,
                            memory_tokens=config.persistent_memory_tokens,
                            fast_tokens=config.persistent_memory_fast_tokens,
                            subgoal_slots=config.persistent_memory_subgoal_slots,
                            action_tokens=config.action_horizon,
                            bottleneck_dim=(
                                config.persistent_conditional_memory_policy_bridge_rank
                            ),
                            rngs=rngs,
                        )
                    )
                if config.persistent_geometry_aux_v1:
                    self.persistent_memory.geometry_aux_v3 = _geometry_nnx.GeometryAuxV3(
                        prefix_dim=paligemma_config.width,
                        hidden_dim=config.persistent_memory_hidden_dim,
                        policy_dim=action_expert_config.width,
                        action_tokens=config.action_horizon,
                        bottleneck_dim=config.persistent_geometry_aux_bottleneck_dim,
                        rngs=rngs,
                    )
                if config.persistent_geometry_external_residual_v1:
                    self.persistent_memory.geometry_external_residual_v1 = (
                        _geometry_nnx.GeometryExternalResidualV1(
                            prefix_dim=paligemma_config.width,
                            hidden_dim=config.persistent_memory_hidden_dim,
                            role_count=5,
                            rngs=rngs,
                        )
                    )
                if config.persistent_temporal_role_memory_v1:
                    self.persistent_memory.action_conditioned_temporal_object_residual_v1 = (
                        _temporal_role_nnx.ActionConditionedTemporalRoleMemoryV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            rngs=rngs,
                        )
                    )
                if config.persistent_cross_view_role_consensus_v1:
                    self.persistent_memory.cross_view_role_consensus_v1 = (
                        _cross_view_role_nnx.CrossViewRoleConsensusV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            rngs=rngs,
                        )
                    )
                    self.cross_view_role_contrastive_temperature = (
                        config.persistent_cross_view_role_contrastive_temperature
                    )
                if config.persistent_contact_risk_calibrated_role_residual_v1:
                    self.persistent_memory.contact_risk_calibrated_role_residual_v1 = (
                        _contact_risk_nnx.ContactRiskCalibratedRoleResidualV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            state_dim=8,
                            rngs=rngs,
                        )
                    )
                    self.contact_risk_auxiliary_loss_weight = (
                        config.persistent_contact_risk_auxiliary_loss_weight
                    )
                    self.contact_risk_class_weights = tuple(
                        config.persistent_contact_risk_class_weights
                    )
                if config.persistent_relational_role_composer_residual_v1:
                    self.persistent_memory.relational_role_composer_residual_v1 = (
                        _relational_role_nnx.RelationalRoleComposerResidualV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            rngs=rngs,
                        )
                    )
                    self.relational_role_class_weights = tuple(
                        config.persistent_relational_role_class_weights
                    )
                if config.persistent_clause_role_binding_verifier_v1:
                    self.persistent_memory.clause_role_binding_verifier_v1 = (
                        _clause_role_binding_nnx.ClauseRoleBindingVerifierV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            slot_count=config.persistent_memory_subgoal_slots,
                            rngs=rngs,
                        )
                    )
                    self.clause_role_binding_source_class_weights = tuple(
                        config.persistent_clause_role_binding_source_class_weights
                    )
                    self.clause_role_binding_destination_class_weights = tuple(
                        config.persistent_clause_role_binding_destination_class_weights
                    )
                if config.persistent_semantic_frontier_completion_verifier_v1:
                    self.persistent_memory.semantic_frontier_completion_verifier_v1 = (
                        _semantic_frontier_nnx.SemanticFrontierCompletionVerifierV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            slot_count=config.persistent_memory_subgoal_slots,
                            rngs=rngs,
                        )
                    )
                    self.semantic_frontier_completion_auxiliary_loss_weight = (
                        config.persistent_semantic_frontier_completion_auxiliary_loss_weight
                    )
                    self.semantic_frontier_completion_class_weights = tuple(
                        tuple(row)
                        for row in config.persistent_semantic_frontier_completion_class_weights
                    )
                if config.persistent_hierarchical_clause_event_alignment_v1:
                    self.persistent_memory.hierarchical_clause_event_alignment_v1 = (
                        _hcea_nnx.HierarchicalClauseEventAlignmentV1(
                            hidden_dim=config.persistent_memory_hidden_dim,
                            slot_count=config.persistent_memory_subgoal_slots,
                            rngs=rngs,
                        )
                    )
                    self.hierarchical_clause_event_alignment_auxiliary_loss_weight = (
                        config.persistent_hierarchical_clause_event_alignment_auxiliary_loss_weight
                    )
                if config.persistent_hcea_causal_recovery_action_experts_v1:
                    self.persistent_memory.hcea_causal_recovery_action_experts_v1 = (
                        _hcea_recovery_nnx.HceaCausalRecoveryActionExpertsV1(
                            action_token_dim=self.action_in_proj.out_features,
                            hidden_dim=config.persistent_memory_hidden_dim,
                            slot_count=config.persistent_memory_subgoal_slots,
                            action_horizon=config.action_horizon,
                            rngs=rngs,
                        )
                    )
                    self.hcea_causal_recovery_intent_loss_weight = (
                        config.persistent_hcea_causal_recovery_intent_loss_weight
                    )
                if config.persistent_hcea_causal_role_identity_transport_expert_v1:
                    self.persistent_memory.hcea_causal_role_identity_transport_expert_v1 = (
                        _hcea_role_transport_nnx.HceaCausalRoleIdentityTransportExpertV1(
                            action_token_dim=self.action_in_proj.out_features,
                            role_dim=config.persistent_memory_hidden_dim,
                            slot_count=config.persistent_memory_subgoal_slots,
                            action_horizon=config.action_horizon,
                            rngs=rngs,
                        )
                    )
                    self.hcea_causal_role_identity_transport_loss_weight = (
                        config.persistent_hcea_causal_role_identity_transport_loss_weight
                    )
                if config.persistent_hmca_v4:
                    self.hierarchical_memory_conditional_adapters = (
                        _hmca_nnx.HierarchicalMemoryConditionalAdapters(
                            model_dim=action_expert_config.width,
                            semantic_dim=5 * config.persistent_memory_hidden_dim,
                            action_horizon=config.action_horizon,
                            rank=config.persistent_hmca_v4_rank,
                            alpha=config.persistent_hmca_v4_alpha,
                            rngs=rngs,
                        )
                    )
                if config.persistent_clause_plan_v1:
                    self.persistent_memory.clause_plan_adapter = (
                        _clause_plan_nnx.ClausePlanAdapterV1(
                            prefix_dim=paligemma_config.width,
                            hidden_dim=config.persistent_memory_hidden_dim,
                            rank=config.persistent_clause_plan_rank,
                            clause_slots=config.persistent_clause_plan_slots,
                            monotonic_strength=(
                                config.persistent_clause_plan_monotonic_strength
                            ),
                            rngs=rngs,
                        )
                    )
                if (
                    config.contact_affordance_predictive_fusion
                    and config.contact_affordance_relation_verification
                    and config.contact_affordance_transition_verification
                ):
                    # Role-specific zero gate: exact Joint51 inheritance at
                    # initialization, then learned protection against
                    # low-confidence visual identity overwrites.
                    self.persistent_memory.role_identity_confidence_gate = (
                        nnx.Param(jnp.zeros((2,), dtype=jnp.float32))
                    )
                self.persistent_memory_policy_gain_warmup_steps = (
                    config.persistent_memory_policy_gain_warmup_steps
                )
                self.persistent_memory_flow_replan_indices = (
                    config.persistent_memory_flow_replan_indices
                )
                self.persistent_memory_auxiliary_decay_steps = (
                    config.persistent_memory_auxiliary_decay_steps
                )
                self.persistent_memory_auxiliary_final_multiplier = (
                    config.persistent_memory_auxiliary_final_multiplier
                )
                self.persistent_memory_role_contrastive_group_size = (
                    config.persistent_memory_role_contrastive_group_size
                )
                self.persistent_memory_loss_weights = {
                    'progress': config.persistent_memory_subgoal_progress_loss_weight,
                    'transition': config.persistent_memory_subgoal_transition_loss_weight,
                    'action': config.persistent_memory_action_verification_loss_weight,
                    'route': config.persistent_memory_causal_route_loss_weight,
                    'cross_camera': config.persistent_memory_cross_camera_loss_weight,
                    'role_distinctness': (
                        config.persistent_memory_role_distinctness_loss_weight
                    ),
                    'role_object_exclusivity': (
                        config.persistent_memory_role_object_exclusivity_loss_weight
                    ),
                    'between_pair_entropy': (
                        config.persistent_memory_between_pair_entropy_loss_weight
                    ),
                    'source_reference': (
                        config.persistent_memory_source_reference_loss_weight
                    ),
                    'destination_reference': (
                        config.persistent_memory_destination_reference_loss_weight
                    ),
                    'condition_state': (
                        config.persistent_memory_condition_state_loss_weight
                    ),
                    'destination_qualifier': (
                        config.persistent_memory_destination_qualifier_loss_weight
                    ),
                    'temporal_role': config.persistent_memory_temporal_role_loss_weight,
                    'cross_view_role_contrastive': (
                        config.persistent_cross_view_role_contrastive_loss_weight
                    ),
                    'contact_risk': (
                        config.persistent_contact_risk_auxiliary_loss_weight
                    ),
                    'relational_role': (
                        config.persistent_relational_role_auxiliary_loss_weight
                    ),
                    'clause_role_binding': (
                        config.persistent_clause_role_binding_auxiliary_loss_weight
                    ),
                    'factorized_role': (
                        config.persistent_memory_factorized_role_loss_weight
                    ),
                    'factor_attention_alignment': (
                        config.persistent_memory_factor_attention_alignment_loss_weight
                    ),
                    'object_slot_reconstruction': (
                        config.persistent_memory_object_slot_reconstruction_loss_weight
                    ),
                    'visual_language_role': (
                        config.persistent_memory_visual_language_role_loss_weight
                    ),
                    'role_contrastive': (
                        config.supervised_role_identity_contrastive_loss_weight
                    ),
                }
                self.persistent_memory_factorized_class_counts = {
                    'operation': config.persistent_memory_operation_class_counts,
                    'source_relation': (
                        config.persistent_memory_source_relation_class_counts
                    ),
                    'destination_relation': (
                        config.persistent_memory_destination_relation_class_counts
                    ),
                    'condition': config.persistent_memory_condition_class_counts,
                    'destination_qualifier': (
                        config.persistent_memory_destination_qualifier_class_counts
                    ),
                }
                self.persistent_memory_semantic_phase_class_counts = (
                    config.persistent_memory_semantic_phase_class_counts
                )
                self.persistent_memory_flow_phase_class_counts = (
                    config.persistent_memory_flow_phase_class_counts
                )
                self.persistent_memory_flow_phase_count_floor = (
                    config.persistent_memory_flow_phase_count_floor
                )
        else:
            self.state_proj = nnx.Linear(
                config.action_dim, action_expert_config.width, rngs=rngs
            )
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
        self.action_out_proj = nnx.Linear(
            action_expert_config.width, config.action_dim, rngs=rngs
        )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, 'b s emb'],
        at.Bool[at.Array, 'b s'],
        at.Bool[at.Array, ' s'],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    'b -> b s',
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method='embed')
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _physical_image_mask(self, observation, name):
        """Return a mask that never treats a retrieved montage as a camera."""
        mask = observation.image_masks[name].astype(jnp.bool_)
        if not self.grounded_demonstration_camera_context_only:
            return mask
        if name != 'right_wrist_0_rgb':
            return mask
        grounded = observation.grounded_context_mask
        if grounded is None:
            raise ValueError(
                'grounded context-only camera mode requires grounded_context_mask'
            )
        grounded = grounded.astype(jnp.bool_)
        if grounded.shape != mask.shape:
            raise ValueError('grounded context mask must match the image batch')
        if not isinstance(mask, jax.core.Tracer):
            if bool(np.any(np.asarray(grounded & ~mask))):
                raise ValueError(
                    'grounded context cannot be enabled without its VLM image'
                )
        return mask & ~grounded

    def _partition_contextual_prefix(
        self,
        observation: _model.Observation,
        contextual_prefix: jax.Array,
        prefix_mask: jax.Array,
    ) -> RACGPrefixView:
        """Partition contextual states using the formal camera ordering."""
        camera_names = tuple(observation.images)
        if camera_names != _model.IMAGE_KEYS:
            raise ValueError(
                f'contextual camera order drifted: {camera_names!r}'
            )
        language_count = (
            observation.tokenized_prompt.shape[-1]
            if observation.tokenized_prompt is not None
            else 0
        )
        visual_count = contextual_prefix.shape[1] - language_count
        if visual_count <= 0 or visual_count % len(camera_names):
            raise ValueError(
                'contextual visual prefix cannot be partitioned by camera'
            )
        patches_per_camera = visual_count // len(camera_names)
        grid = math.isqrt(patches_per_camera)
        if grid * grid != patches_per_camera:
            raise ValueError('RACG camera patch count must form a square grid')
        visual = contextual_prefix[:, :visual_count].reshape(
            contextual_prefix.shape[0],
            len(camera_names),
            patches_per_camera,
            contextual_prefix.shape[-1],
        )
        camera_valid = jnp.stack(
            [self._physical_image_mask(observation, name) for name in camera_names],
            axis=1,
        ).astype(jnp.bool_)
        right_valid = camera_valid[:, camera_names.index('right_wrist_0_rgb')]
        if not isinstance(right_valid, jax.core.Tracer) and bool(
            np.any(np.asarray(right_valid))
        ):
            raise ValueError('RACG-v1 rejects a valid right-wrist camera')
        patch_valid = jnp.broadcast_to(
            camera_valid[:, :, None],
            (
                contextual_prefix.shape[0],
                len(camera_names),
                patches_per_camera,
            ),
        )
        patch_valid = patch_valid & prefix_mask[:, :visual_count].reshape(
            contextual_prefix.shape[0], len(camera_names), patches_per_camera
        )
        return RACGPrefixView(
            visual_states=visual,
            visual_mask=patch_valid,
            language_states=contextual_prefix[:, visual_count:],
            language_mask=prefix_mask[:, visual_count:],
            camera_names=camera_names,
            grid_size=grid,
        )

    def _racg_state_from_observation(
        self,
        observation: _model.Observation,
        prefix_states,
        prefix_mask,
        hetm_private_state,
    ):
        """Encode one RACG-v1 scene and propose an atomic anchor update."""
        if not hasattr(self, 'racg'):
            return None
        if hetm_private_state is None:
            raise ValueError('RACG requires current HETM outputs')
        view = self._partition_contextual_prefix(
            observation, prefix_states, prefix_mask
        )
        batch_size = observation.state.shape[0]
        role_present = observation.racg_role_valid_mask
        relation_kind = observation.racg_relation_kind
        role_span_mask = observation.racg_role_span_mask
        source_reference_span_mask = (
            observation.factorized_source_reference_span_mask
        )
        destination_reference_span_mask = (
            observation.factorized_destination_reference_span_mask
        )
        if (
            role_present is None
            or role_present.shape != (batch_size, _racg.ROLE_COUNT)
        ):
            raise ValueError('RACG role validity must be [batch,6]')
        if relation_kind is None or relation_kind.shape != (batch_size,):
            raise ValueError('RACG relation kind must be [batch]')
        if (
            role_span_mask is None
            or role_span_mask.shape
            != (batch_size, _racg.ROLE_COUNT, view.language_states.shape[1])
        ):
            raise ValueError('RACG role spans must be [batch,6,prompt]')
        if (
            source_reference_span_mask is None
            or source_reference_span_mask.shape
            != (batch_size, view.language_states.shape[1])
        ):
            raise ValueError(
                'RACG source-reference spans must be [batch,prompt]'
            )
        if (
            destination_reference_span_mask is None
            or destination_reference_span_mask.shape
            != (batch_size, 2, view.language_states.shape[1])
        ):
            raise ValueError(
                'RACG destination-reference spans must be [batch,2,prompt]'
            )
        supplied = (
            observation.racg_target_anchor,
            observation.racg_target_geometry,
            observation.racg_target_anchor_valid,
        )
        if sum(value is not None for value in supplied) not in (0, 3):
            raise ValueError('RACG target-anchor state must be supplied atomically')
        if supplied[0] is None:
            previous_anchor = jnp.zeros(
                (batch_size, _hetm.HIDDEN_DIM), jnp.float32
            )
            previous_geometry = jnp.zeros((batch_size, 5), jnp.float32)
            previous_valid = jnp.zeros((batch_size,), jnp.bool_)
        else:
            previous_anchor, previous_geometry, previous_valid = supplied
        episode_start = observation.racg_episode_start
        if episode_start is None:
            episode_start = observation.hetm_episode_start
        if episode_start is None:
            episode_start = jnp.ones((batch_size,), jnp.bool_)
        carried_anchor = jnp.where(
            episode_start[:, None], jnp.zeros_like(previous_anchor), previous_anchor
        )
        carried_geometry = jnp.where(
            episode_start[:, None],
            jnp.zeros_like(previous_geometry),
            previous_geometry,
        )
        carried_valid = previous_valid & ~episode_start.astype(jnp.bool_)
        previous_actions = observation.hetm_previous_actions
        if previous_actions is None:
            previous_actions = jnp.zeros(
                (
                    batch_size,
                    _hetm.PREVIOUS_ACTION_STEPS,
                    self.active_action_dim,
                ),
                jnp.float32,
            )
        previous_actions = previous_actions[
            :, : _hetm.PREVIOUS_ACTION_STEPS, : self.active_action_dim
        ]
        axis = jnp.linspace(-1.0, 1.0, view.grid_size, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(axis, axis, indexing='ij')
        xy = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)
        xy = jnp.broadcast_to(
            xy[None, None], (batch_size, 2, xy.shape[0], 2)
        )
        current = hetm_private_state['outputs']
        racg_view_valid = jnp.stack(
            [
                self._physical_image_mask(observation, name)
                for name in view.camera_names[:2]
            ],
            axis=1,
        )
        external_geometry = None
        if hasattr(self, 'racg_external_geometry'):
            external_geometry = self.racg_external_geometry(
                jax.lax.stop_gradient(view.visual_states[:, :2]),
                view.visual_mask[:, :2],
                racg_view_valid,
                role_present,
            )
        scene = self.racg.encode_scene(
            patches=jax.lax.stop_gradient(view.visual_states[:, :2]),
            patch_mask=view.visual_mask[:, :2],
            patch_xy=xy,
            view_valid=racg_view_valid,
            language_states=jax.lax.stop_gradient(view.language_states),
            language_mask=view.language_mask,
            hetm_role_states=current['role_states'],
            role_present=role_present,
            predicate_probabilities=current['predicate_probabilities'],
            frontier=current['frontier'],
            proprioception=observation.state,
            previous_actions=previous_actions,
            previous_target_anchor=carried_anchor,
            previous_target_geometry=carried_geometry,
            previous_target_valid=carried_valid,
            episode_start=episode_start,
            relation_kind=relation_kind,
            role_span_mask=role_span_mask,
            source_reference_span_mask=source_reference_span_mask,
            destination_reference_span_mask=destination_reference_span_mask,
            external_role_residual=(
                None if external_geometry is None else external_geometry.role_residual
            ),
        )
        proposed_valid = role_present[:, _racg.TARGET_ROLE] & jnp.any(
            view.visual_mask[:, :2], axis=(1, 2)
        )
        return {
            'scene': scene,
            'prefix_view': view,
            'external_geometry': external_geometry,
            'previous_anchor': carried_anchor,
            'previous_geometry': carried_geometry,
            'previous_valid': carried_valid,
            'episode_start': episode_start,
            'next_state': {
                'target_anchor': jnp.where(
                    proposed_valid[:, None],
                    scene.role_nodes[:, _racg.TARGET_ROLE],
                    carried_anchor,
                ),
                'target_geometry': jnp.where(
                    proposed_valid[:, None],
                    scene.role_geometry[:, _racg.TARGET_ROLE],
                    carried_geometry,
                ),
                'target_anchor_valid': jnp.where(
                    proposed_valid, jnp.ones_like(carried_valid), carried_valid
                ),
            },
        }

    @staticmethod
    def _masked_example_mean(values, mask):
        mask = mask.astype(jnp.bool_)
        numerator = jnp.sum(jnp.where(mask, values, 0.0), axis=-1)
        denominator = jnp.maximum(jnp.sum(mask, axis=-1), 1)
        return numerator / denominator

    def _racg_scene_auxiliary_loss(
        self, observation, racg_private_state, hetm_private_state
    ):
        """Compute RACG-v1 scene losses without leaking labels to policy inputs."""
        del hetm_private_state
        scene = racg_private_state['scene']
        batch_size = scene.graph_tokens.shape[0]
        total = jnp.zeros((batch_size,), jnp.float32)
        spans = observation.racg_role_span_mask
        if spans is not None:
            if spans.shape != (
                batch_size,
                _racg.ROLE_COUNT,
                scene.projected_language_tokens.shape[1],
            ):
                raise ValueError('RACG role spans must be [batch,6,prompt]')
            total = total + self.racg_loss_weights['role_align'] * (
                _racg.sample_local_role_alignment_loss(
                    scene.role_nodes,
                    scene.projected_language_tokens,
                    spans,
                )
            )
        identity_labels = observation.racg_role_identity_labels
        if identity_labels is not None:
            if identity_labels.shape != (batch_size, _racg.ROLE_COUNT):
                raise ValueError('RACG role identity labels must be [batch,6]')
            nodes = scene.role_nodes.astype(jnp.float32).reshape(
                batch_size * _racg.ROLE_COUNT, -1
            )
            labels = identity_labels.reshape(-1)
            valid_identity = labels >= 0
            nodes = _l2_normalize(nodes)
            logits = nodes @ nodes.T / 0.1
            eye = jnp.eye(logits.shape[0], dtype=jnp.bool_)
            candidate = (
                valid_identity[:, None] & valid_identity[None, :] & ~eye
            )
            positive = candidate & (labels[:, None] == labels[None, :])
            probabilities = jax.nn.softmax(
                jnp.where(candidate, logits, -1.0e30), axis=-1
            )
            positive_mass = jnp.sum(probabilities * positive, axis=-1)
            anchor_valid = jnp.any(positive, axis=-1)
            identity_loss = jnp.where(
                anchor_valid,
                -jnp.log(jnp.maximum(positive_mass, 1.0e-6)),
                0.0,
            ).reshape(batch_size, _racg.ROLE_COUNT)
            total = total + self.racg_loss_weights['role_align'] * (
                self._masked_example_mean(
                    identity_loss
                    * jnp.asarray(
                        _racg.ROLE_VALID_CLASS_WEIGHTS, dtype=jnp.float32
                    )[None, :],
                    anchor_valid.reshape(batch_size, _racg.ROLE_COUNT),
                )
            )
        cross_valid = observation.racg_crossview_role_valid
        if cross_valid is None:
            views_ok = jnp.all(
                racg_private_state['prefix_view'].visual_mask[:, :2],
                axis=(1, 2),
            )
            cross_valid = (
                observation.racg_role_valid_mask & views_ok[:, None]
            )
        if cross_valid.shape != (batch_size, _racg.ROLE_COUNT):
            raise ValueError('RACG cross-view validity must be [batch,6]')
        total = total + self.racg_loss_weights['crossview'] * (
            _racg.sample_local_cross_view_role_contrastive_loss(
                scene.cross_view_role_embeddings, cross_valid
            )
        )
        relation_target = observation.racg_relation_target
        relation_valid = observation.racg_relation_valid
        if (relation_target is None) != (relation_valid is None):
            raise ValueError('RACG relation targets must be supplied atomically')
        if relation_target is not None:
            total = total + self.racg_loss_weights['relation'] * (
                _racg.sample_local_relation_logits_loss(
                    scene.relation_logits, relation_target, relation_valid
                )
            )
        patch_valid = racg_private_state['prefix_view'].visual_mask[:, :2]
        total = total + self.racg_loss_weights['slot_reconstruction'] * (
            _racg.sample_local_slot_reconstruction_loss(
                scene.projected_patch_targets,
                scene.reconstructed_projected_patches,
                patch_valid,
            )
        )
        slots = scene.object_slots.astype(jnp.float32)
        normalized = _l2_normalize(slots)
        similarity = jnp.einsum('bvkh,bvlh->bvkl', normalized, normalized)
        off_diagonal = 1.0 - jnp.eye(
            similarity.shape[-1], dtype=jnp.float32
        )
        diversity = jnp.sum(
            jnp.square(similarity) * off_diagonal, axis=(-1, -2)
        )
        diversity = jnp.mean(
            diversity
            / (similarity.shape[-1] * (similarity.shape[-1] - 1)),
            axis=1,
        )
        total = total + self.racg_loss_weights['slot_diversity'] * diversity
        current = scene.role_nodes[:, _racg.TARGET_ROLE].astype(jnp.float32)
        previous = racg_private_state['previous_anchor'].astype(jnp.float32)
        cosine = jnp.sum(
            _l2_normalize(current) * _l2_normalize(previous), axis=-1
        )
        identity_valid = (
            racg_private_state['previous_valid']
            & observation.racg_role_valid_mask[:, _racg.TARGET_ROLE]
        )
        return total + self.racg_loss_weights['identity_transition'] * (
            jnp.where(identity_valid, 1.0 - cosine, 0.0)
        )

    def _persistent_phase_context(
        self,
        memory_context,
        ordered_program=None,
        frontier=None,
        slot_valid_mask=None,
    ):
        """Return one scale-stable PSM phase state for every policy consumer."""
        persistent = getattr(self, 'persistent_memory', None)
        memory_summary = (
            persistent.structured_memory_summary(memory_context)
            if persistent is not None
            else jnp.mean(memory_context, axis=1)
        )
        # Memory slots and program tokens are produced by different branches;
        # align their RMS scales before applying the explicit 1:0.5:0.25
        # semantic mixture so encoder norm drift cannot silently override the
        # intended global/current/future balance.
        memory_summary = _rms_normalize(memory_summary).astype(jnp.float32)
        if ordered_program is not None:
            if slot_valid_mask is None:
                slot_valid_mask = jnp.ones(frontier.shape, dtype=jnp.bool_)
            if slot_valid_mask.shape != frontier.shape:
                raise ValueError('persistent phase-context validity shape drifted')
            valid_frontier = frontier.astype(jnp.float32) * slot_valid_mask.astype(
                jnp.float32
            )
            valid_frontier = valid_frontier / jnp.maximum(
                jnp.sum(valid_frontier, axis=-1, keepdims=True), 1.0e-8
            )
            current_program = jnp.einsum(
                'bs,bsh->bh',
                valid_frontier.astype(ordered_program.dtype),
                ordered_program,
            )
            slot_count = ordered_program.shape[1]
            phase_ids = jnp.arange(slot_count, dtype=jnp.float32)
            phase_distance = phase_ids[None, :] - phase_ids[:, None]
            remaining_kernel = jnp.where(
                phase_distance >= 0.0,
                jnp.power(0.75, phase_distance),
                0.0,
            )
            remaining_weights = jnp.einsum(
                'bs,sr->br', valid_frontier, remaining_kernel
            )
            # Clause banks are padded to a fixed slot count.  Invalid trailing
            # clauses must contribute neither content nor normalization mass;
            # otherwise the shared phase summary can hallucinate future work.
            remaining_weights = remaining_weights * slot_valid_mask.astype(
                remaining_weights.dtype
            )
            remaining_weights = remaining_weights / jnp.maximum(
                jnp.sum(remaining_weights, axis=-1, keepdims=True), 1.0e-8
            )
            remaining_program = jnp.einsum(
                'bs,bsh->bh',
                remaining_weights.astype(ordered_program.dtype),
                ordered_program,
            )
            # Preserve the global structured memory while explicitly exposing
            # the phase being executed and its causal future.  No new
            # parameters are needed, so checkpoint inheritance remains exact.
            memory_summary = _rms_normalize(
                memory_summary
                + 0.5 * _rms_normalize(current_program).astype(jnp.float32)
                + 0.25 * _rms_normalize(remaining_program).astype(jnp.float32)
            ).astype(jnp.float32)
        return memory_summary

    def compute_persistent_memory_adarms(
        self,
        memory_context,
        ordered_program=None,
        frontier=None,
        slot_valid_mask=None,
    ):
        """Project structured causal PSM and active-plan state into AdaRMS."""
        if memory_context.ndim != 3:
            raise ValueError('persistent-memory AdaRMS expects [batch, slots, hidden]')
        if (ordered_program is None) != (frontier is None):
            raise ValueError(
                'persistent-memory AdaRMS requires program and frontier together'
            )
        if ordered_program is not None and (
            ordered_program.ndim != 3
            or ordered_program.shape[0] != memory_context.shape[0]
            or ordered_program.shape[2] != memory_context.shape[2]
            or frontier.shape != ordered_program.shape[:2]
        ):
            raise ValueError(
                'persistent-memory AdaRMS program/frontier shape drifted'
            )
        memory_summary = self._persistent_phase_context(
            memory_context, ordered_program, frontier, slot_valid_mask
        )
        memory_emb = self.persistent_memory_adarms_in(
            memory_summary
        )
        return self.persistent_memory_adarms_out(nnx.swish(memory_emb))

    def compute_phase_contact_action_film(
        self,
        action_tokens,
        contact_phase_tokens,
        memory_context,
        ordered_program,
        frontier,
        slot_valid_mask=None,
    ):
        """Modulate each action token by its contact phase and PSM frontier."""
        if action_tokens.ndim != 3 or contact_phase_tokens.shape != action_tokens.shape:
            raise ValueError(
                'phase-contact FiLM requires aligned [batch, action, hidden] tokens'
            )
        if (
            memory_context is None
            or memory_context.ndim != 3
            or memory_context.shape[0] != action_tokens.shape[0]
        ):
            raise ValueError('phase-contact FiLM requires batched persistent memory')
        if ordered_program is None or frontier is None:
            raise ValueError('phase-contact FiLM requires program and frontier')
        if (
            ordered_program.ndim != 3
            or ordered_program.shape[0] != action_tokens.shape[0]
            or ordered_program.shape[2] != memory_context.shape[2]
            or frontier.shape != ordered_program.shape[:2]
        ):
            raise ValueError('phase-contact FiLM program/frontier shape drifted')

        phase_context = self._persistent_phase_context(
            memory_context, ordered_program, frontier, slot_valid_mask
        )
        phase_latent = nnx.swish(
            self.phase_contact_film_memory_in(phase_context)
        )[:, None, :]
        contact_latent = nnx.swish(
            self.phase_contact_film_contact_in(contact_phase_tokens)
        )
        # The product makes contact semantics conditional on the active
        # cross-replan phase instead of merely summing two independent paths.
        joint = contact_latent * (1.0 + jnp.tanh(phase_latent))
        scale, shift = jnp.split(self.phase_contact_film_out(joint), 2, axis=-1)
        return action_tokens * (1.0 + jnp.tanh(scale)) + shift

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, ' b'],
        action_prior_tokens: at.Float[at.Array, 'b ah emb'] | None = None,
        action_reasoning_tokens: at.Float[at.Array, 'b guidance emb'] | None = None,
        action_contexts: at.Float[at.Array, 'b context hidden'] | None = None,
        persistent_memory_context: at.Float[at.Array, 'b memory hidden'] | None = None,
        hetm_condition: Mapping[str, at.Array] | None = None,
        racg_action_residual: at.Float[at.Array, 'b ah emb'] | None = None,
        persistent_ordered_program: at.Float[at.Array, 'b subgoal hidden'] | None = None,
        persistent_frontier: at.Float[at.Array, 'b subgoal'] | None = None,
        contact_phase_tokens: at.Float[at.Array, 'b ah emb'] | None = None,
        persistent_slot_valid_mask: at.Bool[at.Array, 'b subgoal'] | None = None,
        persistent_previous_frontier: at.Float[at.Array, 'b subgoal'] | None = None,
        persistent_previous_actions: at.Float[at.Array, 'b r pad'] | None = None,
        persistent_previous_actions_valid: at.Bool[at.Array, 'b'] | None = None,
        persistent_current_context: at.Float[at.Array, 'b hidden'] | None = None,
        persistent_previous_roles: at.Float[at.Array, 'b roles hidden'] | None = None,
        persistent_current_roles: at.Float[at.Array, 'b roles hidden'] | None = None,
        persistent_role_valid_mask: at.Bool[at.Array, 'b roles'] | None = None,
    ) -> tuple[
        at.Float[at.Array, 'b s emb'],
        at.Bool[at.Array, 'b s'],
        at.Bool[at.Array, ' s'],
        at.Float[at.Array, 'b emb'] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            if action_prior_tokens is not None:
                action_expert_tokens = action_expert_tokens + action_prior_tokens
            if action_reasoning_tokens is not None:
                action_expert_tokens = action_expert_tokens + (
                    self.compute_action_guidance_residual(
                        action_expert_tokens, action_reasoning_tokens
                    )
                )
            if racg_action_residual is not None:
                if racg_action_residual.shape != action_expert_tokens.shape:
                    raise ValueError('RACG residual must match action token shape')
                action_expert_tokens = _racg.apply_graph_action_residual(
                    action_expert_tokens, racg_action_residual
                )
            if hetm_condition is not None:
                hetm_scale = jnp.tanh(hetm_condition['film_scale'])[:, None, :]
                hetm_shift = hetm_condition['film_shift'][:, None, :]
                action_expert_tokens = (
                    action_expert_tokens * (1.0 + hetm_scale) + hetm_shift
                )
            if hasattr(self, 'state_film_in'):
                state_film = nnx.swish(self.state_film_in(obs.state))
                state_scale, state_shift = jnp.split(
                    self.state_film_out(state_film), 2, axis=-1
                )
                # tanh bounds the multiplicative path while the zero-init
                # projection makes this exactly an identity at initialization.
                action_expert_tokens = (
                    action_expert_tokens * (1.0 + jnp.tanh(state_scale)[:, None, :])
                    + state_shift[:, None, :]
                )
            if hasattr(self, 'phase_contact_film_memory_in'):
                # Ordinary auxiliary samples intentionally have no cross-
                # replan memory.  Keep those samples on the inherited policy
                # path, matching persistent-memory AdaRMS below.
                if persistent_memory_context is not None:
                    if contact_phase_tokens is None:
                        raise ValueError(
                            'phase-contact action FiLM requires contact phase tokens'
                        )
                    action_expert_tokens = self.compute_phase_contact_action_film(
                        action_expert_tokens,
                        contact_phase_tokens,
                        persistent_memory_context,
                        persistent_ordered_program,
                        persistent_frontier,
                        persistent_slot_valid_mask,
                    )
            recovery = getattr(
                getattr(self, 'persistent_memory', None),
                'hcea_causal_recovery_action_experts_v1',
                None,
            )
            if recovery is not None:
                required = (
                    persistent_ordered_program,
                    persistent_previous_frontier,
                    persistent_frontier,
                    persistent_previous_actions,
                    persistent_previous_actions_valid,
                    persistent_current_context,
                    persistent_slot_valid_mask,
                )
                if any(value is None for value in required):
                    raise ValueError('HCEA recovery experts require complete causal private state')
                recovery_residual, _ = recovery(
                    action_expert_tokens,
                    persistent_ordered_program,
                    persistent_previous_frontier,
                    persistent_frontier,
                    persistent_previous_actions,
                    persistent_previous_actions_valid,
                    persistent_current_context,
                    persistent_slot_valid_mask,
                )
                action_expert_tokens = action_expert_tokens + recovery_residual
            role_transport = getattr(
                getattr(self, 'persistent_memory', None),
                'hcea_causal_role_identity_transport_expert_v1',
                None,
            )
            if role_transport is not None:
                required = (
                    persistent_previous_roles,
                    persistent_current_roles,
                    persistent_previous_actions,
                    persistent_previous_actions_valid,
                    persistent_current_context,
                    persistent_frontier,
                    persistent_role_valid_mask,
                )
                if any(value is None for value in required):
                    raise ValueError(
                        'HCEA role transport requires complete causal role state'
                    )
                role_transport_residual, _ = role_transport(
                    action_expert_tokens,
                    persistent_previous_roles,
                    persistent_current_roles,
                    persistent_previous_actions,
                    persistent_current_context,
                    persistent_frontier,
                    persistent_role_valid_mask,
                    persistent_previous_actions_valid,
                )
                action_expert_tokens = action_expert_tokens + role_transport_residual
            adarms_cond = time_emb
            if hasattr(self, 'state_adarms_in'):
                state_emb = self.state_adarms_in(obs.state)
                state_emb = nnx.swish(state_emb)
                adarms_cond = adarms_cond + self.state_adarms_out(state_emb)
            if hasattr(self, 'context_adarms_in'):
                if action_contexts is None:
                    raise ValueError(
                        'context AdaRMS requires contextual action-prior states'
                    )
                context_emb = self.context_adarms_in(
                    jnp.mean(action_contexts, axis=1)
                )
                context_emb = nnx.swish(context_emb)
                context_delta = self.context_adarms_out(context_emb)
                if hasattr(self, 'specialist_module_router_score'):
                    specialist_gates = self.compute_specialist_module_gates(
                        action_contexts, obs.state
                    )
                    context_delta = context_delta * specialist_gates[:, :1]
                adarms_cond = adarms_cond + context_delta
            if hasattr(self, 'persistent_memory_adarms_in'):
                # Ordinary auxiliary samples have no cross-replan memory and
                # must remain exactly on the inherited policy path.
                if persistent_memory_context is not None:
                    adarms_cond = adarms_cond + self.compute_persistent_memory_adarms(
                        persistent_memory_context,
                        persistent_ordered_program,
                        persistent_frontier,
                        persistent_slot_valid_mask,
                    )
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(
                time_emb, 'b emb -> b s emb', s=self.action_horizon
            )
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def apply_multimodal_prefix_moe(
        self,
        prefix_tokens,
        prefix_mask,
        observation,
    ):
        """Apply a sparse, identity-initialized FiLM adapter before PaliGemma."""
        if observation.tokenized_prompt is not None:
            language_count = observation.tokenized_prompt.shape[1]
            image_count = prefix_tokens.shape[1] - language_count
            language_tokens = prefix_tokens[:, -language_count:]
            language_mask = observation.tokenized_prompt_mask
        else:
            image_count = prefix_tokens.shape[1]
            language_tokens = prefix_tokens
            language_mask = prefix_mask
        language_weight = language_mask.astype(prefix_tokens.dtype)
        pooled_language = jnp.sum(
            language_tokens * language_weight[..., None], axis=1
        ) / jnp.maximum(jnp.sum(language_weight, axis=1, keepdims=True), 1.0)
        router_hidden = nnx.swish(
            self.prefix_moe_router_in(pooled_language)
            + self.prefix_moe_state_in(observation.state)
        )
        router_logits = self.prefix_moe_router_out(router_hidden).astype(
            jnp.float32
        )
        probabilities = jax.nn.softmax(
            router_logits / self.prefix_moe_temperature, axis=-1
        )
        top_values, top_indices = jax.lax.top_k(
            probabilities, self.prefix_moe_top_k
        )
        top_values = top_values / jnp.maximum(
            jnp.sum(top_values, axis=-1, keepdims=True), 1e-8
        )
        gates = jnp.zeros_like(probabilities)
        batch_indices = jnp.arange(probabilities.shape[0])[:, None]
        gates = gates.at[batch_indices, top_indices].set(top_values)

        expert_scale_shift = jnp.stack(
            [
                expert_out(nnx.swish(expert_in(pooled_language)))
                for expert_in, expert_out in zip(
                    self.prefix_moe_expert_in,
                    self.prefix_moe_expert_out,
                    strict=True,
                )
            ],
            axis=1,
        )
        scale_shift = jnp.einsum(
            'be,bed->bd', gates.astype(expert_scale_shift.dtype), expert_scale_shift
        )
        (
            image_scale,
            image_shift,
            language_scale,
            language_shift,
        ) = jnp.split(scale_shift, 4, axis=-1)
        adapted_image = (
            prefix_tokens[:, :image_count]
            * (1.0 + jnp.tanh(image_scale)[:, None, :])
            + jnp.tanh(image_shift)[:, None, :]
        )
        if image_count < prefix_tokens.shape[1]:
            adapted_language = (
                prefix_tokens[:, image_count:]
                * (1.0 + jnp.tanh(language_scale)[:, None, :])
                + jnp.tanh(language_shift)[:, None, :]
            )
            adapted = jnp.concatenate(
                [adapted_image, adapted_language], axis=1
            )
        else:
            adapted = adapted_image
        adapted = jnp.where(prefix_mask[..., None], adapted, prefix_tokens)

        # Switch-transformer auxiliary load balance: gradients flow through
        # probability mass while the observed top-k assignment is a target.
        importance = jnp.mean(probabilities, axis=0)
        load = jax.lax.stop_gradient(
            jnp.mean((gates > 0).astype(jnp.float32), axis=0)
            / self.prefix_moe_top_k
        )
        balance_loss = self.prefix_moe_expert_count * jnp.sum(importance * load)
        return adapted, balance_loss, gates

    def apply_layerwise_kv_moe(
        self,
        kv_cache,
        prefix_tokens,
        prefix_mask,
        observation,
    ):
        """Apply sparse, layer-specific key/value FiLM to valid prefix memory."""
        if observation.tokenized_prompt is not None:
            language_count = observation.tokenized_prompt.shape[1]
            language_tokens = prefix_tokens[:, -language_count:]
            language_mask = observation.tokenized_prompt_mask
        else:
            language_tokens = prefix_tokens
            language_mask = prefix_mask
        language_weight = language_mask.astype(prefix_tokens.dtype)
        pooled_language = jnp.sum(
            language_tokens * language_weight[..., None], axis=1
        ) / jnp.maximum(jnp.sum(language_weight, axis=1, keepdims=True), 1.0)
        router_hidden = nnx.swish(
            self.kv_moe_router_in(pooled_language)
            + self.kv_moe_state_in(observation.state)
        )
        router_logits = self.kv_moe_router_out(router_hidden).astype(jnp.float32)
        probabilities = jax.nn.softmax(
            router_logits / self.kv_moe_temperature, axis=-1
        )
        top_values, top_indices = jax.lax.top_k(
            probabilities, self.kv_moe_top_k
        )
        top_values = top_values / jnp.maximum(
            jnp.sum(top_values, axis=-1, keepdims=True), 1e-8
        )
        gates = jnp.zeros_like(probabilities)
        batch_indices = jnp.arange(probabilities.shape[0])[:, None]
        gates = gates.at[batch_indices, top_indices].set(top_values)

        expert_transforms = jnp.stack(
            [
                expert_out(nnx.swish(expert_in(pooled_language)))
                for expert_in, expert_out in zip(
                    self.kv_moe_expert_in,
                    self.kv_moe_expert_out,
                    strict=True,
                )
            ],
            axis=1,
        )
        transforms = jnp.einsum(
            'be,bed->bd', gates.astype(expert_transforms.dtype), expert_transforms
        ).reshape(
            prefix_tokens.shape[0],
            self.kv_moe_layer_count,
            4,
            self.kv_moe_head_count,
            self.kv_moe_head_dim,
        )
        transforms = jnp.transpose(transforms, (1, 0, 2, 3, 4))
        cache_k, cache_v = kv_cache
        key_scale = jnp.tanh(transforms[:, :, 0]).astype(cache_k.dtype)
        key_shift = jnp.tanh(transforms[:, :, 1]).astype(cache_k.dtype)
        value_scale = jnp.tanh(transforms[:, :, 2]).astype(cache_v.dtype)
        value_shift = jnp.tanh(transforms[:, :, 3]).astype(cache_v.dtype)
        key_scale = key_scale[:, :, None, :, :]
        key_shift = key_shift[:, :, None, :, :]
        value_scale = value_scale[:, :, None, :, :]
        value_shift = value_shift[:, :, None, :, :]
        if cache_k.shape[0] != self.kv_moe_layer_count:
            raise ValueError(
                'layerwise KV cache depth does not match the configured VLM'
            )
        valid = prefix_mask[None, :, :, None, None]
        adapted_k = jnp.where(
            valid, cache_k * (1.0 + key_scale) + key_shift, cache_k
        )
        adapted_v = jnp.where(
            valid, cache_v * (1.0 + value_scale) + value_shift, cache_v
        )

        importance = jnp.mean(probabilities, axis=0)
        load = jax.lax.stop_gradient(
            jnp.mean((gates > 0).astype(jnp.float32), axis=0)
            / self.kv_moe_top_k
        )
        balance_loss = self.kv_moe_expert_count * jnp.sum(importance * load)
        return (adapted_k, adapted_v), balance_loss, gates

    def compute_persistent_action_prior_condition(
        self, memory_context, ordered_program, frontier, slot_valid_mask=None
    ):
        """Encode the active and remaining causal program for prior queries."""
        if memory_context is None or memory_context.ndim != 3:
            raise ValueError(
                'persistent action-prior conditioning requires batched memory'
            )
        if ordered_program is None or frontier is None:
            raise ValueError(
                'persistent action-prior conditioning requires program and frontier'
            )
        if (
            ordered_program.ndim != 3
            or ordered_program.shape != memory_context.shape
            or frontier.shape != memory_context.shape[:2]
        ):
            raise ValueError('persistent action-prior program/frontier shape drifted')
        phase_summary = self._persistent_phase_context(
            memory_context, ordered_program, frontier, slot_valid_mask
        )
        return self.persistent_action_prior(phase_summary)

    def _action_prior_query_tokens(
        self,
        batch_size,
        state=None,
        memory_context=None,
        ordered_program=None,
        frontier=None,
        slot_valid_mask=None,
    ):
        queries = self.action_prior_queries(jnp.arange(self.action_prior_horizon))
        queries = jnp.broadcast_to(
            queries[None, :, :],
            (batch_size, *queries.shape),
        )
        if hasattr(self, 'action_prior_state'):
            if state is None:
                raise ValueError(
                    'state is required for state-conditioned action priors'
                )
            queries = queries + nnx.swish(self.action_prior_state(state))[:, None, :]
        if hasattr(self, 'persistent_action_prior'):
            queries = queries + self.compute_persistent_action_prior_condition(
                memory_context, ordered_program, frontier, slot_valid_mask
            )[:, None, :]
        return queries

    def compute_action_prior_contexts(
        self,
        prefix_tokens,
        prefix_mask,
        state=None,
        memory_context=None,
        ordered_program=None,
        frontier=None,
        slot_valid_mask=None,
    ):
        """Extract a compact implicit action prior from final prefix states."""
        keys = self.action_prior_key(prefix_tokens)
        values = self.action_prior_value(prefix_tokens)
        queries = self._action_prior_query_tokens(
            prefix_tokens.shape[0],
            state,
            memory_context,
            ordered_program,
            frontier,
            slot_valid_mask,
        )
        scores = jnp.einsum('bph,bsh->bps', queries, keys)
        scores = scores / jnp.sqrt(keys.shape[-1])
        scores = jnp.where(prefix_mask[:, None, :], scores, -1.0e30)
        contexts = jnp.einsum('bps,bsh->bph', jax.nn.softmax(scores, axis=-1), values)
        return nnx.swish(contexts)

    def compute_specialist_module_gates(self, contexts, state):
        """Return unit-mean gates for context, velocity, and language experts."""
        hidden = self.specialist_module_router_context_in(
            jnp.mean(contexts, axis=1)
        )
        hidden = hidden + self.specialist_module_router_state_in(state)
        logits = self.specialist_module_router_score(nnx.swish(hidden))
        weights = jax.nn.softmax(
            logits.astype(jnp.float32)
            / self.specialist_module_router_temperature,
            axis=-1,
        )
        return weights.astype(contexts.dtype) * 3.0

    def compute_specialist_module_router_balance_loss(self, gates):
        """Keep batch-global expert use balanced without blurring each sample."""
        probabilities = gates.astype(jnp.float32) / 3.0
        mean_probabilities = jnp.mean(probabilities, axis=0)
        uniform = jnp.asarray(1.0 / 3.0, dtype=mean_probabilities.dtype)
        return 3.0 * jnp.sum(jnp.square(mean_probabilities - uniform))

    def compute_multilayer_action_prior_contexts(
        self,
        contexts,
        kv_cache,
        prefix_mask,
        state=None,
        memory_context=None,
        ordered_program=None,
        frontier=None,
        slot_valid_mask=None,
        *,
        return_layer_guidance=False,
    ):
        """Add learned-query IAR features from selected VLM KV layers."""
        cache_k, cache_v = kv_cache
        layer_indexes = jnp.asarray(self.action_prior_implicit_layers)
        cache_k = jnp.take(cache_k, layer_indexes, axis=0)
        cache_v = jnp.take(cache_v, layer_indexes, axis=0)
        cache_k = cache_k.reshape(*cache_k.shape[:3], -1)
        cache_v = cache_v.reshape(*cache_v.shape[:3], -1)

        stride = self.action_prior_implicit_pool_stride
        token_count = cache_k.shape[2]
        padding = (-token_count) % stride
        if padding:
            cache_k = jnp.pad(cache_k, ((0, 0), (0, 0), (0, padding), (0, 0)))
            cache_v = jnp.pad(cache_v, ((0, 0), (0, 0), (0, padding), (0, 0)))
            prefix_mask = jnp.pad(prefix_mask, ((0, 0), (0, padding)))
        pooled_tokens = cache_k.shape[2] // stride
        cache_k = cache_k.reshape(
            cache_k.shape[0],
            cache_k.shape[1],
            pooled_tokens,
            stride,
            cache_k.shape[-1],
        )
        cache_v = cache_v.reshape(
            cache_v.shape[0],
            cache_v.shape[1],
            pooled_tokens,
            stride,
            cache_v.shape[-1],
        )
        pooled_mask = prefix_mask.reshape(prefix_mask.shape[0], pooled_tokens, stride)
        pool_weights = pooled_mask[None, :, :, :, None]
        denominator = jnp.maximum(jnp.sum(pool_weights, axis=3), 1)
        cache_k = jnp.sum(cache_k * pool_weights, axis=3) / denominator
        cache_v = jnp.sum(cache_v * pool_weights, axis=3) / denominator
        pooled_mask = jnp.any(pooled_mask, axis=-1)

        keys = self.action_prior_implicit_key(cache_k)
        values = self.action_prior_implicit_value(cache_v)
        queries = self._action_prior_query_tokens(
            contexts.shape[0],
            state,
            memory_context,
            ordered_program,
            frontier,
            slot_valid_mask,
        )
        scores = jnp.einsum('bqh,lbsh->lbqs', queries, keys)
        scores = scores / jnp.sqrt(keys.shape[-1])
        scores = jnp.where(pooled_mask[None, :, None, :], scores, -1.0e30)
        layer_contexts = jnp.einsum(
            'lbqs,lbsh->lbqh', jax.nn.softmax(scores, axis=-1), values
        )
        layer_mix = self.action_prior_implicit_layer_mix(
            jnp.arange(self.action_prior_horizon)
        )
        layer_mix = jax.nn.softmax(layer_mix, axis=-1)
        multilayer_contexts = jnp.einsum('ql,lbqh->bqh', layer_mix, layer_contexts)
        enriched_contexts = contexts + self.action_prior_implicit_out(
            nnx.swish(multilayer_contexts)
        )
        if not return_layer_guidance:
            return enriched_contexts
        layer_guidance = None
        if hasattr(self, 'action_prior_implicit_layer_queries'):
            layer_guidance = self.compute_layerwise_implicit_guidance(
                cache_k, cache_v, pooled_mask
            )
        return enriched_contexts, layer_guidance

    def compute_layerwise_implicit_guidance(self, cache_k, cache_v, pooled_mask):
        """Keep one group-projected learned-query IAR token per VLM layer."""
        layer_count = cache_k.shape[0]
        queries = self.action_prior_implicit_layer_queries(
            jnp.arange(layer_count)
        )
        heads = self.action_prior_implicit_num_heads
        layer_tokens = []
        for layer_position in range(layer_count):
            group = layer_position // self.action_prior_implicit_group_size
            query = self.action_prior_implicit_group_query[group](
                queries[layer_position]
            )
            keys = self.action_prior_implicit_group_key[group](
                cache_k[layer_position]
            )
            values = self.action_prior_implicit_group_value[group](
                cache_v[layer_position]
            )
            head_dim = query.shape[-1] // heads
            query = query.reshape(heads, head_dim)
            keys = keys.reshape(*keys.shape[:-1], heads, head_dim)
            values = values.reshape(*values.shape[:-1], heads, head_dim)
            scores = jnp.einsum('hd,bshd->bhs', query, keys)
            scores = scores / jnp.sqrt(float(head_dim))
            scores = jnp.where(pooled_mask[:, None, :], scores, -1.0e30)
            weights = jax.nn.softmax(scores, axis=-1).astype(values.dtype)
            attended = jnp.einsum('bhs,bshd->bhd', weights, values)
            attended = attended.reshape(attended.shape[0], -1)
            layer_tokens.append(
                self.action_prior_implicit_group_out[group](attended)
            )
        layer_tokens = jnp.stack(layer_tokens, axis=1)
        return self.action_prior_implicit_layer_to_action(
            nnx.swish(layer_tokens)
        )

    def _implicit_action_prior_outputs(self, contexts):
        coarse_actions = self.action_prior_action_out(contexts)
        coarse_tokens = self.action_prior_token_out(contexts)
        return coarse_tokens, coarse_actions

    def compute_action_guidance_residual(self, action_tokens, guidance_tokens):
        """Cross-attend noisy action tokens to distinct EAR/IAR guidance."""
        queries = self.action_prior_guidance_action_in(action_tokens)
        guidance = self.action_prior_guidance_prior_in(guidance_tokens)
        fused = self.action_prior_guidance_block(queries, guidance)
        return self.action_prior_guidance_out(_rms_normalize(fused))

    def compute_spatial_relation_tokens(
        self,
        prefix_tokens,
        prefix_mask,
        observation,
        contextual_prefix_tokens=None,
    ):
        """Read explicit camera/row/column patch structure into waypoint tokens."""
        if observation.tokenized_prompt is None:
            raise ValueError('spatial relation reasoning requires language tokens')
        camera_names = tuple(observation.images)
        camera_count = len(camera_names)
        if camera_count > self.spatial_relation_max_cameras:
            raise ValueError(
                'observation camera count exceeds spatial_relation_max_cameras'
            )
        language_tokens = observation.tokenized_prompt.shape[1]
        image_tokens = prefix_tokens.shape[1] - language_tokens
        if image_tokens <= 0 or image_tokens % camera_count:
            raise ValueError(
                'visual prefix cannot be evenly partitioned across cameras'
            )
        patches_per_camera = image_tokens // camera_count
        grid_size = math.isqrt(patches_per_camera)
        if grid_size * grid_size != patches_per_camera:
            raise ValueError('spatial relation reasoning requires a square patch grid')
        if grid_size > self.spatial_relation_max_grid_size:
            raise ValueError(
                'visual patch grid exceeds spatial_relation_max_grid_size'
            )

        batch_size = prefix_tokens.shape[0]
        visual = prefix_tokens[:, :image_tokens].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        visual = self.spatial_relation_image_in(visual)
        if contextual_prefix_tokens is not None:
            contextual_visual = contextual_prefix_tokens[:, :image_tokens].reshape(
                batch_size, camera_count, patches_per_camera, -1
            )
            visual = visual + self.spatial_relation_contextual_in(
                contextual_visual
            )
        row_ids = jnp.repeat(jnp.arange(grid_size), grid_size)
        column_ids = jnp.tile(jnp.arange(grid_size), grid_size)
        geometry = (
            self.spatial_relation_camera_position(jnp.arange(camera_count))[
                :, None, :
            ]
            + self.spatial_relation_row_position(row_ids)[None, :, :]
            + self.spatial_relation_column_position(column_ids)[None, :, :]
        )
        visual = visual + geometry[None, :, :, :]
        visual = visual + self.spatial_relation_token_type(jnp.asarray(0))
        visual = visual.reshape(batch_size, image_tokens, -1)
        camera_mask = jnp.stack(
            [self._physical_image_mask(observation, name) for name in camera_names],
            axis=1,
        )
        visual_mask = jnp.repeat(
            camera_mask[:, :, None], patches_per_camera, axis=2
        ).reshape(batch_size, image_tokens)

        language_source = (
            contextual_prefix_tokens
            if contextual_prefix_tokens is not None
            else prefix_tokens
        )
        language = self.spatial_relation_language_in(
            language_source[:, image_tokens:]
        )
        language = language + self.spatial_relation_token_type(jnp.asarray(1))
        language_mask = prefix_mask[:, image_tokens:]
        context_tokens = jnp.concatenate([visual, language], axis=1)
        context_mask = jnp.concatenate([visual_mask, language_mask], axis=1)

        relation_tokens = self.spatial_relation_queries(
            jnp.arange(self.spatial_relation_query_count)
        )
        relation_tokens = jnp.broadcast_to(
            relation_tokens[None, :, :],
            (batch_size, *relation_tokens.shape),
        )
        state_token = nnx.swish(self.spatial_relation_state_in(observation.state))
        relation_tokens = relation_tokens + state_token[:, None, :]
        for block in self.spatial_relation_blocks:
            relation_tokens = block(
                relation_tokens, context_tokens, context_mask
            )

        waypoint_tokens = self.spatial_relation_waypoint_queries(
            jnp.arange(self.action_prior_horizon)
        )
        waypoint_tokens = jnp.broadcast_to(
            waypoint_tokens[None, :, :],
            (batch_size, *waypoint_tokens.shape),
        )
        waypoint_tokens = waypoint_tokens + state_token[:, None, :]
        waypoint_tokens = self.spatial_relation_waypoint_block(
            waypoint_tokens, relation_tokens
        )
        waypoint_tokens = _rms_normalize(waypoint_tokens)
        coarse_actions = self.spatial_relation_action_out(waypoint_tokens)
        action_tokens = self.spatial_relation_token_out(waypoint_tokens)
        repeat = self.action_horizon // self.action_prior_horizon
        return jnp.repeat(action_tokens, repeat, axis=1), coarse_actions

    def compute_object_affordance_graph_tokens(
        self,
        prefix_tokens,
        prefix_mask,
        observation,
        contextual_prefix_tokens=None,
        *,
        compute_reconstruction=False,
        routing_contexts=None,
    ):
        """Bind competitive object slots and read their relation graph."""
        if observation.tokenized_prompt is None:
            raise ValueError(
                'object affordance graph reasoning requires language tokens'
            )
        camera_names = tuple(observation.images)
        camera_count = len(camera_names)
        if camera_count > self.object_affordance_max_cameras:
            raise ValueError(
                'observation camera count exceeds object_affordance_max_cameras'
            )
        language_tokens = observation.tokenized_prompt.shape[1]
        image_tokens = prefix_tokens.shape[1] - language_tokens
        if image_tokens <= 0 or image_tokens % camera_count:
            raise ValueError(
                'visual prefix cannot be evenly partitioned across cameras'
            )
        patches_per_camera = image_tokens // camera_count
        grid_size = math.isqrt(patches_per_camera)
        if grid_size * grid_size != patches_per_camera:
            raise ValueError(
                'object affordance graph reasoning requires a square patch grid'
            )
        if grid_size > self.object_affordance_max_grid_size:
            raise ValueError(
                'visual patch grid exceeds object_affordance_max_grid_size'
            )

        batch_size = prefix_tokens.shape[0]
        visual = prefix_tokens[:, :image_tokens].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        visual = self.object_affordance_image_in(visual)
        if contextual_prefix_tokens is not None:
            contextual_visual = contextual_prefix_tokens[
                :, :image_tokens
            ].reshape(batch_size, camera_count, patches_per_camera, -1)
            visual = visual + self.object_affordance_contextual_in(
                contextual_visual
            )
        row_ids = jnp.repeat(jnp.arange(grid_size), grid_size)
        column_ids = jnp.tile(jnp.arange(grid_size), grid_size)
        geometry = (
            self.object_affordance_camera_position(jnp.arange(camera_count))[
                :, None, :
            ]
            + _dense_embedding_lookup(
                self.object_affordance_row_position, row_ids
            )[None, :, :]
            + _dense_embedding_lookup(
                self.object_affordance_column_position, column_ids
            )[None, :, :]
        )
        visual = (visual + geometry[None, :, :, :]).reshape(
            batch_size, image_tokens, -1
        )
        camera_mask = jnp.stack(
            [self._physical_image_mask(observation, name) for name in camera_names],
            axis=1,
        )
        visual_mask = jnp.repeat(
            camera_mask[:, :, None], patches_per_camera, axis=2
        ).reshape(batch_size, image_tokens)

        language_source = (
            contextual_prefix_tokens
            if contextual_prefix_tokens is not None
            else prefix_tokens
        )
        language = self.object_affordance_language_in(
            language_source[:, image_tokens:]
        )
        language_mask = prefix_mask[:, image_tokens:]
        language_weight = language_mask.astype(language.dtype)[..., None]
        pooled_language = jnp.sum(language * language_weight, axis=1) / jnp.maximum(
            jnp.sum(language_weight, axis=1), 1.0
        )
        state_token = nnx.swish(
            self.object_affordance_state_in(observation.state)
        )
        slots = self.object_affordance_slot_queries(
            jnp.arange(self.object_affordance_slot_count)
        )
        slots = jnp.broadcast_to(
            slots[None, :, :], (batch_size, *slots.shape)
        )
        slots = slots + pooled_language[:, None, :] + state_token[:, None, :]

        slot_queries = self.object_affordance_slot_query_in(
            _rms_normalize(slots)
        )
        patch_keys = self.object_affordance_patch_key_in(
            _rms_normalize(visual)
        )
        patch_values = self.object_affordance_patch_value_in(
            _rms_normalize(visual)
        )
        assignment_logits = jnp.einsum(
            'bsd,bvd->bsv',
            slot_queries,
            patch_keys,
            preferred_element_type=jnp.float32,
        ) / jnp.sqrt(float(slot_queries.shape[-1]))
        assignment_logits = assignment_logits / self.object_affordance_temperature
        assignment_logits = jnp.where(
            visual_mask[:, None, :], assignment_logits, -1.0e30
        )
        # Competition is across slots for every patch.  This is the structural
        # distinction from independent relation queries, which can all collapse
        # onto the same salient patch.
        competitive_assignments = jax.nn.softmax(
            assignment_logits, axis=1
        ).astype(patch_values.dtype)
        competitive_assignments = competitive_assignments * visual_mask[
            :, None, :
        ].astype(patch_values.dtype)
        slot_weights = competitive_assignments / jnp.maximum(
            jnp.sum(competitive_assignments, axis=-1, keepdims=True),
            jnp.asarray(1.0e-6, dtype=patch_values.dtype),
        )
        slot_updates = jnp.einsum(
            'bsv,bvd->bsd', slot_weights, patch_values
        )
        slots = slots + self.object_affordance_slot_update(slot_updates)
        for block in self.object_affordance_graph_blocks:
            slots = block(slots, language, language_mask)

        waypoint_tokens = self.object_affordance_waypoint_queries(
            jnp.arange(self.action_prior_horizon)
        )
        waypoint_tokens = jnp.broadcast_to(
            waypoint_tokens[None, :, :],
            (batch_size, *waypoint_tokens.shape),
        )
        waypoint_tokens = (
            waypoint_tokens
            + pooled_language[:, None, :]
            + state_token[:, None, :]
        )
        waypoint_tokens = self.object_affordance_waypoint_block(
            waypoint_tokens, slots
        )
        waypoint_tokens = _rms_normalize(waypoint_tokens)
        coarse_actions = self.object_affordance_action_out(waypoint_tokens)
        action_tokens = self.object_affordance_token_out(waypoint_tokens)
        normalized_slots = _rms_normalize(slots)
        reconstruction_loss = None
        if compute_reconstruction:
            # Decode every visible patch from the slot that competitively owns
            # it.  The projected patch target is detached: gradients must
            # improve the assignment/slot graph rather than move the target to
            # match a collapsed representation.  This branch is training-only.
            patch_reconstruction = jnp.einsum(
                'bsv,bsd->bvd',
                competitive_assignments,
                normalized_slots,
            )
            patch_target = jax.lax.stop_gradient(_rms_normalize(patch_values))
            patch_error = jnp.mean(
                jnp.square(
                    patch_reconstruction.astype(jnp.float32)
                    - patch_target.astype(jnp.float32)
                ),
                axis=-1,
            )
            patch_mask = visual_mask.astype(patch_error.dtype)
            reconstruction_loss = jnp.sum(
                patch_error * patch_mask, axis=-1
            ) / jnp.maximum(jnp.sum(patch_mask, axis=-1), 1.0)
        repeat = self.action_horizon // self.action_prior_horizon
        action_tokens = jnp.repeat(action_tokens, repeat, axis=1)
        if hasattr(self, 'specialist_module_router_score'):
            if routing_contexts is None:
                raise ValueError(
                    'specialist-routed object affordance requires action contexts'
                )
            specialist_gates = self.compute_specialist_module_gates(
                routing_contexts, observation.state
            )
            # Gate zero is the context/grounding capability family.  The
            # router is zero initialized and therefore starts at multiplier
            # one, preserving the inherited object residual exactly.
            action_tokens = action_tokens * specialist_gates[:, :1, None]
        return (
            action_tokens,
            coarse_actions,
            competitive_assignments,
            normalized_slots,
            reconstruction_loss,
        )

    def compute_contact_phase_tokens(self, contexts, state, *, train):
        """Predict a discrete manipulation phase for every continuous action step."""
        tokens = self.contact_phase_queries(jnp.arange(self.action_horizon))
        tokens = jnp.broadcast_to(
            tokens[None, :, :], (contexts.shape[0], *tokens.shape)
        )
        tokens = tokens + nnx.swish(self.contact_phase_state_in(state))[:, None, :]
        phase_context = self.contact_phase_context_in(contexts)
        for block in self.contact_phase_blocks:
            tokens = block(tokens, phase_context)
        logits = self.contact_phase_logits(_rms_normalize(tokens))
        probabilities = jax.nn.softmax(
            logits.astype(jnp.float32) / self.contact_phase_temperature,
            axis=-1,
        )
        hard_phases = jax.nn.one_hot(
            jnp.argmax(probabilities, axis=-1), probabilities.shape[-1]
        )
        phase_weights = (
            probabilities + jax.lax.stop_gradient(hard_phases - probabilities)
            if train
            else hard_phases
        )
        phase_embeddings = self.contact_phase_embeddings(jnp.arange(4))
        selected_phase = jnp.einsum(
            'btc,cd->btd',
            phase_weights.astype(tokens.dtype),
            phase_embeddings,
        )
        fused = jnp.concatenate([tokens, selected_phase], axis=-1)
        fused = nnx.swish(self.contact_phase_fuse(fused))
        residual = self.contact_phase_token_out(fused)
        if hasattr(self, 'specialist_module_router_score'):
            specialist_gates = self.compute_specialist_module_gates(
                contexts, state
            )
            # Gate one already controls velocity refinement; pairing contact
            # phase with it forms a dynamics/control capability family.
            residual = residual * specialist_gates[:, 1:2, None]
        return residual, logits

    def compute_structured_rationale_tokens(
        self, prefix_tokens, prefix_mask, state, *, train
    ):
        """Predict seven verbalizable chunk-motion classes from VLM states."""
        context = self.action_prior_rationale_context_in(prefix_tokens)
        axis_tokens = self.action_prior_rationale_axis_queries(jnp.arange(7))
        axis_tokens = jnp.broadcast_to(
            axis_tokens[None, :, :],
            (prefix_tokens.shape[0], *axis_tokens.shape),
        )
        state_token = nnx.swish(self.action_prior_rationale_state_in(state))
        axis_tokens = axis_tokens + state_token[:, None, :]
        for block in self.action_prior_rationale_blocks:
            axis_tokens = block(axis_tokens, context, prefix_mask)
        logits = self.action_prior_rationale_logits(_rms_normalize(axis_tokens))
        probabilities = jax.nn.softmax(
            logits.astype(jnp.float32)
            / self.action_prior_rationale_temperature,
            axis=-1,
        )
        hard_classes = jax.nn.one_hot(
            jnp.argmax(probabilities, axis=-1), 3, dtype=probabilities.dtype
        )
        class_weights = (
            probabilities
            + jax.lax.stop_gradient(hard_classes - probabilities)
            if train
            else hard_classes
        )
        class_embeddings = self.action_prior_rationale_class_embeddings(
            jnp.arange(3)
        )
        selected_classes = jnp.einsum(
            'bac,ch->bah', class_weights.astype(axis_tokens.dtype), class_embeddings
        )
        semantic_axes = axis_tokens + selected_classes

        waypoint_tokens = self.action_prior_rationale_waypoint_queries(
            jnp.arange(self.action_horizon)
        )
        waypoint_tokens = jnp.broadcast_to(
            waypoint_tokens[None, :, :],
            (prefix_tokens.shape[0], *waypoint_tokens.shape),
        )
        waypoint_tokens = waypoint_tokens + state_token[:, None, :]
        waypoint_tokens = self.action_prior_rationale_fusion(
            waypoint_tokens, semantic_axes
        )
        return (
            self.action_prior_rationale_token_out(
                _rms_normalize(waypoint_tokens)
            ),
            logits,
        )

    def structured_rationale_targets(self, actions):
        """Convert normalized action chunks to negative/steady/positive labels."""
        q01 = jnp.asarray(
            self.action_prior_rationale_action_q01, dtype=actions.dtype
        )
        q99 = jnp.asarray(
            self.action_prior_rationale_action_q99, dtype=actions.dtype
        )
        physical = actions[..., :7]
        physical = (physical + 1.0) * 0.5 * (q99 - q01) + q01
        mean_motion = jnp.mean(physical, axis=-2)
        # A gripper chunk is semantically described by its final command;
        # translation/rotation use mean motion to suppress oscillatory noise.
        mean_motion = mean_motion.at[..., 6].set(physical[..., -1, 6])
        neutral_eps = jnp.asarray(
            self.action_prior_rationale_neutral_eps, dtype=actions.dtype
        )
        targets = jnp.where(
            mean_motion < -neutral_eps,
            0,
            jnp.where(mean_motion > neutral_eps, 2, 1),
        )
        return targets.astype(jnp.int32)

    def contact_phase_targets(self, actions, state):
        """Map gripper commands and current aperture to semantic phases."""
        if self.contact_phase_gripper_indices:
            gripper = actions[..., self.contact_phase_gripper_indices]
            state_gripper = state[..., self.contact_phase_state_scalar_indices]
            if self.contact_phase_open_when_positive:
                current_open = gripper > self.contact_phase_state_open_threshold
                previous_open_first = (
                    state_gripper > self.contact_phase_state_open_threshold
                )
            else:
                current_open = gripper < self.contact_phase_state_open_threshold
                previous_open_first = (
                    state_gripper < self.contact_phase_state_open_threshold
                )
            previous_open = jnp.concatenate(
                [previous_open_first[:, None, :], current_open[:, :-1, :]],
                axis=1,
            )
            close_transition = jnp.any(
                (~current_open) & previous_open, axis=-1
            )
            release_transition = jnp.any(
                current_open & (~previous_open), axis=-1
            )
            any_closed = jnp.any(~current_open, axis=-1)
            # 0=approach/open, 1=close, 2=carry/closed, 3=release.
            # A simultaneous close/release is assigned to close because a new
            # grasp is the safety-critical contact transition.
            targets = jnp.where(any_closed, 2, 0)
            targets = jnp.where(release_transition, 3, targets)
            targets = jnp.where(close_transition, 1, targets)
            return targets.astype(jnp.int32)

        gripper = actions[..., self.contact_phase_gripper_index]
        positive_finger, negative_finger = self.contact_phase_state_gripper_indices
        normalized_aperture = (
            state[..., positive_finger] - state[..., negative_finger]
        )
        # LIBERO uses -1=open and +1=close.  The two normalized finger
        # positions move in opposite directions, so their difference provides
        # the state immediately before the first predicted action.
        previous_first = jnp.where(
            normalized_aperture > self.contact_phase_state_open_threshold,
            -jnp.ones_like(gripper[:, 0]),
            jnp.ones_like(gripper[:, 0]),
        )
        previous = jnp.concatenate(
            [previous_first[:, None], gripper[:, :-1]], axis=1
        )
        targets = jnp.where(gripper < 0, 0, 2)
        targets = jnp.where((gripper >= 0) & (previous < 0), 1, targets)
        targets = jnp.where((gripper < 0) & (previous >= 0), 3, targets)
        return targets.astype(jnp.int32)

    def compute_contact_affordance_predictive_tokens(
        self,
        object_slots,
        contact_phase_tokens,
        contact_phase_logits,
        contexts,
        state,
        persistent_memory,
        persistent_program,
        persistent_frontier,
        clause_plan_attention=None,
        future_object_tokens=None,
        factorized_relation_state=None,
        grounded_relation_phase_state=None,
        bound_roles=None,
        verification_transition_probabilities=None,
        persistent_progress=None,
        verification_state=None,
    ):
        """Fuse objects, contact phase, memory, and the active physical program."""
        if persistent_memory is None:
            raise ValueError(
                'contact-affordance fusion requires private persistent memory'
            )
        if persistent_program is None or persistent_frontier is None:
            raise ValueError(
                'contact-affordance fusion requires program and frontier'
            )
        if (
            persistent_program.ndim != 3
            or persistent_frontier.ndim != 2
            or persistent_program.shape[:2] != persistent_frontier.shape
        ):
            raise ValueError(
                'contact-affordance program/frontier shapes are incompatible'
            )
        object_context = self.contact_affordance_object_in(object_slots)
        task_context = self.contact_affordance_context_in(contexts)
        memory_context = self.contact_affordance_memory_in(
            _rms_normalize(persistent_memory)
        ) + self.contact_affordance_memory_position(
            jnp.arange(persistent_memory.shape[1])
        )[None, :, :]
        program_tokens = self.contact_affordance_program_in(
            _rms_normalize(persistent_program)
        ) + self.contact_affordance_program_position(
            jnp.arange(persistent_program.shape[1])
        )[None, :, :]
        current_program = jnp.einsum(
            'bs,bsd->bd',
            persistent_frontier.astype(program_tokens.dtype),
            program_tokens,
        )
        next_frontier = jnp.zeros_like(persistent_frontier)
        next_frontier = next_frontier.at[:, 1:].set(
            persistent_frontier[:, :-1]
        )
        next_frontier = next_frontier.at[:, -1].add(
            persistent_frontier[:, -1]
        )
        next_program = jnp.einsum(
            'bs,bsd->bd',
            next_frontier.astype(program_tokens.dtype),
            program_tokens,
        )
        current_weight = jnp.full(
            (persistent_program.shape[0],), 0.75, dtype=program_tokens.dtype
        )
        next_weight = jnp.full(
            (persistent_program.shape[0],), 0.25, dtype=program_tokens.dtype
        )
        transition_context = None
        if self.contact_affordance_transition_verification:
            if (
                verification_transition_probabilities is None
                or verification_transition_probabilities.shape
                != persistent_frontier.shape
                or persistent_progress is None
                or persistent_progress.shape != (persistent_program.shape[0],)
                or verification_state is None
                or verification_state.shape
                != (persistent_program.shape[0], persistent_program.shape[-1])
            ):
                raise ValueError(
                    'transition-verified contact requires PSM transition '
                    'probabilities, progress, and verification state'
                )
            transition_probabilities = (
                verification_transition_probabilities.astype(program_tokens.dtype)
            )
            current_weight = jnp.einsum(
                'bs,bs->b', persistent_frontier, transition_probabilities
            )
            next_weight = jnp.einsum(
                'bs,bs->b', next_frontier, transition_probabilities
            )
            transition_mass = jnp.maximum(
                current_weight + next_weight,
                jnp.asarray(1.0e-8, dtype=current_weight.dtype),
            )
            current_weight = current_weight / transition_mass
            next_weight = next_weight / transition_mass
        phase_program_context = (
            current_weight[:, None] * current_program
            + next_weight[:, None] * next_program
        )
        if self.contact_affordance_transition_verification:
            transition_context = _rms_normalize(verification_state) + (
                persistent_progress.astype(program_tokens.dtype)[:, None]
                * _rms_normalize(phase_program_context)
            )
        clause_alignment_context = None
        if hasattr(self, 'contact_affordance_clause_attention_in'):
            expected_attention_shape = (
                persistent_program.shape[0],
                persistent_program.shape[1],
                persistent_program.shape[1],
            )
            if (
                clause_plan_attention is None
                or clause_plan_attention.shape != expected_attention_shape
            ):
                raise ValueError(
                    'verified contact requires slot-to-clause attention'
                )
            current_alignment = jnp.einsum(
                'bs,bsc->bc',
                persistent_frontier.astype(clause_plan_attention.dtype),
                clause_plan_attention,
            )
            next_alignment = jnp.einsum(
                'bs,bsc->bc',
                next_frontier.astype(clause_plan_attention.dtype),
                clause_plan_attention,
            )
            clause_alignment_context = self.contact_affordance_clause_attention_in(
                current_weight[:, None] * current_alignment
                + next_weight[:, None] * next_alignment
            )
        future_context = None
        if hasattr(self, 'contact_affordance_future_in'):
            if (
                future_object_tokens is None
                or future_object_tokens.ndim != 3
                or future_object_tokens.shape[0] != object_slots.shape[0]
            ):
                raise ValueError(
                    'verified contact requires predicted future object tokens'
                )
            future_context = self.contact_affordance_future_in(
                _rms_normalize(future_object_tokens)
            )
        relation_context = None
        relation_alignment_logits = None
        if hasattr(self, 'contact_affordance_relation_state_in'):
            batch_size = object_slots.shape[0]
            hidden_dim = persistent_memory.shape[-1]
            if (
                factorized_relation_state is None
                or factorized_relation_state.shape != (batch_size, hidden_dim)
                or grounded_relation_phase_state is None
                or grounded_relation_phase_state.shape
                != persistent_program.shape
                or bound_roles is None
                or bound_roles.shape != (batch_size, 2, hidden_dim)
            ):
                raise ValueError(
                    'verified contact requires factorized/grounded relation '
                    'states and two bound roles'
                )
            active_grounded_relation = jnp.einsum(
                'bs,bsd->bd',
                (
                    current_weight[:, None] * persistent_frontier
                    + next_weight[:, None] * next_frontier
                ).astype(grounded_relation_phase_state.dtype),
                grounded_relation_phase_state,
            )
            factorized_relation_token = self.contact_affordance_relation_state_in(
                _rms_normalize(factorized_relation_state)
            )
            grounded_relation_token = (
                self.contact_affordance_grounded_relation_in(
                    _rms_normalize(active_grounded_relation)
                )
            )
            bound_role_tokens = self.contact_affordance_bound_role_in(
                _rms_normalize(bound_roles)
            )
            grounded_role_relation_token = (
                grounded_relation_token
                + _ordered_role_pair_summary(bound_role_tokens)
            )
            relation_alignment_logits = jnp.einsum(
                'bd,cd->bc',
                _l2_normalize(factorized_relation_token).astype(jnp.float32),
                _l2_normalize(grounded_role_relation_token).astype(jnp.float32),
            ) / self.contact_affordance_relation_contrastive_temperature
            relation_context = jnp.concatenate(
                [
                    factorized_relation_token[:, None, :],
                    grounded_relation_token[:, None, :],
                    bound_role_tokens,
                ],
                axis=1,
            )
            relation_context = (
                relation_context
                + self.contact_affordance_relation_position(
                    jnp.arange(4)
                )[None, :, :]
            )
        fusion_context = jnp.concatenate(
            [
                object_context,
                task_context,
                memory_context,
                program_tokens,
                phase_program_context[:, None, :],
                *(
                    [clause_alignment_context[:, None, :]]
                    if clause_alignment_context is not None
                    else []
                ),
                *([future_context] if future_context is not None else []),
                *([relation_context] if relation_context is not None else []),
                *(
                    [transition_context[:, None, :]]
                    if transition_context is not None
                    else []
                ),
            ],
            axis=1,
        )
        phase_evidence = jax.nn.softmax(
            contact_phase_logits.astype(jnp.float32)
            / self.contact_phase_temperature,
            axis=-1,
        ).astype(contact_phase_tokens.dtype)
        tokens = (
            self.contact_affordance_phase_in(contact_phase_tokens)
            + self.contact_affordance_phase_logits_in(phase_evidence)
        )
        tokens = (
            tokens
            + nnx.swish(self.contact_affordance_state_in(state))[:, None, :]
            + nnx.swish(phase_program_context)[:, None, :]
            + (
                nnx.swish(clause_alignment_context)[:, None, :]
                if clause_alignment_context is not None
                else 0.0
            )
            + (
                nnx.swish(jnp.mean(relation_context, axis=1))[:, None, :]
                if relation_context is not None
                else 0.0
            )
            + (
                nnx.swish(transition_context)[:, None, :]
                if transition_context is not None
                else 0.0
            )
            + self.contact_affordance_action_position(
                jnp.arange(self.action_horizon)
            )[None, :, :]
        )
        for block in self.contact_affordance_blocks:
            tokens = block(tokens, fusion_context)
        normalized = _rms_normalize(tokens)
        risk_logits = self.contact_affordance_risk_logit(normalized)[..., 0]
        risk = jax.nn.sigmoid(risk_logits).astype(tokens.dtype)
        risk_conditioned = tokens + self.contact_affordance_risk_in(
            risk[..., None]
        )
        return (
            self.contact_affordance_token_out(
                _rms_normalize(risk_conditioned)
            ),
            risk_logits,
            relation_alignment_logits,
        )

    def contact_affordance_risk_targets(
        self, competitive_assignments, actions, state,
        clause_plan_attention=None, persistent_frontier=None,
    ):
        """Supervise caution from object ambiguity and contact transitions."""
        assignments = competitive_assignments.astype(jnp.float32)
        patch_mass = jnp.sum(assignments, axis=1)
        entropy = -jnp.sum(
            jnp.where(
                assignments > 0,
                assignments * jnp.log(jnp.maximum(assignments, 1.0e-8)),
                0.0,
            ),
            axis=1,
        )
        entropy = entropy / jnp.log(float(assignments.shape[1]))
        ambiguity = jnp.sum(entropy * patch_mass, axis=-1) / jnp.maximum(
            jnp.sum(patch_mass, axis=-1), 1.0
        )
        if clause_plan_attention is not None or persistent_frontier is not None:
            if clause_plan_attention is None or persistent_frontier is None:
                raise ValueError(
                    'clause risk supervision requires attention and frontier'
                )
            active_alignment = jnp.einsum(
                'bs,bsc->bc',
                persistent_frontier.astype(clause_plan_attention.dtype),
                clause_plan_attention,
            ).astype(jnp.float32)
            clause_entropy = -jnp.sum(
                jnp.where(
                    active_alignment > 0,
                    active_alignment
                    * jnp.log(jnp.maximum(active_alignment, 1.0e-8)),
                    0.0,
                ),
                axis=-1,
            ) / jnp.log(float(active_alignment.shape[-1]))
            ambiguity = jnp.maximum(ambiguity, clause_entropy)
        phases = self.contact_phase_targets(actions, state)
        transition = (phases == 1) | (phases == 3)
        targets = jnp.maximum(
            ambiguity[:, None], transition.astype(jnp.float32)
        )
        return jax.lax.stop_gradient(targets)

    @staticmethod
    def exact_prompt_positive_mask(tokenized_prompt, tokenized_prompt_mask):
        """Treat repeated ordinary prompts as contrastive co-positives."""
        if tokenized_prompt is None or tokenized_prompt_mask is None:
            raise ValueError(
                'relation contrastive supervision requires prompt tokens and mask'
            )
        if (
            tokenized_prompt.ndim != 2
            or tokenized_prompt_mask.shape != tokenized_prompt.shape
        ):
            raise ValueError(
                'relation contrastive prompt tokens/mask must both be [batch, token]'
            )
        same_tokens = jnp.all(
            tokenized_prompt[:, None, :] == tokenized_prompt[None, :, :],
            axis=-1,
        )
        same_masks = jnp.all(
            tokenized_prompt_mask[:, None, :]
            == tokenized_prompt_mask[None, :, :],
            axis=-1,
        )
        return jax.lax.stop_gradient(same_tokens & same_masks)

    @staticmethod
    def valid_factorized_relation_alignment_loss(
        relation_logits,
        tokenized_prompt,
        tokenized_prompt_mask,
        factorized_valid,
    ):
        """Symmetric multi-positive loss over factorized-valid examples only."""

        batch_size = relation_logits.shape[0]
        if relation_logits.shape != (batch_size, batch_size):
            raise ValueError('relation alignment logits must be square')
        if factorized_valid is None:
            factorized_valid = jnp.zeros((batch_size,), dtype=jnp.bool_)
        factorized_valid = factorized_valid.astype(jnp.bool_)
        if factorized_valid.shape != (batch_size,):
            raise ValueError('factorized relation validity must be [batch]')
        relation_positive = Pi0.exact_prompt_positive_mask(
            tokenized_prompt, tokenized_prompt_mask
        ) & (factorized_valid[:, None] & factorized_valid[None, :])
        relation_positive = relation_positive.astype(jnp.float32)
        language_targets = relation_positive / jnp.maximum(
            jnp.sum(relation_positive, axis=-1, keepdims=True), 1.0
        )
        grounded_targets = relation_positive / jnp.maximum(
            jnp.sum(relation_positive, axis=0, keepdims=True), 1.0
        )
        # Invalid examples must be absent from both softmax denominators;
        # zeroing only their final row loss would still make them negatives.
        masked_language_logits = jnp.where(
            factorized_valid[None, :], relation_logits, -1.0e30
        )
        masked_grounded_logits = jnp.where(
            factorized_valid[:, None], relation_logits, -1.0e30
        )
        language_to_grounded = -jnp.sum(
            language_targets
            * jax.nn.log_softmax(masked_language_logits, axis=-1),
            axis=-1,
        )
        grounded_to_language = -jnp.sum(
            grounded_targets
            * jax.nn.log_softmax(masked_grounded_logits, axis=0),
            axis=0,
        )
        return jnp.where(
            factorized_valid,
            0.5 * (language_to_grounded + grounded_to_language),
            jnp.zeros_like(language_to_grounded),
        )

    def compute_action_prior(
        self,
        prefix_tokens,
        prefix_mask,
        state=None,
        memory_context=None,
        ordered_program=None,
        frontier=None,
        slot_valid_mask=None,
    ):
        """Predict coarse action references from multimodal prefix tokens."""
        contexts = self.compute_action_prior_contexts(
            prefix_tokens,
            prefix_mask,
            state,
            memory_context,
            ordered_program,
            frontier,
            slot_valid_mask,
        )
        coarse_tokens, coarse_actions = self._implicit_action_prior_outputs(contexts)
        repeat = self.action_horizon // self.action_prior_horizon
        return jnp.repeat(coarse_tokens, repeat, axis=1), coarse_actions

    def compute_explicit_action_velocity(self, noisy_waypoints, time, contexts):
        """Predict the coarse reference-trajectory flow velocity."""
        tokens = self.action_prior_explicit_action_in(noisy_waypoints)
        waypoint_tokens = self.action_prior_explicit_waypoints(
            jnp.arange(self.action_prior_horizon)
        )
        tokens = tokens + waypoint_tokens[None, :, :]
        time_tokens = posemb_sincos(
            time,
            tokens.shape[-1],
            min_period=4e-3,
            max_period=4.0,
        )
        time_tokens = self.action_prior_explicit_time_in(time_tokens)
        time_tokens = nnx.swish(time_tokens)
        time_tokens = self.action_prior_explicit_time_out(time_tokens)
        tokens = tokens + time_tokens[:, None, :]
        context_tokens = self.action_prior_explicit_context_in(contexts)
        for block in self.action_prior_explicit_blocks:
            tokens = block(tokens, context_tokens)
        return self.action_prior_explicit_velocity_out(_rms_normalize(tokens))

    def compute_detached_explicit_reference(self, initial_waypoints, contexts):
        """Run the deployed EAR solver without coupling its graph to main flow."""
        steps = self.action_prior_explicit_inference_steps
        dt = -1.0 / steps
        reference_waypoints = initial_waypoints
        batch_size = initial_waypoints.shape[0]
        for step in range(steps):
            reference_time = 1.0 + step * dt
            reference_velocity = self.compute_explicit_action_velocity(
                reference_waypoints,
                jnp.full((batch_size,), reference_time),
                contexts,
            )
            reference_waypoints = reference_waypoints + dt * reference_velocity
        # The EAR retains its own flow-matching objective.  Blocking this path
        # prevents every main-flow sample from adding a duplicate EAR backward
        # graph while still exposing the main policy to the exact reference
        # distribution produced by the configured deployment solver.
        return jax.lax.stop_gradient(reference_waypoints)

    def dual_action_prior_token_paths(self, contexts, reference_waypoints):
        """Return the two inherited dual-reasoner token paths separately."""
        implicit_tokens, coarse_actions = self._implicit_action_prior_outputs(contexts)
        explicit_tokens = self.action_prior_explicit_action_in(reference_waypoints)
        explicit_tokens = self.action_prior_explicit_token_out(
            nnx.swish(explicit_tokens)
        )
        repeat = self.action_horizon // self.action_prior_horizon
        implicit_tokens = jnp.repeat(implicit_tokens, repeat, axis=1)
        explicit_tokens = jnp.repeat(explicit_tokens, repeat, axis=1)
        reasoning_tokens = jnp.concatenate(
            [implicit_tokens, explicit_tokens], axis=1
        )
        return implicit_tokens, explicit_tokens, coarse_actions, reasoning_tokens

    def fuse_dual_action_prior_tokens(self, contexts, reference_waypoints):
        (
            implicit_tokens,
            explicit_tokens,
            coarse_actions,
            reasoning_tokens,
        ) = self.dual_action_prior_token_paths(contexts, reference_waypoints)
        additive_tokens = implicit_tokens + explicit_tokens
        return additive_tokens, coarse_actions, reasoning_tokens

    def route_reasoning_pathway_tokens(self, pathway_tokens, contexts, state):
        """Dynamically fuse four action-token paths while preserving their sum."""
        if len(pathway_tokens) != self.action_prior_pathway_count:
            raise ValueError(
                'reasoning pathway router requires exactly '
                f'{self.action_prior_pathway_count} token paths'
            )
        stacked = jnp.stack(pathway_tokens, axis=2)
        path_features = self.action_prior_pathway_token_in(
            _rms_normalize(stacked)
        )
        context_features = self.action_prior_pathway_context_in(
            jnp.mean(contexts, axis=1)
        )[:, None, None, :]
        state_features = self.action_prior_pathway_state_in(state)[
            :, None, None, :
        ]
        path_embeddings = self.action_prior_pathway_embedding(
            jnp.arange(self.action_prior_pathway_count)
        )[None, None, :, :]
        hidden = nnx.swish(
            path_features
            + context_features
            + state_features
            + path_embeddings
        )
        logits = jnp.einsum(
            'btph,h->btp',
            hidden,
            self.action_prior_pathway_score.value,
            preferred_element_type=jnp.float32,
        )
        weights = (
            jax.nn.softmax(
                logits.astype(jnp.float32)
                / self.action_prior_pathway_temperature,
                axis=-1,
            )
            * self.action_prior_pathway_count
        ).astype(stacked.dtype)
        routed = jnp.sum(stacked * weights[..., None], axis=2)
        return routed, weights

    def interact_reasoning_pathway_tokens(self, pathway_tokens, contexts, state):
        """Add a learned cross-path/time residual to the inherited token sum."""
        if len(pathway_tokens) != self.action_prior_pathway_interaction_count:
            raise ValueError(
                'reasoning pathway interaction requires exactly '
                f'{self.action_prior_pathway_interaction_count} token paths'
            )
        stacked = jnp.stack(pathway_tokens, axis=2)
        base = jnp.sum(stacked, axis=2)
        hidden = self.action_prior_pathway_interaction_token_in(
            _rms_normalize(stacked)
        )
        hidden = hidden + self.action_prior_pathway_interaction_state_in(state)[
            :, None, None, :
        ]
        hidden = hidden + self.action_prior_pathway_interaction_path_embedding(
            jnp.arange(self.action_prior_pathway_interaction_count)
        )[None, None, :, :]
        hidden = hidden + self.action_prior_pathway_interaction_time_embedding(
            jnp.arange(self.action_horizon)
        )[None, :, None, :]
        hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
        interaction_context = self.action_prior_pathway_interaction_context_in(
            contexts
        )
        for block in self.action_prior_pathway_interaction_blocks:
            hidden = block(hidden, interaction_context)
        hidden = hidden.reshape(
            stacked.shape[0],
            stacked.shape[1],
            stacked.shape[2],
            hidden.shape[-1],
        )
        residual_paths = self.action_prior_pathway_interaction_out(hidden)
        residual = jnp.mean(residual_paths, axis=2)
        return base + residual, residual_paths

    def compute_action_chunk_verifier(self, actions, contexts, state):
        """Score and correct a complete action chunk against its context."""
        if actions.shape[-2] != self.action_horizon:
            raise ValueError(
                'action chunk verifier requires the configured action horizon'
            )
        tokens = self.action_chunk_verifier_action_in(actions)
        tokens = tokens + self.action_chunk_verifier_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        tokens = tokens + nnx.swish(
            self.action_chunk_verifier_state_in(state)
        )[:, None, :]
        verifier_context = self.action_chunk_verifier_context_in(contexts)
        for block in self.action_chunk_verifier_blocks:
            tokens = block(tokens, verifier_context)
        tokens = _rms_normalize(tokens)
        scores = jnp.mean(
            self.action_chunk_verifier_score(tokens)[..., 0], axis=-1
        )
        residual = self.action_chunk_verifier_token_out(tokens)
        return residual, scores

    def action_chunk_verifier_candidates(self, actions):
        """Build deterministic hard negatives without evaluator-side labels."""
        if actions.shape[-2] != self.action_horizon:
            raise ValueError(
                'action chunk verifier requires the configured action horizon'
            )
        mismatched = jnp.roll(actions, 1, axis=0)
        if actions.shape[0] == 1:
            active_dim = self.active_action_dim
            active = -actions[..., :active_dim]
            mismatched = jnp.concatenate(
                [active, actions[..., active_dim:]], axis=-1
            )
        phase_shifted = jnp.roll(
            actions, self.action_horizon // 2, axis=-2
        )
        reversed_actions = actions[:, ::-1, :]
        return jnp.stack(
            [actions, mismatched, phase_shifted, reversed_actions], axis=1
        )

    def _latent_future_pool_grid(self, tokens):
        """Deterministically pool a square SigLIP grid to the target grid."""
        source_grid = math.isqrt(tokens.shape[-2])
        target_grid = self.latent_future_grid_size
        if source_grid * source_grid != tokens.shape[-2]:
            raise ValueError('latent future reasoning requires square image tokens')
        if source_grid % target_grid:
            raise ValueError(
                'latent future grid size must divide the SigLIP patch grid'
            )
        stride = source_grid // target_grid
        tokens = tokens.reshape(
            *tokens.shape[:-2],
            target_grid,
            stride,
            target_grid,
            stride,
            tokens.shape[-1],
        )
        tokens = jnp.mean(tokens, axis=(-4, -2))
        return tokens.reshape(
            *tokens.shape[:-3], target_grid**2, tokens.shape[-1]
        )

    def latent_future_current_visual_tokens(
        self, prefix_tokens, observation
    ):
        """Extract the two deployed camera grids from the normal prefix."""
        language_tokens = (
            observation.tokenized_prompt.shape[-1]
            if observation.tokenized_prompt is not None
            else 0
        )
        image_token_count = prefix_tokens.shape[1] - language_tokens
        camera_count = len(observation.images)
        if image_token_count <= 0 or image_token_count % camera_count:
            raise ValueError('cannot partition prefix tokens by camera')
        tokens_per_camera = image_token_count // camera_count
        selected = []
        for camera_index, name in enumerate(observation.images):
            if name not in self.latent_future_camera_names:
                continue
            start = camera_index * tokens_per_camera
            selected.append(
                self._latent_future_pool_grid(
                    prefix_tokens[:, start : start + tokens_per_camera]
                )
            )
        if len(selected) != len(self.latent_future_camera_names):
            raise ValueError('latent future reasoning is missing a deployed camera')
        return jnp.stack(selected, axis=1)

    def latent_future_targets(self, observation, current_visual_tokens):
        """Encode training-only future RGB and build a masked feature delta."""
        if observation.future_images is None:
            raise ValueError('latent future training requires future RGB targets')
        future_masks = observation.future_image_masks or {}
        targets = []
        masks = []
        for name in self.latent_future_camera_names:
            if name not in observation.future_images:
                raise ValueError(f'missing future image target {name!r}')
            future_tokens, _ = self.PaliGemma.img(
                observation.future_images[name], train=False
            )
            targets.append(self._latent_future_pool_grid(future_tokens))
            masks.append(
                jnp.asarray(
                    future_masks.get(
                        name,
                        jnp.ones(
                            (future_tokens.shape[0],), dtype=jnp.bool_
                        ),
                    )
                )
            )
        future_visual_tokens = jnp.stack(targets, axis=1)
        target_delta = jax.lax.stop_gradient(
            _rms_normalize(future_visual_tokens)
            - _rms_normalize(current_visual_tokens)
        )
        target_mask = jnp.stack(masks, axis=1)
        target_mask = jnp.repeat(
            target_mask[:, :, None],
            self.latent_future_grid_size**2,
            axis=2,
        )
        return target_delta, target_mask

    def compute_latent_future_tokens(
        self, current_visual_tokens, actions, contexts, state
    ):
        """Predict latent scene change and return a dynamics-aware residual."""
        camera_count = len(self.latent_future_camera_names)
        patch_count = self.latent_future_grid_size**2
        hidden = self.latent_future_current_in(
            _rms_normalize(current_visual_tokens)
        )
        camera_positions = self.latent_future_camera_position(
            jnp.arange(camera_count)
        )[:, None, :]
        patch_positions = self.latent_future_patch_position(
            jnp.arange(patch_count)
        )[None, :, :]
        hidden = hidden + camera_positions[None] + patch_positions[None]
        hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])

        action_context = self.latent_future_action_in(actions)
        action_context = action_context + self.latent_future_action_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        prior_context = self.latent_future_context_in(contexts)
        state_context = nnx.swish(self.latent_future_state_in(state))[:, None, :]
        prediction_context = jnp.concatenate(
            [action_context, prior_context, state_context], axis=1
        )
        for block in self.latent_future_blocks:
            hidden = block(hidden, prediction_context)
        predicted_delta = self.latent_future_predict_out(
            _rms_normalize(hidden)
        ).reshape(
            hidden.shape[0],
            camera_count,
            patch_count,
            -1,
        )

        action_queries = self.latent_future_action_queries(
            jnp.arange(self.action_horizon)
        )
        action_queries = jnp.broadcast_to(
            action_queries[None, :, :],
            (hidden.shape[0], *action_queries.shape),
        )
        action_queries = action_queries + state_context
        action_queries = self.latent_future_action_block(
            action_queries, hidden
        )
        residual = self.latent_future_token_out(
            _rms_normalize(action_queries)
        )
        return residual, predicted_delta

    def compute_state_rollout_tokens(self, actions, contexts, state):
        """Predict dense proprioceptive change and return a flow residual."""
        state_context = nnx.swish(self.state_rollout_state_in(state))[:, None, :]
        hidden = self.state_rollout_action_in(actions)
        hidden = hidden + self.state_rollout_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        hidden = hidden + state_context
        prediction_context = jnp.concatenate(
            [self.state_rollout_context_in(contexts), state_context], axis=1
        )
        for block in self.state_rollout_blocks:
            hidden = block(hidden, prediction_context)
        normalized = _rms_normalize(hidden)
        predicted_delta = self.state_rollout_predict_out(normalized)
        residual = self.state_rollout_token_out(normalized)
        return residual, predicted_delta

    def state_rollout_targets(self, observation):
        """Build masked normalized state deltas for post-action timesteps."""
        if observation.future_states is None:
            raise ValueError('state rollout training requires future states')
        if observation.future_state_masks is None:
            raise ValueError('state rollout training requires future state masks')
        if observation.future_states.shape[-2] != self.action_horizon:
            raise ValueError('future state horizon must equal the action horizon')
        target = observation.future_states[..., : self.state_rollout_target_dim]
        current = observation.state[..., : self.state_rollout_target_dim]
        return (
            jax.lax.stop_gradient(target - current[:, None, :]),
            observation.future_state_masks,
        )

    def compute_action_moe_tokens(self, actions, contexts, state):
        """Run a top-k task-conditioned expert bank over an action chunk."""
        state_context = nnx.swish(self.action_moe_state_in(state))[:, None, :]
        hidden = self.action_moe_action_in(actions)
        hidden = hidden + self.action_moe_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        hidden = hidden + state_context
        expert_context = jnp.concatenate(
            [self.action_moe_context_in(contexts), state_context], axis=1
        )
        for block in self.action_moe_blocks:
            hidden = block(hidden, expert_context)
        hidden = _rms_normalize(hidden)

        router_features = _action_moe_router_features(
            hidden,
            expert_context,
            self.action_moe_task_consistent_routing,
        )
        router_logits = self.action_moe_router(router_features).astype(jnp.float32)
        router_logits = router_logits / self.action_moe_temperature
        top_values, top_indices = jax.lax.top_k(
            router_logits, self.action_moe_top_k
        )
        top_weights = jax.nn.softmax(top_values, axis=-1)
        routing_weights = jnp.sum(
            jax.nn.one_hot(
                top_indices,
                self.action_moe_num_experts,
                dtype=top_weights.dtype,
            )
            * top_weights[..., None],
            axis=-2,
        )

        expert_outputs = jnp.stack(
            [
                expert_out(nnx.swish(expert_in(hidden)))
                for expert_in, expert_out in zip(
                    self.action_moe_expert_in,
                    self.action_moe_expert_out,
                    strict=True,
                )
            ],
            axis=-2,
        )
        mixed = hidden + jnp.sum(
            expert_outputs * routing_weights[..., None].astype(
                expert_outputs.dtype
            ),
            axis=-2,
        )
        mixed = _rms_normalize(mixed)

        # Balance is differentiable through the full router distribution even
        # though forward execution uses only the selected experts.  Aggregate
        # over both the batch and action-token axes: task-conditioned experts
        # are allowed to specialize for an individual sample as long as the
        # minibatch as a whole does not collapse onto a subset of experts.
        balance_loss = _action_moe_batch_global_balance_loss(
            router_logits, top_indices
        )
        prediction = self.action_moe_predict_out(mixed)
        residual = self.action_moe_token_out(mixed)
        return residual, prediction, balance_loss, routing_weights

    def compute_task_progress_tokens(self, contexts, state):
        """Infer episode progress and expand it into per-action guidance."""
        context_tokens = self.task_progress_context_in(contexts)
        state_token = nnx.swish(self.task_progress_state_in(state))[:, None, :]
        progress_token = self.task_progress_query(jnp.asarray([0]))[None, :, :]
        progress_token = jnp.broadcast_to(
            progress_token,
            (state.shape[0], 1, progress_token.shape[-1]),
        )
        progress_token = progress_token + state_token
        progress_context = jnp.concatenate(
            [context_tokens, state_token], axis=1
        )
        for block in self.task_progress_blocks:
            progress_token = block(progress_token, progress_context)
        progress_token = _rms_normalize(progress_token)
        logits = self.task_progress_logits(progress_token[:, 0, :])
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

        bin_embeddings = self.task_progress_bin_embeddings(
            jnp.arange(self.task_progress_bins)
        )
        latent_progress = jnp.einsum(
            'bk,kh->bh',
            probabilities.astype(bin_embeddings.dtype),
            bin_embeddings,
        )
        action_tokens = self.task_progress_action_queries(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        action_tokens = action_tokens + latent_progress[:, None, :] + state_token
        action_tokens = self.task_progress_action_block(
            action_tokens,
            jnp.concatenate([progress_context, progress_token], axis=1),
        )
        residual = self.task_progress_token_out(
            _rms_normalize(action_tokens)
        )
        progress_anchors = jnp.linspace(
            0.0, 1.0, self.task_progress_bins, dtype=jnp.float32
        )
        expected_progress = jnp.sum(
            probabilities * progress_anchors[None, :], axis=-1
        )
        return residual, logits, expected_progress, probabilities

    def compute_language_subgoal_tokens(
        self,
        contexts,
        state,
        prefix_context=None,
        prefix_mask=None,
    ):
        """Infer an ordered active subgoal and per-action slot guidance."""
        context_tokens = self.language_subgoal_context_in(contexts)
        state_token = nnx.swish(self.language_subgoal_state_in(state))[:, None, :]
        slot_indices = jnp.arange(self.language_subgoal_slot_count)
        slots = self.language_subgoal_slot_queries(slot_indices)
        slots = slots + self.language_subgoal_slot_positions(slot_indices)
        slots = slots[None, :, :] + state_token
        slots = jnp.broadcast_to(
            slots,
            (state.shape[0], self.language_subgoal_slot_count, slots.shape[-1]),
        )
        if (prefix_context is None) != (prefix_mask is None):
            raise ValueError(
                'language subgoal prefix context and mask must be provided together'
            )
        if prefix_context is not None:
            if prefix_context.ndim != 3 or prefix_mask.ndim != 2:
                raise ValueError(
                    'language subgoal prefix context/mask ranks are invalid'
                )
            if prefix_context.shape[:2] != prefix_mask.shape:
                raise ValueError(
                    'language subgoal prefix context and mask shapes differ'
                )
            if prefix_context.shape[0] != state.shape[0]:
                raise ValueError(
                    'language subgoal prefix batch does not match state'
                )
            valid_prefix = prefix_mask.astype(jnp.bool_)
            prefix_tokens = self.language_subgoal_prefix_in(prefix_context)
            prefix_queries = self.language_subgoal_prefix_query(
                _rms_normalize(slots)
            )
            prefix_logits = jnp.einsum(
                'bsh,bph->bsp',
                prefix_queries,
                _rms_normalize(prefix_tokens),
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(prefix_tokens.shape[-1]))
            prefix_logits = jnp.where(
                valid_prefix[:, None, :], prefix_logits, -1.0e30
            )
            prefix_probabilities = jax.nn.softmax(
                prefix_logits.astype(jnp.float32), axis=-1
            ).astype(prefix_tokens.dtype)
            prefix_summaries = jnp.einsum(
                'bsp,bph->bsh', prefix_probabilities, prefix_tokens
            )
            slots = slots + prefix_summaries
        subgoal_context = jnp.concatenate([context_tokens, state_token], axis=1)
        for block in self.language_subgoal_blocks:
            slots = block(slots, subgoal_context)
        slots = _rms_normalize(slots)

        logits = self.language_subgoal_score(slots)[..., 0]
        logits = logits.astype(jnp.float32) / self.language_subgoal_temperature
        probabilities = jax.nn.softmax(logits, axis=-1)
        active_slot = jnp.einsum(
            'bs,bsh->bh', probabilities.astype(slots.dtype), slots
        )

        action_tokens = self.language_subgoal_action_queries(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        action_tokens = action_tokens + active_slot[:, None, :] + state_token
        action_tokens = self.language_subgoal_action_block(
            action_tokens,
            jnp.concatenate([subgoal_context, slots], axis=1),
        )
        action_tokens = _rms_normalize(action_tokens)
        action_prediction = self.language_subgoal_action_predict(action_tokens)
        residual = self.language_subgoal_token_out(action_tokens)
        if hasattr(self, 'specialist_module_router_score'):
            specialist_gates = self.compute_specialist_module_gates(
                contexts, state
            )
            residual = residual * specialist_gates[:, 2:3, None]
        anchors = jnp.linspace(
            0.0,
            1.0,
            self.language_subgoal_slot_count,
            dtype=jnp.float32,
        )
        expected_progress = jnp.sum(probabilities * anchors[None, :], axis=-1)
        return (
            residual,
            action_prediction,
            logits,
            expected_progress,
            probabilities,
            slots,
        )

    def compute_object_subgoal_binding_tokens(
        self,
        object_slots,
        subgoal_slots,
        subgoal_probabilities,
        contexts,
        state,
    ):
        """Ground the active ordered subgoal in competitive visual objects."""
        objects = self.object_subgoal_binding_object_in(object_slots)
        subgoals = self.object_subgoal_binding_subgoal_in(subgoal_slots)
        object_queries = self.object_subgoal_binding_object_query(
            _rms_normalize(objects)
        )
        # The first role identifies the object being manipulated.  A second,
        # target-conditioned role identifies the spatial/receptacle reference
        # object.  Keeping the two roles explicit is important for relational
        # instructions such as "put A left of B"; a single categorical binding
        # cannot represent both entities without overloading one slot.
        target_keys = self.object_subgoal_binding_subgoal_key(
            _rms_normalize(subgoals)
        )
        target_logits = jnp.einsum(
            'bsd,bod->bso',
            target_keys,
            object_queries,
            preferred_element_type=jnp.float32,
        ) / jnp.sqrt(float(object_queries.shape[-1]))
        target_logits = (
            target_logits / self.object_subgoal_binding_temperature
        )
        target_probabilities = jax.nn.softmax(
            target_logits, axis=-1
        ).astype(objects.dtype)
        target_summary = jnp.einsum(
            'bso,boh->bsh', target_probabilities, objects
        )

        reference_keys = self.object_subgoal_binding_reference_key(
            _rms_normalize(subgoals + target_summary)
        )
        reference_logits = jnp.einsum(
            'bsd,bod->bso',
            reference_keys,
            object_queries,
            preferred_element_type=jnp.float32,
        ) / jnp.sqrt(float(object_queries.shape[-1]))
        reference_logits = (
            reference_logits / self.object_subgoal_binding_temperature
        )
        reference_logits = self.object_subgoal_reference_logits(
            reference_logits, target_probabilities
        )
        reference_probabilities = jax.nn.softmax(
            reference_logits, axis=-1
        ).astype(objects.dtype)
        relation_gate = jax.nn.sigmoid(
            self.object_subgoal_binding_relation_gate(subgoals)
        ).astype(objects.dtype)

        subgoal_count = subgoals.shape[1]
        object_count = objects.shape[1]
        object_pairs = jnp.broadcast_to(
            objects[:, None, :, :],
            (
                objects.shape[0],
                subgoal_count,
                object_count,
                objects.shape[-1],
            ),
        )
        subgoal_pairs = jnp.broadcast_to(
            subgoals[:, :, None, :], object_pairs.shape
        )

        target_objects = objects[:, None, :, None, :]
        reference_objects = objects[:, None, None, :, :]
        relation_shape = (
            objects.shape[0],
            subgoals.shape[1],
            objects.shape[1],
            objects.shape[1],
            objects.shape[-1],
        )
        target_objects = jnp.broadcast_to(target_objects, relation_shape)
        reference_objects = jnp.broadcast_to(reference_objects, relation_shape)
        relation_subgoals = jnp.broadcast_to(
            subgoals[:, :, None, None, :], relation_shape
        )
        relation_hypotheses = nnx.swish(
            self.object_subgoal_binding_relation_fuse(
                jnp.concatenate(
                    [
                        relation_subgoals,
                        target_objects,
                        reference_objects,
                        target_objects - reference_objects,
                        target_objects * reference_objects,
                    ],
                    axis=-1,
                )
            )
        )
        relation_probabilities = (
            target_probabilities[:, :, :, None]
            * reference_probabilities[:, :, None, :]
        )
        relation_summary = jnp.einsum(
            'bsto,bstoh->bsh',
            relation_probabilities.astype(relation_hypotheses.dtype),
            relation_hypotheses,
        )
        pair_tokens = nnx.swish(
            self.object_subgoal_binding_pair_fuse(
                jnp.concatenate([subgoal_pairs, object_pairs], axis=-1)
            )
        )
        pair_tokens = pair_tokens + (
            relation_gate[:, :, None, :] * relation_summary[:, :, None, :]
        )
        role_probabilities = (
            target_probabilities
            + relation_gate * reference_probabilities
        ) / (1.0 + relation_gate)
        active_pair_weights = (
            subgoal_probabilities.astype(pair_tokens.dtype)[..., None]
            * role_probabilities
        )
        pair_tokens = pair_tokens * (
            1.0
            + active_pair_weights[..., None]
            * float(subgoal_count * object_count)
        )
        pair_tokens = pair_tokens.reshape(
            pair_tokens.shape[0],
            subgoal_count * object_count,
            pair_tokens.shape[-1],
        )
        state_token = nnx.swish(
            self.object_subgoal_binding_state_in(state)
        )
        binding_context = self.object_subgoal_binding_context_in(contexts)
        binding_context = jnp.concatenate(
            [binding_context, state_token[:, None, :]], axis=1
        )
        for block in self.object_subgoal_binding_blocks:
            pair_tokens = block(pair_tokens, binding_context)
        pair_tokens = _rms_normalize(pair_tokens)

        action_tokens = self.object_subgoal_binding_action_queries(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        action_tokens = action_tokens + state_token[:, None, :]
        action_tokens = jnp.broadcast_to(
            action_tokens,
            (
                state.shape[0],
                self.action_horizon,
                action_tokens.shape[-1],
            ),
        )
        action_tokens = self.object_subgoal_binding_action_block(
            action_tokens,
            jnp.concatenate([pair_tokens, binding_context], axis=1),
        )
        action_tokens = _rms_normalize(action_tokens)
        action_prediction = self.object_subgoal_binding_action_predict(
            action_tokens
        )
        residual = self.object_subgoal_binding_token_out(action_tokens)
        if hasattr(self, 'specialist_module_router_score'):
            specialist_gates = self.compute_specialist_module_gates(
                contexts, state
            )
            # Gate two is the language/subgoal family.  Object grounding of
            # the active subgoal belongs to the same semantic capability.
            residual = residual * specialist_gates[:, 2:3, None]
        return (
            residual,
            action_prediction,
            target_probabilities,
            reference_probabilities,
            relation_gate,
        )

    def object_subgoal_reference_logits(
        self, reference_logits, target_probabilities
    ):
        """Discourage relational target/reference roles from sharing one slot."""
        if not self.object_subgoal_binding_distinct_roles:
            return reference_logits
        available_mass = 1.0 - jax.lax.stop_gradient(
            target_probabilities.astype(jnp.float32)
        )
        distinct_role_bias = jnp.log(
            jnp.clip(available_mass, 1.0e-4, 1.0)
        )
        return reference_logits.astype(jnp.float32) + distinct_role_bias

    def compute_kinematic_action_tokens(self, actions, contexts, state):
        """Reason jointly over factorized translation/rotation/gripper streams."""
        translation = self.kinematic_action_translation_in(actions[..., :3])
        rotation = self.kinematic_action_rotation_in(actions[..., 3:6])
        gripper = self.kinematic_action_gripper_in(actions[..., 6:7])
        group_tokens = jnp.stack([translation, rotation, gripper], axis=2)

        context_tokens = self.kinematic_action_context_in(contexts)
        state_token = nnx.swish(self.kinematic_action_state_in(state))
        pooled_context = jnp.mean(context_tokens, axis=1) + state_token
        time_positions = self.kinematic_action_time_position(
            jnp.arange(self.action_horizon)
        )[None, :, None, :]
        group_positions = self.kinematic_action_group_position(
            jnp.arange(3)
        )[None, None, :, :]
        group_tokens = (
            group_tokens
            + time_positions
            + group_positions
            + pooled_context[:, None, None, :]
        )
        flat_tokens = group_tokens.reshape(
            group_tokens.shape[0],
            self.action_horizon * 3,
            group_tokens.shape[-1],
        )
        reasoner_context = jnp.concatenate(
            [context_tokens, state_token[:, None, :]], axis=1
        )
        for block in self.kinematic_action_blocks:
            flat_tokens = block(flat_tokens, reasoner_context)
        group_tokens = _rms_normalize(flat_tokens).reshape(
            flat_tokens.shape[0], self.action_horizon, 3, flat_tokens.shape[-1]
        )

        active_prediction = jnp.concatenate(
            [
                self.kinematic_action_translation_predict(group_tokens[:, :, 0]),
                self.kinematic_action_rotation_predict(group_tokens[:, :, 1]),
                self.kinematic_action_gripper_predict(group_tokens[:, :, 2]),
            ],
            axis=-1,
        )
        prediction = jnp.pad(
            active_prediction,
            ((0, 0), (0, 0), (0, self.action_dim - 7)),
        )
        fused = self.kinematic_action_group_fusion(
            group_tokens.reshape(
                group_tokens.shape[0],
                self.action_horizon,
                3 * group_tokens.shape[-1],
            )
        )
        residual = self.kinematic_action_token_out(_rms_normalize(fused))
        return residual, prediction, group_tokens

    def spectral_action_basis(self, dtype=jnp.float32):
        """Return the complete orthonormal DCT-II basis for the action horizon."""
        frequency = jnp.arange(self.action_horizon, dtype=jnp.float32)[:, None]
        position = jnp.arange(self.action_horizon, dtype=jnp.float32)[None, :]
        basis = jnp.cos(
            (jnp.pi / self.action_horizon)
            * (position + 0.5)
            * frequency
        )
        scale = jnp.full(
            (self.action_horizon, 1),
            jnp.sqrt(2.0 / self.action_horizon),
            dtype=jnp.float32,
        )
        scale = scale.at[0].set(jnp.sqrt(1.0 / self.action_horizon))
        return (basis * scale).astype(dtype)

    def action_to_spectrum(self, actions):
        """Transform every action coordinate without dropping frequencies."""
        basis = self.spectral_action_basis(actions.dtype)
        return jnp.einsum('ft,bta->bfa', basis, actions)

    def spectrum_to_action_tokens(self, spectral_tokens):
        """Invert all frequency tokens into the original action positions."""
        basis = self.spectral_action_basis(spectral_tokens.dtype)
        return jnp.einsum('ft,bfw->btw', basis, spectral_tokens)

    def compute_spectral_action_tokens(self, actions, contexts, state):
        """Reason over a lossless low/mid/high-frequency action representation."""
        spectrum = self.action_to_spectrum(actions)
        frequency_indices = jnp.arange(self.action_horizon)
        band_indices = jnp.minimum(
            frequency_indices
            * self.spectral_action_band_count
            // self.action_horizon,
            self.spectral_action_band_count - 1,
        )
        spectral_tokens = (
            self.spectral_action_in(spectrum)
            + self.spectral_action_frequency_position(frequency_indices)[None]
            + self.spectral_action_band_position(band_indices)[None]
        )
        context_tokens = self.spectral_action_context_in(contexts)
        state_token = nnx.swish(self.spectral_action_state_in(state))[:, None, :]
        spectral_tokens = spectral_tokens + state_token
        reasoner_context = jnp.concatenate(
            [context_tokens, state_token], axis=1
        )
        for block in self.spectral_action_blocks:
            spectral_tokens = block(spectral_tokens, reasoner_context)
        spectral_tokens = _rms_normalize(spectral_tokens)
        prediction = self.spectral_action_predict(spectral_tokens)
        frequency_residual = self.spectral_action_token_out(spectral_tokens)
        residual = self.spectrum_to_action_tokens(frequency_residual)
        return residual, prediction, spectral_tokens

    def compute_velocity_refinement(
        self,
        noisy_actions,
        base_velocity,
        suffix_hidden,
        contexts,
        state,
        timestep,
    ):
        """Predict and safely gate the error left by the inherited velocity."""
        detached_velocity = jax.lax.stop_gradient(base_velocity)
        detached_hidden = jax.lax.stop_gradient(suffix_hidden)
        time_expanded = timestep[..., None, None]
        implied_clean = jax.lax.stop_gradient(
            noisy_actions - time_expanded * detached_velocity
        )
        tokens = self.velocity_refiner_action_in(
            jnp.concatenate(
                [noisy_actions, detached_velocity, implied_clean], axis=-1
            )
        )
        tokens = tokens + self.velocity_refiner_hidden_in(detached_hidden)

        context_tokens = self.velocity_refiner_context_in(contexts)
        repeat = self.action_horizon // context_tokens.shape[1]
        aligned_context = jnp.repeat(context_tokens, repeat, axis=1)
        state_token = nnx.swish(self.velocity_refiner_state_in(state))[:, None, :]
        time_token = self.velocity_refiner_time_in(
            posemb_sincos(
                timestep,
                self.velocity_refiner_action_in.out_features,
                min_period=4e-3,
                max_period=4.0,
            )
        )[:, None, :]
        positions = self.velocity_refiner_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        tokens = tokens + aligned_context + state_token + time_token + positions
        reasoner_context = jnp.concatenate(
            [context_tokens, state_token, time_token], axis=1
        )
        for block in self.velocity_refiner_blocks:
            tokens = block(tokens, reasoner_context)
        tokens = _rms_normalize(tokens)
        predicted_residual = self.velocity_refiner_predict(tokens)
        gain = jnp.tanh(self.velocity_refiner_gain.value).astype(
            predicted_residual.dtype
        )
        gated_residual = predicted_residual * gain[None, None]
        if hasattr(self, 'specialist_module_router_score'):
            specialist_gates = self.compute_specialist_module_gates(
                contexts, state
            )
            gated_residual = gated_residual * specialist_gates[:, 1:2, None]
        corrected_velocity = base_velocity + gated_residual
        return corrected_velocity, predicted_residual, tokens

    def compute_action_visual_refinement(
        self,
        noisy_actions,
        base_velocity,
        suffix_hidden,
        contextual_prefix,
        prefix_mask,
        state,
        timestep,
    ):
        """Re-read dense visual/language evidence for the proposed trajectory."""
        detached_velocity = jax.lax.stop_gradient(base_velocity)
        detached_hidden = jax.lax.stop_gradient(suffix_hidden)
        detached_prefix = jax.lax.stop_gradient(contextual_prefix)
        time_expanded = timestep[..., None, None]
        implied_clean = jax.lax.stop_gradient(
            noisy_actions - time_expanded * detached_velocity
        )
        queries = self.action_visual_refiner_action_in(
            jnp.concatenate(
                [noisy_actions, detached_velocity, implied_clean], axis=-1
            )
        )
        queries = queries + self.action_visual_refiner_hidden_in(detached_hidden)
        state_token = nnx.swish(
            self.action_visual_refiner_state_in(state)
        )[:, None, :]
        time_token = self.action_visual_refiner_time_in(
            posemb_sincos(
                timestep,
                self.action_visual_refiner_action_in.out_features,
                min_period=4e-3,
                max_period=4.0,
            )
        )[:, None, :]
        positions = self.action_visual_refiner_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        queries = queries + state_token + time_token + positions
        prefix_tokens = self.action_visual_refiner_prefix_in(detached_prefix)
        for block in self.action_visual_refiner_blocks:
            queries = block(queries, prefix_tokens, prefix_mask)
        queries = _rms_normalize(queries)
        predicted_residual = self.action_visual_refiner_predict(queries)
        gain = jnp.tanh(self.action_visual_refiner_gain.value).astype(
            predicted_residual.dtype
        )
        corrected_velocity = base_velocity + predicted_residual * gain[None, None]
        return corrected_velocity, predicted_residual, queries

    def compute_masked_spatial_tokens(
        self,
        prefix_tokens,
        prefix_mask,
        observation,
        contextual_prefix_tokens,
        *,
        train: bool,
        rng=None,
    ):
        """Build scene tokens and reconstruct only masked current-view patches."""
        if observation.tokenized_prompt is None:
            raise ValueError('masked spatial reasoning requires language tokens')
        if train and rng is None:
            raise ValueError('masked spatial training requires an RNG key')
        camera_names = tuple(observation.images)
        camera_count = len(camera_names)
        if camera_count > self.masked_spatial_max_cameras:
            raise ValueError(
                'observation camera count exceeds masked_spatial_max_cameras'
            )
        language_tokens = observation.tokenized_prompt.shape[1]
        image_tokens = prefix_tokens.shape[1] - language_tokens
        if image_tokens <= 0 or image_tokens % camera_count:
            raise ValueError(
                'visual prefix cannot be evenly partitioned across cameras'
            )
        patches_per_camera = image_tokens // camera_count
        grid_size = math.isqrt(patches_per_camera)
        if grid_size * grid_size != patches_per_camera:
            raise ValueError(
                'masked spatial reasoning requires a square patch grid'
            )
        if grid_size > self.masked_spatial_max_grid_size:
            raise ValueError(
                'visual patch grid exceeds masked_spatial_max_grid_size'
            )

        batch_size = prefix_tokens.shape[0]
        raw_visual = prefix_tokens[:, :image_tokens].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        contextual_visual = contextual_prefix_tokens[:, :image_tokens].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        visual = self.masked_spatial_image_in(raw_visual)
        visual = visual + self.masked_spatial_contextual_in(contextual_visual)
        row_ids = jnp.repeat(jnp.arange(grid_size), grid_size)
        column_ids = jnp.tile(jnp.arange(grid_size), grid_size)
        geometry = (
            self.masked_spatial_camera_position(jnp.arange(camera_count))[
                :, None, :
            ]
            + self.masked_spatial_row_position(row_ids)[None, :, :]
            + self.masked_spatial_column_position(column_ids)[None, :, :]
        )
        visual = visual + geometry[None, :, :, :]
        camera_mask = jnp.stack(
            [self._physical_image_mask(observation, name) for name in camera_names],
            axis=1,
        )
        visual_mask = jnp.repeat(
            camera_mask[:, :, None], patches_per_camera, axis=2
        )

        reconstruction_mask = None
        if train:
            reconstruction_mask = (
                jax.random.uniform(
                    rng,
                    (batch_size, camera_count, patches_per_camera),
                )
                < self.masked_spatial_mask_ratio
            ) & visual_mask
            mask_token = self.masked_spatial_mask_token(jnp.asarray(0))
            masked_value = mask_token[None, None, None, :] + geometry[
                None, :, :, :
            ]
            visual = jnp.where(
                reconstruction_mask[..., None], masked_value, visual
            )

        language = self.masked_spatial_language_in(
            contextual_prefix_tokens[:, image_tokens:]
        )
        language_mask = prefix_mask[:, image_tokens:]
        language_weight = language_mask.astype(language.dtype)[..., None]
        pooled_language = jnp.sum(language * language_weight, axis=1) / jnp.maximum(
            jnp.sum(language_weight, axis=1), 1.0
        )
        state_token = nnx.swish(
            self.masked_spatial_state_in(observation.state)
        )
        scene_tokens = self.masked_spatial_scene_queries(
            jnp.arange(self.masked_spatial_query_count)
        )
        scene_tokens = jnp.broadcast_to(
            scene_tokens[None, :, :],
            (batch_size, *scene_tokens.shape),
        )
        scene_tokens = (
            scene_tokens
            + pooled_language[:, None, :]
            + state_token[:, None, :]
        )
        visual_flat = visual.reshape(batch_size, image_tokens, -1)
        visual_mask_flat = visual_mask.reshape(batch_size, image_tokens)
        scene_context = jnp.concatenate([visual_flat, language], axis=1)
        scene_context_mask = jnp.concatenate(
            [visual_mask_flat, language_mask], axis=1
        )
        for block in self.masked_spatial_scene_blocks:
            scene_tokens = block(
                scene_tokens, scene_context, scene_context_mask
            )

        reconstruction = None
        target = None
        if train:
            query = self.masked_spatial_decoder_q(
                _rms_normalize(visual_flat)
            )
            key = self.masked_spatial_decoder_k(
                _rms_normalize(scene_tokens)
            )
            value = self.masked_spatial_decoder_v(
                _rms_normalize(scene_tokens)
            )
            head_dim = query.shape[-1] // self.masked_spatial_num_heads
            query = query.reshape(
                batch_size,
                image_tokens,
                self.masked_spatial_num_heads,
                head_dim,
            )
            key = key.reshape(
                batch_size,
                self.masked_spatial_query_count,
                self.masked_spatial_num_heads,
                head_dim,
            )
            value = value.reshape(key.shape)
            logits = jnp.einsum(
                'bvhd,bshd->bhvs',
                query,
                key,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(head_dim))
            weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
            attended = jnp.einsum(
                'bhvs,bshd->bvhd', weights, value
            ).reshape(batch_size, image_tokens, -1)
            decoded = visual_flat + self.masked_spatial_decoder_out(attended)
            reconstruction = self.masked_spatial_reconstruct(
                _rms_normalize(decoded)
            )
            # The frozen raw SigLIP/PaliGemma image embedding is a stable
            # non-collapsing target; gradients flow through contextual source
            # tokens and hence specialize enabled VLM LoRA weights.
            target = jax.lax.stop_gradient(
                _rms_normalize(
                    raw_visual.reshape(batch_size, image_tokens, -1)
                )
            )
            reconstruction_mask = reconstruction_mask.reshape(
                batch_size, image_tokens
            )

        waypoint_tokens = self.masked_spatial_waypoint_queries(
            jnp.arange(self.action_prior_horizon)
        )
        waypoint_tokens = jnp.broadcast_to(
            waypoint_tokens[None, :, :],
            (batch_size, *waypoint_tokens.shape),
        )
        waypoint_tokens = (
            waypoint_tokens
            + pooled_language[:, None, :]
            + state_token[:, None, :]
        )
        waypoint_tokens = self.masked_spatial_waypoint_block(
            waypoint_tokens, scene_tokens
        )
        waypoint_tokens = _rms_normalize(waypoint_tokens)
        coarse_actions = self.masked_spatial_action_out(waypoint_tokens)
        action_tokens = self.masked_spatial_token_out(waypoint_tokens)
        repeat = self.action_horizon // self.action_prior_horizon
        return (
            jnp.repeat(action_tokens, repeat, axis=1),
            coarse_actions,
            reconstruction,
            target,
            reconstruction_mask,
            scene_tokens,
        )

    def compute_object_future_tokens(
        self,
        prefix_tokens,
        prefix_mask,
        observation,
        contextual_prefix_tokens,
        contexts,
        *,
        train: bool,
        object_affordance_slots=None,
    ):
        """Forecast object slots and dense future frozen-feature targets."""
        if observation.tokenized_prompt is None:
            raise ValueError('object future reasoning requires language tokens')
        camera_names = tuple(observation.images)
        selected_indices = [
            camera_names.index(name)
            for name in self.object_future_camera_names
            if name in camera_names
        ]
        if len(selected_indices) != len(self.object_future_camera_names):
            raise ValueError('object future reasoning is missing a deployed camera')

        language_count = observation.tokenized_prompt.shape[1]
        image_count = prefix_tokens.shape[1] - language_count
        camera_count = len(camera_names)
        if image_count <= 0 or image_count % camera_count:
            raise ValueError(
                'object future visual prefix cannot be partitioned by camera'
            )
        patches_per_camera = image_count // camera_count
        grid_size = math.isqrt(patches_per_camera)
        if grid_size * grid_size != patches_per_camera:
            raise ValueError('object future reasoning requires square patch grids')
        if grid_size > self.object_future_max_grid_size:
            raise ValueError(
                'object future patch grid exceeds configured maximum'
            )

        batch_size = prefix_tokens.shape[0]
        raw_all = prefix_tokens[:, :image_count].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        contextual_all = contextual_prefix_tokens[:, :image_count].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        raw_visual = jnp.stack(
            [raw_all[:, index] for index in selected_indices], axis=1
        )
        contextual_visual = jnp.stack(
            [contextual_all[:, index] for index in selected_indices], axis=1
        )
        row_ids = jnp.repeat(jnp.arange(grid_size), grid_size)
        column_ids = jnp.tile(jnp.arange(grid_size), grid_size)
        geometry = (
            self.object_future_camera_position(
                jnp.arange(len(self.object_future_camera_names))
            )[:, None, :]
            + _dense_embedding_lookup(
                self.object_future_row_position, row_ids
            )[None, :, :]
            + _dense_embedding_lookup(
                self.object_future_column_position, column_ids
            )[None, :, :]
        )
        visual = (
            self.object_future_image_in(raw_visual)
            + self.object_future_contextual_in(contextual_visual)
            + geometry[None, :, :, :]
        )
        visual_mask = jnp.stack(
            [
                self._physical_image_mask(observation, name)
                for name in self.object_future_camera_names
            ],
            axis=1,
        )
        visual_mask = jnp.repeat(
            visual_mask[:, :, None], patches_per_camera, axis=2
        )

        language = self.object_future_language_in(
            contextual_prefix_tokens[:, image_count:]
        )
        language_mask = prefix_mask[:, image_count:]
        language_weight = language_mask.astype(language.dtype)[..., None]
        pooled_language = jnp.sum(language * language_weight, axis=1) / jnp.maximum(
            jnp.sum(language_weight, axis=1), 1.0
        )
        state_token = nnx.swish(
            self.object_future_state_in(observation.state)
        )
        object_tokens = self.object_future_object_queries(
            jnp.arange(self.object_future_query_count)
        )
        object_tokens = jnp.broadcast_to(
            object_tokens[None, :, :],
            (batch_size, *object_tokens.shape),
        )
        object_tokens = (
            object_tokens
            + pooled_language[:, None, :]
            + state_token[:, None, :]
        )
        visual_flat = visual.reshape(batch_size, -1, visual.shape[-1])
        visual_mask_flat = visual_mask.reshape(batch_size, -1)
        object_context = jnp.concatenate([visual_flat, language], axis=1)
        object_context_mask = jnp.concatenate(
            [visual_mask_flat, language_mask], axis=1
        )
        for block in self.object_future_object_blocks:
            object_tokens = block(
                object_tokens, object_context, object_context_mask
            )

        forecast_tokens = self.object_future_forecast_queries(
            jnp.arange(self.object_future_query_count)
        )
        forecast_tokens = jnp.broadcast_to(
            forecast_tokens[None, :, :],
            (batch_size, *forecast_tokens.shape),
        )
        forecast_tokens = (
            forecast_tokens
            + object_tokens
            + pooled_language[:, None, :]
            + state_token[:, None, :]
        )
        forecast_parts = [object_tokens, self.object_future_prior_in(contexts)]
        bridged_affordance = None
        if hasattr(self, 'object_future_affordance_in'):
            if (
                object_affordance_slots is None
                or object_affordance_slots.ndim != 3
                or object_affordance_slots.shape[0] != batch_size
            ):
                raise ValueError(
                    'object future bridge requires competitive affordance slots'
                )
            bridged_affordance = self.object_future_affordance_in(
                _rms_normalize(object_affordance_slots)
            )
            forecast_parts.append(bridged_affordance)
        forecast_context = jnp.concatenate(forecast_parts, axis=1)
        for block in self.object_future_forecast_blocks:
            forecast_tokens = block(forecast_tokens, forecast_context)

        reconstruction = None
        target = None
        target_mask = None
        if train:
            if observation.future_images is None:
                raise ValueError(
                    'object future training requires future RGB targets'
                )
            future_masks = observation.future_image_masks or {}
            targets = []
            masks = []
            for name in self.object_future_camera_names:
                if name not in observation.future_images:
                    raise ValueError(f'missing future image target {name!r}')
                future_tokens, _ = self.PaliGemma.img(
                    observation.future_images[name], train=False
                )
                if future_tokens.shape[1] != patches_per_camera:
                    raise ValueError(
                        'future and current object patch grids must match'
                    )
                targets.append(future_tokens)
                masks.append(
                    jnp.asarray(
                        future_masks.get(
                            name,
                            jnp.ones(
                                (batch_size,), dtype=jnp.bool_
                            ),
                        )
                    )
                    & self._physical_image_mask(observation, name)
                )
            target = jax.lax.stop_gradient(
                _rms_normalize(
                    jnp.stack(targets, axis=1).reshape(
                        batch_size, -1, targets[0].shape[-1]
                    )
                )
            )
            target_mask = jnp.repeat(
                jnp.stack(masks, axis=1)[:, :, None],
                patches_per_camera,
                axis=2,
            ).reshape(batch_size, -1)

            patch_queries = geometry.reshape(-1, geometry.shape[-1])
            patch_queries = jnp.broadcast_to(
                patch_queries[None, :, :],
                (batch_size, *patch_queries.shape),
            )
            query = self.object_future_decoder_q(
                _rms_normalize(patch_queries)
            )
            key = self.object_future_decoder_k(
                _rms_normalize(forecast_tokens)
            )
            value = self.object_future_decoder_v(
                _rms_normalize(forecast_tokens)
            )
            head_dim = query.shape[-1] // self.object_future_num_heads
            query = query.reshape(
                batch_size,
                query.shape[1],
                self.object_future_num_heads,
                head_dim,
            )
            key = key.reshape(
                batch_size,
                self.object_future_query_count,
                self.object_future_num_heads,
                head_dim,
            )
            value = value.reshape(key.shape)
            logits = jnp.einsum(
                'bvhd,bshd->bhvs',
                query,
                key,
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(head_dim))
            weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
            attended = jnp.einsum(
                'bhvs,bshd->bvhd', weights, value
            ).reshape(batch_size, query.shape[1], -1)
            decoded = patch_queries + self.object_future_decoder_out(attended)
            reconstruction = self.object_future_reconstruct(
                _rms_normalize(decoded)
            )

        waypoint_tokens = self.object_future_waypoint_queries(
            jnp.arange(self.action_prior_horizon)
        )
        waypoint_tokens = jnp.broadcast_to(
            waypoint_tokens[None, :, :],
            (batch_size, *waypoint_tokens.shape),
        )
        waypoint_tokens = (
            waypoint_tokens
            + pooled_language[:, None, :]
            + state_token[:, None, :]
        )
        waypoint_tokens = self.object_future_waypoint_block(
            waypoint_tokens,
            jnp.concatenate(
                [
                    object_tokens,
                    forecast_tokens,
                    *(
                        [bridged_affordance]
                        if bridged_affordance is not None
                        else []
                    ),
                ],
                axis=1,
            ),
        )
        waypoint_tokens = _rms_normalize(waypoint_tokens)
        coarse_actions = self.object_future_action_out(waypoint_tokens)
        action_tokens = self.object_future_token_out(waypoint_tokens)
        repeat = self.action_horizon // self.action_prior_horizon
        return (
            jnp.repeat(action_tokens, repeat, axis=1),
            coarse_actions,
            reconstruction,
            target,
            target_mask,
            object_tokens,
            forecast_tokens,
        )

    def compute_predicate_binding_tokens(
        self,
        prefix_tokens,
        prefix_mask,
        observation,
        contextual_prefix_tokens,
        contexts,
    ):
        """Bind target/predicate/reference language roles to visual objects."""
        if observation.tokenized_prompt is None:
            raise ValueError('predicate binding requires language tokens')
        camera_names = tuple(observation.images)
        camera_count = len(camera_names)
        if camera_count > self.predicate_binding_max_cameras:
            raise ValueError(
                'observation camera count exceeds '
                'predicate_binding_max_cameras'
            )
        language_count = observation.tokenized_prompt.shape[1]
        image_count = prefix_tokens.shape[1] - language_count
        if image_count <= 0 or image_count % camera_count:
            raise ValueError(
                'predicate binding visual prefix cannot be partitioned by camera'
            )
        patches_per_camera = image_count // camera_count
        grid_size = math.isqrt(patches_per_camera)
        if grid_size * grid_size != patches_per_camera:
            raise ValueError('predicate binding requires square patch grids')
        if grid_size > self.predicate_binding_max_grid_size:
            raise ValueError(
                'predicate binding patch grid exceeds configured maximum'
            )

        batch_size = prefix_tokens.shape[0]
        raw_visual = prefix_tokens[:, :image_count].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        contextual_visual = contextual_prefix_tokens[:, :image_count].reshape(
            batch_size, camera_count, patches_per_camera, -1
        )
        row_ids = jnp.repeat(jnp.arange(grid_size), grid_size)
        column_ids = jnp.tile(jnp.arange(grid_size), grid_size)
        geometry = (
            self.predicate_binding_camera_position(jnp.arange(camera_count))[
                :, None, :
            ]
            + self.predicate_binding_row_position(row_ids)[None, :, :]
            + self.predicate_binding_column_position(column_ids)[None, :, :]
        )
        visual = (
            self.predicate_binding_image_in(raw_visual)
            + self.predicate_binding_contextual_in(contextual_visual)
            + geometry[None, :, :, :]
        ).reshape(batch_size, image_count, -1)
        visual_mask = jnp.stack(
            [self._physical_image_mask(observation, name) for name in camera_names],
            axis=1,
        )
        visual_mask = jnp.repeat(
            visual_mask[:, :, None], patches_per_camera, axis=2
        ).reshape(batch_size, image_count)

        language = self.predicate_binding_language_in(
            contextual_prefix_tokens[:, image_count:]
        )
        independent_language = self.predicate_binding_language_in(
            prefix_tokens[:, image_count:]
        )
        language_mask = prefix_mask[:, image_count:]
        state_token = nnx.swish(
            self.predicate_binding_state_in(observation.state)
        )

        # First discover reusable visual objects without access to language.
        object_tokens = self.predicate_binding_object_queries(
            jnp.arange(self.predicate_binding_object_count)
        )
        object_tokens = jnp.broadcast_to(
            object_tokens[None, :, :],
            (batch_size, *object_tokens.shape),
        )
        object_queries = self.predicate_binding_object_query_in(
            _rms_normalize(object_tokens)
        )
        patch_keys = self.predicate_binding_patch_key_in(
            _rms_normalize(visual)
        )
        patch_values = self.predicate_binding_patch_value_in(
            _rms_normalize(visual)
        )
        assignment_logits = jnp.einsum(
            'bsd,bvd->bsv',
            object_queries,
            patch_keys,
            preferred_element_type=jnp.float32,
        ) / jnp.sqrt(float(object_queries.shape[-1]))
        assignment_logits = jnp.where(
            visual_mask[:, None, :], assignment_logits, -1.0e30
        )
        object_assignments = jax.nn.softmax(
            assignment_logits, axis=1
        ).astype(patch_values.dtype)
        object_assignments = object_assignments * visual_mask[
            :, None, :
        ].astype(patch_values.dtype)
        object_weights = object_assignments / jnp.maximum(
            jnp.sum(object_assignments, axis=-1, keepdims=True),
            jnp.asarray(1.0e-6, dtype=patch_values.dtype),
        )
        object_updates = jnp.einsum(
            'bsv,bvd->bsd', object_weights, patch_values
        )
        object_tokens = object_tokens + self.predicate_binding_object_update(
            object_updates
        )
        for block in self.predicate_binding_object_blocks:
            object_tokens = block(object_tokens, visual, visual_mask)

        # The fixed three queries represent target object, relation predicate,
        # and reference object.  They parse only the contextualized prompt,
        # then bind to the independently constructed object bank.
        role_tokens = self.predicate_binding_role_queries(
            jnp.arange(self.predicate_binding_role_count)
        )
        role_tokens = jnp.broadcast_to(
            role_tokens[None, :, :],
            (batch_size, *role_tokens.shape),
        )
        for block in self.predicate_binding_role_blocks:
            role_tokens = block(role_tokens, language, language_mask)
        role_queries = self.predicate_binding_role_query_in(
            _rms_normalize(role_tokens)
        )
        object_keys = self.predicate_binding_object_key_in(
            _rms_normalize(object_tokens)
        )
        role_logits = jnp.einsum(
            'brd,bsd->brs',
            role_queries,
            object_keys,
            preferred_element_type=jnp.float32,
        ) / jnp.sqrt(float(role_queries.shape[-1]))
        role_bindings = jax.nn.softmax(role_logits, axis=-1).astype(
            object_tokens.dtype
        )
        bound_objects = jnp.einsum(
            'brs,bsd->brd', role_bindings, object_tokens
        )
        predicate_tokens = role_tokens + self.predicate_binding_role_object_out(
            bound_objects
        )
        for block in self.predicate_binding_graph_blocks:
            predicate_tokens = block(predicate_tokens, object_tokens)

        # Multi-positive contrastive targets treat exact repeated prompts in a
        # batch as equivalent positives instead of false negatives.  The
        # visual embedding sees the language-independent object bank only.
        language_weight = language_mask.astype(independent_language.dtype)[
            ..., None
        ]
        independent_text = jnp.sum(
            independent_language * language_weight, axis=1
        ) / jnp.maximum(jnp.sum(language_weight, axis=1), 1.0)
        text_embedding = _rms_normalize(
            self.predicate_binding_text_match_out(
                independent_text
            )
        )
        visual_embedding = _rms_normalize(
            self.predicate_binding_visual_match_out(
                jnp.mean(object_tokens, axis=1)
            )
        )
        contrastive_logits = jnp.einsum(
            'bd,cd->bc',
            text_embedding,
            visual_embedding,
            preferred_element_type=jnp.float32,
        ) / self.predicate_binding_temperature
        prompt_tokens = observation.tokenized_prompt
        same_tokens = jnp.all(
            prompt_tokens[:, None, :] == prompt_tokens[None, :, :], axis=-1
        )
        same_masks = jnp.all(
            language_mask[:, None, :] == language_mask[None, :, :], axis=-1
        )
        positive_mask = same_tokens & same_masks

        waypoint_tokens = self.predicate_binding_waypoint_queries(
            jnp.arange(self.action_prior_horizon)
        )
        waypoint_tokens = jnp.broadcast_to(
            waypoint_tokens[None, :, :],
            (batch_size, *waypoint_tokens.shape),
        )
        waypoint_tokens = (
            waypoint_tokens
            + state_token[:, None, :]
            + self.predicate_binding_prior_in(contexts)
        )
        waypoint_tokens = self.predicate_binding_waypoint_block(
            waypoint_tokens, predicate_tokens
        )
        waypoint_tokens = _rms_normalize(waypoint_tokens)
        coarse_actions = self.predicate_binding_action_out(waypoint_tokens)
        action_tokens = self.predicate_binding_token_out(waypoint_tokens)
        repeat = self.action_horizon // self.action_prior_horizon
        return (
            jnp.repeat(action_tokens, repeat, axis=1),
            coarse_actions,
            contrastive_logits,
            positive_mask,
            object_assignments,
            role_bindings,
            predicate_tokens,
        )

    def fuse_predictive_world_model_tokens(
        self,
        latent_future_tokens,
        state_rollout_tokens,
        task_progress_tokens,
        contexts,
        state,
        *,
        action_moe_tokens=None,
        persistent_memory=None,
        persistent_program=None,
        persistent_frontier=None,
    ):
        """Fuse calibrated predictive and phase-specialized action branches."""
        if not hasattr(self, 'predictive_world_model_gate'):
            raise ValueError('predictive world-model fusion is disabled')
        tokens = [
            latent_future_tokens,
            state_rollout_tokens,
            task_progress_tokens,
        ]
        if self.predictive_world_model_include_action_moe:
            tokens.append(action_moe_tokens)
        elif action_moe_tokens is not None:
            raise ValueError('action-MoE tokens were supplied to a three-branch gate')
        if any(token is None for token in tokens):
            raise ValueError(
                'predictive world-model fusion requires all three token paths'
            )
        expected_shape = tokens[0].shape
        if any(token.shape != expected_shape for token in tokens[1:]):
            raise ValueError(
                'predictive world-model token shapes must match exactly'
            )
        pooled_context = jnp.mean(contexts, axis=1)
        hidden = nnx.swish(
            self.predictive_world_model_context_in(pooled_context)
            + self.predictive_world_model_state_in(state)
        )[:, None, :]
        positions = self.predictive_world_model_action_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        if hasattr(self, 'predictive_world_model_memory_in'):
            if persistent_memory is None:
                raise ValueError(
                    'predictive-memory fusion requires private persistent memory'
                )
            memory_tokens = self.predictive_world_model_memory_in(
                _rms_normalize(persistent_memory)
            ) + self.predictive_world_model_memory_position(
                jnp.arange(persistent_memory.shape[1])
            )[None, :, :]
            memory_logits = jnp.einsum(
                'bhd,bmd->bhm',
                _rms_normalize(hidden + positions),
                _rms_normalize(memory_tokens),
                preferred_element_type=jnp.float32,
            ) / jnp.sqrt(float(memory_tokens.shape[-1]))
            memory_attention = jax.nn.softmax(
                memory_logits, axis=-1
            ).astype(memory_tokens.dtype)
            memory_context = jnp.einsum(
                'bhm,bmd->bhd', memory_attention, memory_tokens
            )
            hidden = hidden + nnx.swish(
                memory_context
            )
        if hasattr(self, 'predictive_world_model_program_in'):
            if persistent_program is None or persistent_frontier is None:
                raise ValueError(
                    'predictive-memory fusion requires program and frontier'
                )
            if (
                persistent_program.ndim != 3
                or persistent_frontier.ndim != 2
                or persistent_program.shape[:2]
                != persistent_frontier.shape
            ):
                raise ValueError(
                    'persistent program/frontier shapes are incompatible'
                )
            program_tokens = self.predictive_world_model_program_in(
                _rms_normalize(persistent_program)
            ) + self.predictive_world_model_program_position(
                jnp.arange(persistent_program.shape[1])
            )[None, :, :]
            current_program = jnp.einsum(
                'bs,bsd->bd',
                persistent_frontier.astype(program_tokens.dtype),
                program_tokens,
            )
            next_frontier = jnp.zeros_like(persistent_frontier)
            next_frontier = next_frontier.at[:, 1:].set(
                persistent_frontier[:, :-1]
            )
            next_frontier = next_frontier.at[:, -1].add(
                persistent_frontier[:, -1]
            )
            next_program = jnp.einsum(
                'bs,bsd->bd',
                next_frontier.astype(program_tokens.dtype),
                program_tokens,
            )
            phase_program_context = 0.75 * current_program + 0.25 * next_program
            hidden = hidden + nnx.swish(phase_program_context)[:, None, :]
        projected_branch_content = [
            self.predictive_world_model_future_in(latent_future_tokens),
            self.predictive_world_model_rollout_in(state_rollout_tokens),
            self.predictive_world_model_progress_in(task_progress_tokens),
        ]
        reliability_paths = [
            self.predictive_world_model_future_reliability(
                latent_future_tokens
            ),
            self.predictive_world_model_rollout_reliability(
                state_rollout_tokens
            ),
            self.predictive_world_model_progress_reliability(
                task_progress_tokens
            ),
        ]
        if self.predictive_world_model_include_action_moe:
            projected_branch_content.append(
                self.predictive_world_model_expert_in(action_moe_tokens)
            )
            reliability_paths.append(
                self.predictive_world_model_expert_reliability(
                    action_moe_tokens
                )
            )
        reliability_logits = jnp.concatenate(
            reliability_paths, axis=-1
        ).astype(jnp.float32)
        branch_content = sum(projected_branch_content)
        stacked_branch_content = jnp.stack(
            projected_branch_content, axis=-2
        )
        shared_context = hidden + positions
        content_logits = jnp.einsum(
            '...d,d->...',
            _rms_normalize(
                shared_context[..., None, :] + stacked_branch_content
            ),
            self.predictive_world_model_content_score.value,
        ).astype(jnp.float32)
        gate_logits = self.predictive_world_model_gate(
            _rms_normalize(shared_context + branch_content)
        ).astype(jnp.float32) + content_logits + reliability_logits
        gates = jax.nn.softmax(gate_logits, axis=-1)
        stacked = jnp.stack(tokens, axis=-2)
        fused = jnp.sum(
            stacked * gates[..., None].astype(stacked.dtype), axis=-2
        )
        return fused, gates, reliability_logits

    def predictive_world_model_reliability_objective(
        self,
        reliability_logits,
        *,
        latent_prediction,
        latent_target,
        latent_target_mask,
        rollout_prediction,
        rollout_target,
        rollout_target_mask,
        action_moe_prediction,
        actions,
        progress_logits,
        progress_expected,
        progress_target,
    ):
        """Supervise branch reliability on the same noisy flow state as routing.

        The branch tokens and ``reliability_logits`` passed here are produced
        from the current flow-matching ``x_t``.  Future observations and clean
        actions are used only to form stop-gradient quality labels; they never
        enter the routed policy representation.  This avoids training the gate
        on teacher-forced branch tokens while deploying it on noisy tokens.
        """
        latent_error = jnp.mean(
            jnp.square(
                latent_prediction.astype(jnp.float32)
                - latent_target.astype(jnp.float32)
            ),
            axis=-1,
        )
        latent_mask = latent_target_mask.astype(latent_error.dtype)
        latent_error = jnp.sum(latent_error * latent_mask, axis=(1, 2)) / (
            jnp.maximum(jnp.sum(latent_mask, axis=(1, 2)), 1.0)
        )
        latent_valid = jnp.sum(latent_mask, axis=(1, 2)) > 0

        rollout_error = jnp.mean(
            jnp.square(
                rollout_prediction.astype(jnp.float32)
                - rollout_target.astype(jnp.float32)
            ),
            axis=-1,
        )
        rollout_valid = rollout_target_mask.astype(jnp.bool_)
        if progress_target is None:
            raise ValueError(
                'predictive reliability training requires a progress target'
            )
        progress_target = progress_target.astype(jnp.float32)
        progress_valid = jnp.logical_and(
            jnp.isfinite(progress_target), progress_target >= 0
        )
        clipped_progress = jnp.where(
            progress_valid, jnp.clip(progress_target, 0.0, 1.0), 0.0
        )
        progress_distribution = _soft_progress_targets(
            clipped_progress, self.task_progress_bins
        )
        progress_error = -jnp.sum(
            progress_distribution
            * jax.nn.log_softmax(progress_logits.astype(jnp.float32), axis=-1),
            axis=-1,
        ) + 2.0 * jnp.square(
            progress_expected.astype(jnp.float32) - clipped_progress
        )

        horizon_shape = rollout_error.shape
        branch_errors = [
            jnp.broadcast_to(latent_error[:, None], horizon_shape),
            rollout_error,
            jnp.broadcast_to(progress_error[:, None], horizon_shape),
        ]
        if self.predictive_world_model_include_action_moe:
            if action_moe_prediction is None:
                raise ValueError(
                    'four-branch predictive reliability requires action-MoE'
                )
            branch_errors.append(
                self.active_action_mse(
                    action_moe_prediction.astype(jnp.float32),
                    actions.astype(jnp.float32),
                )
            )
        errors = jnp.stack(branch_errors, axis=-1).astype(jnp.float32)
        valid = (
            rollout_valid
            & latent_valid[:, None]
            & progress_valid[:, None]
        )
        valid_weight = valid.astype(errors.dtype)
        masked_errors = jnp.where(valid[..., None], errors, 0.0)
        branch_scales = jnp.sum(masked_errors, axis=(0, 1)) / jnp.maximum(
            jnp.sum(valid_weight), 1.0
        )
        branch_scales = jax.lax.stop_gradient(
            jnp.maximum(branch_scales, 1e-6)
        )
        reliability_targets = jax.nn.softmax(
            -jax.lax.stop_gradient(masked_errors / branch_scales), axis=-1
        )
        reliability_loss = -jnp.sum(
            reliability_targets
            * jax.nn.log_softmax(
                reliability_logits.astype(jnp.float32), axis=-1
            ),
            axis=-1,
        )
        return reliability_loss * valid_weight

    def evidence_combination_component_gates(self, contexts, state):
        """Return identity-initialized task/state gates for Stage-20 modules."""
        if not hasattr(self, 'evidence_combination_gate'):
            raise ValueError('evidence combination routing is disabled')
        pooled_context = jnp.mean(contexts, axis=1)
        shared = nnx.swish(
            self.evidence_combination_context_in(pooled_context)
            + self.evidence_combination_state_in(state)
        )[:, None, :]
        shared = shared + self.evidence_combination_action_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        logits = self.evidence_combination_gate(_rms_normalize(shared))
        # Exactly one at zero logits.  Unlike a softmax, each independently
        # positive component can remain fully active; the learned range [0, 2]
        # can suppress or strengthen it without forcing zero-sum competition.
        return 2.0 * jax.nn.sigmoid(logits.astype(jnp.float32))

    def evidence_combination_content_gate(
        self,
        contexts,
        state,
        component_tokens,
        component_index: int,
    ):
        """Route one component using both task context and its actual content."""
        if not hasattr(self, 'evidence_combination_content_score'):
            raise ValueError('content-aware evidence combination is disabled')
        if not 0 <= component_index < self.evidence_combination_component_count:
            raise ValueError('evidence combination component index is invalid')
        if component_tokens.shape[1] != self.action_horizon:
            raise ValueError(
                'evidence combination tokens must match the action horizon'
            )
        pooled_context = jnp.mean(contexts, axis=1)
        shared = nnx.swish(
            self.evidence_combination_context_in(pooled_context)
            + self.evidence_combination_state_in(state)
        )[:, None, :]
        shared = shared + self.evidence_combination_action_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        base_logits = self.evidence_combination_gate(
            _rms_normalize(shared)
        )[..., component_index]
        content_hidden = (
            shared
            + nnx.swish(
                self.evidence_combination_content_in(
                    _rms_normalize(component_tokens)
                )
            )
            + self.evidence_combination_component_embedding(
                jnp.asarray(component_index)
            )
        )
        content_logits = jnp.einsum(
            '...d,d->...',
            _rms_normalize(content_hidden),
            self.evidence_combination_content_score.value[component_index],
            preferred_element_type=jnp.float32,
        )
        # Both the shared gate and content scorer start at exact zero, so the
        # new content-aware route is exactly the identity at inheritance.
        return (
            2.0
            * jax.nn.sigmoid(
                base_logits.astype(jnp.float32) + content_logits
            )
        )[..., None]

    def compute_retrieved_demo_tokens(self, observation, contexts, *, train=False):
        """Encode local demonstrated actions and sparse whole-task structure."""
        if observation.demonstration_actions is None:
            raise ValueError('demonstration actions are required by this model')
        if observation.demonstration_plan is None:
            raise ValueError('demonstration plan is required by this model')
        if observation.demonstration_mask is None:
            raise ValueError('demonstration mask is required by this model')
        actions = observation.demonstration_actions
        if actions.shape[-2] != self.action_horizon:
            raise ValueError(
                'demonstration action horizon must equal the model action horizon'
            )
        plan = observation.demonstration_plan
        plan_steps = self.action_prior_demo_plan_steps
        if plan.shape[-2] != plan_steps:
            raise ValueError(
                'demonstration plan length must equal retrieved_demo_plan_steps'
            )
        compositional = hasattr(self, 'action_prior_demo_router_blocks')
        if compositional:
            if actions.ndim != 4 or plan.ndim != 4:
                raise ValueError(
                    'compositional demonstrations require candidate axes'
                )
            if actions.shape[1] != self.action_prior_demo_slots:
                raise ValueError(
                    'demonstration candidate count must equal compositional slots'
                )
            if observation.demonstration_slot_mask is None:
                raise ValueError(
                    'compositional demonstrations require a slot mask'
                )
            if observation.demonstration_progress is None:
                raise ValueError(
                    'compositional demonstrations require progress estimates'
                )
            batch_size, candidate_count = actions.shape[:2]
            actions = actions.reshape(
                batch_size * candidate_count, self.action_horizon, actions.shape[-1]
            )
            plan = plan.reshape(
                batch_size * candidate_count, plan_steps, plan.shape[-1]
            )
            encoded_context = self.action_prior_demo_context_in(contexts)
            context_tokens = jnp.repeat(encoded_context, candidate_count, axis=0)
        else:
            batch_size, candidate_count = actions.shape[0], 1
            context_tokens = self.action_prior_demo_context_in(contexts)

        tokens = self.action_prior_demo_action_in(actions)
        tokens = tokens + self.action_prior_demo_position(
            jnp.arange(self.action_horizon)
        )[None, :, :]
        plan_tokens = self.action_prior_demo_plan_in(plan)
        plan_tokens = plan_tokens + self.action_prior_demo_plan_position(
            jnp.arange(plan_steps)
        )[None, :, :]
        # Local action queries can self-attend to all whole-episode keyframes,
        # while only the local positions are projected into additive action
        # tokens. This preserves the inherited action-horizon interface.
        tokens = jnp.concatenate([tokens, plan_tokens], axis=1)
        for block in self.action_prior_demo_blocks:
            tokens = block(tokens, context_tokens)
        reliability = None
        if hasattr(self, 'action_prior_demo_gate_out'):
            reliability = self.compute_retrieved_demo_reliability(
                tokens, context_tokens
            )
        local_tokens = self.action_prior_demo_token_out(
            _rms_normalize(tokens[:, : self.action_horizon])
        )
        if not compositional:
            if reliability is not None:
                local_tokens = local_tokens * reliability[:, None, :]
            return (
                local_tokens * observation.demonstration_mask[:, None, None],
                reliability,
                None,
            )

        router_tokens = jnp.mean(tokens, axis=1).reshape(
            batch_size, candidate_count, -1
        )
        router_tokens = router_tokens + self.action_prior_demo_slot_position(
            jnp.arange(candidate_count)
        )[None, :, :]
        router_tokens = router_tokens + self.action_prior_demo_progress_in(
            observation.demonstration_progress[..., None]
        )
        router_context = self.action_prior_demo_context_in(contexts)
        for block in self.action_prior_demo_router_blocks:
            router_tokens = block(router_tokens, router_context)
        router_logits = self.action_prior_demo_router_out(
            _rms_normalize(router_tokens)
        )[..., 0]
        router_logits = (
            router_logits + self.action_prior_demo_router_order_bias.value[None, :]
        )
        router_logits, soft_router_weights = _masked_demo_router_weights(
            router_logits,
            observation.demonstration_slot_mask,
            temperature=self.action_prior_demo_router_temperature,
        )
        router_weights = _hard_demo_router_weights(
            router_logits, soft_router_weights, train=train
        )
        router_weights = router_weights.astype(local_tokens.dtype)
        local_tokens = local_tokens.reshape(
            batch_size,
            candidate_count,
            self.action_horizon,
            local_tokens.shape[-1],
        )
        if reliability is not None:
            reliability = reliability.reshape(batch_size, candidate_count, 1)
            local_tokens = local_tokens * reliability[:, :, None, :]
            aggregate_reliability = jnp.sum(
                router_weights[:, :, None] * reliability, axis=1
            )
        else:
            aggregate_reliability = None
        mixed_tokens = jnp.einsum(
            'bk,bkhd->bhd', router_weights, local_tokens
        )
        return (
            mixed_tokens * observation.demonstration_mask[:, None, None],
            aggregate_reliability,
            router_logits,
        )

    def compute_retrieved_demo_reliability(self, demo_tokens, context_tokens):
        """Return a bounded learned trust weight for the retrieved trajectory."""
        features = jnp.concatenate(
            [jnp.mean(demo_tokens, axis=1), jnp.mean(context_tokens, axis=1)],
            axis=-1,
        )
        hidden = nnx.swish(self.action_prior_demo_gate_in(features))
        logits = self.action_prior_demo_gate_out(hidden)
        # Twice a sigmoid gives an exact identity multiplier at zero init while
        # retaining a smooth [0, 2] range for suppressing or amplifying context.
        return 2.0 * jax.nn.sigmoid(logits)

    def compute_discrete_action_code_tokens(self, contexts, *, train):
        """Decode hierarchical global/local codes into action-expert tokens."""
        hidden = contexts.reshape(contexts.shape[0], -1)
        hidden = nnx.swish(self.action_prior_discrete_context_in(hidden))
        hidden = nnx.swish(self.action_prior_discrete_context_out(hidden))
        logits = self.action_prior_discrete_logits(hidden)
        probabilities = jax.nn.softmax(
            logits.astype(jnp.float32)
            / self.action_prior_discrete_temperature,
            axis=-1,
        )
        hard_codes = jax.nn.one_hot(
            jnp.argmax(probabilities, axis=-1), probabilities.shape[-1]
        )
        # Use the selected discrete mode in the forward pass, while the
        # straight-through probabilities let continuous flow loss refine the
        # classifier in addition to its explicit cross-entropy objective.
        code_weights = (
            probabilities
            + jax.lax.stop_gradient(hard_codes - probabilities)
            if train
            else hard_codes
        )
        decoded_actions = jnp.einsum(
            'bk,khd->bhd',
            code_weights,
            self.action_prior_discrete_codebook.value,
        )
        global_tokens = nnx.swish(
            self.action_prior_discrete_action_in(decoded_actions)
        )
        global_tokens = self.action_prior_discrete_token_out(global_tokens)

        step_tokens = self.action_prior_discrete_step_queries(
            jnp.arange(self.action_horizon)
        )
        step_tokens = jnp.broadcast_to(
            step_tokens[None, :, :],
            (contexts.shape[0], *step_tokens.shape),
        )
        step_contexts = self.action_prior_discrete_step_context_in(contexts)
        for block in self.action_prior_discrete_step_blocks:
            step_tokens = block(step_tokens, step_contexts)
        step_logits = self.action_prior_discrete_step_logits(
            _rms_normalize(step_tokens)
        )
        step_probabilities = jax.nn.softmax(
            step_logits.astype(jnp.float32)
            / self.action_prior_discrete_temperature,
            axis=-1,
        )
        hard_step_codes = jax.nn.one_hot(
            jnp.argmax(step_probabilities, axis=-1),
            step_probabilities.shape[-1],
        )
        step_weights = (
            step_probabilities
            + jax.lax.stop_gradient(hard_step_codes - step_probabilities)
            if train
            else hard_step_codes
        )
        decoded_steps = jnp.einsum(
            'bhk,kd->bhd',
            step_weights,
            self.action_prior_discrete_codebook_steps.value,
        )
        local_tokens = nnx.swish(
            self.action_prior_discrete_step_action_in(decoded_steps)
        )
        local_tokens = self.action_prior_discrete_step_token_out(local_tokens)
        return global_tokens + local_tokens, logits, step_logits

    def discrete_action_code_targets(self, actions):
        """Assign normalized ground-truth chunks to their nearest fixed code."""
        robot_dim = self.action_prior_discrete_robot_dim
        differences = (
            actions[:, None, :, :robot_dim]
            - self.action_prior_discrete_codebook.value[
                None, :, :, :robot_dim
            ]
        )
        distances = jnp.mean(jnp.square(differences), axis=(-1, -2))
        return jnp.argmin(distances, axis=-1)

    def discrete_action_step_code_targets(self, actions):
        """Assign every action position to its nearest local motion token."""
        robot_dim = self.action_prior_discrete_robot_dim
        differences = (
            actions[:, :, None, :robot_dim]
            - self.action_prior_discrete_codebook_steps.value[
                None, None, :, :robot_dim
            ]
        )
        distances = jnp.mean(jnp.square(differences), axis=-1)
        return jnp.argmin(distances, axis=-1)

    def coarse_action_targets(self, actions):
        """Reduce a dense action chunk to supervised coarse waypoints."""
        repeat = self.action_horizon // self.action_prior_horizon
        action_segments = actions.reshape(
            *actions.shape[:-2],
            self.action_prior_horizon,
            repeat,
            actions.shape[-1],
        )
        if self.action_prior_target == 'endpoint':
            return action_segments[..., -1, :]
        return jnp.mean(action_segments, axis=-2)

    def mask_inactive_actions(self, actions):
        """Keep physical action coordinates and zero model-padding coordinates."""
        mask = jnp.arange(self.action_dim) < self.active_action_dim
        return actions * mask.astype(actions.dtype)

    def active_action_mse(self, prediction, target):
        """MSE normalized by physical, rather than padded, action dimensions."""
        difference = self.mask_inactive_actions(prediction - target)
        return jnp.sum(jnp.square(difference), axis=-1) / self.active_action_dim

    def _encode_geometry_once(self, contextualized_prefix, prefix_mask, observation):
        persistent = getattr(self, 'persistent_memory', None)
        module = getattr(persistent, 'geometry_aux_v3', None)
        if module is None:
            return None
        visual_prefix, visual_mask = _geometry_plumbing.visual_prefix_only(
            contextualized_prefix, prefix_mask, observation
        )
        # Deliberately no stop_gradient: the auxiliary reaches trainable
        # PaliGemma LoRA through this existing contextualized-prefix tape.
        external = getattr(
            persistent, 'geometry_external_residual_v1', None
        )
        if external is not None:
            return external.encode_once(module, visual_prefix, visual_mask)
        return module.encode_once(visual_prefix, visual_mask)

    def _inject_geometry_after_inherited_parent(
        self, inherited_actions, geometry_context, *, policy_scale=1.0
    ):
        persistent = getattr(self, 'persistent_memory', None)
        module = getattr(persistent, 'geometry_aux_v3', None)
        if module is None or geometry_context is None:
            return inherited_actions
        _, tokens, token_mask, enabled = geometry_context
        # Re-read the complete inherited PSM + native-v3 + SDLA output.  This
        # helper is shared verbatim by training flow samples and sampling ODE.
        parent_query = persistent.read_query(
            _rms_normalize(inherited_actions)
        )
        return module.inject_after_parent(
            inherited_actions,
            parent_query,
            tokens,
            token_mask,
            enabled,
            policy_scale=policy_scale,
        )[0]


    def _hmca_adapter_from_persistent_state(
        self,
        persistent_private_state,
        hetm_private_state=None,
        racg_private_state=None,
    ):
        module = getattr(self, 'hierarchical_memory_conditional_adapters', None)
        if module is None or persistent_private_state is None:
            return None
        bridge = getattr(
            self.persistent_memory, 'conditional_memory_policy_bridge', None
        )
        if bridge is None:
            raise RuntimeError('HMCA-v4 requires the native v3 semantic bridge')
        features, enabled = bridge._semantic_features(
            persistent_private_state['memory'],
            persistent_private_state['ordered_subgoals'],
            persistent_private_state['frontier'],
        )
        if hetm_private_state is not None:
            residual = hetm_private_state['outputs'].get(
                'hmca_condition_residual'
            )
            if residual is None or residual.shape != features.shape:
                raise ValueError('HETM-to-HMCA semantic residual shape drifted')
            features = features + residual.astype(features.dtype)
        geometry_hmca = getattr(self, 'racg_external_geometry_hmca', None)
        if geometry_hmca is not None:
            if racg_private_state is None:
                # The helper is called once before contextual prefix/RACG
                # construction and then replaced by the complete payload.
                return module.adapter_payload(features, enabled)
            external = racg_private_state.get('external_geometry')
            if external is None:
                raise ValueError(
                    'geometry-HMCA requires RACG external geometry state'
                )
            geometry_residual, geometry_enabled = geometry_hmca(
                external.external_role_features,
                external.external_role_mask,
            )
            if geometry_residual.shape != features.shape:
                raise ValueError('geometry-HMCA semantic residual shape drifted')
            features = _racg_external_hmca.exact_zero_geometry_add(
                features, geometry_residual.astype(features.dtype)
            )
            # Keep the inherited HMCA availability contract. Missing external
            # evidence produces an exact-zero residual rather than disabling
            # a valid persistent-memory adapter.
            del geometry_enabled
        graph_hmca = getattr(self, 'racg_graph_hmca', None)
        if graph_hmca is not None:
            if racg_private_state is None:
                return module.adapter_payload(features, enabled)
            scene = racg_private_state.get('scene')
            if scene is None:
                raise ValueError('graph-HMCA requires an encoded RACG scene')
            graph_residual, graph_enabled = graph_hmca(
                scene.graph_tokens,
                scene.graph_token_mask,
            )
            if graph_residual.shape != features.shape:
                raise ValueError('graph-HMCA semantic residual shape drifted')
            features = _racg_graph_hmca.exact_zero_graph_add(
                features, graph_residual.astype(features.dtype)
            )
            del graph_enabled
        return module.adapter_payload(features, enabled)

    def _memory_attention_adapter_from_persistent_state(
        self, persistent_private_state
    ):
        """Build token-preserving causal memory reads for the Gemma layer scan."""
        module = getattr(self, 'layerwise_persistent_memory_attention', None)
        if module is None or persistent_private_state is None:
            return None
        memory = persistent_private_state['memory']
        ordered_program = persistent_private_state['ordered_subgoals']
        frontier = persistent_private_state['frontier']
        slot_valid_mask = persistent_private_state.get('slot_valid_mask')
        if slot_valid_mask is None:
            slot_valid_mask = jnp.ones(frontier.shape, dtype=jnp.bool_)
        if memory.ndim != 3 or ordered_program.shape != memory.shape:
            raise ValueError('layerwise memory attention PSM/program shape drifted')
        if frontier.shape != memory.shape[:2]:
            raise ValueError('layerwise memory attention frontier shape drifted')
        if slot_valid_mask.shape != frontier.shape:
            raise ValueError('layerwise memory attention validity shape drifted')

        # The inherited PSM read assigns every recurrent slot a learned
        # physical identity through memory_position (fast, target/reference,
        # verification, ...).  Cross-attention over a bare token set is
        # permutation invariant, so dropping those codes here would force the
        # new layerwise reader to rediscover slot responsibility from content
        # alone.  Normalize content before adding the already-trained codes so
        # raw memory magnitude cannot drown out (or amplify) slot identity.
        persistent_memory = getattr(self, 'persistent_memory', None)
        if persistent_memory is None or not hasattr(
            persistent_memory, 'memory_position'
        ):
            raise ValueError(
                'layerwise memory attention requires PSM memory-position identity'
            )
        memory_positions = persistent_memory.memory_position(
            jnp.arange(memory.shape[1])
        )
        if memory_positions.shape != memory.shape[1:]:
            raise ValueError('layerwise memory position shape drifted')
        structured_memory = (
            _rms_normalize(memory)
            + memory_positions[None, :, :].astype(memory.dtype)
        )

        valid_frontier = frontier.astype(jnp.float32) * slot_valid_mask.astype(
            jnp.float32
        )
        valid_frontier_mass = jnp.sum(
            valid_frontier, axis=-1, keepdims=True
        )
        valid_frontier = valid_frontier / jnp.maximum(
            valid_frontier_mass, 1.0e-8
        )

        current_program = jnp.einsum(
            'bs,bsh->bh',
            valid_frontier.astype(ordered_program.dtype),
            ordered_program,
        )
        slot_count = ordered_program.shape[1]
        phase_ids = jnp.arange(slot_count, dtype=jnp.float32)
        phase_distance = phase_ids[None, :] - phase_ids[:, None]
        remaining_kernel = jnp.where(
            phase_distance >= 0.0, jnp.power(0.75, phase_distance), 0.0
        )
        remaining_weights = jnp.einsum(
            'bs,sr->br', valid_frontier, remaining_kernel
        )
        remaining_weights = remaining_weights * slot_valid_mask.astype(
            remaining_weights.dtype
        )
        remaining_mass = jnp.sum(
            remaining_weights, axis=-1, keepdims=True
        )
        remaining_weights = remaining_weights / jnp.maximum(
            remaining_mass, 1.0e-8
        )
        remaining_program = jnp.einsum(
            'bs,bsh->bh',
            remaining_weights.astype(ordered_program.dtype),
            ordered_program,
        )
        memory_tokens = jnp.concatenate(
            [
                structured_memory,
                current_program[:, None, :],
                remaining_program[:, None, :],
            ],
            axis=1,
        )
        # The action query is RMS-normalized inside Gemma.  Normalize every
        # memory token on the matching boundary so slot magnitude cannot act
        # as an accidental attention temperature or residual gain.
        memory_tokens = _rms_normalize(memory_tokens).astype(jnp.float32)
        memory_mask = jnp.ones(memory.shape[:2], dtype=jnp.bool_)
        current_valid = valid_frontier_mass[:, 0] > 1.0e-8
        remaining_valid = remaining_mass[:, 0] > 1.0e-8
        token_mask = jnp.concatenate(
            [
                memory_mask,
                current_valid[:, None],
                remaining_valid[:, None],
            ],
            axis=1,
        )
        enabled = (
            jnp.all(jnp.isfinite(memory_tokens), axis=(1, 2))
            & current_valid
            & remaining_valid
        )
        return module.adapter_payload(memory_tokens, token_mask, enabled)

    def _geometry_auxiliary_loss(
        self, geometry_context, observation, *, training_step
    ):
        if geometry_context is None or training_step is None:
            return jnp.asarray(0.0, jnp.float32)
        target_bundle = _geometry_plumbing.auxiliary_targets(observation)
        if target_bundle is None:
            return jnp.asarray(0.0, jnp.float32)
        predictions = geometry_context[0]
        targets, supervision_mask = target_bundle
        return self.persistent_memory.geometry_aux_v3.auxiliary_loss(
            predictions,
            targets,
            supervision_mask,
            training_step=training_step,
        )['weighted_total']

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        persistent_policy_scale: float | at.Float[at.Array, ''] = 1.0,
        _preprocessed_observation=None,
        _precomputed_prefix=None,
        _precomputed_contextual_prefix=None,
        _precomputed_persistent_state=None,
        _precomputed_hetm_state=None,
        _precomputed_racg_state=None,
        _geometry_training_step=None,
        _include_racg_scene_auxiliary: bool = True,
    ) -> at.Float[at.Array, '*b ah']:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = (
            _model.preprocess_observation(
                preprocess_rng, observation, train=train
            )
            if _preprocessed_observation is None
            else _preprocessed_observation
        )

        batch_shape = actions.shape[:-2]
        actions = self.mask_inactive_actions(actions)
        noise = self.mask_inactive_actions(
            jax.random.normal(noise_rng, actions.shape)
        )
        time = _sample_beta_1p5_1(time_rng, batch_shape) * 0.999 + 0.001

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = (
            self.embed_prefix(observation)
            if _precomputed_prefix is None
            else _precomputed_prefix
        )
        persistent_private_state = (
            self._persistent_state_from_observation(
                observation,
                prefix_tokens,
                prefix_mask,
                compute_object_reconstruction=train,
            )
            if _precomputed_persistent_state is None
            else _precomputed_persistent_state
        )
        hmca_adapter = self._hmca_adapter_from_persistent_state(
            persistent_private_state
        )
        memory_layer_adapter = self._memory_attention_adapter_from_persistent_state(
            persistent_private_state
        )
        prefix_moe_balance_loss = None
        kv_moe_balance_loss = None
        if hasattr(self, 'prefix_moe_router_in'):
            prefix_tokens, prefix_moe_balance_loss, _ = (
                self.apply_multimodal_prefix_moe(
                    prefix_tokens, prefix_mask, observation
                )
            )
        action_prior_tokens = None
        coarse_actions = None
        explicit_velocity = None
        explicit_velocity_target = None
        discrete_logits = None
        discrete_step_logits = None
        demo_reliability = None
        demo_router_logits = None
        spatial_relation_actions = None
        object_affordance_actions = None
        object_affordance_slots = None
        object_affordance_assignments = None
        object_affordance_reconstruction_loss = None
        masked_spatial_actions = None
        masked_spatial_reconstruction = None
        masked_spatial_target = None
        masked_spatial_reconstruction_mask = None
        object_future_actions = None
        object_future_reconstruction = None
        object_future_target = None
        object_future_target_mask = None
        object_future_forecast_tokens = None
        predicate_binding_actions = None
        predicate_binding_contrastive_logits = None
        predicate_binding_positive_mask = None
        contact_phase_tokens = None
        phase_contact_condition_tokens = None
        contact_phase_logits = None
        contact_affordance_risk_logits = None
        contact_affordance_relation_alignment_logits = None
        structured_rationale_logits = None
        action_chunk_verifier_logits = None
        latent_future_current_visual = None
        latent_future_prediction = None
        latent_future_target = None
        latent_future_target_mask = None
        latent_future_clean_residual = None
        state_rollout_prediction = None
        state_rollout_target = None
        state_rollout_target_mask = None
        state_rollout_clean_residual = None
        action_moe_clean_residual = None
        action_moe_prediction = None
        action_moe_balance_loss = None
        specialist_module_router_balance_loss = None
        task_progress_tokens = None
        task_progress_logits = None
        task_progress_expected = None
        language_subgoal_prediction = None
        language_subgoal_logits = None
        language_subgoal_expected = None
        language_subgoal_probabilities = None
        language_subgoal_slots = None
        object_subgoal_binding_prediction = None
        kinematic_action_clean_residual = None
        kinematic_action_prediction = None
        spectral_action_prediction = None
        prefix_cache = None
        action_reasoning_tokens = None
        contexts = None
        geometry_context = None
        geometry_auxiliary_loss = jnp.asarray(0.0, jnp.float32)
        hetm_private_state = None
        racg_private_state = None
        if hasattr(self, 'action_prior_queries') and self.action_prior_contextual:
            if _precomputed_contextual_prefix is None:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
                prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
                (prefix_out, _), prefix_cache = self.PaliGemma.llm(
                    [prefix_tokens, None],
                    mask=prefix_attn_mask,
                    positions=prefix_positions,
                )
            else:
                prefix_out, prefix_cache = _precomputed_contextual_prefix
            geometry_context = self._encode_geometry_once(
                prefix_out, prefix_mask, observation
            )
            geometry_auxiliary_loss = self._geometry_auxiliary_loss(
                geometry_context,
                observation,
                training_step=_geometry_training_step,
            )
            if hasattr(self, 'kv_moe_router_in'):
                prefix_cache, kv_moe_balance_loss, _ = (
                    self.apply_layerwise_kv_moe(
                        prefix_cache,
                        prefix_tokens,
                        prefix_mask,
                        observation,
                    )
                )
            # The auxiliary reasoners train their compact projections without
            # retaining a second full VLM backward graph. The normal action
            # flow still adapts enabled VLM LoRA weights through prefix_cache.
            prior_source = jax.lax.stop_gradient(prefix_out)
            hetm_private_state = (
                self._hetm_state_from_observation(
                    observation, prior_source, prefix_mask
                )
                if _precomputed_hetm_state is None
                else _precomputed_hetm_state
            )
            racg_private_state = (
                self._racg_state_from_observation(
                    observation,
                    prior_source,
                    prefix_mask,
                    hetm_private_state,
                )
                if _precomputed_racg_state is None
                else _precomputed_racg_state
            )
            hmca_adapter = self._hmca_adapter_from_persistent_state(
                persistent_private_state, hetm_private_state, racg_private_state
            )
            if hasattr(self, 'action_prior_explicit_blocks'):
                contexts = self.compute_action_prior_contexts(
                    prior_source,
                    prefix_mask,
                    observation.state,
                    (
                        persistent_private_state['memory']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        persistent_private_state['ordered_subgoals']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        persistent_private_state['frontier']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        persistent_private_state['slot_valid_mask']
                        if persistent_private_state is not None
                        else None
                    ),
                )
                stopped_cache = jax.tree.map(jax.lax.stop_gradient, prefix_cache)
                contexts, layerwise_implicit_guidance = (
                    self.compute_multilayer_action_prior_contexts(
                        contexts,
                        stopped_cache,
                        prefix_mask,
                        observation.state,
                        (
                            persistent_private_state['memory']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['ordered_subgoals']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['frontier']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['slot_valid_mask']
                            if persistent_private_state is not None
                            else None
                        ),
                        return_layer_guidance=True,
                    )
                )
                coarse_targets = self.coarse_action_targets(actions)
                main_coarse_noise = self.coarse_action_targets(noise)
                flow_samples = self.action_prior_explicit_flow_samples
                explicit_noises = main_coarse_noise[:, None, :, :]
                explicit_times = time[:, None]
                if flow_samples > 1:
                    explicit_noise_rng, explicit_time_rng = jax.random.split(
                        jax.random.fold_in(rng, 37)
                    )
                    extra_shape = (
                        coarse_targets.shape[0],
                        flow_samples - 1,
                        *coarse_targets.shape[1:],
                    )
                    extra_noises = jax.random.normal(
                        explicit_noise_rng, extra_shape
                    )
                    extra_times = (
                        _sample_beta_1p5_1(
                            explicit_time_rng,
                            (
                                coarse_targets.shape[0],
                                flow_samples - 1,
                            ),
                        )
                        * 0.999
                        + 0.001
                    )
                    explicit_noises = jnp.concatenate(
                        [explicit_noises, extra_noises], axis=1
                    )
                    explicit_times = jnp.concatenate(
                        [explicit_times, extra_times], axis=1
                    )
                explicit_noises = self.mask_inactive_actions(explicit_noises)
                target_samples = coarse_targets[:, None, :, :]
                explicit_time_expanded = explicit_times[..., None, None]
                explicit_noisy = (
                    explicit_time_expanded * explicit_noises
                    + (1 - explicit_time_expanded) * target_samples
                )
                explicit_velocity_target = explicit_noises - target_samples
                context_samples = jnp.broadcast_to(
                    contexts[:, None, :, :],
                    (
                        contexts.shape[0],
                        flow_samples,
                        *contexts.shape[1:],
                    ),
                )
                explicit_velocity = self.compute_explicit_action_velocity(
                    explicit_noisy.reshape(
                        -1, *explicit_noisy.shape[2:]
                    ),
                    explicit_times.reshape(-1),
                    context_samples.reshape(
                        -1, *context_samples.shape[2:]
                    ),
                ).reshape(explicit_noisy.shape)
                if self.action_prior_explicit_teacher_forcing:
                    # Keep the main flow objective stable while the auxiliary
                    # EAR is still learning its coarse trajectory. Sampling
                    # uses the EAR prediction because targets are unavailable.
                    reference_waypoints = coarse_targets
                else:
                    # Later progressive stages train the main flow on the
                    # exact configured reference distribution it receives at
                    # inference. Keep the explicit flow objective separate so
                    # this exposure correction does not duplicate its backward
                    # graph through every main-flow sample.
                    reference_waypoints = self.compute_detached_explicit_reference(
                        main_coarse_noise,
                        contexts,
                    )
                pathway_tokens = None
                if hasattr(self, 'action_prior_pathway_score') or hasattr(
                    self, 'action_prior_pathway_interaction_out'
                ):
                    (
                        implicit_tokens,
                        explicit_tokens,
                        coarse_actions,
                        action_reasoning_tokens,
                    ) = self.dual_action_prior_token_paths(
                        contexts, reference_waypoints
                    )
                    pathway_tokens = [implicit_tokens, explicit_tokens]
                    action_prior_tokens = None
                else:
                    (
                        action_prior_tokens,
                        coarse_actions,
                        action_reasoning_tokens,
                    ) = self.fuse_dual_action_prior_tokens(
                        contexts, reference_waypoints
                    )
                if layerwise_implicit_guidance is not None:
                    action_reasoning_tokens = jnp.concatenate(
                        [action_reasoning_tokens, layerwise_implicit_guidance],
                        axis=1,
                    )
                if hasattr(self, 'action_prior_demo_blocks'):
                    (
                        demo_tokens,
                        demo_reliability,
                        demo_router_logits,
                    ) = self.compute_retrieved_demo_tokens(
                        observation, contexts, train=train
                    )
                    if pathway_tokens is None:
                        action_prior_tokens = action_prior_tokens + demo_tokens
                    else:
                        pathway_tokens.append(demo_tokens)
                    action_reasoning_tokens = jnp.concatenate(
                        [action_reasoning_tokens, demo_tokens], axis=1
                    )
                if hasattr(self, 'action_prior_discrete_codebook'):
                    discrete_tokens, discrete_logits, discrete_step_logits = (
                        self.compute_discrete_action_code_tokens(
                            contexts, train=train
                        )
                    )
                    if pathway_tokens is None:
                        action_prior_tokens = action_prior_tokens + discrete_tokens
                    else:
                        pathway_tokens.append(discrete_tokens)
                    action_reasoning_tokens = jnp.concatenate(
                        [action_reasoning_tokens, discrete_tokens], axis=1
                    )
                if pathway_tokens is not None:
                    if hasattr(self, 'action_prior_pathway_score'):
                        action_prior_tokens, _ = self.route_reasoning_pathway_tokens(
                            pathway_tokens, contexts, observation.state
                        )
                    if hasattr(self, 'action_prior_pathway_interaction_out'):
                        interacted_tokens, interaction_residual_paths = (
                            self.interact_reasoning_pathway_tokens(
                                pathway_tokens, contexts, observation.state
                            )
                        )
                        interaction_residual = jnp.mean(
                            interaction_residual_paths, axis=2
                        )
                        if hasattr(self, 'action_prior_pathway_score'):
                            # Hybrid Stage-20 fusion: route the inherited base
                            # paths, then add the independent cross-path/time
                            # residual.  Both operations are exact identities
                            # at initialization (uniform weights and a zero
                            # residual head), preserving the Stage-19 policy.
                            action_prior_tokens = (
                                action_prior_tokens + interaction_residual
                            )
                        else:
                            action_prior_tokens = interacted_tokens
                        if hasattr(self, 'evidence_combination_gate'):
                            interaction_gate = (
                                self.evidence_combination_content_gate(
                                    contexts,
                                    observation.state,
                                    interaction_residual,
                                    2,
                                )
                            )
                            action_prior_tokens = action_prior_tokens + (
                                interaction_gate - 1.0
                            ).astype(interaction_residual.dtype) * interaction_residual
                if hasattr(self, 'spatial_relation_blocks'):
                    spatial_tokens, spatial_relation_actions = (
                        self.compute_spatial_relation_tokens(
                            prefix_tokens,
                            prefix_mask,
                            observation,
                            prior_source,
                        )
                    )
                    if hasattr(self, 'evidence_combination_gate'):
                        spatial_gate = self.evidence_combination_content_gate(
                            contexts,
                            observation.state,
                            spatial_tokens,
                            0,
                        )
                        spatial_tokens = spatial_tokens * (
                            spatial_gate.astype(spatial_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + spatial_tokens
                if hasattr(self, 'object_affordance_graph_blocks'):
                    (
                        object_affordance_tokens,
                        object_affordance_actions,
                        object_affordance_assignments,
                        object_affordance_slots,
                        object_affordance_reconstruction_loss,
                    ) = self.compute_object_affordance_graph_tokens(
                        prefix_tokens,
                        prefix_mask,
                        observation,
                        prior_source,
                        compute_reconstruction=(
                            train
                            and self.object_affordance_reconstruction_loss_weight
                            > 0.0
                        ),
                        routing_contexts=contexts,
                    )
                    if (
                        hasattr(self, 'evidence_combination_gate')
                        and self.evidence_combination_component_count > 4
                    ):
                        object_affordance_tokens = object_affordance_tokens * (
                            self.evidence_combination_content_gate(
                                contexts,
                                observation.state,
                                object_affordance_tokens,
                                4,
                            ).astype(object_affordance_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + object_affordance_tokens
                if hasattr(self, 'masked_spatial_scene_blocks'):
                    (
                        masked_spatial_tokens,
                        masked_spatial_actions,
                        masked_spatial_reconstruction,
                        masked_spatial_target,
                        masked_spatial_reconstruction_mask,
                        _,
                    ) = self.compute_masked_spatial_tokens(
                        prefix_tokens,
                        prefix_mask,
                        observation,
                        prefix_out,
                        train=train,
                        rng=jax.random.fold_in(rng, 2401),
                    )
                    action_prior_tokens = (
                        action_prior_tokens + masked_spatial_tokens
                    )
                if hasattr(self, 'object_future_object_blocks'):
                    (
                        object_future_tokens,
                        object_future_actions,
                        object_future_reconstruction,
                        object_future_target,
                        object_future_target_mask,
                        _,
                        object_future_forecast_tokens,
                    ) = self.compute_object_future_tokens(
                        prefix_tokens,
                        prefix_mask,
                        observation,
                        prefix_out,
                        contexts,
                        train=train,
                        object_affordance_slots=object_affordance_slots,
                    )
                    action_prior_tokens = (
                        action_prior_tokens + object_future_tokens
                    )
                if hasattr(self, 'predicate_binding_object_blocks'):
                    (
                        predicate_binding_tokens,
                        predicate_binding_actions,
                        predicate_binding_contrastive_logits,
                        predicate_binding_positive_mask,
                        _,
                        _,
                        _,
                    ) = self.compute_predicate_binding_tokens(
                        prefix_tokens,
                        prefix_mask,
                        observation,
                        prefix_out,
                        contexts,
                    )
                    action_prior_tokens = (
                        action_prior_tokens + predicate_binding_tokens
                    )
                if hasattr(self, 'contact_phase_blocks'):
                    contact_phase_tokens, contact_phase_logits = (
                        self.compute_contact_phase_tokens(
                            contexts, observation.state, train=train
                        )
                    )
                    if hasattr(self, 'evidence_combination_gate'):
                        contact_gate = self.evidence_combination_content_gate(
                            contexts,
                            observation.state,
                            contact_phase_tokens,
                            1,
                        )
                        contact_phase_tokens = contact_phase_tokens * (
                            contact_gate.astype(contact_phase_tokens.dtype)
                        )
                    phase_contact_condition_tokens = contact_phase_tokens
                    action_prior_tokens = action_prior_tokens + contact_phase_tokens
                    if hasattr(self, 'contact_affordance_blocks'):
                        if object_affordance_slots is None:
                            raise ValueError(
                                'contact-affordance fusion requires object slots'
                            )
                        (
                            contact_affordance_tokens,
                            contact_affordance_risk_logits,
                            contact_affordance_relation_alignment_logits,
                        ) = self.compute_contact_affordance_predictive_tokens(
                            object_affordance_slots,
                            contact_phase_tokens,
                            contact_phase_logits,
                            contexts,
                            observation.state,
                            (
                                persistent_private_state['memory']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['ordered_subgoals']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['frontier']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['clause_plan_attention']
                                if persistent_private_state is not None
                                else None
                            ),
                            object_future_forecast_tokens,
                            (
                                persistent_private_state['factorized_relation_state']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['grounded_relation_phase_state']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['bound_roles']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state[
                                    'verification_transition_probabilities'
                                ]
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['progress']
                                if persistent_private_state is not None
                                else None
                            ),
                            (
                                persistent_private_state['verification_state']
                                if persistent_private_state is not None
                                else None
                            ),
                        )
                        action_prior_tokens = (
                            action_prior_tokens + contact_affordance_tokens
                        )
                        # The multiplicative Phase-Contact path should consume
                        # the same contact representation that has already
                        # fused object ambiguity, gripper state, verified
                        # current/next program state, relation grounding, and
                        # the explicitly supervised risk logit.  The parent
                        # policy keeps the two additive residuals unchanged.
                        phase_contact_condition_tokens = (
                            contact_phase_tokens + contact_affordance_tokens
                        )
                if hasattr(self, 'action_prior_rationale_blocks'):
                    (
                        rationale_tokens,
                        structured_rationale_logits,
                    ) = self.compute_structured_rationale_tokens(
                        prefix_out,
                        prefix_mask,
                        observation.state,
                        train=train,
                    )
                    action_prior_tokens = action_prior_tokens + rationale_tokens
                if hasattr(self, 'task_progress_blocks'):
                    (
                        progress_tokens,
                        task_progress_logits,
                        task_progress_expected,
                        _,
                    ) = self.compute_task_progress_tokens(
                        contexts, observation.state
                    )
                    task_progress_tokens = progress_tokens
                    if not hasattr(self, 'predictive_world_model_gate'):
                        action_prior_tokens = (
                            action_prior_tokens + progress_tokens
                        )
                if hasattr(self, 'language_subgoal_blocks'):
                    (
                        subgoal_tokens,
                        language_subgoal_prediction,
                        language_subgoal_logits,
                        language_subgoal_expected,
                        language_subgoal_probabilities,
                        language_subgoal_slots,
                    ) = self.compute_language_subgoal_tokens(
                        contexts,
                        observation.state,
                        prior_source,
                        prefix_mask,
                    )
                    if (
                        hasattr(self, 'evidence_combination_gate')
                        and self.evidence_combination_component_count > 5
                    ):
                        subgoal_tokens = subgoal_tokens * (
                            self.evidence_combination_content_gate(
                                contexts,
                                observation.state,
                                subgoal_tokens,
                                5,
                            ).astype(subgoal_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + subgoal_tokens
                if hasattr(self, 'object_subgoal_binding_blocks'):
                    if object_affordance_slots is None or language_subgoal_slots is None:
                        raise ValueError(
                            'object-subgoal binding requires both slot banks'
                        )
                    (
                        object_subgoal_tokens,
                        object_subgoal_binding_prediction,
                        _,
                        _,
                        _,
                    ) = self.compute_object_subgoal_binding_tokens(
                        object_affordance_slots,
                        language_subgoal_slots,
                        language_subgoal_probabilities,
                        contexts,
                        observation.state,
                    )
                    if (
                        hasattr(self, 'evidence_combination_gate')
                        and self.evidence_combination_component_count > 7
                    ):
                        object_subgoal_tokens = object_subgoal_tokens * (
                            self.evidence_combination_content_gate(
                                contexts,
                                observation.state,
                                object_subgoal_tokens,
                                7,
                            ).astype(object_subgoal_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + object_subgoal_tokens
            else:
                # BestAnchor ContextAdaRMS deliberately keeps the audited
                # contextual parent without importing the 137-leaf dual
                # reasoner.  The parent already owns the same contextual
                # action-prior query/key/value path, so expose those states to
                # the new zero-gated AdaRMS branch before computing the
                # unchanged parent prior tokens.
                if (
                    hasattr(self, 'context_adarms_in')
                    or hasattr(self, 'velocity_refiner_blocks')
                    or hasattr(self, 'language_subgoal_blocks')
                ):
                    contexts = self.compute_action_prior_contexts(
                        prior_source,
                        prefix_mask,
                        observation.state,
                        (
                            persistent_private_state['memory']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['ordered_subgoals']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['frontier']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['slot_valid_mask']
                            if persistent_private_state is not None
                            else None
                        ),
                    )
                    coarse_tokens, coarse_actions = (
                        self._implicit_action_prior_outputs(contexts)
                    )
                    repeat = self.action_horizon // self.action_prior_horizon
                    action_prior_tokens = jnp.repeat(
                        coarse_tokens, repeat, axis=1
                    )
                    if hasattr(self, 'language_subgoal_blocks'):
                        (
                            subgoal_tokens,
                            language_subgoal_prediction,
                            language_subgoal_logits,
                            language_subgoal_expected,
                            language_subgoal_probabilities,
                            language_subgoal_slots,
                        ) = self.compute_language_subgoal_tokens(
                            contexts,
                            observation.state,
                            prior_source,
                            prefix_mask,
                        )
                        action_prior_tokens = (
                            action_prior_tokens + subgoal_tokens
                        )
                else:
                    action_prior_tokens, coarse_actions = (
                        self.compute_action_prior(
                            prior_source,
                            prefix_mask,
                            observation.state,
                        )
                    )
        elif hasattr(self, 'action_prior_queries'):
            action_prior_tokens, coarse_actions = self.compute_action_prior(
                prefix_tokens, prefix_mask, observation.state
            )
        if hasattr(self, 'latent_future_blocks'):
            if contexts is None:
                raise ValueError(
                    'latent future reasoning requires contextual prior states'
                )
            latent_future_current_visual = (
                self.latent_future_current_visual_tokens(
                    prefix_tokens, observation
                )
            )
            (
                latent_future_clean_residual,
                latent_future_prediction,
            ) = self.compute_latent_future_tokens(
                latent_future_current_visual,
                actions,
                contexts,
                observation.state,
            )
            latent_future_target, latent_future_target_mask = (
                self.latent_future_targets(
                    observation, latent_future_current_visual
                )
            )
        if hasattr(self, 'state_rollout_blocks'):
            if contexts is None:
                raise ValueError(
                    'state rollout reasoning requires contextual prior states'
                )
            (
                state_rollout_clean_residual,
                state_rollout_prediction,
            ) = self.compute_state_rollout_tokens(
                actions, contexts, observation.state
            )
            state_rollout_target, state_rollout_target_mask = (
                self.state_rollout_targets(observation)
            )
        if hasattr(self, 'action_moe_blocks'):
            if contexts is None:
                raise ValueError(
                    'action MoE reasoning requires contextual prior states'
                )
            # Predict the clean chunk from task/state context without exposing
            # the target actions as an identity shortcut.
            (
                action_moe_clean_residual,
                action_moe_prediction,
                action_moe_balance_loss,
                _,
            ) = self.compute_action_moe_tokens(
                jnp.zeros_like(actions), contexts, observation.state
            )
        if hasattr(self, 'specialist_module_router_score'):
            if contexts is None:
                raise ValueError(
                    'specialist module routing requires contextual prior states'
                )
            specialist_module_router_balance_loss = (
                self.compute_specialist_module_router_balance_loss(
                    self.compute_specialist_module_gates(
                        contexts, observation.state
                    )
                )
            )
        if hasattr(self, 'kinematic_action_blocks'):
            if contexts is None:
                raise ValueError(
                    'kinematic action reasoning requires contextual prior states'
                )
            # The clean-action auxiliary path receives no target action input;
            # otherwise its group decoders could learn an identity shortcut.
            (
                kinematic_action_clean_residual,
                kinematic_action_prediction,
                _,
            ) = self.compute_kinematic_action_tokens(
                jnp.zeros_like(actions), contexts, observation.state
            )
        if hasattr(self, 'spectral_action_blocks'):
            if contexts is None:
                raise ValueError(
                    'spectral action reasoning requires contextual prior states'
                )
            # Predict the clean spectrum only from task/state context and
            # learned frequency identities, avoiding an action identity path.
            _, spectral_action_prediction, _ = (
                self.compute_spectral_action_tokens(
                    jnp.zeros_like(actions), contexts, observation.state
                )
            )
        if hasattr(self, 'action_chunk_verifier_blocks'):
            if contexts is None:
                raise ValueError(
                    'action chunk verifier requires contextual prior states'
                )
            verifier_candidates = self.action_chunk_verifier_candidates(actions)
            candidate_count = verifier_candidates.shape[1]
            flat_candidates = verifier_candidates.reshape(
                -1, self.action_horizon, self.action_dim
            )
            candidate_contexts = jnp.repeat(
                contexts[:, None, :, :], candidate_count, axis=1
            ).reshape(-1, contexts.shape[1], contexts.shape[2])
            candidate_states = jnp.repeat(
                observation.state[:, None, :], candidate_count, axis=1
            ).reshape(-1, observation.state.shape[-1])
            _, verifier_scores = self.compute_action_chunk_verifier(
                flat_candidates, candidate_contexts, candidate_states
            )
            action_chunk_verifier_logits = verifier_scores.reshape(
                actions.shape[0], candidate_count
            ) / self.action_chunk_verifier_temperature
        def main_flow_loss(sample):
            sample_noise, sample_time = sample
            sample_time_expanded = sample_time[..., None, None]
            sample_x_t = (
                sample_time_expanded * sample_noise
                + (1 - sample_time_expanded) * actions
            )
            sample_u_t = sample_noise - actions
            flow_action_prior_tokens = action_prior_tokens
            if hetm_private_state is not None:
                hetm_prior = hetm_private_state['outputs']['prior_residual'][:, None, :]
                hetm_prior = jnp.broadcast_to(
                    hetm_prior,
                    (
                        hetm_prior.shape[0],
                        self.action_horizon,
                        hetm_prior.shape[-1],
                    ),
                )
                flow_action_prior_tokens = (
                    hetm_prior
                    if flow_action_prior_tokens is None
                    else flow_action_prior_tokens + hetm_prior
                )
            latent_future_residual = None
            latent_future_flow_prediction = None
            state_rollout_residual = None
            state_rollout_flow_prediction = None
            action_moe_residual = None
            action_moe_flow_prediction = None
            predictive_reliability_loss = 0.0
            kinematic_action_residual = None
            spectral_action_residual = None
            racg_action_residual = None
            racg_contact_loss = 0.0
            if racg_private_state is not None:
                (
                    racg_action_residual,
                    _,
                    racg_contact_logits,
                ) = self.racg.read_actions(
                    racg_private_state['scene'],
                    self.action_in_proj(sample_x_t),
                    sample_time,
                )
                contact_target = observation.racg_contact_target
                contact_valid = observation.racg_contact_valid
                if (contact_target is None) != (contact_valid is None):
                    raise ValueError(
                        'RACG contact targets must be supplied atomically'
                    )
                if contact_target is not None:
                    if contact_target.shape != racg_contact_logits.shape[:2]:
                        raise ValueError(
                            'RACG contact targets must be [batch,horizon]'
                        )
                    racg_contact_loss = (
                        _racg.sample_local_contact_logits_loss(
                            racg_contact_logits.astype(jnp.float32),
                            contact_target,
                            contact_valid,
                            graph_valid=(
                                racg_private_state['scene'].edge_mask[:, 0]
                            ),
                        )[:, None]
                    )
            if hasattr(self, 'action_chunk_verifier_blocks'):
                verifier_residual, _ = self.compute_action_chunk_verifier(
                    sample_x_t, contexts, observation.state
                )
                if getattr(
                    self,
                    'evidence_combination_action_verifier_component',
                    False,
                ):
                    verifier_residual = verifier_residual * (
                        self.evidence_combination_content_gate(
                            contexts,
                            observation.state,
                            verifier_residual,
                            6,
                        ).astype(verifier_residual.dtype)
                    )
                flow_action_prior_tokens = (
                    flow_action_prior_tokens + verifier_residual
                )
            if latent_future_current_visual is not None:
                (
                    latent_future_residual,
                    latent_future_flow_prediction,
                ) = self.compute_latent_future_tokens(
                    latent_future_current_visual,
                    sample_x_t,
                    contexts,
                    observation.state,
                )
                if not hasattr(self, 'predictive_world_model_gate'):
                    flow_action_prior_tokens = (
                        flow_action_prior_tokens + latent_future_residual
                    )
            if hasattr(self, 'state_rollout_blocks'):
                (
                    state_rollout_residual,
                    state_rollout_flow_prediction,
                ) = self.compute_state_rollout_tokens(
                    sample_x_t, contexts, observation.state
                )
                if not hasattr(self, 'predictive_world_model_gate'):
                    flow_action_prior_tokens = (
                        flow_action_prior_tokens + state_rollout_residual
                    )
            if hasattr(self, 'action_moe_blocks'):
                (
                    action_moe_residual,
                    action_moe_flow_prediction,
                    _,
                    _,
                ) = self.compute_action_moe_tokens(
                    sample_x_t, contexts, observation.state
                )
                if not getattr(
                    self, 'predictive_world_model_include_action_moe', False
                ):
                    flow_action_prior_tokens = (
                        flow_action_prior_tokens + action_moe_residual
                    )
            if hasattr(self, 'kinematic_action_blocks'):
                kinematic_action_residual, _, _ = (
                    self.compute_kinematic_action_tokens(
                        sample_x_t, contexts, observation.state
                    )
                )
                flow_action_prior_tokens = (
                    flow_action_prior_tokens + kinematic_action_residual
                )
            if hasattr(self, 'spectral_action_blocks'):
                spectral_action_residual, _, _ = (
                    self.compute_spectral_action_tokens(
                        sample_x_t, contexts, observation.state
                    )
                )
                flow_action_prior_tokens = (
                    flow_action_prior_tokens + spectral_action_residual
                )
            if hasattr(self, 'predictive_world_model_gate'):
                predictive_residual, _, reliability_logits = (
                    self.fuse_predictive_world_model_tokens(
                        latent_future_residual,
                        state_rollout_residual,
                        task_progress_tokens,
                        contexts,
                        observation.state,
                        action_moe_tokens=action_moe_residual,
                        persistent_memory=(
                            persistent_private_state['memory']
                            if persistent_private_state is not None
                            else None
                        ),
                        persistent_program=(
                            persistent_private_state['ordered_subgoals']
                            if persistent_private_state is not None
                            else None
                        ),
                        persistent_frontier=(
                            persistent_private_state['frontier']
                            if persistent_private_state is not None
                            else None
                        ),
                    )
                )
                predictive_reliability_loss = (
                    self.predictive_world_model_reliability_objective(
                        reliability_logits,
                        latent_prediction=latent_future_flow_prediction,
                        latent_target=latent_future_target,
                        latent_target_mask=latent_future_target_mask,
                        rollout_prediction=state_rollout_flow_prediction,
                        rollout_target=state_rollout_target,
                        rollout_target_mask=state_rollout_target_mask,
                        action_moe_prediction=action_moe_flow_prediction,
                        actions=actions,
                        progress_logits=task_progress_logits,
                        progress_expected=task_progress_expected,
                        progress_target=observation.task_progress_target,
                    )
                )
                if hasattr(self, 'evidence_combination_gate'):
                    predictive_gate = self.evidence_combination_content_gate(
                        contexts,
                        observation.state,
                        predictive_residual,
                        3,
                    ).astype(predictive_residual.dtype)
                else:
                    predictive_gate = 1.0
                flow_action_prior_tokens = (
                    flow_action_prior_tokens
                    + predictive_residual * predictive_gate
                )
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
                self.embed_suffix(
                    observation,
                    sample_x_t,
                    sample_time,
                    flow_action_prior_tokens,
                    action_reasoning_tokens,
                    contexts,
                    (
                        persistent_private_state['memory']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        hetm_private_state['outputs']
                        if hetm_private_state is not None
                        else None
                    ),
                    racg_action_residual,
                    persistent_ordered_program=(
                        persistent_private_state['ordered_subgoals']
                        if persistent_private_state is not None
                        else None
                    ),
                    persistent_frontier=(
                        persistent_private_state['frontier']
                        if persistent_private_state is not None
                        else None
                    ),
                    contact_phase_tokens=phase_contact_condition_tokens,
                    persistent_slot_valid_mask=(
                        persistent_private_state['slot_valid_mask']
                        if persistent_private_state is not None
                        else None
                    ),
                    persistent_previous_frontier=(
                        persistent_private_state['previous_frontier_distribution']
                        if persistent_private_state is not None else None
                    ),
                    persistent_previous_actions=(
                        persistent_private_state['previous_actions']
                        if persistent_private_state is not None else None
                    ),
                    persistent_previous_actions_valid=(
                        persistent_private_state['previous_actions_valid']
                        if persistent_private_state is not None else None
                    ),
                    persistent_current_context=(
                        persistent_private_state['current_context']
                        if persistent_private_state is not None else None
                    ),
                    persistent_previous_roles=(
                        persistent_private_state.get('prior_role_identity_anchors')
                        if persistent_private_state is not None else None
                    ),
                    persistent_current_roles=(
                        persistent_private_state.get('role_identity_anchors')
                        if persistent_private_state is not None else None
                    ),
                    persistent_role_valid_mask=(
                        persistent_private_state.get('role_valid_mask')
                        if persistent_private_state is not None else None
                    ),
                )
            )
            if persistent_private_state is not None:
                action_start = suffix_tokens.shape[1] - self.action_horizon
                injected_actions = self.persistent_memory.inject(
                    suffix_tokens[:, action_start:],
                    persistent_private_state['memory'],
                    ordered_program=persistent_private_state[
                        'ordered_subgoals'
                    ],
                    frontier=persistent_private_state['frontier'],
                    policy_scale=persistent_policy_scale,
                )
                if persistent_private_state['structured_demo_tokens'] is not None:
                    injected_actions = (
                        self.persistent_memory.structured_demo.inject_actions(
                            injected_actions,
                            persistent_private_state['structured_demo_tokens'],
                            persistent_private_state[
                                'structured_demo_token_mask'
                            ],
                            self.persistent_memory,
                        )
                    )
                injected_actions = self._inject_geometry_after_inherited_parent(
                    injected_actions,
                    geometry_context,
                    policy_scale=1.0,
                )
                suffix_tokens = suffix_tokens.at[:, action_start:].set(
                    injected_actions
                )
            if prefix_cache is None:
                input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
                ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
                attn_mask = make_attn_mask(input_mask, ar_mask)
                positions = jnp.cumsum(input_mask, axis=1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [prefix_tokens, suffix_tokens],
                    mask=attn_mask,
                    positions=positions,
                    adarms_cond=[None, adarms_cond],
                    action_layer_adapter=hmca_adapter,
                    memory_layer_adapter=memory_layer_adapter,
                )
            else:
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(
                    prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1]
                )
                full_attn_mask = jnp.concatenate(
                    [prefix_attn_mask, suffix_attn_mask], axis=-1
                )
                positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None]
                    + jnp.cumsum(suffix_mask, axis=-1)
                    - 1
                )
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=prefix_cache,
                    adarms_cond=[None, adarms_cond],
                    action_layer_adapter=hmca_adapter,
                    memory_layer_adapter=memory_layer_adapter,
                )
            velocity = self.action_out_proj(
                suffix_out[:, -self.action_horizon :]
            )
            # Each cascaded refiner has its own stop-gradient residual target.
            # Accumulate those auxiliary objectives instead of letting a later
            # refiner overwrite the earlier one's learning signal.  The main
            # flow objective is evaluated once on the final corrected velocity.
            refinement_auxiliary_loss = 0.0
            if hasattr(self, 'velocity_refiner_blocks'):
                base_velocity = velocity
                velocity, predicted_residual, _ = (
                    self.compute_velocity_refinement(
                        sample_x_t,
                        base_velocity,
                        suffix_out[:, -self.action_horizon :],
                        contexts,
                        observation.state,
                        sample_time,
                    )
                )
                residual_target = jax.lax.stop_gradient(
                    sample_u_t - base_velocity
                )
                refinement_auxiliary_loss = (
                    refinement_auxiliary_loss
                    + self.velocity_refiner_loss_weight
                    * self.active_action_mse(
                        predicted_residual, residual_target
                    )
                )
            if hasattr(self, 'action_visual_refiner_blocks'):
                base_velocity = velocity
                velocity, predicted_residual, _ = (
                    self.compute_action_visual_refinement(
                        sample_x_t,
                        base_velocity,
                        suffix_out[:, -self.action_horizon :],
                        prior_source,
                        prefix_mask,
                        observation.state,
                        sample_time,
                    )
                )
                residual_target = jax.lax.stop_gradient(
                    sample_u_t - base_velocity
                )
                refinement_auxiliary_loss = (
                    refinement_auxiliary_loss
                    + self.action_visual_refiner_loss_weight
                    * self.active_action_mse(
                        predicted_residual, residual_target
                    )
                )
            flow_objective = (
                self.active_action_mse(velocity, sample_u_t)
                + refinement_auxiliary_loss
                + getattr(
                    self,
                    'predictive_world_model_reliability_loss_weight',
                    0.0,
                )
                * predictive_reliability_loss
            )
            if racg_private_state is not None:
                flow_objective = flow_objective + (
                    self.racg_loss_weights['contact'] * racg_contact_loss
                )
            return flow_objective

        if self.main_flow_samples == 1:
            flow_loss = main_flow_loss((noise, time))
        else:
            main_noise_rng, main_time_rng = jax.random.split(
                jax.random.fold_in(rng, 73)
            )
            extra_noises = self.mask_inactive_actions(
                jax.random.normal(
                    main_noise_rng,
                    (self.main_flow_samples - 1, *actions.shape),
                )
            )
            extra_times = (
                _sample_beta_1p5_1(
                    main_time_rng,
                    (self.main_flow_samples - 1, *batch_shape),
                )
                * 0.999
                + 0.001
            )
            all_noises = jnp.concatenate([noise[None], extra_noises], axis=0)
            all_times = jnp.concatenate([time[None], extra_times], axis=0)
            # Sequential rematerialized suffix passes retain the statistical
            # benefit of K flow samples without making the per-device action
            # batch K times larger.
            sample_losses = jax.lax.map(
                jax.checkpoint(main_flow_loss),
                (all_noises, all_times),
            )
            flow_loss = jnp.mean(sample_losses, axis=0)
        if coarse_actions is None:
            total_loss = flow_loss
            if geometry_context is not None:
                total_loss = total_loss + geometry_auxiliary_loss
            if (
                racg_private_state is not None
                and _include_racg_scene_auxiliary
            ):
                total_loss = total_loss + self._racg_scene_auxiliary_loss(
                    observation, racg_private_state, hetm_private_state
                )[:, None]
            return total_loss
        repeat = self.action_horizon // self.action_prior_horizon
        coarse_targets = self.coarse_action_targets(actions)
        prior_loss = self.active_action_mse(coarse_actions, coarse_targets)
        prior_loss = jnp.repeat(prior_loss, repeat, axis=-1)
        if geometry_context is None:
            total_loss = flow_loss + self.action_prior_loss_weight * prior_loss
        else:
            total_loss = (
                flow_loss
                + self.action_prior_loss_weight * prior_loss
                + geometry_auxiliary_loss
            )
        if racg_private_state is not None and _include_racg_scene_auxiliary:
            total_loss = total_loss + self._racg_scene_auxiliary_loss(
                observation, racg_private_state, hetm_private_state
            )[:, None]
        if prefix_moe_balance_loss is not None:
            total_loss = (
                total_loss
                + self.prefix_moe_balance_loss_weight
                * prefix_moe_balance_loss
            )
        if kv_moe_balance_loss is not None:
            total_loss = (
                total_loss
                + self.kv_moe_balance_loss_weight * kv_moe_balance_loss
            )
        if explicit_velocity is not None:
            explicit_loss = self.active_action_mse(
                explicit_velocity,
                explicit_velocity_target,
            )
            explicit_loss = jnp.mean(explicit_loss, axis=1)
            explicit_loss = jnp.repeat(explicit_loss, repeat, axis=-1)
            total_loss = (
                total_loss + self.action_prior_explicit_loss_weight * explicit_loss
            )
        if spatial_relation_actions is not None:
            spatial_relation_loss = self.active_action_mse(
                spatial_relation_actions, coarse_targets
            )
            spatial_relation_loss = jnp.repeat(
                spatial_relation_loss, repeat, axis=-1
            )
            total_loss = (
                total_loss
                + self.spatial_relation_loss_weight * spatial_relation_loss
            )
        if object_affordance_actions is not None:
            object_affordance_loss = self.active_action_mse(
                object_affordance_actions, coarse_targets
            )
            object_affordance_loss = jnp.repeat(
                object_affordance_loss, repeat, axis=-1
            )
            total_loss = (
                total_loss
                + self.object_affordance_loss_weight * object_affordance_loss
            )
        if object_affordance_reconstruction_loss is not None:
            total_loss = total_loss + (
                self.object_affordance_reconstruction_loss_weight
                * object_affordance_reconstruction_loss[:, None]
            )
        if masked_spatial_actions is not None:
            masked_spatial_action_loss = self.active_action_mse(
                masked_spatial_actions, coarse_targets
            )
            masked_spatial_action_loss = jnp.repeat(
                masked_spatial_action_loss, repeat, axis=-1
            )
            total_loss = (
                total_loss
                + self.masked_spatial_action_loss_weight
                * masked_spatial_action_loss
            )
        if masked_spatial_reconstruction is not None:
            reconstruction_error = jnp.mean(
                jnp.square(
                    masked_spatial_reconstruction.astype(jnp.float32)
                    - masked_spatial_target.astype(jnp.float32)
                ),
                axis=-1,
            )
            reconstruction_weight = (
                masked_spatial_reconstruction_mask.astype(jnp.float32)
            )
            reconstruction_loss = jnp.sum(
                reconstruction_error * reconstruction_weight, axis=-1
            ) / jnp.maximum(jnp.sum(reconstruction_weight, axis=-1), 1.0)
            reconstruction_loss = jnp.repeat(
                reconstruction_loss[:, None], self.action_horizon, axis=1
            )
            total_loss = (
                total_loss
                + self.masked_spatial_reconstruction_loss_weight
                * reconstruction_loss
            )
        if object_future_actions is not None:
            object_future_action_loss = self.active_action_mse(
                object_future_actions, coarse_targets
            )
            object_future_action_loss = jnp.repeat(
                object_future_action_loss, repeat, axis=-1
            )
            total_loss = (
                total_loss
                + self.object_future_action_loss_weight
                * object_future_action_loss
            )
        if object_future_reconstruction is not None:
            object_future_error = jnp.mean(
                jnp.square(
                    object_future_reconstruction.astype(jnp.float32)
                    - object_future_target.astype(jnp.float32)
                ),
                axis=-1,
            )
            object_future_weight = object_future_target_mask.astype(jnp.float32)
            object_future_loss = jnp.sum(
                object_future_error * object_future_weight, axis=-1
            ) / jnp.maximum(jnp.sum(object_future_weight, axis=-1), 1.0)
            object_future_loss = jnp.repeat(
                object_future_loss[:, None], self.action_horizon, axis=1
            )
            total_loss = (
                total_loss
                + self.object_future_reconstruction_loss_weight
                * object_future_loss
            )
        if predicate_binding_actions is not None:
            predicate_action_loss = self.active_action_mse(
                predicate_binding_actions, coarse_targets
            )
            predicate_action_loss = jnp.repeat(
                predicate_action_loss, repeat, axis=-1
            )
            total_loss = (
                total_loss
                + self.predicate_binding_action_loss_weight
                * predicate_action_loss
            )
        if predicate_binding_contrastive_logits is not None:
            positive = predicate_binding_positive_mask.astype(jnp.float32)
            row_targets = positive / jnp.maximum(
                jnp.sum(positive, axis=-1, keepdims=True), 1.0
            )
            column_targets = positive / jnp.maximum(
                jnp.sum(positive, axis=0, keepdims=True), 1.0
            )
            row_loss = -jnp.sum(
                row_targets
                * jax.nn.log_softmax(
                    predicate_binding_contrastive_logits, axis=-1
                ),
                axis=-1,
            )
            column_loss = -jnp.sum(
                column_targets
                * jax.nn.log_softmax(
                    predicate_binding_contrastive_logits, axis=0
                ),
                axis=0,
            )
            contrastive_loss = 0.5 * (row_loss + column_loss)
            contrastive_loss = jnp.repeat(
                contrastive_loss[:, None], self.action_horizon, axis=1
            )
            total_loss = (
                total_loss
                + self.predicate_binding_contrastive_loss_weight
                * contrastive_loss
            )
        if (
            discrete_logits is not None
            and self.action_prior_discrete_auxiliary_loss
        ):
            code_targets = self.discrete_action_code_targets(actions)
            discrete_loss = -jnp.take_along_axis(
                jax.nn.log_softmax(discrete_logits.astype(jnp.float32), axis=-1),
                code_targets[:, None],
                axis=-1,
            )[:, 0]
            discrete_loss = discrete_loss * jnp.take(
                self.action_prior_discrete_codebook_weights.value,
                code_targets,
                axis=0,
            )
            total_loss = total_loss + self.action_prior_discrete_loss_weight * (
                discrete_loss[:, None]
            )
            step_code_targets = self.discrete_action_step_code_targets(actions)
            step_discrete_loss = -jnp.take_along_axis(
                jax.nn.log_softmax(
                    discrete_step_logits.astype(jnp.float32), axis=-1
                ),
                step_code_targets[:, :, None],
                axis=-1,
            )[:, :, 0]
            step_discrete_loss = step_discrete_loss * jnp.take(
                self.action_prior_discrete_codebook_step_weights.value,
                step_code_targets,
                axis=0,
            )
            total_loss = (
                total_loss
                + self.action_prior_discrete_step_loss_weight
                * step_discrete_loss
            )
        if contact_phase_logits is not None:
            phase_targets = self.contact_phase_targets(actions, observation.state)
            contact_phase_log_probability = jnp.take_along_axis(
                jax.nn.log_softmax(
                    contact_phase_logits.astype(jnp.float32)
                    / self.contact_phase_loss_temperature,
                    axis=-1,
                ),
                phase_targets[..., None],
                axis=-1,
            )[..., 0]
            contact_phase_loss = -contact_phase_log_probability
            if self.contact_phase_focal_gamma > 0:
                contact_phase_probability = jnp.exp(
                    contact_phase_log_probability
                )
                contact_phase_loss = contact_phase_loss * jnp.power(
                    jnp.maximum(1.0 - contact_phase_probability, 0.0),
                    self.contact_phase_focal_gamma,
                )
            class_weights = jnp.asarray(
                self.contact_phase_class_weights,
                dtype=contact_phase_loss.dtype,
            )
            contact_phase_loss = contact_phase_loss * jnp.take(
                class_weights, phase_targets, axis=0
            )
            total_loss = (
                total_loss
                + self.contact_phase_loss_weight * contact_phase_loss
            )
        if contact_affordance_risk_logits is not None:
            if object_affordance_assignments is None:
                raise ValueError(
                    'contact-affordance risk supervision requires assignments'
                )
            risk_targets = self.contact_affordance_risk_targets(
                object_affordance_assignments,
                actions,
                observation.state,
                (
                    persistent_private_state['clause_plan_attention']
                    if persistent_private_state is not None
                    and hasattr(self, 'contact_affordance_clause_attention_in')
                    else None
                ),
                (
                    persistent_private_state['frontier']
                    if persistent_private_state is not None
                    and hasattr(self, 'contact_affordance_clause_attention_in')
                    else None
                ),
            )
            risk_logits = contact_affordance_risk_logits.astype(jnp.float32)
            risk_loss = (
                jnp.maximum(risk_logits, 0.0)
                - risk_logits * risk_targets
                + jnp.log1p(jnp.exp(-jnp.abs(risk_logits)))
            )
            total_loss = (
                total_loss
                + self.contact_affordance_risk_loss_weight * risk_loss
            )
        if contact_affordance_relation_alignment_logits is not None:
            relation_logits = (
                contact_affordance_relation_alignment_logits.astype(jnp.float32)
            )
            relation_alignment_loss = self.valid_factorized_relation_alignment_loss(
                relation_logits,
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.factorized_auxiliary_valid,
            )
            relation_alignment_loss = jnp.repeat(
                relation_alignment_loss[:, None], self.action_horizon, axis=1
            )
            total_loss = total_loss + (
                self.contact_affordance_relation_contrastive_loss_weight
                * relation_alignment_loss
            )
        if structured_rationale_logits is not None:
            rationale_targets = self.structured_rationale_targets(actions)
            rationale_loss = -jnp.take_along_axis(
                jax.nn.log_softmax(
                    structured_rationale_logits.astype(jnp.float32), axis=-1
                ),
                rationale_targets[..., None],
                axis=-1,
            )[..., 0]
            rationale_class_weights = jnp.asarray(
                self.action_prior_rationale_class_weights,
                dtype=rationale_loss.dtype,
            )
            rationale_loss = rationale_loss * jnp.take_along_axis(
                jnp.broadcast_to(
                    rationale_class_weights[None, :, :],
                    structured_rationale_logits.shape,
                ),
                rationale_targets[..., None],
                axis=-1,
            )[..., 0]
            rationale_loss = jnp.mean(rationale_loss, axis=-1)
            total_loss = (
                total_loss
                + self.action_prior_rationale_loss_weight
                * rationale_loss[:, None]
            )
        if action_chunk_verifier_logits is not None:
            verifier_loss = -jax.nn.log_softmax(
                action_chunk_verifier_logits.astype(jnp.float32), axis=-1
            )[:, 0]
            total_loss = (
                total_loss
                + self.action_chunk_verifier_loss_weight
                * verifier_loss[:, None]
            )
        predictive_auxiliary_scale = getattr(
            self, 'predictive_world_model_auxiliary_scale', 1.0
        )
        if latent_future_prediction is not None:
            latent_future_loss = jnp.mean(
                jnp.square(
                    latent_future_prediction.astype(jnp.float32)
                    - latent_future_target.astype(jnp.float32)
                ),
                axis=-1,
            )
            target_mask = latent_future_target_mask.astype(
                latent_future_loss.dtype
            )
            latent_future_loss = jnp.sum(
                latent_future_loss * target_mask, axis=(1, 2)
            ) / jnp.maximum(jnp.sum(target_mask, axis=(1, 2)), 1.0)
            total_loss = (
                total_loss
                + self.latent_future_loss_weight
                * predictive_auxiliary_scale
                * latent_future_loss[:, None]
            )
        if state_rollout_prediction is not None:
            state_rollout_loss = jnp.mean(
                jnp.square(
                    state_rollout_prediction.astype(jnp.float32)
                    - state_rollout_target.astype(jnp.float32)
                ),
                axis=-1,
            )
            target_mask = state_rollout_target_mask.astype(
                state_rollout_loss.dtype
            )
            state_rollout_loss = jnp.sum(
                state_rollout_loss * target_mask, axis=-1
            ) / jnp.maximum(jnp.sum(target_mask, axis=-1), 1.0)
            total_loss = (
                total_loss
                + self.state_rollout_loss_weight
                * predictive_auxiliary_scale
                * state_rollout_loss[:, None]
            )
        if action_moe_prediction is not None:
            action_moe_prediction_loss = self.active_action_mse(
                action_moe_prediction.astype(jnp.float32),
                actions.astype(jnp.float32),
            )
            action_moe_auxiliary_scale = (
                predictive_auxiliary_scale
                if getattr(
                    self, 'predictive_world_model_include_action_moe', False
                )
                else 1.0
            )
            total_loss = (
                total_loss
                + self.action_moe_prediction_loss_weight
                * action_moe_auxiliary_scale
                * action_moe_prediction_loss
                + self.action_moe_balance_loss_weight
                * action_moe_auxiliary_scale
                * action_moe_balance_loss[:, None]
            )
        if specialist_module_router_balance_loss is not None:
            total_loss = (
                total_loss
                + self.specialist_module_router_balance_loss_weight
                * specialist_module_router_balance_loss
            )
        if kinematic_action_prediction is not None:
            kinematic_prediction_loss = self.active_action_mse(
                kinematic_action_prediction.astype(jnp.float32),
                actions.astype(jnp.float32),
            )
            total_loss = (
                total_loss
                + self.kinematic_action_prediction_loss_weight
                * kinematic_prediction_loss
            )
        if spectral_action_prediction is not None:
            spectral_target = self.action_to_spectrum(
                actions.astype(jnp.float32)
            )
            spectral_prediction_loss = self.active_action_mse(
                spectral_action_prediction.astype(jnp.float32),
                spectral_target,
            )
            # Parseval's identity keeps this full-spectrum objective on the
            # same per-position scale as the time-domain action auxiliaries.
            total_loss = (
                total_loss
                + self.spectral_action_prediction_loss_weight
                * spectral_prediction_loss
            )
        if task_progress_logits is not None:
            if observation.task_progress_target is None:
                raise ValueError(
                    'task progress training requires a progress target'
                )
            progress_target = observation.task_progress_target.astype(
                jnp.float32
            )
            progress_mask = jnp.logical_and(
                jnp.isfinite(progress_target), progress_target >= 0
            )
            clipped_progress = jnp.where(
                progress_mask,
                jnp.clip(progress_target, 0.0, 1.0),
                0.0,
            )
            progress_distribution = _soft_progress_targets(
                clipped_progress, self.task_progress_bins
            )
            progress_ce = -jnp.sum(
                progress_distribution
                * jax.nn.log_softmax(
                    task_progress_logits.astype(jnp.float32), axis=-1
                ),
                axis=-1,
            )
            progress_regression = jnp.square(
                task_progress_expected.astype(jnp.float32) - clipped_progress
            )
            progress_loss = (progress_ce + 2.0 * progress_regression) * (
                progress_mask.astype(progress_ce.dtype)
            )
            total_loss = (
                total_loss
                + self.task_progress_loss_weight
                * predictive_auxiliary_scale
                * progress_loss[:, None]
            )
        if language_subgoal_logits is not None:
            subgoal_action_loss = self.active_action_mse(
                language_subgoal_prediction.astype(jnp.float32),
                actions.astype(jnp.float32),
            )
            # Cross-benchmark demonstrations such as RoboDojo provide full
            # action trajectories but no phase/progress annotation. Keep the
            # useful action-supervised subgoal branch trainable and omit only
            # the unavailable progress objective in that case.
            if observation.task_progress_target is None:
                total_loss = (
                    total_loss
                    + self.language_subgoal_action_loss_weight
                    * subgoal_action_loss
                )
            else:
                subgoal_progress_target = (
                    observation.task_progress_target.astype(jnp.float32)
                )
                subgoal_progress_mask = jnp.logical_and(
                    jnp.isfinite(subgoal_progress_target),
                    subgoal_progress_target >= 0,
                )
                clipped_subgoal_progress = jnp.where(
                    subgoal_progress_mask,
                    jnp.clip(subgoal_progress_target, 0.0, 1.0),
                    0.0,
                )
                subgoal_distribution = _soft_progress_targets(
                    clipped_subgoal_progress,
                    self.language_subgoal_slot_count,
                )
                subgoal_ce = -jnp.sum(
                    subgoal_distribution
                    * jax.nn.log_softmax(
                        language_subgoal_logits.astype(jnp.float32), axis=-1
                    ),
                    axis=-1,
                )
                subgoal_regression = jnp.square(
                    language_subgoal_expected.astype(jnp.float32)
                    - clipped_subgoal_progress
                )
                # Full-horizon sampling over-represents approach frames by
                # roughly 5x relative to completion/settle. Weight the
                # interpolated target distribution rather than argmax labels.
                subgoal_phase_weight = jnp.sum(
                    subgoal_distribution
                    * jnp.asarray(
                        self.language_subgoal_phase_class_weights,
                        dtype=jnp.float32,
                    ),
                    axis=-1,
                )
                subgoal_progress_loss = (
                    subgoal_ce + 2.0 * subgoal_regression
                ) * subgoal_phase_weight * subgoal_progress_mask.astype(
                    subgoal_ce.dtype
                )
                total_loss = (
                    total_loss
                    + self.language_subgoal_progress_loss_weight
                    * subgoal_progress_loss[:, None]
                    + self.language_subgoal_action_loss_weight
                    * subgoal_action_loss
                )
        if object_subgoal_binding_prediction is not None:
            object_subgoal_binding_loss = self.active_action_mse(
                object_subgoal_binding_prediction.astype(jnp.float32),
                actions.astype(jnp.float32),
            )
            total_loss = (
                total_loss
                + self.object_subgoal_binding_action_loss_weight
                * object_subgoal_binding_loss
            )
        if (
            demo_reliability is not None
            and observation.demonstration_reliability_target is not None
        ):
            reliability_target = observation.demonstration_reliability_target
            reliability_supervised = reliability_target >= 0
            reliability_loss = jnp.square(
                demo_reliability[:, 0]
                - jnp.clip(reliability_target, 0.0, 1.0)
            )
            reliability_loss = reliability_loss * reliability_supervised.astype(
                reliability_loss.dtype
            )
            total_loss = (
                total_loss
                + self.action_prior_demo_reliability_loss_weight
                * reliability_loss[:, None]
            )
        if (
            demo_router_logits is not None
            and observation.demonstration_router_target is not None
        ):
            router_target = observation.demonstration_router_target
            router_supervised = router_target >= 0
            safe_target = jnp.clip(
                router_target, 0, demo_router_logits.shape[-1] - 1
            )
            router_loss = -jnp.take_along_axis(
                jax.nn.log_softmax(
                    demo_router_logits.astype(jnp.float32), axis=-1
                ),
                safe_target[:, None],
                axis=-1,
            )[:, 0]
            router_loss = router_loss * router_supervised.astype(router_loss.dtype)
            total_loss = (
                total_loss
                + self.action_prior_demo_router_loss_weight * router_loss[:, None]
            )
        if hetm_private_state is not None:
            targets = (
                observation.hetm_event_targets,
                observation.hetm_predicate_targets,
                observation.hetm_frontier_target,
                observation.hetm_next_frontier_target,
                observation.hetm_supervision_valid,
            )
            supplied = sum(target is not None for target in targets)
            if supplied not in (0, len(targets)):
                raise ValueError('HETM supervision fields must be supplied atomically')
            if supplied:
                hetm_loss = _hetm.supervised_auxiliary_loss(
                    hetm_private_state['outputs'],
                    event_targets=observation.hetm_event_targets,
                    predicate_targets=observation.hetm_predicate_targets,
                    frontier_targets=observation.hetm_frontier_target,
                    next_frontier_targets=observation.hetm_next_frontier_target,
                    sample_valid=observation.hetm_supervision_valid,
                    weights=self.hetm_loss_weights,
                )
                total_loss = total_loss + hetm_loss[:, None]
        return total_loss

    def _spatial_language_weight(self, training_step):
        initial, peak, final, warmup, decay_start, total = (
            self.spatial_language_auxiliary_schedule
        )
        step = jnp.clip(
            jnp.asarray(training_step, dtype=jnp.float32), 0.0, float(total)
        )
        warmup_weight = initial + step / float(warmup) * (peak - initial)
        decay_fraction = (step - float(decay_start)) / float(
            total - decay_start
        )
        cosine = 0.5 * (1.0 - jnp.cos(jnp.pi * decay_fraction))
        decay_weight = peak + cosine * (final - peak)
        return jnp.where(
            step <= warmup,
            warmup_weight,
            jnp.where(step <= decay_start, peak, decay_weight),
        )

    def compute_loss_sequence(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: at.Float[at.Array, 'b r ah ad'],
        *,
        training_step: int | at.Int[at.Array, ''],
        train: bool = False,
    ):
        """Unroll the causal 4/8-replan PSM objective without padding dilution."""
        if not hasattr(self, 'persistent_memory'):
            raise ValueError('persistent memory is not enabled for this model')
        if actions.ndim != 4:
            raise ValueError('sequence actions must be [batch, replan, horizon, action]')
        batch_size, replan_count = actions.shape[:2]
        if replan_count != 8:
            raise ValueError('persistent memory sequences must be padded to 8 replans')
        valid_mask = observation.persistent_memory_sequence_valid
        if valid_mask is None or valid_mask.shape != (batch_size, replan_count):
            raise ValueError('persistent memory sequence-valid mask is required')
        initial_memory = observation.persistent_memory_initial_state
        initial_frontier = observation.persistent_subgoal_initial_frontier
        if initial_memory is None or initial_frontier is None:
            raise ValueError('persistent interior windows require cached memory/frontier')
        current_memory = jax.lax.stop_gradient(initial_memory)
        current_frontier = jax.lax.stop_gradient(initial_frontier)
        hetm_valid_mask = observation.hetm_supervision_valid
        if hasattr(self, 'hetm'):
            if hetm_valid_mask is None or hetm_valid_mask.shape != valid_mask.shape:
                raise ValueError('joint sequence requires aligned HETM validity')
            clean_hetm_state = _hetm.initial_state(batch_size)
            history_targets = observation.hetm_history_event_targets
            history_valid = observation.hetm_history_event_valid
            initial_predicates = observation.hetm_initial_predicate_targets
            initial_hetm_frontier = observation.hetm_initial_frontier_target
            initial_state_valid = observation.hetm_initial_state_valid
            initial_fields = (
                initial_predicates,
                initial_hetm_frontier,
                initial_state_valid,
            )
            if (history_targets is None) != (history_valid is None):
                raise ValueError('HETM history fields must be supplied atomically')
            if sum(value is not None for value in initial_fields) not in (0, 3):
                raise ValueError('HETM initial-state fields must be supplied atomically')
            if history_targets is not None and all(
                value is not None for value in initial_fields
            ):
                teacher_state = self.hetm.supervised_initial_state(
                    history_targets,
                    history_valid,
                    initial_predicates,
                    initial_hetm_frontier,
                )
                teacher_state = {
                    name: jnp.where(
                        initial_state_valid.reshape(
                            (batch_size,) + (1,) * (value.ndim - 1)
                        ),
                        value,
                        clean_hetm_state[name],
                    )
                    for name, value in teacher_state.items()
                }
            else:
                teacher_state = clean_hetm_state
            if train:
                teacher_probability = 0.75 - 0.50 * jnp.clip(
                    jnp.asarray(training_step, dtype=jnp.float32) / 15_000.0,
                    0.0,
                    1.0,
                )
                use_teacher = jax.random.bernoulli(
                    jax.random.fold_in(rng, 883),
                    teacher_probability,
                    (batch_size,),
                )
                current_hetm_state = {
                    name: jnp.where(
                        use_teacher.reshape(
                            (batch_size,) + (1,) * (value.ndim - 1)
                        ),
                        value,
                        clean_hetm_state[name],
                    )
                    for name, value in teacher_state.items()
                }
            else:
                current_hetm_state = teacher_state
        else:
            current_hetm_state = None
        if hasattr(self, 'racg'):
            racg_initial = (
                observation.racg_target_anchor,
                observation.racg_target_geometry,
                observation.racg_target_anchor_valid,
            )
            if sum(value is not None for value in racg_initial) not in (0, 3):
                raise ValueError(
                    'RACG pre-window anchor must be supplied atomically'
                )
            if racg_initial[0] is None:
                current_racg_state = {
                    'target_anchor': jnp.zeros(
                        (batch_size, _hetm.HIDDEN_DIM), jnp.float32
                    ),
                    'target_geometry': jnp.zeros(
                        (batch_size, 5), jnp.float32
                    ),
                    'target_anchor_valid': jnp.zeros(
                        (batch_size,), jnp.bool_
                    ),
                }
            else:
                current_racg_state = {
                    'target_anchor': jax.lax.stop_gradient(racg_initial[0]),
                    'target_geometry': jax.lax.stop_gradient(racg_initial[1]),
                    'target_anchor_valid': jax.lax.stop_gradient(
                        racg_initial[2]
                    ),
                }
        else:
            current_racg_state = None
        valid_count = jnp.maximum(jnp.sum(valid_mask, axis=1), 1)
        flow_index_mask = jnp.zeros((replan_count,), dtype=jnp.bool_).at[
            jnp.asarray(self.persistent_memory_flow_replan_indices)
        ].set(True)
        selected_flow_count = jnp.maximum(
            jnp.sum(valid_mask & flow_index_mask[None], axis=1), 1
        )
        policy_scale = jnp.clip(
            jnp.asarray(training_step, dtype=jnp.float32)
            / float(self.persistent_memory_policy_gain_warmup_steps),
            0.0,
            1.0,
        )
        decay_progress = jnp.clip(
            (
                jnp.asarray(training_step, dtype=jnp.float32)
                - float(self.persistent_memory_policy_gain_warmup_steps)
            )
            / float(
                self.persistent_memory_auxiliary_decay_steps
                - self.persistent_memory_policy_gain_warmup_steps
            ),
            0.0,
            1.0,
        )
        auxiliary_multiplier = (
            self.persistent_memory_auxiliary_final_multiplier
            + (1.0 - self.persistent_memory_auxiliary_final_multiplier)
            * 0.5
            * (1.0 + jnp.cos(jnp.pi * decay_progress))
        )
        initial_bound_roles = jnp.zeros(
            (batch_size, 2, self.persistent_memory.hidden_dim),
            dtype=current_memory.dtype,
        )
        initial_role_valid = jnp.zeros((batch_size,), dtype=jnp.bool_)

        def compute_active_replan(carry, replan_index):
            (
                current_memory,
                current_frontier,
                previous_bound_roles,
                role_valid,
                current_hetm_state,
                current_racg_state,
            ) = carry

            initial_window_fields = {
                'persistent_memory_initial_state',
                'persistent_subgoal_initial_frontier',
                'hetm_initial_predicate_targets',
            }

            def take_replan(path, value):
                leaf_name = next(
                    (
                        getattr(key, 'name', None)
                        for key in reversed(path)
                        if getattr(key, 'name', None) is not None
                    ),
                    None,
                )
                if leaf_name in initial_window_fields:
                    return value
                if (
                    hasattr(value, 'shape')
                    and value.ndim >= 2
                    and tuple(value.shape[:2]) == (batch_size, replan_count)
                ):
                    return value[:, replan_index]
                return value

            current_observation = jax.tree_util.tree_map_with_path(
                take_replan, observation
            )
            current_observation = current_observation.replace(
                persistent_memory_initial_state=current_memory,
                persistent_subgoal_initial_frontier=current_frontier,
                **(
                    {
                        'hetm_event_ledger': current_hetm_state['event_ledger'],
                        'hetm_event_valid': current_hetm_state['event_valid'],
                        'hetm_event_write_index': current_hetm_state['event_write_index'],
                        'hetm_last_event_probabilities': current_hetm_state[
                            'last_event_probabilities'
                        ],
                        'hetm_predicate_memory': current_hetm_state['predicate_memory'],
                        'hetm_predicate_probabilities': current_hetm_state[
                            'predicate_probabilities'
                        ],
                        'hetm_frontier': current_hetm_state['frontier'],
                        'hetm_episode_start': jnp.zeros(
                            (batch_size,), dtype=jnp.bool_
                        ),
                    }
                    if current_hetm_state is not None
                    else {}
                ),
                **(
                    {
                        'racg_target_anchor': current_racg_state[
                            'target_anchor'
                        ],
                        'racg_target_geometry': current_racg_state[
                            'target_geometry'
                        ],
                        'racg_target_anchor_valid': current_racg_state[
                            'target_anchor_valid'
                        ],
                        'racg_episode_start': jnp.zeros(
                            (batch_size,), dtype=jnp.bool_
                        ),
                    }
                    if current_racg_state is not None
                    else {}
                ),
            )

            # Preprocess, encode, and update private state exactly once per
            # valid replan.  Selected parent-flow replans reuse these values so
            # the flow objective and recurrent carry see the same augmented
            # observation instead of two independently computed branches.
            step_rng = jax.random.fold_in(rng, replan_index)
            preprocess_rng, _, _ = jax.random.split(step_rng, 3)
            processed = _model.preprocess_observation(
                preprocess_rng, current_observation, train=train
            )
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(processed)
            contextual_prefix = None
            hetm_private = None
            racg_private = None
            if current_hetm_state is not None:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
                prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
                (prefix_out, _), prefix_cache = self.PaliGemma.llm(
                    [prefix_tokens, None],
                    mask=prefix_attn_mask,
                    positions=prefix_positions,
                )
                contextual_prefix = (prefix_out, prefix_cache)
                hetm_private = self._hetm_state_from_observation(
                    processed,
                    jax.lax.stop_gradient(prefix_out),
                    prefix_mask,
                )
                racg_private = self._racg_state_from_observation(
                    processed,
                    jax.lax.stop_gradient(prefix_out),
                    prefix_mask,
                    hetm_private,
                )
            private = self._persistent_state_from_observation(
                processed,
                prefix_tokens,
                prefix_mask,
                compute_object_reconstruction=train,
            )

            # The replan index is static in the production unroll. Keep the
            # selected FSDP parent calls out of XLA conditional control flow.
            if replan_index in self.persistent_memory_flow_replan_indices:
                flow_processed = processed.replace(
                    hetm_event_targets=None,
                    hetm_predicate_targets=None,
                    hetm_frontier_target=None,
                    hetm_next_frontier_target=None,
                    hetm_supervision_valid=None,
                )
                parent_loss = self.compute_loss(
                    step_rng,
                    current_observation,
                    actions[:, replan_index],
                    train=train,
                    persistent_policy_scale=policy_scale,
                    _preprocessed_observation=flow_processed,
                    _precomputed_prefix=(
                        prefix_tokens,
                        prefix_mask,
                        prefix_ar_mask,
                    ),
                    _precomputed_contextual_prefix=contextual_prefix,
                    _precomputed_persistent_state=private,
                    _precomputed_hetm_state=hetm_private,
                    _precomputed_racg_state=racg_private,
                    _geometry_training_step=training_step,
                    _include_racg_scene_auxiliary=False,
                )
            else:
                parent_loss = jnp.zeros(
                    (batch_size, self.action_horizon), dtype=jnp.float32
                )
            valid = valid_mask[:, replan_index].astype(jnp.bool_)
            next_memory = jnp.where(
                valid[:, None, None],
                private['memory'].astype(current_memory.dtype),
                current_memory,
            )
            next_frontier = jnp.where(
                valid[:, None], private['frontier'], current_frontier
            )
            if hetm_private is not None:
                next_hetm_state = {
                    name: jnp.where(
                        valid.reshape(
                            (batch_size,) + (1,) * (value.ndim - 1)
                        ),
                        value,
                        current_hetm_state[name],
                    )
                    for name, value in hetm_private['next_state'].items()
                }
            else:
                next_hetm_state = current_hetm_state
            if racg_private is not None:
                next_racg_state = {
                    name: jnp.where(
                        valid.reshape(
                            (batch_size,) + (1,) * (value.ndim - 1)
                        ),
                        value,
                        current_racg_state[name],
                    )
                    for name, value in racg_private['next_state'].items()
                }
            else:
                next_racg_state = current_racg_state

            auxiliary = jnp.zeros((batch_size,), dtype=jnp.float32)
            if hetm_private is not None:
                targets = (
                    processed.hetm_event_targets,
                    processed.hetm_predicate_targets,
                    processed.hetm_frontier_target,
                    processed.hetm_next_frontier_target,
                )
                if any(target is None for target in targets):
                    raise ValueError('joint HETM supervision is incomplete')
                auxiliary = auxiliary + _hetm.supervised_auxiliary_loss(
                    hetm_private['outputs'],
                    event_targets=processed.hetm_event_targets,
                    predicate_targets=processed.hetm_predicate_targets,
                    frontier_targets=processed.hetm_frontier_target,
                    next_frontier_targets=processed.hetm_next_frontier_target,
                    sample_valid=valid,
                    weights=self.hetm_loss_weights,
                )
                # Direct's frozen recurrent frontier is the trusted teacher.
                # Align HETM to it without allowing the new objective to move
                # the inherited PSM prediction path.
                auxiliary = auxiliary + (
                    self.hetm_psm_frontier_consistency_loss_weight
                    * _psm_hetm_frontier_consistency_loss(
                        private['frontier'],
                        hetm_private['outputs']['frontier'],
                    )
                    * valid.astype(jnp.float32)
                )
            if racg_private is not None:
                auxiliary = auxiliary + self._racg_scene_auxiliary_loss(
                    processed, racg_private, hetm_private
                ) * valid.astype(jnp.float32)
            spatial_language_auxiliary = jnp.zeros(
                (batch_size,), dtype=jnp.float32
            )
            if train and hasattr(self, 'spatial_language_aux'):
                language_ids = current_observation.spatial_language_target_ids
                language_mask = current_observation.spatial_language_target_mask
                if (language_ids is None) != (language_mask is None):
                    raise ValueError(
                        'spatial-language ids and mask must be supplied together'
                    )
                if language_ids is not None:
                    demo_tokens = private['structured_demo_tokens']
                    demo_token_mask = private['structured_demo_token_mask']
                    if demo_tokens is None or demo_token_mask is None:
                        raise ValueError(
                            'spatial-language supervision requires demo context'
                        )
                    language_loss = self.spatial_language_aux.loss(
                        language_ids,
                        language_mask,
                        private['structured_demo_shared_context'],
                        private['phase_evidence_state'],
                        demo_tokens,
                        demo_token_mask,
                    )
                    spatial_language_auxiliary = self._spatial_language_weight(
                        training_step
                    ) * language_loss
            current_target = current_observation.persistent_current_subgoal_target
            next_target = current_observation.persistent_next_subgoal_target
            progress_target = current_observation.persistent_next_progress_target
            recovery = getattr(
                self.persistent_memory,
                'hcea_causal_recovery_action_experts_v1',
                None,
            )
            if recovery is not None:
                _, recovery_state = recovery(
                    jnp.zeros(
                        (batch_size, self.action_horizon, self.action_in_proj.out_features),
                        dtype=private['current_context'].dtype,
                    ),
                    private['ordered_subgoals'],
                    private['previous_frontier_distribution'],
                    private['frontier'],
                    private['previous_actions'],
                    private['previous_actions_valid'],
                    private['current_context'],
                    private['slot_valid_mask'],
                )
                auxiliary = auxiliary + (
                    self.hcea_causal_recovery_intent_loss_weight
                    * _hcea_recovery.transition_intent_auxiliary_per_sample(
                        recovery_state,
                        target_advance=(
                            (next_target > current_target).astype(jnp.float32)
                            if current_target is not None and next_target is not None
                            else None
                        ),
                        target_valid=(
                            (current_target >= 0) & (next_target >= 0)
                            if current_target is not None and next_target is not None
                            else None
                        ),
                    )
                )
            role_transport = getattr(
                self.persistent_memory,
                'hcea_causal_role_identity_transport_expert_v1',
                None,
            )
            if role_transport is not None:
                _, role_transport_state = role_transport(
                    jnp.zeros(
                        (batch_size, self.action_horizon, self.action_in_proj.out_features),
                        dtype=private['current_context'].dtype,
                    ),
                    private['prior_role_identity_anchors'],
                    private['role_identity_anchors'],
                    private['previous_actions'],
                    private['current_context'],
                    private['frontier'],
                    private['role_valid_mask'],
                    private['previous_actions_valid'],
                )
                auxiliary = auxiliary + (
                    self.hcea_causal_role_identity_transport_loss_weight
                    * _hcea_role_transport.role_transport_auxiliary_per_sample(
                        role_transport_state
                    )
                )
            flow_phase_weight = jnp.ones((batch_size,), dtype=jnp.float32)
            if current_target is not None:
                flow_phase_weights = (
                    self.persistent_memory.balanced_factorized_class_weights(
                        self.persistent_memory_flow_phase_class_counts,
                        count_floor=self.persistent_memory_flow_phase_count_floor,
                    )
                )
                flow_phase_weight = jnp.take(
                    flow_phase_weights,
                    jnp.clip(
                        current_target,
                        0,
                        self.persistent_memory.subgoal_slots - 1,
                    ),
                    axis=0,
                )
                prior_slot = private['prior_frontier']
                reachable_current_target = jnp.clip(
                    current_target,
                    prior_slot,
                    jnp.minimum(
                        prior_slot + 1,
                        self.persistent_memory.subgoal_slots - 1,
                    ),
                )
                selected = jnp.take_along_axis(
                    private['frontier'],
                    reachable_current_target[:, None],
                    axis=-1,
                )[:, 0]
                phase_class_weights = (
                    self.persistent_memory.balanced_factorized_class_weights(
                        self.persistent_memory_semantic_phase_class_counts
                    )
                )
                selected_phase_weight = jnp.take(
                    phase_class_weights, reachable_current_target, axis=0
                )
                auxiliary = auxiliary - self.persistent_memory_loss_weights[
                    'route'
                ] * selected_phase_weight * jnp.log(
                    jnp.maximum(selected, 1.0e-8)
                )
                semantic_frontier_state = private[
                    'semantic_frontier_completion_state'
                ]
                if semantic_frontier_state is not None:
                    semantic_reachable_target = jnp.clip(
                        current_target,
                        semantic_frontier_state['current_index'],
                        semantic_frontier_state['following_index'],
                    )
                    semantic_target_allowed = jnp.take_along_axis(
                        semantic_frontier_state['allowed_mask'],
                        semantic_reachable_target[:, None],
                        axis=-1,
                    )[:, 0]
                    semantic_current_allowed = jnp.take_along_axis(
                        semantic_frontier_state['allowed_mask'],
                        semantic_frontier_state['current_index'][:, None],
                        axis=-1,
                    )[:, 0]
                    semantic_fallback_target = jnp.where(
                        semantic_current_allowed,
                        semantic_frontier_state['current_index'],
                        semantic_frontier_state['following_index'],
                    )
                    semantic_reachable_target = jnp.where(
                        semantic_target_allowed,
                        semantic_reachable_target,
                        semantic_fallback_target,
                    )
                    semantic_valid = jnp.any(
                        semantic_frontier_state['allowed_mask'], axis=-1
                    ).astype(jnp.float32)
                    semantic_log_probabilities = jax.nn.log_softmax(
                        semantic_frontier_state['auxiliary_logits'].astype(
                            jnp.float32
                        ),
                        axis=-1,
                    )
                    semantic_selected = jnp.take_along_axis(
                        semantic_log_probabilities,
                        semantic_reachable_target[:, None],
                        axis=-1,
                    )[:, 0]
                    transition_class = (
                        semantic_reachable_target
                        > semantic_frontier_state['current_index']
                    ).astype(jnp.int32)
                    semantic_class_weights = jnp.asarray(
                        self.semantic_frontier_completion_class_weights,
                        jnp.float32,
                    )
                    semantic_weight = semantic_class_weights[
                        semantic_reachable_target, transition_class
                    ]
                    auxiliary = auxiliary - (
                        self.semantic_frontier_completion_auxiliary_loss_weight
                        * semantic_weight
                        * semantic_selected
                        * semantic_valid
                    )
                hcea_state = private[
                    'hierarchical_clause_event_alignment_state'
                ]
                if hcea_state is not None:
                    hcea_target = jnp.clip(
                        current_target,
                        hcea_state['current_index'],
                        hcea_state['following_index'],
                    )
                    hcea_target_allowed = jnp.take_along_axis(
                        hcea_state['allowed_mask'], hcea_target[:, None], axis=-1
                    )[:, 0]
                    hcea_current_allowed = jnp.take_along_axis(
                        hcea_state['allowed_mask'],
                        hcea_state['current_index'][:, None],
                        axis=-1,
                    )[:, 0]
                    hcea_fallback = jnp.where(
                        hcea_current_allowed,
                        hcea_state['current_index'],
                        hcea_state['following_index'],
                    )
                    hcea_target = jnp.where(
                        hcea_target_allowed, hcea_target, hcea_fallback
                    )
                    hcea_log_probabilities = jax.nn.log_softmax(
                        hcea_state['auxiliary_logits'].astype(jnp.float32), axis=-1
                    )
                    hcea_selected = jnp.take_along_axis(
                        hcea_log_probabilities, hcea_target[:, None], axis=-1
                    )[:, 0]
                    hcea_valid = jnp.any(
                        hcea_state['allowed_mask'], axis=-1
                    ).astype(jnp.float32)
                    auxiliary = auxiliary - (
                        self.hierarchical_clause_event_alignment_auxiliary_loss_weight
                        * hcea_selected
                        * hcea_valid
                    )
            if next_target is not None:
                current_slot = jnp.argmax(private['frontier'], axis=-1)
                reachable_next_target = jnp.clip(
                    next_target,
                    current_slot,
                    jnp.minimum(
                        current_slot + 1,
                        self.persistent_memory.subgoal_slots - 1,
                    ),
                )
                transition_indices = jnp.arange(
                    self.persistent_memory.subgoal_slots
                )[None]
                transition_allowed = (
                    transition_indices == current_slot[:, None]
                ) | (
                    transition_indices
                    == jnp.minimum(
                        current_slot + 1,
                        self.persistent_memory.subgoal_slots - 1,
                    )[:, None]
                )
                transition_logits = jnp.where(
                    transition_allowed,
                    private['transition_logits'].astype(jnp.float32),
                    -1.0e30,
                )
                next_log_probs = jax.nn.log_softmax(transition_logits, axis=-1)
                selected = jnp.take_along_axis(
                    next_log_probs,
                    reachable_next_target[:, None],
                    axis=-1,
                )[:, 0]
                phase_class_weights = (
                    self.persistent_memory.balanced_factorized_class_weights(
                        self.persistent_memory_semantic_phase_class_counts
                    )
                )
                selected_phase_weight = jnp.take(
                    phase_class_weights, reachable_next_target, axis=0
                )
                auxiliary = auxiliary - self.persistent_memory_loss_weights[
                    'transition'
                ] * selected_phase_weight * selected
            if progress_target is not None:
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'progress'
                ] * jnp.square(private['progress'] - progress_target)
            next_action_target = (
                current_observation.memory_next_action_summary_target
            )
            if next_action_target is None:
                raise ValueError(
                    'persistent action verification target is required'
                )
            if next_action_target.shape != private['next_action_summary'].shape:
                raise ValueError(
                    'persistent action verification target has an invalid shape'
                )
            auxiliary = auxiliary + self.persistent_memory_loss_weights[
                'action'
            ] * jnp.mean(
                jnp.square(
                    private['next_action_summary'] - next_action_target
                ),
                axis=-1,
            )
            role_error = jnp.mean(
                jnp.square(
                    private['bound_roles']
                    - jax.lax.stop_gradient(previous_bound_roles)
                ),
                axis=(-1, -2),
            )
            auxiliary = auxiliary + self.persistent_memory_loss_weights[
                'temporal_role'
            ] * role_error * role_valid.astype(role_error.dtype)

            role_valid_mask = private['role_valid_mask'].astype(jnp.bool_)
            camera_roles = private['camera_bound_roles']
            camera_mask = private['camera_mask']
            if camera_roles is not None and camera_mask is not None:
                camera_role_valid = (
                    camera_mask[:, :, None].astype(jnp.bool_)
                    & role_valid_mask[:, None, :]
                )
                camera_error = jnp.mean(
                    jnp.square(
                        camera_roles - private['bound_roles'][:, None]
                    ),
                    axis=-1,
                )
                cross_camera_loss = jnp.sum(
                    camera_error * camera_role_valid.astype(camera_error.dtype),
                    axis=(1, 2),
                ) / jnp.maximum(
                    jnp.sum(camera_role_valid, axis=(1, 2)), 1
                )
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'cross_camera'
                ] * cross_camera_loss

            cross_view_state = private['cross_view_role_consensus_state']
            cross_view_module = getattr(
                self.persistent_memory, 'cross_view_role_consensus_v1', None
            )
            if cross_view_state is not None:
                if cross_view_module is None or camera_mask is None:
                    raise ValueError(
                        'cross-view role state requires its production module and camera mask'
                    )
                cross_view_contrastive_loss = cross_view_module.contrastive_loss(
                    cross_view_state['input_roles'],
                    camera_mask,
                    role_valid_mask,
                    temperature=self.cross_view_role_contrastive_temperature,
                )
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'cross_view_role_contrastive'
                ] * cross_view_contrastive_loss

            contact_risk_state = private[
                'contact_risk_calibrated_role_residual_state'
            ]
            if contact_risk_state is not None:
                role_assignment_entropy = private['role_assignment_entropy']
                if role_assignment_entropy is None or camera_mask is None:
                    raise ValueError(
                        'contact-risk supervision requires visual role assignment entropy'
                    )
                camera_role_valid = (
                    camera_mask[:, :, None].astype(jnp.bool_)
                    & role_valid_mask[:, None, :]
                )
                role_ambiguity = jnp.sum(
                    role_assignment_entropy.astype(jnp.float32)
                    * camera_role_valid.astype(jnp.float32),
                    axis=1,
                ) / jnp.maximum(
                    jnp.sum(camera_role_valid.astype(jnp.float32), axis=1), 1.0
                )
                contact_phases = self.contact_phase_targets(
                    actions[:, replan_index], current_observation.state
                )
                contact_event = jnp.any(
                    (contact_phases == 1) | (contact_phases == 3), axis=-1
                )
                risk_targets = jnp.where(
                    contact_event[:, None],
                    jnp.where(role_ambiguity > 0.35, 2, 1),
                    0,
                ).astype(jnp.int32)
                risk_logits = contact_risk_state['risk_logits'].astype(jnp.float32)
                risk_log_probabilities = jax.nn.log_softmax(risk_logits, axis=-1)
                selected_risk_log_probability = jnp.take_along_axis(
                    risk_log_probabilities, risk_targets[..., None], axis=-1
                )[..., 0]
                class_weights = jnp.asarray(
                    self.contact_risk_class_weights, jnp.float32
                )
                risk_weights = (
                    jnp.take(class_weights, risk_targets, axis=0)
                    * role_valid_mask.astype(jnp.float32)
                )
                contact_risk_loss = -jnp.sum(
                    selected_risk_log_probability * risk_weights, axis=-1
                ) / jnp.maximum(jnp.sum(risk_weights, axis=-1), 1.0)
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'contact_risk'
                ] * contact_risk_loss

            relational_role_state = private[
                'relational_role_composer_residual_state'
            ]
            if relational_role_state is not None:
                factorized_valid = current_observation.factorized_auxiliary_valid
                source_labels = (
                    current_observation.factorized_source_relation_label
                )
                destination_labels = (
                    current_observation.factorized_destination_relation_label
                )
                if (
                    factorized_valid is None
                    or source_labels is None
                    or destination_labels is None
                ):
                    raise ValueError(
                        'relational role composer requires current-clause '
                        'source/destination relation labels'
                    )
                relation_valid = (
                    factorized_valid.astype(jnp.bool_)
                    & valid
                    & (source_labels >= 0)
                    & (destination_labels >= 0)
                    & jnp.all(role_valid_mask, axis=-1)
                )
                source_safe = jnp.clip(source_labels, 0, 3).astype(jnp.int32)
                destination_safe = jnp.clip(
                    destination_labels, 0, 3
                ).astype(jnp.int32)
                relation_targets = source_safe * 4 + destination_safe
                relation_logits = relational_role_state[
                    'relation_logits'
                ].astype(jnp.float32)
                relation_log_probabilities = jax.nn.log_softmax(
                    relation_logits, axis=-1
                )
                selected_relation_log_probability = jnp.take_along_axis(
                    relation_log_probabilities,
                    relation_targets[:, None],
                    axis=-1,
                )[:, 0]
                relation_class_weights = jnp.asarray(
                    self.relational_role_class_weights, jnp.float32
                )
                relation_weights = (
                    jnp.take(
                        relation_class_weights, relation_targets, axis=0
                    )
                    * relation_valid.astype(jnp.float32)
                )
                relational_role_loss = -(
                    selected_relation_log_probability * relation_weights
                )
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'relational_role'
                ] * relational_role_loss

            clause_role_binding_state = private[
                'clause_role_binding_verifier_state'
            ]
            if clause_role_binding_state is not None:
                factorized_valid = current_observation.factorized_auxiliary_valid
                source_labels = (
                    current_observation.factorized_source_relation_label
                )
                destination_labels = (
                    current_observation.factorized_destination_relation_label
                )
                if (
                    factorized_valid is None
                    or source_labels is None
                    or destination_labels is None
                ):
                    raise ValueError(
                        'clause-role binding verifier requires current-clause '
                        'source/destination relation labels'
                    )
                relation_valid = (
                    factorized_valid.astype(jnp.bool_)
                    & valid
                    & (source_labels >= 0)
                    & (destination_labels >= 0)
                    & jnp.all(role_valid_mask, axis=-1)
                )
                source_safe = jnp.clip(source_labels, 0, 3).astype(jnp.int32)
                destination_safe = jnp.clip(
                    destination_labels, 0, 3
                ).astype(jnp.int32)
                source_relation_logits = clause_role_binding_state[
                    'source_relation_logits'
                ].astype(jnp.float32)
                destination_relation_logits = clause_role_binding_state[
                    'destination_relation_logits'
                ].astype(jnp.float32)
                grounded_source_relation_logits = clause_role_binding_state[
                    'grounded_source_relation_logits'
                ].astype(jnp.float32)
                grounded_destination_relation_logits = clause_role_binding_state[
                    'grounded_destination_relation_logits'
                ].astype(jnp.float32)
                source_log_probabilities = jax.nn.log_softmax(
                    source_relation_logits, axis=-1
                )
                destination_log_probabilities = jax.nn.log_softmax(
                    destination_relation_logits, axis=-1
                )
                grounded_source_log_probabilities = jax.nn.log_softmax(
                    grounded_source_relation_logits, axis=-1
                )
                grounded_destination_log_probabilities = jax.nn.log_softmax(
                    grounded_destination_relation_logits, axis=-1
                )
                selected_source_log_probability = jnp.take_along_axis(
                    source_log_probabilities,
                    source_safe[:, None],
                    axis=-1,
                )[:, 0]
                selected_destination_log_probability = jnp.take_along_axis(
                    destination_log_probabilities,
                    destination_safe[:, None],
                    axis=-1,
                )[:, 0]
                selected_grounded_source_log_probability = jnp.take_along_axis(
                    grounded_source_log_probabilities,
                    source_safe[:, None],
                    axis=-1,
                )[:, 0]
                selected_grounded_destination_log_probability = (
                    jnp.take_along_axis(
                        grounded_destination_log_probabilities,
                        destination_safe[:, None],
                        axis=-1,
                    )[:, 0]
                )
                source_class_weights = jnp.asarray(
                    self.clause_role_binding_source_class_weights, jnp.float32
                )
                destination_class_weights = jnp.asarray(
                    self.clause_role_binding_destination_class_weights,
                    jnp.float32,
                )
                source_weights = jnp.take(
                    source_class_weights, source_safe, axis=0
                )
                destination_weights = jnp.take(
                    destination_class_weights, destination_safe, axis=0
                )
                clause_role_binding_loss = -0.25 * (
                    selected_source_log_probability * source_weights
                    + selected_destination_log_probability * destination_weights
                    + selected_grounded_source_log_probability * source_weights
                    + selected_grounded_destination_log_probability
                    * destination_weights
                ) * relation_valid.astype(jnp.float32)
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'clause_role_binding'
                ] * clause_role_binding_loss

            normalized_roles = _rms_normalize(private['bound_roles']).astype(
                jnp.float32
            )
            role_cosine = jnp.sum(
                normalized_roles[:, 0] * normalized_roles[:, 1], axis=-1
            ) / float(self.persistent_memory.hidden_dim)
            both_roles_valid = role_valid_mask[:, 0] & role_valid_mask[:, 1]
            role_distinctness = jnp.square(jax.nn.relu(role_cosine - 0.25))
            auxiliary = auxiliary + self.persistent_memory_loss_weights[
                'role_distinctness'
            ] * role_distinctness * both_roles_valid.astype(jnp.float32)

            role_object_overlap = private['role_object_overlap']
            if role_object_overlap is not None and camera_mask is not None:
                overlap_valid = (
                    camera_mask.astype(jnp.bool_)
                    & both_roles_valid[:, None]
                )
                role_object_exclusivity = jnp.sum(
                    role_object_overlap
                    * overlap_valid.astype(role_object_overlap.dtype),
                    axis=1,
                ) / jnp.maximum(jnp.sum(overlap_valid, axis=1), 1)
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'role_object_exclusivity'
                ] * role_object_exclusivity
            source_reference_overlap = private['source_reference_overlap']
            source_reference_valid = private['source_reference_valid'].astype(
                jnp.bool_
            )
            if source_reference_overlap is not None and camera_mask is not None:
                source_overlap_valid = (
                    camera_mask.astype(jnp.bool_)
                    & source_reference_valid[:, None]
                )
                source_reference_exclusivity = jnp.sum(
                    source_reference_overlap
                    * source_overlap_valid.astype(source_reference_overlap.dtype),
                    axis=1,
                ) / jnp.maximum(jnp.sum(source_overlap_valid, axis=1), 1)
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'role_object_exclusivity'
                ] * source_reference_exclusivity
            destination_reference_overlap = private[
                'destination_reference_overlap'
            ]
            destination_reference_valid = private[
                'destination_reference_valid'
            ].astype(jnp.bool_)
            if (
                destination_reference_overlap is not None
                and camera_mask is not None
            ):
                destination_overlap_valid = (
                    camera_mask.astype(jnp.bool_)
                    & jnp.any(destination_reference_valid, axis=-1)[:, None]
                )
                destination_reference_exclusivity = jnp.sum(
                    destination_reference_overlap
                    * destination_overlap_valid.astype(
                        destination_reference_overlap.dtype
                    ),
                    axis=1,
                ) / jnp.maximum(
                    jnp.sum(destination_overlap_valid, axis=1), 1
                )
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'role_object_exclusivity'
                ] * destination_reference_exclusivity
            condition_state_overlap = private['condition_state_overlap']
            condition_state_valid = private['condition_state_valid'].astype(
                jnp.bool_
            )
            if condition_state_overlap is not None and camera_mask is not None:
                condition_overlap_valid = (
                    camera_mask.astype(jnp.bool_)
                    & condition_state_valid[:, None]
                )
                condition_state_exclusivity = jnp.sum(
                    condition_state_overlap
                    * condition_overlap_valid.astype(
                        condition_state_overlap.dtype
                    ),
                    axis=1,
                ) / jnp.maximum(
                    jnp.sum(condition_overlap_valid, axis=1), 1
                )
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'role_object_exclusivity'
                ] * condition_state_exclusivity

            factorized_valid = current_observation.factorized_auxiliary_valid
            if factorized_valid is not None:
                factorized_valid = factorized_valid.astype(jnp.bool_) & valid
                destination_labels = (
                    current_observation.factorized_destination_relation_label
                )
                role_assignment_entropy = private['role_assignment_entropy']
                if (
                    destination_labels is not None
                    and role_assignment_entropy is not None
                    and camera_mask is not None
                ):
                    between_valid = (
                        factorized_valid
                        & (destination_labels == 3)
                        & role_valid_mask[:, 1]
                    )
                    pair_camera_valid = (
                        camera_mask.astype(jnp.bool_)
                        & between_valid[:, None]
                    )
                    two_slot_entropy = math.log(2.0) / math.log(
                        float(self.persistent_memory.object_count)
                    )
                    pair_entropy_error = jnp.square(
                        role_assignment_entropy[:, :, 1] - two_slot_entropy
                    )
                    pair_entropy_loss = jnp.sum(
                        pair_entropy_error
                        * pair_camera_valid.astype(pair_entropy_error.dtype),
                        axis=1,
                    ) / jnp.maximum(jnp.sum(pair_camera_valid, axis=1), 1)
                    auxiliary = auxiliary + self.persistent_memory_loss_weights[
                        'between_pair_entropy'
                    ] * pair_entropy_loss
                teacher_roles = _rms_normalize(
                    private['role_span_teacher']
                ).astype(jnp.float32)
                language_roles = _rms_normalize(
                    private['language_roles']
                ).astype(jnp.float32)
                span_alignment = jnp.mean(
                    jnp.square(language_roles - teacher_roles), axis=-1
                )
                span_alignment = jnp.sum(
                    span_alignment * role_valid_mask.astype(jnp.float32), axis=-1
                ) / jnp.maximum(jnp.sum(role_valid_mask, axis=-1), 1)

                compact_loss = jnp.zeros((batch_size,), dtype=jnp.float32)
                compact_count = jnp.zeros((batch_size,), dtype=jnp.float32)
                compact_supervision = (
                    (
                        private['operation_logits'],
                        current_observation.factorized_operation_label,
                        self.persistent_memory_factorized_class_counts['operation'],
                    ),
                    (
                        private['source_relation_logits'],
                        current_observation.factorized_source_relation_label,
                        self.persistent_memory_factorized_class_counts[
                            'source_relation'
                        ],
                    ),
                    (
                        private['destination_relation_logits'],
                        current_observation.factorized_destination_relation_label,
                        self.persistent_memory_factorized_class_counts[
                            'destination_relation'
                        ],
                    ),
                    (
                        private['condition_logits'],
                        current_observation.factorized_condition_label,
                        self.persistent_memory_factorized_class_counts['condition'],
                    ),
                    (
                        private['destination_qualifier_logits'],
                        current_observation.factorized_destination_qualifier_label,
                        self.persistent_memory_factorized_class_counts[
                            'destination_qualifier'
                        ],
                    ),
                )
                for logits, labels, class_counts in compact_supervision:
                    if labels is None:
                        continue
                    if len(class_counts) != logits.shape[-1]:
                        raise ValueError(
                            'factorized class counts do not match classifier width'
                        )
                    label_valid = factorized_valid & (labels >= 0)
                    safe_labels = jnp.clip(labels, 0, logits.shape[-1] - 1)
                    selected = jnp.take_along_axis(
                        jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
                        safe_labels[:, None],
                        axis=-1,
                    )[:, 0]
                    class_weights = (
                        self.persistent_memory.balanced_factorized_class_weights(
                            class_counts
                        )
                    )
                    selected_class_weight = jnp.take(
                        class_weights, safe_labels, axis=0
                    )
                    compact_loss = compact_loss - (
                        selected
                        * selected_class_weight
                        * label_valid.astype(jnp.float32)
                    )
                    compact_count = compact_count + label_valid.astype(jnp.float32)
                compact_loss = compact_loss / jnp.maximum(compact_count, 1.0)
                factorized_loss = span_alignment + compact_loss
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'factorized_role'
                ] * factorized_loss * factorized_valid.astype(jnp.float32)
                factor_attention_alignment = private[
                    'factor_attention_alignment_loss'
                ]
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'factor_attention_alignment'
                ] * factor_attention_alignment * factorized_valid.astype(
                    jnp.float32
                )
                object_slot_reconstruction_loss = private[
                    'object_slot_reconstruction_loss'
                ]
                if object_slot_reconstruction_loss is not None:
                    auxiliary = auxiliary + self.persistent_memory_loss_weights[
                        'object_slot_reconstruction'
                    ] * object_slot_reconstruction_loss * factorized_valid.astype(
                        jnp.float32
                    )

                source_language = _rms_normalize(
                    private['source_reference_language']
                ).astype(jnp.float32)
                source_teacher = _rms_normalize(
                    private['source_reference_teacher']
                ).astype(jnp.float32)
                source_bound = _rms_normalize(
                    private['source_reference_bound']
                ).astype(jnp.float32)
                source_span_alignment = jnp.mean(
                    jnp.square(source_language - source_teacher), axis=-1
                )
                source_visual_alignment = jnp.mean(
                    jnp.square(
                        source_bound - jax.lax.stop_gradient(source_language)
                    ),
                    axis=-1,
                )
                source_reference_loss = (
                    source_span_alignment + source_visual_alignment
                ) * source_reference_valid.astype(jnp.float32)
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'source_reference'
                ] * source_reference_loss * factorized_valid.astype(jnp.float32)

                destination_language = _rms_normalize(
                    private['destination_reference_language']
                ).astype(jnp.float32)
                destination_teacher = _rms_normalize(
                    private['destination_reference_teacher']
                ).astype(jnp.float32)
                destination_bound = _rms_normalize(
                    private['destination_reference_bound']
                ).astype(jnp.float32)
                destination_span_alignment = jnp.mean(
                    jnp.square(destination_language - destination_teacher),
                    axis=-1,
                )
                destination_visual_alignment = jnp.mean(
                    jnp.square(
                        destination_bound
                        - jax.lax.stop_gradient(destination_language)
                    ),
                    axis=-1,
                )
                destination_reference_loss = jnp.sum(
                    (
                        destination_span_alignment
                        + destination_visual_alignment
                    )
                    * destination_reference_valid.astype(jnp.float32),
                    axis=-1,
                ) / jnp.maximum(
                    jnp.sum(
                        destination_reference_valid, axis=-1
                    ).astype(jnp.float32),
                    1.0,
                )
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'destination_reference'
                ] * destination_reference_loss * factorized_valid.astype(
                    jnp.float32
                )

                condition_language = _rms_normalize(
                    private['condition_state_language']
                ).astype(jnp.float32)
                condition_teacher = _rms_normalize(
                    private['condition_state_teacher']
                ).astype(jnp.float32)
                condition_bound = _rms_normalize(
                    private['condition_state_bound']
                ).astype(jnp.float32)
                condition_span_alignment = jnp.mean(
                    jnp.square(condition_language - condition_teacher), axis=-1
                )
                condition_visual_alignment = jnp.mean(
                    jnp.square(
                        condition_bound
                        - jax.lax.stop_gradient(condition_language)
                    ),
                    axis=-1,
                )
                condition_state_loss = (
                    condition_span_alignment + condition_visual_alignment
                ) * condition_state_valid.astype(jnp.float32)
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'condition_state'
                ] * condition_state_loss * factorized_valid.astype(jnp.float32)

                destination_qualifier_language = _rms_normalize(
                    private['destination_qualifier_language']
                ).astype(jnp.float32)
                destination_qualifier_teacher = _rms_normalize(
                    private['destination_qualifier_teacher']
                ).astype(jnp.float32)
                destination_qualifier_valid = private[
                    'destination_qualifier_valid'
                ].astype(jnp.float32)
                destination_qualifier_loss = jnp.mean(
                    jnp.square(
                        destination_qualifier_language
                        - destination_qualifier_teacher
                    ),
                    axis=-1,
                ) * destination_qualifier_valid
                auxiliary = auxiliary + self.persistent_memory_loss_weights[
                    'destination_qualifier'
                ] * destination_qualifier_loss * factorized_valid.astype(
                    jnp.float32
                )

                if (
                    private['camera_bound_roles'] is not None
                    and private['camera_role_weights'] is not None
                ):
                    visual_language_role_loss = (
                        self.persistent_memory.visual_language_role_contrastive_loss(
                            private['language_roles'],
                            private['camera_bound_roles'],
                            private['camera_role_weights'],
                            role_valid_mask,
                        )
                    )
                    auxiliary = auxiliary + self.persistent_memory_loss_weights[
                        'visual_language_role'
                    ] * visual_language_role_loss * factorized_valid.astype(
                        jnp.float32
                    )

                identity_labels = (
                    current_observation.factorized_role_identity_labels
                )
                if identity_labels is not None:
                    identity_valid = (
                        role_valid_mask
                        & factorized_valid[:, None]
                        & (identity_labels >= 0)
                    )
                    normalized_identity_anchors = _rms_normalize(
                        private['role_identity_anchors']
                    ).astype(jnp.float32)
                    per_example_contrastive = (
                        _persistent_role_identity_contrastive_loss(
                            normalized_roles,
                            normalized_identity_anchors,
                            identity_valid,
                            identity_labels,
                            max_group_size=(
                                self.persistent_memory_role_contrastive_group_size
                            ),
                        )
                    )
                    auxiliary = auxiliary + self.persistent_memory_loss_weights[
                        'role_contrastive'
                    ] * per_example_contrastive
            next_bound_roles = jnp.where(
                valid[:, None, None],
                private['bound_roles'].astype(previous_bound_roles.dtype),
                previous_bound_roles,
            )
            next_role_valid = role_valid | valid
            auxiliary_replan_weight = (
                valid.astype(jnp.float32)
                * float(replan_count)
                / valid_count.astype(jnp.float32)
            )
            flow_replan_weight = (
                valid.astype(jnp.float32)
                * flow_index_mask[replan_index].astype(jnp.float32)
                * float(replan_count)
                / selected_flow_count.astype(jnp.float32)
            )
            loss = (
                parent_loss
                * flow_replan_weight[:, None]
                * flow_phase_weight[:, None]
                + auxiliary_multiplier
                * auxiliary[:, None]
                * auxiliary_replan_weight[:, None]
                + spatial_language_auxiliary[:, None]
                * auxiliary_replan_weight[:, None]
            )
            return (
                next_memory,
                next_frontier,
                next_bound_roles,
                next_role_valid,
                next_hetm_state,
                next_racg_state,
            ), loss

        # Statically unroll the fixed eight-position architecture so no FSDP
        # parameter or activation resharding is nested under XLA scan/cond.
        # Checkpoint each complete body to retain only the causal carry during
        # backward and recompute one replan's large activations at a time.
        rematerialized_active_replan = jax.checkpoint(
            compute_active_replan, static_argnums=(1,)
        )
        carry = (
            current_memory,
            current_frontier,
            initial_bound_roles,
            initial_role_valid,
            current_hetm_state,
            current_racg_state,
        )
        losses = []
        for replan_index in range(replan_count):
            carry, replan_loss = rematerialized_active_replan(
                carry, replan_index
            )
            losses.append(replan_loss)
        return jnp.stack(losses, axis=1)

    def compute_loss_joint_psm_hetm_sequence(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: at.Float[at.Array, 'b r ah ad'],
        *,
        training_step: int | at.Int[at.Array, ''],
        train: bool = False,
    ):
        """Advance Direct PSM and HETM in one causal eight-replan unroll."""
        if not hasattr(self, 'persistent_memory') or not hasattr(self, 'hetm'):
            raise ValueError('joint sequence loss requires Direct PSM and HETM')
        return self.compute_loss_sequence(
            rng,
            observation,
            actions,
            training_step=training_step,
            train=train,
        )

    def _hetm_state_from_observation(
        self,
        observation: _model.Observation,
        prefix_states,
        prefix_mask,
    ):
        """Advance the private HETM state exactly once for this observation."""
        if not hasattr(self, 'hetm'):
            return None
        batch_size = observation.state.shape[0]
        clean = _hetm.initial_state(batch_size)
        supplied = {
            'event_ledger': observation.hetm_event_ledger,
            'event_valid': observation.hetm_event_valid,
            'event_write_index': observation.hetm_event_write_index,
            'last_event_probabilities': observation.hetm_last_event_probabilities,
            'predicate_memory': observation.hetm_predicate_memory,
            'predicate_probabilities': observation.hetm_predicate_probabilities,
            'frontier': observation.hetm_frontier,
        }
        supplied_count = sum(value is not None for value in supplied.values())
        if supplied_count not in (0, len(supplied)):
            raise ValueError('HETM recurrent state must be supplied atomically')
        state = clean if supplied_count == 0 else supplied
        previous_actions = observation.hetm_previous_actions
        if previous_actions is None:
            previous_actions = observation.persistent_previous_actions
        if previous_actions is None:
            previous_actions = jnp.zeros(
                (
                    batch_size,
                    _hetm.PREVIOUS_ACTION_STEPS,
                    _hetm.ACTIVE_ACTION_DIM,
                ),
                dtype=jnp.float32,
            )
        else:
            previous_actions = previous_actions[
                :, : _hetm.PREVIOUS_ACTION_STEPS, : _hetm.ACTIVE_ACTION_DIM
            ]
        episode_start = observation.hetm_episode_start
        if episode_start is None:
            episode_start = observation.persistent_memory_episode_start
        if episode_start is None:
            episode_start = jnp.ones((batch_size,), dtype=jnp.bool_)
        outputs, next_state = self.hetm(
            prefix_states,
            prefix_mask,
            observation.state,
            previous_actions,
            state,
            episode_start,
        )
        return {'outputs': outputs, 'next_state': next_state}

    def _persistent_memory_action_view(self, previous_actions):
        """Project ordered dual-arm actions without changing memory weights."""
        expected = self.persistent_memory.previous_action_dim
        if previous_actions.shape[-1] == expected:
            return previous_actions
        if previous_actions.shape[-1] != 2 * expected:
            raise ValueError(
                'persistent previous actions cannot be projected to the '
                f'audited memory width: got {previous_actions.shape[-1]}, '
                f'expected {expected} or {2 * expected}'
            )
        # ARX-X5 stores the ordered left and right seven-joint chunks as
        # [left_0..left_6, right_0..right_6].  Their symmetric mean keeps the
        # inherited 7-D memory parameter tree exact while making both arms
        # causally visible; slicing the first seven would silently turn the
        # recurrent policy into a left-arm-only model.
        return 0.5 * (
            previous_actions[..., :expected]
            + previous_actions[..., expected : 2 * expected]
        )

    def _persistent_state_from_observation(
        self,
        observation,
        prefix_tokens,
        prefix_mask,
        *,
        compute_object_reconstruction=False,
    ):
        if not hasattr(self, 'persistent_memory'):
            return None
        batch_size = observation.state.shape[0]
        memory = observation.persistent_memory_initial_state
        if memory is None:
            memory = self.persistent_memory.initial_state(
                batch_size, dtype=prefix_tokens.dtype
            )
        frontier = observation.persistent_subgoal_initial_frontier
        if frontier is None:
            frontier = self.persistent_memory.initial_frontier(batch_size)
        previous_actions = observation.persistent_previous_actions
        if previous_actions is None:
            previous_actions = jnp.zeros(
                (batch_size, 5, self.persistent_memory.previous_action_dim),
                dtype=observation.state.dtype,
            )
        else:
            previous_actions = self._persistent_memory_action_view(
                previous_actions
            )
        if previous_actions.ndim != 3:
            raise ValueError(
                'persistent inference previous actions must be [batch, 5, action]'
            )
        previous_actions_valid = observation.persistent_previous_actions_valid
        if previous_actions_valid is None:
            previous_actions_valid = jnp.zeros((batch_size,), dtype=jnp.bool_)
        if previous_actions_valid.ndim != 1:
            raise ValueError(
                'persistent inference previous-action validity must be [batch]'
            )
        episode_start = observation.persistent_memory_episode_start
        if episode_start is None:
            episode_start = jnp.ones((batch_size,), dtype=jnp.bool_)
        if episode_start.ndim != 1:
            raise ValueError(
                'persistent inference episode-start marker must be [batch]'
            )
        role_span_mask = observation.factorized_role_span_mask
        source_reference_span_mask = (
            observation.factorized_source_reference_span_mask
        )
        destination_reference_span_mask = (
            observation.factorized_destination_reference_span_mask
        )
        condition_state_span_mask = observation.factorized_condition_span_mask
        destination_qualifier_span_mask = (
            observation.factorized_destination_qualifier_span_mask
        )
        role_valid_mask = observation.factorized_role_valid_mask
        clause_span_mask = observation.clause_span_mask
        clause_valid_mask = observation.clause_valid_mask
        prompt_length = (
            observation.tokenized_prompt.shape[-1]
            if observation.tokenized_prompt is not None
            else 0
        )
        image_token_count = prefix_tokens.shape[1] - prompt_length
        plan_token_mask = None
        if observation.tokenized_prompt_mask is not None:
            if observation.tokenized_prompt_mask.shape != (
                batch_size,
                prompt_length,
            ):
                raise ValueError(
                    'tokenized prompt mask must be [batch, prompt] for planning'
                )
            plan_token_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, image_token_count), dtype=jnp.bool_
                    ),
                    observation.tokenized_prompt_mask,
                ],
                axis=-1,
            )
        if role_span_mask is not None:
            if role_span_mask.shape != (batch_size, 2, prompt_length):
                raise ValueError(
                    'factorized inference role spans must be [batch, 2, prompt]'
                )
            role_span_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, 2, image_token_count), dtype=jnp.bool_
                    ),
                    role_span_mask,
                ],
                axis=-1,
            )
        if source_reference_span_mask is not None:
            if source_reference_span_mask.shape != (
                batch_size,
                prompt_length,
            ):
                raise ValueError(
                    'factorized source-reference spans must be [batch, prompt]'
                )
            source_reference_span_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, image_token_count), dtype=jnp.bool_
                    ),
                    source_reference_span_mask,
                ],
                axis=-1,
            )
        if destination_reference_span_mask is not None:
            if destination_reference_span_mask.shape != (
                batch_size,
                2,
                prompt_length,
            ):
                raise ValueError(
                    'factorized destination-reference spans must be '
                    '[batch, 2, prompt]'
                )
            destination_reference_span_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, 2, image_token_count), dtype=jnp.bool_
                    ),
                    destination_reference_span_mask,
                ],
                axis=-1,
            )
        if condition_state_span_mask is not None:
            if condition_state_span_mask.shape != (
                batch_size,
                prompt_length,
            ):
                raise ValueError(
                    'factorized condition-state spans must be [batch, prompt]'
                )
            condition_state_span_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, image_token_count), dtype=jnp.bool_
                    ),
                    condition_state_span_mask,
                ],
                axis=-1,
            )
        if destination_qualifier_span_mask is not None:
            if destination_qualifier_span_mask.shape != (
                batch_size,
                prompt_length,
            ):
                raise ValueError(
                    'factorized destination-qualifier spans must be '
                    '[batch, prompt]'
                )
            destination_qualifier_span_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, image_token_count), dtype=jnp.bool_
                    ),
                    destination_qualifier_span_mask,
                ],
                axis=-1,
            )
        if (clause_span_mask is None) != (clause_valid_mask is None):
            raise ValueError('ClausePlan span and validity masks must be paired')
        if clause_span_mask is not None:
            clause_slots = getattr(
                getattr(self.persistent_memory, 'clause_plan_adapter', None),
                'clause_slots',
                clause_span_mask.shape[1],
            )
            if clause_span_mask.shape != (
                batch_size,
                clause_slots,
                prompt_length,
            ):
                raise ValueError(
                    'ClausePlan spans must be [batch, clause, prompt]'
                )
            if clause_valid_mask.shape != (batch_size, clause_slots):
                raise ValueError(
                    'ClausePlan validity must be [batch, clause]'
                )
            clause_span_mask = jnp.concatenate(
                [
                    jnp.zeros(
                        (batch_size, clause_slots, image_token_count),
                        dtype=jnp.bool_,
                    ),
                    clause_span_mask,
                ],
                axis=-1,
            )
        camera_count = len(observation.images)
        visual_tokens = None
        camera_mask = None
        if camera_count:
            if image_token_count % camera_count:
                raise ValueError('image prefix cannot be partitioned by camera')
            visual_tokens = prefix_tokens[:, :image_token_count].reshape(
                batch_size,
                camera_count,
                image_token_count // camera_count,
                prefix_tokens.shape[-1],
            )
            camera_mask = jnp.stack(
                [
                    self._physical_image_mask(observation, name)
                    for name in observation.images
                ],
                axis=1,
            )
        structured_demo = getattr(self.persistent_memory, 'structured_demo', None)
        structured_demo_tokens = None
        structured_demo_token_mask = None
        if structured_demo is not None:
            demo_fields = (
                observation.demonstration_tokenized_prompt,
                observation.demonstration_tokenized_prompt_mask,
                observation.demonstration_semantic_span_mask,
                observation.demonstration_semantic_valid_mask,
                observation.demonstration_plan,
                observation.demonstration_actions,
                observation.demonstration_context_mask,
                observation.demonstration_trajectory_mask,
            )
            supplied = tuple(value is not None for value in demo_fields)
            if any(supplied) and not all(supplied):
                raise ValueError('PSM-SDLA observation fields are incomplete')
            if all(supplied):
                (
                    demo_prompt_ids,
                    demo_prompt_mask,
                    demo_span_mask,
                    demo_semantic_valid,
                    demo_plan,
                    demo_actions,
                    demo_context_mask,
                    demo_trajectory_mask,
                ) = demo_fields
                if jnp.ndim(demo_context_mask) != 1:
                    raise ValueError('PSM-SDLA context mask must be [batch]')
                if jnp.ndim(demo_trajectory_mask) != 1:
                    raise ValueError('PSM-SDLA trajectory mask must be [batch]')
                # L0/L1/L2 admission is derived by the manifest-bound data
                # transform.  This second guard prevents any trajectory from
                # bypassing an absent semantic context.
                if not isinstance(demo_trajectory_mask, jax.core.Tracer):
                    if bool(jnp.any(demo_trajectory_mask & ~demo_context_mask)):
                        raise ValueError(
                            'demonstration trajectory cannot bypass context'
                        )
                frozen_demo_embeddings = jax.lax.stop_gradient(
                    self.PaliGemma.llm(demo_prompt_ids, method='embed')
                )
                demo_semantics = structured_demo.pool_semantics(
                    frozen_demo_embeddings,
                    demo_prompt_mask,
                    demo_span_mask,
                    demo_semantic_valid,
                )
                structured_demo_tokens, structured_demo_token_mask = (
                    structured_demo.encode(
                        demo_semantics,
                        demo_semantic_valid,
                        demo_plan,
                        demo_actions,
                        demo_context_mask,
                        demo_trajectory_mask,
                    )
                )
        return self.persistent_memory.infer_state(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            state=observation.state,
            memory=memory,
            frontier=frontier,
            previous_actions=previous_actions,
            previous_actions_valid=previous_actions_valid,
            episode_start=episode_start,
            plan_token_mask=plan_token_mask,
            role_span_mask=role_span_mask,
            source_reference_span_mask=source_reference_span_mask,
            destination_reference_span_mask=destination_reference_span_mask,
            condition_state_span_mask=condition_state_span_mask,
            destination_qualifier_span_mask=(
                destination_qualifier_span_mask
            ),
            role_valid_mask=role_valid_mask,
            clause_span_mask=clause_span_mask,
            clause_valid_mask=clause_valid_mask,
            visual_tokens=visual_tokens,
            camera_mask=camera_mask,
            compute_object_reconstruction=compute_object_reconstruction,
            structured_demo=(
                structured_demo
                if structured_demo_tokens is not None
                else None
            ),
            structured_demo_tokens=structured_demo_tokens,
            structured_demo_token_mask=structured_demo_token_mask,
        )

    def sample_actions_with_persistent_state(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        **kwargs,
    ):
        """Return actions plus private next state for transactional serving."""
        if not hasattr(self, 'persistent_memory'):
            raise ValueError('persistent memory is not enabled for this model')
        actions, private = self._sample_actions_impl(
            rng,
            observation,
            return_persistent_state=True,
            **kwargs,
        )
        if private is None:
            raise RuntimeError('persistent inference produced no private state')
        return {
            'actions': actions,
            'persistent_memory': private['memory'],
            'persistent_subgoal_frontier': private['frontier'],
            'persistent_progress': private['progress'],
            'persistent_next_action_summary': private['next_action_summary'],
        }

    def sample_actions_with_persistent_hetm_state(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        **kwargs,
    ):
        """Return actions and one atomic Direct-PSM + HETM next state."""
        if not hasattr(self, 'persistent_memory') or not hasattr(self, 'hetm'):
            raise ValueError('joint sampler requires Direct PSM and HETM')
        if hasattr(self, 'racg'):
            raise ValueError(
                'PSM+HETM+RACG requires the three-way atomic sampler'
            )
        actions, private = self._sample_actions_impl(
            rng,
            observation,
            return_persistent_state=True,
            **kwargs,
        )
        if private is None or private.get('hetm_next_state') is None:
            raise RuntimeError('joint inference produced incomplete private state')
        return {
            'actions': actions,
            'persistent_memory': private['memory'],
            'persistent_subgoal_frontier': private['frontier'],
            **private['hetm_next_state'],
        }

    def sample_actions_with_persistent_hetm_racg_state(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        **kwargs,
    ):
        """Return actions and one atomic Direct-PSM + HETM + RACG state."""
        if not (
            hasattr(self, 'persistent_memory')
            and hasattr(self, 'hetm')
            and hasattr(self, 'racg')
        ):
            raise ValueError('three-way sampler requires PSM, HETM and RACG')
        actions, private = self._sample_actions_impl(
            rng,
            observation,
            return_persistent_state=True,
            **kwargs,
        )
        if (
            private is None
            or private.get('hetm_next_state') is None
            or private.get('racg_next_state') is None
        ):
            raise RuntimeError('three-way inference produced incomplete state')
        return {
            'actions': actions,
            'persistent_memory': private['memory'],
            'persistent_subgoal_frontier': private['frontier'],
            **private['hetm_next_state'],
            **private['racg_next_state'],
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ''] = 10,
        noise: at.Float[at.Array, 'b ah ad'] | None = None,
    ) -> _model.Actions:
        return self._sample_actions_impl(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            return_persistent_state=False,
        )

    def _sample_actions_impl(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ''] = 10,
        noise: at.Float[at.Array, 'b ah ad'] | None = None,
        return_persistent_state: bool,
    ):
        persistent = getattr(self, 'persistent_memory', None)
        geometry_module = getattr(persistent, 'geometry_aux_v3', None)
        if geometry_module is not None and (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps != 10
        ):
            raise ValueError('geometry-v1 formal main ODE requires exactly num_steps=10')
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )
        noise = self.mask_inactive_actions(noise)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        persistent_private_state = self._persistent_state_from_observation(
            observation, prefix_tokens, prefix_mask
        )
        hmca_adapter = self._hmca_adapter_from_persistent_state(
            persistent_private_state
        )
        memory_layer_adapter = self._memory_attention_adapter_from_persistent_state(
            persistent_private_state
        )
        if hasattr(self, 'prefix_moe_router_in'):
            prefix_tokens, _, _ = self.apply_multimodal_prefix_moe(
                prefix_tokens, prefix_mask, observation
            )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        hetm_private_state = self._hetm_state_from_observation(
            observation,
            jax.lax.stop_gradient(prefix_out),
            prefix_mask,
        )
        racg_private_state = self._racg_state_from_observation(
            observation,
            jax.lax.stop_gradient(prefix_out),
            prefix_mask,
            hetm_private_state,
        )
        hmca_adapter = self._hmca_adapter_from_persistent_state(
            persistent_private_state, hetm_private_state, racg_private_state
        )
        geometry_context = self._encode_geometry_once(
            prefix_out, prefix_mask, observation
        )
        if hasattr(self, 'kv_moe_router_in'):
            kv_cache, _, _ = self.apply_layerwise_kv_moe(
                kv_cache,
                prefix_tokens,
                prefix_mask,
                observation,
            )
        action_prior_tokens = None
        action_reasoning_tokens = None
        contexts = None
        latent_future_current_visual = None
        task_progress_tokens = None
        object_affordance_slots = None
        object_future_forecast_tokens = None
        contact_phase_tokens = None
        phase_contact_condition_tokens = None
        if hasattr(self, 'action_prior_queries'):
            prior_source = (
                jax.lax.stop_gradient(prefix_out)
                if self.action_prior_contextual
                else prefix_tokens
            )
            if hasattr(self, 'action_prior_explicit_blocks'):
                contexts = self.compute_action_prior_contexts(
                    prior_source,
                    prefix_mask,
                    observation.state,
                    (
                        persistent_private_state['memory']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        persistent_private_state['ordered_subgoals']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        persistent_private_state['frontier']
                        if persistent_private_state is not None
                        else None
                    ),
                    (
                        persistent_private_state['slot_valid_mask']
                        if persistent_private_state is not None
                        else None
                    ),
                )
                contexts, layerwise_implicit_guidance = (
                    self.compute_multilayer_action_prior_contexts(
                        contexts,
                        kv_cache,
                        prefix_mask,
                        observation.state,
                        (
                            persistent_private_state['memory']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['ordered_subgoals']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['frontier']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['slot_valid_mask']
                            if persistent_private_state is not None
                            else None
                        ),
                        return_layer_guidance=True,
                    )
                )
                reference_waypoints = self.compute_detached_explicit_reference(
                    self.coarse_action_targets(noise), contexts
                )
                pathway_tokens = None
                if hasattr(self, 'action_prior_pathway_score') or hasattr(
                    self, 'action_prior_pathway_interaction_out'
                ):
                    (
                        implicit_tokens,
                        explicit_tokens,
                        _,
                        action_reasoning_tokens,
                    ) = self.dual_action_prior_token_paths(
                        contexts, reference_waypoints
                    )
                    pathway_tokens = [implicit_tokens, explicit_tokens]
                    action_prior_tokens = None
                else:
                    (
                        action_prior_tokens,
                        _,
                        action_reasoning_tokens,
                    ) = self.fuse_dual_action_prior_tokens(
                        contexts, reference_waypoints
                    )
                if layerwise_implicit_guidance is not None:
                    action_reasoning_tokens = jnp.concatenate(
                        [action_reasoning_tokens, layerwise_implicit_guidance],
                        axis=1,
                    )
                if hasattr(self, 'action_prior_demo_blocks'):
                    demo_tokens, _, _ = self.compute_retrieved_demo_tokens(
                        observation, contexts, train=False
                    )
                    if pathway_tokens is None:
                        action_prior_tokens = action_prior_tokens + demo_tokens
                    else:
                        pathway_tokens.append(demo_tokens)
                    action_reasoning_tokens = jnp.concatenate(
                        [action_reasoning_tokens, demo_tokens], axis=1
                    )
                if hasattr(self, 'action_prior_discrete_codebook'):
                    discrete_tokens, _, _ = (
                        self.compute_discrete_action_code_tokens(
                            contexts, train=False
                        )
                    )
                    if pathway_tokens is None:
                        action_prior_tokens = action_prior_tokens + discrete_tokens
                    else:
                        pathway_tokens.append(discrete_tokens)
                    action_reasoning_tokens = jnp.concatenate(
                        [action_reasoning_tokens, discrete_tokens], axis=1
                    )
                if pathway_tokens is not None:
                    if hasattr(self, 'action_prior_pathway_score'):
                        action_prior_tokens, _ = self.route_reasoning_pathway_tokens(
                            pathway_tokens, contexts, observation.state
                        )
                    if hasattr(self, 'action_prior_pathway_interaction_out'):
                        interacted_tokens, interaction_residual_paths = (
                            self.interact_reasoning_pathway_tokens(
                                pathway_tokens, contexts, observation.state
                            )
                        )
                        interaction_residual = jnp.mean(
                            interaction_residual_paths, axis=2
                        )
                        if hasattr(self, 'action_prior_pathway_score'):
                            action_prior_tokens = (
                                action_prior_tokens + interaction_residual
                            )
                        else:
                            action_prior_tokens = interacted_tokens
                        if hasattr(self, 'evidence_combination_gate'):
                            interaction_gate = (
                                self.evidence_combination_content_gate(
                                    contexts,
                                    observation.state,
                                    interaction_residual,
                                    2,
                                )
                            )
                            action_prior_tokens = action_prior_tokens + (
                                interaction_gate - 1.0
                            ).astype(interaction_residual.dtype) * interaction_residual
                if hasattr(self, 'spatial_relation_blocks'):
                    spatial_tokens, _ = self.compute_spatial_relation_tokens(
                        prefix_tokens,
                        prefix_mask,
                        observation,
                        prior_source,
                    )
                    if hasattr(self, 'evidence_combination_gate'):
                        spatial_gate = self.evidence_combination_content_gate(
                            contexts,
                            observation.state,
                            spatial_tokens,
                            0,
                        )
                        spatial_tokens = spatial_tokens * (
                            spatial_gate.astype(spatial_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + spatial_tokens
                if hasattr(self, 'object_affordance_graph_blocks'):
                    (
                        object_affordance_tokens,
                        _,
                        _,
                        object_affordance_slots,
                        _,
                    ) = (
                        self.compute_object_affordance_graph_tokens(
                            prefix_tokens,
                            prefix_mask,
                            observation,
                            prior_source,
                            routing_contexts=contexts,
                        )
                    )
                    if (
                        hasattr(self, 'evidence_combination_gate')
                        and self.evidence_combination_component_count > 4
                    ):
                        object_affordance_tokens = object_affordance_tokens * (
                            self.evidence_combination_content_gate(
                                contexts,
                                observation.state,
                                object_affordance_tokens,
                                4,
                            ).astype(object_affordance_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + object_affordance_tokens
                if hasattr(self, 'masked_spatial_scene_blocks'):
                    masked_spatial_tokens, _, _, _, _, _ = (
                        self.compute_masked_spatial_tokens(
                            prefix_tokens,
                            prefix_mask,
                            observation,
                            prior_source,
                            train=False,
                        )
                    )
                    action_prior_tokens = (
                        action_prior_tokens + masked_spatial_tokens
                    )
                if hasattr(self, 'object_future_object_blocks'):
                    (
                        object_future_tokens,
                        _,
                        _,
                        _,
                        _,
                        _,
                        object_future_forecast_tokens,
                    ) = (
                        self.compute_object_future_tokens(
                            prefix_tokens,
                            prefix_mask,
                            observation,
                            prior_source,
                            contexts,
                            train=False,
                            object_affordance_slots=object_affordance_slots,
                        )
                    )
                    action_prior_tokens = (
                        action_prior_tokens + object_future_tokens
                    )
                if hasattr(self, 'predicate_binding_object_blocks'):
                    predicate_binding_tokens, _, _, _, _, _, _ = (
                        self.compute_predicate_binding_tokens(
                            prefix_tokens,
                            prefix_mask,
                            observation,
                            prior_source,
                            contexts,
                        )
                    )
                    action_prior_tokens = (
                        action_prior_tokens + predicate_binding_tokens
                    )
                if hasattr(self, 'contact_phase_blocks'):
                    contact_phase_tokens, contact_phase_logits = (
                        self.compute_contact_phase_tokens(
                            contexts, observation.state, train=False
                        )
                    )
                    if hasattr(self, 'evidence_combination_gate'):
                        contact_gate = self.evidence_combination_content_gate(
                            contexts,
                            observation.state,
                            contact_phase_tokens,
                            1,
                        )
                        contact_phase_tokens = contact_phase_tokens * (
                            contact_gate.astype(contact_phase_tokens.dtype)
                        )
                    phase_contact_condition_tokens = contact_phase_tokens
                    action_prior_tokens = action_prior_tokens + contact_phase_tokens
                    if hasattr(self, 'contact_affordance_blocks'):
                        if object_affordance_slots is None:
                            raise ValueError(
                                'contact-affordance fusion requires object slots'
                            )
                        contact_affordance_tokens, _, _ = (
                            self.compute_contact_affordance_predictive_tokens(
                                object_affordance_slots,
                                contact_phase_tokens,
                                contact_phase_logits,
                                contexts,
                                observation.state,
                                (
                                    persistent_private_state['memory']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['ordered_subgoals']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['frontier']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['clause_plan_attention']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                object_future_forecast_tokens,
                                (
                                    persistent_private_state['factorized_relation_state']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['grounded_relation_phase_state']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['bound_roles']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state[
                                        'verification_transition_probabilities'
                                    ]
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['progress']
                                    if persistent_private_state is not None
                                    else None
                                ),
                                (
                                    persistent_private_state['verification_state']
                                    if persistent_private_state is not None
                                    else None
                                ),
                            )
                        )
                        action_prior_tokens = (
                            action_prior_tokens + contact_affordance_tokens
                        )
                        phase_contact_condition_tokens = (
                            contact_phase_tokens + contact_affordance_tokens
                        )
                if hasattr(self, 'action_prior_rationale_blocks'):
                    rationale_tokens, _ = self.compute_structured_rationale_tokens(
                        prefix_out,
                        prefix_mask,
                        observation.state,
                        train=False,
                    )
                    action_prior_tokens = action_prior_tokens + rationale_tokens
                if hasattr(self, 'task_progress_blocks'):
                    progress_tokens, _, _, _ = (
                        self.compute_task_progress_tokens(
                            contexts, observation.state
                        )
                    )
                    task_progress_tokens = progress_tokens
                    if not hasattr(self, 'predictive_world_model_gate'):
                        action_prior_tokens = (
                            action_prior_tokens + progress_tokens
                        )
                if hasattr(self, 'language_subgoal_blocks'):
                    (
                        subgoal_tokens,
                        _,
                        _,
                        _,
                        subgoal_probabilities,
                        subgoal_slots,
                    ) = (
                        self.compute_language_subgoal_tokens(
                            contexts,
                            observation.state,
                            prior_source,
                            prefix_mask,
                        )
                    )
                    if (
                        hasattr(self, 'evidence_combination_gate')
                        and self.evidence_combination_component_count > 5
                    ):
                        subgoal_tokens = subgoal_tokens * (
                            self.evidence_combination_content_gate(
                                contexts,
                                observation.state,
                                subgoal_tokens,
                                5,
                            ).astype(subgoal_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + subgoal_tokens
                if hasattr(self, 'object_subgoal_binding_blocks'):
                    if object_affordance_slots is None:
                        raise ValueError(
                            'object-subgoal binding requires object slots'
                        )
                    object_subgoal_tokens, _, _, _, _ = (
                        self.compute_object_subgoal_binding_tokens(
                            object_affordance_slots,
                            subgoal_slots,
                            subgoal_probabilities,
                            contexts,
                            observation.state,
                        )
                    )
                    if (
                        hasattr(self, 'evidence_combination_gate')
                        and self.evidence_combination_component_count > 7
                    ):
                        object_subgoal_tokens = object_subgoal_tokens * (
                            self.evidence_combination_content_gate(
                                contexts,
                                observation.state,
                                object_subgoal_tokens,
                                7,
                            ).astype(object_subgoal_tokens.dtype)
                        )
                    action_prior_tokens = action_prior_tokens + object_subgoal_tokens
            else:
                if (
                    hasattr(self, 'context_adarms_in')
                    or hasattr(self, 'velocity_refiner_blocks')
                    or hasattr(self, 'language_subgoal_blocks')
                ):
                    contexts = self.compute_action_prior_contexts(
                        prior_source,
                        prefix_mask,
                        observation.state,
                        (
                            persistent_private_state['memory']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['ordered_subgoals']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['frontier']
                            if persistent_private_state is not None
                            else None
                        ),
                        (
                            persistent_private_state['slot_valid_mask']
                            if persistent_private_state is not None
                            else None
                        ),
                    )
                    coarse_tokens, _ = self._implicit_action_prior_outputs(
                        contexts
                    )
                    repeat = self.action_horizon // self.action_prior_horizon
                    action_prior_tokens = jnp.repeat(
                        coarse_tokens, repeat, axis=1
                    )
                    if hasattr(self, 'language_subgoal_blocks'):
                        subgoal_tokens, _, _, _, _, _ = (
                            self.compute_language_subgoal_tokens(
                                contexts,
                                observation.state,
                                prior_source,
                                prefix_mask,
                            )
                        )
                        action_prior_tokens = (
                            action_prior_tokens + subgoal_tokens
                        )
                else:
                    action_prior_tokens, _ = self.compute_action_prior(
                        prior_source, prefix_mask, observation.state
                    )
        if hasattr(self, 'latent_future_blocks'):
            if contexts is None:
                raise ValueError(
                    'latent future reasoning requires contextual prior states'
                )
            latent_future_current_visual = (
                self.latent_future_current_visual_tokens(
                    prefix_tokens, observation
                )
            )

        def step(carry):
            x_t, time = carry
            flow_action_prior_tokens = action_prior_tokens
            if hetm_private_state is not None:
                hetm_prior = hetm_private_state['outputs']['prior_residual'][:, None, :]
                hetm_prior = jnp.broadcast_to(
                    hetm_prior,
                    (
                        batch_size,
                        self.action_horizon,
                        hetm_prior.shape[-1],
                    ),
                )
                flow_action_prior_tokens = (
                    hetm_prior
                    if flow_action_prior_tokens is None
                    else flow_action_prior_tokens + hetm_prior
                )
            latent_future_residual = None
            state_rollout_residual = None
            action_moe_residual = None
            if hasattr(self, 'action_chunk_verifier_blocks'):
                if contexts is None:
                    raise ValueError(
                        'action chunk verifier requires contextual prior states'
                    )
                verifier_residual, _ = self.compute_action_chunk_verifier(
                    x_t, contexts, observation.state
                )
                if getattr(
                    self,
                    'evidence_combination_action_verifier_component',
                    False,
                ):
                    verifier_residual = verifier_residual * (
                        self.evidence_combination_content_gate(
                            contexts,
                            observation.state,
                            verifier_residual,
                            6,
                        ).astype(verifier_residual.dtype)
                    )
                flow_action_prior_tokens = (
                    flow_action_prior_tokens + verifier_residual
                )
            if latent_future_current_visual is not None:
                latent_future_residual, _ = self.compute_latent_future_tokens(
                    latent_future_current_visual,
                    x_t,
                    contexts,
                    observation.state,
                )
                if not hasattr(self, 'predictive_world_model_gate'):
                    flow_action_prior_tokens = (
                        flow_action_prior_tokens + latent_future_residual
                    )
            if hasattr(self, 'state_rollout_blocks'):
                if contexts is None:
                    raise ValueError(
                        'state rollout reasoning requires contextual prior states'
                    )
                state_rollout_residual, _ = self.compute_state_rollout_tokens(
                    x_t, contexts, observation.state
                )
                if not hasattr(self, 'predictive_world_model_gate'):
                    flow_action_prior_tokens = (
                        flow_action_prior_tokens + state_rollout_residual
                    )
            if hasattr(self, 'action_moe_blocks'):
                if contexts is None:
                    raise ValueError(
                        'action MoE reasoning requires contextual prior states'
                    )
                action_moe_residual, _, _, _ = self.compute_action_moe_tokens(
                    x_t, contexts, observation.state
                )
                if not getattr(
                    self, 'predictive_world_model_include_action_moe', False
                ):
                    flow_action_prior_tokens = (
                        flow_action_prior_tokens + action_moe_residual
                    )
            if hasattr(self, 'kinematic_action_blocks'):
                if contexts is None:
                    raise ValueError(
                        'kinematic action reasoning requires contextual prior states'
                    )
                kinematic_action_residual, _, _ = (
                    self.compute_kinematic_action_tokens(
                        x_t, contexts, observation.state
                    )
                )
                flow_action_prior_tokens = (
                    flow_action_prior_tokens + kinematic_action_residual
                )
            if hasattr(self, 'spectral_action_blocks'):
                if contexts is None:
                    raise ValueError(
                        'spectral action reasoning requires contextual prior states'
                    )
                spectral_action_residual, _, _ = (
                    self.compute_spectral_action_tokens(
                        x_t, contexts, observation.state
                    )
                )
                flow_action_prior_tokens = (
                    flow_action_prior_tokens + spectral_action_residual
                )
            if hasattr(self, 'predictive_world_model_gate'):
                predictive_residual, _, _ = (
                    self.fuse_predictive_world_model_tokens(
                        latent_future_residual,
                        state_rollout_residual,
                        task_progress_tokens,
                        contexts,
                        observation.state,
                        action_moe_tokens=action_moe_residual,
                        persistent_memory=(
                            persistent_private_state['memory']
                            if persistent_private_state is not None
                            else None
                        ),
                        persistent_program=(
                            persistent_private_state['ordered_subgoals']
                            if persistent_private_state is not None
                            else None
                        ),
                        persistent_frontier=(
                            persistent_private_state['frontier']
                            if persistent_private_state is not None
                            else None
                        ),
                    )
                )
                if hasattr(self, 'evidence_combination_gate'):
                    predictive_gate = self.evidence_combination_content_gate(
                        contexts,
                        observation.state,
                        predictive_residual,
                        3,
                    ).astype(predictive_residual.dtype)
                else:
                    predictive_gate = 1.0
                flow_action_prior_tokens = (
                    flow_action_prior_tokens
                    + predictive_residual * predictive_gate
                )
            racg_action_residual = None
            if racg_private_state is not None:
                racg_action_residual, _, _ = self.racg.read_actions(
                    racg_private_state['scene'],
                    self.action_in_proj(x_t),
                    jnp.broadcast_to(time, (batch_size,)),
                )
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                flow_action_prior_tokens,
                action_reasoning_tokens,
                contexts,
                (
                    persistent_private_state['memory']
                    if persistent_private_state is not None
                    else None
                ),
                (
                    hetm_private_state['outputs']
                    if hetm_private_state is not None
                    else None
                ),
                racg_action_residual,
                persistent_ordered_program=(
                    persistent_private_state['ordered_subgoals']
                    if persistent_private_state is not None
                    else None
                ),
                persistent_frontier=(
                    persistent_private_state['frontier']
                    if persistent_private_state is not None
                    else None
                ),
                contact_phase_tokens=phase_contact_condition_tokens,
                persistent_slot_valid_mask=(
                    persistent_private_state['slot_valid_mask']
                    if persistent_private_state is not None
                    else None
                ),
                persistent_previous_frontier=(
                    persistent_private_state['previous_frontier_distribution']
                    if persistent_private_state is not None else None
                ),
                persistent_previous_actions=(
                    persistent_private_state['previous_actions']
                    if persistent_private_state is not None else None
                ),
                persistent_previous_actions_valid=(
                    persistent_private_state['previous_actions_valid']
                    if persistent_private_state is not None else None
                ),
                persistent_current_context=(
                    persistent_private_state['current_context']
                    if persistent_private_state is not None else None
                ),
                persistent_previous_roles=(
                    persistent_private_state.get('prior_role_identity_anchors')
                    if persistent_private_state is not None else None
                ),
                persistent_current_roles=(
                    persistent_private_state.get('role_identity_anchors')
                    if persistent_private_state is not None else None
                ),
                persistent_role_valid_mask=(
                    persistent_private_state.get('role_valid_mask')
                    if persistent_private_state is not None else None
                ),
            )
            if persistent_private_state is not None:
                action_start = suffix_tokens.shape[1] - self.action_horizon
                injected_actions = self.persistent_memory.inject(
                    suffix_tokens[:, action_start:],
                    persistent_private_state['memory'],
                    ordered_program=persistent_private_state[
                        'ordered_subgoals'
                    ],
                    frontier=persistent_private_state['frontier'],
                )
                if persistent_private_state['structured_demo_tokens'] is not None:
                    injected_actions = (
                        self.persistent_memory.structured_demo.inject_actions(
                            injected_actions,
                            persistent_private_state['structured_demo_tokens'],
                            persistent_private_state[
                                'structured_demo_token_mask'
                            ],
                            self.persistent_memory,
                        )
                    )
                injected_actions = self._inject_geometry_after_inherited_parent(
                    injected_actions, geometry_context, policy_scale=1.0
                )
                suffix_tokens = suffix_tokens.at[:, action_start:].set(
                    injected_actions
                )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(
                prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1]
            )
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask, suffix_attn_mask], axis=-1
            )
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                action_layer_adapter=hmca_adapter,
                memory_layer_adapter=memory_layer_adapter,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if hasattr(self, 'velocity_refiner_blocks'):
                v_t, _, _ = self.compute_velocity_refinement(
                    x_t,
                    v_t,
                    suffix_out[:, -self.action_horizon :],
                    contexts,
                    observation.state,
                    jnp.broadcast_to(time, (batch_size,)),
                )
            if hasattr(self, 'action_visual_refiner_blocks'):
                v_t, _, _ = self.compute_action_visual_refinement(
                    x_t,
                    v_t,
                    suffix_out[:, -self.action_horizon :],
                    prior_source,
                    prefix_mask,
                    observation.state,
                    jnp.broadcast_to(time, (batch_size,)),
                )

            return self.mask_inactive_actions(x_t + dt * v_t), time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if return_persistent_state:
            private = dict(persistent_private_state)
            private['hetm_next_state'] = (
                hetm_private_state['next_state']
                if hetm_private_state is not None
                else None
            )
            private['racg_next_state'] = (
                racg_private_state['next_state']
                if racg_private_state is not None
                else None
            )
            return x_0, private
        return x_0
