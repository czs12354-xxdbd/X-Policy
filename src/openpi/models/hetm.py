"""Hierarchical event-transition memory for long-horizon pi0.5 policies.

The recurrent state is explicit and server-owned.  The module only consumes the
current multimodal prefix, proprioception, and actions executed before the
current observation.  Every parent-facing projection is zero initialized so a
checkpoint upgraded with HETM has exactly the same action function initially.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import nnx


HIDDEN_DIM = 256
ROLE_SLOTS = 6
EVENT_SLOTS = 16
EVENT_TYPES = 8
PREDICATE_SLOTS = 8
PREDICATE_STATES = 4
PREVIOUS_ACTION_STEPS = 5
ACTIVE_ACTION_DIM = 7

# Derived from the complete seed-7 production stream (30,000 * 32 sampled
# windows, 5,749,888 valid replan positions).  Unlike the previous symmetric
# channel weights, these factors amplify only positives.  Per-channel
# normalizers preserve the expected zero-logit BCE scale, so rare-event
# supervision does not silently increase the overall HETM loss magnitude.
# Source: experiments/pi05/hetm_production_event_exposure_audit_v1.json.
EVENT_POSITIVE_WEIGHTS = (
    1.210865608238963,
    1.816153310035362,
    2.5052827862771516,
    8.0,
    1.9326788568111182,
    8.0,
    8.0,
    8.0,
)
EVENT_CHANNEL_NORMALIZERS = (
    0.9212324268882492,
    0.8404257007911221,
    0.8285891609596543,
    0.9395909676586489,
    0.8354461953743445,
    0.9840003559583665,
    0.9355788530979166,
    0.9283978205638471,
)
# Target-aware inverse-square-root weights from the same full production
# stream. Unobserved classes retain unit placeholders; observed weights are
# normalized to unit expectation per predicate slot / frontier objective.
PREDICATE_TARGET_WEIGHTS = (
    (1.0, 1.0, 1.0),
    (0.7103156909781795, 1.1154886629134961, 0.9212324268882494),
    (2.01516532877131, 0.8115335974456671, 1.0390641247808126),
    (0.7389286454243044, 0.8378543482739937, 1.5675828187653686),
    (0.8072474234443028, 0.8404441383574242, 2.9008816234802572),
    (0.9236867633215581, 0.9288176897212712, 7.389494106572465),
    (0.895775452706898, 6.55555140969611, 7.099941996788467),
    (0.9354971955966807, 0.9398279328413672, 7.483977564773445),
)
FRONTIER_CLASS_WEIGHTS = (
    0.9327746624360973,
    0.8588713675750165,
    0.8584873703255187,
    1.0296524946444714,
    1.0574437416821287,
    1.561115584858555,
    3.218657248924603,
    3.218657248924603,
)
NEXT_FRONTIER_CLASS_WEIGHTS = (
    1.0363128751239303,
    0.84095770049759,
    0.8194203498055137,
    0.9641223869862812,
    0.9561331007515083,
    1.285049202606698,
    2.6067197994308287,
    3.062561014511411,
)


class _KernelLinear(nnx.Module):
    """Bias-free linear layer with auditable zero-output initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        zero: bool = False,
        rngs: nnx.Rngs,
    ):
        initializer = (
            nnx.initializers.zeros_init()
            if zero
            else nnx.initializers.normal(in_features**-0.5)
        )
        self.kernel = nnx.Param(
            initializer(rngs.params(), (in_features, out_features), jnp.float32)
        )

    def __call__(self, value):
        return jnp.einsum("...d,dh->...h", value, self.kernel.value)


class _LearnedSlots(nnx.Module):
    def __init__(self, count: int, width: int, *, rngs: nnx.Rngs):
        self.value = nnx.Param(
            nnx.initializers.normal(width**-0.5)(
                rngs.params(), (count, width), jnp.float32
            )
        )


def initial_state(batch_size: int) -> dict[str, jax.Array]:
    """Create a clean episode state.  Slot zero is the committed frontier."""
    if batch_size < 1:
        raise ValueError("HETM batch size must be positive")
    return {
        "event_ledger": jnp.zeros(
            (batch_size, EVENT_SLOTS, HIDDEN_DIM), jnp.float32
        ),
        "event_valid": jnp.zeros((batch_size, EVENT_SLOTS), jnp.bool_),
        "event_write_index": jnp.zeros((batch_size,), jnp.int32),
        "last_event_probabilities": jnp.zeros(
            (batch_size, EVENT_TYPES), jnp.float32
        ),
        "predicate_memory": jnp.zeros(
            (batch_size, PREDICATE_SLOTS, HIDDEN_DIM), jnp.float32
        ),
        "predicate_probabilities": jnp.broadcast_to(
            jax.nn.one_hot(0, PREDICATE_STATES, dtype=jnp.float32),
            (batch_size, PREDICATE_SLOTS, PREDICATE_STATES),
        ),
        "frontier": jnp.broadcast_to(
            jax.nn.one_hot(0, PREDICATE_SLOTS, dtype=jnp.float32),
            (batch_size, PREDICATE_SLOTS),
        ),
    }


def _ordered_ledger_read(ledger, valid, write_index, query):
    """Content-addressed ledger read with an explicit chronological bias."""
    slots = jnp.arange(EVENT_SLOTS)[None]
    newest = (write_index - 1) % EVENT_SLOTS
    age = (newest[:, None] - slots) % EVENT_SLOTS
    logits = jnp.einsum('bh,bsh->bs', query, ledger) / jnp.sqrt(
        jnp.asarray(HIDDEN_DIM, jnp.float32)
    )
    # The ring-buffer slot itself has no stable temporal meaning after wrap;
    # age relative to the current write cursor restores chronological order.
    logits = logits - 0.20 * age.astype(logits.dtype)
    logits = jnp.where(valid, logits, -1.0e30)
    attention = jax.nn.softmax(logits, axis=-1) * valid.astype(logits.dtype)
    attention /= jnp.maximum(jnp.sum(attention, axis=-1, keepdims=True), 1.0e-6)
    return jnp.einsum('bs,bsh->bh', attention, ledger)


def _frontier_allowed_mask(current_index, regression_probability):
    """Return the hard topology supported by the recurrent server contract."""
    indices = jnp.arange(PREDICATE_SLOTS)[None]
    normal_allowed = (indices == current_index[:, None]) | (
        indices
        == jnp.minimum(current_index + 1, PREDICATE_SLOTS - 1)[:, None]
    )
    regression_allowed = (
        (indices < current_index[:, None])
        & (regression_probability[:, None] > 0.5)
    )
    return normal_allowed | regression_allowed


class HierarchicalEventTransitionMemory(nnx.Module):
    """Open-vocabulary roles, causal event ledger, and predicate frontier."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        state_dim: int,
        action_hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.prefix_dim = prefix_dim
        self.state_dim = state_dim
        self.action_hidden_dim = action_hidden_dim
        self.role_queries = _LearnedSlots(ROLE_SLOTS, HIDDEN_DIM, rngs=rngs)
        self.prefix_key = _KernelLinear(prefix_dim, HIDDEN_DIM, rngs=rngs)
        self.prefix_value = _KernelLinear(prefix_dim, HIDDEN_DIM, rngs=rngs)
        self.state_in = _KernelLinear(state_dim, HIDDEN_DIM, rngs=rngs)
        self.role_fuse = _KernelLinear(
            ROLE_SLOTS * HIDDEN_DIM, HIDDEN_DIM, rngs=rngs
        )
        self.action_in = _KernelLinear(
            PREVIOUS_ACTION_STEPS * ACTIVE_ACTION_DIM, HIDDEN_DIM, rngs=rngs
        )
        self.event_in = _KernelLinear(5 * HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.event_logits = _KernelLinear(HIDDEN_DIM, EVENT_TYPES, rngs=rngs)
        self.event_type_embeddings = _LearnedSlots(EVENT_TYPES, HIDDEN_DIM, rngs=rngs)
        self.event_content = _KernelLinear(HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.predicate_queries = _LearnedSlots(
            PREDICATE_SLOTS, HIDDEN_DIM, rngs=rngs
        )
        self.predicate_memory_in = _KernelLinear(HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.predicate_transition_in = _KernelLinear(
            4 * HIDDEN_DIM, HIDDEN_DIM, rngs=rngs
        )
        self.predicate_delta = _KernelLinear(HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.predicate_logits = _KernelLinear(
            HIDDEN_DIM, PREDICATE_STATES, rngs=rngs
        )
        self.frontier_in = _KernelLinear(4 * HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.frontier_logits = _KernelLinear(HIDDEN_DIM, PREDICATE_SLOTS, rngs=rngs)
        self.action_context = _KernelLinear(3 * HIDDEN_DIM, HIDDEN_DIM, rngs=rngs)
        self.next_frontier_logits = _KernelLinear(
            HIDDEN_DIM, PREDICATE_SLOTS, rngs=rngs
        )
        self.prior_out = _KernelLinear(
            HIDDEN_DIM, action_hidden_dim, zero=True, rngs=rngs
        )
        self.film_out = _KernelLinear(
            HIDDEN_DIM, 2 * action_hidden_dim, zero=True, rngs=rngs
        )
        # Direct already owns the only Gemma action-layer adapter payload.
        # Project HETM context into that adapter's 5*256 semantic feature
        # space instead of installing a competing second adapter. The closed
        # projection preserves the complete Direct policy at initialization.
        self.hmca_condition_out = _KernelLinear(
            HIDDEN_DIM, 5 * HIDDEN_DIM, zero=True, rngs=rngs
        )

    def supervised_initial_state(
        self,
        history_event_targets: jax.Array,
        history_event_valid: jax.Array,
        predicate_targets: jax.Array,
        frontier_targets: jax.Array,
    ) -> dict[str, jax.Array]:
        """Encode causal pre-window labels into the recurrent training state."""
        batch = history_event_targets.shape[0]
        if history_event_targets.shape != (batch, EVENT_SLOTS, EVENT_TYPES):
            raise ValueError('HETM history event shape drifted')
        if history_event_valid.shape != (batch, EVENT_SLOTS):
            raise ValueError('HETM history validity shape drifted')
        if predicate_targets.shape != (batch, PREDICATE_SLOTS):
            raise ValueError('HETM initial predicate shape drifted')
        if frontier_targets.shape != (batch,):
            raise ValueError('HETM initial frontier shape drifted')
        valid = history_event_valid.astype(jnp.bool_)
        # Inference appends an event atom only when its probability vector is
        # novel.  Mirror that causal topology for hard teacher labels instead
        # of filling the ledger with repeated copies of a stable phase.  This
        # matters for interior windows: the ordered ledger reader must see
        # transitions, not the number of fixed-rate replans spent in a phase.
        previous_targets = jnp.concatenate(
            [
                jnp.zeros_like(history_event_targets[:, :1]),
                history_event_targets[:, :-1],
            ],
            axis=1,
        )
        previous_valid = jnp.concatenate(
            [jnp.zeros_like(valid[:, :1]), valid[:, :-1]], axis=1
        )
        changed = jnp.any(
            history_event_targets != previous_targets, axis=-1
        )
        transition_valid = valid & (~previous_valid | changed)
        transition_count = jnp.sum(transition_valid, axis=-1).astype(jnp.int32)
        transition_index = jnp.cumsum(
            transition_valid.astype(jnp.int32), axis=-1
        ) - 1
        transition_assignment = jax.nn.one_hot(
            jnp.clip(transition_index, 0, EVENT_SLOTS - 1),
            EVENT_SLOTS,
            dtype=jnp.float32,
        ) * transition_valid[..., None]
        event_count = jnp.maximum(
            jnp.sum(history_event_targets, axis=-1, keepdims=True), 1.0
        )
        encoded_history = jnp.tanh(
            jnp.einsum(
                'bse,eh->bsh',
                history_event_targets.astype(jnp.float32),
                self.event_type_embeddings.value.value,
            )
            / event_count
        ) * transition_valid[..., None]
        ledger = jnp.einsum(
            'bst,bsh->bth', transition_assignment, encoded_history
        )
        compact_valid = (
            jnp.arange(EVENT_SLOTS)[None] < transition_count[:, None]
        )
        predicate_indices = jnp.clip(
            predicate_targets.astype(jnp.int32), 0, PREDICATE_STATES - 1
        )
        predicate_probabilities = jax.nn.one_hot(
            predicate_indices, PREDICATE_STATES, dtype=jnp.float32
        )
        predicate_memory = jnp.tanh(
            self.predicate_queries.value.value[None]
            + jnp.einsum(
                'bps,sh->bph',
                predicate_probabilities,
                self.event_type_embeddings.value.value[:PREDICATE_STATES],
            )
        )
        frontier = jax.nn.one_hot(
            jnp.clip(
                frontier_targets.astype(jnp.int32), 0, PREDICATE_SLOTS - 1
            ),
            PREDICATE_SLOTS,
            dtype=jnp.float32,
        )
        history_positions = jnp.where(
            valid,
            jnp.arange(EVENT_SLOTS, dtype=jnp.int32)[None],
            -1,
        )
        last_history_index = jnp.maximum(
            jnp.max(history_positions, axis=-1), 0
        )
        last_event_probabilities = jnp.take_along_axis(
            history_event_targets,
            last_history_index[:, None, None],
            axis=1,
        )[:, 0]
        last_event_probabilities = jnp.where(
            jnp.any(valid, axis=-1)[:, None], last_event_probabilities, 0.0
        ).astype(jnp.float32)
        return {
            'event_ledger': ledger,
            'event_valid': compact_valid,
            'event_write_index': (
                transition_count % EVENT_SLOTS
            ),
            'last_event_probabilities': last_event_probabilities,
            'predicate_memory': predicate_memory,
            'predicate_probabilities': predicate_probabilities,
            'frontier': frontier,
        }

    def __call__(
        self,
        prefix_states: jax.Array,
        prefix_mask: jax.Array,
        proprioception: jax.Array,
        previous_actions: jax.Array,
        state: Mapping[str, jax.Array],
        episode_start: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        batch = prefix_states.shape[0]
        if prefix_states.shape[-1] != self.prefix_dim:
            raise ValueError("HETM prefix width drifted")
        if prefix_mask.shape != prefix_states.shape[:2]:
            raise ValueError("HETM prefix mask shape drifted")
        if proprioception.shape != (batch, self.state_dim):
            raise ValueError("HETM proprioception shape drifted")
        if previous_actions.shape != (
            batch,
            PREVIOUS_ACTION_STEPS,
            ACTIVE_ACTION_DIM,
        ):
            raise ValueError("HETM previous-action shape drifted")
        clean = initial_state(batch)
        if set(state) != set(clean):
            raise ValueError("HETM recurrent state fields drifted")

        reset = episode_start.astype(jnp.bool_)
        reset2, reset3 = reset[:, None], reset[:, None, None]
        ledger = jnp.where(reset3, clean["event_ledger"], state["event_ledger"])
        valid = jnp.where(reset2, clean["event_valid"], state["event_valid"])
        write_index = jnp.where(
            reset, clean["event_write_index"], state["event_write_index"]
        )
        previous_event_probabilities = jnp.where(
            reset2,
            clean["last_event_probabilities"],
            state["last_event_probabilities"],
        )
        predicate_memory = jnp.where(
            reset3, clean["predicate_memory"], state["predicate_memory"]
        )
        previous_predicate_probabilities = jnp.where(
            reset3,
            clean["predicate_probabilities"],
            state["predicate_probabilities"],
        )
        previous_frontier = jnp.where(reset2, clean["frontier"], state["frontier"])

        keys, values = self.prefix_key(prefix_states), self.prefix_value(prefix_states)
        state_token = self.state_in(proprioception)
        role_queries = self.role_queries.value.value[None] + state_token[:, None]
        logits = jnp.einsum("brh,bth->brt", role_queries, keys) / jnp.sqrt(
            jnp.asarray(HIDDEN_DIM, jnp.float32)
        )
        logits = jnp.where(prefix_mask[:, None], logits, -1.0e30)
        role_attention = jax.nn.softmax(logits, axis=-1) * prefix_mask[:, None]
        role_attention /= jnp.maximum(
            jnp.sum(role_attention, axis=-1, keepdims=True), 1.0e-6
        )
        role_states = jnp.einsum("brt,bth->brh", role_attention, values)
        # Keep the six open-vocabulary roles ordered: target/reference swaps
        # must not collapse to the same pooled representation.
        role_summary = jnp.tanh(self.role_fuse(role_states.reshape(batch, -1)))
        predicate_summary = jnp.mean(predicate_memory, axis=1)
        action_state = self.action_in(previous_actions.reshape(batch, -1))
        ledger_summary = _ordered_ledger_read(
            ledger,
            valid,
            write_index,
            role_summary + state_token + action_state,
        )

        event_hidden = jnp.tanh(
            self.event_in(
                jnp.concatenate(
                    [role_summary, state_token, action_state, ledger_summary, predicate_summary],
                    axis=-1,
                )
            )
        )
        event_logits = self.event_logits(event_hidden)
        event_probabilities = jax.nn.sigmoid(event_logits)
        event_candidate = jnp.tanh(
            self.event_content(event_hidden)
            + jnp.einsum(
                "be,eh->bh",
                event_probabilities,
                self.event_type_embeddings.value.value,
            )
        )
        write_slot = jax.nn.one_hot(write_index, EVENT_SLOTS, dtype=jnp.float32)
        # The ledger records event transitions, not every fixed-rate replan.
        # Stable phase probabilities leave the cursor unchanged; a reset or a
        # material probability change appends one event atomically.
        event_novelty = jnp.max(
            jnp.abs(event_probabilities - previous_event_probabilities), axis=-1
        )
        write_strength = jnp.where(
            reset, jnp.max(event_probabilities, axis=-1), event_novelty
        )
        write_decision = reset | (write_strength > 0.05)
        effective_write_strength = jnp.where(
            write_decision, write_strength, 0.0
        )
        soft_write = write_slot * effective_write_strength[:, None]
        next_ledger = (
            ledger * (1.0 - soft_write[..., None])
            + event_candidate[:, None] * soft_write[..., None]
        )
        next_valid = valid | (
            write_slot.astype(jnp.bool_) & write_decision[:, None]
        )
        next_write_index = jnp.where(
            write_decision, (write_index + 1) % EVENT_SLOTS, write_index
        )
        next_ledger_summary = _ordered_ledger_read(
            next_ledger,
            next_valid,
            next_write_index,
            event_candidate + state_token,
        )

        predicate_transition = jnp.tanh(
            self.predicate_transition_in(
                jnp.concatenate(
                    [role_summary, state_token, action_state, event_candidate], axis=-1
                )
            )
        )
        predicate_hidden = jnp.tanh(
            self.predicate_memory_in(predicate_memory)
            + self.predicate_queries.value.value[None]
            + predicate_transition[:, None]
            + jnp.einsum(
                'bps,sh->bph',
                previous_predicate_probabilities,
                self.event_type_embeddings.value.value[:PREDICATE_STATES],
            )
        )
        next_predicate_memory = jnp.tanh(
            predicate_memory + self.predicate_delta(predicate_hidden)
        )
        predicate_logits = self.predicate_logits(next_predicate_memory)
        predicate_probabilities = jax.nn.softmax(predicate_logits, axis=-1)
        frontier_hidden = jnp.tanh(
            self.frontier_in(
                jnp.concatenate(
                    [
                        next_ledger_summary,
                        jnp.mean(next_predicate_memory, axis=1),
                        role_summary,
                        state_token,
                    ],
                    axis=-1,
                )
            )
        )
        raw_frontier_logits = self.frontier_logits(frontier_hidden)
        current_index = jnp.argmax(jax.lax.stop_gradient(previous_frontier), axis=-1)
        regression_probability = jnp.take_along_axis(
            predicate_probabilities[..., 3], current_index[:, None], axis=1
        )[:, 0]
        frontier_allowed = _frontier_allowed_mask(
            current_index, regression_probability
        )
        frontier_logits = jnp.where(
            frontier_allowed, raw_frontier_logits, -1.0e30
        )
        frontier = jax.nn.softmax(frontier_logits, axis=-1)
        committed_frontier = jnp.where(reset2, clean["frontier"], frontier)
        frontier_state = jnp.einsum(
            'bs,sh->bh', committed_frontier, self.predicate_queries.value.value
        )
        action_context = jnp.tanh(
            self.action_context(
                jnp.concatenate(
                    [
                        next_ledger_summary,
                        jnp.mean(next_predicate_memory, axis=1),
                        frontier_hidden + frontier_state,
                    ],
                    axis=-1,
                )
            )
        )
        raw_next_frontier_logits = self.next_frontier_logits(action_context)
        next_frontier_logits = jnp.where(
            _frontier_allowed_mask(
                jnp.argmax(
                    jax.lax.stop_gradient(committed_frontier), axis=-1
                ),
                regression_probability,
            ),
            raw_next_frontier_logits,
            -1.0e30,
        )
        film_scale, film_shift = jnp.split(self.film_out(action_context), 2, axis=-1)
        outputs = {
            "role_states": role_states,
            "role_attention": role_attention,
            "event_logits": event_logits,
            "event_probabilities": event_probabilities,
            "event_write_strength": write_strength,
            "predicate_logits": predicate_logits,
            "predicate_probabilities": predicate_probabilities,
            "frontier_logits": frontier_logits,
            "raw_frontier_logits": raw_frontier_logits,
            "frontier": committed_frontier,
            "next_frontier_logits": next_frontier_logits,
            "raw_next_frontier_logits": raw_next_frontier_logits,
            "previous_frontier_index": current_index,
            "action_context": action_context,
            "prior_residual": self.prior_out(action_context),
            "film_scale": film_scale,
            "film_shift": film_shift,
            # Match the bounded scale of the inherited bridge features while
            # retaining unit derivative at the exact-zero initialization.
            "hmca_condition_residual": jnp.tanh(
                self.hmca_condition_out(action_context)
            ),
        }
        next_state = {
            "event_ledger": next_ledger,
            "event_valid": next_valid,
            "event_write_index": next_write_index,
            "last_event_probabilities": event_probabilities,
            "predicate_memory": next_predicate_memory,
            "predicate_probabilities": predicate_probabilities,
            "frontier": committed_frontier,
        }
        return outputs, next_state


def event_supervision_loss(
    event_logits: jax.Array,
    event_targets: jax.Array,
) -> jax.Array:
    """Per-example exposure-normalized BCE with positive-only balancing."""
    event_targets = event_targets.astype(jnp.float32)
    positive_weights = jnp.asarray(EVENT_POSITIVE_WEIGHTS, dtype=jnp.float32)
    channel_normalizers = jnp.asarray(
        EVENT_CHANNEL_NORMALIZERS, dtype=jnp.float32
    )
    event_bce = jax.nn.softplus(event_logits) - event_targets * event_logits
    positive_factor = 1.0 + event_targets * (positive_weights - 1.0)
    return jnp.mean(
        event_bce * positive_factor * channel_normalizers, axis=-1
    )


def supervised_auxiliary_loss(
    outputs: Mapping[str, jax.Array],
    *,
    event_targets: jax.Array,
    predicate_targets: jax.Array,
    frontier_targets: jax.Array,
    next_frontier_targets: jax.Array,
    sample_valid: jax.Array,
    weights: Mapping[str, float] | None = None,
) -> jax.Array:
    """Per-example weighted event/predicate/frontier supervision."""
    valid = sample_valid.astype(jnp.float32)
    predicate_targets = predicate_targets.astype(jnp.int32)
    frontier_targets = frontier_targets.astype(jnp.int32)
    next_frontier_targets = next_frontier_targets.astype(jnp.int32)
    event_loss = event_supervision_loss(outputs["event_logits"], event_targets)
    # VLA-Arena demonstrations supply unknown/unsatisfied/satisfied only.
    # Normalize over those observed states so the never-labelled regression
    # column is not trained as an implicit negative. It remains available to
    # action/transition gradients instead of being erased by auxiliary CE.
    predicate_logp = jax.nn.log_softmax(
        outputs["predicate_logits"][..., : PREDICATE_STATES - 1], axis=-1
    )
    predicate_selected = jnp.take_along_axis(
        predicate_logp, predicate_targets[..., None], axis=-1
    )[..., 0]
    predicate_weight_table = jnp.asarray(
        PREDICATE_TARGET_WEIGHTS, dtype=jnp.float32
    )
    predicate_weight = predicate_weight_table[
        jnp.arange(PREDICATE_SLOTS)[None, :], predicate_targets
    ]
    predicate_loss = -jnp.mean(predicate_selected * predicate_weight, axis=-1)
    class_weight = jnp.asarray(FRONTIER_CLASS_WEIGHTS, dtype=jnp.float32)
    next_class_weight = jnp.asarray(
        NEXT_FRONTIER_CLASS_WEIGHTS, dtype=jnp.float32
    )
    # Runtime uses the hard-projected logits, while teacher-forced CE must use
    # raw logits because a valid label may lie outside the committed frontier.
    frontier_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(outputs["raw_frontier_logits"], axis=-1),
        frontier_targets[:, None], axis=-1,
    )[:, 0] * class_weight[frontier_targets]
    transition_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(outputs["raw_next_frontier_logits"], axis=-1),
        next_frontier_targets[:, None], axis=-1,
    )[:, 0] * next_class_weight[next_frontier_targets]
    indices = jnp.arange(PREDICATE_SLOTS)[None]
    previous = outputs["previous_frontier_index"][:, None]
    invalid = (indices < previous) | (
        indices > jnp.minimum(previous + 1, PREDICATE_SLOTS - 1)
    )
    monotonic_loss = jnp.sum(
        jax.nn.softmax(outputs["raw_frontier_logits"], axis=-1) * invalid,
        axis=-1,
    )
    weights = weights or {
        "event": 0.08,
        "predicate": 0.08,
        "frontier": 0.05,
        "transition": 0.03,
        "monotonic": 0.02,
    }
    return valid * (
        weights["event"] * event_loss
        + weights["predicate"] * predicate_loss
        + weights["frontier"] * frontier_loss
        + weights["transition"] * transition_loss
        + weights["monotonic"] * monotonic_loss
    )
