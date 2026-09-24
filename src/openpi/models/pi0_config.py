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

import dataclasses
import math
from typing import TYPE_CHECKING
from typing_extensions import override

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import openpi.models.gemma as _gemma
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import model as _model
from openpi.shared import array_typing as at


if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = 'bfloat16'
    paligemma_variant: _gemma.Variant = 'gemma_2b'
    action_expert_variant: _gemma.Variant = 'gemma_300m'

    # Set the model specific defaults.
    action_dim: int = 32
    # Restrict flow matching and denoising to the physical robot action
    # dimensions when the model tensor is padded for cross-robot compatibility.
    # ``None`` preserves the original all-dimension behavior.
    active_action_dim: int | None = None
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Optionally inject the continuous proprioceptive state into the pi0.5
    # action expert through its adaptive RMSNorm conditioning.  The added
    # branch is zero-initialized, so enabling it preserves the behavior of a
    # pretrained pi0.5 checkpoint until the branch is fine-tuned.
    state_adarms: bool = False
    state_adarms_hidden_dim: int = 256
    # Inject pooled contextual multimodal action-prior states into the same
    # adaptive RMSNorm condition used at every action-expert attention and MLP
    # layer.  The output projection is zero initialized for exact inheritance.
    context_adarms: bool = False
    context_adarms_hidden_dim: int = 256
    # Inject the causal cross-replan PSM state into every action-expert
    # AdaRMS layer.  This is distinct from context_adarms, which only reads
    # the current-frame visual-language action-prior states.
    persistent_memory_adarms: bool = False
    persistent_memory_adarms_hidden_dim: int = 256
    # Couple the current persistent execution phase with the per-action
    # risk-conditioned contact-affordance representation through a tokenwise
    # FiLM path.  This supplies the multiplicative interaction that additive
    # PSM/contact residuals cannot express, while a zero-initialized output
    # keeps checkpoint inheritance exact until the new branch is trained.
    phase_contact_action_film: bool = False
    phase_contact_action_film_hidden_dim: int = 256
    # At selected action-expert depths, use each action token as a query over
    # the distinct causal PSM slots plus current/future program tokens.  This
    # retains slot identity that pooled AdaRMS and FiLM necessarily discard.
    layerwise_persistent_memory_attention: bool = False
    layerwise_persistent_memory_attention_rank: int = 32
    layerwise_persistent_memory_attention_alpha: float = 32.0
    layerwise_persistent_memory_attention_layers: tuple[int, ...] = (5, 11, 17)
    # Apply a second, token-level state path to the noisy action embeddings.
    # The FiLM projection is zero initialized in Pi0, preserving checkpoint
    # behavior before fine-tuning.
    state_action_film: bool = False
    state_action_film_hidden_dim: int = 256
    # Learned coarse-action queries cross-attend to multimodal prefix
    # embeddings. Their reference trajectory conditions the flow expert and
    # receives a small auxiliary action-space loss during training.
    action_prior: bool = False
    action_prior_hidden_dim: int = 256
    action_prior_horizon: int = 5
    action_prior_loss_weight: float = 0.1
    # Number of independently sampled noise/time pairs used by the main
    # continuous flow objective for each action chunk. Values above one reuse
    # a single contextual prefix cache and execute checkpointed suffix passes
    # sequentially, improving flow coverage without multiplying peak batch
    # activation memory.
    main_flow_samples: int = 1
    # Read the coarse action prior from the frozen VLM's contextualized prefix
    # states instead of the raw image/text embeddings. This adds no parameters
    # but makes the prior consume the representation after spatial-language
    # reasoning. It requires a cached prefix pass during training, matching the
    # inference path exactly.
    action_prior_contextual: bool = False
    # Condition coarse waypoint queries on the current normalized robot state.
    action_prior_state_conditioning: bool = False
    # Additionally condition each coarse waypoint query on the causal PSM
    # state: global memory, the active ordered subgoal, and a decayed summary
    # of the remaining program.  The projection is zero initialized, so a
    # completed parent policy is preserved exactly at checkpoint inheritance.
    persistent_action_prior_conditioning: bool = False
    # ``mean`` retains the stage-2 pair-average target. ``endpoint`` predicts
    # segment endpoints and avoids blurring gripper or rotational transitions.
    action_prior_target: str = 'mean'
    hierarchical_event_transition_memory: bool = False
    hetm_event_loss_weight: float = 0.08
    hetm_predicate_loss_weight: float = 0.08
    hetm_frontier_loss_weight: float = 0.05
    hetm_transition_loss_weight: float = 0.03
    hetm_monotonic_loss_weight: float = 0.02
    hetm_psm_frontier_consistency_loss_weight: float = 0.02
    # Two-view role-affordance graph successor. It consumes the already
    # contextualized scene and HETM state, and reaches the action expert only
    # through an exactly-zero initialized residual boundary.
    role_affordance_causal_graph: bool = False
    racg_role_align_loss_weight: float = 0.05
    racg_relation_loss_weight: float = 0.04
    racg_crossview_loss_weight: float = 0.03
    racg_slot_reconstruction_loss_weight: float = 0.03
    racg_contact_loss_weight: float = 0.06
    racg_identity_transition_loss_weight: float = 0.02
    racg_slot_diversity_loss_weight: float = 0.01
    # Allow a retrieved demonstration montage to occupy the otherwise absent
    # right-wrist slot in the shared VLM prefix, while excluding it from every
    # physical camera mask used by geometry, memory, and dynamics modules.
    grounded_demonstration_camera_context_only: bool = False
    # Optional Molmo2-ER geometry successor. Five exact-zero gates inject the
    # sealed external role reads before RACG graph propagation, preserving the
    # complete Direct848+HETM+RACG parent function at initialization.
    racg_external_geometry_prior: bool = False
    # Route the five ordered external geometry roles through a zero-initialized
    # low-rank bridge into HMCA's semantic condition at layers 5/11/17.
    racg_external_geometry_hmca: bool = False
    racg_external_geometry_hmca_hidden_dim: int = 64
    # Route the complete ordered RACG role/edge graph into HMCA. Unlike the
    # input-only graph residual, this exposes relation/contact/hazard evidence
    # at every selected action-expert adapter layer.
    racg_graph_hmca: bool = False
    racg_graph_hmca_hidden_dim: int = 128
    # A compact Action-CoT stage.  The implicit branch pools selected VLM KV
    # layers with learned queries, while the explicit branch learns a separate
    # flow field over the short waypoint trajectory.  Both are fused into the
    # normal pi0.5 action tokens, so the original continuous action expert and
    # its per-layer prefix attention remain intact.
    dual_action_reasoner: bool = False
    implicit_action_reasoner_layers: tuple[int, ...] = (3, 7, 11, 17)
    implicit_action_reasoner_pool_stride: int = 1
    # Preserve one learned-query token per selected VLM layer for the final
    # noisy-action cross-attention.  Projections are shared within adjacent
    # layer groups, matching ACoT's released downsample-based IAR instead of
    # collapsing the layer axis before action-conditioned fusion.
    implicit_action_reasoner_layerwise_guidance: bool = False
    implicit_action_reasoner_group_size: int = 3
    implicit_action_reasoner_downsample_dim: int = 128
    implicit_action_reasoner_num_heads: int = 8
    explicit_action_reasoner_hidden_dim: int = 256
    explicit_action_reasoner_layers: int = 2
    explicit_action_reasoner_num_heads: int = 4
    explicit_action_reasoner_mlp_dim: int = 512
    explicit_action_reasoner_loss_weight: float = 0.1
    explicit_action_reasoner_flow_samples: int = 4
    # Condition the main action expert on ground-truth coarse waypoints during
    # training while independently training the EAR flow field. At inference,
    # the main expert consumes the EAR prediction. This follows ACoT-VLA's
    # teacher-forcing stabilization and adds no parameters.
    explicit_action_reasoner_teacher_forcing: bool = True
    explicit_action_reasoner_inference_steps: int = 4
    # Add a retrieved same-task visual plan and progress-aligned demonstrated
    # action chunk. The third (normally padded) camera carries a four-keyframe
    # montage; a temporal reasoner jointly encodes the local action chunk and a
    # sparse whole-episode kinematic plan.
    retrieved_demo_conditioning: bool = False
    retrieved_demo_hidden_dim: int = 256
    retrieved_demo_layers: int = 2
    retrieved_demo_num_heads: int = 4
    retrieved_demo_mlp_dim: int = 512
    retrieved_demo_plan_steps: int = 10
    retrieved_demo_plan_dim: int = 17
    # Supervise the learned context gate on deterministic wrong-task
    # demonstrations injected by the training transform. Correct contexts
    # target the identity multiplier (1), wrong contexts target suppression
    # (0), and absent contexts are ignored.
    retrieved_demo_reliability_loss_weight: float = 0.05
    # Persist a compact ordered-subgoal state across fixed five-action replans.
    # The policy-facing gain starts at exact zero so the compound Stage3/Stage4
    # parent is function preserving before fine-tuning.
    persistent_subgoal_memory: bool = False
    persistent_memory_tokens: int = 8
    persistent_memory_hidden_dim: int = 256
    persistent_memory_subgoal_slots: int = 8
    persistent_memory_fast_tokens: int = 4
    persistent_memory_fast_update_rate: float = 0.50
    persistent_memory_slow_update_rate: float = 0.05
    persistent_memory_previous_action_steps: int = 5
    # Optional shape-preserving embodiment bridge.  A dual-arm 14-D policy can
    # keep an inherited single-arm 7-D recurrent memory kernel by projecting
    # the two ordered arm chunks to their elementwise mean before the memory
    # update.  The policy action head still predicts all 14 physical joints.
    persistent_memory_action_dim: int | None = None
    persistent_memory_short_replans: int = 4
    persistent_memory_long_replans: int = 8
    persistent_memory_long_probability: float = 0.50
    persistent_memory_flow_replan_indices: tuple[int, ...] = (0, 2, 4, 7)
    persistent_memory_cache_refresh_steps: int = 1000
    persistent_memory_max_staleness_steps: int = 1000
    persistent_memory_policy_gain: float = 0.0
    # Bound the learned policy-facing scalar for new recurrent successors.
    # Historical PSM checkpoints retain their original raw-gain semantics.
    persistent_memory_bounded_policy_gain: bool = False
    # Keep exactly one function-preserving gate at the policy boundary.  The
    # compositional phase projection starts small but open so route/flow losses
    # can reach its upstream factor graph immediately instead of crossing two
    # sequential zero-initialized projections.
    persistent_memory_compositional_phase_init_scale: float = 0.01
    persistent_memory_policy_gain_warmup_steps: int = 3000
    # Persistent structured-demonstration language auxiliary (PSM-SDLA).
    # The values are deliberately explicit so checkpoint parameter shapes are
    # immutable and can be audited before an eight-GPU preflight.
    persistent_structured_demo_language: bool = False
    # Direct causal memory/program residual added after the inherited scalar
    # path.  Defaults remain disabled for every historical configuration.
    persistent_conditional_memory_policy_bridge: bool = False
    persistent_conditional_memory_policy_bridge_rank: int = 128
    # Five-role visual geometry action-head residual. Historical configs stay disabled.
    persistent_geometry_aux_v1: bool = False
    persistent_geometry_aux_bottleneck_dim: int = 128
    # Zero-gated Molmo2-ER geometry branch for the completed ClausePlan-v3 parent.
    persistent_geometry_external_residual_v1: bool = False
    # Zero-gated action-conditioned refinement of camera-bound role identity.
    persistent_temporal_role_memory_v1: bool = False
    # Zero-gated same-role attention across ordinary camera views.
    persistent_cross_view_role_consensus_v1: bool = False
    persistent_cross_view_role_contrastive_loss_weight: float = 0.02
    persistent_cross_view_role_contrastive_temperature: float = 0.1
    # Zero-gated latent role refinement calibrated by causal contact risk.
    persistent_contact_risk_calibrated_role_residual_v1: bool = False
    persistent_contact_risk_auxiliary_loss_weight: float = 0.05
    persistent_contact_risk_class_weights: tuple[float, ...] = (1.0, 2.0, 4.0)
    # Zero-gated factorized object/reference/relation role composition.
    persistent_relational_role_composer_residual_v1: bool = False
    persistent_relational_role_auxiliary_loss_weight: float = 0.05
    persistent_relational_role_class_weights: tuple[float, ...] = (1.0,) * 16
    # Zero-gated current-clause x plan-slot x grounded-role verifier.
    persistent_clause_role_binding_verifier_v1: bool = False
    persistent_clause_role_binding_auxiliary_loss_weight: float = 0.05
    persistent_clause_role_binding_source_class_weights: tuple[float, ...] = (1.0,) * 4
    persistent_clause_role_binding_destination_class_weights: tuple[float, ...] = (
        1.0,
    ) * 4
    # Zero-gated semantic stay/advance-one residual over current/next plan slots.
    persistent_semantic_frontier_completion_verifier_v1: bool = False
    persistent_semantic_frontier_completion_auxiliary_loss_weight: float = 0.05
    persistent_semantic_frontier_completion_class_weights: tuple[
        tuple[float, float], ...
    ] = ((1.0, 1.0),) * 8
    # Direct causal stay/advance gate over the inherited recurrent frontier.
    # Its final two-logit projection is exactly zero initialized, preserving
    # the parent policy while allowing route supervision to reach it directly.
    persistent_causal_frontier_transition_gate_v1: bool = False
    # ClausePlan-aware causal stay/advance residual.  The policy path is
    # exactly zero at initialization while its auxiliary head trains all
    # internal alignment leaves from the first update.
    persistent_hierarchical_clause_event_alignment_v1: bool = False
    persistent_hierarchical_clause_event_alignment_auxiliary_loss_weight: float = 0.05
    # Four causal HCEA-conditioned action experts for hold/retry/confirmed/
    # stabilize behavior. The exact-zero policy gate preserves its parent.
    persistent_hcea_causal_recovery_action_experts_v1: bool = False
    persistent_hcea_causal_recovery_intent_loss_weight: float = 0.05
    # Permutation-aware causal transport of manipulated/reference identities
    # across replans, with a bounded exact-zero action-token residual.
    persistent_hcea_causal_role_identity_transport_expert_v1: bool = False
    persistent_hcea_causal_role_identity_transport_loss_weight: float = 0.05
    # Three depth-specific PSM-conditioned adapters in the shared Gemma scan.
    persistent_hmca_v4: bool = False
    persistent_hmca_v4_rank: int = 16
    persistent_hmca_v4_alpha: float = 16.0
    persistent_hmca_v4_layers: tuple[int, ...] = (5, 11, 17)
    # Bind the eight recurrent plan slots to ordered clauses extracted from
    # ordinary natural-language prompts.  The adapter output is zero at init.
    persistent_clause_plan_v1: bool = False
    persistent_clause_plan_rank: int = 128
    persistent_clause_plan_slots: int = 8
    persistent_clause_plan_monotonic_strength: float = 4.0
    structured_demo_hidden_dim: int = 256
    structured_demo_semantic_dim: int = 2048
    structured_demo_semantic_slots: int = 8
    structured_demo_prompt_steps: int = 48
    structured_demo_plan_steps: int = 10
    structured_demo_plan_dim: int = 17
    structured_demo_action_steps: int = 10
    spatial_language_vocabulary_size: int = 128
    spatial_language_steps: int = 32
    spatial_language_bos_token_id: int = 1
    spatial_language_auxiliary_initial_weight: float = 0.05
    spatial_language_auxiliary_peak_weight: float = 0.10
    spatial_language_auxiliary_final_weight: float = 0.02
    spatial_language_auxiliary_warmup_steps: int = 1_000
    spatial_language_auxiliary_decay_start_step: int = 15_000
    spatial_language_auxiliary_total_steps: int = 30_000
    persistent_memory_auxiliary_decay_steps: int = 30_000
    persistent_memory_auxiliary_final_multiplier: float = 0.40
    persistent_memory_subgoal_progress_loss_weight: float = 0.05
    persistent_memory_subgoal_transition_loss_weight: float = 0.05
    persistent_memory_action_verification_loss_weight: float = 0.05
    persistent_memory_causal_route_loss_weight: float = 0.05
    persistent_memory_cross_camera_loss_weight: float = 0.02
    persistent_memory_role_distinctness_loss_weight: float = 0.01
    # Penalize target/reference attention overlap on the latent object slots.
    # The deployed binder also gives the manipulated target first claim on a
    # slot and conditions the reference distribution on that claim.
    persistent_memory_role_object_exclusivity_loss_weight: float = 0.01
    # A "between" reference names two objects, so supervise an effective
    # two-slot reference distribution instead of rewarding single-slot entropy.
    persistent_memory_between_pair_entropy_loss_weight: float = 0.01
    # Bind pickup-clause references (table, drawer, cereal, cutting board) to
    # visual slots and feed their geometry into the manipulated target state.
    persistent_memory_source_reference_loss_weight: float = 0.02
    # Bind up to two destination-side spatial references (for example the two
    # objects in "between", or the cabinet in "top layer of") independently.
    persistent_memory_destination_reference_loss_weight: float = 0.02
    persistent_memory_condition_state_loss_weight: float = 0.02
    # Make spatial composition explicit instead of forcing the policy to
    # rediscover top/top-layer/middle-layer/on/between solely from free text.
    persistent_memory_destination_qualifier_loss_weight: float = 0.02
    persistent_memory_temporal_role_loss_weight: float = 0.01
    persistent_memory_factorized_role_loss_weight: float = 0.02
    # Train the five full-context factor queries to retain probability mass
    # near their parsed language spans.  Inference also receives a bounded
    # span residual, so this loss supervises a deployed architecture path.
    persistent_memory_factor_attention_alignment_loss_weight: float = 0.01
    # Reconstruct every valid visual patch through the competitive PSM object
    # slots so unmentioned and distractor objects also train slot coverage.
    persistent_memory_object_slot_reconstruction_loss_weight: float = 0.01
    # Match each grounded visual role to its own open-vocabulary language role
    # against the other role in the same instruction.  This supplies direct
    # target/reference binding supervision without an object-class vocabulary.
    persistent_memory_visual_language_role_loss_weight: float = 0.02
    # Exact integer masses induced by production's equal-suite/equal-task
    # sampler, in classifier-index order.  The sequence objective converts
    # these to inverse-square-root weights normalized to unit expected scale.
    persistent_memory_operation_class_counts: tuple[int, ...] = (100, 8, 1, 1)
    persistent_memory_source_relation_class_counts: tuple[int, ...] = (40, 10, 2, 3)
    persistent_memory_destination_relation_class_counts: tuple[int, ...] = (
        1,
        38,
        12,
        4,
    )
    persistent_memory_condition_class_counts: tuple[int, ...] = (50, 1, 4)
    # Exact equal-suite/task sampling masses for none, between, at-top-of,
    # top-of, top-layer-of, middle-layer-of, nested-on, and conditioned.
    # Open/close tasks provide transferable top/middle-layer supervision.
    persistent_memory_destination_qualifier_class_counts: tuple[int, ...] = (
        63,
        12,
        4,
        10,
        8,
        1,
        2,
        10,
    )
    # Deployment-grid exposure in the generated semantic-phase manifest.
    # Inverse-square-root balancing keeps short contact/release states visible
    # without making them dominate the continuous action objective.
    persistent_memory_semantic_phase_class_counts: tuple[int, ...] = (
        17557,
        11632,
        10953,
        8134,
        9423,
        7621,
        3261,
        5045,
    )
    # Expected semantic-phase mass at the four parent-flow supervision
    # positions under the production 50/50 short/long, equal-suite sampler.
    # This differs materially from the all-frame phase counts above because a
    # complete action horizon makes release/settle anchors much rarer.  The
    # sequence loss converts these masses to inverse-square-root weights with
    # unit expectation, so late completion behavior receives useful policy
    # gradients without changing the overall parent-flow loss scale.
    persistent_memory_flow_phase_class_counts: tuple[int, ...] = (
        219199,
        212599,
        210598,
        145709,
        140154,
        61698,
        9044,
        999,
    )
    # Smooth only the weighting denominator for extremely rare flow phases.
    # Counts remain the exact audited sampler mass and still normalize the
    # weights to unit empirical expectation.  A 0.4% floor prevents a handful
    # of completion anchors from producing unstable >12x parent-flow losses.
    persistent_memory_flow_phase_count_floor: int = 4000
    # Keep the inherited parent prompt byte-for-byte unchanged.  Earlier design
    # drafts advertised a second paraphrase branch, but the production graph
    # never consumed one; leaving a positive weight here would therefore claim
    # supervision that does not exist.
    persistent_memory_paraphrase_loss_weight: float = 0.0
    supervised_role_identity_contrastive_loss: bool = False
    supervised_role_identity_contrastive_loss_weight: float = 0.01
    # Consecutive examples are emitted as same-task positive pairs.  Restrict
    # contrastive negatives to two adjacent pairs so every four-example group
    # can remain local to one data-parallel device.  A global batch matrix
    # forces an NCCL all-gather on replicated-data meshes, which is not a
    # supported collective on the production host.
    persistent_memory_role_contrastive_group_size: int = 4
    # Read the contextualized VLM prefix with a parallel motion-rationale
    # expert.  Seven axis queries predict a compact negative/steady/positive
    # description of the whole action chunk, then condition the continuous
    # action expert through a zero-initialized residual.  Unlike the older
    # detached action-prior auxiliaries, this loss also shapes enabled VLM
    # LoRA features without adding autoregressive language decoding at
    # inference.
    structured_rationale_reasoner: bool = False
    structured_rationale_hidden_dim: int = 256
    structured_rationale_layers: int = 2
    structured_rationale_num_heads: int = 8
    structured_rationale_mlp_dim: int = 1024
    structured_rationale_temperature: float = 0.7
    structured_rationale_loss_weight: float = 0.1
    structured_rationale_neutral_eps: tuple[float, ...] = (
        0.05,
        0.05,
        0.05,
        0.01,
        0.01,
        0.01,
        0.0,
    )
    structured_rationale_action_q01: tuple[float, ...] = (
        -1.0666238371,
        -1.1667888946,
        -2.2257006435,
        -3.9784434656,
        -2.1655601227,
        -0.3351589038,
        -1.0,
    )
    structured_rationale_action_q99: tuple[float, ...] = (
        1.2838100475,
        1.2818870662,
        0.6552116871,
        -2.1637486581,
        2.2740507436,
        0.9024185783,
        0.9996,
    )
    # Inverse-sqrt weights from all 361,736 full-horizon training windows.
    # The clipped weights have empirical mean one independently per axis.
    structured_rationale_class_weights: tuple[tuple[float, ...], ...] = (
        (1.8455472339, 0.7757396308, 1.1704448968),
        (1.4214770165, 0.8077931055, 1.1049591041),
        (1.1276474808, 0.8085962945, 1.3511953385),
        (7.5017834923, 0.9377229365, 7.5017834923),
        (7.4918593423, 0.9364824178, 7.4918593423),
        (6.9830171469, 0.8728771434, 6.9830171469),
        (1.0098856860, 7.9239230848, 0.9904903856),
    )
    # Score whether a complete noisy action chunk is compatible with the
    # multimodal action-prior context, and feed a zero-initialized corrective
    # token into every flow step.  A contrastive auxiliary objective separates
    # the demonstrated chunk from task-mismatched and temporally corrupted
    # chunks; inference still follows one flow trajectory (no reranking).
    action_chunk_verifier: bool = False
    action_chunk_verifier_hidden_dim: int = 256
    action_chunk_verifier_layers: int = 2
    action_chunk_verifier_num_heads: int = 8
    action_chunk_verifier_mlp_dim: int = 1024
    action_chunk_verifier_temperature: float = 0.5
    action_chunk_verifier_loss_weight: float = 0.1
    # Predict a compact future visual feature grid from the current RGB pair
    # and the noisy action chunk, then feed a zero-initialized dynamics-aware
    # residual back into the same flow trajectory. Future RGB is a training
    # target only and is never part of the inference input.
    latent_future_reasoner: bool = False
    latent_future_hidden_dim: int = 256
    latent_future_layers: int = 2
    latent_future_num_heads: int = 8
    latent_future_mlp_dim: int = 1024
    latent_future_grid_size: int = 4
    latent_future_loss_weight: float = 0.05
    # Roll the noisy action chunk forward into a dense normalized
    # proprioceptive trajectory.  Future states supervise training only; a
    # zero-initialized action-facing head preserves inherited inference.
    state_rollout_reasoner: bool = False
    state_rollout_hidden_dim: int = 256
    state_rollout_layers: int = 2
    state_rollout_num_heads: int = 8
    state_rollout_mlp_dim: int = 1024
    state_rollout_target_dim: int = 8
    state_rollout_loss_weight: float = 0.1
    # Route every denoising-step action token through a language/state-
    # conditioned sparse bank of trainable experts.  A dense clean-action
    # prediction objective starts the experts and router immediately, while a
    # zero-initialized action-facing head preserves the inherited Stage-6
    # policy exactly before fine-tuning.
    action_moe_reasoner: bool = False
    action_moe_hidden_dim: int = 256
    action_moe_layers: int = 2
    action_moe_num_heads: int = 8
    action_moe_mlp_dim: int = 1024
    action_moe_num_experts: int = 8
    action_moe_top_k: int = 2
    action_moe_expert_dim: int = 512
    action_moe_temperature: float = 1.0
    action_moe_prediction_loss_weight: float = 0.05
    action_moe_balance_loss_weight: float = 0.01
    # Opt-in task-level routing: select one top-k expert set for the complete
    # action chunk. Historical action-MoE checkpoints retain token-wise
    # routing because the default is disabled.
    action_moe_task_consistent_routing: bool = False
    # Infer a latent whole-episode progress state from current multimodal
    # context and robot state. Exact frame progress is training supervision
    # only; predicted progress tokens condition the inherited action flow.
    task_progress_reasoner: bool = False
    task_progress_hidden_dim: int = 256
    task_progress_layers: int = 2
    task_progress_num_heads: int = 8
    task_progress_mlp_dim: int = 1024
    task_progress_bins: int = 10
    task_progress_loss_weight: float = 0.05
    # Parse the contextualized instruction/observation representation into an
    # ordered bank of latent subgoals, infer the currently active slot from the
    # current observation and robot state, and condition every action position
    # on that slot.  Dataset frame progress and clean actions supervise the
    # latent automaton during training only; inference predicts the slot from
    # the current inputs.  The action-facing residual is zero initialized.
    language_subgoal_reasoner: bool = False
    language_subgoal_hidden_dim: int = 256
    language_subgoal_slots: int = 8
    language_subgoal_layers: int = 2
    language_subgoal_num_heads: int = 8
    language_subgoal_mlp_dim: int = 1024
    language_subgoal_temperature: float = 1.0
    language_subgoal_progress_loss_weight: float = 0.05
    language_subgoal_action_loss_weight: float = 0.05
    # Per-phase weights for the ordered progress objective.  These are plain
    # configuration values (not model leaves), so balancing semantic phases
    # cannot change checkpoint compatibility or step-zero inference.
    language_subgoal_phase_class_weights: tuple[float, ...] = (1.0,) * 8
    # Explicitly bind every ordered language subgoal to the competitive visual
    # object bank, then let action queries read the active subgoal/object pairs.
    # A dense clean-action auxiliary head trains the binding immediately while
    # its zero-initialized policy projection preserves the inherited policy.
    object_subgoal_binding: bool = False
    object_subgoal_binding_hidden_dim: int = 256
    object_subgoal_binding_layers: int = 2
    object_subgoal_binding_num_heads: int = 8
    object_subgoal_binding_mlp_dim: int = 1024
    object_subgoal_binding_temperature: float = 1.0
    object_subgoal_binding_action_loss_weight: float = 0.05
    # Relational instructions bind a manipulated object and a distinct
    # reference/receptacle object.  The unary/relational gate can suppress the
    # second role for grasp/reach instructions.
    object_subgoal_binding_distinct_roles: bool = True
    # Factor the seven deployed controls into translation, rotation, and
    # gripper streams before joint cross-time reasoning.  Separate auxiliary
    # decoders keep every stream identifiable while a zero-initialized fused
    # token head preserves the inherited continuous-flow policy exactly.
    kinematic_action_reasoner: bool = False
    kinematic_action_hidden_dim: int = 256
    kinematic_action_layers: int = 2
    kinematic_action_num_heads: int = 8
    kinematic_action_mlp_dim: int = 1024
    kinematic_action_prediction_loss_weight: float = 0.05
    # Represent the complete action chunk in an orthonormal DCT basis and let
    # every retained frequency token reason jointly with task context and
    # state.  A full inverse transform maps a zero-initialized residual back to
    # the original action positions; this is not output filtering or smoothing.
    spectral_action_reasoner: bool = False
    spectral_action_hidden_dim: int = 256
    spectral_action_layers: int = 2
    spectral_action_num_heads: int = 8
    spectral_action_mlp_dim: int = 1024
    spectral_action_bands: int = 3
    spectral_action_prediction_loss_weight: float = 0.05
    # Predict the remaining error of the inherited continuous-flow velocity
    # after reading its suffix hidden trajectory and algebraic clean-action
    # estimate.  An exact-zero per-axis gain preserves inherited sampling while
    # an auxiliary residual objective trains the compact refiner immediately.
    velocity_refiner: bool = False
    velocity_refiner_hidden_dim: int = 256
    velocity_refiner_layers: int = 2
    velocity_refiner_num_heads: int = 8
    velocity_refiner_mlp_dim: int = 1024
    velocity_refiner_loss_weight: float = 0.05
    # Re-read the complete contextualized visual/language prefix with the
    # inherited action proposal as ten mask-aware queries.  A residual target
    # trains the verifier immediately while an exact-zero per-axis gain keeps
    # the deployed Stage-6 velocity unchanged at inheritance.
    action_visual_refiner: bool = False
    action_visual_refiner_hidden_dim: int = 256
    action_visual_refiner_layers: int = 2
    action_visual_refiner_num_heads: int = 8
    action_visual_refiner_mlp_dim: int = 1024
    action_visual_refiner_loss_weight: float = 0.05
    # Specialize the contextual visual representation with a current-view,
    # masked-patch objective.  A compact bank of language/state-conditioned
    # scene queries reads camera/row/column-aware patches, reconstructs only
    # hidden frozen SigLIP targets during training, and supplies an exactly
    # zero-initialized action residual at deployment.  No masked/future input
    # or reconstruction target is required at inference.
    masked_spatial_reasoner: bool = False
    masked_spatial_hidden_dim: int = 256
    masked_spatial_queries: int = 16
    masked_spatial_layers: int = 2
    masked_spatial_num_heads: int = 8
    masked_spatial_mlp_dim: int = 1024
    masked_spatial_max_cameras: int = 3
    masked_spatial_max_grid_size: int = 16
    masked_spatial_mask_ratio: float = 0.5
    masked_spatial_reconstruction_loss_weight: float = 0.02
    masked_spatial_action_loss_weight: float = 0.05
    # Predict object-level future visual structure from the deployed current
    # cameras.  Language/state-conditioned object slots read the current patch
    # grid, a compact transition stack forecasts future slots, and a linear
    # cross-attention decoder reconstructs frozen future SigLIP embeddings.
    # Future RGB is training-only; a zero-initialized action head consumes the
    # forecast slots at inference without changing the observation contract.
    object_future_reasoner: bool = False
    object_future_hidden_dim: int = 256
    object_future_queries: int = 12
    object_future_layers: int = 2
    object_future_num_heads: int = 8
    object_future_mlp_dim: int = 1024
    object_future_max_grid_size: int = 16
    object_future_reconstruction_loss_weight: float = 0.02
    object_future_action_loss_weight: float = 0.05
    # Let the future predictor start from the same competitive object slots
    # that drive affordance reasoning.  This turns the two visual branches
    # into a current-object -> future-object transition model instead of two
    # unrelated residual heads.
    object_future_affordance_bridge: bool = False
    # Factor compositional instructions into target, predicate, and reference
    # roles, then bind those roles to a language-independent bank of visual
    # object slots.  A batchwise multi-positive image/text objective makes the
    # binding identifiable without boxes or evaluator-specific annotations;
    # zero-initialized action heads preserve the inherited policy exactly.
    predicate_binding_reasoner: bool = False
    predicate_binding_hidden_dim: int = 256
    predicate_binding_object_slots: int = 12
    predicate_binding_role_slots: int = 3
    predicate_binding_layers: int = 2
    predicate_binding_num_heads: int = 8
    predicate_binding_mlp_dim: int = 1024
    predicate_binding_max_cameras: int = 3
    predicate_binding_max_grid_size: int = 16
    predicate_binding_temperature: float = 0.07
    predicate_binding_contrastive_loss_weight: float = 0.02
    predicate_binding_action_loss_weight: float = 0.05
    # Select a sparse pair of task/state-conditioned low-rank experts and use
    # them to FiLM every valid image/language embedding before the PaliGemma
    # prefix pass.  This changes the layerwise KV representation consumed by
    # the action expert rather than adding another action-side residual.
    # Zero-initialized expert outputs make the adapter exactly identity at
    # inheritance; a Switch-style balance loss keeps the router from collapse.
    multimodal_prefix_moe: bool = False
    multimodal_prefix_moe_hidden_dim: int = 256
    multimodal_prefix_moe_expert_dim: int = 64
    multimodal_prefix_moe_num_experts: int = 8
    multimodal_prefix_moe_top_k: int = 2
    multimodal_prefix_moe_temperature: float = 1.0
    multimodal_prefix_moe_balance_loss_weight: float = 0.01
    # Independently specialize the complete layerwise prefix memory after the
    # PaliGemma pass.  Sparse experts emit distinct key/value FiLM residuals
    # for every VLM layer; zero output heads preserve the inherited cache.
    layerwise_kv_moe: bool = False
    layerwise_kv_moe_hidden_dim: int = 256
    layerwise_kv_moe_expert_dim: int = 64
    layerwise_kv_moe_num_experts: int = 8
    layerwise_kv_moe_top_k: int = 2
    layerwise_kv_moe_temperature: float = 1.0
    layerwise_kv_moe_balance_loss_weight: float = 0.01
    # Jointly fuse the complementary latent-future, state-rollout, and
    # whole-episode-progress reasoners, optionally with a fourth sparse action
    # expert branch. A context/state-conditioned gate is evaluated at every
    # action position. All component token heads remain zero-initialized, so
    # enabling the fusion preserves the inherited policy before fine-tuning.
    predictive_world_model_fusion: bool = False
    predictive_world_model_hidden_dim: int = 256
    predictive_world_model_auxiliary_scale: float = 1.0 / 3.0
    # Add the sparse action-expert pathway as a fourth calibrated branch.  Its
    # dense clean-action error supplies a directly comparable per-step
    # reliability target and its experts can specialize by manipulation phase.
    predictive_world_model_include_action_moe: bool = False
    # Calibrate a branch-reliability distribution from each prediction error
    # and inject the predicted reliability logits into the fusion gate.  The
    # reliability heads are zero initialized, preserving uniform initial gates.
    predictive_world_model_reliability_loss_weight: float = 0.02
    # A predictive branch's zero action-token projection is the sole
    # function-preserving gate.  Persistent descendants open the internal
    # context router at small scale so memory/program gradients do not wait for
    # a second zero projection.  Zero retains the legacy uniform router for
    # non-persistent configurations.
    predictive_world_model_router_init_scale: float = 0.0
    # Route the four evidence-backed Stage-20 components independently at
    # every action position.  The gate emits ``2 * sigmoid(logit)`` and its
    # zero-initialized output therefore starts at the exact identity
    # multiplier one.  A zero-initialized component-specific content scorer
    # additionally reads each actual residual, so routing can respond to local
    # evidence quality rather than task/state context alone while preserving
    # every independently validated residual exactly at initialization.
    evidence_combination_router: bool = False
    evidence_combination_router_hidden_dim: int = 256
    # Extend the evidence router with competitive object-affordance and
    # ordered language-subgoal residuals.  Both heads and both new router
    # logits start at exact zero/identity, so the larger hierarchy remains
    # bitwise function preserving at inheritance.
    evidence_combination_hierarchical_components: bool = False
    # Add the action-chunk verifier as a seventh content-routed Stage-20
    # component.  Its score and policy heads are zero initialized, so both the
    # verifier residual and its new identity gate preserve the inherited
    # policy exactly while hard-negative supervision teaches chunk validity.
    evidence_combination_action_verifier_component: bool = False
    # Route the object/subgoal binding residual as an eighth independently
    # content-gated component.  Both the residual and its gate contribution
    # start at exact zero/identity.
    evidence_combination_object_subgoal_binding_component: bool = False
    # Preserve the camera and 2-D patch topology explicitly after SigLIP and
    # let a small bank of language-conditioned relation queries read that
    # structured grid.  Waypoint queries then turn the relation tokens into a
    # zero-initialized residual for the continuous action expert.  A coarse
    # action auxiliary objective supplies trajectory supervision without
    # requiring object boxes or evaluator-specific relation labels.
    spatial_relation_reasoner: bool = False
    spatial_relation_hidden_dim: int = 256
    spatial_relation_queries: int = 8
    spatial_relation_layers: int = 2
    spatial_relation_num_heads: int = 8
    spatial_relation_mlp_dim: int = 1024
    spatial_relation_max_cameras: int = 3
    spatial_relation_max_grid_size: int = 16
    spatial_relation_loss_weight: float = 0.05
    # Partition the camera-aware patch grid into language-conditioned,
    # competitive object slots before relational action reasoning.  Unlike
    # independent relation queries, patch-to-slot assignments normalize across
    # slots, which encourages distinct target/reference/distractor bindings.
    object_affordance_graph_reasoner: bool = False
    object_affordance_hidden_dim: int = 256
    object_affordance_slots: int = 8
    object_affordance_layers: int = 2
    object_affordance_num_heads: int = 8
    object_affordance_mlp_dim: int = 1024
    object_affordance_max_cameras: int = 3
    object_affordance_max_grid_size: int = 16
    object_affordance_temperature: float = 1.0
    object_affordance_loss_weight: float = 0.05
    # Dense slot-to-patch reconstruction prevents the competitive object slots
    # from collapsing to interchangeable task vectors.  It is training-only
    # and disabled by default for historical object-affordance checkpoints.
    object_affordance_reconstruction_loss_weight: float = 0.0
    # Predict an explicit per-step manipulation phase (approach/open, close,
    # carry/closed, release) and inject its embedding into the continuous
    # action expert.  The rare transition classes receive inverse-frequency
    # weighting derived once from the complete training corpus.
    contact_phase_reasoner: bool = False
    contact_phase_hidden_dim: int = 256
    contact_phase_layers: int = 2
    contact_phase_num_heads: int = 8
    contact_phase_mlp_dim: int = 1024
    contact_phase_temperature: float = 0.5
    contact_phase_loss_weight: float = 0.05
    contact_phase_gripper_index: int = 6
    # Optional scalar gripper indexes for robots with one independently
    # commanded aperture per arm.  When non-empty, phase labels are computed
    # for every listed arm and reduced to one global manipulation phase.  The
    # legacy single-gripper/finger-pair path remains the default so historical
    # checkpoints keep identical behavior and parameter graphs.
    contact_phase_gripper_indices: tuple[int, ...] = ()
    contact_phase_state_scalar_indices: tuple[int, ...] = ()
    # ALOHA-style normalized scalar apertures are positive when open, whereas
    # the historical LIBERO action convention is negative when open.
    contact_phase_open_when_positive: bool = False
    # LIBERO state stores the two normalized finger positions at indexes 6/7.
    # Their signed aperture supplies the pre-chunk gripper state, so a close or
    # release on the first predicted action is not mislabeled as a steady phase.
    contact_phase_state_gripper_indices: tuple[int, int] = (6, 7)
    contact_phase_state_open_threshold: float = 0.0
    contact_phase_class_counts: tuple[int, int, int, int] = (
        1_761_766,
        148_394,
        1_677_941,
        29_259,
    )
    # Gamma zero, temperature one and unit boosts reproduce the inherited
    # inverse-frequency weighted cross entropy exactly.  Positive gamma and
    # transition boosts focus learning on rare close/release mistakes.
    contact_phase_focal_gamma: float = 0.0
    contact_phase_loss_temperature: float = 1.0
    contact_phase_transition_boosts: tuple[float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    # Couple competitive object-affordance slots to the explicit manipulation
    # phase and the private cross-replan memory.  A per-step risk head is
    # supervised by object-assignment entropy plus rare close/release phases;
    # its zero-output policy projection preserves the PPWM parent exactly.
    contact_affordance_predictive_fusion: bool = False
    contact_affordance_clause_plan_verification: bool = False
    # Require contact verification to inspect the action-conditioned future
    # prediction, closing the object/plan/contact loop before policy output.
    contact_affordance_future_verification: bool = False
    # Feed PSM's grounded target/reference bindings and factorized relation
    # state directly into the final plan/contact verifier.  These tensors are
    # produced from ordinary prompt spans and visual patches at inference;
    # no evaluator metadata or compact training labels are consumed here.
    contact_affordance_relation_verification: bool = False
    # Replace the fixed current/next-program blend with PSM's supervised
    # stay/advance distribution and expose its progress-verification state to
    # the final action verifier.
    contact_affordance_transition_verification: bool = False
    contact_affordance_fusion_hidden_dim: int = 256
    contact_affordance_fusion_layers: int = 2
    contact_affordance_fusion_num_heads: int = 8
    contact_affordance_fusion_mlp_dim: int = 1024
    contact_affordance_risk_loss_weight: float = 0.02
    # Directly align the factorized language relation with the grounded
    # visual/role relation.  Without this objective the relation bridge is
    # trained only indirectly through action and contact-risk losses.
    contact_affordance_relation_contrastive_loss_weight: float = 0.0
    contact_affordance_relation_contrastive_temperature: float = 0.1
    # Encode several retrieved atomic demonstrations independently and learn
    # which candidate is the active unfinished subgoal.  The single-demo path
    # remains unchanged unless this flag is enabled.
    compositional_demo_routing: bool = False
    compositional_demo_slots: int = 3
    compositional_demo_router_layers: int = 2
    compositional_demo_router_temperature: float = 0.5
    compositional_demo_router_loss_weight: float = 0.1
    # Replace unconditional addition of the implicit, explicit, retrieved-demo,
    # and discrete-code action-token paths with a state/context-conditioned
    # per-step router.  The score projection is zero initialized and the
    # softmax weights are multiplied by the path count, so the new graph is
    # exactly the inherited four-path sum before fine-tuning.
    reasoning_pathway_router: bool = False
    reasoning_pathway_router_hidden_dim: int = 256
    reasoning_pathway_router_temperature: float = 1.0
    # Dynamically gate the three cumulative best-anchor specialists using the
    # shared contextual prior and proprioceptive state.  Zero score weights
    # produce unit gates (softmax * 3), preserving the ungated specialist
    # computation exactly at initialization.
    specialist_module_router: bool = False
    specialist_module_router_hidden_dim: int = 256
    specialist_module_router_temperature: float = 1.0
    # Penalize only batch-global expert imbalance.  Per-sample gates may stay
    # sharp, so this prevents collapse without forcing every sample uniform.
    specialist_module_router_balance_loss_weight: float = 0.01
    # Independently test a richer alternative to scalar routing: jointly mix
    # every action-step/path token with self-attention and action-prior context,
    # then add a zero-initialized residual to the inherited four-path sum.
    reasoning_pathway_interaction: bool = False
    reasoning_pathway_interaction_hidden_dim: int = 256
    reasoning_pathway_interaction_layers: int = 2
    reasoning_pathway_interaction_num_heads: int = 4
    reasoning_pathway_interaction_mlp_dim: int = 512
    # Jointly predict a global whole-chunk code, a sequence of per-step action
    # codes, and the normal continuous flow. The hierarchical fixed codebooks
    # retain both coordinated motion intent and local gripper/rotation changes
    # without quantizing the final controller output.
    discrete_action_codebook_path: str | None = None
    discrete_action_codebook_hidden_dim: int = 512
    discrete_action_codebook_loss_weight: float = 0.1
    discrete_action_step_codebook_loss_weight: float = 0.05
    # Stage 6 uses supervised code labels to establish the hierarchy. Later
    # stages can retain hard code routing and its straight-through flow
    # gradient without repeatedly optimizing the already learned labels.
    discrete_action_auxiliary_loss: bool = True
    discrete_action_codebook_temperature: float = 0.5
    discrete_action_codebook_robot_dim: int = 7
    # Inverse-sqrt frequency weighting keeps rare coordination/transition
    # codes visible without letting a handful of tiny clusters dominate.
    discrete_action_class_weight_clip: float = 5.0
    discrete_action_step_reasoner_layers: int = 2
    discrete_action_step_reasoner_num_heads: int = 8
    discrete_action_step_reasoner_mlp_dim: int = 1024

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, 'max_token_len', 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, 'discrete_state_input', self.pi05)
        if self.active_action_dim is not None and not (
            1 <= self.active_action_dim <= self.action_dim
        ):
            raise ValueError('active_action_dim must be within action_dim')
        if self.action_prior and self.action_horizon % self.action_prior_horizon:
            raise ValueError('action_horizon must be divisible by action_prior_horizon')
        if self.main_flow_samples < 1:
            raise ValueError('main_flow_samples must be positive')
        if self.main_flow_samples > 1 and not self.action_prior_contextual:
            raise ValueError(
                'multiple main flow samples require a contextual cached prefix'
            )
        if self.action_prior_contextual and not self.action_prior:
            raise ValueError(
                'action_prior must be enabled for a contextual action prior'
            )
        if self.context_adarms:
            if not self.pi05:
                raise ValueError('context AdaRMS requires pi0.5')
            if not self.action_prior_contextual:
                raise ValueError(
                    'context AdaRMS requires a contextual action prior'
                )
            if self.context_adarms_hidden_dim < 1:
                raise ValueError('context_adarms_hidden_dim must be positive')
        if self.persistent_memory_adarms:
            if not self.pi05:
                raise ValueError('persistent-memory AdaRMS requires pi0.5')
            if not self.persistent_subgoal_memory:
                raise ValueError(
                    'persistent-memory AdaRMS requires persistent subgoal memory'
                )
            if self.persistent_memory_adarms_hidden_dim < 1:
                raise ValueError(
                    'persistent_memory_adarms_hidden_dim must be positive'
                )
        if self.phase_contact_action_film:
            if not self.pi05:
                raise ValueError('phase-contact action FiLM requires pi0.5')
            if not self.persistent_subgoal_memory:
                raise ValueError(
                    'phase-contact action FiLM requires persistent subgoal memory'
                )
            if not self.contact_phase_reasoner:
                raise ValueError(
                    'phase-contact action FiLM requires contact phase reasoning'
                )
            if not self.contact_affordance_predictive_fusion:
                raise ValueError(
                    'phase-contact action FiLM requires risk-conditioned '
                    'contact-affordance fusion'
                )
            if self.phase_contact_action_film_hidden_dim < 1:
                raise ValueError(
                    'phase_contact_action_film_hidden_dim must be positive'
                )
        if self.layerwise_persistent_memory_attention:
            if not (
                self.pi05
                and self.persistent_subgoal_memory
                and self.persistent_memory_adarms
                and self.phase_contact_action_film
                and self.layerwise_persistent_memory_attention_rank == 32
                and self.layerwise_persistent_memory_attention_alpha == 32.0
                and self.layerwise_persistent_memory_attention_layers == (5, 11, 17)
                and self.action_expert_variant == 'gemma_300m_lora'
            ):
                raise ValueError(
                    'layerwise persistent-memory attention requires the '
                    'PhaseContact parent, Gemma-300M-LoRA, rank32/alpha32, '
                    'and layers5/11/17'
                )
        if self.action_prior_state_conditioning and not self.action_prior:
            raise ValueError(
                'action_prior must be enabled for state-conditioned queries'
            )
        if self.persistent_action_prior_conditioning:
            if not self.action_prior_contextual:
                raise ValueError(
                    'persistent action-prior conditioning requires a contextual '
                    'action prior'
                )
            if not self.persistent_subgoal_memory:
                raise ValueError(
                    'persistent action-prior conditioning requires persistent '
                    'subgoal memory'
                )
        if self.action_prior_target not in ('mean', 'endpoint'):
            raise ValueError("action_prior_target must be either 'mean' or 'endpoint'")
        if self.hierarchical_event_transition_memory:
            if not self.pi05:
                raise ValueError('HETM requires pi0.5')
            if not self.action_prior_contextual:
                raise ValueError('HETM requires contextual prefix states')
            if self.action_horizon < 5 or self.active_action_dim != 7:
                raise ValueError(
                    'HETM requires at least five actions and active_action_dim=7'
                )
            weights = (
                self.hetm_event_loss_weight,
                self.hetm_predicate_loss_weight,
                self.hetm_frontier_loss_weight,
                self.hetm_transition_loss_weight,
                self.hetm_monotonic_loss_weight,
                self.hetm_psm_frontier_consistency_loss_weight,
            )
            if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
                raise ValueError('HETM auxiliary weights must be finite and positive')
        if self.role_affordance_causal_graph:
            if not self.pi05:
                raise ValueError('RACG requires pi0.5')
            if not self.hierarchical_event_transition_memory:
                raise ValueError('RACG requires HETM')
            if not self.action_prior_contextual:
                raise ValueError('RACG requires contextual prefix states')
            if self.action_horizon != 10:
                raise ValueError('RACG-v1 requires action_horizon=10')
            if (self.active_action_dim or self.action_dim) != 7:
                raise ValueError('RACG-v1 requires seven active action dimensions')
            weights = (
                self.racg_role_align_loss_weight,
                self.racg_relation_loss_weight,
                self.racg_crossview_loss_weight,
                self.racg_slot_reconstruction_loss_weight,
                self.racg_contact_loss_weight,
                self.racg_identity_transition_loss_weight,
                self.racg_slot_diversity_loss_weight,
            )
            if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
                raise ValueError(
                    'production RACG auxiliary weights must be finite and positive'
                )
        if self.racg_external_geometry_prior and not self.role_affordance_causal_graph:
            raise ValueError('external geometry prior requires RACG')
        if self.racg_external_geometry_hmca:
            if not self.racg_external_geometry_prior:
                raise ValueError('geometry-HMCA requires the external geometry prior')
            if not self.persistent_hmca_v4:
                raise ValueError('geometry-HMCA requires HMCA-v4')
            if self.racg_external_geometry_hmca_hidden_dim <= 0:
                raise ValueError('geometry-HMCA hidden_dim must be positive')
        if self.racg_graph_hmca:
            if not self.role_affordance_causal_graph:
                raise ValueError('graph-HMCA requires RACG')
            if not self.persistent_hmca_v4:
                raise ValueError('graph-HMCA requires HMCA-v4')
            if self.racg_graph_hmca_hidden_dim <= 0:
                raise ValueError('graph-HMCA hidden_dim must be positive')
        if self.dual_action_reasoner:
            if not self.action_prior_contextual:
                raise ValueError(
                    'a dual action reasoner requires a contextual action prior'
                )
            if self.action_prior_target != 'endpoint':
                raise ValueError(
                    "a dual action reasoner requires action_prior_target='endpoint'"
                )
            if not self.implicit_action_reasoner_layers:
                raise ValueError('implicit_action_reasoner_layers must not be empty')
            if any(
                layer < 0 or layer >= _gemma.get_config(self.paligemma_variant).depth
                for layer in self.implicit_action_reasoner_layers
            ):
                raise ValueError(
                    'implicit action-reasoner layer indexes must be within the PaliGemma depth'
                )
            if len(set(self.implicit_action_reasoner_layers)) != len(
                self.implicit_action_reasoner_layers
            ):
                raise ValueError(
                    'implicit action-reasoner layer indexes must be unique'
                )
            if self.implicit_action_reasoner_pool_stride < 1:
                raise ValueError(
                    'implicit_action_reasoner_pool_stride must be positive'
                )
            if self.implicit_action_reasoner_layerwise_guidance:
                if self.implicit_action_reasoner_group_size < 1:
                    raise ValueError(
                        'implicit_action_reasoner_group_size must be positive'
                    )
                if len(self.implicit_action_reasoner_layers) % (
                    self.implicit_action_reasoner_group_size
                ):
                    raise ValueError(
                        'implicit action-reasoner layer count must be divisible by group size'
                    )
                if self.implicit_action_reasoner_num_heads < 1:
                    raise ValueError(
                        'implicit_action_reasoner_num_heads must be positive'
                    )
                if self.implicit_action_reasoner_downsample_dim < 1:
                    raise ValueError(
                        'implicit_action_reasoner_downsample_dim must be positive'
                    )
                if self.implicit_action_reasoner_downsample_dim % (
                    self.implicit_action_reasoner_num_heads
                ):
                    raise ValueError(
                        'implicit downsample dimension must be divisible by head count'
                    )
            if self.explicit_action_reasoner_layers < 1:
                raise ValueError('explicit_action_reasoner_layers must be positive')
            if self.explicit_action_reasoner_num_heads < 1:
                raise ValueError('explicit_action_reasoner_num_heads must be positive')
            if (
                self.explicit_action_reasoner_hidden_dim
                % self.explicit_action_reasoner_num_heads
            ):
                raise ValueError(
                    'explicit_action_reasoner_hidden_dim must be divisible by explicit_action_reasoner_num_heads'
                )
            if self.explicit_action_reasoner_mlp_dim < 1:
                raise ValueError('explicit_action_reasoner_mlp_dim must be positive')
            if self.explicit_action_reasoner_loss_weight <= 0:
                raise ValueError(
                    'explicit_action_reasoner_loss_weight must be positive'
                )
            if self.explicit_action_reasoner_flow_samples < 1:
                raise ValueError(
                    'explicit_action_reasoner_flow_samples must be positive'
                )
            if self.explicit_action_reasoner_inference_steps < 1:
                raise ValueError(
                    'explicit_action_reasoner_inference_steps must be positive'
                )
        if self.retrieved_demo_conditioning:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'retrieved demonstration conditioning requires the dual action reasoner'
                )
            if self.retrieved_demo_layers < 1:
                raise ValueError('retrieved_demo_layers must be positive')
            if self.retrieved_demo_num_heads < 1:
                raise ValueError('retrieved_demo_num_heads must be positive')
            if self.retrieved_demo_hidden_dim % self.retrieved_demo_num_heads:
                raise ValueError(
                    'retrieved_demo_hidden_dim must be divisible by retrieved_demo_num_heads'
                )
            if self.retrieved_demo_mlp_dim < 1:
                raise ValueError('retrieved_demo_mlp_dim must be positive')
            if self.retrieved_demo_plan_steps < 2:
                raise ValueError('retrieved_demo_plan_steps must be at least two')
            if self.retrieved_demo_plan_dim < 1:
                raise ValueError('retrieved_demo_plan_dim must be positive')
            if self.retrieved_demo_reliability_loss_weight <= 0:
                raise ValueError(
                    'retrieved_demo_reliability_loss_weight must be positive'
                )
        if self.persistent_subgoal_memory:
            if not self.pi05:
                raise ValueError('persistent subgoal memory requires pi0.5')
            if not self.dual_action_reasoner:
                raise ValueError(
                    'persistent subgoal memory requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'persistent subgoal memory requires a contextual action prior'
                )
            if self.persistent_memory_tokens != 8:
                raise ValueError('persistent_memory_tokens must equal eight')
            if self.persistent_memory_subgoal_slots != 8:
                raise ValueError(
                    'persistent_memory_subgoal_slots must equal eight'
                )
            if self.persistent_memory_hidden_dim < 1:
                raise ValueError(
                    'persistent_memory_hidden_dim must be positive'
                )
            if not 0 < self.persistent_memory_fast_tokens < self.persistent_memory_tokens:
                raise ValueError(
                    'persistent_memory_fast_tokens must split the memory bank'
                )
            if not (
                0.0 < self.persistent_memory_slow_update_rate
                < self.persistent_memory_fast_update_rate
                <= 1.0
            ):
                raise ValueError(
                    'persistent memory update rates must be bounded and slow < fast'
                )
            if self.persistent_memory_previous_action_steps != 5:
                raise ValueError(
                    'persistent previous actions must match replan_steps=5'
                )
            active_action_dim = self.active_action_dim or self.action_dim
            memory_action_dim = (
                active_action_dim
                if self.persistent_memory_action_dim is None
                else self.persistent_memory_action_dim
            )
            if not 1 <= memory_action_dim <= active_action_dim:
                raise ValueError(
                    'persistent_memory_action_dim must be within active action dim'
                )
            if memory_action_dim != active_action_dim and (
                active_action_dim != 2 * memory_action_dim
            ):
                raise ValueError(
                    'a reduced persistent memory action interface requires two '
                    'ordered equal-width arm chunks'
                )
            if (
                self.persistent_memory_short_replans != 4
                or self.persistent_memory_long_replans != 8
            ):
                raise ValueError(
                    'persistent memory requires the audited 4/8 replan mixture'
                )
            if self.persistent_memory_flow_replan_indices != (0, 2, 4, 7):
                raise ValueError(
                    'persistent memory flow supervision must use replans 0/2/4/7'
                )
            if not 0.0 < self.persistent_memory_long_probability < 1.0:
                raise ValueError(
                    'persistent_memory_long_probability must lie in (0, 1)'
                )
            if self.persistent_memory_cache_refresh_steps != 1000:
                raise ValueError(
                    'persistent memory cache refresh must remain 1000 steps'
                )
            if (
                self.persistent_memory_max_staleness_steps
                != self.persistent_memory_cache_refresh_steps
            ):
                raise ValueError(
                    'persistent memory cache staleness must equal refresh interval'
                )
            if self.persistent_memory_policy_gain != 0.0:
                raise ValueError(
                    'persistent_memory_policy_gain must initialize at exact zero'
                )
            if not (
                math.isfinite(self.persistent_memory_compositional_phase_init_scale)
                and 0.0
                < self.persistent_memory_compositional_phase_init_scale
                <= 0.05
            ):
                raise ValueError(
                    'persistent memory compositional phase init scale must lie '
                    'in (0, 0.05]'
                )
            if self.persistent_memory_policy_gain_warmup_steps < 1:
                raise ValueError(
                    'persistent memory policy gain warmup must be positive'
                )
            if (
                self.persistent_memory_auxiliary_decay_steps
                <= self.persistent_memory_policy_gain_warmup_steps
            ):
                raise ValueError(
                    'persistent memory auxiliary decay must end after warmup'
                )
            if not 0.0 < self.persistent_memory_auxiliary_final_multiplier <= 1.0:
                raise ValueError(
                    'persistent memory final auxiliary multiplier must lie in (0, 1]'
                )
            persistent_loss_weights = (
                self.persistent_memory_subgoal_progress_loss_weight,
                self.persistent_memory_subgoal_transition_loss_weight,
                self.persistent_memory_action_verification_loss_weight,
                self.persistent_memory_causal_route_loss_weight,
                self.persistent_memory_cross_camera_loss_weight,
                self.persistent_memory_role_distinctness_loss_weight,
                self.persistent_memory_role_object_exclusivity_loss_weight,
                self.persistent_memory_between_pair_entropy_loss_weight,
                self.persistent_memory_source_reference_loss_weight,
                self.persistent_memory_destination_reference_loss_weight,
                self.persistent_memory_condition_state_loss_weight,
                self.persistent_memory_destination_qualifier_loss_weight,
                self.persistent_memory_temporal_role_loss_weight,
                self.persistent_cross_view_role_contrastive_loss_weight,
                self.persistent_contact_risk_auxiliary_loss_weight,
                self.persistent_relational_role_auxiliary_loss_weight,
                self.persistent_clause_role_binding_auxiliary_loss_weight,
                self.persistent_memory_factorized_role_loss_weight,
                self.persistent_memory_factor_attention_alignment_loss_weight,
                self.persistent_memory_object_slot_reconstruction_loss_weight,
                self.persistent_memory_visual_language_role_loss_weight,
                self.supervised_role_identity_contrastive_loss_weight,
            )
            if any(weight <= 0 for weight in persistent_loss_weights):
                raise ValueError(
                    'persistent memory auxiliary loss weights must be positive'
                )
            if (
                len(self.persistent_memory_semantic_phase_class_counts)
                != self.persistent_memory_subgoal_slots
                or any(
                    count <= 0
                    for count in self.persistent_memory_semantic_phase_class_counts
                )
            ):
                raise ValueError(
                    'persistent semantic phase class counts must match positive slots'
                )
            if (
                len(self.persistent_memory_flow_phase_class_counts)
                != self.persistent_memory_subgoal_slots
                or any(
                    count <= 0
                    for count in self.persistent_memory_flow_phase_class_counts
                )
            ):
                raise ValueError(
                    'persistent flow phase class counts must match positive slots'
                )
            if self.persistent_memory_flow_phase_count_floor < 1:
                raise ValueError(
                    'persistent flow phase count floor must be positive'
                )
            if self.persistent_memory_paraphrase_loss_weight != 0.0:
                raise ValueError(
                    'persistent memory paraphrase loss must remain disabled '
                    'because production preserves the inherited parent prompt'
                )
            if not self.supervised_role_identity_contrastive_loss:
                raise ValueError(
                    'persistent subgoal memory requires supervised role identity contrastive loss'
                )
            if (
                self.persistent_memory_role_contrastive_group_size < 4
                or self.persistent_memory_role_contrastive_group_size % 2
            ):
                raise ValueError(
                    'persistent memory role contrastive groups must contain '
                    'an even number of at least four examples'
                )
            if self.persistent_conditional_memory_policy_bridge:
                if not self.persistent_structured_demo_language:
                    raise ValueError(
                        'conditional memory-policy bridge requires PSM-SDLA'
                    )
                if (
                    self.persistent_conditional_memory_policy_bridge_rank != 128
                    or self.persistent_memory_hidden_dim != 256
                    or self.persistent_memory_tokens != 8
                    or self.persistent_memory_subgoal_slots != 8
                    or self.action_horizon != 10
                ):
                    raise ValueError(
                        'PSM-SDLA-v3 bridge requires rank128/H256/M8/S8/T10'
                    )
            if self.persistent_geometry_aux_v1:
                if not (
                    self.persistent_subgoal_memory
                    and self.persistent_structured_demo_language
                    and self.persistent_conditional_memory_policy_bridge
                    and self.action_prior_contextual
                    and self.main_flow_samples == 1
                    and self.explicit_action_reasoner_flow_samples == 2
                    and self.explicit_action_reasoner_inference_steps == 4
                    and self.persistent_geometry_aux_bottleneck_dim == 128
                    and self.persistent_memory_hidden_dim == 256
                    and self.action_horizon == 10
                ):
                    raise ValueError(
                        'geometry-v1 requires contextual SDLA-v3, one main flow, explicit 2/4, rank128/H256/T10'
                    )
            if self.persistent_geometry_external_residual_v1:
                if not (
                    self.persistent_geometry_aux_v1
                    and self.persistent_clause_plan_v1
                    and self.object_future_reasoner
                    and self.contact_affordance_predictive_fusion
                    and self.contact_affordance_clause_plan_verification
                    and self.contact_affordance_future_verification
                    and self.contact_affordance_relation_verification
                    and self.contact_affordance_transition_verification
                ):
                    raise ValueError(
                        'external geometry residual requires the complete verified-contact ClausePlan-v3 parent'
                    )
            if self.persistent_temporal_role_memory_v1:
                if not (
                    self.persistent_geometry_external_residual_v1
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                    and self.persistent_memory_tokens >= 6
                    and self.persistent_clause_plan_v1
                ):
                    raise ValueError(
                        'temporal role memory requires DualGeometry, H256, replan5, and ClausePlan'
                    )
            if self.persistent_cross_view_role_consensus_v1:
                if not (
                    self.persistent_temporal_role_memory_v1
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                    and self.persistent_clause_plan_v1
                ):
                    raise ValueError(
                        'cross-view role consensus requires TemporalRoleMemory, H256, replan5, and ClausePlan'
                    )
                if self.persistent_cross_view_role_contrastive_temperature <= 0.0:
                    raise ValueError(
                        'cross-view role contrastive temperature must be positive'
                    )
            if self.persistent_contact_risk_calibrated_role_residual_v1:
                if not (
                    self.persistent_temporal_role_memory_v1
                    and self.persistent_cross_view_role_consensus_v1
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                ):
                    raise ValueError(
                        'contact-risk role residual requires TemporalRoleMemory, '
                        'CrossViewConsensus, H256, and replan5'
                    )
                if (
                    self.persistent_contact_risk_auxiliary_loss_weight <= 0.0
                    or len(self.persistent_contact_risk_class_weights) != 3
                    or any(
                        weight <= 0.0
                        for weight in self.persistent_contact_risk_class_weights
                    )
                ):
                    raise ValueError('contact-risk loss contract is invalid')
            if self.persistent_relational_role_composer_residual_v1:
                if not (
                    self.persistent_contact_risk_calibrated_role_residual_v1
                    and self.persistent_temporal_role_memory_v1
                    and self.persistent_cross_view_role_consensus_v1
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                ):
                    raise ValueError(
                        'relational role composer requires ContactRisk, '
                        'TemporalRoleMemory, CrossViewConsensus, H256, and replan5'
                    )
                if (
                    self.persistent_relational_role_auxiliary_loss_weight <= 0.0
                    or len(self.persistent_relational_role_class_weights) != 16
                    or any(
                        weight <= 0.0
                        for weight in self.persistent_relational_role_class_weights
                    )
                ):
                    raise ValueError('relational role composer loss contract is invalid')
            if self.persistent_clause_role_binding_verifier_v1:
                if not (
                    self.persistent_relational_role_composer_residual_v1
                    and self.persistent_clause_plan_v1
                    and self.persistent_memory_subgoal_slots == 8
                    and self.persistent_clause_plan_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                ):
                    raise ValueError(
                        'clause-role binding verifier requires RelationalRole, '
                        'ClausePlan S8, H256, and replan5'
                    )
                if (
                    self.persistent_clause_role_binding_auxiliary_loss_weight <= 0.0
                    or len(
                        self.persistent_clause_role_binding_source_class_weights
                    )
                    != 4
                    or len(
                        self.persistent_clause_role_binding_destination_class_weights
                    )
                    != 4
                    or any(
                        weight <= 0.0
                        for weight in (
                            *self.persistent_clause_role_binding_source_class_weights,
                            *self.persistent_clause_role_binding_destination_class_weights,
                        )
                    )
                ):
                    raise ValueError('clause-role binding loss contract is invalid')
            if self.persistent_semantic_frontier_completion_verifier_v1:
                if not (
                    self.persistent_clause_role_binding_verifier_v1
                    and self.persistent_memory_subgoal_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                ):
                    raise ValueError(
                        'semantic frontier completion requires ClauseRole, S8, '
                        'H256, and replan5'
                    )
                weights = self.persistent_semantic_frontier_completion_class_weights
                if (
                    self.persistent_semantic_frontier_completion_auxiliary_loss_weight
                    <= 0.0
                    or len(weights) != 8
                    or any(
                        len(row) != 2 or any(weight <= 0.0 for weight in row)
                        for row in weights
                    )
                ):
                    raise ValueError(
                        'semantic frontier completion loss contract is invalid'
                    )
            if self.persistent_causal_frontier_transition_gate_v1:
                if not (
                    self.persistent_memory_subgoal_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                ):
                    raise ValueError(
                        'causal frontier transition requires S8, H256, and replan5'
                    )
            if self.persistent_hierarchical_clause_event_alignment_v1:
                if not (
                    self.persistent_clause_plan_v1
                    and self.persistent_semantic_frontier_completion_verifier_v1
                    and self.persistent_memory_subgoal_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                    and self.persistent_hierarchical_clause_event_alignment_auxiliary_loss_weight
                    > 0.0
                ):
                    raise ValueError(
                        'HCEA requires ClausePlan, semantic frontier, S8, H256, '
                        'replan5, and a positive auxiliary weight'
                    )
            if self.persistent_hcea_causal_recovery_action_experts_v1:
                if not (
                    self.persistent_hierarchical_clause_event_alignment_v1
                    and self.persistent_memory_subgoal_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                    and self.action_horizon == 10
                    and self.persistent_hcea_causal_recovery_intent_loss_weight > 0.0
                ):
                    raise ValueError(
                        'HCEA recovery experts require HCEA, S8, H256, T10, '
                        'replan5, and a positive causal intent loss'
                    )
            if self.persistent_hcea_causal_role_identity_transport_expert_v1:
                if not (
                    self.persistent_hierarchical_clause_event_alignment_v1
                    and self.persistent_memory_subgoal_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                    and self.persistent_memory_previous_action_steps == 5
                    and self.action_horizon == 10
                    and self.persistent_hcea_causal_role_identity_transport_loss_weight
                    > 0.0
                ):
                    raise ValueError(
                        'HCEA role transport requires HCEA, S8, H256, T10, '
                        'replan5, and a positive causal transport loss'
                    )
            if self.persistent_hmca_v4:
                if not (
                    self.persistent_geometry_aux_v1
                    and self.persistent_hmca_v4_rank == 16
                    and self.persistent_hmca_v4_alpha == 16.0
                    and self.persistent_hmca_v4_layers == (5, 11, 17)
                    and self.action_expert_variant == 'gemma_300m_lora'
                    and self.action_horizon == 10
                ):
                    raise ValueError(
                        'HMCA-v4 requires Geometry-47, Gemma-300M-LoRA, rank16/alpha16/T10 and layers5/11/17'
                    )
            if self.persistent_clause_plan_v1:
                if not (
                    self.persistent_hmca_v4
                    and self.persistent_clause_plan_rank == 128
                    and self.persistent_clause_plan_slots == 8
                    and self.persistent_clause_plan_monotonic_strength == 4.0
                    and self.persistent_memory_subgoal_slots == 8
                    and self.persistent_memory_hidden_dim == 256
                ):
                    raise ValueError(
                        'ClausePlan-v1 requires HMCA-v4, rank128, monotonic strength4, and eight H256 plan slots'
                    )
            if self.persistent_structured_demo_language:
                exact_dimensions = {
                    'structured_demo_hidden_dim': (
                        self.structured_demo_hidden_dim,
                        self.persistent_memory_hidden_dim,
                    ),
                    'structured_demo_semantic_dim': (
                        self.structured_demo_semantic_dim,
                        2048,
                    ),
                    'structured_demo_semantic_slots': (
                        self.structured_demo_semantic_slots,
                        8,
                    ),
                    'structured_demo_prompt_steps': (
                        self.structured_demo_prompt_steps,
                        48,
                    ),
                    'structured_demo_plan_steps': (
                        self.structured_demo_plan_steps,
                        10,
                    ),
                    'structured_demo_plan_dim': (
                        self.structured_demo_plan_dim,
                        17,
                    ),
                    'structured_demo_action_steps': (
                        self.structured_demo_action_steps,
                        self.action_horizon,
                    ),
                    'spatial_language_vocabulary_size': (
                        self.spatial_language_vocabulary_size,
                        128,
                    ),
                    'spatial_language_steps': (
                        self.spatial_language_steps,
                        32,
                    ),
                }
                drifted = [
                    name
                    for name, (observed, expected) in exact_dimensions.items()
                    if observed != expected
                ]
                if drifted:
                    raise ValueError(
                        'PSM-SDLA audited dimensions drifted: '
                        f'{drifted}'
                    )
                if not (
                    0 <= self.spatial_language_bos_token_id
                    < self.spatial_language_vocabulary_size
                ):
                    raise ValueError('spatial-language BOS id is out of range')
                if (
                    self.spatial_language_auxiliary_initial_weight != 0.05
                    or self.spatial_language_auxiliary_peak_weight != 0.10
                    or self.spatial_language_auxiliary_final_weight != 0.02
                    or self.spatial_language_auxiliary_warmup_steps != 1_000
                    or self.spatial_language_auxiliary_decay_start_step != 15_000
                    or self.spatial_language_auxiliary_total_steps != 30_000
                ):
                    raise ValueError(
                        'PSM-SDLA auxiliary schedule must remain '
                        '0.05->0.10->0.02 over 0/1000/15000/30000'
                    )
        elif (
            self.persistent_structured_demo_language
            or self.persistent_conditional_memory_policy_bridge
            or self.persistent_geometry_aux_v1
        ):
            raise ValueError('PSM-SDLA and its v3 bridge require persistent memory')
        if self.structured_rationale_reasoner:
            if not self.retrieved_demo_conditioning:
                raise ValueError(
                    'structured rationale reasoning requires retrieved demonstrations'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'structured rationale reasoning requires a contextual action prior'
                )
            if self.structured_rationale_hidden_dim < 1:
                raise ValueError(
                    'structured_rationale_hidden_dim must be positive'
                )
            if self.structured_rationale_layers < 1:
                raise ValueError('structured_rationale_layers must be positive')
            if self.structured_rationale_num_heads < 1:
                raise ValueError(
                    'structured_rationale_num_heads must be positive'
                )
            if self.structured_rationale_hidden_dim % (
                self.structured_rationale_num_heads
            ):
                raise ValueError(
                    'structured rationale hidden dimension must be divisible by heads'
                )
            if self.structured_rationale_mlp_dim < 1:
                raise ValueError('structured_rationale_mlp_dim must be positive')
            if self.structured_rationale_temperature <= 0:
                raise ValueError(
                    'structured_rationale_temperature must be positive'
                )
            if self.structured_rationale_loss_weight <= 0:
                raise ValueError(
                    'structured_rationale_loss_weight must be positive'
                )
            axis_count = 7
            if self.active_action_dim is not None and self.active_action_dim < axis_count:
                raise ValueError(
                    'structured rationale reasoning requires at least seven active actions'
                )
            if any(
                len(values) != axis_count
                for values in (
                    self.structured_rationale_neutral_eps,
                    self.structured_rationale_action_q01,
                    self.structured_rationale_action_q99,
                )
            ):
                raise ValueError(
                    'structured rationale action statistics must contain seven values'
                )
            if len(self.structured_rationale_class_weights) != axis_count or any(
                len(weights) != 3 or any(weight <= 0 for weight in weights)
                for weights in self.structured_rationale_class_weights
            ):
                raise ValueError(
                    'structured rationale class weights must be a positive 7x3 matrix'
                )
            if any(
                high <= low
                for low, high in zip(
                    self.structured_rationale_action_q01,
                    self.structured_rationale_action_q99,
                    strict=True,
                )
            ):
                raise ValueError(
                    'structured rationale q99 values must exceed q01 values'
                )
        if self.action_chunk_verifier:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'action chunk verification requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'action chunk verification requires a contextual action prior'
                )
            if self.action_chunk_verifier_hidden_dim < 1:
                raise ValueError(
                    'action_chunk_verifier_hidden_dim must be positive'
                )
            if self.action_chunk_verifier_layers < 1:
                raise ValueError('action_chunk_verifier_layers must be positive')
            if self.action_chunk_verifier_num_heads < 1:
                raise ValueError(
                    'action_chunk_verifier_num_heads must be positive'
                )
            if self.action_chunk_verifier_hidden_dim % (
                self.action_chunk_verifier_num_heads
            ):
                raise ValueError(
                    'action chunk verifier hidden dimension must be divisible by heads'
                )
            if self.action_chunk_verifier_mlp_dim < 1:
                raise ValueError(
                    'action_chunk_verifier_mlp_dim must be positive'
                )
            if self.action_chunk_verifier_temperature <= 0:
                raise ValueError(
                    'action_chunk_verifier_temperature must be positive'
                )
            if self.action_chunk_verifier_loss_weight <= 0:
                raise ValueError(
                    'action_chunk_verifier_loss_weight must be positive'
                )
        if self.latent_future_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'latent future reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'latent future reasoning requires a contextual action prior'
                )
            if self.latent_future_hidden_dim < 1:
                raise ValueError('latent_future_hidden_dim must be positive')
            if self.latent_future_layers < 1:
                raise ValueError('latent_future_layers must be positive')
            if self.latent_future_num_heads < 1:
                raise ValueError('latent_future_num_heads must be positive')
            if self.latent_future_hidden_dim % self.latent_future_num_heads:
                raise ValueError(
                    'latent future hidden dimension must be divisible by heads'
                )
            if self.latent_future_mlp_dim < 1:
                raise ValueError('latent_future_mlp_dim must be positive')
            if self.latent_future_grid_size < 1:
                raise ValueError('latent_future_grid_size must be positive')
            if self.latent_future_loss_weight <= 0:
                raise ValueError('latent_future_loss_weight must be positive')
        if self.state_rollout_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'state rollout reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'state rollout reasoning requires a contextual action prior'
                )
            if self.state_rollout_hidden_dim < 1:
                raise ValueError('state_rollout_hidden_dim must be positive')
            if self.state_rollout_layers < 1:
                raise ValueError('state_rollout_layers must be positive')
            if self.state_rollout_num_heads < 1:
                raise ValueError('state_rollout_num_heads must be positive')
            if self.state_rollout_hidden_dim % self.state_rollout_num_heads:
                raise ValueError(
                    'state rollout hidden dimension must be divisible by heads'
                )
            if self.state_rollout_mlp_dim < 1:
                raise ValueError('state_rollout_mlp_dim must be positive')
            if not 1 <= self.state_rollout_target_dim <= self.action_dim:
                raise ValueError(
                    'state_rollout_target_dim must be within the model state width'
                )
            if self.state_rollout_loss_weight <= 0:
                raise ValueError('state_rollout_loss_weight must be positive')
        if self.action_moe_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'action MoE reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'action MoE reasoning requires a contextual action prior'
                )
            if self.action_moe_hidden_dim < 1:
                raise ValueError('action_moe_hidden_dim must be positive')
            if self.action_moe_layers < 1:
                raise ValueError('action_moe_layers must be positive')
            if self.action_moe_num_heads < 1:
                raise ValueError('action_moe_num_heads must be positive')
            if self.action_moe_hidden_dim % self.action_moe_num_heads:
                raise ValueError(
                    'action MoE hidden dimension must be divisible by heads'
                )
            if self.action_moe_mlp_dim < 1:
                raise ValueError('action_moe_mlp_dim must be positive')
            if self.action_moe_num_experts < 2:
                raise ValueError('action_moe_num_experts must be at least two')
            if not 1 <= self.action_moe_top_k <= self.action_moe_num_experts:
                raise ValueError(
                    'action_moe_top_k must be within the expert count'
                )
            if self.action_moe_expert_dim < 1:
                raise ValueError('action_moe_expert_dim must be positive')
            if self.action_moe_temperature <= 0:
                raise ValueError('action_moe_temperature must be positive')
            if self.action_moe_prediction_loss_weight <= 0:
                raise ValueError(
                    'action_moe_prediction_loss_weight must be positive'
                )
            if self.action_moe_balance_loss_weight <= 0:
                raise ValueError(
                    'action_moe_balance_loss_weight must be positive'
                )
        if self.task_progress_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'task progress reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'task progress reasoning requires a contextual action prior'
                )
            if self.task_progress_hidden_dim < 1:
                raise ValueError('task_progress_hidden_dim must be positive')
            if self.task_progress_layers < 1:
                raise ValueError('task_progress_layers must be positive')
            if self.task_progress_num_heads < 1:
                raise ValueError('task_progress_num_heads must be positive')
            if self.task_progress_hidden_dim % self.task_progress_num_heads:
                raise ValueError(
                    'task progress hidden dimension must be divisible by heads'
                )
            if self.task_progress_mlp_dim < 1:
                raise ValueError('task_progress_mlp_dim must be positive')
            if self.task_progress_bins < 2:
                raise ValueError('task_progress_bins must be at least two')
            if self.task_progress_loss_weight <= 0:
                raise ValueError('task_progress_loss_weight must be positive')
        if self.language_subgoal_reasoner:
            if not self.action_prior_contextual:
                raise ValueError(
                    'language subgoal reasoning requires a contextual action prior'
                )
            if self.language_subgoal_hidden_dim < 1:
                raise ValueError('language_subgoal_hidden_dim must be positive')
            if self.language_subgoal_slots < 2:
                raise ValueError('language_subgoal_slots must be at least two')
            if self.language_subgoal_layers < 1:
                raise ValueError('language_subgoal_layers must be positive')
            if self.language_subgoal_num_heads < 1:
                raise ValueError('language_subgoal_num_heads must be positive')
            if (
                self.language_subgoal_hidden_dim
                % self.language_subgoal_num_heads
            ):
                raise ValueError(
                    'language subgoal hidden dimension must be divisible by heads'
                )
            if self.language_subgoal_mlp_dim < 1:
                raise ValueError('language_subgoal_mlp_dim must be positive')
            if self.language_subgoal_temperature <= 0:
                raise ValueError('language_subgoal_temperature must be positive')
            if self.language_subgoal_progress_loss_weight <= 0:
                raise ValueError(
                    'language_subgoal_progress_loss_weight must be positive'
                )
            if self.language_subgoal_action_loss_weight <= 0:
                raise ValueError(
                    'language_subgoal_action_loss_weight must be positive'
                )
            if (
                len(self.language_subgoal_phase_class_weights)
                != self.language_subgoal_slots
                or any(
                    not math.isfinite(weight) or weight <= 0
                    for weight in self.language_subgoal_phase_class_weights
                )
            ):
                raise ValueError(
                    'language_subgoal_phase_class_weights must contain one '
                    'positive finite weight per slot'
                )
        if self.object_subgoal_binding:
            if not (
                self.object_affordance_graph_reasoner
                and self.language_subgoal_reasoner
            ):
                raise ValueError(
                    'object-subgoal binding requires object-affordance and '
                    'language-subgoal reasoners'
                )
            if self.object_subgoal_binding_hidden_dim < 1:
                raise ValueError(
                    'object_subgoal_binding_hidden_dim must be positive'
                )
            if self.object_subgoal_binding_layers < 1:
                raise ValueError('object_subgoal_binding_layers must be positive')
            if self.object_subgoal_binding_num_heads < 1:
                raise ValueError(
                    'object_subgoal_binding_num_heads must be positive'
                )
            if (
                self.object_subgoal_binding_hidden_dim
                % self.object_subgoal_binding_num_heads
            ):
                raise ValueError(
                    'object-subgoal binding hidden dimension must be divisible '
                    'by heads'
                )
            if self.object_subgoal_binding_mlp_dim < 1:
                raise ValueError('object_subgoal_binding_mlp_dim must be positive')
            if self.object_subgoal_binding_temperature <= 0:
                raise ValueError(
                    'object_subgoal_binding_temperature must be positive'
                )
            if self.object_subgoal_binding_action_loss_weight <= 0:
                raise ValueError(
                    'object_subgoal_binding_action_loss_weight must be positive'
                )
        if self.kinematic_action_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'kinematic action reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'kinematic action reasoning requires a contextual action prior'
                )
            if (self.active_action_dim or self.action_dim) < 7:
                raise ValueError(
                    'kinematic action reasoning requires seven active controls'
                )
            if self.kinematic_action_hidden_dim < 1:
                raise ValueError('kinematic_action_hidden_dim must be positive')
            if self.kinematic_action_layers < 1:
                raise ValueError('kinematic_action_layers must be positive')
            if self.kinematic_action_num_heads < 1:
                raise ValueError('kinematic_action_num_heads must be positive')
            if (
                self.kinematic_action_hidden_dim
                % self.kinematic_action_num_heads
            ):
                raise ValueError(
                    'kinematic action hidden dimension must be divisible by heads'
                )
            if self.kinematic_action_mlp_dim < 1:
                raise ValueError('kinematic_action_mlp_dim must be positive')
            if self.kinematic_action_prediction_loss_weight <= 0:
                raise ValueError(
                    'kinematic_action_prediction_loss_weight must be positive'
                )
        if self.spectral_action_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'spectral action reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'spectral action reasoning requires a contextual action prior'
                )
            if self.spectral_action_hidden_dim < 1:
                raise ValueError('spectral_action_hidden_dim must be positive')
            if self.spectral_action_layers < 1:
                raise ValueError('spectral_action_layers must be positive')
            if self.spectral_action_num_heads < 1:
                raise ValueError('spectral_action_num_heads must be positive')
            if self.spectral_action_hidden_dim % self.spectral_action_num_heads:
                raise ValueError(
                    'spectral action hidden dimension must be divisible by heads'
                )
            if self.spectral_action_mlp_dim < 1:
                raise ValueError('spectral_action_mlp_dim must be positive')
            if not 1 < self.spectral_action_bands <= self.action_horizon:
                raise ValueError(
                    'spectral_action_bands must be between two and action_horizon'
                )
            if self.spectral_action_prediction_loss_weight <= 0:
                raise ValueError(
                    'spectral_action_prediction_loss_weight must be positive'
                )
        if self.velocity_refiner:
            if not self.pi05:
                raise ValueError('velocity refinement requires pi0.5')
            if not self.action_prior_contextual:
                raise ValueError(
                    'velocity refinement requires a contextual action prior'
                )
            if self.velocity_refiner_hidden_dim < 1:
                raise ValueError('velocity_refiner_hidden_dim must be positive')
            if self.velocity_refiner_layers < 1:
                raise ValueError('velocity_refiner_layers must be positive')
            if self.velocity_refiner_num_heads < 1:
                raise ValueError('velocity_refiner_num_heads must be positive')
            if self.velocity_refiner_hidden_dim % self.velocity_refiner_num_heads:
                raise ValueError(
                    'velocity refiner hidden dimension must be divisible by heads'
                )
            if self.velocity_refiner_mlp_dim < 1:
                raise ValueError('velocity_refiner_mlp_dim must be positive')
            if self.velocity_refiner_loss_weight <= 0:
                raise ValueError('velocity_refiner_loss_weight must be positive')
        if self.action_visual_refiner:
            if not self.pi05:
                raise ValueError('action visual refinement requires pi0.5')
            if not self.dual_action_reasoner:
                raise ValueError(
                    'action visual refinement requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'action visual refinement requires a contextual action prior'
                )
            # Standalone architecture screens keep these refiners independent,
            # but the audited evidence-combination candidate composes them in
            # a deliberate cascade: temporal/context velocity correction first,
            # then dense visual-prefix correction of the remaining residual.
            if self.velocity_refiner and not self.evidence_combination_router:
                raise ValueError(
                    'action visual and velocity refiners are independent candidates'
                )
            if self.action_visual_refiner_hidden_dim < 1:
                raise ValueError(
                    'action_visual_refiner_hidden_dim must be positive'
                )
            if self.action_visual_refiner_layers < 1:
                raise ValueError('action_visual_refiner_layers must be positive')
            if self.action_visual_refiner_num_heads < 1:
                raise ValueError(
                    'action_visual_refiner_num_heads must be positive'
                )
            if (
                self.action_visual_refiner_hidden_dim
                % self.action_visual_refiner_num_heads
            ):
                raise ValueError(
                    'action visual refiner hidden dimension must be divisible by heads'
                )
            if self.action_visual_refiner_mlp_dim < 1:
                raise ValueError(
                    'action_visual_refiner_mlp_dim must be positive'
                )
            if self.action_visual_refiner_loss_weight <= 0:
                raise ValueError(
                    'action_visual_refiner_loss_weight must be positive'
                )
        if self.masked_spatial_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'masked spatial reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'masked spatial reasoning requires a contextual action prior'
                )
            if self.masked_spatial_hidden_dim < 1:
                raise ValueError('masked_spatial_hidden_dim must be positive')
            if self.masked_spatial_queries < 1:
                raise ValueError('masked_spatial_queries must be positive')
            if self.masked_spatial_layers < 1:
                raise ValueError('masked_spatial_layers must be positive')
            if self.masked_spatial_num_heads < 1:
                raise ValueError('masked_spatial_num_heads must be positive')
            if self.masked_spatial_hidden_dim % self.masked_spatial_num_heads:
                raise ValueError(
                    'masked_spatial_hidden_dim must be divisible by '
                    'masked_spatial_num_heads'
                )
            if self.masked_spatial_mlp_dim < 1:
                raise ValueError('masked_spatial_mlp_dim must be positive')
            if self.masked_spatial_max_cameras < 1:
                raise ValueError('masked_spatial_max_cameras must be positive')
            if self.masked_spatial_max_grid_size < 1:
                raise ValueError('masked_spatial_max_grid_size must be positive')
            if not 0 < self.masked_spatial_mask_ratio < 1:
                raise ValueError('masked_spatial_mask_ratio must be in (0, 1)')
            if self.masked_spatial_reconstruction_loss_weight <= 0:
                raise ValueError(
                    'masked_spatial_reconstruction_loss_weight must be positive'
                )
            if self.masked_spatial_action_loss_weight <= 0:
                raise ValueError(
                    'masked_spatial_action_loss_weight must be positive'
                )
        if self.object_future_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'object future reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'object future reasoning requires a contextual action prior'
                )
            if self.object_future_hidden_dim < 1:
                raise ValueError('object_future_hidden_dim must be positive')
            if self.object_future_queries < 1:
                raise ValueError('object_future_queries must be positive')
            if self.object_future_layers < 1:
                raise ValueError('object_future_layers must be positive')
            if self.object_future_num_heads < 1:
                raise ValueError('object_future_num_heads must be positive')
            if self.object_future_hidden_dim % self.object_future_num_heads:
                raise ValueError(
                    'object_future_hidden_dim must be divisible by '
                    'object_future_num_heads'
                )
            if self.object_future_mlp_dim < 1:
                raise ValueError('object_future_mlp_dim must be positive')
            if self.object_future_max_grid_size < 1:
                raise ValueError('object_future_max_grid_size must be positive')
            if self.object_future_reconstruction_loss_weight <= 0:
                raise ValueError(
                    'object_future_reconstruction_loss_weight must be positive'
                )
            if self.object_future_action_loss_weight <= 0:
                raise ValueError(
                    'object_future_action_loss_weight must be positive'
                )
            if (
                self.object_future_affordance_bridge
                and not self.object_affordance_graph_reasoner
            ):
                raise ValueError(
                    'object future affordance bridge requires object affordance'
                )
        elif self.object_future_affordance_bridge:
            raise ValueError(
                'object future affordance bridge requires object future reasoning'
            )
        if self.predicate_binding_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'predicate binding reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'predicate binding reasoning requires a contextual action prior'
                )
            if self.predicate_binding_hidden_dim < 1:
                raise ValueError('predicate_binding_hidden_dim must be positive')
            if self.predicate_binding_object_slots < 2:
                raise ValueError(
                    'predicate_binding_object_slots must be at least two'
                )
            if self.predicate_binding_role_slots != 3:
                raise ValueError(
                    'predicate_binding_role_slots must be exactly three'
                )
            if self.predicate_binding_layers < 1:
                raise ValueError('predicate_binding_layers must be positive')
            if self.predicate_binding_num_heads < 1:
                raise ValueError('predicate_binding_num_heads must be positive')
            if (
                self.predicate_binding_hidden_dim
                % self.predicate_binding_num_heads
            ):
                raise ValueError(
                    'predicate_binding_hidden_dim must be divisible by '
                    'predicate_binding_num_heads'
                )
            if self.predicate_binding_mlp_dim < 1:
                raise ValueError('predicate_binding_mlp_dim must be positive')
            if self.predicate_binding_max_cameras < 1:
                raise ValueError(
                    'predicate_binding_max_cameras must be positive'
                )
            if self.predicate_binding_max_grid_size < 1:
                raise ValueError(
                    'predicate_binding_max_grid_size must be positive'
                )
            if self.predicate_binding_temperature <= 0:
                raise ValueError(
                    'predicate_binding_temperature must be positive'
                )
            if self.predicate_binding_contrastive_loss_weight <= 0:
                raise ValueError(
                    'predicate_binding_contrastive_loss_weight must be positive'
                )
            if self.predicate_binding_action_loss_weight <= 0:
                raise ValueError(
                    'predicate_binding_action_loss_weight must be positive'
                )
        if self.multimodal_prefix_moe:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'multimodal prefix MoE requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'multimodal prefix MoE requires a contextual action prior'
                )
            if self.multimodal_prefix_moe_hidden_dim < 1:
                raise ValueError(
                    'multimodal_prefix_moe_hidden_dim must be positive'
                )
            if self.multimodal_prefix_moe_expert_dim < 1:
                raise ValueError(
                    'multimodal_prefix_moe_expert_dim must be positive'
                )
            if self.multimodal_prefix_moe_num_experts < 2:
                raise ValueError(
                    'multimodal_prefix_moe_num_experts must be at least two'
                )
            if not 1 <= self.multimodal_prefix_moe_top_k < (
                self.multimodal_prefix_moe_num_experts
            ):
                raise ValueError(
                    'multimodal_prefix_moe_top_k must be positive and smaller '
                    'than the expert count'
                )
            if self.multimodal_prefix_moe_temperature <= 0:
                raise ValueError(
                    'multimodal_prefix_moe_temperature must be positive'
                )
            if self.multimodal_prefix_moe_balance_loss_weight <= 0:
                raise ValueError(
                    'multimodal_prefix_moe_balance_loss_weight must be positive'
                )
        if self.layerwise_kv_moe:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'layerwise KV MoE requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'layerwise KV MoE requires a contextual action prior'
                )
            if self.layerwise_kv_moe_hidden_dim < 1:
                raise ValueError('layerwise_kv_moe_hidden_dim must be positive')
            if self.layerwise_kv_moe_expert_dim < 1:
                raise ValueError('layerwise_kv_moe_expert_dim must be positive')
            if self.layerwise_kv_moe_num_experts < 2:
                raise ValueError(
                    'layerwise_kv_moe_num_experts must be at least two'
                )
            if not 1 <= self.layerwise_kv_moe_top_k < (
                self.layerwise_kv_moe_num_experts
            ):
                raise ValueError(
                    'layerwise_kv_moe_top_k must be positive and smaller '
                    'than the expert count'
                )
            if self.layerwise_kv_moe_temperature <= 0:
                raise ValueError('layerwise_kv_moe_temperature must be positive')
            if self.layerwise_kv_moe_balance_loss_weight <= 0:
                raise ValueError(
                    'layerwise_kv_moe_balance_loss_weight must be positive'
                )
        if self.predictive_world_model_fusion:
            if not (
                self.latent_future_reasoner
                and self.state_rollout_reasoner
                and self.task_progress_reasoner
            ):
                raise ValueError(
                    'predictive world-model fusion requires latent-future, '
                    'state-rollout, and task-progress reasoners'
                )
            if self.predictive_world_model_hidden_dim < 1:
                raise ValueError(
                    'predictive_world_model_hidden_dim must be positive'
                )
            if not 0 < self.predictive_world_model_auxiliary_scale <= 1:
                raise ValueError(
                    'predictive_world_model_auxiliary_scale must be in (0, 1]'
                )
            if self.predictive_world_model_reliability_loss_weight <= 0:
                raise ValueError(
                    'predictive_world_model_reliability_loss_weight must be positive'
                )
            if not (
                math.isfinite(self.predictive_world_model_router_init_scale)
                and 0.0 <= self.predictive_world_model_router_init_scale <= 0.05
            ):
                raise ValueError(
                    'predictive world-model router init scale must lie in [0, 0.05]'
                )
            if (
                self.predictive_world_model_include_action_moe
                and not self.action_moe_reasoner
            ):
                raise ValueError(
                    'predictive world-model action-MoE fusion requires '
                    'action_moe_reasoner'
                )
        elif self.predictive_world_model_include_action_moe:
            raise ValueError(
                'predictive_world_model_include_action_moe requires predictive fusion'
            )
        if self.evidence_combination_router:
            if not (
                self.predictive_world_model_fusion
                and self.spatial_relation_reasoner
                and self.contact_phase_reasoner
                and self.reasoning_pathway_interaction
            ):
                raise ValueError(
                    'evidence combination routing requires predictive, spatial, '
                    'contact, and pathway-interaction reasoners'
                )
            if self.evidence_combination_router_hidden_dim < 1:
                raise ValueError(
                    'evidence_combination_router_hidden_dim must be positive'
                )
            if self.evidence_combination_hierarchical_components and not (
                self.object_affordance_graph_reasoner
                and self.language_subgoal_reasoner
            ):
                raise ValueError(
                    'hierarchical evidence combination requires object-affordance '
                    'and language-subgoal reasoners'
                )
        elif self.evidence_combination_hierarchical_components:
            raise ValueError(
                'hierarchical evidence components require evidence combination routing'
            )
        if self.evidence_combination_action_verifier_component:
            if not self.evidence_combination_router:
                raise ValueError(
                    'evidence verifier component requires evidence combination routing'
                )
            if not self.action_chunk_verifier:
                raise ValueError(
                    'evidence verifier component requires action chunk verification'
                )
        if self.evidence_combination_object_subgoal_binding_component:
            if not self.evidence_combination_router:
                raise ValueError(
                    'evidence object-subgoal component requires evidence routing'
                )
            if not self.object_subgoal_binding:
                raise ValueError(
                    'evidence object-subgoal component requires object-subgoal '
                    'binding'
                )
        if self.compositional_demo_routing:
            if not self.retrieved_demo_conditioning:
                raise ValueError(
                    'compositional demo routing requires retrieved demo conditioning'
                )
            if self.compositional_demo_slots < 2:
                raise ValueError(
                    'compositional_demo_slots must be at least two'
                )
            if self.compositional_demo_router_layers < 1:
                raise ValueError(
                    'compositional_demo_router_layers must be positive'
                )
            if self.compositional_demo_router_temperature <= 0:
                raise ValueError(
                    'compositional_demo_router_temperature must be positive'
                )
            if self.compositional_demo_router_loss_weight <= 0:
                raise ValueError(
                    'compositional_demo_router_loss_weight must be positive'
                )
        if self.reasoning_pathway_router:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'reasoning pathway routing requires the dual action reasoner'
                )
            if not self.retrieved_demo_conditioning:
                raise ValueError(
                    'reasoning pathway routing requires retrieved demonstrations'
                )
            if self.discrete_action_codebook_path is None:
                raise ValueError(
                    'reasoning pathway routing requires the discrete action codebook'
                )
            if self.reasoning_pathway_router_hidden_dim < 1:
                raise ValueError(
                    'reasoning_pathway_router_hidden_dim must be positive'
                )
            if self.reasoning_pathway_router_temperature <= 0:
                raise ValueError(
                    'reasoning_pathway_router_temperature must be positive'
                )
        if self.specialist_module_router:
            if not (
                self.context_adarms
                and self.velocity_refiner
                and self.language_subgoal_reasoner
            ):
                raise ValueError(
                    'specialist module routing requires ContextAdaRMS, '
                    'VelocityRefiner, and LanguageSubgoal'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'specialist module routing requires contextual priors'
                )
            if self.specialist_module_router_hidden_dim < 1:
                raise ValueError(
                    'specialist_module_router_hidden_dim must be positive'
                )
            if self.specialist_module_router_temperature <= 0:
                raise ValueError(
                    'specialist_module_router_temperature must be positive'
                )
            if self.specialist_module_router_balance_loss_weight <= 0:
                raise ValueError(
                    'specialist_module_router_balance_loss_weight must be positive'
                )
        if self.reasoning_pathway_interaction:
            if self.reasoning_pathway_router and not self.evidence_combination_router:
                raise ValueError(
                    'reasoning pathway interaction and scalar routing may only '
                    'be combined by the evidence-combination candidate'
                )
            if not self.dual_action_reasoner:
                raise ValueError(
                    'reasoning pathway interaction requires the dual action reasoner'
                )
            if not self.retrieved_demo_conditioning:
                raise ValueError(
                    'reasoning pathway interaction requires retrieved demonstrations'
                )
            if self.discrete_action_codebook_path is None:
                raise ValueError(
                    'reasoning pathway interaction requires the discrete action codebook'
                )
            if self.reasoning_pathway_interaction_hidden_dim < 1:
                raise ValueError(
                    'reasoning_pathway_interaction_hidden_dim must be positive'
                )
            if self.reasoning_pathway_interaction_layers < 1:
                raise ValueError(
                    'reasoning_pathway_interaction_layers must be positive'
                )
            if self.reasoning_pathway_interaction_num_heads < 1:
                raise ValueError(
                    'reasoning_pathway_interaction_num_heads must be positive'
                )
            if self.reasoning_pathway_interaction_hidden_dim % (
                self.reasoning_pathway_interaction_num_heads
            ):
                raise ValueError(
                    'reasoning pathway interaction hidden dimension must be divisible by heads'
                )
            if self.reasoning_pathway_interaction_mlp_dim < 1:
                raise ValueError(
                    'reasoning_pathway_interaction_mlp_dim must be positive'
                )
        if self.spatial_relation_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'spatial relation reasoning requires the dual action reasoner'
                )
            if self.spatial_relation_hidden_dim < 1:
                raise ValueError('spatial_relation_hidden_dim must be positive')
            if self.spatial_relation_queries < 1:
                raise ValueError('spatial_relation_queries must be positive')
            if self.spatial_relation_layers < 1:
                raise ValueError('spatial_relation_layers must be positive')
            if self.spatial_relation_num_heads < 1:
                raise ValueError('spatial_relation_num_heads must be positive')
            if self.spatial_relation_hidden_dim % self.spatial_relation_num_heads:
                raise ValueError(
                    'spatial_relation_hidden_dim must be divisible by '
                    'spatial_relation_num_heads'
                )
            if self.spatial_relation_mlp_dim < 1:
                raise ValueError('spatial_relation_mlp_dim must be positive')
            if self.spatial_relation_max_cameras < 1:
                raise ValueError('spatial_relation_max_cameras must be positive')
            if self.spatial_relation_max_grid_size < 1:
                raise ValueError('spatial_relation_max_grid_size must be positive')
            if self.spatial_relation_loss_weight <= 0:
                raise ValueError('spatial_relation_loss_weight must be positive')
        if self.object_affordance_graph_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'object affordance graph reasoning requires the dual action reasoner'
                )
            if not self.action_prior_contextual:
                raise ValueError(
                    'object affordance graph reasoning requires a contextual action prior'
                )
            if self.object_affordance_hidden_dim < 1:
                raise ValueError('object_affordance_hidden_dim must be positive')
            if self.object_affordance_slots < 2:
                raise ValueError('object_affordance_slots must be at least two')
            if self.object_affordance_layers < 1:
                raise ValueError('object_affordance_layers must be positive')
            if self.object_affordance_num_heads < 1:
                raise ValueError('object_affordance_num_heads must be positive')
            if (
                self.object_affordance_hidden_dim
                % self.object_affordance_num_heads
            ):
                raise ValueError(
                    'object_affordance_hidden_dim must be divisible by '
                    'object_affordance_num_heads'
                )
            if self.object_affordance_mlp_dim < 1:
                raise ValueError('object_affordance_mlp_dim must be positive')
            if self.object_affordance_max_cameras < 1:
                raise ValueError('object_affordance_max_cameras must be positive')
            if self.object_affordance_max_grid_size < 1:
                raise ValueError(
                    'object_affordance_max_grid_size must be positive'
                )
            if self.object_affordance_temperature <= 0:
                raise ValueError('object_affordance_temperature must be positive')
            if self.object_affordance_loss_weight <= 0:
                raise ValueError('object_affordance_loss_weight must be positive')
            if not (
                math.isfinite(self.object_affordance_reconstruction_loss_weight)
                and 0.0
                <= self.object_affordance_reconstruction_loss_weight
                <= 1.0
            ):
                raise ValueError(
                    'object_affordance_reconstruction_loss_weight must be finite '
                    'and in [0, 1]'
                )
        if self.contact_phase_reasoner:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'contact phase reasoning requires the dual action reasoner'
                )
            if self.contact_phase_hidden_dim < 1:
                raise ValueError('contact_phase_hidden_dim must be positive')
            if self.contact_phase_layers < 1:
                raise ValueError('contact_phase_layers must be positive')
            if self.contact_phase_num_heads < 1:
                raise ValueError('contact_phase_num_heads must be positive')
            if self.contact_phase_hidden_dim % self.contact_phase_num_heads:
                raise ValueError(
                    'contact_phase_hidden_dim must be divisible by '
                    'contact_phase_num_heads'
                )
            if self.contact_phase_mlp_dim < 1:
                raise ValueError('contact_phase_mlp_dim must be positive')
            if self.contact_phase_temperature <= 0:
                raise ValueError('contact_phase_temperature must be positive')
            if self.contact_phase_loss_weight <= 0:
                raise ValueError('contact_phase_loss_weight must be positive')
            active_dim = self.active_action_dim or self.action_dim
            if not 0 <= self.contact_phase_gripper_index < active_dim:
                raise ValueError(
                    'contact_phase_gripper_index must be within active_action_dim'
                )
            if self.contact_phase_gripper_indices:
                if (
                    len(self.contact_phase_gripper_indices)
                    != len(self.contact_phase_state_scalar_indices)
                    or len(set(self.contact_phase_gripper_indices))
                    != len(self.contact_phase_gripper_indices)
                    or len(set(self.contact_phase_state_scalar_indices))
                    != len(self.contact_phase_state_scalar_indices)
                    or any(
                        not 0 <= index < active_dim
                        for index in self.contact_phase_gripper_indices
                    )
                    or any(
                        not 0 <= index < self.action_dim
                        for index in self.contact_phase_state_scalar_indices
                    )
                ):
                    raise ValueError(
                        'contact_phase_gripper_indices and '
                        'contact_phase_state_scalar_indices must contain '
                        'equally many unique indexes within their action/state '
                        'dimensions'
                    )
            elif self.contact_phase_state_scalar_indices:
                raise ValueError(
                    'contact_phase_state_scalar_indices requires '
                    'contact_phase_gripper_indices'
                )
            if (
                len(self.contact_phase_state_gripper_indices) != 2
                or len(set(self.contact_phase_state_gripper_indices)) != 2
                or any(
                    not 0 <= index < self.action_dim
                    for index in self.contact_phase_state_gripper_indices
                )
            ):
                raise ValueError(
                    'contact_phase_state_gripper_indices must contain two '
                    'distinct indexes within action_dim'
                )
            if not math.isfinite(self.contact_phase_state_open_threshold):
                raise ValueError(
                    'contact_phase_state_open_threshold must be finite'
                )
            if (
                len(self.contact_phase_class_counts) != 4
                or any(count <= 0 for count in self.contact_phase_class_counts)
            ):
                raise ValueError(
                    'contact_phase_class_counts must contain four positive counts'
                )
            if (
                not math.isfinite(self.contact_phase_focal_gamma)
                or self.contact_phase_focal_gamma < 0
            ):
                raise ValueError(
                    'contact_phase_focal_gamma must be finite and nonnegative'
                )
            if (
                not math.isfinite(self.contact_phase_loss_temperature)
                or self.contact_phase_loss_temperature <= 0
            ):
                raise ValueError(
                    'contact_phase_loss_temperature must be finite and positive'
                )
            if (
                len(self.contact_phase_transition_boosts) != 4
                or any(
                    not math.isfinite(value) or value <= 0
                    for value in self.contact_phase_transition_boosts
                )
            ):
                raise ValueError(
                    'contact_phase_transition_boosts must contain four positive values'
                )
        if self.contact_affordance_predictive_fusion:
            if not (
                self.object_affordance_graph_reasoner
                and self.contact_phase_reasoner
                and self.persistent_subgoal_memory
            ):
                raise ValueError(
                    'contact-affordance predictive fusion requires object '
                    'affordance, contact phase, and persistent memory'
                )
            if self.contact_affordance_fusion_hidden_dim < 1:
                raise ValueError(
                    'contact_affordance_fusion_hidden_dim must be positive'
                )
            if self.contact_affordance_fusion_layers < 1:
                raise ValueError(
                    'contact_affordance_fusion_layers must be positive'
                )
            if self.contact_affordance_fusion_num_heads < 1:
                raise ValueError(
                    'contact_affordance_fusion_num_heads must be positive'
                )
            if self.contact_affordance_fusion_hidden_dim % (
                self.contact_affordance_fusion_num_heads
            ):
                raise ValueError(
                    'contact-affordance fusion hidden dimension must be '
                    'divisible by its head count'
                )
            if self.contact_affordance_fusion_mlp_dim < 1:
                raise ValueError(
                    'contact_affordance_fusion_mlp_dim must be positive'
                )
            if self.contact_affordance_risk_loss_weight <= 0:
                raise ValueError(
                    'contact_affordance_risk_loss_weight must be positive'
                )
            if (
                self.contact_affordance_clause_plan_verification
                and not self.persistent_clause_plan_v1
            ):
                raise ValueError(
                    'contact-affordance clause verification requires ClausePlan-v1'
                )
            if (
                self.contact_affordance_future_verification
                and not (
                    self.object_future_reasoner
                    and self.object_future_affordance_bridge
                )
            ):
                raise ValueError(
                    'contact-affordance future verification requires bridged '
                    'object future reasoning'
                )
            if (
                self.contact_affordance_relation_verification
                and not self.persistent_subgoal_memory
            ):
                raise ValueError(
                    'contact-affordance relation verification requires the '
                    'persistent factorized plan'
                )
            if self.contact_affordance_relation_verification:
                if self.contact_affordance_relation_contrastive_loss_weight <= 0:
                    raise ValueError(
                        'relation verification requires a positive direct '
                        'contrastive loss weight'
                    )
                if (
                    not math.isfinite(
                        self.contact_affordance_relation_contrastive_temperature
                    )
                    or self.contact_affordance_relation_contrastive_temperature
                    <= 0
                ):
                    raise ValueError(
                        'relation contrastive temperature must be finite and '
                        'positive'
                    )
            if (
                self.contact_affordance_transition_verification
                and not self.persistent_subgoal_memory
            ):
                raise ValueError(
                    'contact-affordance transition verification requires PSM'
                )
        elif (
            self.contact_affordance_clause_plan_verification
            or self.contact_affordance_future_verification
            or self.contact_affordance_relation_verification
            or self.contact_affordance_transition_verification
        ):
            raise ValueError(
                'contact-affordance verification requires contact fusion'
            )
        if self.discrete_action_codebook_path is not None:
            if not self.dual_action_reasoner:
                raise ValueError(
                    'discrete action-code prediction requires the dual action reasoner'
                )
            if self.discrete_action_codebook_hidden_dim < 1:
                raise ValueError(
                    'discrete_action_codebook_hidden_dim must be positive'
                )
            if self.discrete_action_codebook_loss_weight <= 0:
                raise ValueError(
                    'discrete_action_codebook_loss_weight must be positive'
                )
            if self.discrete_action_step_codebook_loss_weight <= 0:
                raise ValueError(
                    'discrete_action_step_codebook_loss_weight must be positive'
                )
            if self.discrete_action_codebook_temperature <= 0:
                raise ValueError(
                    'discrete_action_codebook_temperature must be positive'
                )
            if self.discrete_action_class_weight_clip < 1:
                raise ValueError(
                    'discrete_action_class_weight_clip must be at least one'
                )
            if not 1 <= self.discrete_action_codebook_robot_dim <= self.action_dim:
                raise ValueError(
                    'discrete_action_codebook_robot_dim must be within action_dim'
                )
            if self.discrete_action_step_reasoner_layers < 1:
                raise ValueError(
                    'discrete_action_step_reasoner_layers must be positive'
                )
            if self.discrete_action_step_reasoner_num_heads < 1:
                raise ValueError(
                    'discrete_action_step_reasoner_num_heads must be positive'
                )
            if self.discrete_action_codebook_hidden_dim % (
                self.discrete_action_step_reasoner_num_heads
            ):
                raise ValueError(
                    'discrete hidden dimension must be divisible by step reasoner heads'
                )
            if self.discrete_action_step_reasoner_mlp_dim < 1:
                raise ValueError(
                    'discrete_action_step_reasoner_mlp_dim must be positive'
                )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> 'Pi0':
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            demonstration_actions = (
                jax.ShapeDtypeStruct(
                    (
                        [
                            batch_size,
                            self.compositional_demo_slots,
                            self.action_horizon,
                            self.action_dim,
                        ]
                        if self.compositional_demo_routing
                        else [batch_size, self.action_horizon, self.action_dim]
                    ),
                    jnp.float32,
                )
                if (
                    self.retrieved_demo_conditioning
                    or self.persistent_structured_demo_language
                )
                else None
            )
            demonstration_plan = (
                jax.ShapeDtypeStruct(
                    (
                        [
                            batch_size,
                            self.compositional_demo_slots,
                            self.retrieved_demo_plan_steps,
                            self.retrieved_demo_plan_dim,
                        ]
                        if self.compositional_demo_routing
                        else [
                            batch_size,
                            self.retrieved_demo_plan_steps,
                            self.retrieved_demo_plan_dim,
                        ]
                    ),
                    jnp.float32,
                )
                if (
                    self.retrieved_demo_conditioning
                    or self.persistent_structured_demo_language
                )
                else None
            )
            demonstration_mask = (
                jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                if self.retrieved_demo_conditioning
                else None
            )
            demonstration_reliability_target = (
                jax.ShapeDtypeStruct([batch_size], jnp.float32)
                if self.retrieved_demo_conditioning
                else None
            )
            demonstration_slot_mask = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.compositional_demo_slots], jnp.bool_
                )
                if self.compositional_demo_routing
                else None
            )
            demonstration_progress = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.compositional_demo_slots], jnp.float32
                )
                if self.compositional_demo_routing
                else None
            )
            demonstration_router_target = (
                jax.ShapeDtypeStruct([batch_size], jnp.int32)
                if self.compositional_demo_routing
                else None
            )
            demonstration_tokenized_prompt = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.structured_demo_prompt_steps], jnp.int32
                )
                if self.persistent_structured_demo_language
                else None
            )
            demonstration_tokenized_prompt_mask = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.structured_demo_prompt_steps], jnp.bool_
                )
                if self.persistent_structured_demo_language
                else None
            )
            demonstration_semantic_span_mask = (
                jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.structured_demo_semantic_slots,
                        self.structured_demo_prompt_steps,
                    ],
                    jnp.bool_,
                )
                if self.persistent_structured_demo_language
                else None
            )
            demonstration_semantic_valid_mask = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.structured_demo_semantic_slots], jnp.bool_
                )
                if self.persistent_structured_demo_language
                else None
            )
            demonstration_context_mask = (
                jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                if self.persistent_structured_demo_language
                else None
            )
            demonstration_trajectory_mask = (
                jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                if self.persistent_structured_demo_language
                else None
            )
            spatial_language_target_ids = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.spatial_language_steps], jnp.int32
                )
                if self.persistent_structured_demo_language
                else None
            )
            spatial_language_target_mask = (
                jax.ShapeDtypeStruct(
                    [batch_size, self.spatial_language_steps], jnp.bool_
                )
                if self.persistent_structured_demo_language
                else None
            )
            observation_spec = _model.Observation(
                images={
                    'base_0_rgb': image_spec,
                    'left_wrist_0_rgb': image_spec,
                    'right_wrist_0_rgb': image_spec,
                },
                image_masks={
                    'base_0_rgb': image_mask_spec,
                    'left_wrist_0_rgb': image_mask_spec,
                    'right_wrist_0_rgb': image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                future_states=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.action_horizon, self.action_dim],
                        jnp.float32,
                    )
                    if self.state_rollout_reasoner
                    else None
                ),
                future_state_masks=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.action_horizon], jnp.bool_
                    )
                    if self.state_rollout_reasoner
                    else None
                ),
                future_images=(
                    {
                        'base_0_rgb': image_spec,
                        'left_wrist_0_rgb': image_spec,
                        'right_wrist_0_rgb': image_spec,
                    }
                    if (
                        self.latent_future_reasoner
                        or self.object_future_reasoner
                    )
                    else None
                ),
                future_image_masks=(
                    {
                        'base_0_rgb': image_mask_spec,
                        'left_wrist_0_rgb': image_mask_spec,
                        'right_wrist_0_rgb': image_mask_spec,
                    }
                    if (
                        self.latent_future_reasoner
                        or self.object_future_reasoner
                    )
                    else None
                ),
                demonstration_actions=demonstration_actions,
                demonstration_plan=demonstration_plan,
                demonstration_mask=demonstration_mask,
                demonstration_reliability_target=(
                    demonstration_reliability_target
                ),
                demonstration_slot_mask=demonstration_slot_mask,
                demonstration_progress=demonstration_progress,
                demonstration_router_target=demonstration_router_target,
                demonstration_tokenized_prompt=demonstration_tokenized_prompt,
                demonstration_tokenized_prompt_mask=(
                    demonstration_tokenized_prompt_mask
                ),
                demonstration_semantic_span_mask=(
                    demonstration_semantic_span_mask
                ),
                demonstration_semantic_valid_mask=(
                    demonstration_semantic_valid_mask
                ),
                demonstration_context_mask=demonstration_context_mask,
                demonstration_trajectory_mask=demonstration_trajectory_mask,
                spatial_language_target_ids=spatial_language_target_ids,
                spatial_language_target_mask=spatial_language_target_mask,
                task_progress_target=(
                    jax.ShapeDtypeStruct([batch_size], jnp.float32)
                    if (
                        self.task_progress_reasoner
                        or self.language_subgoal_reasoner
                    )
                    else None
                ),
                clause_span_mask=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.persistent_clause_plan_slots,
                            self.max_token_len,
                        ],
                        jnp.bool_,
                    )
                    if self.persistent_clause_plan_v1
                    else None
                ),
                clause_valid_mask=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.persistent_clause_plan_slots],
                        jnp.bool_,
                    )
                    if self.persistent_clause_plan_v1
                    else None
                ),
                racg_role_span_mask=(
                    jax.ShapeDtypeStruct(
                        [batch_size, 6, self.max_token_len], jnp.bool_
                    )
                    if self.role_affordance_causal_graph
                    else None
                ),
                racg_role_valid_mask=(
                    jax.ShapeDtypeStruct([batch_size, 6], jnp.bool_)
                    if self.role_affordance_causal_graph
                    else None
                ),
                racg_relation_kind=(
                    jax.ShapeDtypeStruct([batch_size], jnp.int32)
                    if self.role_affordance_causal_graph
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex('.*llm.*')
        action_expert_params_filter = nnx_utils.PathRegex('.*llm.*_1.*')
        if 'lora' in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if 'lora' not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif 'lora' in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex('.*lora.*')),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_state_adarms_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze every parameter except the state-conditioning adapter."""
        if not self.state_adarms:
            raise ValueError('state_adarms must be enabled for adapter training')
        return nnx.Not(nnx_utils.PathRegex('.*state_adarms.*'))

    def get_context_adarms_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and multimodal action-layer conditioning."""
        if not self.context_adarms:
            raise ValueError('context_adarms must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError('context AdaRMS requires the discrete action reasoner')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|context_adarms|action_prior|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_best_anchor_context_adarms_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the audited 51.24 parent and train only four AdaRMS leaves."""
        if not self.context_adarms:
            raise ValueError('context_adarms must be enabled')
        if not self.action_prior_contextual:
            raise ValueError(
                'best-anchor context AdaRMS requires contextual action priors'
            )
        return nnx.Not(nnx_utils.PathRegex('.*context_adarms.*'))

    def get_state_film_lora_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train state fusion, action-expert LoRA and small action heads."""
        if not (self.state_adarms and self.state_action_film and self.action_prior):
            raise ValueError(
                'state_adarms, state_action_film and action_prior must be enabled'
            )
        if 'lora' not in self.action_expert_variant:
            raise ValueError('the action expert must use a LoRA variant')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            # The codebook is an offline k-means artifact, not a learned model
            # parameter. Keeping it in the parameter tree makes checkpoints
            # self-contained while this exclusion prevents codebook drift.
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_persistent_sdla_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train inherited PSM/action paths plus both exact SDLA namespaces."""
        if not (
            self.persistent_subgoal_memory
            and self.persistent_structured_demo_language
        ):
            raise ValueError('persistent PSM-SDLA must be enabled')
        trainable = nnx_utils.PathRegex(
            '.*(persistent_memory|spatial_language_aux|state_adarms|state_film|'
            'action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
        )
        return nnx.Not(trainable)

    def get_persistent_sdla_v3_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train PSM, SDLA-v2, direct v3 bridge, and inherited action paths."""
        if not self.persistent_conditional_memory_policy_bridge:
            raise ValueError('persistent conditional memory bridge must be enabled')
        return self.get_persistent_sdla_freeze_filter()

    def get_persistent_geometry_v1_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train geometry, inherited PSM/SDLA-v3 and PaliGemma LoRA."""
        if not self.persistent_geometry_aux_v1:
            raise ValueError('geometry-v1 must be enabled')
        return self.get_persistent_sdla_v3_freeze_filter()

    def get_persistent_geometry_hmca_v4_freeze_filter(self) -> nnx.filterlib.Filter:
        if not self.persistent_hmca_v4:
            raise ValueError('HMCA-v4 must be enabled')
        return nnx.All(
            self.get_persistent_geometry_v1_freeze_filter(),
            nnx.Not(
                nnx_utils.PathRegex(
                    '.*hierarchical_memory_conditional_adapters.*'
                )
            ),
        )

    def get_persistent_clause_plan_v1_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train ClausePlan plus the inherited Joint51/HMCA action paths."""
        if not self.persistent_clause_plan_v1:
            raise ValueError('ClausePlan-v1 must be enabled')
        return nnx.All(
            self.get_persistent_geometry_hmca_v4_freeze_filter(),
            nnx.Not(nnx_utils.PathRegex('.*clause_plan_adapter.*')),
        )

    def get_clause_plan_object_contact_v2_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Tune Joint51, ClausePlan, competitive objects, and contact phase."""
        if not (
            self.persistent_clause_plan_v1
            and self.object_affordance_graph_reasoner
            and self.contact_phase_reasoner
        ):
            raise ValueError(
                'ClausePlan, object affordance, and contact phase must be enabled'
            )
        return nnx.All(
            self.get_persistent_clause_plan_v1_freeze_filter(),
            nnx.Not(
                nnx_utils.PathRegex('.*(object_affordance|contact_phase).*')
            ),
        )

    def get_clause_plan_verified_contact_v3_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Tune v2 plus object/contact/PSM-frontier predictive fusion."""
        if not self.contact_affordance_predictive_fusion:
            raise ValueError('verified-contact fusion must be enabled')
        return nnx.All(
            self.get_clause_plan_object_contact_v2_freeze_filter(),
            nnx.Not(
                nnx_utils.PathRegex(
                    '.*(contact_affordance|object_future).*'
                )
            ),
        )

    def get_geometry_external_residual_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the exact ClausePlan-v3 parent and train only three new leaves."""
        if not self.persistent_geometry_external_residual_v1:
            raise ValueError('external geometry residual must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/geometry_external_residual_v1.*'
            )
        )

    def get_temporal_role_memory_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze exact DualGeometry parent and train only five new leaves."""
        if not self.persistent_temporal_role_memory_v1:
            raise ValueError('temporal role memory must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/action_conditioned_temporal_object_residual_v1.*'
            )
        )

    def get_cross_view_role_consensus_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze exact TemporalRoleMemory parent and train only five new leaves."""
        if not self.persistent_cross_view_role_consensus_v1:
            raise ValueError('cross-view role consensus must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/cross_view_role_consensus_v1.*'
            )
        )

    def get_temporal_cross_view_coadapt_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the 811-leaf base and jointly adapt both adjacent branches."""
        if not (
            self.persistent_temporal_role_memory_v1
            and self.persistent_cross_view_role_consensus_v1
        ):
            raise ValueError('temporal/cross-view co-adaptation requires both branches')
        trainable = nnx_utils.PathRegex(
            '.*persistent_memory/(action_conditioned_temporal_object_residual_v1|'
            'cross_view_role_consensus_v1).*'
        )
        return nnx.Not(trainable)

    def get_contact_risk_calibrated_role_residual_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the exact CoAdapt parent and train seven contact-risk leaves."""
        if not self.persistent_contact_risk_calibrated_role_residual_v1:
            raise ValueError('contact-risk role residual must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/contact_risk_calibrated_role_residual_v1.*'
            )
        )

    def get_relational_role_composer_residual_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the exact ContactRisk parent and train eight composer leaves."""
        if not self.persistent_relational_role_composer_residual_v1:
            raise ValueError('relational role composer must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/relational_role_composer_residual_v1.*'
            )
        )

    def get_joint_role_geometry_residual_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze ClausePlan-v3 and jointly train the five residual branches."""
        if not (
            self.persistent_geometry_external_residual_v1
            and self.persistent_temporal_role_memory_v1
            and self.persistent_cross_view_role_consensus_v1
            and self.persistent_contact_risk_calibrated_role_residual_v1
            and self.persistent_relational_role_composer_residual_v1
        ):
            raise ValueError('joint role/geometry training requires all branches')
        trainable = nnx_utils.PathRegex(
            '.*persistent_memory/(geometry_external_residual_v1|'
            'action_conditioned_temporal_object_residual_v1|'
            'cross_view_role_consensus_v1|'
            'contact_risk_calibrated_role_residual_v1|'
            'relational_role_composer_residual_v1).*'
        )
        return nnx.Not(trainable)

    def get_clause_role_binding_verifier_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze JointRoleGeometry and train exactly eight verifier leaves."""
        if not self.persistent_clause_role_binding_verifier_v1:
            raise ValueError('clause-role binding verifier must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/clause_role_binding_verifier_v1.*'
            )
        )

    def get_semantic_frontier_completion_verifier_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze ClauseRole and train exactly four semantic-frontier leaves."""
        if not self.persistent_semantic_frontier_completion_verifier_v1:
            raise ValueError('semantic frontier completion verifier must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/semantic_frontier_completion_verifier_v1.*'
            )
        )

    def get_hierarchical_clause_event_alignment_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the inherited parent and train exactly the five HCEA leaves."""
        if not self.persistent_hierarchical_clause_event_alignment_v1:
            raise ValueError('hierarchical clause-event alignment must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/hierarchical_clause_event_alignment_v1.*'
            )
        )

    def get_hcea_causal_recovery_action_experts_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the HCEA parent and train exactly ten recovery leaves."""
        if not self.persistent_hcea_causal_recovery_action_experts_v1:
            raise ValueError('HCEA causal recovery experts must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/hcea_causal_recovery_action_experts_v1.*'
            )
        )

    def get_hcea_recovery_action_cadapt_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Co-adapt Recovery and action paths while freezing prefix producers."""
        required = (
            self.persistent_hierarchical_clause_event_alignment_v1
            and self.persistent_hcea_causal_recovery_action_experts_v1
            and self.state_adarms
            and self.state_action_film
            and self.action_prior
            and self.action_prior_contextual
            and self.action_prior_state_conditioning
            and 'lora' in self.action_expert_variant
            and self.persistent_subgoal_memory
            and self.contact_phase_reasoner
            and self.contact_affordance_predictive_fusion
            and self.latent_future_reasoner
            and self.state_rollout_reasoner
            and self.action_moe_reasoner
            and self.task_progress_reasoner
            and self.predictive_world_model_fusion
        )
        if not required:
            raise ValueError(
                'Recovery action co-adaptation requires the complete recovery graph'
            )
        downstream = nnx_utils.PathRegex(
            '.*(?:latent_future|state_rollout|action_moe|task_progress|'
            'predictive_world_model|contact_phase|contact_affordance|'
            'hierarchical_memory_conditional_adapters|'
            'hierarchical_clause_event_alignment_v1|'
            'hcea_causal_recovery_action_experts_v1|'
            'state_adarms|state_film|action_prior|action_in_proj|'
            'action_out_proj|time_mlp).*'
        )
        action_expert_lora = nnx_utils.PathRegex(
            '.*PaliGemma/llm/layers/(?:attn/(?:attn_vec_einsum_1|'
            'kv_einsum_1|q_einsum_1)/lora_[ab]|'
            'mlp_1/(?:gating_einsum_lora_[ab]|linear_lora_[ab])).*'
        )
        trainable = nnx.All(
            nnx.Any(downstream, action_expert_lora),
            nnx.Not(nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')),
        )
        return nnx.Not(trainable)

    def get_hcea_causal_role_identity_transport_expert_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze HCEA and train exactly the eleven role-transport leaves."""
        if not self.persistent_hcea_causal_role_identity_transport_expert_v1:
            raise ValueError('HCEA causal role identity transport must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/'
                'hcea_causal_role_identity_transport_expert_v1.*'
            )
        )

    def get_joint_clause_role_semantic_frontier_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze JointRole and jointly train the exact twelve successor leaves."""
        if not (
            self.persistent_clause_role_binding_verifier_v1
            and self.persistent_semantic_frontier_completion_verifier_v1
        ):
            raise ValueError('joint clause-role and semantic-frontier modules must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*persistent_memory/(clause_role_binding_verifier_v1|'
                'semantic_frontier_completion_verifier_v1).*'
            )
        )

    def get_direct_psm_cumulative848_v1_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze exact PSM-402 and jointly optimize all 446 successors."""
        required = (
            self.persistent_subgoal_memory
            and self.persistent_clause_plan_v1
            and self.object_affordance_graph_reasoner
            and self.contact_phase_reasoner
            and self.object_future_reasoner
            and self.contact_affordance_predictive_fusion
            and self.contact_affordance_clause_plan_verification
            and self.persistent_geometry_external_residual_v1
            and self.persistent_temporal_role_memory_v1
            and self.persistent_cross_view_role_consensus_v1
            and self.persistent_contact_risk_calibrated_role_residual_v1
            and self.persistent_relational_role_composer_residual_v1
            and self.persistent_clause_role_binding_verifier_v1
            and self.persistent_semantic_frontier_completion_verifier_v1
        )
        if not required:
            raise ValueError('direct cumulative-848 training requires the complete graph')
        # The top-level reasoners did not exist in PSM-402.  The explicit
        # persistent-memory allowlist excludes every inherited PSM namespace.
        trainable = nnx_utils.PathRegex(
            '.*(?:contact_phase.*|object_future.*|contact_affordance.*|'
            'object_affordance.*|spatial_language_aux.*|'
            'hierarchical_memory_conditional_adapters.*|'
            'persistent_memory/(?:structured_demo|'
            'conditional_memory_policy_bridge|geometry_aux_v3|'
            'clause_plan_adapter|role_identity_confidence_gate|'
            'geometry_external_residual_v1|'
            'action_conditioned_temporal_object_residual_v1|'
            'cross_view_role_consensus_v1|'
            'contact_risk_calibrated_role_residual_v1|'
            'relational_role_composer_residual_v1|'
            'clause_role_binding_verifier_v1|'
            'semantic_frontier_completion_verifier_v1).*)'
        )
        return nnx.Not(trainable)

    def get_persistent_memory_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train recurrent memory, compound action heads, and inherited LoRA."""
        if not self.persistent_subgoal_memory:
            raise ValueError('persistent_subgoal_memory must be enabled')
        trainable = nnx_utils.PathRegex(
            '.*(persistent_memory|state_adarms|state_film|action_prior|lora|'
            'action_in_proj|action_out_proj|time_mlp).*'
        )
        return nnx.Not(trainable)

    def get_joint_psm_hetm_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train HETM/action adapters while preserving all Direct PSM paths."""
        if not (
            self.persistent_subgoal_memory
            and self.hierarchical_event_transition_memory
        ):
            raise ValueError('joint PSM+HETM training requires both memories')
        trainable = nnx_utils.PathRegex(
            '.*(hetm|state_adarms|state_film|action_prior|lora|'
            'action_in_proj|action_out_proj|time_mlp).*'
        )
        return nnx.Not(trainable)

    def get_joint_psm_hetm_racg_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train RACG/HETM adapters while preserving the audited Direct PSM."""
        if not (
            self.persistent_subgoal_memory
            and self.hierarchical_event_transition_memory
            and self.role_affordance_causal_graph
        ):
            raise ValueError('joint PSM+HETM+RACG training requires all branches')
        trainable = nnx_utils.PathRegex(
            '.*(racg|hetm|state_adarms|state_film|action_prior|lora|'
            'action_in_proj|action_out_proj|time_mlp).*'
        )
        return nnx.Not(trainable)

    def get_joint_integrated60_ppwm_freeze_filter(self) -> nnx.filterlib.Filter:
        """Co-adapt the full Integrated60 stack with predictive dynamics."""
        if not (
            self.persistent_subgoal_memory
            and self.hierarchical_event_transition_memory
            and self.role_affordance_causal_graph
            and self.racg_external_geometry_prior
            and self.racg_external_geometry_hmca
            and self.racg_graph_hmca
            and self.object_future_reasoner
            and self.latent_future_reasoner
            and self.state_rollout_reasoner
            and self.action_moe_reasoner
            and self.task_progress_reasoner
            and self.predictive_world_model_fusion
        ):
            raise ValueError('Integrated60+PPWM requires every joint capability')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(racg|hetm|object_future|latent_future|state_rollout|'
                'action_moe|task_progress|predictive_world_model|contact_phase|'
                'contact_affordance|state_adarms|'
                'state_film|action_prior|lora|action_in_proj|action_out_proj|'
                'time_mlp).*'
            ),
            nnx.Not(nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')),
        )
        return nnx.Not(trainable)

    def get_joint_integrated60_ppwm_memory_adarms_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Co-adapt PPWM/GroundedRole with layerwise causal-memory conditioning."""
        if not self.persistent_memory_adarms:
            raise ValueError('persistent-memory AdaRMS must be enabled')
        # Reuse the complete capability validation of the parent contract.
        self.get_joint_integrated60_ppwm_freeze_filter()
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(racg|hetm|object_future|latent_future|state_rollout|'
                'action_moe|task_progress|predictive_world_model|contact_phase|'
                'contact_affordance|state_adarms|state_film|action_prior|lora|'
                'action_in_proj|action_out_proj|time_mlp|'
                'persistent_memory_adarms).*'
            ),
            nnx.Not(nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')),
        )
        return nnx.Not(trainable)

    def get_joint_integrated60_ppwm_phase_contact_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Co-adapt Memory-AdaRMS with tokenwise phase/contact modulation."""
        if not self.persistent_memory_adarms:
            raise ValueError('persistent-memory AdaRMS must be enabled')
        if not self.phase_contact_action_film:
            raise ValueError('phase-contact action FiLM must be enabled')
        self.get_joint_integrated60_ppwm_memory_adarms_freeze_filter()
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(racg|hetm|object_future|latent_future|state_rollout|'
                'action_moe|task_progress|predictive_world_model|contact_phase|'
                'contact_affordance|state_adarms|state_film|action_prior|lora|'
                'action_in_proj|action_out_proj|time_mlp|'
                'persistent_memory_adarms|phase_contact_film|'
                'persistent_action_prior).*'
            ),
            nnx.Not(nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')),
        )
        return nnx.Not(trainable)

    def get_joint_integrated60_ppwm_layerwise_memory_attention_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Co-adapt PhaseContact with token-preserving layerwise PSM reads."""
        if not self.layerwise_persistent_memory_attention:
            raise ValueError('layerwise persistent-memory attention must be enabled')
        self.get_joint_integrated60_ppwm_phase_contact_freeze_filter()
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(racg|hetm|object_future|latent_future|state_rollout|'
                'action_moe|task_progress|predictive_world_model|contact_phase|'
                'contact_affordance|state_adarms|state_film|action_prior|lora|'
                'action_in_proj|action_out_proj|time_mlp|persistent_memory_adarms|'
                'phase_contact_film|layerwise_persistent_memory_attention).*'
            ),
            nnx.Not(nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')),
        )
        return nnx.Not(trainable)

    def get_attention_adarms_full_freeze_filter(self) -> nnx.filterlib.Filter:
        """Fully tune attention/AdaRMS while retaining the existing adapters."""
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'progressive attention full tuning requires the '
                'discrete action reasoner'
            )
        trainable = nnx.All(
            nnx.Any(
                nnx_utils.PathRegex(
                    '.*(state_adarms|state_film|action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
                ),
                nnx_utils.PathRegex('.*PaliGemma/llm/layers/attn/.*/w'),
                nnx_utils.PathRegex(
                    '.*PaliGemma/llm/layers/pre_(attention|ffw)_norm_1.*'
                ),
                nnx_utils.PathRegex('.*PaliGemma/llm/final_norm_1.*'),
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_spatial_attention_freeze_filter(self) -> nnx.filterlib.Filter:
        """Tune spatial-relation queries plus Stage-7 attention/reasoners.

        The pretrained SigLIP tower stays frozen. A real full-vision backward
        pass exceeded the single-L20 memory pool even at microbatch eight, and
        the explicit spatial bottleneck can adapt its frozen patch features
        without risking broad vision-representation drift.
        """
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'vision-attention tuning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx.Any(
                nnx_utils.PathRegex('.*spatial_relation.*'),
                nnx_utils.PathRegex(
                    '.*(state_adarms|state_film|action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
                ),
                nnx_utils.PathRegex('.*PaliGemma/llm/layers/attn/.*/w'),
                nnx_utils.PathRegex(
                    '.*PaliGemma/llm/layers/pre_(attention|ffw)_norm_1.*'
                ),
                nnx_utils.PathRegex('.*PaliGemma/llm/final_norm_1.*'),
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_object_affordance_graph_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the competitive object-slot graph."""
        if not self.object_affordance_graph_reasoner:
            raise ValueError('object affordance graph reasoning must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'object affordance graph reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|object_affordance|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_action_mlp_full_freeze_filter(self) -> nnx.filterlib.Filter:
        """Tune the action-expert MLP plus Stage-6 reasoners.

        This independent Stage-6 branch freezes vision and base attention while
        the 300M action expert's feed-forward blocks are refined. Keeping both
        large prefix paths frozen isolates action-side capacity and fits the
        single-device optimizer/activation memory budget.
        """
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'action-MLP tuning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx.Any(
                nnx_utils.PathRegex(
                    '.*(state_adarms|state_film|action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
                ),
                nnx_utils.PathRegex('.*contact_phase.*'),
                nnx_utils.PathRegex(
                    '.*PaliGemma/llm/layers/mlp_1/(gating_einsum|linear)'
                ),
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_reasoning_pathway_router_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train the Stage-6 reasoners and the function-preserving path router."""
        if not self.reasoning_pathway_router:
            raise ValueError('reasoning_pathway_router must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'reasoning pathway routing requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_reasoning_pathway_interaction_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the zero-residual path interaction net."""
        if not self.reasoning_pathway_interaction:
            raise ValueError('reasoning_pathway_interaction must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'reasoning pathway interaction requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_structured_rationale_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners, VLM LoRA, and the rationale expert."""
        if not self.structured_rationale_reasoner:
            raise ValueError('structured_rationale_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'structured rationale reasoning requires the discrete action reasoner'
            )
        if 'lora' not in self.paligemma_variant:
            raise ValueError('the VLM must use a LoRA variant')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|structured_rationale|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_action_chunk_verifier_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the contrastive flow verifier."""
        if not self.action_chunk_verifier:
            raise ValueError('action_chunk_verifier must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'action chunk verification requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|action_chunk_verifier|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_latent_future_reasoner_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the latent visual dynamics expert."""
        if not self.latent_future_reasoner:
            raise ValueError('latent_future_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'latent future reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|latent_future|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_state_rollout_reasoner_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the dense proprioceptive dynamics expert."""
        if not self.state_rollout_reasoner:
            raise ValueError('state_rollout_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'state rollout reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|state_rollout|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_action_moe_reasoner_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the sparse task-conditioned experts."""
        if not self.action_moe_reasoner:
            raise ValueError('action_moe_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'action MoE reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|action_moe|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_task_progress_reasoner_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the latent task-progress state."""
        if not self.task_progress_reasoner:
            raise ValueError('task_progress_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'task progress reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|task_progress|lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_language_subgoal_reasoner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the ordered latent-subgoal automaton."""
        if not self.language_subgoal_reasoner:
            raise ValueError('language_subgoal_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'language subgoal reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|language_subgoal|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_best_anchor_language_subgoal_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train only the ordered subgoal module on a complete best anchor."""
        if not self.language_subgoal_reasoner:
            raise ValueError('language_subgoal_reasoner must be enabled')
        if not self.action_prior_contextual:
            raise ValueError(
                'best-anchor subgoal reasoning requires contextual prior states'
            )
        return nnx.Not(nnx_utils.PathRegex('.*language_subgoal.*'))

    def get_best_anchor_joint_cadapt_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Jointly tune only the three cumulative best-anchor modules."""
        if not (
            self.context_adarms
            and self.velocity_refiner
            and self.language_subgoal_reasoner
        ):
            raise ValueError(
                'joint co-adaptation requires ContextAdaRMS, VelocityRefiner, '
                'and LanguageSubgoal'
            )
        if not self.action_prior_contextual:
            raise ValueError(
                'best-anchor joint co-adaptation requires contextual priors'
            )
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(context_adarms|velocity_refiner|language_subgoal).*'
            )
        )

    def get_best_anchor_specialist_router_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Jointly tune transplanted specialists and their sample-wise router."""
        if not self.specialist_module_router:
            raise ValueError('specialist_module_router must be enabled')
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(context_adarms|velocity_refiner|language_subgoal|'
                'specialist_module_router).*'
            )
        )

    def get_best_anchor_dual_contact_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the cumulative best anchor and train only dual/contact leaves."""
        if not (self.dual_action_reasoner and self.contact_phase_reasoner):
            raise ValueError(
                'best-anchor dual-contact training requires both reasoners'
            )
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(action_prior_implicit|action_prior_explicit|'
                'action_prior_guidance|contact_phase).*'
            )
        )

    def get_best_anchor_grounded_subgoal_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train only visual object slots and their ordered-subgoal binding."""
        if not (
            self.object_affordance_graph_reasoner
            and self.language_subgoal_reasoner
            and self.object_subgoal_binding
        ):
            raise ValueError(
                'grounded subgoal training requires object affordance, '
                'language subgoal, and object-subgoal binding reasoners'
            )
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(object_affordance|object_subgoal_binding).*'
            )
        )

    def get_best_anchor_task_routed_action_moe_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the cumulative anchor and train only routed action experts."""
        if not self.action_moe_reasoner:
            raise ValueError('task-routed action MoE must be enabled')
        if not (self.dual_action_reasoner and self.action_prior_contextual):
            raise ValueError(
                'best-anchor action MoE requires dual contextual action priors'
            )
        return nnx.Not(nnx_utils.PathRegex('.*action_moe.*'))

    def get_robodojo_task_moe_transfer_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Adapt the audited TaskMoE stack to a new dual-arm embodiment.

        Unlike a within-benchmark successor, cross-embodiment transfer must
        update the 14-D state/action interface and the inherited capability
        modules as well as the sparse experts.  The frozen set still contains
        the dense PaliGemma and action-expert base weights; their LoRA leaves,
        compact conditioning modules, and input/output projections train.
        """
        if self.active_action_dim != 14:
            raise ValueError('RoboDojo transfer requires active_action_dim=14')
        if self.action_horizon != 10:
            raise ValueError('RoboDojo TaskMoE transfer preserves the 10-step graph')
        if not (
            self.action_moe_reasoner
            and self.dual_action_reasoner
            and self.language_subgoal_reasoner
            and self.object_affordance_graph_reasoner
            and self.object_subgoal_binding
            and self.contact_phase_reasoner
        ):
            raise ValueError('RoboDojo transfer requires the complete TaskMoE stack')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(lora|action_in_proj|action_out_proj|state_proj|time_mlp|'
                'state_adarms|state_film|context_adarms|action_prior|'
                'implicit_action_reasoner|explicit_action_reasoner|'
                'retrieved_demo|velocity_refiner|language_subgoal|'
                'object_affordance|object_subgoal_binding|contact_phase|'
                'action_moe).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_robodojo_closed_loop_memory_transfer_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Adapt the policy interface while preserving the trained memory core."""
        if self.active_action_dim != 14 or self.persistent_memory_action_dim != 7:
            raise ValueError(
                'RoboDojo recurrent transfer requires a 14D policy and 7D '
                'shape-preserving memory bridge'
            )
        if not self.persistent_subgoal_memory:
            raise ValueError('RoboDojo recurrent transfer requires persistent memory')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(lora|action_in_proj|action_out_proj|state_proj|time_mlp|'
                'state_adarms|state_film|context_adarms|action_prior|'
                'implicit_action_reasoner|explicit_action_reasoner|'
                'retrieved_demo|velocity_refiner|language_subgoal|'
                'object_affordance|object_subgoal_binding|contact_phase|'
                'action_moe).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        # The recurrent parameters retain their audited VLA-Arena geometry and
        # are frozen during cross-embodiment adaptation.  They remain active in
        # both training and serving; only the surrounding 14D interface moves.
        return nnx.Not(trainable)

    def get_robodojo_public_pi05_language_moe_memory_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train the audited X-Policy stack on the public RoboDojo Pi-05.

        The released dense Pi-05 remains frozen.  Only its new LoRA leaves and
        the compact state/context, language-subgoal, routed-action, and causal
        memory paths are optimized.  Every policy-facing adapter starts at an
        exact-zero boundary, so the released checkpoint is the step-zero
        policy rather than merely a similarly configured initialization.
        """
        if self.active_action_dim != 14 or self.action_horizon != 50:
            raise ValueError(
                'public RoboDojo Pi-05 adaptation requires 14D actions and '
                'the released 50-step action horizon'
            )
        if not (
            self.pi05
            and self.state_adarms
            and self.state_action_film
            and self.action_prior_contextual
            and self.dual_action_reasoner
            and self.language_subgoal_reasoner
            and self.action_moe_reasoner
            and self.action_moe_task_consistent_routing
            and self.persistent_subgoal_memory
        ):
            raise ValueError(
                'public Pi-05 successor requires the complete language, '
                'task-routed action-MoE, and closed-loop memory graph'
            )
        trainable = nnx_utils.PathRegex(
            '.*(lora|state_adarms|state_film|action_prior|language_subgoal|'
            'action_moe|persistent_memory).*'
        )
        return nnx.Not(trainable)

    def get_best_anchor_closed_loop_memory_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Freeze the selected best anchor and train only recurrent memory."""
        if not self.persistent_subgoal_memory:
            raise ValueError('persistent_subgoal_memory must be enabled')
        if not self.supervised_role_identity_contrastive_loss:
            raise ValueError(
                'closed-loop memory requires supervised role identity loss'
            )
        return nnx.Not(nnx_utils.PathRegex('.*persistent_memory.*'))

    def get_best_anchor_closed_loop_specialist_coadapt_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Co-adapt the recurrent policy stack while freezing its base anchor."""
        if not self.persistent_subgoal_memory:
            raise ValueError('closed-loop specialist co-adaptation requires memory')
        if not self.action_moe_reasoner:
            raise ValueError('closed-loop specialist co-adaptation requires action MoE')
        if not self.specialist_module_router:
            raise ValueError('closed-loop specialist co-adaptation requires routing')
        if not self.persistent_memory_adarms:
            raise ValueError(
                'closed-loop specialist co-adaptation requires direct memory AdaRMS'
            )
        if not (
            self.context_adarms
            and self.velocity_refiner
            and self.language_subgoal_reasoner
        ):
            raise ValueError(
                'closed-loop specialist co-adaptation requires ContextAdaRMS, '
                'VelocityRefiner, and LanguageSubgoal'
            )
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(context_adarms|velocity_refiner|language_subgoal|'
                'action_moe|persistent_memory|specialist_module_router).*'
            )
        )

    def get_best_anchor_closed_loop_grounded_contact_cadapt_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Co-adapt grounded contact control on the exact closed-loop graph."""
        if not (
            self.persistent_subgoal_memory
            and self.action_moe_reasoner
            and self.specialist_module_router
            and self.dual_action_reasoner
            and self.contact_phase_reasoner
            and self.object_affordance_graph_reasoner
            and self.object_subgoal_binding
        ):
            raise ValueError(
                'grounded-contact co-adaptation requires memory, action MoE, '
                'specialist routing, dual/contact reasoning, object slots, '
                'and object-subgoal binding'
            )
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(context_adarms|velocity_refiner|language_subgoal|'
                'action_prior_implicit|action_prior_explicit|'
                'action_prior_guidance|contact_phase|object_affordance|'
                'object_subgoal_binding|action_moe|persistent_memory|'
                'specialist_module_router).*'
            )
        )

    def get_best_anchor_causal_frontier_transition_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train only the function-preserving causal stay/advance gate."""
        if not (
            self.persistent_subgoal_memory
            and self.persistent_causal_frontier_transition_gate_v1
        ):
            raise ValueError(
                'causal frontier transition training requires recurrent memory'
            )
        return nnx.Not(
            nnx_utils.PathRegex('.*causal_frontier_transition.*')
        )

    def get_best_anchor_closed_loop_predictive_dynamics_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train future/state/progress experts and their causal fusion only."""
        if not (
            self.persistent_subgoal_memory
            and self.persistent_causal_frontier_transition_gate_v1
            and self.latent_future_reasoner
            and self.state_rollout_reasoner
            and self.task_progress_reasoner
            and self.predictive_world_model_fusion
            and self.predictive_world_model_include_action_moe
        ):
            raise ValueError(
                'closed-loop predictive dynamics requires causal memory, '
                'future/state/progress experts, fusion, and action MoE'
            )
        return nnx.Not(
            nnx_utils.PathRegex(
                '.*(latent_future|state_rollout|task_progress|'
                'predictive_world_model).*'
            )
        )

    def get_kinematic_action_reasoner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the factorized kinematic expert."""
        if not self.kinematic_action_reasoner:
            raise ValueError('kinematic_action_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'kinematic action reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|kinematic_action|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_spectral_action_reasoner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the full-band action-spectrum expert."""
        if not self.spectral_action_reasoner:
            raise ValueError('spectral_action_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'spectral action reasoning requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|spectral_action|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_velocity_refiner_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and the residual flow-velocity refiner."""
        if not self.velocity_refiner:
            raise ValueError('velocity_refiner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'velocity refinement requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|velocity_refiner|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_best_anchor_velocity_refiner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train only the new refiner on top of a fully trained best anchor."""
        if not self.velocity_refiner:
            raise ValueError('velocity_refiner must be enabled')
        if not self.action_prior_contextual:
            raise ValueError(
                'best-anchor velocity refinement requires contextual prior states'
            )
        return nnx.Not(nnx_utils.PathRegex('.*velocity_refiner.*'))

    def get_action_visual_refiner_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners and dense-prefix action visual refinement."""
        if not self.action_visual_refiner:
            raise ValueError('action_visual_refiner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'action visual refinement requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|action_visual_refiner|'
                'lora|action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_masked_spatial_reasoner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners, VLM LoRA, and masked spatial expert."""
        if not self.masked_spatial_reasoner:
            raise ValueError('masked_spatial_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'masked spatial reasoning requires the discrete action reasoner'
            )
        if 'lora' not in self.paligemma_variant:
            raise ValueError('masked spatial reasoning requires VLM LoRA')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|masked_spatial|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_object_future_reasoner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6 reasoners, VLM LoRA, and object-future expert."""
        if not self.object_future_reasoner:
            raise ValueError('object_future_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'object future reasoning requires the discrete action reasoner'
            )
        if 'lora' not in self.paligemma_variant:
            raise ValueError('object future reasoning requires VLM LoRA')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|object_future|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_predicate_binding_reasoner_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6, VLM LoRA, and compositional predicate binding."""
        if not self.predicate_binding_reasoner:
            raise ValueError('predicate_binding_reasoner must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'predicate binding requires the discrete action reasoner'
            )
        if 'lora' not in self.paligemma_variant:
            raise ValueError('predicate binding reasoning requires VLM LoRA')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|predicate_binding|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_multimodal_prefix_moe_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6, VLM LoRA, and task-routed prefix FiLM experts."""
        if not self.multimodal_prefix_moe:
            raise ValueError('multimodal_prefix_moe must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'multimodal prefix MoE requires the discrete action reasoner'
            )
        if 'lora' not in self.paligemma_variant:
            raise ValueError('multimodal prefix MoE requires VLM LoRA')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|prefix_moe|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_layerwise_kv_moe_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Train Stage-6, VLM LoRA, and task-routed layerwise KV experts."""
        if not self.layerwise_kv_moe:
            raise ValueError('layerwise_kv_moe must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'layerwise KV MoE requires the discrete action reasoner'
            )
        if 'lora' not in self.paligemma_variant:
            raise ValueError('layerwise KV MoE requires VLM LoRA')
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|kv_moe|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_predictive_world_model_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train Stage-6 plus all jointly gated predictive reasoners."""
        if not self.predictive_world_model_fusion:
            raise ValueError('predictive_world_model_fusion must be enabled')
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'predictive world-model fusion requires the discrete action reasoner'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|latent_future|'
                'state_rollout|action_moe|task_progress|predictive_world_model|lora|'
                'action_in_proj|action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_persistent_predictive_world_model_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Jointly tune persistent memory and memory-routed world dynamics."""
        if not (
            self.persistent_subgoal_memory
            and self.object_future_reasoner
            and self.predictive_world_model_fusion
        ):
            raise ValueError(
                'persistent predictive fusion requires persistent memory, '
                'object-future reasoning, and the predictive world model'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(persistent_memory|state_adarms|state_film|action_prior|'
                'object_future|latent_future|state_rollout|action_moe|task_progress|'
                'predictive_world_model|lora|action_in_proj|action_out_proj|'
                'time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_persistent_contact_affordance_freeze_filter(
        self,
    ) -> nnx.filterlib.Filter:
        """Tune PPWM plus the object/contact uncertainty coupling."""
        if not (
            self.object_future_reasoner
            and self.contact_affordance_predictive_fusion
        ):
            raise ValueError(
                'object-future and contact-affordance predictive fusion must '
                'both be enabled'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(persistent_memory|state_adarms|state_film|action_prior|'
                'object_future|latent_future|state_rollout|action_moe|task_progress|'
                'predictive_world_model|object_affordance|contact_phase|'
                'contact_affordance|lora|action_in_proj|action_out_proj|'
                'time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)

    def get_evidence_combination_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train compact evidence, hierarchy, and residual-flow reasoners."""
        if not (
            self.predictive_world_model_fusion
            and self.spatial_relation_reasoner
            and self.contact_phase_reasoner
            and self.reasoning_pathway_interaction
            and self.evidence_combination_router
        ):
            raise ValueError(
                'evidence combination requires predictive, spatial, contact, '
                'pathway-interaction reasoners, and their component router'
            )
        if self.discrete_action_codebook_path is None:
            raise ValueError(
                'evidence combination requires the discrete action reasoner'
            )
        if self.evidence_combination_hierarchical_components and not (
            self.object_affordance_graph_reasoner
            and self.language_subgoal_reasoner
        ):
            raise ValueError(
                'hierarchical evidence combination requires object-affordance '
                'and language-subgoal reasoners'
            )
        if self.evidence_combination_action_verifier_component and not (
            self.action_chunk_verifier
        ):
            raise ValueError(
                'evidence combination verifier route requires action chunk verification'
            )
        if self.evidence_combination_object_subgoal_binding_component and not (
            self.object_subgoal_binding
        ):
            raise ValueError(
                'evidence combination object-subgoal route requires binding'
            )
        trainable = nnx.All(
            nnx_utils.PathRegex(
                '.*(state_adarms|state_film|action_prior|latent_future|'
                'state_rollout|action_moe|task_progress|predictive_world_model|'
                'spatial_relation|contact_phase|object_affordance|language_subgoal|'
                'object_subgoal_binding|'
                'velocity_refiner|action_visual_refiner|action_chunk_verifier|'
                'evidence_combination|lora|'
                'action_in_proj|'
                'action_out_proj|time_mlp).*'
            ),
            nnx.Not(
                nnx_utils.PathRegex('.*action_prior_discrete_codebook.*')
            ),
        )
        return nnx.Not(trainable)
