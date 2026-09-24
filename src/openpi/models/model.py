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

import abc
import dataclasses
import enum
import logging
import pathlib
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import openpi.shared.array_typing as at
import orbax.checkpoint as ocp
import safetensors
import torch
from flax import nnx, struct, traverse_util
from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
from experiments.pi05 import l0s_geometry_aux_model_plumbing_v1 as _geometry_plumbing


logger = logging.getLogger('openpi')

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar('ArrayT', bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = 'pi0'
    PI0_FAST = 'pi0_fast'
    PI05 = 'pi05'


# The model always expects these images
IMAGE_KEYS = (
    'base_0_rgb',
    'left_wrist_0_rgb',
    'right_wrist_0_rgb',
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


def _align_restored_mapping_keys(expected, restored):
    """Align Orbax numeric mapping keys with the NNX reference tree.

    NNX represents modules created from Python lists as integer-keyed mappings.
    Some Orbax restore paths deserialize those keys as decimal strings.  Use
    the freshly initialized reference tree as the authority, changing only a
    key's representation and never its value or position.
    """
    if not isinstance(expected, Mapping) or not isinstance(restored, Mapping):
        return restored

    aligned = {}
    consumed = set()
    for expected_key, expected_value in expected.items():
        candidates = [expected_key]
        if isinstance(expected_key, int):
            candidates.append(str(expected_key))
        elif isinstance(expected_key, str) and expected_key.isdecimal():
            candidates.append(int(expected_key))
        for candidate in candidates:
            if candidate in restored and candidate not in consumed:
                aligned[expected_key] = _align_restored_mapping_keys(
                    expected_value, restored[candidate]
                )
                consumed.add(candidate)
                break

    # Preserve unknown entries so the existing remove_extra_params behavior
    # remains the only authority deciding whether they are accepted.
    for key, value in restored.items():
        if key not in consumed:
            aligned[key] = value
    return aligned


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, '*b h w c']]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, '*b']]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, '*b s']

    # Optional dense future proprioceptive targets used only by state-rollout
    # auxiliary training.  The current state remains in ``state``; this tensor
    # contains the state after each action in the predicted horizon and is
    # never required by inference.
    future_states: at.Float[ArrayT, '*b ah s'] | None = None
    future_state_masks: at.Bool[ArrayT, '*b ah'] | None = None

    # Optional future RGB targets used only by latent future-visual auxiliary
    # training. They are kept separate from ``images`` so a target can never
    # enter the policy prefix or be required by inference.
    future_images: dict[str, at.Float[ArrayT, '*b h w c']] | None = None
    future_image_masks: dict[str, at.Bool[ArrayT, '*b']] | None = None

    # Optional task-matched demonstration trajectory. It is already normalized
    # into the policy action space by the retrieval transform.
    demonstration_actions: at.Float[ArrayT, '*demo_b ah ad'] | None = None
    # Sparse whole-episode demonstration plan. Each token contains a normalized
    # proprioceptive keyframe, its displacement from the preceding keyframe,
    # and normalized episode progress. This complements the local action chunk
    # above with long-horizon task structure.
    demonstration_plan: at.Float[ArrayT, '*demo_b dp pd'] | None = None
    # Scalar validity/dropout mask for the retrieved visual/action context.
    demonstration_mask: at.Bool[ArrayT, '*b'] | None = None
    # Marks a retrieved montage carried in the right-wrist image slot for the
    # shared VLM only. Physical geometry/dynamics modules must mask this slot.
    grounded_context_mask: at.Bool[ArrayT, '*b'] | None = None
    # Synthetic match target used only while training the retrieval reliability
    # gate: one for the correct same-task demonstration, zero for an injected
    # wrong-task negative, and -1 when no context is present. Inference does not
    # need to supply this field.
    demonstration_reliability_target: at.Float[ArrayT, '*b'] | None = None
    # Optional candidate-level metadata for compositional retrieval.  When
    # present, ``demonstration_actions`` and ``demonstration_plan`` carry an
    # additional candidate axis immediately after the batch dimensions.
    demonstration_slot_mask: at.Bool[ArrayT, '*b k'] | None = None
    # Progress-alignment estimate for every candidate trajectory in [0, 1].
    demonstration_progress: at.Float[ArrayT, '*b k'] | None = None
    # Synthetic active-candidate label used by the auxiliary router loss.
    # Inference and dropped contexts use -1.
    demonstration_router_target: at.Int[ArrayT, '*b'] | None = None
    # PSM-SDLA retrieves a separately tokenized demonstration instruction.
    # These tokens are embedded by the frozen PaliGemma input embedding and
    # never replace or extend the current policy prompt.
    demonstration_tokenized_prompt: at.Int[ArrayT, '*demo_b dl'] | None = None
    demonstration_tokenized_prompt_mask: at.Bool[ArrayT, '*demo_b dl'] | None = None
    demonstration_semantic_span_mask: at.Bool[
        ArrayT, '*demo_b ds dl'
    ] | None = None
    demonstration_semantic_valid_mask: at.Bool[
        ArrayT, '*demo_b ds'
    ] | None = None
    # Semantic context and raw trajectory admission are independent.  In
    # particular, L2 may set context true but trajectory must remain false.
    demonstration_context_mask: at.Bool[ArrayT, '*b'] | None = None
    demonstration_trajectory_mask: at.Bool[ArrayT, '*b'] | None = None
    # Decoder-only teacher-forcing labels.  They are consumed solely by the
    # training sequence objective and never by the deployed policy encoder.
    spatial_language_target_ids: at.Int[ArrayT, '*sequence_b sl'] | None = None
    spatial_language_target_mask: at.Bool[
        ArrayT, '*sequence_b sl'
    ] | None = None
    # Normalized training progress: either continuous frame position or a
    # semantic phase anchor. This auxiliary target is never required by
    # inference; -1 denotes an unavailable target.
    task_progress_target: at.Float[ArrayT, '*b'] | None = None
    # Training-only decoder targets in the jointly augmented agentview frame.
    # None during serving; no membership/key field exists in Observation.
    geometry_target_bbox_xyxy: at.Float[ArrayT, '*b four'] | None = None
    geometry_source_xy: at.Float[ArrayT, '*b two'] | None = None
    geometry_destination_bbox_xyxy: at.Float[ArrayT, '*b four'] | None = None
    geometry_destination_route_endpoint_xy: at.Float[ArrayT, '*b two'] | None = None
    geometry_destination_center_xy_unclipped: at.Float[ArrayT, '*b two'] | None = None
    geometry_obstacle_bboxes_xyxy: at.Float[ArrayT, '*b obstacles four'] | None = None
    geometry_obstacle_mask: at.Bool[ArrayT, '*b obstacles'] | None = None
    geometry_route_polyline_xy: at.Float[ArrayT, '*b route two'] | None = None
    geometry_route_point_mask: at.Bool[ArrayT, '*b route'] | None = None
    geometry_path_class: at.Int[ArrayT, '*b'] | None = None
    geometry_safety_class: at.Int[ArrayT, '*b'] | None = None
    geometry_supervision_mask: at.Bool[ArrayT, '*b'] | None = None

    # Server-private recurrent state and sequence-training metadata for the
    # persistent-subgoal-memory architecture. None of these fields is required
    # by stateless policies or exposed in the websocket action response.
    persistent_memory_initial_state: at.Float[ArrayT, '*memory_b mt mh'] | None = None
    persistent_subgoal_initial_frontier: at.Float[ArrayT, '*memory_b sg'] | None = None
    persistent_memory_sequence_valid: at.Bool[ArrayT, '*sequence_b'] | None = None
    # The leading batch shape is ``[batch]`` during serving and
    # ``[batch, replan]`` during sequence training.
    persistent_previous_actions: at.Float[ArrayT, '*sequence_b pa pad'] | None = None
    persistent_previous_actions_valid: at.Bool[ArrayT, '*sequence_b'] | None = None
    persistent_memory_episode_start: at.Bool[ArrayT, '*sequence_b'] | None = None
    persistent_current_subgoal_target: at.Int[ArrayT, '*sequence_b'] | None = None
    persistent_next_subgoal_target: at.Int[ArrayT, '*sequence_b'] | None = None
    persistent_next_progress_target: at.Float[ArrayT, '*sequence_b'] | None = None
    # Mean of the normalized active robot actions that will be executed after
    # this observation. This is a training-only verification target; padded
    # model action dimensions and future replans are intentionally excluded.
    memory_next_action_summary_target: at.Float[ArrayT, '*sequence_b pad'] | None = None
    factorized_role_span_mask: at.Bool[ArrayT, '*factorized_b roles l'] | None = None
    factorized_source_reference_span_mask: at.Bool[
        ArrayT, '*factorized_b l'
    ] | None = None
    factorized_destination_reference_span_mask: at.Bool[
        ArrayT, '*factorized_b refs l'
    ] | None = None
    factorized_condition_span_mask: at.Bool[
        ArrayT, '*factorized_b l'
    ] | None = None
    factorized_destination_qualifier_span_mask: at.Bool[
        ArrayT, '*factorized_b l'
    ] | None = None
    factorized_role_valid_mask: at.Bool[ArrayT, '*factorized_b roles'] | None = None
    factorized_role_identity_labels: at.Int[ArrayT, '*factorized_b roles'] | None = None
    factorized_operation_label: at.Int[ArrayT, '*factorized_b'] | None = None
    factorized_source_relation_label: at.Int[ArrayT, '*factorized_b'] | None = None
    factorized_destination_relation_label: at.Int[ArrayT, '*factorized_b'] | None = None
    factorized_condition_label: at.Int[ArrayT, '*factorized_b'] | None = None
    factorized_destination_qualifier_label: at.Int[
        ArrayT, '*factorized_b'
    ] | None = None
    factorized_auxiliary_valid: at.Bool[ArrayT, '*factorized_b'] | None = None
    # Ordered natural-language clauses used by ClausePlan-v1.  They are
    # derived solely from the ordinary prompt and contain no benchmark/task
    # identifier.  The leading shape follows the prompt batch dimensions.
    clause_span_mask: at.Bool[ArrayT, '*factorized_b clauses l'] | None = None
    clause_valid_mask: at.Bool[ArrayT, '*factorized_b clauses'] | None = None
    persistent_memory_initial_state_key: at.Int[ArrayT, '*memory_b key'] | None = None
    persistent_memory_requires_cached_initial_state: at.Bool[ArrayT, '*memory_b'] | None = None

    hetm_event_ledger: at.Float[ArrayT, '*hetm_b events hidden'] | None = None
    hetm_event_valid: at.Bool[ArrayT, '*hetm_b events'] | None = None
    hetm_event_write_index: at.Int[ArrayT, '*hetm_b'] | None = None
    hetm_last_event_probabilities: at.Float[ArrayT, '*hetm_b event_types'] | None = None
    hetm_predicate_memory: at.Float[ArrayT, '*hetm_b predicates hidden'] | None = None
    hetm_predicate_probabilities: at.Float[ArrayT, '*hetm_b predicates states'] | None = None
    hetm_frontier: at.Float[ArrayT, '*hetm_b predicates'] | None = None
    hetm_previous_actions: at.Float[ArrayT, '*hetm_b steps action'] | None = None
    hetm_episode_start: at.Bool[ArrayT, '*hetm_b'] | None = None
    hetm_history_event_targets: at.Float[ArrayT, '*history_b history event_types'] | None = None
    hetm_history_event_valid: at.Bool[ArrayT, '*history_b history'] | None = None
    hetm_initial_predicate_targets: at.Float[ArrayT, '*initial_b predicates'] | None = None
    hetm_initial_frontier_target: at.Float[ArrayT, '*initial_b'] | None = None
    hetm_initial_state_valid: at.Bool[ArrayT, '*initial_b'] | None = None
    hetm_event_targets: at.Float[ArrayT, '*hetm_b event_types'] | None = None
    hetm_predicate_targets: at.Float[ArrayT, '*hetm_b predicates'] | None = None
    hetm_frontier_target: at.Float[ArrayT, '*hetm_b'] | None = None
    hetm_next_frontier_target: at.Float[ArrayT, '*hetm_b'] | None = None
    hetm_supervision_valid: at.Bool[ArrayT, '*hetm_b'] | None = None

    # Prompt-derived RACG inputs, server-private target state, and training-only
    # graph supervision. The evaluator supplies none of the private/target fields.
    racg_role_span_mask: at.Bool[ArrayT, '*racg_b racg_roles l'] | None = None
    racg_role_valid_mask: at.Bool[ArrayT, '*racg_b racg_roles'] | None = None
    racg_relation_kind: at.Int[ArrayT, '*racg_b'] | None = None
    racg_target_anchor: at.Float[ArrayT, '*racg_state_b hidden'] | None = None
    racg_target_geometry: at.Float[ArrayT, '*racg_state_b geometry'] | None = None
    racg_target_anchor_valid: at.Bool[ArrayT, '*racg_state_b'] | None = None
    racg_episode_start: at.Bool[ArrayT, '*racg_b'] | None = None
    racg_role_identity_labels: at.Int[ArrayT, '*racg_b racg_roles'] | None = None
    racg_relation_target: at.Int[ArrayT, '*racg_b'] | None = None
    racg_relation_valid: at.Bool[ArrayT, '*racg_b'] | None = None
    racg_crossview_role_valid: at.Bool[ArrayT, '*racg_b racg_roles'] | None = None
    racg_contact_target: at.Int[ArrayT, '*racg_b ah'] | None = None
    racg_contact_valid: at.Bool[ArrayT, '*racg_b ah'] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, '*b l'] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, '*b l'] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, '*b l'] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, '*b l'] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> 'Observation[ArrayT]':
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ('tokenized_prompt' in data) != ('tokenized_prompt_mask' in data):
            raise ValueError(
                'tokenized_prompt and tokenized_prompt_mask must be provided together.'
            )
        # If images are uint8, convert them to [-1, 1] float32.
        for image_group_name in ('image', 'future_image'):
            image_group = data.get(image_group_name)
            if image_group is None:
                continue
            for key in image_group:
                if image_group[key].dtype == np.uint8:
                    image_group[key] = (
                        image_group[key].astype(np.float32) / 255.0 * 2.0 - 1.0
                    )
                elif (
                    hasattr(image_group[key], 'dtype')
                    and image_group[key].dtype == torch.uint8
                ):
                    # Preserve arbitrary leading batch/sequence axes.  Normal
                    # policy batches are [B,H,W,C]; recurrent batches are
                    # [B,R,H,W,C].
                    image_group[key] = (
                        image_group[key].to(torch.float32).movedim(-1, -3)
                        / 255.0
                        * 2.0
                        - 1.0
                    )
        return cls(
            images=data['image'],
            image_masks=data['image_mask'],
            state=data['state'],
            future_states=data.get('future_state'),
            future_state_masks=data.get('future_state_mask'),
            future_images=data.get('future_image'),
            future_image_masks=data.get('future_image_mask'),
            demonstration_actions=data.get('demonstration_actions'),
            demonstration_plan=data.get('demonstration_plan'),
            demonstration_mask=data.get('demonstration_mask'),
            grounded_context_mask=data.get('grounded_context_mask'),
            demonstration_reliability_target=data.get(
                'demonstration_reliability_target'
            ),
            demonstration_slot_mask=data.get('demonstration_slot_mask'),
            demonstration_progress=data.get('demonstration_progress'),
            demonstration_router_target=data.get('demonstration_router_target'),
            demonstration_tokenized_prompt=data.get(
                'demonstration_tokenized_prompt'
            ),
            demonstration_tokenized_prompt_mask=data.get(
                'demonstration_tokenized_prompt_mask'
            ),
            demonstration_semantic_span_mask=data.get(
                'demonstration_semantic_span_mask'
            ),
            demonstration_semantic_valid_mask=data.get(
                'demonstration_semantic_valid_mask'
            ),
            demonstration_context_mask=data.get('demonstration_context_mask'),
            demonstration_trajectory_mask=data.get(
                'demonstration_trajectory_mask'
            ),
            spatial_language_target_ids=data.get('spatial_language_target_ids'),
            spatial_language_target_mask=data.get(
                'spatial_language_target_mask'
            ),
            task_progress_target=data.get('task_progress_target'),
            geometry_target_bbox_xyxy=data.get('geometry_target_bbox_xyxy'),
            geometry_source_xy=data.get('geometry_source_xy'),
            geometry_destination_bbox_xyxy=data.get('geometry_destination_bbox_xyxy'),
            geometry_destination_route_endpoint_xy=data.get('geometry_destination_route_endpoint_xy'),
            geometry_destination_center_xy_unclipped=data.get('geometry_destination_center_xy_unclipped'),
            geometry_obstacle_bboxes_xyxy=data.get('geometry_obstacle_bboxes_xyxy'),
            geometry_obstacle_mask=data.get('geometry_obstacle_mask'),
            geometry_route_polyline_xy=data.get('geometry_route_polyline_xy'),
            geometry_route_point_mask=data.get('geometry_route_point_mask'),
            geometry_path_class=data.get('geometry_path_class'),
            geometry_safety_class=data.get('geometry_safety_class'),
            geometry_supervision_mask=data.get('geometry_supervision_mask'),
            persistent_memory_initial_state=data.get(
                'persistent_memory_initial_state'
            ),
            persistent_subgoal_initial_frontier=data.get(
                'persistent_subgoal_initial_frontier'
            ),
            persistent_memory_sequence_valid=data.get(
                'persistent_memory_sequence_valid'
            ),
            persistent_previous_actions=data.get('persistent_previous_actions'),
            persistent_previous_actions_valid=data.get(
                'persistent_previous_actions_valid'
            ),
            persistent_memory_episode_start=data.get(
                'persistent_memory_episode_start'
            ),
            persistent_current_subgoal_target=data.get(
                'persistent_current_subgoal_target'
            ),
            persistent_next_subgoal_target=data.get(
                'persistent_next_subgoal_target'
            ),
            persistent_next_progress_target=data.get(
                'persistent_next_progress_target'
            ),
            memory_next_action_summary_target=data.get(
                'memory_next_action_summary_target'
            ),
            factorized_role_span_mask=data.get('factorized_role_span_mask'),
            factorized_source_reference_span_mask=data.get(
                'factorized_source_reference_span_mask'
            ),
            factorized_destination_reference_span_mask=data.get(
                'factorized_destination_reference_span_mask'
            ),
            factorized_condition_span_mask=data.get(
                'factorized_condition_span_mask'
            ),
            factorized_destination_qualifier_span_mask=data.get(
                'factorized_destination_qualifier_span_mask'
            ),
            factorized_role_valid_mask=data.get('factorized_role_valid_mask'),
            factorized_role_identity_labels=data.get(
                'factorized_role_identity_labels'
            ),
            factorized_operation_label=data.get('factorized_operation_label'),
            factorized_source_relation_label=data.get(
                'factorized_source_relation_label'
            ),
            factorized_destination_relation_label=data.get(
                'factorized_destination_relation_label'
            ),
            factorized_condition_label=data.get('factorized_condition_label'),
            factorized_destination_qualifier_label=data.get(
                'factorized_destination_qualifier_label'
            ),
            factorized_auxiliary_valid=data.get('factorized_auxiliary_valid'),
            clause_span_mask=data.get('clause_span_mask'),
            clause_valid_mask=data.get('clause_valid_mask'),
            persistent_memory_initial_state_key=data.get(
                'persistent_memory_initial_state_key'
            ),
            persistent_memory_requires_cached_initial_state=data.get(
                'persistent_memory_requires_cached_initial_state'
            ),
            hetm_event_ledger=data.get('hetm_event_ledger'),
            hetm_event_valid=data.get('hetm_event_valid'),
            hetm_event_write_index=data.get('hetm_event_write_index'),
            hetm_last_event_probabilities=data.get('hetm_last_event_probabilities'),
            hetm_predicate_memory=data.get('hetm_predicate_memory'),
            hetm_predicate_probabilities=data.get('hetm_predicate_probabilities'),
            hetm_frontier=data.get('hetm_frontier'),
            hetm_previous_actions=data.get('hetm_previous_actions'),
            hetm_episode_start=data.get('hetm_episode_start'),
            hetm_history_event_targets=(
                None
                if data.get('hetm_history_event_targets') is None
                else np.asarray(data['hetm_history_event_targets'], dtype=np.float32)
            ),
            hetm_history_event_valid=data.get('hetm_history_event_valid'),
            hetm_initial_predicate_targets=(
                None
                if data.get('hetm_initial_predicate_targets') is None
                else np.asarray(data['hetm_initial_predicate_targets'], dtype=np.float32)
            ),
            hetm_initial_frontier_target=(
                None
                if data.get('hetm_initial_frontier_target') is None
                else np.asarray(data['hetm_initial_frontier_target'], dtype=np.float32)
            ),
            hetm_initial_state_valid=data.get('hetm_initial_state_valid'),
            hetm_event_targets=(
                None
                if data.get('hetm_event_targets') is None
                else np.asarray(data['hetm_event_targets'], dtype=np.float32)
            ),
            hetm_predicate_targets=(
                None
                if data.get('hetm_predicate_targets') is None
                else np.asarray(data['hetm_predicate_targets'], dtype=np.float32)
            ),
            hetm_frontier_target=(
                None
                if data.get('hetm_frontier_target') is None
                else np.asarray(data['hetm_frontier_target'], dtype=np.float32)
            ),
            hetm_next_frontier_target=(
                None
                if data.get('hetm_next_frontier_target') is None
                else np.asarray(data['hetm_next_frontier_target'], dtype=np.float32)
            ),
            hetm_supervision_valid=data.get('hetm_supervision_valid'),
            racg_role_span_mask=data.get('racg_role_span_mask'),
            racg_role_valid_mask=data.get('racg_role_valid_mask'),
            racg_relation_kind=data.get('racg_relation_kind'),
            racg_target_anchor=data.get('racg_target_anchor'),
            racg_target_geometry=data.get('racg_target_geometry'),
            racg_target_anchor_valid=data.get('racg_target_anchor_valid'),
            racg_episode_start=data.get('racg_episode_start'),
            racg_role_identity_labels=data.get('racg_role_identity_labels'),
            racg_relation_target=data.get('racg_relation_target'),
            racg_relation_valid=data.get('racg_relation_valid'),
            racg_crossview_role_valid=data.get('racg_crossview_role_valid'),
            racg_contact_target=data.get('racg_contact_target'),
            racg_contact_valid=data.get('racg_contact_valid'),
            tokenized_prompt=data.get('tokenized_prompt'),
            tokenized_prompt_mask=data.get('tokenized_prompt_mask'),
            token_ar_mask=data.get('token_ar_mask'),
            token_loss_mask=data.get('token_loss_mask'),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result['image'] = result.pop('images')
        result['image_mask'] = result.pop('image_masks')
        future_states = result.pop('future_states')
        future_state_masks = result.pop('future_state_masks')
        if future_states is not None:
            result['future_state'] = future_states
            result['future_state_mask'] = future_state_masks
        future_images = result.pop('future_images')
        future_image_masks = result.pop('future_image_masks')
        if future_images is not None:
            result['future_image'] = future_images
            result['future_image_mask'] = future_image_masks
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, '*b ah ad']


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f'images dict missing keys: expected {image_keys}, got {list(observation.images)}'
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    out_future_images = {} if observation.future_images is not None else None
    geometry_updates = _geometry_plumbing.unchanged_geometry_fields(observation)
    geometry_targets_present = _geometry_plumbing.has_geometry_training_targets(observation)
    for key in image_keys:
        image = observation.images[key]
        future_image = (
            observation.future_images.get(key)
            if observation.future_images is not None
            else None
        )
        if (
            key == 'base_0_rgb'
            and geometry_targets_present
            and image.shape[1:3] != image_resolution
            and image.shape[1] * image_resolution[1]
            != image.shape[2] * image_resolution[0]
        ):
            raise ValueError('geometry agentview resize would add padding and drift normalized xy')
        # A pure same-aspect resize (the production 256x256 -> 224x224
        # case) preserves normalized xy exactly; the ordinary image
        # resize below may therefore run before the single joint aug.
        if image.shape[1:3] != image_resolution:
            logger.info(
                f'Resizing image {key} from {image.shape[1:3]} to {image_resolution}'
            )
            image = image_tools.resize_with_pad(image, *image_resolution)
        if future_image is not None and future_image.shape[1:3] != image_resolution:
            logger.info(
                f'Resizing future image {key} from '
                f'{future_image.shape[1:3]} to {image_resolution}'
            )
            future_image = image_tools.resize_with_pad(
                future_image, *image_resolution
            )

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if 'wrist' not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(
                    brightness=0.3, contrast=0.4, saturation=0.5
                ),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            if key == 'base_0_rgb' and geometry_targets_present:
                image, geometry_updates = _geometry_plumbing.augment_agentview_batch(
                    sub_rngs, image, observation
                )
            else:
                image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
            if future_image is not None:
                # Reuse exactly the same geometric and photometric draws for
                # the current/future pair. Otherwise the auxiliary target
                # would learn augmentation differences instead of dynamics.
                future_image = future_image / 2.0 + 0.5
                future_image = jax.vmap(augmax.Chain(*transforms))(
                    sub_rngs, future_image
                )
                future_image = future_image * 2.0 - 1.0

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image
        if out_future_images is not None and future_image is not None:
            out_future_images[key] = future_image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    out_future_masks = None
    if out_future_images is not None:
        out_future_masks = {}
        supplied_future_masks = observation.future_image_masks or {}
        for key in out_future_images:
            out_future_masks[key] = jnp.asarray(
                supplied_future_masks.get(
                    key, jnp.ones(batch_shape, dtype=jnp.bool_)
                )
            )

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        future_states=observation.future_states,
        future_state_masks=observation.future_state_masks,
        future_images=out_future_images,
        future_image_masks=out_future_masks,
        demonstration_actions=observation.demonstration_actions,
        demonstration_plan=observation.demonstration_plan,
        demonstration_mask=observation.demonstration_mask,
        grounded_context_mask=observation.grounded_context_mask,
        demonstration_reliability_target=(
            observation.demonstration_reliability_target
        ),
        demonstration_slot_mask=observation.demonstration_slot_mask,
        demonstration_progress=observation.demonstration_progress,
        demonstration_router_target=observation.demonstration_router_target,
        demonstration_tokenized_prompt=(
            observation.demonstration_tokenized_prompt
        ),
        demonstration_tokenized_prompt_mask=(
            observation.demonstration_tokenized_prompt_mask
        ),
        demonstration_semantic_span_mask=(
            observation.demonstration_semantic_span_mask
        ),
        demonstration_semantic_valid_mask=(
            observation.demonstration_semantic_valid_mask
        ),
        demonstration_context_mask=observation.demonstration_context_mask,
        demonstration_trajectory_mask=(
            observation.demonstration_trajectory_mask
        ),
        spatial_language_target_ids=observation.spatial_language_target_ids,
        spatial_language_target_mask=observation.spatial_language_target_mask,
        task_progress_target=observation.task_progress_target,
        **geometry_updates,
        persistent_memory_initial_state=(
            observation.persistent_memory_initial_state
        ),
        persistent_subgoal_initial_frontier=(
            observation.persistent_subgoal_initial_frontier
        ),
        persistent_memory_sequence_valid=(
            observation.persistent_memory_sequence_valid
        ),
        persistent_previous_actions=observation.persistent_previous_actions,
        persistent_previous_actions_valid=(
            observation.persistent_previous_actions_valid
        ),
        persistent_memory_episode_start=(
            observation.persistent_memory_episode_start
        ),
        persistent_current_subgoal_target=(
            observation.persistent_current_subgoal_target
        ),
        persistent_next_subgoal_target=(
            observation.persistent_next_subgoal_target
        ),
        persistent_next_progress_target=(
            observation.persistent_next_progress_target
        ),
        memory_next_action_summary_target=(
            observation.memory_next_action_summary_target
        ),
        factorized_role_span_mask=observation.factorized_role_span_mask,
        factorized_source_reference_span_mask=(
            observation.factorized_source_reference_span_mask
        ),
        factorized_destination_reference_span_mask=(
            observation.factorized_destination_reference_span_mask
        ),
        factorized_condition_span_mask=(
            observation.factorized_condition_span_mask
        ),
        factorized_destination_qualifier_span_mask=(
            observation.factorized_destination_qualifier_span_mask
        ),
        factorized_role_valid_mask=observation.factorized_role_valid_mask,
        factorized_role_identity_labels=(
            observation.factorized_role_identity_labels
        ),
        factorized_operation_label=observation.factorized_operation_label,
        factorized_source_relation_label=(
            observation.factorized_source_relation_label
        ),
        factorized_destination_relation_label=(
            observation.factorized_destination_relation_label
        ),
        factorized_condition_label=observation.factorized_condition_label,
        factorized_destination_qualifier_label=(
            observation.factorized_destination_qualifier_label
        ),
        factorized_auxiliary_valid=observation.factorized_auxiliary_valid,
        clause_span_mask=observation.clause_span_mask,
        clause_valid_mask=observation.clause_valid_mask,
        persistent_memory_initial_state_key=(
            observation.persistent_memory_initial_state_key
        ),
        persistent_memory_requires_cached_initial_state=(
            observation.persistent_memory_requires_cached_initial_state
        ),
        hetm_event_ledger=observation.hetm_event_ledger,
        hetm_event_valid=observation.hetm_event_valid,
        hetm_event_write_index=observation.hetm_event_write_index,
        hetm_last_event_probabilities=(
            observation.hetm_last_event_probabilities
        ),
        hetm_predicate_memory=observation.hetm_predicate_memory,
        hetm_predicate_probabilities=(
            observation.hetm_predicate_probabilities
        ),
        hetm_frontier=observation.hetm_frontier,
        hetm_previous_actions=observation.hetm_previous_actions,
        hetm_episode_start=observation.hetm_episode_start,
        hetm_history_event_targets=observation.hetm_history_event_targets,
        hetm_history_event_valid=observation.hetm_history_event_valid,
        hetm_initial_predicate_targets=(
            observation.hetm_initial_predicate_targets
        ),
        hetm_initial_frontier_target=(
            observation.hetm_initial_frontier_target
        ),
        hetm_initial_state_valid=observation.hetm_initial_state_valid,
        hetm_event_targets=observation.hetm_event_targets,
        hetm_predicate_targets=observation.hetm_predicate_targets,
        hetm_frontier_target=observation.hetm_frontier_target,
        hetm_next_frontier_target=observation.hetm_next_frontier_target,
        hetm_supervision_valid=observation.hetm_supervision_valid,

        racg_role_span_mask=observation.racg_role_span_mask,
        racg_role_valid_mask=observation.racg_role_valid_mask,
        racg_relation_kind=observation.racg_relation_kind,
        racg_target_anchor=observation.racg_target_anchor,
        racg_target_geometry=observation.racg_target_geometry,
        racg_target_anchor_valid=observation.racg_target_anchor_valid,
        racg_episode_start=observation.racg_episode_start,
        racg_role_identity_labels=observation.racg_role_identity_labels,
        racg_relation_target=observation.racg_relation_target,
        racg_relation_valid=observation.racg_relation_valid,
        racg_crossview_role_valid=observation.racg_crossview_role_valid,
        racg_contact_target=observation.racg_contact_target,
        racg_contact_valid=observation.racg_contact_valid,

        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> 'BaseModel':
        """Create a new model, initializing parameters."""

    def load(
        self, params: at.Params, *, remove_extra_params: bool = True
    ) -> 'BaseModel':
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        expected = state.to_pure_dict()
        params = _align_restored_mapping_keys(expected, params)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(
                expected, params
            )
            # Orbax's intersection currently reconstructs nested containers
            # from the restored tree and therefore reintroduces decimal-string
            # keys.  Align once more after intersection before strict checking.
            params = _align_restored_mapping_keys(expected, params)
        at.check_pytree_equality(
            expected=expected,
            got=params,
            check_shapes=True,
            check_dtypes=False,
        )
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f'train_config: {train_config}')
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(
            lambda x: jnp.ones(x.shape, x.dtype), observation_spec
        )

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, '*b ah']: ...

    @abc.abstractmethod
    def sample_actions(
        self, rng: at.KeyArrayLike, observation: Observation, **kwargs
    ) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = (
        pathlib.Path(params_path).resolve()
        if not str(params_path).startswith('gs://')
        else params_path
    )

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ('x',))
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {'params': metadata['params']}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding,
                        restore_type=restore_type,
                        dtype=dtype,
                    ),
                    item,
                ),
            ),
        )['params']

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == 'value' for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
