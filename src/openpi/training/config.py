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

"""See _CONFIGS for the list of available configs."""

import abc
import dataclasses
import difflib
import logging
import os
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias
from typing_extensions import override

import etils.epath as epath
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import tyro


ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ('actions',)

    # Optional LeRobot observation sequences expressed as frame offsets. A
    # value such as ``{'image': (0, 9)}`` asks the loader for the current image
    # and the image 0.9 s later in a 10 Hz dataset. This is empty for every
    # existing config and is used only by future-visual auxiliary training.
    observation_sequence_offsets: Mapping[str, Sequence[int]] = (
        dataclasses.field(default_factory=dict)
    )

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional, training-only sampling partition computed from exact L0
    # trajectories.  It changes sampler mass only and is never placed in a
    # row, observation, cache key, or model input.
    persistent_memory_exact_alias_artifact_path: str | None = None
    persistent_memory_allow_unpromoted_alias_artifact: bool = False
    # Explicit training-only mmap sidecar; inference configs leave all three None.
    geometry_aux_sidecar_manifest_path: str | None = None
    geometry_aux_sidecar_file_sha256: str | None = None
    geometry_aux_sidecar_internal_sha256: str | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(
        self, model_config: _model.BaseModelConfig
    ) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # External natural-language sources can explicitly retain ordinary prompt
    # tokenization even when the target architecture supports factorized roles.
    factorized_prompt: bool | None = None
    # Request ordered prompt-clause masks without the VLA-Arena-specific
    # factorized-role grammar (used by heterogeneous auxiliary datasets).
    clause_prompt: bool | None = None

    def __call__(
        self, model_config: _model.BaseModelConfig
    ) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(
                                model_config.max_token_len
                            ),
                        ),
                        _transforms.PadStatesAndActions(
                            model_config.action_dim
                        ),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                if self.factorized_prompt and not model_config.persistent_subgoal_memory:
                    raise ValueError(
                        'factorized prompts require a persistent-memory model'
                    )
                clause_plan = model_config.persistent_clause_plan_v1
                if self.clause_prompt and not clause_plan:
                    raise ValueError(
                        'clause prompts require a ClausePlan-v1 model'
                    )
                if clause_plan and self.factorized_prompt is False:
                    tokenize_prompt = _transforms.ClauseTokenizePrompt
                elif (
                    model_config.persistent_subgoal_memory
                    and self.factorized_prompt is not False
                ):
                    tokenize_prompt = _transforms.FactorizedTokenizePrompt
                else:
                    tokenize_prompt = _transforms.TokenizePrompt
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        *(
                            [_transforms.RejectRACGRightCamera()]
                            if model_config.role_affordance_causal_graph
                            else []
                        ),
                        tokenize_prompt(
                            _tokenizer.PaligemmaTokenizer(
                                model_config.max_token_len
                            ),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(
                            model_config.action_dim
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {}
                    if model_config.fast_model_tokenizer_kwargs is None
                    else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs
                            ),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs
                            ),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        """Create a data config."""

    def create_base_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(
                _download.maybe_download(data_assets_dir)
            )
            logging.info(f'Loaded norm stats from {data_assets_dir}')
            return norm_stats
        except FileNotFoundError:
            logging.info(
                f'Norm stats not found in {data_assets_dir}, skipping.'
            )
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = 'fake'

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=GroupFactory
    )
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    # None keeps the architecture default. External datasets such as
    # RoboDojo can opt out of VLA-Arena's factorized-role prompt grammar.
    factorized_prompt: bool | None = None

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = (
        dataclasses.field(
            default=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            'images': {'cam_high': 'observation.images.top'},
                            'state': 'observation.state',
                            'actions': 'action',
                        }
                    )
                ]
            )
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ('action',)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt,
            factorized_prompt=self.factorized_prompt,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    demonstration_bank_path: str | None = None
    # Immutable bank/manifest pair for the native PSM-SDLA consumer.  It is
    # deliberately separate from montage-based retrieved demonstrations.
    structured_demo_manifest_path: str | None = None
    structured_demo_bank_path: str | None = None
    # Optional role-grounded v2 montage/rationale bank used alongside the
    # native structured PSM bank. It has an independent dropout schedule and
    # is visible only to the shared VLM, never to physical camera reasoners.
    grounded_demonstration_bank_path: str | None = None
    grounded_demonstration_dropout: float = 0.5
    grounded_demonstration_retrieval_mode: str = 'role_signature'
    grounded_demonstration_rationale: bool = False
    demonstration_dropout: float = 0.5
    # Conditional probability of replacing an enabled same-task reference with
    # a deterministic wrong-task reference during training. This provides
    # negative examples for the model-side retrieval reliability gate. It is
    # never applied to inference inputs, which lack episode/frame indexes.
    demonstration_mismatch_rate: float = 0.0
    # Add a compact subgoal/motion-language description derived from the same
    # retrieved training reference used by the visual and action branches.
    structured_demonstration_rationale: bool = False
    # Values above one switch to candidate-level compositional retrieval and
    # must agree with ``Pi0Config.compositional_demo_slots``.
    compositional_demonstration_slots: int = 1
    # Load paired current/future RGB at the final action-horizon frame. The
    # specialized transform keeps targets outside the normal policy image
    # prefix and omits them entirely during inference.
    future_visual_supervision: bool = False
    # Load the current state plus the ten states reached after each action in
    # the chunk.  The transform separates these training-only rollout targets
    # from the ordinary single-state inference interface.
    future_state_supervision: bool = False
    # Supervise a model-side progress-state predictor with either exact
    # normalized frame position from LeRobot metadata or event-derived phase
    # anchors from a semantic-phase manifest. The target is omitted at
    # inference; only the predicted latent state conditions actions.
    task_progress_supervision: bool = False
    episode_metadata_path: str | None = None
    persistent_memory_exact_alias_artifact_path: str | None = None
    persistent_memory_allow_unpromoted_alias_artifact: bool = False
    geometry_aux_sidecar_manifest_path: str | None = None
    geometry_aux_sidecar_file_sha256: str | None = None
    geometry_aux_sidecar_internal_sha256: str | None = None
    # None preserves the architecture default. False is required for external
    # instructions that do not carry audited VLA-Arena factorized role spans.
    factorized_prompt: bool | None = None
    clause_prompt: bool | None = None

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_structure = {
            'observation/image': 'image',
            'observation/wrist_image': 'wrist_image',
            'observation/state': 'state',
            'actions': 'actions',
            'prompt': 'prompt',
        }
        if self.future_visual_supervision:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('future visual supervision requires Pi0Config')
            if not (
                model_config.latent_future_reasoner
                or model_config.object_future_reasoner
            ):
                raise ValueError(
                    'data requests future visual targets but the model reasoner '
                    'is disabled'
                )
            if self.compositional_demonstration_slots > 1:
                raise ValueError(
                    'future visual supervision is not combined with the '
                    'compositional retrieval branch'
                )
            repack_structure.update(
                {
                    'observation/image_is_pad': 'image_is_pad',
                    'observation/wrist_image_is_pad': 'wrist_image_is_pad',
                }
            )
        if self.future_state_supervision:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('future state supervision requires Pi0Config')
            if not model_config.state_rollout_reasoner:
                raise ValueError(
                    'data requests future state targets but the rollout '
                    'reasoner is disabled'
                )
            if self.compositional_demonstration_slots > 1:
                raise ValueError(
                    'future state supervision is not combined with the '
                    'compositional retrieval branch'
                )
            repack_structure['observation/state_is_pad'] = 'state_is_pad'
        if self.task_progress_supervision:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('task progress supervision requires Pi0Config')
            if not (
                model_config.task_progress_reasoner
                or model_config.language_subgoal_reasoner
            ):
                raise ValueError(
                    'data requests task progress targets but no compatible '
                    'model reasoner is enabled'
                )
            if self.episode_metadata_path is None:
                raise ValueError(
                    'task progress supervision requires episode_metadata_path'
                )
        if (self.structured_demo_manifest_path is None) != (
            self.structured_demo_bank_path is None
        ):
            raise ValueError(
                'structured demo manifest and bank must be supplied together'
            )
        if self.structured_demo_manifest_path is not None:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('PSM-SDLA data requires Pi0Config')
            if not model_config.persistent_structured_demo_language:
                raise ValueError('PSM-SDLA data requires the model overlay')
            if self.demonstration_bank_path is not None:
                raise ValueError(
                    'PSM-SDLA cannot share the montage demonstration consumer'
                )
            formal_source = f'{self.repo_id} {self.structured_demo_bank_path}'
            if re.search(
                r'vla[_-]arena[_-]l[12](?:[_/\\-]|$)',
                formal_source,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    'formal PSM-SDLA rejects every VLA-Arena L1/L2 bank/repo'
                )
            if 'VLA_Arena_L0_L_lerobot_openpi' not in str(self.repo_id):
                raise ValueError(
                    'formal PSM-SDLA data must use the released L0 train repo'
                )
        if self.grounded_demonstration_bank_path is not None:
            if self.structured_demo_manifest_path is None:
                raise ValueError(
                    'grounded role context requires the structured PSM bank'
                )
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('grounded role context requires Pi0Config')
            if not model_config.grounded_demonstration_camera_context_only:
                raise ValueError(
                    'grounded role context requires context-only camera masking'
                )
            if not self.grounded_demonstration_rationale:
                raise ValueError(
                    'grounded role context requires the audited bank rationale'
                )
            if self.grounded_demonstration_retrieval_mode != 'role_signature':
                raise ValueError(
                    'grounded role context requires role_signature retrieval'
                )
        if self.persistent_memory_exact_alias_artifact_path is not None:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('exact trajectory aliases require Pi0Config')
            if not model_config.persistent_subgoal_memory:
                raise ValueError(
                    'exact trajectory aliases are only valid for persistent '
                    'memory sequence sampling'
                )
            if 'VLA_Arena_L0_L_lerobot_openpi' not in str(self.repo_id):
                raise ValueError(
                    'exact trajectory aliases are bound to the released L0 '
                    'train repo'
                )
        if (
            self.demonstration_bank_path is not None
            or self.structured_demo_manifest_path is not None
            or self.grounded_demonstration_bank_path is not None
            or self.task_progress_supervision
        ):
            repack_structure.update(
                {
                    'episode_index': 'episode_index',
                    'frame_index': 'frame_index',
                }
            )
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(repack_structure)
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        if (
            self.structured_demo_manifest_path is not None
            and self.grounded_demonstration_bank_path is not None
        ):
            assert self.structured_demo_bank_path is not None
            input_transform = (
                libero_policy.GroundedStructuredDemoLanguageLiberoInputs(
                    model_type=model_config.model_type,
                    structured_manifest_path=self.structured_demo_manifest_path,
                    structured_bank_path=self.structured_demo_bank_path,
                    grounded_bank_path=self.grounded_demonstration_bank_path,
                    structured_dropout=self.demonstration_dropout,
                    grounded_dropout=self.grounded_demonstration_dropout,
                    grounded_retrieval_mode=(
                        self.grounded_demonstration_retrieval_mode
                    ),
                    future_visual_supervision=self.future_visual_supervision,
                    future_state_supervision=self.future_state_supervision,
                    task_progress_supervision=self.task_progress_supervision,
                    episode_metadata_path=self.episode_metadata_path,
                )
            )
        elif self.structured_demo_manifest_path is not None:
            assert self.structured_demo_bank_path is not None
            input_transform = libero_policy.StructuredDemoLanguageLiberoInputs(
                model_type=model_config.model_type,
                manifest_path=self.structured_demo_manifest_path,
                bank_path=self.structured_demo_bank_path,
                dropout=self.demonstration_dropout,
                future_visual_supervision=self.future_visual_supervision,
                future_state_supervision=self.future_state_supervision,
                task_progress_supervision=self.task_progress_supervision,
                episode_metadata_path=self.episode_metadata_path,
            )
        elif self.demonstration_bank_path is None:
            if self.structured_demonstration_rationale:
                raise ValueError(
                    'structured rationales require a demonstration bank'
                )
            input_transform = libero_policy.LiberoInputs(
                model_type=model_config.model_type,
                future_visual_supervision=self.future_visual_supervision,
                future_state_supervision=self.future_state_supervision,
                task_progress_supervision=self.task_progress_supervision,
                episode_metadata_path=self.episode_metadata_path,
            )
        elif self.compositional_demonstration_slots > 1:
            if not isinstance(model_config, pi0_config.Pi0Config):
                raise ValueError('compositional demonstrations require Pi0Config')
            if not model_config.compositional_demo_routing:
                raise ValueError(
                    'data requests compositional demonstrations but the model router is disabled'
                )
            if (
                self.compositional_demonstration_slots
                != model_config.compositional_demo_slots
            ):
                raise ValueError(
                    'data and model compositional demonstration slot counts must match'
                )
            input_transform = (
                libero_policy.CompositionalRetrievedDemonstrationLiberoInputs(
                    model_type=model_config.model_type,
                    demonstration_bank_path=self.demonstration_bank_path,
                    max_slots=self.compositional_demonstration_slots,
                    dropout=self.demonstration_dropout,
                )
            )
        else:
            if self.structured_demonstration_rationale:
                if not isinstance(model_config, pi0_config.Pi0Config):
                    raise ValueError('structured rationales require Pi0Config')
                if not model_config.structured_rationale_reasoner:
                    raise ValueError(
                        'data requests structured rationales but the model reasoner is disabled'
                    )
            elif (
                isinstance(model_config, pi0_config.Pi0Config)
                and model_config.structured_rationale_reasoner
            ):
                raise ValueError(
                    'structured rationale model requires structured demonstration text'
                )
            input_transform = libero_policy.RetrievedDemonstrationLiberoInputs(
                model_type=model_config.model_type,
                demonstration_bank_path=self.demonstration_bank_path,
                dropout=self.demonstration_dropout,
                mismatch_rate=self.demonstration_mismatch_rate,
                structured_rationale=self.structured_demonstration_rationale,
                future_visual_supervision=self.future_visual_supervision,
                future_state_supervision=self.future_state_supervision,
                task_progress_supervision=self.task_progress_supervision,
                episode_metadata_path=self.episode_metadata_path,
            )
        data_transforms = _transforms.Group(
            inputs=[input_transform],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory(
            factorized_prompt=self.factorized_prompt,
            clause_prompt=self.clause_prompt,
        )(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        observation_sequence_offsets = {}
        if self.future_visual_supervision:
            observation_sequence_offsets.update(
                {
                    'image': (0, model_config.action_horizon - 1),
                    'wrist_image': (0, model_config.action_horizon - 1),
                }
            )
        if self.future_state_supervision:
            observation_sequence_offsets.update(
                {
                    'state': tuple(range(model_config.action_horizon + 1)),
                }
            )
        base_config = self.create_base_config(assets_dirs, model_config)
        norm_stats = base_config.norm_stats
        if self.future_state_supervision and norm_stats is not None:
            if 'state' not in norm_stats:
                raise ValueError(
                    'future state supervision requires state normalization stats'
                )
            norm_stats = {**norm_stats, 'future_state': norm_stats['state']}
        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            observation_sequence_offsets=observation_sequence_offsets,
            norm_stats=norm_stats,
            persistent_memory_exact_alias_artifact_path=(
                self.persistent_memory_exact_alias_artifact_path
            ),
            persistent_memory_allow_unpromoted_alias_artifact=(
                self.persistent_memory_allow_unpromoted_alias_artifact
            ),
            geometry_aux_sidecar_manifest_path=(
                self.geometry_aux_sidecar_manifest_path
            ),
            geometry_aux_sidecar_file_sha256=(
                self.geometry_aux_sidecar_file_sha256
            ),
            geometry_aux_sidecar_internal_sha256=(
                self.geometry_aux_sidecar_internal_sha256
            ),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = (
        'gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json'
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        'observation/exterior_image_1_left': 'observation/image',
                        'observation/wrist_image_left': 'observation/wrist_image',
                        'observation/joint_position': 'observation/joint_position',
                        'observation/gripper_position': 'observation/gripper_position',
                        'actions': 'actions',
                        'prompt': 'prompt',
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type)
            ],
            outputs=[droid_policy.DroidOutputs()],
        )

        if (
            self.action_space
            == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION
        ):
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert (
            self.rlds_data_dir is not None
        ), 'Need to set rlds data dir for RLDS data loader.'

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        'observation/exterior_image_1_left': 'exterior_image_1_left',
                        'observation/exterior_image_2_left': 'exterior_image_2_left',
                        'observation/wrist_image_left': 'wrist_image_left',
                        'observation/joint_position': 'joint_position',
                        'observation/gripper_position': 'gripper_position',
                        'actions': 'actions',
                        'prompt': 'prompt',
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type)
            ],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = 'vla-arena'
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config
    )

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal['bfloat16', 'float32'] = 'bfloat16'

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=_optimizer.AdamW
    )
    # Multipliers are applied to completed AdamW updates, giving small newly
    # initialized architecture modules a distinct effective learning rate
    # without increasing the conservative backbone learning rate.
    architecture_update_path: str | None = None
    architecture_update_multiplier: float = 1.0
    # Additional independently scaled architecture parameter groups.  This is
    # used by joint candidates whose newly initialized modules do not share a
    # single path substring.  The legacy singular fields remain supported.
    architecture_update_multipliers: tuple[tuple[str, float], ...] = ()
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(
        default_factory=nnx.Nothing
    )

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    # Independently transformed action-supervised source. Its gradient is
    # computed separately and cannot consume target-only sequence labels.
    auxiliary_data: DataConfigFactory | None = None
    auxiliary_batch_size: int = 0
    auxiliary_loss_weight: float = 0.0
    auxiliary_num_workers: int = 0
    auxiliary_task_balanced_sampling: bool = False
    auxiliary_gradient_path_allowlist: tuple[str, ...] = ()
    auxiliary_gradient_merge: Literal[
        'convex',
        'target_preserving_pcgrad',
        'clip_aware_target_preserving_pcgrad',
    ] = 'convex'
    # The target gradient is clipped to this norm before clip-aware auxiliary
    # merging.  The final optimizer threshold must admit this protected target
    # plus the weighted, independently capped auxiliary gradient.
    auxiliary_target_gradient_clip_norm: float = 1.0

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = './assets'
    # Base directory for checkpoints.
    checkpoint_base_dir: str = './checkpoints'

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Split one effective batch into this many sequential microbatches before
    # applying one optimizer update. This reduces activation memory without
    # changing effective batch size, LR steps, or checkpoint semantics.
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Sample frames with inverse-frequency weights over LeRobot task ids.
    task_balanced_sampling: bool = False
    # Sample VLA-Arena frames with equal aggregate mass per benchmark suite and
    # equal task mass within each suite. This matches the public leaderboard's
    # equal weighting over suite-level cells.
    suite_balanced_sampling: bool = False
    # Immutable CPU-built observation rows for persistent-memory causal replay.
    # These fields are admission-bound and are ignored by non-PSM models.
    persistent_memory_static_replay_cache: str | None = None
    persistent_memory_static_replay_binding: str | None = None
    persistent_memory_replay_batch_size: int = 32
    # Tri-state keeps the historical PSM default while allowing an external
    # source batch to use the ordinary single-frame action-flow objective.
    persistent_sequence_training: bool | None = None
    hetm_sequence_training: bool | None = None
    # HCEA-only source-balanced causal sequence mixture.  L0 retains its
    # geometry/demo supervision; L1 explicitly abstains from unavailable
    # fields while sharing the same parent normalization and action space.
    joint_l0_l1_sequence_training: bool = False
    joint_l1_repo_id: str | None = None
    joint_l0_l1_episode_metadata_path: str | None = None
    joint_l1_episode_offset: int = 3018
    joint_l1_task_offset: int = 60
    # HCEA's L1 observations intentionally omit two optional demonstration
    # leaves present in L0, so the sources require separate immutable replay
    # artifacts plus one admission for the atomic dual-source consumer.
    persistent_memory_joint_l1_static_replay_cache: str | None = None
    persistent_memory_joint_l1_static_replay_binding: str | None = None
    persistent_memory_joint_replay_consumer_admission: str | None = None
    # Isolated frozen-prefix + bounded CPU-prefetch consumer.  Enabling it
    # requires every immutable binding below; disabled configurations reject
    # latent non-null bindings instead of silently falling back.
    persistent_memory_frozen_prefix_prefetch_enabled: bool = False
    persistent_memory_frozen_prefix_artifact: str | None = None
    persistent_memory_frozen_prefix_manifest_sha256: str | None = None
    persistent_memory_frozen_prefix_consumer_checkpoint_step: int | None = None
    persistent_memory_frozen_prefix_dependency_merkle_sha256: str | None = None
    persistent_memory_bounded_prefetch_protocol: str | None = None
    persistent_memory_bounded_prefetch_protocol_sha256: str | None = None
    persistent_memory_static_replay_manifest_sha256: str | None = None
    persistent_memory_cpu_prefetch_queue_capacity: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError('--exp_name must be set')
        return (
            pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name
        ).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    @property
    def use_persistent_sequence_training(self) -> bool:
        model_supports_sequences = bool(
            getattr(self.model, 'persistent_subgoal_memory', False)
        )
        if self.persistent_sequence_training is None:
            return model_supports_sequences
        if self.persistent_sequence_training and not model_supports_sequences:
            raise ValueError(
                'persistent_sequence_training requires a persistent-memory model'
            )
        return bool(self.persistent_sequence_training)

    @property
    def use_hetm_sequence_training(self) -> bool:
        supported = bool(
            getattr(self.model, 'hierarchical_event_transition_memory', False)
        )
        enabled = supported if self.hetm_sequence_training is None else bool(
            self.hetm_sequence_training
        )
        if enabled and not supported:
            raise ValueError('hetm_sequence_training requires a HETM model')
        return enabled

    @property
    def use_joint_psm_hetm_sequence_training(self) -> bool:
        return bool(
            self.use_persistent_sequence_training
            and self.use_hetm_sequence_training
        )

    @property
    def optimizer_update_multipliers(self) -> tuple[tuple[str, float], ...]:
        singular = (
            (
                self.architecture_update_path,
                self.architecture_update_multiplier,
            ),
        ) if self.architecture_update_path is not None else ()
        return singular + self.architecture_update_multipliers

    @property
    def uses_auxiliary_data(self) -> bool:
        return self.auxiliary_data is not None

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError('Cannot resume and overwrite at the same time.')
        if self.architecture_update_multiplier <= 0:
            raise ValueError(
                'architecture_update_multiplier must be positive'
            )
        if (
            self.architecture_update_path is None
            and self.architecture_update_multiplier != 1.0
        ):
            raise ValueError(
                'architecture_update_path is required for a non-unit multiplier'
            )
        normalized_paths = tuple(
            str(path) for path, _ in self.optimizer_update_multipliers
        )
        if any(not path for path in normalized_paths):
            raise ValueError(
                'architecture update multipliers require non-empty paths'
            )
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError('architecture update multiplier paths must be unique')
        if any(
            multiplier <= 0
            for _, multiplier in self.optimizer_update_multipliers
        ):
            raise ValueError(
                'architecture update multipliers must be positive'
            )
        _ = self.use_persistent_sequence_training
        _ = self.use_hetm_sequence_training
        _ = self.use_joint_psm_hetm_sequence_training
        if self.joint_l0_l1_sequence_training:
            if not (
                self.use_joint_psm_hetm_sequence_training
                and getattr(
                    self.model,
                    'persistent_hierarchical_clause_event_alignment_v1',
                    False,
                )
                and self.joint_l1_repo_id
                and self.joint_l0_l1_episode_metadata_path
                and self.joint_l1_episode_offset > 0
                and self.joint_l1_task_offset > 0
                and self.persistent_memory_static_replay_cache
                and self.persistent_memory_static_replay_binding
                and self.persistent_memory_joint_l1_static_replay_cache
                and self.persistent_memory_joint_l1_static_replay_binding
                and self.persistent_memory_joint_replay_consumer_admission
                and self.batch_size % 4 == 0
            ):
                raise ValueError(
                    'joint L0/L1 sequence training requires HCEA, PSM+HETM, '
                    'both sealed sources and replay bindings, positive identity '
                    'offsets, and a batch divisible by four'
                )
        elif any(
            value is not None
            for value in (
                self.joint_l1_repo_id,
                self.joint_l0_l1_episode_metadata_path,
                self.persistent_memory_joint_l1_static_replay_cache,
                self.persistent_memory_joint_l1_static_replay_binding,
                self.persistent_memory_joint_replay_consumer_admission,
            )
        ):
            raise ValueError('joint L0/L1 bindings require the joint training flag')
        if self.auxiliary_gradient_merge not in (
            'convex',
            'target_preserving_pcgrad',
            'clip_aware_target_preserving_pcgrad',
        ):
            raise ValueError('unsupported auxiliary_gradient_merge')
        if self.auxiliary_target_gradient_clip_norm <= 0:
            raise ValueError('auxiliary target gradient clip norm must be positive')
        if self.auxiliary_gradient_merge == 'clip_aware_target_preserving_pcgrad':
            optimizer_clip = getattr(self.optimizer, 'clip_gradient_norm', None)
            required_clip = self.auxiliary_target_gradient_clip_norm * (
                1.0 + self.auxiliary_loss_weight
            )
            if optimizer_clip is None or optimizer_clip + 1.0e-8 < required_clip:
                raise ValueError(
                    'clip-aware PCGrad optimizer clip must admit the protected '
                    'target plus the weighted capped auxiliary gradient'
                )
        if self.auxiliary_data is None:
            if (
                self.auxiliary_batch_size != 0
                or self.auxiliary_loss_weight != 0.0
                or self.auxiliary_num_workers != 0
                or self.auxiliary_task_balanced_sampling
                or self.auxiliary_gradient_path_allowlist
            ):
                raise ValueError(
                    'auxiliary training options require auxiliary_data'
                )
        else:
            if self.auxiliary_batch_size < 1:
                raise ValueError('auxiliary_batch_size must be positive')
            if not 0.0 < self.auxiliary_loss_weight < 1.0:
                raise ValueError('auxiliary_loss_weight must be in (0, 1)')
            if self.auxiliary_num_workers < 0:
                raise ValueError('auxiliary_num_workers must be non-negative')
            if not self.auxiliary_gradient_path_allowlist:
                raise ValueError(
                    'auxiliary gradients require a non-empty path allowlist'
                )
            if any(not path for path in self.auxiliary_gradient_path_allowlist):
                raise ValueError('auxiliary gradient paths must be non-empty')
            if self.gradient_accumulation_steps != 1:
                raise ValueError(
                    'auxiliary gradient mixing requires one target batch per update'
                )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name='pi0_aloha',
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id='trossen'),
        ),
        policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name='pi05_aloha',
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id='trossen'),
        ),
        policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name='pi0_aloha_towel',
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id='trossen'),
            default_prompt='fold the towel',
        ),
        policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name='pi0_aloha_tupperware',
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id='trossen'),
            default_prompt='open the tupperware and put the food on the plate',
        ),
        policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name='pi0_droid',
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id='droid'),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name='pi0_fast_droid',
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id='droid'),
            data_transforms=lambda model: _transforms.Group(
                inputs=[
                    droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)
                ],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name='pi05_droid',
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id='droid'),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name='pi0_libero',
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id='physical-intelligence/libero',
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi0_base/params'
        ),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name='pi0_vla_arena',
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi0_base/params',
            )
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name='pi0_libero_low_mem_finetune',
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='physical-intelligence/libero',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi0_base/params'
        ),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    # vla-arena low memory finetune for pi0
    TrainConfig(
        name='pi0_vla_arena_low_mem_finetune',
        model=pi0_config.Pi0Config(
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        # Set OPENPI_VLA_ARENA_CHECKPOINT_PATH environment variable to specify a custom checkpoint path.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi0_base/params',
            )
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name='pi0_fast_vla_arena',
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi0_fast_base/params',
            )
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name='pi0_fast_libero_low_mem_finetune',
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant='gemma_2b_lora',
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='physical-intelligence/libero',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi0_fast_base/params'
        ),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant='gemma_2b_lora',
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    # vla-arena low memory finetune for pi0-fast
    TrainConfig(
        name='pi0_fast_vla_arena_low_mem_finetune',
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant='gemma_2b_lora',
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Set OPENPI_VLA_ARENA_CHECKPOINT_PATH environment variable to specify a custom checkpoint path.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi0_fast_base/params',
            )
        ),
        num_train_steps=60_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant='gemma_2b_lora',
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name='pi05_libero',
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='physical-intelligence/libero',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi05_base/params'
        ),
        pytorch_weight_path='/path/to/your/pytorch_weight_path',
        num_train_steps=30_000,
    ),
    TrainConfig(
        name='pi05_vla_arena',
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi05_base/params'
        ),
        pytorch_weight_path='/path/to/your/pytorch_weight_path',
        num_train_steps=30_000,
    ),
    TrainConfig(
        name='pi05_vla_arena_low_mem_finetune',
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi05_base/params',
            )
        ),
        num_train_steps=60_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name='pi05_vla_arena_state_adarms',
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            state_adarms=True,
            state_adarms_hidden_dim=256,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi05_base/params',
            ),
            missing_regex='.*state_adarms.*',
        ),
        num_train_steps=10_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            state_adarms=True,
            state_adarms_hidden_dim=256,
        ).get_state_adarms_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name='pi05_vla_arena_reasoning_lora',
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                'gs://openpi-assets/checkpoints/pi05_base/params',
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        num_train_steps=12_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
        ).get_state_film_lora_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_contextual_dual_lora',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=2,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                    'gs://openpi-assets/checkpoints/pi05_base/params',
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
        ).get_state_film_lora_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_dual_action_reasoner',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE3_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                    os.getenv(
                        'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                        'gs://openpi-assets/checkpoints/pi05_base/params',
                    ),
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
        ).get_state_film_lora_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_persistent_subgoal_memory',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            # A single unbiased parent-flow draw per supervised replan avoids
            # rematerializing FSDP collectives inside the multi-sample map.
            # Batch 32 and the 50/50 4/8 schedule still provide about 2.88M
            # parent-flow examples over 30k optimizer steps.
            main_flow_samples=1,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=2,
            explicit_action_reasoner_inference_steps=4,
            persistent_subgoal_memory=True,
            persistent_memory_tokens=8,
            persistent_memory_hidden_dim=256,
            persistent_memory_subgoal_slots=8,
            persistent_memory_fast_tokens=4,
            persistent_memory_fast_update_rate=0.50,
            persistent_memory_slow_update_rate=0.05,
            persistent_memory_previous_action_steps=5,
            persistent_memory_short_replans=4,
            persistent_memory_long_replans=8,
            persistent_memory_long_probability=0.50,
            persistent_memory_cache_refresh_steps=1000,
            persistent_memory_max_staleness_steps=1000,
            persistent_memory_policy_gain=0.0,
            persistent_memory_policy_gain_warmup_steps=3000,
            persistent_memory_factor_attention_alignment_loss_weight=0.01,
            persistent_memory_object_slot_reconstruction_loss_weight=0.01,
            supervised_role_identity_contrastive_loss=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            assets=AssetsConfig(
                asset_id='VLA_Arena_L0_L_lerobot_openpi/VLA_Arena'
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CompoundCheckpointWeightLoader(
            primary_params_path=os.getenv(
                'OPENPI_VLA_ARENA_PSM_PRIMARY_PARAMS',
                '/path/to/workspace/VLA-Arena/experiments/'
                'pi05/checkpoints/pi05_vla_arena_contextual_dual_lora/'
                'contextual_dual_lora_full_seed7/29999/params',
            ),
            secondary_params_path=os.getenv(
                'OPENPI_VLA_ARENA_PSM_SECONDARY_PARAMS',
                '/path/to/workspace/VLA-Arena/experiments/'
                'pi05/checkpoints/pi05_vla_arena_dual_action_reasoner/'
                'dual_action_reasoner_from_stage2_seed7/29999/params',
            ),
            secondary_only_regex='action_prior_(explicit|implicit|guidance)_.*',
            missing_regex='persistent_memory.*',
            expected_primary_arrays=90,
            expected_secondary_arrays=227,
            expected_secondary_only_arrays=137,
        ),
        num_train_steps=30_000,
        batch_size=32,
        architecture_update_path='persistent_memory',
        architecture_update_multiplier=5.0,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_action_film=True,
            action_prior=True,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=2,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            explicit_action_reasoner_flow_samples=2,
            persistent_subgoal_memory=True,
            supervised_role_identity_contrastive_loss=True,
        ).get_persistent_memory_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_retrieved_demo_reasoner',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE4_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_STAGE3_CHECKPOINT_PATH',
                    os.getenv(
                        'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                        os.getenv(
                            'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                            'gs://openpi-assets/checkpoints/pi05_base/params',
                        ),
                    ),
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
        ).get_state_film_lora_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_discrete_chunk_reasoner',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE5_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_STAGE4_CHECKPOINT_PATH',
                    os.getenv(
                        'OPENPI_VLA_ARENA_STAGE3_CHECKPOINT_PATH',
                        os.getenv(
                            'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                            os.getenv(
                                'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                                'gs://openpi-assets/checkpoints/pi05_base/params',
                            ),
                        ),
                    ),
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_state_film_lora_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_attention_reasoner',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_STAGE5_CHECKPOINT_PATH',
                    os.getenv(
                        'OPENPI_VLA_ARENA_STAGE4_CHECKPOINT_PATH',
                        os.getenv(
                            'OPENPI_VLA_ARENA_STAGE3_CHECKPOINT_PATH',
                            os.getenv(
                                'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                                os.getenv(
                                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                                    'gs://openpi-assets/checkpoints/pi05_base/params',
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3e-6,
            decay_steps=30_000,
            decay_lr=3e-7,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_attention_adarms_full_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_vision_attention_reasoner',
        suite_balanced_sampling=True,
        architecture_update_path='spatial_relation',
        architecture_update_multiplier=20.0,
        model=pi0_config.Pi0Config(
            spatial_relation_reasoner=True,
            spatial_relation_hidden_dim=256,
            spatial_relation_queries=8,
            spatial_relation_layers=2,
            spatial_relation_num_heads=8,
            spatial_relation_mlp_dim=1024,
            spatial_relation_max_cameras=3,
            spatial_relation_max_grid_size=16,
            spatial_relation_loss_weight=0.05,
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE7_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                    os.getenv(
                        'OPENPI_VLA_ARENA_STAGE5_CHECKPOINT_PATH',
                        os.getenv(
                            'OPENPI_VLA_ARENA_STAGE4_CHECKPOINT_PATH',
                            os.getenv(
                                'OPENPI_VLA_ARENA_STAGE3_CHECKPOINT_PATH',
                                os.getenv(
                                    'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                                    os.getenv(
                                        'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                                        'gs://openpi-assets/checkpoints/pi05_base/params',
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|spatial_relation|lora).*'
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-6,
            decay_steps=30_000,
            decay_lr=1e-7,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            spatial_relation_reasoner=True,
            spatial_relation_hidden_dim=256,
            spatial_relation_queries=8,
            spatial_relation_layers=2,
            spatial_relation_num_heads=8,
            spatial_relation_mlp_dim=1024,
            spatial_relation_max_cameras=3,
            spatial_relation_max_grid_size=16,
            spatial_relation_loss_weight=0.05,
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_spatial_attention_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_action_mlp_reasoner',
        suite_balanced_sampling=True,
        architecture_update_path='contact_phase',
        architecture_update_multiplier=40.0,
        model=pi0_config.Pi0Config(
            contact_phase_reasoner=True,
            contact_phase_hidden_dim=256,
            contact_phase_layers=2,
            contact_phase_num_heads=8,
            contact_phase_mlp_dim=1024,
            contact_phase_temperature=0.5,
            contact_phase_loss_weight=0.05,
            contact_phase_gripper_index=6,
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE8_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_STAGE7_CHECKPOINT_PATH',
                    os.getenv(
                        'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                        os.getenv(
                            'OPENPI_VLA_ARENA_STAGE5_CHECKPOINT_PATH',
                            os.getenv(
                                'OPENPI_VLA_ARENA_STAGE4_CHECKPOINT_PATH',
                                os.getenv(
                                    'OPENPI_VLA_ARENA_STAGE3_CHECKPOINT_PATH',
                                    os.getenv(
                                        'OPENPI_VLA_ARENA_STAGE2_CHECKPOINT_PATH',
                                        os.getenv(
                                            'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                                            'gs://openpi-assets/checkpoints/pi05_base/params',
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|contact_phase|lora).*'
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-7,
            decay_steps=30_000,
            decay_lr=5e-8,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            contact_phase_reasoner=True,
            contact_phase_hidden_dim=256,
            contact_phase_layers=2,
            contact_phase_num_heads=8,
            contact_phase_mlp_dim=1024,
            contact_phase_temperature=0.5,
            contact_phase_loss_weight=0.05,
            contact_phase_gripper_index=6,
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_action_mlp_full_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_compositional_demo_router',
        suite_balanced_sampling=True,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            compositional_demo_routing=True,
            compositional_demo_slots=3,
            compositional_demo_router_layers=2,
            compositional_demo_router_temperature=0.5,
            compositional_demo_router_loss_weight=0.1,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.25,
            compositional_demonstration_slots=3,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                    'gs://openpi-assets/checkpoints/pi05_base/params',
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            compositional_demo_routing=True,
            compositional_demo_slots=3,
            compositional_demo_router_layers=2,
            compositional_demo_router_temperature=0.5,
            compositional_demo_router_loss_weight=0.1,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_state_film_lora_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_reasoning_pathway_router',
        suite_balanced_sampling=True,
        architecture_update_path='action_prior_pathway',
        architecture_update_multiplier=5.0,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            reasoning_pathway_router=True,
            reasoning_pathway_router_hidden_dim=256,
            reasoning_pathway_router_temperature=1.0,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                    'gs://openpi-assets/checkpoints/pi05_base/params',
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            reasoning_pathway_router=True,
            reasoning_pathway_router_hidden_dim=256,
            reasoning_pathway_router_temperature=1.0,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_reasoning_pathway_router_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_reasoning_pathway_interaction',
        suite_balanced_sampling=True,
        architecture_update_path='action_prior_pathway_interaction',
        architecture_update_multiplier=5.0,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            reasoning_pathway_interaction=True,
            reasoning_pathway_interaction_hidden_dim=256,
            reasoning_pathway_interaction_layers=2,
            reasoning_pathway_interaction_num_heads=4,
            reasoning_pathway_interaction_mlp_dim=512,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                    'gs://openpi-assets/checkpoints/pi05_base/params',
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            reasoning_pathway_interaction=True,
            reasoning_pathway_interaction_hidden_dim=256,
            reasoning_pathway_interaction_layers=2,
            reasoning_pathway_interaction_num_heads=4,
            reasoning_pathway_interaction_mlp_dim=512,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_reasoning_pathway_interaction_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_structured_rationale_reasoner',
        suite_balanced_sampling=True,
        architecture_update_path='action_prior_rationale',
        architecture_update_multiplier=5.0,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            structured_rationale_reasoner=True,
            structured_rationale_hidden_dim=256,
            structured_rationale_layers=2,
            structured_rationale_num_heads=8,
            structured_rationale_mlp_dim=1024,
            structured_rationale_temperature=0.7,
            structured_rationale_loss_weight=0.1,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
            structured_demonstration_rationale=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                    'gs://openpi-assets/checkpoints/pi05_base/params',
                ),
            ),
            missing_regex='.*(state_adarms|state_film|action_prior|lora).*',
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            structured_rationale_reasoner=True,
            structured_rationale_hidden_dim=256,
            structured_rationale_layers=2,
            structured_rationale_num_heads=8,
            structured_rationale_mlp_dim=1024,
            structured_rationale_temperature=0.7,
            structured_rationale_loss_weight=0.1,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_structured_rationale_freeze_filter(),
        ema_decay=0.999,
    ),
    TrainConfig(
        name='pi05_vla_arena_action_chunk_verifier',
        suite_balanced_sampling=True,
        architecture_update_path='action_chunk_verifier',
        architecture_update_multiplier=5.0,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            action_chunk_verifier=True,
            action_chunk_verifier_hidden_dim=256,
            action_chunk_verifier_layers=2,
            action_chunk_verifier_num_heads=8,
            action_chunk_verifier_mlp_dim=1024,
            action_chunk_verifier_temperature=0.5,
            action_chunk_verifier_loss_weight=0.1,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id='VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            demonstration_bank_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/demonstrations/vla_arena_demo_bank.npz'
            ),
            demonstration_dropout=0.5,
            demonstration_mismatch_rate=0.25,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_STAGE6_CHECKPOINT_PATH',
                os.getenv(
                    'OPENPI_VLA_ARENA_CHECKPOINT_PATH',
                    'gs://openpi-assets/checkpoints/pi05_base/params',
                ),
            ),
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|'
                'action_chunk_verifier|lora).*'
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            active_action_dim=7,
            discrete_state_input=False,
            paligemma_variant='gemma_2b_lora',
            action_expert_variant='gemma_300m_lora',
            state_adarms=True,
            state_adarms_hidden_dim=256,
            state_action_film=True,
            state_action_film_hidden_dim=256,
            action_prior=True,
            action_prior_hidden_dim=256,
            action_prior_horizon=5,
            action_prior_loss_weight=0.1,
            action_prior_contextual=True,
            action_prior_state_conditioning=True,
            action_prior_target='endpoint',
            main_flow_samples=8,
            dual_action_reasoner=True,
            implicit_action_reasoner_layers=tuple(range(18)),
            implicit_action_reasoner_layerwise_guidance=True,
            implicit_action_reasoner_pool_stride=1,
            explicit_action_reasoner_hidden_dim=256,
            explicit_action_reasoner_layers=2,
            explicit_action_reasoner_num_heads=4,
            explicit_action_reasoner_mlp_dim=512,
            explicit_action_reasoner_loss_weight=0.1,
            explicit_action_reasoner_flow_samples=8,
            explicit_action_reasoner_inference_steps=4,
            explicit_action_reasoner_teacher_forcing=False,
            retrieved_demo_conditioning=True,
            retrieved_demo_hidden_dim=256,
            retrieved_demo_layers=2,
            retrieved_demo_num_heads=4,
            retrieved_demo_mlp_dim=512,
            retrieved_demo_plan_steps=10,
            retrieved_demo_plan_dim=17,
            retrieved_demo_reliability_loss_weight=0.05,
            action_chunk_verifier=True,
            action_chunk_verifier_hidden_dim=256,
            action_chunk_verifier_layers=2,
            action_chunk_verifier_num_heads=8,
            action_chunk_verifier_mlp_dim=1024,
            action_chunk_verifier_temperature=0.5,
            action_chunk_verifier_loss_weight=0.1,
            discrete_action_codebook_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/action_codebooks/'
                'vla_arena_action_hierarchical_k256.npz'
            ),
            discrete_action_codebook_hidden_dim=512,
            discrete_action_codebook_loss_weight=0.1,
            discrete_action_auxiliary_loss=False,
            discrete_action_codebook_temperature=0.5,
            discrete_action_codebook_robot_dim=7,
        ).get_action_chunk_verifier_freeze_filter(),
        ema_decay=0.999,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name='pi0_aloha_pen_uncap',
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id='physical-intelligence/aloha_pen_uncap_diverse',
            assets=AssetsConfig(
                assets_dir='gs://openpi-assets/checkpoints/pi0_base/assets',
                asset_id='trossen',
            ),
            default_prompt='uncap the pen',
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            'images': {
                                'cam_high': 'observation.images.cam_high',
                                'cam_left_wrist': 'observation.images.cam_left_wrist',
                                'cam_right_wrist': 'observation.images.cam_right_wrist',
                            },
                            'state': 'observation.state',
                            'actions': 'action',
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi0_base/params'
        ),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name='pi05_aloha_pen_uncap',
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id='physical-intelligence/aloha_pen_uncap_diverse',
            assets=AssetsConfig(
                assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets',
                asset_id='trossen',
            ),
            default_prompt='uncap the pen',
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            'images': {
                                'cam_high': 'observation.images.cam_high',
                                'cam_left_wrist': 'observation.images.cam_left_wrist',
                                'cam_right_wrist': 'observation.images.cam_right_wrist',
                            },
                            'state': 'observation.state',
                            'actions': 'action',
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi05_base/params'
        ),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name='pi0_fast_full_droid_finetune',
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id='droid',
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir='<path_to_droid_rlds_dataset>',
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi0_fast_base/params'
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name='pi05_full_droid_finetune',
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id='droid',
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            # Set OPENPI_DROID_RLDS_DATA_DIR environment variable to specify a custom dataset path.
            rlds_data_dir=os.getenv('OPENPI_DROID_RLDS_DATA_DIR', ''),
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets/',
                asset_id='droid',
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi05_base/params'
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name='pi05_droid_finetune',
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id='your_hf_username/my_droid_dataset',
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir='gs://openpi-assets/checkpoints/pi05_droid/assets',
                asset_id='droid',
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi05_droid/params'
        ),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name='pi0_aloha_sim',
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id='lerobot/aloha_sim_transfer_cube_human',
            default_prompt='Transfer cube',
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            'gs://openpi-assets/checkpoints/pi0_base/params'
        ),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name='debug',
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(
            paligemma_variant='dummy', action_expert_variant='dummy'
        ),
        save_interval=100,
        overwrite=True,
        exp_name='debug',
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name='debug_restore',
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(
            paligemma_variant='dummy', action_expert_variant='dummy'
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            './checkpoints/debug/debug/9/params'
        ),
        overwrite=True,
        exp_name='debug',
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name='debug_pi05',
        model=pi0_config.Pi0Config(
            pi05=True, paligemma_variant='dummy', action_expert_variant='dummy'
        ),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name='debug_pi05',
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

# Stage 15 is an independent Stage-6 branch. Deriving it from the explicit
# Stage-14 entry keeps every proven Stage-6 reasoning setting identical while
# replacing the verifier-only head with future-visual dynamics supervision.
_stage15_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage15_model = dataclasses.replace(
    _stage15_source_config.model,
    action_chunk_verifier=False,
    latent_future_reasoner=True,
    latent_future_hidden_dim=256,
    latent_future_layers=2,
    latent_future_num_heads=8,
    latent_future_mlp_dim=1024,
    latent_future_grid_size=4,
    latent_future_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage15_source_config,
        name='pi05_vla_arena_latent_future_reasoner',
        architecture_update_path='latent_future',
        architecture_update_multiplier=5.0,
        model=_stage15_model,
        data=dataclasses.replace(
            _stage15_source_config.data,
            future_visual_supervision=True,
        ),
        weight_loader=dataclasses.replace(
            _stage15_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|latent_future|lora).*'
            ),
        ),
        freeze_filter=_stage15_model.get_latent_future_reasoner_freeze_filter(),
        # Two additional target-camera encodes increase forward memory. Keep
        # the formal effective batch at 32 while using microbatches of 16.
        gradient_accumulation_steps=2,
    )
)
del _stage15_model, _stage15_source_config

# Stage 16 is another independent Stage-6 branch.  It predicts the normalized
# proprioceptive state reached after every action in the ten-step chunk and
# injects the learned rollout representation through a zero residual head.
_stage16_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage16_model = dataclasses.replace(
    _stage16_source_config.model,
    action_chunk_verifier=False,
    state_rollout_reasoner=True,
    state_rollout_hidden_dim=256,
    state_rollout_layers=2,
    state_rollout_num_heads=8,
    state_rollout_mlp_dim=1024,
    state_rollout_target_dim=8,
    state_rollout_loss_weight=0.1,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage16_source_config,
        name='pi05_vla_arena_state_rollout_reasoner',
        architecture_update_path='state_rollout',
        architecture_update_multiplier=5.0,
        model=_stage16_model,
        data=dataclasses.replace(
            _stage16_source_config.data,
            future_state_supervision=True,
        ),
        weight_loader=dataclasses.replace(
            _stage16_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|state_rollout|lora).*'
            ),
        ),
        freeze_filter=_stage16_model.get_state_rollout_reasoner_freeze_filter(),
    )
)
del _stage16_model, _stage16_source_config

# Stage 17 is an independent Stage-6 branch.  It adds a sparse top-2 bank of
# action-denoising experts whose router is conditioned by language, state, and
# action-step features.  A clean-action auxiliary head trains the experts from
# the first update while the zero action-facing head preserves Stage 6 at init.
_stage17_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage17_model = dataclasses.replace(
    _stage17_source_config.model,
    action_chunk_verifier=False,
    action_moe_reasoner=True,
    action_moe_hidden_dim=256,
    action_moe_layers=2,
    action_moe_num_heads=8,
    action_moe_mlp_dim=1024,
    action_moe_num_experts=8,
    action_moe_top_k=2,
    action_moe_expert_dim=512,
    action_moe_temperature=1.0,
    action_moe_prediction_loss_weight=0.05,
    action_moe_balance_loss_weight=0.01,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage17_source_config,
        name='pi05_vla_arena_action_moe_reasoner',
        architecture_update_path='action_moe',
        architecture_update_multiplier=5.0,
        model=_stage17_model,
        weight_loader=dataclasses.replace(
            _stage17_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|action_moe|lora).*'
            ),
        ),
        freeze_filter=_stage17_model.get_action_moe_reasoner_freeze_filter(),
    )
)
del _stage17_model, _stage17_source_config

# Stage 18 independently learns a latent whole-episode progress state from the
# current multimodal context. Exact dataset frame progress supervises training,
# while inference uses only the predicted progress token.
_stage18_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage18_model = dataclasses.replace(
    _stage18_source_config.model,
    action_chunk_verifier=False,
    task_progress_reasoner=True,
    task_progress_hidden_dim=256,
    task_progress_layers=2,
    task_progress_num_heads=8,
    task_progress_mlp_dim=1024,
    task_progress_bins=10,
    task_progress_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage18_source_config,
        name='pi05_vla_arena_task_progress_reasoner',
        architecture_update_path='task_progress',
        architecture_update_multiplier=5.0,
        model=_stage18_model,
        data=dataclasses.replace(
            _stage18_source_config.data,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L0_L_lerobot_openpi/meta/episodes.jsonl'
            ),
        ),
        weight_loader=dataclasses.replace(
            _stage18_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|task_progress|lora).*'
            ),
        ),
        freeze_filter=(
            _stage18_model.get_task_progress_reasoner_freeze_filter()
        ),
    )
)
del _stage18_model, _stage18_source_config

# Candidate Stage 19 is an independent Stage-6 branch that learns a compact
# predictive world model from complementary training-only targets and a sparse
# action-expert pathway.  The
# policy never receives future observations at inference: learned visual
# dynamics, proprioceptive rollout, and whole-episode progress tokens are
# combined by a context/state-conditioned per-action gate.
_stage19_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage19_model = dataclasses.replace(
    _stage19_source_config.model,
    action_chunk_verifier=False,
    latent_future_reasoner=True,
    latent_future_hidden_dim=256,
    latent_future_layers=2,
    latent_future_num_heads=8,
    latent_future_mlp_dim=1024,
    latent_future_grid_size=4,
    latent_future_loss_weight=0.05,
    state_rollout_reasoner=True,
    state_rollout_hidden_dim=256,
    state_rollout_layers=2,
    state_rollout_num_heads=8,
    state_rollout_mlp_dim=1024,
    state_rollout_target_dim=8,
    state_rollout_loss_weight=0.1,
    action_moe_reasoner=True,
    action_moe_hidden_dim=256,
    action_moe_layers=2,
    action_moe_num_heads=8,
    action_moe_mlp_dim=1024,
    action_moe_num_experts=8,
    action_moe_top_k=2,
    action_moe_expert_dim=512,
    action_moe_temperature=1.0,
    action_moe_prediction_loss_weight=0.05,
    action_moe_balance_loss_weight=0.01,
    task_progress_reasoner=True,
    task_progress_hidden_dim=256,
    task_progress_layers=2,
    task_progress_num_heads=8,
    task_progress_mlp_dim=1024,
    task_progress_bins=10,
    task_progress_loss_weight=0.05,
    predictive_world_model_fusion=True,
    predictive_world_model_hidden_dim=256,
    predictive_world_model_auxiliary_scale=1.0 / 4.0,
    predictive_world_model_include_action_moe=True,
    predictive_world_model_reliability_loss_weight=0.02,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage19_source_config,
        name='pi05_vla_arena_predictive_world_model',
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('latent_future', 5.0),
            ('state_rollout', 5.0),
            ('action_moe', 5.0),
            ('task_progress', 5.0),
            ('predictive_world_model', 5.0),
        ),
        model=_stage19_model,
        data=dataclasses.replace(
            _stage19_source_config.data,
            future_visual_supervision=True,
            future_state_supervision=True,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L0_L_lerobot_openpi/meta/episodes.jsonl'
            ),
        ),
        weight_loader=dataclasses.replace(
            _stage19_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|latent_future|'
                'state_rollout|action_moe|task_progress|predictive_world_model|lora).*'
            ),
        ),
        freeze_filter=(
            _stage19_model.get_predictive_world_model_freeze_filter()
        ),
        # Two future-image encodes plus four reasoners require a conservative
        # microbatch of eight while retaining the formal effective batch 32.
        # The current JAX trainer consumes one complete global batch per
        # optimizer update; gradient_accumulation_steps is compatibility
        # metadata and is not an execution transform.  Keep it truthful.
        gradient_accumulation_steps=1,
    )
)
_evidence_combination_source_config = _CONFIGS[-1]
_evidence_combination_model = dataclasses.replace(
    _evidence_combination_source_config.model,
    spatial_relation_reasoner=True,
    contact_phase_reasoner=True,
    reasoning_pathway_interaction=True,
    reasoning_pathway_router=True,
    evidence_combination_router=True,
    evidence_combination_hierarchical_components=True,
    object_affordance_graph_reasoner=True,
    language_subgoal_reasoner=True,
    object_subgoal_binding=True,
    velocity_refiner=True,
    # Re-read the complete contextualized image/instruction prefix at every
    # flow-matching step.  Its exactly-zero gain keeps the selected Stage-19
    # trajectory bitwise unchanged before optimization.
    action_visual_refiner=True,
    action_chunk_verifier=True,
    evidence_combination_action_verifier_component=True,
    evidence_combination_object_subgoal_binding_component=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _evidence_combination_source_config,
        name='pi05_vla_arena_evidence_combination',
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('latent_future', 1.0),
            ('state_rollout', 1.0),
            ('action_moe', 1.0),
            ('task_progress', 1.0),
            ('predictive_world_model', 1.0),
            ('action_prior_pathway_token_in', 5.0),
            ('action_prior_pathway_context_in', 5.0),
            ('action_prior_pathway_state_in', 5.0),
            ('action_prior_pathway_embedding', 5.0),
            ('action_prior_pathway_score', 5.0),
            ('action_prior_pathway_interaction', 5.0),
            ('spatial_relation', 1.0),
            ('contact_phase', 1.0),
            ('object_affordance', 5.0),
            ('language_subgoal', 5.0),
            ('object_subgoal_binding', 5.0),
            ('velocity_refiner', 5.0),
            ('action_visual_refiner', 5.0),
            ('action_chunk_verifier', 5.0),
            ('evidence_combination', 5.0),
        ),
        model=_evidence_combination_model,
        weight_loader=dataclasses.replace(
            _evidence_combination_source_config.weight_loader,
            params_path=os.getenv(
                'OPENPI_VLA_ARENA_STAGE19_CHECKPOINT_PATH',
                _evidence_combination_source_config.weight_loader.params_path,
            ),
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|latent_future|'
                'state_rollout|action_moe|task_progress|predictive_world_model|'
                'spatial_relation|contact_phase|object_affordance|'
                'language_subgoal|object_subgoal_binding|velocity_refiner|'
                'action_visual_refiner|action_chunk_verifier|'
                'evidence_combination|lora).*'
            ),
        ),
        freeze_filter=(
            _evidence_combination_model.get_evidence_combination_freeze_filter()
        ),
    )
)

# Candidate Stage 21 independently inherits Stage 6 and adds competitive
# language-conditioned object slots.  It is deliberately not derived from the
# Stage-8 spatial branch: the screen isolates instance binding and slot-graph
# reasoning from broad base-attention tuning.
_stage21_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage21_model = dataclasses.replace(
    _stage21_source_config.model,
    action_chunk_verifier=False,
    object_affordance_graph_reasoner=True,
    object_affordance_hidden_dim=256,
    object_affordance_slots=8,
    object_affordance_layers=2,
    object_affordance_num_heads=8,
    object_affordance_mlp_dim=1024,
    object_affordance_max_cameras=3,
    object_affordance_max_grid_size=16,
    object_affordance_temperature=1.0,
    object_affordance_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage21_source_config,
        name='pi05_vla_arena_object_affordance_graph',
        architecture_update_path='object_affordance',
        architecture_update_multiplier=5.0,
        model=_stage21_model,
        weight_loader=dataclasses.replace(
            _stage21_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|object_affordance|lora).*'
            ),
        ),
        freeze_filter=(
            _stage21_model.get_object_affordance_graph_freeze_filter()
        ),
    )
)

# Candidate Stage 22 independently inherits Stage 6 and represents the
# instruction-conditioned workflow as an ordered latent subgoal automaton.
# Exact episode progress and clean actions are training-only auxiliary targets;
# deployed inference selects the active subgoal from current context and state.
_stage22_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage22_model = dataclasses.replace(
    _stage22_source_config.model,
    action_chunk_verifier=False,
    language_subgoal_reasoner=True,
    language_subgoal_hidden_dim=256,
    language_subgoal_slots=8,
    language_subgoal_layers=2,
    language_subgoal_num_heads=8,
    language_subgoal_mlp_dim=1024,
    language_subgoal_temperature=1.0,
    language_subgoal_progress_loss_weight=0.05,
    language_subgoal_action_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage22_source_config,
        name='pi05_vla_arena_language_subgoal_reasoner',
        architecture_update_path='language_subgoal',
        architecture_update_multiplier=5.0,
        model=_stage22_model,
        data=dataclasses.replace(
            _stage22_source_config.data,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L0_L_lerobot_openpi/meta/episodes.jsonl'
            ),
        ),
        weight_loader=dataclasses.replace(
            _stage22_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|language_subgoal|lora).*'
            ),
        ),
        freeze_filter=(
            _stage22_model.get_language_subgoal_reasoner_freeze_filter()
        ),
    )
)

# Candidate Stage 23 independently inherits Stage 6 and factorizes the seven
# deployed controls into translation, rotation, and gripper streams.  The
# compact streams interact across the complete ten-step horizon before their
# fused zero-residual token conditions the inherited continuous flow.
_stage23_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage23_model = dataclasses.replace(
    _stage23_source_config.model,
    action_chunk_verifier=False,
    kinematic_action_reasoner=True,
    kinematic_action_hidden_dim=256,
    kinematic_action_layers=2,
    kinematic_action_num_heads=8,
    kinematic_action_mlp_dim=1024,
    kinematic_action_prediction_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage23_source_config,
        name='pi05_vla_arena_kinematic_action_reasoner',
        architecture_update_path='kinematic_action',
        architecture_update_multiplier=5.0,
        model=_stage23_model,
        weight_loader=dataclasses.replace(
            _stage23_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|kinematic_action|lora).*'
            ),
        ),
        freeze_filter=(
            _stage23_model.get_kinematic_action_reasoner_freeze_filter()
        ),
    )
)

# Candidate Stage 24 independently inherits Stage 6 and specializes current
# camera patches with masked frozen-feature reconstruction.  Language/state
# scene queries also provide a zero-initialized action residual, while the
# reconstruction decoder is absent from deployed inference compute.
_stage24_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage24_model = dataclasses.replace(
    _stage24_source_config.model,
    action_chunk_verifier=False,
    masked_spatial_reasoner=True,
    masked_spatial_hidden_dim=256,
    masked_spatial_queries=16,
    masked_spatial_layers=2,
    masked_spatial_num_heads=8,
    masked_spatial_mlp_dim=1024,
    masked_spatial_max_cameras=3,
    masked_spatial_max_grid_size=16,
    masked_spatial_mask_ratio=0.5,
    masked_spatial_reconstruction_loss_weight=0.02,
    masked_spatial_action_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage24_source_config,
        name='pi05_vla_arena_masked_spatial_reasoner',
        architecture_update_path='masked_spatial',
        architecture_update_multiplier=5.0,
        model=_stage24_model,
        weight_loader=dataclasses.replace(
            _stage24_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|masked_spatial|lora).*'
            ),
        ),
        freeze_filter=(
            _stage24_model.get_masked_spatial_reasoner_freeze_filter()
        ),
    )
)

# Candidate Stage 25 independently inherits Stage 6.  It compresses the two
# deployed current-camera grids into language/state-conditioned object slots,
# forecasts their future state, and reconstructs frozen per-patch future VLM
# features.  The forecast slots also supply a zero-initialized action residual;
# future RGB remains training-only.
_stage25_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage25_model = dataclasses.replace(
    _stage25_source_config.model,
    action_chunk_verifier=False,
    object_future_reasoner=True,
    object_future_hidden_dim=256,
    object_future_queries=12,
    object_future_layers=2,
    object_future_num_heads=8,
    object_future_mlp_dim=1024,
    object_future_max_grid_size=16,
    object_future_reconstruction_loss_weight=0.02,
    object_future_action_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage25_source_config,
        name='pi05_vla_arena_object_future_reasoner',
        architecture_update_path='object_future',
        architecture_update_multiplier=5.0,
        model=_stage25_model,
        data=dataclasses.replace(
            _stage25_source_config.data,
            future_visual_supervision=True,
        ),
        weight_loader=dataclasses.replace(
            _stage25_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|object_future|lora).*'
            ),
        ),
        freeze_filter=(
            _stage25_model.get_object_future_reasoner_freeze_filter()
        ),
        # Future target encodes match Stage15's memory class.
        gradient_accumulation_steps=2,
    )
)

# Candidate Stage 26 independently inherits Stage 6.  Language-independent
# visual object slots are bound by target/predicate/reference role tokens from
# the instruction.  A multi-positive visual-language objective and coarse
# action supervision train the compositional graph while zero action heads
# preserve the inherited controller exactly at initialization.
_stage26_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage26_model = dataclasses.replace(
    _stage26_source_config.model,
    action_chunk_verifier=False,
    predicate_binding_reasoner=True,
    predicate_binding_hidden_dim=256,
    predicate_binding_object_slots=12,
    predicate_binding_role_slots=3,
    predicate_binding_layers=2,
    predicate_binding_num_heads=8,
    predicate_binding_mlp_dim=1024,
    predicate_binding_max_cameras=3,
    predicate_binding_max_grid_size=16,
    predicate_binding_temperature=0.07,
    predicate_binding_contrastive_loss_weight=0.02,
    predicate_binding_action_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage26_source_config,
        name='pi05_vla_arena_predicate_binding_reasoner',
        architecture_update_path='predicate_binding',
        architecture_update_multiplier=5.0,
        model=_stage26_model,
        weight_loader=dataclasses.replace(
            _stage26_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|predicate_binding|lora).*'
            ),
        ),
        freeze_filter=(
            _stage26_model.get_predicate_binding_reasoner_freeze_filter()
        ),
    )
)

# Candidate Stage 27 independently inherits Stage 6.  A language/state router
# selects two of eight low-rank FiLM experts before the PaliGemma prefix pass,
# adapting every layerwise KV state while exact-zero expert outputs preserve
# the inherited controller before optimization.
_stage27_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage27_model = dataclasses.replace(
    _stage27_source_config.model,
    action_chunk_verifier=False,
    multimodal_prefix_moe=True,
    multimodal_prefix_moe_hidden_dim=256,
    multimodal_prefix_moe_expert_dim=64,
    multimodal_prefix_moe_num_experts=8,
    multimodal_prefix_moe_top_k=2,
    multimodal_prefix_moe_temperature=1.0,
    multimodal_prefix_moe_balance_loss_weight=0.01,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage27_source_config,
        name='pi05_vla_arena_multimodal_prefix_moe',
        architecture_update_path='prefix_moe',
        architecture_update_multiplier=5.0,
        model=_stage27_model,
        weight_loader=dataclasses.replace(
            _stage27_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|prefix_moe|lora).*'
            ),
        ),
        freeze_filter=(
            _stage27_model.get_multimodal_prefix_moe_freeze_filter()
        ),
    )
)

# Candidate Stage 28 independently inherits Stage 6.  The unchanged prefix
# pass builds the full PaliGemma cache, then language/state-routed sparse
# experts apply distinct identity-initialized key/value FiLM residuals at every
# layer before the action expert and layerwise implicit reasoner read it.
_stage28_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage28_model = dataclasses.replace(
    _stage28_source_config.model,
    action_chunk_verifier=False,
    layerwise_kv_moe=True,
    layerwise_kv_moe_hidden_dim=256,
    layerwise_kv_moe_expert_dim=64,
    layerwise_kv_moe_num_experts=8,
    layerwise_kv_moe_top_k=2,
    layerwise_kv_moe_temperature=1.0,
    layerwise_kv_moe_balance_loss_weight=0.01,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage28_source_config,
        name='pi05_vla_arena_layerwise_kv_moe',
        architecture_update_path='kv_moe',
        architecture_update_multiplier=5.0,
        model=_stage28_model,
        weight_loader=dataclasses.replace(
            _stage28_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|kv_moe|lora).*'
            ),
        ),
        freeze_filter=(
            _stage28_model.get_layerwise_kv_moe_freeze_filter()
        ),
    )
)

# Candidate Stage 29 independently inherits Stage 6.  The complete noisy
# action chunk is represented in a lossless orthonormal frequency basis, where
# a compact contextual transformer reasons across low, middle, and high bands
# before an exact inverse transform returns zero-initialized per-step guidance.
_stage29_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage29_model = dataclasses.replace(
    _stage29_source_config.model,
    action_chunk_verifier=False,
    spectral_action_reasoner=True,
    spectral_action_hidden_dim=256,
    spectral_action_layers=2,
    spectral_action_num_heads=8,
    spectral_action_mlp_dim=1024,
    spectral_action_bands=3,
    spectral_action_prediction_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage29_source_config,
        name='pi05_vla_arena_spectral_action_reasoner',
        architecture_update_path='spectral_action',
        architecture_update_multiplier=5.0,
        model=_stage29_model,
        weight_loader=dataclasses.replace(
            _stage29_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|spectral_action|lora).*'
            ),
        ),
        freeze_filter=(
            _stage29_model.get_spectral_action_reasoner_freeze_filter()
        ),
    )
)

# Candidate Stage 30 independently inherits Stage 6.  Contextual multimodal
# action-prior states augment the adaptive RMSNorm condition consumed by every
# action-expert attention and feed-forward layer.  The new final projection is
# exactly zero, preserving all inherited layer behavior before optimization.
_stage30_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage30_model = dataclasses.replace(
    _stage30_source_config.model,
    action_chunk_verifier=False,
    context_adarms=True,
    context_adarms_hidden_dim=256,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage30_source_config,
        name='pi05_vla_arena_context_adarms',
        architecture_update_path='context_adarms',
        architecture_update_multiplier=5.0,
        model=_stage30_model,
        weight_loader=dataclasses.replace(
            _stage30_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|context_adarms|action_prior|lora).*'
            ),
        ),
        freeze_filter=_stage30_model.get_context_adarms_freeze_filter(),
    )
)

# Candidate Stage 31 independently inherits Stage 6.  A compact transformer
# learns the detached residual error left by the inherited flow velocity and
# injects its correction through an exact-zero per-axis gain.
_stage31_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage31_model = dataclasses.replace(
    _stage31_source_config.model,
    action_chunk_verifier=False,
    velocity_refiner=True,
    velocity_refiner_hidden_dim=256,
    velocity_refiner_layers=2,
    velocity_refiner_num_heads=8,
    velocity_refiner_mlp_dim=1024,
    velocity_refiner_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage31_source_config,
        name='pi05_vla_arena_velocity_refiner',
        architecture_update_path='velocity_refiner',
        architecture_update_multiplier=5.0,
        model=_stage31_model,
        weight_loader=dataclasses.replace(
            _stage31_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|velocity_refiner|lora).*'
            ),
        ),
        freeze_filter=_stage31_model.get_velocity_refiner_freeze_filter(),
    )
)

# Candidate Stage 32 independently inherits Stage 6.  Action-conditioned
# queries re-read the complete contextual prefix and predict only the residual
# velocity error behind an exact-zero per-axis deployment gate.
_stage32_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_action_chunk_verifier'
)
_stage32_model = dataclasses.replace(
    _stage32_source_config.model,
    action_chunk_verifier=False,
    action_visual_refiner=True,
    action_visual_refiner_hidden_dim=256,
    action_visual_refiner_layers=2,
    action_visual_refiner_num_heads=8,
    action_visual_refiner_mlp_dim=1024,
    action_visual_refiner_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _stage32_source_config,
        name='pi05_vla_arena_action_visual_refiner',
        architecture_update_path='action_visual_refiner',
        architecture_update_multiplier=5.0,
        model=_stage32_model,
        weight_loader=dataclasses.replace(
            _stage32_source_config.weight_loader,
            missing_regex=(
                '.*(state_adarms|state_film|action_prior|action_visual_refiner|'
                'lora).*'
            ),
        ),
        freeze_filter=_stage32_model.get_action_visual_refiner_freeze_filter(),
    )
)

del (
    _stage19_model,
    _stage19_source_config,
    _evidence_combination_source_config,
    _evidence_combination_model,
    _stage21_source_config,
    _stage21_model,
    _stage22_source_config,
    _stage22_model,
    _stage23_source_config,
    _stage23_model,
    _stage24_source_config,
    _stage24_model,
    _stage25_source_config,
    _stage25_model,
    _stage26_source_config,
    _stage26_model,
    _stage27_source_config,
    _stage27_model,
    _stage28_source_config,
    _stage28_model,
    _stage29_source_config,
    _stage29_model,
    _stage30_source_config,
    _stage30_model,
    _stage31_source_config,
    _stage31_model,
    _stage32_source_config,
    _stage32_model,
)

# Isolated PSM-SDLA successor.  It inherits the complete parent model tree,
# warms from exactly step 29999, and initializes only the two audited new
# namespaces.  Missing data/checkpoint artifacts intentionally fail closed.
_psm_sdla_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_persistent_subgoal_memory'
)
_psm_sdla_model = dataclasses.replace(
    _psm_sdla_source_config.model,
    persistent_structured_demo_language=True,
    structured_demo_hidden_dim=256,
    structured_demo_semantic_dim=2048,
    structured_demo_semantic_slots=8,
    structured_demo_prompt_steps=48,
    structured_demo_plan_steps=10,
    structured_demo_plan_dim=17,
    structured_demo_action_steps=10,
    spatial_language_vocabulary_size=128,
    spatial_language_steps=32,
    spatial_language_bos_token_id=1,
    spatial_language_auxiliary_initial_weight=0.05,
    spatial_language_auxiliary_peak_weight=0.10,
    spatial_language_auxiliary_final_weight=0.02,
    spatial_language_auxiliary_warmup_steps=1_000,
    spatial_language_auxiliary_decay_start_step=15_000,
    spatial_language_auxiliary_total_steps=30_000,
)
_CONFIGS.append(
    dataclasses.replace(
        _psm_sdla_source_config,
        name='pi05_vla_arena_persistent_structured_demo_language',
        model=_psm_sdla_model,
        data=dataclasses.replace(
            _psm_sdla_source_config.data,
            structured_demo_manifest_path=os.getenv(
                'OPENPI_PSM_SDLA_BANK_MANIFEST',
                '/path/to/workspace/VLA-Arena-SDLA-stage/'
                'experiments/pi05/psm_sdla_structured_bank_manifest.json',
            ),
            structured_demo_bank_path=os.getenv(
                'OPENPI_PSM_SDLA_BANK_PATH',
                '/path/to/workspace/VLA-Arena-SDLA-stage/'
                'experiments/pi05/psm_sdla_structured_bank.npz',
            ),
            demonstration_dropout=0.25,
            persistent_memory_exact_alias_artifact_path=os.getenv(
                'OPENPI_PSM_EXACT_ALIAS_ARTIFACT',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/'
                'l0_exact_trajectory_duplicates_candidate_audit.json',
            ),
            persistent_memory_allow_unpromoted_alias_artifact=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_PSM_29999_PARAMS',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/psm_checkpoints/'
                'pi05_vla_arena_persistent_subgoal_memory/'
                'persistent_subgoal_memory_compound_stage3_stage4_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '(?:persistent_memory/structured_demo|spatial_language_aux)/.*'
            ),
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('persistent_memory/structured_demo', 5.0),
            ('spatial_language_aux', 5.0),
        ),
        freeze_filter=_psm_sdla_model.get_persistent_sdla_freeze_filter(),
        resume=False,
        overwrite=False,
    )
)
del _psm_sdla_source_config, _psm_sdla_model


# Apply-ready PSM-SDLA-v3 branch.  It keeps every v2 setting and adds the
# direct conditional memory-policy residual under a separately scaled new
# namespace.  The parent PSM scalar and SDLA-v2 gate are never reset.
_psm_sdla_v3_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_persistent_structured_demo_language'
)
_psm_sdla_v3_model = dataclasses.replace(
    _psm_sdla_v3_source_config.model,
    persistent_conditional_memory_policy_bridge=True,
    persistent_conditional_memory_policy_bridge_rank=128,
)
_CONFIGS.append(
    dataclasses.replace(
        _psm_sdla_v3_source_config,
        name='pi05_vla_arena_persistent_structured_demo_language_conditional_memory_bridge_v3',
        model=_psm_sdla_v3_model,
        weight_loader=weight_loaders.SdlaV3ManifestTransplantWeightLoader(
            primary_params_path=(
                _psm_sdla_v3_source_config.weight_loader.params_path
            ),
            transplant_artifact=os.getenv(
                'OPENPI_PSM_SDLA_V2_TRANSPLANT_ARTIFACT', ''
            ),
            transplant_manifest_sha256=os.getenv(
                'OPENPI_PSM_SDLA_V2_TRANSPLANT_MANIFEST_SHA256', ''
            ),
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('persistent_memory/structured_demo', 5.0),
            ('spatial_language_aux', 5.0),
            ('persistent_memory/conditional_memory_policy_bridge', 5.0),
        ),
        freeze_filter=(
            _psm_sdla_v3_model.get_persistent_sdla_v3_freeze_filter()
        ),
        resume=False,
        overwrite=False,
    )
)
del _psm_sdla_v3_source_config, _psm_sdla_v3_model


# Strict 47-leaf geometry extension of the exact base-v3 config.
_geometry_v1_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_persistent_structured_demo_language_conditional_memory_bridge_v3'
)
_geometry_v1_model = dataclasses.replace(
    _geometry_v1_source_config.model,
    persistent_geometry_aux_v1=True,
    persistent_geometry_aux_bottleneck_dim=128,
)
_CONFIGS.append(
    dataclasses.replace(
        _geometry_v1_source_config,
        name='pi05_vla_arena_persistent_structured_demo_language_conditional_memory_bridge_geometry_v1',
        model=_geometry_v1_model,
        data=dataclasses.replace(
            _geometry_v1_source_config.data,
            structured_demo_manifest_path=os.getenv(
                'OPENPI_PSM_SDLA_BANK_MANIFEST',
                '/path/to/workspace/VLA-Arena-PSM-SDLA-v3-geometry-stage/experiments/pi05/psm_sdla_structured_bank_manifest.json',
            ),
            structured_demo_bank_path=os.getenv(
                'OPENPI_PSM_SDLA_BANK_PATH',
                '/path/to/workspace/VLA-Arena-PSM-SDLA-v3-geometry-stage/experiments/pi05/psm_sdla_structured_bank.npz',
            ),
            geometry_aux_sidecar_manifest_path=os.getenv(
                'OPENPI_L0S_GEOMETRY_SIDECAR_MANIFEST',
                '/path/to/workspace/VLA-Arena/experiments/pi05/l0s_geometry_aux_runtime_sidecar_v1/manifest.json',
            ),
            geometry_aux_sidecar_file_sha256='ef140fcb31fae16d4e4a9bb07fafd1d73858dbf62fefcdedcf8ba3699e9ed98d',
            geometry_aux_sidecar_internal_sha256='a3e0c51f17bc5d28eddce4403e95e62946b558ff272c4ca862e73f0acf07ba91',
        ),
        weight_loader=weight_loaders.GeometrySdlaV3ManifestTransplantWeightLoader(
            primary_params_path=_geometry_v1_source_config.weight_loader.primary_params_path,
            transplant_artifact=_geometry_v1_source_config.weight_loader.transplant_artifact,
            transplant_manifest_sha256=_geometry_v1_source_config.weight_loader.transplant_manifest_sha256,
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('persistent_memory/structured_demo', 5.0),
            ('spatial_language_aux', 5.0),
            ('persistent_memory/conditional_memory_policy_bridge', 5.0),
            ('persistent_memory/geometry_aux_v3', 5.0),
        ),
        freeze_filter=_geometry_v1_model.get_persistent_geometry_v1_freeze_filter(),
        resume=False,
        overwrite=False,
    )
)
del _geometry_v1_source_config, _geometry_v1_model


# Joint-51: exact Geometry-47 parent plus four HMCA-v4 leaves.
_joint51_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_persistent_structured_demo_language_conditional_memory_bridge_geometry_v1'
)
_joint51_model = dataclasses.replace(
    _joint51_source_config.model,
    persistent_hmca_v4=True,
    persistent_hmca_v4_rank=16,
    persistent_hmca_v4_alpha=16.0,
    persistent_hmca_v4_layers=(5, 11, 17),
)
_CONFIGS.append(
    dataclasses.replace(
        _joint51_source_config,
        name='pi05_vla_arena_persistent_structured_demo_language_conditional_memory_bridge_geometry_hmca_v4',
        model=_joint51_model,
        data=dataclasses.replace(
            _joint51_source_config.data,
            structured_demo_manifest_path=os.getenv(
                'OPENPI_PSM_SDLA_BANK_MANIFEST',
                '/path/to/workspace/VLA-Arena-PSM-SDLA-v3-geometry-hmca-v4-stage/experiments/pi05/psm_sdla_structured_bank_manifest.json',
            ),
            structured_demo_bank_path=os.getenv(
                'OPENPI_PSM_SDLA_BANK_PATH',
                '/path/to/workspace/VLA-Arena-PSM-SDLA-v3-geometry-hmca-v4-stage/experiments/pi05/psm_sdla_structured_bank.npz',
            ),
        ),
        weight_loader=weight_loaders.Joint51ManifestTransplantWeightLoader(
            primary_params_path=_joint51_source_config.weight_loader.primary_params_path,
            transplant_artifact=_joint51_source_config.weight_loader.transplant_artifact,
            transplant_manifest_sha256=_joint51_source_config.weight_loader.transplant_manifest_sha256,
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('persistent_memory/structured_demo', 5.0),
            ('spatial_language_aux', 5.0),
            ('persistent_memory/conditional_memory_policy_bridge', 5.0),
            ('persistent_memory/geometry_aux_v3', 5.0),
            ('hierarchical_memory_conditional_adapters', 5.0),
        ),
        freeze_filter=_joint51_model.get_persistent_geometry_hmca_v4_freeze_filter(),
        resume=False,
        overwrite=False,
    )
)

# Joint51 continuation with label-isolated LIBERO action supervision.  Target
# batches retain the complete PSM/SDLA/geometry/HMCA objective; source batches
# use ordinary action flow and can update only shared policy representation
# paths.  This is a strict single-model successor, not a router or ensemble.
_joint51_libero_parent_config = _CONFIGS[-1]
_CONFIGS.append(
    dataclasses.replace(
        _joint51_libero_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_libero_replay'
        ),
        exp_name='joint51_libero_replay_from_joint51_29999_seed7',
        auxiliary_data=LeRobotLiberoDataConfig(
            repo_id='physical-intelligence/libero',
            assets=AssetsConfig(
                assets_dir=(
                    '/path/to/workspace/VLA-Arena/'
                    'experiments/pi05/libero_openpi_source_assets_v1'
                ),
                asset_id='physical-intelligence/libero',
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            factorized_prompt=False,
        ),
        auxiliary_batch_size=8,
        auxiliary_loss_weight=0.25,
        auxiliary_num_workers=4,
        auxiliary_task_balanced_sampling=True,
        auxiliary_gradient_path_allowlist=(
            'lora',
            'state_adarms',
            'state_film',
            'action_prior',
            'action_in_proj',
            'action_out_proj',
            'time_mlp',
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_JOINT51_LIBERO_REPLAY_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Joint51-LIBERO-Replay-v1-stage/'
                'joint51_corrected_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4/'
                'joint51_geometry_hmca_v4_corrected_from_psm_29999_seed7/'
                '29999/params',
            ),
            missing_regex=r'(?!)',
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('persistent_memory/structured_demo', 1.0),
            ('spatial_language_aux', 1.0),
            ('persistent_memory/conditional_memory_policy_bridge', 1.0),
            ('persistent_memory/geometry_aux_v3', 1.0),
            ('hierarchical_memory_conditional_adapters', 1.0),
        ),
        persistent_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        gradient_accumulation_steps=1,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)
del _joint51_libero_parent_config
del _joint51_source_config, _joint51_model


# ClausePlan-v1: function-preserving successor to the full Joint51+LIBERO
# model.  Target batches retain the factorized PSM supervision and both target
# and auxiliary prompts expose ordered clause masks.  Only the ordinary prompt
# is parsed; benchmark level/task identifiers never enter the model.
_clause_plan_parent_config = _CONFIGS[-1]
_clause_plan_model = dataclasses.replace(
    _clause_plan_parent_config.model,
    persistent_clause_plan_v1=True,
    persistent_clause_plan_rank=128,
    persistent_clause_plan_slots=8,
    persistent_clause_plan_monotonic_strength=4.0,
)
_CONFIGS.append(
    dataclasses.replace(
        _clause_plan_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_clause_plan_v1'
        ),
        exp_name='joint51_clause_plan_v1_from_joint51_29999_seed7',
        model=_clause_plan_model,
        auxiliary_data=dataclasses.replace(
            _clause_plan_parent_config.auxiliary_data,
            factorized_prompt=False,
            clause_prompt=True,
        ),
        auxiliary_gradient_path_allowlist=(
            *_clause_plan_parent_config.auxiliary_gradient_path_allowlist,
            'clause_plan_adapter',
        ),
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            os.getenv(
                'OPENPI_CLAUSE_PLAN_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Joint51-LIBERO-Replay-v1-stage/'
                'experiments/pi05/joint51_corrected_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4/'
                'joint51_geometry_hmca_v4_corrected_from_psm_29999_seed7/'
                '29999/params',
            ),
            missing_regex='persistent_memory/clause_plan_adapter.*',
            expected_missing_count=11,
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            *_clause_plan_parent_config.architecture_update_multipliers,
            ('persistent_memory/clause_plan_adapter', 5.0),
        ),
        freeze_filter=_clause_plan_model.get_persistent_clause_plan_v1_freeze_filter(),
        persistent_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# ClausePlan-v2: add competitive language-conditioned object slots for target
# identity under static distractors and an explicit four-state manipulation
# phase for cautious grasp/release.  All three policy projections start at
# exact zero, preserving the Joint51 parent before the first update.
_clause_plan_v2_parent_config = _CONFIGS[-1]
_clause_plan_v2_model = dataclasses.replace(
    _clause_plan_v2_parent_config.model,
    object_affordance_graph_reasoner=True,
    object_affordance_hidden_dim=256,
    object_affordance_slots=8,
    object_affordance_layers=2,
    object_affordance_num_heads=8,
    object_affordance_mlp_dim=1024,
    object_affordance_max_cameras=3,
    object_affordance_max_grid_size=16,
    object_affordance_temperature=1.0,
    object_affordance_loss_weight=0.05,
    object_affordance_reconstruction_loss_weight=0.02,
    contact_phase_reasoner=True,
    contact_phase_hidden_dim=256,
    contact_phase_layers=2,
    contact_phase_num_heads=8,
    contact_phase_mlp_dim=1024,
    contact_phase_temperature=0.5,
    contact_phase_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _clause_plan_v2_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_object_contact_v2'
        ),
        exp_name='joint51_clause_plan_object_contact_v2_from_joint51_29999_seed7',
        model=_clause_plan_v2_model,
        auxiliary_gradient_path_allowlist=(
            *_clause_plan_v2_parent_config.auxiliary_gradient_path_allowlist,
            'object_affordance',
            'contact_phase',
        ),
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            os.getenv(
                'OPENPI_CLAUSE_PLAN_V2_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Joint51-LIBERO-Replay-v1-stage/'
                'experiments/pi05/joint51_corrected_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4/'
                'joint51_geometry_hmca_v4_corrected_from_psm_29999_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '.*(?:persistent_memory/clause_plan_adapter|'
                'object_affordance|contact_phase).*'
            ),
            expected_missing_count=148,
        ),
        architecture_update_multipliers=(
            *_clause_plan_v2_parent_config.architecture_update_multipliers,
            ('object_affordance', 5.0),
            ('contact_phase', 5.0),
        ),
        freeze_filter=(
            _clause_plan_v2_model
            .get_clause_plan_object_contact_v2_freeze_filter()
        ),
        resume=False,
        overwrite=False,
    )
)

# ClausePlan-v3: turn the three parallel v2 pathways into an explicit plan
# verifier.  Competitive object slots, four-state contact evidence, current
# PSM memory, ordered subgoals, and the recurrent frontier are jointly fused
# before a zero-initialized action residual.  This targets the formal
# long-horizon/contact failures without reading evaluator or task metadata.
_clause_plan_v3_parent_config = _CONFIGS[-1]
_clause_plan_v3_model = dataclasses.replace(
    _clause_plan_v3_parent_config.model,
    object_future_reasoner=True,
    object_future_hidden_dim=256,
    object_future_queries=12,
    object_future_layers=2,
    object_future_num_heads=8,
    object_future_mlp_dim=1024,
    object_future_max_grid_size=16,
    object_future_reconstruction_loss_weight=0.02,
    object_future_action_loss_weight=0.05,
    object_future_affordance_bridge=True,
    contact_affordance_predictive_fusion=True,
    contact_affordance_clause_plan_verification=True,
    contact_affordance_future_verification=True,
    contact_affordance_relation_verification=True,
    contact_affordance_transition_verification=True,
    contact_affordance_fusion_hidden_dim=256,
    contact_affordance_fusion_layers=2,
    contact_affordance_fusion_num_heads=8,
    contact_affordance_fusion_mlp_dim=1024,
    contact_affordance_risk_loss_weight=0.02,
    contact_affordance_relation_contrastive_loss_weight=0.02,
    contact_affordance_relation_contrastive_temperature=0.1,
)
_CONFIGS.append(
    dataclasses.replace(
        _clause_plan_v3_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3'
        ),
        exp_name=(
            'joint51_clause_plan_verified_contact_v3_'
            'from_joint51_29999_seed7'
        ),
        model=_clause_plan_v3_model,
        data=dataclasses.replace(
            _clause_plan_v3_parent_config.data,
            future_visual_supervision=True,
        ),
        auxiliary_data=dataclasses.replace(
            _clause_plan_v3_parent_config.auxiliary_data,
            future_visual_supervision=True,
        ),
        auxiliary_gradient_path_allowlist=(
            *_clause_plan_v3_parent_config.auxiliary_gradient_path_allowlist,
            'object_future',
            'contact_affordance',
        ),
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            os.getenv(
                'OPENPI_CLAUSE_PLAN_V3_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Joint51-LIBERO-Replay-v1-stage/'
                'experiments/pi05/joint51_corrected_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4/'
                'joint51_geometry_hmca_v4_corrected_from_psm_29999_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '.*(?:persistent_memory/(?:clause_plan_adapter|'
                'role_identity_confidence_gate)|'
                'object_affordance|object_future|contact_phase|'
                'contact_affordance).*'
            ),
            expected_missing_count=355,
        ),
        architecture_update_multipliers=(
            *_clause_plan_v3_parent_config.architecture_update_multipliers,
            ('persistent_memory/role_identity_confidence_gate', 5.0),
            ('object_future', 5.0),
            ('contact_affordance', 5.0),
        ),
        freeze_filter=(
            _clause_plan_v3_model
            .get_clause_plan_verified_contact_v3_freeze_filter()
        ),
        resume=False,
        overwrite=False,
    )
)

# Function-preserving post-ClausePlan-v3 geometry transfer.  The exact
# 808-leaf parent is frozen; two sealed Molmo2-ER leaves and one five-role
# positive-zero gate are the complete trainable successor namespace.
_dual_geometry_parent_config = _CONFIGS[-1]
_dual_geometry_model = dataclasses.replace(
    _dual_geometry_parent_config.model,
    persistent_geometry_external_residual_v1=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _dual_geometry_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_dual_geometry_v1'
        ),
        exp_name=(
            'clause_plan_verified_contact_v3_dual_geometry_v1_'
            'from_clauseplan_v3_29999_seed7'
        ),
        model=_dual_geometry_model,
        auxiliary_gradient_path_allowlist=(
            *_dual_geometry_parent_config.auxiliary_gradient_path_allowlist,
            'geometry_external_residual_v1',
        ),
        weight_loader=weight_loaders.DualGeometryResidualCheckpointWeightLoader(
            parent_params_path=os.getenv(
                'OPENPI_DUAL_GEOMETRY_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Joint51-ClausePlan-v1-stage/experiments/pi05/'
                'clause_plan_v3_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4_'
                'clause_plan_verified_contact_v3/'
                'joint51_clause_plan_verified_contact_v3_from_joint51_29999_seed7/'
                '29999/params',
            ),
            warmstart_artifact=os.getenv(
                'OPENPI_DUAL_GEOMETRY_WARMSTART_ARTIFACT',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/molmo2_er_geometry_warmstart_transplant_v1',
            ),
            warmstart_manifest_sha256=os.getenv(
                'OPENPI_DUAL_GEOMETRY_WARMSTART_MANIFEST_SHA256', ''
            ),
            warmstart_training_manifest_sha256=os.getenv(
                'OPENPI_DUAL_GEOMETRY_TRAINING_MANIFEST_SHA256', ''
            ),
            repo_root=(
                '/path/to/workspace/'
                'VLA-Arena-ClausePlan-v3-DualGeometry-v1-stage'
            ),
            expected_parent_leaf_count=808,
        ),
        architecture_update_multipliers=(
            ('persistent_memory/geometry_external_residual_v1', 20.0),
        ),
        freeze_filter=(
            _dual_geometry_model
            .get_geometry_external_residual_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Action-conditioned temporal target/destination identity refinement.  The
# exact 811-leaf DualGeometry parent is frozen; only the five zero-gated role
# memory leaves are trained from ordinary replan-5 observations and actions.
_temporal_role_parent_config = _CONFIGS[-1]
_temporal_role_model = dataclasses.replace(
    _temporal_role_parent_config.model,
    persistent_temporal_role_memory_v1=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _temporal_role_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_dual_geometry_v1_'
            'temporal_role_memory_v1'
        ),
        exp_name=(
            'temporal_role_memory_v1_from_dual_geometry_29999_seed7'
        ),
        model=_temporal_role_model,
        auxiliary_gradient_path_allowlist=(
            *_temporal_role_parent_config.auxiliary_gradient_path_allowlist,
            'action_conditioned_temporal_object_residual_v1',
        ),
        weight_loader=weight_loaders.TemporalRoleMemoryCheckpointWeightLoader(
            parent_params_path=os.getenv(
                'OPENPI_TEMPORAL_ROLE_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-ClausePlan-v3-DualGeometry-v1-stage/'
                'experiments/pi05/dual_geometry_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4_'
                'clause_plan_verified_contact_v3_dual_geometry_v1/'
                'clause_plan_verified_contact_v3_dual_geometry_v1_'
                'from_clauseplan_v3_29999_seed7/29999/params',
            ),
            expected_parent_leaf_count=811,
        ),
        architecture_update_multipliers=(
            ('persistent_memory/action_conditioned_temporal_object_residual_v1', 20.0),
        ),
        freeze_filter=(
            _temporal_role_model.get_temporal_role_memory_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Cross-camera same-role consensus.  The exact 816-leaf TemporalRoleMemory
# parent is frozen; only five zero-gated attention/residual leaves are trained.
_cross_view_parent_config = _CONFIGS[-1]
_cross_view_model = dataclasses.replace(
    _cross_view_parent_config.model,
    persistent_cross_view_role_consensus_v1=True,
    persistent_cross_view_role_contrastive_loss_weight=0.02,
    persistent_cross_view_role_contrastive_temperature=0.1,
)
_CONFIGS.append(
    dataclasses.replace(
        _cross_view_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_dual_geometry_v1_'
            'temporal_role_memory_v1_cross_view_consensus_v1'
        ),
        exp_name=(
            'cross_view_consensus_v1_from_temporal_role_memory_29999_seed7'
        ),
        model=_cross_view_model,
        auxiliary_gradient_path_allowlist=(
            *_cross_view_parent_config.auxiliary_gradient_path_allowlist,
            'cross_view_role_consensus_v1',
        ),
        weight_loader=(
            weight_loaders.CrossViewRoleConsensusCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_CROSS_VIEW_ROLE_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-v1-stage/'
                    'experiments/pi05/temporal_role_memory_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_dual_geometry_v1_'
                    'temporal_role_memory_v1/'
                    'temporal_role_memory_v1_from_dual_geometry_29999_seed7/'
                    '29999/params',
                ),
                expected_parent_leaf_count=816,
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/cross_view_role_consensus_v1', 20.0),
        ),
        freeze_filter=(
            _cross_view_model.get_cross_view_role_consensus_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Jointly calibrate the two adjacent role branches after CrossView has been
# trained and both of its fixed full-1700 formals are complete.  The inference
# graph is byte-for-byte the same 821-leaf graph; only optimizer ownership
# changes from CrossView-only five leaves to the disjoint union of ten leaves.
_temporal_cross_view_coadapt_parent_config = _CONFIGS[-1]
_CONFIGS.append(
    dataclasses.replace(
        _temporal_cross_view_coadapt_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_dual_geometry_v1_'
            'temporal_role_memory_v1_cross_view_consensus_v1_coadapt_v1'
        ),
        exp_name='temporal_cross_view_coadapt_v1_from_cross_view_29999_seed7',
        weight_loader=(
            weight_loaders.TemporalCrossViewCoAdaptCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_TEMPORAL_CROSS_VIEW_COADAPT_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewConsensus-v1-stage/experiments/pi05/'
                    'cross_view_role_consensus_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_dual_geometry_v1_'
                    'temporal_role_memory_v1_cross_view_consensus_v1/'
                    'cross_view_consensus_v1_from_temporal_role_memory_29999_seed7/'
                    '29999/params',
                ),
                expected_leaf_count=821,
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/action_conditioned_temporal_object_residual_v1', 2.0),
            ('persistent_memory/cross_view_role_consensus_v1', 20.0),
        ),
        freeze_filter=(
            _temporal_cross_view_coadapt_parent_config.model.
            get_temporal_cross_view_coadapt_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Contact-risk calibrated latent role refinement.  The exact 821-leaf CoAdapt
# parent is frozen; seven zero-gated/auxiliary leaves are trained from causal
# role ambiguity, contact transitions, state, memory, and executed actions.
_contact_risk_parent_config = _CONFIGS[-1]
_contact_risk_model = dataclasses.replace(
    _contact_risk_parent_config.model,
    persistent_contact_risk_calibrated_role_residual_v1=True,
    persistent_contact_risk_auxiliary_loss_weight=0.05,
    persistent_contact_risk_class_weights=(1.0, 2.0, 4.0),
)
_CONFIGS.append(
    dataclasses.replace(
        _contact_risk_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_dual_geometry_v1_'
            'temporal_role_memory_v1_cross_view_consensus_v1_coadapt_v1_'
            'contact_risk_v1'
        ),
        exp_name='contact_risk_v1_from_coadapt_29999_seed7',
        model=_contact_risk_model,
        auxiliary_gradient_path_allowlist=(
            *_contact_risk_parent_config.auxiliary_gradient_path_allowlist,
            'contact_risk_calibrated_role_residual_v1',
        ),
        weight_loader=(
            weight_loaders.ContactRiskCalibratedRoleResidualCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_CONTACT_RISK_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-v1-stage/experiments/pi05/'
                    'temporal_cross_view_coadapt_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_dual_geometry_v1_'
                    'temporal_role_memory_v1_cross_view_consensus_v1_coadapt_v1/'
                    'temporal_cross_view_coadapt_v1_from_cross_view_29999_seed7/'
                    '29999/params',
                ),
                expected_parent_leaf_count=821,
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/contact_risk_calibrated_role_residual_v1', 10.0),
        ),
        freeze_filter=(
            _contact_risk_model.
            get_contact_risk_calibrated_role_residual_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)
_relational_role_parent_config = _CONFIGS[-1]
_relational_role_model = dataclasses.replace(
    _relational_role_parent_config.model,
    persistent_relational_role_composer_residual_v1=True,
    persistent_relational_role_auxiliary_loss_weight=0.05,
    persistent_relational_role_class_weights=(1.0,) * 16,
)
_CONFIGS.append(
    dataclasses.replace(
        _relational_role_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_dual_geometry_v1_'
            'temporal_role_memory_v1_cross_view_consensus_v1_coadapt_v1_'
            'contact_risk_v1_relational_role_composer_v1'
        ),
        exp_name='relational_role_composer_v1_from_contact_risk_29999_seed7',
        model=_relational_role_model,
        auxiliary_gradient_path_allowlist=(
            *_relational_role_parent_config.auxiliary_gradient_path_allowlist,
            'relational_role_composer_residual_v1',
        ),
        weight_loader=(
            weight_loaders.RelationalRoleComposerResidualCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_RELATIONAL_ROLE_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-ContactRisk-v1-stage/experiments/pi05/'
                    'contact_risk_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_dual_geometry_v1_'
                    'temporal_role_memory_v1_cross_view_consensus_v1_coadapt_v1_'
                    'contact_risk_v1/'
                    'contact_risk_v1_from_coadapt_29999_seed7/29999/params',
                ),
                expected_parent_leaf_count=828,
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/relational_role_composer_residual_v1', 10.0),
        ),
        freeze_filter=(
            _relational_role_model.
            get_relational_role_composer_residual_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)
# Priority path: initialize the complete 28-leaf residual stack directly from
# the exact ClausePlan-v3 checkpoint and optimize the interacting branches in
# one run.  The sequential stages above remain registered as independently
# auditable ablations, but no longer impose five serial 30k runs before the
# strongest combined architecture can be tested.
_joint_role_geometry_model = _relational_role_model
_CONFIGS.append(
    dataclasses.replace(
        _dual_geometry_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_joint_role_geometry_v1'
        ),
        exp_name='joint_role_geometry_v1_from_clauseplan_v3_29999_seed7',
        model=_joint_role_geometry_model,
        auxiliary_gradient_path_allowlist=(
            *_dual_geometry_parent_config.auxiliary_gradient_path_allowlist,
            'geometry_external_residual_v1',
            'action_conditioned_temporal_object_residual_v1',
            'cross_view_role_consensus_v1',
            'contact_risk_calibrated_role_residual_v1',
            'relational_role_composer_residual_v1',
        ),
        weight_loader=(
            weight_loaders.JointRoleGeometryResidualCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_JOINT_ROLE_GEOMETRY_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-Joint51-ClausePlan-v1-stage/experiments/pi05/'
                    'clause_plan_v3_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3/'
                    'joint51_clause_plan_verified_contact_v3_'
                    'from_joint51_29999_seed7/29999/params',
                ),
                warmstart_artifact=os.getenv(
                    'OPENPI_JOINT_ROLE_GEOMETRY_WARMSTART_ARTIFACT',
                    '/path/to/workspace/VLA-Arena/'
                    'experiments/pi05/molmo2_er_geometry_warmstart_transplant_v1',
                ),
                warmstart_manifest_sha256=os.getenv(
                    'OPENPI_JOINT_ROLE_GEOMETRY_WARMSTART_MANIFEST_SHA256', ''
                ),
                warmstart_training_manifest_sha256=os.getenv(
                    'OPENPI_JOINT_ROLE_GEOMETRY_TRAINING_MANIFEST_SHA256', ''
                ),
                repo_root=(
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-ContactRisk-RelationalRoleComposer-v1-stage'
                ),
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/geometry_external_residual_v1', 20.0),
            ('persistent_memory/action_conditioned_temporal_object_residual_v1', 20.0),
            ('persistent_memory/cross_view_role_consensus_v1', 20.0),
            ('persistent_memory/contact_risk_calibrated_role_residual_v1', 10.0),
            ('persistent_memory/relational_role_composer_residual_v1', 10.0),
        ),
        freeze_filter=(
            _joint_role_geometry_model.
            get_joint_role_geometry_residual_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)
# First post-JointRoleGeometry successor: bind the active language clause and
# persistent frontier to grounded object/reference roles at the latent plan
# boundary.  The complete 836-leaf parent remains frozen.
_clause_role_binding_parent_config = _CONFIGS[-1]
_clause_role_binding_model = dataclasses.replace(
    _joint_role_geometry_model,
    persistent_clause_role_binding_verifier_v1=True,
    persistent_clause_role_binding_auxiliary_loss_weight=0.05,
    # Independent source/destination heads make all 4x4 combinations
    # representable even though only seven joint combinations occur in L0.
    # Each marginal uses bounded inverse-sqrt weights with E_p[weight] = 1.
    # See clause_role_binding_relation_balance_audit_v1.json.
    persistent_clause_role_binding_source_class_weights=(
        0.7291666666666664,
        1.4583333333333328,
        2.25,
        2.25,
    ),
    persistent_clause_role_binding_destination_class_weights=(
        2.25,
        0.7371008633069434,
        1.3116805995280116,
        2.25,
    ),
)
_CONFIGS.append(
    dataclasses.replace(
        _clause_role_binding_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
            'clause_role_binding_verifier_v1'
        ),
        exp_name='clause_role_binding_v1_from_joint_role_geometry_29999_seed7',
        model=_clause_role_binding_model,
        auxiliary_gradient_path_allowlist=(
            *_clause_role_binding_parent_config.auxiliary_gradient_path_allowlist,
            'clause_role_binding_verifier_v1',
        ),
        weight_loader=(
            weight_loaders.ClauseRoleBindingVerifierCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_CLAUSE_ROLE_BINDING_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-ContactRisk-RelationalRoleComposer-v1-stage/'
                    'experiments/pi05/joint_role_geometry_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_joint_role_geometry_v1/'
                    'joint_role_geometry_v1_from_clauseplan_v3_29999_seed7/'
                    '29999/params',
                )
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/clause_role_binding_verifier_v1', 10.0),
        ),
        freeze_filter=(
            _clause_role_binding_model.
            get_clause_role_binding_verifier_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

_semantic_frontier_parent_config = _CONFIGS[-1]
_semantic_frontier_model = dataclasses.replace(
    _clause_role_binding_model,
    persistent_semantic_frontier_completion_verifier_v1=True,
    persistent_semantic_frontier_completion_auxiliary_loss_weight=0.05,
    persistent_semantic_frontier_completion_class_weights=(
        (0.7654313234768432, 1.2377047095441711),
        (0.7283494802753492, 1.1777432596252013),
        (0.73161495547618, 1.183023542664324),
        (0.8741580619709863, 1.4135161666404261),
        (0.8796901294050828, 1.4224615360115087),
        (1.2620024775511787, 2.0406617315142594),
        (2.5693603327352257, 4.154663242545726),
        (2.5693603327352257, 4.154663242545726),
    ),
)
_CONFIGS.append(
    dataclasses.replace(
        _semantic_frontier_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
            'clause_role_binding_verifier_v1_'
            'semantic_frontier_completion_verifier_v1'
        ),
        exp_name='semantic_frontier_completion_v1_from_clause_role_binding_29999_seed7',
        model=_semantic_frontier_model,
        auxiliary_gradient_path_allowlist=(
            *_semantic_frontier_parent_config.auxiliary_gradient_path_allowlist,
            'semantic_frontier_completion_verifier_v1',
        ),
        weight_loader=(
            weight_loaders.SemanticFrontierCompletionVerifierCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_SEMANTIC_FRONTIER_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-ContactRisk-RelationalRoleComposer-v1-stage/'
                    'experiments/pi05/clause_role_binding_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
                    'clause_role_binding_verifier_v1/'
                    'clause_role_binding_v1_from_joint_role_geometry_29999_seed7/'
                    '29999/params',
                )
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/semantic_frontier_completion_verifier_v1', 10.0),
        ),
        freeze_filter=(
            _semantic_frontier_model.
            get_semantic_frontier_completion_verifier_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

_CONFIGS.append(
    dataclasses.replace(
        _clause_role_binding_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
            'joint_clause_role_semantic_frontier_v1'
        ),
        exp_name='joint_clause_role_semantic_frontier_v1_from_joint_role_29999_seed7',
        model=_semantic_frontier_model,
        auxiliary_gradient_path_allowlist=(
            *_clause_role_binding_parent_config.auxiliary_gradient_path_allowlist,
            'clause_role_binding_verifier_v1',
            'semantic_frontier_completion_verifier_v1',
        ),
        weight_loader=(
            weight_loaders.JointClauseRoleSemanticFrontierCheckpointWeightLoader(
                parent_params_path=os.getenv(
                    'OPENPI_JOINT_CLAUSE_SEMANTIC_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-ContactRisk-RelationalRoleComposer-v1-stage/'
                    'experiments/pi05/joint_role_geometry_checkpoints/'
                    'pi05_vla_arena_persistent_structured_demo_language_'
                    'conditional_memory_bridge_geometry_hmca_v4_'
                    'clause_plan_verified_contact_v3_joint_role_geometry_v1/'
                    'joint_role_geometry_v1_from_clauseplan_v3_29999_seed7/'
                    '29999/params',
                )
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/clause_role_binding_verifier_v1', 10.0),
            ('persistent_memory/semantic_frontier_completion_verifier_v1', 10.0),
        ),
        freeze_filter=(
            _semantic_frontier_model.
            get_joint_clause_role_semantic_frontier_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Priority direct-cumulative branch: preserve the exact audited PSM-402 parent
# and train the complete 446-leaf successor graph in one joint run.  This keeps
# the serial stages above as ablations while removing seven 30k dependencies
# from the fastest path to the strongest single-model architecture.
_direct_psm_cumulative848_parent_config = _CONFIGS[-1]
_CONFIGS.append(
    dataclasses.replace(
        _direct_psm_cumulative848_parent_config,
        name=(
            'pi05_vla_arena_persistent_structured_demo_language_'
            'conditional_memory_bridge_geometry_hmca_v4_'
            'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
            'joint_clause_role_semantic_frontier_v1_'
            'direct_psm_cumulative848_v1'
        ),
        exp_name='direct_psm_cumulative848_v1_from_psm_29999_seed7',
        weight_loader=(
            weight_loaders.DirectPsmCumulative848CheckpointWeightLoader(
                primary_params_path=os.getenv(
                    'OPENPI_DIRECT_CUMULATIVE848_PSM_PARENT',
                    '/path/to/workspace/VLA-Arena/'
                    'experiments/pi05/psm_checkpoints/'
                    'pi05_vla_arena_persistent_subgoal_memory/'
                    'persistent_subgoal_memory_compound_stage3_stage4_seed7/'
                    '29999/params',
                ),
                transplant_artifact=os.getenv(
                    'OPENPI_DIRECT_CUMULATIVE848_TRANSPLANT_ARTIFACT',
                    '/path/to/workspace/'
                    'psm_sdla_v3_control_v2/aux_transplant',
                ),
                transplant_manifest_sha256=os.getenv(
                    'OPENPI_DIRECT_CUMULATIVE848_TRANSPLANT_MANIFEST_SHA256', ''
                ),
                warmstart_artifact=os.getenv(
                    'OPENPI_DIRECT_CUMULATIVE848_WARMSTART_ARTIFACT',
                    '/path/to/workspace/VLA-Arena/'
                    'experiments/pi05/molmo2_er_geometry_warmstart_transplant_v1',
                ),
                warmstart_manifest_sha256=os.getenv(
                    'OPENPI_DIRECT_CUMULATIVE848_WARMSTART_MANIFEST_SHA256', ''
                ),
                warmstart_training_manifest_sha256=os.getenv(
                    'OPENPI_DIRECT_CUMULATIVE848_WARMSTART_TRAINING_MANIFEST_SHA256',
                    '',
                ),
                repo_root=(
                    '/path/to/workspace/'
                    'VLA-Arena-ClausePlan-v3-DualGeometry-TemporalRoleMemory-'
                    'CrossViewCoAdapt-ContactRisk-RelationalRoleComposer-v1-stage'
                ),
            )
        ),
        architecture_update_multipliers=(
            ('persistent_memory/structured_demo', 1.0),
            ('spatial_language_aux', 1.0),
            ('persistent_memory/conditional_memory_policy_bridge', 1.0),
            ('persistent_memory/geometry_aux_v3', 1.0),
            ('hierarchical_memory_conditional_adapters', 1.0),
            ('persistent_memory/clause_plan_adapter', 5.0),
            ('object_affordance', 5.0),
            ('contact_phase', 5.0),
            ('persistent_memory/role_identity_confidence_gate', 5.0),
            ('object_future', 5.0),
            ('contact_affordance', 5.0),
            ('persistent_memory/geometry_external_residual_v1', 20.0),
            ('persistent_memory/action_conditioned_temporal_object_residual_v1', 20.0),
            ('persistent_memory/cross_view_role_consensus_v1', 20.0),
            ('persistent_memory/contact_risk_calibrated_role_residual_v1', 10.0),
            ('persistent_memory/relational_role_composer_residual_v1', 10.0),
            ('persistent_memory/clause_role_binding_verifier_v1', 10.0),
            ('persistent_memory/semantic_frontier_completion_verifier_v1', 10.0),
        ),
        freeze_filter=(
            _direct_psm_cumulative848_parent_config.model.
            get_direct_psm_cumulative848_v1_freeze_filter()
        ),
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Direct-848 successor: retain the complete recurrent PSM graph and add HETM
# event/predicate recurrence through exact-zero prior and FiLM boundaries.
_direct_psm_hetm_parent_config = _CONFIGS[-1]
_direct_psm_hetm_model = dataclasses.replace(
    _direct_psm_hetm_parent_config.model,
    hierarchical_event_transition_memory=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _direct_psm_hetm_parent_config,
        name='pi05_vla_arena_direct_psm_cumulative848_hetm_v1',
        exp_name='direct_psm_cumulative848_hetm_v1_seed7',
        model=_direct_psm_hetm_model,
        weight_loader=weight_loaders.DirectPsmWithHetmWarmstartWeightLoader(
            params_path=os.getenv(
                'OPENPI_DIRECT_PSM_HETM_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-PSM29999-DirectCumulative848-v1-stage/'
                'experiments/pi05/direct_psm_cumulative848_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4_'
                'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
                'joint_clause_role_semantic_frontier_v1_'
                'direct_psm_cumulative848_v1/'
                'direct_psm_cumulative848_v1_from_psm_29999_seed7/'
                '29999/params',
            ),
            hetm_bundle_path=os.getenv(
                'OPENPI_DIRECT_PSM_HETM_WARMSTART',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/hetm_libero_structural_warmstart_v9',
            ),
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-HETM-v1',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
        },
        architecture_update_multipliers=(('hetm', 5.0),),
        auxiliary_gradient_path_allowlist=(
            *_direct_psm_hetm_parent_config.auxiliary_gradient_path_allowlist,
            'hetm',
        ),
        freeze_filter=_direct_psm_hetm_model.get_joint_psm_hetm_freeze_filter(),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=8,
        resume=False,
        overwrite=False,
    )
)

# Exact successor: preserve the complete Direct848+HETM parent and add only
# the role-bound affordance causal graph at a byte-zero action boundary.
_direct_psm_hetm_racg_parent_config = _CONFIGS[-1]
_direct_psm_hetm_racg_model = dataclasses.replace(
    _direct_psm_hetm_racg_parent_config.model,
    role_affordance_causal_graph=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _direct_psm_hetm_racg_parent_config,
        name='pi05_vla_arena_direct_psm_cumulative848_hetm_racg_v1',
        exp_name='direct_psm_cumulative848_hetm_racg_v1_seed7',
        model=_direct_psm_hetm_racg_model,
        weight_loader=(
            weight_loaders.DirectPsmHetmWithRacgInitializerWeightLoader(
                params_path=os.getenv(
                    'OPENPI_DIRECT848_HETM_RACG_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-PSM-HETM-Integration-v1-stage/'
                    'experiments/pi05/direct_psm_hetm_checkpoints/'
                    'pi05_vla_arena_direct_psm_cumulative848_hetm_v1/'
                    'direct_psm_cumulative848_hetm_v1_seed7/29999/params',
                )
            )
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-HETM-RACG-v1',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'racg_warmstart': 'parent_hetm_semantic_transport_v3',
        },
        architecture_update_multipliers=(('racg', 5.0),),
        auxiliary_gradient_path_allowlist=(
            *_direct_psm_hetm_racg_parent_config.auxiliary_gradient_path_allowlist,
            'racg',
        ),
        freeze_filter=(
            _direct_psm_hetm_racg_model.get_joint_psm_hetm_racg_freeze_filter()
        ),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
    )
)

# Exact post-RACG successor: inject the sealed Molmo2-ER geometry role reads
# through five zero gates while retaining every Direct848/PSM/HETM/RACG leaf.
_direct_psm_hetm_racg_egp_model = dataclasses.replace(
    _direct_psm_hetm_racg_model,
    racg_external_geometry_prior=True,
)
_direct_psm_hetm_racg_egp_parent_config = dataclasses.replace(
        _direct_psm_hetm_racg_parent_config,
        name='pi05_vla_arena_direct_psm_cumulative848_hetm_racg_egp_v1',
        exp_name='direct_psm_cumulative848_hetm_racg_egp_v1_seed7',
        model=_direct_psm_hetm_racg_egp_model,
        weight_loader=(
            weight_loaders.DirectPsmHetmRacgWithExternalGeometryWeightLoader(
                params_path=os.getenv(
                    'OPENPI_DIRECT848_HETM_RACG_EGP_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-Direct848-HETM-RACG-v1-stage/experiments/pi05/'
                    'direct_psm_hetm_racg_checkpoints/'
                    'pi05_vla_arena_direct_psm_cumulative848_hetm_racg_v1/'
                    'direct_psm_cumulative848_hetm_racg_v1_seed7/29999/params',
                ),
                external_geometry_bundle_path=os.getenv(
                    'OPENPI_DIRECT848_HETM_RACG_EGP_WARMSTART',
                    '/path/to/workspace/'
                    'VLA-Arena-Direct848-HETM-RACG-EGP-v1-stage/'
                    'experiments/pi05/molmo2_er_geometry_warmstart_transplant_v1',
                ),
            )
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-HETM-RACG-EGP-v1',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'selector_used': False,
            'external_geometry_source': 'sealed_molmo2_er_375432_rows',
        },
        architecture_update_multipliers=(('racg_external_geometry', 5.0),),
        auxiliary_gradient_path_allowlist=(
            *_direct_psm_hetm_racg_parent_config.auxiliary_gradient_path_allowlist,
            'racg_external_geometry',
        ),
        freeze_filter=(
            _direct_psm_hetm_racg_egp_model.get_joint_psm_hetm_racg_freeze_filter()
        ),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
)
_CONFIGS.append(_direct_psm_hetm_racg_egp_parent_config)

# Geometry-conditioned hierarchical memory adapters: preserve the complete
# EGP policy and add a zero-output low-rank bridge from its five ordered
# external roles into HMCA's semantic condition at action layers 5/11/17.
_direct_psm_hetm_racg_egp_ghma_model = dataclasses.replace(
    _direct_psm_hetm_racg_egp_model,
    racg_external_geometry_hmca=True,
    racg_external_geometry_hmca_hidden_dim=64,
)
_CONFIGS.append(
    dataclasses.replace(
        _direct_psm_hetm_racg_egp_parent_config,
        name='pi05_vla_arena_direct_psm_cumulative848_hetm_racg_egp_ghma_v1',
        exp_name='direct_psm_cumulative848_hetm_racg_egp_ghma_v1_seed7',
        model=_direct_psm_hetm_racg_egp_ghma_model,
        weight_loader=(
            weight_loaders.DirectPsmHetmRacgEgpWithGeometryHmcaWeightLoader(
                params_path=os.getenv(
                    'OPENPI_DIRECT848_HETM_RACG_EGP_GHMA_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-Direct848-HETM-RACG-EGP-v1-stage/experiments/pi05/'
                    'direct_psm_hetm_racg_egp_checkpoints/'
                    'pi05_vla_arena_direct_psm_cumulative848_hetm_racg_egp_v1/'
                    'direct_psm_cumulative848_hetm_racg_egp_v1_seed7/29999/params',
                ),
            )
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-HETM-RACG-EGP-GHMA-v1',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'selector_used': False,
            'geometry_hmca_layers': [5, 11, 17],
            'geometry_hmca_rank': 64,
        },
        architecture_update_multipliers=(
            *_direct_psm_hetm_racg_egp_parent_config.architecture_update_multipliers,
            ('racg_external_geometry_hmca', 5.0),
        ),
        auxiliary_gradient_path_allowlist=(
            *_direct_psm_hetm_racg_egp_parent_config.auxiliary_gradient_path_allowlist,
            'racg_external_geometry_hmca',
        ),
        freeze_filter=(
            _direct_psm_hetm_racg_egp_ghma_model.get_joint_psm_hetm_racg_freeze_filter()
        ),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
    )
)
_direct_psm_hetm_racg_egp_ghma_parent_config = _CONFIGS[-1]

# Graph-conditioned HMCA: expose all six RACG role nodes and four typed edges
# (including contact/relation/hazard evidence) at action layers 5/11/17 while
# preserving the complete GHMA policy through a byte-zero output boundary.
_direct_psm_hetm_racg_egp_ghma_gchmca_model = dataclasses.replace(
    _direct_psm_hetm_racg_egp_ghma_model,
    racg_graph_hmca=True,
    racg_graph_hmca_hidden_dim=128,
)
_CONFIGS.append(
    dataclasses.replace(
        _direct_psm_hetm_racg_egp_ghma_parent_config,
        name='pi05_vla_arena_direct_psm_cumulative848_hetm_racg_egp_ghma_gchmca_v1',
        exp_name='direct_psm_cumulative848_hetm_racg_egp_ghma_gchmca_v1_seed7',
        model=_direct_psm_hetm_racg_egp_ghma_gchmca_model,
        weight_loader=(
            weight_loaders.DirectPsmHetmRacgEgpGhmaWithGraphHmcaWeightLoader(
                params_path=os.getenv(
                    'OPENPI_DIRECT848_HETM_RACG_EGP_GHMA_GCHMCA_PARENT',
                    '/path/to/workspace/'
                    'VLA-Arena-Direct848-HETM-RACG-EGP-GHMA-v1-stage/'
                    'experiments/pi05/direct_psm_hetm_racg_egp_ghma_checkpoints/'
                    'pi05_vla_arena_direct_psm_cumulative848_hetm_racg_egp_ghma_v1/'
                    'direct_psm_cumulative848_hetm_racg_egp_ghma_v1_seed7/29999/params',
                ),
            )
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-HETM-RACG-EGP-GHMA-GCHMCA-v1',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'selector_used': False,
            'graph_hmca_layers': [5, 11, 17],
            'graph_hmca_rank': 128,
            'graph_hmca_tokens': 10,
        },
        architecture_update_multipliers=(
            *_direct_psm_hetm_racg_egp_ghma_parent_config.architecture_update_multipliers,
            ('racg_graph_hmca', 5.0),
        ),
        auxiliary_gradient_path_allowlist=(
            *_direct_psm_hetm_racg_egp_ghma_parent_config.auxiliary_gradient_path_allowlist,
            'racg_graph_hmca',
        ),
        freeze_filter=(
            _direct_psm_hetm_racg_egp_ghma_gchmca_model.get_joint_psm_hetm_racg_freeze_filter()
        ),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
    )
)

# Time-to-result candidate: train the complete HETM/RACG/EGP/GHMA/GCHMCA
# capability stack jointly from the audited Direct848 checkpoint.  Every
# policy-writing boundary remains exact-zero at initialization, while the
# sealed HETM and Molmo2-ER tensors provide meaningful internal coordinates.
# This preserves the serial lineage above as a controlled comparison but can
# reach the same full architecture after one 30k run instead of five.
_CONFIGS.append(
    dataclasses.replace(
        _direct_psm_hetm_racg_egp_ghma_parent_config,
        name='pi05_vla_arena_direct848_integrated60_v1',
        exp_name='direct848_integrated60_v1_from_direct29999_seed7',
        model=_direct_psm_hetm_racg_egp_ghma_gchmca_model,
        weight_loader=weight_loaders.DirectPsmIntegrated60WeightLoader(
            params_path=os.getenv(
                'OPENPI_DIRECT848_INTEGRATED60_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-PSM29999-DirectCumulative848-v1-stage/'
                'experiments/pi05/direct_psm_cumulative848_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4_'
                'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
                'joint_clause_role_semantic_frontier_v1_'
                'direct_psm_cumulative848_v1/'
                'direct_psm_cumulative848_v1_from_psm_29999_seed7/'
                '29999/params',
            ),
            hetm_bundle_path=os.getenv(
                'OPENPI_DIRECT848_INTEGRATED60_HETM_WARMSTART',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/hetm_libero_structural_warmstart_v9',
            ),
            external_geometry_bundle_path=os.getenv(
                'OPENPI_DIRECT848_INTEGRATED60_EGP_WARMSTART',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-v1-stage/'
                'experiments/pi05/molmo2_er_geometry_warmstart_transplant_v1',
            ),
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-Integrated60-v1',
            'parent_model': 'X-Policy-Direct848-v1@29999',
            'training_mode': 'joint_higher_order_capability_bootstrap',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
            'auxiliary_gradient_merge': 'clip_aware_target_preserving_pcgrad',
            'auxiliary_target_gradient_clip_norm': 1.0,
            'combined_gradient_clip_norm': 1.25,
            'joint_modules': [
                'hetm',
                'racg',
                'racg_external_geometry',
                'racg_external_geometry_hmca',
                'racg_graph_hmca',
            ],
            'hmca_layers': [5, 11, 17],
        },
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('hetm/', 5.0),
            ('racg/', 5.0),
            ('racg_external_geometry/', 5.0),
            ('racg_external_geometry_hmca/', 5.0),
            ('racg_graph_hmca/', 5.0),
        ),
        auxiliary_gradient_path_allowlist=(
            *_direct_psm_hetm_racg_egp_ghma_parent_config.auxiliary_gradient_path_allowlist,
            'hetm',
            'racg',
            'racg_external_geometry',
            'racg_external_geometry_hmca',
            'racg_graph_hmca',
        ),
        auxiliary_gradient_merge='clip_aware_target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.25),
        freeze_filter=(
            _direct_psm_hetm_racg_egp_ghma_gchmca_model.get_joint_psm_hetm_racg_freeze_filter()
        ),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
    )
)

# Predictive successor to Integrated60.  It preserves all five joint
# capability branches and adds future visual/state rollout, action MoE and
# task-progress experts behind zero-output token heads.  The reliability gate
# reads persistent memory and its ordered program, directly targeting dynamic
# obstacle and long-workflow failures without changing the control protocol.
_integrated60_ppwm_parent_config = _CONFIGS[-1]
_integrated60_ppwm_model = dataclasses.replace(
    _integrated60_ppwm_parent_config.model,
    latent_future_reasoner=True,
    latent_future_hidden_dim=256,
    latent_future_layers=2,
    latent_future_num_heads=8,
    latent_future_mlp_dim=1024,
    latent_future_grid_size=4,
    latent_future_loss_weight=0.05,
    state_rollout_reasoner=True,
    state_rollout_hidden_dim=256,
    state_rollout_layers=2,
    state_rollout_num_heads=8,
    state_rollout_mlp_dim=1024,
    state_rollout_target_dim=8,
    state_rollout_loss_weight=0.1,
    action_moe_reasoner=True,
    action_moe_hidden_dim=256,
    action_moe_layers=2,
    action_moe_num_heads=8,
    action_moe_mlp_dim=1024,
    action_moe_num_experts=8,
    action_moe_top_k=2,
    action_moe_expert_dim=512,
    action_moe_temperature=1.0,
    action_moe_prediction_loss_weight=0.05,
    action_moe_balance_loss_weight=0.01,
    task_progress_reasoner=True,
    task_progress_hidden_dim=256,
    task_progress_layers=2,
    task_progress_num_heads=8,
    task_progress_mlp_dim=1024,
    task_progress_bins=10,
    task_progress_loss_weight=0.05,
    # Co-train the inherited contact experts with focal calibration.  The
    # strongest remaining safety failure is cautious grasp; transition index
    # three (release/terminal contact) receives the largest weight while the
    # predictive branches learn the corresponding future dynamics.
    contact_phase_loss_weight=0.10,
    contact_phase_focal_gamma=1.5,
    contact_phase_loss_temperature=0.5,
    contact_phase_transition_boosts=(1.0, 2.0, 1.0, 4.0),
    predictive_world_model_fusion=True,
    predictive_world_model_hidden_dim=256,
    predictive_world_model_auxiliary_scale=1.0 / 4.0,
    predictive_world_model_include_action_moe=True,
    predictive_world_model_reliability_loss_weight=0.02,
    predictive_world_model_router_init_scale=0.01,
)
_CONFIGS.append(
    dataclasses.replace(
        _integrated60_ppwm_parent_config,
        name='pi05_vla_arena_direct848_integrated60_ppwm_v1',
        exp_name='direct848_integrated60_ppwm_v1_from_integrated60_29999_seed7',
        model=_integrated60_ppwm_model,
        data=dataclasses.replace(
            _integrated60_ppwm_parent_config.data,
            future_visual_supervision=True,
            future_state_supervision=True,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L0_L_lerobot_openpi/meta/episodes.jsonl'
            ),
        ),
        auxiliary_data=dataclasses.replace(
            _integrated60_ppwm_parent_config.auxiliary_data,
            future_visual_supervision=True,
            future_state_supervision=True,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'LIBERO_OPENPI_OFFICIAL/meta/episodes.jsonl'
            ),
        ),
        weight_loader=weight_loaders.Integrated60PpwmWeightLoader(
            params_path=os.getenv(
                'OPENPI_INTEGRATED60_PPWM_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-v1-stage/experiments/pi05/'
                'direct848_integrated60_checkpoints/'
                'pi05_vla_arena_direct848_integrated60_v1/'
                'direct848_integrated60_v1_from_direct29999_seed7/29999/params',
            )
        ),
        policy_metadata={
            'model_name': 'X-Policy-Direct848-Integrated60-PPWM-v1',
            'parent_model': 'X-Policy-Direct848-Integrated60-v1@29999',
            'training_mode': 'joint_predictive_dynamics_coupling',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
            'auxiliary_gradient_merge': 'clip_aware_target_preserving_pcgrad',
            'auxiliary_target_gradient_clip_norm': 1.0,
            'combined_gradient_clip_norm': 1.25,
            'predictive_branches': [
                'latent_future',
                'state_rollout',
                'action_moe',
                'task_progress',
                'persistent_memory_routed_fusion',
                'focal_contact_phase_and_affordance',
            ],
        },
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('hetm/', 1.0),
            ('racg/', 1.0),
            ('racg_external_geometry/', 1.0),
            ('racg_external_geometry_hmca/', 1.0),
            ('racg_graph_hmca/', 1.0),
            ('object_future', 1.0),
            ('latent_future', 5.0),
            ('state_rollout', 5.0),
            ('action_moe', 5.0),
            ('task_progress', 5.0),
            ('predictive_world_model', 5.0),
            ('contact_phase', 5.0),
            ('contact_affordance', 3.0),
        ),
        auxiliary_gradient_path_allowlist=(
            *_integrated60_ppwm_parent_config.auxiliary_gradient_path_allowlist,
            'object_future',
            'latent_future',
            'state_rollout',
            'action_moe',
            'task_progress',
            'predictive_world_model',
            'contact_phase',
            'contact_affordance',
        ),
        freeze_filter=_integrated60_ppwm_model.get_joint_integrated60_ppwm_freeze_filter(),
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
    )
)
# Role-grounded successor to the fully trained Integrated60+PPWM policy.  It
# adds no randomly initialized parameter leaves: the extra montage/rationale
# enters through the shared VLM, while an explicit observation mask prevents
# every physical geometry, memory and dynamics branch from interpreting the
# retrieved montage as a live wrist camera.
_integrated60_ppwm_grounded_parent = _CONFIGS[-1]
_integrated60_ppwm_grounded_model = dataclasses.replace(
    _integrated60_ppwm_grounded_parent.model,
    grounded_demonstration_camera_context_only=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _integrated60_ppwm_grounded_parent,
        name='pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_v1',
        exp_name='direct848_integrated60_ppwm_grounded_role_v1_seed7',
        model=_integrated60_ppwm_grounded_model,
        data=dataclasses.replace(
            _integrated60_ppwm_grounded_parent.data,
            grounded_demonstration_bank_path=(
                '/path/to/workspace/'
                'VLA-Arena-XPolicy-Stella-v3-GroundedTrajectory-stage/'
                'experiments/pi05/demonstrations/'
                'vla_arena_grounded_demo_bank_v2.npz'
            ),
            grounded_demonstration_dropout=0.5,
            grounded_demonstration_retrieval_mode='role_signature',
            grounded_demonstration_rationale=True,
        ),
        weight_loader=weight_loaders.ExactCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_INTEGRATED60_PPWM_GROUNDED_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-PPWM-v1-stage/'
                'experiments/pi05/direct848_integrated60_ppwm_checkpoints/'
                'pi05_vla_arena_direct848_integrated60_ppwm_v1/'
                'direct848_integrated60_ppwm_v1_from_integrated60_29999_seed7/'
                '29999/params',
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3.0e-6,
            decay_steps=30_000,
            decay_lr=3.0e-7,
        ),
        policy_metadata={
            'model_name': 'X-Policy-Integrated60-PPWM-GroundedRole-v1',
            'parent_model': 'X-Policy-Direct848-Integrated60-PPWM-v1@29999',
            'training_mode': 'grounded_role_context_finetune',
            'replan_steps': 5,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
            'auxiliary_gradient_merge': 'clip_aware_target_preserving_pcgrad',
            'auxiliary_target_gradient_clip_norm': 1.0,
            'combined_gradient_clip_norm': 1.25,
            'grounded_context': [
                'role_signature_retrieval',
                'grounded_keyframe_montage',
                'grounded_phase_rationale',
                'vlm_only_demo_camera_mask',
            ],
        },
        architecture_update_multipliers=tuple(
            (path, 1.0)
            for path, _ in (
                _integrated60_ppwm_grounded_parent.architecture_update_multipliers
            )
        ),
        num_train_steps=30_000,
        resume=False,
        overwrite=False,
    )
)
# Post-GroundedRole architecture candidate: route the causal cross-replan PSM
# state into every action-expert AdaRMS layer.  The new output projection is
# exactly zero initialized, so loading the completed GroundedRole checkpoint
# preserves its policy before optimization while opening a genuinely new
# layerwise memory path for the long-horizon L1/L2 failure cluster.
_memory_adarms_parent = _CONFIGS[-1]
_memory_adarms_model = dataclasses.replace(
    _memory_adarms_parent.model,
    persistent_memory_adarms=True,
    persistent_memory_adarms_hidden_dim=256,
)
_CONFIGS.append(
    dataclasses.replace(
        _memory_adarms_parent,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_v1'
        ),
        exp_name=(
            'direct848_integrated60_ppwm_grounded_role_memory_adarms_v1_seed7'
        ),
        model=_memory_adarms_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_INTEGRATED60_PPWM_MEMORY_ADARMS_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-PPWM-GroundedRole-v1-stage/'
                'experiments/pi05/direct848_integrated60_ppwm_grounded_role_'
                'checkpoints/pi05_vla_arena_direct848_integrated60_ppwm_'
                'grounded_role_v1/direct848_integrated60_ppwm_grounded_role_'
                'v1_from_ppwm_29999_seed7/29999/params',
            ),
            missing_regex='.*persistent_memory_adarms.*',
            expected_missing_count=4,
        ),
        freeze_filter=(
            _memory_adarms_model.get_joint_integrated60_ppwm_memory_adarms_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            *_memory_adarms_parent.architecture_update_multipliers,
            ('persistent_memory_adarms', 5.0),
        ),
        policy_metadata={
            **_memory_adarms_parent.policy_metadata,
            'model_name': 'X-Policy-Integrated60-PPWM-GroundedRole-MemoryAdaRMS-v1',
            'parent_model': 'X-Policy-Integrated60-PPWM-GroundedRole-v1@29999',
            'training_mode': 'joint_layerwise_persistent_memory_adarms',
            'architecture_delta': [
                'causal_psm_memory_pool',
                'validity_masked_shared_psm_phase_context',
                'persistent_memory_adarms',
                'zero_initialized_layerwise_memory_residual',
            ],
        },
        resume=False,
        overwrite=False,
    )
)
# Contact-precision successor to phase-aware Memory-AdaRMS.  The parent
# already exposes PSM and risk-conditioned contact-affordance features, but
# only through independent additive paths.  This branch learns their
# multiplicative interaction and applies a distinct scale/shift at every
# action position.  Its output head is exactly zero initialized, preserving
# the completed Memory-AdaRMS policy at the loading boundary.
_phase_contact_parent = _CONFIGS[-1]
_phase_contact_model = dataclasses.replace(
    _phase_contact_parent.model,
    phase_contact_action_film=True,
    phase_contact_action_film_hidden_dim=256,
    persistent_action_prior_conditioning=True,
)
_CONFIGS.append(
    dataclasses.replace(
        _phase_contact_parent,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_phase_contact_v1'
        ),
        exp_name=(
            'direct848_integrated60_ppwm_grounded_role_memory_adarms_'
            'phase_contact_v1_seed7'
        ),
        model=_phase_contact_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_INTEGRATED60_PPWM_PHASE_CONTACT_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-PPWM-GroundedRole-v1-stage/'
                'experiments/pi05/direct848_integrated60_ppwm_grounded_role_'
                'memory_adarms_checkpoints/pi05_vla_arena_direct848_'
                'integrated60_ppwm_grounded_role_memory_adarms_v1/'
                'direct848_integrated60_ppwm_grounded_role_memory_adarms_'
                'v1_seed7/29999/params',
            ),
            missing_regex='.*(phase_contact_film|persistent_action_prior).*',
            expected_missing_count=8,
        ),
        freeze_filter=(
            _phase_contact_model.get_joint_integrated60_ppwm_phase_contact_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            *_phase_contact_parent.architecture_update_multipliers,
            ('phase_contact_film', 5.0),
            ('persistent_action_prior', 5.0),
        ),
        policy_metadata={
            **_phase_contact_parent.policy_metadata,
            'model_name': (
                'X-Policy-Integrated60-PPWM-GroundedRole-'
                'MemoryAdaRMS-PhaseContact-v1'
            ),
            'parent_model': (
                'X-Policy-Integrated60-PPWM-GroundedRole-'
                'MemoryAdaRMS-v1@29999'
            ),
            'training_mode': 'joint_phase_contact_tokenwise_film',
            'architecture_delta': [
                'persistent_frontier_phase_context',
                'per_action_contact_phase_context',
                'risk_conditioned_contact_affordance_context',
                'multiplicative_phase_contact_interaction',
                'zero_initialized_tokenwise_film',
                'persistent_phase_conditioned_contextual_action_prior',
                'zero_initialized_prior_query_residual',
            ],
        },
        resume=False,
        overwrite=False,
    )
)
# Token-preserving long-horizon successor.  Pooled AdaRMS and input FiLM are
# retained, while action tokens additionally read the distinct causal PSM
# slots at three depths of the action expert.  Only the low-rank output banks
# are zero initialized, so the entire new path is an exact parent identity at
# checkpoint load.  The output bank opens on the first update and Q/K/V start
# receiving gradient immediately afterward.
_layerwise_memory_parent = _CONFIGS[-1]
_layerwise_memory_model = dataclasses.replace(
    _layerwise_memory_parent.model,
    layerwise_persistent_memory_attention=True,
    layerwise_persistent_memory_attention_rank=32,
    layerwise_persistent_memory_attention_alpha=32.0,
    layerwise_persistent_memory_attention_layers=(5, 11, 17),
)
_CONFIGS.append(
    dataclasses.replace(
        _layerwise_memory_parent,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_phase_contact_layerwise_memory_v1'
        ),
        exp_name=(
            'direct848_integrated60_ppwm_grounded_role_memory_adarms_'
            'phase_contact_layerwise_memory_v1_seed7'
        ),
        model=_layerwise_memory_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_INTEGRATED60_PPWM_LAYERWISE_MEMORY_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-PPWM-GroundedRole-v1-stage/'
                'experiments/pi05/direct848_integrated60_ppwm_grounded_role_'
                'memory_adarms_phase_contact_checkpoints/'
                'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
                'memory_adarms_phase_contact_v1/'
                'direct848_integrated60_ppwm_grounded_role_memory_adarms_'
                'phase_contact_v1_seed7/29999/params',
            ),
            missing_regex='.*layerwise_persistent_memory_attention.*',
            expected_missing_count=4,
        ),
        freeze_filter=(
            _layerwise_memory_model
            .get_joint_integrated60_ppwm_layerwise_memory_attention_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            *_layerwise_memory_parent.architecture_update_multipliers,
            ('layerwise_persistent_memory_attention', 5.0),
        ),
        policy_metadata={
            **_layerwise_memory_parent.policy_metadata,
            'model_name': (
                'X-Policy-Integrated60-PPWM-GroundedRole-MemoryAdaRMS-'
                'PhaseContact-LayerwiseMemory-v1'
            ),
            'parent_model': (
                'X-Policy-Integrated60-PPWM-GroundedRole-MemoryAdaRMS-'
                'PhaseContact-v1@29999'
            ),
            'training_mode': 'joint_token_preserving_layerwise_psm_attention',
            'architecture_delta': [
                'action_query_to_distinct_psm_slot_attention',
                'inherited_psm_slot_identity_in_layerwise_reads',
                'current_and_remaining_program_memory_tokens',
                'validity_masked_remaining_program',
                'three_depth_low_rank_memory_reads',
                'zero_initialized_layerwise_memory_outputs',
            ],
        },
        resume=False,
        overwrite=False,
    )
)
_hcea_parent_config = _CONFIGS[-1]
# Best-first architecture candidate.  It collapses the full parameterized
# Integrated60 -> PPWM -> MemoryAdaRMS -> PhaseContact -> LayerwiseMemory chain
# into one 30k run from Direct848.  GroundedRole retrieval is intentionally
# excluded because adding un-gated prefix tokens would violate exact step-zero
# policy preservation; that data intervention remains in the serial ablation
# chain.  A lower base LR than the Integrated60-only run protects inherited
# Direct LoRA parameters, while 5x multipliers let every zero-boundary
# successor open at an effective 5e-5 peak rate.
_collapsed_best_first_model = dataclasses.replace(
    _layerwise_memory_model,
    grounded_demonstration_camera_context_only=False,
)
_CONFIGS.append(
    dataclasses.replace(
        _CONFIGS[-1],
        name='pi05_vla_arena_direct848_collapsed_architecture_v1',
        exp_name='direct848_collapsed_architecture_v1_from_direct29999_seed7',
        model=_collapsed_best_first_model,
        data=_integrated60_ppwm_grounded_parent.data,
        auxiliary_data=_integrated60_ppwm_grounded_parent.auxiliary_data,
        weight_loader=weight_loaders.DirectPsmCollapsedArchitectureWeightLoader(
            params_path=os.getenv(
                'OPENPI_DIRECT848_COLLAPSED_ARCHITECTURE_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-PSM29999-DirectCumulative848-v1-stage/'
                'experiments/pi05/direct_psm_cumulative848_checkpoints/'
                'pi05_vla_arena_persistent_structured_demo_language_'
                'conditional_memory_bridge_geometry_hmca_v4_'
                'clause_plan_verified_contact_v3_joint_role_geometry_v1_'
                'joint_clause_role_semantic_frontier_v1_'
                'direct_psm_cumulative848_v1/'
                'direct_psm_cumulative848_v1_from_psm_29999_seed7/'
                '29999/params',
            ),
            hetm_bundle_path=os.getenv(
                'OPENPI_DIRECT848_COLLAPSED_HETM_WARMSTART',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/hetm_libero_structural_warmstart_v9',
            ),
            external_geometry_bundle_path=os.getenv(
                'OPENPI_DIRECT848_COLLAPSED_EGP_WARMSTART',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-v1-stage/'
                'experiments/pi05/molmo2_er_geometry_warmstart_transplant_v1',
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=30_000,
            decay_lr=1.0e-6,
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('hetm/', 5.0),
            ('racg/', 5.0),
            ('racg_external_geometry/', 5.0),
            ('racg_external_geometry_hmca/', 5.0),
            ('racg_graph_hmca/', 5.0),
            ('object_future', 1.0),
            ('latent_future', 5.0),
            ('state_rollout', 5.0),
            ('action_moe', 5.0),
            ('task_progress', 5.0),
            ('predictive_world_model', 5.0),
            ('contact_phase', 5.0),
            ('contact_affordance', 3.0),
            ('persistent_memory_adarms', 5.0),
            ('phase_contact_film', 5.0),
            ('persistent_action_prior', 5.0),
            ('layerwise_persistent_memory_attention', 5.0),
        ),
        freeze_filter=(
            _collapsed_best_first_model
            .get_joint_integrated60_ppwm_layerwise_memory_attention_freeze_filter()
        ),
        policy_metadata={
            **_CONFIGS[-1].policy_metadata,
            'model_name': 'X-Policy-Direct848-CollapsedArchitecture-v1',
            'parent_model': 'X-Policy-Direct848-v1@29999',
            'training_mode': 'best_first_joint_architecture_bootstrap',
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
            'grounded_role_context': False,
            'step_zero_policy_identity': 'Direct848 plus sealed warm starts',
            'joint_modules': [
                'hetm_racg_geometry_hmca',
                'predictive_world_model',
                'memory_adarms',
                'phase_contact_film',
                'persistent_action_prior',
                'layerwise_persistent_memory_attention',
            ],
        },
        persistent_sequence_training=True,
        hetm_sequence_training=True,
        num_train_steps=30_000,
        batch_size=32,
        auxiliary_batch_size=8,
        fsdp_devices=1,
        resume=False,
        overwrite=False,
    )
)
del _collapsed_best_first_model
# Hierarchical clause-event alignment is kept as an independent serial
# successor so its 15k/29999 full1700 results remain attributable.  The five
# new leaves alone are optimized; their zero policy gate preserves the exact
# LayerwiseMemory parent policy at step zero while the auxiliary head gives
# query/key/transition leaves a nonzero first-update gradient.
_hcea_model = dataclasses.replace(
    _hcea_parent_config.model,
    grounded_demonstration_camera_context_only=False,
    persistent_hierarchical_clause_event_alignment_v1=True,
    persistent_hierarchical_clause_event_alignment_auxiliary_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _hcea_parent_config,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_v1'
        ),
        exp_name=(
            'direct848_integrated60_ppwm_grounded_role_memory_adarms_'
            'phase_contact_layerwise_memory_hcea_joint_l01_v1_seed7'
        ),
        model=_hcea_model,
        # CollapsedArchitecture deliberately removed the retrieved montage
        # from the live policy input (see grounded_role_context=False below).
        # Clear the inherited bank as well: leaving a bank configured while
        # context-only camera masking is disabled is an invalid data/model
        # contract and makes a fresh production loader fail before training.
        data=dataclasses.replace(
            _hcea_parent_config.data,
            grounded_demonstration_bank_path=None,
        ),
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_HCEA_JOINT_L01_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-Direct848-Integrated60-PPWM-GroundedRole-v1-stage/'
                'experiments/pi05/direct848_collapsed_architecture_checkpoints/'
                'pi05_vla_arena_direct848_collapsed_architecture_v1/'
                'direct848_collapsed_architecture_v1_from_direct29999_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '.*persistent_memory/'
                'hierarchical_clause_event_alignment_v1.*'
            ),
            expected_missing_count=5,
        ),
        freeze_filter=(
            _hcea_model
            .get_hierarchical_clause_event_alignment_v1_freeze_filter()
        ),
        architecture_update_path=(
            'persistent_memory/hierarchical_clause_event_alignment_v1'
        ),
        architecture_update_multiplier=5.0,
        architecture_update_multipliers=(),
        joint_l0_l1_sequence_training=True,
        joint_l1_repo_id=(
            '/path/to/workspace/datasets/'
            'VLA_Arena_L1_L_lerobot_openpi'
        ),
        joint_l0_l1_episode_metadata_path=(
            '/path/to/workspace/VLA-Arena/experiments/'
            'pi05/l0_l1_joint_sequence_metadata_v1/meta/episodes.jsonl'
        ),
        joint_l1_episode_offset=3018,
        joint_l1_task_offset=60,
        persistent_memory_static_replay_cache=(
            '/dev/shm/vla_arena_psm_static_replay/slot1'
        ),
        persistent_memory_static_replay_binding=(
            '/path/to/workspace/'
            'VLA-Arena-Direct848-Integrated60-PPWM-GroundedRole-v1-stage/'
            'experiments/pi05/'
            'direct848_collapsed_architecture_static_replay_binding_v1.json'
        ),
        persistent_memory_joint_l1_static_replay_cache=(
            '/dev/shm/vla_arena_hcea_joint_l01_l1_static_replay/slot0'
        ),
        persistent_memory_joint_l1_static_replay_binding=(
            '/path/to/workspace/'
            'VLA-Arena-HCEA-JointL01-v1-stage/experiments/pi05/'
            'hcea_joint_l01_l1_static_replay_binding_v1.json'
        ),
        persistent_memory_joint_replay_consumer_admission=(
            '/path/to/workspace/'
            'VLA-Arena-HCEA-JointL01-v1-stage/experiments/pi05/'
            'hcea_joint_l01_dual_source_replay_consumer_admission_v1.json'
        ),
        policy_metadata={
            **_hcea_parent_config.policy_metadata,
            'model_name': 'X-Policy-HCEA-JointL01-v1',
            'parent_model': (
                'X-Policy-Direct848-CollapsedArchitecture-v1@29999'
            ),
            'training_mode': 'joint_l0_l1_hierarchical_clause_event_alignment',
            'architecture_delta': [
                'explicit_slot_to_clause_event_pooling',
                'causal_current_or_next_clause_transition_consistency',
                'stay_or_advance_one_frontier_residual',
                'zero_initialized_policy_blend_gate',
                'ungated_auxiliary_training_behind_zero_policy_boundary',
            ],
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
# Architecture-first successor: preserve the complete HCEA checkpoint and add
# only ten causal recovery-expert leaves.  This remains separate from the
# exact-graph ActionCoAdapt run so gains are attributable to the new module.
_hcea_recovery_parent = _CONFIGS[-1]
_hcea_recovery_model = dataclasses.replace(
    _hcea_recovery_parent.model,
    persistent_hcea_causal_recovery_action_experts_v1=True,
    persistent_hcea_causal_recovery_intent_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _hcea_recovery_parent,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_'
            'causal_recovery_action_experts_v1'
        ),
        exp_name='hcea_joint_l01_causal_recovery_action_experts_v1_from_hcea_29999_seed7',
        model=_hcea_recovery_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_HCEA_RECOVERY_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-HCEA-JointL01-v1-stage/experiments/pi05/'
                'hcea_joint_l01_checkpoints/'
                'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
                'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_v1/'
                'direct848_integrated60_ppwm_grounded_role_memory_adarms_'
                'phase_contact_layerwise_memory_hcea_joint_l01_v1_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '.*persistent_memory/'
                'hcea_causal_recovery_action_experts_v1.*'
            ),
            expected_missing_count=10,
        ),
        freeze_filter=(
            _hcea_recovery_model
            .get_hcea_causal_recovery_action_experts_v1_freeze_filter()
        ),
        architecture_update_path=(
            'persistent_memory/hcea_causal_recovery_action_experts_v1'
        ),
        architecture_update_multiplier=5.0,
        architecture_update_multipliers=(),
        policy_metadata={
            **_hcea_recovery_parent.policy_metadata,
            'model_name': 'X-Policy-HCEA-CausalRecoveryExperts-v1',
            'parent_model': 'X-Policy-HCEA-JointL01-v1@29999',
            'training_mode': 'joint_l0_l1_hcea_causal_recovery_action_experts',
            'architecture_delta': [
                'causal_previous_action_intent_head',
                'observed_frontier_transition_evidence',
                'hold_retry_confirmed_stabilize_action_experts',
                'bounded_zero_initialized_action_token_residual',
                'ungated_transition_intent_auxiliary_training',
            ],
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
# If the attributable Recovery run improves HCEA but remains below 60%, retain
# its exact 1,256-leaf graph and let the causal recovery loss co-adapt the
# downstream action generator. Prefix-producing vision/language weights and
# the discrete action-prior codebook remain frozen.
_hcea_recovery_action_cadapt_parent = _CONFIGS[-1]
_CONFIGS.append(
    dataclasses.replace(
        _hcea_recovery_action_cadapt_parent,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_'
            'causal_recovery_action_experts_action_cadapt_v1'
        ),
        exp_name=(
            'hcea_recovery_action_cadapt_v1_from_recovery29999_seed7'
        ),
        weight_loader=weight_loaders.ExactCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_HCEA_RECOVERY_ACTION_CADAPT_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-HCEA-JointL01-v1-stage/experiments/pi05/'
                'hcea_causal_recovery_action_experts_checkpoints/'
                'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
                'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_'
                'causal_recovery_action_experts_v1/'
                'hcea_joint_l01_causal_recovery_action_experts_v1_'
                'from_hcea_29999_seed7/29999/params',
            )
        ),
        freeze_filter=(
            _hcea_recovery_action_cadapt_parent.model
            .get_hcea_recovery_action_cadapt_v1_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('latent_future', 1.0),
            ('state_rollout', 1.0),
            ('action_moe', 1.0),
            ('task_progress', 1.0),
            ('predictive_world_model', 1.0),
            ('contact_phase', 2.0),
            ('contact_affordance', 2.0),
            ('hierarchical_clause_event_alignment_v1', 1.0),
            ('hcea_causal_recovery_action_experts_v1', 1.0),
            ('hierarchical_memory_conditional_adapters', 0.25),
            ('state_adarms', 0.25),
            ('state_film', 0.25),
            ('action_prior', 0.25),
            ('action_in_proj', 0.25),
            ('action_out_proj', 0.25),
            ('time_mlp', 0.25),
            ('PaliGemma/llm/layers', 0.25),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-6,
            decay_steps=30_000,
            decay_lr=2e-7,
        ),
        gradient_accumulation_steps=1,
        num_train_steps=30_000,
        batch_size=32,
        fsdp_devices=1,
        log_interval=20,
        save_interval=15_000,
        keep_period=15_000,
        num_workers=16,
        auxiliary_num_workers=8,
        policy_metadata={
            **_hcea_recovery_action_cadapt_parent.policy_metadata,
            'model_name': 'X-Policy-HCEA-Recovery-ActionCoAdapt-v1',
            'parent_model': 'X-Policy-HCEA-CausalRecoveryExperts-v1@29999',
            'training_mode': 'joint_l0_l1_hcea_recovery_action_cadapt',
            'architecture_delta': [
                'exact_recovery_graph_downstream_action_reopening',
                'recovery_expert_and_hcea_cadaptation',
                'state_adarms_and_state_action_film_cadaptation',
                'contextual_dual_action_prior_cadaptation',
                'action_expert_only_lora_cadaptation',
                'frozen_visual_language_prefix',
            ],
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del _hcea_recovery_action_cadapt_parent
# Follow the recovery experts with persistent manipulated/reference identity
# transport.  This is a strict eleven-leaf successor: it retains every HCEA
# and recovery parameter and trains only the new permutation-aware module.
_hcea_role_transport_parent = _CONFIGS[-1]
_hcea_role_transport_model = dataclasses.replace(
    _hcea_role_transport_parent.model,
    persistent_hcea_causal_role_identity_transport_expert_v1=True,
    persistent_hcea_causal_role_identity_transport_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _hcea_role_transport_parent,
        name=(
            'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
            'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_'
            'causal_recovery_action_experts_'
            'causal_role_identity_transport_expert_v1'
        ),
        exp_name=(
            'hcea_causal_recovery_role_identity_transport_v1_'
            'from_recovery29999_seed7'
        ),
        model=_hcea_role_transport_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_HCEA_ROLE_TRANSPORT_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-HCEA-JointL01-v1-stage/experiments/pi05/'
                'hcea_causal_recovery_action_experts_checkpoints/'
                'pi05_vla_arena_direct848_integrated60_ppwm_grounded_role_'
                'memory_adarms_phase_contact_layerwise_memory_hcea_joint_l01_'
                'causal_recovery_action_experts_v1/'
                'hcea_joint_l01_causal_recovery_action_experts_v1_'
                'from_hcea_29999_seed7/29999/params',
            ),
            missing_regex=(
                '.*persistent_memory/'
                'hcea_causal_role_identity_transport_expert_v1.*'
            ),
            expected_missing_count=11,
        ),
        freeze_filter=(
            _hcea_role_transport_model
            .get_hcea_causal_role_identity_transport_expert_v1_freeze_filter()
        ),
        architecture_update_path=(
            'persistent_memory/'
            'hcea_causal_role_identity_transport_expert_v1'
        ),
        architecture_update_multiplier=5.0,
        architecture_update_multipliers=(),
        policy_metadata={
            **_hcea_role_transport_parent.policy_metadata,
            'model_name': 'X-Policy-HCEA-CausalRoleIdentityTransport-v1',
            'parent_model': 'X-Policy-HCEA-CausalRecoveryExperts-v1@29999',
            'training_mode': (
                'joint_l0_l1_hcea_causal_role_identity_transport'
            ),
            'architecture_delta': [
                'previous_action_conditioned_role_transport',
                'permutation_aware_role_assignment',
                'aligned_and_swap_recovery_action_experts',
                'bounded_zero_initialized_action_token_residual',
                'ungated_role_transition_auxiliary_training',
            ],
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del _hcea_role_transport_parent, _hcea_role_transport_model
del _hcea_recovery_parent, _hcea_recovery_model
del _hcea_parent_config, _hcea_model
del _layerwise_memory_parent, _layerwise_memory_model
del _phase_contact_parent, _phase_contact_model
del _memory_adarms_parent, _memory_adarms_model
del _integrated60_ppwm_grounded_parent, _integrated60_ppwm_grounded_model
del _integrated60_ppwm_parent_config, _integrated60_ppwm_model
del _direct_psm_hetm_racg_egp_ghma_gchmca_model
del _direct_psm_hetm_racg_egp_ghma_parent_config
del _direct_psm_hetm_racg_egp_ghma_model
del _direct_psm_hetm_racg_egp_parent_config
del _direct_psm_hetm_racg_egp_model
del _direct_psm_hetm_racg_parent_config, _direct_psm_hetm_racg_model
del _direct_psm_hetm_parent_config, _direct_psm_hetm_model
del _direct_psm_cumulative848_parent_config
del _semantic_frontier_parent_config, _semantic_frontier_model
del _clause_role_binding_parent_config, _clause_role_binding_model
del _joint_role_geometry_model
del _relational_role_parent_config, _relational_role_model
del _contact_risk_parent_config, _contact_risk_model
del _temporal_cross_view_coadapt_parent_config
del _cross_view_parent_config, _cross_view_model
del _temporal_role_parent_config, _temporal_role_model
del _dual_geometry_parent_config, _dual_geometry_model
del _clause_plan_v3_parent_config, _clause_plan_v3_model
del _clause_plan_v2_parent_config, _clause_plan_v2_model
del _clause_plan_parent_config, _clause_plan_model

# High-upside compound candidate: retain the production PSM's grounded,
# ordered cross-replan state and add a short-horizon predictive world model.
# Its fusion gate explicitly reads the private memory, so branch selection can
# depend on durable target/reference bindings and subgoal progress rather than
# only the current frame.  A completed PSM checkpoint owns every inherited
# path; only the five predictive groups are newly initialized.
_persistent_predictive_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_persistent_subgoal_memory'
)
_persistent_predictive_model = dataclasses.replace(
    _persistent_predictive_source_config.model,
    object_future_reasoner=True,
    object_future_hidden_dim=256,
    object_future_queries=12,
    object_future_layers=2,
    object_future_num_heads=8,
    object_future_mlp_dim=1024,
    object_future_max_grid_size=16,
    object_future_reconstruction_loss_weight=0.02,
    object_future_action_loss_weight=0.05,
    latent_future_reasoner=True,
    latent_future_hidden_dim=256,
    latent_future_layers=2,
    latent_future_num_heads=8,
    latent_future_mlp_dim=1024,
    latent_future_grid_size=4,
    latent_future_loss_weight=0.05,
    state_rollout_reasoner=True,
    state_rollout_hidden_dim=256,
    state_rollout_layers=2,
    state_rollout_num_heads=8,
    state_rollout_mlp_dim=1024,
    state_rollout_target_dim=8,
    state_rollout_loss_weight=0.1,
    action_moe_reasoner=True,
    action_moe_hidden_dim=256,
    action_moe_layers=2,
    action_moe_num_heads=8,
    action_moe_mlp_dim=1024,
    action_moe_num_experts=8,
    action_moe_top_k=2,
    action_moe_expert_dim=512,
    action_moe_temperature=1.0,
    action_moe_prediction_loss_weight=0.05,
    action_moe_balance_loss_weight=0.01,
    task_progress_reasoner=True,
    task_progress_hidden_dim=256,
    task_progress_layers=2,
    task_progress_num_heads=8,
    task_progress_mlp_dim=1024,
    task_progress_bins=10,
    task_progress_loss_weight=0.05,
    predictive_world_model_fusion=True,
    predictive_world_model_hidden_dim=256,
    predictive_world_model_auxiliary_scale=1.0 / 4.0,
    predictive_world_model_include_action_moe=True,
    predictive_world_model_reliability_loss_weight=0.02,
    predictive_world_model_router_init_scale=0.01,
)
_CONFIGS.append(
    dataclasses.replace(
        _persistent_predictive_source_config,
        name='pi05_vla_arena_persistent_predictive_world_model',
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('object_future', 5.0),
            ('latent_future', 5.0),
            ('state_rollout', 5.0),
            ('action_moe', 5.0),
            ('task_progress', 5.0),
            ('predictive_world_model', 5.0),
        ),
        model=_persistent_predictive_model,
        data=dataclasses.replace(
            _persistent_predictive_source_config.data,
            future_visual_supervision=True,
            future_state_supervision=True,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L0_L_lerobot_openpi/meta/episodes.jsonl'
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_PSM_CHECKPOINT_PATH',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/psm_checkpoints/'
                'pi05_vla_arena_persistent_subgoal_memory/'
                'persistent_subgoal_memory_compound_stage3_stage4_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '.*(object_future|latent_future|state_rollout|action_moe|task_progress|'
                'predictive_world_model).*'
            ),
        ),
        freeze_filter=(
            _persistent_predictive_model
            .get_persistent_predictive_world_model_freeze_filter()
        ),
        # Joint persistent sequences carry up to 8 replans and a second visual
        # encode for future targets.  Preserve global batch 32 while keeping
        # each four-GPU microbatch at one sequence per device.
        gradient_accumulation_steps=8,
    )
)
del _persistent_predictive_source_config, _persistent_predictive_model

# Failure-heatmap successor to PPWM: explicitly couple competitive visual
# object slots and manipulation phase to the private memory.  Assignment
# entropy supplies an object-ambiguity risk signal, while close/release labels
# identify the action positions that require cautious contact control.
_persistent_contact_source_config = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_persistent_predictive_world_model'
)
_persistent_contact_model = dataclasses.replace(
    _persistent_contact_source_config.model,
    object_affordance_graph_reasoner=True,
    object_affordance_hidden_dim=256,
    object_affordance_slots=8,
    object_affordance_layers=2,
    object_affordance_num_heads=8,
    object_affordance_mlp_dim=1024,
    object_affordance_max_cameras=3,
    object_affordance_max_grid_size=16,
    object_affordance_temperature=1.0,
    object_affordance_loss_weight=0.05,
    object_affordance_reconstruction_loss_weight=0.02,
    contact_phase_reasoner=True,
    contact_phase_hidden_dim=256,
    contact_phase_layers=2,
    contact_phase_num_heads=8,
    contact_phase_mlp_dim=1024,
    contact_phase_temperature=0.5,
    contact_phase_loss_weight=0.05,
    contact_affordance_predictive_fusion=True,
    contact_affordance_fusion_hidden_dim=256,
    contact_affordance_fusion_layers=2,
    contact_affordance_fusion_num_heads=8,
    contact_affordance_fusion_mlp_dim=1024,
    contact_affordance_risk_loss_weight=0.02,
)
_CONFIGS.append(
    dataclasses.replace(
        _persistent_contact_source_config,
        name='pi05_vla_arena_persistent_contact_affordance_world_model',
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('object_future', 1.0),
            ('latent_future', 1.0),
            ('state_rollout', 1.0),
            ('action_moe', 1.0),
            ('task_progress', 1.0),
            ('predictive_world_model', 1.0),
            ('object_affordance', 5.0),
            ('contact_phase', 5.0),
            ('contact_affordance', 5.0),
        ),
        model=_persistent_contact_model,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.getenv(
                'OPENPI_VLA_ARENA_PPWM_CHECKPOINT_PATH',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/persistent_predictive_checkpoints/'
                'pi05_vla_arena_persistent_predictive_world_model/'
                'persistent_predictive_world_model_from_psm_seed7/'
                '29999/params',
            ),
            missing_regex=(
                '.*(object_affordance|contact_phase|contact_affordance).*'
            ),
        ),
        freeze_filter=(
            _persistent_contact_model
            .get_persistent_contact_affordance_freeze_filter()
        ),
        gradient_accumulation_steps=8,
    )
)
del _persistent_contact_source_config, _persistent_contact_model

# Exact-successor architecture rooted in the best audited formal policy
# (Contextual Dual LoRA @ 29999, cell-mean SR 0.512424).  The existing
# contextual action-prior states already summarize the multimodal prefix.
# This successor projects their pooled representation into the adaptive
# RMSNorm condition consumed by every action-expert attention/MLP layer.
# The final projection is exactly zero initialized, so step-zero policy output
# is identical to the 51.24 parent.  Only the four new adapter leaves train.
_best_anchor_context_adarms_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_contextual_dual_lora'
)
_best_anchor_context_adarms_model = dataclasses.replace(
    _best_anchor_context_adarms_parent.model,
    context_adarms=True,
    context_adarms_hidden_dim=256,
)
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_context_adarms_parent,
        name='pi05_vla_arena_best_anchor_context_adarms_v1',
        exp_name='best_anchor_context_adarms_v1_from_contextual29999_seed7',
        model=_best_anchor_context_adarms_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_BEST_ANCHOR_CONTEXT_ADARMS_PARENT',
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/checkpoints/'
                'pi05_vla_arena_contextual_dual_lora/'
                'contextual_dual_lora_full_seed7/29999/params',
            ),
            missing_regex='.*context_adarms.*',
            expected_missing_count=4,
        ),
        freeze_filter=(
            _best_anchor_context_adarms_model
            .get_best_anchor_context_adarms_freeze_filter()
        ),
        architecture_update_path='context_adarms',
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_data=LeRobotLiberoDataConfig(
            repo_id=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L1_L_lerobot_openpi'
            ),
            assets=AssetsConfig(
                assets_dir=(
                    '/path/to/workspace/checkpoints/'
                    'pi05-vla-arena-finetuned/assets'
                ),
                asset_id='VLA_Arena_L0_L_lerobot_openpi/VLA_Arena',
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        auxiliary_batch_size=32,
        auxiliary_loss_weight=0.25,
        auxiliary_num_workers=8,
        auxiliary_task_balanced_sampling=True,
        auxiliary_gradient_path_allowlist=('context_adarms',),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3.0e-5,
            decay_steps=30_000,
            decay_lr=3.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-ContextAdaRMS-v1',
            'parent_model': 'X-Policy-ContextualDualLoRA@29999',
            'parent_formal_cell_mean_sr': 0.5124242424242424,
            'training_mode': 'strict_four_leaf_context_adarms_successor',
            'architecture_delta': [
                'pooled_multimodal_action_prior_to_action_layer_adarms',
                'zero_initialized_context_projection',
                'frozen_90_leaf_best_parent',
            ],
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del _best_anchor_context_adarms_parent, _best_anchor_context_adarms_model

# Cumulative best-anchor successor aimed at the weakest formal tier (L2).
# ContextAdaRMS remains frozen after its own complete 30k training.  A compact
# temporal refiner reads the noisy trajectory, inherited flow velocity,
# action-expert hidden states, state, diffusion time and contextual prior.  It
# predicts only the residual velocity error and enters the policy through a
# per-action-axis tanh gate initialized to exactly zero, preserving parent
# inference before optimization while still receiving a direct residual loss.
_best_anchor_velocity_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_context_adarms_v1'
)
_best_anchor_velocity_model = dataclasses.replace(
    _best_anchor_velocity_parent.model,
    velocity_refiner=True,
    velocity_refiner_hidden_dim=256,
    velocity_refiner_layers=2,
    velocity_refiner_num_heads=8,
    velocity_refiner_mlp_dim=1024,
    velocity_refiner_loss_weight=0.05,
)
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_velocity_parent,
        name='pi05_vla_arena_best_anchor_context_velocity_refiner_v1',
        exp_name=(
            'best_anchor_context_velocity_refiner_v1_from_'
            'context_adarms29999_seed7'
        ),
        model=_best_anchor_velocity_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=os.getenv(
                'OPENPI_BEST_ANCHOR_VELOCITY_PARENT',
                '/path/to/workspace/'
                'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
                'experiments/pi05/checkpoints/'
                'pi05_vla_arena_best_anchor_context_adarms_v1/'
                'best_anchor_context_adarms_v1_from_'
                'contextual29999_seed7/29999/params',
            ),
            missing_regex='.*velocity_refiner.*',
            expected_missing_count=54,
        ),
        freeze_filter=(
            _best_anchor_velocity_model
            .get_best_anchor_velocity_refiner_freeze_filter()
        ),
        architecture_update_path='velocity_refiner',
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_gradient_path_allowlist=('velocity_refiner',),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': (
                'X-Policy-BestAnchor-ContextAdaRMS-VelocityRefiner-v1'
            ),
            'parent_model': (
                'X-Policy-BestAnchor-ContextAdaRMS-v1@best_full1700'
            ),
            'training_mode': 'strict_54_leaf_velocity_residual_successor',
            'architecture_delta': [
                'context_state_time_conditioned_velocity_residual_transformer',
                'zero_initialized_per_axis_tanh_gate',
                'frozen_94_leaf_context_adarms_parent',
                'direct_stop_gradient_residual_supervision',
            ],
            'targeted_failure_mode': (
                'L2 long-horizon trajectory drift and contact-phase correction'
            ),
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del _best_anchor_velocity_parent, _best_anchor_velocity_model

# Cumulative best-anchor successor for the largest observed failure cluster:
# Contextual Dual LoRA scores zero on both L1 and L2 long-horizon cells.  Eight
# ordered latent slots read the full instruction/image prefix and current state,
# receive audited semantic-phase supervision during training, and emit an
# action-token residual through an exactly-zero output projection.  The frozen
# ContextAdaRMS + VelocityRefiner parent is therefore preserved at step zero.
_best_anchor_subgoal_parent = next(
    config
    for config in _CONFIGS
    if config.name
    == 'pi05_vla_arena_best_anchor_context_velocity_refiner_v1'
)
_best_anchor_subgoal_model = dataclasses.replace(
    _best_anchor_subgoal_parent.model,
    language_subgoal_reasoner=True,
    language_subgoal_hidden_dim=256,
    language_subgoal_slots=8,
    language_subgoal_layers=2,
    language_subgoal_num_heads=8,
    language_subgoal_mlp_dim=1024,
    language_subgoal_temperature=1.0,
    language_subgoal_progress_loss_weight=0.05,
    language_subgoal_action_loss_weight=0.05,
    # Inverse-sqrt weights from the effective L0 + 0.5*L1 frame mixture.
    # Empirical weighted mean is exactly one; rare completion phases receive
    # a modest 1.63--1.74x emphasis without destabilizing action supervision.
    language_subgoal_phase_class_weights=(
        0.7555964772,
        0.9198929564,
        0.9332401807,
        1.0519303781,
        0.9886850551,
        1.0999199184,
        1.7360514515,
        1.6313828827,
    ),
)
_best_anchor_subgoal_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_context_velocity_refiner_v1/'
    'best_anchor_context_velocity_refiner_v1_from_'
    'context_adarms29999_seed7/29999/params'
)
_best_anchor_subgoal_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_SUBGOAL_PARENT',
    _best_anchor_subgoal_default_parent,
)
_best_anchor_subgoal_reanchored = (
    '/pi05_vla_arena_best_anchor_context_adarms_v1/'
    in _best_anchor_subgoal_parent_path
)
_best_anchor_subgoal_contextual_reanchored = (
    '/pi05_vla_arena_contextual_dual_lora/'
    in _best_anchor_subgoal_parent_path
)
if _best_anchor_subgoal_contextual_reanchored:
    # The raw 51.24 parent predates both cumulative intermediate modules.
    # Their output gates are exactly zero at initialization, so admitting
    # these 58 additional leaves preserves the parent policy byte-for-byte
    # while LanguageSubgoal remains the only trainable namespace.
    _best_anchor_subgoal_missing_regex = (
        '.*(context_adarms|velocity_refiner|language_subgoal).*'
    )
    _best_anchor_subgoal_expected_missing_count = 135
    _best_anchor_subgoal_parent_name = (
        'X-Policy-Contextual-Dual-LoRA@global_full1700_floor'
    )
elif _best_anchor_subgoal_reanchored:
    # The velocity refiner is part of the successor graph but is intentionally
    # absent from the stable ContextAdaRMS checkpoint.  Its per-axis gain is
    # initialized to exact zero and all velocity leaves are frozen, so the
    # missing module is behaviorally inert while LanguageSubgoal trains.
    _best_anchor_subgoal_missing_regex = (
        '.*(velocity_refiner|language_subgoal).*'
    )
    _best_anchor_subgoal_expected_missing_count = 131
    _best_anchor_subgoal_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-v1@best_full1700_floor'
    )
else:
    _best_anchor_subgoal_missing_regex = '.*language_subgoal.*'
    _best_anchor_subgoal_expected_missing_count = 77
    _best_anchor_subgoal_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-VelocityRefiner-v1'
        '@best_full1700'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_subgoal_parent,
        name='pi05_vla_arena_best_anchor_language_subgoal_v1',
        exp_name=(
            'best_anchor_language_subgoal_v1_from_'
            'velocity_refiner_seed7'
        ),
        model=_best_anchor_subgoal_model,
        data=dataclasses.replace(
            _best_anchor_subgoal_parent.data,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/datasets/'
                'VLA_Arena_L0_L_lerobot_openpi/meta/'
                'persistent_semantic_phase_targets.json'
            ),
        ),
        auxiliary_data=dataclasses.replace(
            _best_anchor_subgoal_parent.auxiliary_data,
            task_progress_supervision=True,
            episode_metadata_path=(
                '/path/to/workspace/VLA-Arena/'
                'experiments/pi05/'
                'l1_persistent_semantic_phase_targets_all_event_v1.json'
            ),
        ),
        auxiliary_loss_weight=0.5,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_subgoal_parent_path,
            missing_regex=_best_anchor_subgoal_missing_regex,
            expected_missing_count=(
                _best_anchor_subgoal_expected_missing_count
            ),
        ),
        freeze_filter=(
            _best_anchor_subgoal_model
            .get_best_anchor_language_subgoal_freeze_filter()
        ),
        architecture_update_path='language_subgoal',
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_gradient_path_allowlist=('language_subgoal',),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-LanguageSubgoal-v1',
            'parent_model': _best_anchor_subgoal_parent_name,
            'training_mode': 'strict_77_leaf_ordered_subgoal_successor',
            'architecture_delta': [
                'eight_ordered_instruction_conditioned_subgoal_slots',
                'dense_masked_multimodal_prefix_read',
                'training_only_audited_semantic_phase_supervision',
                'zero_initialized_action_token_residual',
                'frozen_148_leaf_context_adarms_velocity_parent',
            ],
            'targeted_failure_mode': (
                'long_horizon L1/L2 zero-success and workflow composition'
            ),
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_subgoal_default_parent,
    _best_anchor_subgoal_expected_missing_count,
    _best_anchor_subgoal_missing_regex,
    _best_anchor_subgoal_model,
    _best_anchor_subgoal_parent,
    _best_anchor_subgoal_parent_name,
    _best_anchor_subgoal_parent_path,
    _best_anchor_subgoal_reanchored,
    _best_anchor_subgoal_contextual_reanchored,
)

# Low-rate co-adaptation of the complete cumulative best-anchor stack.  The
# graph is identical to LanguageSubgoal, so ExactCheckpointWeightLoader proves
# every leaf is inherited.  The original 90-leaf 51.24 policy remains frozen;
# only ContextAdaRMS (4), VelocityRefiner (54), and SemanticLanguageSubgoal
# (77) may coordinate.  Lower update multipliers protect the earlier modules
# while the semantic controller remains the primary adaptation surface.
_best_anchor_joint_cadapt_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_language_subgoal_v1'
)
_best_anchor_joint_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_language_subgoal_v1/'
    'best_anchor_language_subgoal_v1_from_'
    'velocity_refiner_seed7/29999/params'
)
_best_anchor_joint_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_JOINT_CADAPT_PARENT',
    _best_anchor_joint_default_parent,
)
if '/pi05_vla_arena_best_anchor_language_subgoal_v1/' in (
    _best_anchor_joint_parent_path
):
    _best_anchor_joint_loader = weight_loaders.ExactCheckpointWeightLoader(
        params_path=_best_anchor_joint_parent_path
    )
    _best_anchor_joint_parent_name = (
        'X-Policy-BestAnchor-LanguageSubgoal-v1@best_full1700'
    )
    _best_anchor_joint_training_mode = (
        'exact_graph_low_rate_three_module_joint_coadaptation'
    )
elif '/pi05_vla_arena_best_anchor_context_velocity_refiner_v1/' in (
    _best_anchor_joint_parent_path
):
    _best_anchor_joint_loader = (
        weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_joint_parent_path,
            missing_regex='.*language_subgoal.*',
            expected_missing_count=77,
        )
    )
    _best_anchor_joint_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-VelocityRefiner-v1'
        '@global_full1700_floor'
    )
    _best_anchor_joint_training_mode = (
        'velocity_floor_plus_language_joint_coadaptation'
    )
elif '/pi05_vla_arena_best_anchor_context_adarms_v1/' in (
    _best_anchor_joint_parent_path
):
    _best_anchor_joint_loader = (
        weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_joint_parent_path,
            missing_regex='.*(velocity_refiner|language_subgoal).*',
            expected_missing_count=131,
        )
    )
    _best_anchor_joint_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-v1@global_full1700_floor'
    )
    _best_anchor_joint_training_mode = (
        'context_floor_plus_velocity_language_joint_coadaptation'
    )
elif '/pi05_vla_arena_contextual_dual_lora/' in (
    _best_anchor_joint_parent_path
):
    _best_anchor_joint_loader = (
        weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_joint_parent_path,
            missing_regex=(
                '.*(context_adarms|velocity_refiner|language_subgoal).*'
            ),
            expected_missing_count=135,
        )
    )
    _best_anchor_joint_parent_name = (
        'X-Policy-Contextual-Dual-LoRA@global_full1700_floor'
    )
    _best_anchor_joint_training_mode = (
        'contextual_floor_plus_three_module_joint_coadaptation'
    )
else:
    raise ValueError(
        'unsupported JointCoAdapt parent checkpoint: '
        f'{_best_anchor_joint_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_joint_cadapt_parent,
        name='pi05_vla_arena_best_anchor_joint_cadapt_v1',
        exp_name='best_anchor_joint_cadapt_v1_from_semantic_subgoal_seed7',
        weight_loader=_best_anchor_joint_loader,
        freeze_filter=(
            _best_anchor_joint_cadapt_parent.model
            .get_best_anchor_joint_cadapt_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('context_adarms', 0.25),
            ('velocity_refiner', 0.5),
            ('language_subgoal', 1.0),
        ),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=(
            'context_adarms',
            'velocity_refiner',
            'language_subgoal',
        ),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=30_000,
            decay_lr=1.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-JointCoAdapt-v1',
            'parent_model': _best_anchor_joint_parent_name,
            'training_mode': _best_anchor_joint_training_mode,
            'architecture_delta': [
                'no_new_parameters_exact_semantic_subgoal_parent',
                'joint_context_velocity_semantic_controller_optimization',
                'module_specific_update_rates_0.25_0.5_1.0',
                'frozen_original_90_leaf_contextual_dual_lora_anchor',
                'target_preserving_l1_pcgrad_over_135_owned_leaves',
            ],
            'targeted_failure_mode': (
                'cross-module coordination for long-horizon L1/L2 execution'
            ),
            'trainable_leaf_count': 135,
            'frozen_anchor_leaf_count': 90,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_joint_cadapt_parent,
    _best_anchor_joint_default_parent,
    _best_anchor_joint_loader,
    _best_anchor_joint_parent_name,
    _best_anchor_joint_parent_path,
    _best_anchor_joint_training_mode,
)

# Multi-checkpoint specialist initialization for a later fallback stage.  The
# 51.24% contextual policy owns every shared leaf; only the three explicitly
# audited adapter namespaces come from their best completed formal checkpoint.
# This separate named config leaves the active TaskMoE/ClosedLoop queue intact.
_best_anchor_specialist_root = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/experiments/pi05/checkpoints/'
)
_best_anchor_specialist_base = (
    '/path/to/workspace/VLA-Arena/experiments/pi05/'
    'checkpoints/pi05_vla_arena_contextual_dual_lora/'
    'contextual_dual_lora_full_seed7/29999/params'
)
_best_anchor_specialist_context = (
    _best_anchor_specialist_root
    + 'pi05_vla_arena_best_anchor_context_adarms_v1/'
    'best_anchor_context_adarms_v1_from_contextual29999_seed7/15000/params'
)
_best_anchor_specialist_velocity = (
    _best_anchor_specialist_root
    + 'pi05_vla_arena_best_anchor_context_velocity_refiner_v1/'
    'best_anchor_context_velocity_refiner_v1_from_context_adarms29999_seed7/'
    '29999/params'
)
_best_anchor_specialist_language = os.getenv(
    'OPENPI_BEST_ANCHOR_SPECIALIST_FUSION_LANGUAGE',
    _best_anchor_specialist_root
    + 'pi05_vla_arena_best_anchor_language_subgoal_v1/'
    'best_anchor_language_subgoal_v1_from_velocity_refiner_seed7/29999/params',
)
_best_anchor_specialist_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_joint_cadapt_v1'
)
_best_anchor_specialist_model = dataclasses.replace(
    _best_anchor_specialist_parent.model,
    specialist_module_router=True,
    specialist_module_router_hidden_dim=256,
    specialist_module_router_temperature=1.0,
    specialist_module_router_balance_loss_weight=0.01,
)
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_specialist_parent,
        name='pi05_vla_arena_best_anchor_specialist_fusion_v1',
        exp_name='best_anchor_specialist_fusion_v1_seed7',
        model=_best_anchor_specialist_model,
        weight_loader=weight_loaders.BestAnchorSpecialistFusionWeightLoader(
            base_params_path=_best_anchor_specialist_base,
            context_params_path=_best_anchor_specialist_context,
            velocity_params_path=_best_anchor_specialist_velocity,
            language_params_path=_best_anchor_specialist_language,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5.0e-6,
            decay_steps=30_000,
            decay_lr=5.0e-7,
        ),
        freeze_filter=(
            _best_anchor_specialist_model
            .get_best_anchor_specialist_router_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multipliers=(
            ('context_adarms', 0.25),
            ('velocity_refiner', 0.5),
            ('language_subgoal', 1.0),
            ('specialist_module_router', 1.0),
        ),
        auxiliary_gradient_path_allowlist=(
            'context_adarms',
            'velocity_refiner',
            'language_subgoal',
            'specialist_module_router',
        ),
        policy_metadata={
            **_best_anchor_specialist_parent.policy_metadata,
            'model_name': 'X-Policy-BestAnchor-SpecialistFusion-v1',
            'parent_model': (
                'X-Policy-Contextual-Dual-LoRA@29999 plus audited '
                'ContextAdaRMS@15000 VelocityRefiner@29999 and '
                'best-formal LanguageSubgoal namespace transplants'
            ),
            'training_mode': 'strict_multi_checkpoint_specialist_coadaptation',
            'architecture_delta': [
                'preserve_original_float32_90_leaf_51.24_anchor',
                'transplant_trained_context_adarms_4_leaf_namespace',
                'transplant_trained_velocity_refiner_54_leaf_namespace',
                'transplant_trained_language_subgoal_77_leaf_namespace',
                'context_state_conditioned_three_specialist_router',
                'unit_gate_zero_score_function_preserving_initialization',
                'batch_global_router_balance_without_per_sample_uniformity',
                'close_five_outer_policy_boundaries_for_exact_step_zero_anchor',
                'low_rate_target_preserving_joint_coordination',
            ],
            'parameter_leaf_count': 231,
            'trainable_leaf_count': 141,
            'frozen_anchor_leaf_count': 90,
            'specialist_leaf_count': 135,
            'router_leaf_count': 6,
            'initial_specialist_gates': [1.0, 1.0, 1.0],
            'router_balance_loss_weight': 0.01,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_specialist_base,
    _best_anchor_specialist_context,
    _best_anchor_specialist_language,
    _best_anchor_specialist_model,
    _best_anchor_specialist_parent,
    _best_anchor_specialist_root,
    _best_anchor_specialist_velocity,
)

# Conditional successor reserved for the remaining contact/generalization gap.
# The audited DualActionReasoner is the most complementary existing expert to
# the 51.24 anchor, while cautious-grasp L2 is still zero.  Add its implicit /
# explicit action paths together with a four-state contact-phase controller to
# the cumulative best-anchor graph.  Every action-facing projection is exactly
# zero initialized, so the selected JointCoAdapt parent is preserved at step
# zero; only the 138 new leaves (6.41M parameters) may train.
_best_anchor_dual_contact_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_joint_cadapt_v1'
)
_best_anchor_dual_contact_model = dataclasses.replace(
    _best_anchor_dual_contact_parent.model,
    dual_action_reasoner=True,
    contact_phase_reasoner=True,
    contact_phase_hidden_dim=256,
    contact_phase_layers=2,
    contact_phase_num_heads=8,
    contact_phase_mlp_dim=1024,
    contact_phase_temperature=0.5,
    contact_phase_loss_weight=0.10,
    contact_phase_focal_gamma=1.5,
    contact_phase_loss_temperature=0.5,
    contact_phase_transition_boosts=(1.0, 2.0, 1.0, 4.0),
)
_best_anchor_dual_contact_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_joint_cadapt_v1/'
    'best_anchor_joint_cadapt_v1_from_'
    'semantic_subgoal_seed7/29999/params'
)
_best_anchor_dual_contact_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_DUAL_CONTACT_PARENT',
    _best_anchor_dual_contact_default_parent,
)
_best_anchor_dual_contact_new_regex = (
    'action_prior_implicit|action_prior_explicit|'
    'action_prior_guidance|contact_phase'
)
if any(
    family in _best_anchor_dual_contact_parent_path
    for family in (
        '/pi05_vla_arena_best_anchor_joint_cadapt_v1/',
        '/pi05_vla_arena_best_anchor_language_subgoal_v1/',
    )
):
    _best_anchor_dual_contact_missing_regex = (
        f'.*({_best_anchor_dual_contact_new_regex}).*'
    )
    _best_anchor_dual_contact_expected_missing_count = 138
elif '/pi05_vla_arena_best_anchor_context_velocity_refiner_v1/' in (
    _best_anchor_dual_contact_parent_path
):
    _best_anchor_dual_contact_missing_regex = (
        f'.*(language_subgoal|{_best_anchor_dual_contact_new_regex}).*'
    )
    _best_anchor_dual_contact_expected_missing_count = 215
elif '/pi05_vla_arena_best_anchor_context_adarms_v1/' in (
    _best_anchor_dual_contact_parent_path
):
    _best_anchor_dual_contact_missing_regex = (
        '.*(velocity_refiner|language_subgoal|'
        f'{_best_anchor_dual_contact_new_regex}).*'
    )
    _best_anchor_dual_contact_expected_missing_count = 269
elif '/pi05_vla_arena_contextual_dual_lora/' in (
    _best_anchor_dual_contact_parent_path
):
    _best_anchor_dual_contact_missing_regex = (
        '.*(context_adarms|velocity_refiner|language_subgoal|'
        f'{_best_anchor_dual_contact_new_regex}).*'
    )
    _best_anchor_dual_contact_expected_missing_count = 273
else:
    raise ValueError(
        'unsupported DualContact parent checkpoint: '
        f'{_best_anchor_dual_contact_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_dual_contact_parent,
        name='pi05_vla_arena_best_anchor_dual_contact_v1',
        exp_name='best_anchor_dual_contact_v1_from_joint_cadapt_seed7',
        model=_best_anchor_dual_contact_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_dual_contact_parent_path,
            missing_regex=_best_anchor_dual_contact_missing_regex,
            expected_missing_count=(
                _best_anchor_dual_contact_expected_missing_count
            ),
        ),
        freeze_filter=(
            _best_anchor_dual_contact_model
            .get_best_anchor_dual_contact_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=(
            'action_prior_implicit',
            'action_prior_explicit',
            'action_prior_guidance',
            'contact_phase',
        ),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-DualContact-v1',
            'parent_model': 'X-Policy-BestAnchor-JointCoAdapt-v1@best_full1700',
            'training_mode': 'strict_138_leaf_dual_action_contact_successor',
            'architecture_delta': [
                'implicit_layerwise_action_evidence',
                'explicit_flow_action_reasoner',
                'action_token_cross_guidance',
                'four_state_contact_phase_controller',
                'zero_initialized_action_facing_projections',
                'frozen_225_leaf_joint_cadapt_parent',
            ],
            'targeted_failure_mode': (
                'cautious-grasp contact precision plus complementary L1/L2 '
                'generalization'
            ),
            'new_parameter_leaf_count': 138,
            'new_parameter_count': 6_408_760,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_dual_contact_default_parent,
    _best_anchor_dual_contact_expected_missing_count,
    _best_anchor_dual_contact_missing_regex,
    _best_anchor_dual_contact_model,
    _best_anchor_dual_contact_new_regex,
    _best_anchor_dual_contact_parent,
    _best_anchor_dual_contact_parent_path,
)

# If the cumulative semantic/contact branch still misses 60%, ground its
# ordered language plan in a competitive visual object bank.  The manipulated
# and reference/receptacle roles are selected separately for every subgoal,
# which directly targets relational prepositions and unseen-object transfer.
# Both action-facing projections are exactly zero initialized, so the selected
# DualContact parent remains bit-identical at step zero.
_best_anchor_grounded_subgoal_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_dual_contact_v1'
)
_best_anchor_grounded_subgoal_model = dataclasses.replace(
    _best_anchor_grounded_subgoal_parent.model,
    object_affordance_graph_reasoner=True,
    object_affordance_hidden_dim=256,
    object_affordance_slots=8,
    object_affordance_layers=2,
    object_affordance_num_heads=8,
    object_affordance_mlp_dim=1024,
    object_affordance_temperature=1.0,
    object_affordance_loss_weight=0.05,
    object_affordance_reconstruction_loss_weight=0.0,
    object_subgoal_binding=True,
    object_subgoal_binding_hidden_dim=256,
    object_subgoal_binding_layers=2,
    object_subgoal_binding_num_heads=8,
    object_subgoal_binding_mlp_dim=1024,
    object_subgoal_binding_temperature=1.0,
    object_subgoal_binding_action_loss_weight=0.05,
    object_subgoal_binding_distinct_roles=True,
)
_best_anchor_grounded_subgoal_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_dual_contact_v1/'
    'best_anchor_dual_contact_v1_from_joint_cadapt_seed7/'
    '29999/params'
)
_best_anchor_grounded_subgoal_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_GROUNDED_SUBGOAL_PARENT',
    _best_anchor_grounded_subgoal_default_parent,
)
_best_anchor_grounded_new_regex = (
    'object_affordance|object_subgoal_binding'
)
if '/pi05_vla_arena_best_anchor_dual_contact_v1/' in (
    _best_anchor_grounded_subgoal_parent_path
):
    _best_anchor_grounded_missing_regex = (
        f'.*({_best_anchor_grounded_new_regex}).*'
    )
    _best_anchor_grounded_expected_missing_count = 170
    _best_anchor_grounded_parent_name = (
        'X-Policy-BestAnchor-DualContact-v1@global_full1700'
    )
elif any(
    family in _best_anchor_grounded_subgoal_parent_path
    for family in (
        '/pi05_vla_arena_best_anchor_joint_cadapt_v1/',
        '/pi05_vla_arena_best_anchor_language_subgoal_v1/',
    )
):
    _best_anchor_grounded_missing_regex = (
        '.*(action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|'
        f'{_best_anchor_grounded_new_regex}).*'
    )
    _best_anchor_grounded_expected_missing_count = 308
    _best_anchor_grounded_parent_name = (
        'X-Policy-BestAnchor-SemanticStack-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_context_velocity_refiner_v1/' in (
    _best_anchor_grounded_subgoal_parent_path
):
    _best_anchor_grounded_missing_regex = (
        '.*(language_subgoal|action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|'
        f'{_best_anchor_grounded_new_regex}).*'
    )
    _best_anchor_grounded_expected_missing_count = 385
    _best_anchor_grounded_parent_name = (
        'X-Policy-BestAnchor-VelocityRefiner-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_context_adarms_v1/' in (
    _best_anchor_grounded_subgoal_parent_path
):
    _best_anchor_grounded_missing_regex = (
        '.*(velocity_refiner|language_subgoal|action_prior_implicit|'
        'action_prior_explicit|action_prior_guidance|contact_phase|'
        f'{_best_anchor_grounded_new_regex}).*'
    )
    _best_anchor_grounded_expected_missing_count = 439
    _best_anchor_grounded_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_contextual_dual_lora/' in (
    _best_anchor_grounded_subgoal_parent_path
):
    _best_anchor_grounded_missing_regex = (
        '.*(context_adarms|velocity_refiner|language_subgoal|'
        'action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|'
        f'{_best_anchor_grounded_new_regex}).*'
    )
    _best_anchor_grounded_expected_missing_count = 443
    _best_anchor_grounded_parent_name = (
        'X-Policy-Contextual-Dual-LoRA@global_full1700_floor'
    )
else:
    raise ValueError(
        'unsupported GroundedSubgoal parent checkpoint: '
        f'{_best_anchor_grounded_subgoal_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_grounded_subgoal_parent,
        name='pi05_vla_arena_best_anchor_grounded_subgoal_v1',
        exp_name='best_anchor_grounded_subgoal_v1_from_dual_contact_seed7',
        model=_best_anchor_grounded_subgoal_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_grounded_subgoal_parent_path,
            missing_regex=_best_anchor_grounded_missing_regex,
            expected_missing_count=(
                _best_anchor_grounded_expected_missing_count
            ),
        ),
        freeze_filter=(
            _best_anchor_grounded_subgoal_model
            .get_best_anchor_grounded_subgoal_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=(
            'object_affordance',
            'object_subgoal_binding',
        ),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-GroundedSubgoal-v1',
            'parent_model': _best_anchor_grounded_parent_name,
            'training_mode': 'strict_object_slot_subgoal_binding_successor',
            'architecture_delta': [
                'eight_competitive_multiview_visual_object_slots',
                'distinct_manipulated_and_reference_object_roles',
                'ordered_language_subgoal_to_object_role_binding',
                'zero_initialized_action_facing_projections',
                'frozen_selected_global_best_parent',
            ],
            'targeted_failure_mode': (
                'relational preposition composition, workflow grounding, '
                'and unseen-object L1/L2 generalization'
            ),
            'new_parameter_leaf_count': 170,
            'new_parameter_count': 9_577_025,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_grounded_expected_missing_count,
    _best_anchor_grounded_missing_regex,
    _best_anchor_grounded_new_regex,
    _best_anchor_grounded_parent_name,
    _best_anchor_grounded_subgoal_default_parent,
    _best_anchor_grounded_subgoal_model,
    _best_anchor_grounded_subgoal_parent,
    _best_anchor_grounded_subgoal_parent_path,
)

# Conditional expert successor used only if the cumulative grounded branch is
# still below 60%.  Unlike earlier monolithic combinations, this stage freezes
# every inherited leaf and trains only a sparse top-2/8 bank.  Its router reads
# the contextual language/vision prior, proprioceptive state, noisy action and
# action position, allowing different skills to specialize without forcing a
# single residual to serve all benchmark suites.  Dense clean-action
# prediction trains the router and experts immediately; the only policy-facing
# projection is exactly zero initialized, preserving the selected
# GroundedSubgoal parent bit-for-bit at step zero.
_best_anchor_action_moe_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_grounded_subgoal_v1'
)
_best_anchor_action_moe_model = dataclasses.replace(
    _best_anchor_action_moe_parent.model,
    action_moe_reasoner=True,
    action_moe_hidden_dim=256,
    action_moe_layers=2,
    action_moe_num_heads=8,
    action_moe_mlp_dim=1024,
    action_moe_num_experts=8,
    action_moe_top_k=2,
    action_moe_expert_dim=512,
    action_moe_temperature=1.0,
    action_moe_prediction_loss_weight=0.05,
    action_moe_balance_loss_weight=0.01,
    action_moe_task_consistent_routing=True,
)
_best_anchor_action_moe_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_grounded_subgoal_v1/'
    'best_anchor_grounded_subgoal_v1_from_dual_contact_seed7/'
    '29999/params'
)
_best_anchor_action_moe_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_TASK_ROUTED_ACTION_MOE_PARENT',
    _best_anchor_action_moe_default_parent,
)
if '/pi05_vla_arena_best_anchor_grounded_subgoal_v1/' in (
    _best_anchor_action_moe_parent_path
):
    _best_anchor_action_moe_missing_regex = '.*action_moe.*'
    _best_anchor_action_moe_expected_missing_count = 85
    _best_anchor_action_moe_parent_name = (
        'X-Policy-BestAnchor-GroundedSubgoal-v1@global_full1700'
    )
elif '/pi05_vla_arena_best_anchor_dual_contact_v1/' in (
    _best_anchor_action_moe_parent_path
):
    _best_anchor_action_moe_missing_regex = (
        '.*(object_affordance|object_subgoal_binding|action_moe).*'
    )
    _best_anchor_action_moe_expected_missing_count = 255
    _best_anchor_action_moe_parent_name = (
        'X-Policy-BestAnchor-DualContact-v1@global_full1700_floor'
    )
elif any(
    family in _best_anchor_action_moe_parent_path
    for family in (
        '/pi05_vla_arena_best_anchor_joint_cadapt_v1/',
        '/pi05_vla_arena_best_anchor_language_subgoal_v1/',
    )
):
    _best_anchor_action_moe_missing_regex = (
        '.*(action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|object_affordance|'
        'object_subgoal_binding|action_moe).*'
    )
    _best_anchor_action_moe_expected_missing_count = 393
    _best_anchor_action_moe_parent_name = (
        'X-Policy-BestAnchor-SemanticStack-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_context_velocity_refiner_v1/' in (
    _best_anchor_action_moe_parent_path
):
    _best_anchor_action_moe_missing_regex = (
        '.*(language_subgoal|action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|object_affordance|'
        'object_subgoal_binding|action_moe).*'
    )
    _best_anchor_action_moe_expected_missing_count = 470
    _best_anchor_action_moe_parent_name = (
        'X-Policy-BestAnchor-VelocityRefiner-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_context_adarms_v1/' in (
    _best_anchor_action_moe_parent_path
):
    _best_anchor_action_moe_missing_regex = (
        '.*(velocity_refiner|language_subgoal|action_prior_implicit|'
        'action_prior_explicit|action_prior_guidance|contact_phase|'
        'object_affordance|object_subgoal_binding|action_moe).*'
    )
    _best_anchor_action_moe_expected_missing_count = 524
    _best_anchor_action_moe_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_contextual_dual_lora/' in (
    _best_anchor_action_moe_parent_path
):
    _best_anchor_action_moe_missing_regex = (
        '.*(context_adarms|velocity_refiner|language_subgoal|'
        'action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|object_affordance|'
        'object_subgoal_binding|action_moe).*'
    )
    _best_anchor_action_moe_expected_missing_count = 528
    _best_anchor_action_moe_parent_name = (
        'X-Policy-Contextual-Dual-LoRA@global_full1700_floor'
    )
else:
    raise ValueError(
        'unsupported TaskRoutedActionMoE parent checkpoint: '
        f'{_best_anchor_action_moe_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_action_moe_parent,
        name='pi05_vla_arena_best_anchor_task_routed_action_moe_v1',
        exp_name=(
            'best_anchor_task_routed_action_moe_v1_from_'
            'grounded_subgoal_seed7'
        ),
        model=_best_anchor_action_moe_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_action_moe_parent_path,
            missing_regex=_best_anchor_action_moe_missing_regex,
            expected_missing_count=(
                _best_anchor_action_moe_expected_missing_count
            ),
        ),
        freeze_filter=(
            _best_anchor_action_moe_model
            .get_best_anchor_task_routed_action_moe_freeze_filter()
        ),
        architecture_update_path='action_moe',
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=('action_moe',),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-TaskRoutedActionMoE-v1',
            'parent_model': _best_anchor_action_moe_parent_name,
            'training_mode': 'strict_sparse_top2_of8_action_expert_successor',
            'architecture_delta': [
                'language_vision_state_conditioned_router',
                'action_noise_invariant_chunk_consistent_top2_routing',
                'sparse_top2_of8_action_denoising_experts',
                'dense_clean_action_expert_supervision',
                'load_balancing_router_objective',
                'zero_initialized_action_token_projection',
                'frozen_selected_global_best_parent',
            ],
            'targeted_failure_mode': (
                'negative transfer between safety, distractor, relational, '
                'and long-horizon behaviors'
            ),
            'new_parameter_leaf_count': 85,
            'new_parameter_count': 4_565_800,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_action_moe_default_parent,
    _best_anchor_action_moe_expected_missing_count,
    _best_anchor_action_moe_missing_regex,
    _best_anchor_action_moe_model,
    _best_anchor_action_moe_parent,
    _best_anchor_action_moe_parent_name,
    _best_anchor_action_moe_parent_path,
)

# Cross-benchmark transfer candidate.  This remains a separate configuration
# so the active VLA-Arena trainer and its formal checkpoint contracts are
# immutable.  The 14 physical ARX-X5 joint dimensions occupy the first 14
# channels of the inherited 32-D Pi0 action interface; keeping the 10-step
# horizon preserves every checkpoint parameter shape and permits a strict
# full-tree load of the audited TaskMoE winner.
_robodojo_task_moe_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_task_routed_action_moe_v1'
)
_robodojo_task_moe_model = dataclasses.replace(
    _robodojo_task_moe_parent.model,
    active_action_dim=14,
    contact_phase_gripper_indices=(6, 13),
    contact_phase_state_scalar_indices=(6, 13),
    contact_phase_open_when_positive=True,
)
_robodojo_postfix_dataset = (
    '/path/to/workspace/RoboDojo/.cache/'
    'robodojo_data_hf_postfix_repo/data/'
    'RoboDojo_lerobot_v30_video'
)
_robodojo_task_moe_parent_params = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_task_routed_action_moe_v1/'
    'best_anchor_task_routed_action_moe_v1_from_grounded_subgoal_seed7/'
    '29999/params'
)
_CONFIGS.append(
    dataclasses.replace(
        _robodojo_task_moe_parent,
        name='pi05_robodojo_best_anchor_task_routed_action_moe_v1',
        exp_name='robodojo_task_moe_transfer_seed7_30k',
        model=_robodojo_task_moe_model,
        data=LeRobotAlohaDataConfig(
            repo_id=_robodojo_postfix_dataset,
            adapt_to_pi=False,
            assets=AssetsConfig(
                assets_dir=(
                    '/path/to/workspace/'
                    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
                    'experiments/pi05/robodojo_assets'
                ),
                asset_id='robodojo_arx_x5_v30',
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            'images': {
                                'cam_high': 'observation.images.cam_high',
                                'cam_left_wrist': (
                                    'observation.images.cam_left_wrist'
                                ),
                                'cam_right_wrist': (
                                    'observation.images.cam_right_wrist'
                                ),
                            },
                            'state': 'observation.state',
                            'actions': 'action',
                            'prompt': 'prompt',
                        }
                    )
                ]
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        auxiliary_data=None,
        auxiliary_batch_size=0,
        auxiliary_loss_weight=0.0,
        auxiliary_num_workers=0,
        auxiliary_task_balanced_sampling=False,
        auxiliary_gradient_path_allowlist=(),
        auxiliary_gradient_merge='convex',
        weight_loader=weight_loaders.ExactCheckpointWeightLoader(
            os.getenv(
                'OPENPI_ROBODOJO_TASK_MOE_PARENT',
                _robodojo_task_moe_parent_params,
            )
        ),
        freeze_filter=(
            _robodojo_task_moe_model
            .get_robodojo_task_moe_transfer_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        batch_size=32,
        gradient_accumulation_steps=1,
        num_train_steps=30_000,
        log_interval=20,
        save_interval=15_000,
        keep_period=15_000,
        task_balanced_sampling=True,
        suite_balanced_sampling=False,
        checkpoint_base_dir=(
            '/path/to/workspace/'
            'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
            'experiments/pi05/robodojo_checkpoints'
        ),
        assets_base_dir=(
            '/path/to/workspace/'
            'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
            'experiments/pi05/robodojo_assets'
        ),
        fsdp_devices=1,
        seed=7,
        policy_metadata={
            'model_name': 'X-Policy-TaskMoE-RoboDojo-v1',
            'parent_model': 'X-Policy-BestAnchor-TaskRoutedActionMoE-v1',
            'source_formal_step': 29_999,
            'source_formal_cell_mean_sr': 0.5227272727272727,
            'robot': 'arx_x5',
            'action_type': 'joint_absolute_with_training_delta_transform',
            'physical_action_dim': 14,
            'internal_action_dim': 32,
            'action_horizon': 10,
            'training_steps': 30_000,
            'dataset_snapshot': 'post_2026_09_16_observation_fix',
        },
        resume=False,
        overwrite=False,
        wandb_enabled=False,
    )
)
del (
    _robodojo_postfix_dataset,
    _robodojo_task_moe_model,
    _robodojo_task_moe_parent,
    _robodojo_task_moe_parent_params,
)

# Final closed-loop fallback for the long-horizon gap.  The selected global
# best parent is embedded in the complete cumulative graph and augmented with
# eight fast/slow recurrent memory tokens.  A scalar policy boundary starts at
# exact zero, so the inherited policy is unchanged at step zero; sequence
# supervision trains only the 175 new persistent-memory leaves.
_best_anchor_closed_loop_memory_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_task_routed_action_moe_v1'
)
_best_anchor_closed_loop_memory_model = dataclasses.replace(
    _best_anchor_closed_loop_memory_parent.model,
    main_flow_samples=1,
    explicit_action_reasoner_flow_samples=2,
    persistent_subgoal_memory=True,
    persistent_memory_tokens=8,
    persistent_memory_hidden_dim=256,
    persistent_memory_subgoal_slots=8,
    persistent_memory_fast_tokens=4,
    persistent_memory_fast_update_rate=0.50,
    persistent_memory_slow_update_rate=0.05,
    persistent_memory_previous_action_steps=5,
    persistent_memory_short_replans=4,
    persistent_memory_long_replans=8,
    persistent_memory_long_probability=0.50,
    persistent_memory_cache_refresh_steps=1000,
    persistent_memory_max_staleness_steps=1000,
    persistent_memory_policy_gain=0.0,
    persistent_memory_bounded_policy_gain=True,
    persistent_memory_policy_gain_warmup_steps=3000,
    persistent_memory_factor_attention_alignment_loss_weight=0.01,
    persistent_memory_object_slot_reconstruction_loss_weight=0.01,
    supervised_role_identity_contrastive_loss=True,
)
_best_anchor_closed_loop_memory_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_task_routed_action_moe_v1/'
    'best_anchor_task_routed_action_moe_v1_from_grounded_subgoal_seed7/'
    '29999/params'
)
_best_anchor_closed_loop_memory_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_CLOSED_LOOP_MEMORY_PARENT',
    _best_anchor_closed_loop_memory_default_parent,
)
_best_anchor_closed_loop_memory_new_regex = 'persistent_memory'
if '/pi05_vla_arena_best_anchor_task_routed_action_moe_v1/' in (
    _best_anchor_closed_loop_memory_parent_path
):
    _best_anchor_closed_loop_memory_missing_regex = '.*persistent_memory.*'
    _best_anchor_closed_loop_memory_expected_missing_count = 175
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-BestAnchor-TaskRoutedActionMoE-v1@global_full1700'
    )
elif '/pi05_vla_arena_best_anchor_grounded_subgoal_v1/' in (
    _best_anchor_closed_loop_memory_parent_path
):
    _best_anchor_closed_loop_memory_missing_regex = (
        '.*(action_moe|persistent_memory).*'
    )
    _best_anchor_closed_loop_memory_expected_missing_count = 260
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-BestAnchor-GroundedSubgoal-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_dual_contact_v1/' in (
    _best_anchor_closed_loop_memory_parent_path
):
    _best_anchor_closed_loop_memory_missing_regex = (
        '.*(object_affordance|object_subgoal_binding|action_moe|'
        'persistent_memory).*'
    )
    _best_anchor_closed_loop_memory_expected_missing_count = 430
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-BestAnchor-DualContact-v1@global_full1700_floor'
    )
elif any(
    family in _best_anchor_closed_loop_memory_parent_path
    for family in (
        '/pi05_vla_arena_best_anchor_joint_cadapt_v1/',
        '/pi05_vla_arena_best_anchor_language_subgoal_v1/',
    )
):
    _best_anchor_closed_loop_memory_missing_regex = (
        '.*(action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|object_affordance|'
        'object_subgoal_binding|action_moe|persistent_memory).*'
    )
    _best_anchor_closed_loop_memory_expected_missing_count = 568
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-BestAnchor-SemanticStack-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_context_velocity_refiner_v1/' in (
    _best_anchor_closed_loop_memory_parent_path
):
    _best_anchor_closed_loop_memory_missing_regex = (
        '.*(language_subgoal|action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|object_affordance|'
        'object_subgoal_binding|action_moe|persistent_memory).*'
    )
    _best_anchor_closed_loop_memory_expected_missing_count = 645
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-BestAnchor-VelocityRefiner-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_best_anchor_context_adarms_v1/' in (
    _best_anchor_closed_loop_memory_parent_path
):
    _best_anchor_closed_loop_memory_missing_regex = (
        '.*(velocity_refiner|language_subgoal|action_prior_implicit|'
        'action_prior_explicit|action_prior_guidance|contact_phase|'
        'object_affordance|object_subgoal_binding|action_moe|'
        'persistent_memory).*'
    )
    _best_anchor_closed_loop_memory_expected_missing_count = 699
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-BestAnchor-ContextAdaRMS-v1@global_full1700_floor'
    )
elif '/pi05_vla_arena_contextual_dual_lora/' in (
    _best_anchor_closed_loop_memory_parent_path
):
    _best_anchor_closed_loop_memory_missing_regex = (
        '.*(context_adarms|velocity_refiner|language_subgoal|'
        'action_prior_implicit|action_prior_explicit|'
        'action_prior_guidance|contact_phase|object_affordance|'
        'object_subgoal_binding|action_moe|persistent_memory).*'
    )
    _best_anchor_closed_loop_memory_expected_missing_count = 703
    _best_anchor_closed_loop_memory_parent_name = (
        'X-Policy-Contextual-Dual-LoRA@global_full1700_floor'
    )
else:
    raise ValueError(
        'unsupported ClosedLoopMemory parent checkpoint: '
        f'{_best_anchor_closed_loop_memory_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_closed_loop_memory_parent,
        name='pi05_vla_arena_best_anchor_closed_loop_memory_v1',
        exp_name=(
            'best_anchor_closed_loop_memory_v1_from_action_moe_seed7'
        ),
        model=_best_anchor_closed_loop_memory_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_closed_loop_memory_parent_path,
            missing_regex=_best_anchor_closed_loop_memory_missing_regex,
            expected_missing_count=(
                _best_anchor_closed_loop_memory_expected_missing_count
            ),
        ),
        freeze_filter=(
            _best_anchor_closed_loop_memory_model
            .get_best_anchor_closed_loop_memory_freeze_filter()
        ),
        architecture_update_path='persistent_memory',
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=('persistent_memory',),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        persistent_sequence_training=True,
        hetm_sequence_training=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=30_000,
            decay_lr=1.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': 'X-Policy-BestAnchor-ClosedLoopMemory-v1',
            'parent_model': _best_anchor_closed_loop_memory_parent_name,
            'training_mode': (
                'strict_175_leaf_fast_slow_recurrent_memory_successor'
            ),
            'architecture_delta': [
                'eight_cross_replan_recurrent_memory_tokens',
                'four_fast_and_four_slow_memory_slots',
                'ordered_subgoal_progress_and_transition_supervision',
                'previous_five_actions_condition_memory_update',
                'zero_initialized_scalar_policy_boundary',
                'frozen_selected_global_best_parent',
            ],
            'targeted_failure_mode': (
                'closed-loop phase completion, task-workflow retention, '
                'and long-horizon L1/L2 execution'
            ),
            'sequence_training_contract': {
                'same_episode_only': True,
                'short_replans': 4,
                'long_replans': 8,
                'short_long_probability': [0.5, 0.5],
                'padded_replans': 8,
                'previous_executed_actions': 5,
                'initial_state': 'explicit_zero_truncated_bptt',
                'auxiliary_stream': 'ordinary_single_frame_policy_loss',
            },
            'new_parameter_leaf_count': 175,
            'new_parameter_count': 9_420_152,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_closed_loop_memory_default_parent,
    _best_anchor_closed_loop_memory_expected_missing_count,
    _best_anchor_closed_loop_memory_missing_regex,
    _best_anchor_closed_loop_memory_model,
    _best_anchor_closed_loop_memory_new_regex,
    _best_anchor_closed_loop_memory_parent,
    _best_anchor_closed_loop_memory_parent_name,
    _best_anchor_closed_loop_memory_parent_path,
)

# Shape-preserving dual-arm transfer of the recurrent winner.  The policy
# predicts all 14 ARX-X5 joints, while the already-trained recurrent memory
# keeps its exact seven-channel parameter geometry and observes the symmetric
# mean of the ordered left/right action chunks.  This prevents a silent
# left-arm-only transfer and permits a strict full-tree checkpoint load.
_robodojo_closed_loop_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_closed_loop_memory_v1'
)
_robodojo_closed_loop_model = dataclasses.replace(
    _robodojo_closed_loop_parent.model,
    active_action_dim=14,
    persistent_memory_action_dim=7,
    contact_phase_gripper_indices=(6, 13),
    contact_phase_state_scalar_indices=(6, 13),
    contact_phase_open_when_positive=True,
)
_robodojo_closed_loop_dataset = (
    '/path/to/workspace/RoboDojo/.cache/'
    'robodojo_data_hf_postfix_repo/data/'
    'RoboDojo_lerobot_v30_video'
)
_robodojo_closed_loop_parent_params = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
    'experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_closed_loop_memory_v1/'
    'best_anchor_closed_loop_memory_v1_from_action_moe_seed7/'
    '15000/params'
)
_CONFIGS.append(
    dataclasses.replace(
        _robodojo_closed_loop_parent,
        name='pi05_robodojo_best_anchor_closed_loop_memory_v1',
        exp_name='robodojo_closed_loop_memory_transfer_seed7_30k',
        model=_robodojo_closed_loop_model,
        data=LeRobotAlohaDataConfig(
            repo_id=_robodojo_closed_loop_dataset,
            adapt_to_pi=False,
            # RoboDojo uses unrestricted natural-language task descriptions,
            # not the audited VLA-Arena factorized-role grammar.
            factorized_prompt=False,
            assets=AssetsConfig(
                assets_dir=(
                    '/path/to/workspace/'
                    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
                    'experiments/pi05/robodojo_assets'
                ),
                asset_id='robodojo_arx_x5_v30',
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            'images': {
                                'cam_high': 'observation.images.cam_high',
                                'cam_left_wrist': (
                                    'observation.images.cam_left_wrist'
                                ),
                                'cam_right_wrist': (
                                    'observation.images.cam_right_wrist'
                                ),
                            },
                            'state': 'observation.state',
                            'actions': 'action',
                            'prompt': 'prompt',
                        }
                    )
                ]
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        auxiliary_data=None,
        auxiliary_batch_size=0,
        auxiliary_loss_weight=0.0,
        auxiliary_num_workers=0,
        auxiliary_task_balanced_sampling=False,
        auxiliary_gradient_path_allowlist=(),
        auxiliary_gradient_merge='convex',
        persistent_sequence_training=False,
        hetm_sequence_training=False,
        weight_loader=weight_loaders.ExactCheckpointWeightLoader(
            os.getenv(
                'OPENPI_ROBODOJO_CLOSED_LOOP_PARENT',
                _robodojo_closed_loop_parent_params,
            )
        ),
        freeze_filter=(
            _robodojo_closed_loop_model
            .get_robodojo_closed_loop_memory_transfer_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.0e-5,
            decay_steps=30_000,
            decay_lr=2.0e-6,
        ),
        batch_size=32,
        gradient_accumulation_steps=1,
        num_train_steps=30_000,
        log_interval=20,
        save_interval=15_000,
        keep_period=15_000,
        task_balanced_sampling=True,
        suite_balanced_sampling=False,
        checkpoint_base_dir=(
            '/path/to/workspace/'
            'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
            'experiments/pi05/robodojo_checkpoints'
        ),
        assets_base_dir=(
            '/path/to/workspace/'
            'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
            'experiments/pi05/robodojo_assets'
        ),
        fsdp_devices=1,
        seed=7,
        policy_metadata={
            'model_name': 'X-Policy-ClosedLoopMemory-RoboDojo-v1',
            'parent_model': 'X-Policy-BestAnchor-ClosedLoopMemory-v1',
            'source_formal_step': 15_000,
            'source_selection_status': (
                'provisional_until_29999_full1700_completes'
            ),
            'robot': 'arx_x5',
            'action_type': 'joint_absolute_with_training_delta_transform',
            'physical_action_dim': 14,
            'internal_action_dim': 32,
            'persistent_memory_action_dim': 7,
            'persistent_memory_dual_arm_projection': 'symmetric_mean',
            'persistent_memory_frozen_during_transfer': True,
            'action_horizon': 10,
            'training_steps': 30_000,
            'dataset_snapshot': 'post_2026_09_16_observation_fix',
        },
        resume=False,
        overwrite=False,
        wandb_enabled=False,
    )
)
del (
    _robodojo_closed_loop_dataset,
    _robodojo_closed_loop_model,
    _robodojo_closed_loop_parent,
    _robodojo_closed_loop_parent_params,
)

# Cumulative successor that preserves the complete closed-loop graph and adds
# sample-wise coordination across ContextAdaRMS, VelocityRefiner, and
# LanguageSubgoal.  Unlike the independent SpecialistFusion fallback above,
# this stage keeps the trained ActionMoE and recurrent memory in the policy
# path.  Zero router scores produce unit gates, so step zero is exactly the
# selected ClosedLoopMemory checkpoint.  Sequence training then co-adapts only
# recurrent/action specialist namespaces at conservative per-module rates.
_best_anchor_closed_loop_specialist_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_closed_loop_memory_v1'
)
_best_anchor_closed_loop_specialist_model = dataclasses.replace(
    _best_anchor_closed_loop_specialist_parent.model,
    specialist_module_router=True,
    specialist_module_router_hidden_dim=256,
    specialist_module_router_temperature=1.0,
    specialist_module_router_balance_loss_weight=0.01,
    persistent_memory_adarms=True,
    persistent_memory_adarms_hidden_dim=256,
)
_best_anchor_closed_loop_specialist_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_closed_loop_memory_v1/'
    'best_anchor_closed_loop_memory_v1_from_action_moe_seed7/29999/params'
)
_best_anchor_closed_loop_specialist_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_CLOSED_LOOP_SPECIALIST_PARENT',
    _best_anchor_closed_loop_specialist_default_parent,
)
if '/pi05_vla_arena_best_anchor_closed_loop_memory_v1/' not in (
    _best_anchor_closed_loop_specialist_parent_path
):
    raise ValueError(
        'unsupported ClosedLoopSpecialistCoAdapt parent checkpoint: '
        f'{_best_anchor_closed_loop_specialist_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_closed_loop_specialist_parent,
        name='pi05_vla_arena_best_anchor_closed_loop_specialist_coadapt_v1',
        exp_name='best_anchor_closed_loop_specialist_coadapt_v1_seed7',
        model=_best_anchor_closed_loop_specialist_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_closed_loop_specialist_parent_path,
            missing_regex=(
                '.*(specialist_module_router|persistent_memory_adarms).*'
            ),
            expected_missing_count=10,
        ),
        freeze_filter=(
            _best_anchor_closed_loop_specialist_model
            .get_best_anchor_closed_loop_specialist_coadapt_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('context_adarms', 0.10),
            ('velocity_refiner', 0.25),
            ('language_subgoal', 0.50),
            ('action_moe', 0.50),
            ('persistent_memory', 1.00),
            ('specialist_module_router', 1.00),
        ),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=(
            'context_adarms',
            'velocity_refiner',
            'language_subgoal',
            'action_moe',
            'persistent_memory',
            'specialist_module_router',
        ),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        persistent_sequence_training=True,
        hetm_sequence_training=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5.0e-6,
            decay_steps=30_000,
            decay_lr=5.0e-7,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': (
                'X-Policy-BestAnchor-ClosedLoopSpecialistCoAdapt-v1'
            ),
            'parent_model': (
                'X-Policy-BestAnchor-ClosedLoopMemory-v1@best_full1700'
            ),
            'training_mode': (
                'sequence_trained_closed_loop_specialist_coadaptation'
            ),
            'architecture_delta': [
                'preserve_complete_task_moe_and_recurrent_memory_parent',
                'context_state_conditioned_three_specialist_router',
                'three_gate_grounding_dynamics_semantic_capability_families',
                'direct_phase_conditioned_persistent_memory_adarms',
                'unit_gate_zero_score_function_preserving_initialization',
                'zero_output_memory_adarms_function_preserving_initialization',
                'batch_global_router_balance_without_per_sample_uniformity',
                'same_episode_four_eight_replan_sequence_training',
                'low_rate_context_velocity_language_action_moe_coadaptation',
                'full_rate_memory_and_router_adaptation',
                'frozen_original_contextual_dual_lora_anchor',
            ],
            'targeted_failure_mode': (
                'dynamic-distractor route instability, long-horizon memory, '
                'and cross-module negative transfer in L1/L2'
            ),
            'initial_specialist_gates': [1.0, 1.0, 1.0],
            'router_balance_loss_weight': 0.01,
            'parameter_leaf_count': 803,
            'parent_leaf_count': 793,
            'new_router_leaf_count': 6,
            'new_memory_adarms_leaf_count': 4,
            'new_memory_adarms_parameter_count': 328_960,
            'trainable_leaf_count': 405,
            'trainable_parameter_count': 21_359_364,
            'trainable_namespace_leaf_counts': {
                'context_adarms': 4,
                'velocity_refiner': 54,
                'language_subgoal': 77,
                'action_moe': 85,
                'persistent_memory': 179,
                'specialist_module_router': 6,
            },
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_closed_loop_specialist_default_parent,
    _best_anchor_closed_loop_specialist_model,
    _best_anchor_closed_loop_specialist_parent,
    _best_anchor_closed_loop_specialist_parent_path,
)

# Exact-graph successor of ClosedLoopSpecialistCoAdapt.  The dual-action,
# contact-phase, competitive object-slot, and object/subgoal-binding modules
# already exist behind zero action-facing boundaries in the cumulative graph,
# but were deliberately frozen while routing and recurrent memory learned.
# This stage opens those grounded-control branches together, while updating
# the learned recurrent stack at smaller rates.  Exact checkpoint loading
# proves that no randomly initialized leaf is added at this transition.
_best_anchor_grounded_contact_parent = next(
    config
    for config in _CONFIGS
    if config.name
    == 'pi05_vla_arena_best_anchor_closed_loop_specialist_coadapt_v1'
)
_best_anchor_grounded_contact_model = _best_anchor_grounded_contact_parent.model
_best_anchor_grounded_contact_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_closed_loop_specialist_coadapt_v1/'
    'best_anchor_closed_loop_specialist_coadapt_v1_seed7/29999/params'
)
_best_anchor_grounded_contact_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_CLOSED_LOOP_GROUNDED_CONTACT_PARENT',
    _best_anchor_grounded_contact_default_parent,
)
if '/pi05_vla_arena_best_anchor_closed_loop_specialist_coadapt_v1/' not in (
    _best_anchor_grounded_contact_parent_path
):
    raise ValueError(
        'unsupported ClosedLoopGroundedContactCoAdapt parent checkpoint: '
        f'{_best_anchor_grounded_contact_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_grounded_contact_parent,
        name=(
            'pi05_vla_arena_best_anchor_closed_loop_'
            'grounded_contact_cadapt_v1'
        ),
        exp_name='best_anchor_closed_loop_grounded_contact_cadapt_v1_seed7',
        model=_best_anchor_grounded_contact_model,
        weight_loader=weight_loaders.ExactCheckpointWeightLoader(
            params_path=_best_anchor_grounded_contact_parent_path,
        ),
        freeze_filter=(
            _best_anchor_grounded_contact_model
            .get_best_anchor_closed_loop_grounded_contact_cadapt_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('context_adarms', 0.05),
            ('velocity_refiner', 0.10),
            ('language_subgoal', 0.20),
            ('action_prior_implicit', 0.50),
            ('action_prior_explicit', 0.50),
            ('action_prior_guidance', 0.50),
            ('contact_phase', 1.00),
            ('object_affordance', 1.00),
            ('object_subgoal_binding', 1.00),
            ('action_moe', 0.25),
            ('persistent_memory', 0.50),
            ('specialist_module_router', 0.50),
        ),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=(
            'context_adarms',
            'velocity_refiner',
            'language_subgoal',
            'action_prior_implicit',
            'action_prior_explicit',
            'action_prior_guidance',
            'contact_phase',
            'object_affordance',
            'object_subgoal_binding',
            'action_moe',
            'persistent_memory',
            'specialist_module_router',
        ),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        persistent_sequence_training=True,
        hetm_sequence_training=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3.0e-6,
            decay_steps=30_000,
            decay_lr=3.0e-7,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': (
                'X-Policy-BestAnchor-ClosedLoopGroundedContactCoAdapt-v1'
            ),
            'parent_model': (
                'X-Policy-BestAnchor-ClosedLoopSpecialistCoAdapt-v1'
                '@best_full1700'
            ),
            'training_mode': (
                'exact_graph_sequence_grounded_contact_coadaptation'
            ),
            'architecture_delta': [
                'no_new_parameter_leaf_exact_parent_graph',
                'open_preinitialized_dual_action_reasoner',
                'four_state_contact_phase_supervision',
                'competitive_multiview_object_slots',
                'distinct_source_destination_subgoal_binding',
                'same_episode_four_eight_replan_sequence_training',
                'conservative_recurrent_specialist_retuning',
                'frozen_original_contextual_dual_lora_anchor',
            ],
            'targeted_failure_mode': (
                'cautious-grasp contact precision, relational composition, '
                'unseen-object grounding, and long-horizon L1/L2 execution'
            ),
            'parameter_leaf_count': 803,
            'parent_leaf_count': 803,
            'new_parameter_leaf_count': 0,
            'trainable_leaf_count': 713,
            'trainable_parameter_count': 37_345_149,
            'trainable_namespace_leaf_counts': {
                'context_adarms': 4,
                'velocity_refiner': 54,
                'language_subgoal': 77,
                'action_prior_implicit': 7,
                'action_prior_explicit': 53,
                'action_prior_guidance': 26,
                'contact_phase': 52,
                'object_affordance': 85,
                'object_subgoal_binding': 85,
                'action_moe': 85,
                'persistent_memory': 179,
                'specialist_module_router': 6,
            },
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_grounded_contact_default_parent,
    _best_anchor_grounded_contact_model,
    _best_anchor_grounded_contact_parent,
    _best_anchor_grounded_contact_parent_path,
)

# Isolated long-horizon successor of the cumulative grounded/contact policy.
# The parent route head already predicts eight subgoal slots, but has no direct
# learned comparison between its current slot and the only causally reachable
# next slot.  This four-leaf residual scores stay versus advance from current
# memory/context and the two ordered subgoal embeddings.  Its output layer is
# exactly zero initialized, so strict loading preserves the parent at step 0.
_best_anchor_causal_frontier_parent = next(
    config
    for config in _CONFIGS
    if config.name
    == 'pi05_vla_arena_best_anchor_closed_loop_grounded_contact_cadapt_v1'
)
_best_anchor_causal_frontier_model = dataclasses.replace(
    _best_anchor_causal_frontier_parent.model,
    persistent_causal_frontier_transition_gate_v1=True,
)
_best_anchor_causal_frontier_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_closed_loop_grounded_contact_cadapt_v1/'
    'best_anchor_closed_loop_grounded_contact_cadapt_v1_seed7/29999/params'
)
_best_anchor_causal_frontier_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_CAUSAL_FRONTIER_PARENT',
    _best_anchor_causal_frontier_default_parent,
)
if '/pi05_vla_arena_best_anchor_closed_loop_grounded_contact_cadapt_v1/' not in (
    _best_anchor_causal_frontier_parent_path
):
    raise ValueError(
        'unsupported CausalFrontierTransition parent checkpoint: '
        f'{_best_anchor_causal_frontier_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_causal_frontier_parent,
        name='pi05_vla_arena_best_anchor_causal_frontier_transition_v1',
        exp_name='best_anchor_causal_frontier_transition_v1_seed7',
        model=_best_anchor_causal_frontier_model,
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_causal_frontier_parent_path,
            missing_regex='.*causal_frontier_transition.*',
            expected_missing_count=4,
        ),
        freeze_filter=(
            _best_anchor_causal_frontier_model
            .get_best_anchor_causal_frontier_transition_freeze_filter()
        ),
        architecture_update_path='causal_frontier_transition',
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=('causal_frontier_transition',),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        persistent_sequence_training=True,
        hetm_sequence_training=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=30_000,
            decay_lr=1.0e-6,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': (
                'X-Policy-BestAnchor-CausalFrontierTransition-v1'
            ),
            'parent_model': (
                'X-Policy-BestAnchor-ClosedLoopGroundedContactCoAdapt-v1'
                '@best_full1700'
            ),
            'training_mode': (
                'strict_four_leaf_causal_stay_advance_transition_successor'
            ),
            'architecture_delta': [
                'explicit_current_vs_next_ordered_subgoal_comparison',
                'causal_memory_context_conditioned_stay_advance_scores',
                'reachability_preserving_current_or_next_only_routing',
                'zero_initialized_two_logit_route_residual',
                'same_episode_four_eight_replan_sequence_training',
                'frozen_cumulative_grounded_contact_parent',
            ],
            'targeted_failure_mode': (
                'premature or delayed subgoal transitions in long-horizon '
                'L1/L2 workflows and dynamic replanning'
            ),
            'parent_parameter_leaf_count': 803,
            'parameter_leaf_count': 807,
            'new_parameter_leaf_count': 4,
            'new_parameter_count': 262_914,
            'trainable_leaf_count': 4,
            'trainable_parameter_count': 262_914,
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_causal_frontier_default_parent,
    _best_anchor_causal_frontier_model,
    _best_anchor_causal_frontier_parent,
    _best_anchor_causal_frontier_parent_path,
)

# Predictive-dynamics successor to the learned causal frontier.  The new
# branches predict a compact future visual state, proprioceptive rollout and
# whole-task progress from training-only future targets.  At inference they
# consume only the current observation, executed-action history and private
# recurrent memory.  Their token outputs are zero initialized, preserving the
# causal-frontier parent while directly targeting dynamic distractors,
# obstacle motion and failed-action recovery.
_best_anchor_predictive_parent = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_causal_frontier_transition_v1'
)
_best_anchor_predictive_model = dataclasses.replace(
    _best_anchor_predictive_parent.model,
    latent_future_reasoner=True,
    latent_future_hidden_dim=256,
    latent_future_layers=2,
    latent_future_num_heads=8,
    latent_future_mlp_dim=1024,
    latent_future_grid_size=4,
    latent_future_loss_weight=0.05,
    state_rollout_reasoner=True,
    state_rollout_hidden_dim=256,
    state_rollout_layers=2,
    state_rollout_num_heads=8,
    state_rollout_mlp_dim=1024,
    state_rollout_target_dim=8,
    state_rollout_loss_weight=0.10,
    task_progress_reasoner=True,
    task_progress_hidden_dim=256,
    task_progress_layers=2,
    task_progress_num_heads=8,
    task_progress_mlp_dim=1024,
    task_progress_bins=10,
    task_progress_loss_weight=0.05,
    predictive_world_model_fusion=True,
    predictive_world_model_hidden_dim=256,
    predictive_world_model_auxiliary_scale=0.25,
    predictive_world_model_include_action_moe=True,
    predictive_world_model_reliability_loss_weight=0.02,
    predictive_world_model_router_init_scale=0.01,
)
_best_anchor_predictive_default_parent = (
    '/path/to/workspace/'
    'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/experiments/pi05/checkpoints/'
    'pi05_vla_arena_best_anchor_causal_frontier_transition_v1/'
    'best_anchor_causal_frontier_transition_v1_seed7/29999/params'
)
_best_anchor_predictive_parent_path = os.getenv(
    'OPENPI_BEST_ANCHOR_CLOSED_LOOP_PREDICTIVE_PARENT',
    _best_anchor_predictive_default_parent,
)
if '/pi05_vla_arena_best_anchor_causal_frontier_transition_v1/' not in (
    _best_anchor_predictive_parent_path
):
    raise ValueError(
        'unsupported ClosedLoopPredictiveDynamics parent checkpoint: '
        f'{_best_anchor_predictive_parent_path}'
    )
_CONFIGS.append(
    dataclasses.replace(
        _best_anchor_predictive_parent,
        name='pi05_vla_arena_best_anchor_closed_loop_predictive_dynamics_v1',
        exp_name='best_anchor_closed_loop_predictive_dynamics_v1_seed7',
        model=_best_anchor_predictive_model,
        data=dataclasses.replace(
            _best_anchor_predictive_parent.data,
            future_visual_supervision=True,
            future_state_supervision=True,
            task_progress_supervision=True,
        ),
        auxiliary_data=dataclasses.replace(
            _best_anchor_predictive_parent.auxiliary_data,
            future_visual_supervision=True,
            future_state_supervision=True,
            task_progress_supervision=True,
        ),
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_best_anchor_predictive_parent_path,
            missing_regex=(
                '.*(latent_future|state_rollout|task_progress|'
                'predictive_world_model).*'
            ),
            expected_missing_count=228,
        ),
        freeze_filter=(
            _best_anchor_predictive_model
            .get_best_anchor_closed_loop_predictive_dynamics_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('latent_future', 1.0),
            ('state_rollout', 1.0),
            ('task_progress', 1.0),
            ('predictive_world_model', 1.0),
        ),
        auxiliary_loss_weight=0.35,
        auxiliary_gradient_path_allowlist=(
            'latent_future',
            'state_rollout',
            'task_progress',
            'predictive_world_model',
        ),
        auxiliary_gradient_merge='target_preserving_pcgrad',
        auxiliary_target_gradient_clip_norm=1.0,
        persistent_sequence_training=True,
        hetm_sequence_training=False,
        # Target-preserving PCGrad requires one target batch per optimizer
        # update. Use a smaller true global batch instead of silently changing
        # that contract through gradient accumulation.
        batch_size=16,
        auxiliary_batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5.0e-6,
            decay_steps=30_000,
            decay_lr=5.0e-7,
        ),
        num_train_steps=30_000,
        policy_metadata={
            'model_name': (
                'X-Policy-BestAnchor-ClosedLoopPredictiveDynamics-v1'
            ),
            'parent_model': (
                'X-Policy-BestAnchor-CausalFrontierTransition-v1'
                '@best_full1700'
            ),
            'training_mode': (
                'strict_causal_memory_conditioned_predictive_dynamics_successor'
            ),
            'architecture_delta': [
                'training_only_future_visual_latent_prediction',
                'causal_proprioceptive_state_rollout',
                'whole_task_progress_expert',
                'persistent_memory_and_ordered_program_conditioned_fusion',
                'existing_action_moe_as_fourth_predictive_expert',
                'zero_initialized_action_token_output_boundaries',
                'same_episode_four_eight_replan_sequence_training',
                'frozen_causal_frontier_and_grounded_contact_parent',
            ],
            'targeted_failure_mode': (
                'dynamic distractors, moving obstacles, failed-action '
                'recovery, and long-horizon L1/L2 progress drift'
            ),
            'parent_parameter_leaf_count': 807,
            'parameter_leaf_count': 1_035,
            'new_parameter_leaf_count': 228,
            'new_parameter_count': 11_784_986,
            'trainable_leaf_count': 228,
            'trainable_parameter_count': 11_784_986,
            'trainable_namespace_leaf_counts': {
                'latent_future': 76,
                'state_rollout': 51,
                'task_progress': 71,
                'predictive_world_model': 30,
            },
            'formal_checkpoints': [15_000, 29_999],
            'formal_episodes': 1_700,
            'formal_cells': 33,
            'replan_steps': 5,
            'selector_used': False,
            'action_smoothing': False,
            'adaptive_replanning': False,
        },
        resume=False,
        overwrite=False,
    )
)
del (
    _best_anchor_predictive_default_parent,
    _best_anchor_predictive_model,
    _best_anchor_predictive_parent,
    _best_anchor_predictive_parent_path,
)

# Strict successor of the *released RoboDojo leaderboard Pi-05*, not the
# generic pi05_base checkpoint and not a VLA-Arena checkpoint.  The parent is
# the official seed-0 step-59999 artifact.  We preserve its 50-step action
# horizon and 14-D ARX-X5 interface, add the three architecture improvements
# that form our strongest audited lineage, and train their compact prerequisite
# conditioning paths jointly while freezing every released dense parameter.
_robodojo_public_pi05_root = (
    '/path/to/workspace/RoboDojo/.cache/'
    'official_public_pi05_seed0/ckpt/RoboDojo/Pi_05/'
    'RoboDojo-sim-arx_x5-joint-0/59999'
)
_robodojo_public_pi05_parent_params = os.getenv(
    'OPENPI_ROBODOJO_PUBLIC_PI05_PARENT',
    f'{_robodojo_public_pi05_root}/params',
)
_robodojo_public_pi05_dataset = (
    '/path/to/workspace/RoboDojo/.cache/'
    'robodojo_data_hf_training_view_v1/data/RoboDojo_lerobot_v30_video'
)
_robodojo_public_pi05_model = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=50,
    active_action_dim=14,
    # Match the released Pi-05 observation contract.  The continuous state is
    # still retained for the new zero-gated proprioceptive adapters.
    discrete_state_input=True,
    paligemma_variant='gemma_2b_lora',
    action_expert_variant='gemma_300m_lora',
    state_adarms=True,
    state_adarms_hidden_dim=256,
    state_action_film=True,
    state_action_film_hidden_dim=256,
    action_prior=True,
    action_prior_hidden_dim=256,
    action_prior_horizon=5,
    action_prior_loss_weight=0.10,
    action_prior_contextual=True,
    action_prior_state_conditioning=True,
    action_prior_target='endpoint',
    main_flow_samples=1,
    dual_action_reasoner=True,
    implicit_action_reasoner_layers=(3, 7, 11, 17),
    implicit_action_reasoner_layerwise_guidance=False,
    explicit_action_reasoner_hidden_dim=256,
    explicit_action_reasoner_layers=2,
    explicit_action_reasoner_num_heads=4,
    explicit_action_reasoner_mlp_dim=512,
    explicit_action_reasoner_loss_weight=0.10,
    explicit_action_reasoner_flow_samples=2,
    explicit_action_reasoner_inference_steps=4,
    language_subgoal_reasoner=True,
    language_subgoal_hidden_dim=256,
    language_subgoal_slots=8,
    language_subgoal_layers=2,
    language_subgoal_num_heads=8,
    language_subgoal_mlp_dim=1024,
    language_subgoal_temperature=1.0,
    language_subgoal_progress_loss_weight=0.05,
    language_subgoal_action_loss_weight=0.05,
    action_moe_reasoner=True,
    action_moe_hidden_dim=256,
    action_moe_layers=2,
    action_moe_num_heads=8,
    action_moe_mlp_dim=1024,
    action_moe_num_experts=8,
    action_moe_top_k=2,
    action_moe_expert_dim=512,
    action_moe_temperature=1.0,
    action_moe_prediction_loss_weight=0.05,
    action_moe_balance_loss_weight=0.01,
    action_moe_task_consistent_routing=True,
    persistent_subgoal_memory=True,
    persistent_memory_tokens=8,
    persistent_memory_hidden_dim=256,
    persistent_memory_subgoal_slots=8,
    persistent_memory_fast_tokens=4,
    persistent_memory_fast_update_rate=0.50,
    persistent_memory_slow_update_rate=0.05,
    persistent_memory_previous_action_steps=5,
    persistent_memory_action_dim=14,
    persistent_memory_short_replans=4,
    persistent_memory_long_replans=8,
    persistent_memory_long_probability=0.50,
    persistent_memory_policy_gain=0.0,
    persistent_memory_bounded_policy_gain=True,
    persistent_memory_policy_gain_warmup_steps=3_000,
    supervised_role_identity_contrastive_loss=True,
)
_CONFIGS.append(
    TrainConfig(
        name='pi05_robodojo_public_pi05_language_moe_memory_v1',
        exp_name='public_pi05_59999_language_moe_memory_seed7_30k',
        model=_robodojo_public_pi05_model,
        data=LeRobotAlohaDataConfig(
            repo_id=_robodojo_public_pi05_dataset,
            adapt_to_pi=True,
            factorized_prompt=False,
            assets=AssetsConfig(
                assets_dir=f'{_robodojo_public_pi05_root}/assets',
                asset_id='arx_x5_sim',
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            'images': {
                                'cam_high': 'observation.images.cam_high',
                                'cam_left_wrist': (
                                    'observation.images.cam_left_wrist'
                                ),
                                'cam_right_wrist': (
                                    'observation.images.cam_right_wrist'
                                ),
                            },
                            'state': 'observation.state',
                            'actions': 'action',
                            'prompt': 'prompt',
                        }
                    )
                ]
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.StrictSuccessorCheckpointWeightLoader(
            params_path=_robodojo_public_pi05_parent_params,
            missing_regex=(
                '.*(lora|state_adarms|state_film|action_prior|'
                'language_subgoal|action_moe|persistent_memory).*'
            ),
            expected_missing_count=462,
        ),
        freeze_filter=(
            _robodojo_public_pi05_model
            .get_robodojo_public_pi05_language_moe_memory_freeze_filter()
        ),
        architecture_update_path=None,
        architecture_update_multiplier=1.0,
        architecture_update_multipliers=(
            ('state_adarms', 1.0),
            ('state_film', 1.0),
            ('action_prior', 1.0),
            ('lora', 1.0),
            ('language_subgoal', 1.0),
            ('action_moe', 1.0),
            ('persistent_memory', 1.0),
        ),
        auxiliary_data=None,
        auxiliary_batch_size=0,
        auxiliary_loss_weight=0.0,
        auxiliary_num_workers=0,
        auxiliary_task_balanced_sampling=False,
        auxiliary_gradient_path_allowlist=(),
        persistent_sequence_training=True,
        hetm_sequence_training=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=30_000,
            decay_lr=1.0e-6,
        ),
        batch_size=8,
        gradient_accumulation_steps=4,
        num_train_steps=30_000,
        log_interval=20,
        save_interval=15_000,
        keep_period=15_000,
        # PersistentMemoryWindowSampler already applies exact hierarchical
        # task/episode/window balancing.  A second frame-level sampler is both
        # redundant and explicitly incompatible with recurrent windows.
        task_balanced_sampling=False,
        suite_balanced_sampling=False,
        checkpoint_base_dir=(
            '/path/to/workspace/'
            'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
            'experiments/pi05/robodojo_checkpoints'
        ),
        assets_base_dir=(
            '/path/to/workspace/'
            'VLA-Arena-BestAnchor-ContextAdaRMS-v1-stage/'
            'experiments/pi05/robodojo_assets'
        ),
        fsdp_devices=4,
        seed=7,
        ema_decay=None,
        wandb_enabled=False,
        policy_metadata={
            'model_name': 'X-Policy-Public-Pi05-LanguageMoEMemory-v1',
            'parent_model': 'RoboDojo-Pi-05-seed0@59999',
            'parent_source': 'RoboDojo-Benchmark/RoboDojo',
            'parent_source_revision': (
                '91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd'
            ),
            'parent_path': (
                'ckpt/RoboDojo/Pi_05/'
                'RoboDojo-sim-arx_x5-joint-0/59999'
            ),
            'training_mode': (
                'strict_public_pi05_462_leaf_zero_boundary_successor'
            ),
            'architecture_delta': [
                'state_conditioned_adarms_and_action_film',
                'contextual_dual_action_prior',
                'eight_ordered_language_subgoals',
                'task_consistent_sparse_top2_of8_action_moe',
                'eight_slot_fast_slow_closed_loop_memory',
                'zero_initialized_policy_boundaries',
                'frozen_released_dense_pi05',
            ],
            'parent_parameter_leaf_count': 51,
            'parameter_leaf_count': 513,
            'new_parameter_leaf_count': 462,
            'new_parameter_count': 74_073_372,
            'physical_action_dim': 14,
            'action_horizon': 50,
            'training_steps': 30_000,
            'formal_checkpoints': [15_000, 29_999],
            'replan_steps': 5,
            'action_smoothing': False,
            'adaptive_replanning': False,
            'dataset_snapshot': 'post_2026_09_16_observation_fix',
        },
        resume=False,
        overwrite=False,
    )
)
_robodojo_public_pi05_legacy_config = _CONFIGS[-1]
_robodojo_public_pi05_corrected_metadata = dict(
    _robodojo_public_pi05_legacy_config.policy_metadata
)
_robodojo_public_pi05_corrected_metadata.update(
    {
        'model_name': 'X-Policy-Public-Pi05-LanguageMoEMemory-v2',
        'training_mode': (
            'strict_public_pi05_native_arx_x5_462_leaf_zero_boundary_successor'
        ),
        'adapter_contract': 'native_arx_x5_no_aloha_pi_remap',
        'corrects': (
            'v1 incorrectly enabled the Aloha-to-PI joint/gripper remap even '
            'though the released RoboDojo Pi0.5 and ARX-X5 dataset use the '
            'native action space'
        ),
    }
)
_CONFIGS.append(
    dataclasses.replace(
        _robodojo_public_pi05_legacy_config,
        name='pi05_robodojo_public_pi05_language_moe_memory_v2',
        exp_name='public_pi05_59999_language_moe_memory_native_arx_seed7_30k',
        data=dataclasses.replace(
            _robodojo_public_pi05_legacy_config.data,
            adapt_to_pi=False,
        ),
        policy_metadata=_robodojo_public_pi05_corrected_metadata,
        wandb_enabled=True,
    )
)
del (
    _robodojo_public_pi05_dataset,
    _robodojo_public_pi05_corrected_metadata,
    _robodojo_public_pi05_legacy_config,
    _robodojo_public_pi05_model,
    _robodojo_public_pi05_parent_params,
    _robodojo_public_pi05_root,
)

# Stable public entry point. The research registry above is retained so the
# cumulative architecture can be reconstructed exactly, while users can refer
# to the released policy without depending on an internal experiment name.
_x_policy_vla_arena = next(
    config
    for config in _CONFIGS
    if config.name == 'pi05_vla_arena_best_anchor_closed_loop_memory_v1'
)
_CONFIGS.append(
    dataclasses.replace(
        _x_policy_vla_arena,
        name='x_policy_vla_arena',
        exp_name='x_policy_vla_arena',
        assets_base_dir='./assets',
        checkpoint_base_dir='./checkpoints',
        policy_metadata={
            **(_x_policy_vla_arena.policy_metadata or {}),
            'model_name': 'X-Policy',
            'released_checkpoint_step': 18_000,
        },
        resume=False,
        overwrite=False,
    )
)
del _x_policy_vla_arena

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError('Config names must be unique.')
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ''
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
