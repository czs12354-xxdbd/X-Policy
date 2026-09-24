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

import logging
import pathlib
import time
from collections.abc import Sequence
from typing import Any, TypeAlias
from typing_extensions import override

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi_client import base_policy as _base_policy


BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = 'cpu',
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_with_private_state = (
                nnx_utils.module_jit(model.sample_actions_with_persistent_state)
                if hasattr(model, 'sample_actions_with_persistent_state')
                else None
            )
            self._sample_actions_with_persistent_hetm_state = (
                nnx_utils.module_jit(
                    model.sample_actions_with_persistent_hetm_state
                )
                if (
                    hasattr(model, 'persistent_memory')
                    and hasattr(model, 'hetm')
                    and not hasattr(model, 'racg')
                )
                else None
            )
            self._sample_actions_with_persistent_hetm_racg_state = (
                nnx_utils.module_jit(
                    model.sample_actions_with_persistent_hetm_racg_state
                )
                if (
                    hasattr(model, 'persistent_memory')
                    and hasattr(model, 'hetm')
                    and hasattr(model, 'racg')
                )
                else None
            )
            self._rng = rng or jax.random.key(0)

    def infer_with_persistent_hetm_private_state(
        self,
        obs: dict,
        *,
        persistent_memory: np.ndarray,
        persistent_subgoal_frontier: np.ndarray,
        hetm_state: dict[str, np.ndarray],
        previous_actions: np.ndarray,
        previous_actions_valid: bool,
        episode_start: bool,
        noise: np.ndarray | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """Infer once and commit Direct PSM and HETM as one transaction."""
        sampler = self._sample_actions_with_persistent_hetm_state
        if self._is_pytorch_model or sampler is None:
            raise ValueError('this policy does not support joint PSM+HETM state')
        required_hetm = {
            'event_ledger',
            'event_valid',
            'event_write_index',
            'last_event_probabilities',
            'predicate_memory',
            'predicate_probabilities',
            'frontier',
        }
        if set(hetm_state) != required_hetm:
            raise ValueError('joint PSM+HETM private state fields drifted')
        forbidden = {
            'persistent_memory',
            'persistent_memory_initial_state',
            'persistent_subgoal_frontier',
            'persistent_subgoal_initial_frontier',
            'persistent_previous_actions',
            'persistent_previous_actions_valid',
            'persistent_memory_episode_start',
            'hetm_event_ledger',
            'hetm_event_valid',
            'hetm_event_write_index',
            'hetm_last_event_probabilities',
            'hetm_predicate_memory',
            'hetm_predicate_probabilities',
            'hetm_frontier',
            'hetm_previous_actions',
            'hetm_episode_start',
        }
        leaked = sorted(forbidden & set(obs))
        if leaked:
            raise ValueError(
                f'client supplied server-private PSM+HETM fields: {leaked}'
            )
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs.update(
            {
                'persistent_memory_initial_state': np.asarray(persistent_memory),
                'persistent_subgoal_initial_frontier': np.asarray(
                    persistent_subgoal_frontier
                ),
                'persistent_previous_actions': np.asarray(previous_actions),
                'persistent_previous_actions_valid': np.asarray(
                    previous_actions_valid, dtype=np.bool_
                ),
                'persistent_memory_episode_start': np.asarray(
                    episode_start, dtype=np.bool_
                ),
                'hetm_event_ledger': np.asarray(hetm_state['event_ledger']),
                'hetm_event_valid': np.asarray(hetm_state['event_valid']),
                'hetm_event_write_index': np.asarray(
                    hetm_state['event_write_index']
                ),
                'hetm_last_event_probabilities': np.asarray(
                    hetm_state['last_event_probabilities']
                ),
                'hetm_predicate_memory': np.asarray(
                    hetm_state['predicate_memory']
                ),
                'hetm_predicate_probabilities': np.asarray(
                    hetm_state['predicate_probabilities']
                ),
                'hetm_frontier': np.asarray(hetm_state['frontier']),
                'hetm_previous_actions': np.asarray(previous_actions),
                'hetm_episode_start': np.asarray(episode_start, dtype=np.bool_),
            }
        )
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            sample_kwargs['noise'] = noise[None] if noise.ndim == 2 else noise
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sampled = sampler(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time
        sampled = jax.tree.map(lambda x: np.asarray(x[0, ...]), sampled)
        active_action_dim = int(
            getattr(self._model, 'active_action_dim', sampled['actions'].shape[-1])
        )
        normalized_previous_actions = sampled['actions'][
            :5, :active_action_dim
        ]
        wire_outputs = self._output_transform(
            {'state': np.asarray(inputs['state'][0]), 'actions': sampled['actions']}
        )
        wire_outputs['policy_timing'] = {'infer_ms': model_time * 1000}
        return wire_outputs, {
            'persistent_memory': sampled['persistent_memory'],
            'persistent_subgoal_frontier': sampled[
                'persistent_subgoal_frontier'
            ],
            'hetm_state': {name: sampled[name] for name in required_hetm},
            'previous_actions': normalized_previous_actions,
        }

    def infer_with_persistent_hetm_racg_private_state(
        self,
        obs: dict,
        *,
        persistent_memory: np.ndarray,
        persistent_subgoal_frontier: np.ndarray,
        hetm_state: dict[str, np.ndarray],
        racg_state: dict[str, np.ndarray],
        previous_actions: np.ndarray,
        previous_actions_valid: bool,
        episode_start: bool,
        noise: np.ndarray | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """Infer and commit Direct PSM, HETM and RACG as one transaction."""
        sampler = self._sample_actions_with_persistent_hetm_racg_state
        if self._is_pytorch_model or sampler is None:
            raise ValueError(
                'this policy does not support joint PSM+HETM+RACG state'
            )
        required_hetm = {
            'event_ledger',
            'event_valid',
            'event_write_index',
            'last_event_probabilities',
            'predicate_memory',
            'predicate_probabilities',
            'frontier',
        }
        required_racg = {
            'target_anchor',
            'target_geometry',
            'target_anchor_valid',
        }
        if set(hetm_state) != required_hetm:
            raise ValueError('joint HETM private state fields drifted')
        if set(racg_state) != required_racg:
            raise ValueError('joint RACG private state fields drifted')
        anchor = np.asarray(racg_state['target_anchor'], dtype=np.float32)
        geometry = np.asarray(racg_state['target_geometry'], dtype=np.float32)
        anchor_valid = racg_state['target_anchor_valid']
        if (
            anchor.shape != (256,)
            or geometry.shape != (5,)
            or not np.all(np.isfinite(anchor))
            or not np.all(np.isfinite(geometry))
            or not isinstance(anchor_valid, (bool, np.bool_))
        ):
            raise ValueError('RACG private anchor is malformed or nonfinite')
        if not bool(anchor_valid) and (np.any(anchor) or np.any(geometry)):
            raise ValueError('invalid RACG private anchor must be exact zero')
        forbidden = {
            'persistent_memory',
            'persistent_memory_initial_state',
            'persistent_subgoal_frontier',
            'persistent_subgoal_initial_frontier',
            'persistent_previous_actions',
            'persistent_previous_actions_valid',
            'persistent_memory_episode_start',
            'hetm_event_ledger',
            'hetm_event_valid',
            'hetm_event_write_index',
            'hetm_last_event_probabilities',
            'hetm_predicate_memory',
            'hetm_predicate_probabilities',
            'hetm_frontier',
            'hetm_previous_actions',
            'hetm_episode_start',
            'racg_target_anchor',
            'racg_target_geometry',
            'racg_target_anchor_valid',
            'racg_episode_start',
        }
        leaked = sorted(forbidden & set(obs))
        if leaked:
            raise ValueError(
                f'client supplied server-private PSM+HETM+RACG fields: {leaked}'
            )
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs.update(
            {
                'persistent_memory_initial_state': np.asarray(
                    persistent_memory
                ),
                'persistent_subgoal_initial_frontier': np.asarray(
                    persistent_subgoal_frontier
                ),
                'persistent_previous_actions': np.asarray(previous_actions),
                'persistent_previous_actions_valid': np.asarray(
                    previous_actions_valid, dtype=np.bool_
                ),
                'persistent_memory_episode_start': np.asarray(
                    episode_start, dtype=np.bool_
                ),
                'hetm_event_ledger': np.asarray(hetm_state['event_ledger']),
                'hetm_event_valid': np.asarray(hetm_state['event_valid']),
                'hetm_event_write_index': np.asarray(
                    hetm_state['event_write_index']
                ),
                'hetm_last_event_probabilities': np.asarray(
                    hetm_state['last_event_probabilities']
                ),
                'hetm_predicate_memory': np.asarray(
                    hetm_state['predicate_memory']
                ),
                'hetm_predicate_probabilities': np.asarray(
                    hetm_state['predicate_probabilities']
                ),
                'hetm_frontier': np.asarray(hetm_state['frontier']),
                'hetm_previous_actions': np.asarray(previous_actions),
                'hetm_episode_start': np.asarray(
                    episode_start, dtype=np.bool_
                ),
                'racg_target_anchor': anchor,
                'racg_target_geometry': geometry,
                'racg_target_anchor_valid': np.asarray(
                    anchor_valid, dtype=np.bool_
                ),
                'racg_episode_start': np.asarray(
                    episode_start, dtype=np.bool_
                ),
            }
        )
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            sample_kwargs['noise'] = noise[None] if noise.ndim == 2 else noise
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sampled = sampler(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time
        sampled = jax.tree.map(lambda x: np.asarray(x[0, ...]), sampled)
        active_action_dim = int(
            getattr(
                self._model,
                'active_action_dim',
                sampled['actions'].shape[-1],
            )
        )
        normalized_previous_actions = sampled['actions'][
            :5, :active_action_dim
        ]
        wire_outputs = self._output_transform(
            {
                'state': np.asarray(inputs['state'][0]),
                'actions': sampled['actions'],
            }
        )
        wire_outputs['policy_timing'] = {'infer_ms': model_time * 1000}
        return wire_outputs, {
            'persistent_memory': sampled['persistent_memory'],
            'persistent_subgoal_frontier': sampled[
                'persistent_subgoal_frontier'
            ],
            'hetm_state': {name: sampled[name] for name in required_hetm},
            'racg_state': {
                'target_anchor': sampled['target_anchor'],
                'target_geometry': sampled['target_geometry'],
                'target_anchor_valid': bool(
                    sampled['target_anchor_valid']
                ),
            },
            'previous_actions': normalized_previous_actions,
        }

    def infer_with_private_state(
        self,
        obs: dict,
        *,
        persistent_memory: np.ndarray,
        persistent_subgoal_frontier: np.ndarray,
        previous_actions: np.ndarray,
        previous_actions_valid: bool,
        episode_start: bool,
        noise: np.ndarray | None = None,
    ) -> tuple[dict, dict[str, np.ndarray]]:
        """Infer transactionally without exposing recurrent state on the wire."""
        if self._is_pytorch_model or self._sample_actions_with_private_state is None:
            raise ValueError('this policy does not support persistent private state')
        forbidden = {
            'persistent_memory_initial_state',
            'persistent_subgoal_initial_frontier',
            'persistent_previous_actions',
            'persistent_previous_actions_valid',
            'persistent_memory_episode_start',
            'persistent_subgoal_frontier',
        }
        leaked = sorted(forbidden & set(obs))
        if leaked:
            raise ValueError(f'client supplied server-private fields: {leaked}')

        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs.update(
            {
                'persistent_memory_initial_state': np.asarray(
                    persistent_memory
                ),
                'persistent_subgoal_initial_frontier': np.asarray(
                    persistent_subgoal_frontier
                ),
                'persistent_previous_actions': np.asarray(previous_actions),
                'persistent_previous_actions_valid': np.asarray(
                    previous_actions_valid, dtype=np.bool_
                ),
                'persistent_memory_episode_start': np.asarray(
                    episode_start, dtype=np.bool_
                ),
            }
        )
        inputs = jax.tree.map(
            lambda x: jnp.asarray(x)[np.newaxis, ...], inputs
        )
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None]
            sample_kwargs['noise'] = noise
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sampled = self._sample_actions_with_private_state(
            sample_rng, observation, **sample_kwargs
        )
        model_time = time.monotonic() - start_time
        sampled = jax.tree.map(lambda x: np.asarray(x[0, ...]), sampled)
        # Recurrent training consumes the normalized model-space action chunk:
        # PersistentMemorySequenceDataset constructs previous actions only
        # after Normalize.  Preserve that exact representation privately
        # before the output transform converts the wire response back to
        # physical robot units.
        active_action_dim = int(
            getattr(self._model, 'active_action_dim', sampled['actions'].shape[-1])
        )
        normalized_previous_actions = sampled['actions'][..., :active_action_dim]
        wire_outputs = self._output_transform(
            {'state': np.asarray(inputs['state'][0]), 'actions': sampled['actions']}
        )
        wire_outputs['policy_timing'] = {'infer_ms': model_time * 1000}
        private_outputs = {
            'persistent_memory': sampled['persistent_memory'],
            'persistent_subgoal_frontier': sampled[
                'persistent_subgoal_frontier'
            ],
            'persistent_previous_actions': normalized_previous_actions,
        }
        return wire_outputs, private_outputs

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(
                lambda x: jnp.asarray(x)[np.newaxis, ...], inputs
            )
            self._rng, sample_rng_or_pytorch_device = jax.random.split(
                self._rng
            )
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(
                    self._pytorch_device
                )[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )

            if (
                noise.ndim == 2
            ):  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[
                    None, ...
                ]  # Make it (1, action_horizon, action_dim)
            sample_kwargs['noise'] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            'state': inputs['state'],
            'actions': self._sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            ),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
            )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs['policy_timing'] = {
            'infer_ms': model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f'Dumping policy records to: {record_dir}')
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {'inputs': obs, 'outputs': results}
        data = flax.traverse_util.flatten_dict(data, sep='/')

        output_path = self._record_dir / f'step_{self._record_step}'
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
