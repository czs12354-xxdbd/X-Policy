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
import ctypes
import functools
import itertools
import logging
import multiprocessing
import os
import pathlib
import re
import signal
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.persistent_memory_data as persistent_memory_data
import openpi.transforms as _transforms
import torch
from openpi.training.droid_rlds_dataset import DroidRldsDataset

try:
    # LeRobot >=0.4, required by the official RoboDojo v3 parquet metadata.
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError as exc:
    if exc.name != 'lerobot.datasets':
        raise
    # Preserve the currently running VLA-Arena environment and v2 datasets.
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


T_co = TypeVar('T_co', covariant=True)


@functools.cache
def _resolve_video_backend() -> str:
    """Select a decoder only after proving its native dependencies load."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except (ImportError, OSError, RuntimeError) as torchcodec_error:
        try:
            import av  # noqa: F401
        except ImportError as pyav_error:
            raise RuntimeError(
                'Neither TorchCodec nor PyAV can decode LeRobot videos'
            ) from pyav_error
        logging.warning(
            'TorchCodec is unusable; falling back to PyAV: %s',
            torchcodec_error,
        )
        return 'pyav'
    return 'torchcodec'


def _resolve_local_lerobot_root(repo_id: str) -> pathlib.Path | None:
    """Resolve local repos independently of LeRobot's import-time cache root."""
    raw = pathlib.Path(repo_id).expanduser()
    candidates = [raw]
    configured_home = os.getenv('HF_LEROBOT_HOME')
    if configured_home:
        candidates.append(pathlib.Path(configured_home).expanduser() / raw)
    for parent in pathlib.Path(__file__).resolve().parents:
        candidates.append(parent / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / 'meta' / 'info.json').is_file():
            return resolved
    return None


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError(
            'Subclasses of Dataset should implement __getitem__.'
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            'Subclasses of Dataset should implement __len__.'
        )


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            'Subclasses of IterableDataset should implement __iter__.'
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            'Subclasses of Dataset should implement __len__.'
        )


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError(
            'Subclasses of DataLoader should implement data_config.'
        )

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            'Subclasses of DataLoader should implement __iter__.'
        )


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [
                    jax.tree.map(lambda x: x[i], sample)
                    for i in range(batch_size)
                ]

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(
                    lambda *x: np.stack(x, axis=0), *transformed
                )
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(
                    data_rng, shape=shape, minval=-1.0, maxval=1.0
                )
            if spec.dtype == jnp.int32:
                return jax.random.randint(
                    data_rng, shape=shape, minval=0, maxval=2048
                )
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            'actions': action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError('Repo ID is not set. Cannot create dataset.')
    if repo_id == 'fake':
        return FakeDataset(model_config, num_samples=1024)

    local_root = _resolve_local_lerobot_root(repo_id)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id, root=local_root
    )
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=local_root,
        delta_timestamps=_lerobot_delta_timestamps(
            data_config,
            action_horizon=action_horizon,
            fps=dataset_meta.fps,
        ),
        video_backend=_resolve_video_backend(),
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset


def _lerobot_delta_timestamps(
    data_config: _config.DataConfig,
    *,
    action_horizon: int,
    fps: float,
) -> dict[str, list[float]]:
    """Build every temporal query consumed by LeRobot.

    LeRobot derives both the clamped episode-local query indexes and the
    corresponding ``<key>_is_pad`` mask from ``delta_timestamps``.  Keeping
    observation offsets only in ``DataConfig`` therefore silently drops the
    future observations *and* their padding masks.  Construct the action and
    observation queries together so recurrent target streams and ordinary
    auxiliary streams have identical, episode-safe future supervision.
    """
    if action_horizon <= 0:
        raise ValueError('action_horizon must be positive')
    if fps <= 0:
        raise ValueError('LeRobot dataset fps must be positive')

    frame_offsets: dict[str, tuple[int, ...]] = {
        key: tuple(range(action_horizon))
        for key in data_config.action_sequence_keys
    }
    for key, raw_offsets in data_config.observation_sequence_offsets.items():
        if key in frame_offsets:
            raise ValueError(
                f'temporal query key {key!r} is configured as both an action '
                'and an observation sequence'
            )
        offsets = tuple(int(offset) for offset in raw_offsets)
        if not offsets:
            raise ValueError(
                f'observation sequence {key!r} must contain at least one offset'
            )
        if any(offset != raw for offset, raw in zip(offsets, raw_offsets, strict=True)):
            raise ValueError(
                f'observation sequence {key!r} contains a non-integer offset'
            )
        frame_offsets[key] = offsets

    return {
        key: [offset / fps for offset in offsets]
        for key, offsets in frame_offsets.items()
    }


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != 'fake' and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                'Normalization stats not found. '
                'Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.'
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != 'fake' and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                'Normalization stats not found. '
                'Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.'
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal['jax', 'pytorch'] = 'jax',
    completed_batches: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f'data_config: {data_config}')

    if data_config.rlds_data_dir is not None:
        if config.task_balanced_sampling or config.suite_balanced_sampling:
            raise ValueError(
                'Balanced sampling is supported only for LeRobot datasets.'
            )
        if completed_batches:
            raise NotImplementedError(
                'Exact data-stream resume is not implemented for RLDS datasets.'
            )
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        task_balanced_sampling=config.task_balanced_sampling,
        suite_balanced_sampling=config.suite_balanced_sampling,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        completed_batches=completed_batches,
        persistent_sequence_training=config.use_persistent_sequence_training,
        hetm_sequence_training=config.use_hetm_sequence_training,
    )


def auxiliary_loader_config(config: _config.TrainConfig) -> _config.TrainConfig:
    """Create a recursion-free independently normalized auxiliary stream."""
    if config.auxiliary_data is None:
        raise ValueError('auxiliary_data is not configured')
    return dataclasses.replace(
        config,
        data=config.auxiliary_data,
        batch_size=config.auxiliary_batch_size,
        num_workers=config.auxiliary_num_workers,
        task_balanced_sampling=config.auxiliary_task_balanced_sampling,
        suite_balanced_sampling=False,
        persistent_sequence_training=False,
        hetm_sequence_training=False,
        persistent_memory_static_replay_cache=None,
        persistent_memory_static_replay_binding=None,
        auxiliary_data=None,
        auxiliary_batch_size=0,
        auxiliary_loss_weight=0.0,
        auxiliary_num_workers=0,
        auxiliary_task_balanced_sampling=False,
        auxiliary_gradient_path_allowlist=(),
    )


def create_auxiliary_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: Literal['jax', 'pytorch'] = 'jax',
    completed_batches: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    return create_data_loader(
        auxiliary_loader_config(config),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        framework=framework,
        completed_batches=completed_batches,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    task_balanced_sampling: bool = False,
    suite_balanced_sampling: bool = False,
    framework: str = 'jax',
    completed_batches: int = 0,
    persistent_sequence_training: bool | None = None,
    hetm_sequence_training: bool | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    model_supports_persistent_sequences = bool(
        getattr(model_config, 'persistent_subgoal_memory', False)
    )
    use_persistent_sequences = (
        model_supports_persistent_sequences
        if persistent_sequence_training is None
        else bool(persistent_sequence_training)
    )
    model_supports_hetm_sequences = bool(
        getattr(model_config, 'hierarchical_event_transition_memory', False)
    )
    use_hetm_sequences = (
        model_supports_hetm_sequences
        if hetm_sequence_training is None
        else bool(hetm_sequence_training)
    )
    if use_persistent_sequences and not model_supports_persistent_sequences:
        raise ValueError(
            'persistent sequence loading requires a persistent-memory model'
        )
    if use_hetm_sequences and not model_supports_hetm_sequences:
        raise ValueError('HETM sequence loading requires a HETM model')
    if use_persistent_sequences or use_hetm_sequences:
        if framework == 'pytorch' and torch.distributed.is_initialized():
            raise NotImplementedError(
                'recurrent sequence datasets currently require JAX/FSDP'
            )
        if task_balanced_sampling and not suite_balanced_sampling:
            raise ValueError(
                'recurrent training owns equal suite/task/episode/window sampling'
            )
        if skip_norm_stats:
            raise ValueError(
                'recurrent production sequences require parent normalization'
            )
        dataset = persistent_memory_data.build_persistent_memory_sequence_dataset(
            dataset,
            data_config,
            model_config,
        )
        local_batch_size = (
            batch_size
            if framework == 'pytorch'
            else batch_size // jax.process_count()
        )
        samples_per_epoch = (len(dataset) // local_batch_size) * local_batch_size
        sampler = persistent_memory_data.PersistentMemoryWindowSampler(
            dataset,
            seed=seed,
            stream_offset=completed_batches * local_batch_size,
            samples_per_epoch=samples_per_epoch,
            batch_size=local_batch_size,
        )
        data_loader = TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=None if framework == 'pytorch' else sharding,
            shuffle=False,
            sampler=sampler,
            num_batches=num_batches,
            num_workers=num_workers,
            seed=seed,
            framework=framework,
        )
        return DataLoaderImpl(data_config, data_loader)
    if task_balanced_sampling and suite_balanced_sampling:
        raise ValueError(
            'task_balanced_sampling and suite_balanced_sampling are mutually exclusive'
        )
    balanced_sampler = None
    if task_balanced_sampling or suite_balanced_sampling:
        sampler_dataset = dataset
        while isinstance(sampler_dataset, TransformedDataset):
            sampler_dataset = sampler_dataset._dataset
        if not isinstance(sampler_dataset, lerobot_dataset.LeRobotDataset):
            raise ValueError(
                'Balanced sampling requires a LeRobotDataset with task_index metadata.'
            )
        task_indices = sampler_dataset.hf_dataset['task_index']
        balanced_sampler = (
            create_vla_arena_suite_balanced_sampler(
                task_indices, sampler_dataset.meta.tasks, seed=seed
            )
            if suite_balanced_sampling
            else create_task_balanced_sampler(task_indices, seed=seed)
        )
    dataset = transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats
    )

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = balanced_sampler
    if framework == 'pytorch':
        if torch.distributed.is_initialized():
            if balanced_sampler is not None:
                raise NotImplementedError(
                    'Balanced sampling with PyTorch DDP is not supported.'
                )
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    if completed_batches < 0:
        raise ValueError('completed_batches must be non-negative')
    if completed_batches and sampler is None:
        raise NotImplementedError(
            'Exact data-stream resume requires an explicit sampler.'
        )
    completed_epochs = 0
    if completed_batches:
        sampler = ResumeOffsetSampler(
            sampler,
            completed_batches=completed_batches,
            batch_size=local_batch_size,
        )
        completed_epochs = sampler.completed_epochs

    logging.info(f'local_batch_size: {local_batch_size}')
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == 'pytorch' else sharding,
        shuffle=(
            sampler is None and shuffle
        ),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        completed_epochs=completed_epochs,
    )

    return DataLoaderImpl(data_config, data_loader)


class ResumeOffsetSampler(torch.utils.data.Sampler[int]):
    """Resume a deterministic weighted sampler at a complete-batch boundary."""

    def __init__(self, sampler, *, completed_batches: int, batch_size: int):
        samples_per_epoch = len(sampler)
        batches_per_epoch = samples_per_epoch // batch_size
        if completed_batches < 0 or batch_size < 1 or batches_per_epoch < 1:
            raise ValueError('invalid sampler resume geometry')
        self._sampler = sampler
        self._completed_epochs, batches_in_epoch = divmod(
            completed_batches, batches_per_epoch
        )
        self._skip_samples = batches_in_epoch * batch_size
        self._pending = True

    @property
    def completed_epochs(self) -> int:
        return self._completed_epochs

    def __iter__(self):
        if self._pending:
            self._pending = False
            for _ in range(self._completed_epochs):
                for _ in self._sampler:
                    pass
            yield from itertools.islice(
                iter(self._sampler), self._skip_samples, None
            )
            return
        yield from self._sampler

    def __len__(self) -> int:
        if not self._pending:
            return len(self._sampler)
        remaining = len(self._sampler) - self._skip_samples
        return remaining if remaining else len(self._sampler)


def create_task_balanced_sampler(
    task_indices: Sequence[int], *, seed: int, num_samples: int | None = None
) -> torch.utils.data.WeightedRandomSampler:
    indices = torch.as_tensor(task_indices, dtype=torch.int64)
    if indices.ndim != 1 or indices.numel() == 0 or bool(torch.any(indices < 0)):
        raise ValueError('task_indices must be a non-empty non-negative vector')
    _, inverse, counts = torch.unique(
        indices, sorted=True, return_inverse=True, return_counts=True
    )
    weights = counts[inverse].to(torch.float64).reciprocal()
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=indices.numel() if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )


def _canonical_task_name(task: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', task.lower())).strip('_')


def vla_arena_task_suite_mapping(task_names: Mapping[int, str]) -> dict[int, str]:
    from vla_arena.vla_arena.benchmark.vla_arena_suite_task_map import (
        vla_arena_task_map,
    )

    task_to_suite = {
        _canonical_task_name(task): suite
        for suite, levels in vla_arena_task_map.items()
        for task in levels[0]
    }
    resolved = {}
    for raw_index, task_name in task_names.items():
        canonical = _canonical_task_name(task_name)
        if canonical not in task_to_suite:
            raise ValueError(
                f'task_index {raw_index} is not an exact VLA-Arena L0 task: {canonical}'
            )
        resolved[int(raw_index)] = task_to_suite[canonical]
    return resolved


def create_vla_arena_suite_balanced_sampler(
    task_indices: Sequence[int],
    task_names: Mapping[int, str],
    *,
    seed: int,
    num_samples: int | None = None,
) -> torch.utils.data.WeightedRandomSampler:
    indices = torch.as_tensor(task_indices, dtype=torch.int64)
    if indices.ndim != 1 or indices.numel() == 0 or bool(torch.any(indices < 0)):
        raise ValueError('task_indices must be a non-empty non-negative vector')
    task_suites = vla_arena_task_suite_mapping(task_names)
    unique_tasks, inverse, counts = torch.unique(
        indices, sorted=True, return_inverse=True, return_counts=True
    )
    suite_labels = [task_suites[int(index)] for index in unique_tasks.tolist()]
    suite_task_counts = {
        suite: suite_labels.count(suite) for suite in set(suite_labels)
    }
    per_task_mass = torch.tensor(
        [1.0 / suite_task_counts[suite] for suite in suite_labels],
        dtype=torch.float64,
    )
    weights = per_task_mass[inverse] / counts[inverse].to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=indices.numel() if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = 'jax',
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == 'pytorch':
        raise NotImplementedError(
            'PyTorch RLDS data loader is not supported yet'
        )
    dataset = create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=shuffle
    )
    dataset = transform_iterable_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True
    )

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = 'jax',
        completed_epochs: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError(
                'Data loading with multiple processes is not supported.'
            )

        if len(dataset) < local_batch_size:
            raise ValueError(
                f'Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).'
            )

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == 'jax':
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ('B',)),
                jax.sharding.PartitionSpec('B'),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context('spawn')

        generator = torch.Generator()
        generator.manual_seed(seed)
        if completed_epochs < 0:
            raise ValueError('completed_epochs must be non-negative')
        for _ in range(completed_epochs):
            torch.empty((), dtype=torch.int64).random_(generator=generator)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(
                sampler is None and shuffle
            ),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if (
                    self._num_batches is not None
                    and num_items >= self._num_batches
                ):
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(
                            self._sharding, x
                        ),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # Ask the Linux kernel to terminate this worker if its DataLoader parent
    # disappears.  Persistent workers otherwise survive an abruptly killed
    # trainer as PPID=1 processes while retaining roughly 1 GiB RSS each.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the race in which the parent exits immediately before prctl.
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError(
                'Data loading with multiple processes is not supported.'
            )

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ('B',)),
                jax.sharding.PartitionSpec('B'),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if (
                    self._num_batches is not None
                    and num_items >= self._num_batches
                ):
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(
                        self._sharding, x
                    ),
                    batch,
                )


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch['actions']
