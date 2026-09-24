"""Training-only temporal wrapper for persistent-memory sequence windows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import os
import pathlib
import re
from typing import Any, SupportsIndex

import numpy as np
from openpi import transforms as openpi_transforms
from openpi.models import model as openpi_model

from openpi.training.persistent_memory_sequence_sampler import (
    SequenceWindowTable,
    authoritative_l0_suite_by_task,
    build_sequence_window_table,
    load_episode_records,
    stateless_uniform_sequence,
)
from openpi.training.persistent_memory_state_cache import (
    CausalRecurrentStateCache,
    MemoryCacheEntry,
    MemoryCacheKey,
)
from openpi.training.semantic_phase_targets import (
    SEMANTIC_PHASE_MANIFEST_FILENAME,
    SemanticPhaseEpisode,
    load_semantic_phase_manifest,
    semantic_phase_targets_for_frames,
)


REFRESH_INTERVAL_STEPS = 1000
# Canonical migration alias retained for cache/data audit compatibility.
memory_episode_start_mask = "persistent_memory_episode_start"


def _hetm_event_targets(task_kind: str, phases: np.ndarray) -> np.ndarray:
    result = np.zeros((len(phases), 8), dtype=np.float32)
    result[phases <= 1, 0] = 1.0
    result[phases == 2, 1] = 1.0
    phase3 = phases == 3
    result[phase3, 2 if task_kind == "pick_place" else 5 if task_kind == "articulate" else 1] = 1.0
    result[np.isin(phases, (4, 5)), 4] = 1.0
    phase6 = phases == 6
    if task_kind == "pick_place":
        result[phase6, 3] = 1.0
    if task_kind == "articulate":
        result[phase6, 5] = 1.0
    result[phase6, 6:8] = 1.0
    result[phases == 7, 7] = 1.0
    return result


def _hetm_predicate_targets(
    task_kind: str, phases: np.ndarray, completion_observed: bool
) -> np.ndarray:
    unknown, unsatisfied, satisfied = 0.0, 1.0, 2.0
    result = np.full((len(phases), 8), unknown, dtype=np.float32)
    result[:, 0] = satisfied
    result[:, 1] = np.where(phases >= 2, satisfied, unsatisfied)
    if task_kind == "pick_place":
        result[:, 2] = np.where(phases >= 3, satisfied, unsatisfied)
    result[:, 3] = np.where(phases >= 4, satisfied, unsatisfied)
    result[:, 4] = np.where(phases >= 5, satisfied, unsatisfied)
    result[:, 5] = np.where(phases >= 6, satisfied, unsatisfied)
    if task_kind == "articulate":
        result[:, 6] = np.where(phases >= 6, satisfied, unsatisfied)
    completion_phase = 6 if completion_observed else 7
    result[:, 7] = np.where(phases >= completion_phase, satisfied, unsatisfied)
    return result


@dataclasses.dataclass(frozen=True, order=True)
class CausalReplayAnchor:
    """One deployment-grid cache state immediately before an observation."""

    episode_table_index: int
    episode_index: int
    anchor_frame: int
    source_frame: int
    has_successor: bool

    @property
    def key(self) -> MemoryCacheKey:
        return MemoryCacheKey(self.episode_index, self.anchor_frame)


@dataclasses.dataclass(frozen=True)
class PreserveWindowMetadataTransform:
    """Pickle-safe composed transform for spawned DataLoader workers."""

    pre_model_transform: Callable[[Mapping[str, Any]], dict[str, Any]]
    model_transform: Callable[[Mapping[str, Any]], dict[str, Any]]
    metadata_keys: tuple[str, ...] = (
        "episode_index",
        "memory_window_start",
        "memory_scale_id",
    )

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        metadata = {
            key: sample[key] for key in self.metadata_keys if key in sample
        }
        transformed = self.pre_model_transform(sample)
        # Repack transforms intentionally drop unknown dataset metadata. Add
        # the audited window key back before prompt tokenization so training
        # labels are enabled once per causal sequence and never during serving.
        transformed.update(metadata)
        transformed = self.model_transform(transformed)
        transformed.update(metadata)
        return transformed


def _stack_replan_values(values: list[Any], *, path: str = "sample") -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        keys = tuple(first)
        for value in values[1:]:
            if not isinstance(value, Mapping) or tuple(value) != keys:
                raise ValueError(f"{path} mappings differ across replans")
        return {
            key: _stack_replan_values(
                [value[key] for value in values], path=f"{path}.{key}"
            )
            for key in keys
        }
    if isinstance(first, str):
        if any(value != first for value in values[1:]):
            raise ValueError(f"{path} string differs within one episode")
        return first
    if first is None:
        if any(value is not None for value in values[1:]):
            raise ValueError(f"{path} mixes None and non-None values")
        return None
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} cannot be stacked across replans") from error


class SameEpisodeReplanDataset:
    """Index a transformed frame dataset as fixed, non-padded replan sequences."""

    def __init__(
        self,
        base_dataset,
        window_table: SequenceWindowTable,
        *,
        replan_steps: int,
        action_key: str = "actions",
        frame_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        scale_id: int = 0,
        active_action_dim: int | None = None,
        causal_anchor_history: bool = False,
        subgoal_count: int = 8,
        memory_tokens: int | None = None,
        memory_hidden_dim: int | None = None,
        semantic_phase_episodes: Mapping[int, SemanticPhaseEpisode] | None = None,
        hetm_supervision: bool = False,
    ) -> None:
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        if replan_steps != window_table.frame_stride:
            raise ValueError(
                "replan_steps must equal frame_stride for executed-action alignment"
            )
        expected_frames = max(
            episode.dataset_offset + episode.length
            for episode in window_table.episodes
        )
        if len(base_dataset) < expected_frames:
            raise ValueError(
                f"base dataset has {len(base_dataset)} frames; expected at least {expected_frames}"
            )
        self._base_dataset = base_dataset
        self._table = window_table
        self._replan_steps = replan_steps
        self._action_key = action_key
        self._frame_transform = frame_transform
        if scale_id < 0:
            raise ValueError("scale_id must be non-negative")
        self._scale_id = scale_id
        if active_action_dim is not None and active_action_dim < 1:
            raise ValueError("active_action_dim must be positive when provided")
        self._active_action_dim = active_action_dim
        self._causal_anchor_history = bool(causal_anchor_history)
        if subgoal_count < 2:
            raise ValueError("subgoal_count must be at least two")
        self._subgoal_count = subgoal_count
        if (memory_tokens is None) != (memory_hidden_dim is None):
            raise ValueError(
                "memory_tokens and memory_hidden_dim must be supplied together"
            )
        if memory_tokens is not None and (
            memory_tokens < 1 or memory_hidden_dim is None or memory_hidden_dim < 1
        ):
            raise ValueError("persistent-memory state dimensions must be positive")
        self._memory_tokens = memory_tokens
        self._memory_hidden_dim = memory_hidden_dim
        self._semantic_phase_episodes = semantic_phase_episodes
        self._hetm_supervision = bool(hetm_supervision)
        if self._hetm_supervision and semantic_phase_episodes is None:
            raise ValueError("HETM supervision requires semantic phase episodes")
        if semantic_phase_episodes is not None:
            missing = [
                episode.episode_index
                for episode in window_table.episodes
                if episode.episode_index not in semantic_phase_episodes
            ]
            if missing:
                raise ValueError(
                    "semantic phase supervision is missing episode "
                    f"{missing[0]}"
                )
            for episode in window_table.episodes:
                phase_episode = semantic_phase_episodes[episode.episode_index]
                if (
                    phase_episode.length != episode.length
                    or phase_episode.task != episode.task
                ):
                    raise ValueError("semantic phase episode identity drifted")
        if (
            self._causal_anchor_history
            and window_table.start_frame_stride != replan_steps
        ):
            raise ValueError(
                "causal anchor history requires start_frame_stride == replan_steps"
            )

    @property
    def window_table(self) -> SequenceWindowTable:
        return self._table

    def __len__(self) -> int:
        return len(self._table)

    def _transformed_episode_frame(
        self, episode_table_index: int, frame: int, *, window_start: int
    ) -> Mapping[str, Any]:
        episode = self._table.episodes[episode_table_index]
        if frame < 0 or frame >= episode.length:
            raise IndexError("causal replay frame lies outside its episode")
        sample = self._base_dataset[episode.dataset_offset + frame]
        if not isinstance(sample, Mapping):
            raise TypeError("causal replay sample must be a mapping")
        if self._frame_transform is None:
            return sample
        annotated = dict(sample)
        if "episode_index" not in annotated or "frame_index" not in annotated:
            raise ValueError(
                "raw causal replay samples require episode_index and frame_index"
            )
        if int(np.asarray(annotated["episode_index"])) != episode.episode_index:
            raise ValueError("causal replay sample crossed an episode boundary")
        if int(np.asarray(annotated["frame_index"])) != frame:
            raise ValueError("causal replay sample frame index drifted")
        annotated["memory_window_start"] = np.int64(window_start)
        annotated["memory_scale_id"] = np.int64(self._scale_id)
        return self._frame_transform(annotated)

    def causal_replay_sample(self, anchor: CausalReplayAnchor) -> dict[str, Any]:
        """Return the observation and real preceding executed action prefix."""
        if anchor.episode_table_index < 0 or anchor.episode_table_index >= len(
            self._table.episodes
        ):
            raise IndexError("causal replay episode index is invalid")
        episode = self._table.episodes[anchor.episode_table_index]
        if episode.episode_index != anchor.episode_index:
            raise ValueError("causal replay episode identity drifted")
        if anchor.anchor_frame % self._replan_steps:
            raise ValueError("causal replay anchor is off the deployment grid")
        expected_source = (
            -1
            if anchor.anchor_frame == 0
            else anchor.anchor_frame - self._replan_steps
        )
        if anchor.source_frame != expected_source:
            raise ValueError("causal replay source is not the prior replan")

        current = dict(
            self._transformed_episode_frame(
                anchor.episode_table_index,
                anchor.anchor_frame,
                window_start=anchor.anchor_frame,
            )
        )
        current.pop(self._action_key, None)
        action_dim = self._active_action_dim
        if action_dim is None:
            raise ValueError("causal replay requires an explicit active action dim")
        previous_actions = np.zeros(
            (self._replan_steps, action_dim), dtype=np.float32
        )
        previous_valid = anchor.anchor_frame > 0
        if previous_valid:
            previous = self._transformed_episode_frame(
                anchor.episode_table_index,
                anchor.source_frame,
                window_start=anchor.anchor_frame,
            )
            if self._action_key not in previous:
                raise KeyError("causal replay source is missing actions")
            previous_chunk = np.asarray(
                previous[self._action_key], dtype=np.float32
            )
            if (
                previous_chunk.ndim != 2
                or previous_chunk.shape[0] < self._replan_steps
                or previous_chunk.shape[1] < action_dim
            ):
                raise ValueError("causal replay source action shape is invalid")
            previous_actions = previous_chunk[
                : self._replan_steps, :action_dim
            ].copy()
        current.update(
            {
                "persistent_previous_actions": previous_actions,
                "persistent_previous_actions_valid": np.bool_(previous_valid),
                "persistent_memory_episode_start": np.bool_(
                    anchor.anchor_frame == 0
                ),
            }
        )
        return current

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        window_index = index.__index__()
        if window_index < 0:
            window_index += len(self)
        if window_index < 0 or window_index >= len(self):
            raise IndexError("sequence window index is out of range")
        dataset_indices = self._table.replan_dataset_indices([window_index])[0]
        samples = [self._base_dataset[int(raw_index)] for raw_index in dataset_indices]
        if any(not isinstance(sample, Mapping) for sample in samples):
            raise TypeError("base sequence samples must be mappings")
        episode_id = int(self._table.episode_ids[window_index])
        episode = self._table.episodes[episode_id]
        episode_frames = self._table.replan_episode_frames([window_index])[0]
        if self._frame_transform is not None:
            transformed_samples = []
            window_start = int(self._table.start_frames[window_index])
            for sample, expected_frame in zip(
                samples, episode_frames, strict=True
            ):
                annotated = dict(sample)
                if "episode_index" not in annotated or "frame_index" not in annotated:
                    raise ValueError(
                        "raw sequence samples require episode_index and frame_index"
                    )
                if int(np.asarray(annotated["episode_index"])) != episode.episode_index:
                    raise ValueError("raw sample crossed an episode boundary")
                if int(np.asarray(annotated["frame_index"])) != int(expected_frame):
                    raise ValueError("raw sample frame_index drifted from window table")
                annotated["memory_window_start"] = np.int64(window_start)
                annotated["memory_scale_id"] = np.int64(self._scale_id)
                transformed_samples.append(self._frame_transform(annotated))
            samples = transformed_samples
        stacked = _stack_replan_values(samples)
        if not isinstance(stacked, dict):
            raise TypeError("stacked sequence sample must be a dictionary")
        if self._action_key not in stacked:
            raise KeyError(f"sequence sample is missing {self._action_key!r}")
        actions = np.asarray(stacked[self._action_key], dtype=np.float32)
        expected_replans = self._table.unroll_replans
        if actions.ndim != 3 or actions.shape[0] != expected_replans:
            raise ValueError(
                "stacked actions must be [replan, action_horizon, action_dim]"
            )
        if actions.shape[1] < self._table.action_horizon:
            raise ValueError("action chunk is shorter than the audited action_horizon")
        if actions.shape[1] < self._replan_steps:
            raise ValueError("action chunk is shorter than replan_steps")
        if (
            self._active_action_dim is not None
            and actions.shape[-1] < self._active_action_dim
        ):
            raise ValueError("actions are narrower than active_action_dim")
        memory_actions = actions[
            ..., : self._active_action_dim
        ] if self._active_action_dim is not None else actions

        previous_actions = np.zeros(
            (expected_replans, self._replan_steps, memory_actions.shape[-1]),
            dtype=np.float32,
        )
        previous_actions[1:] = memory_actions[:-1, : self._replan_steps]
        previous_valid = np.ones((expected_replans,), dtype=np.bool_)
        previous_valid[0] = False
        window_start = int(self._table.start_frames[window_index])
        if self._causal_anchor_history and window_start > 0:
            previous_dataset_index = (
                episode.dataset_offset + window_start - self._replan_steps
            )
            previous_sample = self._base_dataset[previous_dataset_index]
            if not isinstance(previous_sample, Mapping):
                raise TypeError("previous anchor sample must be a mapping")
            if self._frame_transform is not None:
                annotated = dict(previous_sample)
                if "episode_index" not in annotated or "frame_index" not in annotated:
                    raise ValueError(
                        "raw previous anchor requires episode_index and frame_index"
                    )
                if int(np.asarray(annotated["episode_index"])) != episode.episode_index:
                    raise ValueError("previous anchor crossed an episode boundary")
                if int(np.asarray(annotated["frame_index"])) != (
                    window_start - self._replan_steps
                ):
                    raise ValueError("previous anchor frame_index is not causal")
                annotated["memory_window_start"] = np.int64(window_start)
                annotated["memory_scale_id"] = np.int64(self._scale_id)
                previous_sample = self._frame_transform(annotated)
            if self._action_key not in previous_sample:
                raise KeyError(
                    f"previous anchor is missing {self._action_key!r}"
                )
            previous_chunk = np.asarray(
                previous_sample[self._action_key], dtype=np.float32
            )
            if previous_chunk.ndim != 2 or previous_chunk.shape[0] < self._replan_steps:
                raise ValueError("previous anchor action chunk is too short")
            if (
                self._active_action_dim is not None
                and previous_chunk.shape[-1] < self._active_action_dim
            ):
                raise ValueError("previous anchor actions are narrower than active_action_dim")
            previous_chunk = (
                previous_chunk[..., : self._active_action_dim]
                if self._active_action_dim is not None
                else previous_chunk
            )
            if previous_chunk.shape[-1] != memory_actions.shape[-1]:
                raise ValueError("previous and current action dimensions differ")
            previous_actions[0] = previous_chunk[: self._replan_steps]
            previous_valid[0] = True
        next_action_summary = np.mean(
            memory_actions[:, : self._replan_steps], axis=1, dtype=np.float32
        )
        episode_start_mask = np.zeros((expected_replans,), dtype=np.bool_)
        episode_start_mask[0] = window_start == 0
        next_frames = np.minimum(
            episode_frames + self._replan_steps, episode.length - 1
        )
        if self._semantic_phase_episodes is None:
            # Synthetic unit fixtures do not own a generated demonstration
            # manifest.  Production construction below always supplies one.
            progress_denominator = max(episode.length - 1, 1)
            current_progress = np.clip(
                episode_frames.astype(np.float32) / float(progress_denominator),
                0.0,
                1.0,
            )
            next_progress = np.clip(
                next_frames.astype(np.float32) / float(progress_denominator),
                0.0,
                1.0,
            )
            current_subgoal_target = np.floor(
                current_progress * float(self._subgoal_count - 1) + 0.5
            ).astype(np.int32)
            next_subgoal_target = np.floor(
                next_progress * float(self._subgoal_count - 1) + 0.5
            ).astype(np.int32)
        else:
            boundaries = self._semantic_phase_episodes[
                episode.episode_index
            ].boundaries
            current_subgoal_target, _ = semantic_phase_targets_for_frames(
                boundaries, episode_frames
            )
            next_subgoal_target, next_progress = semantic_phase_targets_for_frames(
                boundaries, next_frames
            )
        result = {
            **stacked,
            "persistent_previous_actions": previous_actions,
            "persistent_previous_actions_valid": previous_valid,
            "memory_next_action_summary_target": next_action_summary,
            "persistent_memory_sequence_valid": np.ones(
                (expected_replans,), dtype=np.bool_
            ),
            "persistent_memory_episode_start": episode_start_mask,
            "persistent_current_subgoal_target": current_subgoal_target,
            "persistent_next_subgoal_target": next_subgoal_target,
            "persistent_next_progress_target": next_progress.astype(np.float32),
            "memory_episode_index": np.int64(
                self._table.episodes[episode_id].episode_index
            ),
            "memory_episode_length": np.int64(
                self._table.episodes[episode_id].length
            ),
            "memory_replan_frame_indices": episode_frames,
            "memory_dataset_indices": dataset_indices,
            "memory_scale_id": np.int32(self._scale_id),
            "persistent_memory_initial_state_key": np.asarray(
                [episode.episode_index, window_start], dtype=np.int64
            ),
            "persistent_memory_requires_cached_initial_state": np.bool_(
                window_start > 0
            ),
        }
        # ClosedLoopMemory is trained with truncated recurrent windows.  A
        # window owns an explicit zero initial state unless a future replay
        # cache consumer replaces it; this makes the sequence contract total
        # and avoids silently falling back to the ordinary frame objective.
        if self._memory_tokens is not None:
            result.update(
                {
                    "persistent_memory_initial_state": np.zeros(
                        (self._memory_tokens, self._memory_hidden_dim),
                        dtype=np.float32,
                    ),
                    "persistent_subgoal_initial_frontier": np.zeros(
                        (self._subgoal_count,), dtype=np.float32
                    ),
                }
            )
        if self._hetm_supervision:
            semantic = self._semantic_phase_episodes[episode.episode_index]
            history_frames = np.arange(
                max(0, window_start - 16 * self._replan_steps),
                window_start,
                self._replan_steps,
                dtype=np.int64,
            )
            history_events = np.zeros((16, 8), dtype=np.float32)
            history_valid = np.zeros((16,), dtype=np.bool_)
            if history_frames.size:
                history_phases, _ = semantic_phase_targets_for_frames(
                    semantic.boundaries, history_frames
                )
                history_events[: history_frames.size] = _hetm_event_targets(
                    semantic.task_kind, history_phases
                )
                history_valid[: history_frames.size] = True
            initial_predicates = np.zeros((8,), dtype=np.float32)
            initial_frontier = np.float32(0)
            initial_state_valid = np.bool_(window_start > 0)
            if initial_state_valid:
                previous_phase, _ = semantic_phase_targets_for_frames(
                    semantic.boundaries,
                    np.asarray([window_start - self._replan_steps], dtype=np.int64),
                )
                initial_predicates = _hetm_predicate_targets(
                    semantic.task_kind,
                    previous_phase,
                    semantic.completion_observed,
                )[0]
                initial_frontier = np.float32(previous_phase[0])
            result.update(
                {
                    "hetm_previous_actions": previous_actions,
                    "hetm_episode_start": episode_start_mask,
                    "hetm_history_event_targets": history_events,
                    "hetm_history_event_valid": history_valid,
                    "hetm_initial_predicate_targets": initial_predicates,
                    "hetm_initial_frontier_target": initial_frontier,
                    "hetm_initial_state_valid": initial_state_valid,
                    "hetm_event_targets": _hetm_event_targets(
                        semantic.task_kind, current_subgoal_target
                    ),
                    "hetm_predicate_targets": _hetm_predicate_targets(
                        semantic.task_kind,
                        current_subgoal_target,
                        semantic.completion_observed,
                    ),
                    "hetm_frontier_target": current_subgoal_target.astype(
                        np.float32
                    ),
                    "hetm_next_frontier_target": next_subgoal_target.astype(
                        np.float32
                    ),
                    "hetm_supervision_valid": np.ones(
                        (expected_replans,), dtype=np.bool_
                    ),
                }
            )
            if "racg_role_valid_mask" in stacked:
                gripper = memory_actions[:, :, 6]
                prior_gripper = np.empty_like(gripper)
                prior_gripper[:, 1:] = gripper[:, :-1]
                state = np.asarray(stacked["state"], dtype=np.float32)
                initial = np.where(
                    previous_valid,
                    previous_actions[:, -1, 6],
                    np.where(state[:, 6] - state[:, 7] > 0.0, -1.0, 1.0),
                )
                prior_gripper[:, 0] = initial
                contact = np.where(gripper < 0, 0, 2)
                contact = np.where((gripper >= 0) & (prior_gripper < 0), 1, contact)
                contact = np.where((gripper < 0) & (prior_gripper >= 0), 3, contact)
                result.update(
                    {
                        "racg_episode_start": episode_start_mask,
                        "racg_contact_target": contact.astype(np.int32),
                        "racg_contact_valid": np.ones_like(contact, dtype=np.bool_),
                    }
                )
        return result


def _pad_replan_axis(value: Any, *, source_replans: int, target_replans: int) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _pad_replan_axis(
                item,
                source_replans=source_replans,
                target_replans=target_replans,
            )
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == source_replans:
        padding = [(0, target_replans - source_replans)] + [
            (0, 0)
        ] * (value.ndim - 1)
        return np.pad(value, padding, mode="constant")
    return value


class MultiScaleSameEpisodeReplanDataset:
    """Concatenate temporal scales while returning one fixed replan shape."""

    def __init__(
        self,
        datasets: Sequence[SameEpisodeReplanDataset],
        *,
        mixture_weights: Sequence[float],
    ) -> None:
        if not datasets or len(datasets) != len(mixture_weights):
            raise ValueError("datasets and mixture_weights must have equal nonzero length")
        weights = np.asarray(mixture_weights, dtype=np.float64)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("mixture_weights must be finite and positive")
        self._datasets = tuple(datasets)
        self._mixture_weights = weights / weights.sum(dtype=np.float64)
        lengths = np.asarray([len(dataset) for dataset in datasets], dtype=np.int64)
        self._offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(lengths)]
        )
        self._max_replans = max(
            dataset.window_table.unroll_replans for dataset in datasets
        )
        # Direct indexing remains fixed-shape for backwards compatibility.
        # The production sampler enables native lengths only after proving
        # that every emitted optimizer batch contains one temporal scale.
        self._native_replan_length_batches = False
        sampling = []
        for mixture_weight, dataset in zip(
            self._mixture_weights, self._datasets, strict=True
        ):
            sampling.append(
                mixture_weight * dataset.window_table.probabilities
            )
        self._sampling_weights = np.concatenate(sampling)
        self._sampling_weights /= self._sampling_weights.sum(dtype=np.float64)
        self._cumulative_sampling_weights = np.cumsum(
            self._sampling_weights, dtype=np.float64
        )
        self._cumulative_sampling_weights[-1] = 1.0
        self._window_task_names: list[np.ndarray] = []
        self._task_conditional_sampling: list[
            dict[str, tuple[np.ndarray, np.ndarray]]
        ] = []
        for dataset in self._datasets:
            table = dataset.window_table
            task_names = np.asarray(
                [table.episodes[int(episode_id)].task for episode_id in table.episode_ids],
                dtype=object,
            )
            conditional: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for task_name in np.unique(task_names):
                indices = np.flatnonzero(task_names == task_name).astype(np.int64)
                probabilities = table.probabilities[indices].astype(np.float64)
                probabilities /= probabilities.sum(dtype=np.float64)
                cumulative = np.cumsum(probabilities, dtype=np.float64)
                cumulative[-1] = 1.0
                conditional[str(task_name)] = (indices, cumulative)
            self._window_task_names.append(task_names)
            self._task_conditional_sampling.append(conditional)

    @property
    def sampling_weights(self) -> np.ndarray:
        return self._sampling_weights

    @property
    def scale_offsets(self) -> np.ndarray:
        return self._offsets

    def enable_native_replan_length_batches(self) -> None:
        """Drop padding when the caller guarantees scale-homogeneous batches."""
        self._native_replan_length_batches = True

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def sample_window_indices(
        self,
        count: int,
        *,
        seed: int,
        stream_offset: int = 0,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Draw an exactly resumable multiscale window stream.

        Production passes ``batch_size`` so every optimizer batch is temporal-
        scale homogeneous.  The scale is sampled once per batch with the same
        declared mixture mass. Consecutive items form same-task positive pairs;
        each pair first draws a balanced anchor and then an
        independent episode/window conditional on that task. Thus every item
        retains the original suite/task/episode/window marginal while every
        four-example contrastive group activates role-identity supervision.
        A homogeneous short batch also lets the model skip padded replans
        4--11 instead of paying twelve visual-prefix passes for four valid
        replans.

        Omitting ``batch_size`` retains the marginal per-sample stream used by
        lightweight callers and backwards-compatible audits.
        """
        if batch_size is None:
            uniform = stateless_uniform_sequence(
                count, seed=seed, stream_offset=stream_offset
            )
            return np.searchsorted(
                self._cumulative_sampling_weights, uniform, side="right"
            ).astype(np.int64)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if count % batch_size or stream_offset % batch_size:
            raise ValueError(
                "batch-homogeneous sampling requires batch-aligned count and offset"
            )
        if batch_size % 2:
            raise ValueError("contrastive task-paired batches require even batch_size")

        batch_count = count // batch_size
        batch_offset = stream_offset // batch_size
        scale_uniform = stateless_uniform_sequence(
            batch_count,
            seed=seed ^ 0x6A09E667F3BCC909,
            stream_offset=batch_offset,
        )
        scale_ids = np.searchsorted(
            np.cumsum(self._mixture_weights, dtype=np.float64),
            scale_uniform,
            side="right",
        ).astype(np.int64)
        pair_scale_ids = np.repeat(scale_ids, batch_size // 2)
        pair_count = count // 2
        pair_offset = stream_offset // 2
        anchor_uniform = stateless_uniform_sequence(
            pair_count,
            seed=seed ^ 0x3C6EF372FE94F82B,
            stream_offset=pair_offset,
        )
        companion_uniform = stateless_uniform_sequence(
            pair_count,
            seed=seed ^ 0x510E527FADE682D1,
            stream_offset=pair_offset,
        )
        result_pairs = np.empty((pair_count, 2), dtype=np.int64)
        for scale_index, dataset in enumerate(self._datasets):
            selected_pairs = np.flatnonzero(pair_scale_ids == scale_index)
            anchor_indices = np.searchsorted(
                dataset.window_table.cumulative_probabilities,
                anchor_uniform[selected_pairs],
                side="right",
            ).astype(np.int64)
            result_pairs[selected_pairs, 0] = (
                self._offsets[scale_index] + anchor_indices
            )
            anchor_tasks = self._window_task_names[scale_index][anchor_indices]
            conditional = self._task_conditional_sampling[scale_index]
            for task_name in np.unique(anchor_tasks):
                task_pair_positions = selected_pairs[anchor_tasks == task_name]
                task_indices, task_cumulative = conditional[str(task_name)]
                task_draws = companion_uniform[task_pair_positions]
                companion_indices = task_indices[
                    np.searchsorted(task_cumulative, task_draws, side="right")
                ]
                result_pairs[task_pair_positions, 1] = (
                    self._offsets[scale_index] + companion_indices
                )
        return result_pairs.reshape(-1)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        global_index = index.__index__()
        if global_index < 0:
            global_index += len(self)
        if global_index < 0 or global_index >= len(self):
            raise IndexError("multiscale sequence index is out of range")
        scale_index = int(np.searchsorted(self._offsets[1:], global_index, side="right"))
        local_index = global_index - int(self._offsets[scale_index])
        sample = self._datasets[scale_index][local_index]
        source_replans = self._datasets[scale_index].window_table.unroll_replans
        if self._native_replan_length_batches:
            result = dict(sample)
        else:
            result = _pad_replan_axis(
                sample,
                source_replans=source_replans,
                target_replans=self._max_replans,
            )
        result["memory_scale_id"] = np.int32(scale_index)
        result["memory_unroll_replans"] = np.int32(source_replans)
        return result


class PersistentMemorySequenceDataset(MultiScaleSameEpisodeReplanDataset):
    """Production 4/8 sequence dataset with resumable sampling and cache contract.

    Transformed frames may additionally provide
    ``factorized_role_identity_labels``; the generic stacking path preserves
    that training-only field without making it an inference input.
    """

    refresh_interval_steps = REFRESH_INTERVAL_STEPS
    cache_type = CausalRecurrentStateCache
    cache_key_type = MemoryCacheKey
    cache_entry_type = MemoryCacheEntry

    def causal_replay_layers(self) -> tuple[tuple[CausalReplayAnchor, ...], ...]:
        """Group anchors by causal depth so episodes replay in parallel."""
        short = self._datasets[0]
        table = short.window_table
        counts = np.bincount(
            table.episode_ids, minlength=len(table.episodes)
        ).astype(np.int64)
        layers: list[tuple[CausalReplayAnchor, ...]] = []
        for depth in range(int(np.max(counts))):
            layer = []
            for episode_table_index, count in enumerate(counts):
                if depth >= int(count):
                    continue
                episode = table.episodes[episode_table_index]
                anchor_frame = depth * table.start_frame_stride
                layer.append(
                    CausalReplayAnchor(
                        episode_table_index=episode_table_index,
                        episode_index=episode.episode_index,
                        anchor_frame=anchor_frame,
                        source_frame=(
                            -1
                            if anchor_frame == 0
                            else anchor_frame - table.frame_stride
                        ),
                        has_successor=depth + 1 < int(count),
                    )
                )
            layers.append(tuple(layer))
        anchors = tuple(anchor for layer in layers for anchor in layer)
        if len(anchors) != len(short):
            raise RuntimeError("causal replay schedule lost cache keys")
        if len({anchor.key for anchor in anchors}) != len(anchors):
            raise RuntimeError("causal replay schedule contains duplicate keys")
        return tuple(layers)

    def causal_replay_sample(self, anchor: CausalReplayAnchor) -> dict[str, Any]:
        return self._datasets[0].causal_replay_sample(anchor)

    def collate_causal_replay_observations(
        self, anchors: Sequence[CausalReplayAnchor]
    ) -> openpi_model.Observation:
        if not anchors:
            raise ValueError("causal replay batch must be non-empty")
        samples = [self.causal_replay_sample(anchor) for anchor in anchors]
        observation_keys = {
            "image",
            "image_mask",
            "state",
            "future_state",
            "future_state_mask",
            "future_image",
            "future_image_mask",
            "demonstration_actions",
            "demonstration_plan",
            "demonstration_mask",
            "demonstration_reliability_target",
            "demonstration_slot_mask",
            "demonstration_progress",
            "demonstration_router_target",
            "task_progress_target",
            "persistent_previous_actions",
            "persistent_previous_actions_valid",
            "persistent_memory_episode_start",
            "factorized_role_span_mask",
            "factorized_source_reference_span_mask",
            "factorized_destination_reference_span_mask",
            "factorized_condition_span_mask",
            "factorized_destination_qualifier_span_mask",
            "factorized_role_valid_mask",
            "factorized_role_identity_labels",
            "factorized_operation_label",
            "factorized_source_relation_label",
            "factorized_destination_relation_label",
            "factorized_condition_label",
            "factorized_destination_qualifier_label",
            "factorized_auxiliary_valid",
            "racg_role_span_mask",
            "racg_role_valid_mask",
            "racg_relation_kind",
            "racg_role_identity_labels",
            "racg_relation_target",
            "racg_relation_valid",
            "racg_crossview_role_valid",
            "racg_contact_target",
            "racg_contact_valid",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        }
        present = observation_keys.intersection(samples[0])
        if any(observation_keys.intersection(sample) != present for sample in samples):
            raise ValueError("causal replay observation fields differ across batch")
        batched = {
            key: _stack_replan_values(
                [sample[key] for sample in samples], path=f"replay.{key}"
            )
            for key in present
        }
        return openpi_model.Observation.from_dict(
            batched
        )


class PersistentMemoryWindowSampler:
    """Finite epochs cut from one deterministic stateless sampling stream."""

    def __init__(
        self,
        dataset: PersistentMemorySequenceDataset,
        *,
        seed: int,
        stream_offset: int,
        samples_per_epoch: int,
        batch_size: int,
    ) -> None:
        if stream_offset < 0 or samples_per_epoch < 1 or batch_size < 1:
            raise ValueError("sampler offset/count are invalid")
        if stream_offset % batch_size or samples_per_epoch % batch_size:
            raise ValueError("sampler stream must start and end on batch boundaries")
        self.dataset = dataset
        self.seed = seed
        self.stream_offset = stream_offset
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        # This stage's compute_loss_sequence contract is fixed at eight
        # replans.  Four-replan windows stay padded to eight and publish an
        # exact validity mask, so padded rows never dilute the objective.

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self):
        indices = self.dataset.sample_window_indices(
            self.samples_per_epoch,
            seed=self.seed,
            stream_offset=self.stream_offset,
            batch_size=self.batch_size,
        )
        self.stream_offset += self.samples_per_epoch
        return iter(indices.tolist())


def build_persistent_memory_sequence_dataset(
    base_dataset,
    data_config,
    model_config,
    *,
    episode_metadata: pathlib.Path | None = None,
) -> PersistentMemorySequenceDataset:
    """Build the audited equal-suite 4/8-replan production dataset."""
    persistent = bool(getattr(model_config, "persistent_subgoal_memory", False))
    hetm_enabled = bool(
        getattr(model_config, "hierarchical_event_transition_memory", False)
    )
    if not (persistent or hetm_enabled):
        raise ValueError("persistent memory or HETM must be enabled")
    previous_steps = int(
        getattr(model_config, "persistent_memory_previous_action_steps", 5)
    )
    short_replans = int(
        getattr(model_config, "persistent_memory_short_replans", 4)
    )
    long_replans = int(
        getattr(model_config, "persistent_memory_long_replans", 8)
    )
    long_probability = float(
        getattr(model_config, "persistent_memory_long_probability", 0.5)
    )
    subgoal_slots = int(
        getattr(model_config, "persistent_memory_subgoal_slots", 8)
    )
    if episode_metadata is None:
        configured_root = os.getenv("OPENPI_VLA_ARENA_DATASET_ROOT")
        dataset_root = pathlib.Path(
            configured_root if configured_root else data_config.repo_id
        ).expanduser()
        if not (dataset_root / "meta/info.json").is_file():
            lerobot_home = os.getenv("HF_LEROBOT_HOME")
            if lerobot_home:
                local_candidate = (
                    pathlib.Path(lerobot_home).expanduser() / dataset_root
                )
                if (local_candidate / "meta/info.json").is_file():
                    dataset_root = local_candidate
        jsonl_metadata = dataset_root / "meta/episodes.jsonl"
        parquet_metadata = dataset_root / "meta/episodes"
        episode_metadata = (
            jsonl_metadata
            if jsonl_metadata.is_file()
            else parquet_metadata
            if parquet_metadata.is_dir()
            else jsonl_metadata
        )
    episode_metadata = pathlib.Path(episode_metadata)
    records = load_episode_records(episode_metadata)
    metadata_root = (
        episode_metadata.parent
        if episode_metadata.name == "episodes"
        else episode_metadata.parent
    )
    semantic_manifest = metadata_root / SEMANTIC_PHASE_MANIFEST_FILENAME
    if semantic_manifest.is_file():
        semantic_phase_episodes = load_semantic_phase_manifest(
            semantic_manifest,
            records=records,
            episode_metadata_path=episode_metadata,
            replan_steps=previous_steps,
            subgoal_count=subgoal_slots,
        )
    elif hetm_enabled:
        raise FileNotFoundError(
            f"HETM sequence training requires {semantic_manifest}"
        )
    else:
        # RoboDojo publishes exact episode boundaries but not generated
        # semantic-phase labels.  Persistent memory still receives causal
        # same-episode windows and uses the audited normalized-progress
        # fallback targets below.
        semantic_phase_episodes = None
    suite_map = authoritative_l0_suite_by_task()
    for record in records:
        canonical = re.sub(
            r"_+", "_", re.sub(r"[^a-z0-9]+", "_", record.task.lower())
        ).strip("_")
        # RoboDojo is not organized into VLA-Arena suites.  Treat each of its
        # semantic tasks as a one-task suite, which reduces the hierarchical
        # sampler exactly to equal task/episode/window weighting.
        suite_map.setdefault(canonical, canonical)
    short_table = build_sequence_window_table(
        records,
        suite_by_task=suite_map,
        unroll_replans=short_replans,
        frame_stride=previous_steps,
        action_horizon=model_config.action_horizon,
        start_frame_stride=previous_steps,
    )
    minimum_long_length = (
        (long_replans - 1)
        * previous_steps
        + model_config.action_horizon
    )
    long_records = tuple(
        record for record in records if record.length >= minimum_long_length
    )
    long_table = build_sequence_window_table(
        long_records,
        suite_by_task=suite_map,
        unroll_replans=long_replans,
        frame_stride=previous_steps,
        action_horizon=model_config.action_horizon,
        start_frame_stride=previous_steps,
    )
    pre_model_transform = openpi_transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            openpi_transforms.Normalize(
                data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
        ]
    )
    model_transform = openpi_transforms.compose(
        list(data_config.model_transforms.inputs)
    )

    preserve_window_metadata = PreserveWindowMetadataTransform(
        pre_model_transform=pre_model_transform,
        model_transform=model_transform,
    )

    active_action_dim = model_config.active_action_dim or model_config.action_dim
    short = SameEpisodeReplanDataset(
        base_dataset,
        short_table,
        replan_steps=previous_steps,
        frame_transform=preserve_window_metadata,
        scale_id=0,
        active_action_dim=active_action_dim,
        causal_anchor_history=True,
        subgoal_count=subgoal_slots,
        memory_tokens=(model_config.persistent_memory_tokens if persistent else None),
        memory_hidden_dim=(
            model_config.persistent_memory_hidden_dim if persistent else None
        ),
        semantic_phase_episodes=semantic_phase_episodes,
        hetm_supervision=hetm_enabled,
    )
    long = SameEpisodeReplanDataset(
        base_dataset,
        long_table,
        replan_steps=previous_steps,
        frame_transform=preserve_window_metadata,
        scale_id=1,
        active_action_dim=active_action_dim,
        causal_anchor_history=True,
        subgoal_count=subgoal_slots,
        memory_tokens=(model_config.persistent_memory_tokens if persistent else None),
        memory_hidden_dim=(
            model_config.persistent_memory_hidden_dim if persistent else None
        ),
        semantic_phase_episodes=semantic_phase_episodes,
        hetm_supervision=hetm_enabled,
    )
    return PersistentMemorySequenceDataset(
        (short, long),
        mixture_weights=(
            1.0 - long_probability,
            long_probability,
        ),
    )
