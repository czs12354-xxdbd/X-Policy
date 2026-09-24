"""Transactional causal recurrent-state cache for PSM sequence training.
Random window sampling remains suite/task/episode balanced, while every
interior window starts from a state produced by the same model at the previous
deployed five-step replan.  Cache values are stop-gradient training inputs;
they are refreshed periodically and never contain a future observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, order=True)
class MemoryCacheKey:
    episode_index: int
    anchor_frame: int


@dataclass(frozen=True)
class MemoryCacheEntry:
    source_frame: int
    memory: np.ndarray
    subgoal_probabilities: np.ndarray


@dataclass(frozen=True)
class CommittedMemoryCacheGeneration:
    """One immutable, atomically replaceable committed cache generation."""

    model_step: int
    keys: np.ndarray
    source_frames: np.ndarray
    memory: np.ndarray
    subgoal_probabilities: np.ndarray


class CausalRecurrentStateCache:
    """Atomically publish a complete, versioned set of causal memory states."""

    def __init__(
        self,
        *,
        memory_tokens: int,
        model_dim: int,
        subgoal_count: int,
        replan_steps: int,
        memory_dtype: np.dtype = np.dtype(np.float32),
        probability_dtype: np.dtype = np.dtype(np.float32),
    ) -> None:
        if min(memory_tokens, model_dim, subgoal_count, replan_steps) < 1:
            raise ValueError("cache dimensions and replan_steps must be positive")
        if subgoal_count < 2:
            raise ValueError("subgoal_count must be at least two")
        self.memory_tokens = memory_tokens
        self.model_dim = model_dim
        self.subgoal_count = subgoal_count
        self.replan_steps = replan_steps
        self.memory_dtype = np.dtype(memory_dtype)
        self.probability_dtype = np.dtype(probability_dtype)
        if not (
            np.issubdtype(self.memory_dtype, np.floating)
            or self.memory_dtype.name == "bfloat16"
        ):
            raise ValueError("memory_dtype must be floating point")
        if not (
            np.issubdtype(self.probability_dtype, np.floating)
            or self.probability_dtype.name == "bfloat16"
        ):
            raise ValueError("probability_dtype must be floating point")
        # Both committed and unpublished generations use compact arrays.  A
        # refresh fills a preallocated staging generation and atomically swaps
        # one immutable snapshot after every declared key is present.
        self._generation: CommittedMemoryCacheGeneration | None = None
        self._staging_model_step: int | None = None
        self._staging_keys = np.empty((0, 2), dtype=np.int32)
        self._staging_source_frames = np.empty((0,), dtype=np.int32)
        self._staging_memory = np.empty(
            (0, memory_tokens, model_dim), dtype=self.memory_dtype
        )
        self._staging_probabilities = np.empty(
            (0, subgoal_count), dtype=self.probability_dtype
        )
        self._staging_present = np.empty((0,), dtype=np.bool_)

    @staticmethod
    def _find_key_index(keys: np.ndarray, key: MemoryCacheKey) -> int | None:
        episode_left = int(
            np.searchsorted(keys[:, 0], key.episode_index, side="left")
        )
        episode_right = int(
            np.searchsorted(keys[:, 0], key.episode_index, side="right")
        )
        if episode_left == episode_right:
            return None
        frames = keys[episode_left:episode_right, 1]
        relative = int(np.searchsorted(frames, key.anchor_frame, side="left"))
        if relative >= len(frames) or int(frames[relative]) != key.anchor_frame:
            return None
        return episode_left + relative

    def _clear_staging(self) -> None:
        self._staging_model_step = None
        self._staging_keys = np.empty((0, 2), dtype=np.int32)
        self._staging_source_frames = np.empty((0,), dtype=np.int32)
        self._staging_memory = np.empty(
            (0, self.memory_tokens, self.model_dim), dtype=self.memory_dtype
        )
        self._staging_probabilities = np.empty(
            (0, self.subgoal_count), dtype=self.probability_dtype
        )
        self._staging_present = np.empty((0,), dtype=np.bool_)

    def begin_refresh(
        self, *, model_step: int, required_keys: Iterable[MemoryCacheKey]
    ) -> None:
        if model_step < 0:
            raise ValueError("model_step must be non-negative")
        required = tuple(required_keys)
        if not required:
            raise ValueError("required_keys must be non-empty")
        if len(set(required)) != len(required):
            raise ValueError("required_keys must not contain duplicates")
        int32_max = np.iinfo(np.int32).max
        if any(
            key.episode_index < 0
            or key.anchor_frame < 0
            or key.episode_index > int32_max
            or key.anchor_frame > int32_max
            for key in required
        ):
            raise ValueError("cache keys must be non-negative")
        ordered = sorted(required)
        self._staging_model_step = model_step
        self._staging_keys = np.asarray(
            [(key.episode_index, key.anchor_frame) for key in ordered],
            dtype=np.int32,
        )
        entry_count = len(ordered)
        self._staging_source_frames = np.empty((entry_count,), dtype=np.int32)
        self._staging_memory = np.empty(
            (entry_count, self.memory_tokens, self.model_dim),
            dtype=self.memory_dtype,
        )
        self._staging_probabilities = np.empty(
            (entry_count, self.subgoal_count), dtype=self.probability_dtype
        )
        self._staging_present = np.zeros((entry_count,), dtype=np.bool_)

    def stage(
        self,
        key: MemoryCacheKey,
        *,
        source_frame: int,
        memory: np.ndarray,
        subgoal_probabilities: np.ndarray,
    ) -> None:
        if self._staging_model_step is None:
            raise RuntimeError("begin_refresh must be called before stage")
        index = self._find_key_index(self._staging_keys, key)
        if index is None:
            raise KeyError("cache key was not declared for this refresh")
        if self._staging_present[index]:
            raise RuntimeError("cache key was staged more than once")
        expected_source = (
            -1 if key.anchor_frame == 0 else key.anchor_frame - self.replan_steps
        )
        if source_frame != expected_source:
            raise ValueError("cache source is not the immediately preceding replan")
        memory_value = np.asarray(memory, dtype=self.memory_dtype)
        probability_value = np.asarray(
            subgoal_probabilities, dtype=self.probability_dtype
        )
        if memory_value.shape != (self.memory_tokens, self.model_dim):
            raise ValueError("cached memory shape is invalid")
        if probability_value.shape != (self.subgoal_count,):
            raise ValueError("cached subgoal probability shape is invalid")
        if not np.isfinite(memory_value).all() or not np.isfinite(
            probability_value
        ).all():
            raise ValueError("cache values must be finite")
        if np.any(probability_value < 0) or not np.isclose(
            np.sum(probability_value, dtype=np.float64), 1.0, atol=1.0e-6
        ):
            raise ValueError("cached subgoal probabilities must be normalized")
        if key.anchor_frame == 0:
            expected_probabilities = np.zeros(
                (self.subgoal_count,), dtype=self.probability_dtype
            )
            expected_probabilities[0] = 1.0
            if np.any(memory_value != 0) or not np.array_equal(
                probability_value, expected_probabilities
            ):
                raise ValueError("episode-start cache state must be exact reset state")
        self._staging_source_frames[index] = source_frame
        self._staging_memory[index] = memory_value
        self._staging_probabilities[index] = probability_value
        self._staging_present[index] = True

    def commit_refresh(self) -> None:
        if self._staging_model_step is None:
            raise RuntimeError("no cache refresh is active")
        missing_count = int(np.count_nonzero(~self._staging_present))
        if missing_count:
            raise RuntimeError(
                f"cache refresh is incomplete: {missing_count} keys missing"
            )
        for value in (
            self._staging_keys,
            self._staging_source_frames,
            self._staging_memory,
            self._staging_probabilities,
        ):
            value.setflags(write=False)
        generation = CommittedMemoryCacheGeneration(
            model_step=self._staging_model_step,
            keys=self._staging_keys,
            source_frames=self._staging_source_frames,
            memory=self._staging_memory,
            subgoal_probabilities=self._staging_probabilities,
        )
        # CPython reference replacement is atomic.  A reader snapshots this
        # one pointer, so it cannot mix arrays from two generations.
        self._generation = generation
        self._clear_staging()

    def abort_refresh(self) -> None:
        self._clear_staging()

    def get(
        self,
        key: MemoryCacheKey,
        *,
        training_step: int,
        max_staleness_steps: int,
    ) -> MemoryCacheEntry:
        generation = self._generation
        if generation is None:
            raise RuntimeError("no committed cache generation is available")
        if max_staleness_steps < 0:
            raise ValueError("max_staleness_steps must be non-negative")
        age = training_step - generation.model_step
        if age < 0:
            raise ValueError("cache generation is from a future model step")
        if age > max_staleness_steps:
            raise RuntimeError("causal recurrent-state cache is stale")
        index = self._find_key_index(generation.keys, key)
        if index is None:
            raise KeyError("cache key is absent from the committed generation")
        return MemoryCacheEntry(
            source_frame=int(generation.source_frames[index]),
            memory=generation.memory[index].copy(),
            subgoal_probabilities=generation.subgoal_probabilities[index].copy(),
        )

    def export_generation(self) -> dict[str, np.ndarray | int]:
        """Return one compact, deterministic checkpoint payload."""
        generation = self._generation
        if generation is None or not len(generation.keys):
            raise RuntimeError("no committed cache generation is available")
        return {
            "model_step": generation.model_step,
            "keys": generation.keys.copy(),
            "source_frames": generation.source_frames.copy(),
            "memory": generation.memory.copy(),
            "subgoal_probabilities": generation.subgoal_probabilities.copy(),
        }

    def restore_generation(
        self, payload: dict[str, np.ndarray | int]
    ) -> None:
        """Atomically restore a compact checkpoint payload after full validation."""
        try:
            model_step = int(payload["model_step"])
            keys = np.asarray(payload["keys"])
            sources = np.asarray(payload["source_frames"])
            memories = np.asarray(payload["memory"])
            probabilities = np.asarray(payload["subgoal_probabilities"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cache checkpoint payload is incomplete") from error
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError("cache checkpoint keys must be [entry, 2]")
        entry_count = keys.shape[0]
        if (
            sources.shape != (entry_count,)
            or memories.shape
            != (entry_count, self.memory_tokens, self.model_dim)
            or probabilities.shape != (entry_count, self.subgoal_count)
        ):
            raise ValueError("cache checkpoint arrays have inconsistent shapes")
        cache_keys = tuple(
            MemoryCacheKey(int(episode), int(frame)) for episode, frame in keys
        )
        if len(set(cache_keys)) != entry_count:
            raise ValueError("cache checkpoint contains duplicate keys")
        self.begin_refresh(model_step=model_step, required_keys=cache_keys)
        try:
            for index, key in enumerate(cache_keys):
                self.stage(
                    key,
                    source_frame=int(sources[index]),
                    memory=memories[index],
                    subgoal_probabilities=probabilities[index],
                )
            self.commit_refresh()
        except BaseException:
            self.abort_refresh()
            raise

    @property
    def committed_model_step(self) -> int | None:
        return None if self._generation is None else self._generation.model_step

    @property
    def committed_entry_count(self) -> int:
        return 0 if self._generation is None else len(self._generation.keys)

    @property
    def committed_storage_is_contiguous(self) -> bool:
        generation = self._generation
        if generation is None:
            return True
        return all(
            value.flags.c_contiguous
            for value in (
                generation.keys,
                generation.source_frames,
                generation.memory,
                generation.subgoal_probabilities,
            )
        )

    @property
    def committed_keys_are_sorted(self) -> bool:
        generation = self._generation
        if generation is None or len(generation.keys) < 2:
            return True
        previous = generation.keys[:-1]
        following = generation.keys[1:]
        return bool(
            np.all(
                (following[:, 0] > previous[:, 0])
                | (
                    (following[:, 0] == previous[:, 0])
                    & (following[:, 1] > previous[:, 1])
                )
            )
        )

    @property
    def committed_generation_is_single_snapshot(self) -> bool:
        return self._generation is not None and isinstance(
            self._generation, CommittedMemoryCacheGeneration
        )

    @property
    def committed_storage_is_read_only(self) -> bool:
        generation = self._generation
        return generation is not None and all(
            not value.flags.writeable
            for value in (
                generation.keys,
                generation.source_frames,
                generation.memory,
                generation.subgoal_probabilities,
            )
        )

    @property
    def staging_storage_is_contiguous(self) -> bool:
        return all(
            value.flags.c_contiguous
            for value in (
                self._staging_keys,
                self._staging_source_frames,
                self._staging_memory,
                self._staging_probabilities,
                self._staging_present,
            )
        )
