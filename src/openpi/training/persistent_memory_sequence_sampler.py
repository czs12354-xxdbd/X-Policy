"""Deterministic same-episode sampling for persistent-memory training.

This prototype remains outside the frozen OpenPI source tree.  Each draw first
balances VLA-Arena suites, then tasks, episodes, and finally valid temporal
windows.  Consequently neither long demonstrations nor over-represented tasks
silently dominate the memory objective.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import pathlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np


def stateless_uniform_sequence(
    count: int, *, seed: int, stream_offset: int = 0
) -> np.ndarray:
    """Return a deterministic resumable SplitMix64 uniform stream."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if stream_offset < 0:
        raise ValueError("stream_offset must be non-negative")
    positions = np.arange(stream_offset, stream_offset + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = positions + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        values = values ^ (values >> np.uint64(31))
    return (values >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


def _canonical_task_name(task: str) -> str:
    return re.sub(
        r"_+", "_", re.sub(r"[^a-z0-9]+", "_", task.lower())
    ).strip("_")


def authoritative_l0_suite_by_task() -> dict[str, str]:
    # This mapping is pure data, but importing it through the benchmark package
    # executes ``benchmark/__init__.py`` and eagerly imports robosuite.  Training
    # only needs the static task names, and official Pi-05 environments need not
    # contain the simulator.  Read the literal assignment without importing any
    # benchmark runtime dependencies.
    mapping_path = (
        pathlib.Path(__file__).resolve().parents[5]
        / "vla_arena"
        / "benchmark"
        / "vla_arena_suite_task_map.py"
    )
    module = ast.parse(mapping_path.read_text(), filename=str(mapping_path))
    assignment = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "vla_arena_task_map"
                for target in node.targets
            )
        ),
        None,
    )
    if assignment is None:
        raise ValueError(f"vla_arena_task_map assignment is missing: {mapping_path}")
    vla_arena_task_map = ast.literal_eval(assignment.value)

    result: dict[str, str] = {}
    for suite, levels in vla_arena_task_map.items():
        for task in levels[0]:
            canonical = _canonical_task_name(task)
            if canonical in result:
                raise ValueError(f"duplicate authoritative L0 task: {canonical}")
            result[canonical] = suite
    return result


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task: str
    length: int
    dataset_offset: int


def load_episode_records(path: pathlib.Path) -> tuple[EpisodeRecord, ...]:
    path = pathlib.Path(path)
    if path.is_dir():
        # LeRobot v3 stores one or more parquet shards under
        # ``meta/episodes``.  Read only the four identity columns; the files
        # also contain large per-episode statistics that are irrelevant to
        # temporal window construction.
        import pyarrow.parquet as pq

        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"episode parquet directory is empty: {path}")
        payloads = []
        for parquet_file in parquet_files:
            payloads.extend(
                pq.read_table(
                    parquet_file,
                    columns=[
                        "episode_index",
                        "tasks",
                        "length",
                        "dataset_from_index",
                    ],
                ).to_pylist()
            )
        payloads.sort(key=lambda payload: int(payload["episode_index"]))
        records: list[EpisodeRecord] = []
        dataset_offset = 0
        for payload in payloads:
            episode_index = int(payload["episode_index"])
            tasks = payload["tasks"]
            length = int(payload["length"])
            declared_offset = int(payload["dataset_from_index"])
            if episode_index != len(records):
                raise ValueError(
                    "episode indices must be contiguous and ordered; "
                    f"row {len(records)} has {episode_index!r}"
                )
            if not isinstance(tasks, list) or len(tasks) != 1 or not tasks[0]:
                raise ValueError(
                    f"episode {episode_index} must contain exactly one task"
                )
            if length < 1 or declared_offset != dataset_offset:
                raise ValueError(
                    f"episode {episode_index} has invalid length/offset: "
                    f"length={length}, offset={declared_offset}, "
                    f"expected_offset={dataset_offset}"
                )
            records.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    task=str(tasks[0]),
                    length=length,
                    dataset_offset=dataset_offset,
                )
            )
            dataset_offset += length
        return tuple(records)

    records: list[EpisodeRecord] = []
    dataset_offset = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            episode_index = payload.get("episode_index")
            tasks = payload.get("tasks")
            length = payload.get("length")
            if episode_index != len(records):
                raise ValueError(
                    "episode indices must be contiguous and ordered; "
                    f"line {line_number} has {episode_index!r}"
                )
            if not isinstance(tasks, list) or len(tasks) != 1 or not tasks[0]:
                raise ValueError(f"line {line_number} must contain exactly one task")
            if not isinstance(length, int) or length < 1:
                raise ValueError(f"line {line_number} has an invalid length")
            records.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    task=str(tasks[0]),
                    length=length,
                    dataset_offset=dataset_offset,
                )
            )
            dataset_offset += length
    if not records:
        raise ValueError("episode metadata is empty")
    return tuple(records)


@dataclass(frozen=True)
class SequenceWindowTable:
    """Compact table of valid same-episode temporal windows."""

    episodes: tuple[EpisodeRecord, ...]
    unroll_replans: int
    frame_stride: int
    start_frame_stride: int
    action_horizon: int
    episode_ids: np.ndarray
    start_frames: np.ndarray
    anchor_dataset_indices: np.ndarray
    probabilities: np.ndarray
    cumulative_probabilities: np.ndarray
    suites_by_episode: tuple[str, ...]

    @property
    def required_frame_span(self) -> int:
        return (self.unroll_replans - 1) * self.frame_stride + 1

    @property
    def required_target_span(self) -> int:
        return (self.unroll_replans - 1) * self.frame_stride + self.action_horizon

    def __len__(self) -> int:
        return int(self.episode_ids.size)

    def replan_dataset_indices(self, window_indices: Sequence[int]) -> np.ndarray:
        indices = np.asarray(window_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("window_indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("window index is out of range")
        offsets = np.arange(self.unroll_replans, dtype=np.int64) * self.frame_stride
        return self.anchor_dataset_indices[indices, None] + offsets[None, :]

    def replan_episode_frames(self, window_indices: Sequence[int]) -> np.ndarray:
        indices = np.asarray(window_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("window_indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("window index is out of range")
        offsets = np.arange(self.unroll_replans, dtype=np.int64) * self.frame_stride
        return self.start_frames[indices, None] + offsets[None, :]

    def sample_window_indices(
        self, count: int, *, seed: int, stream_offset: int = 0
    ) -> np.ndarray:
        """Draw from a stateless stream that resumes in O(batch) time."""
        uniform = stateless_uniform_sequence(
            count, seed=seed, stream_offset=stream_offset
        )
        return np.searchsorted(
            self.cumulative_probabilities, uniform, side="right"
        ).astype(np.int64)


def build_sequence_window_table(
    episodes: Sequence[EpisodeRecord],
    *,
    suite_by_task: Mapping[str, str],
    unroll_replans: int,
    frame_stride: int,
    action_horizon: int,
    start_frame_stride: int = 1,
) -> SequenceWindowTable:
    if unroll_replans < 2:
        raise ValueError("unroll_replans must be at least two")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if start_frame_stride < 1:
        raise ValueError("start_frame_stride must be positive")
    if not episodes:
        raise ValueError("episodes must be non-empty")
    required_span = (unroll_replans - 1) * frame_stride + action_horizon

    suites_by_episode: list[str] = []
    canonical_tasks: list[str] = []
    windows_per_episode: list[int] = []
    for episode in episodes:
        if episode.length < required_span:
            raise ValueError(
                f"episode {episode.episode_index} is shorter than {required_span} frames"
            )
        canonical = _canonical_task_name(episode.task)
        try:
            suite = suite_by_task[canonical]
        except KeyError as error:
            raise ValueError(
                f"episode {episode.episode_index} task has no suite mapping: {canonical}"
            ) from error
        canonical_tasks.append(canonical)
        suites_by_episode.append(suite)
        windows_per_episode.append(
            (episode.length - required_span) // start_frame_stride + 1
        )

    suite_count = len(set(suites_by_episode))
    tasks_per_suite = Counter(
        (suite, task)
        for suite, task in set(zip(suites_by_episode, canonical_tasks, strict=True))
    )
    # Counter above has unit values; count distinct tasks explicitly per suite.
    distinct_task_counts = Counter(suite for suite, _ in tasks_per_suite)
    episodes_per_task = Counter(
        zip(suites_by_episode, canonical_tasks, strict=True)
    )

    episode_ids = np.repeat(
        np.arange(len(episodes), dtype=np.int64), windows_per_episode
    )
    start_frames = np.concatenate(
        [
            np.arange(count, dtype=np.int64) * start_frame_stride
            for count in windows_per_episode
        ]
    )
    dataset_offsets = np.asarray(
        [episode.dataset_offset for episode in episodes], dtype=np.int64
    )
    anchor_dataset_indices = dataset_offsets[episode_ids] + start_frames

    per_episode_window_weight = np.asarray(
        [
            1.0
            / (
                suite_count
                * distinct_task_counts[suite]
                * episodes_per_task[(suite, task)]
                * window_count
            )
            for suite, task, window_count in zip(
                suites_by_episode,
                canonical_tasks,
                windows_per_episode,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    probabilities = np.repeat(per_episode_window_weight, windows_per_episode)
    probabilities /= probabilities.sum(dtype=np.float64)
    cumulative_probabilities = np.cumsum(probabilities, dtype=np.float64)
    cumulative_probabilities[-1] = 1.0

    return SequenceWindowTable(
        episodes=tuple(episodes),
        unroll_replans=unroll_replans,
        frame_stride=frame_stride,
        start_frame_stride=start_frame_stride,
        action_horizon=action_horizon,
        episode_ids=episode_ids,
        start_frames=start_frames,
        anchor_dataset_indices=anchor_dataset_indices,
        probabilities=probabilities,
        cumulative_probabilities=cumulative_probabilities,
        suites_by_episode=tuple(suites_by_episode),
    )
