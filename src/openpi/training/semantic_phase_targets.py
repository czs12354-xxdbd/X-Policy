"""Validated training-only semantic phase targets for persistent memory.
The deployed policy never reads these targets.  They replace demonstration-time
bins with phase boundaries anchored to robot interaction events and configuration
space motion.  Keeping the manifest loader separate from parquet generation makes
worker startup deterministic and keeps pyarrow out of the training hot path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from openpi.training.persistent_memory_sequence_sampler import (
    EpisodeRecord,
)


SEMANTIC_PHASE_MANIFEST_KIND = "persistent_memory_semantic_phase_targets"
SEMANTIC_PHASE_MANIFEST_VERSION = 1
SEMANTIC_PHASE_MANIFEST_FILENAME = "persistent_semantic_phase_targets.json"
SEMANTIC_PHASE_NAMES = (
    "approach",
    "source_alignment",
    "interaction_onset",
    "post_contact_motion",
    "task_transport",
    "destination_alignment",
    "completion_event",
    "terminal_settle",
)
SEMANTIC_TASK_KINDS = frozenset(("pick_place", "push", "articulate"))


@dataclasses.dataclass(frozen=True)
class SemanticPhaseEpisode:
    episode_index: int
    task: str
    task_kind: str
    length: int
    boundaries: np.ndarray
    interaction_frame: int | None
    completion_frame: int | None
    completion_observed: bool


def episode_metadata_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_optional_frame(value: Any, *, length: int, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"semantic phase {name} must be an integer or null")
    if value < 0 or value >= length:
        raise ValueError(f"semantic phase {name} lies outside its episode")
    return value


def load_semantic_phase_manifest(
    path: pathlib.Path,
    *,
    records: Sequence[EpisodeRecord],
    episode_metadata_path: pathlib.Path,
    replan_steps: int,
    subgoal_count: int,
) -> dict[int, SemanticPhaseEpisode]:
    """Load and strictly bind one generated manifest to episode metadata."""

    if not path.is_file():
        raise FileNotFoundError(
            f"semantic phase manifest is missing: {path}; run "
            "experiments/pi05/build_persistent_semantic_phase_targets.py"
        )
    payload = json.loads(path.read_text())
    if payload.get("kind") != SEMANTIC_PHASE_MANIFEST_KIND:
        raise ValueError("semantic phase manifest kind is invalid")
    if payload.get("format_version") != SEMANTIC_PHASE_MANIFEST_VERSION:
        raise ValueError("semantic phase manifest version is invalid")
    if payload.get("episode_metadata_sha256") != episode_metadata_sha256(
        episode_metadata_path
    ):
        raise ValueError("semantic phase manifest metadata hash is stale")
    if payload.get("replan_steps") != replan_steps:
        raise ValueError("semantic phase manifest replan_steps drifted")
    if payload.get("subgoal_count") != subgoal_count:
        raise ValueError("semantic phase manifest subgoal_count drifted")
    if tuple(payload.get("phase_names", ())) != SEMANTIC_PHASE_NAMES:
        raise ValueError("semantic phase name contract drifted")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != len(records):
        raise ValueError("semantic phase manifest episode count drifted")

    result: dict[int, SemanticPhaseEpisode] = {}
    for record, raw in zip(records, raw_episodes, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("semantic phase episode entry is not a mapping")
        if raw.get("episode_index") != record.episode_index:
            raise ValueError("semantic phase episode order drifted")
        if raw.get("task") != record.task or raw.get("length") != record.length:
            raise ValueError("semantic phase episode identity drifted")
        task_kind = raw.get("task_kind")
        if task_kind not in SEMANTIC_TASK_KINDS:
            raise ValueError("semantic phase task kind is invalid")
        boundaries = np.asarray(raw.get("boundaries"), dtype=np.int64)
        if boundaries.shape != (subgoal_count,):
            raise ValueError("semantic phase boundary count is invalid")
        if boundaries[0] != 0:
            raise ValueError("semantic phase must begin at frame zero")
        if boundaries[-1] >= record.length:
            raise ValueError("semantic phase boundary lies outside its episode")
        if np.any(boundaries % replan_steps):
            raise ValueError("semantic phase boundaries are off the deployment grid")
        if np.any(np.diff(boundaries) < replan_steps):
            raise ValueError("semantic phase boundaries can skip a deployed slot")
        interaction_frame = _validated_optional_frame(
            raw.get("interaction_frame"),
            length=record.length,
            name="interaction_frame",
        )
        completion_frame = _validated_optional_frame(
            raw.get("completion_frame"),
            length=record.length,
            name="completion_frame",
        )
        completion_observed = raw.get("completion_observed")
        if not isinstance(completion_observed, bool):
            raise ValueError("semantic phase completion_observed must be boolean")
        result[record.episode_index] = SemanticPhaseEpisode(
            episode_index=record.episode_index,
            task=record.task,
            task_kind=task_kind,
            length=record.length,
            boundaries=boundaries.astype(np.int32),
            interaction_frame=interaction_frame,
            completion_frame=completion_frame,
            completion_observed=completion_observed,
        )
    return result


def semantic_phase_targets_for_frames(
    boundaries: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return monotonic discrete phase and piecewise phase progress targets."""

    boundaries = np.asarray(boundaries, dtype=np.int64)
    frames = np.asarray(frames, dtype=np.int64)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("semantic phase boundaries must be one-dimensional")
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("semantic phase boundaries must increase strictly")
    if np.any(frames < 0):
        raise ValueError("semantic phase frames must be non-negative")
    phases = np.searchsorted(boundaries[1:], frames, side="right").astype(
        np.int32
    )
    phases = np.minimum(phases, boundaries.size - 1).astype(np.int32)
    progress = np.interp(
        frames.astype(np.float64),
        boundaries.astype(np.float64),
        np.linspace(0.0, 1.0, boundaries.size, dtype=np.float64),
    ).astype(np.float32)
    return phases, progress
