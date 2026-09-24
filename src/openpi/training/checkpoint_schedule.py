"""Checkpoint scheduling helpers that do not import JAX or Orbax."""

from __future__ import annotations

import os
from collections.abc import Mapping


RECOVERY_SAVE_INTERVAL_ENV = "VLA_ARENA_RECOVERY_SAVE_INTERVAL"


def training_log_due(
    *, step: int, num_train_steps: int, log_interval: int
) -> bool:
    """Log regular metrics and always log the exact terminal update."""
    if num_train_steps <= 0:
        raise ValueError("num_train_steps must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    return step % log_interval == 0 or step == num_train_steps - 1


def recovery_save_interval(
    environment: Mapping[str, str] | None = None,
) -> int | None:
    """Read an optional recovery-only checkpoint cadence from the environment."""

    values = os.environ if environment is None else environment
    raw = values.get(RECOVERY_SAVE_INTERVAL_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        interval = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{RECOVERY_SAVE_INTERVAL_ENV} must be a positive integer"
        ) from error
    if interval <= 0:
        raise ValueError(
            f"{RECOVERY_SAVE_INTERVAL_ENV} must be a positive integer"
        )
    return interval


def checkpoint_reason(
    *,
    step: int,
    start_step: int,
    num_train_steps: int,
    save_interval: int,
    recovery_interval: int | None,
) -> str | None:
    """Return why a checkpoint is due, preserving the formal save cadence."""

    if step == num_train_steps - 1:
        return "final"
    if step <= start_step:
        return None
    if step % save_interval == 0:
        return "scheduled"
    if recovery_interval is not None and step % recovery_interval == 0:
        return "recovery-only"
    return None
