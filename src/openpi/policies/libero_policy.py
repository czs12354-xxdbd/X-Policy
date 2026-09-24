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
import functools
import hashlib
import json
import os
import pathlib
import re
import stat

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        'observation/state': np.random.rand(8),
        'observation/image': np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        'observation/wrist_image': np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        'prompt': 'do something',
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, 'c h w -> h w c')
    return image


def _parse_current_future_image(
    image, *, future_visual_supervision: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split an optional two-frame LeRobot query without changing inference."""
    array = np.asarray(image)
    if not future_visual_supervision or array.ndim == 3:
        return _parse_image(array), None
    if array.ndim != 4 or array.shape[0] != 2:
        raise ValueError(
            'future visual supervision expects exactly current/future image pairs'
        )
    return _parse_image(array[0]), _parse_image(array[1])


def _attach_future_visual_inputs(
    inputs: dict,
    data: dict,
    base_future: np.ndarray | None,
    wrist_future: np.ndarray | None,
) -> None:
    """Attach training-only targets and exclude episode-end padding."""
    if base_future is None or wrist_future is None:
        return

    def target_valid(key: str) -> np.bool_:
        padding = np.asarray(data.get(key, (False, False)), dtype=np.bool_)
        if padding.shape != (2,):
            raise ValueError(f'{key} must describe a two-frame image query')
        return np.bool_(not padding[-1])

    inputs['future_image'] = {
        'base_0_rgb': base_future,
        'left_wrist_0_rgb': wrist_future,
        'right_wrist_0_rgb': np.zeros_like(base_future),
    }
    inputs['future_image_mask'] = {
        'base_0_rgb': target_valid('observation/image_is_pad'),
        'left_wrist_0_rgb': target_valid('observation/wrist_image_is_pad'),
        'right_wrist_0_rgb': np.False_,
    }


def _parse_current_future_state(
    data: dict, *, future_state_supervision: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split a current plus ten-step LeRobot state query.

    Inference supplies a single rank-one state and therefore follows the
    unchanged input path even when the training transform enables rollout
    supervision.
    """
    state = np.asarray(data['observation/state'], dtype=np.float32)
    if not future_state_supervision or state.ndim == 1:
        return state, None, None
    if state.ndim != 2 or state.shape[0] != 11:
        raise ValueError(
            'future state supervision expects current state plus ten future states'
        )
    padding = np.asarray(
        data.get('observation/state_is_pad', np.zeros((11,), dtype=np.bool_)),
        dtype=np.bool_,
    )
    if padding.shape != (11,):
        raise ValueError(
            'observation/state_is_pad must describe the eleven-state query'
        )
    return state[0], state[1:], np.logical_not(padding[1:])


def _attach_future_state_inputs(
    inputs: dict,
    future_state: np.ndarray | None,
    future_state_mask: np.ndarray | None,
) -> None:
    if future_state is None:
        return
    if future_state_mask is None:
        raise ValueError('future state targets require a validity mask')
    inputs['future_state'] = future_state
    inputs['future_state_mask'] = future_state_mask


def _prompt_tokens(prompt: str) -> frozenset[str]:
    aliases = {
        'apples': 'apple',
        'drawers': 'cabinet',
        # LIBERO instructions use "drawer" and "layer of the cabinet" for the
        # same articulated storage geometry.
        'drawer': 'cabinet',
        'peaches': 'peach',
        'puts': 'place',
        'put': 'place',
        'placing': 'place',
        'pocelain': 'porcelain',
    }
    return frozenset(
        aliases.get(token, token)
        for token in re.findall(r'[a-z0-9]+', prompt.lower())
    )


def _prompt_token_sequence(prompt: str) -> tuple[str, ...]:
    """Ordered counterpart to ``_prompt_tokens`` for subgoal ordering."""
    normalized = []
    for token in re.findall(r'[a-z0-9]+', prompt.lower()):
        # Reuse the canonicalization above without exposing a second alias map.
        normalized.append(next(iter(_prompt_tokens(token))))
    return tuple(normalized)


def _task_signature(prompt: str) -> frozenset[str]:
    """Collapse wording-only variants while retaining task-critical relations."""
    aliases = {'put': 'place', 'pocelain': 'porcelain'}
    ignored = {'a', 'an', 'and', 'it', 'the', 'up', 'table'}
    return frozenset(
        aliases.get(token, token)
        for token in _prompt_tokens(prompt)
        if token not in ignored and not token.isdigit()
    )


_DEMONSTRATION_RETRIEVAL_MODES = frozenset(
    {'weighted_jaccard', 'role_signature'}
)


def _role_normalized_token_sequence(prompt: str) -> tuple[str, ...]:
    """Normalize auditable LIBERO paraphrases without a runtime encoder."""
    phrase_aliases = (
        (r'\bcould you\b', ' '),
        (r'\bplease\b', ' '),
        (r'\bcarefully\b', ' '),
        (r'\bfirst\b', ' '),
        (r'\bthen stop\b', ' '),
        (r'\bresting upon the work surface\b', 'on the table'),
        (r'\bresting on the work surface\b', 'on the table'),
        (r'\bwork surface\b', 'table'),
        (
            r'\binside the upper drawer(?: of the cabinet)?\b',
            'in the top layer of the cabinet',
        ),
        (r'\bupper drawer\b', 'top layer of the cabinet'),
        (
            r'\bcentral drawer(?: of the cabinet)?\b',
            'middle layer of the cabinet',
        ),
        (r'\bin the space between\b', 'between'),
        (r'\bburner remains active\b', 'stove turned on'),
        (r'\bstove is on\b', 'stove turned on'),
        (r'\bcandle is burning\b', 'candle lit'),
        (r'\batop\b', 'on the top of'),
        (r'\bonto\b', 'on'),
        (r'\binside\b', 'in'),
        (r'\bbeside\b', 'next to'),
        (r'\bamid\b', 'between'),
        (r'\bdish\b', 'plate'),
        (r'\bcontainer\b', 'box'),
        (r'\bdeposit the item\b', 'place it'),
        (r'\bset it\b', 'place it'),
    )
    normalized = prompt.lower()
    for pattern, replacement in phrase_aliases:
        normalized = re.sub(pattern, replacement, normalized)
    word_aliases = {
        'active': 'on',
        'burning': 'lit',
        'central': 'middle',
        'deposit': 'place',
        'drawer': 'cabinet',
        'drawers': 'cabinet',
        'grab': 'pick',
        'grasp': 'pick',
        'lift': 'pick',
        'placing': 'place',
        'pocelain': 'porcelain',
        'put': 'place',
        'puts': 'place',
        'set': 'place',
        'take': 'pick',
        'upon': 'on',
        'upper': 'top',
    }
    ignored = {
        'a', 'an', 'and', 'carefully', 'could', 'first', 'of', 'please',
        'stop', 'the', 'then', 'up', 'while', 'with', 'you',
    }
    return tuple(
        word_aliases.get(token, token)
        for token in re.findall(r'[a-z0-9]+', normalized)
        if token not in ignored
    )


def _role_task_signature(prompt: str) -> tuple | None:
    """Parse action and role-bound clauses; return ``None`` when unknown."""
    tokens = _role_normalized_token_sequence(prompt)
    try:
        if 'open' in tokens or 'close' in tokens:
            action = 'open' if 'open' in tokens else 'close'
            action_index = tokens.index(action)
            target = frozenset(tokens[action_index + 1 :])
            return (action, target) if target else None
        if 'push' in tokens:
            action_index = tokens.index('push')
            goal_index = tokens.index('to', action_index + 1)
            manipulated = frozenset(tokens[action_index + 1 : goal_index])
            goal = frozenset(tokens[goal_index + 1 :])
            if not manipulated or not goal:
                return None
            return ('push', manipulated, goal)
        action_index = tokens.index('pick')
        goal_index = tokens.index('place', action_index + 1)
        manipulated = frozenset(
            token
            for token in tokens[action_index + 1 : goal_index]
            if token != 'it'
        )
        goal = frozenset(
            token for token in tokens[goal_index + 1 :] if token != 'it'
        )
        if not manipulated or not goal:
            return None
        return ('pick_place', manipulated, goal)
    except ValueError:
        return None


def _role_signature_index(bank: dict) -> dict[tuple, int]:
    """Build a strict one-to-one index, rejecting ambiguous training banks."""
    cached = bank.get('role_signature_to_index')
    if cached is not None:
        return cached
    index: dict[tuple, int] = {}
    for task_index, prompt in enumerate(bank['prompts']):
        signature = _role_task_signature(str(prompt))
        if signature is None:
            raise ValueError(
                f'role-signature parser cannot parse bank task {task_index}: {prompt}'
            )
        if signature in index:
            raise ValueError(
                'role-signature retrieval requires unique canonical signatures; '
                f'tasks {index[signature]} and {task_index} collide'
            )
        index[signature] = task_index
    bank['role_signature_to_index'] = index
    return index


def _weighted_token_similarity(
    query: frozenset[str],
    candidate: frozenset[str],
    weights: dict[str, float],
) -> float:
    intersection = query & candidate
    union = query | candidate
    numerator = sum(weights.get(token, 1.0) for token in intersection)
    denominator = sum(weights.get(token, 1.0) for token in union)
    return numerator / denominator if denominator else 0.0


def _weighted_candidate_coverage(
    query: frozenset[str], candidate: frozenset[str], weights: dict[str, float]
) -> float:
    """Fraction of an atomic candidate explained by a longer query."""
    numerator = sum(weights.get(token, 1.0) for token in query & candidate)
    denominator = sum(weights.get(token, 1.0) for token in candidate)
    return numerator / denominator if denominator else 0.0


def _structured_demonstration_plan(
    reference_states: np.ndarray, *, steps: int = 10
) -> np.ndarray:
    """Summarize a whole reference episode as ordered kinematic keyframes."""
    if reference_states.ndim != 2 or len(reference_states) == 0:
        raise ValueError('reference_states must be a non-empty rank-two array')
    if steps < 2:
        raise ValueError('structured demonstration plan needs at least two steps')
    indexes = np.rint(np.linspace(0, len(reference_states) - 1, steps)).astype(
        np.int64
    )
    keyframes = np.asarray(reference_states[indexes], dtype=np.float32)
    displacements = np.zeros_like(keyframes)
    displacements[1:] = keyframes[1:] - keyframes[:-1]
    progress = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    return np.concatenate([keyframes, displacements, progress], axis=-1)


@functools.lru_cache(maxsize=4)
def _load_demonstration_bank(path: str) -> dict:
    resolved = pathlib.Path(path).resolve()
    with np.load(resolved, allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    prompts = [str(prompt) for prompt in bank['prompts']]
    bank['prompt_to_index'] = {prompt: index for index, prompt in enumerate(prompts)}
    bank['prompt_tokens'] = [_prompt_tokens(prompt) for prompt in prompts]
    bank['task_signatures'] = [_task_signature(prompt) for prompt in prompts]
    document_frequency: dict[str, int] = {}
    for tokens in bank['prompt_tokens']:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    task_count = len(prompts)
    bank['token_weights'] = {
        token: np.log((task_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    hard_negative_indexes = []
    negative_candidate_indexes = []
    for query_index, query in enumerate(bank['prompt_tokens']):
        candidates = np.asarray(
            [
                candidate_index
                for candidate_index in range(task_count)
                if candidate_index != query_index
                and bank['task_signatures'][candidate_index]
                != bank['task_signatures'][query_index]
            ],
            dtype=np.int32,
        )
        if not len(candidates):
            candidates = np.asarray(
                [
                    candidate_index
                    for candidate_index in range(task_count)
                    if candidate_index != query_index
                ],
                dtype=np.int32,
            )
        scores = np.asarray(
            [
                _weighted_token_similarity(
                    query,
                    bank['prompt_tokens'][candidate_index],
                    bank['token_weights'],
                )
                for candidate_index in candidates
            ]
        )
        hard_negative_indexes.append(int(candidates[int(np.argmax(scores))]))
        negative_candidate_indexes.append(candidates)
    bank['hard_negative_indexes'] = np.asarray(
        hard_negative_indexes, dtype=np.int32
    )
    bank['negative_candidate_indexes'] = negative_candidate_indexes
    format_version = int(bank['format_version'])
    if format_version not in (1, 2):
        raise ValueError(f'unsupported demonstration-bank format: {resolved}')
    if format_version == 2:
        expected_prefix = (task_count, 2, 10)
        if bank['grounded_demonstration_trajectory'].shape != (*expected_prefix, 7):
            raise ValueError('invalid grounded demonstration trajectory shape')
        if bank['grounded_demonstration_trajectory_mask'].shape != expected_prefix:
            raise ValueError('invalid grounded demonstration trajectory mask shape')
        if bank['grounded_demonstration_rationale'].shape != (task_count, 2):
            raise ValueError('invalid grounded demonstration rationale shape')
    return bank


@functools.lru_cache(maxsize=8)
def _load_episode_progress_metadata(
    path: str,
) -> dict[int, tuple[int, tuple[int, ...] | None]]:
    """Load episode lengths and optional audited semantic phase boundaries.

    Ordinary ``episodes.jsonl`` files retain the legacy continuous frame
    progress target.  A sealed ``persistent_semantic_phase_targets.json``
    supplies eight event-derived boundaries instead, allowing the same
    training-only scalar field to supervise the eight ordered subgoal slots
    with semantic phases.  Neither representation is consumed at inference.
    """
    resolved = pathlib.Path(path).resolve()
    expected_phase_count = None
    with resolved.open() as handle:
        if resolved.suffix == '.json':
            document = json.load(handle)
            if document.get('kind') != 'persistent_memory_semantic_phase_targets':
                raise ValueError('unsupported episode progress metadata JSON')
            expected_phase_count = int(document.get('subgoal_count', 0))
            phase_names = document.get('phase_names', ())
            if expected_phase_count < 2 or len(phase_names) != expected_phase_count:
                raise ValueError('semantic phase manifest has an invalid phase schema')
            records = document.get('episodes', ())
        else:
            records = [
                json.loads(line)
                for line in handle
                if line.strip()
            ]

    metadata: dict[int, tuple[int, tuple[int, ...] | None]] = {}
    for record_number, record in enumerate(records, start=1):
        episode_index = int(record['episode_index'])
        length = int(record['length'])
        if length < 2:
            raise ValueError(
                f'episode {episode_index} has invalid length {length}'
            )
        if episode_index in metadata:
            raise ValueError(
                f'duplicate episode {episode_index} at record {record_number}'
            )
        raw_boundaries = record.get('boundaries')
        boundaries = None
        if raw_boundaries is not None:
            boundaries = tuple(int(value) for value in raw_boundaries)
            if (
                expected_phase_count is not None
                and len(boundaries) != expected_phase_count
            ):
                raise ValueError(
                    f'episode {episode_index} phase count does not match manifest'
                )
            if len(boundaries) < 2:
                raise ValueError(
                    f'episode {episode_index} has fewer than two phase boundaries'
                )
            if boundaries[0] != 0 or any(
                left >= right for left, right in zip(boundaries, boundaries[1:])
            ):
                raise ValueError(
                    f'episode {episode_index} phase boundaries are not strictly ordered'
                )
            if boundaries[-1] >= length:
                raise ValueError(
                    f'episode {episode_index} phase boundary exceeds length {length}'
                )
        metadata[episode_index] = (length, boundaries)
    if not metadata:
        raise ValueError('episode progress metadata is empty')
    return metadata


@functools.lru_cache(maxsize=8)
def _load_episode_lengths(path: str) -> dict[int, int]:
    """Compatibility view used by older structured-demonstration transforms."""
    return {
        episode_index: record[0]
        for episode_index, record in _load_episode_progress_metadata(path).items()
    }


def _task_progress_from_frame(data: dict, *, enabled: bool, metadata_path: str | None):
    """Return exact normalized frame progress for training, absent at inference."""
    if not enabled:
        return None
    if metadata_path is None:
        raise ValueError('task progress supervision requires episode metadata')
    has_episode = 'episode_index' in data
    has_frame = 'frame_index' in data
    if not has_episode and not has_frame and 'actions' not in data:
        return None
    if not (has_episode and has_frame):
        raise ValueError('task-progress training requires episode and frame indexes')
    episode_index = int(np.asarray(data['episode_index']))
    frame_index = int(np.asarray(data['frame_index']))
    metadata = _load_episode_progress_metadata(metadata_path)
    if episode_index not in metadata:
        raise ValueError(f'episode progress metadata is missing episode {episode_index}')
    length, boundaries = metadata[episode_index]
    if not 0 <= frame_index < length:
        raise ValueError(
            f'frame {frame_index} exceeds episode {episode_index} length {length}'
        )
    if boundaries is not None:
        phase_index = int(np.searchsorted(boundaries, frame_index, side='right') - 1)
        phase_index = int(np.clip(phase_index, 0, len(boundaries) - 1))
        return np.float32(phase_index / (len(boundaries) - 1))
    return np.float32(frame_index / (length - 1))


def _retrieve_task_index(
    bank: dict,
    prompt: str,
    *,
    mode: str = 'weighted_jaccard',
) -> int | None:
    if mode not in _DEMONSTRATION_RETRIEVAL_MODES:
        raise ValueError(f'unsupported demonstration retrieval mode: {mode}')
    if prompt in bank['prompt_to_index']:
        return int(bank['prompt_to_index'][prompt])
    if mode == 'role_signature':
        signature = _role_task_signature(prompt)
        if signature is None:
            return None
        return _role_signature_index(bank).get(signature)
    query = _prompt_tokens(prompt)
    if not query:
        return None
    scores = np.asarray(
        [
            _weighted_token_similarity(query, candidate, bank['token_weights'])
            for candidate in bank['prompt_tokens']
        ]
    )
    best = int(np.argmax(scores))
    # Avoid injecting an unrelated trajectory for genuinely novel tasks.
    return best if scores[best] >= 0.55 else None


def _retrieve_compositional_task_indexes(
    bank: dict, prompt: str, *, max_slots: int
) -> tuple[int, ...]:
    """Retrieve a diverse ordered set of atomic demonstrations.

    Exact L0 instructions retain the single-demo behavior.  Novel relational
    or multi-action instructions use weighted lexical coverage to retrieve a
    small set of complementary atoms.  The algorithm depends only on the
    training demonstration bank and never on evaluation task definitions.
    """
    if max_slots < 1:
        raise ValueError('max_slots must be positive')
    exact = bank['prompt_to_index'].get(prompt)
    if exact is not None:
        return (int(exact),)
    query = _prompt_tokens(prompt)
    if not query:
        return ()
    scores = np.asarray(
        [
            0.55
            * _weighted_token_similarity(query, candidate, bank['token_weights'])
            + 0.45
            * _weighted_candidate_coverage(
                query, candidate, bank['token_weights']
            )
            for candidate in bank['prompt_tokens']
        ],
        dtype=np.float32,
    )
    best_score = float(np.max(scores))
    if best_score < 0.35:
        return ()
    candidate_order = list(np.argsort(-scores, kind='stable'))
    selected: list[int] = []
    covered: set[str] = set()
    query_weight = sum(bank['token_weights'].get(token, 1.0) for token in query)
    minimum_score = max(0.25, 0.50 * best_score)
    uninformative = {
        'a',
        'an',
        'and',
        'at',
        'in',
        'it',
        'of',
        'on',
        'pick',
        'place',
        'put',
        'take',
        'the',
        'them',
        'then',
        'to',
        'up',
    }
    action_tokens = {'close', 'open', 'pick', 'place', 'push', 'take'}
    minimum_diverse_slots = 1 if 'all' in query else min(2, max_slots)
    while candidate_order and len(selected) < max_slots:
        ranked = []
        for index in candidate_order:
            index = int(index)
            if float(scores[index]) < minimum_score:
                continue
            candidate_tokens = bank['prompt_tokens'][index]
            new_tokens = (query & candidate_tokens) - covered
            marginal = sum(
                bank['token_weights'].get(token, 1.0) for token in new_tokens
            ) / max(query_weight, 1.0e-6)
            action_coverage_bonus = 0.75 if new_tokens & action_tokens else 0.0
            redundancy = max(
                (
                    _weighted_token_similarity(
                        candidate_tokens,
                        bank['prompt_tokens'][chosen],
                        bank['token_weights'],
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            ranked.append(
                (
                    float(scores[index])
                    + 1.5 * marginal
                    + action_coverage_bonus
                    - 0.35 * redundancy,
                    index,
                )
            )
        if not ranked:
            break
        _, chosen = max(ranked, key=lambda item: (item[0], -item[1]))
        informative_new = (
            (query & bank['prompt_tokens'][chosen]) - covered - uninformative
        )
        if (
            selected
            and not informative_new
            and len(selected) >= minimum_diverse_slots
        ):
            break
        selected.append(chosen)
        covered.update(query & bank['prompt_tokens'][chosen])
        candidate_order.remove(chosen)
    # Put atoms in the order in which their distinctive matched words first
    # occur in the original instruction. This preserves "A, then B" without
    # parsing benchmark-specific templates.
    query_sequence = _prompt_token_sequence(prompt)
    token_positions: dict[str, int] = {}
    for position, token in enumerate(query_sequence):
        token_positions.setdefault(token, position)

    def anchor(index: int) -> tuple[int, float, int]:
        distinctive = (
            query & bank['prompt_tokens'][index]
        ) - uninformative
        rarest_first = sorted(
            distinctive,
            key=lambda token: (
                token_positions.get(token, len(query_sequence)),
                -bank['token_weights'].get(token, 1.0),
            ),
        )
        position = (
            token_positions.get(rarest_first[0], len(query_sequence))
            if rarest_first
            else len(query_sequence)
        )
        return position, -float(scores[index]), index

    return tuple(sorted(selected, key=anchor))


def _training_compositional_candidates(
    bank: dict,
    *,
    matched_task_index: int,
    episode_index: int,
    frame_index: int,
    max_slots: int,
) -> tuple[tuple[int, ...], int]:
    """Build deterministic candidate sets with a non-positional router label."""
    if max_slots < 1:
        raise ValueError('max_slots must be positive')
    task_count = len(bank['prompts'])
    if task_count < 1:
        return (), -1
    slot_hash = (
        episode_index * 2_654_435_761
        + frame_index * 805_459_861
        + matched_task_index * 367_761_217
    ) % 1_000_000
    target_slot = int(slot_hash % min(max_slots, task_count))
    candidates = [matched_task_index]
    for candidate in (
        int(bank['hard_negative_indexes'][matched_task_index]),
        *(
            int(index)
            for index in bank['negative_candidate_indexes'][matched_task_index]
        ),
    ):
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= min(max_slots, task_count):
            break
    ordered = [candidate for candidate in candidates if candidate != matched_task_index]
    ordered.insert(target_slot, matched_task_index)
    return tuple(ordered[:max_slots]), target_slot


def _reference_demonstration(
    bank: dict,
    *,
    task_index: int,
    state: np.ndarray,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    """Return one progress-aligned local chunk and whole-episode reference."""
    reference_episodes = bank['episode_indexes'][task_index]
    slot = 1 if episode_index == int(reference_episodes[0]) else 0
    length = int(bank['lengths'][task_index, slot])
    q01 = bank['state_q01']
    q99 = bank['state_q99']
    normalized_state = (state[:8] - q01) / (q99 - q01 + 1.0e-6) * 2.0 - 1.0
    reference_states = bank['states'][task_index, slot, :length]
    plan = _structured_demonstration_plan(reference_states)
    weights = np.asarray(
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1], dtype=np.float32
    )
    distances = np.mean(
        np.square(
            np.clip(reference_states, -3.0, 3.0)
            - np.clip(normalized_state, -3.0, 3.0)
        )
        * weights,
        axis=-1,
    )
    nearest = int(np.argmin(distances))
    indexes = np.minimum(
        nearest + np.arange(int(bank['action_horizon'])), length - 1
    )
    actions = bank['actions'][task_index, slot, indexes].copy()
    montage = bank['montages'][task_index, slot].copy()
    progress = np.float32(nearest / max(length - 1, 1))
    return actions, plan, montage, progress


def _compose_demonstration_montage(
    montages: list[np.ndarray], template: np.ndarray
) -> np.ndarray:
    """Pack candidate montages into fixed-width strips for the vision prefix."""
    if not montages:
        return np.zeros_like(template)
    output = np.zeros_like(template)
    height, width = output.shape[:2]
    boundaries = np.linspace(0, width, len(montages) + 1, dtype=np.int32)
    for index, montage in enumerate(montages):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        source = np.asarray(montage)
        y = np.linspace(0, source.shape[0] - 1, height).astype(np.int32)
        x = np.linspace(0, source.shape[1] - 1, end - start).astype(np.int32)
        output[:, start:end] = source[y[:, None], x[None, :]]
    return output


def _compositional_router_prompt(
    query: str, retrieved_prompts: list[str], *, training: bool
) -> str:
    """Describe candidate atoms without leaking the supervised training atom."""
    labels = '; '.join(
        f'[{slot + 1}] {candidate_prompt}'
        for slot, candidate_prompt in enumerate(retrieved_prompts)
    )
    instruction = (
        f'Ordered candidate demonstrations: {labels}. '
        'Route actions through the currently unfinished subgoal.'
    )
    if training:
        # The source frame belongs to one atomic task, but naming that task as
        # the query gives away the target slot through exact text matching.
        # Present every candidate symmetrically so the router must use scene
        # compatibility and progress to recover the randomly placed target.
        return f'Synthetic ordered workflow. {instruction}'
    return f'{query}\n{instruction}'


def _training_demonstration_decision(
    *,
    episode_index: int,
    frame_index: int,
    matched_task_index: int,
    task_count: int,
    dropout: float,
    mismatch_rate: float,
    hard_negative_task_index: int | None = None,
    random_negative_task_indexes: tuple[int, ...] | None = None,
) -> tuple[bool, int, bool, str | None]:
    """Return reproducible enable/index/mismatch choices for a training frame."""
    dropout_hash = (
        episode_index * 1_000_003
        + frame_index * 9_176
        + matched_task_index * 611_953
    ) % 1_000_000
    enabled = dropout_hash / 1_000_000 >= dropout
    mismatch_hash = (
        episode_index * 433_494_437
        + frame_index * 297_121_507
        + matched_task_index * 104_729
    ) % 1_000_000
    mismatched = (
        enabled
        and mismatch_rate > 0
        and task_count > 1
        and mismatch_hash / 1_000_000 < mismatch_rate
    )
    selected_task_index = matched_task_index
    negative_kind = None
    if mismatched:
        if hard_negative_task_index is not None and mismatch_hash % 2 == 0:
            selected_task_index = hard_negative_task_index
            negative_kind = 'hard'
        else:
            if random_negative_task_indexes:
                selected_task_index = random_negative_task_indexes[
                    mismatch_hash % len(random_negative_task_indexes)
                ]
            else:
                offset = 1 + mismatch_hash % (task_count - 1)
                selected_task_index = (matched_task_index + offset) % task_count
            negative_kind = 'random'
    return enabled, selected_task_index, mismatched, negative_kind


def _structured_demonstration_rationale(
    retrieved_prompt: str, plan: np.ndarray
) -> str:
    """Verbalize a sparse demonstrated plan without evaluator-side labels."""
    plan = np.asarray(plan, dtype=np.float32)
    if plan.shape != (10, 17):
        raise ValueError('structured demonstration rationale requires a 10x17 plan')
    axis_names = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
    phase_names = ('early', 'middle', 'late')
    phase_descriptions = []
    for phase_name, indexes in zip(
        phase_names, np.array_split(np.arange(plan.shape[0]), 3), strict=True
    ):
        displacement = np.mean(plan[indexes, 8:14], axis=0)
        directions = []
        for axis, value in zip(axis_names, displacement, strict=True):
            if value > 0.05:
                directions.append(f'{axis} positive')
            elif value < -0.05:
                directions.append(f'{axis} negative')
        phase_descriptions.append(
            f'{phase_name} ' + (', '.join(directions) if directions else 'steady')
        )
    return (
        f'Demonstrated subgoal: {retrieved_prompt}. '
        f'Motion rationale: {"; ".join(phase_descriptions)}.'
    )


@dataclasses.dataclass(frozen=True)
class RetrievedDemonstrationLiberoInputs(transforms.DataTransformFn):
    """Inject a task-matched visual/local-action/whole-episode plan."""

    model_type: _model.ModelType
    demonstration_bank_path: str
    demonstration_retrieval_mode: str = 'weighted_jaccard'
    dropout: float = 0.5
    mismatch_rate: float = 0.0
    structured_rationale: bool = False
    grounded_rationale: bool = False
    future_visual_supervision: bool = False
    future_state_supervision: bool = False
    task_progress_supervision: bool = False
    episode_metadata_path: str | None = None

    def __post_init__(self):
        if self.demonstration_retrieval_mode not in _DEMONSTRATION_RETRIEVAL_MODES:
            raise ValueError(
                'unsupported demonstration retrieval mode: '
                f'{self.demonstration_retrieval_mode}'
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('demonstration dropout must be in [0, 1)')
        if not 0.0 <= self.mismatch_rate < 1.0:
            raise ValueError('demonstration mismatch rate must be in [0, 1)')
        if self.structured_rationale and self.grounded_rationale:
            raise ValueError(
                'legacy structured and grounded rationales are mutually exclusive'
            )
        if self.task_progress_supervision and self.episode_metadata_path is None:
            raise ValueError(
                'task progress supervision requires episode metadata'
            )

    def __call__(self, data: dict) -> dict:
        prompt_value = data.get('prompt')
        if prompt_value is None:
            raise ValueError('prompt is required for demonstration retrieval')
        prompt = prompt_value if isinstance(prompt_value, str) else prompt_value.item()
        bank = _load_demonstration_bank(self.demonstration_bank_path)
        matched_task_index = _retrieve_task_index(
            bank,
            prompt,
            mode=self.demonstration_retrieval_mode,
        )
        state, future_state, future_state_mask = _parse_current_future_state(
            data,
            future_state_supervision=self.future_state_supervision,
        )
        actions = np.zeros(
            (int(bank['action_horizon']), int(bank['action_dim'])),
            dtype=np.float32,
        )
        plan = np.zeros((10, 17), dtype=np.float32)
        montage = np.zeros_like(bank['montages'][0, 0])
        retrieved_prompt = None
        grounded_rationale_text = None
        reliability_target = np.float32(-1.0)
        enabled = matched_task_index is not None
        task_index = matched_task_index
        episode_index = int(np.asarray(data.get('episode_index', -1)))
        frame_index = int(np.asarray(data.get('frame_index', -1)))
        task_progress_target = np.float32(-1.0)
        if (
            self.task_progress_supervision
            and episode_index >= 0
            and frame_index >= 0
        ):
            assert self.episode_metadata_path is not None
            episode_lengths = _load_episode_lengths(self.episode_metadata_path)
            if episode_index not in episode_lengths:
                raise ValueError(
                    f'episode progress metadata is missing episode {episode_index}'
                )
            length = episode_lengths[episode_index]
            if frame_index >= length:
                raise ValueError(
                    f'frame {frame_index} exceeds episode {episode_index} length {length}'
                )
            task_progress_target = np.float32(frame_index / (length - 1))
        mismatched = False
        # Training samples carry episode/frame indexes; inference inputs do
        # not. Use two independent deterministic hashes so dropout and negative
        # sampling remain reproducible across data-loader workers and epochs.
        if enabled and episode_index >= 0 and frame_index >= 0:
            enabled, task_index, mismatched, _ = _training_demonstration_decision(
                episode_index=episode_index,
                frame_index=frame_index,
                matched_task_index=int(matched_task_index),
                task_count=len(bank['prompts']),
                dropout=self.dropout,
                mismatch_rate=self.mismatch_rate,
                hard_negative_task_index=int(
                    bank['hard_negative_indexes'][matched_task_index]
                ),
                random_negative_task_indexes=tuple(
                    int(index)
                    for index in bank['negative_candidate_indexes'][
                        matched_task_index
                    ]
                ),
            )
        if enabled:
            assert task_index is not None
            retrieved_prompt = str(bank['prompts'][task_index])
            reference_episodes = bank['episode_indexes'][task_index]
            slot = 1 if episode_index == int(reference_episodes[0]) else 0
            length = int(bank['lengths'][task_index, slot])
            q01 = bank['state_q01']
            q99 = bank['state_q99']
            normalized_state = (state[:8] - q01) / (q99 - q01 + 1.0e-6) * 2.0 - 1.0
            reference_states = bank['states'][task_index, slot, :length]
            plan = _structured_demonstration_plan(reference_states)
            # Position/orientation dimensions define task progress; the final
            # two low-range state channels are deliberately down-weighted.
            weights = np.asarray(
                [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1],
                dtype=np.float32,
            )
            distances = np.mean(
                np.square(
                    np.clip(reference_states, -3.0, 3.0)
                    - np.clip(normalized_state, -3.0, 3.0)
                )
                * weights,
                axis=-1,
            )
            nearest = int(np.argmin(distances))
            indexes = np.minimum(
                nearest + np.arange(int(bank['action_horizon'])), length - 1
            )
            actions = bank['actions'][task_index, slot, indexes].copy()
            montage = bank['montages'][task_index, slot].copy()
            if self.grounded_rationale:
                if int(bank['format_version']) != 2:
                    raise ValueError(
                        'grounded rationale requires a v2 demonstration bank'
                    )
                grounded_rationale_text = str(
                    bank['grounded_demonstration_rationale'][task_index, slot]
                )
            reliability_target = np.float32(0.0 if mismatched else 1.0)
        if not enabled:
            actions.fill(0.0)
            plan.fill(0.0)
            montage.fill(0)

        base_image, base_future = _parse_current_future_image(
            data['observation/image'],
            future_visual_supervision=self.future_visual_supervision,
        )
        wrist_image, wrist_future = _parse_current_future_image(
            data['observation/wrist_image'],
            future_visual_supervision=self.future_visual_supervision,
        )
        inputs = {
            'state': state,
            'image': {
                'base_0_rgb': base_image,
                'left_wrist_0_rgb': wrist_image,
                'right_wrist_0_rgb': montage,
            },
            'image_mask': {
                'base_0_rgb': np.True_,
                'left_wrist_0_rgb': np.True_,
                'right_wrist_0_rgb': np.bool_(enabled),
            },
            'demonstration_actions': actions,
            'demonstration_plan': plan,
            'demonstration_mask': np.bool_(enabled),
            'demonstration_reliability_target': reliability_target,
        }
        if self.task_progress_supervision and task_progress_target >= 0:
            inputs['task_progress_target'] = task_progress_target
        _attach_future_visual_inputs(
            inputs, data, base_future, wrist_future
        )
        _attach_future_state_inputs(
            inputs, future_state, future_state_mask
        )
        if 'actions' in data:
            inputs['actions'] = data['actions']
        # Label the retrieved visual/action context in the VLM's native text
        # space. Without this link the auxiliary trajectory is anonymous; the
        # language label lets the action reasoner distinguish the current task
        # from its nearest L0 procedural example. It is removed together with
        # the demonstration under context dropout.
        if enabled:
            if self.grounded_rationale:
                assert grounded_rationale_text is not None
                retrieved_context = grounded_rationale_text
            elif self.structured_rationale:
                retrieved_context = _structured_demonstration_rationale(
                    retrieved_prompt, plan
                )
            else:
                retrieved_context = f'Retrieved demonstration task: {retrieved_prompt}'
            inputs['prompt'] = f'{prompt}\n{retrieved_context}'
        else:
            inputs['prompt'] = prompt
        return inputs


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _array_payload_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(
        json.dumps(
            list(array.shape), separators=(',', ':'), ensure_ascii=False
        ).encode('utf-8')
    )
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _stat_identity(value) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _immutable_file_identity(
    path: pathlib.Path,
) -> tuple[int, int, int, int, int, int]:
    value = os.lstat(path)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'PSM-SDLA cache path is not a regular file: {path}')
    return _stat_identity(value)


def _open_stable_file(
    path: pathlib.Path,
    identity: tuple[int, int, int, int, int, int],
):
    if _immutable_file_identity(path) != identity:
        raise ValueError(f'PSM-SDLA cache file identity drifted: {path}')
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0),
    )
    if _stat_identity(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        raise ValueError(f'PSM-SDLA cache file changed while opening: {path}')
    return os.fdopen(descriptor, 'rb')


def _read_stable_bytes(
    path: pathlib.Path,
    identity: tuple[int, int, int, int, int, int],
) -> bytes:
    with _open_stable_file(path, identity) as handle:
        payload = handle.read()
        if _stat_identity(os.fstat(handle.fileno())) != identity:
            raise ValueError(f'PSM-SDLA cache file changed while reading: {path}')
    if _immutable_file_identity(path) != identity:
        raise ValueError(f'PSM-SDLA cache file changed after reading: {path}')
    return payload


def _hash_stable_file(
    path: pathlib.Path,
    identity: tuple[int, int, int, int, int, int],
) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with _open_stable_file(path, identity) as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
            byte_count += len(block)
        if _stat_identity(os.fstat(handle.fileno())) != identity:
            raise ValueError(f'PSM-SDLA cache file changed while hashing: {path}')
    if _immutable_file_identity(path) != identity:
        raise ValueError(f'PSM-SDLA cache file changed after hashing: {path}')
    return digest.hexdigest(), byte_count


def _strict_json_bytes(payload: bytes, path: pathlib.Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key in {path}: {key}')
            result[key] = value
        return result

    try:
        result = json.loads(payload.decode('utf-8'), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid PSM-SDLA manifest: {path}') from error
    if not isinstance(result, dict):
        raise ValueError('PSM-SDLA manifest must be a JSON object')
    return result


@functools.lru_cache(maxsize=8)
def _load_structured_demo_language_bank_cached(
    manifest_path: str,
    bank_path: str,
) -> tuple[dict, dict]:
    """Admit one immutable CPU bank exactly once per worker/configuration."""
    manifest_file = pathlib.Path(manifest_path)
    bank_file = pathlib.Path(bank_path)
    manifest_identity = _immutable_file_identity(manifest_file)
    bank_identity = _immutable_file_identity(bank_file)
    manifest = _strict_json_bytes(
        _read_stable_bytes(manifest_file, manifest_identity), manifest_file
    )
    claimed_manifest_hash = manifest.get('manifest_sha256')
    digest_manifest = dict(manifest)
    digest_manifest.pop('manifest_sha256', None)
    if claimed_manifest_hash != _canonical_sha256(digest_manifest):
        raise ValueError('PSM-SDLA manifest self-hash drifted')
    required_constants = {
        'schema_version': 'psm_sdla_structured_bank/v1',
        'status': 'complete_atomic_candidate_not_production_authorized',
        'production_authorized': False,
        'source_repo_id': 'VLA-Arena/VLA_Arena_L0_L_lerobot_openpi',
        'source_revision': '8c6cdcac185c7c97582a78f8ad8c2ef3757ce2c7',
        'source_dataset_level': 'L0',
        'source_level': 'L0',
        'source_split': 'train',
        'split': 'train',
        'contains_l1_or_l2': False,
        'contains_l1_or_l2_bank_rows': False,
        'contains_evaluation_rollouts': False,
        'contains_success_outcomes': False,
        'contains_outcomes': False,
        'labels_are_policy_inputs': False,
    }
    for name, expected in required_constants.items():
        if manifest.get(name) != expected:
            raise ValueError(f'PSM-SDLA manifest constant drifted: {name}')
    contract_constants = {
        'task_count': 60,
        'episode_count': 3018,
        'references_per_task': 2,
        'spatial_language_lookup_rows': 73626,
        'leave_one_out_unit': 'exact_full_observation_alias_group',
        'exact_group_ids_are_model_features': False,
        'spatial_language_labels_are_decoder_only': True,
    }
    manifest_contract = manifest.get('contract', {})
    for name, expected in contract_constants.items():
        if manifest_contract.get(name) != expected:
            raise ValueError(f'PSM-SDLA bank contract drifted: {name}')
    source_access = manifest.get('source_bindings', {}).get('source_access', {})
    if source_access.get('arena_trajectory_levels_read') != [0] or any(
        source_access.get(name) is not False
        for name in (
            'image_columns_read',
            'evaluation_definitions_read',
            'evaluator_task_id_or_level_read',
            'bddl_or_simulator_state_read',
            'reward_success_cost_or_outcome_read',
            'future_evaluation_rollouts_read',
        )
    ):
        raise ValueError('PSM-SDLA manifest reports forbidden source access')
    forbidden_source_pattern = re.compile(
        r'vla[_-]arena[_-]l[12](?:[_/\\-]|$)', re.IGNORECASE
    )
    if forbidden_source_pattern.search(str(bank_file)) or (
        forbidden_source_pattern.search(json.dumps(manifest, sort_keys=True))
    ):
        raise ValueError('L1/L2 dataset path escaped the L0-only boundary')
    if manifest.get('bank_filename') != bank_file.name:
        raise ValueError('PSM-SDLA bank filename drifted')
    bank_sha256, bank_bytes = _hash_stable_file(bank_file, bank_identity)
    if manifest.get('bank_bytes') != bank_bytes:
        raise ValueError('PSM-SDLA bank byte count drifted')
    if manifest.get('bank_sha256') != bank_sha256:
        raise ValueError('PSM-SDLA bank hash drifted')
    # ``benchmark_level`` may be retained as immutable audit metadata, but is
    # intentionally not validated into, returned by, or consulted by runtime
    # retrieval.  One prompt-only algorithm must behave identically at every
    # formal benchmark level.
    with _open_stable_file(bank_file, bank_identity) as handle:
        with np.load(handle, allow_pickle=False) as archive:
            bank = {name: np.asarray(archive[name]) for name in archive.files}
        if _stat_identity(os.fstat(handle.fileno())) != bank_identity:
            raise ValueError('PSM-SDLA bank changed during eager array load')
    if _immutable_file_identity(manifest_file) != manifest_identity:
        raise ValueError('PSM-SDLA manifest changed during cache load')
    if _immutable_file_identity(bank_file) != bank_identity:
        raise ValueError('PSM-SDLA bank changed during cache load')
    required = {
        'prompts',
        'episode_indexes',
        'lengths',
        'states',
        'actions',
        'state_q01',
        'state_q99',
        'demonstration_tokenized_prompt',
        'demonstration_tokenized_prompt_mask',
        'demonstration_semantic_span_mask',
        'demonstration_semantic_valid_mask',
        'demonstration_plans',
        'demonstration_actions',
        'reference_mask',
        'reference_exact_group_index',
        'episode_exact_group_index',
        'exact_group_ids',
        'target_episode_index',
        'target_frame_index',
        'target_task_index',
        'spatial_language_target_ids',
        'spatial_language_target_mask',
    }
    if not required.issubset(bank):
        raise ValueError(
            'PSM-SDLA bank is missing arrays: '
            f'{sorted(required - set(bank))}'
        )
    array_contract = manifest.get('array_contract')
    if not isinstance(array_contract, dict) or set(array_contract) != set(bank):
        raise ValueError('PSM-SDLA bank array set differs from manifest')
    for name, value in bank.items():
        specification = array_contract[name]
        if (
            not isinstance(specification, dict)
            or specification.get('shape') != list(value.shape)
            or specification.get('dtype') != value.dtype.str
            or specification.get('bytes') != value.nbytes
            or specification.get('payload_sha256')
            != _array_payload_sha256(value)
            or value.dtype.hasobject
        ):
            raise ValueError(f'PSM-SDLA bank array contract drifted: {name}')
    task_count = len(bank['prompts'])
    if task_count != 60 or bank['episode_indexes'].shape != (60, 2):
        raise ValueError('formal bank must contain 60 tasks and two references')
    if (
        bank['lengths'].shape != (task_count, 2)
        or bank['states'].ndim != 4
        or bank['states'].shape[:2] != (task_count, 2)
        or bank['states'].shape[-1] != 8
        or bank['actions'].ndim != 4
        or bank['actions'].shape[:3] != bank['states'].shape[:3]
        or bank['actions'].shape[-1] != 32
    ):
        raise ValueError('full reference state/action trajectory shapes drifted')
    if bank['demonstration_tokenized_prompt'].shape != (task_count, 48):
        raise ValueError('demo prompt ids must be [task,48]')
    if bank['demonstration_tokenized_prompt_mask'].shape != (task_count, 48):
        raise ValueError('demo prompt mask must be [task,48]')
    if bank['demonstration_semantic_span_mask'].shape != (task_count, 8, 48):
        raise ValueError('semantic span mask must be [task,8,48]')
    if bank['demonstration_semantic_valid_mask'].shape != (task_count, 8):
        raise ValueError('semantic validity must be [task,8]')
    if bank['demonstration_plans'].shape != (task_count, 2, 10, 17):
        raise ValueError('precomputed demonstration plans must be [task,2,10,17]')
    if bank['demonstration_actions'].shape != (task_count, 2, 10, 32):
        raise ValueError('precomputed demonstration actions must be [task,2,10,32]')
    if bank['reference_mask'].shape != (task_count, 2) or not np.all(
        bank['reference_mask']
    ):
        raise ValueError('every task must have two valid references')
    if bank['episode_exact_group_index'].shape != (3018,):
        raise ValueError('episode exact-group registry must cover all L0 episodes')
    if len(bank['exact_group_ids']) != 1865 or len(set(bank['exact_group_ids'])) != 1865:
        raise ValueError('exact-group registry cardinality or uniqueness drifted')
    episode_groups = bank['episode_exact_group_index'].astype(np.int64)
    reference_groups = bank['reference_exact_group_index'].astype(np.int64)
    if (
        np.any(episode_groups < 0)
        or np.any(episode_groups >= len(bank['exact_group_ids']))
        or np.any(reference_groups < 0)
        or np.any(reference_groups >= len(bank['exact_group_ids']))
        or np.any(reference_groups[:, 0] == reference_groups[:, 1])
    ):
        raise ValueError('exact-group index registry drifted')
    references = bank['episode_indexes'].astype(np.int64)
    if np.any(references < 0) or np.any(references >= len(episode_groups)):
        raise ValueError('reference episode index escaped the L0 registry')
    if np.any(reference_groups != episode_groups[references]):
        raise ValueError('reference episode/group bindings drifted')
    if bank['target_episode_index'].shape != (73626,) or bank[
        'target_frame_index'
    ].shape != (73626,):
        raise ValueError('spatial-language lookup key count drifted')
    if bank['spatial_language_target_ids'].shape != (73626, 32):
        raise ValueError('spatial-language targets must be [row,32]')
    if bank['spatial_language_target_mask'].shape != bank[
        'spatial_language_target_ids'
    ].shape:
        raise ValueError('spatial-language target mask differs from ids')
    target_codes = (
        bank['target_episode_index'].astype(np.int64) << np.int64(32)
    ) | bank['target_frame_index'].astype(np.int64)
    if np.any(target_codes[1:] <= target_codes[:-1]):
        raise ValueError('spatial-language lookup keys must be sorted and unique')
    if bank['target_task_index'].shape != bank['target_episode_index'].shape:
        raise ValueError('target task lookup shape drifted')
    bank['target_codes'] = target_codes
    prompts = [str(value) for value in bank['prompts']]
    bank['prompt_to_index'] = {
        prompt: index for index, prompt in enumerate(prompts)
    }
    if len(bank['prompt_to_index']) != task_count:
        raise ValueError('formal L0 prompts must be unique')
    bank['prompt_tokens'] = [_prompt_tokens(prompt) for prompt in prompts]
    document_frequency: dict[str, int] = {}
    for tokens in bank['prompt_tokens']:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    bank['token_weights'] = {
        token: np.log((task_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    bank['_manifest_identity'] = manifest_identity
    bank['_bank_identity'] = bank_identity
    for value in bank.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return manifest, bank


def _load_structured_demo_language_bank(
    manifest_path: str, bank_path: str
) -> tuple[dict, dict]:
    """Per-worker cache invalidated by any manifest/bank identity drift."""
    manifest_file = pathlib.Path(os.path.abspath(manifest_path))
    bank_file = pathlib.Path(os.path.abspath(bank_path))
    manifest, bank = _load_structured_demo_language_bank_cached(
        str(manifest_file), str(bank_file)
    )
    if _immutable_file_identity(manifest_file) != bank['_manifest_identity']:
        raise ValueError('PSM-SDLA manifest identity drifted after admission')
    if _immutable_file_identity(bank_file) != bank['_bank_identity']:
        raise ValueError('PSM-SDLA bank identity drifted after admission')
    return manifest, bank


def _select_structured_demo_reference_slot(
    bank: dict, *, task_index: int, episode_index: int
) -> int:
    """Select one valid reference, excluding the query's exact alias group."""
    reference_mask = np.asarray(bank['reference_mask'][task_index], dtype=np.bool_)
    references = np.asarray(bank['episode_indexes'][task_index], dtype=np.int64)
    reference_groups = np.asarray(
        bank['reference_exact_group_index'][task_index], dtype=np.int64
    )
    if episode_index < 0:
        eligible = np.flatnonzero(reference_mask)
    else:
        episode_groups = bank['episode_exact_group_index']
        if episode_index >= len(episode_groups):
            raise ValueError('training episode escaped the L0 exact-group registry')
        query_group = int(episode_groups[episode_index])
        eligible = np.flatnonzero(
            reference_mask
            & (references != episode_index)
            & (reference_groups != query_group)
        )
    if not len(eligible):
        raise ValueError('exact-group leave-one-out has no eligible reference')
    return int(eligible[0])


@dataclasses.dataclass(frozen=True)
class StructuredDemoLanguageLiberoInputs(transforms.DataTransformFn):
    """Manifest-bound, exact-group-leave-one-out PSM-SDLA consumer."""

    model_type: _model.ModelType
    manifest_path: str
    bank_path: str
    dropout: float = 0.25
    future_visual_supervision: bool = False
    future_state_supervision: bool = False
    task_progress_supervision: bool = False
    episode_metadata_path: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('structured-demo dropout must lie in [0,1)')
        if self.task_progress_supervision and self.episode_metadata_path is None:
            raise ValueError('task progress supervision requires episode metadata')

    def __call__(self, data: dict) -> dict:
        forbidden_runtime_fields = {
            'benchmark_level',
            'task_id',
            'task_index',
            'evaluator_task_id',
            'evaluator_task_index',
            'bddl_path',
            'bddl_ast',
            'initial_state',
            'simulator_state',
            'reward',
            'success',
            'outcome',
        }
        normalized_input_fields = {
            str(key).lower().replace('-', '_') for key in data
        }
        escaped = sorted(forbidden_runtime_fields & normalized_input_fields)
        if escaped:
            raise ValueError(
                'forbidden formal retrieval fields were supplied: '
                f'{escaped}'
            )
        manifest, bank = _load_structured_demo_language_bank(
            self.manifest_path, self.bank_path
        )
        prompt_value = data.get('prompt')
        if prompt_value is None:
            raise ValueError('prompt is required for structured retrieval')
        prompt = (
            prompt_value
            if isinstance(prompt_value, str)
            else str(prompt_value.item())
        )
        task_index = _retrieve_task_index(bank, prompt)
        has_actions = 'actions' in data
        has_episode_index = 'episode_index' in data
        has_frame_index = 'frame_index' in data
        if has_actions:
            if not (has_episode_index and has_frame_index):
                raise ValueError(
                    'structured-demo training samples require actions, '
                    'episode_index, and frame_index together'
                )
            episode_index = int(np.asarray(data['episode_index']))
            frame_index = int(np.asarray(data['frame_index']))
            if episode_index < 0 or frame_index < 0:
                raise ValueError(
                    'structured-demo training episode/frame indexes must be '
                    'nonnegative'
                )
            training = True
        else:
            if has_episode_index or has_frame_index:
                raise ValueError(
                    'structured-demo inference must not receive episode_index '
                    'or frame_index side channels'
                )
            episode_index = -1
            frame_index = -1
            training = False
        context_enabled = task_index is not None
        if context_enabled and training:
            assert task_index is not None
            dropout_hash = (
                episode_index * 1_000_003
                + frame_index * 9_176
                + task_index * 611_953
            ) % 1_000_000
            context_enabled = dropout_hash / 1_000_000 >= self.dropout
        exact_l0_task_match = bool(
            task_index is not None and prompt in bank['prompt_to_index']
        )
        if training and not exact_l0_task_match:
            raise ValueError(
                'formal PSM-SDLA training requires an exact canonical L0 prompt'
            )
        # Runtime has no benchmark-level side channel.  Only an exact L0
        # prompt may expose raw L0 trajectories; prompt-only semantic matches
        # supply typed semantic context and always mask the 10+2 trajectory
        # tokens.  This makes every novel L1/L2 mapping safe without knowing
        # or inferring its formal level.
        trajectory_enabled = bool(context_enabled and exact_l0_task_match)

        actions = np.zeros((10, 32), dtype=np.float32)
        plan = np.zeros((10, 17), dtype=np.float32)
        demo_ids = np.zeros((48,), dtype=np.int32)
        demo_prompt_mask = np.zeros((48,), dtype=np.bool_)
        semantic_spans = np.zeros((8, 48), dtype=np.bool_)
        semantic_valid = np.zeros((8,), dtype=np.bool_)
        if context_enabled:
            assert task_index is not None
            slot = _select_structured_demo_reference_slot(
                bank, task_index=task_index, episode_index=episode_index
            )
            demo_ids = np.asarray(
                bank['demonstration_tokenized_prompt'][task_index],
                dtype=np.int32,
            )
            demo_prompt_mask = np.asarray(
                bank['demonstration_tokenized_prompt_mask'][task_index],
                dtype=np.bool_,
            )
            semantic_spans = np.asarray(
                bank['demonstration_semantic_span_mask'][task_index],
                dtype=np.bool_,
            )
            semantic_valid = np.asarray(
                bank['demonstration_semantic_valid_mask'][task_index],
                dtype=np.bool_,
            )
            if trajectory_enabled:
                plan = np.asarray(
                    bank['demonstration_plans'][task_index, slot],
                    dtype=np.float32,
                )
                actions = np.asarray(
                    bank['demonstration_actions'][task_index, slot],
                    dtype=np.float32,
                )

        base_image, base_future = _parse_current_future_image(
            data['observation/image'],
            future_visual_supervision=self.future_visual_supervision,
        )
        wrist_image, wrist_future = _parse_current_future_image(
            data['observation/wrist_image'],
            future_visual_supervision=self.future_visual_supervision,
        )
        state, future_state, future_state_mask = _parse_current_future_state(
            data, future_state_supervision=self.future_state_supervision
        )
        inputs = {
            'state': state,
            'image': {
                'base_0_rgb': base_image,
                'left_wrist_0_rgb': wrist_image,
                # Never overload a current-camera slot with a demo montage.
                'right_wrist_0_rgb': np.zeros_like(base_image),
            },
            'image_mask': {
                'base_0_rgb': np.True_,
                'left_wrist_0_rgb': np.True_,
                'right_wrist_0_rgb': np.False_,
            },
            'prompt': prompt,
            'demonstration_actions': actions,
            'demonstration_plan': plan,
            'demonstration_tokenized_prompt': demo_ids,
            'demonstration_tokenized_prompt_mask': demo_prompt_mask,
            'demonstration_semantic_span_mask': semantic_spans,
            'demonstration_semantic_valid_mask': semantic_valid,
            'demonstration_context_mask': np.bool_(context_enabled),
            'demonstration_trajectory_mask': np.bool_(trajectory_enabled),
        }
        _attach_future_visual_inputs(
            inputs, data, base_future, wrist_future
        )
        _attach_future_state_inputs(inputs, future_state, future_state_mask)
        progress_target = _task_progress_from_frame(
            data,
            enabled=self.task_progress_supervision,
            metadata_path=self.episode_metadata_path,
        )
        if progress_target is not None:
            inputs['task_progress_target'] = progress_target
        if 'actions' in data:
            inputs['actions'] = data['actions']
        if training:
            code = (np.int64(episode_index) << np.int64(32)) | np.int64(
                frame_index
            )
            row = int(np.searchsorted(bank['target_codes'], code))
            if row >= len(bank['target_codes']) or bank['target_codes'][row] != code:
                raise KeyError(
                    'spatial-language target is absent for '
                    f'{episode_index}:{frame_index}'
                )
            assert task_index is not None
            if int(bank['target_task_index'][row]) != task_index:
                raise ValueError(
                    'training prompt disagrees with the L0 episode/task binding'
                )
            inputs['spatial_language_target_ids'] = np.asarray(
                bank['spatial_language_target_ids'][row], dtype=np.int32
            )
            inputs['spatial_language_target_mask'] = np.asarray(
                bank['spatial_language_target_mask'][row], dtype=np.bool_
            )
        for name in (
            'demonstration_actions',
            'demonstration_plan',
            'demonstration_tokenized_prompt',
            'demonstration_tokenized_prompt_mask',
            'demonstration_semantic_span_mask',
            'demonstration_semantic_valid_mask',
            'spatial_language_target_ids',
            'spatial_language_target_mask',
        ):
            value = inputs.get(name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        return inputs


@dataclasses.dataclass(frozen=True)
class GroundedStructuredDemoLanguageLiberoInputs(transforms.DataTransformFn):
    """Add grounded VLM context without overwriting native PSM supervision.

    The structured L0-only transform remains authoritative for every PSM
    action, plan, semantic, spatial-language, future-state and progress field.
    A second independently audited bank contributes only a retrieved montage
    in the otherwise absent right-wrist slot and its matched grounded
    rationale. ``grounded_context_mask`` lets physical geometry, memory and
    dynamics branches exclude that montage while the shared VLM can attend it.
    """

    model_type: _model.ModelType
    structured_manifest_path: str
    structured_bank_path: str
    grounded_bank_path: str
    structured_dropout: float = 0.25
    grounded_dropout: float = 0.5
    grounded_retrieval_mode: str = 'role_signature'
    future_visual_supervision: bool = False
    future_state_supervision: bool = False
    task_progress_supervision: bool = False
    episode_metadata_path: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.structured_dropout < 1.0:
            raise ValueError('structured-demo dropout must lie in [0,1)')
        if not 0.0 <= self.grounded_dropout < 1.0:
            raise ValueError('grounded-demo dropout must lie in [0,1)')
        if self.grounded_retrieval_mode != 'role_signature':
            raise ValueError(
                'grounded structured retrieval requires role_signature mode'
            )

    def __call__(self, data: dict) -> dict:
        structured = StructuredDemoLanguageLiberoInputs(
            model_type=self.model_type,
            manifest_path=self.structured_manifest_path,
            bank_path=self.structured_bank_path,
            dropout=self.structured_dropout,
            future_visual_supervision=self.future_visual_supervision,
            future_state_supervision=self.future_state_supervision,
            task_progress_supervision=self.task_progress_supervision,
            episode_metadata_path=self.episode_metadata_path,
        )(data)
        grounded = RetrievedDemonstrationLiberoInputs(
            model_type=self.model_type,
            demonstration_bank_path=self.grounded_bank_path,
            demonstration_retrieval_mode=self.grounded_retrieval_mode,
            dropout=self.grounded_dropout,
            mismatch_rate=0.0,
            grounded_rationale=True,
            future_visual_supervision=self.future_visual_supervision,
            future_state_supervision=self.future_state_supervision,
            task_progress_supervision=self.task_progress_supervision,
            episode_metadata_path=self.episode_metadata_path,
        )(data)
        grounded_valid = np.bool_(
            grounded['image_mask']['right_wrist_0_rgb']
        )
        result = dict(structured)
        result['image'] = dict(structured['image'])
        result['image_mask'] = dict(structured['image_mask'])
        result['image']['right_wrist_0_rgb'] = grounded['image'][
            'right_wrist_0_rgb'
        ]
        result['image_mask']['right_wrist_0_rgb'] = grounded_valid
        result['grounded_context_mask'] = grounded_valid
        result['prompt'] = grounded['prompt']
        return result


@dataclasses.dataclass(frozen=True)
class CompositionalRetrievedDemonstrationLiberoInputs(transforms.DataTransformFn):
    """Inject multiple atomic demonstrations and an active-subgoal target.

    Training frames place the matched atomic demonstration at a deterministic
    but varying slot among hard negatives.  This teaches candidate routing
    without evaluation instructions.  At inference, exact atomic tasks retain
    one candidate while novel/composite prompts retrieve complementary atoms.
    """

    model_type: _model.ModelType
    demonstration_bank_path: str
    max_slots: int = 3
    dropout: float = 0.25

    def __post_init__(self):
        if self.max_slots < 2:
            raise ValueError('compositional retrieval requires at least two slots')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('demonstration dropout must be in [0, 1)')

    def __call__(self, data: dict) -> dict:
        prompt_value = data.get('prompt')
        if prompt_value is None:
            raise ValueError('prompt is required for demonstration retrieval')
        prompt = prompt_value if isinstance(prompt_value, str) else prompt_value.item()
        bank = _load_demonstration_bank(self.demonstration_bank_path)
        state = np.asarray(data['observation/state'], dtype=np.float32)
        episode_index = int(np.asarray(data.get('episode_index', -1)))
        frame_index = int(np.asarray(data.get('frame_index', -1)))
        training = episode_index >= 0 and frame_index >= 0
        matched_task_index = _retrieve_task_index(bank, prompt)

        target_slot = -1
        if training and matched_task_index is not None:
            candidate_indexes, target_slot = _training_compositional_candidates(
                bank,
                matched_task_index=matched_task_index,
                episode_index=episode_index,
                frame_index=frame_index,
                max_slots=self.max_slots,
            )
            dropout_hash = (
                episode_index * 1_000_003
                + frame_index * 9_176
                + matched_task_index * 611_953
            ) % 1_000_000
            if dropout_hash / 1_000_000 < self.dropout:
                candidate_indexes, target_slot = (), -1
        else:
            candidate_indexes = _retrieve_compositional_task_indexes(
                bank, prompt, max_slots=self.max_slots
            )

        action_horizon = int(bank['action_horizon'])
        action_dim = int(bank['action_dim'])
        actions = np.zeros(
            (self.max_slots, action_horizon, action_dim), dtype=np.float32
        )
        plans = np.zeros((self.max_slots, 10, 17), dtype=np.float32)
        slot_mask = np.zeros(self.max_slots, dtype=np.bool_)
        progress = np.zeros(self.max_slots, dtype=np.float32)
        montages: list[np.ndarray] = []
        retrieved_prompts: list[str] = []
        for slot, task_index in enumerate(candidate_indexes):
            candidate_actions, candidate_plan, montage, candidate_progress = (
                _reference_demonstration(
                    bank,
                    task_index=task_index,
                    state=state,
                    episode_index=episode_index,
                )
            )
            actions[slot] = candidate_actions
            plans[slot] = candidate_plan
            slot_mask[slot] = True
            progress[slot] = candidate_progress
            montages.append(montage)
            retrieved_prompts.append(str(bank['prompts'][task_index]))

        if training and target_slot >= 0:
            # Synthetic candidate order supplies a monotonic subgoal curriculum:
            # slots preceding the current atomic behavior are marked complete,
            # later slots are marked not started, and the matched slot retains
            # its real progress-alignment estimate. Across frames the matched
            # task appears at every position, so the router must learn progress
            # and visual compatibility instead of a fixed slot shortcut.
            progress[:target_slot] = 1.0
            progress[target_slot + 1 :] = 0.0

        template = np.zeros_like(bank['montages'][0, 0])
        montage = _compose_demonstration_montage(montages, template)
        enabled = bool(np.any(slot_mask))
        base_image = _parse_image(data['observation/image'])
        wrist_image = _parse_image(data['observation/wrist_image'])
        inputs = {
            'state': state,
            'image': {
                'base_0_rgb': base_image,
                'left_wrist_0_rgb': wrist_image,
                'right_wrist_0_rgb': montage,
            },
            'image_mask': {
                'base_0_rgb': np.True_,
                'left_wrist_0_rgb': np.True_,
                'right_wrist_0_rgb': np.bool_(enabled),
            },
            'demonstration_actions': actions,
            'demonstration_plan': plans,
            'demonstration_mask': np.bool_(enabled),
            'demonstration_reliability_target': np.float32(1.0 if enabled else -1.0),
            'demonstration_slot_mask': slot_mask,
            'demonstration_progress': progress,
            'demonstration_router_target': np.int32(target_slot),
        }
        if 'actions' in data:
            inputs['actions'] = data['actions']
        if enabled:
            inputs['prompt'] = _compositional_router_prompt(
                prompt, retrieved_prompts, training=training
            )
        else:
            inputs['prompt'] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    future_visual_supervision: bool = False
    future_state_supervision: bool = False
    task_progress_supervision: bool = False
    episode_metadata_path: str | None = None

    def __post_init__(self):
        if self.task_progress_supervision and self.episode_metadata_path is None:
            raise ValueError('task progress supervision requires episode metadata')

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image, base_future = _parse_current_future_image(
            data['observation/image'],
            future_visual_supervision=self.future_visual_supervision,
        )
        wrist_image, wrist_future = _parse_current_future_image(
            data['observation/wrist_image'],
            future_visual_supervision=self.future_visual_supervision,
        )

        state, future_state, future_state_mask = _parse_current_future_state(
            data,
            future_state_supervision=self.future_state_supervision,
        )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            'state': state,
            'image': {
                'base_0_rgb': base_image,
                'left_wrist_0_rgb': wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                'right_wrist_0_rgb': np.zeros_like(base_image),
            },
            'image_mask': {
                'base_0_rgb': np.True_,
                'left_wrist_0_rgb': np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                'right_wrist_0_rgb': (
                    np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }
        _attach_future_visual_inputs(
            inputs, data, base_future, wrist_future
        )
        _attach_future_state_inputs(
            inputs, future_state, future_state_mask
        )
        progress_target = _task_progress_from_frame(
            data,
            enabled=self.task_progress_supervision,
            metadata_path=self.episode_metadata_path,
        )
        if progress_target is not None:
            inputs['task_progress_target'] = progress_target

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if 'actions' in data:
            inputs['actions'] = data['actions']

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if 'prompt' in data:
            inputs['prompt'] = data['prompt']

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {'actions': np.asarray(data['actions'][:, :7])}
