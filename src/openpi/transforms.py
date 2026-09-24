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
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from openpi_client import image_tools


DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar('T')
S = TypeVar('S')


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(
        self,
        *,
        inputs: Sequence[DataTransformFn] = (),
        outputs: Sequence[DataTransformFn] = (),
    ) -> 'Group':
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(
            inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs)
        )


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and 'prompt' not in data:
            data['prompt'] = np.asarray(self.prompt)
        return data


def _stat_array_like(value, reference):
    """Materialize a normalization constant on the reference dtype/backend."""
    if isinstance(reference, np.ndarray):
        return np.asarray(value, dtype=reference.dtype)
    if type(reference).__module__.startswith('torch'):
        import torch

        return torch.as_tensor(
            np.asarray(value), dtype=reference.dtype, device=reference.device
        )
    return jax.numpy.asarray(value, dtype=reference.dtype)


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            (
                self._normalize_quantile
                if self.use_quantiles
                else self._normalize
            ),
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = (
            _stat_array_like(stats.mean[..., : x.shape[-1]], x),
            _stat_array_like(stats.std[..., : x.shape[-1]], x),
        )
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01 = _stat_array_like(stats.q01[..., : x.shape[-1]], x)
        q99 = _stat_array_like(stats.q99[..., : x.shape[-1]], x)
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            (
                self._unnormalize_quantile
                if self.use_quantiles
                else self._unnormalize
            ),
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(
            _stat_array_like(stats.mean, x),
            x.shape[-1], axis=-1, value=0.0,
        )
        std = pad_to_dim(
            _stat_array_like(stats.std, x),
            x.shape[-1], axis=-1, value=1.0,
        )
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01 = _stat_array_like(stats.q01, x)
        q99 = _stat_array_like(stats.q99, x)
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [
                    (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01,
                    x[..., dim:],
                ],
                axis=-1,
            )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data['image'] = {
            k: image_tools.resize_with_pad(v, self.height, self.width)
            for k, v in data['image'].items()
        }
        if 'future_image' in data:
            data['future_image'] = {
                k: image_tools.resize_with_pad(v, self.height, self.width)
                for k, v in data['future_image'].items()
            }
        return data


@dataclasses.dataclass(frozen=True)
class RejectRACGRightCamera(DataTransformFn):
    """Fail closed before JIT when RACG-v1's unsupported view is valid."""

    def __call__(self, data: DataDict) -> DataDict:
        masks = data.get('image_mask')
        if masks is None or 'right_wrist_0_rgb' not in masks:
            raise ValueError('RACG input requires an explicit right-wrist mask')
        if bool(np.any(np.asarray(masks['right_wrist_0_rgb']))):
            raise ValueError('RACG-v1 rejects a valid right-wrist camera')
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data['actions'] = data['actions'][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if 'actions' not in data or self.mask is None:
            return data

        state, actions = data['state'], data['actions']
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2
        )
        data['actions'] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if 'actions' not in data or self.mask is None:
            return data

        state, actions = data['state'], data['actions']
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2
        )
        data['actions'] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop('prompt', None)) is None:
            raise ValueError('Prompt is required')

        if self.discrete_state_input:
            if (state := data.get('state', None)) is None:
                raise ValueError('State is required.')
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {
            **data,
            'tokenized_prompt': tokens,
            'tokenized_prompt_mask': token_masks,
        }


_PSM_PICK_PREFIXES = ('pick up the ', 'pick the ', 'pick up ')
_PSM_PICK_SPLITS = (
    (' and place them ', 'place'),
    (' and put them ', 'put'),
    (' and place it ', 'place'),
    (' and put it ', 'put'),
    (' and place ', 'place'),
    (' and put ', 'put'),
)
_PSM_SOURCE_MARKERS = (
    (' next to the ', 'next_to'),
    (' on the ', 'on'),
    (' in the ', 'in'),
)
_PSM_DESTINATION_PREFIXES = (('on the ', 'on'), ('in the ', 'in'))
_PSM_CONDITIONS = (
    (' with the candle lit', 'candle_lit'),
    (' with the stove turned on', 'stove_on'),
)
_PSM_OPERATION_LABELS = {'pick_place': 0, 'push': 1, 'open': 2, 'close': 3}
_PSM_SOURCE_RELATION_LABELS = {'none': 0, 'on': 1, 'in': 2, 'next_to': 3}
_PSM_DESTINATION_RELATION_LABELS = {
    'none': 0,
    'on': 1,
    'in': 2,
    'between': 3,
}
# ``racg_relation_kind`` is an inference-safe prompt-derived bit field.  Its
# low four bits retain the 0..15 source x destination relation class; bit 4
# records that the reference role must bind two distinct object modes.  The
# auxiliary CE target remains the unflagged low-four-bit class.
_RACG_TWO_REFERENCE_FLAG = np.int32(1 << 4)
_PSM_CONDITION_LABELS = {'none': 0, 'candle_lit': 1, 'stove_on': 2}
_PSM_DESTINATION_QUALIFIER_LABELS = {
    'none': 0,
    'between': 1,
    'at_top_of': 2,
    'top_of': 3,
    'top_layer_of': 4,
    'middle_layer_of': 5,
    'on': 6,
    'conditioned': 7,
}
_PSM_CONDITION_PHRASES = {
    'candle_lit': 'candle lit',
    'stove_on': 'stove turned on',
}


@dataclasses.dataclass(frozen=True)
class _FactorizedInstruction:
    operation: str
    manipulated_entity: str
    source_relation: str
    source_reference_phrase: str
    destination_relation: str
    destination_phrase: str
    destination_reference_phrases: tuple[str, ...]
    destination_qualifier: str
    destination_qualifier_phrase: str
    condition: str


def _normalize_factorized_instruction(instruction: str) -> str:
    normalized = ' '.join(instruction.strip().lower().split())
    return normalized.replace('pocelain', 'porcelain')


def _split_factorized_role(
    value: str, markers: tuple[tuple[str, str], ...]
) -> tuple[str, str, str]:
    for marker, label in markers:
        if marker in value:
            left, right = value.split(marker, 1)
            if left and right:
                return left, label, right
    return value, 'none', ''


def _parse_factorized_instruction(instruction: str) -> _FactorizedInstruction:
    text = _normalize_factorized_instruction(instruction)
    for operation in ('open', 'close'):
        prefix = next(
            (
                candidate
                for candidate in (
                    f'{operation} the ',
                    f'{operation} all of the ',
                )
                if text.startswith(candidate) and text.removeprefix(candidate)
            ),
            None,
        )
        if prefix is not None:
            entity = text.removeprefix(prefix)
            qualifier = 'none'
            qualifier_phrase = ''
            for candidate, phrase in (
                ('top_layer_of', 'top layer of'),
                ('middle_layer_of', 'middle layer of'),
                ('top_of', 'top of'),
            ):
                if entity.startswith(f'{phrase} the '):
                    qualifier = candidate
                    qualifier_phrase = phrase
                    break
            return _FactorizedInstruction(
                operation,
                entity,
                'none',
                '',
                'none',
                '',
                (),
                qualifier,
                qualifier_phrase,
                'none',
            )

    # Long-horizon evaluation instructions can start with a causal removal,
    # contain one or more pick/place clauses, and optionally end with a close.
    # The existing two-role interface cannot encode every clause separately,
    # but it can preserve the overall workflow endpoints: bind the first
    # manipulated object and its source, then the final explicit destination.
    # The complete unchanged prompt still drives the ordered subgoal queries.
    take_sequence = re.match(
        r'take(?: out)? the (.+?) (out of|on) the (.+?)(?:,| and pick)',
        text,
    )
    if take_sequence:
        destinations = list(
            re.finditer(
                r'(?:place|put) (?:it|them) (on|in) the '
                r'(.+?)(?=,| then |$)',
                text,
            )
        )
        if destinations:
            destination = destinations[-1]
            return _FactorizedInstruction(
                'pick_place',
                take_sequence.group(1),
                'in' if take_sequence.group(2) == 'out of' else 'on',
                take_sequence.group(3),
                destination.group(1),
                destination.group(2),
                (),
                'none',
                '',
                'none',
            )

    push = re.fullmatch(r'push the (.+?) to the region between the (.+)', text)
    if push:
        push_references = tuple(push.group(2).split(' and ', 1))
        if len(push_references) == 1:
            # ``between the mugs`` names a class twice implicitly.  Preserve
            # the same open-vocabulary span for two ordered instance queries;
            # their sequential object-slot exclusion grounds distinct mugs
            # and makes their visual midpoint available to the policy.
            push_references = (push_references[0], push_references[0])
        return _FactorizedInstruction(
            'push',
            push.group(1),
            'none',
            '',
            'between',
            push.group(2),
            push_references,
            'between',
            'between',
            'none',
        )

    remainder = None
    for prefix in _PSM_PICK_PREFIXES:
        if text.startswith(prefix):
            remainder = text.removeprefix(prefix)
            break
    if remainder is None:
        raise ValueError(f'unsupported instruction operation: {instruction!r}')

    pickup_clause = None
    destination_clause = None
    for marker, _ in _PSM_PICK_SPLITS:
        if marker in remainder:
            pickup_clause, destination_clause = remainder.split(marker, 1)
            break
    if not pickup_clause or not destination_clause:
        raise ValueError(f'cannot split pick/place instruction: {instruction!r}')

    manipulated, source_relation, source_reference_phrase = _split_factorized_role(
        pickup_clause, _PSM_SOURCE_MARKERS
    )
    destination_relation = None
    destination_phrase = None
    for prefix, relation in _PSM_DESTINATION_PREFIXES:
        if destination_clause.startswith(prefix):
            destination_relation = relation
            destination_phrase = destination_clause.removeprefix(prefix)
            break
    if destination_relation is None or not destination_phrase:
        raise ValueError(f'cannot parse destination: {instruction!r}')
    condition = 'none'
    for suffix, label in _PSM_CONDITIONS:
        if destination_phrase.endswith(suffix):
            destination_phrase = destination_phrase[: -len(suffix)]
            condition = label
            break
    destination_reference_phrases: tuple[str, ...] = ()
    destination_qualifier = 'none'
    destination_qualifier_phrase = ''
    between = re.fullmatch(
        r'(.+?) between the (.+?) and the (.+)', destination_phrase
    )
    at_top = re.fullmatch(
        r'(.+?) at the top of the (.+)', destination_phrase
    )
    nested_on = re.fullmatch(r'(.+?) on the (.+)', destination_phrase)
    if between:
        destination_reference_phrases = (between.group(2), between.group(3))
        destination_qualifier = 'between'
        destination_qualifier_phrase = 'between'
    elif at_top:
        destination_reference_phrases = (at_top.group(2),)
        destination_qualifier = 'at_top_of'
        destination_qualifier_phrase = 'at the top of'
    elif destination_phrase.startswith('top of the '):
        destination_reference_phrases = (
            destination_phrase.removeprefix('top of the '),
        )
        destination_qualifier = 'top_of'
        destination_qualifier_phrase = 'top of'
    elif destination_phrase.startswith('top layer of the '):
        destination_reference_phrases = (
            destination_phrase.removeprefix('top layer of the '),
        )
        destination_qualifier = 'top_layer_of'
        destination_qualifier_phrase = 'top layer of'
    elif destination_phrase.startswith('middle layer of the '):
        destination_reference_phrases = (
            destination_phrase.removeprefix('middle layer of the '),
        )
        destination_qualifier = 'middle_layer_of'
        destination_qualifier_phrase = 'middle layer of'
    elif nested_on:
        destination_reference_phrases = (nested_on.group(2),)
        destination_qualifier = 'on'
        destination_qualifier_phrase = 'on'
    elif condition != 'none':
        destination_qualifier = 'conditioned'
        destination_qualifier_phrase = _PSM_CONDITION_PHRASES[condition]
    return _FactorizedInstruction(
        'pick_place',
        manipulated,
        source_relation,
        source_reference_phrase,
        destination_relation,
        destination_phrase,
        destination_reference_phrases,
        destination_qualifier,
        destination_qualifier_phrase,
        condition,
    )


def _stable_factorized_role_identity(role_index: int, value: str) -> np.int32:
    normalized = ' '.join(value.strip().lower().split())
    if not normalized:
        return np.int32(-1)
    digest = hashlib.blake2s(
        f'psm-role-v1:{role_index}:{normalized}'.encode(), digest_size=4
    ).digest()
    return np.int32(int.from_bytes(digest, 'little') & 0x7FFF_FFFF)


def _stable_racg_role_identity(role_name: str, value: str) -> np.int32:
    """Match experiments/pi05/racg_v1_supervision.py exactly."""
    normalized = ' '.join(value.strip().lower().split())
    if not normalized:
        return np.int32(-1)
    digest = hashlib.blake2s(
        f'racg-role-v1:{role_name}:{normalized}'.encode(), digest_size=4
    ).digest()
    return np.int32(int.from_bytes(digest, 'little') & 0x7FFF_FFFF)


def _locate_factorized_span(
    text: str, value: str, *, last: bool
) -> tuple[int, int]:
    candidates = (value, value.replace('porcelain', 'pocelain'))
    matches = []
    for candidate in candidates:
        index = text.rfind(candidate) if last else text.find(candidate)
        if index >= 0:
            matches.append((index, index + len(candidate)))
    if not matches:
        raise ValueError(f'cannot locate role span {value!r} in {text!r}')
    return max(matches) if last else min(matches)


@dataclasses.dataclass(frozen=True)
class FactorizedTokenizePrompt(DataTransformFn):
    """Tokenize the parent prompt and add open-vocabulary PSM role targets.

    The parent token ids are byte-for-byte identical to ``TokenizePrompt``.
    Character spans produce manipulated/destination roles plus independent
    source and destination-reference masks. Equality and compact relation
    labels are marked valid solely for sequence-training samples; serving uses
    the masks but never consumes training labels.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop('prompt', None)) is None:
            raise ValueError('Prompt is required')
        if not isinstance(prompt, str):
            prompt = str(np.asarray(prompt).item())
        state = data.get('state') if self.discrete_state_input else None
        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        parsed = _parse_factorized_instruction(prompt)

        cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
        if state is None:
            character_text = cleaned
            prompt_offset = 0
            newline_ids = self.tokenizer._tokenizer.encode('\n')
        else:
            discretized = np.digitize(
                np.asarray(state), bins=np.linspace(-1, 1, 257)[:-1]
            ) - 1
            state_text = ' '.join(map(str, discretized))
            character_text = f'Task: {cleaned}, State: {state_text};\nAction: '
            prompt_offset = len('Task: ')
            newline_ids = []
        proto = self.tokenizer._tokenizer.encode_as_immutable_proto(
            character_text
        )
        unpadded_ids = [self.tokenizer._tokenizer.bos_id()] + [
            piece.id for piece in proto.pieces
        ] + list(newline_ids)
        valid_length = int(np.sum(token_mask))
        if unpadded_ids[:valid_length] != tokens[:valid_length].tolist():
            raise RuntimeError('factorized role tokenization drifted from parent')
        if len(unpadded_ids) > len(tokens):
            raise ValueError('prompt truncation invalidated factorized role targets')

        lowered = cleaned.lower()
        target_start, target_end = _locate_factorized_span(
            lowered, parsed.manipulated_entity, last=False
        )
        spans: list[tuple[int, int] | None] = [
            (prompt_offset + target_start, prompt_offset + target_end),
            None,
        ]
        if parsed.destination_phrase:
            destination_start, destination_end = _locate_factorized_span(
                lowered, parsed.destination_phrase, last=True
            )
            spans[1] = (
                prompt_offset + destination_start,
                prompt_offset + destination_end,
            )
        source_reference_span = None
        if parsed.source_reference_phrase:
            source_start, source_end = _locate_factorized_span(
                lowered, parsed.source_reference_phrase, last=False
            )
            source_reference_span = (
                prompt_offset + source_start,
                prompt_offset + source_end,
            )
        destination_reference_spans: list[tuple[int, int] | None] = [None, None]
        for reference_index, reference_phrase in enumerate(
            parsed.destination_reference_phrases
        ):
            reference_start, reference_end = _locate_factorized_span(
                lowered, reference_phrase, last=True
            )
            destination_reference_spans[reference_index] = (
                prompt_offset + reference_start,
                prompt_offset + reference_end,
            )
        condition_span = None
        if parsed.condition != 'none':
            condition_start, condition_end = _locate_factorized_span(
                lowered,
                _PSM_CONDITION_PHRASES[parsed.condition],
                last=True,
            )
            condition_span = (
                prompt_offset + condition_start,
                prompt_offset + condition_end,
            )
        destination_qualifier_span = None
        if parsed.destination_qualifier_phrase:
            qualifier_start, qualifier_end = _locate_factorized_span(
                lowered,
                parsed.destination_qualifier_phrase,
                last=True,
            )
            destination_qualifier_span = (
                prompt_offset + qualifier_start,
                prompt_offset + qualifier_end,
            )
        role_mask = np.zeros((2, len(tokens)), dtype=np.bool_)
        for role_index, span in enumerate(spans):
            if span is None:
                continue
            start, end = span
            for piece_index, piece in enumerate(proto.pieces, start=1):
                if piece.begin < end and piece.end > start:
                    role_mask[role_index, piece_index] = True
            if not np.any(role_mask[role_index]):
                raise RuntimeError('a valid factorized role has no token coverage')
        role_mask &= token_mask[None]
        source_reference_mask = np.zeros((len(tokens),), dtype=np.bool_)
        if source_reference_span is not None:
            start, end = source_reference_span
            for piece_index, piece in enumerate(proto.pieces, start=1):
                if piece.begin < end and piece.end > start:
                    source_reference_mask[piece_index] = True
            if not np.any(source_reference_mask):
                raise RuntimeError('a valid source reference has no token coverage')
        source_reference_mask &= token_mask
        destination_reference_mask = np.zeros(
            (2, len(tokens)), dtype=np.bool_
        )
        for reference_index, span in enumerate(destination_reference_spans):
            if span is None:
                continue
            start, end = span
            for piece_index, piece in enumerate(proto.pieces, start=1):
                if piece.begin < end and piece.end > start:
                    destination_reference_mask[reference_index, piece_index] = True
            if not np.any(destination_reference_mask[reference_index]):
                raise RuntimeError(
                    'a valid destination reference has no token coverage'
                )
        destination_reference_mask &= token_mask[None]
        condition_mask = np.zeros((len(tokens),), dtype=np.bool_)
        if condition_span is not None:
            start, end = condition_span
            for piece_index, piece in enumerate(proto.pieces, start=1):
                if piece.begin < end and piece.end > start:
                    condition_mask[piece_index] = True
            if not np.any(condition_mask):
                raise RuntimeError('a valid condition has no token coverage')
        condition_mask &= token_mask
        hazard_mask = np.zeros((len(tokens),), dtype=np.bool_)
        hazard_identity = ''
        if parsed.condition != 'none':
            hazard_identity = {
                'candle_lit': 'candle',
                'stove_on': 'stove',
            }[parsed.condition]
            hazard_start, hazard_end = _locate_factorized_span(
                lowered, hazard_identity, last=True
            )
            hazard_start += prompt_offset
            hazard_end += prompt_offset
            for piece_index, piece in enumerate(proto.pieces, start=1):
                if piece.begin < hazard_end and piece.end > hazard_start:
                    hazard_mask[piece_index] = True
            hazard_mask &= token_mask
            if not np.any(hazard_mask):
                raise RuntimeError('a valid RACG hazard has no token coverage')
        destination_qualifier_mask = np.zeros((len(tokens),), dtype=np.bool_)
        if destination_qualifier_span is not None:
            start, end = destination_qualifier_span
            for piece_index, piece in enumerate(proto.pieces, start=1):
                if piece.begin < end and piece.end > start:
                    destination_qualifier_mask[piece_index] = True
            if not np.any(destination_qualifier_mask):
                raise RuntimeError(
                    'a valid destination qualifier has no token coverage'
                )
        destination_qualifier_mask &= token_mask
        role_valid = np.asarray([True, spans[1] is not None], dtype=np.bool_)
        training_valid = all(
            key in data
            for key in ('episode_index', 'memory_window_start', 'memory_scale_id')
        )
        if any(
            key in data
            for key in ('episode_index', 'memory_window_start', 'memory_scale_id')
        ) and not training_valid:
            raise ValueError('partial persistent-memory window metadata')
        identity_labels = np.asarray(
            [
                _stable_factorized_role_identity(
                    0, parsed.manipulated_entity
                ),
                _stable_factorized_role_identity(
                    1, parsed.destination_phrase
                ),
            ],
            dtype=np.int32,
        )
        if not training_valid:
            identity_labels.fill(-1)
        racg_role_mask = np.zeros((6, len(tokens)), dtype=np.bool_)
        # agent/free-space are always-present structural roles but have no
        # fabricated textual span.  The other masks are exact unions of the
        # deterministic factor parser's existing token spans.
        racg_role_mask[1] = role_mask[0]
        racg_role_mask[2] = source_reference_mask | np.any(
            destination_reference_mask, axis=0
        )
        racg_role_mask[3] = role_mask[1]
        racg_role_mask[4] = hazard_mask
        racg_role_valid = np.asarray(
            [
                True,
                role_valid[0],
                np.any(racg_role_mask[2]),
                role_valid[1],
                np.any(hazard_mask),
                True,
            ],
            dtype=np.bool_,
        )
        racg_relation_target = np.int32(
            _PSM_SOURCE_RELATION_LABELS[parsed.source_relation] * 4
            + _PSM_DESTINATION_RELATION_LABELS[parsed.destination_relation]
        )
        two_references = (
            parsed.destination_relation == 'between'
            or parsed.destination_qualifier == 'between'
        )
        racg_relation_kind = np.int32(
            racg_relation_target
            | (_RACG_TWO_REFERENCE_FLAG if two_references else np.int32(0))
        )
        image_mask = data.get('image_mask', {})
        both_views = bool(
            image_mask.get('base_0_rgb', True)
            and image_mask.get('left_wrist_0_rgb', True)
        )
        racg_identity_values = (
            'robot gripper',
            parsed.manipulated_entity,
            (
                parsed.source_reference_phrase
                or next(iter(parsed.destination_reference_phrases), '')
            ),
            parsed.destination_phrase,
            hazard_identity,
            'free space',
        )
        racg_identity_labels = np.asarray(
            [
                _stable_racg_role_identity(role_name, value)
                for role_name, value in zip(
                    ('agent', 'target', 'reference', 'destination', 'hazard', 'free_space'),
                    racg_identity_values,
                    strict=True,
                )
            ],
            dtype=np.int32,
        )
        racg_identity_labels[~racg_role_valid] = -1
        if not training_valid:
            racg_identity_labels.fill(-1)
        invalid_label = np.int32(-1)
        return {
            **data,
            'tokenized_prompt': tokens,
            'tokenized_prompt_mask': token_mask,
            'factorized_role_span_mask': role_mask,
            'factorized_source_reference_span_mask': source_reference_mask,
            'factorized_destination_reference_span_mask': (
                destination_reference_mask
            ),
            'factorized_condition_span_mask': condition_mask,
            'factorized_destination_qualifier_span_mask': (
                destination_qualifier_mask
            ),
            'factorized_role_valid_mask': role_valid,
            'factorized_role_identity_labels': identity_labels,
            'factorized_operation_label': (
                np.int32(_PSM_OPERATION_LABELS[parsed.operation])
                if training_valid
                else invalid_label
            ),
            'factorized_source_relation_label': (
                np.int32(_PSM_SOURCE_RELATION_LABELS[parsed.source_relation])
                if training_valid
                else invalid_label
            ),
            'factorized_destination_relation_label': (
                np.int32(
                    _PSM_DESTINATION_RELATION_LABELS[
                        parsed.destination_relation
                    ]
                )
                if training_valid
                else invalid_label
            ),
            'factorized_condition_label': (
                np.int32(_PSM_CONDITION_LABELS[parsed.condition])
                if training_valid
                else invalid_label
            ),
            'factorized_destination_qualifier_label': (
                np.int32(
                    _PSM_DESTINATION_QUALIFIER_LABELS[
                        parsed.destination_qualifier
                    ]
                )
                if training_valid
                else invalid_label
            ),
            'factorized_auxiliary_valid': np.bool_(training_valid),
            'racg_role_span_mask': racg_role_mask,
            'racg_role_valid_mask': racg_role_valid,
            'racg_relation_kind': racg_relation_kind,
            'racg_role_identity_labels': racg_identity_labels,
            # These targets remain loss-only; they are deliberately invalid
            # for ordinary inference records.
            'racg_relation_target': (
                racg_relation_target if training_valid else invalid_label
            ),
            'racg_relation_valid': np.bool_(training_valid),
            'racg_crossview_role_valid': (
                racg_role_valid & both_views & training_valid
            ),
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop('prompt', None)) is None:
            raise ValueError('Prompt is required')

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data['state'], data.get('actions')
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(
            prompt, state, actions
        )
        return {
            **data,
            'tokenized_prompt': tokens,
            'tokenized_prompt_mask': token_mask,
            'token_ar_mask': ar_mask,
            'token_loss_mask': loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if 'actions' not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop('actions')
        actions = self.tokenizer.extract_actions(
            tokens.astype(np.int32), self.action_horizon, self.action_dim
        )
        return {
            **data,
            'actions': actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: object

    def _get_prompt(self, task_index: int) -> str | None:
        # LeRobot <=0.3 exposes a dictionary; preserve that path for the
        # currently running VLA-Arena jobs.
        if isinstance(self.tasks, Mapping):
            prompt = self.tasks.get(task_index)
            return None if prompt is None else str(prompt)

        # LeRobot 0.4 / RoboDojo v3 exposes a pandas DataFrame whose textual
        # task is the index and whose integer id is the task_index column.
        columns = getattr(self.tasks, 'columns', None)
        index = getattr(self.tasks, 'index', None)
        if columns is not None and index is not None and 'task_index' in columns:
            matches = index[self.tasks['task_index'] == task_index]
            if len(matches) > 0:
                return str(matches[0])
        return None

    def __call__(self, data: DataDict) -> DataDict:
        if 'task_index' not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data['task_index'])
        if (prompt := self._get_prompt(task_index)) is None:
            raise ValueError(
                f'{task_index=} not found in task mapping: {self.tasks}'
            )

        return {**data, 'prompt': prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data['state'] = pad_to_dim(
            data['state'], self.model_action_dim, axis=-1
        )
        if 'future_state' in data:
            data['future_state'] = pad_to_dim(
                data['future_state'], self.model_action_dim, axis=-1
            )
        if 'actions' in data:
            data['actions'] = pad_to_dim(
                data['actions'], self.model_action_dim, axis=-1
            )
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep='/')


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep='/')


def transform_dict(
    patterns: Mapping[str, str | None], tree: at.PyTree
) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = (
                    pattern.sub(repl, k, count=1) if repl is not None else None
                )
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + '/'):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T],
    selector: at.PyTree[S],
    fn: Callable[[T, S], T],
    *,
    strict: bool = False,
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f'Selector key {k} not found in tree')

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(
    x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0
) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f'quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99.'
            )
