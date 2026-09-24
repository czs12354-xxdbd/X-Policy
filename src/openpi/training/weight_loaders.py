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
import json
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from experiments.pi05 import psm_sdla_v3_aux_transplant_weight_loader as _sdla_v3_transplant
from experiments.pi05 import psm_sdla_v3_geometry_checkpoint_leaf_contract_v1 as _geometry_leaf_contract
from experiments.pi05 import psm_sdla_v3_geometry_hmca_v4_checkpoint_leaf_contract as _joint51_leaf_contract
from experiments.pi05 import molmo2_er_geometry_dual_residual_candidate_v1 as _dual_geometry
from experiments.pi05 import molmo2_er_geometry_dual_residual_checkpoint_contract_v1 as _dual_geometry_checkpoint
from experiments.pi05 import molmo2_er_geometry_next_round_loader_v1 as _geometry_warmstart_loader
from experiments.pi05 import action_conditioned_temporal_object_residual_candidate_v1 as _temporal_role
from experiments.pi05 import action_conditioned_temporal_role_memory_checkpoint_contract_v1 as _temporal_role_checkpoint
from experiments.pi05 import cross_view_role_consensus_candidate_v1 as _cross_view_role
from experiments.pi05 import contact_risk_calibrated_role_residual_candidate_v1 as _contact_risk
from experiments.pi05 import relational_role_composer_residual_candidate_v1 as _relational_role
from experiments.pi05 import clause_role_binding_verifier_candidate_v1 as _clause_role_binding
from experiments.pi05 import semantic_frontier_completion_verifier_candidate_v1 as _semantic_frontier


logger = logging.getLogger(__name__)


def _reference_shape_dtype(value) -> tuple[tuple[int, ...], np.dtype]:
    """Read geometry from an array or the abstract tree used by trainer init."""
    return tuple(value.shape), np.dtype(value.dtype)


def _zeros_from_reference(value) -> np.ndarray:
    shape, dtype = _reference_shape_dtype(value)
    return np.zeros(shape, dtype=dtype)


def _finite_if_materialized(value) -> bool:
    return isinstance(value, jax.ShapeDtypeStruct) or bool(
        np.isfinite(np.asarray(value)).all()
    )


def _byte_zero_if_materialized(value) -> bool:
    """Abstract leaves are initialized later; concrete test leaves stay audited."""
    return isinstance(value, jax.ShapeDtypeStruct) or not np.count_nonzero(
        np.asarray(value).view(np.uint8)
    )


def _same_value_or_abstract(left, right) -> bool:
    if isinstance(left, jax.ShapeDtypeStruct) or isinstance(
        right, jax.ShapeDtypeStruct
    ):
        return _reference_shape_dtype(left) == _reference_shape_dtype(right)
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _index_by_rendered_path(flat, *, source: str):
    """Match Orbax string list indexes to NNX integer indexes without aliasing."""
    indexed = {}
    for path, value in flat.items():
        rendered = '/'.join(map(str, path))
        if rendered in indexed and indexed[rendered][0] != path:
            raise ValueError(
                f'{source} contains ambiguous paths rendering as {rendered!r}'
            )
        indexed[rendered] = (path, value)
    return indexed


def _joint_sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _joint_canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Parameters matching this expression may be absent from the checkpoint
    # and are initialized from the target model. This defaults to the historic
    # LoRA-only behavior and also supports small, backward-compatible adapters.
    missing_regex: str = '.*lora.*'

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        return _merge_params(
            loaded_params, params, missing_regex=self.missing_regex
        )


@dataclasses.dataclass(frozen=True)
class DirectPsmWithHetmWarmstartWeightLoader(WeightLoader):
    """Exact Direct parent, sealed HETM bundle, and closed HMCA bridge."""

    params_path: str
    hetm_bundle_path: str
    expected_parent_arrays: int = 848
    expected_hetm_arrays: int = 21
    expected_target_hetm_arrays: int = 22

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_target = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        target_index = _index_by_rendered_path(
            flat_target, source='Direct PSM+HETM target'
        )
        parent_index = _index_by_rendered_path(
            flat_parent, source='Direct PSM parent'
        )
        if len(flat_parent) != self.expected_parent_arrays:
            raise ValueError('Direct PSM parent leaf count drifted')
        if (
            len(flat_target)
            != self.expected_parent_arrays + self.expected_target_hetm_arrays
        ):
            raise ValueError('Direct PSM+HETM target leaf count drifted')
        if not set(parent_index).issubset(target_index):
            raise ValueError('joint target dropped a Direct parent leaf')
        new_paths = set(target_index) - set(parent_index)
        if len(new_paths) != self.expected_target_hetm_arrays or any(
            not path.startswith('hetm/') for path in new_paths
        ):
            raise ValueError('joint target has a non-HETM initializer leaf')

        bundle = pathlib.Path(self.hetm_bundle_path).expanduser().resolve()
        manifest_path = bundle / 'manifest.json'
        if not bundle.is_dir() or bundle.is_symlink():
            raise ValueError('HETM warm-start directory is invalid')
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError('HETM warm-start manifest is absent')
        manifest = json.loads(manifest_path.read_text())
        core = dict(manifest)
        content_hash = core.pop('manifest_content_sha256', None)
        if content_hash != _joint_canonical_sha256(core):
            raise ValueError('HETM warm-start manifest digest drifted')
        if (
            manifest.get('schema_version') != 'hetm_libero_structural_bundle/v1'
            or manifest.get('status') != 'complete_cpu_structural_pretraining_bundle'
            or manifest.get('target_namespace') != 'hetm'
        ):
            raise ValueError('HETM warm-start contract drifted')
        leaves = manifest.get('leaves')
        if not isinstance(leaves, list) or len(leaves) != self.expected_hetm_arrays:
            raise ValueError('HETM warm-start leaf count drifted')
        if [row.get('index') for row in leaves] != list(range(len(leaves))):
            raise ValueError('HETM warm-start indexes are not canonical')

        target_hetm = {
            rendered.removeprefix('hetm/'): (path, value)
            for rendered, (path, value) in target_index.items()
            if rendered.startswith('hetm/')
        }
        source = {}
        for row in leaves:
            relative = row.get('path')
            payload = (bundle / str(row.get('file'))).resolve()
            if (
                not isinstance(relative, str)
                or not relative
                or relative in source
                or payload.parent != bundle
                or not payload.is_file()
                or payload.is_symlink()
            ):
                raise ValueError('HETM warm-start payload contract drifted')
            if _joint_sha256_file(payload) != row.get('sha256'):
                raise ValueError(f'HETM payload digest drifted: {relative}')
            value = np.load(payload, allow_pickle=False)
            if (
                list(value.shape) != row.get('shape')
                or value.dtype.name != row.get('dtype')
                or not np.isfinite(value).all()
            ):
                raise ValueError(f'HETM payload geometry drifted: {relative}')
            source[relative] = value
        initialized_only = {'hmca_condition_out/kernel'}
        if set(source) | initialized_only != set(target_hetm):
            raise ValueError('HETM warm-start/bridge namespace differs from target')

        merged = dict(flat_target)
        for rendered, (_, value) in parent_index.items():
            path, reference = target_index[rendered]
            array = np.asarray(value)
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f'Direct parent geometry drifted: {path}')
            merged[path] = array.astype(reference.dtype, copy=False)
        for relative, value in source.items():
            path, reference = target_hetm[relative]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f'HETM target geometry drifted: {relative}')
            merged[path] = value
        for relative in (
            'film_out/kernel',
            'prior_out/kernel',
            'hmca_condition_out/kernel',
        ):
            if not _byte_zero_if_materialized(
                merged[target_hetm[relative][0]]
            ):
                raise ValueError(f'HETM policy boundary is open: {relative}')
        for rendered, (_, value) in parent_index.items():
            path, reference = target_index[rendered]
            expected = np.asarray(value).astype(reference.dtype, copy=False)
            if not np.array_equal(np.asarray(merged[path]), expected):
                raise AssertionError(
                    f'HETM merge changed Direct parent: {rendered}'
                )
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class DirectPsmHetmWithRacgInitializerWeightLoader(WeightLoader):
    """Strict 870-leaf Direct848+HETM parent plus a 31-leaf RACG delta."""

    params_path: str
    expected_parent_arrays: int = 870
    expected_racg_arrays: int = 31
    warmstart_status: str = 'parent_hetm_semantic_transport_v3'

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        if self.warmstart_status != 'parent_hetm_semantic_transport_v3':
            raise ValueError('RACG semantic-transport status drifted')
        parent = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_target = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)

        def rendered(flat, source):
            result = {}
            for path, value in flat.items():
                key = '/'.join(map(str, path))
                if key in result:
                    raise ValueError(f'{source} contains an ambiguous path: {key}')
                result[key] = (path, value)
            return result

        target_index = rendered(flat_target, 'RACG target')
        parent_index = rendered(flat_parent, 'Direct848+HETM parent')
        if len(parent_index) != self.expected_parent_arrays:
            raise ValueError('Direct848+HETM parent leaf count drifted')
        if len(target_index) != self.expected_parent_arrays + self.expected_racg_arrays:
            raise ValueError('Direct848+HETM+RACG target leaf count drifted')
        expected_new = {
            key for key in target_index if key.startswith('racg/')
        }
        if len(expected_new) != self.expected_racg_arrays:
            raise ValueError('RACG target namespace cardinality drifted')
        if set(target_index) - set(parent_index) != expected_new:
            raise ValueError('target differs from parent outside the RACG namespace')
        if set(parent_index) - set(target_index):
            raise ValueError('RACG successor dropped an inherited parent leaf')

        merged = dict(flat_target)
        for key, (target_path, reference) in target_index.items():
            if key in expected_new:
                continue
            _, value = parent_index[key]
            array = np.asarray(value)
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f'inherited parent geometry/value drifted: {key}')
            merged[target_path] = array.astype(reference.dtype, copy=False)
        initialized = _initialize_racg_from_hetm_semantic_transport(
            flax.traverse_util.unflatten_dict(merged)
        )
        flat_initialized = flax.traverse_util.flatten_dict(initialized)
        initialized_index = rendered(flat_initialized, 'initialized RACG target')
        for key, (_, parent_value) in parent_index.items():
            _, value = initialized_index[key]
            if not np.array_equal(np.asarray(value), np.asarray(parent_value)):
                raise AssertionError(f'RACG initialization changed parent leaf: {key}')
        for suffix in ('kernel', 'bias'):
            value = np.asarray(initialized_index[f'racg/graph_action_out/{suffix}'][1])
            if np.count_nonzero(value.view(np.uint8)):
                raise ValueError(f'RACG action boundary is not byte-zero: {suffix}')
        return initialized


@dataclasses.dataclass(frozen=True)
class DirectPsmHetmRacgWithExternalGeometryWeightLoader(WeightLoader):
    """Load the exact 901-leaf Direct848+HETM+RACG parent plus sealed EGP."""

    params_path: str
    external_geometry_bundle_path: str
    expected_parent_arrays: int = 901
    expected_external_arrays: int = 3
    expected_manifest_sha256: str = (
        'c1b9d8d9c6535b9f89cb70abe818c8468b3fcadde1f17975352223910b2cc562'
    )

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_target = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)

        def rendered(flat, source):
            result = {}
            for path, value in flat.items():
                key = '/'.join(map(str, path))
                if key in result:
                    raise ValueError(f'{source} contains an ambiguous path: {key}')
                result[key] = (path, value)
            return result

        target_index = rendered(flat_target, 'Direct848 RACG-EGP target')
        parent_index = rendered(flat_parent, 'Direct848 RACG parent')
        additions = {
            'racg_external_geometry/external_blend_gate',
            'racg_external_geometry/external_prefix_out',
            'racg_external_geometry/external_role_query',
        }
        if len(parent_index) != self.expected_parent_arrays:
            raise ValueError('Direct848 RACG parent leaf count drifted')
        if len(target_index) != self.expected_parent_arrays + self.expected_external_arrays:
            raise ValueError('Direct848 RACG-EGP target leaf count drifted')
        if set(target_index) - set(parent_index) != additions:
            raise ValueError('target differs from parent outside the three EGP leaves')
        if set(parent_index) - set(target_index):
            raise ValueError('EGP successor dropped an inherited parent leaf')

        merged = dict(flat_target)
        for key, (parent_path, value) in parent_index.items():
            target_path, reference = target_index[key]
            array = np.asarray(value)
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f'inherited parent geometry/value drifted: {key}')
            merged[target_path] = array.astype(reference.dtype, copy=False)

        bundle = pathlib.Path(self.external_geometry_bundle_path).expanduser().resolve()
        manifest_path = bundle / 'manifest.json'
        if not bundle.is_dir() or bundle.is_symlink():
            raise ValueError('external geometry bundle is invalid')
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError('external geometry manifest is absent')
        if _joint_sha256_file(manifest_path) != self.expected_manifest_sha256:
            raise ValueError('external geometry manifest SHA-256 drifted')
        manifest = json.loads(manifest_path.read_text())
        unsigned = dict(manifest)
        claimed = unsigned.pop('manifest_payload_sha256', None)
        if claimed != _joint_canonical_sha256(unsigned):
            raise ValueError('external geometry manifest seal drifted')
        if not (
            manifest.get('schema_version')
            == 'molmo2_er_geometry_warmstart_transplant/v1'
            and manifest.get('state') == 'committed_manifest_last'
            and manifest.get('leaf_count') == 2
            and manifest.get('optimizer_state_included') is False
            and manifest.get('parent_psm_action_vlm_arrays_included') is False
            and manifest.get('production_policy_out_included') is False
        ):
            raise ValueError('external geometry provenance/ownership drifted')

        specifications = (
            (
                'persistent_memory/geometry_aux_v3/prefix_out/kernel',
                'leaf_000_prefix_out.npy',
                (2048, 256),
                'racg_external_geometry/external_prefix_out',
            ),
            (
                'persistent_memory/geometry_aux_v3/role_query/embedding',
                'leaf_001_role_query.npy',
                (5, 2048),
                'racg_external_geometry/external_role_query',
            ),
        )
        leaves = manifest.get('leaves')
        if not isinstance(leaves, list) or len(leaves) != 2:
            raise ValueError('external geometry leaf manifest drifted')
        expected_files = {'manifest.json'}
        for index, (row, specification) in enumerate(zip(leaves, specifications, strict=True)):
            source_path, filename, shape, target_name = specification
            if not (
                row.get('index') == index
                and row.get('path') == source_path
                and row.get('filename') == filename
                and row.get('shape') == list(shape)
                and row.get('dtype') == 'float32'
            ):
                raise ValueError(f'external geometry leaf schema drifted: {target_name}')
            payload = bundle / filename
            if not (
                payload.resolve().parent == bundle
                and payload.is_file()
                and not payload.is_symlink()
                and _joint_sha256_file(payload) == row.get('file_sha256')
            ):
                raise ValueError(f'external geometry payload drifted: {target_name}')
            value = np.load(payload, allow_pickle=False)
            target_path, reference = target_index[target_name]
            if (
                value.dtype != np.float32
                or value.shape != shape
                or not np.isfinite(value).all()
                or hashlib.sha256(value.tobytes(order='C')).hexdigest()
                != row.get('array_bytes_sha256')
                or reference.shape != value.shape
                or reference.dtype != value.dtype
            ):
                raise ValueError(f'external geometry payload geometry drifted: {target_name}')
            merged[target_path] = value
            expected_files.add(filename)
        if {path.name for path in bundle.iterdir()} != expected_files:
            raise ValueError('unexpected file in external geometry bundle')

        gate_path, gate = target_index['racg_external_geometry/external_blend_gate']
        gate = np.asarray(gate)
        if gate.shape != (5,) or np.count_nonzero(gate.view(np.uint8)):
            raise ValueError('external geometry blend gate is not exact byte-zero')
        merged[gate_path] = gate
        for key, (_, parent_value) in parent_index.items():
            target_path, _ = target_index[key]
            if not np.array_equal(np.asarray(merged[target_path]), np.asarray(parent_value)):
                raise AssertionError(f'EGP merge changed parent leaf: {key}')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class DirectPsmHetmRacgEgpWithGeometryHmcaWeightLoader(WeightLoader):
    """Load every EGP parent leaf and admit only the four GHMA leaves."""

    params_path: str
    expected_parent_arrays: int = 904
    expected_bridge_arrays: int = 4

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_target = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)

        def rendered(flat, source):
            result = {}
            for path, value in flat.items():
                key = '/'.join(map(str, path))
                if key in result:
                    raise ValueError(f'{source} contains an ambiguous path: {key}')
                result[key] = (path, value)
            return result

        target_index = rendered(flat_target, 'Direct848 EGP-GHMA target')
        parent_index = rendered(flat_parent, 'Direct848 EGP parent')
        additions = {
            'racg_external_geometry_hmca/geometry_down/kernel',
            'racg_external_geometry_hmca/geometry_down/bias',
            'racg_external_geometry_hmca/geometry_up/kernel',
            'racg_external_geometry_hmca/geometry_up/bias',
        }
        if len(parent_index) != self.expected_parent_arrays:
            raise ValueError('Direct848 EGP parent leaf count drifted')
        if len(target_index) != self.expected_parent_arrays + self.expected_bridge_arrays:
            raise ValueError('Direct848 EGP-GHMA target leaf count drifted')
        if set(target_index) - set(parent_index) != additions:
            raise ValueError('target differs from EGP parent outside four GHMA leaves')
        if set(parent_index) - set(target_index):
            raise ValueError('EGP-GHMA successor dropped an inherited parent leaf')

        merged = dict(flat_target)
        for key, (_, value) in parent_index.items():
            target_path, reference = target_index[key]
            array = np.asarray(value)
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f'inherited EGP geometry/value drifted: {key}')
            merged[target_path] = array.astype(reference.dtype, copy=False)
        for suffix in ('kernel', 'bias'):
            path = f'racg_external_geometry_hmca/geometry_up/{suffix}'
            value = np.asarray(target_index[path][1])
            if np.count_nonzero(value.view(np.uint8)):
                raise ValueError(f'geometry-HMCA output boundary is not byte-zero: {suffix}')
        for key, (_, parent_value) in parent_index.items():
            target_path, _ = target_index[key]
            if not np.array_equal(np.asarray(merged[target_path]), np.asarray(parent_value)):
                raise AssertionError(f'GHMA merge changed EGP parent leaf: {key}')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class DirectPsmHetmRacgEgpGhmaWithGraphHmcaWeightLoader(WeightLoader):
    """Load every GHMA parent leaf and admit only four graph-HMCA leaves."""

    params_path: str
    expected_parent_arrays: int = 908
    expected_bridge_arrays: int = 4

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_target = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)

        def rendered(flat, source):
            result = {}
            for path, value in flat.items():
                key = '/'.join(map(str, path))
                if key in result:
                    raise ValueError(f'{source} contains an ambiguous path: {key}')
                result[key] = (path, value)
            return result

        target_index = rendered(flat_target, 'Direct848 GHMA-GCHMCA target')
        parent_index = rendered(flat_parent, 'Direct848 GHMA parent')
        additions = {
            'racg_graph_hmca/graph_down/kernel',
            'racg_graph_hmca/graph_down/bias',
            'racg_graph_hmca/graph_up/kernel',
            'racg_graph_hmca/graph_up/bias',
        }
        if len(parent_index) != self.expected_parent_arrays:
            raise ValueError('Direct848 GHMA parent leaf count drifted')
        if len(target_index) != self.expected_parent_arrays + self.expected_bridge_arrays:
            raise ValueError('Direct848 GHMA-GCHMCA target leaf count drifted')
        if set(target_index) - set(parent_index) != additions:
            raise ValueError('target differs from GHMA parent outside four graph-HMCA leaves')
        if set(parent_index) - set(target_index):
            raise ValueError('GHMA-GCHMCA successor dropped an inherited parent leaf')

        merged = dict(flat_target)
        for key, (_, value) in parent_index.items():
            target_path, reference = target_index[key]
            array = np.asarray(value)
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f'inherited GHMA geometry/value drifted: {key}')
            merged[target_path] = array.astype(reference.dtype, copy=False)
        for suffix in ('kernel', 'bias'):
            path = f'racg_graph_hmca/graph_up/{suffix}'
            value = np.asarray(target_index[path][1])
            if np.count_nonzero(value.view(np.uint8)):
                raise ValueError(f'graph-HMCA output boundary is not byte-zero: {suffix}')
        for key, (_, parent_value) in parent_index.items():
            target_path, _ = target_index[key]
            if not np.array_equal(np.asarray(merged[target_path]), np.asarray(parent_value)):
                raise AssertionError(f'GCHMCA merge changed GHMA parent leaf: {key}')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class DirectPsmIntegrated60WeightLoader(WeightLoader):
    """Bootstrap the complete 912-leaf stack directly from Direct848.

    This is the strict joint-training counterpart of the serial
    Direct -> HETM -> RACG -> EGP -> GHMA -> GCHMCA loaders.  It preserves all
    848 learned Direct leaves, installs the sealed HETM and external-geometry
    warm starts, initializes RACG in the HETM semantic basis, and keeps every
    policy-writing boundary byte-zero.  Consequently the initial policy is
    exactly the Direct parent even though all five successor capabilities are
    present and can be optimized together.
    """

    params_path: str
    hetm_bundle_path: str
    external_geometry_bundle_path: str
    expected_direct_arrays: int = 848
    expected_hetm_arrays: int = 22
    expected_racg_arrays: int = 31
    expected_external_arrays: int = 3
    expected_geometry_hmca_arrays: int = 4
    expected_graph_hmca_arrays: int = 4
    expected_external_manifest_sha256: str = (
        'c1b9d8d9c6535b9f89cb70abe818c8468b3fcadde1f17975352223910b2cc562'
    )

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        flat_target = flax.traverse_util.flatten_dict(params)
        expected_total = (
            self.expected_direct_arrays
            + self.expected_hetm_arrays
            + self.expected_racg_arrays
            + self.expected_external_arrays
            + self.expected_geometry_hmca_arrays
            + self.expected_graph_hmca_arrays
        )
        if len(flat_target) != expected_total:
            raise ValueError('Integrated60 target leaf count drifted')

        namespace_counts = {
            namespace: sum(path and path[0] == namespace for path in flat_target)
            for namespace in (
                'hetm',
                'racg',
                'racg_external_geometry',
                'racg_external_geometry_hmca',
                'racg_graph_hmca',
            )
        }
        expected_counts = {
            'hetm': self.expected_hetm_arrays,
            'racg': self.expected_racg_arrays,
            'racg_external_geometry': self.expected_external_arrays,
            'racg_external_geometry_hmca': self.expected_geometry_hmca_arrays,
            'racg_graph_hmca': self.expected_graph_hmca_arrays,
        }
        if namespace_counts != expected_counts:
            raise ValueError(
                'Integrated60 successor namespace drifted: '
                f'{namespace_counts} != {expected_counts}'
            )

        # Reuse the audited Direct+HETM loader on the exact 870-leaf subset.
        hetm_target = flax.traverse_util.unflatten_dict(
            {
                path: value
                for path, value in flat_target.items()
                if not path
                or path[0]
                not in {
                    'racg',
                    'racg_external_geometry',
                    'racg_external_geometry_hmca',
                    'racg_graph_hmca',
                }
            }
        )
        base = DirectPsmWithHetmWarmstartWeightLoader(
            params_path=self.params_path,
            hetm_bundle_path=self.hetm_bundle_path,
            expected_parent_arrays=self.expected_direct_arrays,
            expected_target_hetm_arrays=self.expected_hetm_arrays,
        ).load(hetm_target)
        flat_base = flax.traverse_util.flatten_dict(base)
        if len(flat_base) != self.expected_direct_arrays + self.expected_hetm_arrays:
            raise ValueError('Integrated60 Direct+HETM bootstrap drifted')
        if not set(flat_base).issubset(flat_target):
            raise ValueError('Integrated60 target dropped a Direct+HETM leaf')

        merged = dict(flat_target)
        for path, value in flat_base.items():
            reference = flat_target[path]
            reference_shape, reference_dtype = _reference_shape_dtype(reference)
            value_shape, value_dtype = _reference_shape_dtype(value)
            if (
                value_shape != reference_shape
                or value_dtype != reference_dtype
                or not _finite_if_materialized(value)
            ):
                raise ValueError(f'Integrated60 base geometry drifted: {path}')
            merged[path] = value

        # RACG starts in the already installed HETM basis instead of from an
        # unrelated random coordinate system.
        initialized = _initialize_racg_from_hetm_semantic_transport(
            flax.traverse_util.unflatten_dict(merged)
        )
        merged = dict(flax.traverse_util.flatten_dict(initialized))

        # Install the two sealed Molmo2-ER tensors; the five-way blend gate
        # remains exactly zero so this evidence cannot perturb Direct at step 0.
        bundle = pathlib.Path(self.external_geometry_bundle_path).expanduser().resolve()
        manifest_path = bundle / 'manifest.json'
        if not bundle.is_dir() or bundle.is_symlink():
            raise ValueError('Integrated60 external geometry bundle is invalid')
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError('Integrated60 external geometry manifest is absent')
        if _joint_sha256_file(manifest_path) != self.expected_external_manifest_sha256:
            raise ValueError('Integrated60 external geometry manifest SHA-256 drifted')
        manifest = json.loads(manifest_path.read_text())
        unsigned = dict(manifest)
        claimed = unsigned.pop('manifest_payload_sha256', None)
        if claimed != _joint_canonical_sha256(unsigned):
            raise ValueError('Integrated60 external geometry manifest seal drifted')
        if not (
            manifest.get('schema_version')
            == 'molmo2_er_geometry_warmstart_transplant/v1'
            and manifest.get('state') == 'committed_manifest_last'
            and manifest.get('leaf_count') == 2
            and manifest.get('optimizer_state_included') is False
            and manifest.get('parent_psm_action_vlm_arrays_included') is False
            and manifest.get('production_policy_out_included') is False
        ):
            raise ValueError('Integrated60 external geometry provenance drifted')
        specifications = (
            (
                'persistent_memory/geometry_aux_v3/prefix_out/kernel',
                'leaf_000_prefix_out.npy',
                (2048, 256),
                ('racg_external_geometry', 'external_prefix_out'),
            ),
            (
                'persistent_memory/geometry_aux_v3/role_query/embedding',
                'leaf_001_role_query.npy',
                (5, 2048),
                ('racg_external_geometry', 'external_role_query'),
            ),
        )
        leaves = manifest.get('leaves')
        if not isinstance(leaves, list) or len(leaves) != len(specifications):
            raise ValueError('Integrated60 external geometry leaf manifest drifted')
        expected_files = {'manifest.json'}
        for index, (row, specification) in enumerate(
            zip(leaves, specifications, strict=True)
        ):
            source_path, filename, shape, target_path = specification
            if not (
                row.get('index') == index
                and row.get('path') == source_path
                and row.get('filename') == filename
                and row.get('shape') == list(shape)
                and row.get('dtype') == 'float32'
            ):
                raise ValueError(f'Integrated60 external leaf drifted: {target_path}')
            payload = bundle / filename
            if not (
                payload.resolve().parent == bundle
                and payload.is_file()
                and not payload.is_symlink()
                and _joint_sha256_file(payload) == row.get('file_sha256')
            ):
                raise ValueError(f'Integrated60 external payload drifted: {target_path}')
            value = np.load(payload, allow_pickle=False)
            reference = merged[target_path]
            reference_shape, reference_dtype = _reference_shape_dtype(reference)
            if not (
                value.dtype == np.float32
                and value.shape == shape
                and reference_shape == value.shape
                and reference_dtype == value.dtype
                and np.isfinite(value).all()
                and hashlib.sha256(value.tobytes(order='C')).hexdigest()
                == row.get('array_bytes_sha256')
            ):
                raise ValueError(f'Integrated60 external geometry drifted: {target_path}')
            merged[target_path] = value
            expected_files.add(filename)
        if {path.name for path in bundle.iterdir()} != expected_files:
            raise ValueError('Integrated60 external geometry bundle has extra files')

        zero_boundaries = (
            ('hetm', 'film_out', 'kernel'),
            ('hetm', 'prior_out', 'kernel'),
            ('hetm', 'hmca_condition_out', 'kernel'),
            ('racg', 'graph_action_out', 'kernel'),
            ('racg', 'graph_action_out', 'bias'),
            ('racg_external_geometry', 'external_blend_gate'),
            ('racg_external_geometry_hmca', 'geometry_up', 'kernel'),
            ('racg_external_geometry_hmca', 'geometry_up', 'bias'),
            ('racg_graph_hmca', 'graph_up', 'kernel'),
            ('racg_graph_hmca', 'graph_up', 'bias'),
        )
        for path in zero_boundaries:
            if not _byte_zero_if_materialized(merged[path]):
                raise ValueError(
                    'Integrated60 policy boundary is not byte-zero: '
                    + '/'.join(path)
                )
        for path, value in flat_base.items():
            if not _same_value_or_abstract(merged[path], value):
                raise AssertionError(f'Integrated60 initialization changed base leaf: {path}')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class DirectPsmCollapsedArchitectureWeightLoader(WeightLoader):
    """Bootstrap the complete architecture-only stack from Direct848.

    The target contains the jointly initialized Integrated60 capability stack,
    every PPWM branch, Memory-AdaRMS, PhaseContact and layerwise PSM attention.
    Direct and the two sealed warm starts are installed by the already audited
    Integrated60 loader.  All later branches retain their target initializer,
    but their complete namespaces and every policy-writing boundary are checked
    exactly.  The result is therefore the Direct policy at step zero while all
    architecture branches are available for one best-first 30k training run.

    GroundedRole is deliberately not part of this loader: it has no parameter
    leaves and changes the prefix input itself, so it cannot satisfy the same
    step-zero function-preservation contract.
    """

    params_path: str
    hetm_bundle_path: str
    external_geometry_bundle_path: str
    expected_integrated_arrays: int = 912
    expected_ppwm_arrays: int = 313
    expected_memory_adarms_arrays: int = 4
    expected_phase_contact_arrays: int = 8
    expected_layerwise_memory_arrays: int = 4

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        flat_target = flax.traverse_util.flatten_dict(params)
        expected_total = (
            self.expected_integrated_arrays
            + self.expected_ppwm_arrays
            + self.expected_memory_adarms_arrays
            + self.expected_phase_contact_arrays
            + self.expected_layerwise_memory_arrays
        )
        if len(flat_target) != expected_total:
            raise ValueError('collapsed architecture target leaf count drifted')

        ppwm_prefixes = (
            'latent_future',
            'state_rollout',
            'action_moe',
            'task_progress',
            'predictive_world_model',
        )
        ppwm_paths = {
            path
            for path in flat_target
            if path and str(path[0]).startswith(ppwm_prefixes)
        }
        memory_adarms_paths = {
            path
            for path in flat_target
            if path and str(path[0]).startswith('persistent_memory_adarms')
        }
        phase_contact_paths = {
            path
            for path in flat_target
            if path
            and (
                str(path[0]).startswith('phase_contact_film')
                or str(path[0]).startswith('persistent_action_prior')
            )
        }
        layerwise_memory_paths = {
            path
            for path in flat_target
            if path
            and str(path[0]).startswith(
                'layerwise_persistent_memory_attention'
            )
        }
        observed_counts = (
            len(ppwm_paths),
            len(memory_adarms_paths),
            len(phase_contact_paths),
            len(layerwise_memory_paths),
        )
        expected_counts = (
            self.expected_ppwm_arrays,
            self.expected_memory_adarms_arrays,
            self.expected_phase_contact_arrays,
            self.expected_layerwise_memory_arrays,
        )
        if observed_counts != expected_counts:
            raise ValueError(
                'collapsed architecture successor namespaces drifted: '
                f'{observed_counts} != {expected_counts}'
            )
        successor_paths = (
            ppwm_paths
            | memory_adarms_paths
            | phase_contact_paths
            | layerwise_memory_paths
        )
        if len(successor_paths) != sum(expected_counts):
            raise ValueError('collapsed architecture successor paths overlap')

        integrated_target = flax.traverse_util.unflatten_dict(
            {
                path: value
                for path, value in flat_target.items()
                if path not in successor_paths
            }
        )
        integrated = DirectPsmIntegrated60WeightLoader(
            params_path=self.params_path,
            hetm_bundle_path=self.hetm_bundle_path,
            external_geometry_bundle_path=self.external_geometry_bundle_path,
        ).load(integrated_target)
        flat_integrated = flax.traverse_util.flatten_dict(integrated)
        if (
            len(flat_integrated) != self.expected_integrated_arrays
            or set(flat_integrated) != set(flat_target) - successor_paths
        ):
            raise ValueError('collapsed architecture Integrated60 subset drifted')

        merged = dict(flat_target)
        for path, value in flat_integrated.items():
            reference = flat_target[path]
            reference_shape, reference_dtype = _reference_shape_dtype(reference)
            value_shape, value_dtype = _reference_shape_dtype(value)
            if (
                value_shape != reference_shape
                or value_dtype != reference_dtype
                or not _finite_if_materialized(value)
            ):
                raise ValueError(
                    'collapsed architecture inherited geometry drifted: '
                    + '/'.join(map(str, path))
                )
            merged[path] = value
        for path in successor_paths:
            if not _finite_if_materialized(merged[path]):
                raise ValueError(
                    'collapsed architecture successor is non-finite: '
                    + '/'.join(map(str, path))
                )

        # These are the complete action-facing boundaries of all post-
        # Integrated60 branches.  Internal prediction heads may be randomly
        # initialized, but none can perturb the inherited policy until one of
        # these exact-zero projections opens through training.
        zero_boundaries = (
            ('latent_future_token_out', 'kernel'),
            ('latent_future_token_out', 'bias'),
            ('state_rollout_token_out', 'kernel'),
            ('state_rollout_token_out', 'bias'),
            ('action_moe_token_out', 'kernel'),
            ('action_moe_token_out', 'bias'),
            ('task_progress_token_out', 'kernel'),
            ('task_progress_token_out', 'bias'),
            ('persistent_memory_adarms_out', 'kernel'),
            ('persistent_memory_adarms_out', 'bias'),
            ('phase_contact_film_out', 'kernel'),
            ('phase_contact_film_out', 'bias'),
            ('persistent_action_prior', 'kernel'),
            ('persistent_action_prior', 'bias'),
            ('layerwise_persistent_memory_attention', 'output', 'kernel'),
        )
        for path in zero_boundaries:
            if path not in successor_paths:
                raise ValueError(
                    'collapsed architecture policy boundary is absent: '
                    + '/'.join(path)
                )
            if not _byte_zero_if_materialized(merged[path]):
                raise ValueError(
                    'collapsed architecture policy boundary is not byte-zero: '
                    + '/'.join(path)
                )
        for path, value in flat_integrated.items():
            if not _same_value_or_abstract(merged[path], value):
                raise AssertionError(
                    'collapsed architecture initialization changed inherited leaf: '
                    + '/'.join(map(str, path))
                )
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class Integrated60PpwmWeightLoader(WeightLoader):
    """Load all 912 Integrated60 leaves and initialize only PPWM branches."""

    params_path: str
    expected_parent_arrays: int = 912
    expected_new_arrays: int = 313

    @property
    def strict(self) -> bool:
        return True

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_parent = flax.traverse_util.flatten_dict(parent)
        flat_target = flax.traverse_util.flatten_dict(params)
        if len(flat_parent) != self.expected_parent_arrays:
            raise ValueError('Integrated60 PPWM parent leaf count drifted')
        if len(flat_target) != self.expected_parent_arrays + self.expected_new_arrays:
            raise ValueError('Integrated60 PPWM target leaf count drifted')
        if not set(flat_parent).issubset(flat_target):
            raise ValueError('Integrated60 PPWM target dropped an inherited leaf')

        new_paths = set(flat_target) - set(flat_parent)
        expected_namespaces = {
            'latent_future': 76,
            'state_rollout': 51,
            'action_moe': 85,
            'task_progress': 71,
            'predictive_world_model': 30,
        }
        counts = {
            namespace: sum(
                bool(path) and str(path[0]).startswith(namespace)
                for path in new_paths
            )
            for namespace in expected_namespaces
        }
        if counts != expected_namespaces or sum(counts.values()) != len(new_paths):
            raise ValueError(
                f'Integrated60 PPWM new namespaces drifted: {counts}'
            )

        merged = dict(flat_target)
        for path, value in flat_parent.items():
            reference = flat_target[path]
            array = np.asarray(value)
            if (
                array.shape != reference.shape
                or array.dtype != reference.dtype
                or not np.isfinite(array).all()
            ):
                raise ValueError(f'Integrated60 PPWM parent geometry drifted: {path}')
            merged[path] = array

        # These are the only newly introduced paths that can write into the
        # action-token stream.  Keeping every one byte-zero makes the PPWM
        # successor exactly equal to its Integrated60 parent at step zero;
        # the non-policy prediction heads can still learn immediately.
        zero_boundaries = (
            ('latent_future_token_out', 'kernel'),
            ('latent_future_token_out', 'bias'),
            ('state_rollout_token_out', 'kernel'),
            ('state_rollout_token_out', 'bias'),
            ('action_moe_token_out', 'kernel'),
            ('action_moe_token_out', 'bias'),
            ('task_progress_token_out', 'kernel'),
            ('task_progress_token_out', 'bias'),
        )
        for path in zero_boundaries:
            value = np.asarray(merged[path])
            if np.count_nonzero(value.view(np.uint8)):
                raise ValueError(
                    'Integrated60 PPWM policy boundary is not byte-zero: '
                    + '/'.join(path)
                )
        for path, value in flat_parent.items():
            if not np.array_equal(np.asarray(merged[path]), np.asarray(value)):
                raise AssertionError(
                    f'Integrated60 PPWM initialization changed parent leaf: {path}'
                )
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class StrictSuccessorCheckpointWeightLoader(WeightLoader):
    """Load an exact parent while initializing one audited successor namespace.

    Unlike the historical permissive loader, this rejects missing inherited
    leaves, unexpected extra leaves, shape drift, and a change in the exact
    number of newly initialized leaves.
    """

    params_path: str
    missing_regex: str
    expected_missing_count: int

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        return _merge_strict_successor_params(
            loaded_params,
            params,
            missing_regex=self.missing_regex,
            expected_missing_count=self.expected_missing_count,
        )


@dataclasses.dataclass(frozen=True)
class ExactCheckpointWeightLoader(WeightLoader):
    """Load a parent only when its complete parameter graph is identical."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_reference = flax.traverse_util.flatten_dict(params)
        flat_loaded = flax.traverse_util.flatten_dict(loaded)
        # Orbax may restore list indexes as strings while an NNX abstract tree
        # represents the same indexes as integers.  Comparing the raw tuple
        # keys therefore reports the same rendered path as both missing and
        # extra.  Canonicalize through the already-audited rendered-path index;
        # it rejects aliases before any parameter is accepted.
        reference_index = _index_by_rendered_path(
            flat_reference, source='exact checkpoint target'
        )
        loaded_index = _index_by_rendered_path(
            flat_loaded, source='exact checkpoint source'
        )
        missing = set(reference_index) - set(loaded_index)
        extras = set(loaded_index) - set(reference_index)
        if missing or extras:
            raise ValueError(
                'exact checkpoint parameter paths drifted: '
                f'missing={sorted(missing)}, extras={sorted(extras)}'
            )
        shape_drift = []
        result = {}
        for rendered, (path, reference) in reference_index.items():
            value = loaded_index[rendered][1]
            if tuple(np.shape(value)) != tuple(np.shape(reference)):
                shape_drift.append(
                    (
                        rendered,
                        tuple(np.shape(value)),
                        tuple(np.shape(reference)),
                    )
                )
                continue
            result[path] = (
                value.astype(reference.dtype)
                if value.dtype != reference.dtype
                else value
            )
        if shape_drift:
            raise ValueError(f'exact checkpoint shape drift: {shape_drift}')
        return flax.traverse_util.unflatten_dict(result)


@dataclasses.dataclass(frozen=True)
class DualGeometryResidualCheckpointWeightLoader(WeightLoader):
    """Exact 808-leaf ClausePlan-v3 parent plus sealed warmstart and zero gate."""

    parent_params_path: str
    warmstart_artifact: str
    warmstart_manifest_sha256: str
    warmstart_training_manifest_sha256: str
    repo_root: str
    expected_parent_leaf_count: int = 808

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        warmstart, _ = _geometry_warmstart_loader.load_sealed_warmstart(
            self.warmstart_artifact,
            repo_root=self.repo_root,
            expected_manifest_sha256=self.warmstart_manifest_sha256,
            expected_training_run_manifest_sha256=(
                self.warmstart_training_manifest_sha256
            ),
        )
        initializer = _dual_geometry.init_parameters_from_warmstart(warmstart)
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered checkpoint path')
        merged = _dual_geometry_checkpoint.merge_shape_reference_parameter_maps(
            parent_index,
            {path: row[1] for path, row in ref_index.items()},
            initializer,
            expected_parent_leaf_count=self.expected_parent_leaf_count,
        )
        return flax.traverse_util.unflatten_dict(
            {
                tuple_path: merged[rendered]
                for rendered, (tuple_path, _) in ref_index.items()
            }
        )


@dataclasses.dataclass(frozen=True)
class TemporalRoleMemoryCheckpointWeightLoader(WeightLoader):
    """Load exact 811-leaf DualGeometry parent plus five initialized leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 811

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered temporal-role checkpoint path')
        missing = set(_temporal_role.PARAMETER_PATHS) - set(ref_index)
        if missing:
            raise ValueError(f'temporal-role target leaves are absent: {sorted(missing)}')
        initializer = {
            path: ref_index[path][1] for path in _temporal_role.PARAMETER_PATHS
        }
        merged = _temporal_role_checkpoint.merge_parent_and_overlay(
            parent_index,
            {path: row[1] for path, row in ref_index.items()},
            initializer,
            expected_parent_leaf_count=self.expected_parent_leaf_count,
        )
        return flax.traverse_util.unflatten_dict(
            {
                tuple_path: merged[rendered]
                for rendered, (tuple_path, _) in ref_index.items()
            }
        )


@dataclasses.dataclass(frozen=True)
class CrossViewRoleConsensusCheckpointWeightLoader(WeightLoader):
    """Load exact 816-leaf TemporalRoleMemory parent plus five fresh leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 816

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered cross-view checkpoint path')
        new_paths = set(_cross_view_role.PARAMETER_PATHS)
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('cross-view parent leaf count drifted')
        if set(parent_index) & new_paths:
            raise ValueError('cross-view namespace collides with parent')
        if set(ref_index) != set(parent_index) | new_paths:
            raise ValueError('cross-view target is not exact parent plus five')
        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = reference if rendered in new_paths else parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(f'cross-view checkpoint geometry drifted: {rendered}')
            merged[tuple_path] = array
        gate = np.asarray(merged[ref_index[_cross_view_role.BLEND_GATE_PATH][0]])
        if gate != 0.0 or np.signbit(gate):
            raise ValueError('cross-view gate is not exact positive zero')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class TemporalCrossViewCoAdaptCheckpointWeightLoader(WeightLoader):
    """Load an exact 821-leaf CrossView checkpoint without graph mutation."""

    parent_params_path: str
    expected_leaf_count: int = 821

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered temporal/cross-view checkpoint path')
        if len(parent_index) != self.expected_leaf_count:
            raise ValueError('temporal/cross-view parent leaf count drifted')
        if set(ref_index) != set(parent_index):
            raise ValueError('co-adapt target graph differs from its exact parent')
        required = set(_temporal_role.PARAMETER_PATHS) | set(
            _cross_view_role.PARAMETER_PATHS
        )
        if len(required) != 10 or not required <= set(parent_index):
            raise ValueError('co-adapt parent does not contain both exact branches')
        loaded = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = np.asarray(parent_index[rendered])
            if source.shape != np.asarray(reference).shape or not np.isfinite(source).all():
                raise ValueError(f'co-adapt checkpoint geometry drifted: {rendered}')
            loaded[tuple_path] = source.astype(np.asarray(reference).dtype, copy=False)
        return flax.traverse_util.unflatten_dict(loaded)


@dataclasses.dataclass(frozen=True)
class ContactRiskCalibratedRoleResidualCheckpointWeightLoader(WeightLoader):
    """Load exact 821-leaf CoAdapt parent plus seven fresh contact-risk leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 821

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered contact-risk checkpoint path')
        new_paths = set(_contact_risk.PARAMETER_PATHS)
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('contact-risk parent leaf count drifted')
        if set(parent_index) & new_paths:
            raise ValueError('contact-risk namespace collides with parent')
        if set(ref_index) != set(parent_index) | new_paths:
            raise ValueError('contact-risk target is not exact parent plus seven')
        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = reference if rendered in new_paths else parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(f'contact-risk checkpoint geometry drifted: {rendered}')
            merged[tuple_path] = array
        gate = np.asarray(merged[ref_index[_contact_risk.BLEND_GATE_PATH][0]])
        if gate != 0.0 or np.signbit(gate):
            raise ValueError('contact-risk gate is not exact positive zero')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class RelationalRoleComposerResidualCheckpointWeightLoader(WeightLoader):
    """Load exact 828-leaf ContactRisk parent plus eight fresh composer leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 828

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered relational-role checkpoint path')
        new_paths = set(_relational_role.PARAMETER_PATHS)
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('relational-role parent leaf count drifted')
        if set(parent_index) & new_paths:
            raise ValueError('relational-role namespace collides with parent')
        if set(ref_index) != set(parent_index) | new_paths:
            raise ValueError('relational-role target is not exact parent plus eight')
        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = reference if rendered in new_paths else parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(
                    f'relational-role checkpoint geometry drifted: {rendered}'
                )
            merged[tuple_path] = array
        gate = np.asarray(merged[ref_index[_relational_role.BLEND_GATE_PATH][0]])
        if gate != 0.0 or np.signbit(gate):
            raise ValueError('relational-role gate is not exact positive zero')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class JointRoleGeometryResidualCheckpointWeightLoader(WeightLoader):
    """Load ClausePlan-v3 and initialize all five residual branches jointly.

    This is the direct 808 -> 836 function-preserving path.  Geometry keeps
    its sealed Molmo2-ER warm start; temporal, cross-view, contact-risk, and
    relational-role leaves use the model's audited fresh initialization.  It
    avoids using intermediate trained checkpoints as an implicit selector.
    """

    parent_params_path: str
    warmstart_artifact: str
    warmstart_manifest_sha256: str
    warmstart_training_manifest_sha256: str
    repo_root: str
    expected_parent_leaf_count: int = 808
    expected_target_leaf_count: int = 836

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        warmstart, _ = _geometry_warmstart_loader.load_sealed_warmstart(
            self.warmstart_artifact,
            repo_root=self.repo_root,
            expected_manifest_sha256=self.warmstart_manifest_sha256,
            expected_training_run_manifest_sha256=(
                self.warmstart_training_manifest_sha256
            ),
        )
        geometry_initializer = _dual_geometry.init_parameters_from_warmstart(
            warmstart
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered joint-role checkpoint path')
        branch_paths = (
            set(_dual_geometry.PARAMETER_PATHS)
            | set(_temporal_role.PARAMETER_PATHS)
            | set(_cross_view_role.PARAMETER_PATHS)
            | set(_contact_risk.PARAMETER_PATHS)
            | set(_relational_role.PARAMETER_PATHS)
        )
        if len(branch_paths) != 28:
            raise ValueError('joint-role residual namespaces overlap or drifted')
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('joint-role parent leaf count drifted')
        if len(ref_index) != self.expected_target_leaf_count:
            raise ValueError('joint-role target leaf count drifted')
        if set(parent_index) & branch_paths:
            raise ValueError('joint-role namespace collides with parent')
        if set(ref_index) != set(parent_index) | branch_paths:
            raise ValueError('joint-role target is not exact parent plus 28')

        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            if rendered in geometry_initializer:
                source = geometry_initializer[rendered]
            elif rendered in branch_paths:
                source = reference
            else:
                source = parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(f'joint-role checkpoint geometry drifted: {rendered}')
            merged[tuple_path] = array

        for gate_path in (
            _dual_geometry.BLEND_GATE_PATH,
            _temporal_role.BLEND_GATE_PATH,
            _cross_view_role.BLEND_GATE_PATH,
            _contact_risk.BLEND_GATE_PATH,
            _relational_role.BLEND_GATE_PATH,
        ):
            gate = np.asarray(merged[ref_index[gate_path][0]])
            if np.any(gate != 0.0) or np.any(np.signbit(gate)):
                raise ValueError(f'joint-role gate is not exact positive zero: {gate_path}')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class ClauseRoleBindingVerifierCheckpointWeightLoader(WeightLoader):
    """Load exact 836-leaf JointRoleGeometry plus eight fresh verifier leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 836
    expected_target_leaf_count: int = 844

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        new_paths = set(_clause_role_binding.PARAMETER_PATHS)
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered clause-role checkpoint path')
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('clause-role parent leaf count drifted')
        if len(ref_index) != self.expected_target_leaf_count:
            raise ValueError('clause-role target leaf count drifted')
        if set(parent_index) & new_paths:
            raise ValueError('clause-role namespace collides with parent')
        if set(ref_index) != set(parent_index) | new_paths:
            raise ValueError('clause-role target is not exact parent plus eight')
        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = reference if rendered in new_paths else parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(f'clause-role checkpoint geometry drifted: {rendered}')
            merged[tuple_path] = array
        gate = np.asarray(
            merged[ref_index[_clause_role_binding.BLEND_GATE_PATH][0]]
        )
        if np.any(gate != 0.0) or np.any(np.signbit(gate)):
            raise ValueError('clause-role gate is not exact positive zero')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class SemanticFrontierCompletionVerifierCheckpointWeightLoader(WeightLoader):
    """Load exact 844-leaf ClauseRole parent plus four fresh frontier leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 844
    expected_target_leaf_count: int = 848

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        new_paths = set(_semantic_frontier.PARAMETER_PATHS)
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered semantic-frontier checkpoint path')
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('semantic-frontier parent leaf count drifted')
        if len(ref_index) != self.expected_target_leaf_count:
            raise ValueError('semantic-frontier target leaf count drifted')
        if set(parent_index) & new_paths:
            raise ValueError('semantic-frontier namespace collides with parent')
        if set(ref_index) != set(parent_index) | new_paths:
            raise ValueError('semantic-frontier target is not exact parent plus four')
        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = reference if rendered in new_paths else parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(f'semantic-frontier checkpoint geometry drifted: {rendered}')
            merged[tuple_path] = array
        gate = np.asarray(merged[ref_index[_semantic_frontier.BLEND_GATE_PATH][0]])
        if np.any(gate != 0.0) or np.any(np.signbit(gate)):
            raise ValueError('semantic-frontier gate is not exact positive zero')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class JointClauseRoleSemanticFrontierCheckpointWeightLoader(WeightLoader):
    """Load exact 836-leaf JointRole parent plus twelve fresh joint leaves."""

    parent_params_path: str
    expected_parent_leaf_count: int = 836
    expected_target_leaf_count: int = 848

    def load(self, params: at.Params) -> at.Params:
        parent = _model.restore_params(
            download.maybe_download(self.parent_params_path),
            restore_type=np.ndarray,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_parent = flax.traverse_util.flatten_dict(parent)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        parent_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_parent.items()
        }
        new_paths = set(_clause_role_binding.PARAMETER_PATHS) | set(
            _semantic_frontier.PARAMETER_PATHS
        )
        if len(new_paths) != 12:
            raise ValueError('joint clause-semantic namespace is not exact twelve')
        if len(ref_index) != len(flat_ref) or len(parent_index) != len(flat_parent):
            raise ValueError('ambiguous rendered joint clause-semantic checkpoint path')
        if len(parent_index) != self.expected_parent_leaf_count:
            raise ValueError('joint clause-semantic parent leaf count drifted')
        if len(ref_index) != self.expected_target_leaf_count:
            raise ValueError('joint clause-semantic target leaf count drifted')
        if set(parent_index) & new_paths:
            raise ValueError('joint clause-semantic namespace collides with parent')
        if set(ref_index) != set(parent_index) | new_paths:
            raise ValueError('joint clause-semantic target is not exact parent plus twelve')
        merged = {}
        for rendered, (tuple_path, reference) in ref_index.items():
            source = reference if rendered in new_paths else parent_index[rendered]
            array = np.asarray(source).astype(np.asarray(reference).dtype, copy=False)
            if array.shape != np.asarray(reference).shape or not np.isfinite(array).all():
                raise ValueError(f'joint clause-semantic checkpoint geometry drifted: {rendered}')
            merged[tuple_path] = array
        for gate_path in (
            _clause_role_binding.BLEND_GATE_PATH,
            _semantic_frontier.BLEND_GATE_PATH,
        ):
            gate = np.asarray(merged[ref_index[gate_path][0]])
            if np.any(gate != 0.0) or np.any(np.signbit(gate)):
                raise ValueError(f'joint clause-semantic gate is not exact positive zero: {gate_path}')
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class DirectPsmCumulative848CheckpointWeightLoader(WeightLoader):
    """Initialize the complete 848-leaf graph directly from exact PSM-402.

    The immutable PSM checkpoint owns every inherited leaf.  The sealed SDLA
    auxiliary transplant owns its exact 36 leaves, the sealed Molmo2-ER
    warmstart owns the two external-geometry projections, and the model
    initializer owns every other successor leaf.  Count, path, shape, dtype,
    finiteness, and namespace checks make this a strict 402+446 merge rather
    than a permissive partial checkpoint load.
    """

    primary_params_path: str
    transplant_artifact: str
    transplant_manifest_sha256: str
    warmstart_artifact: str
    warmstart_manifest_sha256: str
    warmstart_training_manifest_sha256: str
    repo_root: str
    expected_parent_leaf_count: int = 402
    expected_target_leaf_count: int = 848
    expected_successor_leaf_count: int = 446

    # The sealed transplant records raw ``nnx.Param`` variables using their
    # checkpoint-state ``/value`` spelling.  Flax's parameter pytree renders
    # those same three variables at the variable path itself.  Keep this
    # representation bridge explicit and closed rather than stripping
    # ``/value`` generically (which could silently alias a real submodule).
    _sdla_raw_param_aliases = {
        f'{_sdla_v3_transplant.SDLA_SHARED_PREFIX}/plan_type/value':
            f'{_sdla_v3_transplant.SDLA_SHARED_PREFIX}/plan_type',
        f'{_sdla_v3_transplant.SDLA_SHARED_PREFIX}/action_type/value':
            f'{_sdla_v3_transplant.SDLA_SHARED_PREFIX}/action_type',
        _sdla_v3_transplant.G_DEMO_PATH:
            _sdla_v3_transplant.G_DEMO_PATH.removesuffix('/value'),
    }

    def load(self, params: at.Params) -> at.Params:
        primary = _model.restore_params(
            download.maybe_download(self.primary_params_path),
            restore_type=np.ndarray,
        )
        transplant, _ = _sdla_v3_transplant.load_transplant_artifact(
            self.transplant_artifact,
            expected_manifest_sha256=self.transplant_manifest_sha256,
        )
        warmstart, _ = _geometry_warmstart_loader.load_sealed_warmstart(
            self.warmstart_artifact,
            repo_root=self.repo_root,
            expected_manifest_sha256=self.warmstart_manifest_sha256,
            expected_training_run_manifest_sha256=(
                self.warmstart_training_manifest_sha256
            ),
        )
        geometry = _dual_geometry.init_parameters_from_warmstart(warmstart)
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_primary = flax.traverse_util.flatten_dict(primary)
        ref_index = {
            '/'.join(map(str, path)): (path, value)
            for path, value in flat_ref.items()
        }
        primary_index = {
            '/'.join(map(str, path)): value
            for path, value in flat_primary.items()
        }
        if len(ref_index) != len(flat_ref) or len(primary_index) != len(flat_primary):
            raise ValueError('ambiguous rendered direct-cumulative checkpoint path')
        if len(primary_index) != self.expected_parent_leaf_count:
            raise ValueError('direct-cumulative PSM parent leaf count drifted')
        if len(ref_index) != self.expected_target_leaf_count:
            raise ValueError('direct-cumulative target leaf count drifted')
        if not set(primary_index).issubset(ref_index):
            raise ValueError('direct-cumulative target dropped a PSM parent leaf')
        successor_paths = set(ref_index) - set(primary_index)
        if len(successor_paths) != self.expected_successor_leaf_count:
            raise ValueError('direct-cumulative successor leaf count drifted')
        if set(transplant) != set(_sdla_v3_transplant.EXPECTED_SDLA_PATHS):
            raise ValueError('direct-cumulative SDLA transplant namespace drifted')
        normalized_transplant = {
            self._sdla_raw_param_aliases.get(path, path): value
            for path, value in transplant.items()
        }
        if len(normalized_transplant) != len(transplant):
            raise ValueError('direct-cumulative SDLA path aliases are ambiguous')
        if not set(normalized_transplant).issubset(successor_paths):
            missing = sorted(set(normalized_transplant) - successor_paths)
            raise ValueError(
                'direct-cumulative SDLA transplant is outside the exact '
                f'successor namespace: {missing}'
            )
        if set(geometry) != set(_dual_geometry.PARAMETER_PATHS):
            raise ValueError('direct-cumulative geometry initializer namespace drifted')
        if not set(geometry).issubset(successor_paths):
            raise ValueError('direct-cumulative geometry initializer collides with parent')

        merged = {}
        for rendered, (tuple_path, reference_value) in ref_index.items():
            reference_shape = tuple(reference_value.shape)
            reference_dtype = np.dtype(reference_value.dtype)
            if rendered in primary_index:
                source = primary_index[rendered]
            elif rendered in normalized_transplant:
                source = normalized_transplant[rendered]
            elif rendered in geometry:
                source = geometry[rendered]
            else:
                source = reference_value
            if isinstance(source, jax.ShapeDtypeStruct):
                source_shape = tuple(source.shape)
                source_finite = True
            else:
                source_array = np.asarray(source)
                source_shape = source_array.shape
                source_finite = bool(np.isfinite(source_array).all())
            if source_shape != reference_shape or not source_finite:
                raise ValueError(
                    f'direct-cumulative checkpoint geometry drifted: {rendered}'
                )
            merged[tuple_path] = (
                source
                if isinstance(source, jax.ShapeDtypeStruct)
                else source_array.astype(reference_dtype, copy=False)
            )

        # Every residual branch that can alter policy actions must be closed at
        # step zero.  The remaining fresh projections may be nonzero because
        # their downstream gate is closed and needs useful first-step gradients.
        closed_gates = (
            self._sdla_raw_param_aliases[_sdla_v3_transplant.G_DEMO_PATH],
            _sdla_v3_transplant.V3_ZERO_PATH,
            _dual_geometry.BLEND_GATE_PATH,
            _temporal_role.BLEND_GATE_PATH,
            _cross_view_role.BLEND_GATE_PATH,
            _contact_risk.BLEND_GATE_PATH,
            _relational_role.BLEND_GATE_PATH,
            _clause_role_binding.BLEND_GATE_PATH,
            _semantic_frontier.BLEND_GATE_PATH,
        )
        for gate_path in closed_gates:
            gate_key = ref_index[gate_path][0]
            if isinstance(merged[gate_key], jax.ShapeDtypeStruct):
                gate_reference = merged[gate_key]
                merged[gate_key] = np.zeros(
                    gate_reference.shape, dtype=gate_reference.dtype
                )
            gate = np.asarray(merged[gate_key])
            if np.any(gate != 0.0) or np.any(np.signbit(gate)):
                raise ValueError(
                    f'direct-cumulative gate is not exact positive zero: {gate_path}'
                )
        return flax.traverse_util.unflatten_dict(merged)


@dataclasses.dataclass(frozen=True)
class SdlaV3ManifestTransplantWeightLoader(WeightLoader):
    """Exact PSM parent + 36-leaf SDLA transplant + native 3-leaf v3 init."""

    primary_params_path: str
    transplant_artifact: str
    transplant_manifest_sha256: str

    def load(self, params: at.Params) -> at.Params:
        primary = _model.restore_params(
            download.maybe_download(self.primary_params_path),
            restore_type=np.ndarray,
        )
        transplant, _ = _sdla_v3_transplant.load_transplant_artifact(
            self.transplant_artifact,
            expected_manifest_sha256=self.transplant_manifest_sha256,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_primary = flax.traverse_util.flatten_dict(primary)

        def index_rendered(flat, *, source):
            indexed = {}
            for tuple_path, value in flat.items():
                rendered = '/'.join(map(str, tuple_path))
                if rendered in indexed and indexed[rendered][0] != tuple_path:
                    raise ValueError(
                        f'{source} has ambiguous rendered path {rendered!r}'
                    )
                indexed[rendered] = (tuple_path, value)
            return indexed

        ref_index = index_rendered(flat_ref, source='target initializer')
        primary_index = index_rendered(flat_primary, source='PSM parent')
        merged = _sdla_v3_transplant.merge_rendered_parameter_maps(
            {path: row[1] for path, row in primary_index.items()},
            {path: row[1] for path, row in ref_index.items()},
            transplant,
        )
        result = {
            tuple_path: merged[rendered]
            for rendered, (tuple_path, _) in ref_index.items()
        }
        return flax.traverse_util.unflatten_dict(result)


@dataclasses.dataclass(frozen=True)
class GeometrySdlaV3ManifestTransplantWeightLoader(WeightLoader):
    """Exact 402-parent + 36 SDLA + 3 native-v3 + 8 geometry init."""

    primary_params_path: str
    transplant_artifact: str
    transplant_manifest_sha256: str

    def load(self, params: at.Params) -> at.Params:
        primary = _model.restore_params(
            download.maybe_download(self.primary_params_path),
            restore_type=np.ndarray,
        )
        transplant, _ = _sdla_v3_transplant.load_transplant_artifact(
            self.transplant_artifact,
            expected_manifest_sha256=self.transplant_manifest_sha256,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_primary = flax.traverse_util.flatten_dict(primary)
        ref_index = {'/'.join(map(str, path)): (path, value) for path, value in flat_ref.items()}
        primary_index = {'/'.join(map(str, path)): value for path, value in flat_primary.items()}
        if len(ref_index) != len(flat_ref) or len(primary_index) != len(flat_primary):
            raise ValueError('ambiguous rendered checkpoint path')
        merged = _geometry_leaf_contract.merge_shape_reference_parameter_maps(
            primary_index,
            {path: row[1] for path, row in ref_index.items()},
            transplant,
        )
        return flax.traverse_util.unflatten_dict(
            {tuple_path: merged[rendered] for rendered, (tuple_path, _) in ref_index.items()}
        )


@dataclasses.dataclass(frozen=True)
class Joint51ManifestTransplantWeightLoader(WeightLoader):
    """Exact 449-leaf Geometry parent initialization plus four native HMCA refs."""

    primary_params_path: str
    transplant_artifact: str
    transplant_manifest_sha256: str

    def load(self, params: at.Params) -> at.Params:
        primary = _model.restore_params(
            download.maybe_download(self.primary_params_path),
            restore_type=np.ndarray,
        )
        transplant, _ = _sdla_v3_transplant.load_transplant_artifact(
            self.transplant_artifact,
            expected_manifest_sha256=self.transplant_manifest_sha256,
        )
        flat_ref = flax.traverse_util.flatten_dict(params)
        flat_primary = flax.traverse_util.flatten_dict(primary)
        ref_index = {'/'.join(map(str, path)): (path, value) for path, value in flat_ref.items()}
        primary_index = {'/'.join(map(str, path)): value for path, value in flat_primary.items()}
        if len(ref_index) != len(flat_ref) or len(primary_index) != len(flat_primary):
            raise ValueError('ambiguous rendered checkpoint path')
        merged = _joint51_leaf_contract.merge_shape_reference_parameter_maps(
            primary_index,
            {path: row[1] for path, row in ref_index.items()},
            transplant,
        )
        return flax.traverse_util.unflatten_dict(
            {tuple_path: merged[rendered] for rendered, (tuple_path, _) in ref_index.items()}
        )


@dataclasses.dataclass(frozen=True)
class CompoundCheckpointWeightLoader(WeightLoader):
    """Build a target tree from a primary checkpoint plus allowlisted secondary leaves.

    The primary checkpoint owns every path that it shares with the target.  The
    secondary checkpoint may only fill paths absent from the primary and matching
    ``secondary_only_regex``.  Newly introduced target modules must match
    ``missing_regex`` and retain their model initializer.  Count checks make the
    declared Stage3/Stage4 PSM parent fail closed if either checkpoint drifts.
    """

    primary_params_path: str
    secondary_params_path: str
    secondary_only_regex: str
    missing_regex: str
    expected_primary_arrays: int | None = None
    expected_secondary_arrays: int | None = None
    expected_secondary_only_arrays: int | None = None

    def load(self, params: at.Params) -> at.Params:
        primary = _model.restore_params(
            download.maybe_download(self.primary_params_path),
            restore_type=np.ndarray,
        )
        secondary = _model.restore_params(
            download.maybe_download(self.secondary_params_path),
            restore_type=np.ndarray,
        )
        return _merge_compound_params(
            primary,
            secondary,
            params,
            secondary_only_regex=self.secondary_only_regex,
            missing_regex=self.missing_regex,
            expected_primary_arrays=self.expected_primary_arrays,
            expected_secondary_arrays=self.expected_secondary_arrays,
            expected_secondary_only_arrays=self.expected_secondary_only_arrays,
        )


@dataclasses.dataclass(frozen=True)
class BestAnchorSpecialistFusionWeightLoader(WeightLoader):
    """Compose the audited best anchor with three trained specialist modules.

    Shared parent leaves always come from the 51.24% contextual anchor.  Each
    specialist checkpoint may contribute only its named namespace; its copy of
    the shared parent must remain value-identical after the target dtype cast.
    This prevents a multi-checkpoint merge from silently becoming a parameter
    soup outside the three explicitly audited adapters and the freshly
    initialized sample-wise router.
    """

    base_params_path: str
    context_params_path: str
    velocity_params_path: str
    language_params_path: str

    def load(self, params: at.Params) -> at.Params:
        paths = {
            'base': self.base_params_path,
            'context_adarms': self.context_params_path,
            'velocity_refiner': self.velocity_params_path,
            'language_subgoal': self.language_params_path,
        }
        restored = {
            name: _model.restore_params(
                download.maybe_download(path), restore_type=np.ndarray
            )
            for name, path in paths.items()
        }
        return _merge_best_anchor_specialist_fusion_params(
            restored['base'],
            restored['context_adarms'],
            restored['velocity_refiner'],
            restored['language_subgoal'],
            params,
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            'gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz',
            gs={'token': 'anon'},
        )
        with path.open('rb') as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {
            'PaliGemma': flax.traverse_util.unflatten_dict(
                flat_params, sep='/'
            )['params']
        }
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex='.*')


def _initialize_racg_from_hetm_multimodal_basis(params: at.Params) -> at.Params:
    """Place RACG visual/language/role inputs in the trained HETM basis."""
    flat = flax.traverse_util.flatten_dict(params)
    source_path = ('hetm', 'prefix_value', 'kernel')
    language_kernel_path = ('racg', 'language_align', 'kernel')
    language_bias_path = ('racg', 'language_align', 'bias')
    patch_kernel_path = ('racg', 'patch_in', 'kernel')
    patch_bias_path = ('racg', 'patch_in', 'bias')
    role_kernel_path = ('racg', 'role_query', 'kernel')
    role_bias_path = ('racg', 'role_query', 'bias')
    required = {
        source_path,
        language_kernel_path,
        language_bias_path,
        patch_kernel_path,
        patch_bias_path,
        role_kernel_path,
        role_bias_path,
    }
    missing = sorted('/'.join(path) for path in required - set(flat))
    if missing:
        raise ValueError(f'RACG multimodal-basis paths are absent: {missing}')
    source = np.asarray(flat[source_path])
    language_kernel = _zeros_from_reference(flat[language_kernel_path])
    language_bias = _zeros_from_reference(flat[language_bias_path])
    patch_kernel = _zeros_from_reference(flat[patch_kernel_path])
    patch_bias = _zeros_from_reference(flat[patch_bias_path])
    role_kernel = _zeros_from_reference(flat[role_kernel_path])
    role_bias = _zeros_from_reference(flat[role_bias_path])
    if source.shape != language_kernel.shape or source.ndim != 2:
        raise ValueError('HETM/RACG multimodal-basis geometry drifted')
    hidden = source.shape[1]
    if not (
        language_bias.shape == (hidden,)
        and patch_kernel.ndim == 2
        and patch_kernel.shape[0] >= source.shape[0]
        and patch_kernel.shape[1] == hidden
        and patch_bias.shape == (hidden,)
        and role_kernel.ndim == 2
        and role_kernel.shape[0] >= hidden
        and role_kernel.shape[1] == hidden
        and role_bias.shape == (hidden,)
        and np.isfinite(source).all()
    ):
        raise ValueError('RACG multimodal/role-query geometry drifted')
    result = dict(flat)
    result[language_kernel_path] = source.astype(language_kernel.dtype, copy=True)
    result[language_bias_path] = np.zeros_like(language_bias)
    visual_bridge = np.zeros_like(patch_kernel)
    visual_bridge[: source.shape[0]] = source.astype(patch_kernel.dtype, copy=False)
    result[patch_kernel_path] = visual_bridge
    result[patch_bias_path] = np.zeros_like(patch_bias)
    identity_bridge = np.zeros_like(role_kernel)
    identity_bridge[:hidden] = np.eye(hidden, dtype=role_kernel.dtype)
    result[role_kernel_path] = identity_bridge
    result[role_bias_path] = np.zeros_like(role_bias)
    return flax.traverse_util.unflatten_dict(result)


def _initialize_racg_from_hetm_semantic_transport(params: at.Params) -> at.Params:
    """Extend the HETM basis through RACG slot and role transports."""
    initialized = _initialize_racg_from_hetm_multimodal_basis(params)
    flat = flax.traverse_util.flatten_dict(initialized)
    modules = ('patch_key_value', 'slot_query', 'slot_update', 'role_key_value')
    required = {
        ('racg', module, leaf)
        for module in modules
        for leaf in ('kernel', 'bias')
    }
    missing = sorted('/'.join(path) for path in required - set(flat))
    if missing:
        raise ValueError(f'RACG semantic-transport paths are absent: {missing}')
    result = dict(flat)
    for module in modules:
        kernel_path = ('racg', module, 'kernel')
        bias_path = ('racg', module, 'bias')
        kernel = _zeros_from_reference(flat[kernel_path])
        bias = _zeros_from_reference(flat[bias_path])
        hidden = bias.shape[0] if module in ('slot_query', 'slot_update') else bias.shape[0] // 2
        expected = {
            'patch_key_value': (hidden, 2 * hidden),
            'slot_query': (hidden, hidden),
            'slot_update': (2 * hidden, hidden),
            'role_key_value': (hidden, 2 * hidden),
        }[module]
        if hidden <= 0 or kernel.shape != expected:
            raise ValueError(f'RACG {module} semantic-transport geometry drifted')
        identity = np.eye(hidden, dtype=kernel.dtype)
        if module in ('patch_key_value', 'role_key_value'):
            mapped = np.concatenate([identity, identity], axis=1)
        elif module == 'slot_query':
            mapped = identity
        else:
            mapped = np.concatenate([identity, identity], axis=0)
        result[kernel_path] = mapped
        result[bias_path] = np.zeros_like(bias)
    return flax.traverse_util.unflatten_dict(result)


def _merge_params(
    loaded_params: at.Params, params: at.Params, *, missing_regex: str
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    # Keep paths as tuples. Later-stage transformer blocks use integer module
    # indexes, which cannot be joined by flax's ``sep='/'`` flattening mode.
    flat_ref = flax.traverse_util.flatten_dict(params)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = (
                v.astype(flat_ref[k].dtype)
                if v.dtype != flat_ref[k].dtype
                else v
            )

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {
        k
        for k in flat_ref
        if pattern.fullmatch('/'.join(map(str, k)))
    }:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result)


def _merge_strict_successor_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    expected_missing_count: int,
) -> at.Params:
    if expected_missing_count < 1:
        raise ValueError('strict successor missing count must be positive')
    flat_ref = flax.traverse_util.flatten_dict(params)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params)

    # Orbax restores sequence indexes as strings, whereas a freshly created
    # NNX graph uses integer list indexes. Compare semantic rendered paths and
    # always emit the target graph's key types.
    def index_by_rendered_path(flat_params, *, source: str):
        indexed = {}
        for path, value in flat_params.items():
            rendered = '/'.join(map(str, path))
            if rendered in indexed and indexed[rendered][0] != path:
                raise ValueError(
                    f'{source} contains ambiguous paths that both render as '
                    f'{rendered!r}: {indexed[rendered][0]!r} and {path!r}'
                )
            indexed[rendered] = (path, value)
        return indexed

    rendered_ref = index_by_rendered_path(
        flat_ref, source='strict successor target'
    )
    rendered_loaded = index_by_rendered_path(
        flat_loaded, source='strict successor parent checkpoint'
    )
    pattern = re.compile(missing_regex)
    allowed_missing = {
        rendered for rendered in rendered_ref if pattern.fullmatch(rendered)
    }
    if len(allowed_missing) != expected_missing_count:
        raise ValueError(
            'strict successor initializer namespace has '
            f'{len(allowed_missing)} leaves, expected {expected_missing_count}'
        )
    actual_missing = set(rendered_ref) - set(rendered_loaded)
    if actual_missing != allowed_missing:
        unexpected = sorted(
            path for path in actual_missing - allowed_missing
        )
        already_present = sorted(
            path for path in allowed_missing - actual_missing
        )
        raise ValueError(
            'strict successor parent mismatch: '
            f'unexpected_missing={unexpected}, successor_present={already_present}'
        )
    extras = set(rendered_loaded) - set(rendered_ref)
    if extras:
        raise ValueError(
            f'strict successor parent has extra leaves: {sorted(extras)}'
        )
    shape_drift = []
    for rendered in set(rendered_loaded) & set(rendered_ref):
        loaded_shape = tuple(np.shape(rendered_loaded[rendered][1]))
        reference_shape = tuple(np.shape(rendered_ref[rendered][1]))
        if loaded_shape != reference_shape:
            shape_drift.append(
                (
                    rendered,
                    loaded_shape,
                    reference_shape,
                )
            )
    if shape_drift:
        raise ValueError(f'strict successor parent shape drift: {shape_drift}')
    merged_flat = {}
    for rendered, (target_path, reference) in rendered_ref.items():
        if rendered in rendered_loaded:
            value = rendered_loaded[rendered][1]
            merged_flat[target_path] = (
                value.astype(reference.dtype)
                if value.dtype != reference.dtype
                else value
            )
        elif rendered in allowed_missing:
            merged_flat[target_path] = reference
    flat_loaded.clear()
    merged = flax.traverse_util.unflatten_dict(merged_flat)
    if set(flax.traverse_util.flatten_dict(merged)) != set(flat_ref):
        raise RuntimeError('strict successor merge did not produce an exact tree')
    return merged


def _merge_compound_params(
    primary_params: at.Params,
    secondary_params: at.Params,
    params: at.Params,
    *,
    secondary_only_regex: str,
    missing_regex: str,
    expected_primary_arrays: int | None = None,
    expected_secondary_arrays: int | None = None,
    expected_secondary_only_arrays: int | None = None,
) -> at.Params:
    """Strictly merge a compound parent into an initialized target tree."""

    flat_ref = flax.traverse_util.flatten_dict(params)
    flat_primary = flax.traverse_util.flatten_dict(primary_params)
    flat_secondary = flax.traverse_util.flatten_dict(secondary_params)

    def index_by_rendered_path(flat_params, *, source: str):
        """Match Orbax string indexes to NNX integer list indexes safely."""
        indexed = {}
        for path, value in flat_params.items():
            rendered = '/'.join(map(str, path))
            if rendered in indexed and indexed[rendered][0] != path:
                raise ValueError(
                    f'{source} contains ambiguous paths that both render as '
                    f'{rendered!r}: {indexed[rendered][0]!r} and {path!r}'
                )
            indexed[rendered] = (path, value)
        return indexed

    # Orbax restores mapping keys as strings, while a freshly constructed NNX
    # Python list uses integer indexes.  Their semantic parameter paths are the
    # same (for example ``blocks/0/cross_k/bias``), so compare a collision-checked
    # rendering while preserving the target tree's original key types.
    rendered_primary = index_by_rendered_path(flat_primary, source='primary checkpoint')
    rendered_secondary = index_by_rendered_path(flat_secondary, source='secondary checkpoint')
    if (
        expected_primary_arrays is not None
        and len(flat_primary) != expected_primary_arrays
    ):
        raise ValueError(
            f'primary checkpoint has {len(flat_primary)} arrays; '
            f'expected {expected_primary_arrays}'
        )
    if (
        expected_secondary_arrays is not None
        and len(flat_secondary) != expected_secondary_arrays
    ):
        raise ValueError(
            f'secondary checkpoint has {len(flat_secondary)} arrays; '
            f'expected {expected_secondary_arrays}'
        )

    primary_only = set(rendered_primary) - set(rendered_secondary)
    if primary_only:
        raise ValueError(
            f'primary-only checkpoint paths are forbidden: {sorted(primary_only)}'
        )

    secondary_only = set(rendered_secondary) - set(rendered_primary)
    secondary_pattern = re.compile(secondary_only_regex)
    unexpected_secondary = sorted(
        path for path in secondary_only if not secondary_pattern.fullmatch(path)
    )
    if unexpected_secondary:
        raise ValueError(
            'secondary-only checkpoint paths escape the allowlist: '
            f'{unexpected_secondary}'
        )
    if (
        expected_secondary_only_arrays is not None
        and len(secondary_only) != expected_secondary_only_arrays
    ):
        raise ValueError(
            f'secondary checkpoint contributes {len(secondary_only)} exclusive '
            f'arrays; expected {expected_secondary_only_arrays}'
        )

    missing_pattern = re.compile(missing_regex)
    result = {}
    for path, reference in flat_ref.items():
        rendered = '/'.join(map(str, path))
        if rendered in rendered_primary:
            value = rendered_primary[rendered][1]
        elif rendered in secondary_only:
            value = rendered_secondary[rendered][1]
        elif missing_pattern.fullmatch(rendered):
            value = reference
        else:
            raise ValueError(
                f'target path {rendered!r} is absent from both parents and does '
                'not match the initialized-module allowlist'
            )
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f'compound parent shape mismatch for {rendered}: '
                f'{value.shape} != {reference.shape}'
            )
        result[path] = (
            value.astype(reference.dtype)
            if value.dtype != reference.dtype
            else value
        )

    return flax.traverse_util.unflatten_dict(result)


def _merge_best_anchor_specialist_fusion_params(
    base_params: at.Params,
    context_params: at.Params,
    velocity_params: at.Params,
    language_params: at.Params,
    params: at.Params,
) -> at.Params:
    """Strictly assemble a 90+4+54+77 specialist graph plus its new router."""

    expected_counts = {
        'base': 90,
        'context_adarms': 94,
        'velocity_refiner': 148,
        'language_subgoal': 225,
        'target': 231,
    }
    expected_owned = {
        'context_adarms': 4,
        'velocity_refiner': 54,
        'language_subgoal': 77,
    }
    expected_initialized = {'specialist_module_router': 6}
    closed_policy_boundaries = {
        'context_adarms_out/kernel',
        'context_adarms_out/bias',
        'velocity_refiner_gain',
        'language_subgoal_token_out/kernel',
        'language_subgoal_token_out/bias',
    }
    flat = {
        'base': flax.traverse_util.flatten_dict(base_params),
        'context_adarms': flax.traverse_util.flatten_dict(context_params),
        'velocity_refiner': flax.traverse_util.flatten_dict(velocity_params),
        'language_subgoal': flax.traverse_util.flatten_dict(language_params),
        'target': flax.traverse_util.flatten_dict(params),
    }
    indexed = {
        name: _index_by_rendered_path(tree, source=f'{name} checkpoint')
        for name, tree in flat.items()
    }
    for name, expected in expected_counts.items():
        if len(indexed[name]) != expected:
            raise ValueError(
                f'specialist-fusion {name} has {len(indexed[name])} leaves; '
                f'expected {expected}'
            )

    base_paths = set(indexed['base'])
    target_paths = set(indexed['target'])
    if not base_paths <= target_paths:
        raise ValueError(
            'specialist-fusion base has target-external leaves: '
            f'{sorted(base_paths - target_paths)}'
        )
    namespace_paths = {}
    for namespace, expected in expected_owned.items():
        source_paths = set(indexed[namespace])
        if not base_paths <= source_paths <= target_paths:
            raise ValueError(
                f'specialist-fusion {namespace} graph is not a valid '
                'base-to-target subset'
            )
        owned = {
            path for path in target_paths if namespace in path
        }
        if len(owned) != expected:
            raise ValueError(
                f'specialist-fusion target namespace {namespace} has '
                f'{len(owned)} leaves; expected {expected}'
            )
        if not owned <= source_paths:
            raise ValueError(
                f'specialist-fusion {namespace} source misses leaves: '
                f'{sorted(owned - source_paths)}'
            )
        namespace_paths[namespace] = owned

    initialized_paths = {
        path
        for path in target_paths
        if any(namespace in path for namespace in expected_initialized)
    }
    if len(initialized_paths) != sum(expected_initialized.values()):
        raise ValueError(
            'specialist-fusion router leaf count drifted: '
            f'{len(initialized_paths)} != {sum(expected_initialized.values())}'
        )
    partition = (
        base_paths
        | set().union(*namespace_paths.values())
        | initialized_paths
    )
    if partition != target_paths:
        raise ValueError(
            'specialist-fusion target is not exactly base plus three '
            f'namespaces: missing={sorted(target_paths - partition)}, '
            f'overlap_or_extra={sorted(partition - target_paths)}'
        )
    if sum(map(len, namespace_paths.values())) != len(
        set().union(*namespace_paths.values())
    ):
        raise ValueError('specialist-fusion namespaces overlap')
    if not closed_policy_boundaries <= target_paths:
        raise ValueError(
            'specialist-fusion target misses policy boundaries: '
            f'{sorted(closed_policy_boundaries - target_paths)}'
        )

    # Every specialist must carry the exact frozen contextual anchor.  Compare
    # in the specialist leaf's stored dtype: mixed-precision successor saves
    # quantize 47 frozen float32 parent arrays to bfloat16, so re-expanding the
    # saved value to float32 cannot reconstruct discarded mantissa bits.  The
    # final merge still takes shared leaves from the original float32 base.
    for namespace in expected_owned:
        for rendered in base_paths:
            source_value = np.asarray(indexed[namespace][rendered][1])
            base_value = np.asarray(indexed['base'][rendered][1]).astype(
                source_value.dtype
            )
            if not np.array_equal(base_value, source_value):
                raise ValueError(
                    f'specialist-fusion {namespace} changed shared parent '
                    f'leaf {rendered}'
                )

    result = {}
    for rendered, (target_path, reference) in indexed['target'].items():
        if rendered in initialized_paths:
            result[target_path] = reference
            continue
        source = 'base'
        for namespace, owned in namespace_paths.items():
            if rendered in owned:
                source = namespace
                break
        value = np.asarray(indexed[source][rendered][1])
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f'specialist-fusion shape mismatch for {rendered}: '
                f'{value.shape} != {reference.shape}'
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f'specialist-fusion non-finite source leaf: {rendered}'
            )
        converted = (
            value.astype(reference.dtype)
            if value.dtype != reference.dtype
            else value
        )
        # Preserve the 51.24% action function at initialization while keeping
        # all internal specialist representations. These five sole outer
        # boundaries receive gradients immediately and reopen during training.
        result[target_path] = (
            np.zeros_like(converted)
            if rendered in closed_policy_boundaries
            else converted
        )
    return flax.traverse_util.unflatten_dict(result)
