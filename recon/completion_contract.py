"""A-04: the file contract between the shape-completion stage and its out-of-process runner.

LaS-Comp (D-057) needs Python 3.10, torch 2.4 and CUDA, so it can't run inside the service's
Python >=3.11 interpreter, and on the M2 Max it can't run at all. The stage therefore talks to a
separate *runner* process through files in one folder (D-058):

    completion/
      request.json             written by the service: what to run
      input/partial_zup.ply    the observed part surface, as points, z up
      input/view.png           one capture photo, cropped to the part
      runner/result.json       written by the runner, rewritten after every sample
      runner/sample_<k>_mesh_zup.ply   one completed mesh per sample, z up

Everything in the folder is relative to `request.json`, so the folder can be copied to a CUDA host
(A-12's Colab check), run there, and copied back to be ingested.

This module is the *only* code the runner shares with the service, so it imports nothing but the
standard library and numpy, and stays Python 3.10 syntax: A-12's runner imports it by putting the
repository root on `sys.path` (`pip install -e .` refuses Python 3.10 because of
`requires-python`). `tests/test_completion_contract.py` enforces both.

The result file carries no paths on purpose. The service derives every file name from the sample
index, so nothing a runner writes can make the service read or overwrite a file outside
`runner/`. A runner may keep its own files in `runner/` too (A-12's `lascomp_metrics.json` and
conditioning image); the service never reads anything there but `result.json` and the meshes.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

REQUEST_FILE = "request.json"
PARTIAL_POINTS_PATH = "input/partial_zup.ply"
VIEW_IMAGE_PATH = "input/view.png"
RUNNER_DIR = "runner"
RESULT_FILE = "result.json"

# D-057's pins. A runner reports what it actually ran, and the service rejects a result whose pins
# differ: a sample from an unpinned model is not the sample A-10 measured.
MODEL_NAME = "LaS-Comp"
MODEL_COMMIT = "0da46f2569b484986d3737d97213428db962161f"
WEIGHTS_ID = "microsoft/TRELLIS-image-large"
WEIGHTS_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"

# The GPU A-12 measured A-04's budget_s and overrun_allowance_s on (recon/completion.py). A runner
# on any other GPU says so in its log, because the budget may fit fewer than K samples there. The
# service only ever learns that the budget ran short; it can't tell why (review on #259).
BUDGET_MEASURED_GPU = "Tesla T4"

# The board frame has +z pointing *into* the table (recon/board_pose.py), and LaS-Comp expects z
# up. A 180-degree rotation about x, diag(1, -1, -1), turns one into the other. It has determinant
# +1, so it is a rotation, not a mirror -- diag(1, 1, -1) would also put z up but would reflect the
# part, a trap I-01 already fell into once (inspect/inspection/synthetic.py's board_frame_transform
# explains it). The rotation is its own inverse, so the same matrix maps samples back.
BOARD_TO_Z_UP = np.diag([1.0, -1.0, -1.0])
FRAME_DESCRIPTION = (
    "board frame rotated 180 degrees about x (diag(1,-1,-1)): millimetres, x along the board's "
    "8-square side, z up out of the board"
)

RESULT_STATUSES = ("running", "complete", "budget_reached", "failed")

_REQUEST_KEYS = {
    "schema_version",
    "partial_points",
    "image",
    "output_dir",
    "seeds",
    "budget_s",
    "units",
    "frame",
    "expected_model",
    "model_options",
}
_MODEL_KEYS = {"name", "commit", "weights", "weights_revision"}
_MODEL_OPTION_KEYS = {"sparse_structure_steps", "slat_steps"}
_RESULT_KEYS = {"schema_version", "status", "model", "load_s", "samples", "error"}
_SAMPLE_KEYS = {"k", "seed", "time_s"}
# Seeds go to torch generators on the runner side; keep them in the range every generator accepts.
_MAX_SEED = 2**31 - 1

_MESH_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {vertex_count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "element face {face_count}\n"
    "property list uchar int vertex_indices\n"
    "end_header\n"
)
_FACE_DTYPE = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
_VERTEX_DTYPE = np.dtype("<f4")


class ContractError(ValueError):
    """A request, result or mesh file that doesn't match this contract exactly."""


def sample_mesh_name(k: int) -> str:
    """The runner's mesh file for sample k, inside `runner/`."""
    return f"sample_{int(k)}_mesh_zup.ply"


def board_to_z_up(points_mm: np.ndarray) -> np.ndarray:
    """Board-frame millimetres -> the runner's z-up frame (still millimetres)."""
    return np.asarray(points_mm, dtype=np.float64) @ BOARD_TO_Z_UP.T


def z_up_to_board(points_mm: np.ndarray) -> np.ndarray:
    """The runner's z-up frame -> board-frame millimetres. The rotation is its own inverse."""
    return np.asarray(points_mm, dtype=np.float64) @ BOARD_TO_Z_UP.T


@dataclass(frozen=True)
class ExpectedModel:
    name: str = MODEL_NAME
    commit: str = MODEL_COMMIT
    weights: str = WEIGHTS_ID
    weights_revision: str = WEIGHTS_REVISION

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "commit": self.commit,
            "weights": self.weights,
            "weights_revision": self.weights_revision,
        }


@dataclass(frozen=True)
class CompletionRequest:
    """What the service asks the runner to do. The file paths are fixed, not chosen per request."""

    seeds: tuple[int, ...]
    budget_s: float
    sparse_structure_steps: int | None = None
    slat_steps: int | None = None
    expected_model: ExpectedModel = ExpectedModel()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "partial_points": PARTIAL_POINTS_PATH,
            "image": VIEW_IMAGE_PATH,
            "output_dir": RUNNER_DIR,
            "seeds": [int(seed) for seed in self.seeds],
            "budget_s": float(self.budget_s),
            "units": "mm",
            "frame": FRAME_DESCRIPTION,
            "expected_model": self.expected_model.as_dict(),
            # None means "LaS-Comp's own default" (A-10 ran 100 sparse-structure and 25 SLAT steps).
            "model_options": {
                "sparse_structure_steps": self.sparse_structure_steps,
                "slat_steps": self.slat_steps,
            },
        }


@dataclass(frozen=True)
class SampleRecord:
    k: int
    seed: int
    time_s: float


@dataclass(frozen=True)
class RunnerResult:
    status: str
    model: dict[str, str]
    load_s: float | None
    samples: tuple[SampleRecord, ...]
    error: str | None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to a temporary file beside `path`, then rename it into place.

    A runner rewrites `result.json` after every sample and may be killed at any moment (the
    service's hard timeout), so a reader must only ever see a complete file: the old one or the
    new one, never half of either.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_request(completion_dir: Path, request: CompletionRequest) -> Path:
    _check_seeds(request.seeds)
    _check_non_negative_number(request.budget_s, "budget_s")
    path = Path(completion_dir) / REQUEST_FILE
    write_json_atomic(path, request.as_dict())
    return path


def read_request(path: Path) -> CompletionRequest:
    """Read and check `request.json` exactly. Used by the runner, and by the service on ingest."""
    data = _load_json_object(path)
    _check_keys(data, _REQUEST_KEYS, "request")
    _check_schema_version(data["schema_version"], "request")
    fixed = {
        "partial_points": PARTIAL_POINTS_PATH,
        "image": VIEW_IMAGE_PATH,
        "output_dir": RUNNER_DIR,
        "units": "mm",
        "frame": FRAME_DESCRIPTION,
    }
    for key, expected in fixed.items():
        if data[key] != expected:
            raise ContractError(
                f"request {key!r} is {data[key]!r}, expected {expected!r}"
            )
    seeds = data["seeds"]
    if not isinstance(seeds, list):
        raise ContractError("request 'seeds' must be a list")
    _check_seeds(seeds)
    budget_s = _check_non_negative_number(data["budget_s"], "request 'budget_s'")
    model = data["expected_model"]
    if not isinstance(model, dict):
        raise ContractError("request 'expected_model' must be an object")
    _check_keys(model, _MODEL_KEYS, "request 'expected_model'")
    for key in _MODEL_KEYS:
        if not isinstance(model[key], str) or not model[key]:
            raise ContractError(
                f"request 'expected_model.{key}' must be a non-empty string"
            )
    options = data["model_options"]
    if not isinstance(options, dict):
        raise ContractError("request 'model_options' must be an object")
    _check_keys(options, _MODEL_OPTION_KEYS, "request 'model_options'")
    steps: dict[str, int | None] = {}
    for key in sorted(_MODEL_OPTION_KEYS):
        value = options[key]
        if value is not None and (not _is_int(value) or value < 1):
            raise ContractError(
                f"request 'model_options.{key}' must be null or an int >= 1"
            )
        steps[key] = value
    return CompletionRequest(
        seeds=tuple(int(seed) for seed in seeds),
        budget_s=budget_s,
        sparse_structure_steps=steps["sparse_structure_steps"],
        slat_steps=steps["slat_steps"],
        expected_model=ExpectedModel(**model),
    )


def result_payload(result: RunnerResult) -> dict[str, Any]:
    """The JSON a runner writes for `result`. Shared so the stub and the real runner can't drift."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": result.status,
        "model": dict(result.model),
        "load_s": result.load_s,
        "samples": [
            {"k": sample.k, "seed": sample.seed, "time_s": sample.time_s}
            for sample in result.samples
        ],
        "error": result.error,
    }


def parse_result(path: Path, request: CompletionRequest) -> RunnerResult:
    """Read and check `runner/result.json` against the request it answers. Raises ContractError.

    Strict on purpose, in the style of A-01's `vlm_check.parse_response`: a result that is
    malformed in any way is rejected whole, rather than half-trusted. Samples must be numbered
    0, 1, 2, ... with exactly the request's seeds in order, so the service can name every file
    itself and say which seed produced which surface.
    """
    data = _load_json_object(path)
    _check_keys(data, _RESULT_KEYS, "result")
    _check_schema_version(data["schema_version"], "result")

    status = data["status"]
    if status not in RESULT_STATUSES:
        raise ContractError(
            f"result 'status' is {status!r}, expected one of {RESULT_STATUSES}"
        )

    model = data["model"]
    if not isinstance(model, dict):
        raise ContractError("result 'model' must be an object")
    _check_keys(model, _MODEL_KEYS, "result 'model'")
    expected_model = request.expected_model.as_dict()
    if model != expected_model:
        raise ContractError(
            f"result model pins {model!r} differ from the requested pins {expected_model!r}"
        )

    load_s = data["load_s"]
    if load_s is not None:
        load_s = _check_non_negative_number(load_s, "result 'load_s'")

    error = data["error"]
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise ContractError("result 'error' must be null or a non-empty string")
    if (status == "failed") != (error is not None):
        raise ContractError(
            "result 'error' must be set exactly when 'status' is 'failed'"
        )

    raw_samples = data["samples"]
    if not isinstance(raw_samples, list):
        raise ContractError("result 'samples' must be a list")
    if len(raw_samples) > len(request.seeds):
        raise ContractError(
            f"result lists {len(raw_samples)} samples, more than the {len(request.seeds)} requested"
        )
    if raw_samples and load_s is None:
        raise ContractError("result lists samples but no 'load_s'")
    samples: list[SampleRecord] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ContractError(f"result sample {index} must be an object")
        _check_keys(raw, _SAMPLE_KEYS, f"result sample {index}")
        if not _is_int(raw["k"]) or raw["k"] != index:
            raise ContractError(
                f"result sample {index} has k={raw['k']!r}, expected {index}"
            )
        if not _is_int(raw["seed"]) or raw["seed"] != request.seeds[index]:
            raise ContractError(
                f"result sample {index} has seed {raw['seed']!r}, "
                f"the request's seed {index} is {request.seeds[index]}"
            )
        time_s = _check_non_negative_number(
            raw["time_s"], f"result sample {index} 'time_s'"
        )
        samples.append(SampleRecord(k=index, seed=raw["seed"], time_s=time_s))

    return RunnerResult(
        status=status,
        model=model,
        load_s=load_s,
        samples=tuple(samples),
        error=error,
    )


def write_mesh_ply(path: Path, vertices_mm: np.ndarray, faces: np.ndarray) -> None:
    """Write the contract's mesh layout: binary little-endian float32 xyz and int32 triangles.

    Byte for byte the layout of `recon.tsdf_fusion.write_mesh_ply` (a test pins that), duplicated
    rather than imported because the runner can't import `recon.tsdf_fusion`'s dependencies.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=_VERTEX_DTYPE)
    faces = np.asarray(faces, dtype="<i4")
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise ValueError(f"vertices_mm must have shape (N,3), got {vertices_mm.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3) (triangles), got {faces.shape}")
    header = _MESH_HEADER_TEMPLATE.format(
        vertex_count=vertices_mm.shape[0], face_count=faces.shape[0]
    )
    face_records = np.empty(faces.shape[0], dtype=_FACE_DTYPE)
    face_records["count"] = 3
    face_records["indices"] = faces
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices_mm.tobytes())
        handle.write(face_records.tobytes())


def read_mesh_ply_strict(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a mesh in exactly `write_mesh_ply`'s layout, or raise ContractError.

    Stricter than `recon.tsdf_fusion.read_mesh_ply`, which trusts its own files and checks neither
    the format line nor the property list. A runner's file is foreign: a PLY exported with normals
    or colours would be misread there as garbage vertices without any error, so this one compares
    the whole header, the exact payload size, every value's finiteness and every face index.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read mesh {path.name}: {exc}") from exc
    header, separator, payload = raw.partition(b"end_header\n")
    if not separator:
        raise ContractError(f"{path.name} has no 'end_header' line")
    header_text = (header + separator).decode("ascii", errors="replace")
    vertex_count = _header_count(header_text, "vertex", path)
    face_count = _header_count(header_text, "face", path)
    expected_header = _MESH_HEADER_TEMPLATE.format(
        vertex_count=vertex_count, face_count=face_count
    )
    if header_text != expected_header:
        raise ContractError(
            f"{path.name} header isn't the contract's layout (binary little-endian float xyz, "
            "uchar/int triangle list, nothing else)"
        )
    if vertex_count < 3 or face_count < 1:
        raise ContractError(
            f"{path.name} has {vertex_count} vertices and {face_count} faces"
        )
    vertex_bytes = vertex_count * 3 * _VERTEX_DTYPE.itemsize
    expected_bytes = vertex_bytes + face_count * _FACE_DTYPE.itemsize
    if len(payload) != expected_bytes:
        raise ContractError(
            f"{path.name} payload is {len(payload)} bytes, expected {expected_bytes}"
        )
    vertices = np.frombuffer(payload[:vertex_bytes], dtype=_VERTEX_DTYPE).reshape(-1, 3)
    face_records = np.frombuffer(payload[vertex_bytes:], dtype=_FACE_DTYPE)
    if not np.all(np.isfinite(vertices)):
        raise ContractError(f"{path.name} has non-finite vertex coordinates")
    if not np.all(face_records["count"] == 3):
        raise ContractError(f"{path.name} has a face that isn't a triangle")
    faces = face_records["indices"].astype(np.int64)
    if faces.min() < 0 or faces.max() >= vertex_count:
        raise ContractError(
            f"{path.name} has a face index outside its {vertex_count} vertices"
        )
    return vertices.astype(np.float64), faces


_POINTS_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {vertex_count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)
_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def read_points_ply(path: Path) -> np.ndarray:
    """Read the partial cloud, `input/partial_zup.ply`, as (N,3) float64 millimetres, z up.

    The service writes it with `recon.vggt_runner.write_ply` (float32 xyz + uchar rgb); this reads
    exactly that layout so the runner needn't import `recon.vggt_runner` and its dependencies.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read points {path.name}: {exc}") from exc
    header, separator, payload = raw.partition(b"end_header\n")
    if not separator:
        raise ContractError(f"{path.name} has no 'end_header' line")
    header_text = (header + separator).decode("ascii", errors="replace")
    vertex_count = _header_count(header_text, "vertex", path)
    if header_text != _POINTS_HEADER_TEMPLATE.format(vertex_count=vertex_count):
        raise ContractError(f"{path.name} header isn't the contract's points layout")
    if vertex_count < 1 or len(payload) != vertex_count * _POINT_DTYPE.itemsize:
        raise ContractError(f"{path.name} payload doesn't hold {vertex_count} points")
    records = np.frombuffer(payload, dtype=_POINT_DTYPE)
    points = np.column_stack([records["x"], records["y"], records["z"]]).astype(
        np.float64
    )
    if not np.all(np.isfinite(points)):
        raise ContractError(f"{path.name} has non-finite coordinates")
    return points


def _header_count(header_text: str, element: str, path: Path) -> int:
    prefix = f"element {element} "
    for line in header_text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :]
            if not value.isdigit():
                raise ContractError(
                    f"{path.name} has a malformed '{prefix.strip()}' line"
                )
            return int(value)
    raise ContractError(f"{path.name} header has no '{prefix.strip()}' line")


def _load_json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"{path.name} does not exist") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path.name} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"{path.name} must hold a JSON object")
    return data


def _check_keys(data: dict[str, Any], expected: set[str], what: str) -> None:
    keys = set(data)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ContractError(
            f"{what} keys differ from the contract: missing {missing}, extra {extra}"
        )


def _check_schema_version(value: Any, what: str) -> None:
    if not _is_int(value) or value != SCHEMA_VERSION:
        raise ContractError(
            f"{what} schema_version is {value!r}, expected {SCHEMA_VERSION}"
        )


def _check_seeds(seeds: Any) -> None:
    if len(seeds) == 0:
        raise ContractError("a request needs at least one seed")
    for seed in seeds:
        if not _is_int(seed) or not 0 <= seed <= _MAX_SEED:
            raise ContractError(f"seed {seed!r} is not an int in [0, {_MAX_SEED}]")
    if len(set(seeds)) != len(seeds):
        raise ContractError(
            "seeds must be distinct: two equal seeds give the same sample"
        )


def _check_non_negative_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ContractError(f"{what} must be finite and >= 0, got {value!r}")
    return value


def _is_int(value: Any) -> bool:
    # bool is an int subclass in Python; a JSON `true` is not a sample index or a seed.
    return isinstance(value, int) and not isinstance(value, bool)
