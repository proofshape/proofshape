"""A-04: the shape-completion stage, service side (proposal §6.6, D-005, D-006, D-057, D-058).

Reconstruction recovers only the surface the cameras saw. A generative model (LaS-Comp, D-057)
samples K plausible completions of the rest, and where those samples disagree is where the system
is guessing -- the disagreement map A-05 builds from them. That map steers capture guidance and
nothing else: completed surface is **Inferred**, never measured, so it never touches the verdict
mesh (D-005, D-006). This module therefore only ever *reads* upstream artifacts and writes inside
its own `completion/` folder; a test hashes every other file before and after to hold it to that.

What it does, after R-10, inside `run_reconstruction_pipeline`:

1. **Isolates the part** from the served mesh's Observed surface. The cleaned mesh still carries a
   thin skin of board 3-5 mm above the paper, connected to the part (A-10), so the partial cloud
   would otherwise be mostly board -- exactly the hallucination the proposal's "segmentation before
   completion is mandatory" is there to prevent.
2. **Samples a partial point cloud** from the kept faces and turns it z-up for the model.
3. **Picks one capture photo** (the one that saw the most kept surface) and crops it to the part.
4. **Writes a request** in `recon.completion_contract`'s format and, if a runner is configured,
   runs it as a separate process under a time budget and a hard timeout (D-057: Python 3.10,
   torch 2.4, CUDA).
5. **Ingests** the runner's meshes: back to the board frame, sampled to points, coloured Inferred,
   written as `sample_<k>.ply`, with everything recorded in `completion.json`.

**Guidance can't break a reconstruction** (A-04's third criterion). `run_completion_stage` never
raises. With no runner configured -- every session on the M2 Max today, since LaS-Comp is
CUDA-only -- it still prepares the inputs and records `unavailable`, with no warning: capture shows
warnings to the supplier, and a stage that isn't installed is nothing they can act on (D-058). A
configured runner that fails adds one fixed warning string. The runner's own output only ever goes
to a capped log file, never into a warning or `session.json`, because the service's warnings reach
the supplier and the runner's output is not something we control.

Every threshold is a declared parameter with a stated default (`CompletionParameters`). K, the
budget and the overrun allowance are **provisional**: they come from A-10's single T4 run, and
A-12 replaces them with measured values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from recon import completion_contract as contract
from recon.board_pose import read_exif_fields
from recon.mesh_cleanup import read_glb
from recon.metric_alignment import project_points_to_camera
from recon.part_segmentation import DEFAULT_FOOTPRINT_MM, BoardSurface
from recon.provenance import (
    INFERRED_COLOR,
    LABEL_INFERRED,
    LABEL_OBSERVED,
    OBSERVED_COLOR,
    colors_to_labels,
    read_observations_npz,
)
from recon.vggt_runner import write_ply

COMPLETION_ENV = "PROOFSHAPE_COMPLETION"
RUNNER_ENV = "PROOFSHAPE_COMPLETION_RUNNER"

COMPLETION_REPORT = "completion.json"
RUNNER_LOG_TAIL = "runner_log_tail.txt"
# Mirrors recon.service's CALIBRATED_BOARD_DIR_NAME / BOARD_CALIBRATION_RECORD for the CLI, which
# rebuilds the pipeline's folder layout without importing the FastAPI service. A test pins them equal.
BOARD_POSE_DIR_NAME = "board_pose"
CALIBRATED_BOARD_DIR_NAME = "board_pose_calibrated"
BOARD_CALIBRATION_RECORD = "board_calibration.json"

# The two fixed warning strings, in the contract's `code: message` warning style. Fixed on
# purpose: whatever went wrong is recorded in completion.json, and nothing a runner prints can
# reach the supplier through them.
WARNING_FAILED = (
    "completion_failed: shape completion produced no samples for this session. The model and its "
    "observed and unobserved colours are unaffected."
)
WARNING_PARTIAL = (
    "completion_partial: shape completion stopped early and produced fewer samples than requested. "
    "The model and its observed and unobserved colours are unaffected."
)

# Secrets the service process may hold that a model runner has no business seeing (A-01's Gemini
# key; R-15's endpoint token). Stripped from the runner's environment.
_SECRET_ENV_NAMES = ("GEMINI_API_KEY", "PROOFSHAPE_ENDPOINT_TOKEN")

# D-057: the model can't share a 15 GB T4 with another sample, so runs are one at a time. A second
# session that arrives while one is running is not queued behind a ~40-minute run inside the
# reconstruction pipeline; it skips completion and says so.
_RUNNER_LOCK = threading.Lock()

# EXIF orientation -> the PIL transpose that shows the photo upright, the same table PIL's own
# ImageOps.exif_transpose uses. Applied explicitly to the *crop*, because the crop box is computed
# in stored sensor pixels -- the frame every pose in this pipeline uses (recon/board_pose.py) -- and
# exif_transpose on a crop would have lost the tag.
_ORIENTATION_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


class CompletionInputError(ValueError):
    """The stage's inputs can't be prepared from this session's artifacts."""


@dataclass(frozen=True)
class CompletionParameters:
    """Every number the stage uses. None of them is measured accuracy; each is an assumption.

    Part isolation (none measured; chosen from A-10's s-04 inspection and the golden set's cuts):
    - skin_allowance_mm 3.5: the board skin left in the cleaned mesh reaches about 6.4 mm on s-04,
      whose fusion cut is 2.5 mm, so the "core" of the part starts 3.5 mm above that cut. A fixed
      core height doesn't work: fusion's cut ranges about 2.5-6 mm across the golden set, and a
      fixed 6 mm core is all skin on s-02.
    - cluster_radius_mm 4.0: A-10's DBSCAN eps; bridges a one-voxel gap, not separate objects.
    - min_core_vertices 50: fewer can't describe a hand-sized part; below it the stage says so.
    - edge_guard_mm 4.0: the skin spans the whole board pattern, so a core reaching its edge means
      skin got into the core, not that the part is that big.
    - min_wall_angle_deg 60: below the core height a vertex is kept only inside a cone under the
      core, so walls steeper than this survive (s-04's sides are 66.6 deg) and the flat skin ring
      round the part doesn't.
    - wall_xy_tolerance_mm 2.0: marching-cubes vertices sit up to a voxel (2 mm, R-07) off the
      true surface.

    Sampling: partial_point_count and sample_point_count 20,000 are A-10's count.

    The photo: view_margin_fraction 0.2 per side, because observed points under-cover the
    silhouette and TRELLIS re-crops to its own foreground mask anyway; view_max_side_px 1024,
    because TRELLIS conditions on 518 px images.

    The runner, **provisional until A-12 measures them**: max_samples 8 is the proposal's final-pass
    K (§6.6); budget_s 2460 fits 8 samples at A-10's ~298 s each plus its 48 s model load on a T4;
    overrun_allowance_s 360 is about one more sample plus 20%, the margin before the hard kill.
    sample_scale_factor_limit 4.0: a completion whose size differs from the observed part's by more
    than this factor is in the wrong units (LaS-Comp normalises to a unit cube), not a completion.
    """

    skin_allowance_mm: float = 3.5
    cluster_radius_mm: float = 4.0
    min_core_vertices: int = 50
    edge_guard_mm: float = 4.0
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM
    min_wall_angle_deg: float = 60.0
    wall_xy_tolerance_mm: float = 2.0
    partial_point_count: int = 20_000
    partial_seed: int = 0
    sample_point_count: int = 20_000
    view_margin_fraction: float = 0.2
    view_max_side_px: int = 1024
    max_samples: int = 8
    base_seed: int = 0
    budget_s: float = 2460.0
    overrun_allowance_s: float = 360.0
    sparse_structure_steps: int | None = None
    slat_steps: int | None = None
    sample_scale_factor_limit: float = 4.0
    log_tail_bytes: int = 64 * 1024

    def as_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["footprint_mm"] = list(self.footprint_mm)
        return report


DEFAULT_PARAMETERS = CompletionParameters()
PROVISIONAL_PARAMETERS = ("max_samples", "budget_s", "overrun_allowance_s")


@dataclass(frozen=True)
class PipelineArtifacts:
    """The upstream folders the stage reads. Built by the pipeline, or by the CLI from a folder."""

    capture_dir: Path
    board_pose_dir: Path
    reconstruction_dir: Path
    fusion_dir: Path
    provenance_dir: Path
    uncertainty_dir: Path

    @classmethod
    def from_artifact_dir(
        cls, artifact_dir: Path, capture_dir: Path | None = None
    ) -> PipelineArtifacts:
        artifact_dir = Path(artifact_dir)
        # The pipeline switches to the calibrated board poses when it had to calibrate the camera
        # from the board itself (C-13), and records that it did in board_calibration.json.
        if (artifact_dir / BOARD_CALIBRATION_RECORD).exists():
            board_pose_dir = artifact_dir / CALIBRATED_BOARD_DIR_NAME
        else:
            board_pose_dir = artifact_dir / BOARD_POSE_DIR_NAME
        return cls(
            capture_dir=Path(capture_dir)
            if capture_dir
            else artifact_dir.parent / "frames",
            board_pose_dir=board_pose_dir,
            reconstruction_dir=artifact_dir / "reconstruction",
            fusion_dir=artifact_dir / "fusion",
            provenance_dir=artifact_dir / "provenance",
            uncertainty_dir=artifact_dir / "uncertainty",
        )


@dataclass(frozen=True)
class PartIsolation:
    kept_vertex_mask: np.ndarray
    kept_face_mask: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class CompletionOutcome:
    status: str
    warning: str | None
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerInvocation:
    exit_code: int | None
    timed_out: bool
    launch_error: str | None
    duration_s: float


# --- settings -------------------------------------------------------------------------------------


def completion_disabled(value: str | None = None) -> bool:
    """True only for the literal "0", the same convention as PROOFSHAPE_PREFETCH_MODEL.

    Anything else runs the stage, and completion.json records the raw value, so a typo like
    "false" shows up there rather than silently doing the opposite of what was meant.
    """
    raw = os.environ.get(COMPLETION_ENV, "") if value is None else value
    return raw.strip() == "0"


def resolve_runner_command(value: str | None = None) -> list[str] | None:
    """The runner command from PROOFSHAPE_COMPLETION_RUNNER, shell-split; None when unset or blank.

    Read per call, not at import, for the same reason as reconstruction_runner.resolve_device: the
    value that matters is the one uvicorn was actually started with.
    """
    raw = os.environ.get(RUNNER_ENV, "") if value is None else value
    if not raw.strip():
        return None
    return shlex.split(raw)


# --- 1. part isolation ----------------------------------------------------------------------------


def isolate_part(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    board_surface: BoardSurface,
    params: CompletionParameters = DEFAULT_PARAMETERS,
) -> PartIsolation:
    """Keep the Observed surface of the part itself, dropping the board skin and stray patches.

    The core is the Observed vertices high enough above the board to be part, not skin: above
    fusion's own cut (`board_surface.height_threshold_mm`) by `skin_allowance_mm`. Its largest
    spatial cluster is the part; smaller clusters are recorded and dropped. Below the core height,
    an Observed vertex down to fusion's cut is kept only if it lies inside a cone under the core
    whose walls are `min_wall_angle_deg` steep -- that is what brings back the bottom of the part's
    sides, which A-10's plain 6 mm cut threw away, without bringing back the flat skin around it.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(labels)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise ValueError(f"vertices_mm must have shape (N,3), got {vertices_mm.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if labels.shape != (len(vertices_mm),):
        raise ValueError(
            f"labels must have one entry per vertex, got shape {labels.shape}"
        )

    heights_mm = board_surface.height_above_board_mm(vertices_mm)
    clearance_mm = float(board_surface.height_threshold_mm)
    core_height_mm = clearance_mm + params.skin_allowance_mm
    observed = labels == LABEL_OBSERVED

    core_candidates = np.flatnonzero(observed & (heights_mm > core_height_mm))
    if len(core_candidates) < params.min_core_vertices:
        raise CompletionInputError(
            f"only {len(core_candidates)} Observed vertices lie more than {core_height_mm:.1f} mm "
            f"above the board (fusion's cut {clearance_mm:.1f} mm + skin_allowance_mm "
            f"{params.skin_allowance_mm}); at least {params.min_core_vertices} are needed to "
            "isolate the part. A part this low, or this little observed, can't be completed."
        )

    component_of, component_sizes = _spatial_components(
        vertices_mm[core_candidates], params.cluster_radius_mm
    )
    order = np.argsort(component_sizes)[::-1]
    largest = int(order[0])
    second_largest_size = int(component_sizes[order[1]]) if len(order) > 1 else 0
    core = core_candidates[component_of == largest]
    if len(core) < params.min_core_vertices:
        raise CompletionInputError(
            f"the largest cluster of high Observed vertices has only {len(core)} vertices "
            f"(minimum {params.min_core_vertices}); the observed surface is too fragmented to "
            "isolate one part"
        )

    core_xy = vertices_mm[core, :2]
    core_min = core_xy.min(axis=0)
    core_max = core_xy.max(axis=0)
    guard = params.edge_guard_mm
    footprint = np.asarray(params.footprint_mm, dtype=np.float64)
    if np.any(core_min < guard) or np.any(core_max > footprint - guard):
        raise CompletionInputError(
            f"the part's core spans x {core_min[0]:.1f}..{core_max[0]:.1f} mm, "
            f"y {core_min[1]:.1f}..{core_max[1]:.1f} mm, reaching within {guard} mm of the board "
            "pattern's edge. That is board skin above the core height, not part: raise "
            "skin_allowance_mm, or keep the part away from the pattern's edge."
        )

    lower = np.flatnonzero(
        observed & (heights_mm >= clearance_mm) & (heights_mm <= core_height_mm)
    )
    kept = np.zeros(len(vertices_mm), dtype=bool)
    kept[core] = True
    kept_lower_count = 0
    if len(lower):
        distance_to_core_mm, _ = cKDTree(core_xy).query(vertices_mm[lower, :2])
        cone_radius_mm = (core_height_mm - heights_mm[lower]) / math.tan(
            math.radians(params.min_wall_angle_deg)
        ) + params.wall_xy_tolerance_mm
        kept_lower = lower[distance_to_core_mm <= cone_radius_mm]
        kept[kept_lower] = True
        kept_lower_count = len(kept_lower)

    # A face is part of the partial only if all three corners are -- the same rule as
    # provenance.label_faces, so no face straddles kept part and dropped skin.
    kept_face_mask = kept[faces].all(axis=1) if len(faces) else np.zeros(0, dtype=bool)
    if not np.any(kept_face_mask):
        raise CompletionInputError(
            "no mesh face has all three vertices on the isolated part"
        )

    kept_heights = heights_mm[kept]
    report = {
        "clearance_mm": clearance_mm,
        "core_height_mm": core_height_mm,
        "observed_vertex_count": int(np.count_nonzero(observed)),
        "core_candidate_count": len(core_candidates),
        "core_vertex_count": len(core),
        "core_cluster_count": len(component_sizes),
        "second_largest_cluster_size": second_largest_size,
        "lower_wall_vertex_count": int(kept_lower_count),
        "kept_vertex_count": int(np.count_nonzero(kept)),
        "dropped_observed_vertex_count": int(np.count_nonzero(observed & ~kept)),
        "kept_face_count": int(np.count_nonzero(kept_face_mask)),
        "kept_height_range_mm": [float(kept_heights.min()), float(kept_heights.max())],
        "core_xy_min_mm": core_min.tolist(),
        "core_xy_max_mm": core_max.tolist(),
    }
    return PartIsolation(
        kept_vertex_mask=kept, kept_face_mask=kept_face_mask, report=report
    )


def _spatial_components(
    points_mm: np.ndarray, radius_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of points closer than `radius_mm`: (component per point, sizes)."""
    pairs = cKDTree(points_mm).query_pairs(r=radius_mm, output_type="ndarray")
    count = len(points_mm)
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(count, count),
    )
    _, component_of = connected_components(graph, directed=False)
    return component_of, np.bincount(component_of)


# --- 2. sampling ----------------------------------------------------------------------------------


def sample_surface_points(
    vertices_mm: np.ndarray, faces: np.ndarray, point_count: int, seed: int
) -> np.ndarray:
    """`point_count` points spread uniformly over the triangles' area, reproducibly from `seed`.

    Area-weighted so a large flat face isn't under-represented next to many small ones -- the model
    sees the surface, not the mesh's triangulation. The repository had no mesh sampler before A-04.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if point_count < 1:
        raise ValueError(f"point_count must be >= 1, got {point_count}")
    corner_a = vertices_mm[faces[:, 0]]
    corner_b = vertices_mm[faces[:, 1]]
    corner_c = vertices_mm[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(
        np.cross(corner_b - corner_a, corner_c - corner_a), axis=1
    )
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0:
        raise CompletionInputError("the faces to sample have no area")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=point_count, p=areas / total_area)
    # Uniform in a triangle: the square root keeps points from bunching at corner a.
    root_u = np.sqrt(rng.random(point_count))[:, None]
    v = rng.random(point_count)[:, None]
    return (
        (1.0 - root_u) * corner_a[chosen]
        + root_u * (1.0 - v) * corner_b[chosen]
        + root_u * v * corner_c[chosen]
    )


# --- 3. the photo ---------------------------------------------------------------------------------


def choose_view_frame(
    observations: dict[str, np.ndarray],
    kept_vertex_mask: np.ndarray,
    reconstruction_frame_names: np.ndarray,
    board_frame_names: np.ndarray,
    pose_found: np.ndarray,
) -> tuple[str, int]:
    """The posed frame that observed the most kept vertices: (frame name, observation count).

    R-09's observations already say which frame saw which vertex, so counting them per frame is a
    free visibility score. Frames are matched to R-02's board poses by name, never by index: the
    two files list frames independently. A frame with no board pose can't be projected, so it
    can't be cropped. Ties go to the earlier frame, so the choice is deterministic.
    """
    vertex_index = np.asarray(observations["vertex_index"], dtype=np.int64)
    frame_index = np.asarray(observations["frame_index"], dtype=np.int64)
    seen_kept = kept_vertex_mask[vertex_index]
    counts = np.bincount(
        frame_index[seen_kept], minlength=len(reconstruction_frame_names)
    )

    posed = {
        str(name): bool(found)
        for name, found in zip(board_frame_names, pose_found, strict=True)
    }
    best_name = None
    best_count = 0
    for index, name in enumerate(reconstruction_frame_names):
        name = str(name)
        if not posed.get(name, False):
            continue
        if counts[index] > best_count:
            best_name = name
            best_count = int(counts[index])
    if best_name is None:
        raise CompletionInputError(
            "no frame with a board pose observed any of the isolated part's vertices"
        )
    return best_name, best_count


def crop_view_image(
    frame_path: Path,
    points_mm: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    params: CompletionParameters = DEFAULT_PARAMETERS,
) -> tuple[Image.Image, dict[str, Any]]:
    """Crop one photo to the part's projected outline, then turn it upright.

    The crop is computed and taken in **stored sensor pixels**, ignoring EXIF orientation, because
    that is the pixel frame R-02's poses use; only the finished crop is rotated by its EXIF
    orientation, so the model sees the part upright as A-10 did by hand. Capture-app frames carry
    no EXIF and are already upright.
    """
    frame_path = Path(frame_path)
    rotation = np.asarray(extrinsic, dtype=np.float64)[:, :3]
    translation = np.asarray(extrinsic, dtype=np.float64)[:, 3]
    u, v, z_cam = project_points_to_camera(points_mm, rotation, translation, intrinsic)
    _, orientation = read_exif_fields(frame_path)

    with Image.open(frame_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    in_view = (
        np.isfinite(u)
        & np.isfinite(v)
        & (z_cam > 0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    if not np.any(in_view):
        raise CompletionInputError(
            f"no isolated part point projects inside {frame_path.name}"
        )

    u_min, u_max = float(u[in_view].min()), float(u[in_view].max())
    v_min, v_max = float(v[in_view].min()), float(v[in_view].max())
    side = max(u_max - u_min, v_max - v_min) * (1.0 + 2.0 * params.view_margin_fraction)
    side = round(min(max(side, 1.0), width, height))
    # Square, centred on the part, then shifted (never shrunk further) to stay inside the photo.
    left = round((u_min + u_max) / 2.0 - side / 2.0)
    top = round((v_min + v_max) / 2.0 - side / 2.0)
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    crop = image.crop((left, top, left + side, top + side))

    transpose = (
        _ORIENTATION_TRANSPOSE.get(orientation) if orientation is not None else None
    )
    if transpose is not None:
        crop = crop.transpose(transpose)
    if max(crop.size) > params.view_max_side_px:
        crop = crop.resize(
            (params.view_max_side_px, params.view_max_side_px), Image.Resampling.LANCZOS
        )

    record = {
        "frame_name": frame_path.name,
        "stored_size_px": [width, height],
        "crop_box_stored_px": [left, top, left + side, top + side],
        "projected_point_count": int(np.count_nonzero(in_view)),
        "exif_orientation": orientation,
        "output_size_px": list(crop.size),
    }
    return crop, record


# --- 4. preparing the request ---------------------------------------------------------------------


def prepare_completion_inputs(
    artifacts: PipelineArtifacts,
    out_dir: Path,
    params: CompletionParameters = DEFAULT_PARAMETERS,
) -> tuple[contract.CompletionRequest, dict[str, Any]]:
    """Write `input/`, the request, and return (request, input summary). Reads upstream only."""
    out_dir = Path(out_dir)
    vertices_mm, faces, colors = read_glb(
        artifacts.uncertainty_dir / "mesh_uncertainty.glb"
    )
    if colors is None:
        raise CompletionInputError("mesh_uncertainty.glb carries no provenance colours")
    # The served post-sigma labels (R-10): a vertex R-10 downgraded for noisy depth isn't trusted
    # surface to complete from either. np.array copies, since read_glb's arrays are read-only views.
    vertices_mm = np.array(vertices_mm, dtype=np.float64)
    labels = colors_to_labels(colors)

    fusion_report = json.loads(
        (artifacts.fusion_dir / "tsdf_fusion.json").read_text(encoding="utf-8")
    )
    if "board_surface" not in fusion_report:
        raise CompletionInputError(
            "tsdf_fusion.json has no board_surface record (pre-R-21 run)"
        )
    board_surface = BoardSurface(**fusion_report["board_surface"])

    isolation = isolate_part(vertices_mm, faces, labels, board_surface, params)
    kept_faces = np.asarray(faces)[isolation.kept_face_mask]
    partial_mm = sample_surface_points(
        vertices_mm, kept_faces, params.partial_point_count, params.partial_seed
    )
    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    partial_colors = np.tile(
        np.asarray(OBSERVED_COLOR, dtype=np.uint8), (len(partial_mm), 1)
    )
    write_ply(
        out_dir / contract.PARTIAL_POINTS_PATH,
        contract.board_to_z_up(partial_mm).astype(np.float32),
        partial_colors,
    )

    observations = read_observations_npz(artifacts.provenance_dir / "observations.npz")
    with np.load(artifacts.reconstruction_dir / "poses.npz") as poses:
        reconstruction_frame_names = poses["frame_names"]
    with np.load(artifacts.board_pose_dir / "board_poses.npz") as board_poses:
        board_frame_names = board_poses["frame_names"]
        pose_found = board_poses["pose_found"]
        board_extrinsic = board_poses["extrinsic"]
        board_intrinsic = board_poses["intrinsic"]
    frame_name, observation_count = choose_view_frame(
        observations,
        isolation.kept_vertex_mask,
        reconstruction_frame_names,
        board_frame_names,
        pose_found,
    )
    board_index = [str(name) for name in board_frame_names].index(frame_name)
    view, view_record = crop_view_image(
        artifacts.capture_dir / frame_name,
        vertices_mm[isolation.kept_vertex_mask],
        board_extrinsic[board_index],
        board_intrinsic[board_index],
        params,
    )
    view.save(out_dir / contract.VIEW_IMAGE_PATH, format="PNG")
    view_record["observation_count"] = observation_count

    seeds = tuple(params.base_seed + k for k in range(params.max_samples))
    request = contract.CompletionRequest(
        seeds=seeds,
        budget_s=params.budget_s,
        sparse_structure_steps=params.sparse_structure_steps,
        slat_steps=params.slat_steps,
    )
    contract.write_request(out_dir, request)

    partial_min = partial_mm.min(axis=0)
    partial_max = partial_mm.max(axis=0)
    summary = {
        "isolation": isolation.report,
        "partial_point_count": len(partial_mm),
        "partial_seed": params.partial_seed,
        "partial_extent_mm": (partial_max - partial_min).tolist(),
        "view": view_record,
    }
    return request, summary


# --- 5. running the runner ------------------------------------------------------------------------


def invoke_runner(
    command: list[str],
    request_path: Path,
    hard_timeout_s: float,
    log_tail_path: Path,
    tail_bytes: int,
) -> RunnerInvocation:
    """Run `command --request <path>` with a hard timeout, keeping only a tail of its output.

    `subprocess.run` with a timeout kills the runner if it overruns; it is used rather than a
    process-group kill because it behaves the same on Windows, where one of us runs the tests. A
    runner that spawns its own children must clean them up itself (A-12).
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _SECRET_ENV_NAMES
    }
    started = time.perf_counter()
    exit_code: int | None = None
    timed_out = False
    launch_error: str | None = None
    try:
        with open(log_tail_path, "wb") as log:
            completed = subprocess.run(
                [*command, "--request", str(Path(request_path).resolve())],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=hard_timeout_s,
                check=False,
            )
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    except OSError as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
    duration_s = time.perf_counter() - started
    _keep_tail(log_tail_path, tail_bytes)
    return RunnerInvocation(
        exit_code=exit_code,
        timed_out=timed_out,
        launch_error=launch_error,
        duration_s=duration_s,
    )


def _keep_tail(path: Path, tail_bytes: int) -> None:
    """Trim a log to its last `tail_bytes`: enough to debug a crash, not a disk-filling transcript."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size <= tail_bytes:
        return
    with open(path, "rb") as handle:
        handle.seek(size - tail_bytes)
        tail = handle.read()
    path.write_bytes(tail)


# --- 6. ingesting the samples ---------------------------------------------------------------------


def ingest_runner_outputs(
    out_dir: Path,
    request: contract.CompletionRequest,
    params: CompletionParameters = DEFAULT_PARAMETERS,
) -> tuple[contract.RunnerResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Check the runner's result and meshes; write `sample_<k>.ply`. Returns (result, kept, rejected).

    Raises ContractError if result.json itself is missing or malformed. A single bad mesh is
    rejected and recorded rather than discarding the samples that are fine.
    """
    out_dir = Path(out_dir)
    result = contract.parse_result(
        out_dir / contract.RUNNER_DIR / contract.RESULT_FILE, request
    )
    partial_mm = contract.z_up_to_board(
        contract.read_points_ply(out_dir / contract.PARTIAL_POINTS_PATH)
    )
    partial_min = partial_mm.min(axis=0)
    partial_max = partial_mm.max(axis=0)
    partial_diagonal = float(np.linalg.norm(partial_max - partial_min))

    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in result.samples:
        mesh_path = out_dir / contract.RUNNER_DIR / contract.sample_mesh_name(record.k)
        try:
            vertices_z_up, faces = contract.read_mesh_ply_strict(mesh_path)
            vertices_mm = contract.z_up_to_board(vertices_z_up)
            _check_sample_scale(
                vertices_mm, partial_min, partial_max, partial_diagonal, params
            )
            points_mm = sample_surface_points(
                vertices_mm, faces, params.sample_point_count, record.seed
            )
        except (contract.ContractError, CompletionInputError) as exc:
            rejected.append({"k": record.k, "seed": record.seed, "reason": str(exc)})
            continue
        sample_name = f"sample_{record.k}.ply"
        colors = np.tile(
            np.asarray(INFERRED_COLOR, dtype=np.uint8), (len(points_mm), 1)
        )
        write_ply(out_dir / sample_name, points_mm.astype(np.float32), colors)
        samples.append(
            {
                "k": record.k,
                "seed": record.seed,
                "path": sample_name,
                "point_count": len(points_mm),
                "mesh_vertex_count": len(vertices_mm),
                "mesh_face_count": len(faces),
                "runner_time_s": record.time_s,
            }
        )
    return result, samples, rejected


def _check_sample_scale(
    vertices_mm: np.ndarray,
    partial_min: np.ndarray,
    partial_max: np.ndarray,
    partial_diagonal: float,
    params: CompletionParameters,
) -> None:
    """Reject a completion in the wrong place or units -- the failure a missed de-normalisation makes.

    LaS-Comp works in a unit cube and maps back (A-10); a runner that forgets to map back returns a
    ~1-unit mesh near the origin, which would otherwise be stored as a "completion" of a 60 mm part.
    """
    sample_min = vertices_mm.min(axis=0)
    sample_max = vertices_mm.max(axis=0)
    if np.any(sample_max < partial_min) or np.any(sample_min > partial_max):
        raise CompletionInputError(
            "the completed mesh doesn't overlap the observed part"
        )
    diagonal = float(np.linalg.norm(sample_max - sample_min))
    limit = params.sample_scale_factor_limit
    if partial_diagonal > 0 and not (
        partial_diagonal / limit <= diagonal <= partial_diagonal * limit
    ):
        raise CompletionInputError(
            f"the completed mesh's size ({diagonal:.1f} mm across) is more than {limit}x off the "
            f"observed part's ({partial_diagonal:.1f} mm); wrong units, not a completion"
        )


def _classify(
    result: contract.RunnerResult,
    samples: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    ended_normally: bool,
) -> tuple[str, str | None]:
    """The stage's status from what came back, and a reason when it isn't plainly complete."""
    if not samples:
        reason = result.error or "the runner returned no valid samples"
        if rejected:
            reason = f"{reason}; rejected: " + "; ".join(
                item["reason"] for item in rejected
            )
        return "failed", reason
    finished = result.status in ("complete", "budget_reached")
    if ended_normally and finished and not rejected:
        return "complete", None
    reasons = []
    if not finished:
        reasons.append(
            f"runner status {result.status!r}"
            + (f": {result.error}" if result.error else "")
        )
    if not ended_normally:
        reasons.append("the runner did not exit cleanly")
    if rejected:
        reasons.append("rejected: " + "; ".join(item["reason"] for item in rejected))
    return "partial", "; ".join(reasons)


# --- 7. the stage ---------------------------------------------------------------------------------


def run_completion_stage(
    artifacts: PipelineArtifacts,
    out_dir: Path,
    runner_command: list[str] | None,
    params: CompletionParameters | None = None,
) -> CompletionOutcome:
    """Prepare, run and ingest. Never raises: every outcome is a status, and at most one warning.

    `out_dir` must be empty or absent; like every other stage, this one never overwrites output.
    `params` defaults to `DEFAULT_PARAMETERS`, looked up at call time rather than bound at import.
    """
    if params is None:
        params = DEFAULT_PARAMETERS
    out_dir = Path(out_dir)
    runner_configured = bool(runner_command)
    started = time.perf_counter()
    report = _base_report(runner_command, params)
    try:
        if out_dir.exists() and any(out_dir.iterdir()):
            # Nothing is written, so a previous run's record survives intact.
            return _fail(
                None,
                report,
                "output_dir",
                f"{out_dir} already has files; refusing to overwrite",
                runner_configured,
                started,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        return _run_stage(artifacts, out_dir, runner_command, params, report, started)
    except Exception as exc:  # noqa: BLE001 -- D-058: no completion outcome fails a reconstruction.
        return _fail(
            out_dir if out_dir.is_dir() else None,
            report,
            report.get("_stage", "internal"),
            f"{type(exc).__name__}: {exc}",
            runner_configured,
            started,
        )


def _run_stage(
    artifacts: PipelineArtifacts,
    out_dir: Path,
    runner_command: list[str] | None,
    params: CompletionParameters,
    report: dict[str, Any],
    started: float,
) -> CompletionOutcome:
    runner_configured = bool(runner_command)
    report["_stage"] = "input_preparation"
    prepare_started = time.perf_counter()
    try:
        request, summary = prepare_completion_inputs(artifacts, out_dir, params)
    except (
        CompletionInputError,
        contract.ContractError,
        OSError,
        KeyError,
        ValueError,
    ) as exc:
        report["timings_s"]["prepare"] = time.perf_counter() - prepare_started
        return _fail(
            out_dir, report, "input_preparation", str(exc), runner_configured, started
        )
    report["timings_s"]["prepare"] = time.perf_counter() - prepare_started
    report["input"] = summary
    report["k_requested"] = len(request.seeds)
    report["seeds"] = list(request.seeds)

    if not runner_configured:
        report["status"] = "unavailable"
        report["reason"] = (
            f"no completion runner is configured on this host ({RUNNER_ENV} is unset); the inputs "
            "are prepared, and `python -m recon.completion ingest` can bring back a run made elsewhere"
        )
        return _finish(out_dir, report, None, started)

    report["_stage"] = "runner"
    if not _RUNNER_LOCK.acquire(blocking=False):
        return _fail(
            out_dir,
            report,
            "runner",
            "another session's completion is already running; one runs at a time (D-057)",
            runner_configured,
            started,
        )
    try:
        hard_timeout_s = params.budget_s + params.overrun_allowance_s
        invocation = invoke_runner(
            runner_command,
            out_dir / contract.REQUEST_FILE,
            hard_timeout_s,
            out_dir / RUNNER_LOG_TAIL,
            params.log_tail_bytes,
        )
    finally:
        _RUNNER_LOCK.release()
    report["timings_s"]["runner"] = invocation.duration_s
    report["runner"].update(
        {
            "exit_code": invocation.exit_code,
            "timed_out": invocation.timed_out,
            "hard_timeout_s": hard_timeout_s,
        }
    )
    if invocation.launch_error is not None:
        return _fail(
            out_dir,
            report,
            "runner",
            f"the runner could not start: {invocation.launch_error}",
            runner_configured,
            started,
        )

    report["_stage"] = "ingest"
    ended_normally = invocation.exit_code == 0 and not invocation.timed_out
    return _ingest_into_report(
        out_dir, request, params, report, ended_normally, started
    )


def _ingest_into_report(
    out_dir: Path,
    request: contract.CompletionRequest,
    params: CompletionParameters,
    report: dict[str, Any],
    ended_normally: bool,
    started: float,
) -> CompletionOutcome:
    ingest_started = time.perf_counter()
    try:
        result, samples, rejected = ingest_runner_outputs(out_dir, request, params)
    except contract.ContractError as exc:
        report["timings_s"]["ingest"] = time.perf_counter() - ingest_started
        reason = f"the runner's result was unusable: {exc}"
        if not ended_normally:
            reason = f"the runner did not exit cleanly, and {reason}"
        return _fail(out_dir, report, "ingest", reason, True, started)
    report["timings_s"]["ingest"] = time.perf_counter() - ingest_started
    report["model"]["reported"] = result.model
    report["model"]["runner_status"] = result.status
    report["timings_s"]["runner_load"] = result.load_s
    report["samples"] = samples
    report["rejected_samples"] = rejected
    report["k_produced"] = len(samples)
    status, reason = _classify(result, samples, rejected, ended_normally)
    report["status"] = status
    report["reason"] = reason
    if status == "failed":
        report["failed_stage"] = "ingest"
        return _finish(out_dir, report, WARNING_FAILED, started)
    return _finish(
        out_dir, report, WARNING_PARTIAL if status == "partial" else None, started
    )


def _base_report(
    runner_command: list[str] | None, params: CompletionParameters
) -> dict[str, Any]:
    return {
        "story": "A-04",
        "schema_version": contract.SCHEMA_VERSION,
        "status": None,
        "failed_stage": None,
        "reason": None,
        "flag": {COMPLETION_ENV: os.environ.get(COMPLETION_ENV)},
        "runner": {"configured": bool(runner_command), "command": runner_command},
        "label": LABEL_INFERRED,
        "color": list(INFERRED_COLOR),
        "units": "mm",
        "frame": "board frame (+z into the table, recon/board_pose.py)",
        "k_requested": 0,
        "k_produced": 0,
        "seeds": [],
        "samples": [],
        "rejected_samples": [],
        "model": {"expected": contract.ExpectedModel().as_dict(), "reported": None},
        "parameters": params.as_report(),
        "provisional_parameters": list(PROVISIONAL_PARAMETERS),
        "timings_s": {},
        "input": None,
    }


def _fail(
    out_dir: Path | None,
    report: dict[str, Any],
    stage: str,
    reason: str,
    runner_configured: bool,
    started: float,
) -> CompletionOutcome:
    report["status"] = "failed"
    report["failed_stage"] = stage
    report["reason"] = reason
    # D-058: without a runner nothing was going to complete anyway, so a failure to prepare
    # inputs is recorded but not shown to the supplier; with a runner it is a real failure.
    warning = WARNING_FAILED if runner_configured else None
    return _finish(out_dir, report, warning, started)


def _finish(
    out_dir: Path | None, report: dict[str, Any], warning: str | None, started: float
) -> CompletionOutcome:
    report.pop("_stage", None)
    report["timings_s"]["total"] = time.perf_counter() - started
    if out_dir is not None:
        try:
            contract.write_json_atomic(out_dir / COMPLETION_REPORT, report)
        except OSError as exc:
            # The record itself couldn't be written. Still never fail the reconstruction; with a
            # runner configured the warning says completion failed, and the stage's return value
            # carries the reason for anyone calling it directly.
            report["status"] = "failed"
            report["reason"] = (
                f"{report.get('reason') or ''} (completion.json not written: {exc})"
            )
            if report["runner"]["configured"]:
                warning = WARNING_FAILED
    return CompletionOutcome(status=report["status"], warning=warning, report=report)


# --- 8. the CLI: prepare here, run on a CUDA host, ingest here (A-12) ------------------------------


def ingest_existing(
    out_dir: Path, params: CompletionParameters = DEFAULT_PARAMETERS
) -> CompletionOutcome:
    """Ingest a runner's outputs copied into `out_dir/runner/` from a run made on another machine.

    Only for a folder the stage prepared without a runner (status `unavailable`) and that holds no
    samples yet: it never replaces a completion this service already ran.
    """
    out_dir = Path(out_dir)
    report_path = out_dir / COMPLETION_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "unavailable":
        raise CompletionInputError(
            f"{report_path} has status {report.get('status')!r}; only an 'unavailable' "
            "(prepared, not run) folder can be ingested"
        )
    if any(out_dir.glob("sample_*.ply")):
        raise CompletionInputError(
            f"{out_dir} already has samples; refusing to overwrite them"
        )
    request = contract.read_request(out_dir / contract.REQUEST_FILE)
    report["runner"]["ran_elsewhere"] = True
    report["parameters"]["sample_point_count"] = params.sample_point_count
    started = time.perf_counter()
    # The run happened elsewhere, so there is no exit code here; the runner's own status is the
    # only evidence of whether it finished.
    return _ingest_into_report(out_dir, request, params, report, True, started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recon.completion",
        description=(
            "A-04: prepare a completion request from a reconstruction's artifacts folder, or "
            "ingest a runner's outputs copied back into it."
        ),
    )
    parser.add_argument("action", choices=("prepare", "ingest"))
    parser.add_argument("artifact_dir", type=Path, help="a session's artifacts/ folder")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="the session's frames (default: artifacts/../frames)",
    )
    args = parser.parse_args(argv)

    out_dir = args.artifact_dir / "completion"
    try:
        if args.action == "prepare":
            if out_dir.exists():
                raise CompletionInputError(
                    f"{out_dir} already exists; refusing to overwrite it"
                )
            artifacts = PipelineArtifacts.from_artifact_dir(
                args.artifact_dir, args.frames_dir
            )
            outcome = run_completion_stage(artifacts, out_dir, runner_command=None)
            passed = outcome.status == "unavailable"
        else:
            outcome = ingest_existing(out_dir)
            passed = outcome.status == "complete"
    except (CompletionInputError, contract.ContractError, OSError, ValueError) as exc:
        print(f"A-04 COMPLETION: FAIL - {exc}", file=sys.stderr)
        return 1

    summary = (
        f"status {outcome.status}, {outcome.report.get('k_produced', 0)} sample(s)"
    )
    if outcome.report.get("reason"):
        summary += f" ({outcome.report['reason']})"
    if passed:
        print(
            f"A-04 COMPLETION: PASS - {args.action}: {summary}; see {out_dir / COMPLETION_REPORT}"
        )
        return 0
    print(f"A-04 COMPLETION: FAIL - {args.action}: {summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
