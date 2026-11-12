"""I-03 global CAD-to-reconstruction registration.

The reconstructed model is already metric, so every pose returned here is rigid: scale is fixed
at 1.0. A free-scale similarity is solved only after the best rigid pose is chosen, using that
pose's geometric correspondences, and is reported as scale - 1 for R-05/D-039.

Buyer CAD is authored z-up from z=0. Reconstruction is in the board frame with the part body at
negative z. The fixed CAD-to-board convention change is the same proper 180-degree x rotation
used by I-01: diag(1, -1, -1). Registration therefore searches only yaw about board z plus xy
shift; it never hides a size error by scaling the CAD.

I-04's `refine` then tightens one candidate with point-to-plane ICP under a Tukey loss, fitted
on fully Observed model geometry only, and recomputes the free-scale residual from the refined
pose.

I-05's `check_ambiguity` refines every candidate, merges only poses whose transforms agree, and
flags the session ambiguous when the runner-up fits nearly as well as the best (D-008, D-061).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import open3d as o3d

from recon.provenance import (
    LABEL_INFERRED,
    LABEL_OBSERVED,
    LABEL_UNOBSERVED,
    label_faces,
)

DEFAULT_VOXEL_FRACTION = 0.03
DEFAULT_NORMAL_RADIUS_FACTOR = 2.5
DEFAULT_FPFH_RADIUS_FACTOR = 5.0
DEFAULT_CORRESPONDENCE_FACTOR = 2.5
DEFAULT_SAMPLE_COUNT = 1600
DEFAULT_YAW_STEPS = 72
DEFAULT_CANDIDATE_COUNT = 8
DEFAULT_RANSAC_ITERATIONS = 20_000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_SCALE_ICP_ITERATIONS = 100
# Algorithmic evidence gate, not a measured inspection tolerance: matched CAD evidence must span
# at least 2% as much in its weakest principal axis as in its strongest before uniform scale is
# reported. This rejects planar-only views without letting observation noise create fake 3-D span.
DEFAULT_SCALE_MIN_AXIS_RATIO = 0.02
DEFAULT_SCALE_TARGET_SAMPLE_COUNT = 20_000
DEFAULT_DISTINCT_ROTATION_DEG = 2.0
DEFAULT_DISTINCT_TRANSLATION_FRACTION = 0.01
# Floor under the translation tolerance above, so a very small part's 1% doesn't shrink below
# numerical noise. An algorithmic choice, not a tolerance (it was inline in I-03; #233 asked for
# it to be named).
DEFAULT_DISTINCT_TRANSLATION_FLOOR_MM = 0.25
# Sanity bound on the free-scale residual, a labelled assumption rather than a tolerance: a
# reading beyond +-25% isn't a plausible scale error for a metric reconstruction but a failed
# similarity fit. Under heavy occlusion (s-04 x1.03 with the bottom and half the part hidden)
# 5 of 24 seeds read +0.29 to +6.36 against a true +0.03 (#233, I-05). Such a reading becomes
# None, so D-039's scale agreement gets no CAD vote rather than a nonsense one. Inside the bound,
# heavy occlusion can still misread (often with the wrong sign); this is not an occlusion gate.
DEFAULT_SCALE_RESIDUAL_SANITY_BOUND = 0.25
# I-04 refinement parameters. Algorithmic choices, not inspection tolerances (D-003):
# Tukey's k is a multiple of the voxel I-03's scale diagnostic uses: 3% of the CAD's extent alone
# (I-03's pose search sizes its voxel from the larger of model and CAD). A residual beyond k gets
# zero weight, which is what stops a defect from dragging the fit. On I-01 output, k from 0.5 to
# 2 voxels refined s-04/s-05 to the same 0.04-0.09 mm surface error, so 1 voxel is a mid choice,
# not a tuned optimum.
DEFAULT_TUKEY_K_FACTOR = 1.0
DEFAULT_REFINE_ICP_ITERATIONS = 100
# I-05. The session is ambiguous when the runner-up's inlier RMSE is within this fraction of the
# best's (D-061, superseding D-008's 10%). The parameter is the decision's, not a tuned value.
DEFAULT_AMBIGUITY_THRESHOLD = 0.20
# The contract's reason code for an ambiguous orientation
# (contracts/examples/getVerdict.response-200-rescan-unobserved.json). I-07 puts it in the verdict.
REASON_ORIENTATION_AMBIGUOUS = "registration_orientation_ambiguous"

CAD_TO_BOARD_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class CandidatePose:
    """One rigid CAD-to-model candidate returned by global_register."""

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    source: str
    correspondence_count: int
    free_scale_residual: float | None = None


@dataclass(frozen=True)
class AmbiguityCheck:
    """I-05's result: every refined pose, ranked, and whether the orientation is ambiguous.

    `best` is the pose I-06/I-07 measure deviations against. `rmse_ratio` is the runner-up's
    inlier RMSE over the best's (None without a runner-up). `failed` lists each candidate that
    couldn't be refined, as (source, reason), so nothing drops out silently.
    """

    orientation_ambiguous: bool
    best: CandidatePose
    runner_up: CandidatePose | None
    rmse_ratio: float | None
    threshold: float
    refined: tuple[CandidatePose, ...]
    failed: tuple[tuple[str, str], ...]

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """The contract's reason codes this result contributes to the verdict (I-07)."""
        return (REASON_ORIENTATION_AMBIGUOUS,) if self.orientation_ambiguous else ()


def _mesh_arrays(
    mesh: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Accept I-01/I-02 mesh shapes while preserving model provenance when present.

    I-01's SyntheticSample exposes per-vertex labels. Tuple callers may supply labels as the
    third item; I-02 CAD tuples keep their units string in that slot and therefore have no
    provenance. A reconstructed model without provenance can still produce pose candidates, but
    it cannot produce a scale diagnostic that could influence inspection.
    """
    labels: np.ndarray | None = None
    if hasattr(mesh, "vertices_mm") and hasattr(mesh, "faces"):
        vertices = mesh.vertices_mm
        faces = mesh.faces
        if hasattr(mesh, "labels"):
            labels = np.asarray(mesh.labels, dtype=object)
    elif isinstance(mesh, (tuple, list)) and len(mesh) >= 2:
        vertices, faces = mesh[0], mesh[1]
        if len(mesh) >= 3 and not isinstance(mesh[2], str):
            candidate_labels = np.asarray(mesh[2], dtype=object)
            if candidate_labels.ndim == 1:
                labels = candidate_labels
    else:
        raise TypeError(
            f"{name} must expose vertices_mm/faces or be a (vertices, faces, ...) tuple."
        )

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"{name} vertices must have shape (N,3) with N >= 3.")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{name} vertices must all be finite.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise ValueError(f"{name} faces must have shape (F,3) with F >= 1.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{name} faces reference a vertex outside the vertex array.")

    if labels is not None:
        if labels.shape != (len(vertices),):
            raise ValueError(
                f"{name} labels must have one entry per vertex, got {labels.shape}."
            )
        known = np.isin(
            labels,
            [LABEL_OBSERVED, LABEL_UNOBSERVED, LABEL_INFERRED],
        )
        if not np.all(known):
            unknown = sorted({str(value) for value in labels[~known]})
            raise ValueError(f"{name} has unknown provenance labels: {unknown[:5]}.")

    return vertices, faces, labels


def _observed_faces(
    faces: np.ndarray,
    labels: np.ndarray | None,
) -> np.ndarray | None:
    """Fully Observed faces, or None when provenance is unavailable."""
    if labels is None:
        return None
    observed = label_faces(faces, labels) == LABEL_OBSERVED
    return faces[observed]


def _triangle_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def _sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> o3d.geometry.PointCloud:
    if sample_count < 50:
        raise ValueError("sample_count must be at least 50.")
    o3d.utility.random.seed(int(seed))
    cloud = _triangle_mesh(vertices, faces).sample_points_uniformly(
        number_of_points=int(sample_count),
        use_triangle_normal=True,
    )
    if len(cloud.points) < 3:
        raise ValueError("Surface sampling produced fewer than three points.")
    return cloud


def _extent_mm(*point_sets: np.ndarray) -> float:
    extent = max(float(np.max(np.ptp(points, axis=0))) for points in point_sets)
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Part extent must be positive and finite.")
    return extent


def _preprocess_fpfh(
    cloud: o3d.geometry.PointCloud,
    *,
    voxel_size_mm: float,
    normal_radius_mm: float,
    fpfh_radius_mm: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    down = cloud.voxel_down_sample(voxel_size_mm)
    if len(down.points) < 4:
        raise ValueError(
            "Voxel downsampling left fewer than four points; reduce voxel_fraction."
        )
    if not down.has_normals():
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius_mm,
                max_nn=60,
            )
        )
    down.normalize_normals()
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius_mm,
            max_nn=120,
        ),
    )
    return down, feature


def _yaw_matrix(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _board_pose_from_yaw(
    cad_points: np.ndarray,
    model_points: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    """Rigid CAD-to-board pose: fixed frame flip, yaw, then xy centroid alignment."""
    transform = _yaw_matrix(yaw_rad) @ CAD_TO_BOARD_FLIP
    rotated = cad_points @ transform[:3, :3].T
    delta_xy = model_points[:, :2].mean(axis=0) - rotated[:, :2].mean(axis=0)
    transform[:2, 3] = delta_xy
    transform[2, 3] = 0.0
    return transform


def _project_ransac_to_board_pose(
    preflipped_transform: np.ndarray,
    cad_points: np.ndarray,
    model_points: np.ndarray,
) -> np.ndarray:
    """Project unconstrained RANSAC onto the known yaw-plus-xy board pose."""
    transform = np.asarray(preflipped_transform, dtype=np.float64)
    yaw = math.atan2(transform[1, 0], transform[0, 0])
    candidate = _board_pose_from_yaw(cad_points, model_points, yaw)
    candidate[:2, 3] = transform[:2, 3]
    candidate[2, 3] = 0.0
    return candidate


def _principal_axis_yaw(points: np.ndarray) -> tuple[float, float]:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    values = np.sort(values)
    anisotropy = (
        0.0 if values[-1] <= 0 else float((values[-1] - values[-2]) / values[-1])
    )
    return math.atan2(float(major[1]), float(major[0])), anisotropy


def _yaw_seed_angles(
    cad_board_points: np.ndarray,
    model_points: np.ndarray,
    yaw_steps: int,
) -> list[float]:
    if yaw_steps < 4:
        raise ValueError("yaw_steps must be at least 4.")

    angles = [2.0 * math.pi * i / yaw_steps for i in range(yaw_steps)]
    source_angle, source_anisotropy = _principal_axis_yaw(cad_board_points)
    target_angle, target_anisotropy = _principal_axis_yaw(model_points)
    if min(source_anisotropy, target_anisotropy) >= 0.02:
        theta = target_angle - source_angle
        angles.extend([theta, theta + math.pi])

    normalized: list[float] = []
    for angle in angles:
        angle = float(angle % (2.0 * math.pi))
        if not any(
            abs(math.atan2(math.sin(angle - old), math.cos(angle - old))) < 1e-8
            for old in normalized
        ):
            normalized.append(angle)
    return normalized


def _transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    transformation = np.asarray(transformation, dtype=np.float64)
    return points @ transformation[:3, :3].T + transformation[:3, 3]


def _evaluate_pose(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    max_correspondence_distance_mm: float,
    *,
    source_name: str,
) -> CandidatePose:
    result = o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        max_correspondence_distance_mm,
        transformation,
    )
    correspondences = np.asarray(result.correspondence_set)
    count = 0 if correspondences.size == 0 else len(correspondences)
    rmse = float(result.inlier_rmse)
    if not math.isfinite(rmse):
        rmse = math.inf
    return CandidatePose(
        transformation=np.asarray(transformation, dtype=np.float64).copy(),
        fitness=float(result.fitness),
        inlier_rmse=rmse,
        source=source_name,
        correspondence_count=count,
    )


def _rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3] @ b[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _same_pose(
    a: CandidatePose,
    b: CandidatePose,
    *,
    translation_tolerance_mm: float,
    rotation_tolerance_deg: float,
) -> bool:
    translation_delta = float(
        np.linalg.norm(a.transformation[:3, 3] - b.transformation[:3, 3])
    )
    return (
        translation_delta <= translation_tolerance_mm
        and _rotation_delta_deg(a.transformation, b.transformation)
        <= rotation_tolerance_deg
    )


def _candidate_sort_key(candidate: CandidatePose) -> tuple[float, float]:
    return (-candidate.fitness, candidate.inlier_rmse)


def _pose_yaw_rad(candidate: CandidatePose) -> float:
    rotation = candidate.transformation[:3, :3]
    return float(math.atan2(rotation[1, 0], rotation[0, 0]) % (2.0 * math.pi))


def _yaw_distance_rad(left: CandidatePose, right: CandidatePose) -> float:
    delta = _pose_yaw_rad(left) - _pose_yaw_rad(right)
    return abs(math.atan2(math.sin(delta), math.cos(delta)))


def _select_orientation_diverse(
    candidates: list[CandidatePose],
    count: int,
) -> list[CandidatePose]:
    """Keep the best pose, then spread the remaining seeds across distinct yaw basins.

    I-05 needs a genuine runner-up orientation, not several 5-degree neighbours of the same
    optimum. Farthest-yaw selection keeps the best-scoring candidate first, then greedily chooses
    the pose farthest from every already-kept yaw; score/RMSE break equal-distance ties.
    """
    if count < 1 or not candidates:
        return []

    ranked = sorted(candidates, key=_candidate_sort_key)
    selected = [ranked[0]]
    remaining = ranked[1:]
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                min(_yaw_distance_rad(remaining[index], kept) for kept in selected),
                remaining[index].fitness,
                -remaining[index].inlier_rmse,
            ),
        )
        selected.append(remaining.pop(best_index))
    return selected


def _deduplicate_candidates(
    candidates: list[CandidatePose],
    *,
    translation_tolerance_mm: float,
    rotation_tolerance_deg: float,
) -> list[CandidatePose]:
    unique: list[CandidatePose] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if not any(
            _same_pose(
                candidate,
                kept,
                translation_tolerance_mm=translation_tolerance_mm,
                rotation_tolerance_deg=rotation_tolerance_deg,
            )
            for kept in unique
        ):
            unique.append(candidate)
    return unique


def _run_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_feature: o3d.pipelines.registration.Feature,
    target_feature: o3d.pipelines.registration.Feature,
    *,
    distance_threshold_mm: float,
    seed: int,
    max_iterations: int,
    confidence: float,
) -> np.ndarray:
    o3d.utility.random.seed(int(seed))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_feature,
        target_feature,
        True,
        distance_threshold_mm,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold_mm
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(max_iterations),
            float(confidence),
        ),
    )
    return np.asarray(result.transformation, dtype=np.float64)


def _observed_scale_seed(
    cad_surface: o3d.geometry.PointCloud,
    observed_model_surface: o3d.geometry.PointCloud,
    *,
    yaw_steps: int,
    max_correspondence_distance_mm: float,
) -> CandidatePose:
    """Choose a rigid scale-diagnostic seed using only CAD and Observed model surface.

    This seed is intentionally separate from the returned global-registration winner. The latter
    is a pose hypothesis for I-04/I-05; the scale diagnostic is inspection evidence and therefore
    must not let Unobserved/Inferred geometry choose its orientation or translation.
    """
    cad_points = np.asarray(cad_surface.points, dtype=np.float64)
    observed_points = np.asarray(observed_model_surface.points, dtype=np.float64)
    cad_board_points = cad_points @ CAD_TO_BOARD_FLIP[:3, :3].T

    candidates = []
    for angle in _yaw_seed_angles(cad_board_points, observed_points, yaw_steps):
        pose = _board_pose_from_yaw(cad_points, observed_points, angle)
        candidates.append(
            _evaluate_pose(
                cad_surface,
                observed_model_surface,
                pose,
                max_correspondence_distance_mm,
                source_name=f"observed-scale-yaw:{math.degrees(angle):.3f}",
            )
        )
    if not candidates:
        raise ValueError("Observed scale diagnostic produced no rigid seed candidates.")
    return min(candidates, key=_candidate_sort_key)


def _free_scale_residual(
    candidate: CandidatePose,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    max_correspondence_distance_mm: float,
    max_iterations: int = DEFAULT_SCALE_ICP_ITERATIONS,
) -> float | None:
    """Fit Observed model samples to CAD and report model scale minus one.

    candidate is still a rigid CAD-to-model pose. For the scale diagnostic we invert that
    seed and register Observed model -> CAD instead. This direction matters under occlusion:
    every Observed model sample has a real CAD counterpart, while the reverse direction lets CAD
    samples over a hidden patch collapse toward the hole edge and biases the fitted scale small.

    A rigid point-to-point ICP first removes remaining pose error; a second ICP then enables
    uniform scale. Planar-only Observed evidence is refused because it cannot robustly establish
    a 3-D uniform scale for inspection.
    """
    source_points = np.asarray(source.points, dtype=np.float64)
    if len(source_points) < 3:
        return None

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if not (
        math.isfinite(max_correspondence_distance_mm)
        and max_correspondence_distance_mm > 0
    ):
        raise ValueError("max_correspondence_distance_mm must be positive and finite.")

    model_to_cad_seed = np.linalg.inv(
        np.asarray(candidate.transformation, dtype=np.float64)
    )
    rigid = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_mm,
        model_to_cad_seed,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iterations)
        ),
    )
    rigid_correspondences = np.asarray(rigid.correspondence_set)
    if rigid_correspondences.ndim != 2 or rigid_correspondences.shape[0] < 3:
        return None

    target_points = np.asarray(target.points, dtype=np.float64)
    matched_target = target_points[rigid_correspondences[:, 1].astype(np.int64)]
    centred_target = matched_target - matched_target.mean(axis=0)
    singular_values = np.linalg.svd(centred_target, compute_uv=False)
    if (
        len(singular_values) < 3
        or singular_values[0] <= 1e-9
        or singular_values[2] / singular_values[0] < DEFAULT_SCALE_MIN_AXIS_RATIO
    ):
        return None

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_mm,
        np.asarray(rigid.transformation, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iterations)
        ),
    )
    correspondences = np.asarray(result.correspondence_set)
    if correspondences.ndim != 2 or correspondences.shape[0] < 3:
        return None

    linear = np.asarray(result.transformation, dtype=np.float64)[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not math.isfinite(determinant) or determinant <= 0.0:
        return None
    model_to_cad_scale = float(np.cbrt(determinant))
    if not math.isfinite(model_to_cad_scale) or model_to_cad_scale <= 1e-9:
        return None
    residual = 1.0 / model_to_cad_scale - 1.0
    if abs(residual) > DEFAULT_SCALE_RESIDUAL_SANITY_BOUND:
        # A failed similarity fit, not a scale measurement (see the constant's comment).
        return None
    return residual


def global_register(
    model: Any,
    cad: Any,
    *,
    voxel_fraction: float = DEFAULT_VOXEL_FRACTION,
    normal_radius_factor: float = DEFAULT_NORMAL_RADIUS_FACTOR,
    fpfh_radius_factor: float = DEFAULT_FPFH_RADIUS_FACTOR,
    correspondence_factor: float = DEFAULT_CORRESPONDENCE_FACTOR,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    yaw_steps: int = DEFAULT_YAW_STEPS,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    ransac_iterations: int = DEFAULT_RANSAC_ITERATIONS,
    ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
    seed: int = 0,
) -> list[CandidatePose]:
    """Return distinct rigid CAD-to-model candidates, best first.

    Defaults are relative to part extent: voxel = 3%, normal radius = 2.5 voxels,
    FPFH radius = 5 voxels, and correspondence distance = 2.5 voxels. These are algorithm
    defaults, not inspection tolerances. candidate_count is the minimum number of distinct poses
    preserved for I-05; if RANSAC plus yaw enumeration yield fewer, RANSAC is reseeded.

    Every returned transform maps the original z-up CAD coordinates directly into the
    reconstruction board frame. The first candidate also carries the free-scale residual.
    """
    if not (math.isfinite(voxel_fraction) and voxel_fraction > 0):
        raise ValueError("voxel_fraction must be positive and finite.")
    if not (math.isfinite(normal_radius_factor) and normal_radius_factor > 0):
        raise ValueError("normal_radius_factor must be positive and finite.")
    if not (math.isfinite(fpfh_radius_factor) and fpfh_radius_factor > 0):
        raise ValueError("fpfh_radius_factor must be positive and finite.")
    if not (math.isfinite(correspondence_factor) and correspondence_factor > 0):
        raise ValueError("correspondence_factor must be positive and finite.")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2.")
    if ransac_iterations < 1:
        raise ValueError("ransac_iterations must be positive.")
    if not (0.0 < ransac_confidence <= 1.0):
        raise ValueError("ransac_confidence must be in (0, 1].")

    model_vertices, model_faces, model_labels = _mesh_arrays(model, name="model")
    cad_vertices, cad_faces, _cad_labels = _mesh_arrays(cad, name="cad")
    observed_model_faces = _observed_faces(model_faces, model_labels)
    extent = _extent_mm(model_vertices, cad_vertices)
    voxel_size = extent * voxel_fraction
    normal_radius = voxel_size * normal_radius_factor
    fpfh_radius = voxel_size * fpfh_radius_factor
    correspondence_distance = voxel_size * correspondence_factor

    cad_surface = _sample_surface(
        cad_vertices,
        cad_faces,
        sample_count=sample_count,
        seed=seed,
    )
    model_surface = _sample_surface(
        model_vertices,
        model_faces,
        sample_count=sample_count,
        seed=seed + 1,
    )
    cad_points = np.asarray(cad_surface.points)
    model_points = np.asarray(model_surface.points)

    cad_board_surface = o3d.geometry.PointCloud(cad_surface)
    cad_board_surface.transform(CAD_TO_BOARD_FLIP)

    source_down, source_feature = _preprocess_fpfh(
        cad_board_surface,
        voxel_size_mm=voxel_size,
        normal_radius_mm=normal_radius,
        fpfh_radius_mm=fpfh_radius,
    )
    target_down, target_feature = _preprocess_fpfh(
        model_surface,
        voxel_size_mm=voxel_size,
        normal_radius_mm=normal_radius,
        fpfh_radius_mm=fpfh_radius,
    )

    all_candidates: list[CandidatePose] = []
    initial_ransac = _run_ransac(
        source_down,
        target_down,
        source_feature,
        target_feature,
        distance_threshold_mm=correspondence_distance,
        seed=seed,
        max_iterations=ransac_iterations,
        confidence=ransac_confidence,
    )
    initial_pose = _project_ransac_to_board_pose(
        initial_ransac,
        cad_points,
        model_points,
    )
    initial_candidate = _evaluate_pose(
        cad_surface,
        model_surface,
        initial_pose,
        correspondence_distance,
        source_name="ransac:0",
    )
    all_candidates.append(initial_candidate)

    cad_board_points = cad_points @ CAD_TO_BOARD_FLIP[:3, :3].T
    for angle in _yaw_seed_angles(cad_board_points, model_points, yaw_steps):
        pose = _board_pose_from_yaw(cad_points, model_points, angle)
        all_candidates.append(
            _evaluate_pose(
                cad_surface,
                model_surface,
                pose,
                correspondence_distance,
                source_name=f"yaw:{math.degrees(angle):.3f}",
            )
        )

    translation_tolerance = max(
        DEFAULT_DISTINCT_TRANSLATION_FLOOR_MM,
        extent * DEFAULT_DISTINCT_TRANSLATION_FRACTION,
    )
    unique = _deduplicate_candidates(
        all_candidates,
        translation_tolerance_mm=translation_tolerance,
        rotation_tolerance_deg=DEFAULT_DISTINCT_ROTATION_DEG,
    )
    selected = _select_orientation_diverse(unique, candidate_count)
    if not any(
        _same_pose(
            initial_candidate,
            candidate,
            translation_tolerance_mm=translation_tolerance,
            rotation_tolerance_deg=DEFAULT_DISTINCT_ROTATION_DEG,
        )
        for candidate in selected
    ):
        selected.append(initial_candidate)

    reseed = 1
    while len(selected) < candidate_count and reseed <= candidate_count * 2:
        ransac = _run_ransac(
            source_down,
            target_down,
            source_feature,
            target_feature,
            distance_threshold_mm=correspondence_distance,
            seed=seed + reseed,
            max_iterations=ransac_iterations,
            confidence=ransac_confidence,
        )
        pose = _project_ransac_to_board_pose(ransac, cad_points, model_points)
        candidate = _evaluate_pose(
            cad_surface,
            model_surface,
            pose,
            correspondence_distance,
            source_name=f"ransac:{reseed}",
        )
        unique = _deduplicate_candidates(
            [*selected, candidate],
            translation_tolerance_mm=translation_tolerance,
            rotation_tolerance_deg=DEFAULT_DISTINCT_ROTATION_DEG,
        )
        selected = _select_orientation_diverse(unique, candidate_count)
        reseed += 1

    if len(selected) < 2:
        raise ValueError(
            "Global registration produced fewer than two distinct candidate poses."
        )

    selected = sorted(selected, key=_candidate_sort_key)
    best = selected[0]
    residual: float | None = None
    if observed_model_faces is not None and len(observed_model_faces):
        observed_model_surface = _sample_surface(
            model_vertices,
            observed_model_faces,
            sample_count=sample_count,
            seed=seed + 10_001,
        )
        scale_cad_surface = _sample_surface(
            cad_vertices,
            cad_faces,
            sample_count=max(sample_count, DEFAULT_SCALE_TARGET_SAMPLE_COUNT),
            seed=seed + 10_002,
        )
        scale_distance = (
            _extent_mm(cad_vertices) * voxel_fraction * correspondence_factor
        )
        scale_seed = _observed_scale_seed(
            scale_cad_surface,
            observed_model_surface,
            yaw_steps=yaw_steps,
            max_correspondence_distance_mm=scale_distance,
        )
        residual = _free_scale_residual(
            scale_seed,
            observed_model_surface,
            scale_cad_surface,
            max_correspondence_distance_mm=scale_distance,
        )
    selected[0] = replace(best, free_scale_residual=residual)
    return selected


def refine(
    candidate: CandidatePose,
    model: Any,
    cad: Any,
    *,
    voxel_fraction: float = DEFAULT_VOXEL_FRACTION,
    tukey_k_factor: float = DEFAULT_TUKEY_K_FACTOR,
    correspondence_factor: float = DEFAULT_CORRESPONDENCE_FACTOR,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    max_iterations: int = DEFAULT_REFINE_ICP_ITERATIONS,
    seed: int = 0,
) -> CandidatePose:
    """I-04: refine one I-03 candidate with point-to-plane ICP under a Tukey loss.

    A least-squares fit would explain a defect away by shifting the whole part toward it, which
    is exactly what inspection must not do (D-008). Tukey's loss gives zero weight to residuals
    beyond k, so a dent, a bump or a shifted hole stops pulling the fit once the rest of the part
    is aligned.

    Only fully Observed model faces drive the fit. This pose is what I-06/I-07 measure deviations
    against, so letting Unobserved or Inferred (completed) geometry move it would let geometry
    nobody measured influence a pass or fail (the hard rule, D-005). A model without provenance,
    or with no fully Observed face, is refused rather than refined on guessed geometry.

    The ICP runs Observed model -> CAD, the same direction as I-03's scale diagnostic. Every
    Observed model point has a real CAD counterpart; the reverse would let CAD samples over a
    hidden patch pull toward the hole. It also puts the face normals point-to-plane needs on the
    target: the sampled CAD surface's own triangle normals (a six-vertex prism's vertex normals
    are corner averages, which point-to-plane can't use).

    The fit is a full rigid 6-DoF pose, as `registration_icp` returns; it is not constrained to
    I-03's yaw-plus-xy board pose, so small real board-plane errors can be absorbed.

    Returns a new rigid CandidatePose (CAD -> model, like I-03's) whose fitness, inlier_rmse and
    correspondence_count describe the refined Observed-model -> CAD fit.
    """
    for name, value in (
        ("voxel_fraction", voxel_fraction),
        ("tukey_k_factor", tukey_k_factor),
        ("correspondence_factor", correspondence_factor),
    ):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive and finite.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    model_vertices, model_faces, model_labels = _mesh_arrays(model, name="model")
    cad_vertices, cad_faces, _cad_labels = _mesh_arrays(cad, name="cad")
    observed_model_faces = _observed_faces(model_faces, model_labels)
    if observed_model_faces is None:
        raise ValueError(
            "refine needs model provenance: without Observed/Unobserved labels there is no way "
            "to keep unmeasured geometry from moving the pose that inspection is judged against."
        )
    if len(observed_model_faces) == 0:
        raise ValueError(
            "refine needs at least one fully Observed model face; refusing to fit the pose to "
            "Unobserved or Inferred geometry alone."
        )

    # CAD extent only, not the model's: a whole-scene reconstruction (floaters, board skin)
    # would otherwise inflate k and the correspondence distance with geometry that isn't the part.
    voxel_size = _extent_mm(cad_vertices) * voxel_fraction
    tukey_k = voxel_size * tukey_k_factor
    correspondence_distance = voxel_size * correspondence_factor

    observed_model_surface = _sample_surface(
        model_vertices,
        observed_model_faces,
        sample_count=sample_count,
        seed=seed,
    )
    cad_surface = _sample_surface(
        cad_vertices,
        cad_faces,
        sample_count=max(sample_count, DEFAULT_SCALE_TARGET_SAMPLE_COUNT),
        seed=seed + 1,
    )

    model_to_cad_initial = np.linalg.inv(
        np.asarray(candidate.transformation, dtype=np.float64)
    )
    result = o3d.pipelines.registration.registration_icp(
        observed_model_surface,
        cad_surface,
        correspondence_distance,
        model_to_cad_initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(tukey_k)
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iterations)
        ),
    )

    correspondences = np.asarray(result.correspondence_set)
    correspondence_count = 0 if correspondences.size == 0 else len(correspondences)
    if correspondence_count < 3:
        # Open3D reports inlier_rmse = 0 when nothing matched, which would read as a perfect
        # fit. A candidate this far from the model is a failed refinement, said out loud.
        raise ValueError(
            f"refinement found {correspondence_count} Observed-model/CAD correspondences within "
            f"{correspondence_distance:.3g} mm; the candidate pose is too far from the model."
        )

    refined = np.linalg.inv(np.asarray(result.transformation, dtype=np.float64))
    rotation = refined[:3, :3]
    if not (
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-6
    ):
        raise ValueError("refinement returned a non-rigid transform.")

    inlier_rmse = float(result.inlier_rmse)
    if not math.isfinite(inlier_rmse):
        inlier_rmse = math.inf
    refined_candidate = CandidatePose(
        transformation=refined,
        fitness=float(result.fitness),
        inlier_rmse=inlier_rmse,
        source=f"refined:{candidate.source}",
        correspondence_count=correspondence_count,
    )

    # The story asks for the free-scale residual "recomputed on the refined correspondences".
    # This re-runs I-03's two-stage similarity ICP (rigid, then with scaling) from the refined
    # pose, on the same Observed-model and dense-CAD samples, rather than solving one similarity
    # on this fit's final pairs: #233's review measured that frozen nearest-neighbour pairs
    # underestimate a real scale error (+3% read as about +1%), because the pairs slide along the
    # surface instead of tracking material points. Same planar-only gate, same None when scale
    # can't be established, same D-039 sign convention.
    residual = _free_scale_residual(
        refined_candidate,
        observed_model_surface,
        cad_surface,
        max_correspondence_distance_mm=correspondence_distance,
    )
    return replace(refined_candidate, free_scale_residual=residual)


def check_ambiguity(
    candidates: Sequence[CandidatePose],
    model: Any,
    cad: Any,
    *,
    ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
    seed: int = 0,
) -> AmbiguityCheck:
    """I-05: refine every I-03 candidate and decide whether the orientation is ambiguous.

    On a near-symmetric part the feature that breaks the symmetry is often the defect, so a fit
    must not silently pick the alignment that explains it away (D-008). Every candidate is
    refined with I-04's `refine`, with the same seed, so all of them are scored on the same
    samples. Poses whose transforms agree within I-03's tolerance (2 degrees, and 1% of the CAD's
    extent with a 0.25 mm floor) are merged. Poses that are equivalent only under the CAD's
    symmetry are kept separate: for the printed part they are different poses, and the
    difference is where the defect would be.

    The refined poses are ranked as I-03 ranks candidates, fitness first, then inlier RMSE. The
    session is ambiguous when the runner-up's inlier RMSE is within `ambiguity_threshold` of the
    best's, relative to the best (D-061: 20%, superseding D-008's 10%). The comparison is
    multiplied out, so a noise-free best RMSE of zero still compares correctly. A single
    surviving pose is not ambiguous.

    A candidate that can't be refined (I-03's `ransac:0` is often an identity pose at fitness 0)
    is recorded in `failed` with its reason; if none can be refined, this raises. A model without
    provenance, or with no fully Observed face, is refused up front rather than being reported as
    a failure of every candidate.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("check_ambiguity needs at least one candidate.")
    if not (math.isfinite(ambiguity_threshold) and ambiguity_threshold >= 0):
        raise ValueError("ambiguity_threshold must be finite and non-negative.")

    _model_vertices, model_faces, model_labels = _mesh_arrays(model, name="model")
    cad_vertices, _cad_faces, _cad_labels = _mesh_arrays(cad, name="cad")
    observed_model_faces = _observed_faces(model_faces, model_labels)
    if observed_model_faces is None:
        raise ValueError(
            "check_ambiguity needs model provenance: without Observed/Unobserved labels no "
            "candidate can be refined on measured geometry."
        )
    if len(observed_model_faces) == 0:
        raise ValueError(
            "check_ambiguity needs at least one fully Observed model face; refusing to judge "
            "the orientation on Unobserved or Inferred geometry alone."
        )

    refined: list[CandidatePose] = []
    failed: list[tuple[str, str]] = []
    for candidate in candidates:
        try:
            refined.append(refine(candidate, model, cad, seed=seed))
        except ValueError as exc:
            failed.append((candidate.source, str(exc)))
    if not refined:
        details = "; ".join(f"{source}: {reason}" for source, reason in failed)
        raise ValueError(f"No candidate could be refined ({details}).")

    translation_tolerance = max(
        DEFAULT_DISTINCT_TRANSLATION_FLOOR_MM,
        _extent_mm(cad_vertices) * DEFAULT_DISTINCT_TRANSLATION_FRACTION,
    )
    # Returned in _candidate_sort_key order: fitness first, then inlier RMSE.
    unique = _deduplicate_candidates(
        refined,
        translation_tolerance_mm=translation_tolerance,
        rotation_tolerance_deg=DEFAULT_DISTINCT_ROTATION_DEG,
    )
    best = unique[0]
    runner_up = unique[1] if len(unique) > 1 else None

    ambiguous = False
    rmse_ratio: float | None = None
    if runner_up is not None:
        ambiguous = runner_up.inlier_rmse <= best.inlier_rmse * (
            1.0 + ambiguity_threshold
        )
        if best.inlier_rmse > 0.0:
            rmse_ratio = runner_up.inlier_rmse / best.inlier_rmse
        else:
            rmse_ratio = 1.0 if runner_up.inlier_rmse == 0.0 else math.inf

    return AmbiguityCheck(
        orientation_ambiguous=bool(ambiguous),
        best=best,
        runner_up=runner_up,
        rmse_ratio=rmse_ratio,
        threshold=float(ambiguity_threshold),
        refined=tuple(unique),
        failed=tuple(failed),
    )
