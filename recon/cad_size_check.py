"""R-22: measure a reconstructed part's size against its reference CAD, and against the board
itself.

R-17 found s-04 and s-05 reconstruct noticeably smaller than their CAD, more in the horizontal
plane than vertically -- too much for R-04's single scale factor to be the whole story, and too
large to be FDM print drift (proposal Section 12). R-17's numbers came from a one-off script;
this module replaces it with something committed, tested, and rerunnable.

Two independent measurements, both against the same board-frame point cloud R-04 writes:

  - ``measure_board_scale``: the board itself is an in-scene ruler (D-034's printed ChArUco
    pattern has an exact, known pitch). Every corner R-02 already detected is lifted through
    VGGT's own depth and camera (R-01), passed through R-04/R-23's transform, and fit against
    the board's true geometry with the same Umeyama solve the alignment uses. If the *board*
    reconstructs at the wrong size, the bug is upstream of the part -- in R-01's depth/intrinsics
    or the metric alignment, not in segmentation or fusion. The report retains R-22's focal-ratio
    measurement as a historical diagnostic for the camera-centre scale bug R-23 replaces.
  - ``measure_prism``: an extruded-prism CAD part (read from a simple ASCII STL -- see
    ``prism_from_cad`` for the exact shape this supports) located in the board-frame cloud and
    measured independently: height from the top-face point cluster, each footprint edge's length
    from points assigned to it by direction, not from an assumed match to the CAD's length. A
    part that is genuinely smaller is reported as smaller; this is a measurement, not a
    comparison with a tolerance -- no pass/fail and no accuracy claim is made anywhere here
    (D-003). ``build_report`` also reports each result divided by the board scale, which is what
    isolates "object-specific" error from whatever the board ruler already explains.

This module is plain NumPy/OpenCV/SciPy, importable and testable on any machine -- no GPU, no
PyTorch, same split as ``recon/vggt_runner.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recon.board_pose import MIN_CORNERS_FOR_POSE
from recon.camera_intrinsics import (
    CalibrationFailure,
    rescale_vggt_intrinsic,
    resolve_vggt_frame_scale,
)
from recon.metric_alignment import (
    align_camera_centres,
    apply_similarity_transform,
    lift_board_corner_observations,
    solve_similarity,
)
from recon.part_segmentation import DEFAULT_FOOTPRINT_MM
from recon.vggt_runner import read_ply, select_point_cloud, unproject_depth_to_world

# --- CAD: a simple extruded-prism STL reader -----------------------------------------------

_STL_VERTEX_PATTERN = re.compile(
    r"vertex\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)"
)

# Vertices that should be the same physical corner sometimes differ in the last float digit
# between facets; round before deduplicating so that doesn't register as a different vertex.
_ROUND_DECIMALS = 3


def read_ascii_stl(path: Path) -> np.ndarray:
    """Return every ``vertex`` line's (x, y, z) from an ASCII STL, as (N, 3) float64.

    Only ASCII STL is supported -- both golden-set CAD files (`fixtures/reference_cad/`) are
    ASCII, and parsing binary STL's packed triangle format is unneeded extra surface area for
    this story. Fails loudly rather than silently misreading a binary file as text.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not ASCII text; this reader only parses ASCII STL (binary STL is not "
            "supported)."
        ) from exc
    if not text.lstrip().startswith("solid"):
        raise ValueError(f"{path} does not start with 'solid'; not an ASCII STL.")
    matches = _STL_VERTEX_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"{path} has no 'vertex' lines; not an ASCII STL.")
    return np.array(
        [[float(x), float(y), float(z)] for x, y, z in matches], dtype=np.float64
    )


@dataclass(frozen=True)
class PrismCAD:
    """A vertical extruded prism: a convex footprint polygon, constant along z."""

    footprint_xy: np.ndarray  # (K, 2) mm, ordered around the hull
    height_mm: float
    edges: list[
        tuple[np.ndarray, np.ndarray]
    ]  # K consecutive (p0, p1) footprint edges, mm


def _sorted_rows(points_xy: np.ndarray) -> np.ndarray:
    """Sort (N,2) rows lexicographically, so two point *sets* can be compared regardless of
    file order (sorting each column independently, the more obvious shortcut, would scramble
    which x goes with which y)."""
    order = np.lexsort((points_xy[:, 1], points_xy[:, 0]))
    return points_xy[order]


def prism_from_cad(vertices: np.ndarray) -> PrismCAD:
    """Extract a vertical prism's footprint and height from a triangle-soup STL's vertices.

    Supports exactly the shape both golden-set CAD files are: a convex polygon extruded along
    z, base and top sharing the same (x, y) outline at two distinct z levels (the proposal's two
    parts with CAD are a box and a triangular prism; nothing in the golden set needs more than
    this). Anything else -- a non-prism mesh, a concave cross-section, more than two z levels --
    raises rather than silently measuring the wrong shape.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"vertices must be (N,3) with N>0, got {vertices.shape}")

    z_levels = np.unique(np.round(vertices[:, 2], _ROUND_DECIMALS))
    if len(z_levels) != 2:
        raise ValueError(
            f"Expected exactly 2 distinct z levels for a vertical prism (base and top), found "
            f"{len(z_levels)}: {z_levels.tolist()}. This reader only supports a simple extruded "
            "prism, not an arbitrary mesh."
        )
    # z_levels is np.unique's sorted output of exactly 2 distinct values (checked above), so
    # z_min < z_max always holds -- no separate height_mm > 0 guard is reachable to write.
    z_min, z_max = float(z_levels.min()), float(z_levels.max())
    height_mm = z_max - z_min

    def footprint_at(z_level: float) -> np.ndarray:
        at_level = vertices[np.round(vertices[:, 2], _ROUND_DECIMALS) == z_level, :2]
        return np.unique(np.round(at_level, _ROUND_DECIMALS), axis=0)

    base_xy = footprint_at(z_min)
    top_xy = footprint_at(z_max)
    if base_xy.shape != top_xy.shape or not np.allclose(
        _sorted_rows(base_xy), _sorted_rows(top_xy), atol=10**-_ROUND_DECIMALS
    ):
        raise ValueError(
            "Base and top footprints differ; this reader only supports a simple extruded prism "
            "with matching top and bottom outlines."
        )
    if len(base_xy) < 3:
        raise ValueError(
            f"Footprint has only {len(base_xy)} distinct vertices; need at least 3."
        )

    hull = cv2.convexHull(base_xy.astype(np.float32))
    footprint = hull.reshape(-1, 2).astype(np.float64)
    if len(footprint) != len(base_xy):
        raise ValueError(
            "Footprint vertices are not all on the convex hull; this reader only supports a "
            "convex prism cross-section."
        )
    edges = [
        (footprint[i], footprint[(i + 1) % len(footprint)])
        for i in range(len(footprint))
    ]
    return PrismCAD(footprint_xy=footprint, height_mm=height_mm, edges=edges)


# --- Board pitch correction (G-04) --------------------------------------------------------------


def apply_board_pitch_correction(
    points_board_frame_mm: np.ndarray,
    measured_span_across_mm: float,
    measured_span_down_mm: float,
    nominal_span_across_mm: float = DEFAULT_FOOTPRINT_MM[0],
    nominal_span_down_mm: float = DEFAULT_FOOTPRINT_MM[1],
) -> np.ndarray:
    """Correct board-frame x/y for G-06's measured board pitch, which is anisotropic (the
    printed sheet is not exactly square): +0.10% across vs +0.46% down from D-034's nominal
    20 mm squares. ``lift_board_corner_observations``'s fit target is an exact nominal grid, so
    R-04/R-23's board-frame coordinates are expressed in nominal-grid units; since the true
    board spans ``measured_span`` physical mm over that same nominal interval, each axis is
    scaled by its own ``measured / nominal`` ratio. A single scalar from the similarity fit
    can't represent two different axis ratios at once, so this stays a deterministic, scoped
    correction here rather than a change to the shared alignment in ``recon/metric_alignment.py``
    (which R-23's live alignment also depends on). z (height above the board) is untouched --
    G-06 only calipered the board's x/y plane, not its flatness.
    """
    points = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points.shape}"
        )
    corrected = points.copy()
    corrected[:, 0] *= measured_span_across_mm / nominal_span_across_mm
    corrected[:, 1] *= measured_span_down_mm / nominal_span_down_mm
    return corrected


# --- Board plane ------------------------------------------------------------------------------


def fit_board_plane(
    points_board_frame_mm: np.ndarray,
    min_points: int = 200,
    max_iterations: int = 4,
) -> tuple[float, float, float]:
    """Robustly fit z = a*x + b*y + c to the board surface near z = 0.

    Candidates are points over the printed pattern's rectangle (``DEFAULT_FOOTPRINT_MM``) within
    15 mm of z = 0 -- loose enough to admit the real board's few-millimetre tilt/offset (R-17),
    but tight enough that a part sitting on the board (height several cm for s-04/s-05) mostly
    falls outside it already. What does leak in near the part's base is then downweighted by a
    few rounds of MAD-trimmed least squares, the same style of robust fit as elsewhere in this
    codebase (``camera_intrinsics``'s reprojection handling) rather than a hard, hand-picked cut.
    """
    points = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points.shape}"
        )

    footprint_width_mm, footprint_height_mm = DEFAULT_FOOTPRINT_MM
    on_board = (
        (points[:, 0] > -5)
        & (points[:, 0] < footprint_width_mm + 5)
        & (points[:, 1] > -5)
        & (points[:, 1] < footprint_height_mm + 5)
        & (np.abs(points[:, 2]) < 15)
    )
    candidates = points[on_board]
    if len(candidates) < min_points:
        raise ValueError(
            f"Only {len(candidates)} board-region points found near z=0; need at least "
            f"{min_points} to fit a plane."
        )

    design = np.column_stack(
        [candidates[:, 0], candidates[:, 1], np.ones(len(candidates))]
    )
    heights = candidates[:, 2]
    coefficients, *_ = np.linalg.lstsq(design, heights, rcond=None)
    for _ in range(max_iterations):
        residual = heights - design @ coefficients
        mad = np.median(np.abs(residual - np.median(residual))) + 1e-9
        inliers = np.abs(residual) < 3.0 * 1.4826 * mad
        if inliers.sum() < min_points:
            break
        coefficients, *_ = np.linalg.lstsq(
            design[inliers], heights[inliers], rcond=None
        )
    return float(coefficients[0]), float(coefficients[1]), float(coefficients[2])


def compute_height_above_board(
    points_board_frame_mm: np.ndarray, board_plane: tuple[float, float, float]
) -> np.ndarray:
    """Height above the fitted board plane, in R-06's sign convention (z increases into the
    table, so height above the board is -z relative to the fitted plane)."""
    points = np.asarray(points_board_frame_mm, dtype=np.float64)
    a, b, c = board_plane
    return -(points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c))


# --- 2D rigid alignment of a point blob against a CAD polygon's edges -------------------------


def _edge_line_distances(
    points_xy: np.ndarray, edges: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Perpendicular distance from every point to every edge's *infinite* line, (N, E).

    Infinite lines, not segments: a point near a corner should not be penalised just because it
    falls slightly past where the nearest edge happens to end -- that is exactly the smear this
    module elsewhere handles by excluding a margin near each edge's own endpoints, not by
    truncating the line itself.
    """
    distances = np.empty((len(points_xy), len(edges)))
    for index, (p0, p1) in enumerate(edges):
        direction = p1 - p0
        length = np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]]) / length
        distances[:, index] = np.abs((points_xy - p0) @ normal)
    return distances


def _edge_segment_distances(
    points_xy: np.ndarray,
    edges: list[tuple[np.ndarray, np.ndarray]],
    slack_fraction: float = 0.3,
) -> np.ndarray:
    """Distance from every point to every edge's *segment*, extended by ``slack_fraction`` of
    its own length past each end, (N, E).

    Used only to choose and refine the orientation/translation in ``_fit_rigid_2d``, never for
    the final per-edge measurement in ``measure_prism`` (that uses ``_edge_line_distances``'s
    unbounded lines instead, deliberately, so a genuinely shrunk edge is not penalised for not
    reaching its CAD-length endpoints). Unbounded lines are wrong here, though: a near-square or
    near-symmetric blob, placed at the *wrong* translation, can still have most of its points
    sitting close to *some* transformed edge's infinite line even though that line's actual
    segment is nowhere near the real data -- this is exactly what let a coarse candidate shifted
    by a whole edge-length score deceptively well before this function existed. Segment distance
    (with generous slack for noise and for a part that really is somewhat smaller than the CAD)
    closes that hole.
    """
    distances = np.empty((len(points_xy), len(edges)))
    for index, (p0, p1) in enumerate(edges):
        direction = p1 - p0
        length = np.linalg.norm(direction)
        unit_direction = direction / length
        normal = np.array([-unit_direction[1], unit_direction[0]])
        relative = points_xy - p0
        projection = relative @ unit_direction
        perpendicular = np.abs(relative @ normal)
        slack = slack_fraction * length
        clamped_projection = np.clip(projection, -slack, length + slack)
        along_excess = np.abs(projection - clamped_projection)
        distances[:, index] = np.hypot(perpendicular, along_excess)
    return distances


def _polygon_centroid(vertices_xy: np.ndarray) -> np.ndarray:
    """Area centroid of an ordered polygon, with a mean fallback for a degenerate outline."""
    vertices = np.asarray(vertices_xy, dtype=np.float64)
    shifted = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * shifted[:, 1] - shifted[:, 0] * vertices[:, 1]
    twice_area = float(cross.sum())
    if abs(twice_area) < 1e-9:
        return vertices.mean(axis=0)
    return ((vertices + shifted) * cross[:, None]).sum(axis=0) / (3.0 * twice_area)


def _approximate_hull_polygon(
    blob_xy: np.ndarray, vertex_count: int
) -> np.ndarray | None:
    """Simplify the observed convex hull to the CAD polygon's vertex count.

    R-37 uses the whole outline rather than trusting whichever one noisy hull edge happens to be
    longest. The scan takes the middle simplification that has the expected vertex count so a
    result does not depend on one brittle epsilon boundary.
    """
    hull = cv2.convexHull(blob_xy.astype(np.float32))
    perimeter = float(cv2.arcLength(hull, True))
    exact: list[np.ndarray] = []
    for fraction in np.linspace(0.001, 0.08, 160):
        approx = cv2.approxPolyDP(hull, fraction * perimeter, True)
        vertices = approx.reshape(-1, 2).astype(np.float64)
        if len(vertices) == vertex_count:
            exact.append(vertices)
    if not exact:
        return None
    return exact[len(exact) // 2]


def _edge_pattern_error_mm(source_xy: np.ndarray, target_xy: np.ndarray) -> float:
    """Scale-free mismatch between two ordered polygons' edge-length sequences.

    Cyclic shifts are geometrically indistinguishable for a square, but not for a scalene
    triangle. Comparing perimeter-normalised edge lengths gives the correspondence search a
    deterministic way to keep a 30/40/50 triangle's physical walls attached to the matching
    CAD edge slots without fitting scale. The result is converted back to an mm-like penalty
    using the CAD perimeter; it is only a correspondence tie-breaker, never a size estimate.
    """
    source_next = np.roll(source_xy, -1, axis=0)
    target_next = np.roll(target_xy, -1, axis=0)
    source_lengths = np.linalg.norm(source_next - source_xy, axis=1)
    target_lengths = np.linalg.norm(target_next - target_xy, axis=1)
    source_perimeter = float(source_lengths.sum())
    target_perimeter = float(target_lengths.sum())
    if source_perimeter <= 0 or target_perimeter <= 0:
        return float("inf")
    source_fraction = source_lengths / source_perimeter
    target_fraction = target_lengths / target_perimeter
    return (
        0.5 * source_perimeter * float(np.abs(source_fraction - target_fraction).sum())
    )


def _rigid_transform_no_scale(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map one ordered polygon correspondence with rotation/reflection but no scale.

    A generic SVD/Kabsch fit is ambiguous for a perfect square because its two planar singular
    values are equal: it can preserve the centroid while losing the specific edge-to-edge
    correspondence. R-37 instead maps the first corresponding edge direction exactly and uses
    the polygons' signed winding to choose whether the perpendicular axis is reflected. The
    candidate scorer still decides which cyclic/reversed correspondence is best.
    """
    source_center = source_xy.mean(axis=0)
    target_center = target_xy.mean(axis=0)

    source_direction = source_xy[1] - source_xy[0]
    target_direction = target_xy[1] - target_xy[0]
    source_direction = source_direction / np.linalg.norm(source_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)

    source_perpendicular = np.array([-source_direction[1], source_direction[0]])
    target_perpendicular = np.array([-target_direction[1], target_direction[0]])

    source_shifted = np.roll(source_xy, -1, axis=0)
    target_shifted = np.roll(target_xy, -1, axis=0)
    source_area = float(
        (
            source_xy[:, 0] * source_shifted[:, 1]
            - source_shifted[:, 0] * source_xy[:, 1]
        ).sum()
    )
    target_area = float(
        (
            target_xy[:, 0] * target_shifted[:, 1]
            - target_shifted[:, 0] * target_xy[:, 1]
        ).sum()
    )
    handedness = 1.0 if source_area * target_area >= 0 else -1.0

    source_basis = np.column_stack((source_direction, source_perpendicular))
    target_basis = np.column_stack(
        (target_direction, handedness * target_perpendicular)
    )
    transform = target_basis @ source_basis.T
    translation = target_center - transform @ source_center
    return transform, translation


def _pose_score(
    blob_xy: np.ndarray,
    edges: list[tuple[np.ndarray, np.ndarray]],
    transform: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float]:
    """Symmetric outline score: observed-to-CAD and CAD-to-observed.

    The old one-way median let a wrong triangle pose look good when most observed points happened
    to sit near some transformed CAD line. R-37 scores a high quantile in both directions, so a
    transformed edge with no observed wall also counts against the candidate.
    """
    transformed = [
        (transform @ p0 + translation, transform @ p1 + translation) for p0, p1 in edges
    ]
    observed_to_cad = _edge_segment_distances(blob_xy, transformed).min(axis=1)

    hull = cv2.convexHull(blob_xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    hull_edges = [
        (hull[index], hull[(index + 1) % len(hull)]) for index in range(len(hull))
    ]
    sample_t = np.linspace(0.0, 1.0, 9)
    cad_samples = np.concatenate(
        [p0[None, :] + sample_t[:, None] * (p1 - p0)[None, :] for p0, p1 in transformed]
    )
    cad_to_observed = _edge_segment_distances(
        cad_samples, hull_edges, slack_fraction=0.0
    ).min(axis=1)

    high_quantile = max(
        float(np.quantile(observed_to_cad, 0.90)),
        float(np.quantile(cad_to_observed, 0.90)),
    )
    median_sum = float(np.median(observed_to_cad) + np.median(cad_to_observed))
    return high_quantile, median_sum


def _fallback_pose_search(
    edges: list[tuple[np.ndarray, np.ndarray]], blob_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic whole-angle fallback when hull simplification cannot recover the polygon."""
    cad_vertices = np.array([p0 for p0, _p1 in edges])
    cad_center = _polygon_centroid(cad_vertices)
    hull = cv2.convexHull(blob_xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    blob_center = _polygon_centroid(hull)

    best: tuple[tuple[float, float, int, float], np.ndarray, np.ndarray] | None = None
    for mirror_index, mirror in enumerate((1.0, -1.0)):
        mirror_matrix = np.array([[mirror, 0.0], [0.0, 1.0]])
        for theta in np.linspace(-math.pi, math.pi, 180, endpoint=False):
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            transform = rotation @ mirror_matrix
            translation = blob_center - transform @ cad_center
            score = _pose_score(blob_xy, edges, transform, translation)
            key = (score[0], score[1], mirror_index, theta)
            if best is None or key < best[0]:
                best = (key, transform, translation)

    assert best is not None
    return best[1], best[2]


def _fit_rigid_2d(
    edges: list[tuple[np.ndarray, np.ndarray]], blob_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[np.ndarray, np.ndarray]]]:
    """Find the no-scale 2D pose mapping the CAD outline onto observed wall points.

    R-37 replaces the old single-hull-edge seed. The convex outline is simplified to the CAD
    polygon's vertex count; every cyclic correspondence in both winding directions gets a
    no-scale orthogonal fit and a symmetric outline score. A deterministic global angle search
    is the fallback when simplification cannot recover the expected polygon.

    A small local angle search removes simplification error, then an edge-balanced translation
    refinement prevents a densely sampled long wall from outvoting a short wall. Scale is never
    fit: a genuinely undersized part must remain undersized for measure_prism.
    """
    cad_vertices = np.array([p0 for p0, _p1 in edges])
    observed_vertices = _approximate_hull_polygon(blob_xy, len(cad_vertices))

    best: tuple[tuple[float, float, float, int, int], np.ndarray, np.ndarray] | None = (
        None
    )
    if observed_vertices is not None:
        for reverse_index, observed in enumerate(
            (observed_vertices, observed_vertices[::-1])
        ):
            for shift in range(len(observed)):
                target = np.roll(observed, -shift, axis=0)
                transform, translation = _rigid_transform_no_scale(cad_vertices, target)
                score = _pose_score(blob_xy, edges, transform, translation)
                pattern_error_mm = _edge_pattern_error_mm(cad_vertices, target)
                combined = score[0] + 0.25 * pattern_error_mm
                key = (combined, score[0], score[1], reverse_index, shift)
                if best is None or key < best[0]:
                    best = (key, transform, translation)

    if best is None:
        transform, translation = _fallback_pose_search(edges, blob_xy)
    else:
        transform, translation = best[1], best[2]

    source_center = cad_vertices.mean(axis=0)
    target_center = transform @ source_center + translation
    refined: tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None = None
    for delta in np.linspace(math.radians(-3.0), math.radians(3.0), 61):
        cos_t, sin_t = math.cos(delta), math.sin(delta)
        adjustment = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        candidate_transform = adjustment @ transform
        candidate_translation = target_center - candidate_transform @ source_center
        score = _pose_score(blob_xy, edges, candidate_transform, candidate_translation)
        key = (score[0], score[1], abs(delta))
        if refined is None or key < refined[0]:
            refined = (key, candidate_transform, candidate_translation)

    assert refined is not None
    transform, translation = refined[1], refined[2]
    transformed = [
        (transform @ p0 + translation, transform @ p1 + translation) for p0, p1 in edges
    ]
    distances = _edge_line_distances(blob_xy, transformed)
    assignment = distances.argmin(axis=1)
    nearest = distances[np.arange(len(blob_xy)), assignment]
    coarse_residual = float(np.median(nearest))
    inliers = nearest < max(5.0, 4.0 * coarse_residual)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for edge_index, (p0, p1) in enumerate(transformed):
        selected = inliers & (assignment == edge_index)
        if selected.sum() < 5:
            continue
        direction = p1 - p0
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        mean_point = blob_xy[selected].mean(axis=0)
        rows.append(normal)
        rhs.append(float(normal @ (mean_point - p0)))
    if len(rows) < 2:
        raise ValueError(
            "Too few inlier points across distinct edge directions to fit the part's orientation."
        )
    delta_translation, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    translation = translation + delta_translation
    transformed = [
        (transform @ p0 + translation, transform @ p1 + translation) for p0, p1 in edges
    ]
    residual = float(np.median(_edge_line_distances(blob_xy, transformed).min(axis=1)))
    return transform, translation, residual, transformed


def _fit_wall_line(
    points_xy: np.ndarray,
    reference_edge: tuple[np.ndarray, np.ndarray],
    max_iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust orthogonal line fit to one observed wall, oriented like its matched CAD edge."""
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit a wall line.")

    reference_direction = reference_edge[1] - reference_edge[0]
    reference_direction = reference_direction / np.linalg.norm(reference_direction)
    keep = np.ones(len(points), dtype=bool)

    for _ in range(max_iterations):
        selected = points[keep]
        center = selected.mean(axis=0)
        _left, _singular_values, right_t = np.linalg.svd(
            selected - center, full_matrices=False
        )
        direction = right_t[0]
        if float(direction @ reference_direction) < 0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        residuals = (points - center) @ normal
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median))) + 1e-9
        updated = np.abs(residuals - median) < max(1e-6, 3.0 * 1.4826 * mad)
        if updated.sum() < 5 or np.array_equal(updated, keep):
            break
        keep = updated

    selected = points[keep]
    center = selected.mean(axis=0)
    _left, _singular_values, right_t = np.linalg.svd(
        selected - center, full_matrices=False
    )
    direction = right_t[0]
    if float(direction @ reference_direction) < 0:
        direction = -direction
    return center, direction


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Intersection of two fitted 2D lines represented as (point, unit direction)."""
    first_point, first_direction = first
    second_point, second_direction = second
    cross = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(cross) < 1e-6:
        raise ValueError(
            "Adjacent fitted wall lines are effectively parallel; cannot locate their corner."
        )
    delta = second_point - first_point
    first_t = (delta[0] * second_direction[1] - delta[1] * second_direction[0]) / cross
    return first_point + first_t * first_direction


def _largest_connected_blob(
    xy_mm: np.ndarray,
    cell_mm: float = 1.0,
    min_points_per_cell: int = 1,
    expected_diameter_mm: float | None = None,
) -> np.ndarray:
    """Return the most plausible connected part blob in an xy slice.

    The original implementation always kept the largest 8-connected component of occupied
    1 mm cells. Real G-04 data showed why that is too permissive: a chain of sparse stray cells
    can connect the real part to a much larger cloud, so the resulting convex hull spans most of
    the board and a 20 mm square can be measured as 70--150 mm.

    When expected_diameter_mm is supplied, inspect components at several density thresholds.
    A one-cell dilation puts sparse wall samples immediately next to a dense core back into the
    candidate, while single-point bridges disappear at the higher thresholds. Components whose
    minimum-area-rectangle diagonal is between 0.35x and 2.5x the CAD footprint diameter are
    considered physically plausible; among those, keep the one containing the most original
    points. The gate is intentionally very broad so it rejects catastrophic non-part blobs
    without forcing the measured part to equal the CAD size. If no component passes that broad
    gate, fall back to the historical largest-component rule rather than fabricating a result.
    """
    xy_mm = np.asarray(xy_mm, dtype=np.float64)
    if xy_mm.ndim != 2 or xy_mm.shape[1] != 2 or len(xy_mm) == 0:
        raise ValueError(f"xy_mm must have shape (N,2) with N>0, got {xy_mm.shape}")

    origin = xy_mm.min(axis=0)
    grid = np.floor((xy_mm - origin) / cell_mm).astype(np.int64)
    width = int(grid[:, 0].max()) + 1
    height = int(grid[:, 1].max()) + 1
    occupancy = np.zeros((height, width), dtype=np.int32)
    np.add.at(occupancy, (grid[:, 1], grid[:, 0]), 1)

    def components(threshold: int) -> list[np.ndarray]:
        binary = (occupancy >= threshold).astype(np.uint8)
        component_count, labels, _stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        masks: list[np.ndarray] = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        for label in range(1, component_count):
            core = (labels == label).astype(np.uint8)
            if int(core.sum()) == 0:
                continue
            support = cv2.dilate(core, kernel, iterations=1)
            point_mask = support[grid[:, 1], grid[:, 0]] > 0
            if int(point_mask.sum()) >= 30:
                masks.append(point_mask)
        return masks

    if expected_diameter_mm is not None and expected_diameter_mm > 0:
        plausible: list[tuple[int, np.ndarray]] = []
        start_threshold = max(1, min_points_per_cell)
        for threshold in range(start_threshold, max(start_threshold, 4) + 1):
            for point_mask in components(threshold):
                selected = xy_mm[point_mask].astype(np.float32)
                _center, (rect_width, rect_height), _angle = cv2.minAreaRect(selected)
                diameter = math.hypot(float(rect_width), float(rect_height))
                ratio = diameter / expected_diameter_mm
                if 0.35 <= ratio <= 2.5:
                    plausible.append((int(point_mask.sum()), point_mask))
        if plausible:
            plausible.sort(key=lambda item: -item[0])
            return plausible[0][1]

    binary = (occupancy >= min_points_per_cell).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count <= 1:
        raise ValueError(
            "No connected blob of points found; check the candidate region."
        )
    largest_label = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    cell_labels = labels[grid[:, 1], grid[:, 0]]
    return cell_labels == largest_label


def _top_face_connected_blob(
    xy_mm: np.ndarray,
    height_mm: np.ndarray,
    expected_height_mm: float,
    *,
    cell_mm: float = 1.0,
    min_top_support: int = 30,
) -> np.ndarray:
    """Choose the connected XY component whose own top-face peak is highest.

    R-39's phone captures exposed a failure in the historical size-first rule: a broad pile of
    low stray points can occupy more 1 mm cells than the real part, so the largest-component
    prefilter keeps the pile and discards the part before height detection even starts. Size is
    therefore the wrong signal for this pre-height choice.

    Each connected component is checked only inside the same broad CAD-relative height range
    used by measure_prism. If the historical largest component already contains a supported top
    peak, keep it exactly: R-39 must not perturb scans that already measure. Only when that old
    choice has no usable top do we choose another component, preferring the highest supported
    peak. This preserves R-37's spatial isolation without using the CAD-diameter plausibility
    gate that regressed g02_02. If no component has a supported peak, fall back to the historical
    rule so later measurement checks keep their existing failure behavior.
    """
    xy = np.asarray(xy_mm, dtype=np.float64)
    heights = np.asarray(height_mm, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError(f"xy_mm must have shape (N,2) with N>0, got {xy.shape}")
    if heights.ndim != 1 or len(heights) != len(xy):
        raise ValueError(
            "height_mm must have shape (N,) and match xy_mm; "
            f"got {heights.shape} for {len(xy)} xy points"
        )
    if expected_height_mm <= 0:
        raise ValueError("expected_height_mm must be positive")

    origin = xy.min(axis=0)
    grid = np.floor((xy - origin) / cell_mm).astype(np.int64)
    width = int(grid[:, 0].max()) + 1
    height = int(grid[:, 1].max()) + 1
    occupancy = np.zeros((height, width), dtype=np.uint8)
    occupancy[grid[:, 1], grid[:, 0]] = 1

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        occupancy, connectivity=8
    )
    if component_count <= 1:
        raise ValueError(
            "No connected blob of points found; check the candidate region."
        )

    # Preserve the historical choice whenever it already has a usable top-face peak. R-39 is a
    # recovery path for the #356 failure, not permission to change which component wins on scans
    # that already measure successfully.
    historical_label = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    point_labels = labels[grid[:, 1], grid[:, 0]]
    search_lo = 0.5 * expected_height_mm
    search_hi = 1.3 * expected_height_mm + 5.0
    bins = np.arange(search_lo, search_hi + 0.5, 0.5)

    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for label in range(1, component_count):
        point_mask = point_labels == label
        component_heights = heights[point_mask]
        in_range = (component_heights >= search_lo) & (component_heights <= search_hi)
        if int(in_range.sum()) == 0:
            continue

        counts, bin_edges = np.histogram(component_heights[in_range], bins=bins)
        if len(counts) == 0 or int(counts.max()) == 0:
            continue
        peak_index = int(counts.argmax())
        peak_center = float(bin_edges[peak_index] + 0.25)
        top_support = int((np.abs(component_heights - peak_center) < 1.75).sum())
        if top_support < min_top_support:
            continue

        if label == historical_label:
            return point_mask
        candidates.append((peak_center, top_support, int(point_mask.sum()), point_mask))

    if not candidates:
        return _largest_connected_blob(xy, cell_mm=cell_mm)

    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    return candidates[0][3]


# --- Part measurement ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeMeasurement:
    cad_length_mm: float
    measured_length_mm: float
    ratio: float
    inlier_count: int
    perpendicular_spread_mm: (
        float  # p90 - p10 of points' distance from the fitted edge line
    )


# R-32's sign convention, stated once and copied into every report that carries a wall profile.
# Each offset is measured along the wall's *inward* normal, found from the fitted footprint itself
# (towards its centre), never from the edge's direction: _fit_rigid_2d tries both handednesses,
# so a mirrored fit reverses the edges' winding, and a direction-based normal would then flip the
# sign of every offset on a run-by-run basis. A positive offset is surface inside the CAD wall (the
# part reads small there); a positive lean is a wall moving inward as it rises. Checked, mirrored
# and not, by test_wall_offset_sign_is_positive_inside_the_cad_wall.
WALL_OFFSET_CONVENTION = (
    "offset_mm is the surface's signed distance from the CAD wall: positive means inside the CAD "
    "footprint (the part reads small there), negative means outside. lean_mm and lean_deg are "
    "positive when the wall moves inward as height increases."
)


@dataclass(frozen=True)
class WallHeightBin:
    height_mm: float
    offset_mm: float
    thickness_mm: float
    point_count: int


@dataclass(frozen=True)
class WallProfileMeasurement:
    edge_index: int
    cad_length_mm: float
    point_count: int
    height_min_mm: float
    height_max_mm: float
    lean_mm: float
    lean_deg: float
    thickness_median_mm: float
    bins: list[WallHeightBin]
    # Where the wall is, in board-frame mm, and which way is inside. A mirror-symmetric footprint
    # (s-05's square) can fit with any of its edges on any physical wall, so edge_index alone
    # doesn't name the same wall in two runs; these do.
    midpoint_xy_mm: tuple[float, float]
    inward_normal_xy: tuple[float, float]


@dataclass(frozen=True)
class PrismMeasurement:
    height_mm: float
    cad_height_mm: float
    height_ratio: float
    alignment_residual_mm: float
    rotation_deg: float
    edges: list[EdgeMeasurement]
    wall_profiles: list[WallProfileMeasurement]
    transformed_edges: list[tuple[np.ndarray, np.ndarray]]


def measure_wall_profiles(
    points_board_frame_mm: np.ndarray,
    transformed_edges: list[tuple[np.ndarray, np.ndarray]],
    board_plane: tuple[float, float, float],
    part_height_mm: float,
    alignment_residual_mm: float,
    *,
    bin_count: int = 5,
    min_points_per_bin: int = 5,
) -> list[WallProfileMeasurement]:
    """Measure each flat wall's signed offset, lean and thickness as a function of height.

    R-32 needs to distinguish a depth error already present in the raw reconstruction from a
    shape error introduced later by fusion. The CAD pose is fixed by the same rigid 2-D fit used
    for the size measurement. For each CAD edge, wall points are assigned to their nearest edge
    line, split into equal height bands from 10% to 90% of the measured part height, and reported
    as the median signed normal offset plus the p10-p90 spread (thickness) in each band.

    lean_mm is the fitted change in signed wall offset from the first valid height bin to the
    last; lean_deg is atan(d offset / d height). Signs follow WALL_OFFSET_CONVENTION: a positive
    offset is inside the CAD wall, and a positive lean moves inward going up. No tolerance or accuracy judgement is made.
    An edge with too little surface evidence is omitted instead of fabricating a zero measurement.
    """
    points = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points.shape}"
        )
    if part_height_mm <= 0:
        raise ValueError("part_height_mm must be positive.")
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    if min_points_per_bin < 1:
        raise ValueError("min_points_per_bin must be positive.")

    heights = compute_height_above_board(points, board_plane)
    wall_band = (heights >= 0.10 * part_height_mm) & (heights <= 0.90 * part_height_mm)
    points = points[wall_band]
    heights = heights[wall_band]
    if len(points) == 0:
        return []

    points_xy = points[:, :2]
    distances = _edge_line_distances(points_xy, transformed_edges)
    assignment = distances.argmin(axis=1)
    distance_limit_mm = max(8.0, 4.0 * alignment_residual_mm)
    height_edges = np.linspace(
        0.10 * part_height_mm, 0.90 * part_height_mm, bin_count + 1
    )

    footprint_centre = np.mean(
        [np.asarray(p0, dtype=np.float64) for p0, _ in transformed_edges], axis=0
    )
    profiles: list[WallProfileMeasurement] = []
    for edge_index, (p0, p1) in enumerate(transformed_edges):
        direction = p1 - p0
        cad_length_mm = float(np.linalg.norm(direction))
        unit_direction = direction / cad_length_mm
        normal = np.array([-unit_direction[1], unit_direction[0]])
        midpoint = (p0 + p1) / 2.0
        if float(np.dot(footprint_centre - midpoint, normal)) < 0.0:
            # Point inward whatever the edges' winding (see WALL_OFFSET_CONVENTION).
            normal = -normal
        selected = (assignment == edge_index) & (
            distances[:, edge_index] <= distance_limit_mm
        )
        if int(np.count_nonzero(selected)) < 2 * min_points_per_bin:
            continue

        edge_heights = heights[selected]
        signed_offsets = (points_xy[selected] - p0) @ normal
        raw_bins: list[tuple[float, float, int, np.ndarray, np.ndarray]] = []
        for bin_index in range(bin_count):
            lo = height_edges[bin_index]
            hi = height_edges[bin_index + 1]
            if bin_index == bin_count - 1:
                in_bin = (edge_heights >= lo) & (edge_heights <= hi)
            else:
                in_bin = (edge_heights >= lo) & (edge_heights < hi)
            count = int(np.count_nonzero(in_bin))
            if count < min_points_per_bin:
                continue
            heights_in_bin = edge_heights[in_bin]
            offsets_in_bin = signed_offsets[in_bin]
            raw_bins.append(
                (
                    float(np.median(heights_in_bin)),
                    float(np.median(offsets_in_bin)),
                    count,
                    heights_in_bin,
                    offsets_in_bin,
                )
            )

        if len(raw_bins) < 2:
            continue
        bin_heights = np.asarray([item[0] for item in raw_bins], dtype=np.float64)
        bin_offsets = np.asarray([item[1] for item in raw_bins], dtype=np.float64)
        slope, _intercept = np.polyfit(bin_heights, bin_offsets, 1)

        bins: list[WallHeightBin] = []
        for height_mm, _offset_mm, count, heights_in_bin, offsets_in_bin in raw_bins:
            # Remove the fitted lean within this height band before measuring thickness.
            # Otherwise a genuinely leaning but razor-thin wall looks artificially thick simply
            # because its centre moves sideways from the bottom to the top of the bin.
            detrended_offsets = offsets_in_bin - slope * (heights_in_bin - height_mm)
            p10, p90 = np.percentile(detrended_offsets, [10, 90])
            bins.append(
                WallHeightBin(
                    height_mm=height_mm,
                    offset_mm=float(np.median(detrended_offsets)),
                    thickness_mm=float(p90 - p10),
                    point_count=count,
                )
            )

        height_span = float(bin_heights[-1] - bin_heights[0])
        profiles.append(
            WallProfileMeasurement(
                edge_index=edge_index,
                cad_length_mm=cad_length_mm,
                point_count=int(np.count_nonzero(selected)),
                height_min_mm=float(bin_heights[0]),
                height_max_mm=float(bin_heights[-1]),
                lean_mm=float(slope * height_span),
                lean_deg=float(math.degrees(math.atan(float(slope)))),
                thickness_median_mm=float(
                    np.median([item.thickness_mm for item in bins])
                ),
                bins=bins,
                midpoint_xy_mm=(float(midpoint[0]), float(midpoint[1])),
                inward_normal_xy=(float(normal[0]), float(normal[1])),
            )
        )
    return profiles


def _wall_profiles_report(
    profiles: list[WallProfileMeasurement],
) -> list[dict[str, Any]]:
    return [
        {
            "edge_index": profile.edge_index,
            "midpoint_xy_mm": list(profile.midpoint_xy_mm),
            "inward_normal_xy": list(profile.inward_normal_xy),
            "cad_length_mm": profile.cad_length_mm,
            "point_count": profile.point_count,
            "height_min_mm": profile.height_min_mm,
            "height_max_mm": profile.height_max_mm,
            "lean_mm": profile.lean_mm,
            "lean_deg": profile.lean_deg,
            "thickness_median_mm": profile.thickness_median_mm,
            "height_bins": [
                {
                    "height_mm": item.height_mm,
                    "offset_mm": item.offset_mm,
                    "thickness_mm": item.thickness_mm,
                    "point_count": item.point_count,
                }
                for item in profile.bins
            ],
        }
        for profile in profiles
    ]


def observed_glb_labels(glb_path: Path) -> str:
    """Which labels a fused GLB carries, named in the report so the two can never be confused.

    R-32 measures the fused surface on the *served* labels, `uncertainty/mesh_uncertainty.glb`:
    R-10 has already downgraded noisy-depth vertices to Unobserved there, so it is the surface
    that can pass or fail a part. `provenance/mesh_provenance.glb` holds R-09's labels from before
    that downgrade, and so keeps the noisiest wall vertices; a run on it is labelled as such
    rather than refused, so the two can be compared deliberately.
    """
    name = Path(glb_path).name
    if name == "mesh_uncertainty.glb":
        return "served: R-10's labels, after the sigma downgrade (uncertainty/mesh_uncertainty.glb)"
    if name == "mesh_provenance.glb":
        return (
            "not served: R-09's labels, before R-10's sigma downgrade "
            "(provenance/mesh_provenance.glb)"
        )
    return f"unrecognised file name {name!r}: labels as given"


def load_observed_surface_vertex_indices(
    provenance_glb: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertex_indices, vertex_positions) for vertices participating in fully Observed
    faces of a provenance GLB.

    Imports provenance lazily so the normal raw-cloud CAD check keeps its lightweight import
    surface. R-09's conservative rule is preserved: a face counts only when all three vertices
    are Observed. The indices are into the GLB's own vertex array -- the same array order
    ``sigma_mm.npz`` (R-10) is indexed by, so callers that need each vertex's own sigma (G-07's
    per-vertex reliability check, for one) can pair them directly.
    """
    from recon import mesh_cleanup, provenance

    vertices, faces, colors = mesh_cleanup.read_glb(Path(provenance_glb))
    if colors is None:
        raise ValueError(f"{provenance_glb} has no provenance vertex colors.")
    vertex_labels = provenance.colors_to_labels(colors)
    face_labels = provenance.label_faces(faces, vertex_labels)
    observed_faces = faces[face_labels == provenance.LABEL_OBSERVED]
    if len(observed_faces) == 0:
        raise ValueError(f"{provenance_glb} has no fully Observed faces.")
    observed_vertex_indices = np.unique(observed_faces.reshape(-1))
    return observed_vertex_indices, vertices[observed_vertex_indices].astype(np.float64)


def load_observed_surface_points(provenance_glb: Path) -> np.ndarray:
    """Return vertices participating in fully Observed faces of a provenance GLB.

    A thin wrapper around ``load_observed_surface_vertex_indices`` for callers that only need
    positions, not indices.
    """
    _indices, positions = load_observed_surface_vertex_indices(provenance_glb)
    return positions


def measure_prism(
    points_board_frame_mm: np.ndarray,
    prism: PrismCAD,
    board_plane: tuple[float, float, float],
) -> PrismMeasurement:
    """Measure a located part's height and per-edge footprint lengths, independently of the
    CAD's assumed size (see ``_fit_rigid_2d``'s docstring for why no scale is fit).
    """
    points = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points.shape}"
        )

    height = compute_height_above_board(points, board_plane)
    footprint_width_mm, footprint_height_mm = DEFAULT_FOOTPRINT_MM
    region = (
        (points[:, 0] > -10)
        & (points[:, 0] < footprint_width_mm + 10)
        & (points[:, 1] > -10)
        & (points[:, 1] < footprint_height_mm + 10)
        & (height > 1.5)
        & (height < max(prism.height_mm * 1.6, 40.0))
    )
    if region.sum() < 200:
        raise ValueError(
            "Too few candidate points above the board to measure the part."
        )

    candidate_points = points[region]
    candidate_xy = candidate_points[:, :2]
    candidate_height = height[region]

    # R-39: choose the spatial component by its own supported top-face height, not by XY area.
    # Phone-style captures can contain a broad low pile of stray points that occupies more cells
    # than the part; choosing the largest component first discards the real top before the
    # height histogram runs. This intentionally does NOT use the CAD-diameter plausibility gate
    # because R-37 showed that gate is too aggressive here on g02_02.
    height_blob_mask = _top_face_connected_blob(
        candidate_xy, candidate_height, prism.height_mm
    )
    candidate_points = candidate_points[height_blob_mask]
    candidate_xy = candidate_xy[height_blob_mask]
    candidate_height = candidate_height[height_blob_mask]

    footprint_rect = cv2.minAreaRect(prism.footprint_xy.astype(np.float32))
    expected_diameter_mm = math.hypot(
        float(footprint_rect[1][0]), float(footprint_rect[1][1])
    )

    # The CAD-size plausibility gate is still intentionally deferred until the mid-height
    # cross-section below. Height uses only the historical connectivity filter above.
    # --- height: the densest height bin in the expected range is the top face --------------
    # Without confidence filtering, low-confidence stray points (R-17) pile up near the board
    # far more densely than the part itself, so searching the *whole* height range for the
    # densest bin finds that pile, not the top face. The CAD's own height bounds the search
    # instead, generously (0.5x-1.3x+5mm) so an under- or over-sized reconstruction still falls
    # inside it -- that generosity is the point, since this function must not assume the answer.
    search_lo = max(float(candidate_height.min()), 0.5 * prism.height_mm)
    search_hi = min(float(candidate_height.max()), 1.3 * prism.height_mm + 5.0)
    if search_hi <= search_lo:
        raise ValueError(
            "No candidate points fall in the expected top-face height range."
        )
    in_search_range = (candidate_height >= search_lo) & (candidate_height <= search_hi)
    bins = np.arange(search_lo, search_hi + 0.5, 0.5)
    counts, bin_edges = np.histogram(candidate_height[in_search_range], bins=bins)
    if counts.max() == 0:
        raise ValueError(
            "No candidate points fall in the expected top-face height range."
        )
    peak_bin_start = bin_edges[int(counts.argmax())]
    top_face = np.abs(candidate_height - (peak_bin_start + 0.25)) < 1.75
    if top_face.sum() < 30:
        raise ValueError("Too few points on the detected top face to measure height.")
    height_mm = float(np.median(candidate_height[top_face]))

    # --- cross-section: a mid-height band avoids both the noisy base and the top-face points ---
    band = (candidate_height > 0.3 * height_mm) & (candidate_height < 0.7 * height_mm)
    if band.sum() < 30:
        raise ValueError(
            "Too few points in the mid-height band to measure the cross-section."
        )
    band_xy = candidate_xy[band]
    band_mask = _largest_connected_blob(
        band_xy, expected_diameter_mm=expected_diameter_mm
    )
    band_xy = band_xy[band_mask]
    if len(band_xy) < 30:
        raise ValueError(
            "Too few points left in the mid-height band after blob filtering."
        )

    transform, _translation, residual, transformed_edges = _fit_rigid_2d(
        prism.edges, band_xy
    )
    distances = _edge_line_distances(band_xy, transformed_edges)
    assignment = distances.argmin(axis=1)

    edge_point_masks: list[np.ndarray] = []
    fitted_wall_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for edge_index, transformed_edge in enumerate(transformed_edges):
        cad_length_mm = float(
            np.linalg.norm(prism.edges[edge_index][1] - prism.edges[edge_index][0])
        )
        on_this_edge = (assignment == edge_index) & (
            distances[:, edge_index] < max(5.0, 3.0 * residual)
        )
        if on_this_edge.sum() < 5:
            raise ValueError(
                f"Too few points assigned to edge {edge_index} (length {cad_length_mm:.1f} mm "
                "in the CAD) to measure it."
            )
        edge_point_masks.append(on_this_edge)
        fitted_wall_lines.append(
            _fit_wall_line(band_xy[on_this_edge], transformed_edge)
        )

    # R-37: a caliper measures physical corners, not percentiles of whichever point samples
    # happened to land on a wall. Corner i is the intersection of wall i-1 with wall i.
    fitted_corners = [
        _line_intersection(
            fitted_wall_lines[(edge_index - 1) % len(fitted_wall_lines)],
            fitted_wall_lines[edge_index],
        )
        for edge_index in range(len(fitted_wall_lines))
    ]

    edge_measurements: list[EdgeMeasurement] = []
    for edge_index, on_this_edge in enumerate(edge_point_masks):
        cad_length_mm = float(
            np.linalg.norm(prism.edges[edge_index][1] - prism.edges[edge_index][0])
        )
        measured_length_mm = float(
            np.linalg.norm(
                fitted_corners[(edge_index + 1) % len(fitted_corners)]
                - fitted_corners[edge_index]
            )
        )
        line_point, unit_direction = fitted_wall_lines[edge_index]
        normal = np.array([-unit_direction[1], unit_direction[0]])
        perpendicular = (band_xy[on_this_edge] - line_point) @ normal
        offset_mm = float(np.median(perpendicular))
        spread_low, spread_high = np.percentile(perpendicular - offset_mm, [10, 90])
        edge_measurements.append(
            EdgeMeasurement(
                cad_length_mm=cad_length_mm,
                measured_length_mm=measured_length_mm,
                ratio=measured_length_mm / cad_length_mm,
                inlier_count=int(on_this_edge.sum()),
                perpendicular_spread_mm=float(spread_high - spread_low),
            )
        )

    wall_profiles = measure_wall_profiles(
        candidate_points,
        transformed_edges,
        board_plane,
        height_mm,
        residual,
    )

    return PrismMeasurement(
        height_mm=height_mm,
        cad_height_mm=prism.height_mm,
        height_ratio=height_mm / prism.height_mm,
        alignment_residual_mm=residual,
        rotation_deg=float(math.degrees(math.atan2(transform[1, 0], transform[0, 0]))),
        edges=edge_measurements,
        wall_profiles=wall_profiles,
        transformed_edges=transformed_edges,
    )


# --- Caliper truth (G-04) ------------------------------------------------------------------------


def match_caliper_edge_truth(
    prism: PrismCAD,
    edge_caliper_features: list[dict[str, Any]],
    length_atol_mm: float = 0.1,
    conflict_atol_mm: float = 0.05,
) -> list[float | None]:
    """Match each of ``prism.edges`` to caliper truth by CAD edge *length*, and return a
    ``mean_mm`` per ``prism.edges`` index (``None`` where no caliper feature's length matches).

    Matching is by length, not by a single edge's identity, because a caliper reading may
    represent every edge of the same nominal length at once -- e.g. a rectangle's two long
    sides measured once (``fixtures/caliper_g-0*.json``'s own "this is the length of the two
    long sides" readings) applies to both of that rectangle's long edges, not just the one
    whose ``cad_vertices`` happen to be given. ``length_atol_mm`` only needs to separate
    genuinely distinct CAD lengths (tens of mm apart in every specimen here), not to resolve
    measurement noise.

    This also makes a genuine ambiguity visible rather than silently guessed: if more than one
    caliper feature's length matches a prism edge's length, and those features' ``mean_mm``
    values differ by more than ``conflict_atol_mm`` (tight -- tighter than the caliper's own
    0.1 mm resolution, since a real conflict here is always much larger in practice: g02_01's
    square reads 19.9 vs 19.81 mm, g02_04's near-regular pentagon and g02_05's near-regular
    hexagon read five/six distinct values each, all several caliper-resolution-steps apart),
    every edge in that length class is left unmatched (``None``) rather than assigned an
    arbitrary one of the conflicting readings -- these are several distinct physical-side
    readings for edges the CAD treats as identical, with no record of which numbered/lettered
    side is which reconstructed edge. Features that agree within ``conflict_atol_mm`` are
    averaged, not treated as a conflict.
    """
    edge_features = [
        feature
        for feature in edge_caliper_features
        if feature.get("kind") == "edge" and feature.get("cad_vertices") is not None
    ]
    feature_lengths = [
        float(
            np.linalg.norm(
                np.asarray(feature["cad_vertices"])[1]
                - np.asarray(feature["cad_vertices"])[0]
            )
        )
        for feature in edge_features
    ]

    truth_mm: list[float | None] = []
    for p0, p1 in prism.edges:
        edge_length = float(np.linalg.norm(p1 - p0))
        matching_means = [
            float(feature["mean_mm"])
            for feature, feature_length in zip(edge_features, feature_lengths)
            if abs(feature_length - edge_length) <= length_atol_mm
        ]
        if not matching_means:
            truth_mm.append(None)
        elif max(matching_means) - min(matching_means) <= conflict_atol_mm:
            truth_mm.append(sum(matching_means) / len(matching_means))
        else:
            truth_mm.append(None)
    return truth_mm


def caliper_edge_errors(
    measurement: PrismMeasurement, caliper_truth_mm: list[float | None]
) -> list[dict[str, Any]]:
    """Per edge with known caliper truth: bias (``caliper_error_mm`` = measured - truth) kept
    apart from the existing ``perpendicular_spread_mm`` (R-22's Finding 2). Edges with no caliper
    truth are omitted, not reported as zero error.
    """
    if len(caliper_truth_mm) != len(measurement.edges):
        raise ValueError(
            f"caliper_truth_mm has {len(caliper_truth_mm)} entries, expected "
            f"{len(measurement.edges)} (one per prism edge)."
        )
    results: list[dict[str, Any]] = []
    for edge_index, (edge, truth_mm) in enumerate(
        zip(measurement.edges, caliper_truth_mm)
    ):
        if truth_mm is None:
            continue
        results.append(
            {
                "edge_index": edge_index,
                "caliper_truth_mm": truth_mm,
                "measured_length_mm": edge.measured_length_mm,
                "caliper_error_mm": edge.measured_length_mm - truth_mm,
                "caliper_ratio": edge.measured_length_mm / truth_mm,
                "perpendicular_spread_mm": edge.perpendicular_spread_mm,
            }
        )
    return results


def per_vertex_height_reliability(
    served_glb: Path,
    sigma_mm: np.ndarray,
    board_plane: tuple[float, float, float],
    caliper_height_truth_mm: float,
    *,
    top_region_percentile: float = 90.0,
) -> list[tuple[float, float]]:
    """Pair each near-top-face Observed vertex's own R-10 sigma with its *signed*
    height-above-board error against an independent caliper height measurement (G-07).

    ``served_glb`` should be R-10's post-downgrade labels (``mesh_uncertainty.glb``) -- the
    labels that actually gate pass/fail in production, per ``observed_glb_labels``'s own
    precedent (R-32). ``sigma_mm`` is R-10's per-vertex array (``uncertainty/sigma_mm.npz``),
    indexed by the same vertex order as ``served_glb``'s own vertex array. ``board_plane`` must
    be the plane R-07's own TSDF fusion already fit and re-anchored while building this mesh
    (``tsdf_fusion.json``'s ``board_surface`` -- see ``scripts/g07_reliability_curve.py``'s
    ``_read_board_plane``), **not** re-fit from this mesh's own vertices: the served mesh only
    contains the segmented-out part (R-07 discards the board layer before meshing), so it has few
    or no real board-level points to fit a plane from at all. A first version of this function
    recommended re-fitting with ``fit_board_plane`` on the mesh's own observed points anyway, and
    it silently produced a badly tilted/offset plane (D-077) -- caught because the resulting
    per-vertex errors came out roughly 10x G-04's own established height floor, implausible as
    real noise.

    Vertices are selected by the top ``100 - top_region_percentile`` percent of this mesh's own
    *observed* height distribution, not ``measure_prism``'s density-peak-bin search: a served GLB
    is R-08's decimated, Observed-only vertex set -- a few hundred to low thousands of vertices
    for a whole part (confirmed on s-04: 2075 total mesh vertices, 1611 Observed) -- orders of
    magnitude sparser than the dense raw point clouds (``points.ply``, hundreds of thousands of
    points) the density-peak search was tuned against; on a vertex set this sparse, a density
    peak is not reliably findable (confirmed empirically: it raised "too few points on the
    detected top face" on s-04's own served GLB). A percentile cut on the mesh's own observed
    heights needs no density peak and adapts to whatever height range this specimen's mesh
    actually reached, rather than assuming the CAD's own height is reached at all -- for a convex
    prism, points this close to the maximum observed height are overwhelmingly on the top face,
    not the walls (which only reach this close to the top at their very rim).

    Returns a list of ``(sigma_mm, signed_error_mm)`` pairs, one per near-top-face Observed
    vertex with a defined sigma -- a NaN sigma (fewer than 2 observations, R-10's own convention)
    is dropped. The error is signed, not absolute: each specimen sits at its own near-constant
    *offset* from caliper truth (a whole-part effect sigma cannot see -- that is
    ``max(sigma, sigma_floor)``'s job, D-073, specifically sigma_floor's), and a caller comparing
    sigma against per-vertex *scatter* needs that offset removed first, which an absolute value
    would make impossible to recover (D-077).
    """
    vertex_indices, vertex_positions = load_observed_surface_vertex_indices(served_glb)
    height_mm = compute_height_above_board(vertex_positions, board_plane)
    threshold_mm = np.percentile(height_mm, top_region_percentile)
    top_region_mask = height_mm >= threshold_mm

    signed_error_mm = height_mm[top_region_mask] - caliper_height_truth_mm
    sigma_at_vertices = np.asarray(sigma_mm, dtype=np.float64)[
        vertex_indices[top_region_mask]
    ]

    finite = np.isfinite(sigma_at_vertices)
    return list(
        zip(sigma_at_vertices[finite].tolist(), signed_error_mm[finite].tolist())
    )


# --- Board scale: the printed pattern as an in-scene ruler -------------------------------------


@dataclass(frozen=True)
class _RunBundle:
    frame_names: list[str]
    poses_extrinsic: np.ndarray  # (S,3,4) camera-from-world, R-01's arbitrary scale
    poses_intrinsic: np.ndarray  # (S,3,3) VGGT K, in processed-image pixels
    depth: np.ndarray  # (S,H,W) R-01's arbitrary-scale depth
    depth_conf: np.ndarray  # (S,H,W)
    processed_size_hw: tuple[int, int]
    preprocess_geometry: list[dict[str, Any]] | None
    alignment_scale: float
    alignment_rotation: np.ndarray
    alignment_translation_mm: np.ndarray
    board_frame_names: list[str]
    board_image_size_hw: np.ndarray  # (S2,2) full-resolution sensor pixels
    board_intrinsic: np.ndarray  # (S2,3,3) the K R-02 actually posed each frame with
    board_extrinsic: np.ndarray  # (S2,3,4) camera-from-board in millimetres
    board_pose_found: np.ndarray  # (S2,) R-02 trusted pose mask
    corner_frame_index: np.ndarray  # (C,) index into board_frame_names
    corner_id: (
        np.ndarray
    )  # (C,) ChArUco corner id, matches make_board().getChessboardCorners()
    corner_px: np.ndarray  # (C,2) full-resolution sensor pixels


def load_bundle(recon_run_dir: Path, board_pose_dir: Path) -> _RunBundle:
    """Load everything R-01/R-02/R-04 already wrote that this module needs."""
    recon_run_dir = Path(recon_run_dir)
    board_pose_dir = Path(board_pose_dir)
    run_json_path = recon_run_dir / "run.json"
    alignment_path = recon_run_dir / "metric_alignment.json"
    poses_path = recon_run_dir / "poses.npz"
    depth_path = recon_run_dir / "depth.npz"
    board_npz_path = board_pose_dir / "board_poses.npz"
    for path in (run_json_path, alignment_path, poses_path, depth_path, board_npz_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; R-22 needs R-01, R-02 and R-04 output already on disk."
            )

    run_info = json.loads(run_json_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    with np.load(poses_path) as poses:
        frame_names = [str(name) for name in poses["frame_names"]]
        poses_extrinsic = poses["extrinsic"].astype(np.float64)
        poses_intrinsic = poses["intrinsic"].astype(np.float64)
    with np.load(depth_path) as depth_data:
        depth = depth_data["depth"].astype(np.float64)
        depth_conf = depth_data["depth_conf"].astype(np.float64)
    with np.load(board_npz_path) as board:
        board_frame_names = [str(name) for name in board["frame_names"]]
        board_image_size_hw = board["image_size_hw"]
        board_intrinsic = board["intrinsic"].astype(np.float64)
        board_extrinsic = board["extrinsic"].astype(np.float64)
        board_pose_found = board["pose_found"].astype(bool)
        corner_frame_index = board["corner_frame_index"]
        corner_id = board["corner_id"]
        corner_px = board["corner_px"]

    processed_size_hw = tuple(
        int(value) for value in run_info["processed_image_size_hw"]
    )
    return _RunBundle(
        frame_names=frame_names,
        poses_extrinsic=poses_extrinsic,
        poses_intrinsic=poses_intrinsic,
        depth=depth,
        depth_conf=depth_conf,
        processed_size_hw=processed_size_hw,
        preprocess_geometry=run_info.get("preprocess_geometry"),
        alignment_scale=float(alignment["scale"]),
        alignment_rotation=np.array(alignment["rotation"], dtype=np.float64),
        alignment_translation_mm=np.array(
            alignment["translation_mm"], dtype=np.float64
        ),
        board_frame_names=board_frame_names,
        board_image_size_hw=board_image_size_hw,
        board_intrinsic=board_intrinsic,
        board_extrinsic=board_extrinsic,
        board_pose_found=board_pose_found,
        corner_frame_index=corner_frame_index,
        corner_id=corner_id,
        corner_px=corner_px,
    )


def reconstruct_confidence_filtered_points(
    bundle: _RunBundle, min_confidence: float
) -> np.ndarray:
    """Rebuild R-01's point cloud directly from ``depth.npz``/``poses.npz``, confidence-filtered.

    ``points.ply``'s rows no longer line up positionally with ``depth_conf.reshape(-1)`` the
    moment R-01's own ``select_point_cloud`` dropped even one non-finite point -- common near an
    edge or occlusion in a real capture -- because ``points.ply`` only has the *kept* rows, not
    one row per (S,H,W) pixel. Indexing ``depth_conf`` by raw position against ``points.ply``
    therefore breaks on exactly the real captures this function's callers need it for. Rebuilding
    points and their confidence together from the same raw depth array, via the identical
    ``unproject_depth_to_world`` + ``select_point_cloud`` R-01 itself uses, keeps a point and its
    confidence value inseparable, so there is no positional assumption left to break -- and it
    needs only ``depth.npz``/``poses.npz``, both stored losslessly regardless of what threshold
    R-01 happened to use when it wrote ``points.ply``.
    """
    world_points = unproject_depth_to_world(
        bundle.depth, bundle.poses_extrinsic, bundle.poses_intrinsic
    )
    dummy_colors = np.zeros(world_points.shape, dtype=np.uint8)
    points, _colors = select_point_cloud(
        world_points, dummy_colors, bundle.depth_conf, min_confidence
    )
    return points.astype(np.float64)


@dataclass(frozen=True)
class BoardScaleResult:
    scale: float  # reconstructed board size / true (nominal) board size
    rotation: np.ndarray
    translation_mm: np.ndarray
    rms_residual_mm: float
    observation_count: int
    frame_count: int
    per_frame_scale_median: float
    per_frame_scale_min: float
    per_frame_scale_max: float
    focal_ratio_board_over_vggt: float


def _measure_board_scale_after_transform(
    bundle: _RunBundle,
    scale_to_board: float,
    rotation_to_board: np.ndarray,
    translation_to_board_mm: np.ndarray,
) -> BoardScaleResult:
    """Measure nominal-vs-reconstructed board size after one supplied metric transform."""
    recon_points, nominal_points, frame_of_point = lift_board_corner_observations(
        bundle.frame_names,
        bundle.poses_extrinsic,
        bundle.poses_intrinsic,
        bundle.depth,
        bundle.processed_size_hw,
        bundle.board_frame_names,
        bundle.board_image_size_hw,
        bundle.corner_frame_index,
        bundle.corner_id,
        bundle.corner_px,
        bundle.preprocess_geometry,
        bundle.board_pose_found,
    )
    recon_points_board = apply_similarity_transform(
        recon_points,
        scale_to_board,
        rotation_to_board,
        translation_to_board_mm,
    )
    scale, rotation, translation = solve_similarity(nominal_points, recon_points_board)
    predicted = apply_similarity_transform(nominal_points, scale, rotation, translation)
    residuals = np.linalg.norm(predicted - recon_points_board, axis=1)

    per_frame_scales: list[float] = []
    matched_frames = sorted(set(frame_of_point.tolist()))
    for frame_index in matched_frames:
        selected = frame_of_point == frame_index
        if selected.sum() < MIN_CORNERS_FOR_POSE:
            continue
        frame_scale, _rotation, _translation = solve_similarity(
            nominal_points[selected], recon_points_board[selected]
        )
        per_frame_scales.append(frame_scale)
    if not per_frame_scales:
        raise ValueError(
            "No single frame had enough corner observations for a per-frame scale check."
        )

    board_index_by_name = {
        str(name): index for index, name in enumerate(bundle.board_frame_names)
    }
    vggt_focals: list[float] = []
    board_focals: list[float] = []
    for frame_index in matched_frames:
        frame_name = str(bundle.frame_names[frame_index])
        board_frame_index = board_index_by_name.get(frame_name)
        if board_frame_index is None or not bundle.board_pose_found[board_frame_index]:
            continue
        full_height, full_width = bundle.board_image_size_hw[board_frame_index]
        scale_x, scale_y, crop_left_px, crop_top_px = resolve_vggt_frame_scale(
            frame_name,
            bundle.preprocess_geometry,
            bundle.processed_size_hw,
            (int(full_height), int(full_width)),
        )
        rescaled = rescale_vggt_intrinsic(
            bundle.poses_intrinsic[frame_index],
            scale_x,
            scale_y,
            crop_left_px,
            crop_top_px,
        )
        vggt_focals.extend([rescaled[0, 0], rescaled[1, 1]])
        board_focals.extend(
            [
                bundle.board_intrinsic[board_frame_index, 0, 0],
                bundle.board_intrinsic[board_frame_index, 1, 1],
            ]
        )

    vggt_focals_arr = np.asarray(vggt_focals, dtype=np.float64)
    board_focals_arr = np.asarray(board_focals, dtype=np.float64)
    valid = np.isfinite(vggt_focals_arr) & np.isfinite(board_focals_arr)
    if valid.sum() < 2:
        raise ValueError(
            "Not enough finite focal-length pairs to compute the focal ratio."
        )
    focal_ratio = float(
        np.median(board_focals_arr[valid]) / np.median(vggt_focals_arr[valid])
    )

    return BoardScaleResult(
        scale=float(scale),
        rotation=rotation,
        translation_mm=translation,
        rms_residual_mm=float(np.sqrt(np.mean(residuals**2))),
        observation_count=len(recon_points),
        frame_count=len(per_frame_scales),
        per_frame_scale_median=float(np.median(per_frame_scales)),
        per_frame_scale_min=float(np.min(per_frame_scales)),
        per_frame_scale_max=float(np.max(per_frame_scales)),
        focal_ratio_board_over_vggt=focal_ratio,
    )


def measure_legacy_camera_centre_board_scale(bundle: _RunBundle) -> BoardScaleResult:
    """Preserve R-22 independent diagnostic using the pre-R-23 camera-centre transform."""
    legacy = align_camera_centres(
        bundle.frame_names,
        bundle.poses_extrinsic,
        bundle.board_frame_names,
        bundle.board_extrinsic,
        bundle.board_pose_found,
    )
    return _measure_board_scale_after_transform(
        bundle,
        legacy.scale,
        legacy.rotation,
        legacy.translation,
    )


def measure_board_scale(bundle: _RunBundle) -> BoardScaleResult:
    """Measure the reconstructed board after R-23 current board-anchored transform.

    This is the explicit R-23 acceptance measurement: after anchoring on the board, report what
    board scale remains. It is expected to be near 1.0, but is not an independent detector of
    uniform scale error. measure_legacy_camera_centre_board_scale preserves the independent
    pre-R-23 diagnostic for that purpose.
    """
    return _measure_board_scale_after_transform(
        bundle,
        bundle.alignment_scale,
        bundle.alignment_rotation,
        bundle.alignment_translation_mm,
    )


# --- Orchestration and CLI ----------------------------------------------------------------------


def build_report(
    recon_run_dir: Path,
    board_pose_dir: Path,
    cad_path: Path | None = None,
    min_confidence: float = 0.0,
    observed_glb: Path | None = None,
    caliper_features: list[dict[str, Any]] | None = None,
    board_pitch_mm: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run the board-ruler check, and the CAD check if a CAD path is given.

    Every figure here is a measurement, never a pass/fail or a tolerance comparison (D-003) --
    that judgement, once a real tolerance exists, is R-16's job, not this one's.

    ``caliper_features``/``board_pitch_mm`` are G-04's extension: when both are given (requires
    ``cad_path`` too, since caliper truth is matched onto the CAD-derived prism's edges),
    ``board_pitch_mm`` (measured_span_across_mm, measured_span_down_mm) corrects the board-frame
    cloud for G-06's measured, anisotropic board pitch before fitting, and the report gains
    ``prism.caliper`` -- each edge's measured length against caliper truth (G-03/G-06) rather
    than CAD, bias kept apart from the existing per-edge spread (R-22 Finding 2).
    """
    bundle = load_bundle(recon_run_dir, board_pose_dir)
    board_scale = measure_board_scale(bundle)
    legacy_board_scale = measure_legacy_camera_centre_board_scale(bundle)
    report: dict[str, Any] = {
        "story": "R-22",
        "recon_run_dir": str(recon_run_dir),
        "board_pose_dir": str(board_pose_dir),
        "board_scale": {
            "scale": board_scale.scale,
            "rotation": board_scale.rotation.tolist(),
            "translation_mm": board_scale.translation_mm.tolist(),
            "rms_residual_mm": board_scale.rms_residual_mm,
            "observation_count": board_scale.observation_count,
            "frame_count": board_scale.frame_count,
            "per_frame_scale_median": board_scale.per_frame_scale_median,
            "per_frame_scale_min": board_scale.per_frame_scale_min,
            "per_frame_scale_max": board_scale.per_frame_scale_max,
            "focal_ratio_board_over_vggt": board_scale.focal_ratio_board_over_vggt,
            "note": (
                "In-scene ruler: nominal ChArUco corners fit against R-01 depth/camera lifted "
                "through R-04's transform. scale < 1 means the reconstruction is smaller than "
                "true size. focal_ratio_board_over_vggt is the median ratio between the K R-02's "
                "board poses were solved with and VGGT's own predicted K (rescaled to the same "
                "pixel frame). R-22 used that ratio to diagnose the old camera-centre anchor; "
                "R-23 no longer uses it to set metric scale. Not a tolerance or accuracy claim "
                "(D-003)."
            ),
        },
        "legacy_camera_centre_board_scale": {
            "scale": legacy_board_scale.scale,
            "rms_residual_mm": legacy_board_scale.rms_residual_mm,
            "observation_count": legacy_board_scale.observation_count,
            "frame_count": legacy_board_scale.frame_count,
            "per_frame_scale_median": legacy_board_scale.per_frame_scale_median,
            "per_frame_scale_min": legacy_board_scale.per_frame_scale_min,
            "per_frame_scale_max": legacy_board_scale.per_frame_scale_max,
            "focal_ratio_board_over_vggt": legacy_board_scale.focal_ratio_board_over_vggt,
            "note": (
                "Independent historical R-22 diagnostic: the same reconstructed board is "
                "measured after the legacy camera-centre transform, not R-23's board anchor. "
                "This preserves visibility into the focal-length-driven scale error that "
                "motivated R-23."
            ),
        },
    }

    if cad_path is not None:
        cad_path = Path(cad_path)
        prism = prism_from_cad(read_ascii_stl(cad_path))
        if min_confidence > 0:
            points = reconstruct_confidence_filtered_points(bundle, min_confidence)
        else:
            points, _colors = read_ply(Path(recon_run_dir) / "points.ply")
            points = points.astype(np.float64)
        points_board = apply_similarity_transform(
            points,
            bundle.alignment_scale,
            bundle.alignment_rotation,
            bundle.alignment_translation_mm,
        )
        if board_pitch_mm is not None:
            points_board = apply_board_pitch_correction(
                points_board, board_pitch_mm[0], board_pitch_mm[1]
            )
        board_plane = fit_board_plane(points_board)
        measurement = measure_prism(points_board, prism, board_plane)
        report["prism"] = {
            "cad_path": str(cad_path),
            "min_confidence": min_confidence,
            "board_pitch_mm": (
                {"across": board_pitch_mm[0], "down": board_pitch_mm[1]}
                if board_pitch_mm is not None
                else None
            ),
            "board_plane": {
                "a": board_plane[0],
                "b": board_plane[1],
                "c": board_plane[2],
            },
            "height_mm": measurement.height_mm,
            "cad_height_mm": measurement.cad_height_mm,
            "height_ratio": measurement.height_ratio,
            "height_ratio_over_board_scale": measurement.height_ratio
            / board_scale.scale,
            "alignment_residual_mm": measurement.alignment_residual_mm,
            "rotation_deg": measurement.rotation_deg,
            "edges": [
                {
                    "cad_length_mm": edge.cad_length_mm,
                    "measured_length_mm": edge.measured_length_mm,
                    "ratio": edge.ratio,
                    "ratio_over_board_scale": edge.ratio / board_scale.scale,
                    "inlier_count": edge.inlier_count,
                    "perpendicular_spread_mm": edge.perpendicular_spread_mm,
                }
                for edge in measurement.edges
            ],
            "raw_wall_profiles": _wall_profiles_report(measurement.wall_profiles),
            "wall_offset_convention": WALL_OFFSET_CONVENTION,
            "note": (
                "Measurements only; no tolerance or pass/fail claim is made here (D-003) -- "
                "that is R-16's job once a real tolerance exists. *_ratio_over_board_scale "
                "divides out the board-ruler's measured scale error, isolating whatever error "
                "is specific to this part rather than shared with the whole scene. R-32 wall "
                "profiles report signed CAD-wall offset by height, lean and p10-p90 thickness."
            ),
        }

        if caliper_features is not None:
            caliper_truth_mm = match_caliper_edge_truth(prism, caliper_features)
            height_features = [
                f
                for f in caliper_features
                if f.get("kind") == "height" and f.get("mean_mm") is not None
            ]
            height_caliper_mm = (
                float(height_features[0]["mean_mm"])
                if len(height_features) == 1
                else None
            )
            report["prism"]["caliper"] = {
                "edges": caliper_edge_errors(measurement, caliper_truth_mm),
                "height": (
                    {
                        "caliper_truth_mm": height_caliper_mm,
                        "measured_height_mm": measurement.height_mm,
                        "caliper_error_mm": measurement.height_mm - height_caliper_mm,
                    }
                    if height_caliper_mm is not None
                    else None
                ),
                "note": (
                    "G-04: each edge's measured length against caliper truth (G-03/G-06), not "
                    "CAD -- caliper_error_mm is the edge's bias, perpendicular_spread_mm (same "
                    "value as the CAD-relative edges above) is its spread, kept apart per R-22 "
                    "Finding 2. Edges with no matching caliper feature are omitted, not zeroed. "
                    "height has no such ambiguity (there is only one height per specimen), so "
                    "it is reported even for specimens whose edges are all excluded."
                ),
            }

        if observed_glb is not None:
            observed_points = load_observed_surface_points(Path(observed_glb))
            observed_profiles = measure_wall_profiles(
                observed_points,
                measurement.transformed_edges,
                board_plane,
                measurement.height_mm,
                measurement.alignment_residual_mm,
            )
            report["prism"]["fused_observed"] = {
                "observed_glb": str(observed_glb),
                "labels": observed_glb_labels(Path(observed_glb)),
                "wall_offset_convention": WALL_OFFSET_CONVENTION,
                "wall_profiles": _wall_profiles_report(observed_profiles),
                "note": (
                    "Only vertices participating in fully Observed provenance faces are used. "
                    "The same CAD wall pose fitted on the raw cloud is reused here, so raw and "
                    "fused offsets are directly comparable and a lean introduced by fusion is "
                    "not absorbed by a second independent alignment."
                ),
            }

    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write the JSON report. Refuses to overwrite, so two runs can never be mixed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Pick a new --out or delete it deliberately; overwriting "
            "silently would let results from two runs get mixed."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure reconstructed-vs-CAD and reconstructed-vs-board size (R-22)."
    )
    parser.add_argument(
        "recon_run_dir",
        type=Path,
        help="R-01/R-04 output folder, e.g. fixtures/data/recon_runs/s-04",
    )
    parser.add_argument(
        "board_pose_dir",
        type=Path,
        help="R-02 output folder, e.g. fixtures/data/board_poses/s-04",
    )
    parser.add_argument(
        "--cad",
        type=Path,
        default=None,
        help="reference CAD STL (ASCII), e.g. fixtures/reference_cad/s04_....stl; omit to run "
        "the board-ruler check only",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop points.ply points whose depth confidence is below this before the CAD check "
        "(default keeps all; does not affect the board-ruler check)",
    )
    parser.add_argument(
        "--observed-glb",
        type=Path,
        default=None,
        help="optional fused, provenance-coloured GLB: repeat R-32's wall lean/thickness "
        "measurement on its fully Observed surface. Use the run's served "
        "uncertainty/mesh_uncertainty.glb (R-10's labels); the report names which labels it got",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    out = args.out or args.recon_run_dir / "cad_size_check.json"
    if args.observed_glb is not None and args.cad is None:
        parser.error(
            "--observed-glb requires --cad so the fused wall profiles have a CAD wall pose."
        )
    try:
        report = build_report(
            args.recon_run_dir,
            args.board_pose_dir,
            args.cad,
            args.min_confidence,
            args.observed_glb,
        )
        write_report(out, report)
    except (FileNotFoundError, FileExistsError, ValueError, CalibrationFailure) as exc:
        print(f"R-22 CAD SIZE CHECK: FAIL - {exc}", file=sys.stderr)
        return 1

    board_scale = report["board_scale"]
    message = (
        f"R-22 CAD SIZE CHECK: PASS - board scale {board_scale['scale']:.4f} "
        f"(focal ratio {board_scale['focal_ratio_board_over_vggt']:.4f})"
    )
    if "prism" in report:
        message += f", height ratio {report['prism']['height_ratio']:.4f}"
    print(message)
    print(f"Outputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
