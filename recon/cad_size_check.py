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
from recon.camera_intrinsics import CalibrationFailure, rescale_vggt_intrinsic
from recon.metric_alignment import (
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


def _longest_hull_edges(
    blob_xy: np.ndarray, max_candidates: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The blob's convex-hull edges, longest first, capped at ``max_candidates``.

    A part's cross-section band, apart from noise, outlines a convex polygon that is literally
    the part's hull -- so its longest hull edges are the most trustworthy direction candidates to
    match against the CAD's own edges below.
    """
    hull = cv2.convexHull(blob_xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        raise ValueError("Point blob's convex hull has fewer than 2 vertices.")
    edges = [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]
    edges.sort(key=lambda edge: -float(np.linalg.norm(edge[1] - edge[0])))
    return edges[:max_candidates]


def _fit_rigid_2d(
    edges: list[tuple[np.ndarray, np.ndarray]], blob_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[np.ndarray, np.ndarray]]]:
    """Find the 2D rotation+mirror and translation mapping the CAD's edges onto the point blob.

    No scale: the whole point of this module is to measure the part's size independently of the
    CAD's assumed size, so letting this fit absorb a scale factor would hide exactly the error
    being investigated.

    Coarse orientation comes from directly matching each CAD edge's direction against each of the
    point blob's longest convex-hull edges (``_longest_hull_edges``) -- not from comparing
    ``cv2.minAreaRect`` of the two, which was tried first and is unreliable for a triangle: the
    *minimum-area* enclosing rectangle of a near-right triangle can align with its hypotenuse
    rather than its legs, and which of the two wins is sensitive to exactly how the noisy
    reconstructed points happen to be distributed -- a rectangle's own minAreaRect has no such
    ambiguity, but this module has to work for both shapes. Each (hull edge, CAD edge) pairing is
    tried both directions along the edge and both chiralities (mirrored or not), since neither the
    blob's point order nor its handedness relative to the CAD is known in advance; each candidate
    seeds its translation by aligning that one matched edge's midpoint, then is scored by median
    point-to-nearest-edge *segment* distance over the whole blob (``_edge_segment_distances``,
    not the unbounded ``_edge_line_distances`` -- see that function's docstring for why: a wrong
    candidate for a near-square or near-symmetric blob can otherwise score deceptively well).
    Translation is then refined by one linear least-squares solve against the winning candidate's
    resulting point-to-edge assignment (rotation itself is not re-optimised further).
    """
    hull_edges = _longest_hull_edges(blob_xy)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for hull_p0, hull_p1 in hull_edges:
        hull_angle = math.atan2(hull_p1[1] - hull_p0[1], hull_p1[0] - hull_p0[0])
        hull_midpoint = (hull_p0 + hull_p1) / 2.0
        for cad_p0, cad_p1 in edges:
            cad_angle = math.atan2(cad_p1[1] - cad_p0[1], cad_p1[0] - cad_p0[0])
            cad_midpoint = (cad_p0 + cad_p1) / 2.0
            for direction_flip in (0.0, math.pi):
                theta = hull_angle - cad_angle + direction_flip
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                for mirror in (1.0, -1.0):
                    transform = rotation @ np.array([[mirror, 0.0], [0.0, 1.0]])
                    translation = hull_midpoint - transform @ cad_midpoint
                    transformed = [
                        (transform @ p0 + translation, transform @ p1 + translation)
                        for p0, p1 in edges
                    ]
                    score = float(
                        np.median(
                            _edge_segment_distances(blob_xy, transformed).min(axis=1)
                        )
                    )
                    if best is None or score < best[0]:
                        best = (score, transform, translation)

    # From here on, the *rotation* (ballpark-correct now) is not re-chosen, only translation is
    # refined -- so this goes back to unbounded line distance, the same as measure_prism's own
    # final per-edge pass, so a genuinely shrunk edge's points are not excluded from refining the
    # fit just for falling short of the CAD's own edge length.
    _, transform, translation = best
    transformed = [
        (transform @ p0 + translation, transform @ p1 + translation) for p0, p1 in edges
    ]
    distances = _edge_line_distances(blob_xy, transformed)
    assignment = distances.argmin(axis=1)
    nearest = distances[np.arange(len(blob_xy)), assignment]
    coarse_residual = float(np.median(nearest))
    inliers = nearest < max(5.0, 4.0 * coarse_residual)

    # One row per *edge* (its inliers' mean offset), not one row per point: a long, densely
    # sampled edge (a wall seen by many frames) would otherwise outvote a short or sparsely
    # sampled one by orders of magnitude in an unweighted per-point solve, pulling the whole pose
    # toward fitting the big edge well at a short edge's expense -- exactly backwards when one
    # edge is genuinely more shrunk than the others, which this module exists to be able to see.
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for edge_index, (p0, _p1) in enumerate(transformed):
        selected = inliers & (assignment == edge_index)
        if selected.sum() < 5:
            continue
        direction = transformed[edge_index][1] - p0
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


def _largest_connected_blob(
    xy_mm: np.ndarray, cell_mm: float = 1.0, min_points_per_cell: int = 1
) -> np.ndarray:
    """Boolean mask keeping only the points in the largest 8-connected cluster of occupied grid
    cells -- the same "largest connected cluster of occupied voxels" idea R-17 used to separate
    a part from stray points, applied here in 2D to an xy slice rather than 3D to the whole cloud.

    The default keeps any occupied cell, not just a densely-populated one: the noise rejection
    here comes from keeping only the *largest* component, which already discards an isolated
    stray cluster regardless of its own per-cell density. Requiring several points per cell was
    tried first and is fragile exactly where it matters most -- a thin line of wall points (an
    edge seen nearly edge-on, or a narrow mid-height band) can have many occupied cells with only
    one or two points each by chance, and a stricter threshold fragments that single wall into
    several small components instead of one connected edge.
    """
    xy_mm = np.asarray(xy_mm, dtype=np.float64)
    origin = xy_mm.min(axis=0)
    grid = np.floor((xy_mm - origin) / cell_mm).astype(np.int64)
    width = int(grid[:, 0].max()) + 1
    height = int(grid[:, 1].max()) + 1
    occupancy = np.zeros((height, width), dtype=np.int32)
    np.add.at(occupancy, (grid[:, 1], grid[:, 0]), 1)
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


@dataclass(frozen=True)
class PrismMeasurement:
    height_mm: float
    cad_height_mm: float
    height_ratio: float
    alignment_residual_mm: float
    rotation_deg: float
    edges: list[EdgeMeasurement]


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

    candidate_xy = points[region, :2]
    candidate_height = height[region]
    blob_mask = _largest_connected_blob(candidate_xy)
    candidate_xy = candidate_xy[blob_mask]
    candidate_height = candidate_height[blob_mask]

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
    band_mask = _largest_connected_blob(band_xy)
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

    edge_measurements: list[EdgeMeasurement] = []
    for edge_index, (p0, p1) in enumerate(transformed_edges):
        cad_length_mm = float(
            np.linalg.norm(prism.edges[edge_index][1] - prism.edges[edge_index][0])
        )
        direction = p1 - p0
        unit_direction = direction / np.linalg.norm(direction)
        normal = np.array([-unit_direction[1], unit_direction[0]])
        # Correspondence by nearest-edge-line only -- no separate margin or corner-ambiguity cut
        # near the endpoints. A margin sized relative to the CAD's edge length was tried first
        # and silently caps how short a genuinely undersized edge can ever measure (it bounds the
        # *window* a shrunk edge's points must fit inside, independent of their real extent); a
        # geometric "close to two edges at once" corner test was tried next and over-excludes a
        # triangle's acute-angled edges, where a third edge's extended line can pass within a few
        # millimetres of points nowhere near the real corner. The 2nd/98th-percentile extent below
        # already discounts the few points that do fall near a corner.
        on_this_edge = (assignment == edge_index) & (
            distances[:, edge_index] < max(5.0, 3.0 * residual)
        )
        if on_this_edge.sum() < 5:
            raise ValueError(
                f"Too few points assigned to edge {edge_index} (length {cad_length_mm:.1f} mm "
                "in the CAD) to measure it."
            )
        projection = (band_xy[on_this_edge] - p0) @ unit_direction
        perpendicular = (band_xy[on_this_edge] - p0) @ normal
        offset_mm = float(np.median(perpendicular))
        low, high = np.percentile(projection, [2, 98])
        measured_length_mm = float(high - low)
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

    return PrismMeasurement(
        height_mm=height_mm,
        cad_height_mm=prism.height_mm,
        height_ratio=height_mm / prism.height_mm,
        alignment_residual_mm=residual,
        rotation_deg=float(math.degrees(math.atan2(transform[1, 0], transform[0, 0]))),
        edges=edge_measurements,
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
    alignment_scale: float
    alignment_rotation: np.ndarray
    alignment_translation_mm: np.ndarray
    board_frame_names: list[str]
    board_image_size_hw: np.ndarray  # (S2,2) full-resolution sensor pixels
    board_intrinsic: np.ndarray  # (S2,3,3) the K R-02 actually posed each frame with
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
        alignment_scale=float(alignment["scale"]),
        alignment_rotation=np.array(alignment["rotation"], dtype=np.float64),
        alignment_translation_mm=np.array(
            alignment["translation_mm"], dtype=np.float64
        ),
        board_frame_names=board_frame_names,
        board_image_size_hw=board_image_size_hw,
        board_intrinsic=board_intrinsic,
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


def measure_board_scale(bundle: _RunBundle) -> BoardScaleResult:
    """Measure the reconstructed board after the current metric transform.

    R-23 shares the lifting primitive with metric alignment itself.  This function deliberately
    remains a *diagnostic*: it applies the already-written alignment, then asks how large the
    reconstructed board is relative to its exact nominal geometry.  A correct R-23 output should
    therefore report a scale near 1.0 without this function correcting anything itself.
    """
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
    )

    recon_points_board = apply_similarity_transform(
        recon_points,
        bundle.alignment_scale,
        bundle.alignment_rotation,
        bundle.alignment_translation_mm,
    )
    # Keep R-22's historical convention: solve nominal -> reconstructed, so scale < 1 means
    # the board (and therefore the reconstructed scene) is smaller than its true printed size.
    scale, rotation, translation = solve_similarity(nominal_points, recon_points_board)
    predicted = scale * (nominal_points @ rotation.T) + translation
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
        if board_frame_index is None:
            continue
        full_height, full_width = bundle.board_image_size_hw[board_frame_index]
        rescaled = rescale_vggt_intrinsic(
            bundle.poses_intrinsic[frame_index],
            bundle.processed_size_hw,
            (int(full_height), int(full_width)),
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


# --- Orchestration and CLI ----------------------------------------------------------------------


def build_report(
    recon_run_dir: Path,
    board_pose_dir: Path,
    cad_path: Path | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Run the board-ruler check, and the CAD check if a CAD path is given.

    Every figure here is a measurement, never a pass/fail or a tolerance comparison (D-003) --
    that judgement, once a real tolerance exists, is R-16's job, not this one's.
    """
    bundle = load_bundle(recon_run_dir, board_pose_dir)
    board_scale = measure_board_scale(bundle)
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
        board_plane = fit_board_plane(points_board)
        measurement = measure_prism(points_board, prism, board_plane)
        report["prism"] = {
            "cad_path": str(cad_path),
            "min_confidence": min_confidence,
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
            "note": (
                "Measurements only; no tolerance or pass/fail claim is made here (D-003) -- "
                "that is R-16's job once a real tolerance exists. *_ratio_over_board_scale "
                "divides out the board-ruler's measured scale error, isolating whatever error "
                "is specific to this part rather than shared with the whole scene."
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
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out = args.out or args.recon_run_dir / "cad_size_check.json"
    try:
        report = build_report(
            args.recon_run_dir, args.board_pose_dir, args.cad, args.min_confidence
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
