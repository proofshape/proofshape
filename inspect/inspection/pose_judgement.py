"""I-15: judge a CAD pose against references that don't come from the pose search.

There is no ground-truth pose for any real capture, and several of `main`'s poses are in doubt,
so I-15 (and R-44's gate) judge a pose against two independent references, defined in
`stories/I/I-15.md` ("How a pose is judged") before anything was compared:

* a **photo reading**: which face is down, which way a named side of the part faces on the
  board, and where the part's centre sits. Coarse (read to about 5-10 degrees and 10 mm), but it
  can't inherit any error from registration;
* the **model's own walls**: the direction of the straight walls in a horizontal slice of the
  reconstructed model, against the same slice of the posed CAD. No fitting of the CAD to the
  model is involved, so this too is independent of the search, and it is fine enough to see the
  10-16 degree turns that the photo reading's 20 degree margin can't.

The margins below are I-15's agreed comparison margins (#440), not inspection tolerances. They
decide whether a pose counts as right in a written check; nothing in the served verdict reads
them.

This module only measures and judges. It never changes a pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d

# I-15's agreed margins (#440). See the story for where each comes from.
GROSS_YAW_MARGIN_DEG = 20.0
WALL_YAW_MARGIN_DEG = 5.0
CENTRE_MARGIN_CAP_MM = 15.0
SAME_POSE_MARGIN_MM = 2.0

# No rotational symmetry about the board normal: a full turn is the only one that maps the CAD
# onto itself.
NO_SYMMETRY_DEG = 360.0

# Below this aligned fraction, the wall offset is too noisy to use as a fine-yaw reference, and
# judge_pose falls back to the photo reading's 20 degrees. 0.65 is a cut in the gap between
# g02_01 base (0.58-0.61, the only part below) and g02_01 R (0.75, the next lowest): an
# empirical gap in the current ten-part sample, not a calibrated noise boundary.
WALL_ALIGNED_FLOOR = 0.65

# How far either side of zero the wall comparison looks. It is a fine-yaw check: gross turns are
# the photo reading's job. The search is narrowed further to half the CAD's own wall-direction
# period (see _direction_period_deg), because directions are lines, compared modulo 180 degrees:
# a pentagon's five walls then repeat every 36 degrees, so beyond 18 a turn reads as its alias.
DEFAULT_WALL_SEARCH_DEG = 45.0
# Each outline point's wall direction is read from a straight-line fit to its neighbours within
# this radius, measured in space rather than along the outline. Along-the-outline measures fail on
# a reconstructed slice: jitter makes the outline zigzag, so its arc length overstates the wall
# and a "2 mm chord" spans a fraction of a millimetre. 2.5 mm is short beside a 20 mm wall and
# long beside sub-millimetre jitter.
WALL_NEIGHBOURHOOD_MM = 2.5
# Outline points are thinned to one per cell this size first, so a densely meshed stretch of
# outline doesn't outvote a sparsely meshed one: each millimetre of wall counts about the same.
WALL_THINNING_MM = 0.5
WALL_HISTOGRAM_BIN_DEG = 0.25
WALL_HISTOGRAM_SIGMA_DEG = 1.0
# A model wall counts as lined up with a CAD wall when its direction is this close after the
# best offset is applied. Used only for the "how straight are these walls" diagnostic.
WALL_ALIGNED_WITHIN_DEG = 3.0


@dataclass(frozen=True)
class PhotoReading:
    """One capture's placement, read from a photograph by its owner, in the board frame.

    named_side_faces_deg: the direction the named side's outward normal points, as an angle in
        the board's xy plane (degrees, counter-clockwise from +x).
    centre_xy_mm: where the part's centre sits on the board.
    face_matches: whether the photo shows the part on the face the CAD placement assumes. None
        where the CAD's faces don't differ (a box flipped onto its opposite face is the same
        surface, and a photo can't tell those faces apart), so face isn't judged.
    """

    named_side_faces_deg: float
    centre_xy_mm: tuple[float, float]
    face_matches: bool | None


# The board frame is OpenCV's ChArUco convention (recon/board_pose.py). Checked on 2026-11-23
# by projecting the frame's axes into an s-04 golden photo (IMG_4093) with its saved board pose:
# with the printed caption nearest you and reading normally (G-06's view), the origin is the
# pattern's far-left corner, +x runs left to right along the caption, and +y runs from the far
# edge toward the caption. z points into the table, so seen from above an increasing board
# angle turns clockwise.
BOARD_SQUARE_MM = 20.0
BOARD_PATTERN_DEPTH_SQUARES = 6.0
_SHEET_EDGE_ANGLE_DEG = {"right": 0.0, "caption": 90.0, "left": 180.0, "far": 270.0}


def reading_from_sheet(
    *,
    faces: str,
    turned_deg: float,
    turned_clockwise: bool,
    squares_from_left: float,
    squares_from_caption: float,
    face_matches: bool | None,
) -> PhotoReading:
    """Convert one row of the I-15 placement sheet into a board-frame PhotoReading.

    The sheet (`runs/i15-placement-sheet.md`) records, in G-06's words: which board edge the
    named side faces (caption, far, left or right, left and right as the caption reads); how far
    it is turned from square to that edge and which way, seen from above; and the centre in
    squares from the pattern's left edge and from its caption edge.
    """
    edge = faces.strip().lower()
    if edge not in _SHEET_EDGE_ANGLE_DEG:
        raise ValueError(
            f"faces must be one of {sorted(_SHEET_EDGE_ANGLE_DEG)}, got {faces!r}."
        )
    if not (math.isfinite(turned_deg) and turned_deg >= 0.0):
        raise ValueError(
            "turned_deg must be a non-negative angle; give the way separately."
        )
    # Seen from above, clockwise is the increasing board angle (see the frame note above).
    signed_turn = turned_deg if turned_clockwise else -turned_deg
    faces_deg = (_SHEET_EDGE_ANGLE_DEG[edge] + signed_turn) % 360.0
    centre_x = squares_from_left * BOARD_SQUARE_MM
    centre_y = (BOARD_PATTERN_DEPTH_SQUARES - squares_from_caption) * BOARD_SQUARE_MM
    return PhotoReading(
        named_side_faces_deg=float(faces_deg),
        centre_xy_mm=(float(centre_x), float(centre_y)),
        face_matches=face_matches,
    )


@dataclass(frozen=True)
class PoseJudgement:
    """Whether a pose is right, with each test's number so the record can show why."""

    right: bool
    face_ok: bool | None
    gross_yaw_error_deg: float
    wall_yaw_error_deg: float | None
    centre_error_mm: float
    centre_margin_mm: float
    reasons: tuple[str, ...]


def symmetric_yaw_difference_deg(
    a_deg: float, b_deg: float, period_deg: float
) -> float:
    """Smallest unsigned angle between two yaws, modulo the CAD's symmetry period.

    A square's pose turned by 90 degrees is the same surface, so it must not read as 90 degrees
    off. period_deg is 360 for a part with no symmetry about the board normal.
    """
    if not (math.isfinite(period_deg) and 0.0 < period_deg <= 360.0):
        raise ValueError("period_deg must be in (0, 360].")
    delta = (a_deg - b_deg) % period_deg
    return float(min(delta, period_deg - delta))


def pose_rotation_of_xy_direction(pose: np.ndarray, direction_xy: np.ndarray) -> float:
    """Board-frame angle (degrees) of a CAD-frame xy direction after the pose's rotation.

    Rotating the vector, rather than reading a yaw off the matrix, keeps this right through
    CAD_TO_BOARD_FLIP (which mirrors y and z) and through refine's full 6-DoF poses.
    """
    pose = np.asarray(pose, dtype=np.float64)
    vector = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    if np.linalg.norm(vector) == 0.0:
        raise ValueError("direction_xy must be non-zero.")
    rotated = pose[:3, :3] @ vector
    return float(math.degrees(math.atan2(rotated[1], rotated[0])) % 360.0)


def posed_centre_xy(cad_vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Centre of the posed CAD's xy bounding box.

    The bounding-box centre, not the vertex mean: a CAD's vertex density is uneven (a notch or a
    hole adds vertices), and the photo reading is of where the part's middle sits.
    """
    posed = _transform(cad_vertices, pose)
    lower = posed[:, :2].min(axis=0)
    upper = posed[:, :2].max(axis=0)
    return (lower + upper) / 2.0


def footprint_min_width_mm(cad_vertices: np.ndarray, step_deg: float = 0.5) -> float:
    """Smallest width of the CAD's top-view footprint, over every direction.

    Measured over directions rather than on the CAD's own x and y axes, so a footprint that is
    drawn turned in its file still gets its true narrowest width.
    """
    xy = np.asarray(cad_vertices, dtype=np.float64)[:, :2]
    smallest = math.inf
    angle = 0.0
    while angle < 180.0:
        theta = math.radians(angle)
        direction = np.array([math.cos(theta), math.sin(theta)])
        projected = xy @ direction
        width = float(projected.max() - projected.min())
        smallest = min(smallest, width)
        angle += step_deg
    return smallest


def centre_margin_mm(
    cad_vertices: np.ndarray, cap_mm: float = CENTRE_MARGIN_CAP_MM
) -> float:
    """The smaller of cap_mm and half the CAD's smallest footprint width (#440).

    A flat 15 mm is three quarters of `g02_01`'s 20 mm width, so a pose that slid onto stray
    surface beside a small part would still count as right. Half the narrowest width means the
    posed part can't have moved off its own footprint and still pass.
    """
    return float(min(cap_mm, footprint_min_width_mm(cad_vertices) / 2.0))


def surface_gap_mm(
    cad_vertices: np.ndarray,
    cad_faces: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    *,
    sample_count: int = 20_000,
    seed: int = 0,
) -> float:
    """Largest distance from one posed CAD surface to the other, both ways.

    This is I-15's "same pose" measure. It is taken surface to surface, not pose to pose, so a
    pose turned by one of the CAD's own symmetries gives a gap near zero with no list of
    symmetries needed. Points are sampled on each posed surface and measured against the other
    posed *mesh* exactly (ray-casting distance), so the result doesn't depend on sample spacing
    beyond where the largest gap happens to be sampled.
    """
    vertices = np.asarray(cad_vertices, dtype=np.float64)
    faces = np.asarray(cad_faces, dtype=np.int64)
    if sample_count < 100:
        raise ValueError("sample_count must be at least 100.")

    largest = 0.0
    for source_pose, target_pose, offset in ((pose_a, pose_b, 0), (pose_b, pose_a, 1)):
        source = _triangle_mesh(_transform(vertices, source_pose), faces)
        o3d.utility.random.seed(int(seed + offset))
        points = np.asarray(
            source.sample_points_uniformly(number_of_points=int(sample_count)).points
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(
                _triangle_mesh(_transform(vertices, target_pose), faces)
            )
        )
        distances = scene.compute_distance(
            o3d.core.Tensor(points.astype(np.float32))
        ).numpy()
        gap = float(distances.max())
        largest = max(largest, gap)
    return largest


def poses_are_same(
    cad_vertices: np.ndarray,
    cad_faces: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    *,
    margin_mm: float = SAME_POSE_MARGIN_MM,
    seed: int = 0,
) -> bool:
    """I-15's "same": the two posed CAD surfaces lie within margin_mm of each other everywhere."""
    return (
        surface_gap_mm(cad_vertices, cad_faces, pose_a, pose_b, seed=seed) <= margin_mm
    )


def outline_at(
    vertices: np.ndarray, faces: np.ndarray, z_mm: float
) -> np.ndarray | None:
    """Largest closed loop where the surface crosses z == z_mm, counter-clockwise, or None.

    Vertices are welded first (to 1 micrometre), because a crossing is keyed by the mesh edge it
    lies on, and an edge only links its two faces if they share vertex indices. Open chains (a
    surface with a hole through the slice) are skipped rather than closed by guesswork: a loop
    that was closed by assumption would put a wall where none was observed.
    """
    welded, welded_faces = _weld(np.asarray(vertices, dtype=np.float64), faces)
    height = welded[:, 2] - float(z_mm)
    # A vertex exactly on the plane would make a face cross it at a point rather than along a
    # segment. Nudging it a hair above keeps every crossing a clean segment.
    height = np.where(height == 0.0, 1e-9, height)

    crossing_point: dict[tuple[int, int], np.ndarray] = {}
    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle in welded_faces:
        crossed: list[tuple[int, int]] = []
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            if (height[first] < 0.0) != (height[second] < 0.0):
                low, high = (int(min(first, second)), int(max(first, second)))
                key = (low, high)
                if key not in crossing_point:
                    fraction = height[low] / (height[low] - height[high])
                    crossing_point[key] = welded[low, :2] + fraction * (
                        welded[high, :2] - welded[low, :2]
                    )
                crossed.append(key)
        if len(crossed) == 2:
            neighbours.setdefault(crossed[0], []).append(crossed[1])
            neighbours.setdefault(crossed[1], []).append(crossed[0])

    loops: list[np.ndarray] = []
    visited: set[tuple[int, int]] = set()
    for start in neighbours:
        if start in visited:
            continue
        # Collect the connected chain first; only a chain where every crossing has exactly two
        # neighbours is a closed loop.
        component: list[tuple[int, int]] = []
        stack = [start]
        while stack:
            key = stack.pop()
            if key in visited:
                continue
            visited.add(key)
            component.append(key)
            for neighbour in neighbours[key]:
                if neighbour not in visited:
                    stack.append(neighbour)
        if any(len(neighbours[key]) != 2 for key in component):
            continue
        order = [start]
        previous = None
        current = start
        while True:
            options = [key for key in neighbours[current] if key != previous]
            following = options[0]
            if following == start or len(order) > len(component):
                break
            order.append(following)
            previous, current = current, following
        if len(order) >= 3:
            loops.append(np.array([crossing_point[key] for key in order]))

    if not loops:
        return None
    largest = max(loops, key=lambda loop: abs(_signed_area(loop)))
    if _signed_area(largest) < 0.0:
        largest = largest[::-1]
    return largest


@dataclass(frozen=True)
class WallOffset:
    """How far the model's walls are turned from the posed CAD's, in one slice.

    offset_deg: counter-clockwise turn that best lines the CAD's wall directions up with the
        model's, within +/- the search range. Its size is the wall yaw error.
    aligned_fraction: weighted share of the model outline whose wall direction is within
        WALL_ALIGNED_WITHIN_DEG of a CAD wall after that turn. Low means the slice has no clear
        straight walls (a rounded part, or a slice through clutter), and the offset shouldn't be
        trusted for that part.
    """

    offset_deg: float
    aligned_fraction: float


def wall_offset(
    model_loop: np.ndarray,
    cad_loop: np.ndarray,
    *,
    search_deg: float = DEFAULT_WALL_SEARCH_DEG,
) -> WallOffset:
    """Turn between two outlines' wall directions, by direction-histogram correlation.

    Every outline point gets a wall direction from a line fit to its spatial neighbours,
    weighted by how straight that neighbourhood is (a corner's neighbourhood isn't a line, so it
    barely counts). The directions are binned, smoothed a little, and the turn that best
    overlays the CAD's histogram on the model's is the offset.

    Directions are lines, so they are compared modulo 180 degrees: opposite walls share one
    direction. That can't tell a 180-degree turn from none, which is fine here, because gross
    turns are the photo reading's job and this check only looks within +/- search_deg.

    Only the posed CAD's own outline is compared, so no shape is assumed (box, triangle,
    hexagon), and no fitting of the CAD to the model is involved.
    """
    if not (math.isfinite(search_deg) and 0.0 < search_deg <= 90.0):
        raise ValueError("search_deg must be in (0, 90].")
    # Both outlines are resampled densely first, so a clean outline that is only its corners
    # still has points along each wall. The spatial line fit and the thinning below don't
    # depend on arc length, so a jittered outline's inflated arc length does no harm here.
    model_directions, model_weights = _wall_directions(
        _resample_by_arc_length(model_loop, WALL_THINNING_MM)
    )
    cad_directions, cad_weights = _wall_directions(
        _resample_by_arc_length(cad_loop, WALL_THINNING_MM)
    )
    if model_weights.sum() <= 0.0 or cad_weights.sum() <= 0.0:
        raise ValueError(
            "an outline has no straight stretch to read a wall direction from."
        )

    bin_count = round(180.0 / WALL_HISTOGRAM_BIN_DEG)
    model_histogram = _smoothed_line_histogram(
        model_directions, model_weights, bin_count
    )
    cad_histogram = _smoothed_line_histogram(cad_directions, cad_weights, bin_count)

    # correlation[k] = sum_a model(a) * cad(a - k bins): how well the CAD, turned by k bins
    # counter-clockwise, overlays the model.
    correlation = np.real(
        np.fft.ifft(np.fft.fft(model_histogram) * np.conj(np.fft.fft(cad_histogram)))
    )
    shifts_deg = np.arange(bin_count) * WALL_HISTOGRAM_BIN_DEG
    signed_shifts = np.where(shifts_deg >= 90.0, shifts_deg - 180.0, shifts_deg)
    # Wider than half the CAD's own direction period, the search would find an alias of the
    # true offset (g02_04, a pentagon, read 35-36 degrees for a pose 1 degree off).
    period = _direction_period_deg(cad_histogram)
    window = min(search_deg, period / 2.0)
    candidate_indices = np.flatnonzero(np.abs(signed_shifts) <= window)
    best_index = int(candidate_indices[np.argmax(correlation[candidate_indices])])

    # Parabolic interpolation between neighbouring bins, so the answer isn't quantised to the
    # bin width.
    left = correlation[(best_index - 1) % bin_count]
    centre = correlation[best_index]
    right = correlation[(best_index + 1) % bin_count]
    denominator = left - 2.0 * centre + right
    refinement = 0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
    offset = float(signed_shifts[best_index] + refinement * WALL_HISTOGRAM_BIN_DEG)

    # Distinct CAD wall directions after the turn, rounded so a straight wall's many points
    # collapse to one entry.
    turned_cad = np.unique(np.round(((cad_directions + offset) % 180.0) * 2.0) / 2.0)
    aligned_weight = 0.0
    for direction, weight in zip(model_directions, model_weights):
        nearest = min(
            symmetric_yaw_difference_deg(direction, cad_direction, 180.0)
            for cad_direction in turned_cad
        )
        if nearest <= WALL_ALIGNED_WITHIN_DEG:
            aligned_weight += float(weight)
    return WallOffset(
        offset_deg=offset,
        aligned_fraction=aligned_weight / float(model_weights.sum()),
    )


def judge_pose(
    cad_vertices: np.ndarray,
    pose: np.ndarray,
    reading: PhotoReading,
    *,
    named_side_normal_cad_xy: tuple[float, float],
    symmetry_deg: float = NO_SYMMETRY_DEG,
    wall_yaw_offset_deg: float | None = None,
    wall_aligned_fraction: float | None = None,
) -> PoseJudgement:
    """Is this pose right, by I-15's definition (#440)?

    Right means all of: the face down matches the photo (only where the CAD's faces differ);
    the named side faces within 20 degrees of the photo reading, modulo the CAD's symmetry; for
    a part with straight walls whose aligned fraction is at least WALL_ALIGNED_FLOOR, the walls
    are within 5 degrees; and the centre is within the smaller of 15 mm and half the CAD's
    narrowest footprint width.

    wall_yaw_offset_deg is `wall_offset(...).offset_deg` for this pose, or None for a part
    without straight walls (then only the photo reading judges yaw). wall_aligned_fraction is
    the matching `aligned_fraction`; below WALL_ALIGNED_FLOOR, the wall reference is too noisy
    to use and the pose falls back to the photo reading's 20-degree margin.
    """
    reasons: list[str] = []

    face_ok = reading.face_matches
    if face_ok is False:
        reasons.append("rests on a different face from the photographs")

    posed_side_deg = pose_rotation_of_xy_direction(pose, named_side_normal_cad_xy)
    gross_error = symmetric_yaw_difference_deg(
        posed_side_deg, reading.named_side_faces_deg, symmetry_deg
    )
    if gross_error > GROSS_YAW_MARGIN_DEG:
        reasons.append(
            f"named side is {gross_error:.1f} deg from the photo reading "
            f"(margin {GROSS_YAW_MARGIN_DEG:g})"
        )

    wall_error: float | None = None
    if wall_yaw_offset_deg is not None:
        aligned = wall_aligned_fraction if wall_aligned_fraction is not None else 1.0
        if aligned >= WALL_ALIGNED_FLOOR:
            wall_error = abs(float(wall_yaw_offset_deg))
            if wall_error > WALL_YAW_MARGIN_DEG:
                reasons.append(
                    f"walls are {wall_error:.1f} deg from the model's "
                    f"(margin {WALL_YAW_MARGIN_DEG:g})"
                )

    centre = posed_centre_xy(cad_vertices, pose)
    centre_error = float(
        np.linalg.norm(centre - np.asarray(reading.centre_xy_mm, dtype=np.float64))
    )
    margin = centre_margin_mm(cad_vertices)
    if centre_error > margin:
        reasons.append(
            f"centre is {centre_error:.1f} mm from the photo reading (margin {margin:.1f})"
        )

    return PoseJudgement(
        right=not reasons,
        face_ok=face_ok,
        gross_yaw_error_deg=gross_error,
        wall_yaw_error_deg=wall_error,
        centre_error_mm=centre_error,
        centre_margin_mm=margin,
        reasons=tuple(reasons),
    )


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]


def _triangle_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )


def _weld(
    vertices: np.ndarray, faces: np.ndarray, resolution_mm: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.round(vertices / resolution_mm).astype(np.int64)
    _unique, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    welded = vertices[first_index]
    return welded, inverse.reshape(-1)[np.asarray(faces, dtype=np.int64)]


def _signed_area(loop: np.ndarray) -> float:
    x = loop[:, 0]
    y = loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _check_outline(loop: np.ndarray) -> np.ndarray:
    loop = np.asarray(loop, dtype=np.float64)
    if loop.ndim != 2 or loop.shape[1] != 2 or len(loop) < 3:
        raise ValueError("an outline must be an (N,2) array with N >= 3.")
    return loop


def _resample_by_arc_length(loop: np.ndarray, step_mm: float) -> np.ndarray:
    """Points every step_mm along a closed outline (fewer if the outline is very short)."""
    loop = _check_outline(loop)
    closed = np.vstack([loop, loop[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance_along = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = float(distance_along[-1])
    if perimeter <= 0.0:
        raise ValueError("an outline must have a positive perimeter.")
    # At least eight points, so a tiny outline still has directions to read.
    count = max(8, int(perimeter // step_mm))
    stations = np.linspace(0.0, perimeter, count, endpoint=False)
    x = np.interp(stations, distance_along, closed[:, 0])
    y = np.interp(stations, distance_along, closed[:, 1])
    return np.column_stack([x, y])


# A shift counts as a repeat of the CAD's wall directions when its overlap is this close to the
# unshifted one.
DIRECTION_PERIOD_MATCH = 0.95


def _direction_period_deg(cad_histogram: np.ndarray) -> float:
    """Smallest turn, modulo 180 degrees, that maps the CAD's wall directions onto themselves.

    180 for a part with no repeat (s-04, g02_03), 90 for a square or a rectangle's walls, 60
    for a hexagon, 36 for a pentagon (its five walls, taken as lines, are 36 degrees apart).
    Only divisors of 180 can be periods, so those are the only ones tried, smallest first.
    """
    zero_shift = float(np.dot(cad_histogram, cad_histogram))
    if zero_shift <= 0.0:
        return 180.0
    for divisions in range(12, 1, -1):
        period = 180.0 / divisions
        shift_bins = round(period / WALL_HISTOGRAM_BIN_DEG)
        if shift_bins * WALL_HISTOGRAM_BIN_DEG != period:
            continue
        overlap = float(np.dot(cad_histogram, np.roll(cad_histogram, shift_bins)))
        if overlap >= DIRECTION_PERIOD_MATCH * zero_shift:
            return period
    return 180.0


def _wall_directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point line direction (degrees, modulo 180) and straightness weight.

    Points are thinned to one per WALL_THINNING_MM cell, then each keeps the principal direction
    of its neighbours within WALL_NEIGHBOURHOOD_MM. The weight is (l1 - l2) / l1 from that
    neighbourhood's spread: 1 on a straight wall, near 0 at a corner or in a blob of noise.
    """
    cells = np.floor(points / WALL_THINNING_MM).astype(np.int64)
    _unique, first_index = np.unique(cells, axis=0, return_index=True)
    thinned = points[np.sort(first_index)]

    directions = np.zeros(len(thinned))
    weights = np.zeros(len(thinned))
    radius_squared = WALL_NEIGHBOURHOOD_MM**2
    for index in range(len(thinned)):
        offsets = thinned - thinned[index]
        nearby = offsets[np.einsum("ij,ij->i", offsets, offsets) <= radius_squared]
        if len(nearby) < 3:
            continue
        centred = nearby - nearby.mean(axis=0)
        covariance = centred.T @ centred / len(centred)
        values, vectors = np.linalg.eigh(covariance)
        larger = float(values[1])
        if larger <= 0.0:
            continue
        major = vectors[:, 1]
        directions[index] = math.degrees(math.atan2(major[1], major[0])) % 180.0
        weights[index] = (larger - float(values[0])) / larger
    return directions, weights


def _hu_seventh_moment(loop: np.ndarray) -> float:
    """Hu's seventh moment of a 2D outline — sign-sensitive to reflection.

    Invariant to translation, scale and rotation, but flips sign when the outline is mirrored.
    A shape that is its own mirror image under some rotation (box, regular polygon) has h7 ≈ 0.
    """
    loop = np.asarray(loop, dtype=np.float64)
    centroid = loop.mean(axis=0)
    centred = loop - centroid
    n = len(centred)

    def mu(p: int, q: int) -> float:
        return float(np.sum(centred[:, 0] ** p * centred[:, 1] ** q)) / n

    mu20 = mu(2, 0)
    mu02 = mu(0, 2)
    mu30 = mu(3, 0)
    mu03 = mu(0, 3)
    mu21 = mu(2, 1)
    mu12 = mu(1, 2)

    # Normalised central moments: η_pq = μ_pq / μ_00^((p+q)/2 + 1).
    # With per-point averaging, μ_00 = 1 so η_pq = μ_pq. Scale-normalise by the second-order
    # norm so the threshold doesn't depend on the outline's physical size.
    norm = mu20 + mu02
    if norm <= 0.0:
        return 0.0
    scale = norm**1.5
    eta30 = mu30 / scale
    eta03 = mu03 / scale
    eta21 = mu21 / scale
    eta12 = mu12 / scale

    return float(
        (3.0 * eta21 - eta03)
        * (eta30 + eta12)
        * ((eta30 + eta12) ** 2 - 3.0 * (eta21 + eta03) ** 2)
        - (eta30 - 3.0 * eta12)
        * (eta21 + eta03)
        * (3.0 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    )


# The CAD outline must be meaningfully chiral for the sign comparison to be reliable.
# A box or regular polygon gives < 1e-10; a plate whose only asymmetry is a small notch gives
# ~ 6e-4; a right triangle or non-equilateral prism gives > 3e-3.  At ~ 6e-4, reconstruction
# noise, partial observation and planted defects can flip the model's h7 sign on a correctly-
# posed part (G-05 defect-ladder regression, 2026-11-25).  The CAD floor sits above that.
_HANDEDNESS_H7_CAD_FLOOR = 1e-3
# The model's h7 must be large enough to trust its sign; below this, noise or missing geometry
# could have produced the sign.  Much lower than the CAD floor because reconstruction always
# weakens the moment, and even a small but nonzero model h7 with opposite sign from a strongly
# chiral CAD is evidence of a face flip (s-04 golden capture h7_model ~ 2e-5).
_HANDEDNESS_H7_MODEL_FLOOR = 1e-7


@dataclass(frozen=True)
class HandednessResult:
    """I-16: whether the model's outline matches the posed CAD's handedness."""

    mismatched: bool
    has_handedness: bool
    reason: str
    h7_model: float
    h7_cad: float


def check_handedness(
    model_vertices: np.ndarray,
    model_faces: np.ndarray,
    cad_vertices: np.ndarray,
    cad_faces: np.ndarray,
    pose: np.ndarray,
) -> HandednessResult:
    """I-16: compare the handedness of the model's outline with the posed CAD's.

    Cuts a mid-height horizontal outline from each mesh and compares Hu's seventh moment.
    A sign mismatch means the part is on a different face from the CAD.
    """
    posed_cad = _transform(cad_vertices, pose)
    z_min = float(posed_cad[:, 2].min())
    z_max = float(posed_cad[:, 2].max())
    z_mid = (z_min + z_max) / 2.0

    cad_outline = outline_at(posed_cad, cad_faces, z_mid)
    if cad_outline is None:
        return HandednessResult(
            mismatched=False,
            has_handedness=False,
            reason="no CAD outline at mid-height",
            h7_model=0.0,
            h7_cad=0.0,
        )

    model_outline = outline_at(model_vertices, model_faces, z_mid)
    if model_outline is None:
        return HandednessResult(
            mismatched=False,
            has_handedness=False,
            reason="no model outline at mid-height",
            h7_model=0.0,
            h7_cad=0.0,
        )

    h7_cad = _hu_seventh_moment(cad_outline)
    h7_model = _hu_seventh_moment(model_outline)

    if abs(h7_cad) < _HANDEDNESS_H7_CAD_FLOOR:
        return HandednessResult(
            mismatched=False,
            has_handedness=False,
            reason="CAD outline has no handedness",
            h7_model=h7_model,
            h7_cad=h7_cad,
        )

    if abs(h7_model) < _HANDEDNESS_H7_MODEL_FLOOR:
        return HandednessResult(
            mismatched=False,
            has_handedness=True,
            reason="model outline moment too small to judge",
            h7_model=h7_model,
            h7_cad=h7_cad,
        )

    mismatched = (h7_model > 0.0) != (h7_cad > 0.0)
    return HandednessResult(
        mismatched=mismatched,
        has_handedness=True,
        reason="handedness mismatch" if mismatched else "handedness matches",
        h7_model=h7_model,
        h7_cad=h7_cad,
    )


def _smoothed_line_histogram(
    directions_deg: np.ndarray, weights: np.ndarray, bin_count: int
) -> np.ndarray:
    histogram = np.zeros(bin_count, dtype=np.float64)
    indices = (
        np.floor(directions_deg / WALL_HISTOGRAM_BIN_DEG).astype(np.int64) % bin_count
    )
    np.add.at(histogram, indices, weights)
    # Circular Gaussian smoothing, so two outlines a fraction of a bin apart still overlap.
    half_width = math.ceil(4.0 * WALL_HISTOGRAM_SIGMA_DEG / WALL_HISTOGRAM_BIN_DEG)
    offsets = np.arange(-half_width, half_width + 1)
    kernel = np.exp(
        -0.5 * (offsets * WALL_HISTOGRAM_BIN_DEG / WALL_HISTOGRAM_SIGMA_DEG) ** 2
    )
    smoothed = np.zeros_like(histogram)
    for offset, weight in zip(offsets, kernel):
        smoothed += weight * np.roll(histogram, int(offset))
    return smoothed
