"""I-15: the pose-judgement helpers measure what the story says they measure.

These are the references every later I-15 and R-44 record is judged against, so they are tested
on shapes whose answer is known exactly: the repository's own s-04 and s-05 CADs, and boxes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from inspection import cad, pose_judgement
from inspection.registration import CAD_TO_BOARD_FLIP

ROOT = Path(__file__).resolve().parents[1]
S04 = ROOT / "fixtures" / "reference_cad" / "s04_triangular_prism_60-26-65x30mm.stl"
S05 = ROOT / "fixtures" / "reference_cad" / "s05_rect_prism_30x30x60mm.stl"


def _box(width: float, depth: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.geometry.TriangleMesh.create_box(width, depth, height)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(
        mesh.triangles, dtype=np.int64
    )


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, _units = cad.load_mesh(path)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _yaw(deg: float) -> np.ndarray:
    theta = math.radians(deg)
    pose = np.eye(4)
    pose[:2, :2] = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]
    return pose


def _translation(x: float, y: float, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


def _turn_about_own_centre(
    vertices: np.ndarray, pose: np.ndarray, deg: float
) -> np.ndarray:
    """`pose`, then a turn about the posed part's own vertical centre line."""
    centre = pose_judgement.posed_centre_xy(vertices, pose)
    return (
        _translation(centre[0], centre[1])
        @ _yaw(deg)
        @ _translation(-centre[0], -centre[1])
        @ pose
    )


def _board_pose(yaw_deg: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """A pose shaped like registration's: frame flip, yaw, then a move on the board."""
    return _translation(x, y) @ _yaw(yaw_deg) @ CAD_TO_BOARD_FLIP


def _mid_height_loop(
    vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    posed = pose_judgement._transform(vertices, pose)
    level = (posed[:, 2].min() + posed[:, 2].max()) / 2.0
    loop = pose_judgement.outline_at(posed, faces, level)
    assert loop is not None
    return loop


# --- symmetric yaw difference ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "period", "expected"),
    [
        (85.0, 0.0, 90.0, 5.0),  # a square turned 85 degrees is 5 degrees off
        (350.0, 10.0, 360.0, 20.0),  # wraps through zero
        (0.0, 180.0, 180.0, 0.0),  # a 180-degree-symmetric part turned 180 is the same
        (0.0, 180.0, 360.0, 180.0),  # an asymmetric one is not
        (31.0, 0.0, 60.0, 29.0),  # hexagon: the nearer image is 60 - 31 away
    ],
)
def test_symmetric_yaw_difference(a, b, period, expected):
    assert pose_judgement.symmetric_yaw_difference_deg(a, b, period) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("period", [0.0, -90.0, 361.0, math.nan])
def test_symmetric_yaw_difference_rejects_a_bad_period(period):
    with pytest.raises(ValueError):
        pose_judgement.symmetric_yaw_difference_deg(0.0, 10.0, period)


# --- centre margin --------------------------------------------------------------------------


def test_centre_margin_is_half_the_narrowest_width_for_a_small_part():
    # g02_01's case: a 20 mm square gets 10 mm, not 15.
    vertices, _faces = _box(20.0, 20.0, 10.0)
    assert pose_judgement.centre_margin_mm(vertices) == pytest.approx(10.0)


def test_centre_margin_is_capped_at_15_mm_for_a_larger_part():
    vertices, _faces = _load(S05)
    assert pose_judgement.centre_margin_mm(vertices) == pytest.approx(15.0)


def test_footprint_width_is_measured_across_directions_not_file_axes():
    # A 40 x 10 rectangle drawn turned 30 degrees in its file is still 10 mm wide.
    vertices, _faces = _box(40.0, 10.0, 5.0)
    turned = pose_judgement._transform(vertices, _yaw(30.0))
    assert pose_judgement.footprint_min_width_mm(turned) == pytest.approx(
        10.0, abs=0.05
    )


def test_posed_centre_is_the_bounding_box_middle():
    vertices, _faces = _box(20.0, 40.0, 10.0)
    centre = pose_judgement.posed_centre_xy(vertices, _translation(100.0, -50.0))
    np.testing.assert_allclose(centre, [110.0, -30.0])


# --- same pose ------------------------------------------------------------------------------


def test_a_pose_is_the_same_as_itself():
    vertices, faces = _load(S04)
    pose = _board_pose(37.0, 12.0, -8.0)
    assert pose_judgement.surface_gap_mm(vertices, faces, pose, pose) < 0.01
    assert pose_judgement.poses_are_same(vertices, faces, pose, pose)


def test_a_symmetric_cad_turned_by_its_own_symmetry_is_the_same_pose():
    # s-05 is square in plan, so a quarter turn gives the same surface. "Same" must say so
    # with no list of symmetries.
    vertices, faces = _load(S05)
    pose = _board_pose(20.0, 5.0, 5.0)
    turned = _turn_about_own_centre(vertices, pose, 90.0)
    assert pose_judgement.surface_gap_mm(vertices, faces, pose, turned) < 0.05
    assert pose_judgement.poses_are_same(vertices, faces, pose, turned)


def test_an_asymmetric_cad_turned_half_way_is_not_the_same_pose():
    vertices, faces = _load(S04)
    pose = _board_pose(0.0)
    turned = _turn_about_own_centre(vertices, pose, 180.0)
    assert pose_judgement.surface_gap_mm(vertices, faces, pose, turned) > 10.0
    assert not pose_judgement.poses_are_same(vertices, faces, pose, turned)


def test_surface_gap_reads_a_known_shift():
    vertices, faces = _load(S05)
    pose = _board_pose(0.0)
    shifted = _translation(3.0, 0.0) @ pose
    gap = pose_judgement.surface_gap_mm(vertices, faces, pose, shifted)
    assert gap == pytest.approx(3.0, abs=0.05)
    assert not pose_judgement.poses_are_same(vertices, faces, pose, shifted)


# --- outline --------------------------------------------------------------------------------


def test_outline_of_a_box_is_its_counter_clockwise_section():
    vertices, faces = _box(30.0, 20.0, 60.0)
    loop = pose_judgement.outline_at(vertices, faces, 30.0)
    assert loop is not None
    assert pose_judgement._signed_area(loop) == pytest.approx(600.0, rel=1e-6)


def test_outline_above_the_part_is_none():
    vertices, faces = _box(30.0, 20.0, 60.0)
    assert pose_judgement.outline_at(vertices, faces, 75.0) is None


def test_outline_keeps_the_largest_loop_when_there_is_clutter():
    # A small loose piece beside the part is cut by the same plane; the part's loop wins.
    part_vertices, part_faces = _box(30.0, 30.0, 40.0)
    clutter_vertices, clutter_faces = _box(4.0, 4.0, 40.0)
    clutter_vertices = clutter_vertices + [50.0, 0.0, 0.0]
    vertices = np.vstack([part_vertices, clutter_vertices])
    faces = np.vstack([part_faces, clutter_faces + len(part_vertices)])
    loop = pose_judgement.outline_at(vertices, faces, 20.0)
    assert loop is not None
    assert pose_judgement._signed_area(loop) == pytest.approx(900.0, rel=1e-6)


def test_outline_skips_a_slice_that_isnt_closed():
    # With one side wall missing, the slice is an open chain. It must not be closed by guesswork.
    vertices, faces = _box(30.0, 30.0, 40.0)
    centroids = vertices[faces].mean(axis=1)
    keep = ~np.isclose(centroids[:, 0], 0.0)
    assert pose_judgement.outline_at(vertices, faces[keep], 20.0) is None


def test_outline_welds_an_unshared_triangle_soup():
    vertices, faces = _box(30.0, 20.0, 60.0)
    soup_vertices = vertices[faces].reshape(-1, 3)
    soup_faces = np.arange(len(soup_vertices)).reshape(-1, 3)
    loop = pose_judgement.outline_at(soup_vertices, soup_faces, 30.0)
    assert loop is not None
    assert pose_judgement._signed_area(loop) == pytest.approx(600.0, rel=1e-6)


# --- wall offset ----------------------------------------------------------------------------


@pytest.mark.parametrize("turn_deg", [7.0, -12.0, 0.6])
@pytest.mark.parametrize("cad_path", [S04, S05])
def test_wall_offset_recovers_a_known_turn(cad_path: Path, turn_deg: float):
    vertices, faces = _load(cad_path)
    pose = _board_pose(25.0, 10.0, 4.0)
    model_pose = _turn_about_own_centre(vertices, pose, turn_deg)
    result = pose_judgement.wall_offset(
        _mid_height_loop(vertices, faces, model_pose),
        _mid_height_loop(vertices, faces, pose),
    )
    assert result.offset_deg == pytest.approx(turn_deg, abs=0.1)
    assert result.aligned_fraction > 0.95


def test_wall_offset_reads_a_square_modulo_its_symmetry():
    # s-05 turned 93 degrees has walls 3 degrees off: the quarter turn is its own symmetry.
    vertices, faces = _load(S05)
    pose = _board_pose(0.0)
    model_pose = _turn_about_own_centre(vertices, pose, 93.0)
    result = pose_judgement.wall_offset(
        _mid_height_loop(vertices, faces, model_pose),
        _mid_height_loop(vertices, faces, pose),
    )
    assert result.offset_deg == pytest.approx(3.0, abs=0.1)


@pytest.mark.parametrize("jitter_mm", [0.05, 0.3])
def test_wall_offset_survives_a_noisy_reconstructed_outline(jitter_mm: float):
    # A reconstructed slice is many short, jittered segments, not four clean edges. 0.3 mm is a
    # deliberately harsh per-point jitter, not a measured noise figure: the answer has to hold
    # well inside the 5 degree wall margin even there.
    vertices, faces = _load(S05)
    pose = _board_pose(10.0)
    model_pose = _turn_about_own_centre(vertices, pose, 4.0)
    clean = _mid_height_loop(vertices, faces, model_pose)
    dense = []
    for index in range(len(clean)):
        start = clean[index]
        end = clean[(index + 1) % len(clean)]
        for fraction in np.linspace(0.0, 1.0, 120, endpoint=False):
            dense.append(start + fraction * (end - start))
    rng = np.random.default_rng(7)
    noisy = np.asarray(dense) + rng.normal(0.0, jitter_mm, size=(len(dense), 2))
    result = pose_judgement.wall_offset(noisy, _mid_height_loop(vertices, faces, pose))
    assert result.offset_deg == pytest.approx(4.0, abs=1.0)


def test_wall_offset_rejects_a_degenerate_outline():
    with pytest.raises(ValueError):
        pose_judgement.wall_offset(np.zeros((2, 2)), np.zeros((4, 2)))


# --- judging a pose -------------------------------------------------------------------------

# s-04's short 26 mm side, as the placement sheet names it. Its outward normal in the CAD frame
# is read off the CAD below rather than typed in, so the test doesn't depend on how the STL is
# drawn.


def _s04_short_side_normal(vertices: np.ndarray) -> tuple[float, float]:
    xy = vertices[:, :2]
    hull_order = []
    # The prism's footprint is its three bottom corners: the distinct xy points.
    corners = np.unique(np.round(xy, 6), axis=0)
    assert len(corners) == 3
    for first in range(3):
        for second in range(first + 1, 3):
            length = float(np.linalg.norm(corners[first] - corners[second]))
            hull_order.append((abs(length - 26.0), first, second))
    _, first, second = min(hull_order)
    edge = corners[second] - corners[first]
    normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
    centroid = corners.mean(axis=0)
    if normal @ (corners[first] - centroid) < 0.0:
        normal = -normal
    return float(normal[0]), float(normal[1])


def _reading_for(
    vertices, pose, normal, *, face_matches=True, turn=0.0, shift=(0.0, 0.0)
):
    faces_deg = pose_judgement.pose_rotation_of_xy_direction(pose, normal) + turn
    centre = pose_judgement.posed_centre_xy(vertices, pose)
    return pose_judgement.PhotoReading(
        named_side_faces_deg=faces_deg,
        centre_xy_mm=(float(centre[0] + shift[0]), float(centre[1] + shift[1])),
        face_matches=face_matches,
    )


def test_a_pose_that_matches_its_reading_is_right():
    vertices, _faces = _load(S04)
    normal = _s04_short_side_normal(vertices)
    pose = _board_pose(40.0, 20.0, -10.0)
    judgement = pose_judgement.judge_pose(
        vertices,
        pose,
        _reading_for(vertices, pose, normal, turn=8.0, shift=(4.0, -3.0)),
        named_side_normal_cad_xy=normal,
        wall_yaw_offset_deg=1.5,
    )
    assert judgement.right, judgement.reasons
    assert judgement.gross_yaw_error_deg == pytest.approx(8.0, abs=1e-6)
    assert judgement.centre_error_mm == pytest.approx(5.0, abs=1e-6)


def test_a_half_turned_pose_is_not_right():
    # G-05's failure shape: turned 180 degrees, otherwise plausible.
    vertices, _faces = _load(S04)
    normal = _s04_short_side_normal(vertices)
    pose = _board_pose(0.0)
    reading = _reading_for(vertices, pose, normal)
    turned = _turn_about_own_centre(vertices, pose, 180.0)
    judgement = pose_judgement.judge_pose(
        vertices, turned, reading, named_side_normal_cad_xy=normal
    )
    assert not judgement.right
    assert judgement.gross_yaw_error_deg == pytest.approx(180.0, abs=1e-6)


def test_a_turn_inside_20_degrees_is_caught_by_the_walls():
    # s-05 and the hexagon's case: 10-16 degrees passes the photo reading, not the walls.
    vertices, _faces = _load(S04)
    normal = _s04_short_side_normal(vertices)
    pose = _board_pose(0.0)
    judgement = pose_judgement.judge_pose(
        vertices,
        pose,
        _reading_for(vertices, pose, normal, turn=12.0),
        named_side_normal_cad_xy=normal,
        wall_yaw_offset_deg=-12.0,
    )
    assert not judgement.right
    assert judgement.gross_yaw_error_deg <= pose_judgement.GROSS_YAW_MARGIN_DEG
    assert judgement.wall_yaw_error_deg == pytest.approx(12.0)


def test_a_pose_beside_a_small_part_is_not_right():
    # 12 mm off is inside a flat 15 mm, but not inside half of a 20 mm square's width.
    vertices, _faces = _box(20.0, 20.0, 10.0)
    normal = (0.0, -1.0)
    pose = _board_pose(0.0)
    judgement = pose_judgement.judge_pose(
        vertices,
        pose,
        _reading_for(vertices, pose, normal, shift=(12.0, 0.0), face_matches=None),
        named_side_normal_cad_xy=normal,
        symmetry_deg=90.0,
    )
    assert not judgement.right
    assert judgement.centre_margin_mm == pytest.approx(10.0)


def test_the_wrong_face_down_is_not_right_even_when_yaw_and_centre_match():
    vertices, _faces = _load(S04)
    normal = _s04_short_side_normal(vertices)
    pose = _board_pose(15.0)
    judgement = pose_judgement.judge_pose(
        vertices,
        pose,
        _reading_for(vertices, pose, normal, face_matches=False),
        named_side_normal_cad_xy=normal,
    )
    assert not judgement.right
    assert judgement.face_ok is False


def test_face_is_not_judged_where_the_cads_faces_dont_differ():
    vertices, _faces = _box(30.0, 30.0, 60.0)
    normal = (0.0, -1.0)
    pose = _board_pose(5.0)
    judgement = pose_judgement.judge_pose(
        vertices,
        pose,
        _reading_for(vertices, pose, normal, face_matches=None),
        named_side_normal_cad_xy=normal,
        symmetry_deg=90.0,
    )
    assert judgement.right
    assert judgement.face_ok is None


def test_a_symmetric_part_is_judged_modulo_its_symmetry():
    vertices, _faces = _box(30.0, 30.0, 60.0)
    normal = (0.0, -1.0)
    pose = _board_pose(0.0)
    judgement = pose_judgement.judge_pose(
        vertices,
        pose,
        _reading_for(vertices, pose, normal, turn=88.0, face_matches=None),
        named_side_normal_cad_xy=normal,
        symmetry_deg=90.0,
    )
    assert judgement.right
    assert judgement.gross_yaw_error_deg == pytest.approx(2.0, abs=1e-6)


# --- the placement sheet's words, in board coordinates --------------------------------------


@pytest.mark.parametrize(
    ("faces", "expected"),
    [("right", 0.0), ("caption", 90.0), ("left", 180.0), ("far", 270.0)],
)
def test_sheet_edges_map_onto_board_angles(faces, expected):
    reading = pose_judgement.reading_from_sheet(
        faces=faces,
        turned_deg=0.0,
        turned_clockwise=True,
        squares_from_left=4.0,
        squares_from_caption=3.0,
        face_matches=None,
    )
    assert reading.named_side_faces_deg == pytest.approx(expected)


def test_a_clockwise_turn_seen_from_above_increases_the_board_angle():
    # Checked against a projected photo: +x is right and +y is toward the caption, so from
    # above, turning from +x toward +y is clockwise.
    clockwise = pose_judgement.reading_from_sheet(
        faces="far",
        turned_deg=10.0,
        turned_clockwise=True,
        squares_from_left=0.0,
        squares_from_caption=0.0,
        face_matches=None,
    )
    anticlockwise = pose_judgement.reading_from_sheet(
        faces="right",
        turned_deg=10.0,
        turned_clockwise=False,
        squares_from_left=0.0,
        squares_from_caption=0.0,
        face_matches=None,
    )
    assert clockwise.named_side_faces_deg == pytest.approx(280.0)
    assert anticlockwise.named_side_faces_deg == pytest.approx(350.0)


def test_sheet_centre_counts_from_the_left_and_the_caption():
    # The pattern is 8 squares along the caption and 6 away from it, 20 mm each. The far-left
    # corner is the board origin, so 6 squares from the caption is y = 0.
    reading = pose_judgement.reading_from_sheet(
        faces="caption",
        turned_deg=0.0,
        turned_clockwise=True,
        squares_from_left=4.5,
        squares_from_caption=2.0,
        face_matches=True,
    )
    assert reading.centre_xy_mm == pytest.approx((90.0, 80.0))
    assert reading.face_matches is True


@pytest.mark.parametrize(
    ("faces", "turned"), [("north", 0.0), ("caption", -5.0), ("far", math.nan)]
)
def test_sheet_reading_rejects_what_the_sheet_cant_say(faces, turned):
    with pytest.raises(ValueError):
        pose_judgement.reading_from_sheet(
            faces=faces,
            turned_deg=turned,
            turned_clockwise=True,
            squares_from_left=1.0,
            squares_from_caption=1.0,
            face_matches=None,
        )
