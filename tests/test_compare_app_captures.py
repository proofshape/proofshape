"""Unit tests for G-15's photographs-against-app comparison. All synthetic: no GPU, no captures.

The figures themselves come from measure_part_accuracy's report_parts, which has its own tests.
What is new here is the pose comparison across two board placements, and the table.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts import compare_app_captures as c

# --- helpers ----------------------------------------------------------------------------------

# An L-shaped plate, 50 x 32 x 20 mm, as two boxes. It has no rotational symmetry about any
# axis, so a pose that is off can't be hidden by the shape looking the same.
_L_BOXES = [
    ((0.0, 0.0, 0.0), (50.0, 16.0, 20.0)),
    ((0.0, 16.0, 0.0), (20.0, 32.0, 20.0)),
]


def _box_mesh(
    lo: tuple[float, ...], hi: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return v, f


def _l_plate() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for lo, hi in _L_BOXES:
        v, f = _box_mesh(lo, hi)
        faces.append(f + sum(len(x) for x in vertices))
        vertices.append(v)
    return np.concatenate(vertices), np.concatenate(faces)


def _sample_surface(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    pick = rng.choice(len(faces), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _placement(yaw_deg: float, x: float, y: float) -> np.ndarray:
    a = math.radians(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = [
        [math.cos(a), -math.sin(a), 0],
        [math.sin(a), math.cos(a), 0],
        [0, 0, 1],
    ]
    pose[:3, 3] = [x, y, 0.0]
    return pose


def _capture(
    true_placement: np.ndarray, served_pose: np.ndarray, seed: int, stray: int = 0
) -> dict[str, np.ndarray]:
    """A capture whose part sits at true_placement, and whose registration returned served_pose."""
    cad_v, cad_f = _l_plate()
    on_part = _sample_surface(cad_v, cad_f, 3000, seed)
    rng = np.random.default_rng(seed + 100)
    on_part = on_part + rng.normal(0.0, 0.2, on_part.shape)
    model = on_part @ true_placement[:3, :3].T + true_placement[:3, 3]
    observed = np.ones(len(model), dtype=bool)
    if stray:
        # Board skin, far from the part: Observed, but never near the CAD.
        skin = np.column_stack(
            [
                rng.uniform(-80, 160, stray),
                rng.uniform(-80, 160, stray),
                np.zeros(stray),
            ]
        )
        skin = skin[np.min(np.abs(skin[:, :2] - true_placement[:2, 3]), axis=1) > 60]
        model = np.vstack([model, skin])
        observed = np.concatenate([observed, np.ones(len(skin), dtype=bool)])
    return {
        "vertices": model,
        "observed": observed,
        "cad_to_model": served_pose,
        "cad_vertices": cad_v,
        "cad_faces": cad_f,
    }


# --- observed_in_cad_frame / near_cad ---------------------------------------------------------


def test_observed_points_are_carried_back_into_the_cad_frame():
    pose = _placement(37.0, 120.0, 60.0)
    cad_points = np.array([[1.0, 2.0, 3.0], [10.0, 0.0, 5.0], [0.0, 30.0, 20.0]])
    model = cad_points @ pose[:3, :3].T + pose[:3, 3]
    back = c.observed_in_cad_frame(model, np.ones(3, dtype=bool), pose)
    np.testing.assert_allclose(back, cad_points, atol=1e-9)


def test_unobserved_vertices_are_left_out():
    back = c.observed_in_cad_frame(
        np.zeros((3, 3)), np.array([True, False, True]), np.eye(4)
    )
    assert back.shape == (2, 3)


def test_near_cad_keeps_only_points_within_the_distance():
    cad_v, cad_f = _l_plate()
    points = np.array([[25.0, 8.0, 21.0], [25.0, 8.0, 30.0], [-4.0, 8.0, 10.0]])
    kept = c.near_cad(points, cad_v, cad_f, 5.0)
    np.testing.assert_allclose(kept, points[[0, 2]])


# --- compare_poses ----------------------------------------------------------------------------


def test_two_right_poses_at_different_placements_are_the_same():
    photos_place, app_place = (
        _placement(10.0, 100.0, 80.0),
        _placement(-95.0, 60.0, 40.0),
    )
    photos = _capture(photos_place, photos_place, seed=1)
    app = _capture(app_place, app_place, seed=2)
    result = c.compare_poses(photos, app)
    assert result.gap_mm < 0.5
    assert result.same
    assert result.icp_fitness > 0.95


def test_a_pose_shifted_off_the_part_is_different():
    photos_place, app_place = (
        _placement(10.0, 100.0, 80.0),
        _placement(-95.0, 60.0, 40.0),
    )
    shifted = app_place @ _placement(
        0.0, 4.0, 0.0
    )  # the CAD served 4 mm along its own x
    photos = _capture(photos_place, photos_place, seed=1)
    app = _capture(app_place, shifted, seed=2)
    result = c.compare_poses(photos, app)
    assert result.gap_mm == pytest.approx(4.0, abs=0.6)
    assert not result.same


def test_a_pose_turned_on_the_part_is_different():
    photos_place, app_place = _placement(0.0, 100.0, 80.0), _placement(50.0, 60.0, 40.0)
    turned = app_place @ _placement(
        5.0, 0.0, 0.0
    )  # 5 degrees about the CAD's own origin
    photos = _capture(photos_place, photos_place, seed=1)
    app = _capture(app_place, turned, seed=2)
    result = c.compare_poses(photos, app)
    # 5 degrees about a corner moves the far end of a 50 mm part by about 4.4 mm.
    assert result.gap_mm > c.SAME_POSE_MARGIN_MM


def test_a_pose_turned_far_on_the_part_is_measured_in_full():
    # The G-15 g02_05 case: started from the identity alone, ICP stopped short of a turn this
    # size and reported a gap under the margin.
    photos_place, app_place = _placement(0.0, 100.0, 80.0), _placement(50.0, 60.0, 40.0)
    centre = np.array([25.0, 16.0, 10.0])
    turned = app_place @ c._turn_about_z(40.0, centre)
    photos = _capture(photos_place, photos_place, seed=1)
    app = _capture(app_place, turned, seed=2)
    cad_v, cad_f = _l_plate()
    expected = c.surface_gap_mm(cad_v, cad_f, np.eye(4), c._turn_about_z(40.0, centre))
    result = c.compare_poses(photos, app)
    assert result.gap_mm == pytest.approx(expected, abs=1.0)
    assert not result.same


def test_a_turn_by_the_parts_own_symmetry_is_the_same_pose():
    cube_v, cube_f = _box_mesh((0.0, 0.0, 0.0), (20.0, 20.0, 15.0))
    photos_place, app_place = _placement(0.0, 100.0, 80.0), _placement(30.0, 60.0, 40.0)
    quarter = c._turn_about_z(90.0, np.array([10.0, 10.0, 7.5]))
    photos = _capture(photos_place, photos_place, seed=1)
    app = _capture(app_place, app_place @ quarter, seed=2)
    for capture in (photos, app):
        on_part = _sample_surface(cube_v, cube_f, 2000, seed=3)
        place = photos_place if capture is photos else app_place
        capture.update(
            vertices=on_part @ place[:3, :3].T + place[:3, 3],
            observed=np.ones(len(on_part), dtype=bool),
            cad_vertices=cube_v,
            cad_faces=cube_f,
        )
    assert c.compare_poses(photos, app).same


def test_stray_board_skin_does_not_drag_the_comparison():
    photos_place, app_place = (
        _placement(10.0, 100.0, 80.0),
        _placement(-95.0, 60.0, 40.0),
    )
    photos = _capture(photos_place, photos_place, seed=1, stray=4000)
    app = _capture(app_place, app_place, seed=2)
    assert c.compare_poses(photos, app).same


def test_too_few_points_near_the_cad_is_an_error_not_a_gap():
    place = _placement(0.0, 0.0, 0.0)
    photos = _capture(place, place, seed=1)
    app = dict(photos, observed=np.zeros(len(photos["vertices"]), dtype=bool))
    with pytest.raises(ValueError, match="fewer than 3"):
        c.compare_poses(photos, app)


# --- service_verdict / markdown_table ---------------------------------------------------------


def test_service_verdict_reads_the_saved_body(tmp_path):
    (tmp_path / "s-05.json").write_text(
        json.dumps(
            {
                "verdict": "Rescan",
                "reason_codes": ["registration_orientation_ambiguous"],
            }
        )
    )
    assert c.service_verdict(tmp_path, "s-05") == (
        "Rescan",
        ["registration_orientation_ambiguous"],
    )
    assert c.service_verdict(tmp_path, "g02_01") == ("—", [])
    assert c.service_verdict(None, "s-05") == ("—", [])


def _figures(wall: float, ambiguous: bool, first_pass: float) -> dict:
    return {
        "summary": {
            "wall": {"abs_p95_mm": wall},
            "top": {"abs_p95_mm": 0.7},
            "worst_reading_mm": 4.28,
            "stray_observed_share": 0.03,
            "unmeasured_observed_share": 0.42,
        },
        "sweep": {"first_pass_mm": first_pass},
        "registration": {"orientation_ambiguous": ambiguous},
    }


def test_table_has_one_row_per_part_with_photographs_first():
    rows = [
        {
            "part": "g02_06",
            "photos": _figures(2.89, True, 3.7),
            "app": _figures(1.2, False, 3.7),
            "pose_gap_mm": 0.87,
            "pose_same": True,
            "service_verdict": "Pass",
            "service_reason_codes": [],
        },
        {
            "part": "s-05",
            "photos": _figures(3.77, True, 5.0),
            "app": _figures(3.12, True, 4.5),
            "pose_gap_mm": 4.85,
            "pose_same": False,
            "service_verdict": "Rescan",
            "service_reason_codes": ["registration_orientation_ambiguous"],
        },
    ]
    lines = c.markdown_table(rows).splitlines()
    assert len(lines) == 4
    assert "| g02_06 | 2.89 / 1.20 |" in lines[2]
    assert "| yes / no | 0.87 (same) | Pass |" in lines[2]
    assert (
        "4.85 (different) | Rescan `registration_orientation_ambiguous` |" in lines[3]
    )
