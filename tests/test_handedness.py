"""I-16: a part on a different face from its CAD is flagged, not served clean.

Tests that the handedness check catches a mirrored outline, passes a matching one, and
never flags a shape with no handedness (box, regular polygon).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open3d as o3d
from inspection import cad
from inspection.pose_judgement import (
    _hu_seventh_moment,
    check_handedness,
)
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


def _mirror_y() -> np.ndarray:
    """Flip y axis — mirrors the shape, changing its handedness."""
    m = np.eye(4)
    m[1, 1] = -1.0
    return m


def _regular_polygon_prism(
    n_sides: int, radius: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """A regular polygon extruded into a prism, centred at origin."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    top = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(n_sides, height / 2)]
    )
    bottom = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(n_sides, -height / 2),
        ]
    )
    vertices = np.vstack([top, bottom])

    faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([i, j, i + n_sides])
        faces.append([j, j + n_sides, i + n_sides])
    for i in range(1, n_sides - 1):
        faces.append([0, i, i + 1])
        faces.append([n_sides, n_sides + i + 1, n_sides + i])

    return vertices, np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Hu's seventh moment: basic properties
# ---------------------------------------------------------------------------


class TestHuSeventhMoment:
    def test_triangle_nonzero(self):
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 8.0]])
        assert abs(_hu_seventh_moment(triangle)) > 1e-10

    def test_mirror_flips_sign(self):
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 8.0]])
        mirrored = triangle.copy()
        mirrored[:, 1] *= -1
        h7 = _hu_seventh_moment(triangle)
        h7_mirror = _hu_seventh_moment(mirrored)
        assert h7 * h7_mirror < 0, "mirroring must flip the sign of h7"

    def test_rotation_invariant(self):
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 8.0]])
        theta = math.radians(47.0)
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        rotated = triangle @ rot.T
        assert abs(_hu_seventh_moment(triangle) - _hu_seventh_moment(rotated)) < 1e-12

    def test_translation_invariant(self):
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 8.0]])
        shifted = triangle + np.array([100.0, -50.0])
        assert abs(_hu_seventh_moment(triangle) - _hu_seventh_moment(shifted)) < 1e-12

    def test_square_near_zero(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert abs(_hu_seventh_moment(square)) < 1e-10

    def test_regular_hexagon_near_zero(self):
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        hexagon = np.column_stack([np.cos(angles), np.sin(angles)])
        assert abs(_hu_seventh_moment(hexagon)) < 1e-10


# ---------------------------------------------------------------------------
# check_handedness: 3D mesh tests
# ---------------------------------------------------------------------------


def _pose_vertices(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 pose to 3D vertices."""
    return (pose[:3, :3] @ vertices.T).T + pose[:3, 3]


class TestCheckHandedness:
    def test_s04_normal_pose_matches(self):
        """s-04 (a triangular prism) posed normally: handedness should match."""
        v, f = _load(S04)
        pose = CAD_TO_BOARD_FLIP.copy()
        model_v = _pose_vertices(v, pose)
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched
        assert result.has_handedness

    def test_s04_mirrored_flags(self):
        """s-04 with the model mirrored (different face from CAD): should flag."""
        v, f = _load(S04)
        pose = CAD_TO_BOARD_FLIP.copy()
        model_v = _pose_vertices(v, pose)
        # Mirror the model in y — as if the part lay on its other face
        model_v[:, 1] *= -1
        result = check_handedness(model_v, f, v, f, pose)
        assert result.mismatched
        assert result.has_handedness

    def test_s04_yaw_does_not_flag(self):
        """A yaw rotation doesn't change handedness."""
        v, f = _load(S04)
        pose = _yaw(73.0) @ CAD_TO_BOARD_FLIP
        model_v = _pose_vertices(v, pose)
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched

    def test_box_no_handedness(self):
        """A box is symmetric — the check can't tell faces apart."""
        v, f = _box(30.0, 20.0, 10.0)
        pose = CAD_TO_BOARD_FLIP.copy()
        model_v = _pose_vertices(v, pose)
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched
        assert not result.has_handedness

    def test_regular_pentagon_no_handedness(self):
        """A regular pentagon prism is symmetric under a turn — no handedness."""
        v, f = _regular_polygon_prism(5, 15.0, 30.0)
        pose = CAD_TO_BOARD_FLIP.copy()
        model_v = _pose_vertices(v, pose)
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched
        assert not result.has_handedness

    def test_s05_rect_prism_no_handedness(self):
        """s-05 (a rectangular prism) has no handedness — it's a box."""
        v, f = _load(S05)
        pose = CAD_TO_BOARD_FLIP.copy()
        model_v = _pose_vertices(v, pose)
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched
        assert not result.has_handedness

    def test_noise_does_not_flip(self):
        """Adding moderate noise to the model doesn't flip the handedness answer."""
        v, f = _load(S04)
        pose = CAD_TO_BOARD_FLIP.copy()
        model_v = _pose_vertices(v, pose)
        rng = np.random.default_rng(42)
        model_v = model_v + rng.normal(0, 0.5, model_v.shape)
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched
        assert result.has_handedness

    def test_no_model_outline_inconclusive(self):
        """When the model has no outline at the slice height, the result is inconclusive."""
        v, f = _load(S04)
        pose = CAD_TO_BOARD_FLIP.copy()
        # Place model far from the posed CAD's z range so no outline is found
        model_v = _pose_vertices(v, pose)
        model_v[:, 2] += 1000.0
        result = check_handedness(model_v, f, v, f, pose)
        assert not result.mismatched
        assert not result.has_handedness
        assert "no model outline" in result.reason
