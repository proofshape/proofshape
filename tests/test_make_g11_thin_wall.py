"""G-11: the thin-walled part is exactly what the generator writes, is a closed solid, and differs
from its control, G-02's solid pentagon, only by its 1.2 mm walls.
"""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest
from inspection.cad import load_mesh

from recon.cad_size_check import prism_from_cad
from recon.part_segmentation import DEFAULT_FOOTPRINT_MM, DEFAULT_MIN_BOARD_CLEARANCE_MM

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "reference_cad"
GENERATOR = runpy.run_path(str(ROOT / "scripts" / "make_g11_thin_wall.py"))

THIN_WALL = FIXTURES / "g11_01_thin_wall_pentagon_r22x25mm.stl"
CONTROL = FIXTURES / "g02_04_pentagon_r22x25mm.stl"

# Written out here rather than read from the generator, so a change to its constants fails these
# tests instead of quietly moving the expectations with it.
OUTER_CIRCUMRADIUS_MM = 22.0
HEIGHT_MM = 25.0
WALL_MM = 1.2
FLOOR_MM = 2.0


def _pentagon_area(circumradius_mm: float) -> float:
    return 2.5 * circumradius_mm**2 * math.sin(2.0 * math.pi / 5.0)


def _signed_volume_about_centroid(vertices: np.ndarray, faces: np.ndarray) -> float:
    # About the centroid, not the origin: the base lies in z = 0, through the origin, where an
    # origin-referenced sum can't see a flipped base facet (G-02's review on #269).
    triangles = vertices[faces] - vertices.mean(axis=0)
    return float(
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6.0
    )


def _is_closed_and_consistently_wound(faces: np.ndarray) -> bool:
    # Every directed edge appears exactly once and its reverse exists (G-02, #269).
    directed_edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            directed_edges[(u, v)] = directed_edges.get((u, v), 0) + 1
    if any(count != 1 for count in directed_edges.values()):
        return False
    return all((v, u) in directed_edges for (u, v) in directed_edges)


def _ring_at(vertices: np.ndarray, z_mm: float) -> np.ndarray:
    at_level = vertices[np.isclose(vertices[:, 2], z_mm)]
    return np.unique(np.round(at_level[:, :2], 6), axis=0)


@pytest.fixture(scope="module")
def thin_wall():
    vertices, faces, units = load_mesh(THIN_WALL)
    assert units == "mm"
    return vertices, faces


def test_committed_file_is_exactly_what_the_generator_writes(capsys):
    assert THIN_WALL.read_bytes() == GENERATOR["thin_wall_stl_text"]().encode("utf-8")
    assert GENERATOR["main"](["--check"]) == 0
    assert "matches the generator" in capsys.readouterr().out


def test_check_fails_on_a_changed_file(tmp_path, monkeypatch, capsys):
    stale = tmp_path / THIN_WALL.name
    stale.write_text(THIN_WALL.read_text().replace("25.000000", "24.000000", 1))
    main = GENERATOR["main"]
    monkeypatch.setitem(main.__globals__, "OUTPUT_DIR", tmp_path)
    assert main(["--check"]) == 1
    assert "differs from the generator" in capsys.readouterr().out


def test_is_a_closed_outward_wound_solid_of_the_designed_volume(thin_wall):
    vertices, faces = thin_wall
    assert _is_closed_and_consistently_wound(faces)
    inner_mm = OUTER_CIRCUMRADIUS_MM - WALL_MM / math.cos(math.pi / 5.0)
    expected = _pentagon_area(OUTER_CIRCUMRADIUS_MM) * HEIGHT_MM - _pentagon_area(
        inner_mm
    ) * (HEIGHT_MM - FLOOR_MM)
    # Positive: outward winding. The tolerance covers the file's six printed decimals.
    assert _signed_volume_about_centroid(vertices, faces) == pytest.approx(
        expected, abs=0.01
    )


def test_outside_is_the_control_specimen_exactly(thin_wall):
    vertices, _ = thin_wall
    control_vertices, _, _ = load_mesh(CONTROL)
    np.testing.assert_array_equal(
        _ring_at(vertices, 0.0), _ring_at(control_vertices, 0.0)
    )
    assert vertices[:, 2].max() == control_vertices[:, 2].max() == HEIGHT_MM


def test_every_wall_is_the_designed_thickness(thin_wall):
    vertices, _ = thin_wall
    outer = _ring_at(vertices, 0.0)
    inner = _ring_at(vertices, FLOOR_MM)
    assert len(outer) == len(inner) == 5
    # Order both rings by angle so outer edge k and inner edge k are the same wall.
    outer = outer[np.argsort(np.arctan2(outer[:, 1], outer[:, 0]))]
    inner = inner[np.argsort(np.arctan2(inner[:, 1], inner[:, 0]))]
    for k in range(5):
        a, b = outer[k], outer[(k + 1) % 5]
        normal = np.array([b[1] - a[1], a[0] - b[0]]) / np.linalg.norm(b - a)
        for point in (inner[k], inner[(k + 1) % 5]):
            assert abs(float(np.dot(point - a, normal))) == pytest.approx(
                WALL_MM, abs=1e-5
            )


def test_only_the_walls_stand_above_the_board_clearance_cut():
    # The case under test: segmentation drops everything within the clearance of the board, so
    # the floor goes and the walls are what reconstruction has to keep.
    assert FLOOR_MM <= DEFAULT_MIN_BOARD_CLEARANCE_MM < HEIGHT_MM


def test_fits_the_board_footprint(thin_wall):
    vertices, _ = thin_wall
    extent_x, extent_y = np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1])
    assert max(extent_x, extent_y) < min(DEFAULT_FOOTPRINT_MM)


def test_is_outside_the_size_checks_prism_shape_on_purpose(thin_wall):
    # A deliberate exception to G-02's convex-prism constraint (G-11's first criterion allows
    # it): the cavity floor is a third z level, so the size check refuses it rather than
    # measuring the wrong shape.
    vertices, _ = thin_wall
    with pytest.raises(ValueError, match="2 distinct z levels"):
        prism_from_cad(vertices)
