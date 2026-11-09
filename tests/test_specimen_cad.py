"""G-02: the specimen batch's CAD files load as valid prisms and fit the board footprint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from inspection.cad import load_mesh

from recon.cad_size_check import prism_from_cad
from recon.part_segmentation import DEFAULT_FOOTPRINT_MM

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "reference_cad"

# (filename, expected footprint vertex count, expected height_mm, expected footprint area_mm2)
SPECIMENS = [
    ("g02_01_square_20x20x15mm.stl", 4, 15.0, 400.0),
    ("g02_02_rect_50x20x25mm.stl", 4, 25.0, 1000.0),
    ("g02_03_right_triangle_40x30x20mm.stl", 3, 20.0, 600.0),
    ("g02_04_pentagon_r22x25mm.stl", 5, 25.0, 1150.7784),
    ("g02_05_hexagon_r28x35mm.stl", 6, 35.0, 2036.8917),
    ("g02_06_rect_70x45x40mm.stl", 4, 40.0, 3150.0),
]


def _footprint_area(footprint_xy: np.ndarray) -> float:
    x, y = footprint_xy[:, 0], footprint_xy[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6.0
    )


@pytest.mark.parametrize(
    ("filename", "expected_vertex_count", "expected_height_mm", "expected_area_mm2"),
    SPECIMENS,
)
def test_specimen_is_a_valid_convex_prism(
    filename, expected_vertex_count, expected_height_mm, expected_area_mm2
):
    vertices, _faces, units = load_mesh(FIXTURES / filename)
    assert units == "mm"

    prism = prism_from_cad(vertices)

    assert len(prism.footprint_xy) == expected_vertex_count
    assert prism.height_mm == pytest.approx(expected_height_mm)
    assert _footprint_area(prism.footprint_xy) == pytest.approx(
        expected_area_mm2, rel=1e-3
    )


@pytest.mark.parametrize(
    ("filename", "expected_vertex_count", "expected_height_mm", "expected_area_mm2"),
    SPECIMENS,
)
def test_specimen_mesh_is_closed_and_outward_wound(
    filename, expected_vertex_count, expected_height_mm, expected_area_mm2
):
    # Catches a winding/vertex mistake in the hand-written STL that loading alone wouldn't --
    # a flipped triangle still loads and still measures the right footprint/height, but the
    # mesh's own signed volume (divergence theorem) only comes out positive and correct if
    # every facet's outward normal is actually outward. Same check as G-01's calibration cube
    # (test_cad_loading.py).
    vertices, faces, _units = load_mesh(FIXTURES / filename)

    signed_volume = _signed_volume(vertices, faces)

    assert signed_volume == pytest.approx(
        expected_area_mm2 * expected_height_mm, rel=1e-3
    )


@pytest.mark.parametrize("filename", [spec[0] for spec in SPECIMENS])
def test_specimen_footprint_fits_the_board_with_margin(filename):
    vertices, _faces, _units = load_mesh(FIXTURES / filename)
    prism = prism_from_cad(vertices)

    bbox_mm = prism.footprint_xy.max(axis=0) - prism.footprint_xy.min(axis=0)
    footprint_width_mm, footprint_height_mm = DEFAULT_FOOTPRINT_MM

    # Not just "fits" -- leaves real margin on every side for R-23's board-corner anchor, the
    # same margin s04's own (largest, already-working) bounding box leaves.
    margin_mm = 40.0
    assert bbox_mm[0] <= footprint_width_mm - margin_mm
    assert bbox_mm[1] <= footprint_height_mm - margin_mm
