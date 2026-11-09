"""G-02: the specimen batch's CAD files load as valid prisms and fit the board footprint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from inspection.cad import load_mesh

from recon.board_pose import MIN_CORNERS_FOR_POSE, make_board
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


def _signed_volume_about_centroid(vertices: np.ndarray, faces: np.ndarray) -> float:
    # About the mesh's own centroid, not the origin (review on #269): every specimen's base lies
    # exactly in z=0, a plane *through* the origin, so an origin-referenced divergence-theorem
    # sum gives every base triangle a zero contribution regardless of its winding -- any flipped
    # base facet is invisible to that version. A prism's centroid sits at its interior (z is
    # about height/2, never on the base or top plane), so no facet's contribution degenerates to
    # zero regardless of which triangle got the vertices swapped.
    centroid = vertices.mean(axis=0)
    triangles = vertices[faces] - centroid
    return float(
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6.0
    )


def _is_closed_and_consistently_wound(faces: np.ndarray) -> bool:
    # Topological check, independent of geometry or reference point (review on #269): a closed,
    # consistently outward-wound manifold mesh has every directed edge (a->b) from one triangle
    # matched by its reverse (b->a) from exactly one other triangle, and no directed edge
    # repeated in the same direction (which would mean two triangles facing the same way across
    # the same edge -- a flip or a hole, not a flip's precise opposite everywhere else).
    directed_edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            directed_edges[(u, v)] = directed_edges.get((u, v), 0) + 1
    if any(count != 1 for count in directed_edges.values()):
        return False
    return all((v, u) in directed_edges for (u, v) in directed_edges)


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
    # Catches a winding/vertex mistake in the hand-written STL that loading alone wouldn't -- a
    # flipped triangle still loads and still measures the right footprint/height. Two
    # independent checks (review on #269: a centroid-only volume check has its own blind spots
    # on some geometries, so the topological edge check is the one actually relied on; the
    # volume check also pins the sign and magnitude, which the edge check alone doesn't).
    vertices, faces, _units = load_mesh(FIXTURES / filename)

    assert _is_closed_and_consistently_wound(faces)
    signed_volume = _signed_volume_about_centroid(vertices, faces)
    assert signed_volume == pytest.approx(
        expected_area_mm2 * expected_height_mm, rel=1e-3
    )


@pytest.mark.parametrize("filename", [spec[0] for spec in SPECIMENS])
def test_specimen_hides_few_enough_board_corners_for_r23(filename):
    # The real safety argument (review on #269): not "smaller than s04's bounding box" (g02_05
    # and g02_06 aren't, on at least one axis), but how many of the board's own 35 ChArUco
    # corners a centred specimen would hide from a straight-overhead frame, against what R-23
    # actually needs -- the same corners R-23 itself reads via make_board().getChessboardCorners().
    all_corners_mm = np.asarray(make_board().getChessboardCorners(), dtype=np.float64)[
        :, :2
    ]
    footprint_width_mm, footprint_height_mm = DEFAULT_FOOTPRINT_MM
    board_center_mm = np.array([footprint_width_mm / 2.0, footprint_height_mm / 2.0])

    vertices, _faces, _units = load_mesh(FIXTURES / filename)
    prism = prism_from_cad(vertices)
    half_bbox_mm = (
        prism.footprint_xy.max(axis=0) - prism.footprint_xy.min(axis=0)
    ) / 2.0
    centered_min = board_center_mm - half_bbox_mm
    centered_max = board_center_mm + half_bbox_mm

    hidden = np.all(
        (all_corners_mm >= centered_min) & (all_corners_mm <= centered_max), axis=1
    )
    visible_count = len(all_corners_mm) - int(np.count_nonzero(hidden))

    # Real margin above the bare per-frame minimum, not just clearing it -- a centred placement
    # is the worst case this checks; off-centre or non-overhead frames hide fewer corners still.
    assert visible_count >= MIN_CORNERS_FOR_POSE * 2
