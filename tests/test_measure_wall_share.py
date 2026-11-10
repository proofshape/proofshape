"""Unit tests for R-33's wall-share measurement script. All synthetic -- no GPU, no fixtures."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import mesh_cleanup, provenance
from recon.part_segmentation import BoardSurface
from scripts import measure_wall_share as m

# A flat board at z=0 (slope 0, offset 0): height_above_board_mm is exactly -z.
FLAT_BOARD = BoardSurface(
    slope_x=0.0,
    slope_y=0.0,
    z_at_origin_mm=0.0,
    offset_mm=0.0,
    tilt_deg=0.0,
    residual_sigma_mm=0.0,
    inlier_count=0,
    candidate_count=0,
    board_layer_top_mm=None,
    height_threshold_mm=0.0,
)
BOUNDS_MIN_MM = np.array([0.0, 0.0, -50.0])
BOUNDS_MAX_MM = np.array([100.0, 100.0, 0.0])  # z=0 is the board (bounds_max_mm[2])
PART_X_MM = (40.0, 60.0)
PART_Y_MM = (40.0, 60.0)
PART_HEIGHT_MM = 33.0

# label: centroid (x, y, z) -> expected category. Covers every category, and the priority
# interaction the module docstring claims (part overrides skin). The board-side override is
# tested separately below, since it needs bounds_max_mm[2] to NOT be excluded from the check.
CENTROIDS_BY_CASE = {
    "wall": (90.0, 90.0, -25.0),  # outside the part's xy, too high for skin, no edge
    "board_skin": (10.0, 10.0, -5.0),  # low, outside the part's xy, no edge nearby
    "part": (50.0, 50.0, -10.0),  # inside the part's xy, under the part height cap
    "volume_boundary_far_z": (50.0, 50.0, -48.0),  # too tall for "part"; touches z=-50
    "part_overrides_skin": (50.0, 50.0, -5.0),  # skin-low, also in the part's xy
    "board_side_not_boundary": (50.0, 50.0, -0.5),  # touches z=0 (board); stays "part"
}


def _triangle_mesh(centroids: list[tuple[float, float, float]]):
    """One flat, tiny triangle per centroid, each centred exactly on it."""
    offsets = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]])
    vertices = []
    faces = []
    for centroid in centroids:
        base = len(vertices)
        vertices.extend(np.asarray(centroid) + offsets)
        faces.append([base, base + 1, base + 2])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _flat_grid_mesh(xs: np.ndarray, ys: np.ndarray, z_mm: float):
    """An len(xs) x len(ys) flat grid mesh at a fixed height, dense enough for
    `completion.isolate_part`'s clustering to treat it as one part."""
    xv, yv = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([xv.ravel(), yv.ravel(), np.full(xv.size, z_mm)], axis=1)
    nx, ny = len(xs), len(ys)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, i * ny + j + 1
            c, d = (i + 1) * ny + j, (i + 1) * ny + j + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices.astype(np.float64), np.asarray(faces, dtype=np.int64)


def test_classify_faces_covers_every_category_and_the_part_over_skin_override():
    labels = list(CENTROIDS_BY_CASE)
    vertices, faces = _triangle_mesh([CENTROIDS_BY_CASE[label] for label in labels])

    categories = m.classify_faces(
        vertices,
        faces,
        FLAT_BOARD,
        BOUNDS_MIN_MM,
        BOUNDS_MAX_MM,
        PART_X_MM,
        PART_Y_MM,
        PART_HEIGHT_MM,
    )

    expected = {
        "wall": m.CATEGORY_WALL,
        "board_skin": m.CATEGORY_BOARD_SKIN,
        "part": m.CATEGORY_PART,
        "volume_boundary_far_z": m.CATEGORY_VOLUME_BOUNDARY,
        "part_overrides_skin": m.CATEGORY_PART,
        "board_side_not_boundary": m.CATEGORY_PART,
    }
    for label, category in zip(labels, categories, strict=True):
        assert category == expected[label], label


def test_classify_faces_volume_boundary_overrides_part_via_a_side_wall():
    # The volume's x=62 side sits just past the part's own x=60 edge, so a face just inside the
    # part's xy box, low enough to also be skin-eligible, is still pulled into volume_boundary.
    vertices, faces = _triangle_mesh([(59.5, 50.0, -5.0)])
    bounds_min_mm = np.array([0.0, 0.0, -50.0])
    bounds_max_mm = np.array([62.0, 100.0, 0.0])

    categories = m.classify_faces(
        vertices,
        faces,
        FLAT_BOARD,
        bounds_min_mm,
        bounds_max_mm,
        PART_X_MM,
        PART_Y_MM,
        PART_HEIGHT_MM,
    )

    assert categories[0] == m.CATEGORY_VOLUME_BOUNDARY


def test_classify_faces_rejects_bad_shape():
    with pytest.raises(ValueError, match="faces"):
        m.classify_faces(
            np.zeros((3, 3)),
            np.zeros((1, 4)),
            FLAT_BOARD,
            BOUNDS_MIN_MM,
            BOUNDS_MAX_MM,
            PART_X_MM,
            PART_Y_MM,
            PART_HEIGHT_MM,
        )


def test_summarize_reports_face_share_observed_share_and_share_of_all_observed():
    # 4 faces: 2 wall (1 Observed), 1 part (Observed), 1 board_skin (Unobserved).
    categories = np.array(
        [m.CATEGORY_WALL, m.CATEGORY_WALL, m.CATEGORY_PART, m.CATEGORY_BOARD_SKIN]
    )
    face_labels = np.array(
        [
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,
        ]
    )

    summary = m.summarize(categories, face_labels)

    assert summary[m.CATEGORY_WALL]["face_count"] == 2
    assert summary[m.CATEGORY_WALL]["face_share"] == pytest.approx(0.5)
    assert summary[m.CATEGORY_WALL]["observed_share"] == pytest.approx(0.5)  # 1 of 2
    assert summary[m.CATEGORY_PART]["face_count"] == 1
    assert summary[m.CATEGORY_PART]["observed_share"] == pytest.approx(1.0)
    assert summary[m.CATEGORY_BOARD_SKIN]["observed_share"] == pytest.approx(0.0)
    # 2 faces total are Observed (one wall, one part); each gets its share of those 2.
    assert summary[m.CATEGORY_WALL]["share_of_all_observed"] == pytest.approx(0.5)
    assert summary[m.CATEGORY_PART]["share_of_all_observed"] == pytest.approx(0.5)
    assert summary[m.CATEGORY_VOLUME_BOUNDARY]["face_count"] == 0
    assert summary[m.CATEGORY_VOLUME_BOUNDARY]["face_share"] == pytest.approx(0.0)


def test_summarize_handles_an_empty_mesh_without_dividing_by_zero():
    summary = m.summarize(np.array([], dtype=object), np.array([], dtype=object))

    for category in m.CATEGORIES:
        assert summary[category] == {
            "face_count": 0,
            "face_share": 0.0,
            "observed_share": 0.0,
            "share_of_all_observed": 0.0,
        }


def test_measure_reads_a_real_fusion_report_and_provenance_glb(tmp_path):
    labels_order = ["part", "board_skin", "wall"]
    vertices, faces = _triangle_mesh(
        [CENTROIDS_BY_CASE[label] for label in labels_order]
    )
    # Every vertex of the "part" face Observed; every vertex of the other two Unobserved.
    colors = np.tile(np.asarray(provenance.UNOBSERVED_COLOR, dtype=np.uint8), (9, 1))
    colors[0:3] = provenance.OBSERVED_COLOR

    fusion_dir = tmp_path / "fusion"
    fusion_dir.mkdir()
    (fusion_dir / "tsdf_fusion.json").write_text(
        json.dumps(
            {
                "bounds_min_mm": BOUNDS_MIN_MM.tolist(),
                "bounds_max_mm": BOUNDS_MAX_MM.tolist(),
                "board_surface": {
                    "slope_x": FLAT_BOARD.slope_x,
                    "slope_y": FLAT_BOARD.slope_y,
                    "z_at_origin_mm": FLAT_BOARD.z_at_origin_mm,
                    "offset_mm": FLAT_BOARD.offset_mm,
                    "tilt_deg": FLAT_BOARD.tilt_deg,
                    "residual_sigma_mm": FLAT_BOARD.residual_sigma_mm,
                    "inlier_count": FLAT_BOARD.inlier_count,
                    "candidate_count": FLAT_BOARD.candidate_count,
                    "board_layer_top_mm": FLAT_BOARD.board_layer_top_mm,
                    "height_threshold_mm": FLAT_BOARD.height_threshold_mm,
                },
            }
        )
    )
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    summary = m.measure(
        fusion_dir, provenance_glb, PART_X_MM, PART_Y_MM, PART_HEIGHT_MM
    )

    assert summary[m.CATEGORY_PART]["face_count"] == 1
    assert summary[m.CATEGORY_PART]["observed_share"] == pytest.approx(1.0)
    assert summary[m.CATEGORY_BOARD_SKIN]["face_count"] == 1
    assert summary[m.CATEGORY_BOARD_SKIN]["observed_share"] == pytest.approx(0.0)
    assert summary[m.CATEGORY_WALL]["face_count"] == 1


def test_derive_part_footprint_reads_the_core_bounds_and_overall_top_padded():
    # 8x8 Observed grid at 10mm above the board, dense enough (1mm spacing, under the 4mm
    # cluster radius) to be one core of 64 vertices, comfortably over min_core_vertices (50).
    xs = np.arange(40.0, 48.0)
    ys = np.arange(40.0, 48.0)
    vertices, faces = _flat_grid_mesh(xs, ys, -10.0)
    vertex_labels = np.full(len(vertices), provenance.LABEL_OBSERVED, dtype=object)

    part_x_mm, part_y_mm, part_height_mm = m.derive_part_footprint(
        vertices, faces, vertex_labels, FLAT_BOARD, pad_mm=3.0
    )

    assert part_x_mm == pytest.approx((37.0, 50.0))  # grid's own 40..47, padded by 3
    assert part_y_mm == pytest.approx((37.0, 50.0))
    assert part_height_mm == pytest.approx(13.0)  # the grid's own 10mm, padded by 3


def test_measure_auto_derives_the_footprint_when_none_is_given(tmp_path):
    xs = np.arange(40.0, 48.0)
    ys = np.arange(40.0, 48.0)
    vertices, faces = _flat_grid_mesh(xs, ys, -10.0)
    colors = np.tile(
        np.asarray(provenance.OBSERVED_COLOR, dtype=np.uint8), (len(vertices), 1)
    )

    fusion_dir = tmp_path / "fusion"
    fusion_dir.mkdir()
    (fusion_dir / "tsdf_fusion.json").write_text(
        json.dumps(
            {
                "bounds_min_mm": [0.0, 0.0, -50.0],
                "bounds_max_mm": [160.0, 120.0, 0.0],
                "board_surface": {
                    "slope_x": 0.0,
                    "slope_y": 0.0,
                    "z_at_origin_mm": 0.0,
                    "offset_mm": 0.0,
                    "tilt_deg": 0.0,
                    "residual_sigma_mm": 0.0,
                    "inlier_count": 0,
                    "candidate_count": 0,
                    "board_layer_top_mm": None,
                    "height_threshold_mm": 0.0,
                },
            }
        )
    )
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    summary = m.measure(fusion_dir, provenance_glb)  # no footprint given at all

    # every face sits inside the derived footprint and under the derived height cap
    assert summary[m.CATEGORY_PART]["face_count"] == len(faces)
    assert summary[m.CATEGORY_WALL]["face_count"] == 0


def test_measure_rejects_a_provenance_glb_with_no_colors(tmp_path):
    vertices, faces = _triangle_mesh([CENTROIDS_BY_CASE["wall"]])
    fusion_dir = tmp_path / "fusion"
    fusion_dir.mkdir()
    (fusion_dir / "tsdf_fusion.json").write_text(
        json.dumps(
            {
                "bounds_min_mm": BOUNDS_MIN_MM.tolist(),
                "bounds_max_mm": BOUNDS_MAX_MM.tolist(),
                "board_surface": {
                    "slope_x": 0.0,
                    "slope_y": 0.0,
                    "z_at_origin_mm": 0.0,
                    "offset_mm": 0.0,
                    "tilt_deg": 0.0,
                    "residual_sigma_mm": 0.0,
                    "inlier_count": 0,
                    "candidate_count": 0,
                    "board_layer_top_mm": None,
                    "height_threshold_mm": 0.0,
                },
            }
        )
    )
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors=None)

    with pytest.raises(ValueError, match="colors"):
        m.measure(fusion_dir, provenance_glb, PART_X_MM, PART_Y_MM, PART_HEIGHT_MM)


def test_main_prints_a_row_per_category_and_returns_zero(tmp_path, capsys):
    vertices, faces = _triangle_mesh([CENTROIDS_BY_CASE["part"]])
    colors = np.tile(np.asarray(provenance.OBSERVED_COLOR, dtype=np.uint8), (3, 1))
    fusion_dir = tmp_path / "fusion"
    fusion_dir.mkdir()
    (fusion_dir / "tsdf_fusion.json").write_text(
        json.dumps(
            {
                "bounds_min_mm": BOUNDS_MIN_MM.tolist(),
                "bounds_max_mm": BOUNDS_MAX_MM.tolist(),
                "board_surface": {
                    "slope_x": 0.0,
                    "slope_y": 0.0,
                    "z_at_origin_mm": 0.0,
                    "offset_mm": 0.0,
                    "tilt_deg": 0.0,
                    "residual_sigma_mm": 0.0,
                    "inlier_count": 0,
                    "candidate_count": 0,
                    "board_layer_top_mm": None,
                    "height_threshold_mm": 0.0,
                },
            }
        )
    )
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    exit_code = m.main(
        [
            str(fusion_dir),
            str(provenance_glb),
            "--part-x-mm",
            "40",
            "60",
            "--part-y-mm",
            "40",
            "60",
            "--part-height-mm",
            str(PART_HEIGHT_MM),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    for category in m.CATEGORIES:
        assert category in out
