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
BOUNDS_MAX_MM = np.array([100.0, 100.0, 0.0])
PART_X_MM = (40.0, 60.0)
PART_Y_MM = (40.0, 60.0)

# label: centroid (x, y, z) -> expected category. Covers every category, and the two priority
# interactions the module docstring claims (part overrides skin; volume_boundary overrides both).
CENTROIDS_BY_CASE = {
    "wall": (
        90.0,
        90.0,
        -25.0,
    ),  # outside the part's xy, too high for skin, no edge nearby
    "board_skin": (10.0, 10.0, -5.0),  # low, outside the part's xy, no edge nearby
    "part": (50.0, 50.0, -10.0),  # inside the part's xy, under the part height cap
    "volume_boundary_tall": (50.0, 50.0, -48.0),  # too tall for "part"; touches z=-50
    "part_overrides_skin": (
        50.0,
        50.0,
        -5.0,
    ),  # low enough for skin AND in the part's xy
    "boundary_overrides_part": (50.0, 50.0, -1.0),  # low, in the part's xy, touches z=0
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


def test_classify_faces_covers_every_category_and_both_priority_overrides():
    labels = list(CENTROIDS_BY_CASE)
    vertices, faces = _triangle_mesh([CENTROIDS_BY_CASE[label] for label in labels])

    categories = m.classify_faces(
        vertices, faces, FLAT_BOARD, BOUNDS_MIN_MM, BOUNDS_MAX_MM, PART_X_MM, PART_Y_MM
    )

    expected = {
        "wall": m.CATEGORY_WALL,
        "board_skin": m.CATEGORY_BOARD_SKIN,
        "part": m.CATEGORY_PART,
        "volume_boundary_tall": m.CATEGORY_VOLUME_BOUNDARY,
        "part_overrides_skin": m.CATEGORY_PART,
        "boundary_overrides_part": m.CATEGORY_VOLUME_BOUNDARY,
    }
    for label, category in zip(labels, categories, strict=True):
        assert category == expected[label], label


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

    summary = m.measure(fusion_dir, provenance_glb, PART_X_MM, PART_Y_MM)

    assert summary[m.CATEGORY_PART]["face_count"] == 1
    assert summary[m.CATEGORY_PART]["observed_share"] == pytest.approx(1.0)
    assert summary[m.CATEGORY_BOARD_SKIN]["face_count"] == 1
    assert summary[m.CATEGORY_BOARD_SKIN]["observed_share"] == pytest.approx(0.0)
    assert summary[m.CATEGORY_WALL]["face_count"] == 1


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
        m.measure(fusion_dir, provenance_glb, PART_X_MM, PART_Y_MM)


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
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    for category in m.CATEGORIES:
        assert category in out
