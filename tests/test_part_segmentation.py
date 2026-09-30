"""Unit tests for R-06's part segmentation. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import part_segmentation, vggt_runner

# --- segment_part: height threshold direction ------------------------------------------------
# board_pose.py's z convention: origin on the paper, z = 0 there, z increasing *into* the
# table, so a camera (and anything else) above the board has negative z. Height above the board
# is therefore -z. These three points pin that direction explicitly so it can never silently
# flip: on the board (z=0), 5mm above it (z=-5), and 5mm "below" it (z=+5, never physically real
# but must still be excluded rather than accidentally kept).


def test_segment_part_keeps_points_above_board_by_negative_z():
    points = np.array(
        [
            [80.0, 60.0, 0.0],  # on the board plane -- not above it
            [80.0, 60.0, -5.0],  # 5 mm above the board
            [80.0, 60.0, 5.0],  # "5 mm into the table" -- must not be kept
        ]
    )

    keep = part_segmentation.segment_part(points, height_threshold_mm=2.0)

    assert list(keep) == [False, True, False]


def test_segment_part_height_threshold_is_configurable():
    points = np.array([[80.0, 60.0, -3.0]])  # 3 mm above the board

    assert part_segmentation.segment_part(points, height_threshold_mm=2.0)[0]
    assert not part_segmentation.segment_part(points, height_threshold_mm=5.0)[0]


# --- segment_part: footprint -------------------------------------------------------------------


def test_segment_part_footprint_keeps_points_inside_pattern_rectangle():
    # All safely above the board (z=-10) so only the footprint check varies.
    points = np.array(
        [
            [80.0, 60.0, -10.0],  # inside the 160x120 pattern rectangle
            [-5.0, 60.0, -10.0],  # outside: x < 0
            [200.0, 60.0, -10.0],  # outside: x beyond 160
            [80.0, 150.0, -10.0],  # outside: y beyond 120
        ]
    )

    keep = part_segmentation.segment_part(points, height_threshold_mm=2.0)

    assert list(keep) == [True, False, False, False]


def test_segment_part_footprint_is_configurable():
    points = np.array(
        [[50.0, 50.0, -10.0]]
    )  # outside a 40x40 footprint, inside a 60x60 one

    assert not part_segmentation.segment_part(points, footprint_mm=(40.0, 40.0))[0]
    assert part_segmentation.segment_part(points, footprint_mm=(60.0, 60.0))[0]


def test_segment_part_default_footprint_matches_printed_pattern():
    assert part_segmentation.DEFAULT_FOOTPRINT_MM == (160.0, 120.0)


# --- segment_part: combined predicate and validation --------------------------------------------


def test_segment_part_requires_both_height_and_footprint():
    points = np.array(
        [
            [80.0, 60.0, -10.0],  # above board, inside footprint -- kept
            [80.0, 60.0, 0.0],  # inside footprint, not above board -- dropped
            [200.0, 60.0, -10.0],  # above board, outside footprint -- dropped
        ]
    )

    keep = part_segmentation.segment_part(points)

    assert list(keep) == [True, False, False]


def test_segment_part_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        part_segmentation.segment_part(np.zeros((3, 2)))


def test_segment_part_rejects_non_finite_points():
    with pytest.raises(ValueError, match="finite"):
        part_segmentation.segment_part(np.array([[np.nan, 0.0, 0.0]]))


def test_segment_part_rejects_non_positive_threshold():
    with pytest.raises(ValueError, match="height_threshold_mm"):
        part_segmentation.segment_part(np.zeros((1, 3)), height_threshold_mm=0.0)


def test_segment_part_rejects_non_positive_footprint():
    with pytest.raises(ValueError, match="footprint_mm"):
        part_segmentation.segment_part(np.zeros((1, 3)), footprint_mm=(0.0, 120.0))


# --- load_alignment ------------------------------------------------------------------------


def test_load_alignment_reads_scale_rotation_translation(tmp_path):
    path = tmp_path / "metric_alignment.json"
    path.write_text(
        json.dumps(
            {
                "scale": 2.5,
                "rotation": np.eye(3).tolist(),
                "translation_mm": [1.0, 2.0, 3.0],
            }
        )
    )

    alignment = part_segmentation.load_alignment(path)

    assert alignment["scale"] == pytest.approx(2.5)
    np.testing.assert_allclose(alignment["rotation"], np.eye(3))
    np.testing.assert_allclose(alignment["translation"], [1.0, 2.0, 3.0])


def test_load_alignment_rejects_missing_field(tmp_path):
    path = tmp_path / "metric_alignment.json"
    path.write_text(json.dumps({"scale": 1.0, "rotation": np.eye(3).tolist()}))

    with pytest.raises(ValueError, match="translation_mm"):
        part_segmentation.load_alignment(path)


# --- segment_run: end to end on synthetic files -------------------------------------------------


def _write_identity_alignment(path) -> None:
    path.write_text(
        json.dumps(
            {
                "scale": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation_mm": [0.0, 0.0, 0.0],
            }
        )
    )


def test_segment_run_writes_filtered_cloud_and_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    points = np.array(
        [
            [80.0, 60.0, -10.0],  # kept: above board, inside footprint
            [80.0, 60.0, 0.0],  # dropped: on the board
            [200.0, 60.0, -10.0],  # dropped: outside footprint
        ],
        dtype=np.float32,
    )
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    vggt_runner.write_ply(run_dir / "points.ply", points, colors)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "out"

    report = part_segmentation.segment_run(run_dir, alignment_json, out_dir)

    assert report["total_point_count"] == 3
    assert report["kept_point_count"] == 1
    assert report["removed_point_count"] == 2
    assert report["height_threshold_mm"] == pytest.approx(
        part_segmentation.DEFAULT_HEIGHT_THRESHOLD_MM
    )

    kept_points, kept_colors = vggt_runner.read_ply(out_dir / "segmented_points.ply")
    np.testing.assert_allclose(kept_points, [[80.0, 60.0, -10.0]])
    np.testing.assert_array_equal(kept_colors, [[10, 20, 30]])

    on_disk_report = json.loads((out_dir / "part_segmentation.json").read_text())
    assert on_disk_report == report


def test_segment_run_refuses_to_overwrite_existing_output(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(
        run_dir / "points.ply",
        np.array([[80.0, 60.0, -10.0]], dtype=np.float32),
        np.array([[1, 2, 3]], dtype=np.uint8),
    )
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "out"

    part_segmentation.segment_run(run_dir, alignment_json, out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        part_segmentation.segment_run(run_dir, alignment_json, out_dir)


def test_segment_run_applies_the_alignment_transform(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A point at (0,0,0) in reconstruction space, scaled/rotated/translated onto the board
    # frame -- must land at (80, 60, -10) *before* segmentation decides anything.
    vggt_runner.write_ply(
        run_dir / "points.ply",
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1, 2, 3]], dtype=np.uint8),
    )
    alignment_json = run_dir / "metric_alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "scale": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation_mm": [80.0, 60.0, -10.0],
            }
        )
    )
    out_dir = tmp_path / "out"

    report = part_segmentation.segment_run(run_dir, alignment_json, out_dir)

    assert report["kept_point_count"] == 1
    kept_points, _ = vggt_runner.read_ply(out_dir / "segmented_points.ply")
    np.testing.assert_allclose(kept_points, [[80.0, 60.0, -10.0]])
