"""Unit tests for R-20's confidence-threshold sweep. All synthetic data -- no GPU, no golden
capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import confidence_sweep

# --- stray_point_fraction --------------------------------------------------------------------


def test_stray_point_fraction_box_plus_scattered_strays():
    # Two adjacent voxels (voxel_size_mm=1.0), each with exactly the minimum 3 points -- a small
    # "box". Two extra points, each alone in its own voxel elsewhere: with min_points_per_voxel=3
    # a lone point's voxel never becomes "occupied" at all, so it's stray by construction, not
    # just by losing a tiebreak against a bigger cluster.
    box = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],  # voxel (0,0,0)
            [1.1, 0.1, 0.1],
            [1.2, 0.2, 0.2],
            [1.3, 0.3, 0.3],  # voxel (1,0,0), face-adjacent to (0,0,0)
        ]
    )
    strays = np.array([[10.5, 10.5, 10.5], [20.5, 20.5, 20.5]])
    points = np.concatenate([box, strays], axis=0)

    result = confidence_sweep.stray_point_fraction(
        points, voxel_size_mm=1.0, min_points_per_voxel=3
    )

    assert result["part_point_count"] == 6
    assert result["stray_point_count"] == 2
    assert result["stray_fraction"] == pytest.approx(2 / 8)
    assert result["part_voxel_keys"] == frozenset({(0, 0, 0), (1, 0, 0)})


def test_stray_point_fraction_smaller_cluster_counts_as_stray():
    # Two separated clusters, both individually "occupied" (>= 3 points each), one twice the
    # size of the other. The smaller one is not noise by any per-voxel measure -- it only
    # becomes stray because it loses to the bigger cluster.
    big_cluster = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],  # voxel (0,0,0)
            [1.1, 0.1, 0.1],
            [1.2, 0.2, 0.2],
            [1.3, 0.3, 0.3],  # voxel (1,0,0)
        ]
    )
    small_cluster = np.array(
        [
            [50.1, 0.1, 0.1],
            [50.2, 0.2, 0.2],
            [50.3, 0.3, 0.3],
        ]  # voxel (50,0,0), isolated
    )
    points = np.concatenate([big_cluster, small_cluster], axis=0)

    result = confidence_sweep.stray_point_fraction(
        points, voxel_size_mm=1.0, min_points_per_voxel=3
    )

    assert result["part_point_count"] == 6
    assert result["stray_point_count"] == 3
    assert result["stray_fraction"] == pytest.approx(3 / 9)
    assert result["part_voxel_keys"] == frozenset({(0, 0, 0), (1, 0, 0)})


def test_stray_point_fraction_reports_cluster_extents():
    # height above the board is -z (recon.part_segmentation's convention); a part 20mm tall at
    # board-frame z=-20 should report top_height_mm=20, not -20.
    points = np.array(
        [
            [0.0, 0.0, -20.0],
            [0.0, 0.0, -20.0],
            [0.0, 0.0, -20.0],
            [10.0, 5.0, -20.0],
            [10.0, 5.0, -20.0],
            [10.0, 5.0, -20.0],
        ]
    )

    result = confidence_sweep.stray_point_fraction(
        points, voxel_size_mm=20.0, min_points_per_voxel=3
    )

    assert result["x_span_mm"] == pytest.approx(10.0)
    assert result["y_span_mm"] == pytest.approx(5.0)
    assert result["top_height_mm"] == pytest.approx(20.0)


def test_stray_point_fraction_empty_input_is_nan_not_an_error():
    result = confidence_sweep.stray_point_fraction(np.empty((0, 3)))

    assert np.isnan(result["stray_fraction"])
    assert result["part_voxel_keys"] == frozenset()
    assert result["total_point_count"] == 0


def test_stray_point_fraction_nothing_reaches_min_points_is_all_stray():
    points = np.array(
        [[0.1, 0.1, 0.1], [50.1, 50.1, 50.1]]
    )  # one point per voxel, each alone

    result = confidence_sweep.stray_point_fraction(
        points, voxel_size_mm=1.0, min_points_per_voxel=3
    )

    assert result["stray_fraction"] == 1.0
    assert result["part_point_count"] == 0
    assert result["part_voxel_keys"] == frozenset()


def test_stray_point_fraction_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(N,3\)"):
        confidence_sweep.stray_point_fraction(np.zeros((4, 2)))


def test_stray_point_fraction_rejects_non_positive_voxel_size():
    with pytest.raises(ValueError, match="voxel_size_mm"):
        confidence_sweep.stray_point_fraction(np.zeros((3, 3)), voxel_size_mm=0.0)


def test_stray_point_fraction_rejects_non_positive_min_points():
    with pytest.raises(ValueError, match="min_points_per_voxel"):
        confidence_sweep.stray_point_fraction(np.zeros((3, 3)), min_points_per_voxel=0)


# --- part_voxel_coverage ----------------------------------------------------------------------


def test_part_voxel_coverage_full_and_partial():
    baseline = frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)})

    assert confidence_sweep.part_voxel_coverage(baseline, baseline) == 1.0
    assert confidence_sweep.part_voxel_coverage(
        frozenset({(0, 0, 0)}), baseline
    ) == pytest.approx(1 / 3)
    assert confidence_sweep.part_voxel_coverage(frozenset(), baseline) == 0.0


def test_part_voxel_coverage_nan_when_baseline_empty():
    assert np.isnan(
        confidence_sweep.part_voxel_coverage(frozenset({(0, 0, 0)}), frozenset())
    )


# --- sweep_run: end-to-end wiring on synthetic files -------------------------------------------
#
# One frame, three pixels. A and B board-frame points land in adjacent voxels (a 2-voxel
# cluster); C lands far away (its own, smaller cluster, i.e. a stray point at threshold 0). A, B,
# C carry confidences 5.0, 1.0 and 0.5. All three are 20mm above the board (z=-20mm), well inside
# the default footprint, so only confidence -- not height or footprint -- decides what survives
# at each threshold. The camera is rotation=diag([1,-1,-1]) (board-frame "above the board" is
# camera-frame "in front", same convention tsdf_fusion's own tests use), identity alignment, so
# board-frame coordinates can be chosen directly and the required pixel/depth values solved for.
#
# R-21: sweep_run now always fits a real board surface (measure_board_surface), which needs at
# least DEFAULT_MIN_BOARD_POINTS candidate points -- A/B/C alone, as R-20's original tests used,
# is far too few. _write_synthetic_sweep_run also writes a flat board layer at board-frame
# z=-14mm (within the 15mm search band), at this same camera's scale, well separated in height
# from A/B/C (z=-20mm, 6mm above the fitted board -- comfortably above the measured 2mm-floor
# threshold) so it never interferes with their own segmentation or clustering.


def _write_synthetic_sweep_run(run_dir, points_confidences, image_hw=(50, 120)):
    """points_confidences: list of (x_mm, y_mm, z_mm, confidence, pixel_row, pixel_col)."""
    run_dir.mkdir()
    height, width = image_hw
    depth = np.zeros((1, height, width))
    depth_conf = np.zeros((1, height, width))

    # Flat board layer: board-frame z=-14mm for every pixel in this block (u in [50,115],
    # v in [0,25] at this camera's f=10/cx=50/cy=25 scale -> x in [0,91], y in [0,35], both well
    # inside the default 160x120mm footprint). 66*26=1716 points, comfortably over
    # DEFAULT_MIN_BOARD_POINTS (1000).
    board_depth_mm = 14.0
    depth[0, 0:26, 50:116] = board_depth_mm

    for _x, _y, z_mm, confidence, row, col in points_confidences:
        depth[
            0, row, col
        ] = -z_mm  # camera-frame Z for rotation=diag([1,-1,-1]) is -z_board
        depth_conf[0, row, col] = confidence

    rotation = np.diag([1.0, -1.0, -1.0])
    extrinsic = np.concatenate([rotation, np.zeros((3, 1))], axis=1)[None]
    intrinsic = np.array([[10.0, 0.0, 50.0], [0.0, 10.0, 25.0], [0.0, 0.0, 1.0]])[None]

    np.savez_compressed(
        run_dir / "poses.npz",
        frame_names=np.array(["frame_0.jpg"]),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
    )
    np.savez_compressed(
        run_dir / "depth.npz",
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
    )


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


# Board point (x, y, -20), pixel (u, v) solved from u = f*x/Z_cam + cx, v = f*(-y)/Z_cam + cy
# with f=10, cx=50, cy=25, Z_cam = -z_board = 20 for all three.
_POINT_A = (10.0, 10.0, -20.0, 5.0, 20, 55)  # voxel (5,5,-10) at voxel_size_mm=2.0
_POINT_B = (
    12.0,
    10.0,
    -20.0,
    1.0,
    20,
    56,
)  # voxel (6,5,-10): face-adjacent to A's voxel
_POINT_C = (80.0, 10.0, -20.0, 0.5, 20, 90)  # voxel (40,5,-10): far from A/B


def test_sweep_run_confidence_drops_a_stray_without_hurting_coverage(tmp_path):
    run_dir = tmp_path / "run"
    _write_synthetic_sweep_run(run_dir, [_POINT_A, _POINT_B, _POINT_C])
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = confidence_sweep.sweep_run(
        run_dir, alignment_json, thresholds=[1.0], min_points_per_voxel=1
    )

    by_threshold = {row["min_confidence"]: row for row in report["thresholds"]}
    assert list(by_threshold) == [0.0, 1.0]  # baseline always first

    # Baseline (min_confidence 0): C is a stray point outside A/B's 2-voxel cluster.
    assert by_threshold[0.0]["stray_fraction"] == pytest.approx(1 / 3)
    assert by_threshold[0.0]["part_voxel_coverage"] == 1.0  # baseline against itself

    # min_confidence 1.0 drops C (0.5 < 1.0) but keeps A and B (5.0, 1.0 both >= 1.0): the
    # stray point is gone and the part cluster is exactly as covered as the baseline.
    assert by_threshold[1.0]["stray_fraction"] == 0.0
    assert by_threshold[1.0]["part_voxel_coverage"] == 1.0


def test_sweep_run_overshoot_drops_coverage_even_though_stray_fraction_looks_clean(
    tmp_path,
):
    # The real failure mode a threshold sweep must catch: pushed high enough, the threshold
    # starts cutting into the part's own low-confidence points, not just the background strays.
    # Once B is gone, there is nothing left to call "stray" either -- stray_fraction alone would
    # read as a clean 0.0, which is exactly why part_voxel_coverage is tracked separately.
    run_dir = tmp_path / "run"
    _write_synthetic_sweep_run(run_dir, [_POINT_A, _POINT_B, _POINT_C])
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = confidence_sweep.sweep_run(
        run_dir, alignment_json, thresholds=[1.5], min_points_per_voxel=1
    )

    by_threshold = {row["min_confidence"]: row for row in report["thresholds"]}
    # B's confidence is exactly 1.0, below 1.5, so only A survives.
    assert by_threshold[1.5]["stray_fraction"] == 0.0
    assert by_threshold[1.5]["part_voxel_coverage"] == pytest.approx(0.5)


def test_sweep_run_excludes_invalid_depth_pixel_even_at_high_confidence(tmp_path):
    # Caught in review (dalwalyk): unproject_depth_to_world maps an invalid (<=0) depth pixel to
    # the camera centre regardless of its pixel coordinates, since camera_point = ray * depth is
    # zero whatever the ray's direction. fuse_run never lets such a pixel reach segment_part --
    # it gates on valid_depth first -- so sweep_run must too, or a stray high-confidence pixel
    # whose camera centre happens to land inside the footprint/height window gets silently
    # counted as real geometry, contradicting the "measures exactly what fuse_run does" premise
    # the sweep's evidence (D-045) relies on.
    #
    # A second frame supplies the spurious pixel: its whole depth map is 0 (invalid) except one
    # pixel's confidence, which is cranked up. Every zero-depth pixel in that frame maps to that
    # frame's own camera centre, so the frame's translation is chosen to land that centre at
    # board-frame (10, 50, -20) -- inside the default 160x120mm footprint and the same 20mm
    # board height A/B/C already use, well clear of their own voxels.
    run_dir = tmp_path / "run"
    _write_synthetic_sweep_run(
        run_dir, [_POINT_A]
    )  # frame 0: board layer + the one real point
    height, width = 50, 120

    rotation = np.diag([1.0, -1.0, -1.0])
    bogus_camera_centre_board = np.array([10.0, 50.0, -20.0])
    # points = -translation @ rotation for a depth=0 pixel (see unproject_depth_to_world), and
    # rotation is self-inverse (a +-1 diagonal), so translation = -bogus_point @ rotation.
    bogus_translation = -(bogus_camera_centre_board @ rotation)
    bogus_extrinsic = np.concatenate(
        [rotation, bogus_translation.reshape(3, 1)], axis=1
    )

    with np.load(run_dir / "poses.npz") as poses:
        extrinsic = np.concatenate(
            [poses["extrinsic"], bogus_extrinsic[None].astype(np.float32)], axis=0
        )
        intrinsic = np.concatenate([poses["intrinsic"], poses["intrinsic"]], axis=0)
    with np.load(run_dir / "depth.npz") as depth_data:
        depth = np.concatenate(
            [depth_data["depth"], np.zeros((1, height, width), dtype=np.float32)],
            axis=0,
        )
        depth_conf = depth_data["depth_conf"]
        bogus_conf = np.zeros((1, height, width), dtype=np.float32)
        bogus_conf[0, 0, 0] = 5.0  # pixel choice is arbitrary -- depth=0 ignores it
        depth_conf = np.concatenate([depth_conf, bogus_conf], axis=0)

    np.savez_compressed(
        run_dir / "poses.npz",
        frame_names=np.array(["frame_0.jpg", "frame_1.jpg"]),
        extrinsic=extrinsic,
        intrinsic=intrinsic,
    )
    np.savez_compressed(run_dir / "depth.npz", depth=depth, depth_conf=depth_conf)

    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = confidence_sweep.sweep_run(
        run_dir, alignment_json, thresholds=[1.0], min_points_per_voxel=1
    )

    by_threshold = {row["min_confidence"]: row for row in report["thresholds"]}
    # Only A (confidence 5.0) should clear threshold 1.0 and reach segmentation -- the bogus
    # pixel's confidence (also 5.0) clears it too, but its depth is invalid, so it must never
    # reach segment_part regardless of confidence. Without the valid_depth gate, this would be 2.
    assert by_threshold[1.0]["segmented_point_count"] == 1


def test_sweep_run_orders_thresholds_with_baseline_first_regardless_of_input_order(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _write_synthetic_sweep_run(run_dir, [_POINT_A, _POINT_B, _POINT_C])
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = confidence_sweep.sweep_run(
        run_dir, alignment_json, thresholds=[1.5, 0.0, 1.0], min_points_per_voxel=1
    )

    assert [row["min_confidence"] for row in report["thresholds"]] == [0.0, 1.0, 1.5]


# --- main / CLI -------------------------------------------------------------------------------


def test_main_refuses_to_overwrite_existing_output(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_synthetic_sweep_run(run_dir, [_POINT_A, _POINT_B, _POINT_C])
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_path = run_dir / "confidence_sweep.json"
    out_path.write_text("{}")

    exit_code = confidence_sweep.main([str(run_dir)])

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err


def test_main_writes_report_and_prints_a_table(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_synthetic_sweep_run(run_dir, [_POINT_A, _POINT_B, _POINT_C])
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    exit_code = confidence_sweep.main(
        [str(run_dir), "--thresholds", "1.0", "--voxel-size-mm", "2.0"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "min_confidence=0.0" in out
    on_disk = json.loads((run_dir / "confidence_sweep.json").read_text())
    assert [row["min_confidence"] for row in on_disk["thresholds"]] == [0.0, 1.0]
