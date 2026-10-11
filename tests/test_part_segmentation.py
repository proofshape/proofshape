"""Unit tests for R-06/R-21's part segmentation. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import part_segmentation, vggt_runner
from recon.part_segmentation import BoardSurface

# --- test helpers --------------------------------------------------------------------------


def _flat_surface(
    threshold_mm: float = part_segmentation.DEFAULT_MIN_BOARD_CLEARANCE_MM,
) -> BoardSurface:
    """A trivial z=0 board surface, for tests of segment_part's own logic in isolation --
    nothing in the production code defaults to this; measure_board_surface is always fit from
    real points."""
    return BoardSurface(
        slope_x=0.0,
        slope_y=0.0,
        z_at_origin_mm=0.0,
        offset_mm=0.0,
        tilt_deg=0.0,
        residual_sigma_mm=0.0,
        inlier_count=1,
        candidate_count=1,
        board_layer_top_mm=threshold_mm,
        height_threshold_mm=threshold_mm,
    )


def _board_and_part_points(
    rng: np.random.Generator,
    offset_mm: float = 0.0,
    slope_x: float = 0.0,
    slope_y: float = 0.0,
    board_noise_mm: float = 0.0,
    board_point_count: int = 4000,
    footprint_mm: tuple[float, float] = (160.0, 120.0),
    part_center: tuple[float, float] = (80.0, 60.0),
    part_half_size_mm: float = 15.0,
    part_height_above_board_mm: float | None = 20.0,
    part_point_count: int = 500,
    second_layer_height_above_board_mm: float | None = None,
    second_layer_fraction: float = 0.2,
) -> np.ndarray:
    """Build a synthetic board-frame point cloud: a board plane, an optional box-shaped part
    sitting on it, and an optional raised second board layer (R-17's finding on s-01/s-02)."""
    width_mm, height_mm = footprint_mm

    def plane_z(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        return slope_x * x_mm + slope_y * y_mm + offset_mm

    board_x = rng.uniform(0.0, width_mm, board_point_count)
    board_y = rng.uniform(0.0, height_mm, board_point_count)
    board_z = plane_z(board_x, board_y)
    if board_noise_mm > 0:
        board_z = board_z + rng.normal(0.0, board_noise_mm, board_point_count)
    points = [np.column_stack([board_x, board_y, board_z])]

    if part_height_above_board_mm is not None:
        cx, cy = part_center
        part_x = rng.uniform(
            cx - part_half_size_mm, cx + part_half_size_mm, part_point_count
        )
        part_y = rng.uniform(
            cy - part_half_size_mm, cy + part_half_size_mm, part_point_count
        )
        part_z = plane_z(part_x, part_y) - part_height_above_board_mm
        points.append(np.column_stack([part_x, part_y, part_z]))

    if second_layer_height_above_board_mm is not None:
        layer_count = int(board_point_count * second_layer_fraction)
        layer_x = rng.uniform(0.0, width_mm, layer_count)
        layer_y = rng.uniform(0.0, height_mm, layer_count)
        layer_z = plane_z(layer_x, layer_y) - second_layer_height_above_board_mm
        points.append(np.column_stack([layer_x, layer_y, layer_z]))

    return np.vstack(points)


# --- measure_board_surface: offset and tilt ------------------------------------------------


def test_measure_board_surface_recovers_offset_below_assumed_z0():
    # offset_mm > 0 means the real board sits below z=0 (deeper into the table), per
    # board_pose.py's convention -- this is R-17's finding on all five golden objects.
    rng = np.random.default_rng(0)
    points = _board_and_part_points(rng, offset_mm=3.0, board_noise_mm=0.3)

    surface = part_segmentation.measure_board_surface(points)

    assert surface.offset_mm == pytest.approx(3.0, abs=0.2)
    assert surface.tilt_deg == pytest.approx(0.0, abs=0.2)
    assert surface.inlier_count > part_segmentation.DEFAULT_MIN_BOARD_POINTS


def test_measure_board_surface_recovers_offset_above_assumed_z0():
    rng = np.random.default_rng(1)
    points = _board_and_part_points(rng, offset_mm=-2.0, board_noise_mm=0.3)

    surface = part_segmentation.measure_board_surface(points)

    assert surface.offset_mm == pytest.approx(-2.0, abs=0.2)


def test_measure_board_surface_recovers_tilt():
    rng = np.random.default_rng(2)
    slope = np.tan(np.radians(1.0))  # a 1-degree tilt
    points = _board_and_part_points(
        rng, offset_mm=1.0, slope_x=slope, board_noise_mm=0.3
    )

    surface = part_segmentation.measure_board_surface(points)

    assert surface.tilt_deg == pytest.approx(1.0, abs=0.1)


def test_measure_board_surface_rejects_tilt_beyond_default_limit():
    rng = np.random.default_rng(3)
    slope = np.tan(np.radians(3.0))  # beyond DEFAULT_MAX_BOARD_TILT_DEG = 2.0
    points = _board_and_part_points(rng, slope_x=slope, board_noise_mm=0.2)

    with pytest.raises(ValueError, match="tilt"):
        part_segmentation.measure_board_surface(points)

    # An explicit, wider limit is still respected -- never a silent fallback.
    surface = part_segmentation.measure_board_surface(points, max_tilt_deg=5.0)
    assert surface.tilt_deg == pytest.approx(3.0, abs=0.1)


# --- measure_board_surface: part base recovered, offset by a few mm either way ---------------


def test_part_base_recovered_when_board_is_below_assumed_z0():
    # The real board sits 3 mm below z=0; the part's base is 3.5 mm above the *real* board.
    # R-06's old z=0-relative rule would have dropped this point (height-above-z0 = 0.5 mm, well
    # under the 2 mm margin); the measured-surface rule must keep it. A single point, not a
    # dense cluster: a cluster this low and this flat would itself look like a second board
    # layer, which is exactly what the next test is about.
    rng = np.random.default_rng(4)
    points = _board_and_part_points(
        rng,
        offset_mm=3.0,
        board_noise_mm=0.2,
        part_height_above_board_mm=3.5,
        part_point_count=1,
    )
    surface = part_segmentation.measure_board_surface(points)

    keep = part_segmentation.segment_part(points, surface)

    assert keep[-1]  # the part's single point, appended last by the helper


def test_part_base_recovered_when_board_tilted():
    rng = np.random.default_rng(5)
    slope = np.tan(np.radians(0.9))  # R-17's measured maximum
    points = _board_and_part_points(
        rng,
        offset_mm=2.0,
        slope_x=slope,
        board_noise_mm=0.2,
        part_height_above_board_mm=3.5,
        part_point_count=1,
    )
    surface = part_segmentation.measure_board_surface(points)

    keep = part_segmentation.segment_part(points, surface)

    assert keep[-1]


# --- measure_board_surface: a raised, noisy second board layer must stay excluded ------------


def test_second_board_layer_is_excluded_from_the_cut():
    # R-17's finding on s-01/s-02: a weaker second board layer several mm above the real one.
    # Moving the cut down to the real board without accounting for this would let it through.
    rng = np.random.default_rng(6)
    points = _board_and_part_points(
        rng,
        offset_mm=2.0,
        board_noise_mm=0.3,
        part_height_above_board_mm=15.0,
        second_layer_height_above_board_mm=5.0,
        second_layer_fraction=0.2,
    )

    surface = part_segmentation.measure_board_surface(points)
    keep = part_segmentation.segment_part(points, surface)

    # The threshold must clear the second layer (5 mm above the real board).
    assert surface.height_threshold_mm > 5.0
    board_count = (
        points.shape[0] - 500 - 800
    )  # board, part(500), second layer(0.2*4000=800)
    second_layer_kept = keep[-800:]
    assert not second_layer_kept.any()
    part_kept = keep[board_count : board_count + 500]
    assert part_kept.all()


# --- measure_board_surface: explicit errors, never a silent fallback to z=0 ------------------


def test_measure_board_surface_rejects_too_few_candidate_points():
    rng = np.random.default_rng(7)
    points = _board_and_part_points(
        rng, board_point_count=50, part_height_above_board_mm=None
    )

    with pytest.raises(ValueError, match="board points"):
        part_segmentation.measure_board_surface(points)


def test_measure_board_surface_rejects_rank_deficient_points():
    # Every candidate point shares the same x -- the fit's design matrix can't span a plane.
    rng = np.random.default_rng(8)
    y = rng.uniform(0.0, 120.0, 2000)
    x = np.full_like(y, 80.0)
    z = np.zeros_like(y)
    points = np.column_stack([x, y, z])

    with pytest.raises(ValueError, match="plane"):
        part_segmentation.measure_board_surface(points)


def test_measure_board_surface_rejects_a_board_layer_that_never_thins_out():
    # A clean board spike at z=0 (so the robust *fit* still locks onto the right plane) plus a
    # dense cluster right at the edge of the search window -- the search has no way to know
    # whether that cluster's density keeps going beyond the window, so it must refuse to guess
    # rather than silently picking the window's edge as "the top".
    rng = np.random.default_rng(9)
    board_x = rng.uniform(0.0, 160.0, 4000)
    board_y = rng.uniform(0.0, 120.0, 4000)
    board_z = rng.normal(0.0, 0.1, 4000)
    edge_x = rng.uniform(0.0, 160.0, 400)
    edge_y = rng.uniform(0.0, 120.0, 400)
    edge_z = -rng.normal(9.75, 0.03, 400)  # a dense cluster right at the search edge
    points = np.vstack(
        [
            np.column_stack([board_x, board_y, board_z]),
            np.column_stack([edge_x, edge_y, edge_z]),
        ]
    )

    with pytest.raises(ValueError, match="top wasn't found"):
        part_segmentation.measure_board_surface(points)


# --- segment_part: height threshold direction, with a trivial flat surface -------------------
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

    keep = part_segmentation.segment_part(
        points, _flat_surface(), height_threshold_mm=2.0
    )

    assert list(keep) == [False, True, False]


def test_segment_part_height_threshold_is_configurable():
    points = np.array([[80.0, 60.0, -3.0]])  # 3 mm above the board

    assert part_segmentation.segment_part(
        points, _flat_surface(), height_threshold_mm=2.0
    )[0]
    assert not part_segmentation.segment_part(
        points, _flat_surface(), height_threshold_mm=5.0
    )[0]


def test_segment_part_defaults_to_the_surface_s_measured_threshold():
    points = np.array([[80.0, 60.0, -3.0]])  # 3 mm above the board
    surface_with_low_threshold = _flat_surface(threshold_mm=2.0)
    surface_with_high_threshold = _flat_surface(threshold_mm=5.0)

    assert part_segmentation.segment_part(points, surface_with_low_threshold)[0]
    assert not part_segmentation.segment_part(points, surface_with_high_threshold)[0]


def test_segment_part_threshold_override_is_relative_to_the_surface_not_z0():
    # A board surface offset 3mm below z=0; a point 4mm above the *real* board (z=-1, which
    # z=0-relative logic would treat as only 1mm above the assumed board).
    surface = BoardSurface(
        slope_x=0.0,
        slope_y=0.0,
        z_at_origin_mm=3.0,
        offset_mm=3.0,
        tilt_deg=0.0,
        residual_sigma_mm=0.2,
        inlier_count=2000,
        candidate_count=2000,
        board_layer_top_mm=1.0,
        height_threshold_mm=2.0,
    )
    points = np.array([[80.0, 60.0, -1.0]])  # height above real board = 3 - (-1) = 4mm

    assert part_segmentation.segment_part(points, surface, height_threshold_mm=3.5)[0]
    assert not part_segmentation.segment_part(points, surface, height_threshold_mm=4.5)[
        0
    ]


def test_segment_part_requires_a_board_surface():
    points = np.array([[80.0, 60.0, -10.0]])

    with pytest.raises(TypeError, match="BoardSurface"):
        part_segmentation.segment_part(points, board_surface=None)


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

    keep = part_segmentation.segment_part(
        points, _flat_surface(), height_threshold_mm=2.0
    )

    assert list(keep) == [True, False, False, False]


def test_segment_part_footprint_is_configurable():
    points = np.array(
        [[50.0, 50.0, -10.0]]
    )  # outside a 40x40 footprint, inside a 60x60 one

    assert not part_segmentation.segment_part(
        points, _flat_surface(), footprint_mm=(40.0, 40.0)
    )[0]
    assert part_segmentation.segment_part(
        points, _flat_surface(), footprint_mm=(60.0, 60.0)
    )[0]


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

    keep = part_segmentation.segment_part(
        points, _flat_surface(), height_threshold_mm=2.0
    )

    assert list(keep) == [True, False, False]


def test_segment_part_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        part_segmentation.segment_part(np.zeros((3, 2)), _flat_surface())


def test_segment_part_rejects_non_finite_points():
    with pytest.raises(ValueError, match="finite"):
        part_segmentation.segment_part(np.array([[np.nan, 0.0, 0.0]]), _flat_surface())


def test_segment_part_rejects_non_positive_threshold():
    with pytest.raises(ValueError, match="height_threshold_mm"):
        part_segmentation.segment_part(
            np.zeros((1, 3)), _flat_surface(), height_threshold_mm=0.0
        )


def test_segment_part_rejects_non_positive_footprint():
    with pytest.raises(ValueError, match="footprint_mm"):
        part_segmentation.segment_part(
            np.zeros((1, 3)), _flat_surface(), footprint_mm=(0.0, 120.0)
        )


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
    rng = np.random.default_rng(10)
    points = _board_and_part_points(
        rng, offset_mm=1.5, board_noise_mm=0.2, part_height_above_board_mm=20.0
    ).astype(np.float32)
    colors = rng.integers(0, 255, size=(points.shape[0], 3), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(run_dir / "points.ply", points, colors)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "out"

    report = part_segmentation.segment_run(run_dir, alignment_json, out_dir)

    assert report["total_point_count"] == points.shape[0]
    assert report["kept_point_count"] == 500  # exactly the part block
    assert report["height_threshold_source"] == "measured"
    assert report["board_surface"]["offset_mm"] == pytest.approx(1.5, abs=0.3)

    kept_points, _kept_colors = vggt_runner.read_ply(out_dir / "segmented_points.ply")
    assert len(kept_points) == 500
    on_disk_report = json.loads((out_dir / "part_segmentation.json").read_text())
    assert on_disk_report == report


def test_segment_run_height_threshold_override_is_recorded(tmp_path):
    rng = np.random.default_rng(11)
    points = _board_and_part_points(
        rng, offset_mm=1.0, board_noise_mm=0.2, part_height_above_board_mm=20.0
    ).astype(np.float32)
    colors = rng.integers(0, 255, size=(points.shape[0], 3), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(run_dir / "points.ply", points, colors)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "out"

    report = part_segmentation.segment_run(
        run_dir, alignment_json, out_dir, height_threshold_mm=10.0
    )

    assert report["height_threshold_mm"] == pytest.approx(10.0)
    assert report["height_threshold_source"] == "override"


def test_segment_run_refuses_to_overwrite_existing_output(tmp_path):
    rng = np.random.default_rng(12)
    points = _board_and_part_points(rng).astype(np.float32)
    colors = rng.integers(0, 255, size=(points.shape[0], 3), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(run_dir / "points.ply", points, colors)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "out"

    part_segmentation.segment_run(run_dir, alignment_json, out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        part_segmentation.segment_run(run_dir, alignment_json, out_dir)


def test_segment_run_applies_the_alignment_transform(tmp_path):
    rng = np.random.default_rng(13)
    points = _board_and_part_points(
        rng, offset_mm=1.0, board_noise_mm=0.2, part_height_above_board_mm=20.0
    )
    # Translate the whole scene by (80, 60, -10) between reconstruction space and the board
    # frame -- the alignment transform must be applied before segmentation decides anything.
    raw_points = (points - np.array([80.0, 60.0, -10.0])).astype(np.float32)
    colors = rng.integers(0, 255, size=(points.shape[0], 3), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(run_dir / "points.ply", raw_points, colors)
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

    assert report["kept_point_count"] == 500


def test_segment_run_fails_loudly_when_the_board_cannot_be_fit(tmp_path):
    rng = np.random.default_rng(14)
    # Far too little of the board visible -- must raise, never fall back to z=0.
    points = _board_and_part_points(
        rng, board_point_count=10, part_height_above_board_mm=None
    ).astype(np.float32)
    colors = rng.integers(0, 255, size=(points.shape[0], 3), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(run_dir / "points.ply", points, colors)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="board points"):
        part_segmentation.segment_run(run_dir, alignment_json, out_dir)


# --- CLI: a fit failure is reported and exits non-zero -------------------------------------


def test_main_reports_a_board_fit_failure_and_exits_nonzero(tmp_path, capsys):
    rng = np.random.default_rng(15)
    points = _board_and_part_points(
        rng, board_point_count=10, part_height_above_board_mm=None
    ).astype(np.float32)
    colors = rng.integers(0, 255, size=(points.shape[0], 3), dtype=np.uint8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vggt_runner.write_ply(run_dir / "points.ply", points, colors)
    _write_identity_alignment(run_dir / "metric_alignment.json")

    exit_code = part_segmentation.main(
        [str(run_dir), "--out-dir", str(tmp_path / "out")]
    )

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().err
