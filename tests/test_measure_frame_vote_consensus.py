"""Unit tests for R-33 criterion 2's frame-vote-consensus script. All synthetic -- no GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from recon import mesh_cleanup, provenance, tsdf_fusion
from recon.part_segmentation import BoardSurface
from scripts import measure_frame_vote_consensus as m

ROOT = Path(__file__).resolve().parents[1]

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
BOUNDS_MIN_MM = [0.0, 0.0, -50.0]
BOUNDS_MAX_MM = [100.0, 100.0, 0.0]
PART_X_MM = (40.0, 60.0)
PART_Y_MM = (40.0, 60.0)
PART_HEIGHT_MM = 33.0


def _board_surface_dict(board: BoardSurface = FLAT_BOARD) -> dict:
    return {
        "slope_x": board.slope_x,
        "slope_y": board.slope_y,
        "z_at_origin_mm": board.z_at_origin_mm,
        "offset_mm": board.offset_mm,
        "tilt_deg": board.tilt_deg,
        "residual_sigma_mm": board.residual_sigma_mm,
        "inlier_count": board.inlier_count,
        "candidate_count": board.candidate_count,
        "board_layer_top_mm": board.board_layer_top_mm,
        "height_threshold_mm": board.height_threshold_mm,
    }


def _write_fusion_dir(
    fusion_dir,
    voxel_size_mm: float = 2.0,
    truncation_mm: float = 25.0,
    bounds_min_mm: list[float] = BOUNDS_MIN_MM,
    bounds_max_mm: list[float] = BOUNDS_MAX_MM,
    board: BoardSurface = FLAT_BOARD,
) -> None:
    fusion_dir.mkdir()
    (fusion_dir / "tsdf_fusion.json").write_text(
        json.dumps(
            {
                "bounds_min_mm": bounds_min_mm,
                "bounds_max_mm": bounds_max_mm,
                "board_surface": _board_surface_dict(board),
                "voxel_size_mm": voxel_size_mm,
                "truncation_mm": truncation_mm,
                "min_confidence": 0.0,
            }
        )
    )


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


# --- sample_points -------------------------------------------------------------------------------


def test_sample_points_separates_wall_and_part_faces(tmp_path):
    centroids = [
        (
            90.0,
            90.0,
            -25.0,
        ),  # wall: outside the part's xy, too high for skin, no edge nearby
        (85.0, 85.0, -25.0),  # wall, same reason
        (50.0, 50.0, -10.0),  # part: inside the part's xy, under the height cap
    ]
    vertices, faces = _triangle_mesh(centroids)
    # "part" samples are Observed-only (review on #281); the part face's own 3 vertices
    # (indices 6-8) are Observed, both wall faces' are not.
    colors = np.tile(np.asarray(provenance.UNOBSERVED_COLOR, dtype=np.uint8), (9, 1))
    colors[6:9] = provenance.OBSERVED_COLOR
    fusion_dir = tmp_path / "fusion"
    _write_fusion_dir(fusion_dir)
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    wall_points, part_points = m.sample_points(
        fusion_dir,
        provenance_glb,
        sample_count=10,
        seed=0,
        part_x_mm=PART_X_MM,
        part_y_mm=PART_Y_MM,
        part_height_mm=PART_HEIGHT_MM,
    )

    assert len(wall_points) == 2
    assert len(part_points) == 1
    np.testing.assert_allclose(sorted(wall_points[:, 0].tolist()), [85.0, 90.0])
    np.testing.assert_allclose(part_points[0], [50.0, 50.0, -10.0])


def test_sample_points_caps_at_sample_count_without_replacement(tmp_path):
    centroids = [(90.0 + i, 90.0, -25.0) for i in range(5)]  # 5 distinct wall faces
    vertices, faces = _triangle_mesh(centroids)
    colors = np.tile(np.asarray(provenance.UNOBSERVED_COLOR, dtype=np.uint8), (15, 1))
    fusion_dir = tmp_path / "fusion"
    _write_fusion_dir(fusion_dir)
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    wall_points, part_points = m.sample_points(
        fusion_dir,
        provenance_glb,
        sample_count=2,
        seed=0,
        part_x_mm=PART_X_MM,
        part_y_mm=PART_Y_MM,
        part_height_mm=PART_HEIGHT_MM,
    )

    assert len(wall_points) == 2
    assert len(part_points) == 0
    assert len(set(wall_points[:, 0].tolist())) == 2  # no duplicate sampled twice


# --- summarize_votes -----------------------------------------------------------------------------


def test_summarize_votes_splits_by_group_and_takes_the_median():
    votes = [
        {
            "contributing_frame_count": 1,
            "near_surface_frame_count": 0,
            "in_front_frame_count": 1,
            "behind_frame_count": 0,
        },
        {
            "contributing_frame_count": 1,
            "near_surface_frame_count": 0,
            "in_front_frame_count": 0,
            "behind_frame_count": 1,
        },
        {
            "contributing_frame_count": 10,
            "near_surface_frame_count": 10,
            "in_front_frame_count": 0,
            "behind_frame_count": 0,
        },
    ]

    summary = m.summarize_votes(votes, wall_count=2)

    assert summary["wall"]["contributing_frame_count"] == 1.0
    assert summary["wall"]["near_surface_frame_count"] == 0.0
    assert summary["part"]["contributing_frame_count"] == 10.0
    assert summary["part"]["near_surface_frame_count"] == 10.0


def test_summarize_votes_handles_an_empty_group():
    summary = m.summarize_votes([], wall_count=0)

    for group in ("wall", "part"):
        for key in m.VOTE_KEYS:
            assert summary[group][key] == 0.0


# --- measure: end to end ---------------------------------------------------------------------


# A real board plane needs >=1000 candidate points to fit (R-21), so the dense-board scene below
# borrows the rotation-flip trick `test_tsdf_fusion.py`'s own fuse_run tests use: with rotation
# diag(1,-1,-1), zero translation and intrinsic fx=fy=10, cx=0, cy=120, a pixel (v, u) at depth D
# unprojects to board-frame (x, y, z) = (u*D/10, (120-v)*D/10, -D). A dense grid of pixels at one
# depth gives a dense flat board; two extra pixels, set after the grid, are this test's own
# "wall" and "part" sample points, computed from the same formula run in reverse: D = -z,
# u = 10*x/D, v = 120 - 10*y/D.
_DENSE_ROTATION = np.diag([1.0, -1.0, -1.0])
_DENSE_EXTRINSIC = np.concatenate([_DENSE_ROTATION, np.zeros((3, 1))], axis=1)[None]
_DENSE_INTRINSIC = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 120.0], [0.0, 0.0, 1.0]])[
    None
]


def _write_dense_reconstruction_run(run_dir, extra_points_board_frame: list[tuple]):
    """A (1, 130, 170) depth image: a dense flat board (z=-10, well over 1000 candidates), a
    100-pixel raised patch (z=-20, clear of the board and of `extra_points_board_frame`'s own
    pixels, solely so `segment_part`'s own min_kept_points=50 default is satisfied), and one
    pixel per `extra_points_board_frame` point, each set to exactly that point's own depth so
    the one frame here votes it `near_surface`."""
    depth = np.zeros((1, 130, 170))
    u_grid = np.arange(0, 161)
    v_grid = np.arange(0, 121)
    depth[0, v_grid[:, None], u_grid[None, :]] = 10.0  # board at board-frame z=-10
    # Raised 10x10 patch, inside the footprint (unlike a block near the image's own top-left,
    # whose unprojected y would fall outside [0, 120]) and away from every sample pixel below.
    depth[0, 100:110, 10:20] = 20.0
    for x, y, z in extra_points_board_frame:
        d = -z
        u = round(10.0 * x / d)
        v = round(120.0 - 10.0 * y / d)
        depth[0, v, u] = d
    return _write_reconstruction_run_with_camera(
        run_dir, depth, _DENSE_EXTRINSIC, _DENSE_INTRINSIC
    )


def _write_reconstruction_run_with_camera(run_dir, depth_mm, extrinsic, intrinsic):
    run_dir.mkdir()
    frame_count = depth_mm.shape[0]
    np.savez_compressed(
        run_dir / "poses.npz",
        frame_names=np.array([f"frame_{i}.jpg" for i in range(frame_count)]),
        extrinsic=np.repeat(extrinsic, frame_count, axis=0).astype(np.float32),
        intrinsic=np.repeat(intrinsic, frame_count, axis=0).astype(np.float32),
    )
    np.savez_compressed(
        run_dir / "depth.npz",
        depth=depth_mm.astype(np.float32),
        depth_conf=np.ones_like(depth_mm, dtype=np.float32),
    )
    (run_dir / "run.json").write_text(json.dumps({"min_confidence": 0.0}))
    alignment_json = run_dir / "metric_alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "scale": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation_mm": [0.0, 0.0, 0.0],
            }
        )
    )
    return alignment_json


def test_measure_reruns_fusion_and_reports_wall_versus_part_votes(tmp_path):
    # Point A (140, 100, -40), height ~30 above the real board (~z=-10): outside the part box
    # below -> "wall". Point B (80, 60, -20), height ~10: inside it -> "part". This test is
    # about the wiring (the right points reach the right group, each carrying a real vote from
    # the one frame that measured it), not the wall/part asymmetry itself, which needs real
    # multi-frame golden data to show.
    wall_point = (140.0, 100.0, -40.0)
    part_point = (80.0, 60.0, -20.0)
    run_dir = tmp_path / "run"
    alignment_json = _write_dense_reconstruction_run(run_dir, [wall_point, part_point])

    vertices, faces = _triangle_mesh([wall_point, part_point])
    # "part" samples are Observed-only (review on #281); part_point's own 3 vertices (indices
    # 3-5) are Observed, wall_point's are not.
    colors = np.tile(np.asarray(provenance.UNOBSERVED_COLOR, dtype=np.uint8), (6, 1))
    colors[3:6] = provenance.OBSERVED_COLOR
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    # A genuine first fuse_run, not a hand-written tsdf_fusion.json -- so its real face_count/
    # vertex_count are there for measure()'s own reproduction check to compare its rerun against
    # (review on #281, @TabeenRaoof). height_threshold_mm must be positive (segment_part's own
    # rule); 1.0mm clears both sample points' real height above the dense board (~30 and ~10mm)
    # with plenty of margin.
    fusion_dir = tmp_path / "fusion"
    tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        fusion_dir,
        voxel_size_mm=2.0,
        truncation_mm=25.0,
        height_threshold_mm=1.0,
        min_confidence=0.0,
    )

    summary = m.measure(
        run_dir,
        alignment_json,
        fusion_dir,
        provenance_glb,
        tmp_path / "vote_out",
        sample_count=10,
        seed=0,
        part_x_mm=(70.0, 90.0),
        part_y_mm=(50.0, 70.0),
        part_height_mm=33.0,
    )

    assert summary["wall"]["contributing_frame_count"] == 1.0
    assert summary["wall"]["near_surface_frame_count"] == 1.0
    assert summary["wall"]["near_surface_le_1_share"] == 1.0
    assert summary["part"]["contributing_frame_count"] == 1.0
    assert summary["part"]["near_surface_frame_count"] == 1.0
    assert (tmp_path / "vote_out" / "vote_consensus.json").exists()

    results = json.loads(
        (tmp_path / "vote_out" / "wall_vs_part_votes.json").read_text()
    )
    assert results["summary"] == summary
    assert [sample["group"] for sample in results["samples"]] == ["wall", "part"]
    assert results["samples"][0]["point_mm"] == list(wall_point)
    assert results["samples"][1]["point_mm"] == list(part_point)


def test_measure_refuses_when_the_rerun_does_not_reproduce_the_first_fusion(tmp_path):
    # Review on #281, @TabeenRaoof: min_kept_points/max_part_height_mm/max_board_tilt_deg/
    # footprint_mm aren't recorded in tsdf_fusion.json, so a first run that didn't leave one of
    # them at its default silently gets a different rerun -- simulated here by corrupting the
    # recorded face_count after a real first run, standing in for that drift.
    wall_point = (140.0, 100.0, -40.0)
    part_point = (80.0, 60.0, -20.0)
    run_dir = tmp_path / "run"
    alignment_json = _write_dense_reconstruction_run(run_dir, [wall_point, part_point])

    vertices, faces = _triangle_mesh([wall_point, part_point])
    colors = np.tile(np.asarray(provenance.UNOBSERVED_COLOR, dtype=np.uint8), (6, 1))
    colors[3:6] = provenance.OBSERVED_COLOR
    provenance_glb = tmp_path / "mesh_provenance.glb"
    mesh_cleanup.write_glb(provenance_glb, vertices, faces, colors)

    fusion_dir = tmp_path / "fusion"
    tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        fusion_dir,
        voxel_size_mm=2.0,
        truncation_mm=25.0,
        height_threshold_mm=1.0,
        min_confidence=0.0,
    )
    report_path = fusion_dir / "tsdf_fusion.json"
    report = json.loads(report_path.read_text())
    report["face_count"] += 1
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="rerun"):
        m.measure(
            run_dir,
            alignment_json,
            fusion_dir,
            provenance_glb,
            tmp_path / "vote_out",
            sample_count=10,
            seed=0,
            part_x_mm=(70.0, 90.0),
            part_y_mm=(50.0, 70.0),
            part_height_mm=33.0,
        )


def test_main_module_runs_as_a_subprocess_with_minus_m(tmp_path):
    # Review on #281, @TabeenRaoof: the documented `python scripts/measure_frame_vote_
    # consensus.py` form fails with ModuleNotFoundError on an install where only `recon` and
    # `inspection` are importable and the repository root isn't on sys.path, because this
    # script (unlike measure_wall_share.py) imports from the `scripts` package itself, which has
    # no __init__.py to put the repo root there. `-m`, run from the repository root, does.
    result = subprocess.run(
        [sys.executable, "-m", "scripts.measure_frame_vote_consensus", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "RECONSTRUCTION_RUN_DIR" in result.stdout
        or "reconstruction_run_dir" in result.stdout
    )
