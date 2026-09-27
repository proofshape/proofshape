from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from recon import camera_intrinsics
from recon.board_pose import make_board


def _write_board_session(
    out_dir: Path,
    frame_names: list[str],
    image_size_hw: tuple[int, int],
    corners_by_frame: list[tuple[np.ndarray, np.ndarray]],
    provisional_k: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True)
    frame_index_parts: list[np.ndarray] = []
    corner_id_parts: list[np.ndarray] = []
    corner_px_parts: list[np.ndarray] = []

    for frame_index, (corner_ids, corner_px) in enumerate(corners_by_frame):
        frame_index_parts.append(np.full(len(corner_ids), frame_index, dtype=np.int32))
        corner_id_parts.append(corner_ids.astype(np.int32))
        corner_px_parts.append(corner_px.astype(np.float64))

    np.savez_compressed(
        out_dir / "board_poses.npz",
        frame_names=np.asarray(frame_names),
        image_size_hw=np.tile(
            np.asarray(image_size_hw, dtype=np.int32),
            (len(frame_names), 1),
        ),
        intrinsic=np.tile(provisional_k, (len(frame_names), 1, 1)),
        corner_frame_index=np.concatenate(frame_index_parts),
        corner_id=np.concatenate(corner_id_parts),
        corner_px=np.concatenate(corner_px_parts),
    )


def _synthetic_board_session(
    tmp_path: Path,
    view_count: int = 5,
) -> tuple[Path, np.ndarray]:
    height, width = 900, 1200
    true_k = np.array(
        [
            [950.0, 0.0, 604.0],
            [0.0, 940.0, 446.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [0.01, -0.005, 0.0005, -0.0003, 0.0],
        dtype=np.float64,
    )
    board_points = np.asarray(
        make_board().getChessboardCorners(),
        dtype=np.float32,
    )
    ids = np.arange(len(board_points), dtype=np.int32)

    corners_by_frame: list[tuple[np.ndarray, np.ndarray]] = []
    names: list[str] = []
    for index in range(view_count):
        rvec = np.array(
            [
                0.08 + index * 0.03,
                -0.12 + index * 0.02,
                0.02 * index,
            ],
            dtype=np.float64,
        )
        tvec = np.array(
            [
                -55.0 + index * 18.0,
                -35.0 + index * 10.0,
                460.0 + index * 25.0,
            ],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(
            board_points,
            rvec,
            tvec,
            true_k,
            distortion,
        )
        corners_by_frame.append((ids, projected.reshape(-1, 2)))
        names.append(f"frame_{index:02d}.jpg")

    provisional = true_k.copy()
    provisional[0, 0] *= 0.97
    provisional[1, 1] *= 0.97

    board_dir = tmp_path / "board_poses" / "s-test"
    _write_board_session(
        board_dir,
        names,
        (height, width),
        corners_by_frame,
        provisional,
    )
    return board_dir, true_k


def _write_vggt_run(
    model_dir: Path,
    frame_names: list[str],
    intrinsic: np.ndarray,
    processed_size_hw: tuple[int, int],
) -> None:
    model_dir.mkdir(parents=True)
    np.savez_compressed(
        model_dir / "poses.npz",
        frame_names=np.asarray(frame_names),
        extrinsic=np.zeros(
            (len(frame_names), 3, 4),
            dtype=np.float32,
        ),
        intrinsic=np.tile(
            intrinsic,
            (len(frame_names), 1, 1),
        ),
    )
    (model_dir / "run.json").write_text(
        json.dumps({"processed_image_size_hw": list(processed_size_hw)}),
        encoding="utf-8",
    )


def test_calibrates_intrinsics_from_r02_corners_across_session(
    tmp_path,
):
    board_dir, true_k = _synthetic_board_session(
        tmp_path,
        view_count=6,
    )

    session = camera_intrinsics.load_board_session(board_dir)
    result = camera_intrinsics.calibrate_from_board(session)

    assert result.source == camera_intrinsics.SOURCE_CHARUCO
    assert result.reprojection_rms_px is not None
    assert result.reprojection_rms_px < 0.01
    assert result.intrinsic[0, 0] == pytest.approx(
        true_k[0, 0],
        abs=1.0,
    )
    assert result.intrinsic[1, 1] == pytest.approx(
        true_k[1, 1],
        abs=1.0,
    )
    assert result.intrinsic[0, 2] == pytest.approx(
        true_k[0, 2],
        abs=1.0,
    )
    assert result.intrinsic[1, 2] == pytest.approx(
        true_k[1, 2],
        abs=1.0,
    )


def test_falls_back_to_rescaled_vggt_intrinsics_when_views_insufficient(
    tmp_path,
):
    board_dir, _ = _synthetic_board_session(
        tmp_path,
        view_count=1,
    )
    model_dir = tmp_path / "recon_runs" / "s-test"
    model_k = np.array(
        [
            [400.0, 0.0, 260.0],
            [0.0, 410.0, 196.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _write_vggt_run(
        model_dir,
        ["frame_00.jpg"],
        model_k,
        processed_size_hw=(450, 600),
    )

    session, result = camera_intrinsics.estimate_session_intrinsics(
        board_dir,
        model_dir,
    )

    assert result.source == camera_intrinsics.SOURCE_VGGT_FALLBACK
    assert result.fallback_reason is not None
    assert "at least 3 usable board views" in (result.fallback_reason)
    assert result.intrinsic[0, 0] == pytest.approx(800.0)
    assert result.intrinsic[1, 1] == pytest.approx(820.0)
    assert result.intrinsic[0, 2] == pytest.approx(520.0)
    assert result.intrinsic[1, 2] == pytest.approx(392.0)
    assert session.image_size_hw == (900, 1200)


def test_missing_vggt_outputs_make_failed_calibration_loud(
    tmp_path,
):
    board_dir, _ = _synthetic_board_session(
        tmp_path,
        view_count=1,
    )

    with pytest.raises(
        FileNotFoundError,
        match="VGGT fallback requires poses.npz",
    ):
        camera_intrinsics.estimate_session_intrinsics(
            board_dir,
            tmp_path / "missing-model-run",
        )


def test_board_session_rejects_mixed_image_sizes(
    tmp_path,
):
    board_dir, _ = _synthetic_board_session(
        tmp_path,
        view_count=3,
    )
    path = board_dir / "board_poses.npz"
    with np.load(path) as old:
        arrays = {}
        for key in old.files:
            arrays[key] = old[key]

    arrays["image_size_hw"][1] = [800, 1200]
    np.savez_compressed(path, **arrays)

    with pytest.raises(
        camera_intrinsics.CalibrationFailure,
        match="one stored-sensor image size",
    ):
        camera_intrinsics.load_board_session(board_dir)


def test_run_writes_intrinsics_and_reprojection_report(
    tmp_path,
):
    board_dir, _ = _synthetic_board_session(
        tmp_path,
        view_count=5,
    )
    out_dir = tmp_path / "camera_intrinsics" / "s-test"

    info = camera_intrinsics.run_intrinsics(
        board_dir,
        tmp_path / "unused-model-run",
        out_dir,
    )

    assert info["source"] == camera_intrinsics.SOURCE_CHARUCO
    assert info["reprojection_rms_px"] is not None
    assert "not a metric-accuracy" in (info["reprojection_note"])

    with np.load(out_dir / "intrinsics.npz") as saved:
        assert saved["intrinsic"].shape == (3, 3)
        assert str(saved["source"]) == (camera_intrinsics.SOURCE_CHARUCO)

    assert (out_dir / "intrinsics.json").is_file()

    with pytest.raises(
        FileExistsError,
        match="already has files",
    ):
        camera_intrinsics.run_intrinsics(
            board_dir,
            tmp_path / "unused-model-run",
            out_dir,
        )
