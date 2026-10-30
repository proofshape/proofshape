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
    # R-28: 520.5/392.5, not 520.0/392.0 -- rescale_vggt_intrinsic now matches
    # lift_board_corner_observations's own half-pixel-correct convention
    # ((px + 0.5) * scale - 0.5), which shifts the principal point by half a pixel versus the
    # old direct-ratio formula. Focal length is unaffected by that shift.
    assert result.intrinsic[0, 2] == pytest.approx(520.5)
    assert result.intrinsic[1, 2] == pytest.approx(392.5)
    assert session.image_size_hw == (900, 1200)


# --- R-28: rescale_vggt_intrinsic's own crop-aware contract (#182) -----------------------------


def test_rescale_vggt_intrinsic_identity_is_a_noop():
    k = np.array([[500.0, 0.0, 260.0], [0.0, 510.0, 190.0], [0.0, 0.0, 1.0]])
    rescaled = camera_intrinsics.rescale_vggt_intrinsic(k, 1.0, 1.0)
    np.testing.assert_allclose(rescaled, k)


def test_rescale_vggt_intrinsic_rejects_non_positive_scale():
    k = np.eye(3)
    with pytest.raises(ValueError, match="scale_x and scale_y"):
        camera_intrinsics.rescale_vggt_intrinsic(k, 0.0, 1.0)
    with pytest.raises(ValueError, match="scale_x and scale_y"):
        camera_intrinsics.rescale_vggt_intrinsic(k, 1.0, -1.0)


def test_rescale_vggt_intrinsic_recovers_a_square_pixel_camera_through_a_real_portrait_crop(
    tmp_path,
):
    # #182: a 1080x1920 portrait frame is resized (width to 518) then centre-cropped (height to
    # at most 518) by VGGT's own preprocessing -- describe_vggt_crop_preprocess, already on
    # main (R-23), records exactly that geometry. A plain full/processed ratio (the old,
    # buggy rescale_vggt_intrinsic) ignores the crop and returns fy about 1.78x too large.
    from PIL import Image

    from recon.vggt_runner import describe_vggt_crop_preprocess

    frame_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (1080, 1920)).save(frame_path)
    [geometry] = describe_vggt_crop_preprocess([frame_path])
    assert geometry["crop_top_px"] > 0  # sanity: this frame really does get cropped

    # A known square-pixel full-resolution camera, centred in the frame.
    full_fx = full_fy = 1500.0
    full_cx, full_cy = 540.0, 960.0

    # The forward map VGGT's own preprocessing induces on a projected pixel (same pixel-centre
    # convention as recon.metric_alignment.lift_board_corner_observations's own forward formula),
    # applied to the principal point; focal length scales by the same per-axis factor.
    scale_x, scale_y = geometry["scale_x"], geometry["scale_y"]
    crop_left, crop_top = geometry["crop_left_px"], geometry["crop_top_px"]
    processed_k = np.array(
        [
            [full_fx * scale_x, 0.0, (full_cx + 0.5) * scale_x - 0.5 - crop_left],
            [0.0, full_fy * scale_y, (full_cy + 0.5) * scale_y - 0.5 - crop_top],
            [0.0, 0.0, 1.0],
        ]
    )

    rescaled = camera_intrinsics.rescale_vggt_intrinsic(
        processed_k, scale_x, scale_y, crop_left, crop_top
    )

    assert rescaled[0, 0] == pytest.approx(rescaled[1, 1])  # square pixels recovered
    assert rescaled[0, 0] == pytest.approx(full_fx)
    assert rescaled[1, 1] == pytest.approx(full_fy)
    assert rescaled[0, 2] == pytest.approx(full_cx)
    assert rescaled[1, 2] == pytest.approx(
        full_cy
    )  # principal point lands where the crop puts it


def test_resolve_vggt_frame_scale_prefers_the_frames_own_recorded_geometry():
    geometry = [
        {
            "frame_name": "frame_00.jpg",
            "scale_x": 0.48,
            "scale_y": 0.50,
            "crop_left_px": 0,
            "crop_top_px": 203,
        }
    ]
    scale_x, scale_y, crop_left, crop_top = camera_intrinsics.resolve_vggt_frame_scale(
        "frame_00.jpg", geometry, (518, 518), (1920, 1080)
    )
    assert (scale_x, scale_y, crop_left, crop_top) == (0.48, 0.50, 0.0, 203.0)


def test_resolve_vggt_frame_scale_refuses_an_unsupported_recorded_frame():
    geometry = [
        {
            "frame_name": "frame_00.jpg",
            "supported_for_metric_alignment": False,
            "unsupported_reason": "mixed-shape batch padding",
        }
    ]
    with pytest.raises(camera_intrinsics.CalibrationFailure, match="frame_00.jpg"):
        camera_intrinsics.resolve_vggt_frame_scale(
            "frame_00.jpg", geometry, (518, 518), (1920, 1080)
        )


def test_resolve_vggt_frame_scale_falls_back_to_plain_resize_for_landscape_without_geometry():
    scale_x, scale_y, crop_left, crop_top = camera_intrinsics.resolve_vggt_frame_scale(
        "frame_00.jpg", None, (450, 600), (900, 1200)
    )
    assert (scale_x, scale_y, crop_left, crop_top) == (0.5, 0.5, 0.0, 0.0)


def test_resolve_vggt_frame_scale_refuses_a_portrait_frame_with_no_recorded_geometry():
    # #182/AC: without preprocess_geometry, a portrait run is refused rather than guessed, the
    # same backward-compat boundary recon.metric_alignment.lift_board_corner_observations
    # already draws.
    with pytest.raises(camera_intrinsics.CalibrationFailure, match="frame_00.jpg"):
        camera_intrinsics.resolve_vggt_frame_scale(
            "frame_00.jpg", None, (518, 518), (1920, 1080)
        )


def test_load_vggt_fallback_refuses_an_ambiguous_run_with_no_recorded_geometry(
    tmp_path,
):
    board_dir, _ = _synthetic_board_session(tmp_path, view_count=1)
    model_dir = tmp_path / "recon_runs" / "s-test"
    model_k = np.array(
        [[400.0, 0.0, 260.0], [0.0, 410.0, 196.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    # The board session (_synthetic_board_session) is landscape, (900, 1200), but this run's
    # processed_image_size_hw is portrait-shaped, (1200, 900) -- not a "clearly uncropped
    # landscape" run by resolve_vggt_frame_scale's own check -- and it carries no
    # preprocess_geometry, so the real load_vggt_fallback path must refuse it rather than guess,
    # naming the frame.
    _write_vggt_run(model_dir, ["frame_00.jpg"], model_k, processed_size_hw=(1200, 900))

    with pytest.raises(camera_intrinsics.CalibrationFailure, match="frame_00.jpg"):
        camera_intrinsics.estimate_session_intrinsics(board_dir, model_dir)


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
