"""Unit tests for R-01's GPU-free logic. No PyTorch, VGGT, GPU or capture photos are needed."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from recon import vggt_runner


def _rotation_about_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _make_intrinsic(width: int, height: int) -> np.ndarray:
    return np.array([[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]])


# --- find_capture_frames -------------------------------------------------------------------


def test_find_capture_frames_sorts_and_filters(tmp_path):
    for name in ["b.JPG", "a.jpg", "c.jpeg", "notes.txt", "d.png"]:
        (tmp_path / name).write_bytes(b"x")
    (
        tmp_path / "subdir.jpg"
    ).mkdir()  # a directory with a jpg-looking name is not a frame

    names = [path.name for path in vggt_runner.find_capture_frames(tmp_path)]

    assert names == ["a.jpg", "b.JPG", "c.jpeg"]


def test_find_capture_frames_missing_folder_points_at_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_golden_capture"):
        vggt_runner.find_capture_frames(tmp_path / "nope")


def test_find_capture_frames_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .jpg"):
        vggt_runner.find_capture_frames(tmp_path)


# --- choose_dtype_name ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_type", "capability", "expected"),
    [
        ("cuda", (7, 5), "float16"),  # Tesla T4 on the shared Lightning Studio
        ("cuda", (8, 0), "bfloat16"),  # A100
        ("cuda", (8, 9), "bfloat16"),
        ("mps", None, "float16"),  # D-011: bf16 is half speed on Metal
    ],
)
def test_choose_dtype_name(device_type, capability, expected):
    assert vggt_runner.choose_dtype_name(device_type, capability) == expected


def test_choose_dtype_name_rejects_unknown_device_and_missing_capability():
    with pytest.raises(ValueError, match="Unsupported device"):
        vggt_runner.choose_dtype_name("cpu", None)
    with pytest.raises(ValueError, match="compute capability"):
        vggt_runner.choose_dtype_name("cuda", None)


# --- unproject_depth_to_world --------------------------------------------------------------


def test_unproject_recovers_known_world_points_from_two_posed_cameras():
    """Project known 3D points into rotated/translated cameras, then unproject them back."""
    width, height = 8, 6
    intrinsic = _make_intrinsic(width, height)
    world_points = np.array(
        [[0.2, -0.1, 2.0], [-0.3, 0.2, 2.5], [0.0, 0.0, 3.0], [0.4, 0.3, 2.2]]
    )
    pixel_uv = [(1, 1), (6, 2), (4, 3), (7, 5)]

    extrinsics = []
    for degrees, translation in [(0.0, [0.0, 0.0, 0.0]), (25.0, [0.3, -0.2, 0.1])]:
        extrinsics.append(
            np.hstack([_rotation_about_z(degrees), np.array(translation)[:, None]])
        )
    extrinsic = np.stack(extrinsics)
    intrinsics = np.stack([intrinsic, intrinsic])

    depth = np.zeros((2, height, width))
    expected = {}
    for frame_index in range(2):
        rotation = extrinsic[frame_index, :, :3]
        translation = extrinsic[frame_index, :, 3]
        for world_point, (u, v) in zip(world_points, pixel_uv):
            camera_point = rotation @ world_point + translation
            # Build a depth map whose (v,u) pixel sees this exact world point: choose the depth
            # so that the pixel's ray hits the camera-frame point.
            ray = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0])
            depth_value = camera_point[2] / ray[2]
            depth[frame_index, v, u] = depth_value
            expected[(frame_index, v, u)] = ray * depth_value

    recovered = vggt_runner.unproject_depth_to_world(depth, extrinsic, intrinsics)

    for (frame_index, v, u), camera_point in expected.items():
        rotation = extrinsic[frame_index, :, :3]
        translation = extrinsic[frame_index, :, 3]
        expected_world = rotation.T @ (camera_point - translation)
        np.testing.assert_allclose(
            recovered[frame_index, v, u], expected_world, atol=1e-5
        )


def test_unproject_identity_camera_reproduces_pinhole_geometry():
    width, height = 5, 4
    intrinsic = _make_intrinsic(width, height)
    depth = np.full((1, height, width), 2.0)
    extrinsic = np.hstack([np.eye(3), np.zeros((3, 1))])[None]

    points = vggt_runner.unproject_depth_to_world(depth, extrinsic, intrinsic[None])

    # With an identity pose the world frame is the camera frame, so z equals the depth and the
    # principal-point pixel sits on the optical axis.
    np.testing.assert_allclose(points[..., 2], 2.0)
    # Pinhole model: x = (u - cx) * z / fx and y = (v - cy) * z / fy, here with f=50 and the
    # principal point at (2.5, 2.0).
    np.testing.assert_allclose(
        points[0, 2, 2, :2], [(2 - 2.5) * 2.0 / 50, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        points[0, 0, 4, :2], [(4 - 2.5) * 2.0 / 50, -2.0 * 2.0 / 50]
    )


def test_unproject_rejects_mismatched_shapes():
    depth = np.ones((2, 3, 3))
    with pytest.raises(ValueError, match="extrinsic"):
        vggt_runner.unproject_depth_to_world(
            depth, np.zeros((1, 3, 4)), np.zeros((2, 3, 3))
        )
    with pytest.raises(ValueError, match="intrinsic"):
        vggt_runner.unproject_depth_to_world(
            depth, np.zeros((2, 3, 4)), np.zeros((1, 3, 3))
        )


# --- frames_chw_to_hwc ---------------------------------------------------------------------


def test_frames_chw_to_hwc_keeps_the_frame_axis_for_a_single_frame():
    # The #31 review case: a one-frame capture must stay (1,H,W,3), not lose its frame axis.
    images = np.zeros((1, 3, 4, 6), dtype=np.float32)
    images[0, 0, 1, 2] = 0.25  # red channel of pixel (row 1, column 2)

    frames = vggt_runner.frames_chw_to_hwc(images)

    assert frames.shape == (1, 4, 6, 3)
    assert frames[0, 1, 2, 0] == pytest.approx(0.25)


def test_frames_chw_to_hwc_rejects_a_missing_frame_axis():
    with pytest.raises(ValueError, match=r"\(S,3,H,W\)"):
        vggt_runner.frames_chw_to_hwc(np.zeros((3, 4, 6)))


# --- select_point_cloud --------------------------------------------------------------------


def test_select_point_cloud_drops_non_finite_and_low_confidence():
    points = np.zeros((1, 2, 2, 3), dtype=np.float32)
    points[0, 0, 0] = [1, 2, 3]
    points[0, 0, 1] = [np.nan, 0, 0]
    points[0, 1, 0] = [4, 5, 6]
    points[0, 1, 1] = [7, 8, 9]
    colors = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    confidence = np.array([[[5.0, 5.0], [0.5, 5.0]]])

    kept_points, kept_colors = vggt_runner.select_point_cloud(
        points, colors, confidence, min_confidence=1.0
    )

    # Pixel (0,1) is NaN and pixel (1,0) is below the confidence floor.
    np.testing.assert_array_equal(kept_points, [[1, 2, 3], [7, 8, 9]])
    np.testing.assert_array_equal(kept_colors, [[0, 1, 2], [9, 10, 11]])


def test_select_point_cloud_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="share"):
        vggt_runner.select_point_cloud(
            np.zeros((1, 2, 2, 3)),
            np.zeros((1, 2, 2, 3), np.uint8),
            np.zeros((1, 3, 2)),
            0.0,
        )


# --- write_ply / write_run_outputs ---------------------------------------------------------


def test_write_ply_header_and_payload_round_trip(tmp_path):
    points = np.array([[0.0, 1.0, 2.0], [3.5, -4.0, 5.25]], dtype=np.float32)
    colors = np.array([[255, 0, 10], [1, 2, 3]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"

    vggt_runner.write_ply(path, points, colors)

    raw = path.read_bytes()
    header, _, payload = raw.partition(b"end_header\n")
    assert b"element vertex 2" in header
    assert b"format binary_little_endian 1.0" in header
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
    )
    vertices = np.frombuffer(payload, dtype=vertex_dtype)
    np.testing.assert_allclose(vertices["x"], points[:, 0])
    np.testing.assert_allclose(vertices["z"], points[:, 2])
    np.testing.assert_array_equal(vertices["r"], colors[:, 0])
    np.testing.assert_array_equal(vertices["b"], colors[:, 2])


def test_write_ply_rejects_bad_colours(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        vggt_runner.write_ply(tmp_path / "x.ply", np.zeros((2, 3)), np.zeros((2, 3)))


def _write_fake_run(out_dir):
    frame_count, height, width = 2, 3, 4
    vggt_runner.write_run_outputs(
        out_dir,
        ["a.jpg", "b.jpg"],
        np.zeros((frame_count, 3, 4)),
        np.zeros((frame_count, 3, 3)),
        np.ones((frame_count, height, width)),
        np.full((frame_count, height, width), 2.0),
        np.zeros((5, 3), np.float32),
        np.zeros((5, 3), np.uint8),
        {"story": "R-01"},
    )


def test_write_run_outputs_writes_all_four_files(tmp_path):
    out_dir = tmp_path / "run"

    _write_fake_run(out_dir)

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "depth.npz",
        "points.ply",
        "poses.npz",
        "run.json",
    ]
    poses = np.load(out_dir / "poses.npz")
    assert list(poses["frame_names"]) == ["a.jpg", "b.jpg"]
    assert poses["extrinsic"].shape == (2, 3, 4)
    depth = np.load(out_dir / "depth.npz")
    assert depth["depth"].shape == (2, 3, 4)
    assert json.loads((out_dir / "run.json").read_text())["story"] == "R-01"


def test_write_run_outputs_refuses_to_overwrite_a_previous_run(tmp_path):
    out_dir = tmp_path / "run"
    _write_fake_run(out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        _write_fake_run(out_dir)


# --- run_on_capture / main, with the model stubbed out --------------------------------------


def _fake_model_result(frame_count, height=4, width=6):
    intrinsic = np.tile(_make_intrinsic(width, height), (frame_count, 1, 1))
    extrinsic = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (frame_count, 1, 1))
    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": np.full((frame_count, height, width), 2.0),
        "depth_conf": np.full((frame_count, height, width), 3.0),
        "images": np.full((frame_count, height, width, 3), 0.5),
        "timings": {"model_load_s": 1.0, "preprocess_s": 0.5, "inference_s": 2.0},
        "device": "Fake GPU",
        "dtype": "float16",
        "torch_version": "0.0",
    }


def test_run_on_capture_end_to_end_with_stubbed_model(tmp_path, monkeypatch):
    capture_dir = tmp_path / "s-99"
    capture_dir.mkdir()
    for name in ["2.jpg", "1.jpg"]:
        (capture_dir / name).write_bytes(b"x")

    seen = {}

    def fake_run(frame_paths, model_id, device_type):
        seen["frames"] = [path.name for path in frame_paths]
        return _fake_model_result(len(frame_paths))

    monkeypatch.setattr(vggt_runner, "run_vggt_model", fake_run)

    run_info = vggt_runner.run_on_capture(
        capture_dir, tmp_path / "out", device_type="cuda"
    )

    assert seen["frames"] == ["1.jpg", "2.jpg"]
    assert run_info["frame_count"] == 2
    assert run_info["point_count"] == 2 * 4 * 6
    assert run_info["processed_image_size_hw"] == [4, 6]
    # Per-run wall-clock timings are an R-01 acceptance criterion and must be on disk.
    on_disk = json.loads((tmp_path / "out" / "run.json").read_text())
    for stage in ["model_load_s", "inference_s", "unproject_s", "write_s", "total_s"]:
        assert on_disk["timings"][stage] >= 0.0
    assert "not metric" in on_disk["scale"]


def test_main_reports_a_missing_capture_folder_and_exits_nonzero(tmp_path, capsys):
    exit_code = vggt_runner.main(
        [str(tmp_path / "missing"), "--out-dir", str(tmp_path / "o")]
    )

    assert exit_code == 1
    assert "R-01 RUN: FAIL" in capsys.readouterr().err


def test_run_vggt_model_names_the_missing_dependency(tmp_path, monkeypatch):
    # Force the lazy imports to fail whether or not torch/vggt happen to be installed here.
    monkeypatch.setitem(sys.modules, "vggt", None)
    monkeypatch.setitem(sys.modules, "vggt.models", None)
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", None)

    with pytest.raises(RuntimeError, match="requirements-gpu.txt"):
        vggt_runner.run_vggt_model([tmp_path / "a.jpg"], "facebook/VGGT-1B", "cuda")
