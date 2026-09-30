"""Unit tests for R-11's GPU-free adapter logic."""

import json

import numpy as np
import pytest
from PIL import Image, ImageOps

from recon import mast3r_runner


def test_open_stored_rgb_does_not_apply_exif_rotation(tmp_path):
    path = tmp_path / "portrait-tagged.jpg"
    image = Image.new("RGB", (6, 4), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)

    stored = mast3r_runner.open_stored_rgb(path)
    with Image.open(path) as reopened:
        displayed = ImageOps.exif_transpose(reopened)

    assert stored.size == (6, 4)
    assert displayed.size == (4, 6)


def test_camera_to_world_to_extrinsic_inverts_pose():
    cam2world = np.eye(4, dtype=np.float32)[None]
    cam2world[0, :3, 3] = [1.0, -2.0, 3.0]

    extrinsic = mast3r_runner.camera_to_world_to_extrinsic(cam2world)

    assert extrinsic.shape == (1, 3, 4)
    np.testing.assert_allclose(extrinsic[0, :3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(extrinsic[0, :, 3], [-1.0, 2.0, -3.0], atol=1e-6)


def test_camera_to_world_to_extrinsic_rejects_bad_shape_and_singular():
    with pytest.raises(ValueError, match="shape"):
        mast3r_runner.camera_to_world_to_extrinsic(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="singular"):
        mast3r_runner.camera_to_world_to_extrinsic(np.zeros((1, 4, 4)))


def test_reshape_dense_outputs_uses_confidence_shape():
    depthmaps = [np.arange(12), np.arange(12) + 20]
    confidences = [np.ones((3, 4)), np.full((3, 4), 2.0)]
    images = [np.zeros((3, 4, 3)), np.ones((3, 4, 3))]

    depth, confidence, rgb = mast3r_runner.reshape_dense_outputs(
        depthmaps, confidences, images
    )

    assert depth.shape == (2, 3, 4)
    assert confidence.shape == (2, 3, 4)
    assert rgb.shape == (2, 3, 4, 3)
    assert depth[0, 2, 3] == 11


def test_reshape_dense_outputs_rejects_mismatch():
    with pytest.raises(ValueError, match="same frame count"):
        mast3r_runner.reshape_dense_outputs([np.arange(4)], [], [])
    with pytest.raises(ValueError, match="depth has"):
        mast3r_runner.reshape_dense_outputs(
            [np.arange(3)],
            [np.ones((2, 2))],
            [np.zeros((2, 2, 3))],
        )


def _fake_result(frame_count, height=3, width=4):
    intrinsic = np.tile(
        np.array(
            [
                [5.0, 0.0, 1.5],
                [0.0, 5.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        (frame_count, 1, 1),
    )
    extrinsic = np.tile(
        np.hstack([np.eye(3), np.zeros((3, 1))]),
        (frame_count, 1, 1),
    )
    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": np.full((frame_count, height, width), 2.0),
        "depth_conf": np.full((frame_count, height, width), 3.0),
        "images": np.full((frame_count, height, width, 3), 0.5),
        "timings": {"model_load_s": 1.0, "alignment_s": 2.0},
        "device": "Fake T4",
        "dtype": "float32",
        "torch_version": "0.0",
        "mast3r_upstream_commit": mast3r_runner.MAST3R_UPSTREAM_COMMIT,
        "pair_count": 6,
        "peak_gpu_memory_gib": 3.0,
    }


def test_run_on_capture_writes_r01_compatible_contract(tmp_path, monkeypatch):
    capture = tmp_path / "s-99"
    capture.mkdir()
    for name in ["b.JPG", "a.jpeg"]:
        (capture / name).write_bytes(b"x")

    seen = {}

    def fake_run(
        frame_paths,
        model_id,
        device_type,
        mast3r_root,
        scene_graph,
        coarse_iters,
        fine_iters,
        image_size,
    ):
        seen["names"] = [path.name for path in frame_paths]
        return _fake_result(len(frame_paths))

    monkeypatch.setattr(mast3r_runner, "run_mast3r_model", fake_run)
    out = tmp_path / "run"

    info = mast3r_runner.run_on_capture(capture, out)

    assert seen["names"] == ["a.jpeg", "b.JPG"]
    assert sorted(path.name for path in out.iterdir()) == [
        "depth.npz",
        "points.ply",
        "poses.npz",
        "run.json",
    ]
    poses = np.load(out / "poses.npz")
    depth = np.load(out / "depth.npz")
    assert poses["extrinsic"].shape == (2, 3, 4)
    assert poses["intrinsic"].shape == (2, 3, 3)
    assert depth["depth"].shape == (2, 3, 4)
    assert depth["depth_conf"].shape == (2, 3, 4)

    on_disk = json.loads((out / "run.json").read_text())
    assert on_disk["backend"] == "mast3r"
    assert on_disk["processed_image_size_hw"] == [3, 4]
    assert "not metric" in on_disk["scale"]
    assert on_disk["pair_count"] == 6
    assert on_disk["timings"]["total_s"] >= 0
    assert info == on_disk
