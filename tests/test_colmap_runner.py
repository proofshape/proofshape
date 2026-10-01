import json
from pathlib import Path

import numpy as np
import pytest

from recon.colmap_runner import (
    camera_center_from_extrinsic,
    load_board_position_priors,
)


def test_camera_center_from_extrinsic_identity_rotation():
    extrinsic = np.array(
        [
            [1.0, 0.0, 0.0, -10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, -30.0],
        ]
    )

    center = camera_center_from_extrinsic(extrinsic)

    np.testing.assert_allclose(
        center,
        [10.0, -20.0, 30.0],
    )


def test_load_board_position_priors_converts_mm_to_metres(
    tmp_path: Path,
):
    path = tmp_path / "board_poses.npz"

    extrinsic = np.zeros((3, 3, 4), dtype=np.float64)

    for index in range(3):
        extrinsic[index, :, :3] = np.eye(3)

    extrinsic[0, :, 3] = [-100.0, -200.0, 300.0]
    extrinsic[1, :, 3] = [-200.0, -300.0, 400.0]
    extrinsic[2, :, 3] = [-300.0, -400.0, 500.0]

    np.savez(
        path,
        frame_names=np.array(["a.jpeg", "b.jpeg", "c.jpeg"]),
        pose_found=np.array([True, True, True]),
        extrinsic=extrinsic,
    )

    priors = load_board_position_priors(path)

    np.testing.assert_allclose(
        priors["a.jpeg"],
        [0.1, 0.2, -0.3],
    )
    np.testing.assert_allclose(
        priors["b.jpeg"],
        [0.2, 0.3, -0.4],
    )


def test_load_board_position_priors_ignores_failed_pose(
    tmp_path: Path,
):
    path = tmp_path / "board_poses.npz"

    extrinsic = np.zeros((4, 3, 4), dtype=np.float64)

    for index in range(4):
        extrinsic[index, :, :3] = np.eye(3)

    np.savez(
        path,
        frame_names=np.array(["a.jpeg", "b.jpeg", "c.jpeg", "d.jpeg"]),
        pose_found=np.array([True, False, True, True]),
        extrinsic=extrinsic,
    )

    priors = load_board_position_priors(path)

    assert set(priors) == {
        "a.jpeg",
        "c.jpeg",
        "d.jpeg",
    }


def test_load_board_position_priors_rejects_too_few(
    tmp_path: Path,
):
    path = tmp_path / "board_poses.npz"

    extrinsic = np.zeros((3, 3, 4), dtype=np.float64)

    for index in range(3):
        extrinsic[index, :, :3] = np.eye(3)

    np.savez(
        path,
        frame_names=np.array(["a.jpeg", "b.jpeg", "c.jpeg"]),
        pose_found=np.array([True, False, True]),
        extrinsic=extrinsic,
    )

    with pytest.raises(
        ValueError,
        match="at least 3 usable board position priors",
    ):
        load_board_position_priors(path)


class FakeImage:
    def __init__(self, name, data_id):
        self.name = name
        self.data_id = data_id


class FakePosePrior:
    def __init__(
        self,
        corr_data_id,
        position,
        coordinate_system,
    ):
        self.corr_data_id = corr_data_id
        self.position = position
        self.coordinate_system = coordinate_system


class FakePosePriorCoordinateSystem:
    CARTESIAN = "cartesian"


class FakePycolmap:
    PosePrior = FakePosePrior
    PosePriorCoordinateSystem = FakePosePriorCoordinateSystem


class FakeDatabase:
    def __init__(self):
        self.cleared = False
        self.written = []

    def clear_pose_priors(self):
        self.cleared = True
        self.written = []

    def write_pose_prior(self, prior):
        self.written.append(prior)


class FakeReconstruction:
    def __init__(self, registered, points):
        self.registered = registered
        self.points = points

    def num_reg_images(self):
        return self.registered

    def num_points3D(self):
        return self.points


def test_write_board_position_priors_clears_gps_and_writes_board_only():
    from recon.colmap_runner import write_board_position_priors

    database = FakeDatabase()

    images = [
        FakeImage("a.jpeg", 1),
        FakeImage("b.jpeg", 2),
        FakeImage("c.jpeg", 3),
        FakeImage("d.jpeg", 4),
    ]

    priors = {
        "a.jpeg": np.array([0.1, 0.2, -0.3]),
        "b.jpeg": np.array([0.2, 0.3, -0.4]),
        "c.jpeg": np.array([0.3, 0.4, -0.5]),
    }

    written, missing = write_board_position_priors(
        database,
        images,
        priors,
        FakePycolmap,
    )

    assert database.cleared
    assert written == 3
    assert missing == ["d.jpeg"]
    assert len(database.written) == 3

    assert database.written[0].corr_data_id == 1
    assert database.written[0].coordinate_system == "cartesian"

    np.testing.assert_allclose(
        database.written[0].position,
        [0.1, 0.2, -0.3],
    )


def test_write_board_position_priors_rejects_too_few_capture_matches():
    from recon.colmap_runner import write_board_position_priors

    database = FakeDatabase()

    images = [
        FakeImage("a.jpeg", 1),
        FakeImage("b.jpeg", 2),
        FakeImage("c.jpeg", 3),
    ]

    priors = {
        "a.jpeg": np.array([0.0, 0.0, 0.0]),
        "b.jpeg": np.array([0.1, 0.0, 0.0]),
    }

    with pytest.raises(
        ValueError,
        match="at least 3 board priors",
    ):
        write_board_position_priors(
            database,
            images,
            priors,
            FakePycolmap,
        )


def test_choose_best_reconstruction_prefers_registered_images():
    from recon.colmap_runner import choose_best_reconstruction

    models = {
        0: FakeReconstruction(6, 9000),
        1: FakeReconstruction(8, 3000),
        2: FakeReconstruction(7, 12000),
    }

    model_id, reconstruction = choose_best_reconstruction(models)

    assert model_id == 1
    assert reconstruction.num_reg_images() == 8


def test_choose_best_reconstruction_returns_none_for_refusal():
    from recon.colmap_runner import choose_best_reconstruction

    assert choose_best_reconstruction({}) is None


def test_depth_validity_mask_is_binary_not_probability():
    from recon.colmap_runner import depth_validity_mask

    depth = np.array(
        [
            [
                [1.0, 0.0, np.nan],
                [2.5, -1.0, np.inf],
            ]
        ],
        dtype=np.float32,
    )

    confidence = depth_validity_mask(depth)

    assert confidence.dtype == np.float32

    np.testing.assert_array_equal(
        confidence,
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ],
    )


class FakeFusedPoint:
    def __init__(self, xyz, color):
        self.xyz = np.asarray(xyz)
        self.color = np.asarray(color)


class FakeFusedReconstruction:
    def __init__(self, points):
        self.points3D = {index: point for index, point in enumerate(points)}


def test_point_cloud_from_reconstruction_extracts_xyz_rgb():
    from recon.colmap_runner import point_cloud_from_reconstruction

    reconstruction = FakeFusedReconstruction(
        [
            FakeFusedPoint(
                [1.0, 2.0, 3.0],
                [255, 10, 20],
            ),
            FakeFusedPoint(
                [-1.0, 4.0, 5.0],
                [1, 2, 3],
            ),
        ]
    )

    points, colors = point_cloud_from_reconstruction(reconstruction)

    assert points.dtype == np.float32
    assert colors.dtype == np.uint8

    np.testing.assert_allclose(
        points,
        [
            [1.0, 2.0, 3.0],
            [-1.0, 4.0, 5.0],
        ],
    )

    np.testing.assert_array_equal(
        colors,
        [
            [255, 10, 20],
            [1, 2, 3],
        ],
    )


def _fake_colmap_success(frame_names):
    frame_count = len(frame_names)
    height = 3
    width = 4

    extrinsic = np.tile(
        np.hstack(
            [
                np.eye(3),
                np.zeros((3, 1)),
            ]
        ),
        (frame_count, 1, 1),
    )

    intrinsic = np.tile(
        np.array(
            [
                [5.0, 0.0, 2.0],
                [0.0, 5.0, 1.5],
                [0.0, 0.0, 1.0],
            ]
        ),
        (frame_count, 1, 1),
    )

    depth = np.full(
        (frame_count, height, width),
        2.0,
        dtype=np.float32,
    )

    return {
        "status": "success",
        "reason": None,
        "model_id": 0,
        "frame_names": list(frame_names),
        "frame_count": frame_count,
        "registered_frame_count": frame_count,
        "sparse_point_count": 100,
        "board_prior_count": frame_count,
        "frames_without_board_prior": [],
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": depth,
        "depth_conf": np.ones_like(
            depth,
            dtype=np.float32,
        ),
        "camera_sizes_hw": [[3, 4] for _ in frame_names],
        "points": np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "point_colors": np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        ),
        "max_image_size": 1600,
        "pycolmap_version": "4.2.0",
        "timings": {
            "feature_extraction_s": 1.0,
            "matching_s": 1.0,
            "mapping_s": 1.0,
            "patch_match_s": 1.0,
        },
    }


def test_run_on_capture_writes_r01_compatible_contract(
    tmp_path,
    monkeypatch,
):
    from recon import colmap_runner

    capture = tmp_path / "s-99"
    capture.mkdir()

    for name in [
        "b.jpeg",
        "a.jpeg",
        "c.jpeg",
    ]:
        (capture / name).write_bytes(b"x")

    board = tmp_path / "board_poses.npz"
    board.write_bytes(b"x")

    frame_names = [
        "a.jpeg",
        "b.jpeg",
        "c.jpeg",
    ]

    def fake_run_colmap_model(
        capture_dir,
        board_pose_path,
        device_type,
        max_image_size,
    ):
        assert capture_dir == capture
        assert board_pose_path == board
        assert device_type == "cuda"

        return _fake_colmap_success(frame_names)

    monkeypatch.setattr(
        colmap_runner,
        "run_colmap_model",
        fake_run_colmap_model,
    )

    out = tmp_path / "out"

    info = colmap_runner.run_on_capture(
        capture,
        out,
        board_pose_path=board,
    )

    assert sorted(path.name for path in out.iterdir()) == [
        "depth.npz",
        "points.ply",
        "poses.npz",
        "run.json",
    ]

    poses = np.load(out / "poses.npz")

    depth = np.load(out / "depth.npz")

    assert list(poses["frame_names"]) == frame_names

    assert poses["extrinsic"].shape == (
        3,
        3,
        4,
    )

    assert poses["intrinsic"].shape == (
        3,
        3,
        3,
    )

    assert depth["depth"].shape == (
        3,
        3,
        4,
    )

    assert depth["depth_conf"].shape == (
        3,
        3,
        4,
    )

    on_disk = json.loads((out / "run.json").read_text())

    assert on_disk["story"] == "R-12"
    assert on_disk["backend"] == "colmap"
    assert on_disk["status"] == "success"

    assert on_disk["registered_frame_count"] == 3

    assert on_disk["processed_image_size_hw"] == [3, 4]

    assert "binary validity mask" in on_disk["depth_conf_semantics"]

    assert on_disk["point_count"] == 2

    assert on_disk["timings"]["total_s"] >= 0.0

    assert info == on_disk


def test_run_on_capture_records_refusal_without_fake_geometry(
    tmp_path,
    monkeypatch,
):
    from recon import colmap_runner

    capture = tmp_path / "s-99"
    capture.mkdir()

    for name in [
        "a.jpeg",
        "b.jpeg",
        "c.jpeg",
    ]:
        (capture / name).write_bytes(b"x")

    board = tmp_path / "board_poses.npz"
    board.write_bytes(b"x")

    def fake_run_colmap_model(
        capture_dir,
        board_pose_path,
        device_type,
        max_image_size,
    ):
        return {
            "status": "refused",
            "reason": ("colmap_registered_no_model"),
            "frame_names": [
                "a.jpeg",
                "b.jpeg",
                "c.jpeg",
            ],
            "frame_count": 3,
            "registered_frame_count": 0,
            "sparse_point_count": 0,
            "board_prior_count": 3,
            "frames_without_board_prior": [],
            "pycolmap_version": "4.2.0",
            "timings": {
                "mapping_s": 1.0,
            },
        }

    monkeypatch.setattr(
        colmap_runner,
        "run_colmap_model",
        fake_run_colmap_model,
    )

    out = tmp_path / "out"

    info = colmap_runner.run_on_capture(
        capture,
        out,
        board_pose_path=board,
    )

    assert info["status"] == "refused"

    assert sorted(path.name for path in out.iterdir()) == [
        "run.json",
    ]

    assert not (out / "poses.npz").exists()

    assert not (out / "depth.npz").exists()

    assert not (out / "points.ply").exists()

    on_disk = json.loads((out / "run.json").read_text())

    assert on_disk["reason"] == "colmap_registered_no_model"

    assert on_disk["point_count"] == 0
