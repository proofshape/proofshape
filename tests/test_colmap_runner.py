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
