"""A-04: the shape-completion stage, service side (recon.completion), against a stub runner."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recon import completion, mesh_cleanup, provenance, service
from recon import completion_contract as contract
from recon.part_segmentation import BoardSurface
from recon.vggt_runner import read_ply

STUB_RUNNER = Path(__file__).with_name("completion_stub_runner.py")

# Fusion's cut on this synthetic board; the core then starts at 2.5 + 3.5 = 6 mm.
CUT_MM = 2.5
BOX_X = (60.0, 100.0)
BOX_Y = (40.0, 70.0)
BOX_HEIGHT_MM = 20.0
IMAGE_SIZE = (320, 240)
FOCAL_PX = 600.0


@pytest.fixture(autouse=True)
def _no_completion_environment(monkeypatch):
    # A developer's own runner setting must never leak into these tests.
    monkeypatch.delenv(completion.COMPLETION_ENV, raising=False)
    monkeypatch.delenv(completion.RUNNER_ENV, raising=False)


# --- a synthetic scene ----------------------------------------------------------------------------
# Board frame: millimetres, +z into the table, so a point h mm above the paper has z = -h.


def _flat_board(cut_mm: float = CUT_MM) -> BoardSurface:
    return BoardSurface(
        slope_x=0.0,
        slope_y=0.0,
        z_at_origin_mm=0.0,
        offset_mm=0.0,
        tilt_deg=0.0,
        residual_sigma_mm=0.1,
        inlier_count=1000,
        candidate_count=1000,
        board_layer_top_mm=cut_mm,
        height_threshold_mm=cut_mm,
    )


def _patch(origin, u_vector, v_vector, u_steps: int, v_steps: int):
    """A triangulated grid patch: origin + i*u + j*v. Returns (vertices, faces)."""
    origin = np.asarray(origin, dtype=np.float64)
    vertices = [
        origin + i * np.asarray(u_vector) + j * np.asarray(v_vector)
        for j in range(v_steps + 1)
        for i in range(u_steps + 1)
    ]
    faces = []
    width = u_steps + 1
    for j in range(v_steps):
        for i in range(u_steps):
            a = j * width + i
            faces.append([a, a + 1, a + width + 1])
            faces.append([a, a + width + 1, a + width])
    return np.array(vertices), np.array(faces)


class _MeshBuilder:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.labels = []
        self.count = 0

    def add(self, vertices, faces, label=provenance.LABEL_OBSERVED):
        self.vertices.append(vertices)
        self.faces.append(np.asarray(faces) + self.count)
        self.labels.extend([label] * len(vertices))
        self.count += len(vertices)
        return self

    def build(self):
        return (
            np.vstack(self.vertices),
            np.vstack(self.faces),
            np.array(self.labels, dtype=object),
        )


def _box_scene(skin_height_mm: float = 3.0, unobserved_x_max_wall: bool = True):
    """A 40 x 30 x 20 mm box on a board skin, one wall Unobserved, plus a stray bump.

    The skin stops 5 mm short of the box, so this scene tests isolation, not the cone's edge (the
    probe test below does that). The bump rises above the core height far from the box: board
    skin that pokes above the allowance, the second cluster isolation must drop.
    """
    builder = _MeshBuilder()
    x0, x1 = BOX_X
    y0, y1 = BOX_Y
    top = -BOX_HEIGHT_MM
    builder.add(*_patch((x0, y0, top), (2, 0, 0), (0, 2, 0), 20, 15))
    builder.add(*_patch((x0, y0, 0), (2, 0, 0), (0, 0, -2), 20, 10))  # y = y0 wall
    builder.add(*_patch((x0, y1, 0), (2, 0, 0), (0, 0, -2), 20, 10))  # y = y1 wall
    builder.add(*_patch((x0, y0, 0), (0, 2, 0), (0, 0, -2), 15, 10))  # x = x0 wall
    label = (
        provenance.LABEL_UNOBSERVED
        if unobserved_x_max_wall
        else provenance.LABEL_OBSERVED
    )
    builder.add(
        *_patch((x1, y0, 0), (0, 2, 0), (0, 0, -2), 15, 10), label
    )  # x = x1 wall

    skin_vertices, skin_faces = _patch(
        (0, 0, -skin_height_mm), (4, 0, 0), (0, 4, 0), 40, 30
    )
    near_box = (
        (skin_vertices[:, 0] > x0 - 5) & (skin_vertices[:, 0] < x1 + 5)
        & (skin_vertices[:, 1] > y0 - 5) & (skin_vertices[:, 1] < y1 + 5)
    )  # fmt: skip
    keep = ~near_box
    remap = -np.ones(len(skin_vertices), dtype=np.int64)
    remap[keep] = np.arange(np.count_nonzero(keep))
    kept_faces = skin_faces[keep[skin_faces].all(axis=1)]
    builder.add(skin_vertices[keep], remap[kept_faces])

    bump_vertices, bump_faces = _patch((20, 90, -6.4), (1, 0, 0), (0, 1, 0), 3, 3)
    builder.add(bump_vertices, bump_faces)
    return builder.build()


def _heights(vertices):
    return -vertices[:, 2]


def _small_params(**overrides) -> completion.CompletionParameters:
    defaults = {
        "partial_point_count": 2000,
        "sample_point_count": 500,
        "max_samples": 3,
        "budget_s": 60.0,
        "overrun_allowance_s": 30.0,
    }
    defaults.update(overrides)
    return completion.CompletionParameters(**defaults)


# --- part isolation -------------------------------------------------------------------------------


def test_isolation_keeps_the_part_and_its_lower_walls_but_not_the_skin():
    vertices, faces, labels = _box_scene()

    isolation = completion.isolate_part(vertices, faces, labels, _flat_board())

    kept = isolation.kept_vertex_mask
    heights = _heights(vertices)
    in_box = (
        (vertices[:, 0] >= BOX_X[0]) & (vertices[:, 0] <= BOX_X[1])
        & (vertices[:, 1] >= BOX_Y[0]) & (vertices[:, 1] <= BOX_Y[1])
    )  # fmt: skip
    observed = labels == provenance.LABEL_OBSERVED
    # Every Observed box vertex above fusion's cut is kept, down to the first wall row above it:
    # A-10's plain 6 mm cut lost exactly these.
    assert np.all(kept[in_box & observed & (heights >= CUT_MM)])
    assert heights[kept].min() <= CUT_MM + 2.0
    # The skin, the Unobserved wall and everything below the cut are not.
    assert not np.any(kept[~in_box])
    assert not np.any(kept[~observed])
    assert not np.any(kept[heights < CUT_MM])
    assert isolation.report["core_height_mm"] == pytest.approx(CUT_MM + 3.5)


def test_isolation_drops_a_stray_cluster_and_records_its_size():
    vertices, faces, labels = _box_scene()

    isolation = completion.isolate_part(vertices, faces, labels, _flat_board())

    bump = (
        (vertices[:, 0] <= 23.0) & (vertices[:, 1] >= 90.0) & (_heights(vertices) > 6.0)
    )
    assert not np.any(isolation.kept_vertex_mask[bump])
    assert isolation.report["core_cluster_count"] == 2
    assert isolation.report["second_largest_cluster_size"] == np.count_nonzero(bump)


@pytest.mark.parametrize(
    ("wall_angle_deg", "kept"),
    [(90.0, True), (66.6, True), (30.0, False)],
)
def test_isolation_keeps_steep_lower_walls_and_drops_shallow_ones(wall_angle_deg, kept):
    # A probe vertex at 3 mm, on a wall of the given steepness running down from the box's x0 edge
    # at the core height. s-04's sides are 66.6 deg; a 30 deg flange is skin-like and dropped.
    vertices, faces, labels = _box_scene(unobserved_x_max_wall=False)
    core_height = CUT_MM + 3.5
    probe_height = 3.0
    offset = (core_height - probe_height) / math.tan(math.radians(wall_angle_deg))
    probe = np.array([[BOX_X[0] - offset, 55.0, -probe_height]])
    vertices = np.vstack([vertices, probe])
    labels = np.append(labels, provenance.LABEL_OBSERVED)

    isolation = completion.isolate_part(vertices, faces, labels, _flat_board())

    assert bool(isolation.kept_vertex_mask[-1]) is kept


def test_isolation_refuses_a_part_too_low_to_have_a_core():
    vertices, faces, labels = _box_scene()
    flat = vertices[:, 2] < -5.0
    vertices[flat, 2] = -4.0  # squash everything to 4 mm: lower than the 6 mm core

    with pytest.raises(completion.CompletionInputError, match="at least 50 are needed"):
        completion.isolate_part(vertices, faces, labels, _flat_board())


def test_isolation_refuses_when_board_skin_reaches_the_core():
    # Skin at 7 mm, above the 6 mm core height: the "core" now spans the whole pattern.
    vertices, faces, labels = _box_scene(skin_height_mm=7.0)

    with pytest.raises(completion.CompletionInputError, match="pattern's edge"):
        completion.isolate_part(vertices, faces, labels, _flat_board())


# --- sampling -------------------------------------------------------------------------------------


def _two_triangles():
    # Areas 3:1, in the z = 0 plane.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 10.0, 0.0], [100.0, 0.0, 0.0],
         [110.0, 0.0, 0.0], [100.0, 10.0, 0.0]]
    )  # fmt: skip
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    return vertices, faces


def test_sampling_is_area_weighted_on_the_surface_and_reproducible():
    vertices, faces = _two_triangles()

    points = completion.sample_surface_points(vertices, faces, 20_000, seed=7)

    assert points.shape == (20_000, 3)
    np.testing.assert_array_equal(points[:, 2], 0.0)
    on_big = points[:, 0] < 50.0
    assert np.count_nonzero(on_big) / np.count_nonzero(~on_big) == pytest.approx(
        3.0, rel=0.05
    )
    # Inside each triangle: x/30 + y/10 <= 1 for the big one.
    big = points[on_big]
    assert np.all(big[:, 0] / 30.0 + big[:, 1] / 10.0 <= 1.0 + 1e-9)
    np.testing.assert_array_equal(
        points, completion.sample_surface_points(vertices, faces, 20_000, seed=7)
    )
    assert not np.array_equal(
        points, completion.sample_surface_points(vertices, faces, 20_000, seed=8)
    )


# --- choosing and cropping the photo --------------------------------------------------------------


def test_choose_view_frame_picks_the_posed_frame_that_saw_most_by_name():
    kept = np.array([True, True, False])
    observations = {
        # frame 1 saw the most kept vertices but has no board pose; frame 2 is next.
        "vertex_index": np.array([0, 1, 0, 1, 2, 2, 2, 0, 1]),
        "frame_index": np.array([1, 1, 1, 1, 0, 0, 0, 2, 2]),
    }
    reconstruction_names = np.array(["a.jpg", "b.jpg", "c.jpg"])
    # A different order in the board-pose file: matching must go by name.
    board_names = np.array(["c.jpg", "a.jpg", "b.jpg"])
    pose_found = np.array([True, True, False])

    name, count = completion.choose_view_frame(
        observations, kept, reconstruction_names, board_names, pose_found
    )

    assert (name, count) == ("c.jpg", 2)


def test_choose_view_frame_breaks_ties_by_frame_order_and_refuses_when_none_posed():
    kept = np.array([True])
    observations = {"vertex_index": np.array([0, 0]), "frame_index": np.array([0, 1])}
    names = np.array(["a.jpg", "b.jpg"])

    assert completion.choose_view_frame(
        observations, kept, names, names, np.array([True, True])
    ) == ("a.jpg", 1)
    with pytest.raises(
        completion.CompletionInputError, match="no frame with a board pose"
    ):
        completion.choose_view_frame(
            observations, kept, names, names, np.array([False, False])
        )


def _overhead_camera():
    """Camera-from-board extrinsic for a camera 400 mm above the board centre, looking down."""
    rotation = np.eye(3)  # camera z = board +z, i.e. into the table
    centre = np.array([80.0, 55.0, -400.0])
    extrinsic = np.hstack([rotation, (-rotation @ centre)[:, None]])
    intrinsic = np.array(
        [
            [FOCAL_PX, 0.0, IMAGE_SIZE[0] / 2],
            [0.0, FOCAL_PX, IMAGE_SIZE[1] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    return extrinsic, intrinsic


def _write_frame(path: Path, orientation: int | None = None) -> None:
    """A dark frame with a red block where the box projects, and a green dot at its x0/y0 corner."""
    extrinsic, intrinsic = _overhead_camera()
    image = Image.new("RGB", IMAGE_SIZE, (10, 10, 10))
    corners = np.array(
        [[BOX_X[0], BOX_Y[0], -BOX_HEIGHT_MM], [BOX_X[1], BOX_Y[1], -BOX_HEIGHT_MM]]
    )
    pixels = (intrinsic @ (extrinsic[:, :3] @ corners.T + extrinsic[:, 3:])).T
    u = pixels[:, 0] / pixels[:, 2]
    v = pixels[:, 1] / pixels[:, 2]
    image.paste((220, 20, 20), (int(u[0]), int(v[0]), int(u[1]), int(v[1])))
    image.paste((20, 220, 20), (int(u[0]), int(v[0]), int(u[0]) + 4, int(v[0]) + 4))
    if orientation is None:
        image.save(path, quality=95)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(path, quality=95, exif=exif)


def _box_points():
    vertices, _, _labels = _box_scene(unobserved_x_max_wall=False)
    in_box = (vertices[:, 0] >= BOX_X[0]) & (vertices[:, 0] <= BOX_X[1])
    return vertices[in_box & (_heights(vertices) > 3.0)]


def test_crop_is_centred_on_the_projected_part(tmp_path):
    _write_frame(tmp_path / "frame.jpg")
    extrinsic, intrinsic = _overhead_camera()

    crop, record = completion.crop_view_image(
        tmp_path / "frame.jpg", _box_points(), extrinsic, intrinsic
    )

    width, height = crop.size
    assert width == height
    red, green, _blue = crop.getpixel((width // 2, height // 2))
    assert red > 150 and green < 80
    left, top, right, bottom = record["crop_box_stored_px"]
    assert 0 <= left < right <= IMAGE_SIZE[0] and 0 <= top < bottom <= IMAGE_SIZE[1]
    assert record["exif_orientation"] is None


def test_crop_is_turned_upright_by_its_exif_orientation(tmp_path):
    _write_frame(tmp_path / "plain.jpg")
    _write_frame(tmp_path / "rotated.jpg", orientation=6)
    extrinsic, intrinsic = _overhead_camera()

    plain, _ = completion.crop_view_image(
        tmp_path / "plain.jpg", _box_points(), extrinsic, intrinsic
    )
    rotated, record = completion.crop_view_image(
        tmp_path / "rotated.jpg", _box_points(), extrinsic, intrinsic
    )

    # The crop box is found in stored pixels, the same for both; only the result is rotated.
    assert record["exif_orientation"] == 6
    np.testing.assert_array_equal(
        np.asarray(rotated), np.asarray(plain.transpose(Image.Transpose.ROTATE_270))
    )


def test_crop_is_clamped_inside_the_image_and_downscaled(tmp_path):
    _write_frame(tmp_path / "frame.jpg")
    extrinsic, intrinsic = _overhead_camera()
    # In view, but spread ~300 px across a 240 px-tall image: the square crop can't fit, so it is
    # clamped to the image's short side and kept inside it.
    wide = np.array([[-15.0, 55.0, -20.0], [175.0, 55.0, -20.0], [80.0, 56.0, -20.0]])
    params = completion.CompletionParameters(view_max_side_px=100)

    crop, record = completion.crop_view_image(
        tmp_path / "frame.jpg", wide, extrinsic, intrinsic, params
    )

    left, _top, right, _bottom = record["crop_box_stored_px"]
    assert right - left == IMAGE_SIZE[1] and 0 <= left and right <= IMAGE_SIZE[0]
    assert crop.size == (100, 100)


# --- a whole artifacts folder, for the stage --------------------------------------------------------


FRAME_NAMES = ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]


def _write_artifacts(root: Path) -> completion.PipelineArtifacts:
    """Every upstream file the stage reads, built from the synthetic box scene."""
    vertices, faces, labels = _box_scene()
    artifacts = completion.PipelineArtifacts.from_artifact_dir(
        root / "artifacts", root / "frames"
    )
    for folder in (
        artifacts.capture_dir,
        artifacts.board_pose_dir,
        artifacts.reconstruction_dir,
        artifacts.fusion_dir,
        artifacts.provenance_dir,
        artifacts.uncertainty_dir,
    ):
        folder.mkdir(parents=True)

    mesh_cleanup.write_glb(
        artifacts.uncertainty_dir / "mesh_uncertainty.glb",
        vertices,
        faces,
        provenance.labels_to_colors(labels),
    )
    np.savez(
        artifacts.uncertainty_dir / "sigma_mm.npz",
        sigma_mm=np.full(len(vertices), 0.5),
        residual_count=np.full(len(vertices), 3),
    )
    (artifacts.fusion_dir / "tsdf_fusion.json").write_text(
        json.dumps({"board_surface": _flat_board().as_report()}), encoding="utf-8"
    )
    # Frame 1 sees every vertex; frames 0 and 2 see half each.
    vertex_ids = np.arange(len(vertices))
    observations = {
        "vertex_index": np.concatenate([vertex_ids[::2], vertex_ids, vertex_ids[1::2]]),
        "frame_index": np.concatenate(
            [
                np.zeros(len(vertex_ids[::2]), dtype=np.int64),
                np.ones(len(vertex_ids), dtype=np.int64),
                np.full(len(vertex_ids[1::2]), 2, dtype=np.int64),
            ]
        ),
    }
    observations["z_cam_mm"] = np.full(len(observations["vertex_index"]), 400.0)
    observations["measured_depth_mm"] = observations["z_cam_mm"]
    provenance.write_observations_npz(
        artifacts.provenance_dir / "observations.npz", observations
    )
    np.savez(
        artifacts.reconstruction_dir / "poses.npz", frame_names=np.array(FRAME_NAMES)
    )

    extrinsic, intrinsic = _overhead_camera()
    np.savez(
        artifacts.board_pose_dir / "board_poses.npz",
        frame_names=np.array(FRAME_NAMES),
        pose_found=np.array([True, True, True]),
        extrinsic=np.stack([extrinsic] * 3),
        intrinsic=np.stack([intrinsic] * 3),
    )
    for name in FRAME_NAMES:
        _write_frame(artifacts.capture_dir / name)
    return artifacts


def _stub(mode: str = "ok", delay_s: float = 0.0) -> list[str]:
    return [
        sys.executable,
        str(STUB_RUNNER),
        "--mode",
        mode,
        "--sample-delay-s",
        str(delay_s),
    ]


def _upstream_hashes(artifact_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file() and "completion" not in path.relative_to(artifact_dir).parts:
            hashes[str(path.relative_to(artifact_dir))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return hashes


def _report(out_dir: Path) -> dict:
    return json.loads(
        (out_dir / completion.COMPLETION_REPORT).read_text(encoding="utf-8")
    )


# --- the stage, with no runner (every session on the Mac) ----------------------------------------


def test_no_runner_prepares_inputs_records_unavailable_and_warns_nobody(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    out_dir = tmp_path / "artifacts" / "completion"

    outcome = completion.run_completion_stage(artifacts, out_dir, None, _small_params())

    assert outcome.status == "unavailable"
    assert outcome.warning is None
    report = _report(out_dir)
    assert report["status"] == "unavailable"
    assert completion.RUNNER_ENV in report["reason"]
    assert report["provisional_parameters"] == [
        "max_samples",
        "budget_s",
        "overrun_allowance_s",
    ]
    assert (
        report["input"]["view"]["frame_name"] == "frame_0001.jpg"
    )  # it saw every vertex
    request = contract.read_request(out_dir / contract.REQUEST_FILE)
    assert request.seeds == (0, 1, 2)
    partial = contract.read_points_ply(out_dir / contract.PARTIAL_POINTS_PATH)
    assert len(partial) == 2000
    # z up: everything kept sits above the board's cut.
    assert partial[:, 2].min() >= CUT_MM - 1e-6
    assert (out_dir / contract.VIEW_IMAGE_PATH).exists()


def test_input_preparation_failure_without_a_runner_is_recorded_not_warned(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    (artifacts.fusion_dir / "tsdf_fusion.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "artifacts" / "completion"

    outcome = completion.run_completion_stage(artifacts, out_dir, None, _small_params())

    assert (outcome.status, outcome.warning) == ("failed", None)
    report = _report(out_dir)
    assert report["failed_stage"] == "input_preparation"
    assert "board_surface" in report["reason"]


def test_input_preparation_failure_with_a_runner_is_one_warning(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    (artifacts.fusion_dir / "tsdf_fusion.json").write_text("{}", encoding="utf-8")

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub(), _small_params()
    )

    assert (outcome.status, outcome.warning) == ("failed", completion.WARNING_FAILED)


def test_the_stage_refuses_a_folder_that_already_has_output(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    out_dir = tmp_path / "artifacts" / "completion"
    out_dir.mkdir()
    (out_dir / "completion.json").write_text('{"status": "complete"}', encoding="utf-8")

    outcome = completion.run_completion_stage(artifacts, out_dir, None, _small_params())

    assert outcome.status == "failed"
    assert json.loads((out_dir / "completion.json").read_text()) == {
        "status": "complete"
    }


# --- the stage with the stub runner ---------------------------------------------------------------


def test_stub_runner_samples_are_written_inferred_in_board_frame_and_upstream_is_untouched(
    tmp_path,
):
    artifacts = _write_artifacts(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    before = _upstream_hashes(artifact_dir)

    outcome = completion.run_completion_stage(
        artifacts, artifact_dir / "completion", _stub(), _small_params()
    )

    assert (outcome.status, outcome.warning) == ("complete", None)
    assert _upstream_hashes(artifact_dir) == before
    report = _report(artifact_dir / "completion")
    assert report["k_requested"] == report["k_produced"] == 3
    assert [sample["seed"] for sample in report["samples"]] == [0, 1, 2]
    assert report["model"]["reported"] == contract.ExpectedModel().as_dict()
    assert report["timings_s"]["runner"] > 0
    partial = contract.z_up_to_board(
        contract.read_points_ply(
            artifact_dir / "completion" / contract.PARTIAL_POINTS_PATH
        )
    )
    for sample in report["samples"]:
        points, colors = read_ply(artifact_dir / "completion" / sample["path"])
        assert len(points) == 500
        assert np.all(colors == provenance.INFERRED_COLOR)
        # Back in the board frame: the box sits above the paper (negative z), around the part.
        assert points[:, 2].max() < 0
        assert abs(points[:, 0].mean() - partial[:, 0].mean()) < 5.0


def test_same_seed_gives_the_same_sample_bytes_and_different_seeds_differ(tmp_path):
    first = _write_artifacts(tmp_path / "first")
    second = _write_artifacts(tmp_path / "second")
    for artifacts, root in ((first, tmp_path / "first"), (second, tmp_path / "second")):
        completion.run_completion_stage(
            artifacts, root / "artifacts" / "completion", _stub(), _small_params()
        )

    def sample_bytes(root, k):
        return (root / "artifacts" / "completion" / f"sample_{k}.ply").read_bytes()

    assert sample_bytes(tmp_path / "first", 0) == sample_bytes(tmp_path / "second", 0)
    assert sample_bytes(tmp_path / "first", 0) != sample_bytes(tmp_path / "first", 1)


def test_the_runner_stops_at_the_budget_and_that_counts_as_complete(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    # Samples take 0.5 s and the budget is 1.25 s: two fit; a third would end past it.
    params = _small_params(max_samples=4, budget_s=1.25)

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub(delay_s=0.5), params
    )

    report = _report(tmp_path / "artifacts" / "completion")
    assert (outcome.status, outcome.warning) == ("complete", None)
    assert report["model"]["runner_status"] == "budget_reached"
    assert (report["k_requested"], report["k_produced"]) == (4, 2)


@pytest.mark.parametrize(
    "mode", ["crash_first", "zero_samples", "invalid_result", "wrong_seed", "unit_cube"]
)
def test_a_runner_that_produces_nothing_usable_is_one_failed_warning(tmp_path, mode):
    artifacts = _write_artifacts(tmp_path)

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub(mode), _small_params()
    )

    assert (outcome.status, outcome.warning) == ("failed", completion.WARNING_FAILED)
    report = _report(tmp_path / "artifacts" / "completion")
    assert report["k_produced"] == 0
    assert not list((tmp_path / "artifacts" / "completion").glob("sample_*.ply"))


def test_a_runner_that_crashes_after_one_sample_keeps_it_as_partial(tmp_path):
    artifacts = _write_artifacts(tmp_path)

    outcome = completion.run_completion_stage(
        artifacts,
        tmp_path / "artifacts" / "completion",
        _stub("crash_after_first"),
        _small_params(),
    )

    assert (outcome.status, outcome.warning) == ("partial", completion.WARNING_PARTIAL)
    report = _report(tmp_path / "artifacts" / "completion")
    assert report["k_produced"] == 1
    assert report["runner"]["exit_code"] == 3


def test_a_hung_runner_is_killed_at_the_hard_timeout_and_its_sample_kept(tmp_path):
    artifacts = _write_artifacts(tmp_path)
    params = _small_params(budget_s=0.2, overrun_allowance_s=3.0)

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub("hang"), params
    )

    report = _report(tmp_path / "artifacts" / "completion")
    assert report["runner"]["timed_out"] is True
    assert report["timings_s"]["runner"] < 3.2 + 5.0  # killed, not waited out for 600 s
    assert (outcome.status, report["k_produced"]) == ("partial", 1)


def test_a_runner_that_cannot_start_is_one_warning(tmp_path):
    artifacts = _write_artifacts(tmp_path)

    outcome = completion.run_completion_stage(
        artifacts,
        tmp_path / "artifacts" / "completion",
        [str(tmp_path / "no-such-runner")],
        _small_params(),
    )

    assert (outcome.status, outcome.warning) == ("failed", completion.WARNING_FAILED)
    assert "could not start" in _report(tmp_path / "artifacts" / "completion")["reason"]


def test_a_second_run_while_one_is_running_is_skipped_with_a_warning(
    tmp_path, monkeypatch
):
    artifacts = _write_artifacts(tmp_path)
    busy = threading.Lock()
    busy.acquire()
    monkeypatch.setattr(completion, "_RUNNER_LOCK", busy)

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub(), _small_params()
    )

    assert (outcome.status, outcome.warning) == ("failed", completion.WARNING_FAILED)
    assert (
        "one runs at a time" in _report(tmp_path / "artifacts" / "completion")["reason"]
    )


def test_runner_output_stays_in_a_capped_log_never_the_report_or_warning(
    tmp_path, monkeypatch
):
    artifacts = _write_artifacts(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "key-that-must-not-reach-the-runner")
    params = _small_params(log_tail_bytes=4096)

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub("huge_stderr"), params
    )

    out_dir = tmp_path / "artifacts" / "completion"
    sentinel = "STUB-RUNNER-STDERR-SENTINEL"
    assert outcome.status == "complete"
    assert sentinel not in (out_dir / completion.COMPLETION_REPORT).read_text()
    log = (out_dir / completion.RUNNER_LOG_TAIL).read_bytes()
    assert sentinel.encode() in log
    assert len(log) <= 4096


def test_the_runner_environment_has_no_service_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("PROOFSHAPE_ENDPOINT_TOKEN", "secret-token")
    probe = tmp_path / "env-probe.py"
    probe.write_text(
        "import os, sys\nprint(os.environ.get('GEMINI_API_KEY'), os.environ.get("
        "'PROOFSHAPE_ENDPOINT_TOKEN'))\n",
        encoding="utf-8",
    )
    log = tmp_path / "log.txt"

    completion.invoke_runner(
        [sys.executable, str(probe)], tmp_path / "request.json", 30, log, 4096
    )

    assert log.read_text().strip() == "None None"


def test_an_unexpected_internal_error_still_never_raises(tmp_path, monkeypatch):
    artifacts = _write_artifacts(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("a bug, not an input problem")

    monkeypatch.setattr(completion, "sample_surface_points", broken)

    outcome = completion.run_completion_stage(
        artifacts, tmp_path / "artifacts" / "completion", _stub(), _small_params()
    )

    assert (outcome.status, outcome.warning) == ("failed", completion.WARNING_FAILED)
    assert "a bug" in _report(tmp_path / "artifacts" / "completion")["reason"]


# --- settings -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "disabled"),
    [("0", True), (" 0 ", True), ("", False), ("1", False), ("false", False)],
)
def test_only_a_literal_zero_disables_the_stage(value, disabled):
    assert completion.completion_disabled(value) is disabled


def test_runner_command_is_shell_split_and_blank_means_none():
    assert completion.resolve_runner_command("") is None
    assert completion.resolve_runner_command("   ") is None
    assert completion.resolve_runner_command("/opt/venv/bin/python 'run me.py'") == [
        "/opt/venv/bin/python",
        "run me.py",
    ]


def test_cli_folder_names_match_the_service():
    assert completion.CALIBRATED_BOARD_DIR_NAME == service.CALIBRATED_BOARD_DIR_NAME
    assert completion.BOARD_CALIBRATION_RECORD == service.BOARD_CALIBRATION_RECORD


# --- the CLI: prepare here, run elsewhere, ingest here ---------------------------------------------


def test_cli_prepare_then_ingest_a_run_made_elsewhere(tmp_path, capsys):
    import subprocess

    _write_artifacts(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    assert completion.main(["prepare", str(artifact_dir)]) == 0
    assert "PASS" in capsys.readouterr().out
    out_dir = artifact_dir / "completion"
    # What A-12's Colab check does by hand: run the runner on the prepared folder.
    subprocess.run(
        [*_stub(), "--request", str(out_dir / contract.REQUEST_FILE)], check=True
    )

    assert completion.main(["ingest", str(artifact_dir)]) == 0
    report = _report(out_dir)
    assert report["status"] == "complete"
    assert report["runner"]["ran_elsewhere"] is True
    assert report["k_produced"] == completion.DEFAULT_PARAMETERS.max_samples


def test_cli_refuses_to_prepare_over_or_ingest_into_a_finished_folder(tmp_path, capsys):
    import subprocess

    _write_artifacts(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    assert completion.main(["prepare", str(artifact_dir)]) == 0

    assert completion.main(["prepare", str(artifact_dir)]) == 1
    assert "already exists" in capsys.readouterr().err

    out_dir = artifact_dir / "completion"
    subprocess.run(
        [*_stub(), "--request", str(out_dir / contract.REQUEST_FILE)], check=True
    )
    assert completion.main(["ingest", str(artifact_dir)]) == 0
    assert completion.main(["ingest", str(artifact_dir)]) == 1
    assert "only an 'unavailable'" in capsys.readouterr().err


# --- inside the reconstruction pipeline -----------------------------------------------------------


def _fake_upstream_stages(monkeypatch, source: completion.PipelineArtifacts) -> None:
    """Every stage before completion, replaced by a copy of the synthetic scene's files."""
    import shutil

    from recon import (
        board_pose,
        metric_alignment,
        reconstruction_runner,
        tsdf_fusion,
        uncertainty,
    )

    def copy_into(source_dir: Path, out_dir: Path) -> None:
        shutil.copytree(source_dir, out_dir, dirs_exist_ok=True)

    def fake_board(capture, out):
        copy_into(source.board_pose_dir, out)
        return {"posed_count": 3, "failure_counts": {}}

    def fake_reconstruction(capture, out, **kwargs):
        copy_into(source.reconstruction_dir, out)
        return {"status": "ok"}

    def fake_fusion(reconstruction_dir, alignment_path, out):
        copy_into(source.fusion_dir, out)
        return {}

    def fake_provenance(model_path, reconstruction_dir, alignment_path, out):
        copy_into(source.provenance_dir, out)
        return {}

    def fake_uncertainty(provenance_dir, out):
        copy_into(source.uncertainty_dir, out)
        return {"observed_fraction": 0.5}

    monkeypatch.setattr(board_pose, "estimate_board_poses", fake_board)
    monkeypatch.setattr(reconstruction_runner, "run_on_capture", fake_reconstruction)
    monkeypatch.setattr(
        metric_alignment, "load_and_align", lambda *a, **k: {"scale": 1.0}
    )
    monkeypatch.setattr(metric_alignment, "write_alignment", lambda *a, **k: None)
    monkeypatch.setattr(tsdf_fusion, "fuse_run", fake_fusion)
    monkeypatch.setattr(mesh_cleanup, "cleanup_run", lambda *a, **k: {})
    monkeypatch.setattr(provenance, "provenance_run", fake_provenance)
    monkeypatch.setattr(uncertainty, "uncertainty_run", fake_uncertainty)


def _small_defaults(monkeypatch) -> None:
    # The pipeline uses the stage's defaults; keep the stub's work small and fast.
    monkeypatch.setattr(completion, "DEFAULT_PARAMETERS", _small_params())


def test_pipeline_with_completion_disabled_never_runs_the_stage(tmp_path, monkeypatch):
    source = _write_artifacts(tmp_path / "source")
    _fake_upstream_stages(monkeypatch, source)
    monkeypatch.setenv(completion.COMPLETION_ENV, "0")

    def must_not_run(*args, **kwargs):
        raise AssertionError(
            "the completion stage ran although PROOFSHAPE_COMPLETION=0"
        )

    monkeypatch.setattr(completion, "run_completion_stage", must_not_run)

    result = service.run_reconstruction_pipeline(
        source.capture_dir, tmp_path / "artifacts"
    )

    assert "completion" not in result["stage_timings_s"]
    assert result["warnings"] == []
    assert not (tmp_path / "artifacts" / "completion").exists()


def test_pipeline_without_a_runner_times_the_stage_and_warns_nobody(
    tmp_path, monkeypatch
):
    source = _write_artifacts(tmp_path / "source")
    _fake_upstream_stages(monkeypatch, source)
    _small_defaults(monkeypatch)

    result = service.run_reconstruction_pipeline(
        source.capture_dir, tmp_path / "artifacts"
    )

    assert "completion" in result["stage_timings_s"]
    assert result["warnings"] == []
    assert _report(tmp_path / "artifacts" / "completion")["status"] == "unavailable"


def test_pipeline_upstream_files_and_served_glb_are_identical_with_completion_on_or_off(
    tmp_path, monkeypatch
):
    import shlex

    source = _write_artifacts(tmp_path / "source")
    _fake_upstream_stages(monkeypatch, source)
    _small_defaults(monkeypatch)

    monkeypatch.setenv(completion.COMPLETION_ENV, "0")
    off = service.run_reconstruction_pipeline(source.capture_dir, tmp_path / "off")
    monkeypatch.setenv(completion.COMPLETION_ENV, "1")
    monkeypatch.setenv(completion.RUNNER_ENV, shlex.join(_stub()))
    on = service.run_reconstruction_pipeline(source.capture_dir, tmp_path / "on")

    assert _report(tmp_path / "on" / "completion")["status"] == "complete"
    assert on["warnings"] == off["warnings"] == []
    # Every upstream file -- mesh_uncertainty.glb and sigma_mm.npz included -- byte for byte.
    assert _upstream_hashes(tmp_path / "on") == _upstream_hashes(tmp_path / "off")
    assert Path(on["glb_path"]).relative_to(tmp_path / "on") == Path(
        off["glb_path"]
    ).relative_to(tmp_path / "off")
    # And the served GLB carries no Inferred vertex: only the samples do.
    _, _, colors = mesh_cleanup.read_glb(Path(on["glb_path"]))
    assert not np.any(np.all(colors == provenance.INFERRED_COLOR, axis=1))


def test_a_failing_runner_leaves_the_session_complete_with_one_warning_and_no_leak(
    tmp_path, monkeypatch
):
    import shlex

    from fastapi.testclient import TestClient

    source = _write_artifacts(tmp_path / "source")
    _fake_upstream_stages(monkeypatch, source)
    _small_defaults(monkeypatch)
    failing_runner = tmp_path / "failing_runner.py"
    failing_runner.write_text(
        "import os, sys\n"
        "sys.stderr.write('RUNNER-STDERR ' + str(os.environ.get('GEMINI_API_KEY')))\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        completion.RUNNER_ENV, shlex.join([sys.executable, str(failing_runner)])
    )

    def pipeline_with_real_frames(capture_dir: Path, artifact_dir: Path) -> dict:
        # The upload test helper sends placeholder bytes; the stage crops a real photo.
        for name in FRAME_NAMES:
            (capture_dir / name).write_bytes((source.capture_dir / name).read_bytes())
        return service.run_reconstruction_pipeline(capture_dir, artifact_dir)

    data_dir = tmp_path / "data"
    app = service.create_app(
        data_dir, pipeline_runner=pipeline_with_real_frames, known_order_codes={"PO-1"}
    )
    # Set only after the app is built, so the VLM supervisor keeps its no-key path (no network);
    # the service process now holds a secret the runner must not inherit.
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    with TestClient(app) as client:
        session_id = client.post("/v1/sessions", json={"order_code": "PO-1"}).json()[
            "session_id"
        ]
        for index in range(3):
            response = client.post(
                f"/v1/sessions/{session_id}/frames",
                files={
                    "frame": (f"frame{index}.jpg", b"jpeg-data", "image/jpeg"),
                    "gyro": (
                        "blob",
                        json.dumps({"alpha": 0, "beta": 0, "gamma": 0}),
                        "application/json",
                    ),
                },
            )
            assert response.status_code == 202
        assert client.post(f"/v1/sessions/{session_id}/finish").status_code == 202
        body = client.get(f"/v1/sessions/{session_id}/reconstruction").json()

    assert body["status"] == "complete"
    assert body["warnings"] == [completion.WARNING_FAILED]
    session_json = (data_dir / session_id / "session.json").read_text(encoding="utf-8")
    assert "RUNNER-STDERR" not in session_json
    assert "secret-gemini-key" not in session_json
    log = (
        data_dir / session_id / "artifacts" / "completion" / completion.RUNNER_LOG_TAIL
    )
    assert log.read_text() == "RUNNER-STDERR None"  # the key never reached the runner
