"""Unit tests for A-05's disagreement map. Synthetic completion folders only: no GPU, no golden data.

Board frame (recon/board_pose.py): +z points INTO the table, so a part on the board spans z in
[-height, 0]. The samples' meshes are written z-up, as a runner writes them, so a missing or wrong
frame conversion shows up as spread everywhere instead of only where it was planted.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from recon import completion, disagreement, next_view
from recon import completion_contract as contract
from recon.part_segmentation import BoardSurface
from recon.vggt_runner import write_ply

ROOT = Path(__file__).resolve().parents[1]

# An asymmetric box, so a mirrored or unconverted frame can't land back on itself.
BOX_X = (0.0, 40.0)
BOX_Y = (0.0, 24.0)
BOX_HEIGHT_MM = 16.0
CLEARANCE_MM = 3.0
# The camera "saw" only x <= SEEN_X_MAX; the +x end is guessed, and that is where seeds disagree.
SEEN_X_MAX = 37.0
POINTS_PER_SAMPLE = 6000


def _box(x_range, y_range, height_mm):
    """A closed 12-triangle box on the board, wound outward."""
    x0, x1 = x_range
    y0, y1 = y_range
    z_top, z_bottom = -height_mm, 0.0
    vertices = np.array(
        [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z_top, z_bottom)],
        dtype=np.float64,
    )
    quads = [
        (0, 1, 3, 2),  # x = x0
        (4, 6, 7, 5),  # x = x1
        (0, 4, 5, 1),  # y = y0
        (2, 3, 7, 6),  # y = y1
        (0, 2, 6, 4),  # top, z = -height
        (1, 5, 7, 3),  # bottom, on the board
    ]
    faces = []
    for a, b, c, d in quads:
        faces.extend([[a, b, c], [a, c, d]])
    faces = np.array(faces, dtype=np.int64)
    # Make every triangle face away from the centre (the box is convex).
    centre = vertices.mean(axis=0)
    for index, (a, b, c) in enumerate(faces):
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        if np.dot(normal, vertices[a] - centre) < 0:
            faces[index] = [a, c, b]
    return vertices, faces


def _level_board() -> BoardSurface:
    return BoardSurface(
        slope_x=0.0,
        slope_y=0.0,
        z_at_origin_mm=0.0,
        offset_mm=0.0,
        tilt_deg=0.0,
        residual_sigma_mm=0.3,
        inlier_count=1000,
        candidate_count=1000,
        board_layer_top_mm=None,
        height_threshold_mm=CLEARANCE_MM,
    )


def _write_fusion_dir(path: Path, board: BoardSurface) -> Path:
    path.mkdir(parents=True)
    (path / disagreement.FUSION_REPORT).write_text(
        json.dumps({"board_surface": board.as_report()}), encoding="utf-8"
    )
    return path


def _write_completion_folder(
    folder: Path,
    x_extensions_mm,
    *,
    accepted=None,
    status="complete",
    points_per_sample=POINTS_PER_SAMPLE,
    invert_winding=(),
    partial_offset_mm=(0.0, 0.0, 0.0),
) -> Path:
    """A completion folder as A-04 leaves it: one box per seed, whose +x end is pushed out by that
    seed's extension (the planted disagreement), sampled to `sample_<k>.ply` in the board frame,
    with its mesh written z-up in `runner/`."""
    folder.mkdir(parents=True)
    (folder / contract.RUNNER_DIR).mkdir()
    (folder / "input").mkdir()
    accepted = range(len(x_extensions_mm)) if accepted is None else accepted
    samples = []
    for k, extension in enumerate(x_extensions_mm):
        vertices, faces = _box((BOX_X[0], BOX_X[1] + extension), BOX_Y, BOX_HEIGHT_MM)
        if k in invert_winding:
            faces = faces[:, [0, 2, 1]]
        contract.write_mesh_ply(
            folder / contract.RUNNER_DIR / contract.sample_mesh_name(k),
            contract.board_to_z_up(vertices).astype(np.float32),
            faces,
        )
        points = completion.sample_surface_points(vertices, faces, points_per_sample, k)
        write_ply(
            folder / f"sample_{k}.ply",
            points.astype(np.float32),
            np.zeros((len(points), 3), dtype=np.uint8),
        )
        if k in accepted:
            samples.append({"k": k, "seed": k, "path": f"sample_{k}.ply"})

    # The partial: what the camera saw of the unextended box, above the board cut.
    vertices, faces = _box(BOX_X, BOX_Y, BOX_HEIGHT_MM)
    partial = completion.sample_surface_points(vertices, faces, 20000, 99)
    heights = -partial[:, 2]
    partial = partial[(partial[:, 0] <= SEEN_X_MAX) & (heights >= CLEARANCE_MM)]
    partial = partial + np.asarray(partial_offset_mm)
    write_ply(
        folder / contract.PARTIAL_POINTS_PATH,
        contract.board_to_z_up(partial).astype(np.float32),
        np.zeros((len(partial), 3), dtype=np.uint8),
    )
    report = {
        "status": status,
        "samples": samples,
        "input": {
            "isolation": {
                "clearance_mm": CLEARANCE_MM,
                "kept_height_range_mm": [float(-partial[:, 2].max()), BOX_HEIGHT_MM],
            }
        },
    }
    (folder / disagreement.COMPLETION_REPORT).write_text(
        json.dumps(report), encoding="utf-8"
    )
    return folder


def _report(completion_dir: Path) -> dict:
    return json.loads(
        (
            completion_dir
            / disagreement.DISAGREEMENT_DIR_NAME
            / disagreement.REPORT_FILE
        ).read_text(encoding="utf-8")
    )


def _targets(completion_dir: Path) -> disagreement.DisagreementTargets:
    targets = disagreement.load_targets(
        completion_dir / disagreement.DISAGREEMENT_DIR_NAME
    )
    assert targets is not None
    return targets


def _on_the_plus_x_end(points_mm: np.ndarray) -> np.ndarray:
    """Targets on the seeds' +x end faces, not on the top or side strips that reach past x = 40."""
    return (
        (points_mm[:, 0] > BOX_X[1] + 0.5)
        & (points_mm[:, 1] > BOX_Y[0] + 1.0)
        & (points_mm[:, 1] < BOX_Y[1] - 1.0)
        & (points_mm[:, 2] > -BOX_HEIGHT_MM + 1.0)
    )


# --- the map -------------------------------------------------------------------------------------


def test_spread_concentrates_where_the_seeds_disagree_and_vanishes_where_they_agree(
    tmp_path,
):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0, 3.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())

    outcome = disagreement.disagreement_stage(folder, fusion_dir=fusion)

    assert outcome.status == "complete", outcome.report["reason"]
    report = _report(folder)
    assert report["guidance_only"] is True
    assert report["k_used"] == [0, 1, 2, 3]
    # Every seed agrees on the seen surface, so the session's own noise level is ~0 (float32).
    assert report["noise_floor_mm"] < 0.01
    # The +x end moves 0..3 mm between seeds: the median distance to the other seeds is 1 or 2 mm.
    assert report["unseen_median_spread_mm"] >= 0.9
    # And only the guessed +x end carries weight.
    targets = _targets(folder)
    assert len(targets.weights) > 0
    assert np.all(targets.weights > 0)
    assert np.all(targets.points_mm[:, 0] > SEEN_X_MAX)
    assert report["total_weight_mm3"] == pytest.approx(targets.weights.sum())


def test_the_seen_samples_agree_only_after_the_runner_frame_is_turned_back(tmp_path):
    # The I-01 trap: diag(1,1,-1) also puts z up but mirrors the part. Read the meshes back with
    # that instead of the contract's rotation and the "agreeing" surface would disagree everywhere.
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 0.0, 0.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())
    assert (
        disagreement.disagreement_stage(folder, fusion_dir=fusion).status == "complete"
    )
    assert _report(folder)["noise_floor_mm"] < 0.01

    mirrored = tmp_path / "mirrored"
    _write_completion_folder(mirrored, (0.0, 0.0, 0.0))
    for k in range(3):
        path = mirrored / contract.RUNNER_DIR / contract.sample_mesh_name(k)
        vertices_z_up, faces = contract.read_mesh_ply_strict(path)
        contract.write_mesh_ply(
            path, (vertices_z_up * [1.0, -1.0, 1.0]).astype(np.float32), faces
        )
    assert (
        disagreement.disagreement_stage(mirrored, fusion_dir=fusion).status
        == "complete"
    )
    assert _report(mirrored)["noise_floor_mm"] > 1.0


def test_identical_samples_give_a_complete_map_with_nothing_worth_aiming_at(tmp_path):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 0.0, 0.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())

    assert (
        disagreement.disagreement_stage(folder, fusion_dir=fusion).status == "complete"
    )

    # Only float32 rounding is left (spreads of ~1e-6 mm), far below what the scorer's rounding to
    # 6 decimals can tell apart, so it ranks views in C-08's order.
    report = _report(folder)
    assert report["unseen_p90_spread_mm"] < 1e-4
    assert report["total_weight_mm3"] < 1e-3


def test_the_board_band_is_never_a_target(tmp_path):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0, 3.0))
    # A board tilted along x: the band follows the plane, not z = 0.
    tilted = replace(_level_board(), slope_x=-0.05)
    fusion = _write_fusion_dir(tmp_path / "fusion", tilted)

    assert (
        disagreement.disagreement_stage(folder, fusion_dir=fusion).status == "complete"
    )

    report = _report(folder)
    assert report["board"]["method"] == "board_surface"
    assert report["board_band_point_count"] > 0
    targets = _targets(folder)
    # Weighted means of points on one side of a plane stay on that side.
    heights = tilted.height_above_board_mm(targets.points_mm)
    assert np.all(heights >= CLEARANCE_MM - 1e-9)


def test_a_folder_without_its_fusion_record_falls_back_to_a_level_plane(tmp_path):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0, 3.0))

    outcome = disagreement.disagreement_stage(folder)

    assert outcome.status == "complete"
    board = _report(folder)["board"]
    assert board["method"] == "level_plane_from_partial"
    assert board["plane_z_mm"] == pytest.approx(0.0, abs=1e-4)
    heights = -_targets(folder).points_mm[:, 2]
    assert np.all(heights >= CLEARANCE_MM - 1e-4)


def test_one_outlier_seed_does_not_raise_the_others_spread(tmp_path):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 0.0, 0.0, 5.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())

    assert (
        disagreement.disagreement_stage(folder, fusion_dir=fusion).status == "complete"
    )

    shares = {
        sample["k"]: sample["weight_share"] for sample in _report(folder)["samples"]
    }
    # Three seeds agree, so the median over the others keeps their +x end at ~0; only the outlier's
    # own points carry weight, and the report shows whose they are.
    assert shares[3] > 0.99
    assert shares[0] + shares[1] + shares[2] < 0.01


def test_a_rejected_sample_never_votes_even_with_its_files_on_disk(tmp_path):
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())
    with_rejected = _write_completion_folder(
        tmp_path / "with_rejected", (0.0, 1.0, 2.0, 30.0), accepted=[0, 1, 2]
    )
    without = _write_completion_folder(tmp_path / "without", (0.0, 1.0, 2.0))

    disagreement.disagreement_stage(with_rejected, fusion_dir=fusion)
    disagreement.disagreement_stage(without, fusion_dir=fusion)

    assert _report(with_rejected)["k_used"] == [0, 1, 2]
    assert _report(with_rejected)["total_weight_mm3"] == pytest.approx(
        _report(without)["total_weight_mm3"]
    )


def test_weight_is_per_area_so_a_denser_sample_does_not_outvote_the_others(tmp_path):
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())
    sparse = _write_completion_folder(tmp_path / "sparse", (0.0, 1.0, 2.0, 3.0))
    dense = _write_completion_folder(
        tmp_path / "dense",
        (0.0, 1.0, 2.0, 3.0),
        points_per_sample=2 * POINTS_PER_SAMPLE,
    )

    disagreement.disagreement_stage(sparse, fusion_dir=fusion)
    disagreement.disagreement_stage(dense, fusion_dir=fusion)

    assert _report(dense)["total_weight_mm3"] == pytest.approx(
        _report(sparse)["total_weight_mm3"], rel=0.1
    )


def test_pairwise_figures_use_a12s_split_and_the_map_leaves_out_the_board_band(
    tmp_path,
):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 2.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())

    disagreement.disagreement_stage(folder, fusion_dir=fusion)

    report = _report(folder)
    (pair,) = report["pairs"]
    assert pair["k"] == [0, 1]
    assert pair["seen_median_mm"] < 0.01
    # A-12's pairwise U counts every point beyond 2 mm of the partial as unseen, the board-contact
    # face included. Here that face is large and shared by both boxes, so it pulls U to ~0; the
    # map's own targets leave the band out and see the 2 mm the end faces moved.
    assert pair["unseen_median_mm"] < 0.01
    assert report["unseen_median_spread_mm"] == pytest.approx(2.0, abs=0.01)


def test_back_face_culling_needs_every_mesh_consistently_wound(tmp_path):
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())
    good = _write_completion_folder(tmp_path / "good", (0.0, 1.0, 2.0))
    disagreement.disagreement_stage(good, fusion_dir=fusion)
    assert _report(good)["cull_back_faces"] is True
    targets = _targets(good)
    # Outward: on the +x end face the normals point +x.
    end_face = _on_the_plus_x_end(targets.points_mm)
    assert np.any(end_face)
    assert np.all(targets.normals[end_face, 0] > 0.9)

    # An inside-out mesh is still consistent, and its normals are turned back outward.
    inside_out = _write_completion_folder(
        tmp_path / "inside_out", (0.0, 1.0, 2.0), invert_winding=(1,)
    )
    disagreement.disagreement_stage(inside_out, fusion_dir=fusion)
    assert _report(inside_out)["cull_back_faces"] is True
    targets = _targets(inside_out)
    end_face = _on_the_plus_x_end(targets.points_mm)
    assert np.any(end_face)
    assert np.all(targets.normals[end_face, 0] > 0.9)

    # One triangle flipped on its own: the winding can't be trusted, so no culling at all.
    flipped = _write_completion_folder(tmp_path / "flipped", (0.0, 1.0, 2.0))
    path = flipped / contract.RUNNER_DIR / contract.sample_mesh_name(2)
    vertices, faces = contract.read_mesh_ply_strict(path)
    faces = faces.copy()
    faces[0] = faces[0, [0, 2, 1]]
    contract.write_mesh_ply(path, vertices, faces)
    disagreement.disagreement_stage(flipped, fusion_dir=fusion)
    report = _report(flipped)
    assert report["cull_back_faces"] is False
    assert report["samples"][2]["winding"]["consistent"] is False
    assert np.all(_targets(flipped).normals == 0.0)


# --- statuses: never raises ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extensions", "accepted", "status"),
    [
        ((0.0, 1.0), [0], "complete"),  # one accepted sample
        ((0.0, 1.0), [0, 1], "unavailable"),  # A-04 prepared the folder but nothing ran
        ((0.0, 1.0), [0, 1], "failed"),
    ],
)
def test_too_few_samples_or_no_completion_is_unavailable_with_no_targets(
    tmp_path, extensions, accepted, status
):
    folder = _write_completion_folder(
        tmp_path / "completion", extensions, accepted=accepted, status=status
    )

    outcome = disagreement.disagreement_stage(folder)

    assert outcome.status == "unavailable"
    out_dir = folder / disagreement.DISAGREEMENT_DIR_NAME
    assert sorted(path.name for path in out_dir.iterdir()) == [disagreement.REPORT_FILE]
    assert disagreement.load_targets(out_dir) is None


def test_a_missing_completion_folder_is_a_failed_record_not_an_exception(tmp_path):
    outcome = disagreement.disagreement_stage(
        tmp_path / "nothing", out_dir=tmp_path / "out"
    )
    assert outcome.status == "failed"
    assert "completion.json" in outcome.report["reason"]
    assert (
        json.loads((tmp_path / "out" / disagreement.REPORT_FILE).read_text())["status"]
        == "failed"
    )


def test_a_corrupt_mesh_fails_the_map_and_leaves_no_targets(tmp_path):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0))
    (folder / contract.RUNNER_DIR / contract.sample_mesh_name(1)).write_bytes(
        b"not a ply"
    )

    outcome = disagreement.disagreement_stage(folder)

    assert outcome.status == "failed"
    assert "ContractError" in outcome.report["reason"]
    out_dir = folder / disagreement.DISAGREEMENT_DIR_NAME
    assert not (out_dir / disagreement.TARGETS_FILE).exists()
    assert disagreement.load_targets(out_dir) is None


def test_no_seen_surface_means_no_noise_level_so_the_map_fails(tmp_path):
    folder = _write_completion_folder(
        tmp_path / "completion", (0.0, 1.0, 2.0), partial_offset_mm=(500.0, 0.0, 0.0)
    )

    outcome = disagreement.disagreement_stage(folder)

    assert outcome.status == "failed"
    assert "noise level" in outcome.report["reason"]


def test_a_finished_map_is_never_overwritten_but_an_unavailable_record_is_replaced(
    tmp_path,
):
    folder = _write_completion_folder(
        tmp_path / "completion", (0.0, 1.0, 2.0), status="unavailable"
    )
    assert disagreement.disagreement_stage(folder).status == "unavailable"

    # The A-12 route: the samples arrive later from another machine (completion.ingest_existing).
    report = json.loads((folder / disagreement.COMPLETION_REPORT).read_text())
    report["status"] = "complete"
    (folder / disagreement.COMPLETION_REPORT).write_text(json.dumps(report))
    assert disagreement.disagreement_stage(folder).status == "complete"

    out_dir = folder / disagreement.DISAGREEMENT_DIR_NAME
    before = {path.name: path.read_bytes() for path in out_dir.iterdir()}
    again = disagreement.disagreement_stage(folder)
    assert again.status == "failed"
    assert "refusing to overwrite" in again.report["reason"]
    assert {path.name: path.read_bytes() for path in out_dir.iterdir()} == before


def test_the_cli_maps_a_folder_and_exits_zero_only_when_complete(tmp_path, capsys):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())

    assert disagreement.main([str(folder), "--fusion-dir", str(fusion)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"

    empty = _write_completion_folder(tmp_path / "empty", (0.0, 1.0), accepted=[0])
    assert disagreement.main([str(empty)]) == 1


# --- the hand-off to the next-view scorer ---------------------------------------------------------


def _served_box_without_its_plus_x_end():
    """The served mesh: the part's seen surface, Observed, with no +x end at all (never seen)."""
    vertices, faces = _box(BOX_X, BOX_Y, BOX_HEIGHT_MM)
    keep = ~np.all(np.isclose(vertices[faces][:, :, 0], BOX_X[1]), axis=1)
    labels = np.full(len(vertices), next_view.LABEL_OBSERVED, dtype=object)
    return vertices, faces[keep], labels


def test_the_scorer_ranks_the_view_facing_the_disagreement_first(tmp_path):
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0, 3.0))
    fusion = _write_fusion_dir(tmp_path / "fusion", _level_board())
    disagreement.disagreement_stage(folder, fusion_dir=fusion)
    vertices, faces, labels = _served_box_without_its_plus_x_end()

    ranked = next_view.score_next_views(
        vertices, faces, labels, disagreement=_targets(folder)
    )

    best = next_view.best_by_disagreement(ranked)
    assert best["disagreement_visible_weight"] > 0
    # Looking at the +x end from outside it. C-08 counts what a view sees rather than how squarely,
    # so an oblique view that also takes in a side strip can win; it is still on the +x side.
    assert best["position_mm"][0] > BOX_X[1]
    assert best["azimuth_deg"] <= 60.0 or best["azimuth_deg"] >= 300.0
    # The top-down view sees the +x end edge-on at best; it sees less of the disagreement.
    straight_down = next(
        view for view in ranked if view["angle_from_vertical_deg"] == 0.0
    )
    assert (
        straight_down["disagreement_visible_weight"]
        < best["disagreement_visible_weight"]
    )


def test_the_scorer_without_a_map_is_exactly_c08(tmp_path):
    vertices, faces, labels = _served_box_without_its_plus_x_end()
    labels[np.isclose(vertices[:, 1], BOX_Y[0])] = next_view.LABEL_UNOBSERVED

    plain = next_view.score_next_views(vertices, faces, labels)
    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0, 3.0))
    disagreement.disagreement_stage(folder)
    with_map = next_view.score_next_views(
        vertices, faces, labels, disagreement=_targets(folder)
    )

    # Same order, same C-08 scores: the map adds fields and a rank, nothing else.
    c08_keys = set(plain[0])
    assert [view["candidate_index"] for view in with_map] == [
        view["candidate_index"] for view in plain
    ]
    assert [{key: view[key] for key in c08_keys} for view in with_map] == plain
    assert next_view.best_by_disagreement(plain) is None


def test_served_observed_surface_hides_a_target_and_guessed_surface_never_does(
    tmp_path,
):
    vertices, faces, labels = _served_box_without_its_plus_x_end()
    camera = np.array([[200.0, 12.0, -8.0]])  # level with the box, beyond its +x end
    candidates = {"position_mm": camera}
    inside = np.array([[39.0, 12.0, -8.0]])  # just inside the (missing) +x end
    behind_far_wall = np.array([[-3.0, 12.0, -8.0]])  # beyond the Observed x = 0 wall

    def visible_weight(points, labels_for_mesh, cull=False, normals=None):
        targets = disagreement.DisagreementTargets(
            points,
            np.zeros_like(points) if normals is None else normals,
            np.ones(len(points)),
            cull,
        )
        scores = next_view.score_candidate_views(
            vertices, faces, labels_for_mesh, candidates, disagreement=targets
        )
        return scores["disagreement_visible_weight"][0]

    # Nothing served stands between the camera and a point inside the open end.
    assert visible_weight(inside, labels) == 1.0
    # The Observed x = 0 wall hides what lies behind it.
    assert visible_weight(behind_far_wall, labels) == 0.0
    # 1.5 mm behind served *Unobserved* surface is the same surface (A-12's 2 mm), so it counts;
    # the same distance behind Observed surface does not.
    unobserved_labels = labels.copy()
    unobserved_labels[np.isclose(vertices[:, 0], BOX_X[0])] = next_view.LABEL_UNOBSERVED
    camera_beyond_x0 = {"position_mm": np.array([[-200.0, 12.0, -8.0]])}
    targets = disagreement.DisagreementTargets(
        np.array([[1.5, 12.0, -8.0]]), np.zeros((1, 3)), np.ones(1), False
    )
    for mesh_labels, expected in ((unobserved_labels, 1.0), (labels, 0.0)):
        scores = next_view.score_candidate_views(
            vertices, faces, mesh_labels, camera_beyond_x0, disagreement=targets
        )
        assert scores["disagreement_visible_weight"][0] == expected
    # With trusted normals, a target facing away from the camera is culled.
    facing_away = np.array([[-1.0, 0.0, 0.0]])
    assert visible_weight(inside, labels, cull=True, normals=facing_away) == 0.0
    assert visible_weight(inside, labels, cull=True, normals=-facing_away) == 1.0


def test_the_scorer_aims_at_targets_even_when_nothing_served_is_unobserved():
    vertices, faces, labels = _served_box_without_its_plus_x_end()
    assert not np.any(labels == next_view.LABEL_UNOBSERVED)
    targets = disagreement.DisagreementTargets(
        np.array([[39.0, 12.0, -8.0]]), np.zeros((1, 3)), np.ones(1), False
    )

    scores = next_view.score_candidate_views(
        vertices,
        faces,
        labels,
        {"position_mm": np.array([[200.0, 12.0, -8.0]])},
        disagreement=targets,
    )

    assert scores["under_observed_visible_count"][0] == 0
    assert scores["disagreement_visible_weight"][0] == 1.0


def test_the_same_surface_allowance_is_a12s_seen_distance():
    assert (
        next_view.DISAGREEMENT_SAME_SURFACE_TOLERANCE_MM
        == disagreement.UNSEEN_BEYOND_MM
    )


def test_next_view_run_reports_the_map_it_used(tmp_path):
    from recon import mesh_cleanup, provenance

    folder = _write_completion_folder(tmp_path / "completion", (0.0, 1.0, 2.0, 3.0))
    disagreement.disagreement_stage(folder)
    vertices, faces, labels = _served_box_without_its_plus_x_end()
    uncertainty_dir = tmp_path / "uncertainty"
    uncertainty_dir.mkdir()
    mesh_cleanup.write_glb(
        uncertainty_dir / "mesh_uncertainty.glb",
        vertices,
        faces,
        provenance.labels_to_colors(labels),
    )

    report = next_view.next_view_run(
        uncertainty_dir,
        tmp_path / "next_view",
        disagreement_dir=folder / disagreement.DISAGREEMENT_DIR_NAME,
    )

    assert report["disagreement"]["used"] is True
    assert report["best_by_disagreement"]["disagreement_rank"] == 1
    written = json.loads((tmp_path / "next_view" / "next_view.json").read_text())
    assert written["best_by_disagreement"] == report["best_by_disagreement"]

    # A folder with no complete map is reported as not used, and C-08 ranks alone.
    empty = _write_completion_folder(tmp_path / "empty", (0.0, 1.0), accepted=[0])
    disagreement.disagreement_stage(empty)
    report = next_view.next_view_run(
        uncertainty_dir,
        tmp_path / "next_view_empty",
        disagreement_dir=empty / disagreement.DISAGREEMENT_DIR_NAME,
    )
    assert report["disagreement"]["used"] is False
    assert report["best_by_disagreement"] is None


# --- guidance only (D-005, D-006) -----------------------------------------------------------------


def test_nothing_in_the_verdict_lane_imports_the_disagreement_map():
    importers = []
    for path in sorted((ROOT / "inspect").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            if any(name.startswith("recon.disagreement") for name in names):
                importers.append(str(path.relative_to(ROOT)))
    assert importers == []
