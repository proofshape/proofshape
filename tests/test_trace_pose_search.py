"""I-15: the stage-by-stage trace records what the search actually did.

The trace is evidence for I-15's first criterion, so it is tested on I-01 synthetic samples whose
true pose is known: a reading taken from the true pose must trace as right all the way through,
and a reading turned half way must be reported lost at the seeds. `global_register`'s `stages`
record must not change what the search returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from inspection import cad, pose_judgement, registration, synthetic

from scripts import trace_pose_search as trace_module

ROOT = Path(__file__).resolve().parents[1]
S04 = ROOT / "fixtures" / "reference_cad" / "s04_triangular_prism_60-26-65x30mm.stl"

# The same reduced search test_registration.py uses, so these run in seconds.
SMALL_SEARCH = {
    "sample_count": 700,
    "yaw_steps": 36,
    "candidate_count": 6,
    "ransac_iterations": 3_000,
}
# Any CAD-frame direction works as the "named side" for a synthetic part: the reading is taken
# from the true pose along the same direction.
NAMED_SIDE = (0.0, -1.0)


def _sample(seed: int = 41):
    vertices, faces, units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(vertices, faces, seed)
    true_pose = np.eye(4)
    true_pose[:3, :3] = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    true_pose[:3, 3] = np.asarray(
        sample.sidecar["transform"]["translation_mm"], dtype=float
    )
    model = (sample.vertices_mm, sample.faces, sample.labels)
    return sample, model, (vertices, faces, units), true_pose


def _reading(cad_vertices, pose, *, turn_deg: float = 0.0):
    centre = pose_judgement.posed_centre_xy(cad_vertices, pose)
    return pose_judgement.PhotoReading(
        named_side_faces_deg=pose_judgement.pose_rotation_of_xy_direction(
            pose, NAMED_SIDE
        )
        + turn_deg,
        centre_xy_mm=(float(centre[0]), float(centre[1])),
        face_matches=True,
    )


def test_recording_stages_does_not_change_the_search():
    _, model, cad_mesh, _true_pose = _sample()
    plain = registration.global_register(model, cad_mesh, seed=3, **SMALL_SEARCH)
    stages: dict = {}
    recorded = registration.global_register(
        model, cad_mesh, seed=3, stages=stages, **SMALL_SEARCH
    )
    assert len(plain) == len(recorded)
    for left, right in zip(plain, recorded):
        np.testing.assert_array_equal(left.transformation, right.transformation)
        assert left.source == right.source
        assert left.fitness == right.fitness
    # What it recorded as selected is what it returned (before the residual is attached).
    assert [c.source for c in stages["selected"]] == [c.source for c in recorded]
    assert len(stages["seeded"]) >= len(stages["deduplicated"]) >= 1
    assert stages["correspondence_distance_mm"] > 0.0


def test_a_reading_from_the_true_pose_traces_as_right_throughout():
    _, model, cad_mesh, true_pose = _sample()
    record = trace_module.trace(
        model,
        cad_mesh,
        _reading(cad_mesh[0], true_pose),
        named_side_normal_cad_xy=NAMED_SIDE,
        register_kwargs=SMALL_SEARCH,
    )
    assert record["lost_at"] is None
    assert record["outcome"] == "right"
    assert all(record["right_candidate_present_at"].values())
    best = record["final"]["best"]
    assert best["gross_yaw_error_deg"] < 5.0
    assert best["wall_yaw_error_deg"] is not None
    assert best["wall_yaw_error_deg"] < 2.0
    # A clean synthetic part has no stray surface.
    assert record["final"]["observed_stray_share"] < 0.02


def test_a_half_turned_reading_is_reported_lost_where_refinement_leaves_it():
    # The yaw seeds cover every direction, so a candidate matching a reading 180 degrees from
    # the truth does leave the seeds and survive selection. Refinement then walks it back to the
    # true pose, and the trace has to say so, rather than blame the seeds or the ranking.
    _, model, cad_mesh, true_pose = _sample()
    record = trace_module.trace(
        model,
        cad_mesh,
        _reading(cad_mesh[0], true_pose, turn_deg=180.0),
        named_side_normal_cad_xy=NAMED_SIDE,
        register_kwargs=SMALL_SEARCH,
    )
    assert record["search"]["by_path"]["yaw"]["right"] >= 1
    assert record["right_candidate_present_at"]["seeds"]
    assert record["right_candidate_present_at"]["selection"]
    assert record["lost_at"] == "refinement"
    assert record["outcome"] != "right"
    # And the pose it ended on is the true one, half a turn from this reading.
    assert record["final"]["best"]["gross_yaw_error_deg"] > 170.0


def test_every_seed_is_recorded_with_its_fate():
    _, model, cad_mesh, true_pose = _sample()
    record = trace_module.trace(
        model,
        cad_mesh,
        _reading(cad_mesh[0], true_pose),
        named_side_normal_cad_xy=NAMED_SIDE,
        register_kwargs=SMALL_SEARCH,
    )
    seeds = record["seeds"]
    assert len(seeds) == record["search"]["seeded"]
    assert [seed["fitness_rank"] for seed in seeds] == list(range(1, len(seeds) + 1))
    assert {seed["path"] for seed in seeds} == {"ransac", "yaw"}
    for seed in seeds:
        # A seed can't be selected without surviving deduplication first.
        assert not (seed["selected"] and not seed["survived_deduplication"])
    assert len(record["refinement"]) == record["search"]["selected"]


def test_without_straight_walls_only_the_photo_judges_yaw():
    _, model, cad_mesh, true_pose = _sample()
    record = trace_module.trace(
        model,
        cad_mesh,
        _reading(cad_mesh[0], true_pose),
        named_side_normal_cad_xy=NAMED_SIDE,
        straight_walls=False,
        register_kwargs=SMALL_SEARCH,
    )
    assert record["final"]["best"]["wall_yaw_error_deg"] is None
    assert record["outcome"] == "right"


def test_command_line_writes_a_record_and_refuses_to_overwrite(tmp_path: Path):
    sample, _model, cad_mesh, true_pose = _sample()
    glb_path = tmp_path / "model.glb"
    synthetic.write_synthetic_sample(sample, glb_path, tmp_path / "model.json")
    reading = _reading(cad_mesh[0], true_pose)
    centre = reading.centre_xy_mm
    out = tmp_path / "trace.json"
    argv = [
        "--model",
        str(glb_path),
        "--cad",
        str(S04),
        "--out",
        str(out),
        "--faces",
        "right",
        # "right" is board angle 0, and seen from above clockwise increases the angle.
        "--turned",
        f"{reading.named_side_faces_deg % 360.0:.4f}",
        "--clockwise",
        "--squares-from-left",
        f"{centre[0] / pose_judgement.BOARD_SQUARE_MM:.4f}",
        "--squares-from-caption",
        f"{pose_judgement.BOARD_PATTERN_DEPTH_SQUARES - centre[1] / pose_judgement.BOARD_SQUARE_MM:.4f}",
        "--face",
        "matches",
        "--side-normal",
        str(NAMED_SIDE[0]),
        str(NAMED_SIDE[1]),
        "--straight-walls",
    ]
    assert trace_module.main(argv) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["outcome"] == "right"
    assert written["inputs"]["argv"] == argv
    np.testing.assert_allclose(written["reading"]["centre_xy_mm"], centre, atol=1e-3)

    with pytest.raises(FileExistsError):
        trace_module.main(argv)


def test_command_line_requires_a_reading(tmp_path: Path):
    with pytest.raises(SystemExit):
        trace_module.main(
            [
                "--model",
                str(tmp_path / "model.glb"),
                "--cad",
                str(S04),
                "--out",
                str(tmp_path / "trace.json"),
                "--side-normal",
                "0",
                "-1",
                "--straight-walls",
            ]
        )


def _present(*stages_with_right: str) -> dict[str, bool]:
    return {stage: stage in stages_with_right for stage in trace_module.STAGES}


def test_lost_at_is_none_when_the_final_pick_is_right_even_after_a_gap():
    # A right seed missed selection, but another selected seed refined into the right pose.
    assert (
        trace_module.stage_lost_at(
            _present("seeds", "deduplication", "refinement", "final ranking")
        )
        is None
    )


def test_lost_at_names_the_stage_after_the_last_one_with_a_right_pose():
    assert (
        trace_module.stage_lost_at(
            _present("seeds", "deduplication", "selection", "refinement")
        )
        == "final ranking"
    )
    # Present at the seeds, missing at selection, briefly back after refinement, lost at the
    # end: the loss that mattered is the final ranking, not selection.
    assert (
        trace_module.stage_lost_at(_present("seeds", "deduplication", "refinement"))
        == "final ranking"
    )
    assert trace_module.stage_lost_at(_present("seeds")) == "deduplication"


def test_lost_at_is_the_seeds_when_no_stage_ever_held_a_right_pose():
    assert trace_module.stage_lost_at(_present()) == "seeds"
