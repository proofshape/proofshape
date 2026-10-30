"""Unit tests for A-06's five agent tools. No language model, no GPU -- every tool is
deterministic, so these are pure function tests plus a couple of service-level wiring checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recon import agent_tools, service
from recon.mesh_cleanup import write_glb
from recon.provenance import LABEL_OBSERVED, LABEL_UNOBSERVED, labels_to_colors

# --- a tiny watertight mesh, for score_next_view / coverage_report's post-reconstruction path ---


def _tetrahedron_glb(path: Path) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, -10.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    labels = np.array(
        [LABEL_UNOBSERVED, LABEL_OBSERVED, LABEL_OBSERVED, LABEL_OBSERVED], dtype=object
    )
    write_glb(path, vertices, faces, labels_to_colors(labels))


def _write_gyro_files(session_dir: Path, gyros: list[dict[str, float]]) -> None:
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, gyro in enumerate(gyros):
        (frames_dir / f"frame_{index:04d}.gyro.json").write_text(
            json.dumps(gyro, sort_keys=True) + "\n", encoding="utf-8"
        )


# --- compute_coverage, cross-checked against capture/src/orbit.test.ts's own cases -------------


def _gyro(alpha: float, beta: float, gamma: float = 0.0) -> dict[str, float]:
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def _elevation_to_beta(elevation_deg: float) -> float:
    return 90.0 - elevation_deg  # the inverse orbit.test.ts's own helper uses


def test_coverage_nothing_with_no_readings():
    state = agent_tools.compute_coverage([])
    assert not any(state["covered"])
    assert state["next_target_index"] == 0


def test_coverage_covers_the_first_slot_within_tolerance():
    # DEFAULT_ORBIT[0] = azimuth 0, elevation 45. Anchor at alpha=0 -> relative azimuth 0.
    readings = [_gyro(0.0, _elevation_to_beta(45.0))]
    state = agent_tools.compute_coverage(readings)
    assert state["covered"][0] is True
    assert state["next_target_index"] == 1


def test_coverage_does_not_cover_just_outside_tolerance():
    anchor_reading = _gyro(0.0, _elevation_to_beta(10.0))
    just_outside = agent_tools.DEFAULT_COVERAGE_TOLERANCE_DEG + 1
    tested_reading = _gyro(just_outside, _elevation_to_beta(45.0))
    state = agent_tools.compute_coverage([anchor_reading, tested_reading])
    assert state["covered"][0] is False
    assert state["next_target_index"] == 0


def test_coverage_next_target_is_none_once_everything_is_covered():
    anchor = 0.0
    readings = [_gyro(anchor, _elevation_to_beta(45.0))]
    readings += [
        _gyro(anchor + slot.azimuth_deg, _elevation_to_beta(slot.elevation_deg))
        for slot in agent_tools.DEFAULT_ORBIT[1:]
    ]
    state = agent_tools.compute_coverage(readings)
    assert all(state["covered"])
    assert state["next_target_index"] is None


def test_coverage_exempts_azimuth_for_the_straight_down_slot():
    stray_azimuth_reading = _gyro(999 % 360, _elevation_to_beta(90.0))
    state = agent_tools.compute_coverage([_gyro(0.0, 0.0), stray_azimuth_reading])
    straight_down_index = next(
        i
        for i, slot in enumerate(agent_tools.DEFAULT_ORBIT)
        if slot.elevation_deg == 90.0
    )
    assert state["covered"][straight_down_index] is True


def test_coverage_assigns_a_reading_to_only_its_nearest_slot():
    mid_ring_index = 0  # azimuth 0, elevation 45
    high_ring_index = 8  # azimuth 0, elevation 70
    anchor = _gyro(0.0, _elevation_to_beta(45.0))
    ambiguous_reading = _gyro(0.0, _elevation_to_beta(57.0))
    state = agent_tools.compute_coverage([anchor, ambiguous_reading])
    assert state["covered"][mid_ring_index] is True
    assert state["covered"][high_ring_index] is False


# --- coverage_report / score_next_view ----------------------------------------------------------


def test_coverage_report_before_reconstruction_has_no_under_observed_count(tmp_path):
    session_dir = tmp_path / "sess_1"
    _write_gyro_files(session_dir, [_gyro(0.0, _elevation_to_beta(45.0))])
    record = {"reconstruction": {"status": "not_started"}}

    result = agent_tools.coverage_report(session_dir, record)

    assert result.available is True
    assert result.data["orbit_coverage"]["covered"][0] is True
    assert "under_observed_vertex_count" not in result.data


def test_coverage_report_after_reconstruction_adds_the_under_observed_count(tmp_path):
    session_dir = tmp_path / "sess_2"
    _write_gyro_files(session_dir, [])
    glb_path = tmp_path / "model.glb"
    _tetrahedron_glb(glb_path)
    record = {"reconstruction": {"status": "complete", "glb_path": str(glb_path)}}

    result = agent_tools.coverage_report(session_dir, record)

    assert result.data["under_observed_vertex_count"] >= 0


def test_score_next_view_unavailable_before_reconstruction():
    result = agent_tools.score_next_view({"reconstruction": {"status": "not_started"}})
    assert result.available is False
    assert result.reason == "no_reconstruction"


def test_score_next_view_wraps_the_real_mesh_after_reconstruction(tmp_path):
    glb_path = tmp_path / "model.glb"
    _tetrahedron_glb(glb_path)
    record = {"reconstruction": {"status": "complete", "glb_path": str(glb_path)}}

    result = agent_tools.score_next_view(record)

    assert result.available is True
    assert len(result.data["ranked"]) > 0
    assert result.data["best"] == result.data["ranked"][0]
    assert result.data["best"]["under_observed_visible_count"] >= 0


# --- check_frame ---------------------------------------------------------------------------


def test_check_frame_is_always_unavailable():
    # D-054, revised after review: the gate's actual decision never reaches the server (a
    # rejected frame is never uploaded), so this tool does not invent a different answer under
    # its name -- same honesty vlm_inspect already applies to A-01 not existing yet.
    assert agent_tools.check_frame({}).available is False
    assert agent_tools.check_frame({"frame_count": 5}).available is False


def test_check_frame_is_deterministic():
    record = {"frame_count": 3}
    assert agent_tools.check_frame(record) == agent_tools.check_frame(record)


# --- vlm_inspect ---------------------------------------------------------------------------


def test_vlm_inspect_needs_an_explicit_session_directory():
    assert agent_tools.vlm_inspect({}).available is False
    assert (
        agent_tools.vlm_inspect({"anything": "here"}).reason
        == "session directory is required for VLM evidence"
    )


# --- finish_session --------------------------------------------------------------------------


def test_finish_session_not_found(tmp_path):
    store = service.SessionStore(tmp_path)
    result = agent_tools.finish_session(
        store, "sess_missing", lambda sid: None, min_frames=3
    )
    assert result.ok is False
    assert result.error_code == "session_not_found"


def test_finish_session_too_few_frames(tmp_path):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    result = agent_tools.finish_session(
        store, record["session_id"], lambda sid: None, min_frames=3
    )
    assert result.ok is False
    assert result.error_code == "too_few_frames"


def test_finish_session_already_finished(tmp_path):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    for _ in range(3):
        store.add_frame(
            record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        )
    first = agent_tools.finish_session(
        store, record["session_id"], lambda sid: None, min_frames=3
    )
    assert first.ok is True

    second = agent_tools.finish_session(
        store, record["session_id"], lambda sid: None, min_frames=3
    )
    assert second.ok is False
    assert second.error_code == "session_already_finished"


def test_finish_session_handles_a_genuine_mark_finished_race(tmp_path, monkeypatch):
    # Distinct from test_finish_session_already_finished above: that test hits the early
    # `record["status"] != "capturing"` check on its second call, never reaching
    # `store.mark_finished()` at all. This test makes the early checks pass (record still says
    # "capturing") and has `mark_finished` itself raise, as a real concurrent race would.
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    for _ in range(3):
        store.add_frame(
            record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        )

    def losing_race(session_id):
        raise service.SessionAlreadyFinishedError("session_already_finished")

    monkeypatch.setattr(store, "mark_finished", losing_race)

    result = agent_tools.finish_session(
        store, record["session_id"], lambda sid: None, min_frames=3
    )

    assert result.ok is False
    assert result.error_code == "session_already_finished"


def test_finish_session_success_schedules_reconstruction(tmp_path):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    for _ in range(3):
        store.add_frame(
            record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        )
    scheduled: list[str] = []

    result = agent_tools.finish_session(
        store, record["session_id"], scheduled.append, min_frames=3
    )

    assert result.ok is True
    assert result.record["status"] == "finished"
    assert scheduled == [record["session_id"]]


@pytest.mark.parametrize("min_frames", [1])
def test_finish_session_respects_custom_min_frames(tmp_path, min_frames):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    store.add_frame(
        record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
    )
    result = agent_tools.finish_session(
        store, record["session_id"], lambda sid: None, min_frames=min_frames
    )
    assert result.ok is True


def test_vlm_inspect_uses_separate_evidence_not_fabricated_record(tmp_path):
    from recon.vlm_check import FrameCheck, FrameObservation, atomic_json, unchecked

    assert agent_tools.vlm_inspect({}, tmp_path).reason == "no VLM result yet"
    result = FrameCheck(
        frame_index=0,
        status="checked",
        model_version="real-version",
        observation=FrameObservation(
            same_part=None,
            hands_occluding=False,
            screen_photo=True,
            lighting_adequate=True,
            hint="Photograph the physical part.",
        ),
    )
    atomic_json(tmp_path / "vlm" / "frame_0000.json", result.model_dump())
    tool = agent_tools.vlm_inspect({}, tmp_path)
    assert tool.available and tool.data["observation"]["screen_photo"] is True
    atomic_json(
        tmp_path / "vlm" / "frame_0005.json", unchecked(5, "api_timeout").model_dump()
    )
    tool = agent_tools.vlm_inspect({"vlm": "pretend pass"}, tmp_path)
    assert not tool.available and tool.reason == "unchecked: api_timeout"
