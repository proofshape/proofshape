"""Unit tests for A-07's tool-calling loop and its fallback to A-08's deterministic policy.
Same harness style as test_agent_policy.py (hand-built records, no mocking framework) plus
test_agent_tools.py's real `service.SessionStore` for the finish_session integration tests."""

from __future__ import annotations

import json
import time
from threading import Event
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from recon import service
from recon.agent_loop import (
    describe_action_for_supplier,
    parse_model_decision,
    run_agent_turn,
)
from recon.agent_policy import Finish, ReportComplete, ShootNextSlot
from recon.agent_tools import ToolResult


def _capturing_record() -> dict:
    return {"status": "capturing", "reconstruction": {"status": "not_started"}}


def _reply(**decision) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(decision))


def _fixed_clock(values: list[float]):
    it = iter(values)

    def clock() -> float:
        return next(it)

    return clock


# --- parse_model_decision -------------------------------------------------------------------


def test_parses_a_tool_call():
    decision = parse_model_decision(_reply(tool="finish_session"))
    assert decision.tool == "finish_session"
    assert decision.done is False


def test_parses_a_done_decision_with_a_message():
    decision = parse_model_decision(_reply(done=True, message="All set."))
    assert decision.done is True
    assert decision.message == "All set."


def test_rejects_a_decision_naming_no_tool_and_not_done():
    with pytest.raises(ValidationError):
        parse_model_decision(_reply())


def test_rejects_a_decision_that_both_names_a_tool_and_is_done():
    with pytest.raises(ValidationError):
        parse_model_decision(_reply(tool="finish_session", done=True))


def test_rejects_an_empty_reply():
    with pytest.raises(ValueError):
        parse_model_decision(SimpleNamespace(text=""))


def test_rejects_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        parse_model_decision(SimpleNamespace(text="not json"))


# --- describe_action_for_supplier -------------------------------------------------------------


def test_describes_every_action_without_raising():
    for action in (
        ShootNextSlot(orbit_index=2),
        Finish(),
        ReportComplete(under_observed_vertex_count=None),
        ReportComplete(under_observed_vertex_count=7),
    ):
        assert isinstance(describe_action_for_supplier(action), str)


# --- run_agent_turn: the three forced fallback triggers ---------------------------------------


def test_falls_back_on_a_model_error(tmp_path):
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        raise RuntimeError("boom")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.fallback_reason == "model_error"
    assert result.finished is False


def test_falls_back_on_a_malformed_reply(tmp_path):
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        return SimpleNamespace(text="not json")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.fallback_reason == "model_error"


def test_model_path_runs_only_the_requested_tool_and_caches_its_result(
    tmp_path, monkeypatch
):
    store = service.SessionStore(tmp_path)
    calls: list[str] = []
    states = []

    def score_next_view(record):
        calls.append("score_next_view")
        return ToolResult(
            available=True, data={"best": {"under_observed_visible_count": 2}}
        )

    def unexpected_tool(*_args, **_kwargs):
        raise AssertionError("an unselected tool ran")

    monkeypatch.setattr("recon.agent_loop.score_next_view", score_next_view)
    monkeypatch.setattr("recon.agent_loop.check_frame", unexpected_tool)
    monkeypatch.setattr("recon.agent_loop.vlm_inspect", unexpected_tool)
    monkeypatch.setattr("recon.agent_loop.coverage_report", unexpected_tool)

    def model_caller(state):
        states.append(state)
        if len(states) <= 2:
            return _reply(tool="score_next_view")
        return _reply(done=True, message="Two spots still need a clearer view.")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )

    assert result.fallback_reason is None
    assert calls == ["score_next_view"]
    assert states[0].tools.score_next_view.reason == "not_requested"
    assert states[1].tools.score_next_view.data == {
        "best": {"under_observed_visible_count": 2}
    }
    assert states[1].tools.coverage_report.reason == "not_requested"


def test_falls_back_on_timeout(tmp_path):
    store = service.SessionStore(tmp_path)
    clock = _fixed_clock(
        [0.0, 100.0]
    )  # started=0.0; the first budget check is already spent

    def model_caller(state):
        raise AssertionError("the model must never be called once the budget is spent")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
        timeout_s=1.0,
        now=clock,
    )
    assert result.fallback_reason == "timeout"


def test_falls_back_on_timeout_even_when_the_call_itself_never_returns_in_time(
    tmp_path,
):
    # Real clock, on purpose (caught on review, @TabeenRaoof, #318): the fake-clock test above
    # only proves the budget is checked *between* calls. This proves the call itself is bounded
    # -- a synchronous model_caller that blocks past the deadline must still come back as
    # "timeout" promptly, not after however long the call actually takes (or never, if it never
    # returns at all). Bounded to 1s, not unbounded, so the worker thread (which Python cannot
    # kill once started) still finishes and this process can exit cleanly after the suite ends.
    store = service.SessionStore(tmp_path)

    def hangs_past_the_deadline(state):
        Event().wait(timeout=1.0)

    started = time.monotonic()
    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        hangs_past_the_deadline,
        min_frames=3,
        timeout_s=0.2,
    )
    elapsed = time.monotonic() - started
    assert result.fallback_reason == "timeout"
    assert elapsed < 0.5  # a small bound well under the call's own 1s internal wait


def test_falls_back_on_a_repeated_tool_call(tmp_path):
    store = service.SessionStore(tmp_path)
    calls: list[int] = []

    def model_caller(state):
        calls.append(1)
        return _reply(tool="score_next_view")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
        max_repeated_calls=2,
    )
    assert result.fallback_reason == "loop_detected"
    # Two identical calls are recorded, and the model is asked a third time -- only then does
    # its answer (still the same tool) confirm the stall and trip the fallback.
    assert len(calls) == 3


def test_falls_back_when_max_steps_is_exhausted_without_ever_repeating(tmp_path):
    store = service.SessionStore(tmp_path)
    # Alternating, never two in a row the same -- max_repeated_calls never trips, only max_steps.
    script = iter(
        ["score_next_view", "coverage_report", "score_next_view", "coverage_report"]
    )

    def model_caller(state):
        return _reply(tool=next(script))

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
        max_steps=4,
        max_repeated_calls=10,
    )
    assert result.fallback_reason == "loop_detected"


# --- run_agent_turn: fallback still finishes the session when the action is Finish() ----------


def test_fallback_still_calls_finish_session_when_the_action_is_finish(
    tmp_path, monkeypatch
):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    for _ in range(3):
        store.add_frame(
            record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        )
    record = store.load(record["session_id"])

    # Force next_action's "still capturing" branch to see a fully-covered orbit, without
    # depending on real gyro-angle math to happen to produce exactly that -- this test is about
    # the fallback-still-finishes wiring, not about compute_coverage.
    fully_covered = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [True], "next_target_index": None}},
    )
    monkeypatch.setattr(
        "recon.agent_loop.coverage_report", lambda session_dir, rec: fully_covered
    )

    def model_caller(state):
        raise RuntimeError("force the fallback path")

    scheduled: list[str] = []
    result = run_agent_turn(
        record,
        store,
        record["session_id"],
        scheduled.append,
        model_caller,
        min_frames=1,
    )
    assert result.fallback_reason == "model_error"
    assert result.finished is True
    assert scheduled == [record["session_id"]]


def test_fallback_finish_reports_the_real_error_when_no_session_exists(
    tmp_path, monkeypatch
):
    # The fallback action is Finish() (forced via the same monkeypatch as the success case
    # above), but there is no real session for finish_session to find -- the fallback's
    # narration must be replaced by finish_session's own error, not silently say "Finishing up."
    store = service.SessionStore(tmp_path)
    fully_covered = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [True], "next_target_index": None}},
    )
    monkeypatch.setattr(
        "recon.agent_loop.coverage_report", lambda session_dir, rec: fully_covered
    )

    def model_caller(state):
        raise RuntimeError("force the fallback path")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_missing",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.fallback_reason == "model_error"
    assert result.finished is False
    assert result.message == "No session with that id."


def test_fallback_with_no_real_session_still_reports_rather_than_crashing(tmp_path):
    # No real session exists for "sess_x", and coverage_report is genuinely available (an empty
    # gyro glob is a valid, if uncovered, orbit) -- next_action's real answer here is
    # ShootNextSlot, not Finish(), so finish_session is never even reached. Confirms the
    # fallback integrates cleanly with a nonexistent session either way.
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        raise RuntimeError("force the fallback path")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.fallback_reason == "model_error"
    assert result.finished is False
    assert (
        result.message
        == "Keep going -- more of the part still needs to be photographed."
    )


# --- run_agent_turn: the normal path ------------------------------------------------------------


def test_normal_path_calls_finish_session_when_the_model_asks_to(tmp_path):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    for _ in range(3):
        store.add_frame(
            record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        )
    record = store.load(record["session_id"])

    def model_caller(state):
        return _reply(tool="finish_session", message="Looks complete, finishing up.")

    scheduled: list[str] = []
    result = run_agent_turn(
        record,
        store,
        record["session_id"],
        scheduled.append,
        model_caller,
        min_frames=1,
    )
    assert result.finished is True
    assert result.fallback_reason is None
    assert result.message == "Looks complete, finishing up."
    assert scheduled == [record["session_id"]]


def test_normal_path_takes_several_steps_before_finishing(tmp_path):
    store = service.SessionStore(tmp_path)
    record = store.create("order-1")
    for _ in range(3):
        store.add_frame(
            record["session_id"], b"jpeg", {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        )
    record = store.load(record["session_id"])
    script = iter(
        [
            _reply(tool="coverage_report"),
            _reply(tool="score_next_view"),
            _reply(tool="finish_session", message="done"),
        ]
    )

    def model_caller(state):
        return next(script)

    result = run_agent_turn(
        record,
        store,
        record["session_id"],
        lambda sid: None,
        model_caller,
        min_frames=1,
    )
    assert result.finished is True
    assert [step.tool for step in result.transcript] == [
        "coverage_report",
        "score_next_view",
        "finish_session",
    ]


def test_normal_path_can_end_with_done_and_no_tool_call(tmp_path):
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        return _reply(done=True, message="Still thinking -- nothing to report yet.")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.finished is False
    assert result.fallback_reason is None
    assert result.message == "Still thinking -- nothing to report yet."


# --- unavailable tools (D-054's real, current state) don't crash either path ------------------


def test_unavailable_check_frame_and_vlm_inspect_do_not_crash_the_normal_path(tmp_path):
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        assert state.tools.check_frame.available is False
        assert state.tools.vlm_inspect.available is False
        return _reply(done=True, message="ok")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.fallback_reason is None
    assert result.message == "ok"


def test_unavailable_tools_do_not_crash_the_fallback_path(tmp_path):
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        raise RuntimeError("force fallback")

    result = run_agent_turn(
        _capturing_record(),
        store,
        "sess_x",
        lambda sid: None,
        model_caller,
        min_frames=3,
    )
    assert result.fallback_reason == "model_error"


# --- determinism of the fallback path, same spirit as A-08's own test -------------------------


def test_fallback_is_deterministic_for_the_same_input(tmp_path):
    store = service.SessionStore(tmp_path)

    def model_caller(state):
        raise RuntimeError("force fallback")

    record = _capturing_record()
    first = run_agent_turn(
        record, store, "sess_x", lambda sid: None, model_caller, min_frames=3
    )
    second = run_agent_turn(
        record, store, "sess_x", lambda sid: None, model_caller, min_frames=3
    )
    assert first == second
