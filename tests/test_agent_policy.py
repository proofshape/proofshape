"""Unit tests for A-08's deterministic capture policy. No language model, no GPU: every scenario
is a hand-built (record, ToolResults) pair, asserting the same input always gives the same
action."""

from __future__ import annotations

from recon.agent_policy import (
    AwaitReconstruction,
    ExplainRejection,
    Finish,
    ReportComplete,
    ReportFailure,
    ShootNextSlot,
    ToolResults,
    next_action,
)
from recon.agent_tools import ToolResult

UNAVAILABLE = ToolResult(available=False, reason="not available")


def _tools(
    *,
    check_frame=UNAVAILABLE,
    vlm_inspect=UNAVAILABLE,
    coverage_report=UNAVAILABLE,
    score_next_view=UNAVAILABLE,
) -> ToolResults:
    return ToolResults(
        check_frame=check_frame,
        vlm_inspect=vlm_inspect,
        coverage_report=coverage_report,
        score_next_view=score_next_view,
    )


def _capturing_record() -> dict:
    return {"status": "capturing", "reconstruction": {"status": "not_started"}}


def _finished_record(reconstruction: dict) -> dict:
    return {"status": "finished", "reconstruction": reconstruction}


# --- normal session: coverage drives ShootNextSlot, then Finish once covered ------------------


def test_shoots_the_next_uncovered_orbit_slot():
    coverage = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [True, False], "next_target_index": 1}},
    )
    action = next_action(_capturing_record(), _tools(coverage_report=coverage))
    assert action == ShootNextSlot(orbit_index=1)


def test_finishes_once_the_orbit_is_fully_covered():
    coverage = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [True, True], "next_target_index": None}},
    )
    action = next_action(_capturing_record(), _tools(coverage_report=coverage))
    assert action == Finish()


def test_is_deterministic_for_the_same_input():
    coverage = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [False], "next_target_index": 0}},
    )
    record = _capturing_record()
    tools = _tools(coverage_report=coverage)
    assert next_action(record, tools) == next_action(record, tools)


# --- repeated rejections: ExplainRejection, inert today but real and tested -------------------


def test_explains_a_rejection_when_check_frame_reports_one():
    # check_frame is always `unavailable` in production today (D-054) -- this constructs the
    # hypothetical shape it produced before that fix, to prove the branch itself is correct and
    # ready for whenever a future story adds real gate telemetry back.
    rejection = ToolResult(
        available=True,
        data={
            "accepted": False,
            "code": "frame_rejected_blur",
            "message": "too blurry",
        },
    )
    action = next_action(_capturing_record(), _tools(check_frame=rejection))
    assert action == ExplainRejection(code="frame_rejected_blur", message="too blurry")


def test_explains_the_same_rejection_every_time_it_repeats():
    rejection = ToolResult(
        available=True,
        data={
            "accepted": False,
            "code": "frame_rejected_duplicate",
            "message": "no movement",
        },
    )
    record = _capturing_record()
    tools = _tools(check_frame=rejection)
    first = next_action(record, tools)
    second = next_action(record, tools)
    assert (
        first
        == second
        == ExplainRejection(code="frame_rejected_duplicate", message="no movement")
    )


def test_an_accepted_check_frame_does_not_trigger_an_explanation():
    accepted = ToolResult(available=True, data={"accepted": True, "frame_index": 3})
    coverage = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [False], "next_target_index": 0}},
    )
    action = next_action(
        _capturing_record(), _tools(check_frame=accepted, coverage_report=coverage)
    )
    assert action == ShootNextSlot(orbit_index=0)


# --- a failed reconstruction --------------------------------------------------------------------


def test_reports_a_failed_reconstruction():
    record = _finished_record(
        {
            "status": "failed",
            "failure": {"code": "metric_alignment_failed", "message": "boom"},
        }
    )
    action = next_action(record, _tools())
    assert action == ReportFailure(code="metric_alignment_failed", message="boom")


def test_failure_report_survives_a_missing_failure_dict():
    record = _finished_record({"status": "failed"})
    action = next_action(record, _tools())
    assert action == ReportFailure(code="unknown", message="")


# --- a completed reconstruction ------------------------------------------------------------------


def test_reports_remaining_under_observed_vertices_after_completion():
    record = _finished_record({"status": "complete"})
    score = ToolResult(
        available=True, data={"best": {"under_observed_visible_count": 42}}
    )
    action = next_action(record, _tools(score_next_view=score))
    assert action == ReportComplete(under_observed_vertex_count=42)


def test_reports_nothing_outstanding_once_fully_observed():
    record = _finished_record({"status": "complete"})
    score = ToolResult(
        available=True, data={"best": {"under_observed_visible_count": 0}}
    )
    action = next_action(record, _tools(score_next_view=score))
    assert action == ReportComplete(under_observed_vertex_count=None)


def test_reports_nothing_outstanding_when_score_next_view_is_unavailable():
    record = _finished_record({"status": "complete"})
    action = next_action(record, _tools())
    assert action == ReportComplete(under_observed_vertex_count=None)


# --- finished but reconstruction still pending: the bug @TabeenRaoof caught in review ----------


def test_awaits_reconstruction_when_finished_but_still_pending():
    record = _finished_record({"status": "pending"})
    action = next_action(record, _tools())
    assert action == AwaitReconstruction()


def test_awaits_reconstruction_rather_than_shooting_more_slots():
    # Before the fix: a finished session with an uncovered orbit fell through to the
    # still-capturing branch and asked for more photos on a session that can no longer accept
    # uploads.
    record = _finished_record({"status": "pending"})
    coverage = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [False, False], "next_target_index": 0}},
    )
    action = next_action(record, _tools(coverage_report=coverage))
    assert action == AwaitReconstruction()


def test_awaits_reconstruction_rather_than_finishing_a_second_time():
    # Before the fix: a finished session with a fully-covered orbit fell through to Finish(),
    # so the caller would call finish_session again and get session_already_finished.
    record = _finished_record({"status": "pending"})
    coverage = ToolResult(
        available=True,
        data={"orbit_coverage": {"covered": [True, True], "next_target_index": None}},
    )
    action = next_action(record, _tools(coverage_report=coverage))
    assert action == AwaitReconstruction()


# --- every tool unavailable: never raises, never stalls -----------------------------------------


def test_falls_through_to_finish_when_every_tool_is_unavailable():
    action = next_action(_capturing_record(), _tools())
    assert action == Finish()


def test_falls_through_to_finish_when_coverage_report_is_unavailable_mid_capture():
    # Defensive: A-06's real coverage_report is never actually unavailable (it only needs gyro
    # files that always exist), but the story's own Tests section asks for this edge case.
    action = next_action(_capturing_record(), _tools(coverage_report=UNAVAILABLE))
    assert action == Finish()
