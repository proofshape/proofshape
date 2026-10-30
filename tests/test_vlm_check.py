"""A-01: deterministic fake-client checks, no credentials or network."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from recon.vlm_check import (
    MODEL,
    GeminiInspector,
    VlmSupervisor,
    parse_response,
    sampled_indices,
)


def reply(index=0, **changes):
    observation = {
        "same_part": None if index == 0 else True,
        "hands_occluding": False,
        "screen_photo": False,
        "lighting_adequate": True,
        "hint": "Keep the part visible.",
    }
    observation.update(changes)
    return SimpleNamespace(
        text=json.dumps(observation),
        model_version="returned-model-version",
        candidates=[SimpleNamespace(finish_reason="STOP")],
        prompt_feedback=None,
        usage_metadata=SimpleNamespace(prompt_token_count=10, total_token_count=20),
    )


@pytest.mark.parametrize(
    "count,expected",
    [(0, []), (1, [0]), (5, [0, 4]), (6, [0, 5]), (12, [0, 5, 10, 11])],
)
def test_sample_indices_include_final_exactly_once(count, expected):
    assert sampled_indices(count) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"same_part": True},
        {"hands_occluding": "false"},
        {"hint": ""},
        {"hint": "line\nline"},
        {"hint": "a" * 241},
        {"extra": True},
    ],
)
def test_invalid_reply_is_unchecked_not_coerced(changes):
    result = parse_response(0, reply(**changes))
    assert result.status == "unchecked"
    assert result.reason == "invalid_reply"
    assert result.observation is None


def test_schema_requires_comparison_on_later_frames_and_records_real_metadata():
    result = parse_response(5, reply(5))
    assert result.status == "checked"
    assert result.model_version == "returned-model-version"
    assert result.token_usage == {"prompt_token_count": 10, "total_token_count": 20}
    assert parse_response(5, reply()).reason == "invalid_reply"


def test_empty_blocked_incomplete_and_malformed_replies():
    response = reply()
    response.candidates = []
    assert parse_response(0, response).reason == "empty_reply"
    response.prompt_feedback = SimpleNamespace(block_reason="SAFETY")
    assert parse_response(0, response).reason == "blocked_reply"
    response = reply()
    response.candidates[0].finish_reason = "MAX_TOKENS"
    assert parse_response(0, response).reason == "incomplete_reply"
    response = reply()
    response.text = ""
    assert parse_response(0, response).reason == "empty_reply"
    response.text = "not json"
    assert parse_response(0, response).reason == "invalid_reply"


def test_no_key_and_missing_result_are_explicit(tmp_path):
    supervisor = VlmSupervisor()
    try:
        supervisor.submit(tmp_path, 0)
        supervisor.submit(tmp_path, 1)  # not a sampled index
        assert not supervisor.result_path(tmp_path, 1).exists()
        report = supervisor.join(tmp_path, 3)
        assert [c["reason"] for c in report["checks"]] == [
            "missing_api_key",
            "missing_result",
        ]
    finally:
        supervisor.close()


def test_bounded_executor_dedup_and_timeout_cannot_be_overwritten(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(frame, reference):
        calls.append((frame.name, reference))
        entered.set()
        assert release.wait(5)
        return reply()

    supervisor = VlmSupervisor(
        inspector=blocked, max_workers=1, max_pending=1, join_timeout_s=0.02
    )
    try:
        supervisor.submit(tmp_path, 0)
        assert entered.wait(1)
        supervisor.submit(tmp_path, 0)
        started = time.monotonic()
        supervisor.submit(tmp_path, 5)
        assert time.monotonic() - started < 0.5
        report = supervisor.join(tmp_path, 6)
        assert [c["reason"] for c in report["checks"]] == ["join_timeout", "queue_full"]
        release.set()
        supervisor._executor.shutdown(wait=True)
        assert (
            json.loads(supervisor.result_path(tmp_path, 0).read_text())["reason"]
            == "join_timeout"
        )
        assert len(calls) == 1
    finally:
        release.set()
        supervisor.close()


@pytest.mark.parametrize(
    "error,reason",
    [
        (TimeoutError("key=secret"), "api_timeout"),
        (RuntimeError("key=secret"), "api_error"),
        (FileNotFoundError("secret"), "frame_read_error"),
    ],
)
def test_worker_errors_are_recorded_without_exception_secrets(tmp_path, error, reason):
    def broken(frame, reference):
        raise error

    supervisor = VlmSupervisor(inspector=broken)
    try:
        supervisor.schedule_finish(tmp_path, 1)
        report = supervisor.join(tmp_path, 1)
        assert report["checks"][0]["reason"] == reason
        assert "secret" not in json.dumps(report)
    finally:
        supervisor.close()


def test_sdk_adapter_sends_reference_only_after_frame_zero_and_closes_client(
    tmp_path, monkeypatch
):
    from google import genai
    from google.genai import types

    frame = tmp_path / "frame.jpg"
    reference = tmp_path / "reference.jpg"
    frame.write_bytes(b"current")
    reference.write_bytes(b"reference")
    requests, clients = [], []

    class FakeClient:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.models = SimpleNamespace(generate_content=self.generate)

        def generate(self, **kwargs):
            requests.append(kwargs)
            return reply()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            clients[-1]["closed"] = True

    monkeypatch.setattr(genai, "Client", FakeClient)
    inspector = GeminiInspector("private-key", timeout_s=1.25)
    inspector(frame, None)
    inspector(frame, reference)
    assert requests[0]["model"] == MODEL
    assert len(requests[0]["contents"]) == 2
    assert len(requests[1]["contents"]) == 4
    assert requests[1]["contents"][1].inline_data.data == b"reference"
    assert requests[1]["contents"][3].inline_data.data == b"current"
    for client in clients:
        assert client["api_key"] == "private-key"
        assert client["vertexai"] is False
        assert client["http_options"].timeout == 1250
        assert client["http_options"].retry_options.attempts == 1
        assert client["closed"]
    # Parse a real SDK object too, not just a duck-typed fake response.
    sdk_reply = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(parts=[types.Part.from_text(text=reply().text)]),
            )
        ],
        model_version=MODEL,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, total_token_count=20
        ),
    )
    assert parse_response(0, sdk_reply).status == "checked"


def test_join_timeout_is_one_deadline_and_sessions_do_not_mix(tmp_path):
    release = threading.Event()

    def blocked(frame, reference):
        assert release.wait(5)
        return reply()

    first, second = tmp_path / "first", tmp_path / "second"
    supervisor = VlmSupervisor(
        inspector=blocked, max_workers=2, max_pending=4, join_timeout_s=0.03
    )
    try:
        supervisor.schedule_finish(first, 11)
        supervisor.submit(second, 0)
        started = time.monotonic()
        report = supervisor.join(first, 11)
        assert time.monotonic() - started < 0.3
        assert len(report["checks"]) == 3
        assert all(c["reason"] == "join_timeout" for c in report["checks"])
        assert not (second / "vlm_checks.json").exists()
    finally:
        release.set()
        supervisor.close()


def test_restart_preserves_existing_evidence_and_corrupt_result_is_explicit(tmp_path):
    supervisor = VlmSupervisor()
    supervisor.schedule_finish(tmp_path, 3)
    supervisor.close()
    path = supervisor.result_path(tmp_path, 0)
    original = path.read_text()
    supervisor = VlmSupervisor()
    try:
        supervisor.schedule_finish(tmp_path, 3)
        assert path.read_text() == original
        path.write_text("{broken")
        report = supervisor.join(tmp_path, 3)
        assert report["checks"][0]["reason"] == "invalid_result_file"
        assert json.loads(path.read_text()) == report["checks"][0]
    finally:
        supervisor.close()


def test_shutdown_cancels_queued_work_and_records_closed_submissions(tmp_path):
    release, entered = threading.Event(), threading.Event()

    def blocked(frame, reference):
        entered.set()
        assert release.wait(5)
        return reply()

    supervisor = VlmSupervisor(inspector=blocked, max_workers=1, max_pending=2)
    try:
        supervisor.submit(tmp_path, 0)
        assert entered.wait(1)
        supervisor.submit(tmp_path, 5)
        supervisor.close()
        supervisor.submit(tmp_path, 10)
        for index in (5, 10):
            assert (
                json.loads(supervisor.result_path(tmp_path, index).read_text())[
                    "reason"
                ]
                == "supervisor_closed"
            )
    finally:
        release.set()
        supervisor.close()
