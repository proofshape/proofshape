"""A-15 service wiring tests for the real phone-facing capture-agent route."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from recon import service


@pytest.fixture(autouse=True)
def _no_real_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    glb_path = artifact_dir / "model.glb"
    glb_path.write_bytes(b"glTF-agent-test")
    return {
        "glb_path": str(glb_path),
        "reference_tier_used": "charuco_board",
        "metric": True,
        "stage_timings_s": {},
        "warnings": [],
    }


def _reply(**decision):
    return SimpleNamespace(text=json.dumps(decision))


def _create(tmp_path, *, caller=None, timeout_s=1.0):
    return service.create_app(
        tmp_path,
        pipeline_runner=_pipeline,
        known_order_codes={"PO-1"},
        agent_model_caller=caller,
        agent_timeout_s=timeout_s,
    )


def _start(client: TestClient) -> str:
    response = client.post("/v1/sessions", json={"order_code": "PO-1"})
    assert response.status_code == 201
    return response.json()["session_id"]


def _upload(client: TestClient, session_id: str, index: int = 0):
    gyro = json.dumps({"alpha": float(index * 45), "beta": 45.0, "gamma": 0.0})
    return client.post(
        f"/v1/sessions/{session_id}/frames",
        files={
            "frame": (f"frame{index}.jpg", b"jpeg-data", "image/jpeg"),
            "gyro": ("blob", gyro, "application/json"),
        },
    )


def test_model_reply_is_persisted_and_get_is_read_only(tmp_path):
    calls = []

    def caller(state):
        calls.append(state)
        if state.tools.coverage_report.reason == "not_requested":
            return _reply(tool="coverage_report", message="Checking coverage.")
        return _reply(done=True, message="Move to the next guided view.")

    app = _create(tmp_path, caller=caller)
    with TestClient(app) as client:
        sid = _start(client)
        assert _upload(client, sid).status_code == 202
        first_call_count = len(calls)

        result = client.get(f"/v1/sessions/{sid}/agent-message")
        assert result.status_code == 200
        assert result.json() == {
            "finished": False,
            "fallback_reason": None,
            "message": "Move to the next guided view.",
        }

        # Polling the GET must never run another model turn.
        assert client.get(f"/v1/sessions/{sid}/agent-message").status_code == 200
        assert len(calls) == first_call_count


@pytest.mark.parametrize(
    ("caller_factory", "timeout_s", "reason"),
    [
        (lambda: (lambda _state: (_ for _ in ()).throw(RuntimeError("boom"))), 1.0, "model_error"),
        (
            lambda: (lambda _state: (time.sleep(0.05), _reply(done=True, message="late"))[1]),
            0.01,
            "timeout",
        ),
        (lambda: (lambda _state: _reply(tool="coverage_report")), 1.0, "loop_detected"),
    ],
)
def test_each_a07_fallback_reason_is_served(
    tmp_path, caller_factory, timeout_s, reason
):
    app = _create(tmp_path, caller=caller_factory(), timeout_s=timeout_s)
    with TestClient(app) as client:
        sid = _start(client)
        assert _upload(client, sid).status_code == 202
        result = client.get(f"/v1/sessions/{sid}/agent-message")
        assert result.status_code == 200
        body = result.json()
        assert body["finished"] is False
        assert body["fallback_reason"] == reason
        assert isinstance(body["message"], str)
        assert body["message"]


def test_no_key_uses_deterministic_fallback_instead_of_hiding_route(tmp_path):
    app = _create(tmp_path)
    with TestClient(app) as client:
        sid = _start(client)
        assert _upload(client, sid).status_code == 202
        body = client.get(f"/v1/sessions/{sid}/agent-message").json()
        assert body["finished"] is False
        assert body["fallback_reason"] == "model_error"
        assert "more of the part" in body["message"].lower()


def test_unknown_session_is_contract_404(tmp_path):
    app = _create(tmp_path)
    with TestClient(app) as client:
        result = client.get("/v1/sessions/sess_missing/agent-message")
        assert result.status_code == 404
        assert result.json() == {
            "code": "session_not_found",
            "message": "No session with that id.",
        }


def test_agent_can_finish_and_schedule_reconstruction_after_minimum_frames(tmp_path):
    def caller(_state):
        return _reply(tool="finish_session", message="Coverage is complete. Finishing.")

    app = _create(tmp_path, caller=caller)
    with TestClient(app) as client:
        sid = _start(client)
        assert _upload(client, sid, 0).status_code == 202
        assert _upload(client, sid, 1).status_code == 202
        third = _upload(client, sid, 2)
        assert third.status_code == 202

        agent = client.get(f"/v1/sessions/{sid}/agent-message")
        assert agent.status_code == 200
        assert agent.json()["finished"] is True
        assert agent.json()["fallback_reason"] is None

        record = app.state.store.load(sid)
        assert record["status"] == "finished"
        assert record["reconstruction"]["status"] in {"pending", "complete"}
