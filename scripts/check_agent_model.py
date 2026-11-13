"""A-13: run A-07 twice against the pinned model using synthetic capture state only."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from recon import service
from recon.agent_loop import AgentLoopResult, parse_model_decision, run_agent_turn
from recon.agent_model import DEFAULT_TEMPERATURE, MODEL, GeminiAgentCaller


def _synthetic_session(store: service.SessionStore) -> dict[str, Any]:
    record = store.create("A-13-live-check")
    session_id = record["session_id"]
    frame_bytes = b"\xff\xd8\xffsynthetic-check-frame"
    gyros = (
        {"alpha": 0.0, "beta": 45.0, "gamma": 0.0},
        {"alpha": 45.0, "beta": 45.0, "gamma": 0.0},
        {"alpha": 90.0, "beta": 45.0, "gamma": 0.0},
    )
    for gyro in gyros:
        store.add_frame(session_id, frame_bytes, gyro)
    updated = store.load(session_id)
    if updated is None:
        raise RuntimeError("synthetic session disappeared")
    return updated


def _result_summary(result: AgentLoopResult) -> dict[str, Any]:
    return {
        "finished": result.finished,
        "fallback_reason": result.fallback_reason,
        "message": result.message,
        "steps": [
            {"tool": step.tool, "message": step.message} for step in result.transcript
        ],
    }


def run_live_check(api_key: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    signatures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="proofshape-a13-") as temporary:
        for run_number in (1, 2):
            store = service.SessionStore(Path(temporary) / f"run-{run_number}")
            record = _synthetic_session(store)
            caller = GeminiAgentCaller(api_key)
            model_errors: list[dict[str, str]] = []

            def checked_caller(state, model_caller=caller, errors=model_errors):
                try:
                    response = model_caller(state)
                except Exception as error:
                    errors.append(
                        {
                            "type": type(error).__name__,
                            "detail": str(error).replace(api_key, "[redacted]"),
                        }
                    )
                    raise
                try:
                    parse_model_decision(response)
                except Exception as error:
                    errors.append({"type": type(error).__name__, "detail": str(error)})
                    raise
                return response

            result = run_agent_turn(
                record,
                store,
                record["session_id"],
                lambda _session_id: None,
                checked_caller,
                min_frames=3,
                max_steps=6,
                timeout_s=65.0,
            )
            summary = _result_summary(result)
            model_output = {"steps": summary["steps"]}
            results.append(
                {
                    **summary,
                    "responses": caller.response_metadata,
                    "model_errors": model_errors,
                }
            )
            signatures.append(json.dumps(model_output, sort_keys=True))

    both_runs_reached_model = all(result["responses"] for result in results)
    return {
        "requested_model": MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "input": "same three synthetic gyro-only accepted frames in separate sessions",
        "runs": results,
        "both_runs_reached_model": both_runs_reached_model,
        "answers_differ": both_runs_reached_model and len(set(signatures)) > 1,
    }


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "Set GEMINI_API_KEY in the process environment; its value is never printed."
        )
        return 2
    try:
        evidence = run_live_check(api_key)
    except Exception as error:  # noqa: BLE001 - report the live-check failure without the key
        message = str(error).replace(api_key, "[redacted]")
        print(
            json.dumps(
                {
                    "requested_model": MODEL,
                    "error_type": type(error).__name__,
                    "error": message,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["answers_differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
