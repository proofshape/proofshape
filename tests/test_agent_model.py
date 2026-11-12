"""A-13: test the pinned Gemini caller without network access or credentials."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from recon.agent_loop import AgentLoopState
from recon.agent_model import (
    MODEL,
    GeminiAgentCaller,
    GeminiAgentOutput,
    _allowed_actions,
    _output_schema,
    _request_payload,
)
from recon.agent_policy import ToolResults
from recon.agent_tools import ToolResult


def _state() -> AgentLoopState:
    return AgentLoopState(
        record={
            "session_id": "must-not-be-sent",
            "status": "capturing",
            "reconstruction": {
                "status": "not_started",
                "glb_path": "/private/model.glb",
            },
        },
        tools=ToolResults(
            check_frame=ToolResult(False, reason="not_requested"),
            vlm_inspect=ToolResult(False, reason="not_requested"),
            coverage_report=ToolResult(False, reason="not_requested"),
            score_next_view=ToolResult(False, reason="not_requested"),
        ),
        history=(),
    )


def _coverage_obtained(next_target_index: int | None) -> AgentLoopState:
    state = _state()
    tools = replace(
        state.tools,
        coverage_report=ToolResult(
            True,
            data={
                "orbit_coverage": {
                    "covered": [False],
                    "next_target_index": next_target_index,
                }
            },
        ),
    )
    return replace(state, tools=tools)


def _response(
    text: str = '{"action":"done","message":"Keep the whole part in view."}',
    *,
    finish_reason: str = "STOP",
    block_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
        model_version=MODEL,
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=5,
            total_token_count=17,
        ),
    )


def test_request_payload_contains_only_agent_state_and_tool_results():
    payload = json.loads(_request_payload(_state()))

    assert payload["session"] == {
        "status": "capturing",
        "reconstruction_status": "not_started",
    }
    assert payload["tools"]["coverage_report"]["reason"] == "not_requested"
    assert payload["tools"]["score_next_view"]["reason"] == "not_requested"
    assert payload["history"] == []
    serialized = json.dumps(payload)
    assert "must-not-be-sent" not in serialized
    assert "/private/model.glb" not in serialized


def test_gemini_agent_caller_uses_pinned_structured_output(monkeypatch):
    from google import genai

    response = _response()
    calls: list[dict] = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return response

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "not-a-real-secret"
            assert kwargs["vertexai"] is False
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(genai, "Client", FakeClient)
    caller = GeminiAgentCaller("not-a-real-secret")

    result = caller(_coverage_obtained(next_target_index=3))
    assert json.loads(result.text) == {
        "tool": None,
        "message": "Keep the whole part in view.",
        "done": True,
    }
    assert calls[0]["model"] == "gemini-3.5-flash-lite"
    config = calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["properties"]["action"]["enum"] == ["done"]
    assert "coverage" in calls[0]["contents"]
    assert caller.response_metadata == [
        {
            "requested_model": MODEL,
            "returned_model": MODEL,
            "token_usage": {
                "prompt_token_count": 12,
                "candidates_token_count": 5,
                "total_token_count": 17,
            },
        }
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(finish_reason="MAX_TOKENS"), "incomplete"),
        (_response(block_reason="SAFETY"), "blocked"),
        (_response(text=""), "empty"),
        (SimpleNamespace(candidates=[], text="{}"), "no agent candidate"),
    ],
)
def test_gemini_agent_caller_rejects_nonterminal_responses(
    monkeypatch, response, message
):
    from google import genai

    class FakeModels:
        def generate_content(self, **_kwargs):
            return response

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(genai, "Client", FakeClient)
    caller = GeminiAgentCaller("not-a-real-secret")

    with pytest.raises(ValueError, match=message):
        caller(_state())


def test_gemini_agent_caller_requires_an_explicit_key():
    with pytest.raises(ValueError, match="API key is required"):
        GeminiAgentCaller("")


def test_provider_output_maps_one_action_to_one_loop_decision(monkeypatch):
    response = _response(
        '{"action":"coverage_report","message":"Checking the view coverage."}'
    )

    class FakeModels:
        def generate_content(self, **_kwargs):
            return response

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeClient)
    result = GeminiAgentCaller("not-a-real-secret")(_state())

    assert json.loads(result.text) == {
        "tool": "coverage_report",
        "message": "Checking the view coverage.",
        "done": False,
    }


def test_action_schema_tracks_unresolved_tool_state():
    initial = _state()
    assert _allowed_actions(initial) == ("coverage_report", "vlm_inspect")
    assert _output_schema(initial)["properties"]["action"]["enum"] == [
        "coverage_report",
        "vlm_inspect",
    ]

    vlm_unavailable = replace(
        initial,
        tools=replace(
            initial.tools,
            vlm_inspect=ToolResult(False, reason="no VLM result yet"),
        ),
    )
    assert _allowed_actions(vlm_unavailable) == ("coverage_report",)

    coverage_complete = _coverage_obtained(next_target_index=None)
    assert _allowed_actions(coverage_complete) == ("finish_session",)


def test_provider_response_outside_dynamic_action_enum_is_rejected(monkeypatch):
    from google import genai

    class FakeModels:
        def generate_content(self, **_kwargs):
            return _response('{"action":"check_frame","message":"Checking the gate."}')

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(ValueError, match="not available this step"):
        GeminiAgentCaller("not-a-real-secret")(_state())


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "done", "message": "Two lines\nare invalid."},
        {"action": "done", "message": ""},
        {"action": "done", "message": "Done.", "done": True},
    ],
)
def test_provider_schema_rejects_invalid_actions_and_messages(payload):
    with pytest.raises(ValueError):
        GeminiAgentOutput.model_validate(payload)
