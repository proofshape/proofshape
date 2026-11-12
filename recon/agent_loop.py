"""A-07: the tool-calling agent loop (proposal §6.7), and its fallback to A-08's deterministic
policy. This is the "caller" both `recon.agent_tools` and `recon.agent_policy` already name in
their own docstrings but don't implement themselves.

Scope, per this story's own acceptance criteria: the loop and its fallback, built and tested
against a recorded or mock model response -- no live API call (A-13's job), no contract change
to show any of this on the phone (named as a future need, not built here; A-13/C-22's job).

**Why "errors, times out, or loops" is one internal loop per turn, not one model call per
turn.** A model can only meaningfully "loop" if it can chain several tool calls before giving a
final answer -- the shape every real LLM function-calling API actually has. So `run_agent_turn`
asks the model repeatedly (up to `max_steps`), executing whatever it asks for, until it says it
is done, asks to finish the session, or one of the three proposal-named failures trips:

- **errors** -- `model_caller` raises, or its reply doesn't parse into a `ToolCallDecision`.
- **times out** -- the wall-clock budget (`timeout_s`, a declared default, not a measured
  latency claim -- D-055 set the same precedent for A-01's own timeouts) runs out.
- **loops** -- the model asks for the same tool more than `max_repeated_calls` times in a row.
  Since nothing in the session changes mid-turn (no new frame can arrive without a separate
  `upload_frame` call, outside this function's scope), every tool's result is already fixed for
  the whole turn -- asking for the same one again is never productive, so repeating it is a real
  stall, not legitimate re-checking.

On any of the three, this function calls `agent_policy.next_action` exactly once and executes
*that* action directly -- never retrying the model. If the model's own decision, or the
fallback's action, is to finish the session, this function is what actually calls
`agent_tools.finish_session` -- making "the session always completes" structural, the same
two-step contract `agent_policy.Finish`'s own docstring describes ("the caller now calls
`agent_tools.finish_session`").
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from recon.agent_policy import (
    Action,
    AwaitReconstruction,
    ExplainRejection,
    Finish,
    ReportComplete,
    ReportFailure,
    ShootNextSlot,
    ToolResults,
    next_action,
)
from recon.agent_tools import (
    check_frame,
    coverage_report,
    finish_session,
    score_next_view,
    vlm_inspect,
)

ToolName = Literal[
    "score_next_view", "check_frame", "vlm_inspect", "coverage_report", "finish_session"
]
FallbackReason = Literal["model_error", "timeout", "loop_detected"]

DEFAULT_MAX_STEPS = 8
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_REPEATED_CALLS = 2


class ToolCallDecision(BaseModel):
    """What the model decided this step: call one named tool, or be done (with a message for
    the supplier either way, optional on a tool call, required to be meaningful when done)."""

    model_config = ConfigDict(extra="forbid")

    tool: ToolName | None = None
    message: str | None = None
    done: bool = False

    @model_validator(mode="after")
    def names_a_tool_or_is_done(self) -> ToolCallDecision:
        if self.tool is None and not self.done:
            raise ValueError("a decision must either name a tool or be done")
        if self.tool is not None and self.done:
            raise ValueError("a decision cannot both name a tool and be done")
        return self


def parse_model_decision(response: Any) -> ToolCallDecision:
    """Mirrors `recon.vlm_check.parse_response`: duck-types on the response object (a `.text`
    attribute holding JSON), the same shape a mock or recorded reply in tests provides with a
    plain `SimpleNamespace`, no SDK and no network. Raising here (malformed JSON, a missing
    `.text`, a schema violation) is deliberate -- the caller's `except Exception` around this
    call is what turns a malformed reply into a `model_error` fallback, the same way an
    outright exception from `model_caller` itself does."""
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("empty model response")
    return ToolCallDecision.model_validate(json.loads(text))


@dataclass(frozen=True)
class LoopStep:
    """One step already taken this turn, for the model's own history and the eventual
    transcript (C-22's job to display, not built here)."""

    tool: ToolName | None
    message: str | None


@dataclass(frozen=True)
class AgentLoopState:
    """What a `ModelCaller` sees each step: the session record, every tool's result (already
    gathered once, since nothing changes mid-turn), and the steps already taken this turn."""

    record: dict[str, Any]
    tools: ToolResults
    history: tuple[LoopStep, ...]


ModelCaller = Callable[[AgentLoopState], Any]


@dataclass(frozen=True)
class AgentLoopResult:
    finished: bool
    fallback_reason: FallbackReason | None
    message: str | None
    transcript: tuple[LoopStep, ...]


def _gather_tool_results(record: dict[str, Any], session_dir: Any) -> ToolResults:
    return ToolResults(
        check_frame=check_frame(record),
        vlm_inspect=vlm_inspect(record, session_dir),
        coverage_report=coverage_report(session_dir, record),
        score_next_view=score_next_view(record),
    )


def describe_action_for_supplier(action: Action) -> str:
    """Plain-language narration for the fallback path, where there is no model left to ask to
    do this -- the proposal's own job description for the agent ("interprets why a frame
    failed... explains at the end what was captured and what was not"), done deterministically
    instead. The normal path never calls this: a mock/recorded model supplies its own message,
    which this function would only duplicate or override."""
    if isinstance(action, ShootNextSlot):
        return "Keep going -- more of the part still needs to be photographed."
    if isinstance(action, ExplainRejection):
        return action.message
    if isinstance(action, Finish):
        return "That's enough photos. Finishing up."
    if isinstance(action, AwaitReconstruction):
        return "Still working on the result."
    if isinstance(action, ReportFailure):
        return action.message
    if isinstance(action, ReportComplete):
        if action.under_observed_vertex_count is None:
            return "Capture complete. Everything needed was seen."
        return (
            f"Capture complete. {action.under_observed_vertex_count} vertices were not "
            "fully observed."
        )
    raise TypeError(f"unhandled action: {action!r}")  # exhaustive by construction


def run_agent_turn(
    record: dict[str, Any],
    store: Any,
    session_id: str,
    schedule_reconstruction: Callable[[str], None],
    model_caller: ModelCaller,
    *,
    min_frames: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_repeated_calls: int = DEFAULT_MAX_REPEATED_CALLS,
    now: Callable[[], float] = time.monotonic,
) -> AgentLoopResult:
    """Run the agent for one turn: up to `max_steps` model calls, executing whatever tool each
    one asks for, until it is done, asks to finish the session, or falls back (see module
    docstring for the three fallback triggers). `store`/`session_id`/`schedule_reconstruction`/
    `min_frames` are exactly `agent_tools.finish_session`'s own parameters -- this function is
    the one place that actually calls it, on both the normal and the fallback path."""
    session_dir = store.session_dir(session_id)
    tools = _gather_tool_results(record, session_dir)
    started = now()
    history: list[LoopStep] = []

    def fall_back(reason: FallbackReason) -> AgentLoopResult:
        action = next_action(record, tools)
        message = describe_action_for_supplier(action)
        finished = False
        if isinstance(action, Finish):
            result = finish_session(
                store, session_id, schedule_reconstruction, min_frames=min_frames
            )
            finished = result.ok
            if not result.ok:
                message = result.error_message or message
        return AgentLoopResult(
            finished=finished,
            fallback_reason=reason,
            message=message,
            transcript=tuple(history),
        )

    for _ in range(max_steps):
        if now() - started > timeout_s:
            return fall_back("timeout")

        try:
            raw = model_caller(
                AgentLoopState(record=record, tools=tools, history=tuple(history))
            )
            decision = parse_model_decision(raw)
        except Exception:  # noqa: BLE001 - an injected model_caller can fail any way at all
            return fall_back("model_error")

        if now() - started > timeout_s:
            return fall_back("timeout")

        recent = history[-max_repeated_calls:] if max_repeated_calls > 0 else []
        if (
            decision.tool is not None
            and len(recent) == max_repeated_calls
            and all(step.tool == decision.tool for step in recent)
        ):
            return fall_back("loop_detected")

        history.append(LoopStep(tool=decision.tool, message=decision.message))

        if decision.tool == "finish_session":
            result = finish_session(
                store, session_id, schedule_reconstruction, min_frames=min_frames
            )
            message = decision.message
            if not result.ok:
                message = result.error_message or message
            return AgentLoopResult(
                finished=result.ok,
                fallback_reason=None,
                message=message,
                transcript=tuple(history),
            )

        if decision.done:
            return AgentLoopResult(
                finished=False,
                fallback_reason=None,
                message=decision.message,
                transcript=tuple(history),
            )

        # decision.tool names one of the four read tools; its result is already in `tools`
        # (gathered once, static for the whole turn) -- loop back and let the model see it
        # recorded in history, then decide its next step.

    return fall_back("loop_detected")
