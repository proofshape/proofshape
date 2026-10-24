"""A-08: the deterministic capture policy -- proposal §6.7's fallback, "built first, at M1, so
the agent is layered onto something that already works." Not generative, on purpose: it is the
floor the real agent (A-07) stands on and the baseline it is judged against ("does the agent
actually beat the deterministic policy?"). Same input always gives the same action, and nothing
here ever calls a language model.

`next_action` is a pure function of state: it does not call A-06's five tools itself. The caller
(eventually A-07's loop, or a test) calls them and hands in the results, the same no-I/O-in-the-
decision-logic style `recon.agent_tools` already uses.

Two decisions this story's own acceptance criteria required before any code (D-054 follow-on,
confirmed with the team):

- `ExplainRejection` stays in the closed action set, driven by `check_frame`'s result, but that
  tool is always `unavailable` today (D-054: the client-side gate's rejection never reaches the
  server). So this branch is real and tested, but practically dead in production until a future
  story adds real gate telemetry -- not silently dropped, not silently pretended to work. Its
  `accepted`/`code`/`message` shape is the same one `check_frame` actually produced before D-054
  reverted it for a different reason (the wrong *source*, not the wrong *shape*), not invented
  from nothing.
- No recapture round. The frozen v1 contract is single-shot per session; an actual second
  capture cycle would need a new D-031 contract change, which nothing today needs badly enough
  to spend speculatively (same call A-06 already made). So once a reconstruction exists, the
  policy only ever *reports* what `score_next_view` still finds under-observed.

`vlm_inspect` gets no decision branch at all, not even an inert one: A-01 doesn't exist yet, so
there is no real result shape to branch on, and guessing one would be designing A-01's output
format here instead of there.
"""

from __future__ import annotations

from dataclasses import dataclass

from recon.agent_tools import ToolResult


@dataclass(frozen=True)
class ToolResults:
    """One snapshot of calling A-06's tools, handed to `next_action` by its caller."""

    check_frame: ToolResult
    vlm_inspect: ToolResult
    coverage_report: ToolResult
    score_next_view: ToolResult


@dataclass(frozen=True)
class ShootNextSlot:
    """Still capturing: aim at this orbit slot next."""

    orbit_index: int


@dataclass(frozen=True)
class ExplainRejection:
    """Still capturing: the gate rejected the last frame. Inert today -- see module docstring."""

    code: str
    message: str


@dataclass(frozen=True)
class Finish:
    """The orbit is covered. The caller now calls `agent_tools.finish_session`."""


@dataclass(frozen=True)
class ReportFailure:
    """Reconstruction already failed. The deterministic policy can't fix a pipeline failure --
    it only reports it, from the stored failure record."""

    code: str
    message: str


@dataclass(frozen=True)
class ReportComplete:
    """Reconstruction already succeeded. `under_observed_vertex_count` is `None` once nothing is
    left under-observed, otherwise the count `score_next_view`'s best candidate still sees --
    reported, never re-captured (the no-recapture-round decision above)."""

    under_observed_vertex_count: int | None


Action = ShootNextSlot | ExplainRejection | Finish | ReportFailure | ReportComplete


def next_action(record: dict, tools: ToolResults) -> Action:
    """Deterministic: the same `(record, tools)` always returns the same `Action`, and nothing
    here ever raises -- an unavailable or missing tool result just falls through to the next
    check, with `Finish` as the final, safe default rather than a stall."""
    reconstruction = record.get("reconstruction") or {}
    status = reconstruction.get("status")

    if status == "failed":
        failure = reconstruction.get("failure") or {}
        return ReportFailure(
            code=failure.get("code", "unknown"), message=failure.get("message", "")
        )

    if status == "complete":
        under_observed = None
        if tools.score_next_view.available:
            count = tools.score_next_view.data["best"]["under_observed_visible_count"]
            under_observed = count if count > 0 else None
        return ReportComplete(under_observed_vertex_count=under_observed)

    # Still capturing.
    if tools.check_frame.available and tools.check_frame.data.get("accepted") is False:
        return ExplainRejection(
            code=tools.check_frame.data["code"],
            message=tools.check_frame.data["message"],
        )

    if tools.coverage_report.available:
        next_index = tools.coverage_report.data["orbit_coverage"]["next_target_index"]
        if next_index is not None:
            return ShootNextSlot(orbit_index=next_index)

    return Finish()
