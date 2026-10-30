"""A-06: the capture agent's five tools (proposal §6.7), as typed, deterministic, no-language-
model functions wrapping code that already exists. The agent (A-07) chooses which tool to call
and when; nothing in this module does. Same input always gives the same output -- that is what
lets A-08's deterministic fallback call these directly, with no model involved at all.

Two of the five tools have a phone-side source (`capture/src/frame_gate.ts`,
`capture/src/orbit.ts`), and how each reaches the server is D-054:

- `coverage_report` recomputes orbit coverage server-side from every accepted frame's gyro
  reading, already persisted at `<session_dir>/frames/frame_NNNN.gyro.json`
  (`SessionStore.add_frame`) -- no contract change.
- `check_frame` is always `unavailable`. A frame the gate rejects is never uploaded, so the
  server never learns the gate's actual decision, and nothing here answers a different
  question under this tool's name instead. Reporting the real gate result would need a new
  contract field and a D-031 sprint-boundary request -- deliberately not added here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recon.mesh_cleanup import read_glb
from recon.provenance import colors_to_labels

# `recon.next_view` is imported lazily, inside score_next_view() below, not at module level --
# it pulls in Open3D (D-047) at import time, which every caller of this module would otherwise
# pay for even when only check_frame/vlm_inspect/finish_session are needed.


@dataclass(frozen=True)
class ToolResult:
    """Every tool returns this: either real data, or an explicit, named reason it couldn't."""

    available: bool
    reason: str | None = None
    data: dict[str, Any] | None = None


# --- orbit coverage, ported from capture/src/orbit.ts (D-054) --------------------------------

DEFAULT_MID_ELEVATION_DEG = 45.0
DEFAULT_HIGH_ELEVATION_DEG = 70.0
DEFAULT_COVERAGE_TOLERANCE_DEG = 15.0
_AZIMUTH_EXEMPT_ELEVATION_DEG = 85.0


@dataclass(frozen=True)
class OrbitSlot:
    azimuth_deg: float
    elevation_deg: float


def _build_default_orbit() -> tuple[OrbitSlot, ...]:
    slots = [
        OrbitSlot(azimuth_deg=i * 45.0, elevation_deg=DEFAULT_MID_ELEVATION_DEG)
        for i in range(8)
    ]
    slots += [
        OrbitSlot(azimuth_deg=i * 90.0, elevation_deg=DEFAULT_HIGH_ELEVATION_DEG)
        for i in range(4)
    ]
    slots.append(OrbitSlot(azimuth_deg=0.0, elevation_deg=90.0))
    return tuple(slots)


DEFAULT_ORBIT: tuple[OrbitSlot, ...] = _build_default_orbit()


def _angular_distance_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def _relative_azimuth_deg(raw_alpha_deg: float, anchor_deg: float) -> float:
    return ((raw_alpha_deg - anchor_deg) % 360.0 + 360.0) % 360.0


def _elevation_from_beta(beta_deg: float) -> float:
    return 90.0 - beta_deg


def _anchor_alpha_deg(accepted_alphas: list[float]) -> float | None:
    return accepted_alphas[0] if accepted_alphas else None


def _nearest_eligible_slot_index(
    alpha_deg: float,
    beta_deg: float,
    anchor_deg: float,
    orbit: tuple[OrbitSlot, ...],
    tolerance_deg: float,
) -> int | None:
    elevation_deg = _elevation_from_beta(beta_deg)
    azimuth_deg = _relative_azimuth_deg(alpha_deg, anchor_deg)

    best_index: int | None = None
    best_elevation_dist = float("inf")
    for index, slot in enumerate(orbit):
        elevation_dist = abs(elevation_deg - slot.elevation_deg)
        if elevation_dist > tolerance_deg:
            continue
        azimuth_eligible = (
            slot.elevation_deg >= _AZIMUTH_EXEMPT_ELEVATION_DEG
            or _angular_distance_deg(azimuth_deg, slot.azimuth_deg) <= tolerance_deg
        )
        if not azimuth_eligible:
            continue
        if elevation_dist < best_elevation_dist:
            best_elevation_dist = elevation_dist
            best_index = index
    return best_index


def compute_coverage(
    accepted_gyros: list[dict[str, float]],
    orbit: tuple[OrbitSlot, ...] = DEFAULT_ORBIT,
    tolerance_deg: float = DEFAULT_COVERAGE_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Direct port of `orbit.ts`'s `computeCoverage`, cross-checked against
    `capture/src/orbit.test.ts`'s own cases in `tests/test_agent_tools.py`."""
    alphas = [float(g["alpha"]) for g in accepted_gyros]
    anchor_deg = _anchor_alpha_deg(alphas)
    covered = [False] * len(orbit)
    if anchor_deg is not None:
        for gyro in accepted_gyros:
            index = _nearest_eligible_slot_index(
                float(gyro["alpha"]),
                float(gyro["beta"]),
                anchor_deg,
                orbit,
                tolerance_deg,
            )
            if index is not None:
                covered[index] = True
    next_target_index = covered.index(False) if False in covered else None
    return {"covered": covered, "next_target_index": next_target_index}


def _read_accepted_gyros(session_dir: Path) -> list[dict[str, float]]:
    frames_dir = session_dir / "frames"
    gyro_paths = sorted(frames_dir.glob("frame_*.gyro.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in gyro_paths]


def coverage_report(
    session_dir: Path,
    record: dict[str, Any],
    orbit: tuple[OrbitSlot, ...] = DEFAULT_ORBIT,
    tolerance_deg: float = DEFAULT_COVERAGE_TOLERANCE_DEG,
) -> ToolResult:
    """Orbit coverage alone before a reconstruction exists; combined with
    `score_next_view`'s under-observed count once one does (D-054)."""
    coverage = compute_coverage(_read_accepted_gyros(session_dir), orbit, tolerance_deg)
    data: dict[str, Any] = {"orbit_coverage": coverage}

    next_view = score_next_view(record)
    if next_view.available:
        data["under_observed_vertex_count"] = next_view.data["best"][
            "under_observed_visible_count"
        ]
    return ToolResult(available=True, data=data)


# --- score_next_view ---------------------------------------------------------------------------


def score_next_view(record: dict[str, Any]) -> ToolResult:
    """Wraps `recon.next_view.score_next_views` exactly as it exists -- no reimplementation.
    Only available once a reconstruction has actually completed (batch-only, D-002)."""
    reconstruction = record.get("reconstruction") or {}
    if reconstruction.get("status") != "complete":
        return ToolResult(available=False, reason="no_reconstruction")

    from recon.next_view import score_next_views  # local import: avoids a hard Open3D

    # import for every caller of this module, matching D-047's own lazy-import precedent.

    glb_path = Path(reconstruction["glb_path"])
    vertices_mm, faces, colors = read_glb(glb_path)
    labels = colors_to_labels(colors)
    ranked = score_next_views(vertices_mm, faces, labels)
    return ToolResult(available=True, data={"ranked": ranked, "best": ranked[0]})


# --- check_frame ---------------------------------------------------------------------------


def check_frame(record: dict[str, Any]) -> ToolResult:
    """Always `unavailable` (D-054, revised after review): the acceptance criteria name this
    tool "over the gate result" -- `capture/src/frame_gate.ts`'s blur/exposure/duplicate
    decision -- but a frame the gate rejects is never uploaded, so the server never learns the
    gate's actual answer, and nothing here invents one. Reporting it would need a new contract
    field (a D-031 request), deliberately not added in this story. Returning `unavailable` is
    honest about that gap, the same way `vlm_inspect` is honest about A-01 not existing yet --
    rather than answering a different, already-knowable question (`upload_frame`'s own
    accept/reject outcome) under this tool's name, which would look complete without being
    complete."""
    del record
    return ToolResult(
        available=False, reason="frame-gate result is not reported to the server"
    )


# --- vlm_inspect -------------------------------------------------------------------------------


def vlm_inspect(record: dict[str, Any], session_dir: Path | None = None) -> ToolResult:
    """Read A-01's separate evidence, including during capture; never trust session.json.

    The path is explicit because the data root is app-owned and cannot be inferred from
    a record. Old callers remain unavailable until they supply that path (A-07).
    """
    from recon.vlm_check import FrameCheck

    del record
    if session_dir is None:
        return ToolResult(
            available=False, reason="session directory is required for VLM evidence"
        )
    paths = sorted((Path(session_dir) / "vlm").glob("frame_*.json"))
    if not paths:
        return ToolResult(available=False, reason="no VLM result yet")
    try:
        result = FrameCheck.model_validate_json(paths[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ToolResult(available=False, reason="invalid VLM result file")
    if result.status == "unchecked":
        return ToolResult(
            available=False,
            reason=f"unchecked: {result.reason}",
            data=result.model_dump(),
        )
    return ToolResult(available=True, data=result.model_dump())


# --- finish_session ------------------------------------------------------------------------


@dataclass(frozen=True)
class FinishResult:
    ok: bool
    record: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


def finish_session(
    store: Any,
    session_id: str,
    schedule_reconstruction: Callable[[str], None],
    *,
    min_frames: int,
) -> FinishResult:
    """The same validation `recon/service.py`'s `finish_session` route already does, extracted
    so the route and this tool call one shared implementation instead of two. `store` is a
    `recon.service.SessionStore`; `schedule_reconstruction` is how the caller actually runs
    `run_session` (the route passes `background_tasks.add_task`, a test passes a list append)."""
    from recon.service import SessionAlreadyFinishedError

    record = store.load(session_id)
    if record is None:
        return FinishResult(
            ok=False,
            error_code="session_not_found",
            error_message="No session with that id.",
        )
    if record["status"] != "capturing":
        return FinishResult(
            ok=False,
            error_code="session_already_finished",
            error_message="The session is already finished.",
        )
    if int(record["frame_count"]) < min_frames:
        return FinishResult(
            ok=False,
            error_code="too_few_frames",
            error_message=(
                f"At least {min_frames} accepted frames are required before finishing."
            ),
        )
    try:
        record = store.mark_finished(session_id)
    except SessionAlreadyFinishedError:
        return FinishResult(
            ok=False,
            error_code="session_already_finished",
            error_message="The session is already finished.",
        )
    schedule_reconstruction(session_id)
    return FinishResult(ok=True, record=record)
