"""A-02: gates the verdict on A-01's already-recorded `vlm_checks.json`, which A-01 itself
builds but deliberately never consults (`recon/vlm_check.py`'s own docstring: "never a verdict
or an upload gate"). `recon/README.md` and this story's own file both refuse to let the
`unchecked` policy be invented here without a recorded decision -- **D-084: `unchecked` blocks a
Pass**, the same way a failed semantic check does. Proposal §6.5's literal wording is "no session
reaches Pass until every sampled frame has passed" -- an unchecked frame has not passed, so an
outage or a missing key downgrades affected sessions to Rescan too, a named trade-off against a
live demo's availability, not an overlooked one (D-084).

A veto only ever raises the verdict toward Rescan, reusing `inspect.inspection.verdict.compute_
verdict`'s own `{"Pass": 0, "Rescan": 1, "Fail": 2}` rank scheme so a veto can never lower an
already-`Fail` result, and never invents a `Fail` on its own: AGENTS.md's hard rule is that a
model's guess must never approve or condemn a component, so a VLM semantic flag can only ever ask
for another look.

**A `Fail` stands even under a `screen_photo`/wrong-part flag -- a real design choice, not the
only hard-rule-compliant one, recorded rather than left implicit** (review, @TabeenRaoof, PR
#396): the flag casts some doubt on whether the measured geometry is even this supplier's real
part, and *overriding* `Fail` to `Rescan` specifically for those two flags would also be
hard-rule-compliant (it approves nothing either). Keeping `Fail` was chosen because this module's
one rule -- a veto only ever raises rank, never lowers it -- stays uniform across every flag
rather than special-casing two of them, and because a geometric `Fail` already requires an
Observed, measured deviation past tolerance (`inspection.verdict`'s own rule): the VLM's doubt is
about the photographed session, not a claim that the measurement itself is fabricated. D-084
records this as an open option, not a closed one, if the team weighs it differently.

The veto covers exactly A-01's sampled frames (frame 0, every fifth index, and the final frame --
`recon.vlm_check.sampled_indices`), not every accepted frame -- R-36/#354 found a screen photo on
an unsampled frame, and `recon/README.md` is explicit that this stays a known limitation of
sampled, not all-frame, supervision pending a measured change to that sampling rule.

**What this module deliberately does not do, left for a follow-up:** whether a vetoed sampled
frame is also kept out of the reconstruction itself (today, it is not -- `run_session` schedules
VLM checks and starts reconstruction without waiting, per `recon/service.py`). #354's own
acceptance criteria ask for two real device sessions to be re-run on current `main` first
(`sess_fb1b028da4cd`, `sess_1a215dae88d7`) to see whether R-38's outlier rejection (D-075, which
landed after those sessions' numbers were recorded) already drops a screen-photo frame's effect
on the *reconstruction* on its own. `@TabeenRaoof`'s review (PR #396) confirmed, from the
sessions' own already-recorded `vlm_checks.json` (on their Mac, outside this repo), that this
veto behaves exactly as designed against the real data: `sess_fb1b028da4cd`'s sampled frame 5
(`screen_photo: true`) correctly vetoes; `sess_1a215dae88d7`'s screen photo on unsampled frame 6
is, as documented above, correctly missed. **That confirms this module's own behaviour, not the
separate, still-open question**: whether R-38 already keeps a bad frame's *geometry* out of the
fused mesh is a reconstruction-level question this module cannot answer and does not attempt to
-- it needs an actual re-run of reconstruction on current `main` against both sessions' raw
photos, which remains a follow-up (see `stories/A/A-02.md`), not silently skipped.
"""

from __future__ import annotations

from typing import Any

_RANK = {"Pass": 0, "Rescan": 1, "Fail": 2}


def _unchecked_reason_codes(check: dict[str, Any]) -> list[str]:
    return [f"vlm_unchecked_frame_{check['frame_index']}_{check['reason']}"]


def _checked_reason_codes(check: dict[str, Any]) -> list[str]:
    observation = check["observation"]
    index = check["frame_index"]
    codes = []
    if observation["screen_photo"]:
        codes.append(f"vlm_screen_photo_frame_{index}")
    if observation["hands_occluding"]:
        codes.append(f"vlm_hands_occluding_frame_{index}")
    if observation["lighting_adequate"] is False:
        codes.append(f"vlm_lighting_inadequate_frame_{index}")
    if observation["same_part"] is False:
        codes.append(f"vlm_wrong_part_frame_{index}")
    return codes


def _raise_to_rescan(
    verdict_result: dict[str, Any], veto_reason_codes: list[str]
) -> dict[str, Any]:
    combined_rank = max(_RANK[verdict_result["verdict"]], _RANK["Rescan"])
    combined_verdict = {rank: name for name, rank in _RANK.items()}[combined_rank]
    return {
        **verdict_result,
        "verdict": combined_verdict,
        "reason_codes": [*verdict_result.get("reason_codes", []), *veto_reason_codes],
    }


def apply_vlm_veto(
    verdict_result: dict[str, Any], vlm_checks_report: dict[str, Any] | None
) -> dict[str, Any]:
    """Combines `verdict_result` (`inspection.verdict.compute_verdict`'s own return shape --
    `verdict`/`orientation_ambiguous`/`regions`/`reason_codes`) with A-01's `vlm_checks_report`
    (the parsed `vlm_checks.json`, A-01's own `{"schema_version", "frame_count",
    "sampled_indices", "checks"}` shape). Returns a dict in the same shape, `reason_codes`
    extended (never replaced) with every VLM-vetoed sampled frame's own code, `verdict` raised to
    at least `"Rescan"` if any fired -- never lowered, and never raised past the input's own
    `verdict`.

    `vlm_checks_report=None` means the file itself was missing (review, @TabeenRaoof, PR #396):
    this shouldn't happen -- `VlmSupervisor.join` writes it unconditionally -- but a missing
    file is itself unchecked evidence, not evidence of a clean session, so it vetoes with its
    own named `vlm_checks_missing` code rather than crashing the endpoint on a state this policy
    can still answer safely."""
    if vlm_checks_report is None:
        return _raise_to_rescan(verdict_result, ["vlm_checks_missing"])

    veto_reason_codes: list[str] = []
    for check in vlm_checks_report["checks"]:
        if check["status"] == "unchecked":
            veto_reason_codes.extend(_unchecked_reason_codes(check))
        else:
            veto_reason_codes.extend(_checked_reason_codes(check))

    if not veto_reason_codes:
        return verdict_result

    return _raise_to_rescan(verdict_result, veto_reason_codes)
