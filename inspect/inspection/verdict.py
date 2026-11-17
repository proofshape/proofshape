"""I-07: proposal 7.2's four-row verdict table, for `declared_at: post_reconstruction`
(Unverifiable's intake-time path is I-09's job, not this one's).

| Verdict | Condition |
|---|---|
| Pass | Every critical region Observed with sigma <= tolerance/k, and no Observed deviation above tolerance |
| Fail | An Observed deviation exceeds tolerance by more than k*sigma at that location |
| Rescan | A critical region is Unobserved or high-sigma; registration is ambiguous; or the only deviation lies in an Unobserved region |

The hard rule governs every row: geometry that was not observed never fails a part and never
contributes to a Pass -- it can only trigger a Rescan (proposal 7.2; AGENTS.md).

**What a "region" is in v1, resolved (caught on review, @TabeenRaoof, #302):** no code anywhere
binds a named spec feature to actual CAD geometry -- `inspect/inspection/cad.py` only loads
coordinates/units, and I-06's `compute_deviation` is one global per-vertex array over the whole
posed mesh. v1 therefore does what proposal 7.4 literally says ("stops at the structured spec
and applies tolerances globally, naming each feature in the report"): every named feature is
judged against the *same* global measurement, using only that feature's own `tolerance_mm` to
decide its own Pass/Fail/Rescan. Binding a feature to real geometry is the Tier 3 extension, not
this story's.

**`max(sigma, sigma_floor)`, never raw sigma, for out-of-plane comparisons (issue #157):**
`compute_vertex_sigma` can return sigma close to 0 from two near-identical residuals at n=2
observations -- the "zero spread means infinite precision" trap. Every out-of-plane comparison
floors sigma at `D073_SIGMA_FLOOR_BOARD_MM` first, via `np.fmax` (not `np.maximum` -- `fmax`
treats a NaN sigma, R-10's own "fewer than two observations" marker, as "use the floor", where
`maximum` would propagate the NaN into the comparison and silently break it).

**sigma_floor covers out-of-plane (height) comparisons only (D-073) -- in-plane gets raw sigma,
no floor, and two pools are judged independently, not one global worst vertex:** D-073 states
the restriction explicitly -- "0.86 mm is sigma_floor for out-of-plane (height) features on the
board reference only... I-07's `max(sigma, sigma_floor)` does not apply 0.86 to in-plane
deviation (most Observed vertices are wall vertices, i.e. in-plane) until R-37 lands." It
forbids *flooring* in-plane sigma with this figure -- it does not say in-plane evidence must
always be discarded. Each Observed, measured vertex is classified out-of-plane (top `100 -
TOP_REGION_PERCENTILE` percent of this session's own height-above-board distribution) or
in-plane (everything else), using `recon.cad_size_check.compute_height_above_board` and the same
percentile convention G-07/D-077's `per_vertex_height_reliability` already established for this
exact board/wall distinction -- reused here, not reinvented. The two pools are then evaluated
independently, each against its own worst-deviation vertex (generalising v1's original
single-worst-vertex simplification to two separate worsts, one per pool, since they can no
longer share one floor):

- **Out-of-plane:** unchanged Fail/Pass/high-sigma logic, sigma floored at `D073_SIGMA_FLOOR_BOARD_MM`.
- **In-plane:** no floor exists (pending R-37), so raw sigma is used as given. If the pool's own
  worst-deviation vertex's raw sigma is NaN (R-10's "can't even estimate this" marker -- not
  merely noisy, genuinely unassessed, the literal #157 scenario with no floor available to
  neutralise it), that is reported as insufficient evidence (Rescan, its own reason code) --
  matching R-16's own `insufficient_evidence` resolution to this identical ambiguity, mapped onto
  the nearest safe verdict value (D-003: never invent a number to dress up a judgement the
  evidence can't support). Otherwise, a deviation that clearly exceeds tolerance by more than
  `k * raw_sigma` still Fails -- a genuine wall-side defect is real evidence of a problem
  regardless of which pool's floor convention applies, and this project's whole purpose is
  catching gross defects, not suppressing ones outside the one pool a floor happens to cover.
  A raw in-plane sigma that is merely large (not NaN) doesn't by itself force anything: it can't
  grant a Pass (no floor to trust one with), but it isn't treated as alarming on its own either --
  it simply contributes nothing, leaving the out-of-plane pool's own result (if any) to stand.

**`k`/`sigma_floor` are consumed as given, never re-measured or re-derived here** (G-07's D-077,
G-04's D-073) -- defined independently in this module rather than imported from
`inspection.intake_check` (I-09), matching this project's own existing precedent of each
consumer keeping its own named copy of a team-agreed figure (G-07's own script does the same
rather than importing G-04's).

**`reason_codes` vocabulary, defined here (caught on review, #302: none existed anywhere):**
`f"{feature_name}_not_seen"` (Unobserved), `f"{feature_name}_deviation_out_of_spec"` (Fail --
matches the contract's own committed example naming exactly, from either pool),
`f"{feature_name}_high_sigma"` (out-of-plane, Observed, but too noisy to trust even with the
floor applied), `f"{feature_name}_in_plane_insufficient_evidence"` (the in-plane pool's own
worst-deviation vertex has no usable sigma estimate at all -- D-073), plus the already-defined
`registration.REASON_ORIENTATION_AMBIGUOUS` and `"registration_failed"` (this module's own, for
when registration/ambiguity/deviation couldn't be computed at all -- a best-effort post-
reconstruction step, not a reconstruction failure).

A feature with a missing or null `tolerance_mm` is refused outright (raises
`RefusedToleranceError`), never defaulted or silently skipped -- the same discipline A-03/I-09
apply, and the contract's `VerdictResult` has no "refused" verdict value to represent it, so the
caller (`getVerdict`) turns this into an error response, not a verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from recon.cad_size_check import compute_height_above_board

# G-04's measured height sigma_floor_mm (D-073), board reference only.
D073_SIGMA_FLOOR_BOARD_MM = 0.86

# G-07's consumable_k_estimate (measured 0.688, D-077) -- total error against
# max(sigma, sigma_floor), the exact quantity this module's Fail/Pass rows use.
D077_K = 0.70

# D-073's own out-of-plane/in-plane split, reusing G-07/D-077's percentile convention
# (recon.cad_size_check.per_vertex_height_reliability) rather than reinventing it: the top
# 100 - TOP_REGION_PERCENTILE percent of this session's own Observed-measured height
# distribution is out-of-plane (top face); everything else is in-plane (wall).
TOP_REGION_PERCENTILE = 90.0

LABEL_OBSERVED = "Observed"

REASON_REGISTRATION_FAILED = "registration_failed"


class RefusedToleranceError(ValueError):
    """A spec feature's tolerance_mm is missing or null -- refused, never defaulted."""


def _pool_worst(
    deviation_mm: np.ndarray, sigma_mm: np.ndarray
) -> tuple[float, float, float] | None:
    """For one pool (out-of-plane or in-plane, already sigma-prepared by the caller -- floored
    or raw): the worst (largest |deviation|) vertex's own deviation and sigma, plus the pool's
    own worst (largest) sigma. `None` if the pool is empty."""
    if len(deviation_mm) == 0:
        return None
    worst_index = int(np.argmax(np.abs(deviation_mm)))
    return (
        float(deviation_mm[worst_index]),
        float(sigma_mm[worst_index]),
        float(np.max(sigma_mm)),
    )


def compute_verdict(
    deviation_mm: np.ndarray,
    deviation_measured: np.ndarray,
    labels: np.ndarray,
    sigma_mm: np.ndarray,
    vertex_positions_mm: np.ndarray,
    board_plane: tuple[float, float, float],
    spec: dict[str, Any],
    *,
    orientation_ambiguous: bool = False,
    ambiguity_reason_codes: Sequence[str] = (),
    registration_failed: bool = False,
) -> dict[str, Any]:
    """Compute a `declared_at: post_reconstruction` verdict. `deviation_mm`, `deviation_measured`,
    `labels`, `sigma_mm` and `vertex_positions_mm` are all per-vertex, same order, same length
    (I-06's `_DEVIATION_MM`/`_DEVIATION_MEASURED`, R-09's labels, R-10's sigma_mm, and the served
    mesh's own board-frame-mm vertex positions). `board_plane` is `(slope_x, slope_y,
    z_at_origin_mm)` -- `BoardSurface`'s own plane fields, the same tuple shape
    `scripts/g07_reliability_curve.py`'s `_read_board_plane` builds -- used only to classify each
    vertex out-of-plane vs in-plane (D-073); it is not re-fit here. `registration_failed=True`
    means I-05/I-06 could not be run at all for this session (e.g. a degenerate CAD or no
    Observed faces) -- distinct from `orientation_ambiguous`, which means they ran but the result
    can't be trusted.

    Returns a dict matching `VerdictResult` minus `declared_at` (always
    `"post_reconstruction"` here -- the caller sets it).
    """
    for feature in spec.get("features", []):
        if feature.get("tolerance_mm") is None:
            raise RefusedToleranceError(
                f"feature {feature.get('name', '<unnamed feature>')!r} has a missing or "
                "null tolerance_mm; refused, not defaulted."
            )

    observed_measured = (np.asarray(labels) == LABEL_OBSERVED) & (
        np.asarray(deviation_measured) > 0.5
    )
    has_observed_data = bool(np.any(observed_measured))

    out_of_plane_worst: tuple[float, float, float] | None = None
    in_plane_worst: tuple[float, float, float] | None = None
    if has_observed_data:
        observed_indices = np.flatnonzero(observed_measured)
        observed_deviation = np.asarray(deviation_mm)[observed_indices]
        observed_sigma_raw = np.asarray(sigma_mm, dtype=np.float64)[observed_indices]

        # D-073: classify each Observed, measured vertex out-of-plane (top face) vs in-plane
        # (wall) before deciding anything -- only the former may be floored.
        height_above_board_mm = compute_height_above_board(
            np.asarray(vertex_positions_mm)[observed_indices], board_plane
        )
        top_region_threshold_mm = np.percentile(
            height_above_board_mm, TOP_REGION_PERCENTILE
        )
        out_of_plane_mask = height_above_board_mm >= top_region_threshold_mm
        in_plane_mask = ~out_of_plane_mask

        out_of_plane_worst = _pool_worst(
            observed_deviation[out_of_plane_mask],
            np.fmax(observed_sigma_raw[out_of_plane_mask], D073_SIGMA_FLOOR_BOARD_MM),
        )
        # In-plane: raw sigma, never floored (D-073) -- a NaN here poisons `_pool_worst`'s own
        # np.max on purpose, so "the pool's worst sigma is NaN" below catches it either way.
        in_plane_worst = _pool_worst(
            observed_deviation[in_plane_mask], observed_sigma_raw[in_plane_mask]
        )

    def _overall_worst_deviation_mm() -> float | None:
        """For reporting only (unobserved/ambiguous/registration-failed regions still want some
        `deviation_mm` to show) -- whichever pool's worst reading is larger in magnitude."""
        candidates = [
            w[0] for w in (out_of_plane_worst, in_plane_worst) if w is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=abs)

    rank = {"Pass": 0, "Rescan": 1, "Fail": 2}
    verdict_rank = 0
    regions: list[dict[str, Any]] = []
    reason_codes: list[str] = list(ambiguity_reason_codes)
    if registration_failed:
        reason_codes.append(REASON_REGISTRATION_FAILED)

    for feature in spec.get("features", []):
        name = feature.get("name", "<unnamed feature>")
        tolerance_mm = feature.get("tolerance_mm")

        if not has_observed_data:
            regions.append(
                {
                    "name": name,
                    "provenance": "unobserved",
                    "deviation_mm": None,
                    "contributes_to_verdict": False,
                }
            )
            reason_codes.append(f"{name}_not_seen")
            verdict_rank = max(verdict_rank, rank["Rescan"])
            continue

        if orientation_ambiguous or registration_failed:
            regions.append(
                {
                    "name": name,
                    "provenance": "observed",
                    "deviation_mm": _overall_worst_deviation_mm(),
                    "contributes_to_verdict": False,
                }
            )
            verdict_rank = max(verdict_rank, rank["Rescan"])
            continue

        feature_rank = rank["Pass"]
        feature_reason_codes: list[str] = []
        feature_deviation_mm: float | None = None
        feature_contributes = False

        if out_of_plane_worst is not None:
            oop_deviation_mm, oop_vertex_sigma_mm, oop_pool_sigma_mm = (
                out_of_plane_worst
            )
            if abs(oop_deviation_mm) > tolerance_mm + D077_K * oop_vertex_sigma_mm:
                feature_rank = max(feature_rank, rank["Fail"])
                feature_reason_codes.append(f"{name}_deviation_out_of_spec")
                feature_deviation_mm = oop_deviation_mm
                feature_contributes = True
            elif oop_pool_sigma_mm <= tolerance_mm / D077_K:
                feature_deviation_mm = oop_deviation_mm
                feature_contributes = True
            else:
                feature_rank = max(feature_rank, rank["Rescan"])
                feature_reason_codes.append(f"{name}_high_sigma")

        if in_plane_worst is not None:
            ip_deviation_mm, ip_vertex_sigma_mm, ip_pool_sigma_mm = in_plane_worst
            if np.isnan(ip_pool_sigma_mm):
                # The in-plane pool has no usable sigma estimate at all for its own worst
                # vertex (R-10's NaN marker) -- genuinely insufficient evidence, not merely
                # noisy, and no floor exists to neutralise it the way out-of-plane's does.
                feature_rank = max(feature_rank, rank["Rescan"])
                feature_reason_codes.append(f"{name}_in_plane_insufficient_evidence")
            elif abs(ip_deviation_mm) > tolerance_mm + D077_K * ip_vertex_sigma_mm:
                # A real, well-measured wall-side defect is still evidence of a problem --
                # D-073 withholds the floor, not the ability to Fail on an obvious defect.
                feature_rank = max(feature_rank, rank["Fail"])
                feature_reason_codes.append(f"{name}_deviation_out_of_spec")
                if feature_deviation_mm is None or abs(ip_deviation_mm) > abs(
                    feature_deviation_mm
                ):
                    feature_deviation_mm = ip_deviation_mm
                feature_contributes = True
            # else: in-plane looks fine on raw sigma alone, but raw sigma has no floor behind
            # it -- D-073 gives this no standing to grant a Pass, so it contributes nothing
            # and leaves the out-of-plane pool's own result (if any) to stand as-is.

        if feature_deviation_mm is None:
            feature_deviation_mm = _overall_worst_deviation_mm()

        regions.append(
            {
                "name": name,
                "provenance": "observed",
                "deviation_mm": feature_deviation_mm,
                "contributes_to_verdict": feature_contributes,
            }
        )
        reason_codes.extend(feature_reason_codes)
        verdict_rank = max(verdict_rank, feature_rank)

    verdict = {0: "Pass", 1: "Rescan", 2: "Fail"}[verdict_rank]
    return {
        "verdict": verdict,
        "orientation_ambiguous": orientation_ambiguous,
        "regions": regions,
        "reason_codes": reason_codes,
    }
