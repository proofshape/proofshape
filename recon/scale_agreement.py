"""R-05: check that the three independent scale sources agree (D-007).

The reconstruction is scaled by the printed ChArUco board pitch (the 20 mm
squares R-02 assumes when it solves camera pose). That assumption can be
silently wrong -- a home or office printer can rescale a page invisibly (see
`docs/manual.md` section 3.2) -- so two more independent readings are checked
against it whenever they are available: a caliper measurement of one printed
square, typed in by the supplier, and the CAD registration residual produced
once a part is aligned to its design file (inspect/, not yet built).

All three are expressed as a scale ratio relative to the nominal board pitch,
so they can be compared on the same footing even though they start out as
different kinds of quantities (a length, a length, and a fractional
alignment error). Ratio 1.0 means "matches the nominal board pitch exactly."
The board-pitch source is always the anchor at 1.0: it is the assumption the
whole reconstruction is built on, not an independent measurement of itself.

The caliper reading and the CAD registration residual are each optional --
neither is guaranteed to be present on a given session -- but at least one of
them must be supplied alongside the board pitch, matching the
`ScaleAgreement` schema in `contracts/openapi.yaml` (`minItems: 2` on
`sources_compared`). The caliper reading is untrusted input typed by the
supplier and must never be the only source (D-007); this module does not
special-case that further, since the "at least one of two other sources"
rule already prevents it from being used alone.
"""

from __future__ import annotations

import math
from itertools import combinations

DEFAULT_DISAGREEMENT_THRESHOLD = 0.02
BOARD_PITCH_SOURCE = "board_pitch"
CALIPER_SOURCE = "caliper_reading"
CAD_RESIDUAL_SOURCE = "cad_registration_residual"


def check_scale_agreement(
    board_pitch_mm: float,
    caliper_reading_mm: float | None = None,
    cad_registration_residual: float | None = None,
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> dict[str, object]:
    """Compare the scale sources from D-007 and flag if any two disagree.

    Args:
        board_pitch_mm: The nominal printed square size the reconstruction
            assumes (`recon.board_pose.BOARD_SQUARE_MM` for the D-034 board).
            This is the anchor the other two sources are compared against; it
            always contributes a ratio of exactly 1.0.
        caliper_reading_mm: An independent caliper measurement of one printed
            square, typed in by the supplier. Optional and untrusted (D-007).
        cad_registration_residual: The fractional scale error reported once
            the reconstructed part is aligned to its CAD design (inspect/).
            A residual of 0.0 means the model matched the design exactly;
            0.03 means the model came out 3% larger than the design implies.
        disagreement_threshold: The fractional disagreement that flags the
            session. D-007 states "about 2%"; this is a team rule, not a
            measured value, so it is a parameter rather than a hardcoded
            constant.

    Returns:
        A dict with `sources_compared` (the source names actually used),
        `ratios` (each source's ratio to the nominal board pitch), and
        `flagged` (True if any two of the compared ratios disagree by more
        than `disagreement_threshold`).

    Raises:
        ValueError: if `board_pitch_mm` is not positive, if a supplied
            reading is not finite, or if fewer than two sources end up being
            compared (board pitch alone is never sufficient -- D-007).
    """
    if not (math.isfinite(board_pitch_mm) and board_pitch_mm > 0):
        raise ValueError("board_pitch_mm must be a positive, finite number.")
    if not (math.isfinite(disagreement_threshold) and disagreement_threshold >= 0):
        raise ValueError(
            "disagreement_threshold must be a non-negative, finite number."
        )

    ratios: dict[str, float] = {BOARD_PITCH_SOURCE: 1.0}

    if caliper_reading_mm is not None:
        if not (math.isfinite(caliper_reading_mm) and caliper_reading_mm > 0):
            raise ValueError(
                "caliper_reading_mm must be a positive, finite number when supplied."
            )
        ratios[CALIPER_SOURCE] = caliper_reading_mm / board_pitch_mm

    if cad_registration_residual is not None:
        if not math.isfinite(cad_registration_residual):
            raise ValueError("cad_registration_residual must be finite when supplied.")
        cad_ratio = 1.0 + cad_registration_residual
        if not (cad_ratio > 0):
            raise ValueError(
                "cad_registration_residual implies a non-positive scale ratio "
                "(must be greater than -1.0)."
            )
        ratios[CAD_RESIDUAL_SOURCE] = cad_ratio

    if len(ratios) < 2:
        raise ValueError(
            "At least one of caliper_reading_mm or cad_registration_residual "
            "must be supplied; the board pitch alone is not an independent "
            "check of itself (D-007)."
        )

    flagged = False
    for (_, ratio_a), (_, ratio_b) in combinations(ratios.items(), 2):
        relative_disagreement = abs(ratio_a - ratio_b) / min(ratio_a, ratio_b)
        if relative_disagreement > disagreement_threshold:
            flagged = True
            break

    return {
        "sources_compared": list(ratios.keys()),
        "ratios": ratios,
        "flagged": flagged,
    }
