from __future__ import annotations

import pytest

from recon.scale_agreement import (
    BOARD_PITCH_SOURCE,
    CAD_RESIDUAL_SOURCE,
    CALIPER_SOURCE,
    check_scale_agreement,
)


def test_agreeing_caliper_reading_is_not_flagged() -> None:
    result = check_scale_agreement(board_pitch_mm=20.0, caliper_reading_mm=20.05)

    assert result["sources_compared"] == [BOARD_PITCH_SOURCE, CALIPER_SOURCE]
    assert result["ratios"][BOARD_PITCH_SOURCE] == pytest.approx(1.0)
    assert result["ratios"][CALIPER_SOURCE] == pytest.approx(20.05 / 20.0)
    assert result["flagged"] is False


def test_disagreeing_caliper_reading_is_flagged() -> None:
    # A caliper reading of 27 mm against a nominal 20 mm square is the exact
    # "Scale to Fit" misprint the manual warns about (docs/manual.md 3.2) --
    # about 35% off, far past the ~2% D-007 threshold. Injecting this
    # disagreement is what R-05's acceptance criteria asks the tests to do.
    result = check_scale_agreement(board_pitch_mm=20.0, caliper_reading_mm=27.0)

    assert result["flagged"] is True


def test_agreeing_cad_registration_residual_is_not_flagged() -> None:
    result = check_scale_agreement(board_pitch_mm=20.0, cad_registration_residual=0.005)

    assert result["sources_compared"] == [BOARD_PITCH_SOURCE, CAD_RESIDUAL_SOURCE]
    assert result["ratios"][CAD_RESIDUAL_SOURCE] == pytest.approx(1.005)
    assert result["flagged"] is False


def test_disagreeing_cad_registration_residual_is_flagged() -> None:
    result = check_scale_agreement(board_pitch_mm=20.0, cad_registration_residual=0.05)

    assert result["flagged"] is True


def test_all_three_sources_present_and_agreeing() -> None:
    result = check_scale_agreement(
        board_pitch_mm=20.0,
        caliper_reading_mm=20.1,
        cad_registration_residual=0.005,
    )

    assert result["sources_compared"] == [
        BOARD_PITCH_SOURCE,
        CALIPER_SOURCE,
        CAD_RESIDUAL_SOURCE,
    ]
    assert result["flagged"] is False


def test_two_agreeing_sources_do_not_mask_a_third_that_disagrees() -> None:
    # Board pitch and CAD residual agree closely; the caliper reading is far
    # off. Flagging must not require *all* pairs to disagree -- any one
    # disagreeing pair is enough (D-007: "any two disagreeing").
    result = check_scale_agreement(
        board_pitch_mm=20.0,
        caliper_reading_mm=25.0,
        cad_registration_residual=0.0,
    )

    assert result["flagged"] is True


def test_caliper_reading_alone_is_a_sufficient_second_source() -> None:
    result = check_scale_agreement(board_pitch_mm=20.0, caliper_reading_mm=20.1)

    assert len(result["sources_compared"]) == 2


def test_board_pitch_alone_is_rejected() -> None:
    # D-007: board pitch alone can never be sufficient -- it is the
    # assumption under test, not an independent check of itself.
    with pytest.raises(ValueError, match="independent check"):
        check_scale_agreement(board_pitch_mm=20.0)


def test_non_positive_board_pitch_is_rejected() -> None:
    with pytest.raises(ValueError, match="board_pitch_mm"):
        check_scale_agreement(board_pitch_mm=0.0, caliper_reading_mm=20.0)


def test_non_finite_caliper_reading_is_rejected() -> None:
    with pytest.raises(ValueError, match="caliper_reading_mm"):
        check_scale_agreement(board_pitch_mm=20.0, caliper_reading_mm=float("nan"))


def test_non_positive_caliper_reading_is_rejected() -> None:
    with pytest.raises(ValueError, match="caliper_reading_mm"):
        check_scale_agreement(board_pitch_mm=20.0, caliper_reading_mm=-1.0)


def test_cad_registration_residual_at_or_below_negative_one_is_rejected() -> None:
    # A residual of -1.0 or less implies the model has zero or negative
    # scale relative to the design, which is not a physically meaningful
    # alignment result.
    with pytest.raises(ValueError, match="non-positive scale ratio"):
        check_scale_agreement(board_pitch_mm=20.0, cad_registration_residual=-1.0)


def test_disagreement_threshold_is_configurable() -> None:
    # A looser threshold accepts a difference the default ~2% would flag.
    result = check_scale_agreement(
        board_pitch_mm=20.0,
        caliper_reading_mm=20.5,
        disagreement_threshold=0.05,
    )

    assert result["flagged"] is False
