"""A-02/D-084: deterministic policy over a synthetic vlm_checks.json -- no credentials, no
network, same harness style as tests/test_vlm_check.py."""

import pytest

from recon.vlm_veto import apply_vlm_veto


def checked(frame_index, **observation_changes):
    observation = {
        "same_part": None if frame_index == 0 else True,
        "hands_occluding": False,
        "screen_photo": False,
        "lighting_adequate": True,
        "hint": "Keep the part visible.",
    }
    observation.update(observation_changes)
    return {
        "frame_index": frame_index,
        "status": "checked",
        "reason": None,
        "model_version": "returned-model-version",
        "token_usage": {},
        "observation": observation,
    }


def unchecked(frame_index, reason):
    return {
        "frame_index": frame_index,
        "status": "unchecked",
        "reason": reason,
        "model_version": None,
        "token_usage": {},
        "observation": None,
    }


def report(*checks):
    return {
        "schema_version": 1,
        "frame_count": len(checks),
        "sampled_indices": [check["frame_index"] for check in checks],
        "checks": list(checks),
    }


def pass_result(**changes):
    result = {
        "verdict": "Pass",
        "orientation_ambiguous": False,
        "regions": [
            {
                "name": "overall_A",
                "provenance": "observed",
                "deviation_mm": 0.1,
                "contributes_to_verdict": True,
            }
        ],
        "reason_codes": [],
    }
    result.update(changes)
    return result


def test_every_sampled_frame_clean_leaves_the_result_untouched():
    vlm_checks_report = report(checked(0), checked(5), checked(10))
    verdict_result = pass_result()
    assert apply_vlm_veto(verdict_result, vlm_checks_report) == verdict_result


def test_screen_photo_on_a_sampled_frame_downgrades_pass_to_rescan():
    vlm_checks_report = report(checked(0), checked(5, screen_photo=True))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert result["verdict"] == "Rescan"
    assert "vlm_screen_photo_frame_5" in result["reason_codes"]


def test_hands_occluding_on_a_sampled_frame_downgrades_pass_to_rescan():
    vlm_checks_report = report(checked(0, hands_occluding=True))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert result["verdict"] == "Rescan"
    assert "vlm_hands_occluding_frame_0" in result["reason_codes"]


def test_inadequate_lighting_on_a_sampled_frame_downgrades_pass_to_rescan():
    vlm_checks_report = report(checked(5, lighting_adequate=False))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert result["verdict"] == "Rescan"
    assert "vlm_lighting_inadequate_frame_5" in result["reason_codes"]


def test_wrong_part_on_a_non_reference_frame_downgrades_pass_to_rescan():
    vlm_checks_report = report(checked(0), checked(5, same_part=False))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert result["verdict"] == "Rescan"
    assert "vlm_wrong_part_frame_5" in result["reason_codes"]


# D-084: unchecked blocks a Pass, the same as a failed semantic check -- an outage or missing
# key is a named trade-off (Rescan during the outage), not leniency invented here.
def test_unchecked_sampled_frame_downgrades_pass_to_rescan():
    vlm_checks_report = report(checked(0), unchecked(5, "missing_api_key"))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert result["verdict"] == "Rescan"
    assert "vlm_unchecked_frame_5_missing_api_key" in result["reason_codes"]


@pytest.mark.parametrize(
    "reason", ["api_timeout", "blocked_reply", "invalid_result_file", "join_timeout"]
)
def test_every_unchecked_reason_is_named_in_its_own_code(reason):
    vlm_checks_report = report(unchecked(0, reason))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert f"vlm_unchecked_frame_0_{reason}" in result["reason_codes"]


def test_veto_never_downgrades_an_already_fail_result():
    vlm_checks_report = report(checked(0, screen_photo=True))
    fail_result = pass_result(
        verdict="Fail", reason_codes=["overall_A_deviation_out_of_spec"]
    )
    result = apply_vlm_veto(fail_result, vlm_checks_report)
    assert result["verdict"] == "Fail"
    assert "overall_A_deviation_out_of_spec" in result["reason_codes"]
    assert "vlm_screen_photo_frame_0" in result["reason_codes"]


def test_veto_never_raises_rescan_to_fail():
    vlm_checks_report = report(checked(0, screen_photo=True))
    rescan_result = pass_result(verdict="Rescan", reason_codes=["overall_A_not_seen"])
    result = apply_vlm_veto(rescan_result, vlm_checks_report)
    assert result["verdict"] == "Rescan"


def test_multiple_simultaneous_problems_on_one_frame_each_get_their_own_code():
    vlm_checks_report = report(checked(0, screen_photo=True, hands_occluding=True))
    result = apply_vlm_veto(pass_result(), vlm_checks_report)
    assert "vlm_screen_photo_frame_0" in result["reason_codes"]
    assert "vlm_hands_occluding_frame_0" in result["reason_codes"]


def test_existing_reason_codes_are_kept_alongside_the_new_ones():
    vlm_checks_report = report(checked(0), checked(5, screen_photo=True))
    rescan_result = pass_result(verdict="Rescan", reason_codes=["overall_A_high_sigma"])
    result = apply_vlm_veto(rescan_result, vlm_checks_report)
    assert "overall_A_high_sigma" in result["reason_codes"]
    assert "vlm_screen_photo_frame_5" in result["reason_codes"]


# Review, @TabeenRaoof, PR #396: a screen photo or wrong-part flag casts doubt on whether the
# geometry a Fail is based on is even this supplier's real part. Keeping Fail (rather than
# overriding it to Rescan specifically for those two flags) is a deliberate choice, not the only
# hard-rule-compliant one -- D-084 records both options and why this one was kept. Covered
# generally by test_veto_never_downgrades_an_already_fail_result above; this test pins the
# specific screen_photo/wrong_part case the reviewer raised.
def test_screen_photo_under_a_fail_result_still_leaves_it_fail():
    vlm_checks_report = report(checked(0, screen_photo=True))
    fail_result = pass_result(
        verdict="Fail", reason_codes=["overall_A_deviation_out_of_spec"]
    )
    result = apply_vlm_veto(fail_result, vlm_checks_report)
    assert result["verdict"] == "Fail"


def test_missing_vlm_checks_file_vetoes_instead_of_crashing():
    result = apply_vlm_veto(pass_result(), None)
    assert result["verdict"] == "Rescan"
    assert "vlm_checks_missing" in result["reason_codes"]


def test_missing_vlm_checks_file_never_downgrades_an_already_fail_result():
    fail_result = pass_result(
        verdict="Fail", reason_codes=["overall_A_deviation_out_of_spec"]
    )
    result = apply_vlm_veto(fail_result, None)
    assert result["verdict"] == "Fail"
