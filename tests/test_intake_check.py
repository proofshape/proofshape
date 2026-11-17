"""I-09: intake pre-check, board tier only (D-073/D-077), kind-aware (found on review,
@mbj1994, PR #372) and total against malformed opaque specs (found on a second review,
@TabeenRaoof, PR #372)."""

from __future__ import annotations

import pytest
from inspection.intake_check import (
    D073_SIGMA_FLOOR_BOARD_MM,
    D077_K,
    D078_SIGMA_FLOOR_IN_PLANE_MM,
    UNVERIFIABLE_THRESHOLD_HEIGHT_MM,
    UNVERIFIABLE_THRESHOLD_IN_PLANE_MM,
    check_intake_spec,
)


def test_threshold_matches_k_times_sigma_floor() -> None:
    assert UNVERIFIABLE_THRESHOLD_HEIGHT_MM == D077_K * D073_SIGMA_FLOOR_BOARD_MM
    assert UNVERIFIABLE_THRESHOLD_IN_PLANE_MM == D077_K * D078_SIGMA_FLOOR_IN_PLANE_MM


@pytest.mark.parametrize("kind", ["face", "overall"])
@pytest.mark.parametrize(
    "tolerance_mm", [0.01, UNVERIFIABLE_THRESHOLD_HEIGHT_MM, 100.0]
)
def test_axis_ambiguous_kind_is_refused_regardless_of_tolerance(
    kind: str, tolerance_mm: float
) -> None:
    # Found on review (@TabeenRaoof, PR #372): a face or overall-size check doesn't say which
    # axis it means, so neither the height nor the in-plane floor is assumed -- refused at every
    # tolerance, not just the ones a height-floor comparison would have flagged.
    spec = {
        "features": [
            {"name": "f", "kind": kind, "tolerance_mm": tolerance_mm, "critical": True}
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "refused"
    assert results[0]["tolerance_mm"] == tolerance_mm
    assert "does not state which dimension" in results[0]["reason"]


def test_in_plane_edge_below_its_measured_floor_is_unverifiable() -> None:
    # Found on review (@mbj1994, PR #372): a 1.0mm edge was wrongly ok against the height floor.
    # D-078 now provides the measured in-plane floor (5.278mm), so this must use k*5.278=3.6946.
    spec = {
        "features": [
            {"name": "edge_1", "kind": "edge", "tolerance_mm": 1.0, "critical": True}
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "unverifiable"
    assert "3.6946 mm" in results[0]["reason"]


def test_in_plane_edge_at_and_above_its_measured_floor_is_ok() -> None:
    for tolerance in (UNVERIFIABLE_THRESHOLD_IN_PLANE_MM, 5.0):
        spec = {
            "features": [{"name": "edge_1", "kind": "edge", "tolerance_mm": tolerance}]
        }

        results = check_intake_spec(spec)

        assert results[0]["status"] == "ok", repr(tolerance)


def test_in_plane_hole_uses_the_measured_in_plane_floor() -> None:
    spec = {
        "features": [
            {"name": "hole_1", "kind": "hole", "tolerance_mm": 1.0, "critical": True}
        ]
    }

    results = check_intake_spec(spec)

    assert results[0]["status"] == "unverifiable"
    assert "3.6946 mm" in results[0]["reason"]


def test_unrecognised_kind_is_refused_not_assumed() -> None:
    spec = {
        "features": [
            {
                "name": "mystery",
                "kind": "surface_finish",
                "tolerance_mm": 5.0,
                "critical": True,
            }
        ]
    }

    results = check_intake_spec(spec)

    assert results[0]["status"] == "refused"
    assert "unrecognised" in results[0]["reason"]


def test_missing_kind_is_refused_not_assumed() -> None:
    spec = {"features": [{"name": "no_kind", "tolerance_mm": 5.0, "critical": True}]}

    results = check_intake_spec(spec)

    assert results[0]["status"] == "refused"


def test_non_string_kind_is_refused_not_assumed() -> None:
    spec = {"features": [{"name": "bad_kind", "kind": [], "tolerance_mm": 5.0}]}

    results = check_intake_spec(spec)

    assert results[0]["status"] == "refused"
    assert "kind is missing or not a string" in results[0]["reason"]


def test_null_tolerance_is_refused_before_kind_is_even_checked() -> None:
    spec = {
        "features": [
            {
                "name": "overall_D",
                "kind": "overall",
                "tolerance_mm": None,
                "critical": False,
            }
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "refused"
    assert results[0]["tolerance_mm"] is None
    assert "tolerance_mm is missing or null" in results[0]["reason"]


def test_missing_tolerance_key_is_also_refused() -> None:
    spec = {"features": [{"name": "edge_E", "kind": "edge", "critical": True}]}

    results = check_intake_spec(spec)

    assert results[0]["status"] == "refused"


def test_non_numeric_non_finite_and_non_positive_tolerances_are_refused() -> None:
    for tolerance in (
        "1.0",
        True,
        float("nan"),
        float("inf"),
        -float("inf"),
        0.0,
        -1.0,
    ):
        spec = {
            "features": [
                {
                    "name": "height",
                    "kind": "face",
                    "tolerance_mm": tolerance,
                    "critical": True,
                }
            ]
        }

        results = check_intake_spec(spec)

        assert results[0]["status"] == "refused", repr(tolerance)
        assert "finite positive number" in results[0]["reason"]


def test_spec_with_no_features_key_produces_no_flags() -> None:
    # D-004: no spec, no verdict machinery -- nothing to check yet.
    assert check_intake_spec({}) == []


def test_spec_with_null_features_produces_no_flags() -> None:
    # Found on review (@TabeenRaoof, PR #372): an explicit null is the same as a missing key,
    # not a malformed value -- `main` already returns 201 for this spec and must keep doing so.
    assert check_intake_spec({"features": None}) == []


def test_spec_with_empty_features_list_produces_no_flags() -> None:
    assert check_intake_spec({"features": []}) == []


@pytest.mark.parametrize("features", ["abc", {"a": 1}, 5, True, 1.5])
def test_non_list_features_value_is_refused_not_raised(features: object) -> None:
    # Found on review (@TabeenRaoof, PR #372): each of these previously raised inside
    # check_intake_spec, which recon/service.py does not catch, turning one malformed order
    # into a 500 on every future read of it.
    results = check_intake_spec({"features": features})

    assert len(results) == 1
    assert results[0]["status"] == "refused"
    assert results[0]["tolerance_mm"] is None
    assert "spec['features'] must be a list" in results[0]["reason"]


@pytest.mark.parametrize("item", [5, "abc", [1, 2], None, True])
def test_non_dict_feature_entry_is_refused_not_raised(item: object) -> None:
    # Found on review (@TabeenRaoof, PR #372): a list entry that isn't an object previously
    # raised from `feature.get(...)`.
    results = check_intake_spec({"features": [item]})

    assert len(results) == 1
    assert results[0]["status"] == "refused"
    assert results[0]["tolerance_mm"] is None
    assert "feature entry is not an object" in results[0]["reason"]


def test_malformed_feature_entries_do_not_stop_the_well_formed_ones() -> None:
    spec = {
        "features": [
            5,
            {"name": "edge_1", "kind": "edge", "tolerance_mm": 1.0},
            "abc",
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 3
    assert results[0]["status"] == "refused"
    assert results[1]["feature_name"] == "edge_1"
    assert results[1]["status"] == "unverifiable"
    assert results[2]["status"] == "refused"


@pytest.mark.parametrize("name", [7, "", "   ", None, [], {}])
def test_non_string_or_empty_name_gets_the_placeholder_not_pydantic_validation_error(
    name: object,
) -> None:
    # Found on review (@TabeenRaoof, PR #372): {"name": 7, ...} previously reached
    # `IntakeCheckResult(feature_name=7, ...)`, which raises because `feature_name` is `str`.
    spec = {"features": [{"name": name, "kind": "edge", "tolerance_mm": 1.0}]}

    results = check_intake_spec(spec)

    assert results[0]["feature_name"] == "<unnamed feature>"
    assert results[0]["status"] == "unverifiable"


def test_multiple_features_each_judged_independently() -> None:
    spec = {
        "features": [
            {
                "name": "ok_feature",
                "kind": "edge",
                "tolerance_mm": 5.0,
                "critical": True,
            },
            {
                "name": "bad_feature",
                "kind": "hole",
                "tolerance_mm": 0.05,
                "critical": True,
            },
            {
                "name": "null_feature",
                "kind": "face",
                "tolerance_mm": None,
                "critical": False,
            },
            {
                "name": "ambiguous_feature",
                "kind": "overall",
                "tolerance_mm": 50.0,
                "critical": True,
            },
        ]
    }

    results = check_intake_spec(spec)

    statuses = {r["feature_name"]: r["status"] for r in results}
    assert statuses == {
        "ok_feature": "ok",
        "bad_feature": "unverifiable",
        "null_feature": "refused",
        "ambiguous_feature": "refused",
    }
