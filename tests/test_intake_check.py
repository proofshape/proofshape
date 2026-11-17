"""I-09: intake pre-check, board tier only (D-073/D-077)."""

from __future__ import annotations

from inspection.intake_check import (
    D073_SIGMA_FLOOR_BOARD_MM,
    D077_K,
    UNVERIFIABLE_THRESHOLD_MM,
    check_intake_spec,
)


def test_threshold_matches_k_times_sigma_floor() -> None:
    assert UNVERIFIABLE_THRESHOLD_MM == D077_K * D073_SIGMA_FLOOR_BOARD_MM


def test_feature_below_threshold_is_unverifiable() -> None:
    spec = {"features": [{"name": "hole_A", "tolerance_mm": 0.1, "critical": True}]}

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["feature_name"] == "hole_A"
    assert results[0]["status"] == "unverifiable"
    assert results[0]["reason"] is not None


def test_feature_above_threshold_is_ok() -> None:
    spec = {"features": [{"name": "face_B", "tolerance_mm": 5.0, "critical": True}]}

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["reason"] is None


def test_feature_exactly_at_threshold_is_ok() -> None:
    # The comparison is strict (tolerance < threshold), so a tolerance equal to the threshold
    # itself is not Unverifiable -- the boundary belongs to the method's own measured coverage.
    spec = {
        "features": [
            {
                "name": "edge_C",
                "tolerance_mm": UNVERIFIABLE_THRESHOLD_MM,
                "critical": True,
            }
        ]
    }

    results = check_intake_spec(spec)

    assert results[0]["status"] == "ok"


def test_null_tolerance_is_refused_not_defaulted() -> None:
    spec = {
        "features": [{"name": "overall_D", "tolerance_mm": None, "critical": False}]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "refused"
    assert results[0]["tolerance_mm"] is None


def test_missing_tolerance_key_is_also_refused() -> None:
    spec = {"features": [{"name": "edge_E", "critical": True}]}

    results = check_intake_spec(spec)

    assert results[0]["status"] == "refused"


def test_spec_with_no_features_key_produces_no_flags() -> None:
    # D-004: no spec, no verdict machinery -- nothing to check yet.
    assert check_intake_spec({}) == []


def test_spec_with_empty_features_list_produces_no_flags() -> None:
    assert check_intake_spec({"features": []}) == []


def test_multiple_features_each_judged_independently() -> None:
    spec = {
        "features": [
            {"name": "ok_feature", "tolerance_mm": 5.0, "critical": True},
            {"name": "bad_feature", "tolerance_mm": 0.05, "critical": True},
            {"name": "null_feature", "tolerance_mm": None, "critical": False},
        ]
    }

    results = check_intake_spec(spec)

    statuses = {r["feature_name"]: r["status"] for r in results}
    assert statuses == {
        "ok_feature": "ok",
        "bad_feature": "unverifiable",
        "null_feature": "refused",
    }
