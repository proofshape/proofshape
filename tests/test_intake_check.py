"""I-09: intake pre-check, board tier only (D-073/D-077), kind-aware (found on review,
@mbj1994, PR #372)."""

from __future__ import annotations

from inspection.intake_check import (
    D073_SIGMA_FLOOR_BOARD_MM,
    D077_K,
    UNVERIFIABLE_THRESHOLD_MM,
    check_intake_spec,
)


def test_threshold_matches_k_times_sigma_floor() -> None:
    assert UNVERIFIABLE_THRESHOLD_MM == D077_K * D073_SIGMA_FLOOR_BOARD_MM


def test_height_like_feature_below_threshold_is_unverifiable() -> None:
    spec = {
        "features": [
            {"name": "top_face", "kind": "face", "tolerance_mm": 0.1, "critical": True}
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["feature_name"] == "top_face"
    assert results[0]["status"] == "unverifiable"
    assert results[0]["reason"] is not None


def test_height_like_feature_above_threshold_is_ok() -> None:
    spec = {
        "features": [
            {
                "name": "overall_height",
                "kind": "overall",
                "tolerance_mm": 5.0,
                "critical": True,
            }
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["reason"] is None


def test_height_like_feature_exactly_at_threshold_is_ok() -> None:
    # The comparison is strict (tolerance < threshold), so a tolerance equal to the threshold
    # itself is not Unverifiable -- the boundary belongs to the method's own measured coverage.
    spec = {
        "features": [
            {
                "name": "face_C",
                "kind": "face",
                "tolerance_mm": UNVERIFIABLE_THRESHOLD_MM,
                "critical": True,
            }
        ]
    }

    results = check_intake_spec(spec)

    assert results[0]["status"] == "ok"


def test_in_plane_edge_is_refused_not_checked_against_height_floor() -> None:
    # Found on review (@mbj1994, PR #372): the height floor (0.602mm effective threshold) does
    # not apply to an in-plane edge/hole tolerance. No verified-on-main in-plane floor exists
    # yet (R-37/D-078 pending review, PR #364), so this is refused, not silently marked "ok" --
    # a 1.0mm edge tolerance would be wrongly "ok" against the height floor (1.0 >= 0.602), but
    # is well within D-078's own proposed in-plane range (0.602-3.695mm) that must not be
    # under-flagged.
    spec = {
        "features": [
            {"name": "edge_1", "kind": "edge", "tolerance_mm": 1.0, "critical": True}
        ]
    }

    results = check_intake_spec(spec)

    assert len(results) == 1
    assert results[0]["status"] == "refused"
    assert "in-plane" in results[0]["reason"]


def test_in_plane_hole_is_refused_regardless_of_tolerance_size() -> None:
    # Even a generous tolerance is refused, not passed -- there is no verified floor to check it
    # against at all, not just a risk of a too-small one.
    spec = {
        "features": [
            {"name": "hole_1", "kind": "hole", "tolerance_mm": 50.0, "critical": True}
        ]
    }

    results = check_intake_spec(spec)

    assert results[0]["status"] == "refused"


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


def test_spec_with_empty_features_list_produces_no_flags() -> None:
    assert check_intake_spec({"features": []}) == []


def test_multiple_features_each_judged_independently() -> None:
    spec = {
        "features": [
            {
                "name": "ok_feature",
                "kind": "face",
                "tolerance_mm": 5.0,
                "critical": True,
            },
            {
                "name": "bad_feature",
                "kind": "overall",
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
                "name": "edge_feature",
                "kind": "edge",
                "tolerance_mm": 1.0,
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
        "edge_feature": "refused",
    }
