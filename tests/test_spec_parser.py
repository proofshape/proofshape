"""A-03: deterministic fake-reply checks, no credentials or network -- same harness style as
tests/test_vlm_check.py."""

import json
from types import SimpleNamespace

import pytest

from recon.spec_parser import InspectionSpec, SpecFeature, parse_spec_response


def reply(**spec_changes) -> SimpleNamespace:
    spec = {
        "features": [
            {
                "name": "mounting holes",
                "kind": "hole",
                "tolerance_mm": 2.0,
                "critical": True,
            },
            {
                "name": "sealing face",
                "kind": "face",
                "tolerance_mm": 0.1,
                "critical": True,
            },
        ],
        "assumptions": [],
    }
    spec.update(spec_changes)
    return SimpleNamespace(
        text=json.dumps(spec),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        prompt_feedback=None,
    )


# --- the proposal's own example sentence, parsed correctly --------------------------------------


def test_parses_the_proposals_own_example():
    # "the four mounting holes need to line up within about 2 mm, and the sealing face has to
    # be flat" (proposal §7.4) -- this is the plausible structured reply a model would give it.
    result = parse_spec_response(reply())
    assert result.status == "parsed"
    assert result.reason is None
    assert result.spec == InspectionSpec(
        features=[
            SpecFeature(
                name="mounting holes", kind="hole", tolerance_mm=2.0, critical=True
            ),
            SpecFeature(
                name="sealing face", kind="face", tolerance_mm=0.1, critical=True
            ),
        ],
        assumptions=[],
    )


def test_assumptions_pass_through_as_plain_strings():
    result = parse_spec_response(
        reply(assumptions=["Assumed the part is rigid, not a thin sheet."])
    )
    assert result.status == "parsed"
    assert result.spec.assumptions == ["Assumed the part is rigid, not a thin sheet."]


def test_an_overall_feature_with_no_named_part_is_valid():
    # Not every spec names discrete features -- "the whole part needs to be within 1mm" is a
    # single `overall`-kind feature, per the proposal's four-value `kind` enum.
    result = parse_spec_response(
        reply(
            features=[
                {
                    "name": "overall",
                    "kind": "overall",
                    "tolerance_mm": 1.0,
                    "critical": True,
                }
            ]
        )
    )
    assert result.status == "parsed"
    assert result.spec.features[0].kind == "overall"


# --- a structurally invalid reply is rejected, not silently coerced ----------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {
            "features": [
                {"name": "hole A", "kind": "hole", "tolerance_mm": 0, "critical": True}
            ]
        },
        {
            "features": [
                {
                    "name": "hole A",
                    "kind": "hole",
                    "tolerance_mm": -1.0,
                    "critical": True,
                }
            ]
        },
        {
            "features": [
                {
                    "name": "hole A",
                    "kind": "slot",
                    "tolerance_mm": 1.0,
                    "critical": True,
                }
            ]
        },
        {
            "features": [
                {"name": "", "kind": "hole", "tolerance_mm": 1.0, "critical": True}
            ]
        },
        {
            "features": [
                {
                    "name": "hole A",
                    "kind": "hole",
                    "tolerance_mm": "loose",
                    "critical": True,
                }
            ]
        },
        {
            "features": [{"name": "hole A", "kind": "hole", "tolerance_mm": 1.0}]
        },  # missing `critical`
        {
            "features": [
                {
                    "name": "hole A",
                    "kind": "hole",
                    "tolerance_mm": 1.0,
                    "critical": True,
                }
            ],
            "extra_field": "not in the schema",
        },
    ],
)
def test_invalid_spec_is_unparsed_not_coerced(changes):
    result = parse_spec_response(reply(**changes))
    assert result.status == "unparsed"
    assert result.reason == "invalid_reply"
    assert result.spec is None


def test_empty_blocked_incomplete_and_malformed_replies():
    response = reply()
    response.candidates = []
    assert parse_spec_response(response).reason == "empty_reply"

    response = reply()
    response.prompt_feedback = SimpleNamespace(block_reason="SAFETY")
    assert parse_spec_response(response).reason == "blocked_reply"

    response = reply()
    response.candidates[0].finish_reason = "MAX_TOKENS"
    assert parse_spec_response(response).reason == "incomplete_reply"

    response = reply()
    response.text = ""
    assert parse_spec_response(response).reason == "empty_reply"

    response = reply()
    response.text = "not json"
    assert parse_spec_response(response).reason == "invalid_reply"
