"""A-09: deterministic fake-reply checks, no credentials or network -- same harness style as
tests/test_vlm_check.py and tests/test_spec_parser.py. Fixtures are the real contract examples
(contracts/examples/, capture/stub/fixtures/) wherever one exists, not hand-invented data."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recon.report_narrative import (
    MODEL,
    VerdictInput,
    generate_narrative,
    template_narrative,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PASS_FIXTURE = json.loads(
    (REPO_ROOT / "contracts" / "examples" / "getVerdict.response-200.json").read_text()
)
RESCAN_FIXTURE = json.loads(
    (
        REPO_ROOT
        / "contracts"
        / "examples"
        / "getVerdict.response-200-rescan-unobserved.json"
    ).read_text()
)
FAIL_FIXTURE = json.loads(
    (REPO_ROOT / "capture" / "stub" / "fixtures" / "verdict-fail.json").read_text()
)
# No real intake-time Unverifiable example exists yet -- I-10 (the story that defines the
# authoritative intake reason-code vocabulary) is still Claimed, not Done. Illustrative only.
UNVERIFIABLE_FIXTURE = {
    "verdict": "Unverifiable",
    "declared_at": "intake",
    "orientation_ambiguous": False,
    "regions": [],
    "reason_codes": ["sealing_face_below_reference_floor"],
}

ALL_FIXTURES = [PASS_FIXTURE, FAIL_FIXTURE, RESCAN_FIXTURE, UNVERIFIABLE_FIXTURE]
FIXTURE_IDS = ["pass", "fail", "rescan", "unverifiable"]


def reply(text="A short narrative.", finish_reason="STOP", block_reason=None):
    return SimpleNamespace(
        text=text,
        model_version="returned-model-version",
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        prompt_feedback=SimpleNamespace(block_reason=block_reason)
        if block_reason
        else None,
    )


def narrator(response):
    return lambda verdict_input: response


def raising_narrator(exc):
    def _narrator(verdict_input):
        raise exc

    return _narrator


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
def test_llm_reply_is_used_when_valid(fixture):
    verdict_input = VerdictInput.model_validate(fixture)
    # The template's own output never invents a figure, so reusing it as the "model's reply"
    # proves the guard accepts a legitimate reply instead of only ever falling back.
    safe_text = template_narrative(verdict_input)
    result = generate_narrative(verdict_input, narrator=narrator(reply(safe_text)))
    assert result.source == "llm"
    assert result.text == safe_text
    assert result.model_version == "returned-model-version"
    assert result.reason is None


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
def test_no_narrator_and_no_api_key_falls_back_to_template(fixture):
    verdict_input = VerdictInput.model_validate(fixture)
    result = generate_narrative(verdict_input)
    assert result.source == "template"
    assert result.reason == "no_api_key"
    assert result.text == template_narrative(verdict_input)


@pytest.mark.parametrize(
    "exc,expected_reason",
    [
        (TimeoutError("boom"), "model_error"),
        (ValueError("boom"), "model_error"),
    ],
)
def test_narrator_exception_falls_back_to_template(exc, expected_reason):
    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(verdict_input, narrator=raising_narrator(exc))
    assert result.source == "template"
    assert result.reason == expected_reason
    assert result.text == template_narrative(verdict_input)


def test_httpx_timeout_falls_back_with_its_own_reason():
    from httpx import TimeoutException

    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(
        verdict_input, narrator=raising_narrator(TimeoutException("timed out"))
    )
    assert result.source == "template"
    assert result.reason == "timeout"


@pytest.mark.parametrize(
    "response,expected_reason",
    [
        (reply(block_reason="SAFETY"), "blocked_reply"),
        (
            SimpleNamespace(
                text="x", model_version=None, candidates=[], prompt_feedback=None
            ),
            "empty_reply",
        ),
        (reply(finish_reason="MAX_TOKENS"), "incomplete_reply"),
        (reply(text=""), "empty_reply"),
        (reply(text="   "), "empty_reply"),
    ],
)
def test_invalid_envelope_falls_back_to_template(response, expected_reason):
    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(verdict_input, narrator=narrator(response))
    assert result.source == "template"
    assert result.reason == expected_reason
    assert result.text == template_narrative(verdict_input)


@pytest.mark.parametrize(
    "text",
    [
        "The system achieves 98% accuracy.",
        "The measurement sigma was low.",
        "Confidence is high, σ is small.",
    ],
)
def test_invented_figure_or_banned_word_falls_back_to_template(text):
    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(verdict_input, narrator=narrator(reply(text)))
    assert result.source == "template"
    assert result.reason == "unmeasured_claim"


@pytest.mark.parametrize(
    "text",
    [
        "This is within tolerance.",
        "Confidence is high due to good precision.",
    ],
)
def test_plain_language_without_a_figure_is_accepted(text):
    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(verdict_input, narrator=narrator(reply(text)))
    assert result.source == "llm"
    assert result.text == text


def test_invented_mm_figure_falls_back_to_template():
    # PASS_FIXTURE's own regions deviate by 0.14 mm and -0.06 mm -- 0.30 mm appears nowhere.
    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(
        verdict_input,
        narrator=narrator(reply("The sealing face deviated by 0.30 mm.")),
    )
    assert result.source == "template"
    assert result.reason == "unmeasured_claim"


def test_real_measured_mm_figure_is_accepted():
    verdict_input = VerdictInput.model_validate(PASS_FIXTURE)
    result = generate_narrative(
        verdict_input,
        narrator=narrator(
            reply("The mounting hole deviated by 0.14 mm, which passed.")
        ),
    )
    assert result.source == "llm"


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
def test_template_mentions_only_given_facts(fixture):
    verdict_input = VerdictInput.model_validate(fixture)
    text = template_narrative(verdict_input)
    assert verdict_input.verdict in text
    lowered = text.lower()
    assert "sigma" not in lowered and "σ" not in lowered
    assert "%" not in text
    for region in verdict_input.regions:
        if region.contributes_to_verdict:
            assert region.name in text
    for code in verdict_input.reason_codes:
        assert code in text


def test_model_is_a_pinned_snapshot_not_latest():
    assert MODEL == "gemini-3.5-flash-lite"
    assert "latest" not in MODEL
