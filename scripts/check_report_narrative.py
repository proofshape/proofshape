"""A-09: run the real Gemini call once against a sample VerdictResult, against the groundwork
`generate_llm_narrative` -- D-082 means the actual v1 entry point (`generate_narrative`) never
calls the model at all, so this script checks the path a future story would resume, not what is
served today. Not part of pytest or CI (same as scripts/check_agent_model.py) -- a manual,
operator-run check, since it spends real API quota."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from recon.report_narrative import (
    MODEL,
    GeminiNarrator,
    VerdictInput,
    generate_llm_narrative,
)

SAMPLE_VERDICT = {
    "verdict": "Fail",
    "declared_at": "post_reconstruction",
    "orientation_ambiguous": False,
    "regions": [
        {
            "name": "mounting_hole_A",
            "provenance": "observed",
            "deviation_mm": 1.8,
            "contributes_to_verdict": True,
        },
        {
            "name": "sealing_face",
            "provenance": "unobserved",
            "contributes_to_verdict": False,
        },
    ],
    "reason_codes": ["mounting_hole_A_deviation_out_of_spec"],
}


def run_live_check(api_key: str) -> dict[str, Any]:
    verdict_input = VerdictInput.model_validate(SAMPLE_VERDICT)
    result = generate_llm_narrative(verdict_input, narrator=GeminiNarrator(api_key))
    return {
        "requested_model": MODEL,
        "returned_model_version": result.model_version,
        "source": result.source,
        "reason": result.reason,
        "text": result.text,
    }


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "Set GEMINI_API_KEY in the process environment; its value is never printed."
        )
        return 2
    try:
        evidence = run_live_check(api_key)
    except Exception as error:  # noqa: BLE001 - report the live-check failure without the key
        message = str(error).replace(api_key, "[redacted]")
        print(
            json.dumps(
                {
                    "requested_model": MODEL,
                    "error_type": type(error).__name__,
                    "error": message,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["source"] == "llm" else 1


if __name__ == "__main__":
    raise SystemExit(main())
