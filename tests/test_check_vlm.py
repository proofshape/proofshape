"""Manual check orchestration is tested with synthetic bytes and a fake client only."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

RUN_CHECK = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_vlm.py")
)["run_check"]


def test_manual_check_records_calls_replies_and_account_without_the_key(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    for i in range(7):
        (capture / f"frame_{i:04d}.jpg").write_bytes(b"golden")
    screen, wrong = tmp_path / "screen.jpg", tmp_path / "wrong.jpg"
    screen.write_bytes(b"screen")
    wrong.write_bytes(b"wrong")

    def fake(frame, reference):
        raw = frame.read_bytes()
        return SimpleNamespace(
            text=json.dumps(
                {
                    "same_part": None if reference is None else raw != b"wrong",
                    "hands_occluding": False,
                    "screen_photo": raw == b"screen",
                    "lighting_adequate": True,
                    "hint": "Keep the part visible.",
                }
            ),
            model_version="fake-model",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            prompt_feedback=None,
            usage_metadata=SimpleNamespace(total_token_count=12),
        )

    output = tmp_path / "evidence.json"
    evidence = RUN_CHECK(
        capture,
        screen,
        wrong,
        output,
        api_key="private-key",
        api_tier="paid",
        billing_account="test-account",
        inspector=fake,
    )
    assert [run["api_requests_started"] for run in evidence["runs"]] == [3, 2, 2]
    assert evidence["runs"][1]["checks"][1]["observation"]["screen_photo"]
    assert evidence["runs"][2]["checks"][1]["observation"]["same_part"] is False
    assert "private-key" not in output.read_text()


def test_manual_check_refuses_to_send_without_setup(tmp_path):
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        RUN_CHECK(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path / "out.json",
            api_key="",
            api_tier="paid",
            billing_account="unknown",
        )
