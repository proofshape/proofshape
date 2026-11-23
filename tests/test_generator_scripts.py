"""Smoke tests for scripts/build_*.js (#201).

These are Node, not Python, so each is exercised as a subprocess -- the same
pattern tests/test_download_golden_capture.py uses for its bash script: run
the real tool end to end and check its real output, rather than mocking.

python-ci.yml's pytest job now installs scripts/node_modules before running
tests, so these run for real in CI, not just locally. A contributor running
pytest locally without having run `npm install` in scripts/ yet gets a skip,
not a failure -- this is the only place in the Python suite that depends on
Node tooling at all, so the skip only has to protect that one, local-only gap.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

pytestmark = pytest.mark.skipif(
    not (SCRIPTS_DIR / "node_modules").is_dir(),
    reason="scripts/node_modules not installed; run `npm install` in scripts/",
)

# PPTX_APPLY_THEME is a personal, opt-in theming tool (#201) -- whether it's set depends on
# the machine, not on anything these tests are checking, so it's removed from every subprocess
# here. Without this, these tests would pass or fail differently depending on whose machine
# runs them (review on #229, @TabeenRaoof): confirmed on a machine where the variable points at
# a real but broken theme setup, build_midterm_deck.js's exit code and output both changed.
_ENV_WITHOUT_THEME = {k: v for k, v in os.environ.items() if k != "PPTX_APPLY_THEME"}


@pytest.mark.parametrize(
    "script_name,out_name",
    [
        ("build_review_deck.js", "review.pptx"),
        ("build_midterm_deck.js", "midterm.pptx"),
        ("build_midterm_report.js", "report.docx"),
        ("build_expo_poster.js", "poster.pptx"),
    ],
)
def test_generator_script_builds_a_nonempty_file(tmp_path, script_name, out_name):
    out_path = tmp_path / out_name

    result = subprocess.run(
        ["node", script_name, str(out_path)],
        cwd=SCRIPTS_DIR,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_ENV_WITHOUT_THEME,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_build_midterm_deck_skips_the_theme_pass_without_crashing(tmp_path):
    """#201: PPTX_APPLY_THEME is a personal, opt-in theming tool, unset for most team
    members and for CI. Confirms the fix's no-op fallback actually runs end to end -- not
    just that the deck gets written -- by checking for its one-line skip message."""
    out_path = tmp_path / "midterm.pptx"

    result = subprocess.run(
        ["node", "build_midterm_deck.js", str(out_path)],
        cwd=SCRIPTS_DIR,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_ENV_WITHOUT_THEME,
    )

    assert result.returncode == 0, result.stderr
    assert "PPTX_APPLY_THEME not set" in result.stdout
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def _run_poster(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "build_expo_poster.js", *args],
        cwd=SCRIPTS_DIR,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_ENV_WITHOUT_THEME,
    )


def _slide_text(pptx_path: Path) -> str:
    """Every text run on every slide, joined."""
    with zipfile.ZipFile(pptx_path) as archive:
        xml = "".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
    return "\n".join(re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.DOTALL))


# Made-up IDs: the real ones live only in an untracked file (P-40).
FAKE_IDS = {
    "Mamadou Jallow": "900001",
    "Yashi Dalwala": "900002",
    "Tabeen Raoof": "900003",
}


def test_expo_poster_has_student_ids_only_when_given_an_ids_file(tmp_path):
    """P-40: the repository is public (D-013), so the committed poster carries names only.
    Student IDs come from an untracked file, for the print copy."""
    ids_path = tmp_path / "student-ids.json"
    ids_path.write_text(json.dumps(FAKE_IDS))
    plain = tmp_path / "plain.pptx"
    printed = tmp_path / "print.pptx"

    for args in ([str(plain)], [str(printed), "--ids", str(ids_path)]):
        result = _run_poster(*args)
        assert result.returncode == 0, result.stderr

    plain_text = _slide_text(plain)
    printed_text = _slide_text(printed)
    for name, student_id in FAKE_IDS.items():
        assert name in plain_text
        assert student_id not in plain_text
        assert f"{name} · {student_id}" in printed_text
    # nothing in the committed build reads like a student ID
    assert not re.search(r"(?<!\d)\d{6}(?!\d)", plain_text)


def test_expo_poster_refuses_an_ids_file_that_misses_a_member(tmp_path):
    ids_path = tmp_path / "student-ids.json"
    ids_path.write_text(json.dumps({"Mamadou Jallow": "900001"}))
    out_path = tmp_path / "poster.pptx"

    result = _run_poster(str(out_path), "--ids", str(ids_path))

    assert result.returncode != 0
    assert "Yashi Dalwala" in result.stderr
    assert not out_path.exists()
