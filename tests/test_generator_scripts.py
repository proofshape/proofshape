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

import os
import subprocess
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
