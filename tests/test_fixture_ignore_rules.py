"""Guard the F-06 fixture ignore rules.

Captured images must remain outside git, while reference CAD under
fixtures/reference_cad/ must be addable normally.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_reference_cad_stl_is_not_ignored() -> None:
    assert not _is_ignored("fixtures/reference_cad/example.stl")


def test_capture_jpeg_remains_ignored() -> None:
    assert _is_ignored("fixtures/data/golden_capture/s-01/example.jpeg")
