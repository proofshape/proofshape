"""Guard the F-06 fixture ignore rules.

Captured images must remain outside git, while reference CAD under
fixtures/reference_cad/ must be addable normally.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(path: str) -> bool:
    # F-08: forces case-sensitive matching regardless of the local git config. Without this,
    # core.ignorecase=true (common on a Mac or Windows checkout, the default here) makes every
    # pattern match case-insensitively, so test_uppercase_photo_extensions_outside_fixtures_are_
    # ignored below would pass even with the uppercase patterns missing from .gitignore -- it
    # would only ever catch a real regression by accident of running on Linux CI, not locally.
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.ignorecase=false",
            "check-ignore",
            "--no-index",
            "-q",
            path,
        ],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_reference_cad_stl_is_not_ignored() -> None:
    assert not _is_ignored("fixtures/reference_cad/example.stl")


def test_capture_jpeg_remains_ignored() -> None:
    assert _is_ignored("fixtures/data/golden_capture/s-01/example.jpeg")


def test_uppercase_photo_extensions_outside_fixtures_are_ignored() -> None:
    # F-08 (#197): the patterns were lowercase-only, so these weren't ignored on a
    # case-sensitive filesystem (Linux, the Lightning Studio) even though they were on a Mac's
    # default case-insensitive one.
    for path in (
        "capture/IMG_0001.JPG",
        "capture/IMG_0002.JPEG",
        "capture/IMG_0003.PNG",
        "capture/IMG_0004.HEIC",
    ):
        assert _is_ignored(path), path
