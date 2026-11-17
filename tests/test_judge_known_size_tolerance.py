"""Unit tests for R-16's known-size tolerance judgment. All synthetic -- `fixtures/data/` is
gitignored and no test may depend on it (same convention as `test_measure_g04_board_error.py`):
`load_height_sigma_floor_mm` is tested against a `tmp_path` evidence file, never the real one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import judge_known_size_tolerance as m


def test_judge_known_size_comparison_within_tolerance() -> None:
    assert (
        m.judge_known_size_comparison("height", difference_mm=0.5, sigma_floor_mm=0.86)
        == "within_tolerance"
    )


def test_judge_known_size_comparison_at_the_boundary_is_within_tolerance() -> None:
    # abs(difference) == sigma_floor_mm exactly: the boundary itself passes, not fails.
    assert (
        m.judge_known_size_comparison("height", difference_mm=0.86, sigma_floor_mm=0.86)
        == "within_tolerance"
    )


def test_judge_known_size_comparison_out_of_tolerance() -> None:
    assert (
        m.judge_known_size_comparison("height", difference_mm=-1.2, sigma_floor_mm=0.86)
        == "out_of_tolerance"
    )


def test_judge_known_size_comparison_uses_absolute_difference() -> None:
    # A negative difference larger in magnitude than the floor is still out_of_tolerance.
    assert (
        m.judge_known_size_comparison("height", difference_mm=-0.26, sigma_floor_mm=0.2)
        == "out_of_tolerance"
    )


def test_judge_known_size_comparison_with_no_measured_tolerance_is_insufficient_evidence() -> (
    None
):
    # The real case this story exists for: an in-plane comparison, no in-plane floor yet
    # (pending R-37) -- never judged against the height floor, which measures something else.
    assert (
        m.judge_known_size_comparison(
            "in_plane", difference_mm=0.26, sigma_floor_mm=None
        )
        == "insufficient_evidence"
    )


def test_judge_all_separates_height_from_in_plane() -> None:
    comparisons = (
        {
            "specimen_id": "s-04",
            "feature": "B-C leg",
            "feature_kind": "in_plane",
            "difference_mm": 0.26,
            "source": "test",
        },
        {
            "specimen_id": "g02_01",
            "feature": "height",
            "feature_kind": "height",
            "difference_mm": -0.76,
            "source": "test",
        },
    )

    results = m.judge_all(comparisons, height_sigma_floor_mm=0.86)

    verdict_by_specimen = {c["specimen_id"]: v for c, v in results}
    assert verdict_by_specimen["s-04"] == "insufficient_evidence"
    assert verdict_by_specimen["g02_01"] == "within_tolerance"


def test_load_height_sigma_floor_mm_reads_from_the_evidence_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "g04-board-error-evidence.json"
    path.write_text(json.dumps({"summary": {"sigma_floor_mm": 0.8564702691269304}}))

    assert m.load_height_sigma_floor_mm(path) == pytest.approx(0.8564702691269304)


def test_known_size_comparisons_includes_s04_and_is_in_plane() -> None:
    # Confirms the real comparison this story judges (R-04, PR #39) is wired up as in_plane --
    # the whole point of this story's judgment.
    s04 = [c for c in m.KNOWN_SIZE_COMPARISONS if c["specimen_id"] == "s-04"]
    assert len(s04) == 1
    assert s04[0]["feature_kind"] == "in_plane"
    assert s04[0]["difference_mm"] == pytest.approx(60.27 - 60.011)
