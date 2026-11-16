"""Unit tests for G-04's board-error aggregation script. All synthetic -- no GPU, no fixtures
(`fixtures/data/` is gitignored and no test may depend on it; `ensure_reconstruction` /
`measure_specimen` / `main` drive the real R-01/R-02/R-04/cad_size_check chain and are exercised
by the real run itself, the same D-052 manual-check exception `smoke_r14.py`/`smoke_r13.py` use).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import measure_g04_board_error as m


def _write_caliper_g06(path: Path, across_mm: float, down_mm: float) -> None:
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "object": "board",
                        "kind": "board_span",
                        "square_count": 8,
                        "mean_mm": across_mm,
                    },
                    {
                        "object": "board",
                        "kind": "board_span",
                        "square_count": 6,
                        "mean_mm": down_mm,
                    },
                    # A decoy single-square reading (square_count=1) shouldn't be picked up.
                    {
                        "object": "board",
                        "kind": "board_span",
                        "square_count": 1,
                        "mean_mm": 20.03,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_board_pitch_mm_reads_across_and_down(tmp_path: Path) -> None:
    path = tmp_path / "caliper_g-06.json"
    _write_caliper_g06(path, across_mm=160.16, down_mm=120.55)

    across, down = m.load_board_pitch_mm(path)

    assert across == pytest.approx(160.16)
    assert down == pytest.approx(120.55)


def test_load_board_pitch_mm_rejects_missing_span(tmp_path: Path) -> None:
    path = tmp_path / "caliper_g-06.json"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "object": "board",
                        "kind": "board_span",
                        "square_count": 8,
                        "mean_mm": 160.16,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="square_count 6 and 8"):
        m.load_board_pitch_mm(path)


def test_reference_cad_path_finds_the_one_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m, "REFERENCE_CAD_DIR", tmp_path)
    (tmp_path / "g02_01_square_20x20x15mm.stl").write_text("solid x\nendsolid x\n")
    (tmp_path / "g02_02_rect_50x20x25mm.stl").write_text("solid x\nendsolid x\n")

    found = m._reference_cad_path("g02_01")

    assert found.name == "g02_01_square_20x20x15mm.stl"


def test_reference_cad_path_rejects_no_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m, "REFERENCE_CAD_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="g02_09"):
        m._reference_cad_path("g02_09")


def _result(
    edges: list[dict],
    height_error_mm: float | None,
    board_alignment_ok: bool = True,
    specimen_id: str = "g02_00",
) -> dict:
    return {
        "specimen_id": specimen_id,
        "board_alignment_ok": board_alignment_ok,
        "caliper": {
            "edges": [{"caliper_error_mm": e} for e in edges],
            "height": (
                {"caliper_error_mm": height_error_mm}
                if height_error_mm is not None
                else None
            ),
        },
    }


def test_aggregate_computes_mean_and_p95_sigma_floor() -> None:
    results = [
        _result([0.1, -0.2, 0.3], height_error_mm=0.05),
        _result(
            [0.5], height_error_mm=None
        ),  # g02_01-style: no height-independent edges here
    ]

    summary = m.aggregate(results)

    assert summary["feature_count"] == 5
    assert summary["excluded_specimen_ids"] == []
    errors = [0.1, -0.2, 0.3, 0.05, 0.5]
    assert summary["mean_error_mm"] == pytest.approx(sum(errors) / len(errors))
    # p95 of |errors| sorted [0.05, 0.1, 0.2, 0.3, 0.5] -> round(0.95*4)=4th index -> 0.5
    assert summary["sigma_floor_mm"] == pytest.approx(0.5)


def test_aggregate_excludes_specimens_with_failed_board_alignment() -> None:
    results = [
        _result(
            [0.1, -0.2],
            height_error_mm=0.05,
            board_alignment_ok=True,
            specimen_id="g02_01",
        ),
        _result(
            [-80.0, 90.0],
            height_error_mm=-40.0,
            board_alignment_ok=False,
            specimen_id="g02_02",
        ),  # g02_02-style: huge errors from a failed alignment, not real part error
    ]

    summary = m.aggregate(results)

    assert summary["excluded_specimen_ids"] == ["g02_02"]
    assert summary["feature_count"] == 3
    assert (
        summary["sigma_floor_mm"] < 1.0
    )  # the excluded specimen's huge errors never enter


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No specimen"):
        m.aggregate([_result([], height_error_mm=None)])


def test_measure_g04_board_error_module_is_importable() -> None:
    assert m.SPECIMEN_IDS == (
        "g02_01",
        "g02_02",
        "g02_03",
        "g02_04",
        "g02_05",
        "g02_06",
    )
