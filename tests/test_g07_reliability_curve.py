"""Unit tests for G-07's reliability-curve analysis functions.

Only the pure analysis functions (`bin_by_sigma`, `measure_k`, `pool_pairs`,
`specimen_offsets_mm`, `specimen_summary`, `scatter_pairs`, `_read_board_plane`) are tested here,
on synthetic (sigma, error) pairs -- the script's own pipeline orchestration (`ensure_pipeline`,
`measure_specimen`) needs the real cached recon fixtures (`fixtures/data/`, gitignored) and is
exercised by actually running the script, not by a test, same convention
`test_measure_g04_board_error.py` already established for its own aggregate-only testing.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from scripts.g07_reliability_curve import (
    _read_board_plane,
    bin_by_sigma,
    measure_k,
    pool_pairs,
    scatter_pairs,
    specimen_offsets_mm,
    specimen_summary,
)


def test_bin_by_sigma_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No .* pairs"):
        bin_by_sigma([])


def test_bin_by_sigma_sorts_vertices_into_increasing_sigma_ranges() -> None:
    rng = np.random.default_rng(0)
    sigmas = np.concatenate(
        [
            rng.uniform(0.1, 0.5, size=200),
            rng.uniform(1.0, 1.5, size=200),
            rng.uniform(2.0, 2.5, size=200),
        ]
    )
    errors = sigmas * 2.0
    pairs = list(zip(sigmas.tolist(), errors.tolist()))

    bins = bin_by_sigma(pairs, bin_count=3)

    assert len(bins) == 3
    assert sum(b["vertex_count"] for b in bins) == len(pairs)
    # sigma ranges are non-decreasing across bins.
    for earlier, later in itertools.pairwise(bins):
        assert earlier["sigma_range_mm"][1] <= later["sigma_range_mm"][1]
    # error = 2*sigma exactly, so each bin's mean error tracks its mean sigma.
    for b in bins:
        assert b["mean_abs_error_mm"] == pytest.approx(
            2.0 * b["mean_sigma_mm"], abs=1e-6
        )


def test_measure_k_recovers_a_known_linear_relationship() -> None:
    rng = np.random.default_rng(1)
    sigma_mm = rng.uniform(0.1, 2.0, size=2000)
    error_mm = 3.0 * sigma_mm  # error = k*sigma exactly, k=3
    pairs = list(zip(sigma_mm.tolist(), error_mm.tolist()))

    result = measure_k(pairs, coverage=0.95)

    assert result["k_at_coverage"] == pytest.approx(3.0, abs=1e-6)
    assert result["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)


def test_measure_k_reports_a_weak_correlation_when_error_is_unrelated_to_sigma() -> (
    None
):
    rng = np.random.default_rng(2)
    sigma_mm = rng.uniform(0.1, 2.0, size=2000)
    error_mm = rng.uniform(0.1, 2.0, size=2000)  # independent of sigma
    pairs = list(zip(sigma_mm.tolist(), error_mm.tolist()))

    result = measure_k(pairs, coverage=0.95)

    # Not a fixed numeric assertion (that would re-introduce a hardcoded "reliable" threshold the
    # module docstring explicitly avoids) -- just confirms the correlation correctly reads as
    # near zero for two independent distributions, not near 1 like the linear case above.
    assert abs(result["pearson_correlation"]) < 0.15


def test_measure_k_handles_every_sigma_being_zero() -> None:
    pairs = [(0.0, 0.5), (0.0, 1.2), (0.0, 0.1)]

    result = measure_k(pairs)

    assert result["k_at_coverage"] is None
    assert "cannot be computed" in result["note"]


def _result(specimen_id: str, pairs: list[tuple[float, float]]) -> dict:
    return {"specimen_id": specimen_id, "pairs": pairs}


def test_pool_pairs_takes_absolute_value_of_signed_error() -> None:
    results = [
        _result("g02_01", [(1.0, -0.1), (1.5, 0.2)]),
        _result("g02_02", [(2.0, -3.0)]),
    ]

    pairs = pool_pairs(results)

    assert pairs == [(1.0, 0.1), (1.5, 0.2), (2.0, 3.0)]


def test_specimen_offsets_mm_is_the_median_signed_error_per_specimen() -> None:
    results = [
        _result("s-04", [(1.0, -0.7), (1.2, -0.6), (0.9, -0.8)]),
        _result("g02_06", [(1.0, 0.6), (1.1, 0.7)]),
    ]

    offsets = specimen_offsets_mm(results)

    assert offsets == {"s-04": -0.7, "g02_06": pytest.approx(0.65)}


def test_scatter_pairs_removes_each_specimens_own_offset() -> None:
    results = [
        _result("s-04", [(1.0, -0.7), (1.2, -0.5)]),
        _result("g02_06", [(1.0, 0.7)]),
    ]
    offsets = {"s-04": -0.6, "g02_06": 0.7}

    pairs = scatter_pairs(results, offsets)

    # abs(error - offset) for each pair, offset looked up by the pair's own specimen.
    assert len(pairs) == 3
    expected = [(1.0, 0.1), (1.2, 0.1), (1.0, 0.0)]
    for actual_pair, expected_pair in zip(pairs, expected):
        assert actual_pair == pytest.approx(expected_pair)


def test_scatter_pairs_skips_specimens_with_no_offset() -> None:
    results = [
        _result("s-04", [(1.0, -0.7)]),
        _result("g02_06", [(1.0, 0.7)]),
    ]
    offsets = {"s-04": -0.7}  # g02_06 missing, e.g. it had zero pairs

    pairs = scatter_pairs(results, offsets)

    assert pairs == [(1.0, 0.0)]


def test_specimen_summary_reports_offset_scatter_and_median_sigma() -> None:
    results = [_result("s-04", [(1.0, -0.7), (1.2, -0.5), (0.8, -0.9)])]
    offsets = specimen_offsets_mm(results)

    summary = specimen_summary(results, offsets)

    assert len(summary) == 1
    row = summary[0]
    assert row["specimen_id"] == "s-04"
    assert row["vertex_count"] == 3
    assert row["median_signed_error_mm"] == pytest.approx(-0.7)
    assert row["median_sigma_mm"] == pytest.approx(1.0)
    # scatter = abs(error - offset): |-0.7-(-0.7)|=0, |-0.5-(-0.7)|=0.2, |-0.9-(-0.7)|=0.2
    assert row["mad_scatter_mm"] == pytest.approx(0.2)


def test_read_board_plane_reads_tsdf_fusion_jsons_board_surface(tmp_path) -> None:
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    (mesh_dir / "tsdf_fusion.json").write_text(
        json.dumps(
            {
                "board_surface": {
                    "slope_x": 0.0123,
                    "slope_y": -0.0456,
                    "z_at_origin_mm": -0.789,
                }
            }
        )
    )

    board_plane = _read_board_plane(mesh_dir)

    assert board_plane == (0.0123, -0.0456, -0.789)
