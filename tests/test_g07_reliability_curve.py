"""Unit tests for G-07's reliability-curve analysis functions.

Only the pure analysis functions (`bin_by_sigma`, `measure_k`, `pool_pairs`) are tested here, on
synthetic (sigma, error) pairs -- the script's own pipeline orchestration (`ensure_pipeline`,
`measure_specimen`) needs the real cached recon fixtures (`fixtures/data/`, gitignored) and is
exercised by actually running the script, not by a test, same convention
`test_measure_g04_board_error.py` already established for its own aggregate-only testing.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scripts.g07_reliability_curve import bin_by_sigma, measure_k, pool_pairs


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


def _result(specimen_id: str, pairs: list[tuple[float, float]], *, ok: bool) -> dict:
    return {
        "specimen_id": specimen_id,
        "legacy_alignment_ok": ok,
        "pairs": pairs,
    }


def test_pool_pairs_includes_everything_by_default() -> None:
    results = [
        _result("g02_01", [(1.0, 0.1), (1.5, 0.2)], ok=True),
        _result("g02_02", [(2.0, 3.0)], ok=False),
    ]

    pairs = pool_pairs(results)

    assert pairs == [(1.0, 0.1), (1.5, 0.2), (2.0, 3.0)]


def test_pool_pairs_excludes_bad_alignment_specimens_when_required() -> None:
    results = [
        _result("g02_01", [(1.0, 0.1), (1.5, 0.2)], ok=True),
        _result("g02_02", [(2.0, 3.0)], ok=False),
    ]

    pairs = pool_pairs(results, require_legacy_alignment_ok=True)

    assert pairs == [(1.0, 0.1), (1.5, 0.2)]
