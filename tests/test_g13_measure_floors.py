"""Tests for scripts/g13_measure_floors.py — the floor-coverage and pool-split logic."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.g13_measure_floors import (
    _floor_coverage,
    _smallest_covering_floor,
    _verdict_pool_split,
)


class TestFloorCoverage:
    def test_all_within(self):
        dev = np.array([0.1, 0.2, 0.3])
        sigma = np.array([0.5, 0.5, 0.5])
        assert _floor_coverage(dev, sigma, candidate_floor=1.0, k=0.70) == 1.0

    def test_none_within(self):
        dev = np.array([10.0, 11.0, 12.0])
        sigma = np.array([0.1, 0.1, 0.1])
        assert _floor_coverage(dev, sigma, candidate_floor=0.1, k=0.70) == 0.0

    def test_partial(self):
        dev = np.array([0.1, 0.5, 5.0, 10.0])
        sigma = np.array([0.1, 0.1, 0.1, 0.1])
        cov = _floor_coverage(dev, sigma, candidate_floor=1.0, k=0.70)
        assert cov == pytest.approx(0.5)

    def test_empty(self):
        assert _floor_coverage(np.array([]), np.array([]), candidate_floor=1.0) == 0.0

    def test_sigma_dominates(self):
        dev = np.array([2.0, 2.0])
        sigma = np.array([5.0, 5.0])
        cov = _floor_coverage(dev, sigma, candidate_floor=0.1, k=0.70)
        assert cov == 1.0

    def test_floor_dominates(self):
        dev = np.array([0.5])
        sigma = np.array([0.01])
        cov = _floor_coverage(dev, sigma, candidate_floor=1.0, k=0.70)
        assert cov == 1.0


class TestSmallestCoveringFloor:
    def test_found(self):
        dev = np.array([0.5, 0.6, 0.7])
        sigma = np.array([0.1, 0.1, 0.1])
        floor = _smallest_covering_floor(dev, sigma, target_coverage=0.95, k=0.70)
        assert floor is not None
        assert floor == 1.0

    def test_already_covered(self):
        dev = np.array([0.01, 0.02])
        sigma = np.array([0.5, 0.5])
        floor = _smallest_covering_floor(dev, sigma, target_coverage=0.95, k=0.70)
        assert floor == 0.1

    def test_not_found(self):
        dev = np.array([100.0, 200.0])
        sigma = np.array([0.01, 0.01])
        floor = _smallest_covering_floor(dev, sigma, target_coverage=0.95, k=0.70)
        assert floor is None


class TestVerdictPoolSplit:
    def test_basic_split(self):
        vertices = np.array(
            [
                [0, 0, -10],
                [0, 0, -20],
                [0, 0, -30],
                [0, 0, -40],
                [0, 0, -50],
                [0, 0, -60],
                [0, 0, -70],
                [0, 0, -80],
                [0, 0, -90],
                [0, 0, -100],
            ],
            dtype=np.float64,
        )
        board_plane = (0.0, 0.0, 0.0)
        judged = np.ones(10, dtype=bool)
        oop, ip = _verdict_pool_split(vertices, board_plane, judged)
        assert oop.sum() == 1
        assert ip.sum() == 9
        assert oop[9]

    def test_unjudged_excluded(self):
        vertices = np.array(
            [
                [0, 0, -10],
                [0, 0, -20],
                [0, 0, -100],
            ],
            dtype=np.float64,
        )
        board_plane = (0.0, 0.0, 0.0)
        judged = np.array([True, True, False])
        oop, ip = _verdict_pool_split(vertices, board_plane, judged)
        assert oop.sum() + ip.sum() == 2
        assert not oop[2]
        assert not ip[2]
