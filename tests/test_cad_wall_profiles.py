"""R-32 tests for CAD-wall lean and thickness measurements."""

import math

import numpy as np
import pytest

from recon.cad_size_check import measure_wall_profiles


FLAT_BOARD = (0.0, 0.0, 0.0)


def _synthetic_wall(
    *,
    lean_deg: float,
    half_thickness_mm: float,
) -> np.ndarray:
    """Dense x-axis wall whose signed y offset changes linearly with height."""
    slope = math.tan(math.radians(lean_deg))
    rows = []
    for height_mm in np.linspace(5.0, 45.0, 17):
        centre_offset_mm = slope * (height_mm - 25.0)
        for x_mm in np.linspace(5.0, 45.0, 9):
            for thickness_offset_mm in np.linspace(
                -half_thickness_mm, half_thickness_mm, 9
            ):
                rows.append(
                    [
                        x_mm,
                        centre_offset_mm + thickness_offset_mm,
                        -height_mm,
                    ]
                )
    return np.asarray(rows, dtype=np.float64)


def test_measure_wall_profiles_recovers_known_lean_and_thickness() -> None:
    points = _synthetic_wall(lean_deg=4.0, half_thickness_mm=1.0)
    edge = (np.array([0.0, 0.0]), np.array([50.0, 0.0]))

    profiles = measure_wall_profiles(
        points,
        [edge],
        FLAT_BOARD,
        part_height_mm=50.0,
        alignment_residual_mm=0.2,
        bin_count=5,
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.edge_index == 0
    assert profile.lean_deg == pytest.approx(4.0, abs=0.2)
    expected_span = profile.height_max_mm - profile.height_min_mm
    expected_lean_mm = math.tan(math.radians(4.0)) * expected_span
    assert profile.lean_mm == pytest.approx(expected_lean_mm, abs=0.15)
    assert profile.thickness_median_mm == pytest.approx(1.6, abs=0.15)
    assert len(profile.bins) == 5
    assert all(item.point_count >= 5 for item in profile.bins)


def test_measure_wall_profiles_flat_wall_has_near_zero_lean() -> None:
    points = _synthetic_wall(lean_deg=0.0, half_thickness_mm=0.5)
    edge = (np.array([0.0, 0.0]), np.array([50.0, 0.0]))

    profile = measure_wall_profiles(
        points,
        [edge],
        FLAT_BOARD,
        part_height_mm=50.0,
        alignment_residual_mm=0.2,
    )[0]

    assert profile.lean_deg == pytest.approx(0.0, abs=1e-9)
    assert profile.lean_mm == pytest.approx(0.0, abs=1e-9)
    assert profile.thickness_median_mm == pytest.approx(0.8, abs=0.1)


def test_measure_wall_profiles_skips_edges_without_enough_surface_evidence() -> None:
    points = np.array(
        [
            [10.0, 0.0, -10.0],
            [20.0, 0.0, -20.0],
            [30.0, 0.0, -30.0],
        ]
    )
    edge = (np.array([0.0, 0.0]), np.array([50.0, 0.0]))

    assert (
        measure_wall_profiles(
            points,
            [edge],
            FLAT_BOARD,
            part_height_mm=50.0,
            alignment_residual_mm=0.2,
        )
        == []
    )
