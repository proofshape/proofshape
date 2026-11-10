"""R-32 tests for CAD-wall lean and thickness measurements."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from recon import mesh_cleanup, provenance
from recon.cad_size_check import (
    WALL_OFFSET_CONVENTION,
    load_observed_surface_points,
    measure_wall_profiles,
    observed_glb_labels,
    prism_from_cad,
    read_ascii_stl,
)

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
                -half_thickness_mm, half_thickness_mm, 11
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


# --- the sign, the Observed filter and the label name (review of #280) ----------------------------

S05_STL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "reference_cad"
    / "s05_rect_prism_30x30x60mm.stl"
)


WALL_HEIGHTS_MM = np.linspace(4.0, 56.0, 27)


def _wall_points_offset(edges, outward_mm_at_height, heights=WALL_HEIGHTS_MM):
    """Points on every CAD wall, pushed outward by outward_mm_at_height(h) at height h."""
    rows = []
    footprint_centre = np.mean([p0 for p0, _ in edges], axis=0)
    for p0, p1 in edges:
        direction = (p1 - p0) / np.linalg.norm(p1 - p0)
        outward = np.array([direction[1], -direction[0]])
        if np.dot(outward, (p0 + p1) / 2.0 - footprint_centre) < 0:
            outward = -outward
        for height_mm in heights:
            for t in np.linspace(0.1, 0.9, 15):
                xy = p0 + t * (p1 - p0) + outward_mm_at_height(height_mm) * outward
                rows.append([xy[0], xy[1], -height_mm])
    return np.asarray(rows, dtype=np.float64)


def test_wall_offset_sign_is_positive_inside_the_cad_wall() -> None:
    # Through the real CAD path, so the footprint's vertex order is cv2.convexHull's own.
    prism = prism_from_cad(read_ascii_stl(S05_STL))
    edges = [(np.asarray(p0, float), np.asarray(p1, float)) for p0, p1 in prism.edges]

    outside = measure_wall_profiles(
        _wall_points_offset(edges, lambda h: 1.0), edges, FLAT_BOARD, 60.0, 0.2
    )
    inside = measure_wall_profiles(
        _wall_points_offset(edges, lambda h: -1.0), edges, FLAT_BOARD, 60.0, 0.2
    )
    # Inward by 1 mm at the bottom of the band, 0 at the top: a wall leaning *outward* going up.
    leaning_out = measure_wall_profiles(
        _wall_points_offset(edges, lambda h: (h - 30.0) / 24.0),
        edges,
        FLAT_BOARD,
        60.0,
        0.2,
    )

    # The same walls with their winding reversed, as a mirrored CAD fit leaves them: the sign
    # must not change, because _fit_rigid_2d may return either handedness.
    mirrored = [(p1, p0) for p0, p1 in reversed(edges)]
    outside_mirrored = measure_wall_profiles(
        _wall_points_offset(mirrored, lambda h: 1.0), mirrored, FLAT_BOARD, 60.0, 0.2
    )

    assert len(outside) == len(inside) == len(leaning_out) == len(outside_mirrored) == 4
    for profile in outside_mirrored:
        assert all(
            item.offset_mm == pytest.approx(-1.0, abs=1e-6) for item in profile.bins
        )
    for profile in outside:
        assert all(
            item.offset_mm == pytest.approx(-1.0, abs=1e-6) for item in profile.bins
        )
    for profile in inside:
        assert all(
            item.offset_mm == pytest.approx(1.0, abs=1e-6) for item in profile.bins
        )
    for profile in leaning_out:
        assert profile.lean_mm < 0 and profile.lean_deg < 0
    assert "positive means inside" in WALL_OFFSET_CONVENTION


def _labelled_glb(path: Path) -> tuple[np.ndarray, set[int]]:
    """Two triangles share an edge: one fully Observed, one with an Unobserved corner."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [20.0, 20.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 3, 4]])
    labels = np.array(
        [
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,
        ],
        dtype=object,
    )
    mesh_cleanup.write_glb(path, vertices, faces, provenance.labels_to_colors(labels))
    return vertices, {0, 1, 2}


def test_only_vertices_of_fully_observed_faces_are_measured(tmp_path) -> None:
    vertices, expected = _labelled_glb(tmp_path / "mesh_uncertainty.glb")

    points = load_observed_surface_points(tmp_path / "mesh_uncertainty.glb")

    # Vertex 3 is Observed, but its only face has an Unobserved corner, so it is left out too.
    assert sorted(map(tuple, points)) == sorted(tuple(vertices[i]) for i in expected)


def test_the_report_names_which_labels_the_fused_glb_carries() -> None:
    assert observed_glb_labels(Path("a/uncertainty/mesh_uncertainty.glb")).startswith(
        "served"
    )
    assert observed_glb_labels(Path("a/provenance/mesh_provenance.glb")).startswith(
        "not served"
    )
    assert "unrecognised" in observed_glb_labels(Path("model.glb"))
