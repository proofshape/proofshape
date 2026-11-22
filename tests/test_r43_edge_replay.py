"""Synthetic checks for R-43's offline signed edge measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recon import mesh_cleanup, provenance
from scripts.measure_r43_edges import fit_wall_edges, observed_mesh, signed_edge_errors


def _edges(footprint: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(zip(footprint, np.roll(footprint, -1, axis=0), strict=True))


def test_refit_corrects_mixed_wall_assignments_without_reordering_edges() -> None:
    # An asymmetric outline distinguishes all three physical edges. The fixture is 3% shorter
    # than CAD in each direction, so simply returning the seed edge lengths cannot pass.
    nominal = np.array([[0.0, 0.0], [60.0, 0.0], [0.0, 26.0]])
    actual = 0.97 * nominal
    centre = actual.mean(axis=0)
    yaw = np.deg2rad(15.0)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    posed_edges = _edges((nominal - nominal.mean(axis=0)) @ rotation.T + centre)

    rng = np.random.default_rng(37)
    wall_points = []
    for start, end in _edges(actual):
        tangent = end - start
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        fractions = np.linspace(0.01, 0.99, 200)
        noise_mm = rng.normal(scale=0.25, size=len(fractions))
        wall_points.append(
            start + fractions[:, None] * tangent + noise_mm[:, None] * normal
        )

    result = fit_wall_edges(np.concatenate(wall_points), posed_edges, gate_mm=12.0)

    actual_lengths = tuple(np.linalg.norm(end - start) for start, end in _edges(actual))
    assert result.lengths_mm == pytest.approx(actual_lengths, abs=0.25)
    assert result.angle_offsets_deg == pytest.approx((15.0, 15.0, 15.0), abs=0.5)
    assert result.first_fit_angle_offsets_deg != pytest.approx(
        result.angle_offsets_deg, abs=1.0
    )


def test_unmeasurable_wall_preserves_other_edge_identities() -> None:
    footprint = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]])
    edges = _edges(footprint)
    parts = []
    for index, (start, end) in enumerate(edges):
        # Keep one wall below the support minimum. Edges that use either corner on that wall
        # cannot be measured, while the far edge still has two independently supported corners.
        samples = 4 if index == 2 else 20
        parts.append(start + np.linspace(0.2, 0.8, samples)[:, None] * (end - start))
    parts.append(np.array([[100.0, 100.0]]))

    result = fit_wall_edges(np.concatenate(parts), edges, gate_mm=5.0)

    assert result.wall_point_counts == (20, 20, 4, 20)
    assert result.lengths_mm[0] == pytest.approx(30.0)
    assert result.lengths_mm[1:] == (None, None, None)
    assert result.angle_offsets_deg[2] is None
    assert result.first_fit_angle_offsets_deg[2] is None
    assert signed_edge_errors(result.lengths_mm, (31.0, 20.0, 30.0, 20.0)) == (
        pytest.approx(-1.0),
        None,
        None,
        None,
    )


def test_signed_errors_keep_missing_references_and_direction() -> None:
    assert signed_edge_errors((31.0, 20.0, None, 27.5), (30.0, None, 25.0, 30.0)) == (
        1.0,
        None,
        None,
        -2.5,
    )
    with pytest.raises(ValueError):
        signed_edge_errors((31.0,), (30.0, 25.0))


def test_refit_excludes_only_points_in_a_wall_assignment_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    footprint = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]])
    edges = _edges(footprint)
    wall_points = [
        start + np.linspace(0.2, 0.8, 20)[:, None] * (end - start)
        for start, end in edges
    ]
    corner_point = np.array([29.0, 1.0])

    def fit_walls_that_alternate(
        points_xy: np.ndarray, reference_edge: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        # Model a corner sample with enough leverage to push either of its assigned lines
        # away. It switches to the other wall on the next round, while each wall's 20
        # unambiguous samples stay put. Other walls retain their original positions.
        origin = reference_edge[0].copy()
        direction = reference_edge[1] - origin
        direction /= np.linalg.norm(direction)
        contains_corner = bool(np.any(np.all(points_xy == corner_point, axis=1)))
        if direction[0] > 0.9:
            origin[1] = -2.0 if contains_corner else 0.0
        elif direction[1] > 0.9:
            origin[0] = 32.0 if contains_corner else 30.0
        return origin, direction

    monkeypatch.setattr(
        "scripts.measure_r43_edges.size._fit_wall_line", fit_walls_that_alternate
    )
    band_xy = np.concatenate([*wall_points, corner_point[None, :]])

    result = fit_wall_edges(band_xy, edges, gate_mm=5.0, max_rounds=10)

    assert result.unstable_point_count == 1
    assert result.wall_point_counts == (20, 20, 20, 20)
    assert result.lengths_mm == pytest.approx((30.0, 20.0, 30.0, 20.0))


def test_observed_mesh_counts_area_only_for_fully_observed_faces(
    tmp_path: Path,
) -> None:
    vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [6, 0, 0], [9, 0, 0], [6, 4, 0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    labels = np.array([provenance.LABEL_OBSERVED] * 5 + [provenance.LABEL_UNOBSERVED])
    path = tmp_path / "mixed-labels.glb"
    mesh_cleanup.write_glb(path, vertices, faces, provenance.labels_to_colors(labels))

    points, observed_area_mm2, total_area_mm2 = observed_mesh(path)

    # The larger triangle has two Observed vertices, but no fully Observed face.
    np.testing.assert_array_equal(points, vertices[:3])
    assert observed_area_mm2 == pytest.approx(2.0)
    assert total_area_mm2 == pytest.approx(8.0)
