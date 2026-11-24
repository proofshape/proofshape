"""P-40: what the expo poster's figure renderer counts as Observed, measured and not evaluated."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recon import mesh_cleanup, provenance
from scripts.render_expo_figures import (
    NOT_EVALUATED,
    face_colours,
    observed_and_measured,
    observed_area_fraction,
)

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
FACES = np.array([[0, 1, 2], [1, 3, 2]])


def _labels(*observed: bool) -> np.ndarray:
    return np.array(
        [
            provenance.LABEL_OBSERVED if o else provenance.LABEL_UNOBSERVED
            for o in observed
        ]
    )


def test_a_mesh_without_provenance_colours_is_refused(tmp_path: Path) -> None:
    # Treating every vertex as Observed would put unseen geometry into the heatmap.
    path = tmp_path / "no-colours.glb"
    mesh_cleanup.write_glb(
        path,
        VERTICES,
        FACES,
        extra_vertex_attributes={
            "_DEVIATION_MM": np.zeros(4),
            "_DEVIATION_MEASURED": np.ones(4),
        },
    )
    with pytest.raises(ValueError, match="no provenance colours"):
        observed_and_measured(path)


def test_a_mesh_without_deviation_attributes_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "no-deviation.glb"
    mesh_cleanup.write_glb(
        path,
        VERTICES,
        FACES,
        provenance.labels_to_colors(_labels(True, True, True, True)),
    )
    with pytest.raises(ValueError, match="_DEVIATION_MM"):
        observed_and_measured(path)


def test_measured_means_observed_and_flagged_as_measured(tmp_path: Path) -> None:
    # Vertex 1 is Observed but has no measured deviation; vertex 2 carries a measured flag but
    # is Unobserved, so it doesn't count either.
    path = tmp_path / "mixed.glb"
    mesh_cleanup.write_glb(
        path,
        VERTICES,
        FACES,
        provenance.labels_to_colors(_labels(True, True, False, True)),
        {
            "_DEVIATION_MM": np.array([0.5, -1.0, 2.0, 0.1]),
            "_DEVIATION_MEASURED": np.array([1.0, 0.0, 1.0, 1.0]),
        },
    )

    _, _, observed, measured, dev = observed_and_measured(path)

    np.testing.assert_array_equal(observed, [True, True, False, True])
    np.testing.assert_array_equal(measured, [True, False, False, True])
    np.testing.assert_allclose(dev, [0.5, -1.0, 2.0, 0.1])


def test_a_face_is_grey_unless_all_three_vertices_are_measured() -> None:
    measured = np.array([True, True, True, False])
    dev = np.zeros(4)

    rgb = face_colours(FACES, measured, dev, rng=4.0)

    assert not np.allclose(rgb[0], NOT_EVALUATED)  # vertices 0, 1, 2: all measured
    np.testing.assert_allclose(rgb[1], NOT_EVALUATED)  # vertex 3 isn't


def test_coverage_is_the_services_area_figure(tmp_path: Path) -> None:
    # The poster quotes uncertainty.json's observed_fraction (an area share), not a vertex count.
    folder = tmp_path / "artifacts" / "uncertainty"
    folder.mkdir(parents=True)
    (folder / "uncertainty.json").write_text(
        json.dumps({"observed_fraction": 0.42, "vertex_count": 10})
    )

    assert observed_area_fraction(tmp_path) == pytest.approx(0.42)
