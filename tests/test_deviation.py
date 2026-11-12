"""I-06: deviation computation and its GLB writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from inspection import cad, synthetic
from inspection.deviation import (
    DEVIATION_MEASURED_ATTRIBUTE,
    DEVIATION_MM_ATTRIBUTE,
    compute_deviation,
    write_deviation_glb,
)

from recon.mesh_cleanup import read_glb, read_glb_extra_attributes, write_glb
from recon.provenance import LABEL_UNOBSERVED

ROOT = Path(__file__).resolve().parents[1]
G01_CUBE = ROOT / "fixtures" / "reference_cad" / "g01_calibration_cube_30mm.stl"


def _ground_truth_transform(sample: synthetic.SyntheticSample) -> np.ndarray:
    """The exact CAD -> model transform I-01's generator used, the same pattern
    tests/test_registration.py's own _expected_transform helper uses."""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    transform[:3, 3] = np.asarray(
        sample.sidecar["transform"]["translation_mm"], dtype=float
    )
    return transform


def _outward_offset_direction(
    cad_vertices_mm: np.ndarray, sample: synthetic.SyntheticSample, vertex_index: int
) -> np.ndarray:
    """A unit vector pointing away from the cube's centre, at one posed vertex, expressed in the
    posed (model) frame. Deliberately not `recon.next_view.vertex_normals`: that function's own
    docstring says its sign "follows the mesh's winding order, which nothing upstream
    guarantees" -- fine for its own callers (which only ever use the absolute value), wrong for
    a test that needs a *known*, unambiguous sign to plant a deliberate offset against. Bounding-
    box-centre-to-vertex is unambiguous for a convex cube and needs no winding assumption.
    """
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertex = (sample.vertices_mm[vertex_index] - translation) @ rotation
    bbox_center = (cad_vertices_mm.min(axis=0) + cad_vertices_mm.max(axis=0)) / 2.0
    outward_cad = cad_frame_vertex - bbox_center
    outward_cad = outward_cad / np.linalg.norm(outward_cad)
    return outward_cad @ rotation.T


def test_compute_deviation_sign_matches_a_known_planted_offset():
    # I-06's acceptance criteria: "the sign convention is checked against a synthetic part
    # (I-01's generator) with a known planted offset." sigma_mm=0 and patch_fraction=0 give a
    # noise-free, fully-Observed posed sample with an exact ground-truth transform, so any
    # residual deviation measured elsewhere is purely floating-point roundtrip noise, not
    # ambiguity about what the "right" answer should be.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)

    # The vertex nearest the +x face's centre in CAD-frame coordinates (30, 15, 15) -- a flat
    # face, well away from any edge, where "push straight out by the CAD's own face normal"
    # has one unambiguous answer.
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertices = (sample.vertices_mm - translation) @ rotation
    face_center_target = np.array([30.0, 15.0, 15.0])
    vertex_index = int(
        np.argmin(np.linalg.norm(cad_frame_vertices - face_center_target, axis=1))
    )
    outward_direction = _outward_offset_direction(cad_vertices, sample, vertex_index)

    for planted_offset_mm in (0.6, -0.6):
        model_vertices = sample.vertices_mm.copy()
        model_vertices[vertex_index] = (
            model_vertices[vertex_index] + planted_offset_mm * outward_direction
        )

        deviation_mm, deviation_measured = compute_deviation(
            (model_vertices, sample.faces, sample.labels),
            (cad_vertices, cad_faces),
            transform,
        )

        assert deviation_measured[vertex_index] == 1.0
        assert deviation_mm[vertex_index] == pytest.approx(planted_offset_mm, abs=1e-3)
        # Every untouched vertex still sits on (or extremely close to) the CAD surface --
        # confirms the planted vertex is what moved the reading, not a systematic offset.
        untouched = np.ones(len(model_vertices), dtype=bool)
        untouched[vertex_index] = False
        assert np.abs(deviation_mm[untouched]).max() < 1e-3


def test_compute_deviation_leaves_unobserved_and_registration_only_model_vertices_unmeasured():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=1, sigma_mm=0.0, patch_fraction=0.3
    )
    assert LABEL_UNOBSERVED in set(sample.labels.tolist())

    transform = _ground_truth_transform(sample)
    deviation_mm, deviation_measured = compute_deviation(
        (sample.vertices_mm, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
    )

    is_unobserved = sample.labels == LABEL_UNOBSERVED
    assert np.any(is_unobserved)
    assert np.all(deviation_measured[is_unobserved] == 0.0)
    assert np.all(deviation_mm[is_unobserved] == 0.0)
    assert np.all(deviation_measured[~is_unobserved] == 1.0)


def test_compute_deviation_rejects_a_model_with_no_provenance():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)
    transform = _ground_truth_transform(sample)

    with pytest.raises(ValueError, match="provenance"):
        compute_deviation(
            (sample.vertices_mm, sample.faces),  # no labels
            (cad_vertices, cad_faces),
            transform,
        )


def test_compute_deviation_rejects_a_malformed_transform():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)

    with pytest.raises(ValueError, match="4x4"):
        compute_deviation(
            (sample.vertices_mm, sample.faces, sample.labels),
            (cad_vertices, cad_faces),
            np.eye(3),
        )


def test_write_deviation_glb_round_trips_through_read_glb_with_no_nan(tmp_path):
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=2, sigma_mm=0.1, patch_fraction=0.2
    )
    existing_glb_path = tmp_path / "model.glb"
    write_glb(existing_glb_path, sample.vertices_mm, sample.faces, colors=sample.colors)

    output_path = tmp_path / "model_with_deviation.glb"
    transform = _ground_truth_transform(sample)
    write_deviation_glb(
        existing_glb_path, (cad_vertices, cad_faces), transform, output_path
    )

    vertices, faces, colors = read_glb(output_path)
    assert np.allclose(vertices, sample.vertices_mm.astype(np.float32))
    assert np.array_equal(faces, sample.faces)
    assert np.array_equal(colors, sample.colors)

    extras = read_glb_extra_attributes(
        output_path, [DEVIATION_MM_ATTRIBUTE, DEVIATION_MEASURED_ATTRIBUTE]
    )
    assert set(extras) == {DEVIATION_MM_ATTRIBUTE, DEVIATION_MEASURED_ATTRIBUTE}
    for array in extras.values():
        assert array.shape == (len(vertices),)
        assert np.isfinite(array).all()
    assert set(np.unique(extras[DEVIATION_MEASURED_ATTRIBUTE]).tolist()) <= {0.0, 1.0}

    # The original file is untouched -- write_deviation_glb writes beside it, not over it.
    original_vertices, _original_faces, original_colors = read_glb(existing_glb_path)
    assert np.allclose(original_vertices, sample.vertices_mm.astype(np.float32))
    assert np.array_equal(original_colors, sample.colors)


def test_write_deviation_glb_rejects_a_mesh_with_no_color0(tmp_path):
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)
    existing_glb_path = tmp_path / "model.glb"
    write_glb(existing_glb_path, sample.vertices_mm, sample.faces)  # no colors

    with pytest.raises(ValueError, match="COLOR_0"):
        write_deviation_glb(
            existing_glb_path,
            (cad_vertices, cad_faces),
            _ground_truth_transform(sample),
            tmp_path / "out.glb",
        )
