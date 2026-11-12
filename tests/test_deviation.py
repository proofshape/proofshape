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
    publish_deviation_glb,
    write_deviation_glb,
)

from recon import service
from recon.mesh_cleanup import read_glb, read_glb_extra_attributes, write_glb
from recon.part_segmentation import BoardSurface
from recon.provenance import LABEL_INFERRED, LABEL_UNOBSERVED

ROOT = Path(__file__).resolve().parents[1]
G01_CUBE = ROOT / "fixtures" / "reference_cad" / "g01_calibration_cube_30mm.stl"

# A flat board at z=0 (slope 0, offset 0): height_above_board_mm is exactly -z, matching
# inspection.synthetic's own convention (the fixed CAD->board flip keeps the CAD's z=0 base at
# model z=0 too, so this is the board surface itself in sample.vertices_mm's frame). 2.0 mm
# mirrors recon.part_segmentation.DEFAULT_MIN_BOARD_CLEARANCE_MM without importing it, since a
# synthetic BoardSurface's own resolved height_threshold_mm is what compute_deviation actually
# reads -- the same tests/test_measure_wall_share.py pattern.
FLAT_BOARD = BoardSurface(
    slope_x=0.0,
    slope_y=0.0,
    z_at_origin_mm=0.0,
    offset_mm=0.0,
    tilt_deg=0.0,
    residual_sigma_mm=0.0,
    inlier_count=0,
    candidate_count=0,
    board_layer_top_mm=None,
    height_threshold_mm=2.0,
)


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


def _face_center_vertex_index(
    cad_vertices_mm: np.ndarray, sample: synthetic.SyntheticSample
):
    """The model-vertex index nearest the +x face's centre in CAD-frame coordinates (30, 15,
    15) -- a flat face, well away from any edge, where "push straight out" has one unambiguous
    answer."""
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertices = (sample.vertices_mm - translation) @ rotation
    face_center_target = np.array([30.0, 15.0, 15.0])
    return int(
        np.argmin(np.linalg.norm(cad_frame_vertices - face_center_target, axis=1))
    )


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
    vertex_index = _face_center_vertex_index(cad_vertices, sample)
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
            FLAT_BOARD,
        )

        assert deviation_measured[vertex_index] == 1.0
        assert deviation_mm[vertex_index] == pytest.approx(planted_offset_mm, abs=1e-3)
        # Every untouched vertex still sits on (or extremely close to) the CAD surface --
        # confirms the planted vertex is what moved the reading, not a systematic offset.
        untouched = np.ones(len(model_vertices), dtype=bool)
        untouched[vertex_index] = False
        assert np.abs(deviation_mm[untouched]).max() < 1e-3


def test_compute_deviation_does_not_hide_a_real_defect_at_g05s_largest_size():
    # Review on #309: a spatial footprint margin must not sit below the largest defect the buyer
    # cares about (G-05's bumps go to 8 mm) the same way a plain distance cutoff can't. Pushing
    # the face-centre vertex outward along the *model frame's own x axis* (not its CAD-surface
    # normal) is deliberate: a push along the surface normal can land anywhere relative to the
    # footprint's axis-aligned bounding box depending on the random yaw, so it doesn't reliably
    # stress the margin at all (confirmed while writing this test: an 8 mm surface-normal push
    # here still passed even with the margin shrunk to 1 mm, because that particular push
    # happened to be diagonal to the box). A pure +x push from the box's own x-max edge is a
    # deterministic, worst-case probe of the margin in the direction that actually matters.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    posed_cad_vertices = cad_vertices @ rotation.T + translation
    x_max_index = int(np.argmax(posed_cad_vertices[:, 0]))
    probe_xy = posed_cad_vertices[x_max_index, :2] + np.array([8.0, 0.0])
    probe = np.array(
        [probe_xy[0], probe_xy[1], -15.0]
    )  # 15 mm above the board, well clear.

    model_vertices = np.vstack([sample.vertices_mm, probe])
    model_labels = np.concatenate([sample.labels, [sample.labels[0]]])

    _deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, model_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    probe_index = len(sample.vertices_mm)
    assert deviation_measured[probe_index] == 1.0


def test_compute_deviation_excludes_board_skin_and_stray_geometry_outside_the_footprint():
    # Review on #309: D-062's "measures Observed vertices only" excludes "surface outside the
    # registered part" too -- not just Unobserved surface. Two orphan probe vertices (not
    # referenced by any face; compute_deviation only needs per-vertex labels/positions) stand in
    # for the two real failure modes named in review: board skin peeking up within the part's
    # own xy footprint, and stray Observed geometry far outside it.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    footprint_center_xy = sample.vertices_mm[:, :2].mean(axis=0)

    board_skin_probe = np.array(
        [footprint_center_xy[0], footprint_center_xy[1], -0.5]
    )  # within the footprint, 0.5 mm above the board -- under FLAT_BOARD's 2.0 mm threshold.
    stray_probe = np.array(
        [footprint_center_xy[0] + 500.0, footprint_center_xy[1], -15.0]
    )  # 500 mm outside the footprint, at a legitimate part-like height.

    model_vertices = np.vstack([sample.vertices_mm, board_skin_probe, stray_probe])
    model_labels = np.concatenate(
        [sample.labels, [sample.labels[0], sample.labels[0]]]
    )  # both probes Observed -- the exclusion must come from geometry, not provenance.

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, model_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    board_skin_index, stray_index = len(sample.vertices_mm), len(sample.vertices_mm) + 1
    assert deviation_measured[board_skin_index] == 0.0
    assert deviation_mm[board_skin_index] == 0.0
    assert deviation_measured[stray_index] == 0.0
    assert deviation_mm[stray_index] == 0.0
    # The real part surface well above the board is unaffected by adding the two probes --
    # the cube's own base face sits exactly at board height (0 mm) and is excluded by the same
    # height gate, for the same reason, which is expected rather than a regression here.
    heights_mm = -sample.vertices_mm[:, 2]
    well_above_board = heights_mm > FLAT_BOARD.height_threshold_mm + 1.0
    assert np.any(well_above_board)
    assert np.all(
        deviation_measured[: len(sample.vertices_mm)][well_above_board] == 1.0
    )


def test_compute_deviation_leaves_unobserved_vertices_unmeasured():
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
        FLAT_BOARD,
    )

    is_unobserved = sample.labels == LABEL_UNOBSERVED
    assert np.any(is_unobserved)
    assert np.all(deviation_measured[is_unobserved] == 0.0)
    assert np.all(deviation_mm[is_unobserved] == 0.0)
    # Every Observed vertex well above the board (clear of the height gate -- the cube's own
    # base face sits exactly at board height and is excluded by it regardless of label) is
    # measured.
    heights_mm = -sample.vertices_mm[:, 2]
    well_above_board = heights_mm > FLAT_BOARD.height_threshold_mm + 1.0
    assert np.any(~is_unobserved & well_above_board)
    assert np.all(deviation_measured[~is_unobserved & well_above_board] == 1.0)


def test_compute_deviation_leaves_inferred_vertices_unmeasured():
    # Review on #309: pins I-08's half of the rule here too, even though no served file carries
    # Inferred yet (A-04) -- the code already only measures LABEL_OBSERVED, so this confirms
    # that stays true rather than relying on no caller ever passing Inferred labels.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    labels = sample.labels.copy()
    labels[0] = LABEL_INFERRED

    deviation_mm, deviation_measured = compute_deviation(
        (sample.vertices_mm, sample.faces, labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    assert deviation_measured[0] == 0.0
    assert deviation_mm[0] == 0.0


def test_compute_deviation_rejects_a_model_with_no_provenance():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)
    transform = _ground_truth_transform(sample)

    with pytest.raises(ValueError, match="provenance"):
        compute_deviation(
            (sample.vertices_mm, sample.faces),  # no labels
            (cad_vertices, cad_faces),
            transform,
            FLAT_BOARD,
        )


def test_compute_deviation_rejects_a_malformed_transform():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)

    with pytest.raises(ValueError, match="4x4"):
        compute_deviation(
            (sample.vertices_mm, sample.faces, sample.labels),
            (cad_vertices, cad_faces),
            np.eye(3),
            FLAT_BOARD,
        )


def test_compute_deviation_rejects_a_non_watertight_cad():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)
    transform = _ground_truth_transform(sample)
    open_cad_faces = cad_faces[
        :-1
    ]  # drop one triangle -- no longer every edge shared twice.

    with pytest.raises(ValueError, match="watertight"):
        compute_deviation(
            (sample.vertices_mm, sample.faces, sample.labels),
            (cad_vertices, open_cad_faces),
            transform,
            FLAT_BOARD,
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
        existing_glb_path, (cad_vertices, cad_faces), transform, FLAT_BOARD, output_path
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
            FLAT_BOARD,
            tmp_path / "out.glb",
        )


def test_publish_deviation_glb_flips_the_session_record_atomically(tmp_path):
    # Review on #309: the third acceptance criterion ("flips the session record to it in one
    # atomic write, the same pattern SessionStore already uses") needs its own test against a
    # real SessionStore, not just a documented future hook.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=3, sigma_mm=0.1, patch_fraction=0.2
    )

    store = service.SessionStore(tmp_path)
    record = store.create("PO-48213-A")
    session_id = record["session_id"]
    original_glb_path = store.session_dir(session_id) / "model.glb"
    write_glb(original_glb_path, sample.vertices_mm, sample.faces, colors=sample.colors)
    record["reconstruction"] = {
        "status": "complete",
        "glb_path": str(original_glb_path),
    }
    store.save(record)

    transform = _ground_truth_transform(sample)
    output_path = publish_deviation_glb(
        store, session_id, (cad_vertices, cad_faces), transform, FLAT_BOARD
    )

    assert output_path != original_glb_path
    assert output_path.parent == original_glb_path.parent
    updated_record = store.load(session_id)
    assert updated_record["reconstruction"]["glb_path"] == str(output_path)
    assert updated_record["reconstruction"]["status"] == "complete"

    # The new file is a real, complete deviation GLB, and the original is untouched.
    vertices, _faces, _colors = read_glb(output_path)
    assert np.allclose(vertices, sample.vertices_mm.astype(np.float32))
    assert read_glb(original_glb_path)[2] is not None


def test_publish_deviation_glb_rejects_an_incomplete_session(tmp_path):
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)
    store = service.SessionStore(tmp_path)
    record = store.create("PO-48213-A")  # reconstruction.status is "not_started"

    with pytest.raises(ValueError, match="completed reconstruction"):
        publish_deviation_glb(
            store,
            record["session_id"],
            (cad_vertices, cad_faces),
            _ground_truth_transform(sample),
            FLAT_BOARD,
        )


def test_publish_deviation_glb_rejects_an_unknown_session(tmp_path):
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(cad_vertices, cad_faces, seed=0)
    store = service.SessionStore(tmp_path)

    with pytest.raises(KeyError):
        publish_deviation_glb(
            store,
            "sess_does_not_exist",
            (cad_vertices, cad_faces),
            _ground_truth_transform(sample),
            FLAT_BOARD,
        )
