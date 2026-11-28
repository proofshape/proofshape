"""I-06: deviation computation and its GLB writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from inspection import cad, synthetic
from inspection.deviation import (
    DEFAULT_FOOTPRINT_MARGIN_MM,
    DEVIATION_MEASURED_ATTRIBUTE,
    DEVIATION_MM_ATTRIBUTE,
    _cad_outer_outline_loops,
    _within_cad_outer_outline,
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
G05_CONTROL = ROOT / "fixtures" / "reference_cad" / "g05_00_control_50x32x20mm.stl"
REFERENCE_CADS = sorted((ROOT / "fixtures" / "reference_cad").glob("*.stl"))

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


def _top_face_center_vertex_index(
    cad_vertices_mm: np.ndarray, sample: synthetic.SyntheticSample
) -> int:
    """The model vertex nearest the CAD top-face centre (15, 15, 30)."""
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertices = (sample.vertices_mm - translation) @ rotation
    target = np.array([15.0, 15.0, 30.0])
    return int(np.argmin(np.linalg.norm(cad_frame_vertices - target, axis=1)))


def _top_face_outward_direction(sample: synthetic.SyntheticSample) -> np.ndarray:
    """CAD-frame +z mapped into the model frame; stays inside the xy footprint."""
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    direction = np.array([0.0, 0.0, 1.0]) @ rotation.T
    return direction / np.linalg.norm(direction)


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
    # I-14 deliberately leaves surface outside the xy footprint unmeasured, so keep this
    # sign-convention regression on the top face: +/- z changes signed distance without moving
    # the probe outside the registered xy footprint.
    vertex_index = _top_face_center_vertex_index(cad_vertices, sample)
    outward_direction = _top_face_outward_direction(sample)

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


def _cad_point_to_model(
    sample: synthetic.SyntheticSample, point_cad_mm: np.ndarray
) -> np.ndarray:
    transform = _ground_truth_transform(sample)
    point_h = np.append(np.asarray(point_cad_mm, dtype=float), 1.0)
    return (transform @ point_h)[:3]


@pytest.mark.parametrize("height_mm", [4.0, 8.0])
def test_compute_deviation_keeps_top_face_defects_inside_the_footprint_measured(
    height_mm: float,
):
    # I-14: narrowing the xy footprint must not hide the defects G-05 actually plants. Its bump
    # ladder grows upward from the top face, inside the outer footprint; the largest is 8 mm.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)

    # The cube top is z=30 mm in CAD coordinates. Probe its centre, raised by the planted amount,
    # so the xy position remains safely inside the registered footprint.
    probe = _cad_point_to_model(sample, np.array([15.0, 15.0, 30.0 + height_mm]))
    model_vertices = np.vstack([sample.vertices_mm, probe])
    model_labels = np.concatenate([sample.labels, [sample.labels[0]]])

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, model_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    probe_index = len(sample.vertices_mm)
    assert deviation_measured[probe_index] == 1.0
    assert deviation_mm[probe_index] == pytest.approx(height_mm, abs=1e-3)


def test_compute_deviation_excludes_outward_flare_beyond_the_cad_outer_outline():
    # I-14 / #407: the false base flare is multi-millimetre geometry beyond the CAD outline.
    # Put the probe beyond the CAD's +x outline + margin in CAD-local coordinates, then transform
    # it with I-01's nonzero-yaw pose. This specifically prevents a rotated board-frame AABB from
    # re-admitting outside-outline geometry at its enlarged corners.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    flare_probe = _cad_point_to_model(
        sample,
        np.array([30.0 + DEFAULT_FOOTPRINT_MARGIN_MM + 1.0, 15.0, 5.4]),
    )

    model_vertices = np.vstack([sample.vertices_mm, flare_probe])
    model_labels = np.concatenate([sample.labels, [sample.labels[0]]])
    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, model_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    probe_index = len(sample.vertices_mm)
    assert deviation_measured[probe_index] == 0.0
    assert deviation_mm[probe_index] == 0.0


def test_compute_deviation_pins_the_cad_outer_outline_boundary():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    cad_max_x = float(cad_vertices[:, 0].max())
    cad_mid_y = float((cad_vertices[:, 1].min() + cad_vertices[:, 1].max()) / 2.0)
    at_boundary = _cad_point_to_model(
        sample, np.array([cad_max_x + DEFAULT_FOOTPRINT_MARGIN_MM, cad_mid_y, 15.0])
    )
    beyond_boundary = _cad_point_to_model(
        sample,
        np.array([cad_max_x + DEFAULT_FOOTPRINT_MARGIN_MM + 1e-4, cad_mid_y, 15.0]),
    )

    model_vertices = np.vstack([sample.vertices_mm, at_boundary, beyond_boundary])
    model_labels = np.concatenate([sample.labels, [sample.labels[0], sample.labels[0]]])
    _deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, model_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    at_index = len(sample.vertices_mm)
    beyond_index = at_index + 1
    assert deviation_measured[at_index] == 1.0
    assert deviation_measured[beyond_index] == 0.0


def test_outline_margin_pins_at_2mm():
    # Probes at 1.9 mm (just inside) and 2.1 mm (just outside) the outline pin the 2 mm constant.
    # Literal distances, not expressions of DEFAULT_FOOTPRINT_MARGIN_MM.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    cad_max_x = float(cad_vertices[:, 0].max())
    cad_mid_y = float((cad_vertices[:, 1].min() + cad_vertices[:, 1].max()) / 2.0)
    within_margin = _cad_point_to_model(
        sample, np.array([cad_max_x + 1.9, cad_mid_y, 15.0])
    )
    beyond_margin = _cad_point_to_model(
        sample, np.array([cad_max_x + 2.1, cad_mid_y, 15.0])
    )

    model_vertices = np.vstack([sample.vertices_mm, within_margin, beyond_margin])
    model_labels = np.concatenate([sample.labels, [sample.labels[0], sample.labels[0]]])
    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, model_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )

    within_idx = len(sample.vertices_mm)
    beyond_idx = within_idx + 1
    assert deviation_measured[within_idx] == 1.0
    assert deviation_mm[within_idx] == pytest.approx(1.9, abs=0.1)
    assert deviation_measured[beyond_idx] == 0.0


@pytest.mark.parametrize("cad_path", REFERENCE_CADS, ids=lambda path: path.stem)
def test_outer_outline_builder_handles_every_reference_cad(cad_path: Path):
    # Review on #416 found that the first silhouette implementation raised on pockets, bumps and
    # the open-top tray because feature edges at different z levels projected to the same xy
    # segment with opposite winding. Every shipped reference CAD must produce a usable outer
    # outline instead of turning the whole order into Rescan.
    cad_vertices, cad_faces, _units = cad.load_mesh(cad_path)

    loops = _cad_outer_outline_loops(cad_vertices, cad_faces)

    assert loops
    assert all(len(loop) >= 3 for loop in loops)


def test_g05_outer_outline_excludes_notch_but_keeps_through_hole_measurable():
    # Tabeen's I-14 review: a CAD-local bounding box still admits the 16 x 11 corner notch.
    # The eligibility outline must follow that concavity, but it must *not* turn the 12 x 12
    # through-hole into "unmeasured" geometry because X-03 needs the hole shift to stay measurable.
    cad_vertices, cad_faces, _units = cad.load_mesh(G05_CONTROL)
    loops = _cad_outer_outline_loops(cad_vertices, cad_faces)
    probes_xy = np.array(
        [
            [42.0, 26.0],  # inside the designed corner notch: outside the outer outline
            [12.0, 16.0],  # centre of the through-hole: keep measurable
            [40.0, 10.0],  # ordinary material inside the outline
            [53.0, 16.0],  # beyond the plate's +x edge + 2 mm margin
        ],
        dtype=float,
    )

    within = _within_cad_outer_outline(probes_xy, loops, DEFAULT_FOOTPRINT_MARGIN_MM)

    assert within.tolist() == [False, True, True, False]


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


def test_publish_deviation_glb_flips_the_session_record_atomically(
    tmp_path, monkeypatch
):
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

    from inspection import deviation

    def write_with_agent_update(*args, **kwargs):
        # The publisher has loaded its input record, but a late agent turn writes now.
        store.update_agent_message(
            session_id, fallback_reason=None, message="Latest guidance"
        )
        return write_deviation_glb(*args, **kwargs)

    monkeypatch.setattr(deviation, "write_deviation_glb", write_with_agent_update)
    transform = _ground_truth_transform(sample)
    output_path = publish_deviation_glb(
        store, session_id, (cad_vertices, cad_faces), transform, FLAT_BOARD
    )

    assert output_path != original_glb_path
    assert output_path.parent == original_glb_path.parent
    updated_record = store.load(session_id)
    assert updated_record["reconstruction"]["glb_path"] == str(output_path)
    assert updated_record["reconstruction"]["status"] == "complete"
    assert updated_record["agent_message"]["message"] == "Latest guidance"

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
