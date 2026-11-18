"""I-07: compute_verdict against proposal 7.2's four-row table, plus the #157 regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from inspection import cad, synthetic
from inspection.deviation import compute_deviation
from inspection.verdict import (
    D073_SIGMA_FLOOR_BOARD_MM,
    D077_K,
    D078_SIGMA_FLOOR_IN_PLANE_MM,
    RefusedToleranceError,
    compute_verdict,
)

from recon.part_segmentation import BoardSurface
from recon.provenance import LABEL_OBSERVED, LABEL_UNOBSERVED

ROOT = Path(__file__).resolve().parents[1]
G01_CUBE = ROOT / "fixtures" / "reference_cad" / "g01_calibration_cube_30mm.stl"

# Same flat-board fixture test_deviation.py uses, duplicated rather than shared across test
# files (this project's own established convention -- test_registration.py does likewise).
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
FLAT_BOARD_PLANE = (FLAT_BOARD.slope_x, FLAT_BOARD.slope_y, FLAT_BOARD.z_at_origin_mm)


def _ground_truth_transform(sample: synthetic.SyntheticSample) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    transform[:3, 3] = np.asarray(
        sample.sidecar["transform"]["translation_mm"], dtype=float
    )
    return transform


def _outward_offset_direction(
    cad_vertices_mm: np.ndarray, sample: synthetic.SyntheticSample, vertex_index: int
) -> np.ndarray:
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertex = (sample.vertices_mm[vertex_index] - translation) @ rotation
    bbox_center = (cad_vertices_mm.min(axis=0) + cad_vertices_mm.max(axis=0)) / 2.0
    outward_cad = cad_frame_vertex - bbox_center
    outward_cad = outward_cad / np.linalg.norm(outward_cad)
    return outward_cad @ rotation.T


def _face_center_vertex_index(
    cad_vertices_mm: np.ndarray, sample: synthetic.SyntheticSample
) -> int:
    """A side-wall face's center vertex -- in-plane under D-073's board/wall split (confirmed:
    height-above-board 15mm, well below this fixture's ~30mm top-region threshold)."""
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
    """The top face's own center vertex -- out-of-plane under D-073's board/wall split
    (confirmed: height-above-board 30mm, exactly this fixture's top-region threshold, since a
    perfect cube's top face is flat). Needed because Pass/Fail can only fire from the
    out-of-plane pool now (D-073) -- `_face_center_vertex_index`'s side-wall vertex no longer
    qualifies."""
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertices = (sample.vertices_mm - translation) @ rotation
    top_face_center_target = np.array([15.0, 15.0, 30.0])
    return int(
        np.argmin(np.linalg.norm(cad_frame_vertices - top_face_center_target, axis=1))
    )


def _second_top_face_vertex_index(
    cad_vertices_mm: np.ndarray, sample: synthetic.SyntheticSample
) -> int:
    """A second, distinct out-of-plane (top-face) vertex, near a corner rather than the center
    -- the whole top face sits at the same cad-frame z=30mm as `_top_face_center_vertex_index`
    (this fixture's cube), strictly above every wall vertex's z=15mm, so any point on it clears
    the top-10%-of-height-above-board threshold the same way. Needed for PR #379's regression
    tests, which plant two different top-face vertices to show Fail/Rescan must be decided from
    every vertex in a pool, not just its single largest-deviation one."""
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    translation = np.asarray(sample.sidecar["transform"]["translation_mm"], dtype=float)
    cad_frame_vertices = (sample.vertices_mm - translation) @ rotation
    corner_target = np.array([3.0, 3.0, 30.0])
    return int(np.argmin(np.linalg.norm(cad_frame_vertices - corner_target, axis=1)))


def _top_face_outward_direction(sample: synthetic.SyntheticSample) -> np.ndarray:
    """The top face's own normal (CAD-frame +z, mapped into model frame) -- unlike
    `_outward_offset_direction`, this is the same direction for *every* top-face vertex, center
    or corner, since the whole face is flat. `_outward_offset_direction`'s bbox-center-relative
    heuristic only coincides with the true normal for a vertex directly above the bbox center
    (confirmed: a corner vertex offset that way measures well under its own planted magnitude,
    since part of the displacement is tangent to the face, not normal to it)."""
    rotation = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    direction = np.array([0.0, 0.0, 1.0]) @ rotation.T
    return direction / np.linalg.norm(direction)


def _clean_sample_deviation(seed: int = 0, patch_fraction: float = 0.0):
    """A noise-free, (fully or partially) Observed synthetic cube, with its real per-vertex
    deviation against the CAD -- essentially zero everywhere since nothing was planted."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=seed, sigma_mm=0.0, patch_fraction=patch_fraction
    )
    transform = _ground_truth_transform(sample)
    deviation_mm, deviation_measured = compute_deviation(
        (sample.vertices_mm, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    return sample, cad_vertices, cad_faces, deviation_mm, deviation_measured


def _spec(name: str = "overall", tolerance_mm: float = 5.0) -> dict:
    return {
        "features": [{"name": name, "kind": "overall", "tolerance_mm": tolerance_mm}]
    }


def test_clean_part_passes():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)  # below the floor; floor applies

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=5.0),
    )

    assert result["verdict"] == "Pass"
    assert result["orientation_ambiguous"] is False
    assert result["regions"][0]["contributes_to_verdict"] is True
    assert result["reason_codes"] == []


def test_planted_deviation_beyond_tolerance_fails():
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    # Top-region (out-of-plane) vertex, not a wall one -- D-073 restricts Fail/Pass to that pool.
    vertex_index = _top_face_center_vertex_index(cad_vertices, sample)
    outward = _outward_offset_direction(cad_vertices, sample, vertex_index)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[vertex_index] = model_vertices[vertex_index] + 2.0 * outward

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    sigma_mm = np.full(len(sample.labels), 0.3)

    # 2.0mm planted deviation; tolerance 0.5mm + k*floor (0.70*0.86=0.602mm) margin = 1.102mm,
    # well under 2.0mm -- must Fail.
    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=0.5),
    )

    assert result["verdict"] == "Fail"
    assert result["regions"][0]["contributes_to_verdict"] is True
    assert "overall_deviation_out_of_spec" in result["reason_codes"]


def test_in_plane_deviation_beyond_the_measured_floor_margin_fails():
    """D-078/R-37 gives in-plane (wall) comparisons their own real measured floor (5.278mm) --
    a genuine, large wall-side defect, well beyond even that floor's margin, must still Fail.
    tolerance 0.5mm + k*D078_SIGMA_FLOOR_IN_PLANE_MM (0.70*5.278=3.6946mm) margin = 4.1946mm;
    a 6.0mm planted deviation clears it comfortably."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    vertex_index = _face_center_vertex_index(cad_vertices, sample)
    outward = _outward_offset_direction(cad_vertices, sample, vertex_index)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[vertex_index] = model_vertices[vertex_index] + 6.0 * outward

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    sigma_mm = np.full(len(sample.labels), 0.3)

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=0.5),
    )

    assert result["verdict"] == "Fail"
    assert result["regions"][0]["contributes_to_verdict"] is True
    assert "overall_deviation_out_of_spec" in result["reason_codes"]


def test_in_plane_deviation_within_the_measured_floor_margin_rescans_not_passes():
    """The flip side of D-078's floor being real and large: a wall-side defect too small to
    clear the floor's own margin cannot be told apart from noise, so it must Rescan as
    high-sigma, never silently Pass -- the same honesty D-073's height floor already required,
    now true for in-plane too (caught on review, @TabeenRaoof, PR #379: the first version of
    this story left in-plane unfloored and would have Passed this exact case on raw sigma
    alone)."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    vertex_index = _face_center_vertex_index(cad_vertices, sample)
    outward = _outward_offset_direction(cad_vertices, sample, vertex_index)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[vertex_index] = model_vertices[vertex_index] + 2.0 * outward

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    sigma_mm = np.full(len(sample.labels), 0.3)

    # 2.0mm planted deviation; tolerance 0.5mm + k*floor (0.70*5.278=3.6946mm) margin = 4.1946mm
    # -- 2.0mm sits well inside it, so this cannot Fail; D-078's floor also blocks a Pass
    # (5.278 > 0.5/0.70=0.714mm), so it must Rescan.
    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=0.5),
    )

    assert result["verdict"] == "Rescan"
    assert "overall_in_plane_high_sigma" in result["reason_codes"]


def test_157_nan_sigma_in_plane_is_also_clamped_via_fmax_not_passed_through():
    """The in-plane analogue of the #157 regression below: a NaN raw sigma (R-10's
    fewer-than-two-observations marker) on a wall vertex must be floored to
    D078_SIGMA_FLOOR_IN_PLANE_MM via np.fmax, not propagated into the comparison -- the same
    defence out-of-plane already had, now true for in-plane since D-078 gives it a real floor
    too."""
    sample, cad_vertices, _cf, deviation_mm, deviation_measured = (
        _clean_sample_deviation()
    )
    wall_vertex_index = _face_center_vertex_index(cad_vertices, sample)
    deviation_mm = deviation_mm.copy()
    deviation_mm[wall_vertex_index] = 0.05
    sigma_mm = np.full(len(sample.labels), 0.3)
    sigma_mm[wall_vertex_index] = np.nan

    # tolerance 10.0mm is generous enough that D078_SIGMA_FLOOR_IN_PLANE_MM (5.278mm) itself
    # clears tolerance/k (10.0/0.70=14.29mm) -- isolates "does NaN crash or silently misbehave"
    # from the separate, much stricter floor-vs-tight-tolerance Rescan case above.
    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=10.0),
    )

    assert result["verdict"] == "Pass"


def test_fail_can_be_driven_by_a_vertex_other_than_the_largest_deviation_one():
    """PR #379 review (@TabeenRaoof): evaluating only the pool's single largest-deviation vertex
    missed a *different* vertex independently failing its own margin. Two top-face vertices:
    3.3mm deviation with sigma 2.0 (its own margin: 2.0 + 0.70*2.0 = 3.4mm, 3.3 does not clear
    it) and 3.0mm deviation with sigma 0.5 -- floored to D073_SIGMA_FLOOR_BOARD_MM, 0.86mm
    (margin: 2.0 + 0.70*0.86 = 2.602mm, 3.0 clears it). The pool's single worst vertex by
    deviation (3.3mm) does not itself Fail, but the per-vertex rule must still Fail from the
    second vertex -- the single-worst-vertex design (only checking 3.3mm's own margin) would
    have Passed this instead."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    first_index = _top_face_center_vertex_index(cad_vertices, sample)
    second_index = _second_top_face_vertex_index(cad_vertices, sample)
    assert first_index != second_index
    outward = _top_face_outward_direction(sample)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[first_index] = model_vertices[first_index] + 3.3 * outward
    model_vertices[second_index] = model_vertices[second_index] + 3.0 * outward

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    sigma_mm = np.full(len(sample.labels), 0.3)
    sigma_mm[first_index] = 2.0
    sigma_mm[second_index] = 0.5

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=2.0),
    )

    assert result["verdict"] == "Fail"
    assert "overall_deviation_out_of_spec" in result["reason_codes"]


def test_fail_is_not_masked_as_rescan_by_a_lower_sigma_vertex_elsewhere():
    """A second PR #379 review counterexample: the pool's single worst-deviation vertex (2.5mm,
    sigma 2.2 -- its own margin 1.0 + 0.70*2.2 = 2.54mm, 2.5 does not clear it, and
    2.2 > tolerance/k = 1.4286mm so that vertex alone would only Rescan) must not let a second
    vertex's own Fail go unseen: 2.3mm deviation with sigma 0.5, floored to 0.86mm (margin
    1.0 + 0.70*0.86 = 1.602mm, 2.3 clears it). The single-worst-vertex design evaluated only the
    first vertex and returned Rescan (high-sigma); the per-vertex rule must Fail instead."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    first_index = _top_face_center_vertex_index(cad_vertices, sample)
    second_index = _second_top_face_vertex_index(cad_vertices, sample)
    assert first_index != second_index
    outward = _top_face_outward_direction(sample)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[first_index] = model_vertices[first_index] + 2.5 * outward
    model_vertices[second_index] = model_vertices[second_index] + 2.3 * outward

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    sigma_mm = np.full(len(sample.labels), 0.3)
    sigma_mm[first_index] = 2.2
    sigma_mm[second_index] = 0.5

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=1.0),
    )

    assert result["verdict"] == "Fail"
    assert "overall_deviation_out_of_spec" in result["reason_codes"]


def test_deviation_within_sigma_margin_but_above_raw_tolerance_rescans_not_passes():
    """A third PR #379 review counterexample, needing only one vertex: a deviation that clears
    its own Fail margin (1.5mm vs. 1.0 + 0.70*0.9 = 1.63mm) is not automatically a Pass just
    because its sigma is low enough -- proposal 7.2's Pass row also requires "no Observed
    deviation above tolerance" outright, and 1.5mm is itself above the 1.0mm tolerance. The
    single-worst-vertex design's Pass branch checked only the pool's sigma against
    tolerance/k, never the deviation itself against tolerance -- it would have Passed this."""
    sample, cad_vertices, _cf, deviation_mm, deviation_measured = (
        _clean_sample_deviation()
    )
    top_vertex_index = _top_face_center_vertex_index(cad_vertices, sample)
    deviation_mm = deviation_mm.copy()
    deviation_mm[top_vertex_index] = 1.5
    sigma_mm = np.full(len(sample.labels), 0.3)
    sigma_mm[top_vertex_index] = 0.9

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=1.0),
    )

    assert result["verdict"] == "Rescan"
    assert "overall_high_sigma" in result["reason_codes"]


def test_fully_unobserved_mesh_rescans_not_seen():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation(
        patch_fraction=1.0
    )
    assert not np.any(
        np.asarray(sample.labels) == LABEL_OBSERVED
    )  # sanity on the fixture
    sigma_mm = np.full(len(sample.labels), 0.3)

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(),
    )

    assert result["verdict"] == "Rescan"
    assert result["regions"][0]["provenance"] == "unobserved"
    assert result["regions"][0]["deviation_mm"] is None
    assert result["regions"][0]["contributes_to_verdict"] is False
    assert "overall_not_seen" in result["reason_codes"]


def test_deviation_only_in_an_unobserved_region_does_not_fail():
    # The hard rule, directly: compute_deviation itself never measures an Unobserved vertex
    # (D-062), so a defect planted there is structurally invisible to the verdict -- it must
    # not turn a clean part into a Fail, or even a Rescan, on its own.
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=1, sigma_mm=0.0, patch_fraction=0.3
    )
    unobserved_mask = np.asarray(sample.labels) == LABEL_UNOBSERVED
    assert np.any(
        unobserved_mask
    )  # sanity: the patch actually left something Unobserved
    unobserved_index = int(np.flatnonzero(unobserved_mask)[0])

    transform = _ground_truth_transform(sample)
    outward = _outward_offset_direction(cad_vertices, sample, unobserved_index)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[unobserved_index] = (
        model_vertices[unobserved_index] + 5.0 * outward
    )  # a large defect, but on an Unobserved vertex

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    assert (
        deviation_measured[unobserved_index] == 0.0
    )  # confirms the plant is truly invisible
    sigma_mm = np.full(len(sample.labels), 0.3)

    # tolerance 10.0mm, not 1.0mm: D078_SIGMA_FLOOR_IN_PLANE_MM (5.278mm) needs
    # tolerance >= 5.278*0.70 = 3.6946mm to ever Pass the in-plane pool on its own -- a tight
    # tolerance would Rescan this clean part on the floor alone, unrelated to what this test is
    # actually checking (that an Unobserved-only defect can't fail or rescan a part by itself).
    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=10.0),
    )

    assert result["verdict"] == "Pass"


def test_ambiguous_orientation_rescans_every_feature():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(),
        orientation_ambiguous=True,
        ambiguity_reason_codes=["registration_orientation_ambiguous"],
    )

    assert result["verdict"] == "Rescan"
    assert result["orientation_ambiguous"] is True
    assert result["regions"][0]["contributes_to_verdict"] is False
    assert "registration_orientation_ambiguous" in result["reason_codes"]


def test_registration_failed_rescans_with_its_own_reason_code():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(),
        registration_failed=True,
    )

    assert result["verdict"] == "Rescan"
    assert "registration_failed" in result["reason_codes"]


def test_157_zero_sigma_is_clamped_to_the_floor_not_treated_as_infinite_precision():
    """Issue #157: compute_vertex_sigma can return sigma~=0 from two agreeing observations at
    n=2. Without max(sigma, sigma_floor), a tolerance of 0.1mm (well below the true floor-
    scaled margin) would incorrectly Pass on raw sigma=0 (0 <= 0.1/k always true). With the
    floor correctly applied, D073_SIGMA_FLOOR_BOARD_MM (0.86mm) is far above tolerance/k
    (0.1/0.70=0.143mm), so this must Rescan as high-sigma instead.

    A tiny (0.05mm, below the Fail margin) deviation is planted on a known top-region vertex so
    the worst-deviation vertex is deterministically out-of-plane -- D-073 restricts the floor
    (and therefore this whole trap) to that pool. Sigma is zero at that one vertex only (the
    actual #157 scenario: one specific vertex's own residuals happen to agree, not literally
    every vertex in the mesh); every other (in-plane) vertex keeps an ordinary, non-alarming
    0.3mm sigma, so the in-plane pool stays a no-op and doesn't interfere with what this test is
    actually checking."""
    sample, cad_vertices, _cf, deviation_mm, deviation_measured = (
        _clean_sample_deviation()
    )
    top_vertex_index = _top_face_center_vertex_index(cad_vertices, sample)
    deviation_mm = deviation_mm.copy()
    deviation_mm[top_vertex_index] = 0.05
    sigma_mm = np.full(len(sample.labels), 0.3)
    sigma_mm[top_vertex_index] = 0.0  # the exact #157 trap, at this one vertex

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=0.1),
    )

    assert result["verdict"] == "Rescan"
    assert "overall_high_sigma" in result["reason_codes"]


def test_157_nan_sigma_is_also_clamped_via_fmax_not_propagated():
    sample, cad_vertices, _cf, deviation_mm, deviation_measured = (
        _clean_sample_deviation()
    )
    top_vertex_index = _top_face_center_vertex_index(cad_vertices, sample)
    deviation_mm = deviation_mm.copy()
    deviation_mm[top_vertex_index] = (
        0.05  # same deterministic out-of-plane plant as above
    )
    # Fewer than 2 observations (R-10's marker) at this one vertex only -- every other
    # (in-plane) vertex keeps an ordinary 0.3mm sigma, so the in-plane pool stays a no-op
    # rather than independently Rescanning on its own, unrelated NaN-handling path.
    sigma_mm = np.full(len(sample.labels), 0.3)
    sigma_mm[top_vertex_index] = np.nan

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(tolerance_mm=5.0),
    )

    # Floored to D073_SIGMA_FLOOR_BOARD_MM (0.86mm) <= tolerance/k (5.0/0.70=7.14mm) -> Pass,
    # not an unhandled NaN propagating into a comparison and raising or silently misbehaving.
    assert result["verdict"] == "Pass"


def test_threshold_constants_match_g04_g07_and_d078():
    assert D073_SIGMA_FLOOR_BOARD_MM == 0.86
    assert D077_K == 0.70
    assert D078_SIGMA_FLOOR_IN_PLANE_MM == 5.278


def test_null_tolerance_is_refused_not_defaulted():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {"features": [{"name": "edge_1", "kind": "edge", "tolerance_mm": None}]}

    with pytest.raises(RefusedToleranceError):
        compute_verdict(
            deviation_mm,
            deviation_measured,
            sample.labels,
            sigma_mm,
            sample.vertices_mm,
            FLAT_BOARD_PLANE,
            spec,
        )


def test_missing_tolerance_key_is_also_refused():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {"features": [{"name": "edge_1", "kind": "edge"}]}

    with pytest.raises(RefusedToleranceError):
        compute_verdict(
            deviation_mm,
            deviation_measured,
            sample.labels,
            sigma_mm,
            sample.vertices_mm,
            FLAT_BOARD_PLANE,
            spec,
        )


def test_non_critical_null_tolerance_is_not_refused_but_contributes_nothing():
    """The contract's own 422 description restricts refusal to a *critical* feature
    (caught on review, @TabeenRaoof, PR #379: the first version refused on any feature's null
    tolerance, critical or not). A non-critical feature with no tolerance has nothing to judge
    it against -- listed as its own region, contributing nothing, not aborting the verdict."""
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {
        "features": [
            {
                "name": "critical_one",
                "kind": "overall",
                "tolerance_mm": 5.0,
                "critical": True,
            },
            {
                "name": "optional_one",
                "kind": "overall",
                "tolerance_mm": None,
                "critical": False,
            },
        ]
    }

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        spec,
    )

    assert result["verdict"] == "Pass"
    statuses = {r["name"]: r for r in result["regions"]}
    assert statuses["critical_one"]["contributes_to_verdict"] is True
    assert statuses["optional_one"]["contributes_to_verdict"] is False
    assert not any(code.startswith("optional_one_") for code in result["reason_codes"])


def test_missing_critical_key_defaults_to_refused_same_as_a_missing_tolerance():
    """A missing `critical` key is treated as critical (refuse on a null tolerance) -- the same
    "refuse, don't assume" discipline a missing tolerance itself already gets. Only an explicit
    `critical: false` is read as non-critical."""
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {"features": [{"name": "edge_1", "kind": "edge", "tolerance_mm": None}]}

    with pytest.raises(RefusedToleranceError):
        compute_verdict(
            deviation_mm,
            deviation_measured,
            sample.labels,
            sigma_mm,
            sample.vertices_mm,
            FLAT_BOARD_PLANE,
            spec,
        )


def test_non_list_features_is_refused_not_crashed():
    """`recon/order_store.py` stores a buyer-supplied spec as an opaque, unvalidated dict
    (I-09's own precedent) -- a malformed `features` value must be refused, not left to crash
    with an uncaught AttributeError/TypeError `getVerdict` would have to let through as a 500."""
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {"features": "not-a-list"}

    with pytest.raises(RefusedToleranceError):
        compute_verdict(
            deviation_mm,
            deviation_measured,
            sample.labels,
            sigma_mm,
            sample.vertices_mm,
            FLAT_BOARD_PLANE,
            spec,
        )


def test_non_dict_feature_entry_is_refused_not_crashed():
    sample, _cv, _cf, deviation_mm, deviation_measured = _clean_sample_deviation()
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {"features": ["not-a-dict"]}

    with pytest.raises(RefusedToleranceError):
        compute_verdict(
            deviation_mm,
            deviation_measured,
            sample.labels,
            sigma_mm,
            sample.vertices_mm,
            FLAT_BOARD_PLANE,
            spec,
        )


def test_overall_verdict_is_the_worst_across_features():
    # Both features are judged against the same global deviation (v1 has no per-feature
    # geometry) -- a loose tolerance can still Pass while a tight one Fails on that same
    # reading, and the *overall* verdict must be the worst of the two (Fail).
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    transform = _ground_truth_transform(sample)
    vertex_index = _top_face_center_vertex_index(cad_vertices, sample)
    outward = _outward_offset_direction(cad_vertices, sample, vertex_index)
    model_vertices = sample.vertices_mm.copy()
    model_vertices[vertex_index] = model_vertices[vertex_index] + 1.5 * outward

    deviation_mm, deviation_measured = compute_deviation(
        (model_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    sigma_mm = np.full(len(sample.labels), 0.3)
    spec = {
        "features": [
            {"name": "passer", "kind": "overall", "tolerance_mm": 5.0},
            {"name": "failer", "kind": "overall", "tolerance_mm": 0.1},
        ]
    }

    result = compute_verdict(
        deviation_mm,
        deviation_measured,
        sample.labels,
        sigma_mm,
        model_vertices,
        FLAT_BOARD_PLANE,
        spec,
    )

    assert result["verdict"] == "Fail"
    statuses = {r["name"]: r["contributes_to_verdict"] for r in result["regions"]}
    assert statuses["passer"] is True
    assert statuses["failer"] is True
    assert "failer_deviation_out_of_spec" in result["reason_codes"]
    assert "passer_deviation_out_of_spec" not in result["reason_codes"]
