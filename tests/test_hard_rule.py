"""I-08: the hard rule (proposal 7.2; AGENTS.md) -- "geometry that was not observed never
fails a part and never contributes to a Pass; it can only trigger a Rescan" -- made structural
on `compute_verdict`'s own live output, across every provenance that must never contribute
(Unobserved, Inferred), not just over the committed contract examples R-29's own
`tests/test_contract.py` check already covers.

Confirmed by reading the code, not assumed: `inspect/inspection/deviation.py`'s `is_observed =
model_labels == LABEL_OBSERVED` already excludes any non-Observed label (Unobserved or Inferred
alike) from measurement by construction, and `inspect/inspection/verdict.py`'s own
`observed_measured` mask does the same before either pool is built -- this module's job is to
pin that down with a direct, reusable check on real `compute_verdict` results, not to add a new
rule.

**D-080 (v1 has no per-feature CAD-region binding, so the hard rule's "a critical region is
Unobserved" row can only be expressed mesh-wide):** `compute_verdict` only ever emits
`provenance: "unobserved"` on a region when the *whole* session has no Observed, measured data
at all (`has_observed_data` is `False`) -- as soon as any exists anywhere, every region comes
back `provenance: "observed"`, regardless of whether the mesh also contains Unobserved/Inferred
patches (caught on review, @mbj1994, PR #384: a first version of the two partial-patch tests
below checked `provenance`/`contributes_to_verdict` directly and passed vacuously for exactly
this reason -- there were no non-`"observed"` regions in the result to inspect). The real,
non-vacuous proof for a *partial* patch is numeric: `compute_verdict` run on the same sample
with and without a hidden defect on non-Observed geometry must return identical results, since
D-062/I-06 guarantee that geometry is never measured either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from inspection import cad, synthetic
from inspection.deviation import compute_deviation
from inspection.verdict import compute_verdict

from recon.part_segmentation import BoardSurface
from recon.provenance import LABEL_INFERRED, LABEL_OBSERVED, LABEL_UNOBSERVED

ROOT = Path(__file__).resolve().parents[1]
G01_CUBE = ROOT / "fixtures" / "reference_cad" / "g01_calibration_cube_30mm.stl"

# Same flat-board fixture test_verdict.py/test_deviation.py use, duplicated rather than shared
# across test files (this project's own established convention).
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


def _spec(name: str = "overall", tolerance_mm: float = 5.0) -> dict:
    return {
        "features": [{"name": name, "kind": "overall", "tolerance_mm": tolerance_mm}]
    }


def _non_observed_contributing_regions(result: dict) -> list[str]:
    """Mirrors `tests/test_contract.py`'s own helper of the same name, applied to
    `compute_verdict`'s live return value instead of a committed JSON example -- duplicated, not
    imported, same convention as this file's other fixtures. D-005, the hard rule, made
    structural: a region whose provenance isn't `observed` must never have
    `contributes_to_verdict: true`."""
    return [
        region["name"]
        for region in result.get("regions", [])
        if region["provenance"] != "observed"
        and region["contributes_to_verdict"] is not False
    ]


def test_hard_rule_holds_with_an_unobserved_patch_and_a_hidden_defect():
    """Same construction as `test_verdict.py`'s own
    `test_deviation_only_in_an_unobserved_region_does_not_fail`: a defect planted on an
    Unobserved vertex, with the rest of the mesh clean and fully Observed. That test already
    owns the specific verdict value (Pass). This one proves the *generic* invariant instead --
    not by checking `provenance`/`contributes_to_verdict` on the result (vacuous here: with any
    Observed data present, every region comes back `provenance: "observed"` regardless of the
    Unobserved patch, per D-080), but by comparing `compute_verdict`'s own output with and
    without the hidden defect and requiring it to be identical."""
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
    defective_vertices = sample.vertices_mm.copy()
    defective_vertices[unobserved_index] = (
        defective_vertices[unobserved_index] + 5.0 * outward
    )  # a large defect, but on an Unobserved vertex

    _, defective_deviation_measured = compute_deviation(
        (defective_vertices, sample.faces, sample.labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    assert (
        defective_deviation_measured[unobserved_index] == 0.0
    )  # confirms the plant is truly invisible
    sigma_mm = np.full(len(sample.labels), 0.3)

    def _verdict(model_vertices: np.ndarray) -> dict:
        deviation_mm, deviation_measured = compute_deviation(
            (model_vertices, sample.faces, sample.labels),
            (cad_vertices, cad_faces),
            transform,
            FLAT_BOARD,
        )
        return compute_verdict(
            deviation_mm,
            deviation_measured,
            sample.labels,
            sigma_mm,
            model_vertices,
            FLAT_BOARD_PLANE,
            _spec(tolerance_mm=10.0),
        )

    clean_result = _verdict(sample.vertices_mm)
    defective_result = _verdict(defective_vertices)

    assert _non_observed_contributing_regions(defective_result) == []
    assert defective_result == clean_result


def test_hard_rule_holds_with_inferred_vertices_stapled_in():
    """A-04's Inferred label, run all the way through both I-06 and I-07 -- today only
    `test_deviation.py`'s own `test_compute_deviation_leaves_inferred_vertices_unmeasured` runs
    Inferred through `compute_deviation`; nothing runs it through `compute_verdict` yet. A clean,
    fully-Observed sample has one vertex relabeled Inferred and given a large (12mm) planted
    displacement; as in the Unobserved-patch test above, the generic invariant is proved by
    comparing `compute_verdict` with and without that treatment, not by inspecting
    `provenance`/`contributes_to_verdict` (vacuous here for the same D-080 reason)."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=0.0
    )
    assert np.all(np.asarray(sample.labels) == LABEL_OBSERVED)  # sanity on the fixture
    transform = _ground_truth_transform(sample)

    inferred_index = 0
    inferred_labels = sample.labels.copy()
    inferred_labels[inferred_index] = LABEL_INFERRED
    outward = _outward_offset_direction(cad_vertices, sample, inferred_index)
    defective_vertices = sample.vertices_mm.copy()
    defective_vertices[inferred_index] = (
        defective_vertices[inferred_index] + 12.0 * outward
    )

    _, defective_deviation_measured = compute_deviation(
        (defective_vertices, sample.faces, inferred_labels),
        (cad_vertices, cad_faces),
        transform,
        FLAT_BOARD,
    )
    assert (
        defective_deviation_measured[inferred_index] == 0.0
    )  # confirms the plant is truly invisible

    def _verdict(model_vertices: np.ndarray, labels: np.ndarray) -> dict:
        deviation_mm, deviation_measured = compute_deviation(
            (model_vertices, sample.faces, labels),
            (cad_vertices, cad_faces),
            transform,
            FLAT_BOARD,
        )
        sigma_mm = np.full(len(labels), 0.3)
        return compute_verdict(
            deviation_mm,
            deviation_measured,
            labels,
            sigma_mm,
            model_vertices,
            FLAT_BOARD_PLANE,
            _spec(tolerance_mm=5.0),
        )

    clean_result = _verdict(sample.vertices_mm, sample.labels)
    defective_result = _verdict(defective_vertices, inferred_labels)

    assert _non_observed_contributing_regions(defective_result) == []
    assert defective_result == clean_result
    # The rest of the mesh is clean and fully Observed -- the 12mm planted defect on the one
    # Inferred vertex must not have dragged the verdict down at all.
    assert defective_result["verdict"] == "Pass"


def test_hard_rule_holds_with_a_fully_unobserved_mesh():
    """The mesh-level case -- the real, v1-expressible form of proposal 7.2's "a critical region
    is Unobserved" row (D-080: this, not a per-feature Rescan on a hidden defect, is the right
    granularity for v1). Unlike the two tests above, `provenance: "unobserved"` really is
    emitted here, so the `_non_observed_contributing_regions` check is not vacuous -- this
    duplicates `test_verdict.py`'s own `test_fully_unobserved_mesh_rescans_not_seen` (same
    construction, same verdict assertion) because the hard rule, not that specific test, is
    this module's own subject."""
    cad_vertices, cad_faces, _units = cad.load_mesh(G01_CUBE)
    sample = synthetic.generate_synthetic_sample(
        cad_vertices, cad_faces, seed=0, sigma_mm=0.0, patch_fraction=1.0
    )
    assert not np.any(
        np.asarray(sample.labels) == LABEL_OBSERVED
    )  # sanity on the fixture
    transform = _ground_truth_transform(sample)
    deviation_mm, deviation_measured = compute_deviation(
        (sample.vertices_mm, sample.faces, sample.labels),
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
        sample.vertices_mm,
        FLAT_BOARD_PLANE,
        _spec(),
    )

    assert _non_observed_contributing_regions(result) == []
    assert result["verdict"] == "Rescan"
