# inspect

CAD registration, verdict logic, and ground-truth tooling — the inspection lane.

This directory itself is **not** registered as an installed Python package, unlike `recon/`. Its
name collides with Python's own standard-library `inspect` module; installing it as a top-level
package would shadow that module for the whole environment. See `docs/decisions.md` D-030.
D-053 settles the import name D-030 left open: `inspect/` stays a bare directory with no
`__init__.py`, and the importable package is `inspection`, living at `inspect/inspection/` and
registered in `pyproject.toml`. I-01 built it: `inspection.synthetic` generates the synthetic
CAD-derived meshes the rest of this lane tests against (`tests/test_synthetic.py`).


## I-03 · global registration

`inspection.registration.global_register(model, cad)` produces multiple rigid CAD-to-model
candidate poses for downstream ambiguity handling. CAD is first put into the board convention
with the same proper 180° x rotation used by I-01, then Open3D FPFH + RANSAC supplies a global
candidate while yaw/xy enumeration preserves alternative board-compatible poses. Scale is fixed
at 1 in every candidate. The best candidate separately reports `free_scale_residual` from fully
Observed model evidence only: Observed model samples are registered to a dense CAD target with a
rigid ICP followed by similarity ICP, then the inverse fitted scale is converted back to the
D-039 sign convention (+0.03 means the model is 3% larger). Hidden geometry cannot pull the fit;
missing or planar-only Observed evidence reports `None` rather than a false scale result, and so
does a reading beyond ±25%, which is a failed fit rather than a scale error (I-05).

Defaults scale with part extent: voxel 3%, normal radius 2.5 voxels, FPFH radius 5 voxels and
correspondence distance 2.5 voxels. These are registration parameters, not inspection
tolerances.


## I-04 · robust refinement

`inspection.registration.refine(candidate, model, cad)` tightens one I-03 candidate with Open3D
point-to-plane ICP under a Tukey loss (D-008). A residual beyond Tukey's `k` gets zero weight,
so a bump, dent or shifted hole stops pulling the pose once the rest of the part is aligned. A
least-squares fit would explain the defect away by moving the whole part toward it.

- **Observed only.** The fit uses fully Observed model faces, so Unobserved or Inferred geometry
  can't move the pose that deviations are measured against (D-005). A model without provenance,
  or with no fully Observed face, raises.
- **Observed model → CAD.** The target is the sampled CAD surface with its own triangle normals,
  which point-to-plane needs. Every Observed point has a CAD counterpart.
- **A full rigid 6-DoF pose**, returned as a `CandidatePose` (CAD → model, like I-03's), with the
  refined fit's `fitness`, `inlier_rmse` and `correspondence_count`.
- **`free_scale_residual` is recomputed from the refined pose** with I-03's two-stage similarity
  ICP, with the same planar-only gate and `None` cases. It isn't solved on the fit's frozen final
  pairs, which underestimate scale.
- **No match, no result.** A candidate with fewer than three correspondences raises; Open3D's
  `inlier_rmse = 0` for "nothing matched" would otherwise look like a perfect fit.

Defaults use the CAD's extent: `k` = 1 voxel (3%), correspondence distance 2.5 voxels, and 100
ICP iterations. On I-01 output, `k` from 0.5 to 2 voxels refined s-04 and s-05 to the same
0.04–0.09 mm surface error. These are algorithm parameters, not tolerances.


## I-05 · orientation ambiguity

`inspection.registration.check_ambiguity(candidates, model, cad)` refines every I-03 candidate
with `refine` and decides whether the part's orientation is ambiguous (D-008). On a near-symmetric
part the feature that breaks the symmetry is often the defect, so a fit must not quietly pick the
alignment that explains it away.

- **Every candidate, the same samples.** Each candidate is refined with the same seed, so their
  RMSEs compare like for like. Refining all of them also corrects a wrong I-03 pick: on a plate
  whose only asymmetry is a hole, I-03's best can be the 180° mirror.
- **Merged only when the transforms agree** (I-03's 2° and 1% of CAD extent, 0.25 mm floor).
  Poses that are equivalent only under the CAD's symmetry stay separate: for the printed part
  they are different poses, and the difference is where a defect would be.
- **Ranked fitness first, then RMSE.** The session is ambiguous when the runner-up's inlier RMSE
  is within 20% of the best's (D-061, which raised D-008's 10%). One surviving pose isn't
  ambiguous.
- **The result** is an `AmbiguityCheck`: `orientation_ambiguous`, the `best` pose to measure
  against, the `runner_up`, `rmse_ratio`, every `refined` pose, the `failed` candidates with their
  reasons, and `reason_codes` (`registration_orientation_ambiguous` when ambiguous). I-07 turns it
  into the verdict's `orientation_ambiguous` and reason code.
- **Failures are listed, not hidden.** A candidate that can't be refined goes into `failed`;
  if none can be, it raises. A model without provenance is refused up front.

On I-01 output with its 20% hidden patch: s-05 (four-fold) is ambiguous and s-04 (scalene) isn't,
every time; a plate whose symmetry-breaking hole is missing is caught 12/12 at 20%, against 9/12
at D-008's 10%. D-061 has the full table and its narrow margins.


## I-06 · deviation computation

`inspection.deviation.compute_deviation(model, cad, cad_to_model_transform, board_surface)` is
the signed point-to-mesh distance from each Observed model vertex to the CAD (D-062, proposal
7.2), using I-05's `AmbiguityCheck.best.transformation` as the pose to measure against. Returns
`(deviation_mm, deviation_measured)`, one value per model vertex, in the model's own vertex order.

- **Observed, above the board cut, inside the registered footprint.** `deviation_measured` is
  1.0 only where all three hold, 0.0 everywhere else; `deviation_mm` is 0.0 wherever
  `deviation_measured` is 0.0 (D-062: "I-06 measures Observed vertices only" — "that excludes any
  it leaves out, such as surface outside the registered part"). **What the two gates actually do
  on real data, re-reviewed on #309's second round rather than assumed from the logic alone:**
  on real golden s-04/s-05, neither gate excluded anything (100% of Observed vertices measured
  on both). The board cut reuses `board_surface`'s own already-resolved `height_threshold_mm`
  (fusion's real D-063 cut for that board), but on a real served mesh this excludes essentially
  nothing — fusion's own cut already removed that surface before this function ever sees it, so
  its real job is defending against a degenerate input, not filtering real board skin. The
  footprint gate (the posed CAD's own xy bounding box plus `DEFAULT_FOOTPRINT_MARGIN_MM`, 10 mm)
  is what does real work, but only beyond the margin — 31% of s-04's measured vertices and 15%
  of s-05's sit outside the CAD's own hull but within the margin (mostly base flare), so the
  margin is doing real work keeping legitimate geometry in, but board skin sitting within that
  same margin of the hull (#292's s-02 case) would also still be measured today. A declared,
  recorded gap (`stories/I/I-06.md`'s notes), not something either gate actually closes — #292's
  or I-07's call. The margin itself is a declared parameter: G-05's defects grow upward on the
  top face, not outward in xy, so they can't justify this number the way an earlier version of
  this note claimed; real data just shows 10 mm isn't obviously too tight for s-04's own base
  flare.
- **Sign convention.** Positive means the part has extra material there, negative means material
  is missing. This is Open3D's own `RaycastingScene.compute_signed_distance` convention
  (negative inside a closed mesh, positive outside) applied directly once the CAD is posed into
  the model frame — a model vertex outside the CAD solid *is* extra material, one inside *is*
  missing material, so no sign flip is needed. Checked against a known planted offset on I-01
  output, not just read from Open3D's docs (`tests/test_deviation.py`). The CAD must be
  watertight (`recon.provenance._mesh_is_watertight`) — Open3D's inside/outside decision is by
  ray parity, which is only meaningful for a closed mesh, so an open CAD raises rather than
  silently misreading its own sign.
- **Flexible inputs.** `model` and `cad` accept anything `_mesh_arrays` does — I-01's
  `SyntheticSample`, `inspection.cad.load_mesh`'s tuples, or raw `(vertices_mm, faces, labels)`.
  `model` without provenance labels raises: there would be no way to find the Observed vertices.

`inspection.deviation.write_deviation_glb(existing_glb_path, cad, cad_to_model_transform,
board_surface, output_path)` reads a served GLB's COLOR_0 (`recon.provenance.colors_to_labels`,
R-09's own reader) to find the Observed vertices, computes the deviation, and writes a new GLB
beside the original via `recon.mesh_cleanup.write_glb`'s existing `extra_vertex_attributes`
parameter — COLOR_0 carried through unchanged, `_DEVIATION_MM`/`_DEVIATION_MEASURED` added.

`inspection.deviation.publish_deviation_glb(store, session_id, cad, cad_to_model_transform,
board_surface)` is D-062's third criterion, implemented against a real
`recon.service.SessionStore` (review on #309): it calls `write_deviation_glb`, then sets
`record["reconstruction"]["glb_path"]` to the new path and calls `store.save(record)` — reusing
`SessionStore`'s own existing atomicity (a temp `session.json`, then `.replace()`) rather than
inventing a second mechanism, so `glb_url` never changes and no viewer reads a half-written
file. `recon.service` is imported lazily inside this function, not at module level, so a caller
who only wants the computation doesn't pay for FastAPI/pydantic. Not called from the real
pipeline yet: no session has a per-order CAD to measure against until R-34 (order store) exists;
until then this module is a library function, a GLB writer, and a tested session-flip helper,
not a wired-up pipeline stage.

Round-tripping the two new attributes needs `recon.mesh_cleanup.read_glb_extra_attributes`
(I-06): `read_glb` itself keeps its existing 3-tuple return (`vertices, faces, colors`) rather
than growing a fourth element, since six existing callers unpack it positionally. It checks each
accessor's `componentType`/`type` and honours `accessor.byteOffset`, so a foreign GLB that
happens to reuse one of these attribute names is refused rather than silently misread.
