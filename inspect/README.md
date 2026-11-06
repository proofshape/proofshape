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
missing or planar-only Observed evidence reports `None` rather than a false scale result.

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
