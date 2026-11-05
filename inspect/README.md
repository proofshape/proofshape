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
