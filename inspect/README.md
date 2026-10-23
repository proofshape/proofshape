# inspect

CAD registration, verdict logic, and ground-truth tooling — the inspection lane.

This directory itself is **not** registered as an installed Python package, unlike `recon/`. Its
name collides with Python's own standard-library `inspect` module; installing it as a top-level
package would shadow that module for the whole environment. See `docs/decisions.md` D-030.
D-053 settles the import name D-030 left open: `inspect/` stays a bare directory with no
`__init__.py`, and the importable package is `inspection`, living at `inspect/inspection/` and
registered in `pyproject.toml`. I-01 built it: `inspection.synthetic` generates the synthetic
CAD-derived meshes the rest of this lane tests against (`tests/test_synthetic.py`).
