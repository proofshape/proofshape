# inspect

CAD registration, verdict logic, and ground-truth tooling — the inspection lane.

Nothing runs yet, and this directory is **not** registered as an installed Python package,
unlike `recon/`. Its name collides with Python's own standard-library `inspect` module;
installing it as a top-level package would shadow that module for the whole environment. See
`docs/decisions.md` D-030. D-053 settles the import name D-030 left open: this directory stays
`inspect/` with no `__init__.py`, and the importable package is `inspection`, living at
`inspect/inspection/` and registered in `pyproject.toml`. I-01 builds it; nothing runs until it
merges.
