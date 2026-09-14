# inspect

CAD registration, verdict logic, and ground-truth tooling — the inspection lane.

Nothing runs yet, and this directory is **not** registered as an installed Python package,
unlike `recon/`. Its name collides with Python's own standard-library `inspect` module;
installing it as a top-level package would shadow that module for the whole environment. See
`docs/decisions.md` D-030. The actual import name for this lane's code (a prefixed name, a
namespace package, or something else) is a decision for whoever starts I-01, once there is real
code to hang it off.
