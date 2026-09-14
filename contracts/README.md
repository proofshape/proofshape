# contracts

The frozen interface between `capture/` and `recon/`: `openapi.yaml`, describing session
creation, frame upload, session finish, reconstruction result, and verdict result. One example
request and response is committed per endpoint under `examples/`.

Frozen at v1 as of F-03. Tier 1 (session lifecycle, frame upload, reconstruction) is specified in
full; the verdict is an **envelope only** — the four values, per-region provenance and reason
codes that existing decisions already settle. Tolerance-table internals, `k` and `σ_floor` are
deliberately left as extension points, marked `x-proofshape-extends-at` in the spec, for the
inspection stories (Phase 3, not yet written) once the October calibration study has real numbers.

**Changing this file needs both other team members' approval, at a sprint boundary only** — see
`CONTRIBUTING.md` and D-031.
