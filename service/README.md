# service — API, sessions, and the order store

The FastAPI application: session creation, the gated capture link, frame upload, the endpoints
described in `../contracts/openapi.yaml`, and the per-order store (a folder and a JSON file per
purchase order this semester).

**Why separate from `recon/`.** Reconstruction is an algorithm; this is a web service. Keeping them
apart means the reconstruction code can be run from a script, a notebook or a test without starting
a server, and the service can be tested without a GPU.

**Tests:** `service/tests/`. Endpoint tests use a fake reconstruction backend.

**Not yet populated.** Work begins with story R-14; see `../stories/README.md`.
