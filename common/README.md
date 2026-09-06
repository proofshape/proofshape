# common — shared types and schemas

Types used by more than one module: the mesh and provenance representation, the per-vertex
uncertainty record, the parsed inspection spec, the verdict enum, and the request and response
models generated from `../contracts/openapi.yaml`.

**Why it exists.** `recon/`, `inspect/`, `ai/` and `service/` all handle the same objects. Without a
shared module you get either duplicated definitions that drift apart, or an import cycle when one
module reaches into another for a type.

**Rule:** `common/` imports from nothing else in this repository. If something here needs to import
from `recon/` or `inspect/`, it does not belong here.

**Tests:** `common/tests/`.

**Not yet populated.**
