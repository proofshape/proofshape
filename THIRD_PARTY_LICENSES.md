# Third-party components

Everything this project builds on, its licence, and two questions that matter for a public
repository: whether we redistribute its weights, and whether its code is vendored here.

**The rule for both is no.** Weights are downloaded at setup and gitignored; code is pinned as a
dependency, never copied in. This file is the record that we checked.

Verify each licence at M0 against the version actually pinned — licences change between releases,
and this table is a starting point, not a substitute for reading them.

## Models

| Component | Purpose | Licence | Weights redistributed here? | Code vendored here? |
|---|---|---|---|---|
| **VGGT** | Primary reconstruction backbone (§6.3) | Non-commercial research licence — **verify at M0** | **No** — downloaded at setup, gitignored | No — pinned dependency |
| **MASt3R** | Reconstruction fallback | Non-commercial research licence — **verify at M0** | **No** | No — pinned dependency |
| **COLMAP** | Classical reconstruction fallback | BSD-style (new BSD) | Not applicable — no learned weights | No — installed binary or pinned package |
| **SAM 2** | Optional segmentation masks (§6.4) | Apache-2.0 (code); check the weights licence separately | **No** | No — pinned dependency |
| **Shape completion model** | Generative completion (§6.6) | **Not yet selected** — licence to be recorded here when it is | **No** | No |
| **Gemini Flash-Lite** | VLM supervision and LLM spec parsing (§6.5, §7.4) | Hosted API, governed by provider terms | Not applicable — no local weights | No |
| **Qwen2.5-VL-3B** | Documented local VLM fallback (extension) | Apache-2.0 — verify for the specific checkpoint | **No** | No |

## Libraries

| Component | Purpose | Licence | Vendored here? |
|---|---|---|---|
| **Open3D** | TSDF fusion, meshing, registration, raycasting | MIT | No — pinned dependency |
| **trimesh** | CAD input/output, mesh operations | MIT | No — pinned dependency |
| **manifold3d** | Boolean operations for defect synthesis | Apache-2.0 | No — pinned dependency |
| **OpenCV** | ChArUco detection, pose, QR | Apache-2.0 (4.5+) | No — pinned dependency |
| **PyTorch** | Model runtime | BSD-style | No — pinned dependency |
| **FastAPI / uvicorn** | Service layer | MIT / BSD | No — pinned dependency |
| **three.js** | Browser 3D viewer | MIT | No — pinned dependency |

## What this means in practice

**Non-commercial weights constrain deployment, not publication.** They are fine for coursework and
for a public repository that does not redistribute them. They would block commercialising a
deployed system without swapping the backbone, which is one config flag (D-009).

**If a licence turns out to prohibit something we assumed,** record it in `docs/decisions.md` rather
than working around it silently.
