import {
  resolveAuthToken,
  resolveBaseUrl,
  resolveViewerDemo,
} from "./config.js";
import { mountModelViewer } from "./model-viewer.js";
import { mountAppShell } from "./shell.js";

const shellRoot = document.getElementById("shell-app");
if (!shellRoot) {
  throw new Error("main: #shell-app element not found in index.html");
}

// C-11 device-check helper: the contract stub serves its canned GLB at this route without
// needing a finished session. C-21 makes this explicitly development-only because a hosted
// production build has no dev stub behind /v1.
const viewerDemo = resolveViewerDemo(
  new URLSearchParams(window.location.search).get("viewer-demo"),
  import.meta.env.DEV,
);

if (viewerDemo === "unavailable") {
  shellRoot.innerHTML =
    '<p role="status">Viewer demo is available only in development builds. Open the normal capture link instead.</p>';
} else if (viewerDemo === "1" || viewerDemo === "c12" || viewerDemo === "c14") {
  mountModelViewer(shellRoot, {
    status: "complete",
    glb_url:
      viewerDemo === "c12"
        ? "/v1/sessions/c12-provenance-check/model.glb"
        : viewerDemo === "c14"
          ? "/v1/sessions/c14-deviation-check/model.glb"
          : "/v1/sessions/c11-device-check/model.glb",
    metric: false,
  });
} else {
  // C-13: VITE_RECON_BASE_URL/VITE_RECON_TOKEN point this build at the deployed service instead
  // of the local contract stub. C-21's Pages workflow leaves both blank until O-005 is settled,
  // so publishing the static app cannot accidentally expose a token or start paid GPU work.
  const baseUrl = resolveBaseUrl(import.meta.env.VITE_RECON_BASE_URL);
  const authToken = resolveAuthToken(import.meta.env.VITE_RECON_TOKEN);
  // A capture link pre-fills the order code so a supplier is not typing one in. This remains
  // relative to the current page, so the same ?order= link works at / locally and under the
  // /proofshape/ GitHub Pages base path.
  const initialOrderCode = new URLSearchParams(window.location.search).get("order") ?? undefined;

  mountAppShell(shellRoot, {
    baseUrl,
    ...(authToken !== undefined ? { authToken } : {}),
    ...(initialOrderCode !== undefined ? { initialOrderCode } : {}),
  });
}
