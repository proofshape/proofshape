import { resolveAuthToken, resolveBaseUrl } from "./config.js";
import { mountModelViewer } from "./model-viewer.js";
import { mountAppShell } from "./shell.js";

const shellRoot = document.getElementById("shell-app");
if (!shellRoot) {
  throw new Error("main: #shell-app element not found in index.html");
}

// C-11 device-check helper: the contract stub serves its canned GLB at this route without
// needing a finished session. Keeping this behind an explicit query parameter makes the real
// app flow unchanged while giving a phone a direct WebGL/orbit/pinch check with no devtools.
const viewerDemo = new URLSearchParams(window.location.search).get("viewer-demo");
if (viewerDemo === "1" || viewerDemo === "c12") {
  mountModelViewer(shellRoot, {
    status: "complete",
    glb_url:
      viewerDemo === "c12"
        ? "/v1/sessions/c12-provenance-check/model.glb"
        : "/v1/sessions/c11-device-check/model.glb",
    metric: false,
  });
} else {
  // C-13: VITE_RECON_BASE_URL/VITE_RECON_TOKEN point this build at the deployed service instead
  // of the local contract stub; both are unset in every existing dev/test run, so baseUrl stays
  // "" and authToken stays undefined -- today's behaviour is the default.
  const baseUrl = resolveBaseUrl(import.meta.env.VITE_RECON_BASE_URL);
  const authToken = resolveAuthToken(import.meta.env.VITE_RECON_TOKEN);
  // Same reasoning as viewer-demo above: a capture link pre-fills the order code so a supplier
  // isn't typing one in.
  const initialOrderCode = new URLSearchParams(window.location.search).get("order") ?? undefined;

  // Return value not captured: nothing consumes it yet until C-05 exists to read/write it.
  // mountAppShell still returns it for tests and for whenever C-05 is wired in here. Keys are
  // omitted rather than set to `undefined` -- tsconfig's exactOptionalPropertyTypes treats those
  // as different things.
  mountAppShell(shellRoot, {
    baseUrl,
    ...(authToken !== undefined ? { authToken } : {}),
    ...(initialOrderCode !== undefined ? { initialOrderCode } : {}),
  });
}
