import { mountModelViewer } from "./model-viewer.js";
import { mountAppShell } from "./shell.js";

const shellRoot = document.getElementById("shell-app");
if (!shellRoot) {
  throw new Error("main: #shell-app element not found in index.html");
}

// C-11 device-check helper: the contract stub serves its canned GLB at this route without
// needing a finished session. Keeping this behind an explicit query parameter makes the real
// app flow unchanged while giving a phone a direct WebGL/orbit/pinch check with no devtools.
const viewerDemo = new URLSearchParams(window.location.search).get("viewer-demo") === "1";
if (viewerDemo) {
  mountModelViewer(shellRoot, {
    status: "complete",
    glb_url: "/v1/sessions/c11-device-check/model.glb",
    metric: false,
  });
} else {
  // Return value not captured: nothing consumes it yet until C-05 exists to read/write it.
  // mountAppShell still returns it for tests and for whenever C-05 is wired in here.
  mountAppShell(shellRoot, { baseUrl: "" });
}
