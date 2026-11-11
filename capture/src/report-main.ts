import { resolveAuthToken, resolveBaseUrl } from "./config.js";
import { mountReportPage } from "./report.js";

const root = document.getElementById("report-app");
if (!root) {
  throw new Error("report-main: #report-app element not found in report.html");
}

const sessionId = new URLSearchParams(window.location.search).get("session");
if (!sessionId) {
  root.textContent = "A session id is required. Open the report from a completed capture.";
} else {
  const baseUrl = resolveBaseUrl(import.meta.env.VITE_RECON_BASE_URL);
  const authToken = resolveAuthToken(import.meta.env.VITE_RECON_TOKEN);
  mountReportPage(root, {
    baseUrl,
    sessionId,
    ...(authToken !== undefined ? { authToken } : {}),
  });
}
