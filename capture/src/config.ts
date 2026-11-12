// C-13: resolves the build-time config (`import.meta.env.VITE_RECON_*`, see vite-env.d.ts) that
// points capture/ at the deployed service instead of the local contract stub. Pure functions so
// main.ts's own wiring stays trivial and these rules are unit-tested directly.

// Stripped so api.ts's `${baseUrl}/v1/sessions` never double-slashes if the configured value
// has a trailing slash (an easy copy-paste mistake with a Lightning URL from a run summary).
export function resolveBaseUrl(raw: string | undefined): string {
  const value = raw ?? "";
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

// "" (unset/blank) means "no token" -- api.ts only attaches an Authorization header when this
// resolves to a real value, so the contract stub (which expects no auth) keeps working.
export function resolveAuthToken(raw: string | undefined): string | undefined {
  const trimmed = (raw ?? "").trim();
  return trimmed === "" ? undefined : trimmed;
}

export const HOSTED_RECON_UNAVAILABLE_MESSAGE =
  "Reconstruction service is not configured for this hosted build yet. Capture is unavailable until the project team enables the public service.";

// C-21: a local development build with no base URL intentionally talks to Vite's contract stub.
// A production Pages build with no base URL is different: O-005 deliberately leaves the paid
// service disconnected, so mounting the shell would POST /v1/sessions back to GitHub Pages and
// expose a raw 405. Stop before any request instead.
export function hostedReconUnavailable(baseUrl: string, isDevelopment: boolean): boolean {
  return !isDevelopment && baseUrl === "";
}

export type ViewerDemo = "1" | "c12" | "c14";
export type ViewerDemoMode = ViewerDemo | "unavailable" | undefined;

// C-21: viewer-demo relies on the Vite dev stub, so it must never try to run from a hosted
// production build. Unknown values are ignored and fall through to the normal capture app.
export function resolveViewerDemo(raw: string | null, isDevelopment: boolean): ViewerDemoMode {
  if (raw !== "1" && raw !== "c12" && raw !== "c14") {
    return undefined;
  }
  return isDevelopment ? raw : "unavailable";
}
