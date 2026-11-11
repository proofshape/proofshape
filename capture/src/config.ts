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
