// C-13: resolves the build-time config (`import.meta.env.VITE_RECON_*`, see vite-env.d.ts) that
// points capture/ at the deployed service instead of the local contract stub. Pure functions so
// main.ts's own wiring stays trivial and these two rules are unit-tested directly, per this
// story's own Tests section.

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
