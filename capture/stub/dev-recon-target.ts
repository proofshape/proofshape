import type { ProxyOptions } from "vite";

// C-13 / D-049: which reconstruction service the dev server's `/v1` routes reach.
//
// By default that is the in-process contract stub (C-01/C-10), so `npm run dev` still needs no
// second process. For C-13's end-to-end check the phone instead has to reach the real
// `recon.service` running natively on the M2 Max (D-010). The phone loads this page over HTTPS
// (Safari only allows the camera in a secure context), so it cannot call `http://<mac>:8000`
// directly -- the browser blocks that as mixed content -- and a second self-signed certificate on
// another port would be one more thing to accept on the phone. Proxying `/v1` through this same
// dev server keeps every request same-origin: no CORS, no second certificate, no token.
//
// Deliberately not a VITE_-prefixed variable: it configures the dev server only and must never be
// compiled into the client bundle.
export const RECON_PROXY_ENV = "PROOFSHAPE_RECON_PROXY";

export type DevReconTarget = { mode: "stub" } | { mode: "proxy"; target: string };

export function resolveDevReconTarget(env: Record<string, string | undefined>): DevReconTarget {
  const raw = env[RECON_PROXY_ENV]?.trim() ?? "";
  if (raw === "") {
    return { mode: "stub" };
  }

  // A malformed value fails the dev server at start-up rather than quietly falling back to the
  // stub: a phone check that silently ran against canned responses would look like a pass.
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(
      `${RECON_PROXY_ENV}=${JSON.stringify(raw)} is not a URL. ` +
        "Expected the recon service's origin, e.g. http://127.0.0.1:8000.",
    );
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(
      `${RECON_PROXY_ENV} must be an http:// or https:// URL, got ${JSON.stringify(raw)}.`,
    );
  }
  // The proxy appends the request path (/v1/...) to the target, so a path here would silently
  // become a prefix the service has no routes for. Accept an origin only.
  if ((url.pathname !== "/" && url.pathname !== "") || url.search !== "" || url.hash !== "") {
    throw new Error(
      `${RECON_PROXY_ENV} must be an origin only (scheme, host, port), got ${JSON.stringify(raw)}.`,
    );
  }
  return { mode: "proxy", target: url.origin };
}

export function devReconProxy(target: string): Record<string, ProxyOptions> {
  return {
    // xfwd adds X-Forwarded-Host/-Proto, which recon/service.py's public_model_url already reads,
    // so the glb_url it returns is https://<this dev server>/v1/sessions/<id>/model.glb -- a URL
    // the phone can load through this same proxy -- rather than the service's own
    // http://127.0.0.1:8000, which on the phone would point at the phone itself.
    "/v1": { target, xfwd: true },
  };
}
