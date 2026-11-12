# capture

The progressive web app (TypeScript) — camera capture, local frame gate, gyro overlay,
next-view scoring, the model viewer, and the report page. Talks to `recon/` over the interface
in `../contracts/`.

C-01 is the first story built here: a minimal page that proves `contracts/openapi.yaml`'s
session lifecycle (`createSession`, `uploadFrame`, `finishSession`) actually works end to end —
no live camera, no real gyro reading, no UI polish. Those arrive with C-02, C-03 and C-04.

## Running it

```
npm install
npm run dev          # serves the page at http://localhost:5173, plus a contract stub (see below)
```

## Testing on a real device (C-02)

Safari refuses `getUserMedia` outside a secure context, and `localhost` doesn't count once a
second device (an iPhone) is involved — it needs real HTTPS. `npm run dev` stays plain HTTP by
default (no self-signed-cert warning for ordinary desktop work); for on-device testing, use:

```
npm run dev:device   # HTTPS, bound to your LAN — e.g. https://10.0.0.5:5173
```

Open the printed `https://<LAN-IP>:PORT` URL on a phone on the same WiFi. The browser will warn
about the self-signed certificate (from `@vitejs/plugin-basic-ssl`) — accept it once per device.

## Testing it

```
npm test             # vitest
npm run typecheck    # tsc --noEmit
npm run lint         # eslint
npm run format:check # prettier --check
```

## The contract stub

`npm run dev` and the test suite both talk to a small stub (`stub/contract-stub.ts`) covering
only the three routes C-01 needs, with every response body loaded from the frozen
`contracts/examples/*.json` at startup so it cannot silently drift from the contract (D-040).
This is **not** the fuller "mock service" `docs/sprint-plan.md` describes for S1, which will
answer every operation (including reconstruction and verdict) with a canned GLB and verdict —
that is separate, later work. Once R-14 (the real reconstruction endpoint) exists, this stub is
no longer needed.

See `docs/decisions.md` D-040 (tooling choice) and D-041 (how `gyro` travels inside the
multipart `uploadFrame` request) for the reasoning behind this setup.

## Running a real session against the recon service on the Mac (C-13)

By default every command above talks to the in-process contract stub. C-13's end-to-end check
instead runs the real `recon.service` natively on the M2 Max and points the dev server's `/v1` at
it (`docs/decisions.md` D-049). Two terminals, both from a clean checkout with the Python and
capture environments set up:

```bash
# 1. repo root, Python venv active: the service, on Metal, reachable only from this Mac
PROOFSHAPE_RECON_DEVICE=mps uvicorn recon.service:app --host 127.0.0.1 --port 8000 --workers 1

# 2. capture/: the phone-facing HTTPS dev server, forwarding /v1 to that service
PROOFSHAPE_RECON_PROXY=http://127.0.0.1:8000 npm run dev:device
```

Open the printed `https://<LAN-IP>:5173/?order=PO-TEST` on the phone (same Wi-Fi, accept the
certificate once). `?order=` pre-fills the order code. Both variables go on the same line as the
command, as above, or are `export`ed first; a plain `NAME=value` line on its own is not seen by
the process started on the next line.

- `PROOFSHAPE_RECON_DEVICE` (`cuda` by default, `mps` on the Mac) picks the torch device the
  service reconstructs on.
- `PROOFSHAPE_RECON_PROXY` takes the service's origin only. When it is set, the contract stub is
  not mounted at all, and the `?viewer-demo=` links, which need the stub's canned GLBs, will 404.
  A malformed value stops the dev server at start-up rather than quietly falling back to the stub.
- Stop both with Ctrl-C when the check is done. The service only listens on `127.0.0.1`, but the
  dev server is on the LAN for as long as it runs, and the service accepts any non-empty order
  code.

### Lightning (not used by C-13)

`VITE_RECON_BASE_URL` and `VITE_RECON_TOKEN` point a build at a remote service instead, with
`Authorization: Bearer <token>` on every request (`src/api.ts`). The token ends up in the built
JS, visible to anyone who opens dev tools on the page (D-048). This path has never run against
Lightning; what is still unknown about it is recorded as O-005 in `docs/decisions.md`. If you use
it, set both on the command line or export them, and stop the deployment afterwards
(`recon/README.md`, R-15 runbook): it bills a GPU until `stop` runs.

## GitHub Pages hosting (C-21)

The capture app is built and deployed by `.github/workflows/capture-pages.yml`
whenever a push to `main` changes `capture/**`. The workflow uses the base path
reported by `actions/configure-pages`, so repository Pages and a future custom
domain both build the correct asset URLs, then publishes `capture/dist`.

An organisation/repository admin must set
**Settings → Pages → Build and deployment → Source** to **GitHub Actions** once.
No laptop dev server is required to open the resulting HTTPS page.

The hosted page is deliberately disconnected from the paid reconstruction service
until O-005 is answered and recorded in `docs/decisions.md`. While it is
disconnected, the production page stops before mounting the capture shell and says
"Reconstruction service is not configured for this hosted build yet" instead of
sending `/v1/sessions` to GitHub Pages and exposing a raw HTTP 405. Local
development is unchanged and still uses the Vite contract stub when the base URL
is blank. Public recon can only be enabled when all three O-005 confirmations are
explicitly set to `true`:

- `PROOFSHAPE_CORS_PREFLIGHT_CONFIRMED` — the real Lightning deployment accepted
  the browser's token-less CORS preflight.
- `PROOFSHAPE_PUBLIC_TOKEN_ACCEPTED` — the team explicitly accepted that the
  bearer token is visible in the public client bundle.
- `PROOFSHAPE_ORDER_GATE_CONFIRMED` — the deployed service rejects unknown order
  codes before they can start paid GPU work, through R-34 or configured
  `known_order_codes`.

Only after those are true may `PROOFSHAPE_PUBLIC_RECON_ENABLED` be set to
`true`, together with `PROOFSHAPE_RECON_BASE_URL` and the
`PROOFSHAPE_RECON_TOKEN` secret. The workflow fails closed when public recon is
enabled without any O-005 prerequisite or without the URL/token.

The `?viewer-demo=1`, `?viewer-demo=c12`, and `?viewer-demo=c14` helpers
remain development only because they depend on Vite's contract stub. A production
Pages build displays a clear message instead of trying and failing to call a
nonexistent stub.

The Mac path in the C-13 section above remains the fallback when Pages or
Lightning is unavailable.
