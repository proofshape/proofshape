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
