import basicSsl from "@vitejs/plugin-basic-ssl";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";
import { createContractStub } from "./stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "./stub/page-test-port.js";

// Serves the capture mock service as dev middleware, so `npm run dev` needs no second process
// and no CORS setup. C-10 extends C-01's original contract stub with reconstruction polling and a
// canned GLB; see stub/contract-stub.ts.
function contractStubPlugin(): Plugin {
  return {
    name: "proofshape-contract-stub",
    configureServer(server) {
      const stub = createContractStub();
      server.middlewares.use(stub.handle);
    },
  };
}

// Default environment is plain Node — most of this suite is HTTP/fetch tests with no DOM.
// src/page.test.ts and src/shell.test.ts (C-04) opt into happy-dom (via a
// `// @vitest-environment happy-dom` docblock) and need to fetch a real session; happy-dom
// enforces the same-origin policy like a real browser, so both start their own contract stub on
// the one shared PAGE_TEST_STUB_PORT, matching the window's origin below — an ephemeral port
// would make every fetch cross-origin and silently blocked before it reaches the stub.
// fileParallelism is off for the same reason: Vitest runs test files in parallel worker
// processes by default, and two files both trying to bind that one fixed port at the same time
// collides (EADDRINUSE) rather than queuing — confirmed directly when C-04's shell.test.ts was
// added alongside page.test.ts. camera.test.ts/gyro.test.ts/dom.test.ts also use happy-dom but
// never fetch, so they were never affected by this until a second fetching file existed. This
// turns off parallelism for the whole suite, not just these two files — Vitest has no
// per-file-pair parallelism knob — but the suite is small enough (well under 5s) that this
// costs nothing worth trading away. If a future test becomes slow, that is not this setting;
// the fix for this specific problem, if it's ever worth narrowing, is giving page.test.ts and
// shell.test.ts their own distinct fixed ports/origins rather than re-enabling parallelism.
// C-02 needs a real iPhone to reach this dev server over HTTPS (Safari refuses getUserMedia
// outside a secure context, and "localhost" doesn't help once a second device is involved) —
// but C-01's own desktop-only workflow doesn't need the self-signed-cert browser warning that
// comes with that, so it's opt-in via `npm run dev:device` (HTTPS=true), not the default.
const servingForDeviceTesting = process.env.HTTPS === "true";

export default defineConfig({
  plugins: [contractStubPlugin(), ...(servingForDeviceTesting ? [basicSsl()] : [])],
  ...(servingForDeviceTesting ? { server: { host: true } } : {}),
  test: {
    environment: "node",
    environmentOptions: {
      happyDOM: {
        url: `http://127.0.0.1:${String(PAGE_TEST_STUB_PORT)}`,
      },
    },
    globals: true,
    fileParallelism: false,
  },
});
