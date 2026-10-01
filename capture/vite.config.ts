import basicSsl from "@vitejs/plugin-basic-ssl";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";
import { createContractStub } from "./stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "./stub/page-test-port.js";

// Serves the C-01 contract stub as dev middleware, so `npm run dev` needs no second process and
// no CORS setup. This is not the S1 mock service (docs/sprint-plan.md) — see stub/contract-stub.ts.
function contractStubPlugin(): Plugin {
  return {
    name: "proofshape-contract-stub",
    configureServer(server) {
      const stub = createContractStub();
      server.middlewares.use(stub.handle);
    },
  };
}

// Default environment is plain Node — most of this suite is HTTP/fetch tests with no DOM. Only
// src/page.test.ts opts into happy-dom (via a `// @vitest-environment happy-dom` docblock).
// happy-dom enforces the same-origin policy like a real browser, so that file's stub is started
// on PAGE_TEST_STUB_PORT instead of an ephemeral one, matching the window's origin below —
// otherwise every fetch from the mounted page is blocked as cross-origin before it reaches
// the stub.
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
  },
});
