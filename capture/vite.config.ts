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
export default defineConfig({
  plugins: [contractStubPlugin()],
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
