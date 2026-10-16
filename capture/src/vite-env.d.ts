/// <reference types="vite/client" />

// C-13: build-time config that points capture/ at the deployed service instead of the local
// contract stub (main.ts resolves these through config.ts). Unset in every existing dev/test
// run, which is what keeps today's stub-talking behaviour the default.
interface ImportMetaEnv {
  readonly VITE_RECON_BASE_URL?: string;
  readonly VITE_RECON_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
