import { fileURLToPath } from "node:url";
import type { OutputAsset, RollupOutput } from "rollup";
import { build } from "vite";
import { describe, expect, it } from "vitest";

describe("GitHub Pages production build", () => {
  it("rewrites entry assets under the configured base path", async () => {
    const result = await build({
      configFile: false,
      root: fileURLToPath(new URL("..", import.meta.url)),
      base: "/proofshape/",
      logLevel: "silent",
      build: {
        write: false,
        rollupOptions: {
          input: fileURLToPath(new URL("../index.html", import.meta.url)),
        },
      },
    });

    const output = (result as RollupOutput).output;
    const html = output.find(
      (entry): entry is OutputAsset => entry.type === "asset" && entry.fileName === "index.html",
    );

    expect(html).toBeDefined();
    expect(String(html?.source)).toContain("/proofshape/assets/");
  });
});
