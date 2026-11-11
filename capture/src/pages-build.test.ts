import { fileURLToPath } from "node:url";
import type { OutputAsset, RollupOutput } from "rollup";
import { build } from "vite";
import { describe, expect, it } from "vitest";

function isRollupOutput(value: unknown): value is RollupOutput {
  return typeof value === "object" && value !== null && "output" in value;
}

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

    const rollupOutputs = Array.isArray(result) ? result : [result];
    const outputs = rollupOutputs
      .filter(isRollupOutput)
      .flatMap((entry) => entry.output);

    const html = outputs.find(
      (entry): entry is OutputAsset =>
        entry.type === "asset" && entry.fileName === "index.html",
    );

    expect(html).toBeDefined();
    if (!html) {
      throw new Error("Pages build test: index.html asset missing");
    }
    expect(String(html.source)).toContain("/proofshape/assets/");
  });
});
