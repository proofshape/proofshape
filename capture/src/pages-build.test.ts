import { fileURLToPath } from "node:url";
import type { OutputAsset, RollupOutput } from "rollup";
import { build } from "vite";
import { describe, expect, it } from "vitest";

function isRollupOutput(value: unknown): value is RollupOutput {
  return typeof value === "object" && value !== null && "output" in value;
}

describe("GitHub Pages production build", () => {
  it("rewrites both HTML entries under the configured base path", async () => {
    const result = await build({
      configFile: false,
      root: fileURLToPath(new URL("..", import.meta.url)),
      base: "/proofshape/",
      logLevel: "silent",
      build: {
        write: false,
        rollupOptions: {
          input: {
            capture: fileURLToPath(new URL("../index.html", import.meta.url)),
            report: fileURLToPath(new URL("../report.html", import.meta.url)),
          },
        },
      },
    });

    const rollupOutputs = Array.isArray(result) ? result : [result];
    const outputs = rollupOutputs.filter(isRollupOutput).flatMap((entry) => entry.output);

    const htmlAssets = new Map(
      outputs
        .filter((entry): entry is OutputAsset => entry.type === "asset")
        .map((entry) => [entry.fileName, String(entry.source)]),
    );

    expect(htmlAssets.get("index.html")).toContain("/proofshape/assets/");
    expect(htmlAssets.get("report.html")).toContain("/proofshape/assets/");
  });
});
