import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import type { OutputAsset, RollupOutput } from "rollup";
import { build } from "vite";
import { describe, expect, it } from "vitest";

function isRollupOutput(value: unknown): value is RollupOutput {
  return typeof value === "object" && value !== null && "output" in value;
}

describe("GitHub Pages production build", () => {
  it("rewrites all HTML entries under the configured base path", async () => {
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
            buyer: fileURLToPath(new URL("../buyer.html", import.meta.url)),
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
    expect(htmlAssets.get("buyer.html")).toContain("/proofshape/assets/");
    expect(htmlAssets.get("report.html")).toContain("/proofshape/assets/");

    const boardAsset = outputs.find(
      (entry): entry is OutputAsset =>
        entry.type === "asset" && entry.fileName === "charuco_board.pdf",
    );
    expect(boardAsset).toBeDefined();
    const boardBytes =
      typeof boardAsset?.source === "string"
        ? new TextEncoder().encode(boardAsset.source)
        : boardAsset?.source;
    expect(boardBytes).toBeDefined();
    expect(
      createHash("sha256")
        .update(boardBytes ?? new Uint8Array())
        .digest("hex"),
    ).toBe("1b64d0328a7551da20fee6ccf5c6edacd5f48858be4004dffb093a60c9c9a354");
  });
});
