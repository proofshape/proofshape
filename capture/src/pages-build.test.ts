import { fileURLToPath } from "node:url";
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

    const outputs = Array.isArray(result)
      ? result.flatMap((entry) => entry.output)
      : result.output;
    const html = outputs.find(
      (entry) => entry.type === "asset" && entry.fileName === "index.html",
    );

    expect(html).toBeDefined();
    if (!html || html.type !== "asset") {
      throw new Error("Pages build test: index.html asset missing");
    }
    expect(String(html.source)).toContain("/proofshape/assets/");
  });
});
