import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { describe, expect, it } from "vitest";

describe("GitHub Pages production build", () => {
  it("rewrites all HTML entries and ships the pinned board asset", async () => {
    const outDir = await mkdtemp(join(tmpdir(), "proofshape-pages-"));
    try {
      await build({
        configFile: false,
        root: fileURLToPath(new URL("..", import.meta.url)),
        base: "/proofshape/",
        logLevel: "silent",
        build: {
          outDir,
          emptyOutDir: true,
          rollupOptions: {
            input: {
              capture: fileURLToPath(new URL("../index.html", import.meta.url)),
              buyer: fileURLToPath(new URL("../buyer.html", import.meta.url)),
              report: fileURLToPath(new URL("../report.html", import.meta.url)),
            },
          },
        },
      });

      const [captureHtml, buyerHtml, reportHtml, boardBytes] = await Promise.all([
        readFile(join(outDir, "index.html"), "utf-8"),
        readFile(join(outDir, "buyer.html"), "utf-8"),
        readFile(join(outDir, "report.html"), "utf-8"),
        readFile(join(outDir, "charuco_board.pdf")),
      ]);

      expect(captureHtml).toContain("/proofshape/assets/");
      expect(buyerHtml).toContain("/proofshape/assets/");
      expect(reportHtml).toContain("/proofshape/assets/");
      expect(createHash("sha256").update(boardBytes).digest("hex")).toBe(
        "1b64d0328a7551da20fee6ccf5c6edacd5f48858be4004dffb093a60c9c9a354",
      );
    } finally {
      await rm(outDir, { recursive: true, force: true });
    }
  });
});
