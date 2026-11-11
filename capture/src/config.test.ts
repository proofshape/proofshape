import { describe, expect, it } from "vitest";
import { resolveAuthToken, resolveBaseUrl, resolveViewerDemo } from "./config.js";

describe("resolveBaseUrl", () => {
  it("falls back to empty string when unset", () => {
    expect(resolveBaseUrl(undefined)).toBe("");
  });

  it("passes a bare base URL through unchanged", () => {
    expect(resolveBaseUrl("https://example.lightning.ai")).toBe("https://example.lightning.ai");
  });

  it("strips exactly one trailing slash", () => {
    expect(resolveBaseUrl("https://example.lightning.ai/")).toBe("https://example.lightning.ai");
  });
});

describe("resolveAuthToken", () => {
  it("resolves undefined to undefined", () => {
    expect(resolveAuthToken(undefined)).toBeUndefined();
  });

  it("resolves an empty or blank string to undefined", () => {
    expect(resolveAuthToken("")).toBeUndefined();
    expect(resolveAuthToken("   ")).toBeUndefined();
  });

  it("trims surrounding whitespace from a real token", () => {
    expect(resolveAuthToken("  secret-token  ")).toBe("secret-token");
  });
});

describe("resolveViewerDemo", () => {
  it("allows the known viewer demos in development", () => {
    expect(resolveViewerDemo("1", true)).toBe("1");
    expect(resolveViewerDemo("c12", true)).toBe("c12");
    expect(resolveViewerDemo("c14", true)).toBe("c14");
  });

  it("marks a known viewer demo unavailable in production", () => {
    expect(resolveViewerDemo("1", false)).toBe("unavailable");
  });

  it("ignores unknown viewer-demo values", () => {
    expect(resolveViewerDemo("other", true)).toBeUndefined();
  });
});
