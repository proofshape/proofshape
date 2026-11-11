import { describe, expect, it } from "vitest";
import {
  resolveAppBasePath,
  resolveAuthToken,
  resolveBaseUrl,
  resolveViewerDemo,
} from "./config.js";

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


describe("resolveAppBasePath", () => {
  it("keeps root builds at root", () => {
    expect(resolveAppBasePath(undefined)).toBe("/");
    expect(resolveAppBasePath("/")).toBe("/");
  });

  it("normalizes a GitHub Pages repository path", () => {
    expect(resolveAppBasePath("proofshape")).toBe("/proofshape/");
    expect(resolveAppBasePath("/proofshape/")).toBe("/proofshape/");
  });
});

describe("resolveViewerDemo", () => {
  it("allows the canned viewer demos only in development", () => {
    expect(resolveViewerDemo("c14", true)).toBe("c14");
    expect(resolveViewerDemo("c14", false)).toBe("unavailable");
  });

  it("ignores unrelated query values", () => {
    expect(resolveViewerDemo(null, false)).toBeNull();
    expect(resolveViewerDemo("other", true)).toBeNull();
  });
});
