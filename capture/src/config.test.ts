import { describe, expect, it } from "vitest";
import { resolveAuthToken, resolveBaseUrl } from "./config.js";

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
