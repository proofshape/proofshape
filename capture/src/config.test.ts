import { describe, expect, it } from "vitest";
import {
  HOSTED_RECON_UNAVAILABLE_MESSAGE,
  hostedReconUnavailable,
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

describe("hostedReconUnavailable", () => {
  it("fails closed in production when O-005 leaves the public backend disconnected", () => {
    expect(hostedReconUnavailable("", false)).toBe(true);
    expect(HOSTED_RECON_UNAVAILABLE_MESSAGE).toMatch(
      /reconstruction service is not configured/i,
    );
  });

  it("keeps the local development stub available with an empty base URL", () => {
    expect(hostedReconUnavailable("", true)).toBe(false);
  });

  it("allows a configured production backend", () => {
    expect(hostedReconUnavailable("https://example.lightning.ai", false)).toBe(false);
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
