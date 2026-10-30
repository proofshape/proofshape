import { describe, expect, it } from "vitest";
import { RECON_PROXY_ENV, devReconProxy, resolveDevReconTarget } from "./dev-recon-target.js";

describe("resolveDevReconTarget (C-13 / D-049)", () => {
  it("keeps the contract stub when the variable is unset or blank", () => {
    expect(resolveDevReconTarget({})).toEqual({ mode: "stub" });
    expect(resolveDevReconTarget({ [RECON_PROXY_ENV]: "   " })).toEqual({ mode: "stub" });
  });

  it("proxies to the given origin, normalising a trailing slash", () => {
    expect(resolveDevReconTarget({ [RECON_PROXY_ENV]: "http://127.0.0.1:8000" })).toEqual({
      mode: "proxy",
      target: "http://127.0.0.1:8000",
    });
    expect(resolveDevReconTarget({ [RECON_PROXY_ENV]: " http://127.0.0.1:8000/ " })).toEqual({
      mode: "proxy",
      target: "http://127.0.0.1:8000",
    });
  });

  it("refuses a value that is not a URL instead of silently using the stub", () => {
    expect(() => resolveDevReconTarget({ [RECON_PROXY_ENV]: "127.0.0.1:8000x" })).toThrow(
      RECON_PROXY_ENV,
    );
    expect(() => resolveDevReconTarget({ [RECON_PROXY_ENV]: "not a url" })).toThrow(/is not a URL/);
  });

  it("refuses a non-http scheme, including a scheme-less host:port that parses as one", () => {
    expect(() => resolveDevReconTarget({ [RECON_PROXY_ENV]: "ftp://127.0.0.1:8000" })).toThrow(
      /http:\/\/ or https:\/\//,
    );
    // new URL("localhost:8000") parses with protocol "localhost:", an easy mistake to make.
    expect(() => resolveDevReconTarget({ [RECON_PROXY_ENV]: "localhost:8000" })).toThrow(
      /http:\/\/ or https:\/\//,
    );
  });

  it("refuses a path, query or fragment, which would become a prefix the service lacks", () => {
    for (const value of [
      "http://127.0.0.1:8000/v1",
      "http://127.0.0.1:8000/?a=1",
      "http://127.0.0.1:8000/#x",
    ]) {
      expect(() => resolveDevReconTarget({ [RECON_PROXY_ENV]: value })).toThrow(/origin only/);
    }
  });
});

describe("devReconProxy", () => {
  it("forwards only /v1, with X-Forwarded headers so glb_url points back at the dev server", () => {
    expect(devReconProxy("http://127.0.0.1:8000")).toEqual({
      "/v1": { target: "http://127.0.0.1:8000", xfwd: true },
    });
  });
});
