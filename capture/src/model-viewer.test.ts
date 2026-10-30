// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountModelViewer, type ModelLoader } from "./model-viewer.js";
import type { ReconstructionResult } from "./types.js";

function makeRoot(): HTMLElement {
  const element = document.createElement("div");
  document.body.replaceChildren(element);
  return element;
}

function complete(glbUrl?: string): ReconstructionResult {
  return glbUrl === undefined
    ? { status: "complete", metric: true }
    : { status: "complete", metric: true, glb_url: glbUrl };
}

describe("mountModelViewer", () => {
  it("invokes the loader with the completed reconstruction URL", () => {
    const load = vi.fn();
    const loaderFactory = vi.fn((): ModelLoader => ({ load }));
    const glbUrl = "https://example.test/reconstruction/model.glb";
    const target = makeRoot();

    mountModelViewer(target, complete(glbUrl), { loaderFactory });

    expect(loaderFactory).toHaveBeenCalledOnce();
    expect(load).toHaveBeenCalledOnce();
    expect(load.mock.calls[0]?.[0]).toBe(glbUrl);
    const download = target.querySelector("a");
    expect(download?.getAttribute("href")).toBe(glbUrl);
    expect(download?.textContent).toBe("Download GLB");
    expect(target.querySelector('[aria-label="3D model provenance legend"]')).toBeNull();
  });

  it("shows a failed-reconstruction fallback without constructing a loader", () => {
    const loaderFactory = vi.fn((): ModelLoader => ({ load: vi.fn() }));
    const target = makeRoot();

    mountModelViewer(target, { status: "failed", warnings: ["backend failed"] }, { loaderFactory });

    expect(target.textContent).toContain("unavailable because reconstruction failed");
    expect(loaderFactory).not.toHaveBeenCalled();
    expect(target.querySelector("canvas")).toBeNull();
  });

  it("shows a missing-URL fallback without constructing a loader", () => {
    const loaderFactory = vi.fn((): ModelLoader => ({ load: vi.fn() }));
    const target = makeRoot();

    mountModelViewer(target, complete(), { loaderFactory });

    expect(target.textContent).toContain("no GLB model URL was returned");
    expect(loaderFactory).not.toHaveBeenCalled();
    expect(target.querySelector("a")).toBeNull();
  });

  it("keeps the download control when the in-browser loader fails", () => {
    let rejectLoad: ((error: unknown) => void) | undefined;
    const loaderFactory = (): ModelLoader => ({
      load: (_url, _onLoad, _onProgress, onError) => {
        rejectLoad = onError;
      },
    });
    const glbUrl = "https://example.test/reconstruction/model.glb";
    const target = makeRoot();

    mountModelViewer(target, complete(glbUrl), { loaderFactory });
    rejectLoad?.(new Error("decoder failed"));

    expect(target.textContent).toContain("3D preview could not load");
    expect(target.textContent).toContain("Download GLB is still available");
    expect(target.querySelector("a")?.getAttribute("href")).toBe(glbUrl);
    expect(target.querySelector('[aria-label="3D model provenance legend"]')).toBeNull();
  });

  // C-13: the deployed service's glb_url is behind the same bearer-token gate as every other
  // route; a plain loader.load(url)/<a href> would 401 against it.
  describe("authToken (C-13)", () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("sets the loader's request header and omits the raw href up front", () => {
      const setRequestHeader = vi.fn();
      const load = vi.fn();
      const loaderFactory = vi.fn((): ModelLoader => ({ load, setRequestHeader }));
      const glbUrl = "https://example.test/reconstruction/model.glb";
      const target = makeRoot();
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(new Blob(["glb bytes"]), { status: 200 }),
      );

      mountModelViewer(target, complete(glbUrl), { loaderFactory, authToken: "the-token" });

      expect(setRequestHeader).toHaveBeenCalledWith({ Authorization: "Bearer the-token" });
      expect(load.mock.calls[0]?.[0]).toBe(glbUrl);
      // Not set to the raw (token-gated) URL -- that would 401 on a plain click before the
      // authenticated fetch below finishes.
      expect(target.querySelector("a")?.getAttribute("href")).toBeNull();
    });

    it("fetches the GLB with the bearer token and points the download link at the result", async () => {
      const loaderFactory = (): ModelLoader => ({ load: vi.fn() });
      const glbUrl = "https://example.test/reconstruction/model.glb";
      const target = makeRoot();
      const fetchSpy = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(new Response(new Blob(["glb bytes"]), { status: 200 }));

      mountModelViewer(target, complete(glbUrl), { loaderFactory, authToken: "the-token" });
      await vi.waitFor(() => {
        if (target.querySelector("a")?.getAttribute("href") === null) {
          throw new Error("still pending");
        }
      });

      expect(fetchSpy).toHaveBeenCalledWith(glbUrl, {
        headers: { Authorization: "Bearer the-token" },
      });
      const href = target.querySelector("a")?.getAttribute("href");
      expect(href).toMatch(/^blob:/);
    });

    it("disables the download link, without crashing, when the authenticated fetch fails", async () => {
      const loaderFactory = (): ModelLoader => ({ load: vi.fn() });
      const glbUrl = "https://example.test/reconstruction/model.glb";
      const target = makeRoot();
      vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 401 }));

      mountModelViewer(target, complete(glbUrl), { loaderFactory, authToken: "the-token" });
      const link = target.querySelector("a");
      await vi.waitFor(() => {
        if (link?.getAttribute("aria-disabled") !== "true") {
          throw new Error("still pending");
        }
      });

      expect(link?.getAttribute("href")).toBeNull();
      expect(link?.textContent).toContain("Download GLB unavailable");
    });
  });

  it("renders an explicit unavailable state without constructing a loader", () => {
    const loaderFactory = vi.fn((): ModelLoader => ({ load: vi.fn() }));
    const target = makeRoot();

    mountModelViewer(
      target,
      {
        status: "unavailable",
        message: "3D model unavailable because reconstruction timed out.",
      },
      { loaderFactory },
    );

    expect(target.textContent).toContain("reconstruction timed out");
    const status = target.querySelector('[role="status"]');
    expect(status).toBeInstanceOf(HTMLElement);
    expect((status as HTMLElement).style.fontWeight).toBe("bold");
    expect(loaderFactory).not.toHaveBeenCalled();
  });
});
