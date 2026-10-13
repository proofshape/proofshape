// @vitest-environment happy-dom
import { describe, expect, it, vi } from "vitest";
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
    const loaderFactory = vi.fn(
      (): ModelLoader => ({
        load,
      }),
    );
    const glbUrl = "https://example.test/reconstruction/model.glb";
    const target = makeRoot();

    mountModelViewer(target, complete(glbUrl), { loaderFactory });

    expect(loaderFactory).toHaveBeenCalledOnce();
    expect(load).toHaveBeenCalledOnce();
    expect(load.mock.calls[0]?.[0]).toBe(glbUrl);
    const download = target.querySelector("a");
    expect(download?.getAttribute("href")).toBe(glbUrl);
    expect(download?.textContent).toBe("Download GLB");
  });

  it("shows a failed-reconstruction fallback without constructing a loader", () => {
    const loaderFactory = vi.fn((): ModelLoader => ({ load: vi.fn() }));
    const target = makeRoot();

    mountModelViewer(
      target,
      { status: "failed", warnings: ["backend failed"] },
      { loaderFactory },
    );

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
  });
});
