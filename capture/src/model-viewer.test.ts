// @vitest-environment happy-dom
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountModelViewer, type ModelLoader } from "./model-viewer.js";
import type { ReconstructionResult } from "./types.js";

// C-19: new THREE.WebGLRenderer() throws under this project's happy-dom test environment
// ("Error creating WebGL context") -- there is no real GPU context to give it. Mocking only
// this one browser capability (not three.js itself) lets mountModelViewer's real success path
// (provenance remap, legend, status text) run end to end in the tests below.
vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof import("three")>();
  class FakeWebGLRenderer {
    domElement = document.createElement("canvas");
    setPixelRatio(): void {}
    setSize(): void {}
    setAnimationLoop(): void {}
    render(): void {}
    dispose(): void {}
  }
  return { ...actual, WebGLRenderer: FakeWebGLRenderer };
});

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "stub", "fixtures");

// C-19: wraps the same fixture GLBs the mock service serves through a real GLTFLoader.parse(),
// so these tests exercise mountModelViewer's actual success path instead of a mocked loader
// that only ever records its own calls.
function realGltfLoaderFor(fixtureFileName: string): ModelLoader {
  const bytes = readFileSync(join(FIXTURES_DIR, fixtureFileName));
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return {
    load: (_url, onLoad, _onProgress, onError) => {
      new GLTFLoader().parse(
        arrayBuffer,
        "",
        (gltf) => {
          onLoad({ scene: gltf.scene });
        },
        onError,
      );
    },
  };
}

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

  // C-19 (#192): a real GLB through the real success path, not a mocked loader -- see the
  // fixture/WebGLRenderer-mock setup at the top of this file.
  describe("provenance warning and legend, against real fixture GLBs (C-19)", () => {
    it("shows the legend and no warning for a model that has provenance colors", async () => {
      const target = makeRoot();

      mountModelViewer(target, complete("https://example.test/model.glb"), {
        loaderFactory: () => realGltfLoaderFor("model-provenance.glb"),
      });

      await vi.waitFor(() => {
        if (!target.textContent.includes("3D model loaded")) {
          throw new Error("still pending");
        }
      });
      expect(target.querySelector('[aria-label="3D model provenance legend"]')).not.toBeNull();
      expect(target.textContent).not.toContain("has no provenance colours");
    });

    it("shows a no-provenance warning and no legend for a model with no provenance colors", async () => {
      const target = makeRoot();

      mountModelViewer(target, complete("https://example.test/model.glb"), {
        loaderFactory: () => realGltfLoaderFor("model.glb"),
      });

      await vi.waitFor(() => {
        if (!target.textContent.includes("3D model loaded")) {
          throw new Error("still pending");
        }
      });
      expect(target.querySelector('[aria-label="3D model provenance legend"]')).toBeNull();
      expect(target.textContent).toContain("has no provenance colours");
    });
  });
  it("offers the C-14 colour mode when measured Observed deviation data exists", async () => {
    const THREE = await import("three");
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(
        new Uint8Array([255, 255, 255, 128, 128, 128, 255, 255, 255]),
        3,
        true,
      ),
    );
    geometry.setAttribute(
      "_deviation_mm",
      new THREE.BufferAttribute(new Float32Array([-2, 99, 1]), 1),
    );
    geometry.setAttribute(
      "_deviation_measured",
      new THREE.BufferAttribute(new Float32Array([1, 1, 0]), 1),
    );
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial()));
    const target = makeRoot();

    mountModelViewer(target, complete("https://example.test/model.glb"), {
      loaderFactory: () => ({
        load: (_url, onLoad) => {
          onLoad({ scene });
        },
      }),
    });

    await vi.waitFor(() => {
      if (!target.textContent.includes("3D model loaded")) {
        throw new Error("still pending");
      }
    });
    const select = target.querySelector('select[aria-label="Colour 3D model by"]');
    expect(select).toBeInstanceOf(HTMLSelectElement);
    (select as HTMLSelectElement).value = "deviation";
    select?.dispatchEvent(new Event("change"));
    expect(target.textContent).toContain("Deviation from CAD (mm)");
    expect(target.textContent).toContain("not a tolerance");
  });

  it("can open a deviation-capable model directly in C-14 heatmap mode", async () => {
    const target = makeRoot();

    mountModelViewer(target, complete("https://example.test/model.glb"), {
      loaderFactory: () => realGltfLoaderFor("model-deviation.glb"),
      initialColourMode: "deviation",
    });

    await vi.waitFor(() => {
      if (!target.textContent.includes("3D model loaded")) {
        throw new Error("still pending");
      }
    });
    const select = target.querySelector('select[aria-label="Colour 3D model by"]');
    expect(select).toBeInstanceOf(HTMLSelectElement);
    expect((select as HTMLSelectElement).value).toBe("deviation");
    expect(target.textContent).toContain("Deviation from CAD (mm)");
  });

  it("loads the committed C-14 fixture through the real GLTFLoader", async () => {
    const target = makeRoot();

    mountModelViewer(target, complete("https://example.test/model.glb"), {
      loaderFactory: () => realGltfLoaderFor("model-deviation.glb"),
    });

    await vi.waitFor(() => {
      if (!target.textContent.includes("3D model loaded")) {
        throw new Error("still pending");
      }
    });
    expect(target.querySelector('select[aria-label="Colour 3D model by"]')).not.toBeNull();
  });
});
