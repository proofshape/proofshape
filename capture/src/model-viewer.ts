import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { setStatus } from "./camera.js";
import { createProvenanceLegend, remapModelProvenanceColors } from "./provenance-colors.js";
import type { ReconstructionResult } from "./types.js";

interface LoadedModel {
  scene: THREE.Object3D;
}

export interface ModelLoader {
  load(
    url: string,
    onLoad: (model: LoadedModel) => void,
    onProgress: ((event: ProgressEvent) => void) | undefined,
    onError: (error: unknown) => void,
  ): void;
  // C-13: three.js's real GLTFLoader has this (it propagates to its internal file/texture
  // loaders); optional so the mocked ModelLoader in tests doesn't have to implement it.
  setRequestHeader?(headers: Record<string, string>): void;
}

export interface ModelViewerOptions {
  loaderFactory?: () => ModelLoader;
  // C-13: the deployed service's glb_url is behind Lightning's bearer-token gate (D-042/D-048)
  // just like every other route -- a plain `loader.load(url)`/`<a href>` would 401.
  authToken?: string;
}

interface ModelViewerUnavailable {
  status: "unavailable";
  message: string;
}

export type ModelViewerState = ReconstructionResult | ModelViewerUnavailable;

function describeLoadError(error: unknown): string {
  return error instanceof Error && error.message ? error.message : "unknown loading error";
}

// C-13: `href` is omitted (not set to the raw URL) when a token is required -- a plain
// navigation/download can't attach an Authorization header, so clicking it before the
// authenticated blob below is ready would just 401 against the real deployed service.
function createDownloadLink(href: string | undefined): HTMLAnchorElement {
  const link = document.createElement("a");
  if (href !== undefined) {
    link.href = href;
  }
  link.download = "proofshape-model.glb";
  link.textContent = "Download GLB";
  link.setAttribute("aria-label", "Download reconstructed GLB model");
  return link;
}

// C-13: fetches the GLB with the bearer token attached and points the link at an object URL,
// since <a href> itself can never carry a header. Revoked by the caller on dispose.
async function attachAuthenticatedDownload(
  link: HTMLAnchorElement,
  url: string,
  authToken: string,
): Promise<string | undefined> {
  try {
    const response = await fetch(url, { headers: { Authorization: `Bearer ${authToken}` } });
    if (!response.ok) {
      throw new Error(`download request failed with status ${String(response.status)}`);
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    link.href = objectUrl;
    return objectUrl;
  } catch (error) {
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
    link.textContent = `Download GLB unavailable (${describeLoadError(error)})`;
    return undefined;
  }
}

function viewportSize(viewport: HTMLElement): { width: number; height: number } {
  const rect = viewport.getBoundingClientRect();
  return {
    width: Math.max(1, Math.round(rect.width || window.innerWidth || 1)),
    height: Math.max(1, Math.round(rect.height || window.innerHeight * 0.6 || 1)),
  };
}

function disposeModelResources(model: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();

  model.traverse((object) => {
    const mesh = object as Partial<THREE.Mesh>;
    if (mesh.geometry) {
      geometries.add(mesh.geometry);
    }

    const material = mesh.material;
    if (Array.isArray(material)) {
      for (const item of material) {
        materials.add(item);
      }
    } else if (material) {
      materials.add(material);
    }
  });

  for (const material of materials) {
    for (const value of Object.values(material as unknown as Record<string, unknown>)) {
      if (value instanceof THREE.Texture) {
        textures.add(value);
      }
    }
  }

  for (const texture of textures) {
    texture.dispose();
  }
  for (const material of materials) {
    material.dispose();
  }
  for (const geometry of geometries) {
    geometry.dispose();
  }
}

function createThreePreview(viewport: HTMLElement, model: THREE.Object3D): () => void {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f5f5);
  scene.add(model);

  const bounds = new THREE.Box3().setFromObject(model);
  if (bounds.isEmpty()) {
    throw new Error("GLB contains no renderable geometry.");
  }
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  model.position.sub(center);

  const maxDimension = Math.max(size.x, size.y, size.z, 1);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, maxDimension * 100);
  const halfFovRadians = (camera.fov * Math.PI) / 360;
  const distance = (maxDimension / (2 * Math.tan(halfFovRadians))) * 1.6;
  camera.near = Math.max(distance / 1000, 0.01);
  camera.far = Math.max(distance * 100, maxDimension * 100);
  camera.position.set(distance * 0.8, distance * 0.6, distance);
  camera.lookAt(new THREE.Vector3());
  camera.updateProjectionMatrix();

  const hemisphere = new THREE.HemisphereLight(0xffffff, 0x444444, 2.2);
  scene.add(hemisphere);
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
  keyLight.position.set(1, 2, 3);
  scene.add(keyLight);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  let controls: OrbitControls | null = null;
  let resizeAttached = false;
  const resize = (): void => {
    const { width, height } = viewportSize(viewport);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };

  try {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.touchAction = "none";
    viewport.append(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.target.set(0, 0, 0);
    controls.touches.ONE = THREE.TOUCH.ROTATE;
    controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE;
    controls.update();

    resize();
    window.addEventListener("resize", resize);
    resizeAttached = true;

    const activeControls = controls;
    renderer.setAnimationLoop(() => {
      activeControls.update();
      renderer.render(scene, camera);
    });
  } catch (error) {
    if (resizeAttached) {
      window.removeEventListener("resize", resize);
    }
    renderer.setAnimationLoop(null);
    controls?.dispose();
    disposeModelResources(model);
    renderer.dispose();
    renderer.domElement.remove();
    throw error;
  }

  return () => {
    window.removeEventListener("resize", resize);
    renderer.setAnimationLoop(null);
    controls.dispose();
    disposeModelResources(model);
    renderer.dispose();
    renderer.domElement.remove();
  };
}

export function mountModelViewer(
  root: HTMLElement,
  result: ModelViewerState,
  options: ModelViewerOptions = {},
): () => void {
  root.replaceChildren();

  const status = document.createElement("p");
  status.setAttribute("role", "status");
  root.append(status);

  if (result.status === "unavailable") {
    setStatus(status, result.message, "error");
    return () => undefined;
  }

  if (result.status === "pending") {
    setStatus(status, "3D model will appear when reconstruction completes.", "info");
    return () => undefined;
  }

  if (result.status === "failed") {
    const detail = result.warnings?.length ? " " + result.warnings.join(" ") : "";
    setStatus(status, "3D model unavailable because reconstruction failed." + detail, "error");
    return () => undefined;
  }

  if (!result.glb_url) {
    setStatus(
      status,
      "Reconstruction completed, but no GLB model URL was returned. The 3D viewer cannot start.",
      "error",
    );
    return () => undefined;
  }

  const controlsContainer = document.createElement("div");
  const downloadLink = createDownloadLink(
    options.authToken === undefined ? result.glb_url : undefined,
  );
  controlsContainer.append(downloadLink);
  root.append(controlsContainer);

  let downloadObjectUrl: string | undefined;
  if (options.authToken !== undefined) {
    void attachAuthenticatedDownload(downloadLink, result.glb_url, options.authToken).then(
      (objectUrl) => {
        downloadObjectUrl = objectUrl;
      },
    );
  }

  const viewport = document.createElement("div");
  viewport.setAttribute("aria-label", "3D reconstruction viewer");
  viewport.style.width = "100%";
  viewport.style.height = "clamp(280px, 60vh, 720px)";
  viewport.style.overflow = "hidden";
  viewport.style.touchAction = "none";
  root.append(viewport);

  setStatus(status, "Loading 3D model…", "info");

  const loader = options.loaderFactory?.() ?? new GLTFLoader();
  if (options.authToken !== undefined) {
    loader.setRequestHeader?.({ Authorization: `Bearer ${options.authToken}` });
  }
  let disposed = false;
  let disposePreview: (() => void) | null = null;

  loader.load(
    result.glb_url,
    (loaded) => {
      if (disposed) {
        return;
      }
      try {
        const provenance = remapModelProvenanceColors(loaded.scene);
        if (provenance.foundProvenance) {
          controlsContainer.append(createProvenanceLegend());
        }
        disposePreview = createThreePreview(viewport, loaded.scene);
        const warning =
          provenance.unknownVertices > 0
            ? ` ${String(provenance.unknownVertices)} vertex colour(s) are unknown and shown in magenta.`
            : "";
        setStatus(
          status,
          "3D model loaded. Drag to orbit; pinch to zoom." + warning,
          provenance.unknownVertices > 0 ? "error" : "info",
        );
      } catch (error) {
        setStatus(
          status,
          "3D preview could not start: " +
            describeLoadError(error) +
            " Download GLB is still available.",
          "error",
        );
      }
    },
    undefined,
    (error) => {
      if (disposed) {
        return;
      }
      setStatus(
        status,
        "3D preview could not load: " +
          describeLoadError(error) +
          " Download GLB is still available.",
        "error",
      );
    },
  );

  return () => {
    disposed = true;
    disposePreview?.();
    if (downloadObjectUrl !== undefined) {
      URL.revokeObjectURL(downloadObjectUrl);
    }
  };
}
