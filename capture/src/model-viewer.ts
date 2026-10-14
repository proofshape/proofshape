import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { setStatus } from "./camera.js";
import type { ReconstructionResult } from "./types.js";

interface LoadedModel {
  scene: THREE.Object3D;
}

export interface ModelLoader {
  load(
    url: string,
    onLoad: (model: LoadedModel) => void,
    onProgress: ((event: ProgressEvent<EventTarget>) => void) | undefined,
    onError: (error: unknown) => void,
  ): void;
}

export interface ModelViewerOptions {
  loaderFactory?: () => ModelLoader;
}

interface ModelViewerUnavailable {
  status: "unavailable";
  message: string;
}

export type ModelViewerState = ReconstructionResult | ModelViewerUnavailable;

function describeLoadError(error: unknown): string {
  return error instanceof Error && error.message ? error.message : "unknown loading error";
}

function createDownloadLink(url: string): HTMLAnchorElement {
  const link = document.createElement("a");
  link.href = url;
  link.download = "proofshape-model.glb";
  link.textContent = "Download GLB";
  link.setAttribute("aria-label", "Download reconstructed GLB model");
  return link;
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

    renderer.setAnimationLoop(() => {
      controls?.update();
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
    controls?.dispose();
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
  controlsContainer.append(createDownloadLink(result.glb_url));
  root.append(controlsContainer);

  const viewport = document.createElement("div");
  viewport.setAttribute("aria-label", "3D reconstruction viewer");
  viewport.style.width = "100%";
  viewport.style.height = "clamp(280px, 60vh, 720px)";
  viewport.style.overflow = "hidden";
  viewport.style.touchAction = "none";
  root.append(viewport);

  setStatus(status, "Loading 3D model…", "info");

  const loader = options.loaderFactory?.() ?? new GLTFLoader();
  let disposed = false;
  let disposePreview: (() => void) | null = null;

  loader.load(
    result.glb_url,
    (loaded) => {
      if (disposed) {
        return;
      }
      try {
        disposePreview = createThreePreview(viewport, loaded.scene);
        setStatus(status, "3D model loaded. Drag to orbit; pinch to zoom.", "info");
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
  };
}
