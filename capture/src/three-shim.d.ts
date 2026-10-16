declare module "three" {
  export class Vector3 {
    x: number;
    y: number;
    z: number;
    constructor(x?: number, y?: number, z?: number);
    set(x: number, y: number, z: number): this;
    sub(value: Vector3): this;
  }

  export class Object3D {
    position: Vector3;
    traverse(callback: (object: Object3D) => void): void;
  }

  export class Scene extends Object3D {
    background: Color | null;
    add(...objects: Object3D[]): this;
  }

  export class PerspectiveCamera extends Object3D {
    fov: number;
    aspect: number;
    near: number;
    far: number;
    constructor(fov?: number, aspect?: number, near?: number, far?: number);
    lookAt(target: Vector3): void;
    updateProjectionMatrix(): void;
  }

  export class Color {
    readonly isColor: true;
    constructor(value: number | string);
  }

  export class Box3 {
    setFromObject(object: Object3D): this;
    isEmpty(): boolean;
    getCenter(target: Vector3): Vector3;
    getSize(target: Vector3): Vector3;
  }

  export class HemisphereLight extends Object3D {
    constructor(skyColor?: number, groundColor?: number, intensity?: number);
  }

  export class DirectionalLight extends Object3D {
    constructor(color?: number, intensity?: number);
  }

  export class BufferAttribute {
    itemSize: number;
    count: number;
    needsUpdate: boolean;
    constructor(array: ArrayLike<number>, itemSize: number, normalized?: boolean);
    getX(index: number): number;
    getY(index: number): number;
    getZ(index: number): number;
    setXYZ(index: number, x: number, y: number, z: number): this;
  }

  export class BufferGeometry {
    setAttribute(name: string, attribute: BufferAttribute): this;
    getAttribute(name: string): BufferAttribute | undefined;
    dispose(): void;
  }

  export class Texture {
    dispose(): void;
  }

  export class Material {
    vertexColors: boolean;
    needsUpdate: boolean;
    dispose(): void;
  }

  export class MeshStandardMaterial extends Material {
    metalness: number;
    roughness: number;
    constructor(parameters?: { vertexColors?: boolean; metalness?: number; roughness?: number });
  }

  export class Mesh extends Object3D {
    readonly isMesh: true;
    geometry: BufferGeometry;
    material: Material | Material[];
    constructor(geometry?: BufferGeometry, material?: Material | Material[]);
  }

  export class WebGLRenderer {
    domElement: HTMLCanvasElement;
    constructor(parameters?: { antialias?: boolean });
    setPixelRatio(value: number): void;
    setSize(width: number, height: number, updateStyle?: boolean): void;
    render(scene: Scene, camera: PerspectiveCamera): void;
    setAnimationLoop(callback: (() => void) | null): void;
    dispose(): void;
  }

  export const TOUCH: {
    ROTATE: number;
    PAN: number;
    DOLLY_ROTATE: number;
    DOLLY_PAN: number;
  };
}

declare module "three/addons/loaders/GLTFLoader.js" {
  import type { Object3D } from "three";

  interface GLTF {
    scene: Object3D;
  }

  export class GLTFLoader {
    load(
      url: string,
      onLoad: (model: GLTF) => void,
      onProgress?: (event: ProgressEvent) => void,
      onError?: (error: unknown) => void,
    ): void;
  }
}

declare module "three/addons/controls/OrbitControls.js" {
  import type { PerspectiveCamera, Vector3 } from "three";

  export class OrbitControls {
    enableDamping: boolean;
    enablePan: boolean;
    target: Vector3;
    touches: { ONE: number; TWO: number };
    constructor(camera: PerspectiveCamera, domElement: HTMLElement);
    update(): boolean;
    dispose(): void;
  }
}
