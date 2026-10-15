import * as THREE from "three";

export type SourceRgb = readonly [number, number, number];
export type DisplayRgb = readonly [number, number, number];

export const PROVENANCE_SOURCE_COLORS = {
  observed: [255, 255, 255] as const,
  unobserved: [128, 128, 128] as const,
} satisfies Record<string, SourceRgb>;

export const PROVENANCE_DISPLAY_COLORS = {
  observed: [34, 197, 94] as const,
  unobserved: [71, 85, 105] as const,
  inferred: [249, 115, 22] as const,
  unknown: [217, 70, 239] as const,
} satisfies Record<string, DisplayRgb>;

export type ProvenanceDisplayClass = keyof typeof PROVENANCE_DISPLAY_COLORS;

function matchesSource(rgb: SourceRgb, expected: SourceRgb): boolean {
  return rgb[0] === expected[0] && rgb[1] === expected[1] && rgb[2] === expected[2];
}

export function mapProvenanceRgb(rgb: SourceRgb): {
  kind: ProvenanceDisplayClass;
  rgb: DisplayRgb;
} {
  if (matchesSource(rgb, PROVENANCE_SOURCE_COLORS.observed)) {
    return { kind: "observed", rgb: PROVENANCE_DISPLAY_COLORS.observed };
  }
  if (matchesSource(rgb, PROVENANCE_SOURCE_COLORS.unobserved)) {
    return { kind: "unobserved", rgb: PROVENANCE_DISPLAY_COLORS.unobserved };
  }
  return { kind: "unknown", rgb: PROVENANCE_DISPLAY_COLORS.unknown };
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function channelToByte(value: number): number {
  return Math.round(clamp01(value) * 255);
}

function srgbChannelToLinear(value: number): number {
  const channel = clamp01(value);
  return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
}

function displayRgbToLinear(rgb: DisplayRgb): readonly [number, number, number] {
  return [
    srgbChannelToLinear(rgb[0] / 255),
    srgbChannelToLinear(rgb[1] / 255),
    srgbChannelToLinear(rgb[2] / 255),
  ];
}

interface MutableColorAttribute {
  itemSize: number;
  count: number;
  getX(index: number): number;
  getY(index: number): number;
  getZ(index: number): number;
  setXYZ(index: number, x: number, y: number, z: number): void;
  needsUpdate: boolean;
}

interface ColorGeometry {
  getAttribute(name: string): MutableColorAttribute | undefined;
}

interface VertexColorMaterial {
  vertexColors?: boolean;
  needsUpdate?: boolean;
}

export function remapModelProvenanceColors(model: THREE.Object3D): void {
  model.traverse((object) => {
    const mesh = object as THREE.Mesh & { isMesh?: boolean };
    if (!mesh.isMesh) {
      return;
    }

    const geometry = mesh.geometry as THREE.BufferGeometry & ColorGeometry;
    const color = geometry.getAttribute("color");
    if (!color || color.itemSize < 3) {
      return;
    }

    for (let index = 0; index < color.count; index += 1) {
      const mapped = mapProvenanceRgb([
        channelToByte(color.getX(index)),
        channelToByte(color.getY(index)),
        channelToByte(color.getZ(index)),
      ]);
      const linear = displayRgbToLinear(mapped.rgb);
      color.setXYZ(index, linear[0], linear[1], linear[2]);
    }
    color.needsUpdate = true;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      const vertexMaterial = material as THREE.Material & VertexColorMaterial;
      vertexMaterial.vertexColors = true;
      vertexMaterial.needsUpdate = true;
    }
  });
}

export function createProvenanceLegend(): HTMLElement {
  const legend = document.createElement("div");
  legend.setAttribute("aria-label", "3D model provenance legend");

  const entries: Array<[string, DisplayRgb]> = [
    ["Observed", PROVENANCE_DISPLAY_COLORS.observed],
    ["Unobserved", PROVENANCE_DISPLAY_COLORS.unobserved],
    ["Inferred (reserved)", PROVENANCE_DISPLAY_COLORS.inferred],
  ];

  for (const [label, rgb] of entries) {
    const item = document.createElement("span");
    item.style.display = "inline-flex";
    item.style.alignItems = "center";
    item.style.gap = "0.35rem";
    item.style.marginRight = "0.9rem";

    const swatch = document.createElement("span");
    swatch.setAttribute("aria-hidden", "true");
    swatch.style.display = "inline-block";
    swatch.style.width = "0.8rem";
    swatch.style.height = "0.8rem";
    swatch.style.border = "1px solid currentColor";
    swatch.style.backgroundColor = "rgb(" + rgb.join(", ") + ")";

    const text = document.createElement("span");
    text.textContent = label;

    item.append(swatch, text);
    legend.append(item);
  }

  return legend;
}
