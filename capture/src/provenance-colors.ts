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

function channelToByte(value: number): number {
  return Math.round(THREE.MathUtils.clamp(value, 0, 1) * 255);
}

function byteToChannel(value: number): number {
  return value / 255;
}

export function remapModelProvenanceColors(model: THREE.Object3D): void {
  model.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) {
      return;
    }

    const color = mesh.geometry.getAttribute("color");
    if (!(color instanceof THREE.BufferAttribute) || color.itemSize < 3) {
      return;
    }

    for (let index = 0; index < color.count; index += 1) {
      const mapped = mapProvenanceRgb([
        channelToByte(color.getX(index)),
        channelToByte(color.getY(index)),
        channelToByte(color.getZ(index)),
      ]);
      color.setXYZ(
        index,
        byteToChannel(mapped.rgb[0]),
        byteToChannel(mapped.rgb[1]),
        byteToChannel(mapped.rgb[2]),
      );
    }
    color.needsUpdate = true;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (material instanceof THREE.MeshStandardMaterial || material instanceof THREE.MeshBasicMaterial) {
        material.vertexColors = true;
        material.needsUpdate = true;
      }
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
    swatch.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

    const text = document.createElement("span");
    text.textContent = label;

    item.append(swatch, text);
    legend.append(item);
  }

  return legend;
}
