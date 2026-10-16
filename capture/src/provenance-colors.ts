import * as THREE from "three";

export type Rgb = readonly [number, number, number];

export const PROVENANCE_SOURCE_COLORS = {
  observed: [255, 255, 255] as const,
  unobserved: [128, 128, 128] as const,
} satisfies Record<string, Rgb>;

// Keep Observed and Inferred separated by both hue and lightness so the reserved Tier-2 class
// cannot be mistaken for observed geometry by red-green colour-blind viewers.
export const PROVENANCE_DISPLAY_COLORS = {
  observed: [37, 99, 235] as const,
  unobserved: [100, 116, 139] as const,
  inferred: [249, 115, 22] as const,
  unknown: [217, 70, 239] as const,
} satisfies Record<string, Rgb>;

export type ProvenanceDisplayClass = keyof typeof PROVENANCE_DISPLAY_COLORS;

export interface ProvenanceRemapResult {
  foundProvenance: boolean;
  unknownVertices: number;
}

function matchesSource(rgb: Rgb, expected: Rgb): boolean {
  return rgb[0] === expected[0] && rgb[1] === expected[1] && rgb[2] === expected[2];
}

export function mapProvenanceRgb(rgb: Rgb): {
  kind: ProvenanceDisplayClass;
  rgb: Rgb;
} {
  if (matchesSource(rgb, PROVENANCE_SOURCE_COLORS.observed)) {
    return { kind: "observed", rgb: PROVENANCE_DISPLAY_COLORS.observed };
  }
  if (matchesSource(rgb, PROVENANCE_SOURCE_COLORS.unobserved)) {
    return { kind: "unobserved", rgb: PROVENANCE_DISPLAY_COLORS.unobserved };
  }
  return { kind: "unknown", rgb: PROVENANCE_DISPLAY_COLORS.unknown };
}

function srgbChannelToLinear(channelByte: number): number {
  const channel = channelByte / 255;
  return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
}

function displayRgbToLinear(rgb: Rgb): Rgb {
  return [srgbChannelToLinear(rgb[0]), srgbChannelToLinear(rgb[1]), srgbChannelToLinear(rgb[2])];
}

// Three.js stores vertex colours in linear working space while the CSS legend uses sRGB.
// Precompute the four palette colours once instead of doing three Math.pow calls per vertex.
const LINEAR_DISPLAY_COLORS = Object.fromEntries(
  Object.entries(PROVENANCE_DISPLAY_COLORS).map(([key, rgb]) => [key, displayRgbToLinear(rgb)]),
) as Record<ProvenanceDisplayClass, Rgb>;

function channelToByte(value: number): number {
  return Math.round(Math.min(1, Math.max(0, value)) * 255);
}

export function remapModelProvenanceColors(model: THREE.Object3D): ProvenanceRemapResult {
  const remappedAttributes = new WeakSet<THREE.BufferAttribute>();
  let foundProvenance = false;
  let unknownVertices = 0;

  model.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) {
      return;
    }

    const color = object.geometry.getAttribute("color");
    if (!color || color.itemSize < 3) {
      return;
    }
    foundProvenance = true;

    // GLTFLoader can reuse one BufferGeometry for multiple nodes. Remap each shared attribute only
    // once; otherwise the second pass would interpret display colours as unknown source colours.
    if (!remappedAttributes.has(color)) {
      remappedAttributes.add(color);
      for (let index = 0; index < color.count; index += 1) {
        // write_glb stores normalized uint8 vertex colours, so converting the normalized floats
        // back to bytes gives the exact committed 255/128 provenance values.
        const mapped = mapProvenanceRgb([
          channelToByte(color.getX(index)),
          channelToByte(color.getY(index)),
          channelToByte(color.getZ(index)),
        ]);
        if (mapped.kind === "unknown") {
          unknownVertices += 1;
        }
        const linear = LINEAR_DISPLAY_COLORS[mapped.kind];
        color.setXYZ(index, linear[0], linear[1], linear[2]);
      }
      color.needsUpdate = true;
    }

    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      material.vertexColors = true;
      if (material instanceof THREE.MeshStandardMaterial) {
        // write_glb emits no material. glTF's default is fully metallic, which suppresses the
        // diffuse vertex colour and can make the provenance palette look almost black.
        material.metalness = 0;
      }
      material.needsUpdate = true;
    }
  });

  return { foundProvenance, unknownVertices };
}

export function createProvenanceLegend(): HTMLElement {
  const legend = document.createElement("div");
  legend.setAttribute("role", "group");
  legend.setAttribute("aria-label", "3D model provenance legend");

  const entries: Array<[string, Rgb]> = [
    ["Observed", PROVENANCE_DISPLAY_COLORS.observed],
    ["Unobserved", PROVENANCE_DISPLAY_COLORS.unobserved],
    ["Inferred (reserved)", PROVENANCE_DISPLAY_COLORS.inferred],
    ["Unknown", PROVENANCE_DISPLAY_COLORS.unknown],
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
