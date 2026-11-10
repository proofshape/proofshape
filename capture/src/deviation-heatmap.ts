import * as THREE from "three";
import {
  PROVENANCE_DISPLAY_COLORS,
  PROVENANCE_SOURCE_COLORS,
  displayRgbToLinear,
  type Rgb,
} from "./provenance-colors.js";

export type HeatmapMode = "provenance" | "deviation";

export interface DeviationRange {
  rangeMm: number;
  source: "tolerance" | "largest-measured";
}

export interface DeviationHeatmapController {
  available: boolean;
  measuredObservedVertices: number;
  range: DeviationRange;
  captureProvenanceColors: () => void;
  setMode: (mode: HeatmapMode) => void;
}

interface HeatmapGeometry {
  color: THREE.BufferAttribute;
  deviation: THREE.BufferAttribute;
  measured: THREE.BufferAttribute;
  observed: Uint8Array;
  provenanceSnapshot: Float32Array;
}

const NEGATIVE_RGB: Rgb = [37, 99, 235];
const ZERO_RGB: Rgb = [250, 250, 250];
const POSITIVE_RGB: Rgb = [185, 28, 28];

function channelToByte(value: number): number {
  return Math.round(Math.min(1, Math.max(0, value)) * 255);
}

function isSourceObserved(color: THREE.BufferAttribute, index: number): boolean {
  const expected = PROVENANCE_SOURCE_COLORS.observed;
  return (
    channelToByte(color.getX(index)) === expected[0] &&
    channelToByte(color.getY(index)) === expected[1] &&
    channelToByte(color.getZ(index)) === expected[2]
  );
}

function mix(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

export function deviationToRgb(deviationMm: number, rangeMm: number): Rgb {
  if (!Number.isFinite(deviationMm)) throw new Error("deviation must be finite");
  if (!Number.isFinite(rangeMm) || rangeMm < 0) {
    throw new Error("range must be finite and non-negative");
  }
  if (rangeMm === 0 || deviationMm === 0) return ZERO_RGB;
  const clamped = Math.max(-rangeMm, Math.min(rangeMm, deviationMm));
  if (clamped < 0) {
    const t = 1 + clamped / rangeMm;
    return [
      mix(NEGATIVE_RGB[0], ZERO_RGB[0], t),
      mix(NEGATIVE_RGB[1], ZERO_RGB[1], t),
      mix(NEGATIVE_RGB[2], ZERO_RGB[2], t),
    ];
  }
  const t = clamped / rangeMm;
  return [
    mix(ZERO_RGB[0], POSITIVE_RGB[0], t),
    mix(ZERO_RGB[1], POSITIVE_RGB[1], t),
    mix(ZERO_RGB[2], POSITIVE_RGB[2], t),
  ];
}

export function chooseDeviationRange(
  measuredDeviationsMm: readonly number[],
  toleranceMm?: number,
): DeviationRange {
  if (toleranceMm !== undefined && Number.isFinite(toleranceMm) && toleranceMm > 0) {
    return { rangeMm: toleranceMm, source: "tolerance" };
  }
  const finite = measuredDeviationsMm.filter(Number.isFinite);
  const largest = finite.reduce((value, item) => Math.max(value, Math.abs(item)), 0);
  return { rangeMm: largest, source: "largest-measured" };
}

export function createDeviationHeatmapController(
  model: THREE.Object3D,
  toleranceMm?: number,
): DeviationHeatmapController {
  const geometries: HeatmapGeometry[] = [];
  const seenColors = new WeakSet<THREE.BufferAttribute>();
  const measuredValues: number[] = [];
  let measuredObservedVertices = 0;

  model.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    const color = object.geometry.getAttribute("color");
    const deviation = object.geometry.getAttribute("_deviation_mm");
    const measured = object.geometry.getAttribute("_deviation_measured");
    if (
      !(color instanceof THREE.BufferAttribute) ||
      !(deviation instanceof THREE.BufferAttribute) ||
      !(measured instanceof THREE.BufferAttribute) ||
      color.itemSize < 3 ||
      deviation.itemSize !== 1 ||
      measured.itemSize !== 1 ||
      color.count !== deviation.count ||
      color.count !== measured.count ||
      seenColors.has(color)
    ) return;
    seenColors.add(color);

    const observed = new Uint8Array(color.count);
    const provenanceSnapshot = new Float32Array(color.count * 3);
    for (let index = 0; index < color.count; index += 1) {
      if (isSourceObserved(color, index)) {
        observed[index] = 1;
        if (measured.getX(index) >= 0.5) {
          const value = deviation.getX(index);
          if (Number.isFinite(value)) {
            measuredValues.push(value);
            measuredObservedVertices += 1;
          }
        }
      }
    }
    geometries.push({ color, deviation, measured, observed, provenanceSnapshot });
  });

  const range = chooseDeviationRange(measuredValues, toleranceMm);
  const available = measuredObservedVertices > 0;
  let provenanceCaptured = false;

  const captureProvenanceColors = (): void => {
    for (const geometry of geometries) {
      for (let index = 0; index < geometry.color.count; index += 1) {
        geometry.provenanceSnapshot[index * 3] = geometry.color.getX(index);
        geometry.provenanceSnapshot[index * 3 + 1] = geometry.color.getY(index);
        geometry.provenanceSnapshot[index * 3 + 2] = geometry.color.getZ(index);
      }
    }
    provenanceCaptured = true;
  };

  const setMode = (mode: HeatmapMode): void => {
    if (!available || !provenanceCaptured) return;
    for (const geometry of geometries) {
      for (let index = 0; index < geometry.color.count; index += 1) {
        if (mode === "provenance") {
          geometry.color.setXYZ(
            index,
            geometry.provenanceSnapshot[index * 3] ?? 0,
            geometry.provenanceSnapshot[index * 3 + 1] ?? 0,
            geometry.provenanceSnapshot[index * 3 + 2] ?? 0,
          );
          continue;
        }
        const gated = geometry.observed[index] === 1 && geometry.measured.getX(index) >= 0.5;
        const rgb = gated
          ? deviationToRgb(geometry.deviation.getX(index), range.rangeMm)
          : PROVENANCE_DISPLAY_COLORS.unobserved;
        const linear = displayRgbToLinear(rgb);
        geometry.color.setXYZ(index, linear[0], linear[1], linear[2]);
      }
      geometry.color.needsUpdate = true;
    }
  };
  return {
    available,
    measuredObservedVertices,
    range,
    captureProvenanceColors,
    setMode,
  };
}

function formatMm(value: number): string {
  return Number.isInteger(value)
    ? String(value)
    : value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

export function createDeviationLegend(range: DeviationRange): HTMLElement {
  const legend = document.createElement("div");
  legend.setAttribute("role", "group");
  legend.setAttribute("aria-label", "Deviation from CAD legend");
  const r = formatMm(range.rangeMm);
  const source =
    range.source === "tolerance"
      ? "scale saturates at the order tolerance"
      : "scale: largest measured deviation, not a tolerance";
  legend.textContent = `Deviation from CAD (mm): −${r} mm · 0 mm · +${r} mm — ${source}. Grey = Unobserved or not measured. Colours do not mean pass or fail.`;
  return legend;
}
