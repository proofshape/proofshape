// @vitest-environment happy-dom
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  chooseDeviationRange,
  createDeviationHeatmapController,
  createDeviationLegend,
  deviationToRgb,
} from "./deviation-heatmap.js";
import { remapModelProvenanceColors } from "./provenance-colors.js";

function makeScene(includeMeasured = true): { scene: THREE.Scene; color: THREE.BufferAttribute } {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), 3),
  );
  const color = new THREE.BufferAttribute(
    new Uint8Array([
      255, 255, 255,
      128, 128, 128,
      255, 255, 255,
      255, 255, 255,
    ]),
    3,
    true,
  );
  geometry.setAttribute("color", color);
  geometry.setAttribute(
    "_deviation_mm",
    new THREE.BufferAttribute(new Float32Array([-2, 99, 1, 5]), 1),
  );
  if (includeMeasured) {
    geometry.setAttribute(
      "_deviation_measured",
      new THREE.BufferAttribute(new Float32Array([1, 1, 0, 1]), 1),
    );
  }
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial()));
  remapModelProvenanceColors(scene);
  return { scene, color };
}

function snapshot(color: THREE.BufferAttribute): number[] {
  const out: number[] = [];
  for (let index = 0; index < color.count; index += 1) {
    out.push(color.getX(index), color.getY(index), color.getZ(index));
  }
  return out;
}

describe("C-14 deviation heatmap", () => {
  it("maps zero, both ends, and values beyond the range", () => {
    expect(deviationToRgb(0, 2)).toEqual([250, 250, 250]);
    expect(deviationToRgb(-2, 2)).toEqual([37, 99, 235]);
    expect(deviationToRgb(-20, 2)).toEqual([37, 99, 235]);
    expect(deviationToRgb(2, 2)).toEqual([185, 28, 28]);
    expect(deviationToRgb(20, 2)).toEqual([185, 28, 28]);
  });

  it("uses tolerance when supplied, otherwise the largest measured absolute deviation", () => {
    expect(chooseDeviationRange([-2, 1, 5], 3)).toEqual({ rangeMm: 3, source: "tolerance" });
    expect(chooseDeviationRange([-2, 1, 5])).toEqual({
      rangeMm: 5,
      source: "largest-measured",
    });
  });

  it("requires both Observed provenance and measured=1, and restores provenance exactly", () => {
    const { scene, color } = makeScene();
    const before = snapshot(color);
    const controller = createDeviationHeatmapController(scene);
    expect(controller.available).toBe(true);
    expect(controller.measuredObservedVertices).toBe(2);
    expect(controller.range).toEqual({ rangeMm: 5, source: "largest-measured" });

    controller.setMode("deviation");
    const deviation = snapshot(color);
    expect(deviation.slice(3, 6)).toEqual(deviation.slice(6, 9));

    controller.setMode("provenance");
    expect(snapshot(color)).toEqual(before);
    controller.setMode("deviation");
    controller.setMode("provenance");
    expect(snapshot(color)).toEqual(before);
  });

  it("is unavailable without both custom attributes", () => {
    const { scene } = makeScene(false);
    expect(createDeviationHeatmapController(scene).available).toBe(false);
  });

  it("legend states the fallback is not a tolerance and never labels colours pass/fail", () => {
    const legend = createDeviationLegend({ rangeMm: 5, source: "largest-measured" });
    expect(legend.textContent).toContain("largest measured deviation, not a tolerance");
    expect(legend.textContent).toContain("do not mean pass or fail");
  });
});
