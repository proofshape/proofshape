// @vitest-environment happy-dom
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  PROVENANCE_DISPLAY_COLORS,
  PROVENANCE_SOURCE_COLORS,
  createProvenanceLegend,
  mapProvenanceRgb,
  remapModelProvenanceColors,
} from "./provenance-colors.js";

function findProvenanceSource(): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, "recon/provenance.py");
    if (existsSync(candidate)) {
      return candidate;
    }
    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error("Could not find recon/provenance.py above the current working directory.");
    }
    directory = parent;
  }
}

function srgbByteToLinear(value: number): number {
  const channel = value / 255;
  return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
}

describe("C-12 provenance colors", () => {
  it("maps exact Observed and Unobserved source RGB values to the display palette", () => {
    expect(mapProvenanceRgb(PROVENANCE_SOURCE_COLORS.observed)).toEqual({
      kind: "observed",
      rgb: PROVENANCE_DISPLAY_COLORS.observed,
    });
    expect(mapProvenanceRgb(PROVENANCE_SOURCE_COLORS.unobserved)).toEqual({
      kind: "unobserved",
      rgb: PROVENANCE_DISPLAY_COLORS.unobserved,
    });
  });

  it("maps every other source RGB to a named unknown display case", () => {
    expect(mapProvenanceRgb([255, 128, 0])).toEqual({
      kind: "unknown",
      rgb: PROVENANCE_DISPLAY_COLORS.unknown,
    });
  });

  it("remaps the vertex buffer, enables vertex colors, and neutralizes metallic defaults", () => {
    const geometry = new THREE.BufferGeometry();
    const colors = new THREE.BufferAttribute(
      new Uint8Array([255, 255, 255, 128, 128, 128, 1, 2, 3]),
      3,
      true,
    );
    geometry.setAttribute("color", colors);
    const material = new THREE.MeshStandardMaterial({ metalness: 1, roughness: 1 });
    const mesh = new THREE.Mesh(geometry, material);
    const scene = new THREE.Scene();
    scene.add(mesh);

    const result = remapModelProvenanceColors(scene);

    expect(result).toEqual({ foundProvenance: true, unknownVertices: 1 });
    expect(material.vertexColors).toBe(true);
    expect(material.metalness).toBe(0);
    expect(colors.getX(0)).toBeCloseTo(srgbByteToLinear(PROVENANCE_DISPLAY_COLORS.observed[0]));
    expect(colors.getY(1)).toBeCloseTo(srgbByteToLinear(PROVENANCE_DISPLAY_COLORS.unobserved[1]));
    expect(colors.getZ(2)).toBeCloseTo(srgbByteToLinear(PROVENANCE_DISPLAY_COLORS.unknown[2]));
  });

  it("remaps shared geometry only once", () => {
    const geometry = new THREE.BufferGeometry();
    const colors = new THREE.BufferAttribute(new Uint8Array([255, 255, 255]), 3, true);
    geometry.setAttribute("color", colors);
    const scene = new THREE.Scene();
    scene.add(
      new THREE.Mesh(geometry, new THREE.MeshStandardMaterial()),
      new THREE.Mesh(geometry, new THREE.MeshStandardMaterial()),
    );

    const result = remapModelProvenanceColors(scene);

    expect(result).toEqual({ foundProvenance: true, unknownVertices: 0 });
    expect(colors.getX(0)).toBeCloseTo(srgbByteToLinear(PROVENANCE_DISPLAY_COLORS.observed[0]));
  });

  it("reports no provenance when a model has no color attribute", () => {
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial()));

    expect(remapModelProvenanceColors(scene)).toEqual({
      foundProvenance: false,
      unknownVertices: 0,
    });
  });

  it("reserves inferred and unknown slots in the visible legend", () => {
    const legend = createProvenanceLegend();
    expect(legend.getAttribute("role")).toBe("group");
    expect(legend.textContent).toContain("Observed");
    expect(legend.textContent).toContain("Unobserved");
    expect(legend.textContent).toContain("Inferred (reserved)");
    expect(legend.textContent).toContain("Unknown");
  });

  it("stays equal to recon/provenance.py's committed source-color constants", () => {
    const source = readFileSync(findProvenanceSource(), "utf8");
    const observed = source.match(
      /^OBSERVED_COLOR\b(?:\s*:[^=]+)?\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)/m,
    );
    const unobserved = source.match(
      /^UNOBSERVED_COLOR\b(?:\s*:[^=]+)?\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)/m,
    );

    expect(observed).not.toBeNull();
    expect(unobserved).not.toBeNull();
    expect(observed?.slice(1).map(Number)).toEqual([...PROVENANCE_SOURCE_COLORS.observed]);
    expect(unobserved?.slice(1).map(Number)).toEqual([...PROVENANCE_SOURCE_COLORS.unobserved]);
  });
});
