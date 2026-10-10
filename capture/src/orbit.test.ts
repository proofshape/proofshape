import { describe, expect, it } from "vitest";
import {
  angularDistanceDeg,
  anchorAlphaDeg,
  computeArrowGuidance,
  computeCoverage,
  DEFAULT_COVERAGE_TOLERANCE_DEG,
  DEFAULT_ORBIT,
  elevationFromBeta,
  relativeAzimuthDeg,
  type OrbitSlot,
} from "./orbit.js";
import type { Gyro } from "./types.js";

// No `// @vitest-environment happy-dom` -- orbit.ts is pure math, no DOM, the whole point.

function gyro(alpha: number, beta: number, gamma = 0): Gyro {
  return { alpha, beta, gamma };
}

describe("angularDistanceDeg", () => {
  it("wraps around 0/360 instead of taking the naive difference", () => {
    expect(angularDistanceDeg(350, 10)).toBe(20);
  });

  it("is 180 for exact opposites", () => {
    expect(angularDistanceDeg(0, 180)).toBe(180);
  });

  it("is 0 for identical values", () => {
    expect(angularDistanceDeg(45, 45)).toBe(0);
  });
});

describe("relativeAzimuthDeg", () => {
  it("is 0 when the reading equals the anchor", () => {
    expect(relativeAzimuthDeg(100, 100)).toBe(0);
  });

  it("wraps correctly when the raw reading is behind the anchor", () => {
    expect(relativeAzimuthDeg(10, 350)).toBe(20);
  });

  it("wraps correctly when the raw reading is ahead of a near-360 anchor", () => {
    expect(relativeAzimuthDeg(350, 10)).toBe(340);
  });
});

describe("elevationFromBeta", () => {
  it("maps the upright mid-elevation pose (beta ~90) to elevation 0", () => {
    expect(elevationFromBeta(90)).toBe(0);
  });

  it("maps the flat straight-down pose (beta 0) to elevation 90", () => {
    expect(elevationFromBeta(0)).toBe(90);
  });
});

describe("anchorAlphaDeg", () => {
  it("is null with no accepted readings", () => {
    expect(anchorAlphaDeg([])).toBeNull();
  });

  it("is the first accepted reading's alpha, not any later one", () => {
    expect(anchorAlphaDeg([gyro(10, 45), gyro(99, 45)])).toBe(10);
  });
});

describe("computeCoverage", () => {
  it("covers nothing and targets index 0 with no accepted readings", () => {
    const state = computeCoverage([]);
    expect(state.covered.every((c) => !c)).toBe(true);
    expect(state.nextTargetIndex).toBe(0);
  });

  it("covers the first slot when a reading lands within tolerance of it", () => {
    // DEFAULT_ORBIT[0] = { azimuthDeg: 0, elevationDeg: 45 }. Anchor at alpha=0, so relative
    // azimuth of the same reading is 0 -- exactly on slot 0.
    const readings = [gyro(0, elevationFromBetaInverse(45))];
    const state = computeCoverage(readings);
    expect(state.covered[0]).toBe(true);
    expect(state.nextTargetIndex).toBe(1);
  });

  it("does not cover a slot just outside tolerance", () => {
    // The first reading sets the anchor (alpha=0) but at an elevation far from every orbit slot
    // (45/70/90), so it covers nothing itself -- isolating the azimuth-tolerance check below to
    // the second reading alone.
    const anchorReading = gyro(0, elevationFromBetaInverse(10));
    const justOutside = DEFAULT_COVERAGE_TOLERANCE_DEG + 1;
    const testedReading = gyro(justOutside, elevationFromBetaInverse(45)); // matches slot 0's elevation
    const state = computeCoverage([anchorReading, testedReading]);
    expect(state.covered[0]).toBe(false);
    expect(state.nextTargetIndex).toBe(0);
  });

  it("returns nextTargetIndex null once every slot is covered", () => {
    const anchor = 0;
    const readings = [
      gyro(anchor, elevationFromBetaInverse(45)),
      ...DEFAULT_ORBIT.slice(1).map((slot) =>
        gyro(anchor + slot.azimuthDeg, elevationFromBetaInverse(slot.elevationDeg)),
      ),
    ];
    const state = computeCoverage(readings);
    expect(state.covered.every(Boolean)).toBe(true);
    expect(state.nextTargetIndex).toBeNull();
  });

  it("exempts azimuth for the straight-down slot: a wildly different azimuth still covers it", () => {
    const strayAzimuthReading = gyro(999 % 360, elevationFromBetaInverse(90));
    const state = computeCoverage([gyro(0, 0), strayAzimuthReading], DEFAULT_ORBIT);
    const straightDownIndex = DEFAULT_ORBIT.findIndex((s) => s.elevationDeg === 90);
    expect(state.covered[straightDownIndex]).toBe(true);
  });
});

describe("computeArrowGuidance", () => {
  const target: OrbitSlot = { azimuthDeg: 90, elevationDeg: 45 };

  it("is onTarget when both azimuth and elevation are within tolerance", () => {
    const live = gyro(90, elevationFromBetaInverse(45));
    expect(computeArrowGuidance(live, 0, target)).toEqual({ onTarget: true });
  });

  it("gives a positive azimuthDeltaDeg when the target is clockwise of the current reading", () => {
    const live = gyro(60, elevationFromBetaInverse(45)); // target at 90, live at 60 -> +30
    const result = computeArrowGuidance(live, 0, target);
    expect(result.onTarget).toBe(false);
    if (!result.onTarget) {
      expect(result.azimuthDeltaDeg).toBeCloseTo(30);
      expect(result.elevationDeltaDeg).toBeCloseTo(0);
    }
  });

  it("gives a negative azimuthDeltaDeg when the target is counterclockwise of the current reading", () => {
    const live = gyro(120, elevationFromBetaInverse(45)); // target at 90, live at 120 -> -30
    const result = computeArrowGuidance(live, 0, target);
    expect(result.onTarget).toBe(false);
    if (!result.onTarget) {
      expect(result.azimuthDeltaDeg).toBeCloseTo(-30);
    }
  });

  it("picks the shortest path around the wrap instead of the long way", () => {
    const wrapTarget: OrbitSlot = { azimuthDeg: 350, elevationDeg: 45 };
    const live = gyro(10, elevationFromBetaInverse(45)); // naive diff = 340, shortest = -20
    const result = computeArrowGuidance(live, 0, wrapTarget);
    expect(result.onTarget).toBe(false);
    if (!result.onTarget) {
      expect(result.azimuthDeltaDeg).toBeCloseTo(-20);
    }
  });

  it("skips the azimuth delta entirely for a near-vertical target", () => {
    const straightDown: OrbitSlot = { azimuthDeg: 0, elevationDeg: 90 };
    const live = gyro(177, elevationFromBetaInverse(50)); // wildly off azimuth, off elevation too
    const result = computeArrowGuidance(live, 0, straightDown);
    expect(result.onTarget).toBe(false);
    if (!result.onTarget) {
      expect(result.azimuthDeltaDeg).toBe(0);
      expect(result.elevationDeltaDeg).toBeCloseTo(40);
    }
  });
});

// Inverse of elevationFromBeta, for building synthetic Gyro fixtures from a desired elevation.
function elevationFromBetaInverse(elevationDeg: number): number {
  return 90 - elevationDeg;
}
