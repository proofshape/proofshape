import { describe, expect, it } from "vitest";
import {
  angularDistanceDeg,
  anchorHeadingDeg,
  computeArrowGuidance,
  computeCoverage,
  computeTargetGuidance,
  DEFAULT_COVERAGE_TOLERANCE_DEG,
  DEFAULT_ORBIT,
  elevationFromBeta,
  guidanceLevel,
  NEAR_TOLERANCE_MULTIPLIER,
  relativeAzimuthDeg,
  viewAnglesFromGyro,
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
  it("maps the upright, camera-level pose (beta ~90) to elevation 0 -- not the mid-elevation ring, which is 45 deg higher", () => {
    expect(elevationFromBeta(90)).toBe(0);
  });

  it("maps the flat straight-down pose (beta 0) to elevation 90", () => {
    expect(elevationFromBeta(0)).toBe(90);
  });
});

describe("viewAnglesFromGyro", () => {
  it("uses non-zero roll in portrait for both heading and elevation", () => {
    const angles = viewAnglesFromGyro(gyro(0, 45, 10));

    // From R(0,45,10)*(0,0,-1): heading shift ~= 14.0 deg and downward elevation ~= 44.1 deg.
    expect(angles.headingDeg).toBeCloseTo(14.0019, 3);
    expect(angles.elevationDeg).toBeCloseTo(44.136, 3);
  });

  it("treats an upright landscape phone as camera-level, not straight-down", () => {
    const angles = viewAnglesFromGyro(gyro(0, 0, -90));

    expect(angles.elevationDeg).toBeCloseTo(0, 6);
    expect(angles.headingDeg).toBeCloseTo(270, 6);
  });
});

describe("anchorHeadingDeg", () => {
  it("is null with no accepted readings", () => {
    expect(anchorHeadingDeg([])).toBeNull();
  });

  it("uses the first accepted reading's full roll-aware heading", () => {
    expect(anchorHeadingDeg([gyro(0, 45, 10), gyro(99, 45)])).toBeCloseTo(14.0019, 3);
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

  it("assigns one reading to only its single nearest slot, even when it falls within tolerance of two rings", () => {
    // TabeenRaoof's and mbj1994's independently-caught regression: the 45/70 ring gap is 25 deg
    // and the default tolerance is 15 deg, so a reading at elevation 57 (12 from the mid ring at
    // 45, 13 from the high ring at 70) is within tolerance of both -- it must only cover the
    // nearer one (the mid ring, at distance 12), not both.
    const midRingIndex = 0; // DEFAULT_ORBIT[0]: azimuth 0, elevation 45
    const highRingIndex = 8; // DEFAULT_ORBIT[8]: azimuth 0, elevation 70
    const anchor = gyro(0, elevationFromBetaInverse(45)); // sets anchor, also covers the mid ring itself
    const ambiguousReading = gyro(0, elevationFromBetaInverse(57));

    const state = computeCoverage([anchor, ambiguousReading], DEFAULT_ORBIT);

    expect(state.covered[midRingIndex]).toBe(true);
    expect(state.covered[highRingIndex]).toBe(false);
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

  it("uses roll-aware heading/elevation in arrow guidance", () => {
    const rolled = gyro(0, 45, 10);
    const rollTarget: OrbitSlot = { azimuthDeg: 0, elevationDeg: 45 };
    const result = computeArrowGuidance(rolled, 0, rollTarget, 1);

    expect(result.onTarget).toBe(false);
    if (!result.onTarget) {
      expect(result.azimuthDeltaDeg).toBeCloseTo(-14.0019, 3);
      expect(result.elevationDeltaDeg).toBeCloseTo(0.864, 3);
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

// C-26: the viewfinder's frame colour.
describe("guidanceLevel", () => {
  const near = DEFAULT_COVERAGE_TOLERANCE_DEG * NEAR_TOLERANCE_MULTIPLIER;

  it("is on-target exactly when computeArrowGuidance says onTarget", () => {
    expect(guidanceLevel({ onTarget: true })).toBe("on-target");
  });

  it("is near up to twice the coverage tolerance on the worse axis, and far just past it", () => {
    expect(near).toBe(30);
    expect(guidanceLevel({ onTarget: false, azimuthDeltaDeg: 30, elevationDeltaDeg: 0 })).toBe(
      "near",
    );
    expect(guidanceLevel({ onTarget: false, azimuthDeltaDeg: 30.01, elevationDeltaDeg: 0 })).toBe(
      "far",
    );
    expect(guidanceLevel({ onTarget: false, azimuthDeltaDeg: 0, elevationDeltaDeg: -30 })).toBe(
      "near",
    );
    expect(guidanceLevel({ onTarget: false, azimuthDeltaDeg: 5, elevationDeltaDeg: -30.01 })).toBe(
      "far",
    );
  });

  it("uses the same 15 degree boundary as coverage for green, via computeArrowGuidance", () => {
    const target: OrbitSlot = { azimuthDeg: 90, elevationDeg: 45 };
    const at15 = computeArrowGuidance(gyro(75, elevationFromBetaInverse(45)), 0, target);
    const past15 = computeArrowGuidance(gyro(74.99, elevationFromBetaInverse(45)), 0, target);
    expect(guidanceLevel(at15)).toBe("on-target");
    expect(guidanceLevel(past15)).toBe("near");
  });

  it("measures across the azimuth wrap the short way, not the long way", () => {
    const target: OrbitSlot = { azimuthDeg: 350, elevationDeg: 45 };
    // naive diff 340 (far), shortest -25 (near)
    const live = gyro(15, elevationFromBetaInverse(45));
    expect(guidanceLevel(computeArrowGuidance(live, 0, target))).toBe("near");
    // shortest -40 -> far
    const farther = gyro(30, elevationFromBetaInverse(45));
    expect(guidanceLevel(computeArrowGuidance(farther, 0, target))).toBe("far");
  });
});

// C-26 device check, 2026-11-20: on @TabeenRaoof's iPhone the frame went green at about 58 deg,
// but each photo there was credited to the 70 deg slot the first photo had already covered.
describe("computeTargetGuidance", () => {
  // Recorded readings from that run (session sess_d4dfa462dddb, frames 0 and 1).
  const firstPhoto: Gyro = { alpha: 294, beta: 12, gamma: 3 };
  const heldAt58: Gyro = { alpha: 297, beta: 32, gamma: 1 };

  it("is not on target where coverage would credit an already-covered slot", () => {
    const accepted = [firstPhoto];
    const coverage = computeCoverage(accepted);
    const anchor = anchorHeadingDeg(accepted) ?? Number.NaN;
    const targetIndex = coverage.nextTargetIndex ?? -1;
    expect(targetIndex).toBe(0);
    const target = DEFAULT_ORBIT[targetIndex] as OrbitSlot;

    // The old check said green here, and the photo would not have counted.
    expect(computeArrowGuidance(heldAt58, anchor, target).onTarget).toBe(true);
    expect(computeCoverage([...accepted, heldAt58]).covered[targetIndex]).toBe(false);

    const guidance = computeTargetGuidance(heldAt58, anchor, DEFAULT_ORBIT, targetIndex);
    expect(guidance.onTarget).toBe(false);
    expect(guidanceLevel(guidance)).toBe("near");
    // The target ring is lower (45 deg), so the arrow asks for less tilt.
    expect(guidance.onTarget ? 0 : guidance.elevationDeltaDeg).toBeLessThan(0);
  });

  it("is on target once the reading would be credited to the target", () => {
    const accepted = [firstPhoto];
    const anchor = anchorHeadingDeg(accepted) ?? Number.NaN;
    const at50: Gyro = { alpha: 297, beta: 40, gamma: 1 };
    expect(computeTargetGuidance(at50, anchor, DEFAULT_ORBIT, 0).onTarget).toBe(true);
    expect(computeCoverage([...accepted, at50]).covered[0]).toBe(true);
  });

  it("is on target exactly when the photo would cover the target, across orientations", () => {
    // gyro() takes beta; these histories are written as (azimuth, elevation).
    const at = (azimuthDeg: number, elevationDeg: number): Gyro =>
      gyro(azimuthDeg, elevationFromBetaInverse(elevationDeg));
    const histories: Gyro[][] = [
      [firstPhoto],
      [at(0, 45)],
      // Part of the 45 deg ring done, from an anchor taken high up.
      [at(0, 80), at(0, 45), at(45, 45), at(90, 45)],
      // The whole 45 deg ring and two of the 70 deg slots done: the next target is a 70 deg slot.
      [...[0, 45, 90, 135, 180, 225, 270, 315].map((az) => at(az, 45)), at(0, 70), at(90, 70)],
    ];
    let checked = 0;
    let onTargetCount = 0;
    for (const accepted of histories) {
      const coverage = computeCoverage(accepted);
      const anchor = anchorHeadingDeg(accepted) ?? Number.NaN;
      const targetIndex = coverage.nextTargetIndex ?? -1;
      expect(targetIndex).toBeGreaterThanOrEqual(0);
      for (let alpha = 0; alpha < 360; alpha += 5) {
        for (let beta = 0; beta <= 90; beta += 1) {
          for (const gamma of [0, 7]) {
            const live: Gyro = { alpha, beta, gamma };
            const onTarget = computeTargetGuidance(
              live,
              anchor,
              DEFAULT_ORBIT,
              targetIndex,
            ).onTarget;
            const counts = computeCoverage([...accepted, live]).covered[targetIndex] === true;
            if (onTarget !== counts) {
              throw new Error(
                `mismatch at ${JSON.stringify(live)} for target ${String(targetIndex)}: ` +
                  `onTarget=${String(onTarget)}, counts=${String(counts)}`,
              );
            }
            checked += 1;
            if (onTarget) onTargetCount += 1;
          }
        }
      }
    }
    expect(checked).toBe(4 * 72 * 91 * 2);
    // The sweep must actually hit the on-target region, not pass by never being green.
    expect(onTargetCount).toBeGreaterThan(100);
  });

  it("rejects a target index outside the orbit", () => {
    expect(() => computeTargetGuidance(firstPhoto, 0, DEFAULT_ORBIT, 13)).toThrow(RangeError);
  });
});
