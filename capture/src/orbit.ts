// C-07: the fixed batch orbit (proposal §6.2 -- D-002 makes batch the guaranteed path, so this
// can be built without waiting for reconstruction), plus the pure logic to track which slots are
// covered and compute the arrow direction toward the next one. No DOM, no device dependency --
// capture/src/overlay.ts is what actually renders this.

import type { Gyro } from "./types.js";

export interface OrbitSlot {
  readonly azimuthDeg: number; // relative to the first accepted frame's alpha, 0-360
  readonly elevationDeg: number; // 0 = level with the part, 90 = straight down
}

// Starting assumptions, not measured/calibrated values -- same honesty this project already
// applies to recon/uncertainty.py's DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM. Tuning against a real
// walked orbit is follow-up work this story's manual check starts, not finishes.
export const DEFAULT_MID_ELEVATION_DEG = 45;
export const DEFAULT_HIGH_ELEVATION_DEG = 70;
// One shared tolerance for both azimuth and elevation -- the acceptance criteria says "the
// angular tolerance" (singular), not two separate ones.
export const DEFAULT_COVERAGE_TOLERANCE_DEG = 15;

// Azimuth is meaningless at the pole -- a target this close to straight down exempts azimuth
// entirely, in both computeCoverage and computeArrowGuidance.
const AZIMUTH_EXEMPT_ELEVATION_DEG = 85;

function buildDefaultOrbit(): OrbitSlot[] {
  const slots: OrbitSlot[] = [];
  // Eight around the part at mid elevation (proposal §6.2).
  for (let i = 0; i < 8; i++) {
    slots.push({ azimuthDeg: i * 45, elevationDeg: DEFAULT_MID_ELEVATION_DEG });
  }
  // The rest higher.
  for (let i = 0; i < 4; i++) {
    slots.push({ azimuthDeg: i * 90, elevationDeg: DEFAULT_HIGH_ELEVATION_DEG });
  }
  // Including straight down.
  slots.push({ azimuthDeg: 0, elevationDeg: 90 });
  return slots;
}

// 8 + 4 + 1 = 13, within the proposal's 12-16 target views.
export const DEFAULT_ORBIT: readonly OrbitSlot[] = buildDefaultOrbit();

export function angularDistanceDeg(a: number, b: number): number {
  const diff = Math.abs(a - b) % 360;
  return diff > 180 ? 360 - diff : diff;
}

export function relativeAzimuthDeg(rawHeadingDeg: number, anchorDeg: number): number {
  return (((rawHeadingDeg - anchorDeg) % 360) + 360) % 360;
}

export interface ViewAngles {
  readonly headingDeg: number; // app azimuth convention: gamma=0 preserves heading=alpha
  readonly elevationDeg: number; // 0 = horizontal, +90 = rear camera straight down
}

function radiansToDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

function normalizeHeadingDeg(value: number): number {
  return ((value % 360) + 360) % 360;
}

// W3C Device Orientation is intrinsic Z-X'-Y''. The rear camera looks out of the back of the
// screen, i.e. along body-frame (0, 0, -1). Compute that vector after the full rotation rather
// than treating alpha/beta independently: gamma (roll) changes both heading and elevation, and
// in landscape beta/gamma exchange the roles a portrait-only shortcut would assume.
//
// For R = Rz(alpha) Rx(beta) Ry(gamma), d = R * (0, 0, -1):
//   x = -cos(alpha) sin(gamma) - sin(alpha) sin(beta) cos(gamma)
//   y = -sin(alpha) sin(gamma) + cos(alpha) sin(beta) cos(gamma)
//   z = -cos(beta) cos(gamma)
//
// ProofShape's existing azimuth convention increases with alpha when gamma=0, the opposite sign
// of the W3C compass-heading example, so heading is atan2(-x, y). Downward elevation is asin(-z).
export function viewAnglesFromGyro(reading: Gyro): ViewAngles {
  const alpha = (reading.alpha * Math.PI) / 180;
  const beta = (reading.beta * Math.PI) / 180;
  const gamma = (reading.gamma * Math.PI) / 180;

  const cosAlpha = Math.cos(alpha);
  const sinAlpha = Math.sin(alpha);
  const cosBeta = Math.cos(beta);
  const sinBeta = Math.sin(beta);
  const cosGamma = Math.cos(gamma);
  const sinGamma = Math.sin(gamma);

  const x = -cosAlpha * sinGamma - sinAlpha * sinBeta * cosGamma;
  const y = -sinAlpha * sinGamma + cosAlpha * sinBeta * cosGamma;
  const z = -cosBeta * cosGamma;

  const headingDeg = normalizeHeadingDeg(radiansToDegrees(Math.atan2(-x, y)));
  // Floating-point trig can produce 1 +/- epsilon at exact poles; clamp before asin.
  const downward = Math.max(-1, Math.min(1, -z));
  const elevationDeg = radiansToDegrees(Math.asin(downward));
  return { headingDeg, elevationDeg };
}

// Compatibility helper for synthetic fixtures that intentionally set gamma=0. Orbit coverage and
// arrow guidance use viewAnglesFromGyro above; this helper is no longer the runtime orientation
// model.
export function elevationFromBeta(betaDeg: number): number {
  return 90 - betaDeg;
}

// null when there are no accepted frames yet -- nothing to anchor azimuth to.
export function anchorHeadingDeg(acceptedReadings: readonly Gyro[]): number | null {
  const first = acceptedReadings[0];
  return first ? viewAnglesFromGyro(first).headingDeg : null;
}

// Kept as a compatibility alias for older callers/tests; despite the historical name, C-20 makes
// the anchor the camera's full roll-aware heading, not raw alpha.
export function anchorAlphaDeg(acceptedReadings: readonly Gyro[]): number | null {
  return anchorHeadingDeg(acceptedReadings);
}

// A reading counts toward its single nearest *eligible* slot only, not every slot within
// tolerance -- caught in review (TabeenRaoof and mbj1994, independently): the tolerance (15 deg)
// is wider than half the gap between rings (45/70 is a 25 deg gap, 70/90 is 20 deg), so "within
// tolerance of every nearby slot" let one reading mark two different rings covered at once,
// reaching "13/13" with fewer distinct views than the orbit is meant to require. Ranking by
// elevation distance (the dimension that was actually ambiguous between rings) and taking the
// closest fixes it without having to shrink the tolerance and lose aiming forgiveness.
function nearestEligibleSlotIndex(
  reading: Gyro,
  anchorDeg: number,
  orbit: readonly OrbitSlot[],
  toleranceDeg: number,
): number | null {
  // Plain subtraction, not angularDistanceDeg: unlike azimuth, elevation isn't a wraparound
  // quantity (computeArrowGuidance's elevationDiff uses the same plain subtraction).
  const { headingDeg, elevationDeg } = viewAnglesFromGyro(reading);
  const azimuthDeg = relativeAzimuthDeg(headingDeg, anchorDeg);

  let bestIndex: number | null = null;
  let bestElevationDist = Infinity;
  orbit.forEach((slot, index) => {
    const elevationDist = Math.abs(elevationDeg - slot.elevationDeg);
    if (elevationDist > toleranceDeg) {
      return;
    }
    const azimuthEligible =
      slot.elevationDeg >= AZIMUTH_EXEMPT_ELEVATION_DEG ||
      angularDistanceDeg(azimuthDeg, slot.azimuthDeg) <= toleranceDeg;
    if (!azimuthEligible) {
      return;
    }
    if (elevationDist < bestElevationDist) {
      bestElevationDist = elevationDist;
      bestIndex = index;
    }
  });
  return bestIndex;
}

export interface CoverageState {
  readonly covered: readonly boolean[]; // parallel to the orbit array
  readonly nextTargetIndex: number | null; // lowest-index uncovered slot; null when all covered
}

// "Next uncovered" is the lowest orbit-array index not yet covered, not nearest-to-current --
// this function is only ever given the accepted-frame history, not a live/current position to
// measure distance from. Fixed-order is simpler, fully testable without extra plumbing, and the
// orbit's own construction order (ring, then ring, then down) is already a sensible walking path.
export function computeCoverage(
  acceptedReadings: readonly Gyro[],
  orbit: readonly OrbitSlot[] = DEFAULT_ORBIT,
  toleranceDeg: number = DEFAULT_COVERAGE_TOLERANCE_DEG,
): CoverageState {
  const anchorDeg = anchorHeadingDeg(acceptedReadings);
  const covered = orbit.map(() => false);
  if (anchorDeg !== null) {
    for (const reading of acceptedReadings) {
      const index = nearestEligibleSlotIndex(reading, anchorDeg, orbit, toleranceDeg);
      if (index !== null) {
        covered[index] = true;
      }
    }
  }
  const nextTargetIndex = covered.indexOf(false);
  return { covered, nextTargetIndex: nextTargetIndex === -1 ? null : nextTargetIndex };
}

export type ArrowGuidance =
  { onTarget: true } | { onTarget: false; azimuthDeltaDeg: number; elevationDeltaDeg: number };

// Signed deltas: positive azimuthDeltaDeg means the target is clockwise (to the right) of the
// current reading; positive elevationDeltaDeg means the target is higher (tilt further toward
// straight down). The "clockwise" direction assumes W3C alpha increases in a specific rotation
// sense that hasn't been confirmed on a real device -- worth checking alongside
// elevationFromBeta's own flagged assumption during the manual check below, not just trusting
// the spec's wording.
export function computeArrowGuidance(
  liveGyro: Gyro,
  anchorHeadingDegValue: number,
  target: OrbitSlot,
  toleranceDeg: number = DEFAULT_COVERAGE_TOLERANCE_DEG,
): ArrowGuidance {
  const { headingDeg: liveHeadingDeg, elevationDeg: liveElevationDeg } =
    viewAnglesFromGyro(liveGyro);
  const elevationDiff = target.elevationDeg - liveElevationDeg;
  const elevationOnTarget = Math.abs(elevationDiff) <= toleranceDeg;

  if (target.elevationDeg >= AZIMUTH_EXEMPT_ELEVATION_DEG) {
    if (elevationOnTarget) {
      return { onTarget: true };
    }
    return { onTarget: false, azimuthDeltaDeg: 0, elevationDeltaDeg: elevationDiff };
  }

  const liveAzimuthDeg = relativeAzimuthDeg(liveHeadingDeg, anchorHeadingDegValue);
  const rawAzimuthDiff = target.azimuthDeg - liveAzimuthDeg;
  // Signed shortest path around the circle, in (-180, 180].
  const azimuthDiff = ((((rawAzimuthDiff + 180) % 360) + 360) % 360) - 180;
  const azimuthOnTarget = Math.abs(azimuthDiff) <= toleranceDeg;

  if (azimuthOnTarget && elevationOnTarget) {
    return { onTarget: true };
  }
  return { onTarget: false, azimuthDeltaDeg: azimuthDiff, elevationDeltaDeg: elevationDiff };
}
