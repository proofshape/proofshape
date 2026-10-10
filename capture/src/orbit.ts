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

export function relativeAzimuthDeg(rawAlphaDeg: number, anchorDeg: number): number {
  return (((rawAlphaDeg - anchorDeg) % 360) + 360) % 360;
}

// beta: W3C front-back tilt, 0 deg when the device lies flat screen-up, ~90 deg when held
// upright facing the user. The normal "mid elevation" photo pose -- standing, rear camera
// pointing roughly horizontally at the part -- is beta ~= 90; tilting the phone forward/down to
// point the camera at the part from above moves beta toward 0. This is a defensible first pass,
// not a verified one -- isolated here specifically so it's a one-line fix if the real-device
// manual check shows it's backwards.
export function elevationFromBeta(betaDeg: number): number {
  return 90 - betaDeg;
}

// null when there are no accepted frames yet -- nothing to anchor azimuth to.
export function anchorAlphaDeg(acceptedReadings: readonly Gyro[]): number | null {
  return acceptedReadings[0]?.alpha ?? null;
}

function slotMatches(
  reading: Gyro,
  anchorDeg: number,
  slot: OrbitSlot,
  toleranceDeg: number,
): boolean {
  // Plain subtraction, not angularDistanceDeg: unlike azimuth, elevation isn't a wraparound
  // quantity (computeArrowGuidance's elevationDiff uses the same plain subtraction).
  const elevationDeg = elevationFromBeta(reading.beta);
  if (Math.abs(elevationDeg - slot.elevationDeg) > toleranceDeg) {
    return false;
  }
  if (slot.elevationDeg >= AZIMUTH_EXEMPT_ELEVATION_DEG) {
    return true;
  }
  const azimuthDeg = relativeAzimuthDeg(reading.alpha, anchorDeg);
  return angularDistanceDeg(azimuthDeg, slot.azimuthDeg) <= toleranceDeg;
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
  const anchorDeg = anchorAlphaDeg(acceptedReadings);
  const covered = orbit.map((slot) => {
    if (anchorDeg === null) {
      return false;
    }
    return acceptedReadings.some((reading) => slotMatches(reading, anchorDeg, slot, toleranceDeg));
  });
  const nextTargetIndex = covered.indexOf(false);
  return { covered, nextTargetIndex: nextTargetIndex === -1 ? null : nextTargetIndex };
}

export type ArrowGuidance =
  { onTarget: true } | { onTarget: false; azimuthDeltaDeg: number; elevationDeltaDeg: number };

// Signed deltas: positive azimuthDeltaDeg means the target is clockwise (to the right) of the
// current reading; positive elevationDeltaDeg means the target is higher (tilt further toward
// straight down).
export function computeArrowGuidance(
  liveGyro: Gyro,
  anchorAlphaDegValue: number,
  target: OrbitSlot,
  toleranceDeg: number = DEFAULT_COVERAGE_TOLERANCE_DEG,
): ArrowGuidance {
  const liveElevationDeg = elevationFromBeta(liveGyro.beta);
  const elevationDiff = target.elevationDeg - liveElevationDeg;
  const elevationOnTarget = Math.abs(elevationDiff) <= toleranceDeg;

  if (target.elevationDeg >= AZIMUTH_EXEMPT_ELEVATION_DEG) {
    if (elevationOnTarget) {
      return { onTarget: true };
    }
    return { onTarget: false, azimuthDeltaDeg: 0, elevationDeltaDeg: elevationDiff };
  }

  const liveAzimuthDeg = relativeAzimuthDeg(liveGyro.alpha, anchorAlphaDegValue);
  const rawAzimuthDiff = target.azimuthDeg - liveAzimuthDeg;
  // Signed shortest path around the circle, in (-180, 180].
  const azimuthDiff = ((((rawAzimuthDiff + 180) % 360) + 360) % 360) - 180;
  const azimuthOnTarget = Math.abs(azimuthDiff) <= toleranceDeg;

  if (azimuthOnTarget && elevationOnTarget) {
    return { onTarget: true };
  }
  return { onTarget: false, azimuthDeltaDeg: azimuthDiff, elevationDeltaDeg: elevationDiff };
}
