// C-27: decides when the capture screen takes a photo by itself. Pure and timer-free: the shell
// feeds it every guidance update (one per deviceorientation event, ~60/s on an iPhone) with the
// current time, and it answers whether to capture now. It never captures by itself and knows
// nothing about the gates, the upload or the session -- the shell checks those (the same
// conditions that enable the shutter) before acting on `fire`.

import { angularDistanceDeg, type GuidanceLevel } from "./orbit.js";
import type { Gyro } from "./types.js";

// How long the reading must stay on target before a photo is taken.
export const AUTO_DWELL_MS = 600;
// How far (per gyro component) the phone may drift during that hold and still count as steady.
export const AUTO_STEADY_DEG = 3;
// Minimum gap between automatic attempts, so a frame the gate rejects (blur, exposure) is retried
// at a calm pace instead of on every orientation event.
export const AUTO_RETRY_MS = 1500;

export interface AutoCaptureState {
  holdStartMs: number | null;
  holdStartGyro: Gyro | null;
  lastAttemptMs: number | null;
}

export const INITIAL_AUTO_CAPTURE_STATE: AutoCaptureState = {
  holdStartMs: null,
  holdStartGyro: null,
  lastAttemptMs: null,
};

export interface AutoCaptureInput {
  nowMs: number;
  level: GuidanceLevel;
  gyro: Gyro | null;
  enabled: boolean;
}

export interface AutoCaptureStep {
  state: AutoCaptureState;
  fire: boolean;
}

function maxComponentDriftDeg(a: Gyro, b: Gyro): number {
  return Math.max(
    angularDistanceDeg(a.alpha, b.alpha),
    angularDistanceDeg(a.beta, b.beta),
    angularDistanceDeg(a.gamma, b.gamma),
  );
}

export function stepAutoCapture(state: AutoCaptureState, input: AutoCaptureInput): AutoCaptureStep {
  const { nowMs, level, gyro, enabled } = input;
  // Only "on-target" can start or continue a hold. "idle" (no first photo yet, so no target) and
  // "complete" (nothing left to cover) never capture automatically.
  if (!enabled || level !== "on-target" || gyro === null) {
    return { state: { ...state, holdStartMs: null, holdStartGyro: null }, fire: false };
  }
  if (
    state.holdStartMs === null ||
    state.holdStartGyro === null ||
    maxComponentDriftDeg(gyro, state.holdStartGyro) > AUTO_STEADY_DEG
  ) {
    return { state: { ...state, holdStartMs: nowMs, holdStartGyro: gyro }, fire: false };
  }
  const held = nowMs - state.holdStartMs >= AUTO_DWELL_MS;
  const rested = state.lastAttemptMs === null || nowMs - state.lastAttemptMs >= AUTO_RETRY_MS;
  if (held && rested) {
    return { state: { holdStartMs: null, holdStartGyro: null, lastAttemptMs: nowMs }, fire: true };
  }
  return { state, fire: false };
}
