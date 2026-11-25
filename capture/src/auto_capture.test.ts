import { describe, expect, it } from "vitest";
import {
  AUTO_DWELL_MS,
  AUTO_RETRY_MS,
  AUTO_STEADY_DEG,
  INITIAL_AUTO_CAPTURE_STATE,
  stepAutoCapture,
  type AutoCaptureInput,
  type AutoCaptureState,
} from "./auto_capture.js";
import type { GuidanceLevel } from "./orbit.js";
import type { Gyro } from "./types.js";

// No happy-dom: auto_capture.ts is pure.

const HERE: Gyro = { alpha: 45, beta: 45, gamma: 0 };

// Feeds a sequence of [timeMs, gyro, level?] readings through stepAutoCapture and returns the
// times at which it fired.
function run(
  readings: readonly (readonly [number, Gyro, GuidanceLevel?])[],
  options: { enabled?: boolean; start?: AutoCaptureState } = {},
): { fired: number[]; state: AutoCaptureState } {
  let state = options.start ?? INITIAL_AUTO_CAPTURE_STATE;
  const fired: number[] = [];
  for (const [nowMs, gyro, level] of readings) {
    const input: AutoCaptureInput = {
      nowMs,
      gyro,
      level: level ?? "on-target",
      enabled: options.enabled ?? true,
    };
    const step = stepAutoCapture(state, input);
    state = step.state;
    if (step.fire) fired.push(nowMs);
  }
  return { fired, state };
}

function every(stepMs: number, untilMs: number, gyro: Gyro = HERE): [number, Gyro][] {
  const out: [number, Gyro][] = [];
  for (let t = 0; t <= untilMs; t += stepMs) out.push([t, gyro]);
  return out;
}

describe("stepAutoCapture", () => {
  it("does not fire before the hold reaches AUTO_DWELL_MS, and fires when it does", () => {
    expect(AUTO_DWELL_MS).toBe(600);
    expect(run(every(100, 599)).fired).toEqual([]);
    expect(run([...every(100, 500), [600, HERE]]).fired).toEqual([600]);
  });

  it("restarts the hold when the phone drifts more than AUTO_STEADY_DEG, but not at exactly it", () => {
    expect(AUTO_STEADY_DEG).toBe(3);
    const drifted = { ...HERE, beta: HERE.beta + 3.01 };
    // Drift at 400 ms restarts the hold there, so nothing by 600 ms; it fires 600 ms later.
    expect(
      run([
        [0, HERE],
        [400, drifted],
        [600, drifted],
        [999, drifted],
      ]).fired,
    ).toEqual([]);
    expect(
      run([
        [0, HERE],
        [400, drifted],
        [1000, drifted],
      ]).fired,
    ).toEqual([1000]);
    const atLimit = { ...HERE, alpha: HERE.alpha + 3 };
    expect(
      run([
        [0, HERE],
        [300, atLimit],
        [600, atLimit],
      ]).fired,
    ).toEqual([600]);
  });

  it("measures drift around the wrap, so 359 -> 1 degree is a 2 degree move", () => {
    const a = { alpha: 359, beta: 45, gamma: 0 };
    const b = { alpha: 1, beta: 45, gamma: 0 };
    expect(
      run([
        [0, a],
        [600, b],
      ]).fired,
    ).toEqual([600]);
  });

  it("resets the hold when the reading leaves the target", () => {
    expect(
      run([
        [0, HERE],
        [300, HERE, "near"],
        [600, HERE],
        [1100, HERE],
      ]).fired,
    ).toEqual([]);
    expect(
      run([
        [0, HERE],
        [300, HERE, "near"],
        [600, HERE],
        [1200, HERE],
      ]).fired,
    ).toEqual([1200]);
  });

  it("waits AUTO_RETRY_MS after an attempt before the next one, even if held on target", () => {
    expect(AUTO_RETRY_MS).toBe(1500);
    const { fired } = run(every(50, 2700));
    // First at 600; the hold restarts after firing (650 -> 1250 held) but the retry gap holds it
    // until 2100; the next hold restarts at 2150 and would complete at 2750, past this run.
    expect(fired).toEqual([600, 2100]);
  });

  it("never fires when disabled, before the first photo (idle), or once the orbit is complete", () => {
    expect(run(every(100, 3000), { enabled: false }).fired).toEqual([]);
    for (const level of ["idle", "complete", "far", "near"] as const) {
      const readings = every(100, 3000).map(([t, g]) => [t, g, level] as const);
      expect(run(readings).fired).toEqual([]);
    }
  });

  it("does not fire without a live reading", () => {
    const step = stepAutoCapture(INITIAL_AUTO_CAPTURE_STATE, {
      nowMs: 1000,
      level: "on-target",
      gyro: null,
      enabled: true,
    });
    expect(step.fire).toBe(false);
  });
});
