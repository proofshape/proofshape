// C-06: catch bad frames on-device, before upload, per the proposal's own description of this
// feature ("blurry, dark or duplicate frames are rejected on the phone... milliseconds, no
// server round trip"). R-14 already returns these three frozen codes when R-02/R-04 catch a
// problem server-side, after upload; nothing on the client has computed them until now.

import type { ErrorCode, Gyro } from "./types.js";

// Not the DOM `ImageData` class: constructing one needs a real browser or jsdom/happy-dom, which
// would force an unnecessary environment dependency onto device-independent pure functions. A
// real ImageData from a canvas satisfies this structurally (extra properties like `colorSpace`
// are fine), so no adapter is needed when C-05 eventually calls these with a real captured frame.
export interface FramePixels {
  readonly data: Uint8ClampedArray; // RGBA, same byte layout as canvas ImageData.data
  readonly width: number;
  readonly height: number;
}

export type FrameGateResult = { ok: true } | { ok: false; code: ErrorCode; message: string };

// All four thresholds below are starting assumptions, not measured/calibrated values -- same
// honesty this project already applies to recon/uncertainty.py's DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM.
// Tuning them against real capture data is follow-up work this story was never going to do.
export const DEFAULT_BLUR_VARIANCE_THRESHOLD = 100;
export const DEFAULT_EXPOSURE_CLIP_FRACTION = 0.05;
export const DEFAULT_DUPLICATE_GYRO_DELTA_DEG = 2;
export const DEFAULT_DUPLICATE_IMAGE_DIFF_THRESHOLD = 5;

function toGrayscale(frame: FramePixels): Float64Array {
  const pixelCount = frame.width * frame.height;
  const gray = new Float64Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const offset = i * 4;
    const r = frame.data[offset] ?? 0;
    const g = frame.data[offset + 1] ?? 0;
    const b = frame.data[offset + 2] ?? 0;
    // Standard perceptual luminance weights.
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return gray;
}

// 3x3 Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]] over interior pixels only (border pixels
// skipped rather than clamped/replicated -- simpler, and a 1px border has no effect on the
// variance of an otherwise-meaningful sample).
function laplacianVariance(gray: Float64Array, width: number, height: number): number {
  const responses: number[] = [];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const center = gray[y * width + x] ?? 0;
      const up = gray[(y - 1) * width + x] ?? 0;
      const down = gray[(y + 1) * width + x] ?? 0;
      const left = gray[y * width + (x - 1)] ?? 0;
      const right = gray[y * width + (x + 1)] ?? 0;
      responses.push(up + down + left + right - 4 * center);
    }
  }
  if (responses.length === 0) {
    return 0;
  }
  const mean = responses.reduce((sum, value) => sum + value, 0) / responses.length;
  const variance =
    responses.reduce((sum, value) => sum + (value - mean) ** 2, 0) / responses.length;
  return variance;
}

export function checkBlur(
  frame: FramePixels,
  threshold = DEFAULT_BLUR_VARIANCE_THRESHOLD,
): FrameGateResult {
  const gray = toGrayscale(frame);
  const variance = laplacianVariance(gray, frame.width, frame.height);
  if (variance < threshold) {
    return {
      ok: false,
      code: "frame_rejected_blur",
      message: `Laplacian variance ${variance.toFixed(2)} is below the blur threshold ${String(threshold)}.`,
    };
  }
  return { ok: true };
}

export function checkExposure(
  frame: FramePixels,
  clipFraction = DEFAULT_EXPOSURE_CLIP_FRACTION,
): FrameGateResult {
  const gray = toGrayscale(frame);
  let clipped = 0;
  for (const value of gray) {
    const rounded = Math.round(value);
    if (rounded <= 0 || rounded >= 255) {
      clipped += 1;
    }
  }
  const fraction = gray.length === 0 ? 0 : clipped / gray.length;
  if (fraction > clipFraction) {
    return {
      ok: false,
      code: "frame_rejected_exposure",
      message: `${(fraction * 100).toFixed(1)}% of pixels are clipped, above the ${(clipFraction * 100).toFixed(1)}% threshold.`,
    };
  }
  return { ok: true };
}

function meanAbsoluteGrayscaleDiff(a: FramePixels, b: FramePixels): number {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(
      `checkDuplicate: frame dimensions must match (${String(a.width)}x${String(a.height)} vs ${String(b.width)}x${String(b.height)}) -- a real capture session never changes resolution mid-session.`,
    );
  }
  const grayA = toGrayscale(a);
  const grayB = toGrayscale(b);
  let total = 0;
  for (let i = 0; i < grayA.length; i++) {
    total += Math.abs((grayA[i] ?? 0) - (grayB[i] ?? 0));
  }
  return grayA.length === 0 ? 0 : total / grayA.length;
}

// Per-axis delta rather than a combined Euclidean distance: alpha/beta/gamma are different
// physical rotation axes, and combining them into one number via naive Euclidean distance would
// need a real rotation-geometry justification this story doesn't need. The max of the three
// per-axis deltas is a simple, defensible "how far did any single axis move" measure.
function maxGyroDeltaDeg(a: Gyro, b: Gyro): number {
  return Math.max(
    Math.abs(a.alpha - b.alpha),
    Math.abs(a.beta - b.beta),
    Math.abs(a.gamma - b.gamma),
  );
}

export function checkDuplicate(
  frame: FramePixels,
  gyro: Gyro,
  lastAccepted: { frame: FramePixels; gyro: Gyro } | null,
  gyroThresholdDeg = DEFAULT_DUPLICATE_GYRO_DELTA_DEG,
  imageDiffThreshold = DEFAULT_DUPLICATE_IMAGE_DIFF_THRESHOLD,
): FrameGateResult {
  if (lastAccepted === null) {
    return { ok: true };
  }
  const gyroDelta = maxGyroDeltaDeg(gyro, lastAccepted.gyro);
  const imageDiff = meanAbsoluteGrayscaleDiff(frame, lastAccepted.frame);
  // Design call: duplicate is flagged only when BOTH signals agree nothing meaningfully
  // changed, not either alone. A single signal is a weaker/noisier proxy -- gyro can jitter
  // slightly with the phone essentially still, and two visually-near-identical frames can still
  // represent a real small translation worth keeping. Requiring both to agree before discarding
  // a capture avoids false-positives that would silently waste a real, useful frame, which
  // matters more for a one-shot physical capture than being slightly less aggressive about
  // trimming genuine duplicates.
  if (gyroDelta < gyroThresholdDeg && imageDiff < imageDiffThreshold) {
    return {
      ok: false,
      code: "frame_rejected_duplicate",
      message: `Gyro delta ${gyroDelta.toFixed(2)}° and image diff ${imageDiff.toFixed(2)} are both below their thresholds -- the phone hasn't moved enough for a second photo to be worth taking.`,
    };
  }
  return { ok: true };
}
