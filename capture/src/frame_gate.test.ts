import { describe, expect, it } from "vitest";
import {
  checkBlur,
  checkDuplicate,
  checkExposure,
  measureBlurVariance,
  DEFAULT_BLUR_VARIANCE_THRESHOLD,
  DEFAULT_DUPLICATE_GYRO_DELTA_DEG,
  DEFAULT_DUPLICATE_IMAGE_DIFF_THRESHOLD,
  DEFAULT_EXPOSURE_CLIP_FRACTION,
  type FramePixels,
} from "./frame_gate.js";
import type { Gyro } from "./types.js";

// No `// @vitest-environment happy-dom` needed -- frame_gate.ts is pure math over typed arrays,
// no DOM involved, which is the whole point of this story.

function makeFrame(
  width: number,
  height: number,
  gray: (x: number, y: number) => number,
): FramePixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = gray(x, y);
      const offset = (y * width + x) * 4;
      data[offset] = value;
      data[offset + 1] = value;
      data[offset + 2] = value;
      data[offset + 3] = 255;
    }
  }
  return { data, width, height };
}

const FLAT_GRAY = makeFrame(8, 8, () => 128);
const CHECKERBOARD = makeFrame(8, 8, (x, y) => ((x + y) % 2 === 0 ? 255 : 0));
const FLAT_WHITE = makeFrame(8, 8, () => 255);

const GYRO_A: Gyro = { alpha: 10, beta: 20, gamma: 30 };

function applyDiagonalMotionBlur(frame: FramePixels, kernelWidth: number): FramePixels {
  if (kernelWidth < 1 || kernelWidth % 2 === 0) {
    throw new Error("motion-blur kernel width must be a positive odd integer");
  }
  const radius = Math.floor(kernelWidth / 2);
  const data = new Uint8ClampedArray(frame.data.length);
  for (let y = 0; y < frame.height; y++) {
    for (let x = 0; x < frame.width; x++) {
      const targetOffset = (y * frame.width + x) * 4;
      for (let channel = 0; channel < 3; channel++) {
        let total = 0;
        for (let offset = -radius; offset <= radius; offset++) {
          const sourceX = Math.min(frame.width - 1, Math.max(0, x + offset));
          const sourceY = Math.min(frame.height - 1, Math.max(0, y + offset));
          const sourceOffset = (sourceY * frame.width + sourceX) * 4;
          total += frame.data[sourceOffset + channel] ?? 0;
        }
        data[targetOffset + channel] = Math.round(total / kernelWidth);
      }
      data[targetOffset + 3] = 255;
    }
  }
  return { data, width: frame.width, height: frame.height };
}

describe("checkBlur", () => {
  it("passes a sharp, high-contrast image", () => {
    expect(checkBlur(CHECKERBOARD)).toEqual({ ok: true });
  });

  it("rejects a flat, uniform image with frame_rejected_blur", () => {
    const result = checkBlur(FLAT_GRAY);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe("frame_rejected_blur");
    }
  });

  it("honors an overridden threshold", () => {
    // Default threshold (100) rejects the flat image (variance 0); a threshold of 0 does not,
    // since 0 is not strictly less than 0 -- demonstrates the override is actually read, not
    // just accepted and ignored.
    expect(checkBlur(FLAT_GRAY, DEFAULT_BLUR_VARIANCE_THRESHOLD).ok).toBe(false);
    expect(checkBlur(FLAT_GRAY, 0).ok).toBe(true);
  });

  it("rejects directional motion blur at the calibrated threshold", () => {
    const board = makeFrame(64, 64, (x, y) =>
      (Math.floor(x / 16) + Math.floor(y / 16)) % 2 === 0 ? 255 : 0,
    );
    const motionBlurred = applyDiagonalMotionBlur(board, 31);
    const sharpVariance = measureBlurVariance(board);
    const blurredVariance = measureBlurVariance(motionBlurred);

    // C-25's paired real-device run established that JPEG encoding shifts the metric only about
    // 1–2% upward and preserves ordering, so the R-36 blur/sharp gap supports the calibrated 200
    // threshold. This synthetic regression now pins the intended reject behavior.
    expect(checkBlur(board)).toEqual({ ok: true });
    expect(blurredVariance).toBeLessThan(sharpVariance);
    expect(blurredVariance).toBeLessThan(DEFAULT_BLUR_VARIANCE_THRESHOLD);
    const result = checkBlur(motionBlurred);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe("frame_rejected_blur");
    }
  });
});

describe("checkExposure", () => {
  it("passes a mid-range, unclipped image", () => {
    expect(checkExposure(FLAT_GRAY)).toEqual({ ok: true });
  });

  it("rejects a fully clipped image with frame_rejected_exposure", () => {
    const result = checkExposure(FLAT_WHITE);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe("frame_rejected_exposure");
    }
  });

  it("honors an overridden clip fraction", () => {
    expect(checkExposure(FLAT_WHITE, DEFAULT_EXPOSURE_CLIP_FRACTION).ok).toBe(false);
    expect(checkExposure(FLAT_WHITE, 1).ok).toBe(true);
  });
});

describe("checkDuplicate", () => {
  it("always passes the first frame of a session (nothing to compare against)", () => {
    expect(checkDuplicate(FLAT_GRAY, GYRO_A, null)).toEqual({ ok: true });
  });

  it("rejects an identical frame with an unchanged gyro reading", () => {
    const result = checkDuplicate(FLAT_GRAY, GYRO_A, { frame: FLAT_GRAY, gyro: GYRO_A });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe("frame_rejected_duplicate");
    }
  });

  it("wraps a 359/1 heading pair instead of treating it as a 358-degree move", () => {
    const before: Gyro = { alpha: 359, beta: 20, gamma: 30 };
    const after: Gyro = { alpha: 1, beta: 20, gamma: 30 };

    // The wrapped alpha delta is 2 degrees. Use a 2.1-degree threshold so the strict
    // "below threshold" duplicate rule is exercised without changing that existing boundary.
    const result = checkDuplicate(FLAT_GRAY, after, { frame: FLAT_GRAY, gyro: before }, 2.1);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.code).toBe("frame_rejected_duplicate");
    }
  });

  it("passes when the gyro moved meaningfully, even if the image looks identical", () => {
    const movedGyro: Gyro = { ...GYRO_A, alpha: GYRO_A.alpha + 10 };
    expect(checkDuplicate(FLAT_GRAY, movedGyro, { frame: FLAT_GRAY, gyro: GYRO_A }).ok).toBe(true);
  });

  it("passes when the image changed meaningfully, even if the gyro looks identical (AND, not OR)", () => {
    const changedFrame = makeFrame(8, 8, () => 138); // mean abs diff of 10 from FLAT_GRAY's 128
    expect(checkDuplicate(changedFrame, GYRO_A, { frame: FLAT_GRAY, gyro: GYRO_A }).ok).toBe(true);
  });

  it("throws on mismatched frame dimensions -- a caller bug, not a real capture scenario", () => {
    const wrongSize = makeFrame(4, 4, () => 128);
    expect(() => checkDuplicate(wrongSize, GYRO_A, { frame: FLAT_GRAY, gyro: GYRO_A })).toThrow(
      /dimensions must match/,
    );
  });

  it("honors overridden thresholds", () => {
    const movedGyro: Gyro = { ...GYRO_A, alpha: GYRO_A.alpha + 10 };
    // Default gyro threshold (2) does not flag this as a duplicate -- the gyro moved too far.
    expect(
      checkDuplicate(
        FLAT_GRAY,
        movedGyro,
        { frame: FLAT_GRAY, gyro: GYRO_A },
        DEFAULT_DUPLICATE_GYRO_DELTA_DEG,
      ).ok,
    ).toBe(true);
    // A wider gyro threshold (20) now tolerates that same delta, and the image is unchanged
    // (diff 0, below the default image-diff threshold), so both signals agree -- duplicate.
    expect(checkDuplicate(FLAT_GRAY, movedGyro, { frame: FLAT_GRAY, gyro: GYRO_A }, 20).ok).toBe(
      false,
    );
    expect(DEFAULT_DUPLICATE_IMAGE_DIFF_THRESHOLD).toBeGreaterThan(0);
  });
});
