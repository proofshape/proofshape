import { describe, expect, it } from "vitest";
import {
  describeCreateSession,
  describeFinishSession,
  describeUploadFrame,
  describeVlmFeedback,
} from "./messages.js";

describe("describeCreateSession", () => {
  it("shows the session id on success", () => {
    expect(
      describeCreateSession({
        ok: true,
        status: 201,
        body: { session_id: "sess_9f2c1d", order_code: "PO-48213-A", status: "capturing" },
      }),
    ).toBe("Session started: sess_9f2c1d");
  });

  it("shows the contract's message on a 400", () => {
    expect(
      describeCreateSession({
        ok: false,
        status: 400,
        error: { code: "order_code_unknown", message: "That order code was not recognised." },
      }),
    ).toBe("Could not start session — That order code was not recognised.");
  });
});

describe("describeUploadFrame", () => {
  it("shows the frame index on success", () => {
    expect(
      describeUploadFrame({ ok: true, status: 202, body: { accepted: true, frame_index: 14 } }),
    ).toBe("Frame accepted (frame 14)");
  });

  it("shows the contract's message on a 422", () => {
    expect(
      describeUploadFrame({
        ok: false,
        status: 422,
        error: {
          code: "frame_rejected_blur",
          message: "That frame was too blurry to use.",
          frame_indices: [14],
        },
      }),
    ).toBe("Frame rejected — That frame was too blurry to use.");
  });
});

describe("describeVlmFeedback", () => {
  it("names the earlier frame the tip is about", () => {
    expect(
      describeVlmFeedback({
        frame_index: 10,
        hint: "Hold the part by its base.",
        same_part: true,
        hands_occluding: true,
        screen_photo: false,
        lighting_adequate: true,
      }),
    ).toBe("Tip (from frame 10): Hold the part by its base.");
  });

  it("works for the reference frame, whose same_part is null", () => {
    expect(
      describeVlmFeedback({
        frame_index: 0,
        hint: "Good start.",
        same_part: null,
        hands_occluding: false,
        screen_photo: false,
        lighting_adequate: true,
      }),
    ).toBe("Tip (from frame 0): Good start.");
  });
});

describe("describeFinishSession", () => {
  it("shows the session id on success", () => {
    expect(
      describeFinishSession({
        ok: true,
        status: 202,
        body: { session_id: "sess_9f2c1d", order_code: "PO-48213-A", status: "finished" },
      }),
    ).toBe("Session finished: sess_9f2c1d");
  });

  it("shows the contract's message on a 409", () => {
    expect(
      describeFinishSession({
        ok: false,
        status: 409,
        error: { code: "too_few_frames", message: "Not enough frames were captured yet." },
      }),
    ).toBe("Could not finish session — Not enough frames were captured yet.");
  });
});
