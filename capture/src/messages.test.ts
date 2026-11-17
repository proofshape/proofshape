import { describe, expect, it } from "vitest";
import { ServiceUnreachableError } from "./api.js";
import {
  describeContinuedSession,
  describeCreateSession,
  describeFinishSession,
  describeNetworkError,
  describeSavedSession,
  describeUploadError,
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

describe("describeNetworkError", () => {
  it("gives an actionable message for a fetch TypeError, regardless of the browser's own wording", () => {
    expect(describeNetworkError(new TypeError("Load failed"))).toBe(
      "No connection to the service — check your network and try again.",
    );
    expect(describeNetworkError(new TypeError("Failed to fetch"))).toBe(
      "No connection to the service — check your network and try again.",
    );
  });

  // C-24 (#351) changed what reaches this pass-through on purpose: a non-JSON *5xx* is now a
  // ServiceUnreachableError (next test), so the raw "not valid JSON" text a supplier saw during
  // R-36's service restart no longer gets here. A non-JSON 2xx/4xx is still a broken contract,
  // still a plain Error, and its own specific message is still the right thing to show.
  it("passes through a broken-contract Error's own message unchanged", () => {
    expect(
      describeNetworkError(
        new Error("createSession: response body was not valid JSON (status 201)"),
      ),
    ).toBe("createSession: response body was not valid JSON (status 201)");
  });

  it("says to wait and retry when the service is unreachable, instead of the parser's text", () => {
    expect(describeNetworkError(new ServiceUnreachableError("uploadFrame", 500))).toBe(
      "The service isn't reachable right now — wait a moment and try again.",
    );
  });

  it("stringifies a non-Error throw", () => {
    expect(describeNetworkError("plain string throw")).toBe("plain string throw");
  });
});

describe("describeUploadError (C-24)", () => {
  it("adds that accepted frames are kept when the service is unreachable", () => {
    expect(describeUploadError(new ServiceUnreachableError("uploadFrame", 500))).toBe(
      "The service isn't reachable right now — wait a moment and try again." +
        " Frames already accepted are kept.",
    );
  });

  it("adds the same for a dropped connection", () => {
    expect(describeUploadError(new TypeError("Load failed"))).toBe(
      "No connection to the service — check your network and try again." +
        " Frames already accepted are kept.",
    );
  });

  it("says nothing about frames for a broken-contract error", () => {
    expect(describeUploadError(new Error("uploadFrame: unexpected status 418"))).toBe(
      "uploadFrame: unexpected status 418",
    );
  });
});

describe("resume messages (C-24)", () => {
  const saved = {
    sessionId: "sess_2e59418b1077",
    orderCode: "PO-YH4E183E",
    phase: "capturing" as const,
    acceptedFrameCount: 5,
    acceptedGyro: [],
  };

  it("names an unfinished session, its order and its frames, and offers both choices", () => {
    expect(describeSavedSession(saved)).toBe(
      "An unfinished session is saved in this tab: sess_2e59418b1077 for PO-YH4E183E," +
        " 5 frames accepted. Continue it, or start a new one.",
    );
  });

  it("uses the singular for one frame", () => {
    expect(describeSavedSession({ ...saved, acceptedFrameCount: 1 })).toContain("1 frame accepted");
    expect(describeContinuedSession({ ...saved, acceptedFrameCount: 1 })).toBe(
      "Continuing session sess_2e59418b1077 (1 frame accepted so far).",
    );
  });

  it("says a finished session's reconstruction may still be running", () => {
    expect(describeSavedSession({ ...saved, phase: "finished" })).toBe(
      "Session sess_2e59418b1077 for PO-YH4E183E was finished." +
        " Its reconstruction may still be running.",
    );
  });
});
