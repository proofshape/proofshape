import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createSession, finishSession, uploadFrame } from "./api.js";
import { startContractStub, type ContractStub } from "../stub/contract-stub.js";

const FAKE_JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46]);

describe("api against the contract stub", () => {
  let baseUrl: string;
  let stub: ContractStub;
  let close: () => Promise<void>;

  beforeEach(async () => {
    const started = await startContractStub();
    baseUrl = started.baseUrl;
    stub = started.stub;
    close = started.close;
  });

  afterEach(async () => {
    await close();
  });

  it("returns a session id for the known order code", async () => {
    const result = await createSession(baseUrl, "PO-48213-A");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.body.session_id).toMatch(/^sess_/);
      expect(result.body.status).toBe("capturing");
    }
  });

  it("returns the contract's 400 message for an unknown order code", async () => {
    const result = await createSession(baseUrl, "PO-NOT-REAL");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe("order_code_unknown");
    }
  });

  it("accepts a frame and returns an incrementing index", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });

    const first = await uploadFrame(baseUrl, session.body.session_id, frame, {
      alpha: 1,
      beta: 2,
      gamma: 3,
    });
    expect(first.ok).toBe(true);
    if (first.ok) expect(first.body.frame_index).toBe(0);

    const second = await uploadFrame(baseUrl, session.body.session_id, frame, {
      alpha: 1,
      beta: 2,
      gamma: 3,
    });
    expect(second.ok).toBe(true);
    if (second.ok) expect(second.body.frame_index).toBe(1);
  });

  it("returns the contract's 422 message when the stub rejects the frame", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    stub.rejectNextFrame();
    const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });

    const result = await uploadFrame(baseUrl, session.body.session_id, frame, {
      alpha: 1,
      beta: 2,
      gamma: 3,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe("frame_rejected_blur");
      expect(result.error.frame_indices).toEqual([0]);
    }
  });

  it("returns the contract's 409 message when finishing before any accepted frame", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");

    const result = await finishSession(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe("too_few_frames");
    }
  });

  it("finishes successfully after at least one accepted frame", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });
    await uploadFrame(baseUrl, session.body.session_id, frame, { alpha: 1, beta: 2, gamma: 3 });

    const result = await finishSession(baseUrl, session.body.session_id);
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.body.status).toBe("finished");
  });

  it("returns a 409 for finishing an already-finished session", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });
    await uploadFrame(baseUrl, session.body.session_id, frame, { alpha: 1, beta: 2, gamma: 3 });
    await finishSession(baseUrl, session.body.session_id);

    const result = await finishSession(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe("session_already_finished");
    }
  });

  it("fails loudly, not as a contract status, when the frame isn't a JPEG", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    const notAJpeg = new Blob([new Uint8Array([0, 1, 2, 3])], { type: "image/jpeg" });

    await expect(
      uploadFrame(baseUrl, session.body.session_id, notAJpeg, { alpha: 1, beta: 2, gamma: 3 }),
    ).rejects.toThrow(/unexpected status 500/);
  });
});
