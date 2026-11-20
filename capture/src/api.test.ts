import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createOrder,
  createSession,
  finishSession,
  getReconstruction,
  getVerdict,
  parseInspectionSpec,
  ServiceUnreachableError,
  uploadFrame,
} from "./api.js";
import { MIN_FINISH_FRAMES, startContractStub, type ContractStub } from "../stub/contract-stub.js";

const FAKE_JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46]);

// C-19: finishSession now requires MIN_FINISH_FRAMES accepted frames, matching
// recon/service.py -- used wherever a test needs a session that can actually finish.
async function uploadMinimumFrames(baseUrl: string, sessionId: string): Promise<void> {
  const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });
  for (let i = 0; i < MIN_FINISH_FRAMES; i += 1) {
    await uploadFrame(baseUrl, sessionId, frame, { alpha: 1, beta: 2, gamma: 3 });
  }
}

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

  it("creates an order through the R-34 mock route", async () => {
    const result = await createOrder(
      baseUrl,
      {
        spec: {
          features: [{ name: "mounting hole", kind: "hole", tolerance_mm: 2, critical: true }],
          assumptions: [],
        },
        placementNote: "Arrow up",
        buyerReference: "BUYER-42",
      },
      "buyer-key",
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.body.order_code).toMatch(/^PO-/);
      expect(result.body.spec).toEqual({
        features: [{ name: "mounting hole", kind: "hole", tolerance_mm: 2, critical: true }],
        assumptions: [],
      });
    }
  });

  it("parses buyer prose through the A-03 mock route", async () => {
    const result = await parseInspectionSpec(
      baseUrl,
      "The mounting holes need 2 mm; the sealing face must be flat.",
      "buyer-key",
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.body.status).toBe("parsed");
      if (result.body.status === "parsed") {
        expect(result.body.spec.features.some((feature) => feature.tolerance_mm === null)).toBe(
          true,
        );
      }
    }
  });

  it("sends the spec as a JSON multipart file when creating an order", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await createOrder(
      baseUrl,
      {
        spec: {
          features: [{ name: "overall", kind: "overall", tolerance_mm: 1, critical: false }],
          assumptions: ["manual"],
        },
      },
      "buyer-key",
    );

    const init = fetchSpy.mock.calls[0]?.[1];
    const form = init?.body;
    expect(form).toBeInstanceOf(FormData);
    const specPart = (form as FormData).get("spec");
    expect(specPart).toBeInstanceOf(Blob);
    expect(JSON.parse(await (specPart as Blob).text())).toEqual({
      features: [{ name: "overall", kind: "overall", tolerance_mm: 1, critical: false }],
      assumptions: ["manual"],
    });
    expect(new Headers(init?.headers).get("X-ProofShape-Buyer-Key")).toBe("buyer-key");
    vi.restoreAllMocks();
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

  // C-19 (#193): the mock used to finish after just one accepted frame; the real service
  // (recon/service.py's MIN_FINISH_FRAMES) needs three.
  it("returns the contract's 409 message when finishing below the minimum frame count", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });
    await uploadFrame(baseUrl, session.body.session_id, frame, { alpha: 1, beta: 2, gamma: 3 });

    const result = await finishSession(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe("too_few_frames");
    }
  });

  it("finishes successfully after the minimum accepted frames", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    await uploadMinimumFrames(baseUrl, session.body.session_id);

    const result = await finishSession(baseUrl, session.body.session_id);
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.body.status).toBe("finished");
  });

  it("returns a 409 for finishing an already-finished session", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    await uploadMinimumFrames(baseUrl, session.body.session_id);
    await finishSession(baseUrl, session.body.session_id);

    const result = await finishSession(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe("session_already_finished");
    }
  });

  it("polls reconstruction from pending to complete after finish", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    await uploadMinimumFrames(baseUrl, session.body.session_id);
    await finishSession(baseUrl, session.body.session_id);

    const pending = await getReconstruction(baseUrl, session.body.session_id);
    expect(pending.ok).toBe(true);
    if (pending.ok) expect(pending.body.status).toBe("pending");

    const complete = await getReconstruction(baseUrl, session.body.session_id);
    expect(complete.ok).toBe(true);
    if (complete.ok) {
      expect(complete.body.status).toBe("complete");
      expect(complete.body.metric).toBe(true);
      expect(complete.body.glb_url).toMatch(/\/model\.glb$/);
    }
  });

  it("returns the contract 404 before reconstruction has started", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");

    const result = await getReconstruction(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(404);
      expect(result.error.code).toBe("session_not_found");
    }
  });

  // C-19 (#193): no typed api.ts wrapper exists for model.glb -- model-viewer.ts fetches it
  // directly too, through GLTFLoader/fetch, not through api.ts.
  it("returns the contract's session_not_found fetching model.glb for a session that isn't ready", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");

    const response = await fetch(`${baseUrl}/v1/sessions/${session.body.session_id}/model.glb`);

    expect(response.status).toBe(404);
    const body = (await response.json()) as { code?: string };
    expect(body.code).toBe("session_not_found");
  });

  it("returns the contract's session_not_found fetching model.glb for an unknown session_id", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions/does-not-exist/model.glb`);

    expect(response.status).toBe(404);
    const body = (await response.json()) as { code?: string };
    expect(body.code).toBe("session_not_found");
  });

  it("returns the contract Pass verdict from getVerdict", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    stub.setVerdictScenario("pass");

    const result = await getVerdict(baseUrl, session.body.session_id);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.body.verdict).toBe("Pass");
      expect(result.body.regions.length).toBeGreaterThan(0);
    }
  });

  it("returns no_cad_uploaded as the contract 404 from getVerdict", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");

    const result = await getVerdict(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(404);
      expect(result.error.code).toBe("no_cad_uploaded");
    }
  });

  // I-07 (PR #379 review, @TabeenRaoof): getVerdict previously threw unexpectedStatus on a 422,
  // which report.ts's own generic error branch never gets a chance to show.
  it("returns tolerance_missing as the contract 422 from getVerdict, not an unexpected status", async () => {
    const session = await createSession(baseUrl, "PO-48213-A");
    if (!session.ok) throw new Error("setup: createSession failed");
    stub.setVerdictScenario("tolerance_missing");

    const result = await getVerdict(baseUrl, session.body.session_id);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(422);
      expect(result.error.code).toBe("tolerance_missing");
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

  // C-13: the deployed service is behind Lightning's bearer-token gate (D-042); the stub is not.
  // The spy wraps the real fetch (not a mock replacement) so the stub still answers every call —
  // only the request actually sent is inspected.
  describe("authorization header (C-13)", () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    function authorizationHeaderOf(init: RequestInit | undefined): string | null {
      return new Headers(init?.headers).get("Authorization");
    }

    it("attaches Authorization: Bearer <token> to every call when a token is given", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const session = await createSession(baseUrl, "PO-48213-A", "the-token");
      if (!session.ok) throw new Error("setup: createSession failed");
      const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });
      await uploadFrame(
        baseUrl,
        session.body.session_id,
        frame,
        { alpha: 1, beta: 2, gamma: 3 },
        "the-token",
      );
      await getReconstruction(baseUrl, session.body.session_id, "the-token");
      await getVerdict(baseUrl, session.body.session_id, "the-token");
      await finishSession(baseUrl, session.body.session_id, "the-token");

      expect(fetchSpy).toHaveBeenCalledTimes(5);
      for (const call of fetchSpy.mock.calls) {
        expect(authorizationHeaderOf(call[1])).toBe("Bearer the-token");
      }
    });

    it("sends no Authorization header when no token is given", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const session = await createSession(baseUrl, "PO-48213-A");
      if (!session.ok) throw new Error("setup: createSession failed");

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(authorizationHeaderOf(fetchSpy.mock.calls[0]?.[1])).toBeNull();
    });
  });
});

// C-24 (#351): what a proxy in front of an unreachable recon.service answers. Vite's dev proxy
// (the D-049 Mac path R-36 measured) gives an empty 500; a hosting proxy gives an HTML 5xx. Only
// a non-JSON 5xx is "unreachable": a non-JSON 2xx/4xx still means a broken contract, and a 5xx
// with a JSON body came from the service itself.
describe("a non-JSON 5xx means the service is unreachable (C-24, #351)", () => {
  const frame = new Blob([FAKE_JPEG_BYTES], { type: "image/jpeg" });
  const gyro = { alpha: 1, beta: 2, gamma: 3 };

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function answerEveryFetchWith(body: string, status: number, contentType = "text/plain"): void {
    vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      Promise.resolve(new Response(body, { status, headers: { "Content-Type": contentType } })),
    );
  }

  it("throws ServiceUnreachableError for Vite's empty 500 on every session call", async () => {
    answerEveryFetchWith("", 500);
    await expect(createSession("http://svc", "PO-1")).rejects.toBeInstanceOf(
      ServiceUnreachableError,
    );
    await expect(uploadFrame("http://svc", "sess_1", frame, gyro)).rejects.toBeInstanceOf(
      ServiceUnreachableError,
    );
    await expect(finishSession("http://svc", "sess_1")).rejects.toBeInstanceOf(
      ServiceUnreachableError,
    );
    await expect(getReconstruction("http://svc", "sess_1")).rejects.toBeInstanceOf(
      ServiceUnreachableError,
    );
  });

  it("throws ServiceUnreachableError for a proxy's HTML 502, and keeps the status", async () => {
    answerEveryFetchWith("<html><body>Bad Gateway</body></html>", 502, "text/html");
    const error: unknown = await uploadFrame("http://svc", "sess_1", frame, gyro).catch(
      (thrown: unknown) => thrown,
    );
    expect(error).toBeInstanceOf(ServiceUnreachableError);
    expect((error as ServiceUnreachableError).status).toBe(502);
    expect((error as ServiceUnreachableError).operation).toBe("uploadFrame");
  });

  it("still fails loudly as a broken contract for a non-JSON 2xx", async () => {
    answerEveryFetchWith("not json", 202);
    const error: unknown = await uploadFrame("http://svc", "sess_1", frame, gyro).catch(
      (thrown: unknown) => thrown,
    );
    expect(error).toBeInstanceOf(Error);
    expect(error).not.toBeInstanceOf(ServiceUnreachableError);
    expect((error as Error).message).toMatch(/response body was not valid JSON \(status 202\)/);
  });

  it("does not call a 5xx with a JSON body unreachable: that answer came from the service", async () => {
    answerEveryFetchWith(
      JSON.stringify({ code: "internal_error", message: "boom" }),
      500,
      "application/json",
    );
    const error: unknown = await uploadFrame("http://svc", "sess_1", frame, gyro).catch(
      (thrown: unknown) => thrown,
    );
    expect(error).toBeInstanceOf(Error);
    expect(error).not.toBeInstanceOf(ServiceUnreachableError);
  });
});
