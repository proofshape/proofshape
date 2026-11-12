import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createSession, getAgentMessage } from "./api.js";
import { startContractStub, type ContractStub } from "../stub/contract-stub.js";

describe("C-22 mock agent message route", () => {
  let close: () => Promise<void>;
  let baseUrl: string;
  let stub: ContractStub;
  let sessionId: string;

  beforeEach(async () => {
    const started = await startContractStub();
    baseUrl = started.baseUrl;
    close = started.close;
    stub = started.stub;

    const created = await createSession(baseUrl, "PO-48213-A");
    if (!created.ok) throw new Error("test session did not start");
    sessionId = created.body.session_id;
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    await close();
  });

  it("returns a supplier-facing mock message using A-07's result shape", async () => {
    const result = await getAgentMessage(baseUrl, sessionId);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body).toEqual({
      finished: false,
      fallback_reason: null,
      message: "Capture your first guided frame.",
    });
  });

  it("returns A-07 deterministic fallback narration and reason", async () => {
    stub.setAgentFallback(true, "timeout");
    const result = await getAgentMessage(baseUrl, sessionId);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body).toEqual({
      finished: false,
      fallback_reason: "timeout",
      message: "Keep going -- more of the part still needs to be photographed.",
    });
  });

  it("aborts a stalled agent request at the configured deadline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      }),
    );

    await expect(getAgentMessage(baseUrl, sessionId, undefined, 5)).rejects.toMatchObject({
      name: "AbortError",
    });
  });
});
