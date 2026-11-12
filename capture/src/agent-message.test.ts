import { afterEach, beforeEach, describe, expect, it } from "vitest";
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
    await close();
  });

  it("returns a supplier-facing mock message", async () => {
    const result = await getAgentMessage(baseUrl, sessionId);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body).toEqual({
      message: "Capture your first guided frame.",
      fallback: false,
    });
  });

  it("returns deterministic fallback state when the agent is unavailable", async () => {
    stub.setAgentFallback(true, "model_timeout");
    const result = await getAgentMessage(baseUrl, sessionId);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.body).toEqual({
      message: null,
      fallback: true,
      fallback_reason: "model_timeout",
    });
  });
});
