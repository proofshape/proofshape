import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { startContractStub, type ContractStub } from "./contract-stub.js";

// See contract-stub.ts for why this is node:path, not `new URL(x, import.meta.url)`.
const EXAMPLES_DIR = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "contracts",
  "examples",
);

function loadExample(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(EXAMPLES_DIR, `${name}.json`), "utf-8")) as Record<
    string,
    unknown
  >;
}

describe("contract-stub shapes match the committed examples", () => {
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

  it("createSession 201 has the same keys as the committed example", async () => {
    const example = loadExample("createSession.response-201");
    const response = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    expect(response.status).toBe(201);
    const body = (await response.json()) as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual(Object.keys(example).sort());
    expect(body["status"]).toBe(example["status"]);
  });

  it("createSession 400 matches the committed example exactly", async () => {
    const example = loadExample("createSession.response-400");
    const response = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "not-the-known-code" }),
    });
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual(example);
  });

  it("uploadFrame 422 reuses the committed example's code and message", async () => {
    const example = loadExample("uploadFrame.response-422");
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    stub.rejectNextFrame();
    const form = new FormData();
    form.set("frame", new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" }));
    form.set("gyro", new Blob([JSON.stringify({ alpha: 1, beta: 2, gamma: 3 })]));
    const response = await fetch(`${baseUrl}/v1/sessions/${sessionId}/frames`, {
      method: "POST",
      body: form,
    });
    expect(response.status).toBe(422);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body["code"]).toBe(example["code"]);
    expect(body["message"]).toBe(example["message"]);
  });

  it("finishSession 409 (too few frames) matches the committed example exactly", async () => {
    const example = loadExample("finishSession.response-409");
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    const response = await fetch(`${baseUrl}/v1/sessions/${sessionId}/finish`, { method: "POST" });
    expect(response.status).toBe(409);
    expect(await response.json()).toEqual(example);
  });


  it("serves pending twice, then the canned reconstruction and GLB fixture", async () => {
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    const form = new FormData();
    form.set("frame", new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" }));
    form.set("gyro", new Blob([JSON.stringify({ alpha: 1, beta: 2, gamma: 3 })]));
    await fetch(`${baseUrl}/v1/sessions/${sessionId}/frames`, { method: "POST", body: form });
    await fetch(`${baseUrl}/v1/sessions/${sessionId}/finish`, { method: "POST" });

    stub.setReconstructionPendingPolls(2);
    const first = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);
    const second = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);
    const third = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);
    expect((await first.json()) as Record<string, unknown>).toEqual({ status: "pending" });
    expect((await second.json()) as Record<string, unknown>).toEqual({ status: "pending" });

    const complete = (await third.json()) as Record<string, unknown>;
    const example = loadExample("getReconstruction.response-200");
    expect(complete["status"]).toBe("complete");
    expect(complete["reference_tier_used"]).toBe(example["reference_tier_used"]);
    expect(complete["metric"]).toBe(example["metric"]);
    expect(complete["observed_fraction"]).toBe(example["observed_fraction"]);

    const glbUrl = complete["glb_url"];
    expect(typeof glbUrl).toBe("string");
    const glbResponse = await fetch(glbUrl as string);
    expect(glbResponse.status).toBe(200);
    expect(glbResponse.headers.get("content-type")).toBe("model/gltf-binary");
    const bytes = new Uint8Array(await glbResponse.arrayBuffer());
    expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe("glTF");
  });

  it("fails loudly on a request missing order_code, instead of guessing", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(response.status).toBe(500);
  });

  it("fails loudly on an unknown session_id, instead of guessing", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions/does-not-exist/finish`, {
      method: "POST",
    });
    expect(response.status).toBe(500);
  });
});
