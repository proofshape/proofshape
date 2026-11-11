import { existsSync, readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  MIN_FINISH_FRAMES,
  createContractStub,
  startContractStub,
  type ContractStub,
} from "./contract-stub.js";

// C-19: same pattern as provenance-colors.test.ts's findProvenanceSource -- walks up from cwd
// to find the repo-root file, since a shared constant can't cross Python/TypeScript directly.
function findReconServiceSource(): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = join(directory, "recon", "service.py");
    if (existsSync(candidate)) {
      return candidate;
    }
    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error("Could not find recon/service.py above the current working directory.");
    }
    directory = parent;
  }
}

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

// C-19: finishSession now requires MIN_FINISH_FRAMES (3) accepted frames, matching
// recon/service.py -- used wherever a test needs a session that can actually finish.
async function uploadFrame(baseUrl: string, sessionId: string): Promise<Response> {
  const form = new FormData();
  form.set("frame", new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" }));
  form.set("gyro", new Blob([JSON.stringify({ alpha: 1, beta: 2, gamma: 3 })]));
  return fetch(`${baseUrl}/v1/sessions/${sessionId}/frames`, { method: "POST", body: form });
}

async function uploadMinimumFrames(baseUrl: string, sessionId: string): Promise<void> {
  for (let i = 0; i < 3; i += 1) {
    await uploadFrame(baseUrl, sessionId);
  }
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

  it("serves pending twice, then a renderable canned GLB", async () => {
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    await uploadMinimumFrames(baseUrl, sessionId);
    await fetch(`${baseUrl}/v1/sessions/${sessionId}/finish`, { method: "POST" });

    stub.setReconstructionPendingPolls(2);
    const first = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);
    const second = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);
    const third = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);

    expect(await first.json()).toEqual({ status: "pending" });
    expect(await second.json()).toEqual({ status: "pending" });

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
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(view.getUint32(4, true)).toBe(2);
    const jsonLength = view.getUint32(12, true);
    const jsonText = new TextDecoder().decode(bytes.slice(20, 20 + jsonLength)).trim();
    const gltf = JSON.parse(jsonText) as {
      asset?: { version?: string };
      meshes?: unknown[];
      scenes?: unknown[];
    };
    expect(gltf.asset?.version).toBe("2.0");
    expect(gltf.meshes?.length).toBeGreaterThan(0);
    expect(gltf.scenes?.length).toBeGreaterThan(0);
  });

  it("preserves HTTPS in the canned GLB URL behind the device dev server", async () => {
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    await uploadMinimumFrames(baseUrl, sessionId);
    await fetch(`${baseUrl}/v1/sessions/${sessionId}/finish`, { method: "POST" });

    stub.setReconstructionPendingPolls(0);
    const response = await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`, {
      headers: { "X-Forwarded-Proto": "https" },
    });
    const body = (await response.json()) as { glb_url?: string };
    expect(body.glb_url).toMatch(/^https:\/\//);
  });

  it("fails loudly on a request missing order_code, instead of guessing", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(response.status).toBe(500);
  });

  // C-19 (#193): recon/service.py returns its own contract-shaped session_not_found for an
  // unknown session on these two routes, not a generic 500 -- a stale/unknown session id is a
  // real scenario the real service handles gracefully, unlike the genuinely malformed requests
  // the other "fails loudly" tests above cover.
  it("returns the contract's session_not_found for finishSession on an unknown session_id", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions/does-not-exist/finish`, {
      method: "POST",
    });
    expect(response.status).toBe(409);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body["code"]).toBe("session_not_found");
  });

  it("returns the contract's session_not_found for uploadFrame on an unknown session_id", async () => {
    const response = await uploadFrame(baseUrl, "does-not-exist");
    expect(response.status).toBe(422);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body["code"]).toBe("session_not_found");
  });

  it("returns the contract's session_not_found fetching model.glb for a session that isn't ready", async () => {
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    const response = await fetch(`${baseUrl}/v1/sessions/${sessionId}/model.glb`);

    expect(response.status).toBe(404);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body["code"]).toBe("session_not_found");
  });

  it("returns the contract's session_not_found fetching model.glb for an unknown session_id", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions/does-not-exist/model.glb`);

    expect(response.status).toBe(404);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body["code"]).toBe("session_not_found");
  });

  it("still serves the C-11/C-12 device-check fixtures with no session created", async () => {
    const response = await fetch(`${baseUrl}/v1/sessions/c11-device-check/model.glb`);
    expect(response.status).toBe(200);
  });

  it("stays equal to recon/service.py's committed MIN_FINISH_FRAMES", () => {
    const source = readFileSync(findReconServiceSource(), "utf8");
    const match = source.match(/^MIN_FINISH_FRAMES\s*=\s*(\d+)/m);

    expect(match).not.toBeNull();
    expect(Number(match?.[1])).toBe(MIN_FINISH_FRAMES);
  });

  it("serves the Pass and Rescan verdict contract examples", async () => {
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    for (const [scenario, exampleName] of [
      ["pass", "getVerdict.response-200"],
      ["rescan", "getVerdict.response-200-rescan-unobserved"],
    ] as const) {
      stub.setVerdictScenario(scenario);
      const response = await fetch(`${baseUrl}/v1/sessions/${sessionId}/verdict`);
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual(loadExample(exampleName));
    }
  });

  it("serves no_cad_uploaded by default and the canned Fail fixture on request", async () => {
    const createResponse = await fetch(`${baseUrl}/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order_code: "PO-48213-A" }),
    });
    const { session_id: sessionId } = (await createResponse.json()) as { session_id: string };

    const noCad = await fetch(`${baseUrl}/v1/sessions/${sessionId}/verdict`);
    expect(noCad.status).toBe(404);
    expect(await noCad.json()).toEqual(loadExample("getVerdict.response-404"));

    stub.setVerdictScenario("fail");
    const failed = await fetch(`${baseUrl}/v1/sessions/${sessionId}/verdict`);
    expect(failed.status).toBe(200);
    const body = (await failed.json()) as { verdict?: string; reason_codes?: string[] };
    expect(body.verdict).toBe("Fail");
    expect(body.reason_codes).toContain("observed_deviation_out_of_spec");
  });

  it("keeps verdict and deviation-model scenarios scoped to one session", async () => {
    stub.setReconstructionPendingPolls(0);

    async function createReadySession(): Promise<string> {
      const created = await fetch(`${baseUrl}/v1/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order_code: "PO-48213-A" }),
      });
      const { session_id: sessionId } = (await created.json()) as { session_id: string };
      await uploadMinimumFrames(baseUrl, sessionId);
      await fetch(`${baseUrl}/v1/sessions/${sessionId}/finish`, { method: "POST" });
      await fetch(`${baseUrl}/v1/sessions/${sessionId}/reconstruction`);
      return sessionId;
    }

    const sessionA = await createReadySession();
    const sessionB = await createReadySession();
    stub.setVerdictScenario("fail", sessionB);

    const verdictA = await fetch(`${baseUrl}/v1/sessions/${sessionA}/verdict`);
    const verdictB = await fetch(`${baseUrl}/v1/sessions/${sessionB}/verdict`);
    expect(verdictA.status).toBe(404);
    expect(verdictB.status).toBe(200);
    expect(((await verdictB.json()) as { verdict?: string }).verdict).toBe("Fail");

    const modelA = new Uint8Array(
      await (await fetch(`${baseUrl}/v1/sessions/${sessionA}/model.glb`)).arrayBuffer(),
    );
    const modelB = new Uint8Array(
      await (await fetch(`${baseUrl}/v1/sessions/${sessionB}/model.glb`)).arrayBuffer(),
    );
    expect(modelA).not.toEqual(modelB);
  });
});

// C-09: `npm run dev:device` serves HTTPS over HTTP/2 (Vite's resolveHttpServer uses
// http2.createSecureServer with allowHTTP1:true whenever basicSsl is active), so a real request
// from a phone carries HTTP/2 pseudo-headers (:method, :path, :authority, :scheme) in req.headers
// and no `host` header at all -- confirmed against a real Node http2.createServer(). `handle`
// is called directly here, in-process, with a hand-built fake IncomingMessage/ServerResponse --
// createContractStub()'s own `handle` is designed to work this way ("usable both as a bare
// http.createServer handler... and as Vite dev middleware"), so this needs no real socket or TLS.
describe("contract stub over HTTP/2 (no real socket)", () => {
  function fakeHttp2Request(
    method: string,
    path: string,
    authority: string,
    extraHeaders: Record<string, string> = {},
    body: Buffer | null = null,
  ): IncomingMessage {
    const chunks = body ? [body] : [];
    return {
      method,
      url: path,
      headers: {
        ":method": method,
        ":path": path,
        ":authority": authority,
        ":scheme": "http",
        ...extraHeaders,
      },
      socket: {},
      [Symbol.asyncIterator]: () => {
        let index = 0;
        return {
          next: () =>
            Promise.resolve(
              index < chunks.length
                ? { value: chunks[index++], done: false }
                : { value: undefined, done: true },
            ),
        };
      },
    } as unknown as IncomingMessage;
  }

  function fakeServerResponse(): {
    res: ServerResponse;
    waitForResponse: () => Promise<{ status: number; body: unknown }>;
  } {
    let statusCode = 0;
    const chunks: Buffer[] = [];
    let resolveDone: (() => void) | undefined;
    const done = new Promise<void>((resolve) => {
      resolveDone = resolve;
    });
    const res = {
      writeHead(code: number) {
        statusCode = code;
        return res;
      },
      end(chunk?: string | Buffer) {
        if (chunk) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        }
        resolveDone?.();
      },
    } as unknown as ServerResponse;
    return {
      res,
      waitForResponse: async () => {
        await done;
        return {
          status: statusCode,
          body: JSON.parse(Buffer.concat(chunks).toString("utf-8")) as unknown,
        };
      },
    };
  }

  // Built from a real FormData/Request, not hand-rolled multipart bytes -- the raw JPEG magic
  // bytes (0xFF 0xD8 0xFF) aren't valid UTF-8, and a hand-written boundary around them turned out
  // to desync undici's own multipart parser (it found "frame" but never "gyro"). Letting the
  // fetch implementation that will later *parse* this body also *generate* it guarantees the two
  // agree byte-for-byte.
  async function multipartFrameBody(): Promise<{ body: Buffer; contentType: string }> {
    const form = new FormData();
    form.set(
      "frame",
      new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" }),
      "frame.jpg",
    );
    form.set("gyro", new Blob([JSON.stringify({ alpha: 1, beta: 2, gamma: 3 })]));
    const probeRequest = new Request("http://stub.local", { method: "POST", body: form });
    const contentType = probeRequest.headers.get("content-type");
    if (!contentType) {
      throw new Error("test setup: no content-type generated for the multipart body");
    }
    return { body: Buffer.from(await probeRequest.arrayBuffer()), contentType };
  }

  it("createSession answers per the contract, not a 500, given HTTP/2 pseudo-headers", async () => {
    const stub = createContractStub();
    const body = Buffer.from(JSON.stringify({ order_code: "PO-48213-A" }));
    const req = fakeHttp2Request(
      "POST",
      "/v1/sessions",
      "192.168.1.42:5173",
      {
        "content-type": "application/json",
      },
      body,
    );
    const { res, waitForResponse } = fakeServerResponse();

    stub.handle(req, res);
    const { status, body: responseBody } = await waitForResponse();

    expect(status).toBe(201);
    expect((responseBody as { session_id?: string }).session_id).toMatch(/^sess_/);
  });

  it("uploadFrame answers per the contract, not a 500, given an HTTP/2-shaped multipart request", async () => {
    const stub = createContractStub();
    const createReq = fakeHttp2Request(
      "POST",
      "/v1/sessions",
      "192.168.1.42:5173",
      { "content-type": "application/json" },
      Buffer.from(JSON.stringify({ order_code: "PO-48213-A" })),
    );
    const created = fakeServerResponse();
    stub.handle(createReq, created.res);
    const { body: session } = await created.waitForResponse();
    const sessionId = (session as { session_id: string }).session_id;

    const { body: multipartBody, contentType } = await multipartFrameBody();
    const uploadReq = fakeHttp2Request(
      "POST",
      `/v1/sessions/${sessionId}/frames`,
      "192.168.1.42:5173",
      { "content-type": contentType },
      multipartBody,
    );
    const uploaded = fakeServerResponse();

    stub.handle(uploadReq, uploaded.res);
    const { status } = await uploaded.waitForResponse();

    expect(status).toBe(202);
  });

  it("glb_url names the phone's real :authority, not 127.0.0.1, when host is absent", async () => {
    const stub = createContractStub();
    stub.setReconstructionPendingPolls(0);

    const created = fakeServerResponse();
    stub.handle(
      fakeHttp2Request(
        "POST",
        "/v1/sessions",
        "192.168.1.42:5173",
        { "content-type": "application/json" },
        Buffer.from(JSON.stringify({ order_code: "PO-48213-A" })),
      ),
      created.res,
    );
    const { body: session } = await created.waitForResponse();
    const sessionId = (session as { session_id: string }).session_id;

    // C-19: finishSession now requires MIN_FINISH_FRAMES (3) accepted frames.
    for (let i = 0; i < 3; i += 1) {
      const { body: multipartBody, contentType } = await multipartFrameBody();
      const uploaded = fakeServerResponse();
      stub.handle(
        fakeHttp2Request(
          "POST",
          `/v1/sessions/${sessionId}/frames`,
          "192.168.1.42:5173",
          { "content-type": contentType },
          multipartBody,
        ),
        uploaded.res,
      );
      await uploaded.waitForResponse();
    }

    const finished = fakeServerResponse();
    stub.handle(
      fakeHttp2Request("POST", `/v1/sessions/${sessionId}/finish`, "192.168.1.42:5173"),
      finished.res,
    );
    await finished.waitForResponse();

    const reconstructed = fakeServerResponse();
    stub.handle(
      fakeHttp2Request("GET", `/v1/sessions/${sessionId}/reconstruction`, "192.168.1.42:5173"),
      reconstructed.res,
    );
    const { body } = await reconstructed.waitForResponse();

    const glbUrl = (body as { glb_url?: string }).glb_url ?? "";
    expect(glbUrl).toContain("192.168.1.42:5173");
    expect(glbUrl).not.toContain("127.0.0.1");
  });
});
