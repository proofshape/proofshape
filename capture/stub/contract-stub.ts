import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ErrorDetail, ReconstructionResult, Session, UploadFrameResult } from "../src/types.js";

// C-01 started this as a three-route contract stub. C-10 extends it into the capture-side mock
// reconstruction service: finish a session, poll reconstruction, and fetch a canned GLB without
// needing the paid R-14 deployment. Later stories can extend the same stub for verdict-specific
// UI rather than creating another mock server.
//
// Every canned JSON body is loaded from contracts/examples/*.json at startup rather than typed in by
// hand, so this stub cannot silently drift from the frozen contract (F-03).
//
// Path built with node:path, not `new URL(x, import.meta.url)` — that literal pattern is
// statically rewritten by Vite into an asset-URL import, which is then subject to
// `server.fs.allow` and denies reading contracts/, since it sits outside the capture/ vite root.
// This is a plain Node file read, not a browser asset, so it should never go through Vite at all.
const EXAMPLES_DIR = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "contracts",
  "examples",
);

function loadExample(name: string): unknown {
  return JSON.parse(readFileSync(join(EXAMPLES_DIR, `${name}.json`), "utf-8"));
}

const KNOWN_ORDER_CODE = (loadExample("createSession.request") as { order_code: string })
  .order_code;
const CREATE_SESSION_400 = loadExample("createSession.response-400") as ErrorDetail;
const UPLOAD_FRAME_422 = loadExample("uploadFrame.response-422") as ErrorDetail;
const FINISH_SESSION_409_TOO_FEW = loadExample("finishSession.response-409") as ErrorDetail;
const RECONSTRUCTION_COMPLETE = loadExample(
  "getReconstruction.response-200",
) as ReconstructionResult;
const RECONSTRUCTION_NO_REFERENCE = loadExample(
  "getReconstruction.response-200-no-reference",
) as ReconstructionResult;
const RECONSTRUCTION_404 = loadExample("getReconstruction.response-404") as ErrorDetail;
const MODEL_FIXTURE_PATH = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "model.glb");

interface StubSession {
  session_id: string;
  order_code: string;
  status: "capturing" | "finished";
  acceptedFrameCount: number;
  frameAttempts: number;
  reconstructionPollCount: number;
}

export interface ContractStub {
  // A Connect-style listener: usable both as a bare `http.createServer` handler (tests) and as
  // Vite dev middleware (`server.middlewares.use`, so `npm run dev` needs no separate process
  // and no CORS).
  handle: (req: IncomingMessage, res: ServerResponse, next?: (err?: unknown) => void) => void;
  // Test control: the next accepted-otherwise frame upload is rejected instead, using the
  // contract's committed 422 example.
  rejectNextFrame: () => void;
  // Test controls for C-10's terminal-state polling cases.
  setReconstructionPendingPolls: (count: number) => void;
  failReconstruction: () => void;
  useNoReferenceReconstruction: () => void;
}

export function createContractStub(): ContractStub {
  const sessions = new Map<string, StubSession>();
  let rejectNext = false;
  let sessionCounter = 0;
  let reconstructionPendingPolls = 1;
  let reconstructionShouldFail = false;
  let reconstructionNoReference = false;

  function sendJson(res: ServerResponse, status: number, body: unknown): void {
    const text = JSON.stringify(body);
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(text);
  }

  // A 500 with a JSON body (rather than plain text) so the client-side code path under test is
  // the same one a real unexpected status would take — api.ts always expects a JSON body.
  function violation(res: ServerResponse, message: string): void {
    sendJson(res, 500, { stub_violation: `stub: request violates contract: ${message}` });
  }

  async function toWebRequest(req: IncomingMessage): Promise<Request> {
    const chunks: Buffer[] = [];
    for await (const chunk of req as AsyncIterable<Buffer>) {
      chunks.push(chunk);
    }
    const body = chunks.length > 0 ? Buffer.concat(chunks) : null;
    const headers = new Headers();
    for (const [key, value] of Object.entries(req.headers)) {
      if (typeof value === "string") {
        headers.set(key, value);
      } else if (Array.isArray(value)) {
        headers.set(key, value.join(", "));
      }
    }
    return new Request(new URL(req.url ?? "/", "http://stub.local"), {
      method: req.method ?? "GET",
      headers,
      body,
    });
  }

  async function handleCreateSession(req: Request, res: ServerResponse): Promise<void> {
    let parsed: unknown;
    try {
      parsed = await req.json();
    } catch {
      violation(res, "createSession body was not valid JSON");
      return;
    }
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      typeof (parsed as { order_code?: unknown }).order_code !== "string"
    ) {
      violation(res, 'createSession body is missing a string "order_code"');
      return;
    }
    const orderCode = (parsed as { order_code: string }).order_code;

    if (orderCode !== KNOWN_ORDER_CODE) {
      sendJson(res, 400, CREATE_SESSION_400);
      return;
    }

    sessionCounter += 1;
    const session: StubSession = {
      session_id: `sess_stub-${String(sessionCounter)}`,
      order_code: orderCode,
      status: "capturing",
      acceptedFrameCount: 0,
      frameAttempts: 0,
      reconstructionPollCount: 0,
    };
    sessions.set(session.session_id, session);
    const body: Session = {
      session_id: session.session_id,
      order_code: session.order_code,
      status: session.status,
    };
    sendJson(res, 201, body);
  }

  async function handleUploadFrame(
    req: Request,
    res: ServerResponse,
    sessionId: string,
  ): Promise<void> {
    const session = sessions.get(sessionId);
    if (!session) {
      violation(res, `uploadFrame called with unknown session_id "${sessionId}"`);
      return;
    }

    let form: FormData;
    try {
      form = await req.formData();
    } catch {
      violation(res, "uploadFrame body was not valid multipart/form-data");
      return;
    }

    const frame = form.get("frame");
    if (!(frame instanceof Blob)) {
      violation(res, 'uploadFrame is missing a "frame" file part');
      return;
    }
    const frameBytes = new Uint8Array(await frame.arrayBuffer());
    const looksLikeJpeg =
      frameBytes.length >= 3 &&
      frameBytes[0] === 0xff &&
      frameBytes[1] === 0xd8 &&
      frameBytes[2] === 0xff;
    if (!looksLikeJpeg) {
      violation(res, '"frame" part does not start with the JPEG magic bytes (FF D8 FF)');
      return;
    }

    const gyroPart = form.get("gyro");
    if (!(gyroPart instanceof Blob)) {
      violation(res, 'uploadFrame is missing a "gyro" part');
      return;
    }
    let gyro: unknown;
    try {
      gyro = JSON.parse(await gyroPart.text());
    } catch {
      violation(res, '"gyro" part was not valid JSON');
      return;
    }
    const g = gyro as Partial<Record<"alpha" | "beta" | "gamma", unknown>>;
    if (typeof g.alpha !== "number" || typeof g.beta !== "number" || typeof g.gamma !== "number") {
      violation(res, '"gyro" part must have numeric alpha, beta and gamma');
      return;
    }

    session.frameAttempts += 1;
    const frameIndex = session.frameAttempts - 1;

    if (rejectNext) {
      rejectNext = false;
      const error: ErrorDetail = { ...UPLOAD_FRAME_422, frame_indices: [frameIndex] };
      sendJson(res, 422, error);
      return;
    }

    session.acceptedFrameCount += 1;
    const body: UploadFrameResult = { accepted: true, frame_index: frameIndex };
    sendJson(res, 202, body);
  }

  function handleFinishSession(res: ServerResponse, sessionId: string): void {
    const session = sessions.get(sessionId);
    if (!session) {
      violation(res, `finishSession called with unknown session_id "${sessionId}"`);
      return;
    }

    if (session.status === "finished") {
      const error: ErrorDetail = {
        code: "session_already_finished",
        message: "This session has already been finished.",
      };
      sendJson(res, 409, error);
      return;
    }

    if (session.acceptedFrameCount === 0) {
      sendJson(res, 409, FINISH_SESSION_409_TOO_FEW);
      return;
    }

    session.status = "finished";
    const body: Session = {
      session_id: session.session_id,
      order_code: session.order_code,
      status: session.status,
    };
    sendJson(res, 202, body);
  }

  function handleReconstruction(
    req: IncomingMessage,
    res: ServerResponse,
    sessionId: string,
  ): void {
    const session = sessions.get(sessionId);
    if (!session || session.status !== "finished") {
      sendJson(res, 404, RECONSTRUCTION_404);
      return;
    }

    session.reconstructionPollCount += 1;
    if (session.reconstructionPollCount <= reconstructionPendingPolls) {
      const pending: ReconstructionResult = { status: "pending" };
      sendJson(res, 200, pending);
      return;
    }

    if (reconstructionShouldFail) {
      const failed: ReconstructionResult = {
        status: "failed",
        warnings: ["Stub reconstruction failure."],
      };
      sendJson(res, 200, failed);
      return;
    }

    const host = req.headers.host ?? "127.0.0.1";
    const canned = reconstructionNoReference
      ? RECONSTRUCTION_NO_REFERENCE
      : RECONSTRUCTION_COMPLETE;
    const body: ReconstructionResult = {
      ...canned,
      glb_url: `http://${host}/v1/sessions/${encodeURIComponent(sessionId)}/model.glb`,
    };
    sendJson(res, 200, body);
  }

  function handleModel(res: ServerResponse): void {
    const bytes = readFileSync(MODEL_FIXTURE_PATH);
    res.writeHead(200, {
      "Content-Type": "model/gltf-binary",
      "Content-Length": String(bytes.length),
    });
    res.end(bytes);
  }

  function handle(req: IncomingMessage, res: ServerResponse, next?: (err?: unknown) => void): void {
    const path = new URL(req.url ?? "/", "http://stub.local").pathname;
    const framesMatch = /^\/v1\/sessions\/([^/]+)\/frames$/.exec(path);
    const finishMatch = /^\/v1\/sessions\/([^/]+)\/finish$/.exec(path);
    const reconstructionMatch = /^\/v1\/sessions\/([^/]+)\/reconstruction$/.exec(path);
    const modelMatch = /^\/v1\/sessions\/([^/]+)\/model\.glb$/.exec(path);

    if (req.method === "POST" && path === "/v1/sessions") {
      toWebRequest(req)
        .then((webReq) => handleCreateSession(webReq, res))
        .catch((error: unknown) => {
          violation(res, `unhandled error in createSession: ${String(error)}`);
        });
      return;
    }
    if (req.method === "POST" && framesMatch) {
      const sessionId = framesMatch[1] as string;
      toWebRequest(req)
        .then((webReq) => handleUploadFrame(webReq, res, sessionId))
        .catch((error: unknown) => {
          violation(res, `unhandled error in uploadFrame: ${String(error)}`);
        });
      return;
    }
    if (req.method === "POST" && finishMatch) {
      const sessionId = finishMatch[1] as string;
      handleFinishSession(res, sessionId);
      return;
    }
    if (req.method === "GET" && reconstructionMatch) {
      const sessionId = reconstructionMatch[1] as string;
      handleReconstruction(req, res, sessionId);
      return;
    }
    if (req.method === "GET" && modelMatch) {
      handleModel(res);
      return;
    }

    if (next) {
      next();
      return;
    }
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end(`stub: no route for ${req.method ?? "?"} ${path}`);
  }

  return {
    handle,
    rejectNextFrame: () => {
      rejectNext = true;
    },
    setReconstructionPendingPolls: (count: number) => {
      reconstructionPendingPolls = Math.max(0, Math.floor(count));
    },
    failReconstruction: () => {
      reconstructionShouldFail = true;
    },
    useNoReferenceReconstruction: () => {
      reconstructionNoReference = true;
    },
  };
}

// Starts the stub on a real socket, for tests that need one (fetch/FormData against `http://`
// rather than calling the handler in-process). Defaults to an ephemeral port (0); pass a fixed
// port when the caller's fetches must match a pre-configured same-origin URL (see
// src/page.test.ts, which runs under happy-dom and needs the window's own origin to match, or
// the browser's same-origin policy blocks the request before it ever reaches this stub).
export async function startContractStub(port = 0): Promise<{
  baseUrl: string;
  stub: ContractStub;
  close: () => Promise<void>;
}> {
  const stub = createContractStub();
  const server = createServer(stub.handle);
  await new Promise<void>((resolve) => {
    server.listen(port, "127.0.0.1", resolve);
  });
  const address = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${String(address.port)}`,
    stub,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error) reject(error);
          else resolve();
        });
      }),
  };
}
