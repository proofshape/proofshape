import type { ErrorDetail, Gyro, ReconstructionResult, Session, UploadFrameResult } from "./types.js";

export type ApiResult<TBody> =
  { ok: true; status: number; body: TBody } | { ok: false; status: number; error: ErrorDetail };

// A response the contract does not define for this operation is not a "rejection" — it means
// the request or the stub/server is broken, and that must fail loudly rather than be displayed
// as if it were a normal 4xx from the contract.
async function readJsonBody(response: Response, operation: string): Promise<unknown> {
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(
      `${operation}: response body was not valid JSON (status ${String(response.status)}): ${text}`,
    );
  }
}

function unexpectedStatus(operation: string, response: Response, body: unknown): never {
  throw new Error(
    `${operation}: unexpected status ${String(response.status)} — not defined by the contract. Body: ${JSON.stringify(body)}`,
  );
}

export async function createSession(
  baseUrl: string,
  orderCode: string,
): Promise<ApiResult<Session>> {
  const response = await fetch(`${baseUrl}/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order_code: orderCode }),
  });
  const body = await readJsonBody(response, "createSession");
  if (response.status === 201) {
    return { ok: true, status: response.status, body: body as Session };
  }
  if (response.status === 400) {
    return { ok: false, status: response.status, error: body as ErrorDetail };
  }
  unexpectedStatus("createSession", response, body);
}

export async function uploadFrame(
  baseUrl: string,
  sessionId: string,
  frame: Blob,
  gyro: Gyro,
): Promise<ApiResult<UploadFrameResult>> {
  const form = new FormData();
  form.set("frame", frame, "frame.jpg");
  form.set("gyro", new Blob([JSON.stringify(gyro)], { type: "application/json" }));

  const response = await fetch(`${baseUrl}/v1/sessions/${encodeURIComponent(sessionId)}/frames`, {
    method: "POST",
    body: form,
  });
  const body = await readJsonBody(response, "uploadFrame");
  if (response.status === 202) {
    return { ok: true, status: response.status, body: body as UploadFrameResult };
  }
  if (response.status === 422) {
    return { ok: false, status: response.status, error: body as ErrorDetail };
  }
  unexpectedStatus("uploadFrame", response, body);
}

export async function finishSession(
  baseUrl: string,
  sessionId: string,
): Promise<ApiResult<Session>> {
  const response = await fetch(`${baseUrl}/v1/sessions/${encodeURIComponent(sessionId)}/finish`, {
    method: "POST",
  });
  const body = await readJsonBody(response, "finishSession");
  if (response.status === 202) {
    return { ok: true, status: response.status, body: body as Session };
  }
  if (response.status === 409) {
    return { ok: false, status: response.status, error: body as ErrorDetail };
  }
  unexpectedStatus("finishSession", response, body);
}


export async function getReconstruction(
  baseUrl: string,
  sessionId: string,
): Promise<ApiResult<ReconstructionResult>> {
  const response = await fetch(
    `${baseUrl}/v1/sessions/${encodeURIComponent(sessionId)}/reconstruction`,
  );
  const body = await readJsonBody(response, "getReconstruction");
  if (response.status === 200) {
    return { ok: true, status: response.status, body: body as ReconstructionResult };
  }
  if (response.status === 404) {
    return { ok: false, status: response.status, error: body as ErrorDetail };
  }
  unexpectedStatus("getReconstruction", response, body);
}
