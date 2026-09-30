import type { ApiResult } from "./api.js";
import type { Session, UploadFrameResult } from "./types.js";

// Pure functions: an ApiResult in, display text out. Kept separate from src/page.ts's DOM
// wiring so they can be unit-tested without a document.

export function describeCreateSession(result: ApiResult<Session>): string {
  if (result.ok) {
    return `Session started: ${result.body.session_id}`;
  }
  return `Could not start session — ${result.error.message}`;
}

export function describeUploadFrame(result: ApiResult<UploadFrameResult>): string {
  if (result.ok) {
    return `Frame accepted (frame ${String(result.body.frame_index)})`;
  }
  return `Frame rejected — ${result.error.message}`;
}

export function describeFinishSession(result: ApiResult<Session>): string {
  if (result.ok) {
    return `Session finished: ${result.body.session_id}`;
  }
  return `Could not finish session — ${result.error.message}`;
}
