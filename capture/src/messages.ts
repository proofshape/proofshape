import type { ApiResult } from "./api.js";
import type { Session, UploadFrameResult, VlmFeedback } from "./types.js";

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

// A-11: the hint is about an earlier sampled frame (the check runs in the background), so the
// text names that frame rather than implying it's about the photo just taken.
export function describeVlmFeedback(feedback: VlmFeedback): string {
  return `Tip (from frame ${String(feedback.frame_index)}): ${feedback.hint}`;
}

export function describeFinishSession(result: ApiResult<Session>): string {
  if (result.ok) {
    return `Session finished: ${result.body.session_id}`;
  }
  return `Could not finish session — ${result.error.message}`;
}

// C-18: `fetch` rejects with a `TypeError` for a genuine network failure (no connection, the
// service unreachable) -- distinct from the `Error`s api.ts itself throws for a malformed or
// undefined-by-the-contract response, whose own messages are already specific. Without this, the
// supplier saw the browser's own wording verbatim ("Load failed" on Safari, "Failed to fetch" on
// Chrome), which names no next step (caught on review, @TabeenRaoof, #306).
export function describeNetworkError(error: unknown): string {
  if (error instanceof TypeError) {
    return "No connection to the service — check your network and try again.";
  }
  return error instanceof Error ? error.message : String(error);
}
