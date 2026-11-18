import { ServiceUnreachableError, type ApiResult } from "./api.js";
import type { StoredCaptureSession } from "./session-persistence.js";
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
  // C-24 (#351): R-36 saw the raw parser text here ("uploadFrame: response body was not valid
  // JSON (status 500):") while the service was restarting. It names no next step, and the next
  // step was simply to wait: the session continued once the service was back.
  if (error instanceof ServiceUnreachableError) {
    return "The service isn't reachable right now — wait a moment and try again.";
  }
  return error instanceof Error ? error.message : String(error);
}

// C-24: an upload that failed for connectivity reasons, not because the frame was rejected, is
// the moment a supplier wonders whether to start over. The frames the server already accepted
// are on its disk (SessionStore.add_frame writes each one before replying), so they are kept.
// That's said only for the two connectivity cases; a broken-contract Error says nothing about it.
export function describeUploadError(error: unknown): string {
  const description = describeNetworkError(error);
  if (error instanceof TypeError || error instanceof ServiceUnreachableError) {
    return `${description} Frames already accepted are kept.`;
  }
  return description;
}

function frameCountText(count: number): string {
  return count === 1 ? "1 frame" : `${String(count)} frames`;
}

// C-24 (#352): what the resume panel says after a reload. "capturing" offers Continue; "finished"
// offers Show result, since the reconstruction may still be running on the server.
export function describeSavedSession(saved: StoredCaptureSession): string {
  if (saved.phase === "finished") {
    return (
      `Session ${saved.sessionId} for ${saved.orderCode} was finished.` +
      " Its reconstruction may still be running."
    );
  }
  return (
    `An unfinished session is saved in this tab: ${saved.sessionId} for ${saved.orderCode},` +
    ` ${frameCountText(saved.acceptedFrameCount)} accepted. Continue it, or start a new one.`
  );
}

export function describeContinuedSession(saved: StoredCaptureSession): string {
  return (
    `Continuing session ${saved.sessionId} (${frameCountText(saved.acceptedFrameCount)}` +
    " accepted so far)."
  );
}
