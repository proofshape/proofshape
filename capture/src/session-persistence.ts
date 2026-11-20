// C-24 (#352): keep an in-progress capture session across a page reload.
//
// R-36's home pre-check lost a session when the supplier switched to Messages mid-capture: iOS
// reloaded the Safari tab on return, and everything about the session (its id, the accepted
// readings coverage is rebuilt from) lived only in memory. The server side was fine -- it numbers
// frames from its own count (SessionStore.add_frame) and R-36 resumed sessions after two service
// restarts -- so all the phone needs to remember is enough to carry on.
//
// sessionStorage, not a `?session=` URL parameter (decided with the owner, 2026-11-17): it
// survives the same-tab reload iOS does, and it keeps the session id out of the address bar,
// history and any link the supplier copies. It does not survive opening a brand-new tab; that's
// accepted.
//
// Pure functions over a small storage interface, so they test without a DOM. Every storage call
// is wrapped: Safari can throw on sessionStorage access (private browsing, storage disabled), and
// losing resume support must never stop a capture.

import type { Gyro } from "./types.js";

export const CAPTURE_SESSION_STORAGE_KEY = "proofshape.captureSession.v1";

// "capturing": frames can still be added; the page offers Continue session.
// "finished": Finish succeeded, so the server is reconstructing (or has). The page offers Show
// result, which resumes polling. The record is cleared once a terminal result has been shown.
export type StoredCapturePhase = "capturing" | "finished";

export interface StoredCaptureSession {
  sessionId: string;
  orderCode: string;
  phase: StoredCapturePhase;
  // The server's own count (the last 202's frame_index + 1), for display only. The server never
  // trusts it: it numbers each upload itself.
  acceptedFrameCount: number;
  // Every server-accepted frame's gyro reading, in order. Replaying them through the overlay
  // rebuilds orbit coverage, which is anchored on the first one.
  acceptedGyro: Gyro[];
}

// The subset of the DOM's Storage this module uses, so tests can pass a plain object.
export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

// The browser's sessionStorage, or null where reading the property itself throws (Safari with
// storage blocked) or doesn't exist (a non-browser environment).
export function browserSessionStorage(): KeyValueStorage | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function parseGyroList(value: unknown): Gyro[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const readings: Gyro[] = [];
  for (const item of value as unknown[]) {
    if (typeof item !== "object" || item === null) {
      return null;
    }
    const candidate = item as Record<string, unknown>;
    if (
      !isFiniteNumber(candidate.alpha) ||
      !isFiniteNumber(candidate.beta) ||
      !isFiniteNumber(candidate.gamma)
    ) {
      return null;
    }
    readings.push({ alpha: candidate.alpha, beta: candidate.beta, gamma: candidate.gamma });
  }
  return readings;
}

// Anything that doesn't parse into exactly this shape is treated as absent, never half-used: a
// wrong session id would send frames to someone else's session, and bad readings would draw a
// wrong coverage arrow. Explicit field checks rather than a cast, for that reason.
export function parseStoredCaptureSession(raw: string): StoredCaptureSession | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) {
    return null;
  }
  const record = parsed as Record<string, unknown>;
  if (!isNonEmptyString(record.sessionId) || !isNonEmptyString(record.orderCode)) {
    return null;
  }
  if (record.phase !== "capturing" && record.phase !== "finished") {
    return null;
  }
  if (
    !isFiniteNumber(record.acceptedFrameCount) ||
    !Number.isInteger(record.acceptedFrameCount) ||
    record.acceptedFrameCount < 0
  ) {
    return null;
  }
  const acceptedGyro = parseGyroList(record.acceptedGyro);
  if (acceptedGyro === null) {
    return null;
  }
  return {
    sessionId: record.sessionId,
    orderCode: record.orderCode,
    phase: record.phase,
    acceptedFrameCount: record.acceptedFrameCount,
    acceptedGyro,
  };
}

export function loadCaptureSession(storage: KeyValueStorage | null): StoredCaptureSession | null {
  if (storage === null) {
    return null;
  }
  let raw: string | null;
  try {
    raw = storage.getItem(CAPTURE_SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
  if (raw === null) {
    return null;
  }
  const session = parseStoredCaptureSession(raw);
  if (session === null) {
    // A corrupt or outdated record would otherwise be offered again on every reload.
    clearCaptureSession(storage);
  }
  return session;
}

// Returns whether the record was written, so a caller could say resume won't work; today the
// shell carries on either way, since losing resume support must not stop a capture.
export function saveCaptureSession(
  storage: KeyValueStorage | null,
  session: StoredCaptureSession,
): boolean {
  if (storage === null) {
    return false;
  }
  try {
    storage.setItem(CAPTURE_SESSION_STORAGE_KEY, JSON.stringify(session));
    return true;
  } catch {
    return false;
  }
}

export function clearCaptureSession(storage: KeyValueStorage | null): void {
  if (storage === null) {
    return;
  }
  try {
    storage.removeItem(CAPTURE_SESSION_STORAGE_KEY);
  } catch {
    // Nothing else to do: the next load parses whatever is left and discards it if invalid.
  }
}
