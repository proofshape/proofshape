import { describe, expect, it } from "vitest";
import {
  CAPTURE_SESSION_STORAGE_KEY,
  clearCaptureSession,
  loadCaptureSession,
  parseStoredCaptureSession,
  saveCaptureSession,
  type KeyValueStorage,
  type StoredCaptureSession,
} from "./session-persistence.js";

function memoryStorage(initial: Record<string, string> = {}): KeyValueStorage & {
  values: Map<string, string>;
} {
  const values = new Map(Object.entries(initial));
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      values.set(key, value);
    },
    removeItem: (key) => {
      values.delete(key);
    },
  };
}

// Safari can throw on any storage call when storage is blocked.
const throwingStorage: KeyValueStorage = {
  getItem: () => {
    throw new Error("SecurityError");
  },
  setItem: () => {
    throw new Error("QuotaExceededError");
  },
  removeItem: () => {
    throw new Error("SecurityError");
  },
};

const SAMPLE: StoredCaptureSession = {
  sessionId: "sess_2e59418b1077",
  orderCode: "PO-YH4E183E",
  phase: "capturing",
  acceptedFrameCount: 5,
  acceptedGyro: [
    { alpha: 10, beta: 45, gamma: 0 },
    { alpha: 40, beta: 46, gamma: 1 },
  ],
};

describe("session persistence (C-24, #352)", () => {
  it("round-trips a saved session", () => {
    const storage = memoryStorage();
    expect(saveCaptureSession(storage, SAMPLE)).toBe(true);
    expect(loadCaptureSession(storage)).toEqual(SAMPLE);
  });

  it("round-trips the finished phase", () => {
    const storage = memoryStorage();
    saveCaptureSession(storage, { ...SAMPLE, phase: "finished" });
    expect(loadCaptureSession(storage)?.phase).toBe("finished");
  });

  it("returns null when nothing is stored", () => {
    expect(loadCaptureSession(memoryStorage())).toBeNull();
  });

  it("clears the record", () => {
    const storage = memoryStorage();
    saveCaptureSession(storage, SAMPLE);
    clearCaptureSession(storage);
    expect(storage.values.has(CAPTURE_SESSION_STORAGE_KEY)).toBe(false);
  });

  it.each([
    ["not JSON", "{not json"],
    ["a JSON array", "[]"],
    ["a missing session id", JSON.stringify({ ...SAMPLE, sessionId: undefined })],
    ["a blank session id", JSON.stringify({ ...SAMPLE, sessionId: "  " })],
    ["a blank order code", JSON.stringify({ ...SAMPLE, orderCode: "" })],
    ["an unknown phase", JSON.stringify({ ...SAMPLE, phase: "reconstructing" })],
    ["a negative frame count", JSON.stringify({ ...SAMPLE, acceptedFrameCount: -1 })],
    ["a fractional frame count", JSON.stringify({ ...SAMPLE, acceptedFrameCount: 2.5 })],
    ["readings that aren't a list", JSON.stringify({ ...SAMPLE, acceptedGyro: {} })],
    [
      "a reading with a missing angle",
      JSON.stringify({ ...SAMPLE, acceptedGyro: [{ alpha: 1, beta: 2 }] }),
    ],
    [
      "a reading with a non-numeric angle",
      JSON.stringify({ ...SAMPLE, acceptedGyro: [{ alpha: 1, beta: "2", gamma: 3 }] }),
    ],
  ])("ignores and removes a stored record with %s", (_label, raw) => {
    expect(parseStoredCaptureSession(raw)).toBeNull();
    const storage = memoryStorage({ [CAPTURE_SESSION_STORAGE_KEY]: raw });
    expect(loadCaptureSession(storage)).toBeNull();
    expect(storage.values.has(CAPTURE_SESSION_STORAGE_KEY)).toBe(false);
  });

  it("keeps only the three angles of each reading", () => {
    const raw = JSON.stringify({
      ...SAMPLE,
      acceptedGyro: [{ alpha: 1, beta: 2, gamma: 3, extra: "x" }],
    });
    expect(parseStoredCaptureSession(raw)?.acceptedGyro).toEqual([{ alpha: 1, beta: 2, gamma: 3 }]);
  });

  it("never throws when storage itself throws, and reports the failed save", () => {
    expect(loadCaptureSession(throwingStorage)).toBeNull();
    expect(saveCaptureSession(throwingStorage, SAMPLE)).toBe(false);
    expect(() => {
      clearCaptureSession(throwingStorage);
    }).not.toThrow();
  });

  it("treats missing storage as nothing stored", () => {
    expect(loadCaptureSession(null)).toBeNull();
    expect(saveCaptureSession(null, SAMPLE)).toBe(false);
    expect(() => {
      clearCaptureSession(null);
    }).not.toThrow();
  });
});
