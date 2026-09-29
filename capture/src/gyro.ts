import type { Gyro } from "./types.js";

// C-01 stands in for a real sensor reading with three plain number inputs (a real reading is
// C-03's job) — this only has to reject non-numeric or empty input loudly, not validate range.
export function parseGyroInput(raw: { alpha: string; beta: string; gamma: string }): Gyro {
  const alpha = parseGyroField("alpha", raw.alpha);
  const beta = parseGyroField("beta", raw.beta);
  const gamma = parseGyroField("gamma", raw.gamma);
  return { alpha, beta, gamma };
}

function parseGyroField(name: keyof Gyro, value: string): number {
  const trimmed = value.trim();
  if (trimmed === "") {
    throw new Error(`Gyro reading is missing a value for "${name}".`);
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Gyro reading for "${name}" is not a number: "${value}".`);
  }
  return parsed;
}
