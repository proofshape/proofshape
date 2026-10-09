import { createSession, finishSession, getReconstruction, uploadFrame } from "./api.js";
import { parseGyroInput } from "./gyro.js";
import { describeCreateSession, describeFinishSession, describeUploadFrame } from "./messages.js";
import type { ReconstructionResult } from "./types.js";

// The single source of truth for the page's markup — index.html mounts it at runtime, and
// page.test.ts mounts the same string, so the two can never drift apart the way a hand-copied
// fixture would.
export const PAGE_HTML = `
  <section>
    <h2>1. Start session</h2>
    <label>Order code <input id="order-code" type="text" /></label>
    <button id="start-button" type="button">Start</button>
    <p id="start-result"></p>
  </section>

  <section>
    <h2>2. Upload frame</h2>
    <label>Frame (JPEG) <input id="frame-file" type="file" accept="image/jpeg" /></label>
    <label>alpha <input id="gyro-alpha" type="number" /></label>
    <label>beta <input id="gyro-beta" type="number" /></label>
    <label>gamma <input id="gyro-gamma" type="number" /></label>
    <button id="upload-button" type="button" disabled>Upload</button>
    <p id="upload-result"></p>
  </section>

  <section>
    <h2>3. Finish session</h2>
    <button id="finish-button" type="button" disabled>Finish</button>
    <p id="finish-result"></p>
  </section>

  <section>
    <h2>4. Reconstruction result</h2>
    <p id="reconstruction-result"></p>
  </section>
`;

export interface UploadPageOptions {
  baseUrl: string;
  reconstructionPollIntervalMs?: number;
  reconstructionMaxAttempts?: number;
}

const DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS = 250;
const DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS = 20;

// Deliberately not a live camera, not a real gyro reading, no UI polish — proving the wire
// format (C-01) is the whole job; C-02/C-03/C-04 replace these inputs with the real thing.
export function mountUploadPage(root: HTMLElement, options: UploadPageOptions): void {
  root.innerHTML = PAGE_HTML;

  const orderCodeInput = requireInput(root, "#order-code");
  const startButton = requireButton(root, "#start-button");
  const startResult = requireElement(root, "#start-result");

  const frameFileInput = requireInput(root, "#frame-file");
  const alphaInput = requireInput(root, "#gyro-alpha");
  const betaInput = requireInput(root, "#gyro-beta");
  const gammaInput = requireInput(root, "#gyro-gamma");
  const uploadButton = requireButton(root, "#upload-button");
  const uploadResult = requireElement(root, "#upload-result");

  const finishButton = requireButton(root, "#finish-button");
  const finishResult = requireElement(root, "#finish-result");
  const reconstructionResult = requireElement(root, "#reconstruction-result");

  let sessionId: string | null = null;

  startButton.addEventListener("click", () => {
    void (async () => {
      const result = await createSession(options.baseUrl, orderCodeInput.value);
      startResult.textContent = describeCreateSession(result);
      if (result.ok) {
        sessionId = result.body.session_id;
        uploadButton.disabled = false;
        finishButton.disabled = false;
      }
    })();
  });

  uploadButton.addEventListener("click", () => {
    void (async () => {
      if (sessionId === null) {
        uploadResult.textContent = "Start a session first.";
        return;
      }
      const file = frameFileInput.files?.[0];
      if (!file) {
        uploadResult.textContent = "Choose a JPEG file first.";
        return;
      }
      let gyro;
      try {
        gyro = parseGyroInput({
          alpha: alphaInput.value,
          beta: betaInput.value,
          gamma: gammaInput.value,
        });
      } catch (error) {
        uploadResult.textContent = error instanceof Error ? error.message : String(error);
        return;
      }
      const result = await uploadFrame(options.baseUrl, sessionId, file, gyro);
      uploadResult.textContent = describeUploadFrame(result);
    })();
  });

  finishButton.addEventListener("click", () => {
    void (async () => {
      if (sessionId === null) {
        finishResult.textContent = "Start a session first.";
        return;
      }
      const result = await finishSession(options.baseUrl, sessionId);
      finishResult.textContent = describeFinishSession(result);
      if (!result.ok) {
        return;
      }
      finishButton.disabled = true;
      await pollReconstruction(
        options.baseUrl,
        sessionId,
        reconstructionResult,
        options.reconstructionMaxAttempts ?? DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS,
        options.reconstructionPollIntervalMs ?? DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS,
      );
    })();
  });
}

function requireElement(root: HTMLElement, selector: string): HTMLElement {
  const element = root.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`mountUploadPage: expected an element at "${selector}"`);
  }
  return element;
}

function requireInput(root: HTMLElement, selector: string): HTMLInputElement {
  const element = requireElement(root, selector);
  if (!(element instanceof HTMLInputElement)) {
    throw new Error(`mountUploadPage: expected an <input> at "${selector}"`);
  }
  return element;
}

function requireButton(root: HTMLElement, selector: string): HTMLButtonElement {
  const element = requireElement(root, selector);
  if (!(element instanceof HTMLButtonElement)) {
    throw new Error(`mountUploadPage: expected a <button> at "${selector}"`);
  }
  return element;
}


async function pollReconstruction(
  baseUrl: string,
  sessionId: string,
  target: HTMLElement,
  maxAttempts: number,
  intervalMs: number,
): Promise<void> {
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    let result;
    try {
      result = await getReconstruction(baseUrl, sessionId);
    } catch (error) {
      target.textContent = `Could not fetch reconstruction — ${
        error instanceof Error ? error.message : String(error)
      }`;
      return;
    }

    if (!result.ok) {
      target.textContent = `Could not fetch reconstruction — ${result.error.message}`;
      return;
    }

    if (result.body.status === "pending") {
      target.textContent = "Reconstruction pending…";
      if (attempt < maxAttempts) {
        await delay(intervalMs);
      }
      continue;
    }

    renderReconstruction(target, result.body);
    return;
  }

  target.textContent = `Reconstruction timed out after ${String(maxAttempts)} attempts.`;
}

function renderReconstruction(target: HTMLElement, result: ReconstructionResult): void {
  if (result.status === "failed") {
    const warnings = result.warnings?.length ? ` Warnings: ${result.warnings.join("; ")}` : "";
    target.textContent = `Reconstruction failed.${warnings}`;
    return;
  }

  const referenceTier = result.reference_tier_used ?? "unknown";
  const metric = result.metric === true ? "yes" : "no";
  const observed =
    result.observed_fraction === undefined ? "unknown" : result.observed_fraction.toFixed(3);
  const warnings = result.warnings?.length ? result.warnings.join("; ") : "none";
  const dimensionalClaim = result.metric === false ? " No dimensional claims." : "";
  target.textContent =
    `Reconstruction complete. Reference tier: ${referenceTier}. Metric: ${metric}.` +
    ` Observed fraction: ${observed}. Warnings: ${warnings}.${dimensionalClaim}`;
}

function delay(ms: number): Promise<void> {
  if (ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}
