import { createSession, finishSession, uploadFrame } from "./api.js";
import { parseGyroInput } from "./gyro.js";
import { describeCreateSession, describeFinishSession, describeUploadFrame } from "./messages.js";

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
`;

export interface UploadPageOptions {
  baseUrl: string;
}

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
