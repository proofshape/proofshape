// C-04: replaces C-01/C-02/C-03's three isolated demo sections (#app/#camera-app/#gyro-app) with
// one real flow — create a session, request both permissions together — and gives C-05 onward
// one shared place to read/write app state instead of per-module globals.

import { requestCameraPermission, setStatus } from "./camera.js";
import { createSession, finishSession, getReconstruction } from "./api.js";
import { requireElement, requireTyped } from "./dom.js";
import { requestGyroPermission } from "./gyro.js";
import { describeCreateSession, describeFinishSession } from "./messages.js";
import type { Gyro, ReconstructionResult } from "./types.js";

export interface AppState {
  sessionId: string | null;
  cameraGranted: boolean;
  gyroGranted: boolean;
  // null components from a real deviceorientation event are dropped, never coerced to 0 -- Gyro
  // requires `number`, the DOM event reports `number | null`.
  latestGyro: Gyro | null;
}

export interface AppShellOptions {
  baseUrl: string;
  reconstructionPollIntervalMs?: number;
  reconstructionMaxAttempts?: number;
}

const DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS = 2000;
const DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS = 300;

export const APP_SHELL_HTML = `
  <section>
    <h2>Start capture session (C-04)</h2>
    <label>Order code <input id="shell-order-code" type="text" /></label>
    <button id="shell-start-button" type="button">Start</button>
    <p id="shell-session-status"></p>
    <p id="shell-camera-status"></p>
    <p id="shell-gyro-status"></p>
  </section>
  <section>
    <h2>Finish and reconstruct (C-10)</h2>
    <button id="shell-finish-button" type="button" disabled>Finish session</button>
    <button id="shell-retry-reconstruction-button" type="button" hidden>
      Retry reconstruction
    </button>
    <p id="shell-finish-status"></p>
    <p id="shell-reconstruction-status"></p>
  </section>
`;

// Not a module-level singleton: gyro.ts's own `listening` flag was deliberately moved from
// module-level to closure-scoped during C-03, specifically because a module-level mutable flag
// leaked across repeated mounts in tests. AppState gets created fresh per mountAppShell call for
// the same reason, and is returned to whoever calls it rather than imported as shared global
// state.
export function mountAppShell(root: HTMLElement, options: AppShellOptions): AppState {
  root.innerHTML = APP_SHELL_HTML;

  const orderCodeInput = requireTyped(root, "#shell-order-code", HTMLInputElement, "<input>");
  const startButton = requireTyped(root, "#shell-start-button", HTMLButtonElement, "<button>");
  const sessionStatus = requireElement(root, "#shell-session-status");
  const cameraStatus = requireElement(root, "#shell-camera-status");
  const gyroStatus = requireElement(root, "#shell-gyro-status");
  const finishButton = requireTyped(root, "#shell-finish-button", HTMLButtonElement, "<button>");
  const retryReconstructionButton = requireTyped(
    root,
    "#shell-retry-reconstruction-button",
    HTMLButtonElement,
    "<button>",
  );
  const finishStatus = requireElement(root, "#shell-finish-status");
  const reconstructionStatus = requireElement(root, "#shell-reconstruction-status");

  const state: AppState = {
    sessionId: null,
    cameraGranted: false,
    gyroGranted: false,
    latestGyro: null,
  };

  // Scoped to this mount, same reasoning as gyro.ts's own `listening` flag.
  let listeningForGyro = false;

  function startListeningForGyro(): void {
    if (listeningForGyro) {
      return;
    }
    listeningForGyro = true;
    window.addEventListener("deviceorientation", (event) => {
      if (event.alpha === null || event.beta === null || event.gamma === null) {
        return;
      }
      state.latestGyro = { alpha: event.alpha, beta: event.beta, gamma: event.gamma };
    });
  }

  startButton.addEventListener("click", () => {
    setStatus(sessionStatus, "", "info");
    setStatus(cameraStatus, "", "info");
    setStatus(gyroStatus, "", "info");
    startButton.disabled = true;

    // All three started synchronously, before awaiting any of them: getUserMedia and
    // requestPermission (inside requestCameraPermission/requestGyroPermission) must have no
    // leading `await` to preserve the iOS gesture that triggered this click; createSession has
    // no such constraint, but starting it here too keeps this one real flow from one click,
    // rather than literally serializing session-creation ahead of the gesture-critical calls.
    const sessionPromise = createSession(options.baseUrl, orderCodeInput.value);
    const cameraPromise = requestCameraPermission();
    const gyroPromise = requestGyroPermission();

    void (async () => {
      // createSession can genuinely reject, not just resolve with ok:false -- a real network
      // failure, a non-JSON response, or any status outside 201/400 all throw inside api.ts.
      // Catching here (TabeenRaoof's PR #102 review, reproduced: an unstubbed rejection left the
      // button disabled forever with camera/gyro never reflected, even though their prompts had
      // already fired) is what keeps that from silently stopping the user with no explanation.
      let sessionOk = false;
      try {
        const sessionResult = await sessionPromise;
        setStatus(
          sessionStatus,
          describeCreateSession(sessionResult),
          sessionResult.ok ? "info" : "error",
        );
        if (sessionResult.ok) {
          state.sessionId = sessionResult.body.session_id;
          finishButton.disabled = false;
          sessionOk = true;
        }
      } catch (error) {
        setStatus(
          sessionStatus,
          `Could not start session — ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
      }

      const cameraResult = await cameraPromise;
      state.cameraGranted = cameraResult.granted;
      setStatus(
        cameraStatus,
        cameraResult.granted ? "Camera permission granted." : cameraResult.message,
        cameraResult.granted ? "info" : "error",
      );

      const gyroResult = await gyroPromise;
      state.gyroGranted = gyroResult.granted;
      setStatus(
        gyroStatus,
        gyroResult.granted ? "Gyro permission granted." : gyroResult.message,
        gyroResult.granted ? "info" : "error",
      );
      if (gyroResult.granted) {
        startListeningForGyro();
      }

      // Nothing left for "Start" to do on this screen once a session exists -- only re-enable
      // if session creation itself failed (rejected or resolved ok:false), so the order code
      // can be fixed and retried.
      startButton.disabled = sessionOk;
    })();
  });

  async function runReconstructionPolling(): Promise<void> {
    if (state.sessionId === null) {
      return;
    }
    retryReconstructionButton.hidden = true;
    const outcome = await pollReconstruction(
      options.baseUrl,
      state.sessionId,
      reconstructionStatus,
      options.reconstructionMaxAttempts ?? DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS,
      options.reconstructionPollIntervalMs ?? DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS,
    );
    retryReconstructionButton.hidden = outcome !== "timeout" && outcome !== "error";
  }

  finishButton.addEventListener("click", () => {
    void (async () => {
      if (state.sessionId === null) {
        setStatus(finishStatus, "Start a session first.", "error");
        return;
      }

      finishButton.disabled = true;
      setStatus(reconstructionStatus, "", "info");
      try {
        const result = await finishSession(options.baseUrl, state.sessionId);
        setStatus(
          finishStatus,
          describeFinishSession(result),
          result.ok ? "info" : "error",
        );
        if (!result.ok) {
          finishButton.disabled = false;
          return;
        }
        await runReconstructionPolling();
      } catch (error) {
        setStatus(
          finishStatus,
          `Could not finish session — ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
        finishButton.disabled = false;
      }
    })();
  });

  retryReconstructionButton.addEventListener("click", () => {
    void runReconstructionPolling();
  });

  return state;
}

async function pollReconstruction(
  baseUrl: string,
  sessionId: string,
  target: HTMLElement,
  maxAttempts: number,
  intervalMs: number,
): Promise<"complete" | "failed" | "timeout" | "error"> {
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    let result;
    try {
      result = await getReconstruction(baseUrl, sessionId);
    } catch (error) {
      setStatus(
        target,
        `Could not fetch reconstruction — ${
          error instanceof Error ? error.message : String(error)
        }`,
        "error",
      );
      return "error";
    }

    if (!result.ok) {
      setStatus(target, `Could not fetch reconstruction — ${result.error.message}`, "error");
      return "error";
    }

    if (result.body.status === "pending") {
      setStatus(target, "Reconstruction pending…", "info");
      if (attempt < maxAttempts) {
        await delay(intervalMs);
      }
      continue;
    }

    renderReconstruction(target, result.body);
    return result.body.status === "failed" ? "failed" : "complete";
  }

  setStatus(target, `Reconstruction timed out after ${String(maxAttempts)} attempts.`, "error");
  return "timeout";
}

function renderReconstruction(target: HTMLElement, result: ReconstructionResult): void {
  if (result.status === "failed") {
    const warnings = result.warnings?.length ? ` Warnings: ${result.warnings.join("; ")}` : "";
    setStatus(target, `Reconstruction failed.${warnings}`, "error");
    return;
  }

  const referenceTier = result.reference_tier_used ?? "unknown";
  const metric = result.metric === true ? "yes" : "no";
  const observed =
    result.observed_fraction === undefined ? "unknown" : result.observed_fraction.toFixed(3);
  const warnings = result.warnings?.length ? result.warnings.join("; ") : "none";
  const dimensionalClaim = result.metric === false ? " No dimensional claims." : "";
  setStatus(
    target,
    `Reconstruction complete. Reference tier: ${referenceTier}. Metric: ${metric}.` +
      ` Observed fraction: ${observed}. Warnings: ${warnings}.${dimensionalClaim}`,
    "info",
  );
}

function delay(ms: number): Promise<void> {
  if (ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}
