// C-04: replaces C-01/C-02/C-03's three isolated demo sections (#app/#camera-app/#gyro-app) with
// one real flow — create a session, request both permissions together — and gives C-05 onward
// one shared place to read/write app state instead of per-module globals.

import {
  captureFrame,
  describeCameraError,
  requestCameraPermission,
  setStatus,
  stopCameraStream,
  type CapturedFrame,
} from "./camera.js";
import { createSession, finishSession, getReconstruction } from "./api.js";
import { requireElement, requireTyped } from "./dom.js";
import { requestGyroPermission } from "./gyro.js";
import { describeCreateSession, describeFinishSession } from "./messages.js";
import {
  mountModelViewer,
  type ModelViewerOptions,
  type ModelViewerState,
} from "./model-viewer.js";
import type { Gyro, ReconstructionResult } from "./types.js";

export interface AppState {
  sessionId: string | null;
  cameraGranted: boolean;
  gyroGranted: boolean;
  // null components from a real deviceorientation event are dropped, never coerced to 0 -- Gyro
  // requires `number`, the DOM event reports `number | null`.
  latestGyro: Gyro | null;
  // C-05: the live stream, kept running for the length of a session -- null once never granted,
  // or after a successful finish releases it.
  cameraStream: MediaStream | null;
  // C-05: overwritten on each capture, not accumulated -- C-09 (not yet built) is what will
  // actually consume/accumulate captured frames; this story only proves one can be grabbed.
  lastCapturedFrame: CapturedFrame | null;
}

export interface AppShellOptions {
  baseUrl: string;
  reconstructionPollIntervalMs?: number;
  reconstructionMaxAttempts?: number;
  modelViewerOptions?: ModelViewerOptions;
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
    <video id="shell-camera-preview" muted playsinline autoplay style="width: 100%; max-width: 480px;"></video>
    <button id="shell-capture-button" type="button" disabled>Capture frame</button>
    <p id="shell-capture-status"></p>
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
  <section>
    <h2>3D model (C-11)</h2>
    <div id="shell-model-viewer"></div>
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
  const cameraPreview = requireTyped(root, "#shell-camera-preview", HTMLVideoElement, "<video>");
  const captureButton = requireTyped(root, "#shell-capture-button", HTMLButtonElement, "<button>");
  const captureStatus = requireElement(root, "#shell-capture-status");
  const finishButton = requireTyped(root, "#shell-finish-button", HTMLButtonElement, "<button>");
  const retryReconstructionButton = requireTyped(
    root,
    "#shell-retry-reconstruction-button",
    HTMLButtonElement,
    "<button>",
  );
  const finishStatus = requireElement(root, "#shell-finish-status");
  const reconstructionStatus = requireElement(root, "#shell-reconstruction-status");
  const modelViewer = requireElement(root, "#shell-model-viewer");

  const state: AppState = {
    sessionId: null,
    cameraGranted: false,
    gyroGranted: false,
    latestGyro: null,
    cameraStream: null,
    lastCapturedFrame: null,
  };

  // Scoped to this mount, same reasoning as gyro.ts's own `listening` flag.
  let listeningForGyro = false;
  let disposeModelViewer = mountModelViewer(
    modelViewer,
    { status: "pending" },
    options.modelViewerOptions,
  );
  function remountModelViewer(result: ModelViewerState): void {
    disposeModelViewer();
    disposeModelViewer = mountModelViewer(modelViewer, result, options.modelViewerOptions);
  }

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

  // D-046: requestCameraPermission no longer stops the stream itself -- this is the one place
  // that releases it, called on every exit path (a new stream about to replace this one, a
  // granted-but-unusable stream after play() rejects, or a confirmed-successful finish), not
  // just the happy path. Resets every piece of camera-related state together, caught missing
  // individually in TabeenRaoof's PR #134 review: the stream's tracks, state.cameraStream,
  // state.cameraGranted (stale "granted" after release is misleading, not just unused), the
  // preview's srcObject (a stale frozen frame otherwise), and the capture button.
  function releaseCamera(): void {
    if (state.cameraStream !== null) {
      stopCameraStream(state.cameraStream);
    }
    state.cameraStream = null;
    state.cameraGranted = false;
    cameraPreview.srcObject = null;
    updateCaptureAvailability();
  }

  // Capture needs both a live stream to draw from and a gyro reading to pair with (the "never a
  // stale or default reading" acceptance criterion) -- gating on camera alone (the bug
  // TabeenRaoof's review caught: capture enabled the moment the camera started, even with gyro
  // denied) left captureFrame's own "No gyro reading yet" message permanently misleading rather
  // than the transient-delay message it's meant to be.
  function updateCaptureAvailability(): void {
    captureButton.disabled = !(state.cameraStream !== null && state.gyroGranted);
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
      // Releases any stream left over from a previous Start click before this one replaces it --
      // without this, a second Start (e.g. retrying after session creation failed) overwrote
      // state.cameraStream with a new stream and left the first one's tracks running forever
      // (TabeenRaoof's PR #134 review).
      releaseCamera();
      state.cameraGranted = cameraResult.granted;
      if (cameraResult.granted) {
        state.cameraStream = cameraResult.stream;
        cameraPreview.srcObject = cameraResult.stream;
        // Captured so the .then() callbacks below can tell whether this is still the current
        // stream by the time they run -- otherwise a slow play() resolving after a later
        // Start/Finish already released or replaced it could resurrect "Camera started." or
        // re-enable capture on a stream that no longer exists (TabeenRaoof's PR #134 review).
        const thisStream = cameraResult.stream;
        // Same "autoplay alone isn't reliable once a stream is attached after the element
        // already exists" fix mountCameraCheck's preview needed (C-02) -- play() explicitly so a
        // silently-stalled preview fails loudly instead.
        cameraPreview.play().then(
          () => {
            if (state.cameraStream !== thisStream) {
              return;
            }
            setStatus(cameraStatus, "Camera started.", "info");
            updateCaptureAvailability();
          },
          (playError: unknown) => {
            if (state.cameraStream !== thisStream) {
              return;
            }
            setStatus(
              cameraStatus,
              `Camera permission granted, but the preview failed to play: ${describeCameraError(playError)}`,
              "error",
            );
            // Granted but unusable -- holding it open would be exactly the camera-stays-on bug
            // this story exists to prevent, just reached through a different path than a normal
            // finish (TabeenRaoof's PR #134 review).
            releaseCamera();
          },
        );
      } else {
        setStatus(cameraStatus, cameraResult.message, "error");
      }

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
      updateCaptureAvailability();

      // Nothing left for "Start" to do on this screen once a session exists -- only re-enable
      // if session creation itself failed (rejected or resolved ok:false), so the order code
      // can be fixed and retried.
      startButton.disabled = sessionOk;
    })();
  });

  captureButton.addEventListener("click", () => {
    if (state.latestGyro === null) {
      setStatus(captureStatus, "No gyro reading yet -- try again in a moment.", "error");
      return;
    }
    try {
      state.lastCapturedFrame = captureFrame(cameraPreview, state.latestGyro);
      const { width, height } = state.lastCapturedFrame.pixels;
      setStatus(captureStatus, `Captured a ${String(width)}x${String(height)} frame.`, "info");
    } catch (error) {
      setStatus(captureStatus, error instanceof Error ? error.message : String(error), "error");
    }
  });

  async function runReconstructionPolling(): Promise<void> {
    if (state.sessionId === null) {
      return;
    }
    retryReconstructionButton.hidden = true;
    remountModelViewer({ status: "pending" });
    const outcome = await pollReconstruction(
      options.baseUrl,
      state.sessionId,
      reconstructionStatus,
      options.reconstructionMaxAttempts ?? DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS,
      options.reconstructionPollIntervalMs ?? DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS,
      (terminalResult) => {
        remountModelViewer(terminalResult);
      },
    );
    retryReconstructionButton.hidden = outcome !== "timeout" && outcome !== "error";
    if (outcome === "timeout") {
      remountModelViewer({
        status: "unavailable",
        message: "3D model unavailable because reconstruction timed out. Retry reconstruction.",
      });
    } else if (outcome === "error") {
      remountModelViewer({
        status: "unavailable",
        message:
          "3D model unavailable because reconstruction status could not be fetched. Retry reconstruction.",
      });
    }
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
        setStatus(finishStatus, describeFinishSession(result), result.ok ? "info" : "error");
        if (!result.ok) {
          finishButton.disabled = false;
          return;
        }
        // The session has actually ended now -- release the camera rather than leaving a real
        // camera-stays-on bug running (this story's own acceptance criteria).
        releaseCamera();
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
  onTerminalResult: (result: ReconstructionResult) => void,
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
    onTerminalResult(result.body);
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
