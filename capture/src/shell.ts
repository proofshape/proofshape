// C-04: replaces C-01/C-02/C-03's three isolated demo sections (#app/#camera-app/#gyro-app) with
// one real flow — create a session, request both permissions together — and gives C-05 onward
// one shared place to read/write app state instead of per-module globals.

import {
  captureFrame,
  describeCameraError,
  encodeFrameAsJpeg,
  requestCameraPermission,
  setStatus,
  stopCameraStream,
  type CapturedFrame,
} from "./camera.js";
import { createSession, finishSession, getReconstruction, uploadFrame } from "./api.js";
import { requireElement, requireTyped } from "./dom.js";
import { checkBlur, checkDuplicate, checkExposure, type FramePixels } from "./frame_gate.js";
import { requestGyroPermission } from "./gyro.js";
import { describeCreateSession, describeFinishSession, describeUploadFrame } from "./messages.js";
import {
  mountModelViewer,
  type ModelViewerOptions,
  type ModelViewerState,
} from "./model-viewer.js";
import { mountGyroOverlay, type GyroOverlayHandle } from "./overlay.js";
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
  // C-09: true only once the preview is actually rendering live frames (play() resolved), not
  // merely once a stream was granted -- a granted-but-not-yet-playing (or no-longer-playing)
  // stream has nothing a capture could draw from (TabeenRaoof's PR #137 review).
  cameraPlaying: boolean;
  // C-09: overwritten on each capture attempt (accepted or not) -- the capture loop's own
  // gate/upload bookkeeping (lastAcceptedForGate, below mountAppShell) is separate from this.
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
    <div id="shell-gyro-overlay"></div>
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
  const gyroOverlayContainer = requireElement(root, "#shell-gyro-overlay");
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
    cameraPlaying: false,
    lastCapturedFrame: null,
  };

  // Scoped to this mount, same reasoning as gyro.ts's own `listening` flag.
  let listeningForGyro = false;
  // C-09: mounted once here (so the shell shows "Capture the first frame to start." immediately),
  // and remounted at the top of every Start click (below) -- overlay.ts's own comment already
  // anticipates this ("a second mount on the same container... can't leak the first mount's
  // listener"), so a retry after an abandoned session starts coverage tracking fresh rather than
  // carrying over slots covered under a session that never actually happened.
  let gyroOverlayHandle: GyroOverlayHandle = mountGyroOverlay(gyroOverlayContainer);
  // C-09: the last frame that passed the *local* gate (frame_gate.ts's own "lastAccepted" naming
  // means "accepted by the gate," not "accepted by the server") -- separate from
  // state.lastCapturedFrame, which is just the most recent capture attempt, accepted or not.
  // Reset alongside gyroOverlayHandle on every Start click, for the same reason.
  let lastAcceptedForGate: { frame: FramePixels; gyro: Gyro } | null = null;
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
      // The *first* reading (not just the grant) is what can flip capture on -- see
      // updateCaptureAvailability's own comment.
      updateCaptureAvailability();
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
    state.cameraPlaying = false;
    cameraPreview.srcObject = null;
    // "Camera started." otherwise stayed on screen after the camera was actually off (TabeenRaoof's
    // PR #137 review). A caller with a more specific reason (e.g. play() rejecting) calls
    // releaseCamera() first, then overwrites this with that reason.
    setStatus(cameraStatus, "Camera stopped.", "info");
    updateCaptureAvailability();
  }

  // Capture needs a preview that's actually rendering live frames, not merely a stream that was
  // granted (play() can still be pending, or have rejected), and a real gyro reading, not just
  // permission granted -- gyroGranted can be true with no sensor ever firing an event (desktop,
  // Android, iOS without motion access ever prompted), which left captureFrame's own "No gyro
  // reading yet" message permanently misleading rather than the transient delay it's meant to be
  // (TabeenRaoof's PR #137 review).
  function updateCaptureAvailability(): void {
    captureButton.disabled = !(state.cameraPlaying && state.latestGyro !== null);
  }

  startButton.addEventListener("click", () => {
    setStatus(sessionStatus, "", "info");
    setStatus(cameraStatus, "", "info");
    setStatus(gyroStatus, "", "info");
    startButton.disabled = true;
    // Fresh coverage tracking and gate-dedup state for this attempt -- without this, a frame
    // locally accepted under an abandoned session (e.g. a bad order code, then retried) would
    // still anchor duplicate-detection and orbit coverage for the next, real session
    // (overlay.ts's own comment anticipates this exact remount).
    gyroOverlayHandle = mountGyroOverlay(gyroOverlayContainer);
    lastAcceptedForGate = null;

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
        if (!sessionOk) {
          // The gesture-preserving getUserMedia call already happened (it has to fire before we
          // know whether the session succeeded), but there's no session to capture into -- rather
          // than attach an unusable stream to the preview and leave it running until the user
          // notices and retries, stop it immediately (TabeenRaoof's PR #137 review, item 2).
          stopCameraStream(cameraResult.stream);
          setStatus(
            cameraStatus,
            "Camera granted, but no session is active -- fix the order code and try again.",
            "info",
          );
        } else {
          state.cameraStream = cameraResult.stream;
          cameraPreview.srcObject = cameraResult.stream;
          // Captured so the .then() callbacks below can tell whether this is still the current
          // stream by the time they run -- otherwise a slow play() resolving after a later
          // Start/Finish already released or replaced it could resurrect "Camera started." or
          // re-enable capture on a stream that no longer exists (TabeenRaoof's PR #134 review).
          const thisStream = cameraResult.stream;
          // Same "autoplay alone isn't reliable once a stream is attached after the element
          // already exists" fix mountCameraCheck's preview needed (C-02) -- play() explicitly so
          // a silently-stalled preview fails loudly instead.
          cameraPreview.play().then(
            () => {
              if (state.cameraStream !== thisStream) {
                return;
              }
              state.cameraPlaying = true;
              setStatus(cameraStatus, "Camera started.", "info");
              updateCaptureAvailability();
            },
            (playError: unknown) => {
              if (state.cameraStream !== thisStream) {
                return;
              }
              // Granted but unusable -- holding it open would be exactly the camera-stays-on bug
              // this story exists to prevent, just reached through a different path than a normal
              // finish (TabeenRaoof's PR #134 review). releaseCamera() first (it sets a generic
              // "Camera stopped." status), then this specific reason overwrites it.
              releaseCamera();
              setStatus(
                cameraStatus,
                `Camera permission granted, but the preview failed to play: ${describeCameraError(playError)}`,
                "error",
              );
            },
          );
        }
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
    // Two more layers of defense beyond the disabled attribute itself (fails loudly rather than
    // silently misbehaving if this handler is ever reached in a state the button shouldn't allow):
    if (state.latestGyro === null) {
      setStatus(captureStatus, "No gyro reading yet -- try again in a moment.", "error");
      return;
    }
    if (state.sessionId === null) {
      setStatus(captureStatus, "Start a session first.", "error");
      return;
    }
    const sessionId = state.sessionId;

    let captured: CapturedFrame;
    try {
      captured = captureFrame(cameraPreview, state.latestGyro);
    } catch (error) {
      setStatus(captureStatus, error instanceof Error ? error.message : String(error), "error");
      return;
    }
    state.lastCapturedFrame = captured;

    // C-06's three checks, in the story's own stated order (sharp, well-exposed, not a
    // duplicate), first failure wins -- nothing is uploaded on a gate rejection. The code is
    // shown alongside the message, not just the message alone -- checkDuplicate's own message
    // ("Gyro delta ... are both below their thresholds") never actually says "duplicate," so the
    // code is what makes the specific reason identifiable (this story's own acceptance criteria
    // name all three codes explicitly).
    const blurResult = checkBlur(captured.pixels);
    if (!blurResult.ok) {
      setStatus(
        captureStatus,
        `Frame rejected (${blurResult.code}) — ${blurResult.message}`,
        "error",
      );
      return;
    }
    const exposureResult = checkExposure(captured.pixels);
    if (!exposureResult.ok) {
      setStatus(
        captureStatus,
        `Frame rejected (${exposureResult.code}) — ${exposureResult.message}`,
        "error",
      );
      return;
    }
    const duplicateResult = checkDuplicate(captured.pixels, captured.gyro, lastAcceptedForGate);
    if (!duplicateResult.ok) {
      setStatus(
        captureStatus,
        `Frame rejected (${duplicateResult.code}) — ${duplicateResult.message}`,
        "error",
      );
      return;
    }

    // Accepted by the local gate -- anchors the next duplicate check, independent of whatever the
    // server goes on to say about this particular upload.
    lastAcceptedForGate = { frame: captured.pixels, gyro: captured.gyro };

    captureButton.disabled = true;
    setStatus(captureStatus, "Uploading…", "info");
    void (async () => {
      try {
        const jpeg = await encodeFrameAsJpeg(captured.pixels);
        const result = await uploadFrame(options.baseUrl, sessionId, jpeg, captured.gyro);
        setStatus(captureStatus, describeUploadFrame(result), result.ok ? "info" : "error");
        if (result.ok) {
          // Only a real server acceptance marks orbit coverage -- a 422 shows the server's
          // reason but must not advance the arrow.
          gyroOverlayHandle.reportAcceptedFrame(captured.gyro);
        }
      } catch (error) {
        setStatus(
          captureStatus,
          `Could not upload frame — ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
      } finally {
        updateCaptureAvailability();
      }
    })();
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
      let finished = false;
      try {
        const result = await finishSession(options.baseUrl, state.sessionId);
        setStatus(finishStatus, describeFinishSession(result), result.ok ? "info" : "error");
        finished = result.ok;
      } catch (error) {
        setStatus(
          finishStatus,
          `Could not finish session — ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
      }
      if (!finished) {
        finishButton.disabled = false;
        return;
      }
      // Everything from here on runs only once finishSession has genuinely resolved ok:true, and
      // outside the try above -- a failure past this point (e.g. the reconstruction poll, or the
      // viewer it rebuilds) is not a "could not finish session" error, and mislabeling it as one
      // also wrongly re-enabled Finish on an already-finished session (TabeenRaoof's PR #137
      // review, item 9).
      releaseCamera();
      await runReconstructionPolling();
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
