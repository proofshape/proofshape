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
  type StatusKind,
} from "./camera.js";
import {
  createSession,
  finishSession,
  getAgentMessage,
  getReconstruction,
  uploadFrame,
} from "./api.js";
import { mountAgentConversation } from "./agent-conversation.js";
import { requireElement, requireTyped } from "./dom.js";
import {
  checkBlur,
  checkDuplicate,
  checkExposure,
  measureBlurVariance,
  type FramePixels,
} from "./frame_gate.js";
import { requestGyroPermission } from "./gyro.js";
import {
  describeCreateSession,
  describeFinishSession,
  describeContinuedSession,
  describeNetworkError,
  describeSavedSession,
  describeUploadError,
  describeUploadFrame,
  describeVlmFeedback,
} from "./messages.js";
import {
  browserSessionStorage,
  clearCaptureSession,
  loadCaptureSession,
  saveCaptureSession,
  type KeyValueStorage,
  type StoredCaptureSession,
} from "./session-persistence.js";
import {
  mountModelViewer,
  type ModelViewerOptions,
  type ModelViewerState,
} from "./model-viewer.js";
import {
  INITIAL_AUTO_CAPTURE_STATE,
  stepAutoCapture,
  type AutoCaptureState,
} from "./auto_capture.js";
import { DEFAULT_ORBIT } from "./orbit.js";
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
  // C-13: the bearer token the deployed service's Lightning proxy requires (D-042); absent
  // against the local contract stub, which expects no auth header.
  authToken?: string;
  // C-13: pre-fills the order-code input from a capture link's `?order=` query param, so a
  // supplier doesn't have to type it in.
  initialOrderCode?: string;
  reconstructionPollIntervalMs?: number;
  reconstructionMaxAttempts?: number;
  modelViewerOptions?: ModelViewerOptions;
  // C-24: where an in-progress session is kept across a reload. Defaults to the tab's
  // sessionStorage; tests pass their own, and null turns resume off.
  captureSessionStorage?: KeyValueStorage | null;
  // C-25 review calibration aid. When true, store the exact pre-JPEG Laplacian variance for each
  // server-accepted frame in sessionStorage, keyed by the server frame index. No pixels/photos are
  // retained. main.ts enables this only with ?blur-debug=1.
  blurDebug?: boolean;
  // Test/debug injection only. When omitted, blur-debug uses the browser's sessionStorage.
  // A null value deliberately represents unavailable storage so the warning path is testable.
  blurDebugStorage?: KeyValueStorage | null;
  // C-27: the clock auto-capture's hold is timed against. Tests pass their own; the browser
  // default is performance.now().
  now?: () => number;
}

export const C25_BLUR_DEBUG_STORAGE_KEY = "proofshape.c25BlurDebug.v1";

export interface C25BlurDebugMeasurement {
  session_id: string;
  frame_index: number;
  pre_jpeg_variance: number;
  width: number;
  height: number;
}

function appendC25BlurDebugMeasurement(
  storage: KeyValueStorage | null,
  measurement: C25BlurDebugMeasurement,
): number | null {
  if (storage === null) {
    return null;
  }
  let measurements: C25BlurDebugMeasurement[] = [];
  try {
    const raw = storage.getItem(C25_BLUR_DEBUG_STORAGE_KEY);
    if (raw !== null) {
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        measurements = parsed as C25BlurDebugMeasurement[];
      }
    }
    measurements.push(measurement);
    storage.setItem(C25_BLUR_DEBUG_STORAGE_KEY, JSON.stringify(measurements));
    return measurements.filter((item) => item.session_id === measurement.session_id).length;
  } catch {
    // Debug evidence must never interrupt a supplier capture. The caller surfaces the failure in
    // the debug-only status line so an operator cannot mistake a partial table for complete data.
    return null;
  }
}

const DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS = 2000;
const DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS = 300;

export const APP_SHELL_HTML = `
  <section>
    <h2>Start capture session (C-04)</h2>
    <label>Order code <input id="shell-order-code" type="text" /></label>
    <button id="shell-start-button" type="button">Start</button>
    <div id="shell-resume" hidden>
      <p id="shell-resume-text"></p>
      <button id="shell-continue-button" type="button" hidden>Continue session</button>
      <button id="shell-show-result-button" type="button" hidden>Show result</button>
      <button id="shell-discard-button" type="button">Start new</button>
    </div>
    <p id="shell-session-status"></p>
    <p id="shell-camera-status"></p>
    <p id="shell-gyro-status"></p>
    <div id="shell-viewfinder" class="viewfinder" data-guidance="idle">
      <video id="shell-camera-preview" muted playsinline autoplay></video>
      <div id="shell-gyro-overlay"></div>
      <button
        id="shell-capture-button"
        class="shutter"
        type="button"
        aria-label="Capture frame"
        disabled
      ><span class="visually-hidden">Capture frame</span></button>
    </div>
    <label class="auto-capture-toggle">
      <input id="shell-auto-capture" type="checkbox" />
      Take the photo automatically when on target
    </label>
    <p id="shell-capture-status"></p>
    <p id="shell-blur-debug-status" hidden></p>
    <p id="shell-vlm-hint"></p>
  </section>
  <section>
    <div id="shell-agent-conversation"></div>
  </section>
  <section>
    <h2>Finish and reconstruct (C-10)</h2>
    <button id="shell-finish-button" class="primary-action" type="button" disabled>
      Finish session
    </button>
    <button id="shell-retry-reconstruction-button" type="button" hidden>
      Retry reconstruction
    </button>
    <p id="shell-finish-status"></p>
    <p id="shell-reconstruction-status"></p>
    <a id="shell-report-link" hidden>Open condition report</a>
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
  orderCodeInput.value = options.initialOrderCode ?? "";
  const startButton = requireTyped(root, "#shell-start-button", HTMLButtonElement, "<button>");
  const resumePanel = requireElement(root, "#shell-resume");
  const resumeText = requireElement(root, "#shell-resume-text");
  const continueButton = requireTyped(
    root,
    "#shell-continue-button",
    HTMLButtonElement,
    "<button>",
  );
  const showResultButton = requireTyped(
    root,
    "#shell-show-result-button",
    HTMLButtonElement,
    "<button>",
  );
  const discardButton = requireTyped(root, "#shell-discard-button", HTMLButtonElement, "<button>");
  const sessionStatus = requireElement(root, "#shell-session-status");
  const cameraStatus = requireElement(root, "#shell-camera-status");
  const gyroStatus = requireElement(root, "#shell-gyro-status");
  const cameraPreview = requireTyped(root, "#shell-camera-preview", HTMLVideoElement, "<video>");
  const captureButton = requireTyped(root, "#shell-capture-button", HTMLButtonElement, "<button>");
  const viewfinder = requireElement(root, "#shell-viewfinder");
  const autoCaptureToggle = requireTyped(root, "#shell-auto-capture", HTMLInputElement, "<input>");
  const captureStatus = requireElement(root, "#shell-capture-status");
  const blurDebugStatus = requireElement(root, "#shell-blur-debug-status");
  const vlmHint = requireElement(root, "#shell-vlm-hint");
  const gyroOverlayContainer = requireElement(root, "#shell-gyro-overlay");
  const agentConversationContainer = requireElement(root, "#shell-agent-conversation");
  const finishButton = requireTyped(root, "#shell-finish-button", HTMLButtonElement, "<button>");
  const retryReconstructionButton = requireTyped(
    root,
    "#shell-retry-reconstruction-button",
    HTMLButtonElement,
    "<button>",
  );
  const finishStatus = requireElement(root, "#shell-finish-status");
  const reconstructionStatus = requireElement(root, "#shell-reconstruction-status");
  const reportLink = requireTyped(root, "#shell-report-link", HTMLAnchorElement, "<a>");
  const modelViewer = requireElement(root, "#shell-model-viewer");

  const agentConversation = mountAgentConversation(agentConversationContainer);

  const blurDebugStorage =
    options.blurDebug === true
      ? options.blurDebugStorage === undefined
        ? browserSessionStorage()
        : options.blurDebugStorage
      : null;
  if (options.blurDebug === true) {
    blurDebugStatus.hidden = false;
    if (blurDebugStorage === null) {
      setStatus(
        blurDebugStatus,
        "Blur debug warning: measurement storage is unavailable; paired evidence will be incomplete.",
        "error",
      );
    } else {
      setStatus(
        blurDebugStatus,
        "Blur debug active: measurement storage is available; stored count will update after accepted frames.",
        "info",
      );
    }
  }

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
  // C-26: every overlay mount colours the viewfinder's frame by the overlay's guidance level
  // (capture.css reads data-guidance: idle, far, near, on-target, complete).
  //
  // C-27: each update also steps auto-capture. stepAutoCapture only says "the phone has been on
  // target and steady long enough"; whether a capture may start at all is the shutter's own
  // enabled state (session, camera, gyro, no upload in flight, not finishing), checked here so
  // the two paths can never disagree. Before the first photo the level is "idle", so the first
  // frame is always a tap: it anchors the orbit's heading.
  //
  // The switch is off by default (the freeze exception on #436): until the person capturing turns
  // it on, every photo is a tap, as before C-27.
  const now = options.now ?? (() => performance.now());
  let autoCaptureState: AutoCaptureState = INITIAL_AUTO_CAPTURE_STATE;
  function mountOverlay(): GyroOverlayHandle {
    autoCaptureState = INITIAL_AUTO_CAPTURE_STATE;
    return mountGyroOverlay(gyroOverlayContainer, DEFAULT_ORBIT, {
      onGuidance: (update) => {
        viewfinder.dataset.guidance = update.level;
        // #420 review (@dalwalyk): the overlay listens from page load, before the motion prompt,
        // so its readings count only once the shell's own gyro listener runs (after the grant).
        // Its listener also runs before the shell's for the same event, so the shutter's state is
        // refreshed here; otherwise auto-capture would read the previous event's.
        if (listeningForGyro && update.gyro !== null) {
          state.latestGyro = update.gyro;
          updateCaptureAvailability();
        }
        const step = stepAutoCapture(autoCaptureState, {
          nowMs: now(),
          level: update.level,
          gyro: update.gyro,
          enabled: autoCaptureToggle.checked && !captureButton.disabled,
        });
        autoCaptureState = step.state;
        if (step.fire) {
          attemptCapture("auto");
        }
      },
      onTargetText: () =>
        autoCaptureToggle.checked ? "On target — hold steady…" : "On target — take the photo.",
    });
  }

  let flashTimer: ReturnType<typeof setTimeout> | null = null;
  function flashViewfinder(): void {
    viewfinder.classList.add("viewfinder-flash");
    if (flashTimer !== null) clearTimeout(flashTimer);
    flashTimer = setTimeout(() => {
      viewfinder.classList.remove("viewfinder-flash");
      flashTimer = null;
    }, 150);
  }
  // C-09: mounted once here (so the shell shows the first-photo prompt immediately),
  // and remounted at the top of every Start click (below) -- overlay.ts's own comment already
  // anticipates this ("a second mount on the same container... can't leak the first mount's
  // listener"), so a retry after an abandoned session starts coverage tracking fresh rather than
  // carrying over slots covered under a session that never actually happened.
  let gyroOverlayHandle: GyroOverlayHandle = mountOverlay();
  // C-09: the last frame that passed the *local* gate (frame_gate.ts's own "lastAccepted" naming
  // means "accepted by the gate," not "accepted by the server") -- separate from
  // state.lastCapturedFrame, which is just the most recent capture attempt, accepted or not.
  // Reset alongside gyroOverlayHandle on every Start click, for the same reason.
  let lastAcceptedForGate: { frame: FramePixels; gyro: Gyro } | null = null;
  // TabeenRaoof's PR #139 review: the deviceorientation listener calls updateCaptureAvailability
  // on every reading (~60/s on a phone), so without this flag the very next reading re-enabled
  // Capture a few milliseconds after a tap disabled it -- a second tap started a second, concurrent
  // upload. Also gates Finish, which could otherwise run while an upload was still in flight.
  let uploadInFlight = false;
  // C-25: once any blur-debug measurement fails to store, that run is unusable for calibration.
  // Keep the warning sticky across later successful writes so the operator cannot miss it.
  let blurDebugStorageFailures = 0;
  // The reverse race of uploadInFlight, caught on a second pass of the same review: clicking
  // Finish only ever disabled Finish, not Capture, so a capture started right after could still
  // be mid-upload when finishSession resolved -- and that upload's own `finally` called
  // updateFinishAvailability(), which re-enabled Finish on a session that had already finished.
  // Set the moment Finish is clicked (before finishSession is even awaited) and reset to false
  // only if finishSession itself fails (so a retry is possible); left `true` forever once it
  // actually succeeds, since there's nothing left in this session to capture into or finish again.
  let sessionFinishing = false;
  // C-22: only the newest assistant request may update the panel. Accepted frames can trigger
  // overlapping optional guidance calls, so an older reply must never land after a newer one.
  let latestAgentRequestId = 0;
  // C-24 (#352): the tab's saved copy of the current session, so a reload (iOS does one when the
  // supplier comes back from another app) can carry on instead of starting over. `savedSession`
  // is what was found at mount and is offered in the resume panel; `activeSession` is the
  // session this page is working on now, rewritten after every accepted frame.
  const captureStorage =
    options.captureSessionStorage === undefined
      ? browserSessionStorage()
      : options.captureSessionStorage;
  let savedSession: StoredCaptureSession | null = loadCaptureSession(captureStorage);
  let activeSession: StoredCaptureSession | null = null;

  function persistActiveSession(): void {
    if (activeSession !== null) {
      saveCaptureSession(captureStorage, activeSession);
    }
  }

  function forgetSession(): void {
    clearCaptureSession(captureStorage);
    activeSession = null;
    savedSession = null;
  }

  function hideResumePanel(): void {
    resumePanel.hidden = true;
    continueButton.hidden = true;
    showResultButton.hidden = true;
  }

  function showResumePanel(saved: StoredCaptureSession): void {
    resumeText.textContent = describeSavedSession(saved);
    continueButton.hidden = saved.phase !== "capturing";
    showResultButton.hidden = saved.phase !== "finished";
    resumePanel.hidden = false;
  }
  // C-13: the deployed service's glb_url is behind the same bearer-token gate as every other
  // route (D-042/D-048) -- the model viewer needs authToken too, not just the session calls.
  const modelViewerOptions: ModelViewerOptions = {
    ...options.modelViewerOptions,
    ...(options.authToken !== undefined ? { authToken: options.authToken } : {}),
  };
  let disposeModelViewer = mountModelViewer(modelViewer, { status: "pending" }, modelViewerOptions);
  function remountModelViewer(result: ModelViewerState): void {
    disposeModelViewer();
    disposeModelViewer = mountModelViewer(modelViewer, result, modelViewerOptions);
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
    // Only claim "stopped" (and only touch the status) when something was actually running --
    // otherwise this ran unconditionally at the top of every camera-grant attempt (including the
    // very first one, with nothing to release yet) and "Camera stopped." flashed on screen during
    // a normal, successful Start, until play() overwrote it a moment later (TabeenRaoof's PR #139
    // review).
    if (state.cameraStream !== null) {
      stopCameraStream(state.cameraStream);
      setStatus(cameraStatus, "Camera stopped.", "info");
    }
    state.cameraStream = null;
    state.cameraGranted = false;
    state.cameraPlaying = false;
    cameraPreview.srcObject = null;
    updateCaptureAvailability();
  }

  // Capture needs a preview that's actually rendering live frames, not merely a stream that was
  // granted (play() can still be pending, or have rejected), and a real gyro reading, not just
  // permission granted -- gyroGranted can be true with no sensor ever firing an event (desktop,
  // Android, iOS without motion access ever prompted), which left captureFrame's own "No gyro
  // reading yet" message permanently misleading rather than the transient delay it's meant to be
  // (TabeenRaoof's PR #137 review). Also gated on uploadInFlight (TabeenRaoof's PR #139 review) --
  // without it, a deviceorientation reading arriving a few milliseconds after a tap disabled this
  // button would immediately re-enable it, mid-upload.
  // C-24 review (@dalwalyk, PR #375): Capture also needs a live session, as Finish does. Dropping
  // a dead session nulls `state.sessionId` while the camera and gyro stay live, and Capture
  // stayed clickable with nothing to upload to. The click handler's "Start a session first."
  // guard remains as the backstop.
  function updateCaptureAvailability(): void {
    captureButton.disabled =
      uploadInFlight ||
      sessionFinishing ||
      state.sessionId === null ||
      !(state.cameraPlaying && state.latestGyro !== null);
  }

  // Finish must not run while an upload is still in flight (TabeenRaoof's PR #139 review): a tap
  // right after a first good frame could get a false 409 "too few frames" just before that frame
  // is accepted, and a session finishing while a later upload is still pending would land that
  // upload on an already-finished session. sessionFinishing covers the reverse: once Finish has
  // been clicked, Finish itself must not become clickable again from some other, unrelated state
  // change (a lagging capture's own finally) while it's still resolving or already succeeded.
  function updateFinishAvailability(): void {
    finishButton.disabled = uploadInFlight || sessionFinishing || state.sessionId === null;
  }

  type CaptureStart = { kind: "new" } | { kind: "continue"; saved: StoredCaptureSession };

  // C-24: Start and Continue share this, so both keep the iOS gesture rule below (the camera and
  // motion prompts fire synchronously from the tap) and the same camera/gyro handling. Continue
  // skips createSession -- the session already exists on the server, which numbers the next
  // frame from its own count -- and rebuilds coverage from the saved readings instead.
  function beginCapture(start: CaptureStart): void {
    setStatus(sessionStatus, "", "info");
    setStatus(cameraStatus, "", "info");
    setStatus(gyroStatus, "", "info");
    // A-11: a previous session's tip is about a different part. Start stays disabled once a
    // session exists, so today this only guards a future flow that re-enables it.
    setStatus(vlmHint, "", "info");
    reportLink.hidden = true;
    reportLink.removeAttribute("href");
    startButton.disabled = true;
    // Fresh coverage tracking and gate-dedup state for this attempt -- without this, a frame
    // locally accepted under an abandoned session (e.g. a bad order code, then retried) would
    // still anchor duplicate-detection and orbit coverage for the next, real session
    // (overlay.ts's own comment anticipates this exact remount).
    gyroOverlayHandle = mountOverlay();
    agentConversation.clear();
    lastAcceptedForGate = null;
    // C-24: replaying the accepted readings in order rebuilds coverage exactly, since the overlay
    // anchors on the first one. The duplicate gate's anchor (the last frame's pixels) isn't
    // saved, so the first frame after continuing is checked for blur and exposure only.
    if (start.kind === "continue") {
      for (const reading of start.saved.acceptedGyro) {
        gyroOverlayHandle.reportAcceptedFrame(reading);
      }
    }
    // A previous session's finish (success or in-flight-then-abandoned) must not carry over and
    // leave Capture/Finish permanently disabled for this new attempt.
    sessionFinishing = false;

    // All three started synchronously, before awaiting any of them: getUserMedia and
    // requestPermission (inside requestCameraPermission/requestGyroPermission) must have no
    // leading `await` to preserve the iOS gesture that triggered this click; createSession has
    // no such constraint, but starting it here too keeps this one real flow from one click,
    // rather than literally serializing session-creation ahead of the gesture-critical calls.
    const sessionPromise =
      start.kind === "new"
        ? createSession(options.baseUrl, orderCodeInput.value, options.authToken)
        : null;
    const cameraPromise = requestCameraPermission();
    const gyroPromise = requestGyroPermission();

    void (async () => {
      // createSession can genuinely reject, not just resolve with ok:false -- a real network
      // failure, a non-JSON response, or any status outside 201/400 all throw inside api.ts.
      // Catching here (TabeenRaoof's PR #102 review, reproduced: an unstubbed rejection left the
      // button disabled forever with camera/gyro never reflected, even though their prompts had
      // already fired) is what keeps that from silently stopping the user with no explanation.
      let sessionOk = false;
      if (start.kind === "continue") {
        state.sessionId = start.saved.sessionId;
        activeSession = { ...start.saved, acceptedGyro: [...start.saved.acceptedGyro] };
        orderCodeInput.value = start.saved.orderCode;
        setStatus(sessionStatus, describeContinuedSession(start.saved), "info");
        updateCaptureAvailability();
        updateFinishAvailability();
        sessionOk = true;
      } else if (sessionPromise !== null) {
        try {
          const sessionResult = await sessionPromise;
          setStatus(
            sessionStatus,
            describeCreateSession(sessionResult),
            sessionResult.ok ? "info" : "error",
          );
          if (sessionResult.ok) {
            state.sessionId = sessionResult.body.session_id;
            activeSession = {
              sessionId: sessionResult.body.session_id,
              orderCode: sessionResult.body.order_code,
              phase: "capturing",
              acceptedFrameCount: 0,
              acceptedGyro: [],
            };
            persistActiveSession();
            updateCaptureAvailability();
            updateFinishAvailability();
            sessionOk = true;
          }
        } catch (error) {
          setStatus(
            sessionStatus,
            `Could not start session — ${describeNetworkError(error)}`,
            "error",
          );
        }
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
          // Consistent with every other "stream got stopped without ever really being usable"
          // path (releaseCamera() does the same) -- cameraGranted means "there's a usable camera
          // right now," not just "permission was granted at some point" (TabeenRaoof's PR #139
          // review).
          state.cameraGranted = false;
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
              // C-26: bring the whole viewfinder (arrow and shutter included) into view, since
              // the session controls above it can push it below the fold on a phone.
              viewfinder.scrollIntoView({ block: "start", behavior: "smooth" });
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
  }

  startButton.addEventListener("click", () => {
    // Start always means a new session: a saved one from earlier in this tab is dropped. Its
    // server-side session stays "capturing" (expiring those is out of C-24's scope).
    forgetSession();
    hideResumePanel();
    beginCapture({ kind: "new" });
  });

  continueButton.addEventListener("click", () => {
    if (savedSession === null || savedSession.phase !== "capturing") {
      return;
    }
    const saved = savedSession;
    savedSession = null;
    hideResumePanel();
    beginCapture({ kind: "continue", saved });
  });

  discardButton.addEventListener("click", () => {
    forgetSession();
    hideResumePanel();
  });

  // C-24: a reload during reconstruction (Finish succeeded, the page was polling) comes back
  // here. The server is still reconstructing or already done, so this only resumes polling; the
  // camera and motion sensors aren't needed and aren't asked for.
  showResultButton.addEventListener("click", () => {
    if (savedSession === null || savedSession.phase !== "finished") {
      return;
    }
    const saved = savedSession;
    savedSession = null;
    hideResumePanel();
    state.sessionId = saved.sessionId;
    activeSession = { ...saved, acceptedGyro: [...saved.acceptedGyro] };
    orderCodeInput.value = saved.orderCode;
    sessionFinishing = true;
    startButton.disabled = true;
    updateCaptureAvailability();
    updateFinishAvailability();
    gyroOverlayHandle.dispose();
    setStatus(finishStatus, `Session finished: ${saved.sessionId}`, "info");
    void runReconstructionPolling();
  });

  if (savedSession !== null) {
    showResumePanel(savedSession);
  }

  async function refreshAgentConversation(sessionId: string): Promise<void> {
    const requestId = ++latestAgentRequestId;
    try {
      const result = await getAgentMessage(options.baseUrl, sessionId, options.authToken);
      if (state.sessionId !== sessionId || requestId !== latestAgentRequestId) {
        return;
      }
      if (!result.ok) {
        // Today's real service has no agent route yet. Keep the panel hidden until a real 200 has
        // existed; after one has existed, transportUnavailable changes only the status line.
        agentConversation.transportUnavailable();
        return;
      }
      // C-22 mirrors A-07's finished field for the future D-031 wire, but does not make that
      // mock-only field drive session lifecycle yet. The real phone contract must define whether
      // the service has already finished the session and how reconstruction should be resumed.
      agentConversation.show(result.body);
    } catch {
      if (state.sessionId !== sessionId || requestId !== latestAgentRequestId) {
        return;
      }
      // Transport failure is not an A-07 fallback reason. The view keeps that client-side state
      // separate and leaves deterministic capture guidance usable.
      agentConversation.transportUnavailable();
    }
  }

  // C-27: one capture path for the shutter and for auto-capture. `trigger` only changes how the
  // status line reads ("Auto: ..."), so a supplier can tell an automatic rejection from their own.
  function attemptCapture(trigger: "manual" | "auto"): void {
    const say = (message: string, kind: StatusKind): void => {
      setStatus(captureStatus, trigger === "auto" ? `Auto: ${message}` : message, kind);
    };
    // #420 review (@dalwalyk): any photo that fails to upload (network, or the server rejects it)
    // turns auto-capture off, automatic or tapped. Holding on target would otherwise send a new
    // photo every retry gap for as long as the phone stays there. Local gate rejections still
    // retry: the retry gap is for those. When the session is gone, neither a tap nor auto can
    // help, so the message doesn't offer them.
    const switchAutoOffAfterUploadFailure = (sessionGone: boolean): void => {
      if (!autoCaptureToggle.checked) return;
      autoCaptureToggle.checked = false;
      const suffix = sessionGone
        ? "Auto-capture is off now."
        : "Auto-capture is off now — tap to retry, or turn it back on.";
      captureStatus.textContent = `${captureStatus.textContent} ${suffix}`;
      gyroOverlayHandle.refresh();
    };
    // Layers of defense beyond the disabled attribute itself (fails loudly rather than silently
    // misbehaving if this handler is ever reached in a state the button shouldn't allow):
    if (sessionFinishing) {
      say("Session is finishing -- cannot capture now.", "error");
      return;
    }
    if (state.latestGyro === null) {
      say("No gyro reading yet -- try again in a moment.", "error");
      return;
    }
    if (state.sessionId === null) {
      say("Start a session first.", "error");
      return;
    }
    const sessionId = state.sessionId;

    let captured: CapturedFrame;
    try {
      captured = captureFrame(cameraPreview, state.latestGyro);
    } catch (error) {
      say(error instanceof Error ? error.message : String(error), "error");
      return;
    }
    state.lastCapturedFrame = captured;
    const c25PreJpegVariance =
      options.blurDebug === true ? measureBlurVariance(captured.pixels) : null;

    // C-20: exposure runs before blur so a uniformly black/white frame reports the lighting
    // problem instead of being mislabelled as blur. Duplicate stays last; first failure wins and
    // nothing is uploaded on a gate rejection. The code is
    // shown alongside the message, not just the message alone -- checkDuplicate's own message
    // ("Gyro delta ... are both below their thresholds") never actually says "duplicate," so the
    // code is what makes the specific reason identifiable (this story's own acceptance criteria
    // name all three codes explicitly). Wrapped in a try -- checkDuplicate throws, not returns a
    // rejection, on a dimension mismatch (frame_gate.ts's own frozen behavior), and nothing here
    // caught that (TabeenRaoof's PR #139 review).
    try {
      const exposureResult = checkExposure(captured.pixels);
      if (!exposureResult.ok) {
        say(`Frame rejected (${exposureResult.code}) — ${exposureResult.message}`, "error");
        return;
      }
      const blurResult = checkBlur(captured.pixels);
      if (!blurResult.ok) {
        say(`Frame rejected (${blurResult.code}) — ${blurResult.message}`, "error");
        return;
      }
      // A real resolution change (e.g. the phone rotating -- iOS swaps videoWidth/videoHeight,
      // which the cached-canvas rework in camera.ts already handles) is genuine motion, not a
      // frozen duplicate -- checkDuplicate itself assumes resolution never changes mid-session
      // and throws rather than compares across sizes. Treating it as "not a duplicate" here (and
      // re-anchoring below) is what keeps a rotation from permanently bricking capture for the
      // rest of the session against a now-stale-sized anchor (TabeenRaoof's PR #139 review).
      const dimensionsChanged =
        lastAcceptedForGate !== null &&
        (lastAcceptedForGate.frame.width !== captured.pixels.width ||
          lastAcceptedForGate.frame.height !== captured.pixels.height);
      const duplicateResult = dimensionsChanged
        ? { ok: true as const }
        : checkDuplicate(captured.pixels, captured.gyro, lastAcceptedForGate);
      if (!duplicateResult.ok) {
        say(`Frame rejected (${duplicateResult.code}) — ${duplicateResult.message}`, "error");
        return;
      }
    } catch (error) {
      say(error instanceof Error ? error.message : String(error), "error");
      return;
    }

    // C-27: a brief flash confirms a shot was taken, which matters most when nobody tapped.
    flashViewfinder();

    // Accepted by the local gate -- anchors the next duplicate check, independent of whatever the
    // server goes on to say about this particular upload. Rolled back below if the upload itself
    // doesn't succeed (network error or a 422): otherwise a retry from the exact same spot was
    // rejected as a duplicate of an upload the server never actually accepted (TabeenRaoof's PR
    // #139 review).
    const previousAcceptedForGate = lastAcceptedForGate;
    lastAcceptedForGate = { frame: captured.pixels, gyro: captured.gyro };

    uploadInFlight = true;
    updateCaptureAvailability();
    updateFinishAvailability();
    say("Uploading…", "info");
    void (async () => {
      try {
        const jpeg = await encodeFrameAsJpeg(captured.pixels);
        const result = await uploadFrame(
          options.baseUrl,
          sessionId,
          jpeg,
          captured.gyro,
          options.authToken,
        );
        say(describeUploadFrame(result), result.ok ? "info" : "error");
        if (result.ok) {
          if (c25PreJpegVariance !== null) {
            const storedCount = appendC25BlurDebugMeasurement(blurDebugStorage, {
              session_id: sessionId,
              frame_index: result.body.frame_index,
              pre_jpeg_variance: c25PreJpegVariance,
              width: captured.pixels.width,
              height: captured.pixels.height,
            });
            if (storedCount === null) {
              blurDebugStorageFailures += 1;
              setStatus(
                blurDebugStatus,
                `Blur debug warning: frame ${String(result.body.frame_index)} was accepted, but its pre-JPEG measurement could not be stored; ${String(blurDebugStorageFailures)} failure(s), paired evidence is incomplete.`,
                "error",
              );
            } else if (blurDebugStorageFailures > 0) {
              setStatus(
                blurDebugStatus,
                `Blur debug: ${String(storedCount)} measurement(s) stored for this session, ${String(blurDebugStorageFailures)} failed; paired evidence is incomplete.`,
                "error",
              );
            } else {
              setStatus(
                blurDebugStatus,
                `Blur debug: ${String(storedCount)} measurement(s) stored for this session; latest frame index ${String(result.body.frame_index)}.`,
                "info",
              );
            }
          }
          // Only a real server acceptance marks orbit coverage -- a 422 shows the server's
          // reason but must not advance the arrow.
          gyroOverlayHandle.reportAcceptedFrame(captured.gyro);
          // C-24: saved after every accepted frame, so a reload at any point loses nothing the
          // server has. The count comes from the server's own frame_index.
          if (activeSession !== null && activeSession.sessionId === sessionId) {
            activeSession.acceptedFrameCount = result.body.frame_index + 1;
            activeSession.acceptedGyro.push(captured.gyro);
            persistActiveSession();
          }
          // A-11: the newest VLM tip, if a check has finished. It stays until a newer one
          // arrives (a later 202 without feedback leaves it alone) and never blocks capture.
          // setStatus writes textContent: the hint is model output, never HTML.
          const feedback = result.body.vlm_feedback;
          if (feedback !== undefined) {
            setStatus(vlmHint, describeVlmFeedback(feedback), "info");
          }
          // Guidance is optional and must never hold the capture/finish lock. Run it separately;
          // getAgentMessage also bounds the request so a stalled route cannot linger forever.
          void refreshAgentConversation(sessionId);
        } else {
          lastAcceptedForGate = previousAcceptedForGate;
          // C-24: a continued session can turn out to be over -- finished from this tab before
          // the reload, or gone because the service lost its data. Retrying can't help, so drop
          // the saved copy and let Start make a new session.
          const code = result.error.code;
          if (code === "session_not_found" || code === "session_already_finished") {
            forgetSession();
            state.sessionId = null;
            startButton.disabled = false;
            say(
              `${describeUploadFrame(result)} This session can't take more frames — start a new session.`,
              "error",
            );
            switchAutoOffAfterUploadFailure(true);
          } else {
            switchAutoOffAfterUploadFailure(false);
          }
        }
      } catch (error) {
        lastAcceptedForGate = previousAcceptedForGate;
        say(`Could not upload frame — ${describeUploadError(error)}`, "error");
        switchAutoOffAfterUploadFailure(false);
      } finally {
        uploadInFlight = false;
        updateCaptureAvailability();
        updateFinishAvailability();
      }
    })();
  }

  captureButton.addEventListener("click", () => {
    // #420 review (@dalwalyk): a tap restarts auto-capture's hold and its retry gap. Otherwise a
    // tap the gate rejects could be followed by an automatic photo on the very next reading,
    // overwriting the person's own rejection message.
    autoCaptureState = { holdStartMs: null, holdStartGyro: null, lastAttemptMs: now() };
    attemptCapture("manual");
  });

  // #420 review (@dalwalyk): the on-target line depends on the switch ("hold steady" or "take the
  // photo"), so a change re-renders it at once instead of on the phone's next movement.
  autoCaptureToggle.addEventListener("change", () => {
    gyroOverlayHandle.refresh();
  });

  async function runReconstructionPolling(): Promise<void> {
    if (state.sessionId === null) {
      return;
    }
    const sessionId = state.sessionId;
    reportLink.hidden = true;
    reportLink.removeAttribute("href");
    retryReconstructionButton.hidden = true;
    remountModelViewer({ status: "pending" });
    const outcome = await pollReconstruction(
      options.baseUrl,
      sessionId,
      reconstructionStatus,
      options.reconstructionMaxAttempts ?? DEFAULT_RECONSTRUCTION_MAX_ATTEMPTS,
      options.reconstructionPollIntervalMs ?? DEFAULT_RECONSTRUCTION_POLL_INTERVAL_MS,
      (terminalResult) => {
        remountModelViewer(terminalResult);
      },
      options.authToken,
    );
    retryReconstructionButton.hidden = outcome !== "timeout" && outcome !== "error";
    // C-24: once a terminal result is on screen, there's nothing left to come back to. A timeout
    // or fetch error keeps the saved copy, since Retry (or a reload) still needs it.
    if (
      (outcome === "complete" || outcome === "failed") &&
      activeSession !== null &&
      activeSession.sessionId === sessionId
    ) {
      forgetSession();
    }
    if (outcome === "complete") {
      reportLink.href = `report.html?session=${encodeURIComponent(sessionId)}`;
      reportLink.hidden = false;
    }
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
      // Defense in depth beyond the disabled attribute itself, same pattern as the capture
      // handler's own guards (TabeenRaoof's PR #139 review) -- an upload still in flight must
      // not race finishSession.
      if (uploadInFlight) {
        setStatus(
          finishStatus,
          "An upload is still in progress -- try again in a moment.",
          "error",
        );
        return;
      }

      // Set before finishSession is even awaited -- the window this closes is exactly the one
      // between this click and that call resolving, during which Capture must not be startable
      // (TabeenRaoof's second-pass review, the reverse of the uploadInFlight race above).
      sessionFinishing = true;
      updateCaptureAvailability();
      finishButton.disabled = true;
      setStatus(reconstructionStatus, "", "info");
      let finished = false;
      try {
        const result = await finishSession(options.baseUrl, state.sessionId, options.authToken);
        setStatus(finishStatus, describeFinishSession(result), result.ok ? "info" : "error");
        finished = result.ok;
      } catch (error) {
        setStatus(
          finishStatus,
          `Could not finish session — ${describeNetworkError(error)}`,
          "error",
        );
      }
      if (!finished) {
        // Genuinely retryable -- finishSession itself failed, so this session is not finishing
        // after all. Left `true` only on the success path below.
        sessionFinishing = false;
        updateFinishAvailability();
        updateCaptureAvailability();
        return;
      }
      // C-24: kept until a result has been shown, so a reload while reconstructing offers
      // "Show result" instead of losing the session.
      if (activeSession !== null) {
        activeSession.phase = "finished";
        persistActiveSession();
      }
      // Everything from here on runs only once finishSession has genuinely resolved ok:true, and
      // outside the try above -- a failure past this point (e.g. the reconstruction poll, or the
      // viewer it rebuilds) is not a "could not finish session" error, and mislabeling it as one
      // also wrongly re-enabled Finish on an already-finished session (TabeenRaoof's PR #137
      // review, item 9). Still inside its own try (TabeenRaoof's PR #139 review) -- moving this
      // out of the first try fixed the mislabeling, but left failures here genuinely silent (a
      // fire-and-forget void (async () => {...})() has nothing else to report them to).
      try {
        releaseCamera();
        // The arrow's guidance is stale the moment the camera is off -- dispose rather than leave
        // it saying "move to the next position" with nothing left to move to (TabeenRaoof's PR
        // #139 review).
        gyroOverlayHandle.dispose();
        await runReconstructionPolling();
      } catch (error) {
        setStatus(
          reconstructionStatus,
          `Could not load the reconstruction — ${describeNetworkError(error)}`,
          "error",
        );
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
  authToken?: string,
): Promise<"complete" | "failed" | "timeout" | "error"> {
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    let result;
    try {
      result = await getReconstruction(baseUrl, sessionId, authToken);
    } catch (error) {
      setStatus(target, `Could not fetch reconstruction — ${describeNetworkError(error)}`, "error");
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
