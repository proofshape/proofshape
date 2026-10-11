// C-04: replaces C-01/C-02/C-03's three isolated demo sections (#app/#camera-app/#gyro-app) with
// one real flow — create a session, request both permissions together — and gives C-05 onward
// one shared place to read/write app state instead of per-module globals.

import { requestCameraPermission, setStatus } from "./camera.js";
import { createSession } from "./api.js";
import { requireElement, requireTyped } from "./dom.js";
import { requestGyroPermission } from "./gyro.js";
import { describeCreateSession } from "./messages.js";
import type { Gyro } from "./types.js";

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
}

export const APP_SHELL_HTML = `
  <section>
    <h2>Start capture session (C-04)</h2>
    <label>Order code <input id="shell-order-code" type="text" /></label>
    <button id="shell-start-button" type="button">Start</button>
    <p id="shell-session-status"></p>
    <p id="shell-camera-status"></p>
    <p id="shell-gyro-status"></p>
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
          sessionOk = true;
        }
      } catch (error) {
        setStatus(
          sessionStatus,
          `Could not start session — ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
      }

      // Plain textContent here was the exact bug TabeenRaoof's PR #117 review caught: camera.ts's
      // own mountCameraCheck already learned (from real on-device testing) that unstyled text is
      // easy to miss as an error -- setStatus carries that same fix into this shell.
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

  return state;
}
