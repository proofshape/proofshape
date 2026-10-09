import { requireElement, requireTyped } from "./dom.js";
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

// iOS 13+ Safari gates DeviceOrientationEvent behind an explicit, gesture-triggered permission
// request — a non-standard extension TypeScript's own DOM lib doesn't know about. Desktop,
// Android and pre-13 iOS have no such method at all, and read orientation events directly.
interface IOSDeviceOrientationEventConstructor {
  requestPermission?: () => Promise<"granted" | "denied">;
}

function getIOSDeviceOrientationEventConstructor():
  IOSDeviceOrientationEventConstructor | undefined {
  if (typeof DeviceOrientationEvent === "undefined") {
    return undefined;
  }
  // Cast, not `any`: the real constructor may or may not carry this iOS-only static method.
  return DeviceOrientationEvent as unknown as IOSDeviceOrientationEventConstructor;
}

// The one piece of this story's logic that's unit-testable without real hardware: whether this
// platform needs an explicit permission request at all, or should skip straight to listening.
export function isOrientationPermissionRequestNeeded(): boolean {
  return typeof getIOSDeviceOrientationEventConstructor()?.requestPermission === "function";
}

// For a genuinely unexpected rejection from requestPermission() — NOT for a user denial, which
// the API reports as a *resolved* value ("denied"), not a rejection.
export function describeOrientationError(error: unknown): string {
  if (error instanceof Error) {
    return `Could not request gyro permission — ${error.message}`;
  }
  return "Could not request gyro permission — unexpected error.";
}

export type GyroPermissionResult = { granted: true } | { granted: false; message: string };

// C-04: requests permission and reports granted/denied, reusing the exact same logic
// mountGyroCheck's own click handler already proves out -- not a new implementation. The
// `requestPermission()` call is the first statement in this function's body (no `await` before
// it), so calling this with no leading `await` from a click handler preserves the gesture the
// same way mountGyroCheck's `.then()`-based handler does; capture/src/shell.test.ts proves this
// in CI rather than relying on the reasoning alone.
export async function requestGyroPermission(): Promise<GyroPermissionResult> {
  const ctor = getIOSDeviceOrientationEventConstructor();
  if (typeof ctor?.requestPermission !== "function") {
    return { granted: true };
  }
  try {
    const permissionState = await ctor.requestPermission();
    if (permissionState === "granted") {
      return { granted: true };
    }
    return {
      granted: false,
      message: "Gyro permission denied. Reload the page and tap Start to try again.",
    };
  } catch (error) {
    return { granted: false, message: describeOrientationError(error) };
  }
}

export const GYRO_CHECK_HTML = `
  <section>
    <h2>Gyro check (C-03)</h2>
    <button id="gyro-button" type="button">Enable gyro</button>
    <p id="gyro-status"></p>
    <dl>
      <dt>alpha</dt>
      <dd id="gyro-alpha-value">—</dd>
      <dt>beta</dt>
      <dd id="gyro-beta-value">—</dd>
      <dt>gamma</dt>
      <dd id="gyro-gamma-value">—</dd>
    </dl>
  </section>
`;

function formatOrientationComponent(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

// Deliberately not mounted into page.ts's existing markup — same reasoning C-02's camera.ts
// used: this proves the permission mechanism in isolation without touching already-shipped,
// already-reviewed code for an unrelated concern.
export function mountGyroCheck(root: HTMLElement): void {
  root.innerHTML = GYRO_CHECK_HTML;

  const button = requireTyped(root, "#gyro-button", HTMLButtonElement, "<button>");
  const status = requireElement(root, "#gyro-status");
  const alphaValue = requireElement(root, "#gyro-alpha-value");
  const betaValue = requireElement(root, "#gyro-beta-value");
  const gammaValue = requireElement(root, "#gyro-gamma-value");

  // Scoped to this mount, not module-level — a module-level flag would leak across repeated
  // mounts (e.g. in tests), the same way a stray module-level camera stream would in camera.ts.
  let listening = false;

  function startListening(): void {
    if (listening) {
      return;
    }
    listening = true;
    window.addEventListener("deviceorientation", (event) => {
      alphaValue.textContent = formatOrientationComponent(event.alpha);
      betaValue.textContent = formatOrientationComponent(event.beta);
      gammaValue.textContent = formatOrientationComponent(event.gamma);
    });
    status.textContent = "Reading live gyro values.";
  }

  button.addEventListener("click", () => {
    status.textContent = "";

    const ctor = getIOSDeviceOrientationEventConstructor();
    if (typeof ctor?.requestPermission !== "function") {
      startListening();
      return;
    }

    // Called synchronously, directly inside the tap handler, with no `await` before it — iOS
    // Safari treats the user gesture as expired if this call is deferred even one microtask.
    button.disabled = true;
    void ctor.requestPermission().then(
      (permissionState) => {
        button.disabled = false;
        if (permissionState === "granted") {
          startListening();
        } else {
          status.textContent =
            "Gyro permission denied. Reload the page and tap Enable gyro to try again.";
        }
      },
      (error: unknown) => {
        button.disabled = false;
        status.textContent = describeOrientationError(error);
      },
    );
  });
}
