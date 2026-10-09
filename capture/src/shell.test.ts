// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountAppShell, type AppState } from "./shell.js";
import { startContractStub } from "../stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "../stub/page-test-port.js";

// Same fixed-port reasoning as page.test.ts: happy-dom enforces same-origin, matching
// vite.config.ts's environmentOptions.happyDOM.url.

describe("mountAppShell", () => {
  let close: () => Promise<void>;
  let baseUrl: string;

  beforeEach(async () => {
    const started = await startContractStub(PAGE_TEST_STUB_PORT);
    baseUrl = started.baseUrl;
    close = started.close;
  });

  afterEach(async () => {
    await close();
    // @ts-expect-error -- test-only cleanup of a property this file itself may have set.
    delete navigator.mediaDevices;
    vi.unstubAllGlobals();
  });

  function mount(): {
    state: AppState;
    orderCode: HTMLInputElement;
    button: HTMLButtonElement;
    sessionStatus: HTMLElement;
    cameraStatus: HTMLElement;
    gyroStatus: HTMLElement;
  } {
    document.body.innerHTML = '<main id="shell-app"></main>';
    const root = document.getElementById("shell-app");
    if (!root) throw new Error("test setup: #shell-app missing");
    const state = mountAppShell(root, { baseUrl });
    const orderCode = document.getElementById("shell-order-code");
    const button = document.getElementById("shell-start-button");
    const sessionStatus = document.getElementById("shell-session-status");
    const cameraStatus = document.getElementById("shell-camera-status");
    const gyroStatus = document.getElementById("shell-gyro-status");
    if (
      !(orderCode instanceof HTMLInputElement) ||
      !(button instanceof HTMLButtonElement) ||
      !sessionStatus ||
      !cameraStatus ||
      !gyroStatus
    ) {
      throw new Error("test setup: mounted elements missing");
    }
    return { state, orderCode, button, sessionStatus, cameraStatus, gyroStatus };
  }

  function stubGrantedCamera(): void {
    // happy-dom's own `MediaStream` class doesn't implement `getTracks()` -- a real browser's
    // does, which is what requestCameraPermission's stop-the-tracks-immediately logic calls. A
    // plain object with a working getTracks() stands in for it here, rather than the incomplete
    // real class, the same class of happy-dom gap this project has run into before.
    const fakeStream = {
      getTracks: () => [{ stop: () => undefined }],
    };
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };
  }

  function stubDeniedCamera(): void {
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => Promise.reject(new DOMException("", "NotAllowedError")),
    };
  }

  function stubGyroPermission(state: "granted" | "denied"): void {
    vi.stubGlobal("DeviceOrientationEvent", {
      requestPermission: () => Promise.resolve(state),
    });
  }

  function dispatchOrientation(
    alpha: number | null,
    beta: number | null,
    gamma: number | null,
  ): void {
    const event = new Event("deviceorientation") as DeviceOrientationEvent;
    (event as unknown as { alpha: number | null }).alpha = alpha;
    (event as unknown as { beta: number | null }).beta = beta;
    (event as unknown as { gamma: number | null }).gamma = gamma;
    window.dispatchEvent(event);
  }

  it("requests camera and gyro permission synchronously, inside the click handler -- the regression this story exists to prevent", async () => {
    let cameraCalledSynchronously = false;
    let gyroCalledSynchronously = false;
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => {
        cameraCalledSynchronously = true;
        return new Promise(() => {
          /* never resolves -- only call timing matters for this test */
        });
      },
    };
    vi.stubGlobal("DeviceOrientationEvent", {
      requestPermission: () => {
        gyroCalledSynchronously = true;
        return new Promise(() => {
          /* never resolves -- only call timing matters for this test */
        });
      },
    });
    const { orderCode, button, sessionStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click(); // no `await` here either -- proves neither call is deferred past this tick

    expect(cameraCalledSynchronously).toBe(true);
    expect(gyroCalledSynchronously).toBe(true);
    // The click handler also starts a real createSession fetch (camera/gyro are stubbed to
    // never resolve, but the session request is real against this test's own stub). Waiting for
    // it to finish here -- even though this test doesn't care about its result -- keeps it from
    // racing against afterEach's server teardown and leaking an unhandled rejection into a
    // later test.
    await vi.waitFor(() => {
      if (sessionStatus.textContent === "") throw new Error("still pending");
    });
  });

  it("on full success: creates the session, grants both permissions, and reports all three", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, sessionStatus, cameraStatus, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.sessionId).toMatch(/^sess_/);
    expect(state.cameraGranted).toBe(true);
    expect(state.gyroGranted).toBe(true);
    expect(sessionStatus.textContent).toMatch(/Session started/);
    expect(cameraStatus.textContent).toBe("Camera permission granted.");
    expect(gyroStatus.textContent).toBe("Gyro permission granted.");
  });

  it("on a platform with no gyro requestPermission, grants gyro directly with no prompt", async () => {
    stubGrantedCamera();
    // Deliberately not stubbing DeviceOrientationEvent -- happy-dom's default shape has no
    // requestPermission, the desktop/Android case.
    const { state, orderCode, button, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.gyroGranted).toBe(true);
  });

  it("records a camera denial without blocking the session or gyro from proceeding", async () => {
    stubDeniedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, cameraStatus, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.cameraGranted).toBe(false);
    expect(cameraStatus.textContent).toContain("permission denied");
    expect(state.sessionId).toMatch(/^sess_/);
    expect(state.gyroGranted).toBe(true);
  });

  it("records a gyro denial without blocking the session or camera from proceeding", async () => {
    stubGrantedCamera();
    stubGyroPermission("denied");
    const { state, orderCode, button, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.gyroGranted).toBe(false);
    expect(gyroStatus.textContent).toMatch(/denied/);
    expect(state.cameraGranted).toBe(true);
  });

  it("records a failed session creation, and re-enables the button, while permissions still proceed", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, sessionStatus, gyroStatus } = mount();
    orderCode.value = "PO-NOT-REAL"; // unknown to the stub -> 400 order_code_unknown

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.sessionId).toBeNull();
    expect(sessionStatus.textContent).toMatch(/Could not start session/);
    expect(state.cameraGranted).toBe(true);
    expect(state.gyroGranted).toBe(true);
    expect(button.disabled).toBe(false);
  });

  it("recovers from createSession actually rejecting (not just resolving ok:false), still reflects camera/gyro, and re-enables the button", async () => {
    // TabeenRaoof's PR #102 review, reproduced here as a permanent regression test: a real
    // network failure makes createSession's fetch reject, not resolve with an error body.
    // Before the fix, nothing caught it, so the async IIFE aborted at that `await` -- the
    // button stayed disabled forever and the camera/gyro results (whose real prompts had
    // already fired) were never reflected in state or the DOM.
    stubGrantedCamera();
    stubGyroPermission("granted");
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))),
    );
    const { state, orderCode, button, sessionStatus, cameraStatus, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.sessionId).toBeNull();
    expect(sessionStatus.textContent).toMatch(/Could not start session/);
    expect(state.cameraGranted).toBe(true);
    expect(cameraStatus.textContent).toBe("Camera permission granted.");
    expect(state.gyroGranted).toBe(true);
    expect(button.disabled).toBe(false);
  });

  it("disables the button on click and leaves it disabled once a session is successfully created", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { orderCode, button, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    expect(button.disabled).toBe(true);
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(button.disabled).toBe(true);
  });

  it("updates latestGyro from a dispatched orientation event once gyro is granted", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });
    dispatchOrientation(12.3, -45.6, 78.9);

    expect(state.latestGyro).toEqual({ alpha: 12.3, beta: -45.6, gamma: 78.9 });
  });

  it("does not overwrite latestGyro with a reading that has a null component", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });
    dispatchOrientation(1, 2, 3);
    dispatchOrientation(null, 9, 9);

    expect(state.latestGyro).toEqual({ alpha: 1, beta: 2, gamma: 3 });
  });
});
