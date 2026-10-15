// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { uploadFrame } from "./api.js";
import { mountAppShell, type AppState } from "./shell.js";
import { startContractStub, type ContractStub } from "../stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "../stub/page-test-port.js";

// Same fixed-port reasoning as page.test.ts: happy-dom enforces same-origin, matching
// vite.config.ts's environmentOptions.happyDOM.url.

describe("mountAppShell", () => {
  let close: () => Promise<void>;
  let baseUrl: string;
  let stub: ContractStub;

  beforeEach(async () => {
    const started = await startContractStub(PAGE_TEST_STUB_PORT);
    baseUrl = started.baseUrl;
    close = started.close;
    stub = started.stub;
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
    cameraPreview: HTMLVideoElement;
    captureButton: HTMLButtonElement;
    captureStatus: HTMLElement;
    finishButton: HTMLButtonElement;
    finishStatus: HTMLElement;
    reconstructionStatus: HTMLElement;
    retryReconstructionButton: HTMLButtonElement;
  } {
    document.body.innerHTML = '<main id="shell-app"></main>';
    const root = document.getElementById("shell-app");
    if (!root) throw new Error("test setup: #shell-app missing");
    const state = mountAppShell(root, {
      baseUrl,
      reconstructionPollIntervalMs: 0,
      reconstructionMaxAttempts: 3,
      modelViewerOptions: { loaderFactory: () => ({ load: () => undefined }) },
    });
    const orderCode = document.getElementById("shell-order-code");
    const button = document.getElementById("shell-start-button");
    const sessionStatus = document.getElementById("shell-session-status");
    const cameraStatus = document.getElementById("shell-camera-status");
    const gyroStatus = document.getElementById("shell-gyro-status");
    const cameraPreview = document.getElementById("shell-camera-preview");
    const captureButton = document.getElementById("shell-capture-button");
    const captureStatus = document.getElementById("shell-capture-status");
    const finishButton = document.getElementById("shell-finish-button");
    const finishStatus = document.getElementById("shell-finish-status");
    const reconstructionStatus = document.getElementById("shell-reconstruction-status");
    const retryReconstructionButton = document.getElementById("shell-retry-reconstruction-button");
    if (
      !(orderCode instanceof HTMLInputElement) ||
      !(button instanceof HTMLButtonElement) ||
      !(cameraPreview instanceof HTMLVideoElement) ||
      !(captureButton instanceof HTMLButtonElement) ||
      !(finishButton instanceof HTMLButtonElement) ||
      !(retryReconstructionButton instanceof HTMLButtonElement) ||
      !sessionStatus ||
      !cameraStatus ||
      !gyroStatus ||
      !captureStatus ||
      !finishStatus ||
      !reconstructionStatus
    ) {
      throw new Error("test setup: mounted elements missing");
    }
    return {
      state,
      orderCode,
      button,
      sessionStatus,
      cameraStatus,
      gyroStatus,
      cameraPreview,
      captureButton,
      captureStatus,
      finishButton,
      finishStatus,
      reconstructionStatus,
      retryReconstructionButton,
    };
  }

  function stubGrantedCamera(): { trackStopSpy: ReturnType<typeof vi.fn> } {
    // C-05 actually assigns the granted stream to `srcObject` now, and happy-dom's own
    // `srcObject` setter (unlike its incomplete `MediaStream.getTracks()`) does validate the
    // value is a real `MediaStream` instance -- so the fake has to be a real one, with
    // `getTracks` overridden on the instance to make track-stop calls observable.
    const trackStopSpy = vi.fn();
    const fakeStream = new MediaStream();
    fakeStream.getTracks = () => [{ stop: trackStopSpy } as unknown as MediaStreamTrack];
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };
    return { trackStopSpy };
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

  async function startSuccessfulSession(): Promise<
    ReturnType<typeof mount> & { trackStopSpy: ReturnType<typeof vi.fn> }
  > {
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount();
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    return { ...mounted, trackStopSpy };
  }

  // happy-dom's canvas has no real 2D rendering, so capturePixelsFromVideo's drawImage/getImageData
  // calls need a stub -- this is what "a fake video element" in C-05's own Tests section maps to
  // on the <canvas> side. Returns the drawImage spy and the pixel data captureFrame will read back.
  function stubCanvasContext(): {
    drawImageSpy: ReturnType<typeof vi.fn>;
    imageData: ImageData;
  } {
    const drawImageSpy = vi.fn();
    const imageData = {
      data: new Uint8ClampedArray([1, 2, 3, 4]),
      width: 1,
      height: 1,
    } as ImageData;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: drawImageSpy,
      getImageData: () => imageData,
    } as unknown as CanvasRenderingContext2D);
    return { drawImageSpy, imageData };
  }

  function setVideoDimensions(video: HTMLVideoElement, width: number, height: number): void {
    // happy-dom's own videoWidth/videoHeight are always 0 (no real decode pipeline) --
    // overridden here the same way a real playing video would report its frame size.
    Object.defineProperty(video, "videoWidth", { value: width, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: height, configurable: true });
  }

  async function uploadAcceptedFrame(sessionId: string): Promise<void> {
    const frame = new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" });
    const result = await uploadFrame(baseUrl, sessionId, frame, {
      alpha: 1,
      beta: 2,
      gamma: 3,
    });
    if (!result.ok) throw new Error("test setup: uploadFrame failed");
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
      if (gyroStatus.textContent === "" || cameraStatus.textContent === "") {
        throw new Error("still pending");
      }
    });

    expect(state.sessionId).toMatch(/^sess_/);
    expect(state.cameraGranted).toBe(true);
    expect(state.gyroGranted).toBe(true);
    expect(sessionStatus.textContent).toMatch(/Session started/);
    // C-05: the camera doesn't just report permission granted anymore -- it keeps the stream
    // running and confirms the preview actually started playing, same wording C-02's own
    // mountCameraCheck already uses for this.
    expect(cameraStatus.textContent).toBe("Camera started.");
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

  it("styles a camera denial as a visible error, not plain unstyled text", async () => {
    stubDeniedCamera();
    stubGyroPermission("granted");
    const { orderCode, button, cameraStatus, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(cameraStatus.style.color).not.toBe("");
    expect(cameraStatus.style.fontWeight).toBe("bold");
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
    expect(gyroStatus.style.color).not.toBe("");
    expect(gyroStatus.style.fontWeight).toBe("bold");
    expect(state.cameraGranted).toBe(true);
  });

  it("does not style a success status as an error", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { orderCode, button, cameraStatus, gyroStatus } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(cameraStatus.style.fontWeight).not.toBe("bold");
    expect(gyroStatus.style.fontWeight).not.toBe("bold");
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
      if (gyroStatus.textContent === "" || cameraStatus.textContent === "") {
        throw new Error("still pending");
      }
    });

    expect(state.sessionId).toBeNull();
    expect(sessionStatus.textContent).toMatch(/Could not start session/);
    expect(state.cameraGranted).toBe(true);
    expect(cameraStatus.textContent).toBe("Camera started.");
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

  it("finishes from the live shell and renders pending then complete reconstruction", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.setReconstructionPendingPolls(2);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Reconstruction complete")) {
        throw new Error("reconstruction not complete yet");
      }
    });

    expect(mounted.finishStatus.textContent).toMatch(/^Session finished: sess_/);
    expect(mounted.reconstructionStatus.textContent).toContain("Reference tier: charuco_board");
    expect(mounted.reconstructionStatus.textContent).toContain("Metric: yes");
    expect(mounted.reconstructionStatus.textContent).toContain("Observed fraction: 0.910");
  });

  it("states that a non-metric result makes no dimensional claims", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.useNoReferenceReconstruction();

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Reconstruction complete")) {
        throw new Error("reconstruction not complete yet");
      }
    });

    expect(mounted.reconstructionStatus.textContent).toContain("Reference tier: none");
    expect(mounted.reconstructionStatus.textContent).toContain("Metric: no");
    expect(mounted.reconstructionStatus.textContent).toContain("No dimensional claims");
  });

  it("shows failed reconstruction instead of staying pending", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.failReconstruction();

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Reconstruction failed")) {
        throw new Error("failure not rendered yet");
      }
    });

    expect(mounted.reconstructionStatus.textContent).toContain("Stub reconstruction failure");
    expect(mounted.reconstructionStatus.style.fontWeight).toBe("bold");
  });

  it("times out after the bounded polling budget", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.setReconstructionPendingPolls(99);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("timed out after 3 attempts")) {
        throw new Error("timeout not rendered yet");
      }
    });

    expect(mounted.reconstructionStatus.textContent).toContain("3 attempts");
    expect(mounted.reconstructionStatus.style.fontWeight).toBe("bold");
  });

  it("offers retry after timeout and resumes polling without finishing the session again", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.setReconstructionPendingPolls(3);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("timed out after 3 attempts")) {
        throw new Error("timeout not rendered yet");
      }
    });
    expect(mounted.retryReconstructionButton.hidden).toBe(false);

    mounted.retryReconstructionButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Reconstruction complete")) {
        throw new Error("reconstruction did not complete after retry");
      }
    });
    expect(mounted.retryReconstructionButton.hidden).toBe(true);
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

  it("keeps the camera stream running on grant -- not stopped immediately like C-02 -- and enables the capture button", async () => {
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, gyroStatus, captureButton } = mount();
    orderCode.value = "PO-48213-A";

    expect(captureButton.disabled).toBe(true);

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(state.cameraStream).not.toBeNull();
    expect(trackStopSpy).not.toHaveBeenCalled();
    await vi.waitFor(() => {
      if (captureButton.disabled) throw new Error("still disabled");
    });
  });

  it("captures a frame paired with the live gyro reading, without restarting the stream", async () => {
    const mounted = await startSuccessfulSession();
    const { imageData } = stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(12.3, -45.6, 78.9);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    mounted.captureButton.click();

    expect(mounted.state.lastCapturedFrame).toEqual({
      pixels: imageData,
      gyro: { alpha: 12.3, beta: -45.6, gamma: 78.9 },
    });
    expect(mounted.captureStatus.textContent).toMatch(/captured/i);
  });

  it("shows an error and does not capture when there is no gyro reading yet", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    mounted.captureButton.click();

    expect(mounted.state.lastCapturedFrame).toBeNull();
    expect(mounted.captureStatus.textContent).toMatch(/no gyro reading/i);
  });

  it("stops every camera track and disables capture once the session actually finishes", async () => {
    const mounted = await startSuccessfulSession();
    await uploadAcceptedFrame(mounted.state.sessionId as string);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (mounted.finishStatus.textContent === "") throw new Error("still pending");
    });

    expect(mounted.trackStopSpy).toHaveBeenCalledTimes(1);
    expect(mounted.state.cameraStream).toBeNull();
    expect(mounted.captureButton.disabled).toBe(true);
  });

  it("does not touch the camera stream when finishSession resolves not-ok", async () => {
    const mounted = await startSuccessfulSession();
    // No frame uploaded -- the stub's own finishSession rejects with 409 "too few frames" when
    // acceptedFrameCount is 0, giving a real not-ok response without a separate stub method.

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (mounted.finishStatus.textContent === "") throw new Error("still pending");
    });

    expect(mounted.finishStatus.style.fontWeight).toBe("bold");
    expect(mounted.trackStopSpy).not.toHaveBeenCalled();
    expect(mounted.state.cameraStream).not.toBeNull();
  });
});
