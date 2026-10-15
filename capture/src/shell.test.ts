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
    // C-09: vi.spyOn calls (canvas context/toBlob, fetch) were never restored between tests here
    // -- harmless while nothing spied on a shared prototype method with per-test return values,
    // but stubCanvasContext's HTMLCanvasElement.prototype.getContext/toBlob spies do exactly
    // that, and an unrestored one from an earlier test silently won the next test's calls.
    vi.restoreAllMocks();
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
  // A real checkerboard, not a 1x1 flat fixture -- C-09 actually runs the captured pixels through
  // C-06's gate now, and a 1x1 image has no interior pixels at all (checkBlur's variance is 0,
  // always below the threshold). 60/200 (not 0/255) keeps every pixel away from checkExposure's
  // clip bounds, while every interior pixel's four neighbors are the opposite checker value, so
  // the Laplacian response is a constant +-560 -- comfortably above DEFAULT_BLUR_VARIANCE_THRESHOLD.
  function sharpWellExposedImageData(width = 4, height = 4): ImageData {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = (x + y) % 2 === 0 ? 60 : 200;
        const offset = (y * width + x) * 4;
        data[offset] = value;
        data[offset + 1] = value;
        data[offset + 2] = value;
        data[offset + 3] = 255;
      }
    }
    return { data, width, height } as ImageData;
  }

  // Stubs both capturePixelsFromVideo's canvas (drawImage/getImageData) and encodeFrameAsJpeg's
  // (putImageData/toBlob) -- a capture that clears the gate now goes on to actually encode and
  // upload, so both need to be in place, not just the capture half C-05's own tests needed.
  // toBlob hands back real JPEG-magic-byte bytes, since the real contract stub (not a mock)
  // checks for them before accepting an upload.
  function stubCanvasContext(imageData: ImageData = sharpWellExposedImageData()): {
    drawImageSpy: ReturnType<typeof vi.fn>;
    imageData: ImageData;
    // capturePixelsFromVideo caches one canvas+context per video element (C-09's own memory-
    // pressure fix), so getContext is only ever called once per video -- a second
    // stubCanvasContext() call wouldn't be consulted again. This mutates what the *already
    // cached* context's getImageData returns, so a test can still simulate a later capture
    // seeing different pixels (e.g. a resolution change) without fighting that cache.
    setImageData: (next: ImageData) => void;
  } {
    const drawImageSpy = vi.fn();
    let currentImageData = imageData;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: drawImageSpy,
      getImageData: () => currentImageData,
      putImageData: () => undefined,
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (
      callback: BlobCallback,
    ) {
      callback(new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" }));
    });
    return {
      drawImageSpy,
      imageData,
      setImageData: (next: ImageData) => {
        currentImageData = next;
      },
    };
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
    // C-09: cameraGranted means "there's a usable camera right now," not just "permission was
    // granted at some point" -- consistent with releaseCamera()'s own semantics. No session means
    // the just-granted stream is stopped immediately, so this is false, not true.
    expect(state.cameraGranted).toBe(false);
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
    // C-09: no session means nothing to capture into -- the just-granted stream is stopped
    // immediately rather than left attached to the preview (TabeenRaoof's PR #137 review, item 2),
    // and cameraGranted goes back to false, consistent with every other "stopped without ever
    // being usable" path (TabeenRaoof's PR #139 review).
    expect(state.cameraGranted).toBe(false);
    expect(cameraStatus.textContent).toMatch(/no session is active/);
    expect(state.cameraStream).toBeNull();
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
    // Capture also needs a real gyro reading now, not just a playing camera (TabeenRaoof's PR
    // #137 review, item 6) -- without this, the button would never enable.
    dispatchOrientation(1, 2, 3);
    await vi.waitFor(() => {
      if (captureButton.disabled) throw new Error("still disabled");
    });
  });

  it("captures a frame paired with the live gyro reading, gates it, and uploads it without restarting the stream", async () => {
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
    await vi.waitFor(() => {
      if (mounted.captureStatus.textContent === "Uploading…") throw new Error("still uploading");
    });
    expect(mounted.captureStatus.textContent).toMatch(/accepted/i);
  });

  it("captures stay available but never enable without a real gyro reading, even with the camera playing", async () => {
    // Before this fix, capture enabled on a playing camera alone -- with gyro denied (or no
    // sensor ever firing an event), tapping it could only ever show "No gyro reading yet," which
    // is misleading for a permanent state, not the transient one it's meant to describe
    // (TabeenRaoof's PR #137 review, item 6). The button itself now simply never enables.
    stubGrantedCamera();
    stubGyroPermission("denied");
    const { orderCode, button, cameraStatus, captureButton } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (cameraStatus.textContent !== "Camera started.") throw new Error("still pending");
    });

    expect(captureButton.disabled).toBe(true);
  });

  it("the no-gyro-reading guard inside the click handler still fires as defense in depth", async () => {
    // The button being disabled is the primary guard (test above); this proves the handler's own
    // internal check still holds too, for whatever bypasses the disabled attribute.
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);

    mounted.captureButton.disabled = false;
    mounted.captureButton.click();

    expect(mounted.state.lastCapturedFrame).toBeNull();
    expect(mounted.captureStatus.textContent).toMatch(/no gyro reading/i);
  });

  it("stops every camera track and fully resets camera state once the session actually finishes", async () => {
    // TabeenRaoof's PR #134 review: the original teardown stopped the stream's tracks but left
    // srcObject and cameraGranted stale -- a frozen last frame still showing in the preview, and
    // state claiming the camera was still granted when it had actually been released.
    const mounted = await startSuccessfulSession();
    await uploadAcceptedFrame(mounted.state.sessionId as string);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (mounted.finishStatus.textContent === "") throw new Error("still pending");
    });

    expect(mounted.trackStopSpy).toHaveBeenCalledTimes(1);
    expect(mounted.state.cameraStream).toBeNull();
    expect(mounted.state.cameraGranted).toBe(false);
    expect(mounted.cameraPreview.srcObject).toBeNull();
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

  it("a failed session stops the just-granted stream immediately, rather than leaving it live for a retry to leak", async () => {
    // Before C-09's item-2 fix, a failed session left the granted stream attached and running
    // (nothing to capture into, no way to reach Finish); a second Start then overwrote
    // state.cameraStream with a fresh stream, leaking the first one's tracks forever
    // (TabeenRaoof's PR #134 and #137 reviews -- #134 added the overwrite-releases-the-old-one
    // defense below; #137 pointed out the failed-session case still never reached it, since the
    // stream was never assigned to state.cameraStream in the first place).
    const streamStopSpies: ReturnType<typeof vi.fn>[] = [];
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => {
        const spy = vi.fn();
        streamStopSpies.push(spy);
        const stream = new MediaStream();
        stream.getTracks = () => [{ stop: spy } as unknown as MediaStreamTrack];
        return Promise.resolve(stream);
      },
    };
    stubGyroPermission("granted");
    const { state, orderCode, button, gyroStatus, cameraStatus } = mount();
    orderCode.value = "PO-NOT-REAL"; // fails session creation

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "" || cameraStatus.textContent === "") {
        throw new Error("still pending");
      }
    });

    expect(state.cameraStream).toBeNull();
    expect(streamStopSpies).toHaveLength(1);
    expect(streamStopSpies[0]).toHaveBeenCalledTimes(1);

    orderCode.value = "PO-48213-A"; // a real order code this time
    button.click();
    await vi.waitFor(() => {
      if (state.cameraStream === null) throw new Error("still pending");
    });

    // The retry's own stream is live, not also stopped.
    expect(streamStopSpies).toHaveLength(2);
    expect(streamStopSpies[1]).not.toHaveBeenCalled();
  });

  it("releaseCamera still stops a stream a new grant is about to replace, as defense in depth", async () => {
    // With item 2 fixed, a failed session never lets a stream reach state.cameraStream at all,
    // so the two-real-Start-clicks path that originally caught this (PR #134) can no longer
    // happen through the UI -- startButton stays disabled once a session succeeds. This proves
    // the release-before-replace guard in the camera-grant branch still holds regardless, by
    // invoking it directly the way a future retry path might.
    const streamStopSpies: ReturnType<typeof vi.fn>[] = [];
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => {
        const spy = vi.fn();
        streamStopSpies.push(spy);
        const stream = new MediaStream();
        stream.getTracks = () => [{ stop: spy } as unknown as MediaStreamTrack];
        return Promise.resolve(stream);
      },
    };
    stubGyroPermission("granted");
    const { state, orderCode, button } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (state.cameraStream === null) throw new Error("still pending");
    });
    const firstStream = state.cameraStream;
    expect(streamStopSpies[0]).not.toHaveBeenCalled();

    button.disabled = false; // bypass -- startButton.disabled correctly blocks this in real use
    button.click();
    await vi.waitFor(() => {
      if (state.cameraStream === firstStream || state.cameraStream === null) {
        throw new Error("still pending");
      }
    });

    expect(streamStopSpies).toHaveLength(2);
    expect(streamStopSpies[0]).toHaveBeenCalledTimes(1); // the first stream was released
    expect(streamStopSpies[1]).not.toHaveBeenCalled(); // the second (current) one wasn't
  });

  it("releases a granted-but-unusable stream when the preview's play() rejects", async () => {
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, cameraStatus, gyroStatus, captureButton } = mount();
    // A bad order code the first time around -- session creation fails, so the Start button is
    // re-enabled for the real second click below (a disabled button doesn't dispatch click()).
    orderCode.value = "PO-NOT-REAL";

    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });
    const preview = document.getElementById("shell-camera-preview") as HTMLVideoElement;
    preview.play = () => Promise.reject(new DOMException("", "NotAllowedError"));
    // Trigger the rejecting play() path by starting over -- releaseCamera() runs first (the
    // regression above), then the new grant's play() call is the one that rejects this time.
    orderCode.value = "PO-48213-A";
    button.click();

    await vi.waitFor(() => {
      if (!cameraStatus.textContent.includes("failed to play")) {
        throw new Error("still pending");
      }
    });

    expect(state.cameraStream).toBeNull();
    expect(state.cameraGranted).toBe(false);
    expect(captureButton.disabled).toBe(true);
    expect(trackStopSpy).toHaveBeenCalled();
  });

  it("keeps capture disabled with a misleading-free state when camera is granted but gyro is denied", async () => {
    // Before this fix, captureButton was enabled on camera grant alone -- with gyro denied (or
    // no sensor), tapping it could only ever show "No gyro reading yet -- try again in a
    // moment," forever, which is misleading for a permanent state, not a transient one
    // (TabeenRaoof's PR #134 review).
    stubGrantedCamera();
    stubGyroPermission("denied");
    const { orderCode, button, cameraStatus, captureButton } = mount();
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (cameraStatus.textContent === "") throw new Error("still pending");
    });
    await vi.waitFor(() => {
      if (cameraStatus.textContent !== "Camera started.") throw new Error("still pending");
    });

    expect(captureButton.disabled).toBe(true);
  });

  it("ignores a stale play() resolution that arrives after the stream was already released", async () => {
    // TabeenRaoof's PR #134 review: a slow play() resolving after a later Finish already
    // released the stream could resurrect "Camera started." and re-enable capture on a stream
    // that no longer exists.
    let resolvePlay: (() => void) | undefined;
    const fakeStream = new MediaStream();
    const trackStopSpy = vi.fn();
    fakeStream.getTracks = () => [{ stop: trackStopSpy } as unknown as MediaStreamTrack];
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };
    stubGyroPermission("granted");
    const { state, orderCode, button, gyroStatus, cameraStatus, captureButton, finishStatus } =
      mount();
    // A bad order code the first time around -- session creation fails, so the Start button is
    // re-enabled for the real second click below (a disabled button doesn't dispatch click()).
    orderCode.value = "PO-NOT-REAL";
    button.click();
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("still pending");
    });
    const preview = document.getElementById("shell-camera-preview") as HTMLVideoElement;
    preview.play = () =>
      new Promise((resolve) => {
        resolvePlay = resolve;
      });
    // Start over, this time with a real order code -- this run's play() is the controlled,
    // not-yet-resolved one, and session creation actually succeeds.
    orderCode.value = "PO-48213-A";
    button.click();
    await vi.waitFor(() => {
      if (resolvePlay === undefined) throw new Error("play() not called yet");
    });

    if (state.sessionId === null) throw new Error("test setup: session id missing");
    await uploadAcceptedFrame(state.sessionId);
    const captureStatusBefore = document.getElementById("shell-capture-status")?.textContent;
    document.getElementById("shell-finish-button")?.dispatchEvent(new Event("click"));
    await vi.waitFor(() => {
      if (finishStatus.textContent === "") throw new Error("still pending");
    });
    expect(state.cameraStream).toBeNull();

    // The stale play() call finally resolves, after release -- its success callback must not
    // resurrect any camera UI state.
    resolvePlay?.();
    await Promise.resolve();
    await Promise.resolve();

    expect(cameraStatus.textContent).not.toBe("Camera started.");
    expect(captureButton.disabled).toBe(true);
    expect(state.cameraStream).toBeNull();
    expect(document.getElementById("shell-capture-status")?.textContent).toBe(captureStatusBefore);
  });

  // C-09: the overlay is remounted into #shell-gyro-overlay on every Start click (fresh coverage
  // tracking per session), so its covered-count element is queried fresh each time rather than
  // cached at mount() time, which would go stale after a remount.
  function coveredCountText(): string | null {
    return document.querySelector("#shell-gyro-overlay .gyro-overlay-count")?.textContent ?? null;
  }

  function overlayStatusText(): string | null {
    return document.querySelector("#shell-gyro-overlay .gyro-overlay-status")?.textContent ?? null;
  }

  async function captureAndWaitForStatus(
    mounted: Awaited<ReturnType<typeof startSuccessfulSession>>,
    match: RegExp,
  ): Promise<void> {
    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (!match.test(mounted.captureStatus.textContent)) {
        throw new Error(`still waiting for captureStatus to match ${String(match)}`);
      }
    });
  }

  it("a gate rejection for blur never calls uploadFrame", async () => {
    const mounted = await startSuccessfulSession();
    // Flat/uniform -- zero interior-pixel contrast, so the Laplacian variance is exactly 0.
    const flat = sharpWellExposedImageData();
    flat.data.fill(128);
    stubCanvasContext(flat);
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await captureAndWaitForStatus(mounted, /blur/i);

    expect(
      fetchSpy.mock.calls.some(([url]) => typeof url === "string" && url.includes("/frames")),
    ).toBe(false);
    expect(coveredCountText()).toMatch(/^Covered: 0\//);
  });

  it("a gate rejection for exposure never calls uploadFrame", async () => {
    const mounted = await startSuccessfulSession();
    // Sharp (strong contrast -> clears the blur check) but every pixel is clipped (0 or 255).
    const clipped = sharpWellExposedImageData();
    for (let i = 0; i < clipped.data.length; i += 4) {
      const value = (i / 4) % 2 === 0 ? 0 : 255;
      clipped.data[i] = value;
      clipped.data[i + 1] = value;
      clipped.data[i + 2] = value;
      clipped.data[i + 3] = 255;
    }
    stubCanvasContext(clipped);
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await captureAndWaitForStatus(mounted, /clipped/i);

    expect(
      fetchSpy.mock.calls.some(([url]) => typeof url === "string" && url.includes("/frames")),
    ).toBe(false);
  });

  it("a gate rejection for duplicate never calls uploadFrame, and only the first capture uploads", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // First capture: passes the gate, uploads, accepted.
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    // Second capture: identical pixels and gyro reading -- both gate signals agree nothing moved.
    await captureAndWaitForStatus(mounted, /duplicate/i);

    expect(
      fetchSpy.mock.calls.some(([url]) => typeof url === "string" && url.includes("/frames")),
    ).toBe(false);
  });

  it("a 202 marks exactly one orbit slot covered and advances the arrow", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    expect(coveredCountText()).toMatch(/^Covered: 0\//);

    await captureAndWaitForStatus(mounted, /accepted/i);

    expect(coveredCountText()).toMatch(/^Covered: 1\//);
  });

  it("a 422 shows the server's reason and does not mark any slot covered", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    stub.rejectNextFrame();

    await captureAndWaitForStatus(mounted, /rejected/i);

    expect(coveredCountText()).toMatch(/^Covered: 0\//);
  });

  it("finishSession shows the too_few_frames message before any frame is accepted, and succeeds after one", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (mounted.finishStatus.textContent === "") throw new Error("still pending");
    });
    expect(mounted.finishStatus.textContent).toMatch(/not enough frames/i);

    await captureAndWaitForStatus(mounted, /accepted/i);
    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.finishStatus.textContent.startsWith("Session finished")) {
        throw new Error("still pending");
      }
    });

    expect(mounted.finishStatus.textContent).toMatch(/^Session finished: sess_/);
  });

  // Holds any fetch whose URL contains urlSubstring pending until resolve() is called, passing
  // every other fetch through to the real implementation (the contract stub itself) -- lets
  // these tests observe button state *while* a specific request is genuinely still in flight.
  function pendingFetchMatching(urlSubstring: string): {
    fetchSpy: ReturnType<typeof vi.fn>;
    resolve: (status: number, body: unknown) => void;
  } {
    const realFetch = globalThis.fetch.bind(globalThis);
    let resolveResponse: ((response: Response) => void) | undefined;
    const fetchSpy = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.includes(urlSubstring)) {
        return new Promise<Response>((resolve) => {
          resolveResponse = resolve;
        });
      }
      return realFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchSpy);
    return {
      fetchSpy,
      resolve: (status: number, body: unknown) => {
        resolveResponse?.(
          new Response(JSON.stringify(body), {
            status,
            headers: { "content-type": "application/json" },
          }),
        );
      },
    };
  }

  // Holds any POST to .../frames pending until resolve() is called, passing every other fetch
  // through to the real implementation (the contract stub itself, for session/finish/etc) --
  // lets these tests observe button state *while* an upload is genuinely still in flight.
  function pendingFramesFetch(): ReturnType<typeof pendingFetchMatching> {
    return pendingFetchMatching("/frames");
  }

  // Same, for .../finish -- lets these tests observe button state while finishSession is
  // genuinely still in flight.
  function pendingFinishFetch(): ReturnType<typeof pendingFetchMatching> {
    return pendingFetchMatching("/finish");
  }

  it("capture is disabled the moment Finish is clicked, and stays disabled after a stale completion -- the reverse race TabeenRaoof's second-pass review caught", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("test setup: session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    const { fetchSpy, resolve } = pendingFinishFetch();
    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (fetchSpy.mock.calls.length === 0) throw new Error("finish fetch not called yet");
    });

    // The window the original bug left open: before finishSession even resolves, Capture must
    // already be disabled, not just Finish.
    expect(mounted.captureButton.disabled).toBe(true);

    mounted.captureButton.disabled = false; // bypass -- proves the handler's own guard too
    mounted.captureButton.click();
    expect(mounted.captureStatus.textContent).toMatch(/session is finishing/i);

    resolve(202, {
      session_id: mounted.state.sessionId,
      order_code: "PO-48213-A",
      status: "finished",
    });
    await vi.waitFor(() => {
      if (mounted.finishStatus.textContent === "") throw new Error("still pending");
    });

    // Still disabled afterward -- nothing (e.g. a lagging, unrelated state change) should
    // re-enable either button on a session that has already finished.
    expect(mounted.captureButton.disabled).toBe(true);
    expect(mounted.finishButton.disabled).toBe(true);
  });

  it("capture stays disabled for the whole upload, even as new gyro readings arrive mid-upload -- the overlapping-upload regression TabeenRaoof's PR #139 review caught", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const { fetchSpy, resolve } = pendingFramesFetch();

    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (mounted.captureStatus.textContent !== "Uploading…") throw new Error("still pending");
    });
    expect(mounted.captureButton.disabled).toBe(true);

    // The real trigger: a gyro reading (the listener fires ~60/s on a phone) arriving while the
    // upload is still in flight must not re-enable the button.
    dispatchOrientation(1, 45, 0);
    expect(mounted.captureButton.disabled).toBe(true);

    // Wait for the /frames request to actually be in flight before resolving it -- encodeFrameAsJpeg
    // takes at least one microtask, so resolving too early would resolve nothing (resolveResponse
    // not yet assigned) and the test would hang on the next waitFor instead of proving anything.
    await vi.waitFor(() => {
      if (fetchSpy.mock.calls.length === 0) throw new Error("fetch not called yet");
    });
    resolve(202, { accepted: true, frame_index: 0 });
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const framesCalls = fetchSpy.mock.calls.filter(([url]) => {
      const u = typeof url === "string" ? url : String(url);
      return u.includes("/frames");
    });
    expect(framesCalls).toHaveLength(1);
  });

  it("Finish is disabled while an upload is in flight, and refuses to run even if the disabled attribute is bypassed", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const { fetchSpy, resolve } = pendingFramesFetch();

    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (mounted.captureStatus.textContent !== "Uploading…") throw new Error("still pending");
    });
    expect(mounted.finishButton.disabled).toBe(true);

    mounted.finishButton.disabled = false; // bypass -- proves the handler's own internal guard too
    mounted.finishButton.click();
    expect(mounted.finishStatus.textContent).toMatch(/upload is still in progress/i);

    await vi.waitFor(() => {
      if (fetchSpy.mock.calls.length === 0) throw new Error("fetch not called yet");
    });
    resolve(202, { accepted: true, frame_index: 0 });
    await vi.waitFor(() => {
      if (mounted.finishButton.disabled) throw new Error("still disabled");
    });
  });

  it("a resolution change mid-session (e.g. the phone rotating) doesn't throw or brick capture for the rest of the session", async () => {
    const mounted = await startSuccessfulSession();
    const { setImageData } = stubCanvasContext(sharpWellExposedImageData(4, 4));
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // iOS swaps videoWidth/videoHeight on rotation -- a differently-sized frame at the same spot.
    setImageData(sharpWellExposedImageData(4, 6));

    expect(() => {
      mounted.captureButton.click();
    }).not.toThrow();
    await vi.waitFor(() => {
      if (mounted.captureStatus.textContent === "Uploading…") throw new Error("still uploading");
    });

    expect(mounted.captureStatus.textContent).toMatch(/accepted/i);
  });

  it("a 202 marks the specific orbit slot the gyro reading falls in, not an adjacent one", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    // Anchors azimuth at 0 and covers slot 0 (azimuth 0, elevation 45 -- DEFAULT_ORBIT's first
    // mid-elevation slot).
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    await captureAndWaitForStatus(mounted, /accepted/i);
    expect(coveredCountText()).toMatch(/^Covered: 1\//);

    // Slot 1 (azimuth 45, same elevation) is the lowest-index uncovered slot only if slot 0 --
    // not some adjacent slot -- was the one actually marked covered. Moving the live reading
    // exactly there should read "on target."
    dispatchOrientation(45, 45, 0);
    await vi.waitFor(() => {
      if (!overlayStatusText()?.match(/on target/i)) throw new Error("not on target yet");
    });
  });

  it("rolls back the gate's duplicate anchor when an upload is rejected, so a retry from the same spot isn't gate-rejected as a duplicate of itself", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext(sharpWellExposedImageData(4, 4));
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // A different position (90 deg away -- well outside checkDuplicate's gyro threshold against
    // frame 1), so this passes the local gate on its own merits, but the server rejects it.
    dispatchOrientation(90, 45, 0);
    stub.rejectNextFrame();
    await captureAndWaitForStatus(mounted, /rejected/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // Retrying the exact same shot (same pixels, same gyro as the rejected attempt): without the
    // rollback, lastAcceptedForGate would still point at that rejected attempt's own data, so
    // this identical retry would be gate-rejected as a duplicate of itself and never reach the
    // network at all.
    await captureAndWaitForStatus(mounted, /accepted/i);
  });

  it("disposes the gyro overlay's listener once the session finishes, so stale guidance doesn't keep reacting to new readings", async () => {
    const mounted = await startSuccessfulSession();
    await uploadAcceptedFrame(mounted.state.sessionId as string);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (mounted.finishStatus.textContent === "") throw new Error("still pending");
    });
    const statusAfterFinish = overlayStatusText();

    // If the listener were still live, a new reading would very likely change the arrow's status
    // (e.g. between "on target" and "move toward the next position").
    dispatchOrientation(123, 7, 55);
    await Promise.resolve();

    expect(overlayStatusText()).toBe(statusAfterFinish);
  });
});
