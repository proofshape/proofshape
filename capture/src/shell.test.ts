// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { createSession, finishSession, uploadFrame } from "./api.js";
import { measureBlurVariance } from "./frame_gate.js";
import {
  browserSessionStorage,
  CAPTURE_SESSION_STORAGE_KEY,
  loadCaptureSession,
  saveCaptureSession,
  type StoredCaptureSession,
} from "./session-persistence.js";
import {
  C25_BLUR_DEBUG_STORAGE_KEY,
  mountAppShell,
  type AppShellOptions,
  type AppState,
  type C25BlurDebugMeasurement,
} from "./shell.js";
import { MIN_FINISH_FRAMES, startContractStub, type ContractStub } from "../stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "../stub/page-test-port.js";

// Same fixed-port reasoning as page.test.ts: happy-dom enforces same-origin, matching
// vite.config.ts's environmentOptions.happyDOM.url.

describe("mountAppShell", () => {
  let close: () => Promise<void>;
  let baseUrl: string;
  let stub: ContractStub;

  beforeEach(async () => {
    // C-24: the shell saves sessions in the tab's sessionStorage by default, and happy-dom's is
    // shared by every test in this file, so each test starts from an empty one.
    window.sessionStorage.clear();
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

  function mount(overrides: Partial<AppShellOptions> = {}): {
    state: AppState;
    orderCode: HTMLInputElement;
    button: HTMLButtonElement;
    sessionStatus: HTMLElement;
    cameraStatus: HTMLElement;
    gyroStatus: HTMLElement;
    cameraPreview: HTMLVideoElement;
    captureButton: HTMLButtonElement;
    captureStatus: HTMLElement;
    blurDebugStatus: HTMLElement;
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
      ...overrides,
    });
    const orderCode = document.getElementById("shell-order-code");
    const button = document.getElementById("shell-start-button");
    const sessionStatus = document.getElementById("shell-session-status");
    const cameraStatus = document.getElementById("shell-camera-status");
    const gyroStatus = document.getElementById("shell-gyro-status");
    const cameraPreview = document.getElementById("shell-camera-preview");
    const captureButton = document.getElementById("shell-capture-button");
    const captureStatus = document.getElementById("shell-capture-status");
    const blurDebugStatus = document.getElementById("shell-blur-debug-status");
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
      !blurDebugStatus ||
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
      blurDebugStatus,
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

  async function startSuccessfulSession(
    overrides: Partial<AppShellOptions> = {},
  ): Promise<ReturnType<typeof mount> & { trackStopSpy: ReturnType<typeof vi.fn> }> {
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount(overrides);
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

  // C-19 (#193): finishSession now requires MIN_FINISH_FRAMES accepted frames, matching
  // recon/service.py -- every caller of this helper wants a session finish to actually
  // succeed afterward.
  async function uploadAcceptedFrame(sessionId: string): Promise<void> {
    const frame = new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" });
    for (let i = 0; i < MIN_FINISH_FRAMES; i += 1) {
      const result = await uploadFrame(baseUrl, sessionId, frame, {
        alpha: 1,
        beta: 2,
        gamma: 3,
      });
      if (!result.ok) throw new Error("test setup: uploadFrame failed");
    }
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

  it("pre-fills the order code from initialOrderCode (C-13: a capture link's ?order=)", () => {
    const { orderCode } = mount({ initialOrderCode: "PO-48213-A" });
    expect(orderCode.value).toBe("PO-48213-A");
  });

  it("leaves the order code blank when initialOrderCode is absent", () => {
    const { orderCode } = mount();
    expect(orderCode.value).toBe("");
  });

  it("sends authToken as a bearer header on the real createSession request (C-13)", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const { orderCode, button, state } = mount({ authToken: "the-token" });
    orderCode.value = "PO-48213-A";

    button.click();
    await vi.waitFor(() => {
      if (state.sessionId === null) throw new Error("still pending");
    });

    const sessionCall = fetchSpy.mock.calls.find(([input]) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      return url.endsWith("/v1/sessions");
    });
    expect(sessionCall).toBeDefined();
    expect(new Headers(sessionCall?.[1]?.headers).get("Authorization")).toBe("Bearer the-token");
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

  it("hides the previous session report link immediately when a new Start begins", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.setReconstructionPendingPolls(0);

    mounted.finishButton.click();
    const reportLink = document.getElementById("shell-report-link");
    expect(reportLink).toBeInstanceOf(HTMLAnchorElement);
    await vi.waitFor(() => {
      if ((reportLink as HTMLAnchorElement).hidden) throw new Error("report link not shown yet");
    });
    const firstHref = (reportLink as HTMLAnchorElement).getAttribute("href");
    expect(firstHref).toMatch(/^report\.html\?session=/);

    mounted.button.disabled = false;
    mounted.button.click();

    expect((reportLink as HTMLAnchorElement).hidden).toBe(true);
    expect((reportLink as HTMLAnchorElement).hasAttribute("href")).toBe(false);
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
    expect(window.sessionStorage.getItem(C25_BLUR_DEBUG_STORAGE_KEY)).toBeNull();
  });

  it("blur-debug records the exact pre-JPEG gate metric beside the accepted server frame index", async () => {
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount({ blurDebug: true });
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });
    expect(trackStopSpy).not.toHaveBeenCalled();

    const { imageData } = stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(12.3, -45.6, 78.9);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (!/accepted/i.test(mounted.captureStatus.textContent)) throw new Error("still uploading");
    });

    const raw = window.sessionStorage.getItem(C25_BLUR_DEBUG_STORAGE_KEY);
    expect(raw).not.toBeNull();
    const measurements = JSON.parse(raw ?? "[]") as C25BlurDebugMeasurement[];
    expect(measurements).toHaveLength(1);
    expect(measurements[0]).toMatchObject({
      session_id: mounted.state.sessionId,
      frame_index: 0,
      width: imageData.width,
      height: imageData.height,
    });
    expect(measurements[0]?.pre_jpeg_variance).toBe(measureBlurVariance(imageData));
    expect(mounted.blurDebugStatus.textContent).toMatch(
      /1 measurement\(s\) stored for this session/i,
    );
  });

  it("blur-debug reports a per-session count when the storage key already has another session", async () => {
    window.sessionStorage.setItem(
      C25_BLUR_DEBUG_STORAGE_KEY,
      JSON.stringify([
        {
          session_id: "sess_previous",
          frame_index: 0,
          pre_jpeg_variance: 123,
          width: 640,
          height: 480,
        },
      ]),
    );
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount({ blurDebug: true });
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });
    expect(trackStopSpy).not.toHaveBeenCalled();

    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(12.3, -45.6, 78.9);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (!/accepted/i.test(mounted.captureStatus.textContent)) throw new Error("still uploading");
    });

    expect(mounted.blurDebugStatus.textContent).toMatch(
      /1 measurement\(s\) stored for this session/i,
    );
  });

  it("blur-debug warns immediately when measurement storage is unavailable", () => {
    const mounted = mount({ blurDebug: true, blurDebugStorage: null });

    expect(mounted.blurDebugStatus.hidden).toBe(false);
    expect(mounted.blurDebugStatus.textContent).toMatch(/storage is unavailable/i);
    expect(mounted.blurDebugStatus.style.fontWeight).toBe("bold");
  });

  it("blur-debug warns after an accepted frame when measurement storage throws", async () => {
    const throwingStorage = {
      getItem: () => null,
      setItem: () => {
        throw new Error("storage blocked");
      },
      removeItem: () => undefined,
    };
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount({ blurDebug: true, blurDebugStorage: throwingStorage });
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });
    expect(trackStopSpy).not.toHaveBeenCalled();

    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(12.3, -45.6, 78.9);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (!/accepted/i.test(mounted.captureStatus.textContent)) throw new Error("still uploading");
    });

    expect(mounted.blurDebugStatus.textContent).toMatch(/could not be stored/i);
    expect(mounted.blurDebugStatus.textContent).toMatch(/evidence is incomplete/i);
    expect(mounted.blurDebugStatus.style.fontWeight).toBe("bold");
  });

  it("blur-debug keeps an earlier storage failure visible after a later successful write", async () => {
    let writes = 0;
    let saved = "[]";
    const flakyStorage = {
      getItem: () => saved,
      setItem: (_key: string, value: string) => {
        writes += 1;
        if (writes === 1) {
          throw new Error("first write blocked");
        }
        saved = value;
      },
      removeItem: () => undefined,
    };
    const { trackStopSpy } = stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount({ blurDebug: true, blurDebugStorage: flakyStorage });
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });
    expect(trackStopSpy).not.toHaveBeenCalled();

    const canvas = stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(12.3, -45.6, 78.9);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (!/accepted/i.test(mounted.captureStatus.textContent)) throw new Error("still uploading");
    });
    expect(mounted.blurDebugStatus.textContent).toMatch(/1 failure\(s\)/i);

    canvas.setImageData(sharpWellExposedImageData(4, 6));
    setVideoDimensions(mounted.cameraPreview, 480, 640);
    dispatchOrientation(30, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (!/accepted/i.test(mounted.captureStatus.textContent)) throw new Error("still uploading");
    });

    expect(mounted.blurDebugStatus.textContent).toMatch(/1 failed/i);
    expect(mounted.blurDebugStatus.textContent).toMatch(/evidence is incomplete/i);
    expect(mounted.blurDebugStatus.style.fontWeight).toBe("bold");
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

  it("an all-black frame reports exposure before blur and never calls uploadFrame", async () => {
    const mounted = await startSuccessfulSession();
    // Uniform black fails both checks: it is fully clipped *and* has zero Laplacian variance.
    // Exposure must run first so the user gets the actionable lighting error, not "hold steadier."
    const black = sharpWellExposedImageData();
    for (let i = 0; i < black.data.length; i += 4) {
      black.data[i] = 0;
      black.data[i + 1] = 0;
      black.data[i + 2] = 0;
      black.data[i + 3] = 255;
    }
    stubCanvasContext(black);
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await captureAndWaitForStatus(mounted, /frame_rejected_exposure/i);

    expect(mounted.captureStatus.textContent).not.toMatch(/frame_rejected_blur/i);
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

  it("finishSession shows the too_few_frames message before enough frames are accepted, and succeeds after the minimum", async () => {
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

    // Each capture needs a distinct orbit slot -- the coverage gate rejects a repeat at the
    // same one as a duplicate (see "rolls back the gate's duplicate anchor..." below).
    for (let slot = 0; slot < MIN_FINISH_FRAMES; slot += 1) {
      dispatchOrientation(slot * 45, 45, 0);
      await vi.waitFor(() => {
        if (mounted.captureButton.disabled) throw new Error("still disabled");
      });
      await captureAndWaitForStatus(mounted, /accepted/i);
    }
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

  function vlmHintText(): string | null {
    return document.getElementById("shell-vlm-hint")?.textContent ?? null;
  }

  const FEEDBACK = {
    frame_index: 0,
    hint: "Good start: the part and the reference sheet are both in view.",
    same_part: null,
    hands_occluding: false,
    screen_photo: false,
    lighting_adequate: true,
  };

  it("A-11: shows the VLM tip once an accepted upload carries one, naming the earlier frame", async () => {
    stub.useVlmFeedback();
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // Frame 0: no check has finished yet, so there's no tip.
    await captureAndWaitForStatus(mounted, /accepted/i);
    expect(vlmHintText()).toBe("");
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // Frame 1 (a different spot, so the local duplicate gate lets it through): the stub now
    // reports frame 0's check.
    dispatchOrientation(90, 45, 0);
    await captureAndWaitForStatus(mounted, /frame 1/i);
    expect(vlmHintText()).toBe(
      "Tip (from frame 0): Good start: the part and the reference sheet are both in view.",
    );
  });

  it("A-11: keeps the tip through a later 202 without one, shows it as text, and clears it on Start", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    const { fetchSpy, resolve } = pendingFramesFetch();

    // Model output that looks like markup must stay text.
    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (fetchSpy.mock.calls.length === 0) throw new Error("fetch not called yet");
    });
    resolve(202, {
      accepted: true,
      frame_index: 0,
      vlm_feedback: { ...FEEDBACK, hint: "<img src=x onerror=alert(1)> move closer" },
    });
    await vi.waitFor(() => {
      if (vlmHintText() === "") throw new Error("no tip yet");
    });
    const hint = document.getElementById("shell-vlm-hint");
    expect(hint?.textContent).toBe("Tip (from frame 0): <img src=x onerror=alert(1)> move closer");
    expect(hint?.querySelector("img")).toBeNull();
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    // A later 202 without feedback (no newer check yet) leaves the tip alone.
    dispatchOrientation(90, 45, 0);
    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (fetchSpy.mock.calls.filter(([u]) => String(u).includes("/frames")).length < 2) {
        throw new Error("second upload not sent yet");
      }
    });
    resolve(202, { accepted: true, frame_index: 1 });
    await vi.waitFor(() => {
      if (!/frame 1/.test(mounted.captureStatus.textContent)) throw new Error("still pending");
    });
    expect(vlmHintText()).toMatch(/^Tip \(from frame 0\)/);

    // A new session is a new part: the old tip goes the moment Start is clicked. Start is
    // disabled once a session exists, so bypass that to prove the handler's own clear.
    mounted.button.disabled = false;
    mounted.button.click();
    expect(vlmHintText()).toBe("");
    await vi.waitFor(() => {
      if (mounted.sessionStatus.textContent === "") throw new Error("still pending");
    });
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

  it("releases Capture and Finish while the optional agent request is still pending", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    const { fetchSpy, resolve } = pendingFetchMatching("/agent-message");
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      const agentCalls = fetchSpy.mock.calls.filter(([url]) => {
        const value = typeof url === "string" ? url : String(url);
        return value.includes("/agent-message");
      });
      if (agentCalls.length === 0) throw new Error("agent request not started");
    });

    expect(mounted.captureButton.disabled).toBe(false);
    expect(mounted.finishButton.disabled).toBe(false);

    resolve(200, {
      finished: false,
      fallback_reason: null,
      message: "Frame accepted. Continue to the next guided view.",
    });
  });

  it("keeps the assistant panel hidden when today's real-style service has no agent route", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });

    const realFetch = globalThis.fetch.bind(globalThis);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url =
          typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        if (url.includes("/agent-message")) {
          return Promise.resolve(
            new Response(JSON.stringify({ code: "session_not_found", message: "Not Found" }), {
              status: 404,
              headers: { "content-type": "application/json" },
            }),
          );
        }
        return realFetch(input, init);
      }),
    );

    await captureAndWaitForStatus(mounted, /accepted/i);
    await Promise.resolve();
    await Promise.resolve();

    const panel = document.getElementById("shell-agent-conversation");
    expect(panel?.hidden).toBe(true);
  });
  it("drops an older agent reply that arrives after a newer reply for the same session", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);

    const realFetch = globalThis.fetch.bind(globalThis);
    const agentResolvers: Array<(response: Response) => void> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url =
          typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        if (url.includes("/agent-message")) {
          return new Promise<Response>((resolve) => {
            agentResolvers.push(resolve);
          });
        }
        return realFetch(input, init);
      }),
    );

    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (agentResolvers.length < 1) throw new Error("first agent request not started");
    });

    dispatchOrientation(90, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (agentResolvers.length < 2) throw new Error("second agent request not started");
    });

    agentResolvers[1]?.(
      new Response(
        JSON.stringify({
          finished: false,
          fallback_reason: null,
          message: "Newer guidance.",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    await vi.waitFor(() => {
      if (!document.body.textContent.includes("Newer guidance.")) {
        throw new Error("newer guidance not shown");
      }
    });

    const olderResponse = new Response(
      JSON.stringify({
        finished: false,
        fallback_reason: null,
        message: "Older guidance.",
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
    agentResolvers[0]?.(olderResponse);
    await vi.waitFor(() => {
      if (!olderResponse.bodyUsed) throw new Error("older response body not read yet");
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(document.body.textContent).toContain("Newer guidance.");
    expect(document.body.textContent).not.toContain("Older guidance.");
  });

  // C-24 (#352): a reload is simulated by mounting the shell again in the same document, which
  // is what a real reload leaves behind: a fresh page with the tab's sessionStorage intact.
  function savedSessionInTab(): StoredCaptureSession | null {
    return loadCaptureSession(browserSessionStorage());
  }

  function resumeElements(): {
    panel: HTMLElement;
    text: HTMLElement;
    continueButton: HTMLButtonElement;
    showResultButton: HTMLButtonElement;
    discardButton: HTMLButtonElement;
  } {
    const panel = document.getElementById("shell-resume");
    const text = document.getElementById("shell-resume-text");
    const continueButton = document.getElementById("shell-continue-button");
    const showResultButton = document.getElementById("shell-show-result-button");
    const discardButton = document.getElementById("shell-discard-button");
    if (
      !panel ||
      !text ||
      !(continueButton instanceof HTMLButtonElement) ||
      !(showResultButton instanceof HTMLButtonElement) ||
      !(discardButton instanceof HTMLButtonElement)
    ) {
      throw new Error("test setup: resume panel elements missing");
    }
    return { panel, text, continueButton, showResultButton, discardButton };
  }

  async function captureOneAcceptedFrame(
    mounted: Awaited<ReturnType<typeof startSuccessfulSession>>,
    gyro: [number, number, number] = [1, 45, 0],
  ): Promise<void> {
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(...gyro);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("upload still in flight");
    });
  }

  it("C-24: saves the session at Start and after every accepted frame", async () => {
    const mounted = await startSuccessfulSession();
    const sessionId = mounted.state.sessionId;
    expect(savedSessionInTab()).toEqual({
      sessionId,
      orderCode: "PO-48213-A",
      phase: "capturing",
      acceptedFrameCount: 0,
      acceptedGyro: [],
    });

    await captureOneAcceptedFrame(mounted);

    const saved = savedSessionInTab();
    expect(saved?.acceptedFrameCount).toBe(1);
    expect(saved?.acceptedGyro).toEqual([{ alpha: 1, beta: 45, gamma: 0 }]);
  });

  it("C-24: does not save a frame the server rejected", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    stub.rejectNextFrame();
    await captureAndWaitForStatus(mounted, /rejected/i);

    expect(savedSessionInTab()?.acceptedFrameCount).toBe(0);
    expect(savedSessionInTab()?.acceptedGyro).toEqual([]);
  });

  it("C-24: after a reload, offers Continue for an unfinished session, naming it and its frames", async () => {
    const first = await startSuccessfulSession();
    await captureOneAcceptedFrame(first);

    mount();
    const resume = resumeElements();

    expect(resume.panel.hidden).toBe(false);
    expect(resume.text.textContent).toContain(first.state.sessionId ?? "missing");
    expect(resume.text.textContent).toContain("1 frame accepted");
    expect(resume.continueButton.hidden).toBe(false);
    expect(resume.showResultButton.hidden).toBe(true);
  });

  it("C-24: Continue sends the next frame to the same session, without creating one, and rebuilds coverage", async () => {
    const first = await startSuccessfulSession();
    await captureOneAcceptedFrame(first);
    const originalSessionId = first.state.sessionId;

    const reloaded = mount();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    resumeElements().continueButton.click();
    await vi.waitFor(() => {
      if (reloaded.gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(reloaded.state.sessionId).toBe(originalSessionId);
    expect(reloaded.sessionStatus.textContent).toBe(
      `Continuing session ${originalSessionId ?? ""} (1 frame accepted so far).`,
    );
    expect(reloaded.button.disabled).toBe(true);
    expect(resumeElements().panel.hidden).toBe(true);
    expect(coveredCountText()).toMatch(/^Covered: 1\//);
    expect(
      fetchSpy.mock.calls.some(
        ([url, init]) =>
          typeof url === "string" && url.endsWith("/v1/sessions") && init?.method === "POST",
      ),
    ).toBe(false);

    await captureOneAcceptedFrame({ ...reloaded, trackStopSpy: vi.fn() }, [90, 45, 0]);
    expect(reloaded.captureStatus.textContent).toBe("Frame accepted (frame 1)");
    expect(savedSessionInTab()?.acceptedFrameCount).toBe(2);
    expect(savedSessionInTab()?.sessionId).toBe(originalSessionId);
  });

  it("C-24: Start new drops the saved session", async () => {
    await startSuccessfulSession();
    mount();

    resumeElements().discardButton.click();

    expect(resumeElements().panel.hidden).toBe(true);
    expect(window.sessionStorage.getItem(CAPTURE_SESSION_STORAGE_KEY)).toBeNull();
  });

  it("C-24: Start replaces a saved session with the new one", async () => {
    const first = await startSuccessfulSession();
    const reloaded = mount();
    reloaded.orderCode.value = "PO-48213-A";

    reloaded.button.click();
    await vi.waitFor(() => {
      if (reloaded.gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(resumeElements().panel.hidden).toBe(true);
    expect(reloaded.state.sessionId).not.toBe(first.state.sessionId);
    expect(savedSessionInTab()?.sessionId).toBe(reloaded.state.sessionId);
  });

  it("C-24: a continued session the server no longer has is dropped, and Start comes back", async () => {
    saveCaptureSession(browserSessionStorage(), {
      sessionId: "sess_gone",
      orderCode: "PO-48213-A",
      phase: "capturing",
      acceptedFrameCount: 4,
      acceptedGyro: [],
    });
    stubGrantedCamera();
    stubGyroPermission("granted");
    const reloaded = mount();
    resumeElements().continueButton.click();
    await vi.waitFor(() => {
      if (reloaded.gyroStatus.textContent === "") throw new Error("still pending");
    });

    stubCanvasContext();
    setVideoDimensions(reloaded.cameraPreview, 640, 480);
    dispatchOrientation(1, 45, 0);
    await vi.waitFor(() => {
      if (reloaded.captureButton.disabled) throw new Error("still disabled");
    });
    reloaded.captureButton.click();
    await vi.waitFor(() => {
      if (!/start a new session/.test(reloaded.captureStatus.textContent)) {
        throw new Error("dead-session message not shown yet");
      }
    });

    expect(reloaded.captureStatus.textContent).toContain("No session with that id.");
    expect(window.sessionStorage.getItem(CAPTURE_SESSION_STORAGE_KEY)).toBeNull();
    expect(reloaded.state.sessionId).toBeNull();
    expect(reloaded.button.disabled).toBe(false);
    expect(reloaded.finishButton.disabled).toBe(true);
    // Review (@dalwalyk, PR #375): the camera and gyro are still live here, so only the missing
    // session keeps Capture disabled.
    expect(reloaded.state.cameraPlaying).toBe(true);
    expect(reloaded.state.latestGyro).not.toBeNull();
    expect(reloaded.captureButton.disabled).toBe(true);
  });

  it("C-24: keeps a finished session saved while reconstruction is still pending", async () => {
    const mounted = await startSuccessfulSession();
    if (mounted.state.sessionId === null) throw new Error("session id missing");
    await uploadAcceptedFrame(mounted.state.sessionId);
    stub.setReconstructionPendingPolls(99);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("timed out")) {
        throw new Error("still polling");
      }
    });

    expect(savedSessionInTab()?.phase).toBe("finished");
  });

  it("C-24: after a reload during reconstruction, Show result resumes polling and then forgets the session", async () => {
    const created = await createSession(baseUrl, "PO-48213-A");
    if (!created.ok) throw new Error("test setup: createSession failed");
    const sessionId = created.body.session_id;
    await uploadAcceptedFrame(sessionId);
    const finished = await finishSession(baseUrl, sessionId);
    if (!finished.ok) throw new Error("test setup: finishSession failed");
    stub.setReconstructionPendingPolls(1);
    saveCaptureSession(browserSessionStorage(), {
      sessionId,
      orderCode: "PO-48213-A",
      phase: "finished",
      acceptedFrameCount: MIN_FINISH_FRAMES,
      acceptedGyro: [],
    });

    const reloaded = mount();
    const resume = resumeElements();
    expect(resume.text.textContent).toContain("was finished");
    expect(resume.continueButton.hidden).toBe(true);
    expect(resume.showResultButton.hidden).toBe(false);

    resume.showResultButton.click();
    await vi.waitFor(() => {
      if (!reloaded.reconstructionStatus.textContent.includes("Reconstruction complete")) {
        throw new Error("reconstruction not shown yet");
      }
    });

    expect(reloaded.finishStatus.textContent).toBe(`Session finished: ${sessionId}`);
    expect(reloaded.captureButton.disabled).toBe(true);
    expect(reloaded.finishButton.disabled).toBe(true);
    expect(document.getElementById("shell-report-link")?.getAttribute("href")).toBe(
      `report.html?session=${sessionId}`,
    );
    expect(window.sessionStorage.getItem(CAPTURE_SESSION_STORAGE_KEY)).toBeNull();
  });

  it("C-24: a corrupt saved value is ignored and removed, and Start still works", async () => {
    window.sessionStorage.setItem(CAPTURE_SESSION_STORAGE_KEY, "{not json");

    const mounted = mount();

    expect(resumeElements().panel.hidden).toBe(true);
    expect(window.sessionStorage.getItem(CAPTURE_SESSION_STORAGE_KEY)).toBeNull();
    stubGrantedCamera();
    stubGyroPermission("granted");
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });
    expect(mounted.state.sessionId).not.toBeNull();
  });

  it("C-24: with storage turned off, a session still runs and nothing is saved", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const mounted = mount({ captureSessionStorage: null });
    mounted.orderCode.value = "PO-48213-A";
    mounted.button.click();
    await vi.waitFor(() => {
      if (mounted.gyroStatus.textContent === "") throw new Error("still pending");
    });

    expect(mounted.state.sessionId).not.toBeNull();
    expect(window.sessionStorage.getItem(CAPTURE_SESSION_STORAGE_KEY)).toBeNull();
  });

  // C-26: the viewfinder's frame colour and the round shutter.
  function viewfinderGuidance(): string | undefined {
    return document.getElementById("shell-viewfinder")?.dataset.guidance;
  }

  it("C-26: the viewfinder frame is idle before the first photo, then follows far, near and on-target", async () => {
    const mounted = await startSuccessfulSession();
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    // alpha 0, beta 45 -> heading 0, elevation 45: DEFAULT_ORBIT's slot 0.
    dispatchOrientation(0, 45, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    expect(viewfinderGuidance()).toBe("idle");

    await captureAndWaitForStatus(mounted, /accepted/i);
    // Next target is slot 1 (azimuth 45); the live reading is still at 0, 45 degrees away.
    expect(viewfinderGuidance()).toBe("far");

    dispatchOrientation(20, 45, 0);
    expect(viewfinderGuidance()).toBe("near");

    dispatchOrientation(45, 45, 0);
    expect(viewfinderGuidance()).toBe("on-target");
  });

  it("C-26: the shutter is inside the viewfinder and keeps its id and accessible name", () => {
    const mounted = mount();
    expect(mounted.captureButton.closest("#shell-viewfinder")).not.toBeNull();
    expect(mounted.captureButton.getAttribute("aria-label")).toBe("Capture frame");
    expect(mounted.captureButton.textContent.trim()).toBe("Capture frame");
    expect(mounted.captureButton.disabled).toBe(true);
    expect(document.querySelector("#shell-viewfinder #shell-gyro-overlay")).not.toBeNull();
  });

  // C-27: auto-capture. The clock is injected; orientation events are dispatched by hand, as a
  // phone would deliver them. The switch is off by default (#436), so these sessions turn it on,
  // as the person capturing would, unless a test says otherwise.
  function autoCaptureToggle(): HTMLInputElement {
    const toggle = document.getElementById("shell-auto-capture");
    if (!(toggle instanceof HTMLInputElement)) throw new Error("test: toggle missing");
    return toggle;
  }

  async function startAutoSession(
    firstPhotoBeta = 45,
    { autoOn = true }: { autoOn?: boolean } = {},
  ): Promise<{
    mounted: Awaited<ReturnType<typeof startSuccessfulSession>>;
    clock: { t: number };
    canvas: ReturnType<typeof stubCanvasContext>;
  }> {
    const clock = { t: 0 };
    const mounted = await startSuccessfulSession({ now: () => clock.t });
    // Left alone when autoOn is false, so a test can see the default.
    if (autoOn) autoCaptureToggle().checked = true;
    const canvas = stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    // alpha 0, beta 45 by default: DEFAULT_ORBIT's slot 0. The first photo is a tap.
    dispatchOrientation(0, firstPhotoBeta, 0);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("still disabled");
    });
    await captureAndWaitForStatus(mounted, /accepted/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("upload still in flight");
    });
    return { mounted, clock, canvas };
  }

  function holdAt(
    clock: { t: number },
    alpha: number,
    fromMs: number,
    untilMs: number,
    beta = 45,
  ): void {
    for (let t = fromMs; t <= untilMs; t += 100) {
      clock.t = t;
      dispatchOrientation(alpha, beta, 0);
    }
  }

  function frameUploads(fetchSpy: MockInstance<typeof fetch>): number {
    return fetchSpy.mock.calls.filter(([url]) => typeof url === "string" && url.includes("/frames"))
      .length;
  }

  it("C-27: after the first tapped photo, holding on the next target takes the next photo by itself", async () => {
    const { mounted, clock } = await startAutoSession();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    expect(coveredCountText()).toMatch(/^Covered: 1\//);

    // Slot 1 is azimuth 45. 500 ms of holding is not enough.
    holdAt(clock, 45, 1000, 1500);
    expect(frameUploads(fetchSpy)).toBe(0);
    expect(overlayStatusText()).toMatch(/hold steady/i);

    holdAt(clock, 45, 1600, 1600);
    await vi.waitFor(() => {
      if (!/^Covered: 2\//.test(coveredCountText() ?? "")) throw new Error("not yet covered");
    });
    expect(frameUploads(fetchSpy)).toBe(1);
    expect(mounted.captureStatus.textContent).toMatch(/^Auto: .*accepted/i);
  });

  // The C-26 device-check bug (2026-11-20): after a first photo taken high, the frame went green
  // at about 58 deg, and auto-capture took a photo every 1.5 s that coverage credited to the
  // already-covered 70 deg slot, so the count never moved.
  it("C-27: does not fire where the photo would count for an already-covered slot", async () => {
    // beta 12 = elevation 78: the first photo covers the 70 deg slot, the next target is slot 0.
    const { clock } = await startAutoSession(12);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    expect(coveredCountText()).toMatch(/^Covered: 1\//);

    // beta 32 = elevation 58, within 15 deg of both rings. Held for 3 s: nothing.
    holdAt(clock, 0, 1000, 4000, 32);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(0);
    expect(document.getElementById("shell-viewfinder")?.dataset.guidance).toBe("near");

    // beta 40 = elevation 50: credited to slot 0, so it goes green and fires once.
    holdAt(clock, 0, 4100, 4800, 40);
    await vi.waitFor(() => {
      if (!/^Covered: 2\//.test(coveredCountText() ?? "")) throw new Error("not yet covered");
    });
    expect(frameUploads(fetchSpy)).toBe(1);
  });

  it("C-27: never fires before the first photo, however long the phone is held", async () => {
    const clock = { t: 0 };
    const mounted = await startSuccessfulSession({ now: () => clock.t });
    autoCaptureToggle().checked = true;
    stubCanvasContext();
    setVideoDimensions(mounted.cameraPreview, 640, 480);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    holdAt(clock, 0, 0, 5000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(0);
    expect(document.getElementById("shell-viewfinder")?.dataset.guidance).toBe("idle");
  });

  it("C-27: the switch starts off, and a held on-target reading takes nothing until it is turned on", async () => {
    const { clock } = await startAutoSession(45, { autoOn: false });
    expect(autoCaptureToggle().checked).toBe(false);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    holdAt(clock, 45, 1000, 5000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(0);
    expect(overlayStatusText()).toMatch(/take the photo/i);

    autoCaptureToggle().checked = true;
    holdAt(clock, 45, 5100, 5800);
    await vi.waitFor(() => {
      if (!/^Covered: 2\//.test(coveredCountText() ?? "")) throw new Error("not yet covered");
    });
    expect(frameUploads(fetchSpy)).toBe(1);
  });

  it("C-27: switched off again, holding on target takes nothing and the status says to tap", async () => {
    const { clock } = await startAutoSession();
    holdAt(clock, 45, 1000, 1100);
    expect(overlayStatusText()).toMatch(/hold steady/i);
    autoCaptureToggle().checked = false;
    autoCaptureToggle().dispatchEvent(new Event("change"));
    // Updated by the switch itself, before the phone moves again.
    expect(overlayStatusText()).toMatch(/take the photo/i);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    holdAt(clock, 45, 1200, 5000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(0);
    expect(overlayStatusText()).toMatch(/take the photo/i);
  });

  it("C-27: never starts a second capture while an upload is still in flight", async () => {
    const { clock } = await startAutoSession();
    const realFetch = globalThis.fetch;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      if (typeof input === "string" && input.includes("/frames")) {
        return new Promise<Response>(() => undefined); // never settles: the upload hangs
      }
      return realFetch(input, init);
    });
    holdAt(clock, 45, 1000, 1600);
    await vi.waitFor(() => {
      if (frameUploads(fetchSpy) === 0) throw new Error("first auto upload not started");
    });
    // Well past both the hold and the retry gap, 5 degrees over: still on target, and far enough
    // from the hanging frame (2 degree duplicate threshold) that the duplicate gate would pass it.
    // Only the shutter-enabled guard (no upload in flight) can stop this second capture.
    holdAt(clock, 50, 1700, 8000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(1);
  });

  // #420 review (@dalwalyk): a failed automatic upload used to be retried every retry gap for as
  // long as the phone stayed on target.
  it("C-27: an automatic photo the server rejects turns auto-capture off instead of retrying", async () => {
    const { mounted, clock } = await startAutoSession();
    stub.rejectNextFrame();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    holdAt(clock, 45, 1000, 1600);
    await vi.waitFor(() => {
      if (autoCaptureToggle().checked) throw new Error("auto still on");
    });
    expect(mounted.captureStatus.textContent).toMatch(/^Auto: .*Auto-capture is off now/);
    // The phone hasn't moved since the failure, and the on-target line already says to tap.
    expect(overlayStatusText()).toMatch(/take the photo/i);
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("upload still in flight");
    });
    holdAt(clock, 45, 1700, 9000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(1);
    expect(coveredCountText()).toMatch(/^Covered: 1\//);
  });

  it("C-27: an automatic photo that can't reach the server turns auto-capture off instead of retrying", async () => {
    const { mounted, clock } = await startAutoSession();
    const realFetch = globalThis.fetch;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      if (typeof input === "string" && input.includes("/frames")) {
        return Promise.reject(new TypeError("Failed to fetch"));
      }
      return realFetch(input, init);
    });
    holdAt(clock, 45, 1000, 1600);
    await vi.waitFor(() => {
      if (autoCaptureToggle().checked) throw new Error("auto still on");
    });
    expect(mounted.captureStatus.textContent).toMatch(
      /^Auto: Could not upload frame.*Auto-capture is off now/,
    );
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("upload still in flight");
    });
    holdAt(clock, 45, 1700, 9000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(1);
  });

  // #420 review, round 2 (@dalwalyk): when the session is gone, the message must not offer a tap
  // or auto-capture, since neither can help.
  it("C-27: an automatic photo on a session that's gone turns auto-capture off without offering a retry", async () => {
    const { mounted, clock } = await startAutoSession();
    const realFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      if (typeof input === "string" && input.includes("/frames")) {
        return Promise.resolve(
          new Response(JSON.stringify({ code: "session_not_found", message: "Not Found" }), {
            status: 422,
            headers: { "content-type": "application/json" },
          }),
        );
      }
      return realFetch(input, init);
    });
    holdAt(clock, 45, 1000, 1600);
    await vi.waitFor(() => {
      if (autoCaptureToggle().checked) throw new Error("auto still on");
    });
    expect(mounted.captureStatus.textContent).toMatch(
      /start a new session\. Auto-capture is off now\.$/,
    );
    expect(mounted.captureStatus.textContent).not.toMatch(/tap to retry/);
  });

  // #420 review, round 2 (@dalwalyk): a tapped photo that failed to upload was retried once by
  // auto-capture 1.5 s later before the switch went off.
  it("C-27: a tapped photo that fails to upload turns auto-capture off too", async () => {
    const { mounted, clock } = await startAutoSession();
    const realFetch = globalThis.fetch;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      if (typeof input === "string" && input.includes("/frames")) {
        return Promise.reject(new TypeError("Failed to fetch"));
      }
      return realFetch(input, init);
    });
    holdAt(clock, 45, 1000, 1100);
    mounted.captureButton.click();
    await vi.waitFor(() => {
      if (autoCaptureToggle().checked) throw new Error("auto still on");
    });
    expect(mounted.captureStatus.textContent).toMatch(
      /^Could not upload frame.*Auto-capture is off now/,
    );
    await vi.waitFor(() => {
      if (mounted.captureButton.disabled) throw new Error("upload still in flight");
    });
    holdAt(clock, 45, 1200, 9000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(frameUploads(fetchSpy)).toBe(1);
  });

  // #420 review (@dalwalyk): a tap the gate rejected could be followed ~16 ms later by an
  // automatic photo, overwriting the person's own rejection message.
  it("C-27: a tap restarts auto-capture's hold and retry gap", async () => {
    const { mounted, clock, canvas } = await startAutoSession();
    const flat = sharpWellExposedImageData();
    flat.data.fill(128);
    canvas.setImageData(flat);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    holdAt(clock, 45, 1000, 1500);
    mounted.captureButton.click();
    expect(mounted.captureStatus.textContent).toMatch(/^Frame rejected/);
    // Past the 0.6 s hold that started at 1000 ms, but within 1.5 s of the tap.
    holdAt(clock, 45, 1600, 2900);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(mounted.captureStatus.textContent).toMatch(/^Frame rejected/);
    expect(frameUploads(fetchSpy)).toBe(0);
  });

  // #420 review, round 2 (@dalwalyk): the case the fix is for -- a reading before the grant, then
  // the grant. Capture waits for a reading that arrives after it.
  it("C-27: a reading before the gyro grant doesn't count once permission is granted", async () => {
    stubGrantedCamera();
    stubGyroPermission("granted");
    const { state, orderCode, button, cameraStatus, gyroStatus, captureButton } = mount();
    dispatchOrientation(0, 45, 0);
    orderCode.value = "PO-48213-A";
    button.click();
    await vi.waitFor(() => {
      if (cameraStatus.textContent !== "Camera started.") throw new Error("still pending");
    });
    await vi.waitFor(() => {
      if (gyroStatus.textContent === "") throw new Error("gyro still pending");
    });
    expect(state.latestGyro).toBeNull();
    expect(captureButton.disabled).toBe(true);
    dispatchOrientation(10, 45, 0);
    expect(state.latestGyro).not.toBeNull();
    expect(captureButton.disabled).toBe(false);
  });

  // #420 review (@dalwalyk): the overlay listens from page load, so a reading before the motion
  // permission was granted used to count as the shell's gyro reading.
  it("C-27: a reading before the gyro is granted doesn't enable capture", async () => {
    stubGrantedCamera();
    stubGyroPermission("denied");
    const { state, orderCode, button, cameraStatus, captureButton } = mount();
    dispatchOrientation(0, 45, 0);
    orderCode.value = "PO-48213-A";
    button.click();
    await vi.waitFor(() => {
      if (cameraStatus.textContent !== "Camera started.") throw new Error("still pending");
    });
    dispatchOrientation(10, 45, 0);
    expect(state.latestGyro).toBeNull();
    expect(captureButton.disabled).toBe(true);
  });

  it("C-27: an automatic frame the gate rejects says so, uploads nothing and leaves coverage alone", async () => {
    const { mounted, clock, canvas } = await startAutoSession();
    const flat = sharpWellExposedImageData();
    flat.data.fill(128);
    canvas.setImageData(flat);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    holdAt(clock, 45, 1000, 1600);
    await vi.waitFor(() => {
      if (!/^Auto: Frame rejected/.test(mounted.captureStatus.textContent)) {
        throw new Error("no automatic rejection yet");
      }
    });
    expect(mounted.captureStatus.textContent).toMatch(/blur/i);
    expect(frameUploads(fetchSpy)).toBe(0);
    expect(coveredCountText()).toMatch(/^Covered: 1\//);
  });

  it("C-26: Finish session is the large primary button and keeps its id and name", () => {
    const mounted = mount();
    expect(mounted.finishButton.classList.contains("primary-action")).toBe(true);
    expect(mounted.finishButton.textContent.trim()).toBe("Finish session");
    expect(mounted.finishButton.disabled).toBe(true);
  });
});
