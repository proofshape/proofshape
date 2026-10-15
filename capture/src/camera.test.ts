// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  capturePixelsFromVideo,
  captureFrame,
  describeCameraError,
  encodeFrameAsJpeg,
  isGetUserMediaSupported,
  mountCameraCheck,
  requestCameraPermission,
  stopCameraStream,
} from "./camera.js";
import type { FramePixels } from "./frame_gate.js";
import type { Gyro } from "./types.js";

describe("describeCameraError", () => {
  it("names the specific reason for each known denial", () => {
    expect(describeCameraError(new DOMException("", "NotAllowedError"))).toContain(
      "permission denied",
    );
    expect(describeCameraError(new DOMException("", "NotFoundError"))).toContain("No camera found");
    expect(describeCameraError(new DOMException("", "NotReadableError"))).toContain(
      "already in use",
    );
    expect(describeCameraError(new DOMException("", "OverconstrainedError"))).toContain(
      "rear camera isn't available",
    );
  });

  it("falls back to a generic-but-specific message for anything else", () => {
    expect(describeCameraError(new Error("disk on fire"))).toBe(
      "Could not start the camera: disk on fire",
    );
    expect(describeCameraError("not even an Error")).toContain("not even an Error");
  });
});

describe("isGetUserMediaSupported", () => {
  afterEach(() => {
    // @ts-expect-error -- test-only cleanup of a property this file itself may have set.
    delete navigator.mediaDevices;
  });

  it("is false when navigator.mediaDevices is absent (the plain-HTTP-over-LAN case)", () => {
    expect(isGetUserMediaSupported()).toBe(false);
  });

  it("is true when getUserMedia exists", () => {
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve() };
    expect(isGetUserMediaSupported()).toBe(true);
  });
});

describe("mountCameraCheck", () => {
  afterEach(() => {
    // @ts-expect-error -- test-only cleanup.
    delete navigator.mediaDevices;
  });

  function mount(): { root: HTMLElement; button: HTMLButtonElement; status: HTMLElement } {
    document.body.innerHTML = '<main id="camera-app"></main>';
    const root = document.getElementById("camera-app");
    if (!root) throw new Error("test setup: #camera-app missing");
    mountCameraCheck(root);
    const button = document.getElementById("camera-start-button");
    const status = document.getElementById("camera-status");
    if (!(button instanceof HTMLButtonElement) || !status) {
      throw new Error("test setup: mounted elements missing");
    }
    return { root, button, status };
  }

  it("renders a start button, a preview video and a status element", () => {
    document.body.innerHTML = '<main id="camera-app"></main>';
    const root = document.getElementById("camera-app");
    if (!root) throw new Error("test setup: #camera-app missing");
    mountCameraCheck(root);
    expect(root.querySelector("#camera-start-button")).toBeInstanceOf(HTMLButtonElement);
    expect(root.querySelector("#camera-preview")).toBeInstanceOf(HTMLVideoElement);
    expect(root.querySelector("#camera-status")).not.toBeNull();
  });

  it("calls getUserMedia synchronously inside the click handler -- the regression this story exists to prevent", () => {
    const { button } = mount();
    let calledSynchronously = false;
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => {
        calledSynchronously = true;
        return new Promise(() => {
          /* never resolves -- only the call timing matters for this test */
        });
      },
    };

    button.click();

    // No await anywhere above this line: if the real handler had an `await` (or any deferral)
    // before calling getUserMedia, `calledSynchronously` would still be false right here.
    expect(calledSynchronously).toBe(true);
  });

  it("shows a specific message, sets the preview, and actually starts playback on grant", async () => {
    const { button, status, root } = mount();
    const fakeStream = new MediaStream();
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };
    const preview = root.querySelector("#camera-preview") as HTMLVideoElement;
    // On a real device, `autoplay` alone wasn't enough for a stream attached after the element
    // already existed -- the preview stayed blank despite a granted permission (caught by manual
    // testing on a real iPhone, not by this suite, since happy-dom doesn't render video at all).
    // This spy is what actually guards the fix: it fails if `.play()` stops being called.
    const playCalls: number[] = [];
    const originalPlay = preview.play.bind(preview);
    preview.play = () => {
      playCalls.push(1);
      return originalPlay();
    };

    button.click();
    await vi.waitFor(() => {
      if (status.textContent === "") throw new Error("still empty");
    });

    expect(status.textContent).toBe("Camera started.");
    expect(preview.srcObject).toBe(fakeStream);
    expect(playCalls).toHaveLength(1);
    // A successful grant isn't an error -- shouldn't be styled like one.
    expect(status.style.fontWeight).not.toBe("bold");
  });

  it("shows the specific deny message, not a blank screen, on rejection", async () => {
    const { button, status } = mount();
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => Promise.reject(new DOMException("", "NotAllowedError")),
    };

    button.click();
    await vi.waitFor(() => {
      if (status.textContent === "") throw new Error("still empty");
    });

    expect(status.textContent).toContain("permission denied");
    // "Visible" per this story's own acceptance criteria turned out to mean more than
    // non-empty text -- caught by manual testing, where plain body-colored text was easy to
    // miss. A real error now has to actually look like one.
    expect(status.style.color).not.toBe("");
    expect(status.style.fontWeight).toBe("bold");
  });

  it("shows the unsupported message instead of throwing when mediaDevices is absent", () => {
    const { button, status } = mount();

    expect(() => {
      button.click();
    }).not.toThrow();
    expect(status.textContent).toContain("needs HTTPS");
    expect(status.style.color).not.toBe("");
    expect(status.style.fontWeight).toBe("bold");
  });
});

describe("requestCameraPermission", () => {
  afterEach(() => {
    // @ts-expect-error -- test-only cleanup of a property this file itself may have set.
    delete navigator.mediaDevices;
  });

  it("hands back the live stream on grant, without stopping it -- C-05's whole point", async () => {
    const fakeStream = new MediaStream();
    const stopSpy = vi.fn();
    fakeStream.getTracks = () => [{ stop: stopSpy } as unknown as MediaStreamTrack];
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };

    const result = await requestCameraPermission();

    expect(result).toEqual({ granted: true, stream: fakeStream });
    expect(stopSpy).not.toHaveBeenCalled();
  });

  it("still reports a specific denial message", async () => {
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = {
      getUserMedia: () => Promise.reject(new DOMException("", "NotAllowedError")),
    };

    const result = await requestCameraPermission();

    expect(result.granted).toBe(false);
    expect(!result.granted && result.message).toContain("permission denied");
  });
});

describe("capturePixelsFromVideo", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function fakeVideo(width: number, height: number): HTMLVideoElement {
    const video = document.createElement("video");
    // happy-dom's real videoWidth/videoHeight are always 0 (no decode pipeline) -- overridden to
    // stand in for a real playing video's reported frame size.
    Object.defineProperty(video, "videoWidth", { value: width, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: height, configurable: true });
    return video;
  }

  it("draws the video into a canvas sized to its dimensions and reads the pixels back", () => {
    const drawImageSpy = vi.fn();
    const imageData = { data: new Uint8ClampedArray([9, 8, 7, 6]), width: 1, height: 1 };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: drawImageSpy,
      getImageData: () => imageData,
    } as unknown as CanvasRenderingContext2D);
    const video = fakeVideo(640, 480);

    const pixels = capturePixelsFromVideo(video);

    expect(drawImageSpy).toHaveBeenCalledWith(video, 0, 0, 640, 480);
    expect(pixels).toBe(imageData);
  });

  it("throws rather than capturing a garbage frame when the video has no dimensions yet", () => {
    const video = fakeVideo(0, 0);

    expect(() => {
      capturePixelsFromVideo(video);
    }).toThrow(/no dimensions yet/);
  });

  it("reuses one canvas context across repeated captures of the same video -- not a fresh one per call", () => {
    // TabeenRaoof's PR #137 review: a capture loop calls this on every attempt, not just accepted
    // ones, and a fresh full-resolution canvas+context per call put real memory pressure on iOS
    // Safari's strict canvas limits.
    const getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: () => undefined,
      getImageData: () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }),
    } as unknown as CanvasRenderingContext2D);
    const video = fakeVideo(640, 480);

    capturePixelsFromVideo(video);
    capturePixelsFromVideo(video);
    capturePixelsFromVideo(video);

    expect(getContextSpy).toHaveBeenCalledTimes(1);
  });

  it("gives two different video elements their own canvas, not a shared one", () => {
    const getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: () => undefined,
      getImageData: () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }),
    } as unknown as CanvasRenderingContext2D);
    const videoA = fakeVideo(640, 480);
    const videoB = fakeVideo(320, 240);

    capturePixelsFromVideo(videoA);
    capturePixelsFromVideo(videoB);

    expect(getContextSpy).toHaveBeenCalledTimes(2);
  });

  it("resizes the cached canvas when the video's dimensions change, instead of keeping a stale size", () => {
    const drawImageSpy = vi.fn();
    const getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: drawImageSpy,
      getImageData: () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }),
    } as unknown as CanvasRenderingContext2D);
    const video = fakeVideo(640, 480);

    capturePixelsFromVideo(video);
    Object.defineProperty(video, "videoWidth", { value: 1280, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: 720, configurable: true });
    capturePixelsFromVideo(video);

    // getContext called only once total (the same cached canvas handled both calls, the cache
    // key is the video element, not the dimensions), but drawImage's second call reflects the
    // new, larger dimensions -- proving the cached canvas was resized, not left stale at 640x480.
    expect(getContextSpy).toHaveBeenCalledTimes(1);
    expect(drawImageSpy).toHaveBeenNthCalledWith(1, video, 0, 0, 640, 480);
    expect(drawImageSpy).toHaveBeenNthCalledWith(2, video, 0, 0, 1280, 720);
  });
});

describe("captureFrame", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("pairs the captured pixels with exactly the gyro value passed in, each call independently", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: () => undefined,
      getImageData: () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 }),
    } as unknown as CanvasRenderingContext2D);
    const video = document.createElement("video");
    Object.defineProperty(video, "videoWidth", { value: 10, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: 10, configurable: true });
    const gyroA: Gyro = { alpha: 1, beta: 2, gamma: 3 };
    const gyroB: Gyro = { alpha: 40, beta: 50, gamma: 60 };

    const resultA = captureFrame(video, gyroA);
    const resultB = captureFrame(video, gyroB);

    expect(resultA.gyro).toEqual(gyroA);
    expect(resultB.gyro).toEqual(gyroB);
  });
});

describe("stopCameraStream", () => {
  it("stops every track on the stream", () => {
    const stopA = vi.fn();
    const stopB = vi.fn();
    const fakeStream = {
      getTracks: () => [{ stop: stopA }, { stop: stopB }],
    } as unknown as MediaStream;

    stopCameraStream(fakeStream);

    expect(stopA).toHaveBeenCalledTimes(1);
    expect(stopB).toHaveBeenCalledTimes(1);
  });

  it("does not throw on a stream with no tracks", () => {
    const fakeStream = { getTracks: () => [] } as unknown as MediaStream;

    expect(() => {
      stopCameraStream(fakeStream);
    }).not.toThrow();
  });
});

describe("encodeFrameAsJpeg", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const pixels: FramePixels = { data: new Uint8ClampedArray([1, 2, 3, 4]), width: 1, height: 1 };

  function stubCanvasToBlob(blob: Blob | null): { putImageDataSpy: ReturnType<typeof vi.fn> } {
    const putImageDataSpy = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      putImageData: putImageDataSpy,
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (
      this: HTMLCanvasElement,
      callback: BlobCallback,
    ) {
      callback(blob);
    });
    return { putImageDataSpy };
  }

  it("resolves with the blob toBlob's callback produces, from pixels written via putImageData", async () => {
    const fakeBlob = new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" });
    const { putImageDataSpy } = stubCanvasToBlob(fakeBlob);

    const result = await encodeFrameAsJpeg(pixels);

    expect(result).toBe(fakeBlob);
    expect(putImageDataSpy).toHaveBeenCalledTimes(1);
    const writtenImageData = putImageDataSpy.mock.calls[0]?.[0] as ImageData;
    expect(Array.from(writtenImageData.data)).toEqual([1, 2, 3, 4]);
  });

  it("rejects, rather than resolving undefined, when canvas.toBlob produces no blob", async () => {
    stubCanvasToBlob(null);

    await expect(encodeFrameAsJpeg(pixels)).rejects.toThrow(/no blob/);
  });

  it("encodes as image/jpeg at the given quality, or the default when omitted", async () => {
    const toBlobSpy = vi.fn((callback: BlobCallback) => {
      callback(new Blob());
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      putImageData: () => undefined,
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(toBlobSpy);

    await encodeFrameAsJpeg(pixels);
    await encodeFrameAsJpeg(pixels, 0.5);

    expect(toBlobSpy).toHaveBeenNthCalledWith(1, expect.any(Function), "image/jpeg", 0.92);
    expect(toBlobSpy).toHaveBeenNthCalledWith(2, expect.any(Function), "image/jpeg", 0.5);
  });
});
