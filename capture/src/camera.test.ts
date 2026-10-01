// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { describeCameraError, isGetUserMediaSupported, mountCameraCheck } from "./camera.js";

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

  it("shows a specific message and sets the preview on grant", async () => {
    const { button, status, root } = mount();
    const fakeStream = new MediaStream();
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };

    button.click();
    await vi.waitFor(() => {
      if (status.textContent === "") throw new Error("still empty");
    });

    expect(status.textContent).toBe("Camera started.");
    const preview = root.querySelector("#camera-preview");
    expect(preview).toBeInstanceOf(HTMLVideoElement);
    expect((preview as HTMLVideoElement).srcObject).toBe(fakeStream);
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
  });

  it("shows the unsupported message instead of throwing when mediaDevices is absent", () => {
    const { button, status } = mount();

    expect(() => {
      button.click();
    }).not.toThrow();
    expect(status.textContent).toContain("needs HTTPS");
  });
});
