// C-02: proves the camera-permission mechanics work on real iOS Safari, before C-05 builds the
// real capture flow on an unverified assumption. Two iOS-specific gotchas drive this file's
// shape: getUserMedia needs a secure context (HTTPS, or localhost — which doesn't help once an
// iPhone is reaching this page over LAN), and it must be called synchronously inside the click
// handler, or Safari can treat the user gesture as expired before the call happens.

export const REAR_CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: "environment" },
};

export function isGetUserMediaSupported(): boolean {
  // TypeScript's DOM lib types `navigator.mediaDevices` as always present, but in a non-secure
  // context (plain HTTP over LAN -- exactly the case this check exists for) it's genuinely
  // `undefined` at runtime, not just absent `getUserMedia`. Cast to match reality, not the type.
  const mediaDevices = navigator.mediaDevices as MediaDevices | undefined;
  return typeof mediaDevices?.getUserMedia === "function";
}

// Pure: a getUserMedia rejection in, a specific human message out — this is what keeps the deny
// path from being a blank screen or a silent failure.
export function describeCameraError(error: unknown): string {
  const name = error instanceof Error ? error.name : undefined;
  switch (name) {
    case "NotAllowedError":
      return "Camera permission denied. Enable camera access for this site in Settings and try again.";
    case "NotFoundError":
      return "No camera found on this device.";
    case "NotReadableError":
      return "The camera is already in use by another app.";
    case "OverconstrainedError":
      return "The rear camera isn't available with these settings.";
    default: {
      const message = error instanceof Error ? error.message : String(error);
      return `Could not start the camera: ${message}`;
    }
  }
}

export const CAMERA_PAGE_HTML = `
  <section>
    <h2>Camera permission check (C-02)</h2>
    <button id="camera-start-button" type="button">Start camera</button>
    <video id="camera-preview" muted playsinline autoplay style="width: 100%; max-width: 480px;"></video>
    <p id="camera-status"></p>
  </section>
`;

// A denial is "visible" per this story's own acceptance criteria only if a tester would actually
// notice it, not just technically find a non-empty string somewhere in the DOM -- confirmed by
// real on-device testing that plain body-text color blended in enough to almost be missed. Not
// styling choices for their own sake (that's C-04's UI-polish job); this is the minimum needed so
// an error reads as an error.
type StatusKind = "info" | "error";

function setStatus(status: HTMLElement, message: string, kind: StatusKind): void {
  status.textContent = message;
  status.style.color = kind === "error" ? "#b00020" : "";
  status.style.fontWeight = kind === "error" ? "bold" : "";
}

export function mountCameraCheck(root: HTMLElement): void {
  root.innerHTML = CAMERA_PAGE_HTML;

  const startButton = requireButton(root, "#camera-start-button");
  const preview = requireVideo(root, "#camera-preview");
  const status = requireElement(root, "#camera-status");

  startButton.addEventListener("click", () => {
    if (!isGetUserMediaSupported()) {
      setStatus(
        status,
        "Camera not available — this needs HTTPS (or localhost) and a supported browser.",
        "error",
      );
      return;
    }
    // The synchronous call itself: no `await` before this line, and the result is handled with
    // .then/.catch rather than `await`, so nothing delays it past the end of this click handler.
    navigator.mediaDevices.getUserMedia(REAR_CAMERA_CONSTRAINTS).then(
      (stream) => {
        preview.srcObject = stream;
        // `autoplay` alone isn't reliably honored for a stream attached after the element
        // already exists -- call play() explicitly too, so a silently-stalled preview (frozen on
        // the camera permission granted but nothing visible) fails loudly instead.
        preview.play().then(
          () => {
            setStatus(status, "Camera started.", "info");
          },
          (playError: unknown) => {
            setStatus(
              status,
              `Camera permission granted, but the preview failed to play: ${describeCameraError(playError)}`,
              "error",
            );
          },
        );
      },
      (error: unknown) => {
        setStatus(status, describeCameraError(error), "error");
      },
    );
  });
}

function requireElement(root: HTMLElement, selector: string): HTMLElement {
  const element = root.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`mountCameraCheck: expected an element at "${selector}"`);
  }
  return element;
}

function requireButton(root: HTMLElement, selector: string): HTMLButtonElement {
  const element = requireElement(root, selector);
  if (!(element instanceof HTMLButtonElement)) {
    throw new Error(`mountCameraCheck: expected a <button> at "${selector}"`);
  }
  return element;
}

function requireVideo(root: HTMLElement, selector: string): HTMLVideoElement {
  const element = requireElement(root, selector);
  if (!(element instanceof HTMLVideoElement)) {
    throw new Error(`mountCameraCheck: expected a <video> at "${selector}"`);
  }
  return element;
}
