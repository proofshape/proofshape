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
    <video id="camera-preview" muted playsinline></video>
    <p id="camera-status"></p>
  </section>
`;

export function mountCameraCheck(root: HTMLElement): void {
  root.innerHTML = CAMERA_PAGE_HTML;

  const startButton = requireButton(root, "#camera-start-button");
  const preview = requireVideo(root, "#camera-preview");
  const status = requireElement(root, "#camera-status");

  startButton.addEventListener("click", () => {
    if (!isGetUserMediaSupported()) {
      status.textContent =
        "Camera not available — this needs HTTPS (or localhost) and a supported browser.";
      return;
    }
    // The synchronous call itself: no `await` before this line, and the result is handled with
    // .then/.catch rather than `await`, so nothing delays it past the end of this click handler.
    navigator.mediaDevices.getUserMedia(REAR_CAMERA_CONSTRAINTS).then(
      (stream) => {
        preview.srcObject = stream;
        status.textContent = "Camera started.";
      },
      (error: unknown) => {
        status.textContent = describeCameraError(error);
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
