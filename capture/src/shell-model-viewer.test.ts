// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { uploadFrame } from "./api.js";
import { mountAppShell, type AppState } from "./shell.js";
import { startContractStub, type ContractStub } from "../stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "../stub/page-test-port.js";

describe("C-11 model viewer shell wiring", () => {
  let close: () => Promise<void>;
  let baseUrl: string;
  let stub: ContractStub;

  beforeEach(async () => {
    const started = await startContractStub(PAGE_TEST_STUB_PORT);
    close = started.close;
    baseUrl = started.baseUrl;
    stub = started.stub;
  });

  afterEach(async () => {
    await close();
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    delete navigator.mediaDevices;
    vi.unstubAllGlobals();
  });

  function grantPermissions(): void {
    const fakeStream = {
      getTracks: () => [{ stop: () => undefined }],
    };
    // @ts-expect-error -- happy-dom has no navigator.mediaDevices of its own to extend.
    navigator.mediaDevices = { getUserMedia: () => Promise.resolve(fakeStream) };
    vi.stubGlobal("DeviceOrientationEvent", {
      requestPermission: () => Promise.resolve("granted"),
    });
  }

  function mount(): {
    state: AppState;
    orderCode: HTMLInputElement;
    startButton: HTMLButtonElement;
    finishButton: HTMLButtonElement;
    reconstructionStatus: HTMLElement;
    modelViewer: HTMLElement;
    retryButton: HTMLButtonElement;
  } {
    document.body.innerHTML = '<main id="shell-app"></main>';
    const root = document.getElementById("shell-app");
    if (!root) throw new Error("test setup: #shell-app missing");

    const state = mountAppShell(root, {
      baseUrl,
      reconstructionPollIntervalMs: 0,
      reconstructionMaxAttempts: 2,
      modelViewerOptions: { loaderFactory: () => ({ load: () => undefined }) },
    });

    const orderCode = document.getElementById("shell-order-code");
    const startButton = document.getElementById("shell-start-button");
    const finishButton = document.getElementById("shell-finish-button");
    const reconstructionStatus = document.getElementById("shell-reconstruction-status");
    const modelViewer = document.getElementById("shell-model-viewer");
    const retryButton = document.getElementById("shell-retry-reconstruction-button");

    if (
      !(orderCode instanceof HTMLInputElement) ||
      !(startButton instanceof HTMLButtonElement) ||
      !(finishButton instanceof HTMLButtonElement) ||
      !(retryButton instanceof HTMLButtonElement) ||
      !reconstructionStatus ||
      !modelViewer
    ) {
      throw new Error("test setup: mounted elements missing");
    }

    return {
      state,
      orderCode,
      startButton,
      finishButton,
      reconstructionStatus,
      modelViewer,
      retryButton,
    };
  }

  async function prepareSession(): Promise<ReturnType<typeof mount>> {
    grantPermissions();
    const mounted = mount();
    mounted.orderCode.value = "PO-48213-A";
    mounted.startButton.click();

    await vi.waitFor(() => {
      if (mounted.state.sessionId === null) {
        throw new Error("session not created yet");
      }
    });

    const sessionId = mounted.state.sessionId;
    if (sessionId === null) {
      throw new Error("test setup: session id missing after wait");
    }

    const frame = new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" });
    const uploaded = await uploadFrame(baseUrl, sessionId, frame, {
      alpha: 1,
      beta: 2,
      gamma: 3,
    });
    if (!uploaded.ok) {
      throw new Error("test setup: upload failed");
    }

    return mounted;
  }

  it("remounts the viewer with the completed reconstruction GLB URL", async () => {
    const mounted = await prepareSession();
    stub.setReconstructionPendingPolls(0);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Reconstruction complete")) {
        throw new Error("reconstruction not complete yet");
      }
    });

    expect(mounted.modelViewer.textContent).toContain("Loading 3D model");
    expect(mounted.modelViewer.querySelector("a")?.getAttribute("href")).toMatch(/model\.glb$/);
  });

  it("remounts the viewer with the backend failed state", async () => {
    const mounted = await prepareSession();
    stub.setReconstructionPendingPolls(0);
    stub.failReconstruction();

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Reconstruction failed")) {
        throw new Error("reconstruction failure not rendered yet");
      }
    });

    expect(mounted.modelViewer.textContent).toContain(
      "3D model unavailable because reconstruction failed",
    );
    expect(mounted.modelViewer.textContent).toContain("Stub reconstruction failure");
  });

  it("does not leave the viewer promising a future model after polling times out", async () => {
    const mounted = await prepareSession();
    stub.setReconstructionPendingPolls(99);

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("timed out after 2 attempts")) {
        throw new Error("timeout not rendered yet");
      }
    });

    expect(mounted.modelViewer.textContent).toContain("reconstruction timed out");
    expect(mounted.retryButton.hidden).toBe(false);
  });

  it("does not leave the viewer pending when reconstruction fetch rejects", async () => {
    const mounted = await prepareSession();
    const nativeFetch = globalThis.fetch.bind(globalThis);

    vi.stubGlobal(
      "fetch",
      (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url =
          typeof input === "string"
            ? input
            : input instanceof URL
              ? input.href
              : input.url;
        if (url.includes("/reconstruction")) {
          return Promise.reject(new TypeError("Failed to fetch"));
        }
        return nativeFetch(input, init);
      },
    );

    mounted.finishButton.click();
    await vi.waitFor(() => {
      if (!mounted.reconstructionStatus.textContent.includes("Could not fetch reconstruction")) {
        throw new Error("fetch failure not rendered yet");
      }
    });

    expect(mounted.modelViewer.textContent).toContain(
      "reconstruction status could not be fetched",
    );
    expect(mounted.retryButton.hidden).toBe(false);
  });
});
