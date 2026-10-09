// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountUploadPage } from "./page.js";
import { startContractStub, type ContractStub } from "../stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "../stub/page-test-port.js";

const FAKE_JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46]);

describe("mountUploadPage", () => {
  let close: () => Promise<void>;
  let baseUrl: string;
  let stub: ContractStub;

  beforeEach(async () => {
    // Fixed port, matching vite.config.ts's environmentOptions.happyDOM.url — happy-dom enforces
    // the same-origin policy, so an ephemeral port here would make every fetch cross-origin and
    // silently blocked.
    const started = await startContractStub(PAGE_TEST_STUB_PORT);
    baseUrl = started.baseUrl;
    close = started.close;
    stub = started.stub;

    document.body.innerHTML = '<main id="app"></main>';
    const root = document.getElementById("app");
    if (!root) throw new Error("test setup: #app missing");
    mountUploadPage(root, {
      baseUrl,
      reconstructionPollIntervalMs: 0,
      reconstructionMaxAttempts: 3,
    });
  });

  afterEach(async () => {
    await close();
  });

  function text(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) throw new Error(`test: #${id} missing`);
    return element;
  }

  function input(id: string): HTMLInputElement {
    const element = text(id);
    if (!(element instanceof HTMLInputElement)) {
      throw new Error(`test: #${id} is not an <input>`);
    }
    return element;
  }

  function button(id: string): HTMLButtonElement {
    const element = text(id);
    if (!(element instanceof HTMLButtonElement)) {
      throw new Error(`test: #${id} is not a <button>`);
    }
    return element;
  }

  // The click handlers fire a real fetch over a real socket (unlike api.test.ts's in-process
  // stub calls), so completion is scheduled via actual I/O, not a fixed number of microtask
  // ticks — poll for the result instead of guessing how long that takes.
  async function waitForNonEmptyText(id: string): Promise<string> {
    await vi.waitFor(() => {
      if (text(id).textContent === "") {
        throw new Error(`#${id} is still empty`);
      }
    });
    return text(id).textContent;
  }



  async function waitForText(id: string, pattern: RegExp): Promise<string> {
    await vi.waitFor(() => {
      const value = text(id).textContent;
      if (!pattern.test(value)) {
        throw new Error(`#${id} did not match ${String(pattern)}: ${value}`);
      }
    });
    return text(id).textContent;
  }

  async function startAndUploadOneFrame(): Promise<void> {
    input("order-code").value = "PO-48213-A";
    button("start-button").click();
    await waitForNonEmptyText("start-result");

    const file = new File([FAKE_JPEG_BYTES], "frame.jpg", { type: "image/jpeg" });
    const fileInput = input("frame-file");
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    input("gyro-alpha").value = "1";
    input("gyro-beta").value = "2";
    input("gyro-gamma").value = "3";
    button("upload-button").click();
    await waitForNonEmptyText("upload-result");
  }

  it("walks the golden path: start a session, upload a frame, finish", async () => {
    input("order-code").value = "PO-48213-A";
    button("start-button").click();
    expect(await waitForNonEmptyText("start-result")).toMatch(/^Session started: sess_/);
    expect(button("upload-button").disabled).toBe(false);
    expect(button("finish-button").disabled).toBe(false);

    const file = new File([FAKE_JPEG_BYTES], "frame.jpg", { type: "image/jpeg" });
    const fileInput = input("frame-file");
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    input("gyro-alpha").value = "1";
    input("gyro-beta").value = "2";
    input("gyro-gamma").value = "3";
    button("upload-button").click();
    expect(await waitForNonEmptyText("upload-result")).toBe("Frame accepted (frame 0)");

    button("finish-button").click();
    expect(await waitForNonEmptyText("finish-result")).toMatch(/^Session finished: sess_/);
    const reconstruction = await waitForText("reconstruction-result", /Reconstruction complete/);
    expect(reconstruction).toContain("Reference tier: charuco_board");
    expect(reconstruction).toContain("Metric: yes");
    expect(reconstruction).toContain("Observed fraction: 0.910");
    expect(reconstruction).toContain("Warnings: none");
  });



  it("states that no dimensional claims are made when the reconstruction is non-metric", async () => {
    await startAndUploadOneFrame();
    stub.useNoReferenceReconstruction();

    button("finish-button").click();
    await waitForNonEmptyText("finish-result");
    const reconstruction = await waitForText("reconstruction-result", /Reconstruction complete/);
    expect(reconstruction).toContain("Reference tier: none");
    expect(reconstruction).toContain("Metric: no");
    expect(reconstruction).toContain("No dimensional claims");
  });

  it("shows a failed reconstruction instead of staying pending", async () => {
    await startAndUploadOneFrame();
    stub.failReconstruction();

    button("finish-button").click();
    await waitForNonEmptyText("finish-result");
    const reconstruction = await waitForText("reconstruction-result", /Reconstruction failed/);
    expect(reconstruction).toContain("Stub reconstruction failure");
  });

  it("times out after the bounded reconstruction polling budget", async () => {
    await startAndUploadOneFrame();
    stub.setReconstructionPendingPolls(99);

    button("finish-button").click();
    await waitForNonEmptyText("finish-result");
    expect(await waitForText("reconstruction-result", /timed out after 3 attempts/)).toContain(
      "3 attempts",
    );
  });

  it("shows the contract's 400 message for an unknown order code", async () => {
    input("order-code").value = "PO-UNKNOWN";
    button("start-button").click();
    expect(await waitForNonEmptyText("start-result")).toContain("not recognised");
    expect(button("upload-button").disabled).toBe(true);
  });

  it("shows a local error instead of calling the API when the gyro input is bad", async () => {
    input("order-code").value = "PO-48213-A";
    button("start-button").click();
    await waitForNonEmptyText("start-result");

    const file = new File([FAKE_JPEG_BYTES], "frame.jpg", { type: "image/jpeg" });
    const fileInput = input("frame-file");
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    input("gyro-alpha").value = "not a number";
    input("gyro-beta").value = "2";
    input("gyro-gamma").value = "3";
    button("upload-button").click();
    expect(await waitForNonEmptyText("upload-result")).toContain('"alpha"');
  });

  it("shows the contract's 409 message when finishing before any accepted frame", async () => {
    input("order-code").value = "PO-48213-A";
    button("start-button").click();
    await waitForNonEmptyText("start-result");

    button("finish-button").click();
    expect(await waitForNonEmptyText("finish-result")).toContain("Not enough frames");
  });
});
