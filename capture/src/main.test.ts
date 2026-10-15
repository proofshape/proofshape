// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  mountModelViewer: vi.fn(() => () => undefined),
  mountAppShell: vi.fn(() => ({})),
}));

vi.mock("./model-viewer.js", () => ({
  mountModelViewer: mocks.mountModelViewer,
}));

vi.mock("./shell.js", () => ({
  mountAppShell: mocks.mountAppShell,
}));

async function loadMain(path: string): Promise<HTMLElement> {
  document.body.innerHTML = '<main id="shell-app"></main>';
  window.history.replaceState({}, "", path);
  await import("./main.js");
  const root = document.getElementById("shell-app");
  if (!root) {
    throw new Error("test setup: #shell-app missing");
  }
  return root;
}

describe("main entrypoint", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    window.history.replaceState({}, "", "/");
  });

  it("mounts the direct C-11 viewer demo only when viewer-demo=1 is present", async () => {
    const root = await loadMain("/?viewer-demo=1");

    expect(mocks.mountModelViewer).toHaveBeenCalledWith(root, {
      status: "complete",
      glb_url: "/v1/sessions/c11-device-check/model.glb",
      metric: false,
    });
    expect(mocks.mountAppShell).not.toHaveBeenCalled();
  });

  it("mounts the normal app shell when viewer-demo is absent", async () => {
    const root = await loadMain("/");

    expect(mocks.mountAppShell).toHaveBeenCalledWith(root, { baseUrl: "" });
    expect(mocks.mountModelViewer).not.toHaveBeenCalled();
  });
});
