// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from "vitest";
import { mountAgentConversation } from "./agent-conversation.js";

describe("mountAgentConversation", () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="agent"></div>';
  });

  function mount() {
    const root = document.getElementById("agent");
    if (!(root instanceof HTMLElement)) throw new Error("test root missing");
    return { root, handle: mountAgentConversation(root) };
  }

  it("stays hidden until a real agent result is shown", () => {
    const { root, handle } = mount();
    expect(root.hidden).toBe(true);

    handle.transportUnavailable();
    expect(root.hidden).toBe(true);

    handle.show({
      finished: false,
      fallback_reason: null,
      message: "Frame accepted. Move slightly to your right.",
    });
    expect(root.hidden).toBe(false);
  });

  it("renders representative agent messages in order without duplicating the last line", () => {
    const { root, handle } = mount();
    const first = {
      finished: false,
      fallback_reason: null,
      message: "Frame accepted. Move slightly to your right.",
    } as const;
    handle.show(first);
    handle.show(first);
    handle.show({
      finished: false,
      fallback_reason: null,
      message: "Good coverage. Capture one higher view.",
    });

    const messages = [...root.querySelectorAll("li")].map((node) => node.textContent);
    expect(messages).toEqual([
      "Frame accepted. Move slightly to your right.",
      "Good coverage. Capture one higher view.",
    ]);
    expect(root.textContent).toContain("Assistant guidance updated.");
  });

  it("shows A-07 fallback narration without exposing raw fallback codes", () => {
    const { root, handle } = mount();
    handle.show({
      finished: false,
      fallback_reason: "timeout",
      message: "Keep going -- more of the part still needs to be photographed.",
    });

    const item = root.querySelector("li");
    const status = root.querySelector("#agent-conversation-status");
    expect(item?.dataset.agentMode).toBe("fallback");
    expect(item?.textContent).toMatch(/more of the part still needs to be photographed/i);
    expect(root.textContent).not.toMatch(/timeout/i);
    expect(status?.getAttribute("data-agent-fallback-reason")).toBe("timeout");
  });

  it("shows transport failure only in the status line once the panel has appeared", () => {
    const { root, handle } = mount();
    handle.show({
      finished: false,
      fallback_reason: null,
      message: "Try another angle.",
    });
    handle.transportUnavailable();

    expect(root.querySelectorAll("li")).toHaveLength(1);
    expect(root.textContent).toMatch(/temporarily unavailable/i);
    expect(root.textContent).toMatch(/continue following the capture arrows/i);
  });

  it("clear removes prior conversation state and hides the panel again", () => {
    const { root, handle } = mount();
    handle.show({
      finished: false,
      fallback_reason: null,
      message: "Try another angle.",
    });
    handle.clear();
    expect(root.querySelectorAll("li")).toHaveLength(0);
    expect(root.hidden).toBe(true);
    expect(root.textContent).toMatch(/no assistant message yet/i);
  });
});
