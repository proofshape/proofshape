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

  it("renders representative agent messages in order", () => {
    const { root, handle } = mount();
    handle.show({ message: "Frame accepted. Move slightly to your right.", fallback: false });
    handle.show({ message: "Good coverage. Capture one higher view.", fallback: false });

    const messages = [...root.querySelectorAll("li")].map((node) => node.textContent);
    expect(messages).toEqual([
      "Frame accepted. Move slightly to your right.",
      "Good coverage. Capture one higher view.",
    ]);
    expect(root.textContent).toContain("Assistant guidance updated.");
  });

  it("degrades to deterministic fallback guidance when no model message is available", () => {
    const { root, handle } = mount();
    handle.show({ message: null, fallback: true, fallback_reason: "model_timeout" });

    const item = root.querySelector("li");
    expect(item?.dataset.agentMode).toBe("fallback");
    expect(item?.textContent).toMatch(/automatic guidance is active/i);
    expect(root.textContent).toMatch(/model_timeout/i);
  });

  it("clear removes prior conversation state", () => {
    const { root, handle } = mount();
    handle.show({ message: "Try another angle.", fallback: false });
    handle.clear();
    expect(root.querySelectorAll("li")).toHaveLength(0);
    expect(root.textContent).toMatch(/no assistant message yet/i);
  });
});
