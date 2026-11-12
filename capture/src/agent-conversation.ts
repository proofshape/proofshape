import { requireElement, requireTyped } from "./dom.js";
import type { AgentMessageResult } from "./types.js";

export interface AgentConversationHandle {
  show: (result: AgentMessageResult) => void;
  clear: () => void;
}

export function mountAgentConversation(container: HTMLElement): AgentConversationHandle {
  container.innerHTML = `
    <section aria-labelledby="agent-conversation-title">
      <h2 id="agent-conversation-title">Capture assistant</h2>
      <p id="agent-conversation-status">No assistant message yet.</p>
      <ol id="agent-conversation-messages"></ol>
    </section>
  `;

  const status = requireElement(container, "#agent-conversation-status");
  const list = requireTyped(
    container,
    "#agent-conversation-messages",
    HTMLOListElement,
    "<ol>",
  );

  function clear(): void {
    list.replaceChildren();
    status.textContent = "No assistant message yet.";
  }

  function show(result: AgentMessageResult): void {
    const item = document.createElement("li");
    if (result.fallback) {
      item.textContent =
        result.message?.trim() ||
        "Automatic guidance is active. Continue following the capture arrows.";
      item.dataset.agentMode = "fallback";
      status.textContent = result.fallback_reason
        ? `Fallback guidance active: ${result.fallback_reason}.`
        : "Fallback guidance active.";
    } else if (result.message?.trim()) {
      item.textContent = result.message.trim();
      item.dataset.agentMode = "agent";
      status.textContent = "Assistant guidance updated.";
    } else {
      status.textContent = "No assistant message to show.";
      return;
    }
    list.append(item);
  }

  return { show, clear };
}
