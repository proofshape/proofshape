import { requireElement, requireTyped } from "./dom.js";
import type { AgentMessageResult } from "./types.js";

export interface AgentConversationHandle {
  show: (result: AgentMessageResult) => void;
  transportUnavailable: () => void;
  clear: () => void;
}

export function mountAgentConversation(container: HTMLElement): AgentConversationHandle {
  container.hidden = true;
  container.innerHTML = `
    <section aria-labelledby="agent-conversation-title">
      <h2 id="agent-conversation-title">Capture assistant</h2>
      <p id="agent-conversation-status" aria-live="polite">No assistant message yet.</p>
      <ol id="agent-conversation-messages" role="log"></ol>
    </section>
  `;

  const status = requireElement(container, "#agent-conversation-status");
  const list = requireTyped(container, "#agent-conversation-messages", HTMLOListElement, "<ol>");

  function clear(): void {
    container.hidden = true;
    list.replaceChildren();
    status.textContent = "No assistant message yet.";
    delete status.dataset.agentFallbackReason;
  }

  function transportUnavailable(): void {
    if (container.hidden) {
      return;
    }
    status.textContent =
      "Capture assistant is temporarily unavailable. Continue following the capture arrows.";
    delete status.dataset.agentFallbackReason;
  }

  function show(result: AgentMessageResult): void {
    container.hidden = false;

    const fallbackReason = result.fallback_reason;
    if (fallbackReason !== null) {
      status.dataset.agentFallbackReason = fallbackReason;
      status.textContent =
        "Automatic guidance is active. Continue following the capture arrows.";
    } else {
      delete status.dataset.agentFallbackReason;
      status.textContent = result.finished
        ? "Capture assistant says the session is finished."
        : "Assistant guidance updated.";
    }

    const message = result.message?.trim();
    if (!message) {
      return;
    }

    const last = list.lastElementChild;
    if (last?.textContent === message) {
      return;
    }

    const item = document.createElement("li");
    item.textContent = message;
    item.dataset.agentMode = fallbackReason === null ? "agent" : "fallback";
    list.append(item);
  }

  return { show, transportUnavailable, clear };
}
