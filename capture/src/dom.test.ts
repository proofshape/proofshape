// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";
import { requireElement, requireTyped } from "./dom.js";

function mount(html: string): HTMLElement {
  const root = document.createElement("div");
  root.innerHTML = html;
  return root;
}

describe("requireElement", () => {
  it("returns the matching element", () => {
    const root = mount('<p id="target">hi</p>');
    expect(requireElement(root, "#target").textContent).toBe("hi");
  });

  it("throws when nothing matches", () => {
    const root = mount("<p>hi</p>");
    expect(() => requireElement(root, "#missing")).toThrow(/"#missing"/);
  });
});

describe("requireTyped", () => {
  it("returns the element when it matches the constructor", () => {
    const root = mount('<button id="go"></button>');
    const button = requireTyped(root, "#go", HTMLButtonElement, "<button>");
    expect(button).toBeInstanceOf(HTMLButtonElement);
  });

  it("throws, naming the expected type, when the element is the wrong type", () => {
    const root = mount('<div id="go"></div>');
    expect(() => requireTyped(root, "#go", HTMLButtonElement, "<button>")).toThrow(/<button>/);
  });

  it("throws when nothing matches the selector at all", () => {
    const root = mount("<div></div>");
    expect(() => requireTyped(root, "#missing", HTMLButtonElement, "<button>")).toThrow(
      /"#missing"/,
    );
  });
});
