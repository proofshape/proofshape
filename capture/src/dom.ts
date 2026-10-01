// Shared DOM-assertion helpers. page.ts has its own inline copies of this exact pattern
// (pre-dating this file, left untouched deliberately — not this story's concern to retrofit),
// and C-02's camera.ts copied them a second time. Rather than add a third copy here, this is the
// one shared place new capture/ code can pull `requireElement`/`requireTyped` from.

export function requireElement(root: ParentNode, selector: string): HTMLElement {
  const element = root.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`requireElement: expected an element at "${selector}"`);
  }
  return element;
}

export function requireTyped<T extends HTMLElement>(
  root: ParentNode,
  selector: string,
  ctor: new () => T,
  label: string,
): T {
  const element = requireElement(root, selector);
  if (!(element instanceof ctor)) {
    throw new Error(`requireTyped: expected a ${label} at "${selector}"`);
  }
  return element;
}
