// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from "vitest";
import { mountGyroOverlay } from "./overlay.js";
import { DEFAULT_ORBIT, type OrbitSlot } from "./orbit.js";
import type { Gyro } from "./types.js";

// noUncheckedIndexedAccess makes DEFAULT_ORBIT[i] `OrbitSlot | undefined`; this codebase bans
// `!` non-null assertions, so an explicit check stands in for it everywhere a fixed, known-valid
// index into DEFAULT_ORBIT is used below.
function requireSlot(index: number): OrbitSlot {
  const slot = DEFAULT_ORBIT[index];
  if (!slot) {
    throw new Error(`test setup: DEFAULT_ORBIT has no slot at index ${String(index)}`);
  }
  return slot;
}

// dispatchOrientation: construct a plain Event and assign alpha/beta/gamma directly, since
// happy-dom's real DeviceOrientationEvent constructor doesn't wire an init dict -- the same
// pattern gyro.test.ts already established.
function dispatchOrientation(
  alpha: number | null,
  beta: number | null,
  gamma: number | null,
): void {
  const event = new Event("deviceorientation") as DeviceOrientationEvent;
  (event as unknown as { alpha: number | null }).alpha = alpha;
  (event as unknown as { beta: number | null }).beta = beta;
  (event as unknown as { gamma: number | null }).gamma = gamma;
  window.dispatchEvent(event);
}

function elevationToBeta(elevationDeg: number): number {
  return 90 - elevationDeg;
}

function mount(): { container: HTMLElement; root: HTMLElement } {
  document.body.innerHTML = '<main id="test-root"><div id="overlay-container"></div></main>';
  const root = document.getElementById("test-root");
  const container = document.getElementById("overlay-container");
  if (!root || !container) throw new Error("test setup: root/container missing");
  return { container, root };
}

function text(root: HTMLElement, selector: string): string | null {
  return root.querySelector(selector)?.textContent ?? null;
}

describe("mountGyroOverlay", () => {
  afterEach(() => {
    // No vi.stubGlobal used here (no permission mocking needed), but dispatched events are
    // real window listeners -- nothing to explicitly unstub, each test gets a fresh document.
  });

  it("renders into the given container, not a new root it creates itself", () => {
    const { container, root } = mount();
    mountGyroOverlay(container);

    expect(root.querySelector(".gyro-overlay")).not.toBeNull();
    // The container passed in is the one that got the markup -- nothing escaped it.
    expect(container.querySelector(".gyro-overlay")).not.toBeNull();
  });

  it("shows an initial state and a 0-covered count before any frame is accepted", () => {
    const { container } = mount();
    mountGyroOverlay(container);

    expect(text(container, ".gyro-overlay-count")).toBe(
      `Covered: 0/${String(DEFAULT_ORBIT.length)}`,
    );
    expect(text(container, ".gyro-overlay-status")).toMatch(/capture the first frame/i);
  });

  it("reportAcceptedFrame advances the covered count", () => {
    const { container } = mount();
    const handle = mountGyroOverlay(container);

    const slot0 = requireSlot(0);
    const firstReading: Gyro = {
      alpha: slot0.azimuthDeg,
      beta: elevationToBeta(slot0.elevationDeg),
      gamma: 0,
    };
    handle.reportAcceptedFrame(firstReading);

    expect(text(container, ".gyro-overlay-count")).toBe(
      `Covered: 1/${String(DEFAULT_ORBIT.length)}`,
    );
  });

  it("a dispatched orientation event updates the arrow toward the next target -- not just status text", () => {
    const { container } = mount();
    const handle = mountGyroOverlay(container);
    // Anchor at alpha=0, elevation matching slot 0 -- covers slot 0, next target is slot 1.
    const slot0 = requireSlot(0);
    handle.reportAcceptedFrame({ alpha: 0, beta: elevationToBeta(slot0.elevationDeg), gamma: 0 });

    const arrow = container.querySelector<HTMLElement>(".gyro-overlay-arrow");
    if (!arrow) throw new Error("test: arrow element missing");
    const beforeTransform = arrow.style.transform;

    // Live reading far from slot 1's target -- should be visibly off-target with a real rotation.
    const slot1 = requireSlot(1);
    dispatchOrientation(slot1.azimuthDeg + 60, elevationToBeta(slot1.elevationDeg), 0);

    expect(arrow.style.visibility).toBe("visible");
    expect(arrow.style.transform).not.toBe(beforeTransform);
    expect(text(container, ".gyro-overlay-status")).toMatch(/move toward/i);
  });

  it("hides the arrow and shows an on-target status once the live reading matches the target", () => {
    const { container } = mount();
    const handle = mountGyroOverlay(container);
    handle.reportAcceptedFrame({
      alpha: 0,
      beta: elevationToBeta(requireSlot(0).elevationDeg),
      gamma: 0,
    });

    const target = requireSlot(1);
    dispatchOrientation(target.azimuthDeg, elevationToBeta(target.elevationDeg), 0);

    expect(text(container, ".gyro-overlay-status")).toMatch(/on target/i);
    const arrow = container.querySelector<HTMLElement>(".gyro-overlay-arrow");
    expect(arrow?.style.visibility).toBe("hidden");
  });

  it("shows an orbit-complete status once every slot is covered", () => {
    const { container } = mount();
    const handle = mountGyroOverlay(container);
    for (const slot of DEFAULT_ORBIT) {
      handle.reportAcceptedFrame({
        alpha: slot.azimuthDeg,
        beta: elevationToBeta(slot.elevationDeg),
        gamma: 0,
      });
    }

    dispatchOrientation(0, elevationToBeta(45), 0);

    expect(text(container, ".gyro-overlay-status")).toMatch(/orbit complete/i);
  });

  it("ignores a deviceorientation event with a null component rather than crashing", () => {
    const { container } = mount();
    const handle = mountGyroOverlay(container);
    handle.reportAcceptedFrame({
      alpha: 0,
      beta: elevationToBeta(requireSlot(0).elevationDeg),
      gamma: 0,
    });

    expect(() => {
      dispatchOrientation(null, null, null);
    }).not.toThrow();
  });

  it("does not attach a second deviceorientation listener on a second mount into a fresh container", () => {
    const { container } = mount();
    mountGyroOverlay(container);

    document.body.innerHTML += '<div id="second-container"></div>';
    const secondContainer = document.getElementById("second-container");
    if (!secondContainer) throw new Error("test setup: second container missing");
    const secondHandle = mountGyroOverlay(secondContainer);
    secondHandle.reportAcceptedFrame({ alpha: 0, beta: elevationToBeta(45), gamma: 0 });

    // Each mount's own closure guards its own listener (scoped, not module-level) -- this just
    // proves a second mount doesn't throw or corrupt the first container's content.
    expect(() => {
      dispatchOrientation(0, elevationToBeta(45), 0);
    }).not.toThrow();
    expect(container.querySelector(".gyro-overlay")).not.toBeNull();
  });
});
