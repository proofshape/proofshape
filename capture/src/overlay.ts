// C-07: renders the orbit coverage count and an arrow toward the next uncovered target, into a
// container element it's given -- it never creates its own page root the way mountGyroCheck
// does, since this is meant to sit inside C-05's eventual live view, not stand alone. Does not
// own the camera stream or gyro permission-granting; by the time this mounts, gyro.ts's
// permission flow has already run.

import { computeArrowGuidance, computeCoverage, DEFAULT_ORBIT, type OrbitSlot } from "./orbit.js";
import { requireElement } from "./dom.js";
import type { Gyro } from "./types.js";

export interface GyroOverlayHandle {
  reportAcceptedFrame(gyro: Gyro): void;
}

const OVERLAY_HTML = `
  <div class="gyro-overlay">
    <p class="gyro-overlay-status"></p>
    <div class="gyro-overlay-arrow" aria-hidden="true">&#8593;</div>
    <p class="gyro-overlay-count"></p>
  </div>
`;

function formatCount(coveredCount: number, total: number): string {
  return `Covered: ${String(coveredCount)}/${String(total)}`;
}

// A simple first-pass mapping from the signed azimuth/elevation deltas to an on-screen rotation
// -- azimuth leans the arrow left/right, elevation leans it up/down. Not meant to be pixel-exact;
// refining it is exactly what the real-device manual check is for.
function arrowRotationDeg(azimuthDeltaDeg: number, elevationDeltaDeg: number): number {
  return (Math.atan2(azimuthDeltaDeg, elevationDeltaDeg) * 180) / Math.PI;
}

export function mountGyroOverlay(
  container: HTMLElement,
  orbit: readonly OrbitSlot[] = DEFAULT_ORBIT,
): GyroOverlayHandle {
  container.innerHTML = OVERLAY_HTML;

  const status = requireElement(container, ".gyro-overlay-status");
  const arrow = requireElement(container, ".gyro-overlay-arrow");
  const count = requireElement(container, ".gyro-overlay-count");

  const acceptedReadings: Gyro[] = [];
  count.textContent = formatCount(0, orbit.length);
  status.textContent = "Capture the first frame to start.";
  arrow.style.visibility = "hidden";

  function render(): void {
    const coverage = computeCoverage(acceptedReadings, orbit);
    count.textContent = formatCount(coverage.covered.filter(Boolean).length, orbit.length);
  }

  function updateArrow(liveGyro: Gyro): void {
    const coverage = computeCoverage(acceptedReadings, orbit);
    if (coverage.nextTargetIndex === null) {
      status.textContent = "Orbit complete.";
      arrow.style.visibility = "hidden";
      return;
    }
    const anchorDeg = acceptedReadings[0]?.alpha;
    if (anchorDeg === undefined) {
      // No accepted frames yet -- nothing to anchor azimuth to, so no arrow guidance yet.
      return;
    }
    const target = orbit[coverage.nextTargetIndex];
    if (!target) {
      return;
    }
    const guidance = computeArrowGuidance(liveGyro, anchorDeg, target);
    if (guidance.onTarget) {
      status.textContent = "On target.";
      arrow.style.visibility = "hidden";
      return;
    }
    status.textContent = "Move toward the next position.";
    arrow.style.visibility = "visible";
    arrow.style.transform = `rotate(${String(arrowRotationDeg(guidance.azimuthDeltaDeg, guidance.elevationDeltaDeg))}deg)`;
  }

  // Scoped to this mount, not module-level -- the same reasoning gyro.ts's own `listening` flag
  // uses, to avoid leaking across repeated mounts (e.g. in tests).
  let listening = false;

  function startListening(): void {
    if (listening) {
      return;
    }
    listening = true;
    window.addEventListener("deviceorientation", (event) => {
      if (event.alpha === null || event.beta === null || event.gamma === null) {
        return;
      }
      updateArrow({ alpha: event.alpha, beta: event.beta, gamma: event.gamma });
    });
  }
  startListening();

  return {
    reportAcceptedFrame(gyro: Gyro): void {
      acceptedReadings.push(gyro);
      render();
      status.textContent =
        acceptedReadings.length === 1 ? "First position captured." : "Position captured.";
    },
  };
}
