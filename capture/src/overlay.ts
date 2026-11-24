// C-07: renders the orbit coverage count and an arrow toward the next uncovered target, into a
// container element it's given -- it never creates its own page root the way mountGyroCheck
// does, since this is meant to sit inside C-05's eventual live view, not stand alone. Does not
// own the camera stream or gyro permission-granting; by the time this mounts, gyro.ts's
// permission flow has already run.

import {
  anchorHeadingDeg,
  computeTargetGuidance,
  computeCoverage,
  DEFAULT_ORBIT,
  guidanceLevel,
  type GuidanceLevel,
  type OrbitSlot,
} from "./orbit.js";
import { requireElement } from "./dom.js";
import type { Gyro } from "./types.js";

// C-26: what the viewfinder needs to colour its frame (and C-27 to decide when to capture).
// `gyro` is the live reading the level was computed from; null when no live reading exists yet.
export interface GuidanceUpdate {
  level: GuidanceLevel;
  gyro: Gyro | null;
}

export interface GyroOverlayOptions {
  onGuidance?: (update: GuidanceUpdate) => void;
  // C-27: the on-target status line. The shell words it by whether auto-capture is on.
  onTargetText?: () => string;
}

export interface GyroOverlayHandle {
  reportAcceptedFrame(gyro: Gyro): void;
  // C-27 (#420 review): re-render from the last live reading, so text that depends on state the
  // overlay can't see (onTargetText, read from the auto-capture switch) updates without waiting
  // for the phone to move.
  refresh(): void;
  dispose(): void;
}

// C-26: the arrow is an SVG shape, not a text glyph, so it can be sized and coloured (capture.css)
// and stays crisp at any size. It still points "up" at rotate(0) and is turned by the same
// transform as before. The completion banner (hidden until every slot is covered) sits in the
// middle of the preview, so the end of the orbit can't be missed in the small top-left chip.
const OVERLAY_HTML = `
  <div class="gyro-overlay">
    <div class="gyro-overlay-chips">
      <p class="gyro-overlay-status"></p>
      <p class="gyro-overlay-count"></p>
    </div>
    <div class="gyro-overlay-arrow" aria-hidden="true">
      <svg viewBox="0 0 100 100" focusable="false">
        <path d="M50 6 L92 50 L64 50 L64 94 L36 94 L36 50 L8 50 Z" />
      </svg>
    </div>
    <div class="gyro-overlay-complete" role="status" hidden>
      <svg viewBox="0 0 48 48" aria-hidden="true" focusable="false">
        <circle cx="24" cy="24" r="22" />
        <path d="M13 25 L21 33 L35 17" />
      </svg>
      <p class="gyro-overlay-complete-title">Orbit complete</p>
      <p class="gyro-overlay-complete-hint">Tap Finish session below.</p>
    </div>
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

// Per-container teardown, so a second mount on the same container (e.g. C-09's capture loop
// re-mounting on retry) can't leak the first mount's listener -- caught in review (TabeenRaoof,
// PR #120): disposing the *returned* handle only helps a caller that remembers to call it; a
// caller that just calls mountGyroOverlay again on the same element had no such guard.
const activeDisposalsByContainer = new WeakMap<HTMLElement, () => void>();

export function mountGyroOverlay(
  container: HTMLElement,
  orbit: readonly OrbitSlot[] = DEFAULT_ORBIT,
  options: GyroOverlayOptions = {},
): GyroOverlayHandle {
  activeDisposalsByContainer.get(container)?.();

  container.innerHTML = OVERLAY_HTML;

  const status = requireElement(container, ".gyro-overlay-status");
  const arrow = requireElement(container, ".gyro-overlay-arrow");
  const count = requireElement(container, ".gyro-overlay-count");
  const completeBanner = requireElement(container, ".gyro-overlay-complete");

  const acceptedReadings: Gyro[] = [];
  let latestLiveGyro: Gyro | null = null;
  count.textContent = formatCount(0, orbit.length);
  status.textContent = "Tap the shutter to take the first photo.";
  arrow.style.visibility = "hidden";

  function emit(level: GuidanceLevel, gyro: Gyro | null): void {
    options.onGuidance?.({ level, gyro });
  }
  emit("idle", null);

  function showComplete(gyro: Gyro | null): void {
    status.textContent = "Orbit complete.";
    arrow.style.visibility = "hidden";
    completeBanner.hidden = false;
    emit("complete", gyro);
  }

  function render(): void {
    const coverage = computeCoverage(acceptedReadings, orbit);
    count.textContent = formatCount(coverage.covered.filter(Boolean).length, orbit.length);
  }

  function updateArrow(liveGyro: Gyro): void {
    const coverage = computeCoverage(acceptedReadings, orbit);
    if (coverage.nextTargetIndex === null) {
      showComplete(liveGyro);
      return;
    }
    const anchorDeg = anchorHeadingDeg(acceptedReadings);
    if (anchorDeg === null) {
      // No accepted frames yet -- nothing to anchor azimuth to, so no arrow guidance yet.
      emit("idle", liveGyro);
      return;
    }
    // computeTargetGuidance, not computeArrowGuidance: green only when coverage would credit
    // this reading to the target (C-26 device check, 2026-11-20).
    const guidance = computeTargetGuidance(liveGyro, anchorDeg, orbit, coverage.nextTargetIndex);
    if (guidance.onTarget) {
      status.textContent = options.onTargetText?.() ?? "On target — take the photo.";
      arrow.style.visibility = "hidden";
      emit("on-target", liveGyro);
      return;
    }
    status.textContent = "Move toward the next position.";
    arrow.style.visibility = "visible";
    arrow.style.transform = `rotate(${String(arrowRotationDeg(guidance.azimuthDeltaDeg, guidance.elevationDeltaDeg))}deg)`;
    emit(guidanceLevel(guidance), liveGyro);
  }

  // A named reference, not an inline arrow function, specifically so it can be removed again --
  // caught in review (TabeenRaoof and mbj1994, independently): this mount previously attached the
  // listener with no way to remove it, and the old `listening` flag it was guarded by never
  // actually did anything (it was only ever checked once per mount, since nothing re-called
  // `startListening`), so every remount (a new session, or each test) leaked one more permanent
  // `window` listener.
  function handleOrientation(event: DeviceOrientationEvent): void {
    if (event.alpha === null || event.beta === null || event.gamma === null) {
      return;
    }
    latestLiveGyro = { alpha: event.alpha, beta: event.beta, gamma: event.gamma };
    updateArrow(latestLiveGyro);
  }
  window.addEventListener("deviceorientation", handleOrientation);

  // C-27 (#420): once disposed (Finish, or a remount that reused this container), refresh() must
  // not redraw -- the guidance is stale with the camera off, and after a remount the DOM belongs to
  // the new mount.
  let disposed = false;
  function dispose(): void {
    disposed = true;
    window.removeEventListener("deviceorientation", handleOrientation);
    activeDisposalsByContainer.delete(container);
  }
  activeDisposalsByContainer.set(container, dispose);

  return {
    reportAcceptedFrame(gyro: Gyro): void {
      acceptedReadings.push(gyro);
      render();
      status.textContent =
        acceptedReadings.length === 1 ? "First position captured." : "Position captured.";
      // Without this, the arrow stays stale (pointing at whatever was the target before this
      // frame was accepted) until the next deviceorientation event happens to fire -- caught in
      // review (TabeenRaoof, PR #120). Re-run on the last live reading, if there's been one, so
      // the arrow reflects the new target immediately.
      if (latestLiveGyro) {
        updateArrow(latestLiveGyro);
      } else if (computeCoverage(acceptedReadings, orbit).nextTargetIndex === null) {
        // No live reading to judge position by, but a complete orbit is complete regardless.
        showComplete(null);
      }
    },
    refresh(): void {
      if (!disposed && latestLiveGyro) {
        updateArrow(latestLiveGyro);
      }
    },
    dispose,
  };
}
