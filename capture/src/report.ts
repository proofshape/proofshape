import { getReconstruction, getVerdict } from "./api.js";
import {
  mountModelViewer,
  type ModelViewerOptions,
  type ModelViewerState,
} from "./model-viewer.js";
import type { ReconstructionResult, Region, Verdict, VerdictResult } from "./types.js";

export const VERDICT_ACTIONS: Record<Verdict, string> = {
  Pass: "Ship it.",
  Fail: "The part is out of spec at a place the camera genuinely saw.",
  Rescan: "Shoot it again, covering the named region. Not a judgement on the part.",
  Unverifiable:
    "Route to a CMM. ProofShape cannot answer this question and says so before more capture work.",
};

const REASON_TEXT: Record<string, string> = {
  sealing_face_not_seen: "The sealing face was not seen in enough usable views.",
  registration_orientation_ambiguous:
    "More than one CAD orientation is plausible for this reconstruction.",
  observed_deviation_out_of_spec: "A measured, Observed region is outside the order specification.",
  scale_disagreement: "The independent scale sources disagreed.",
};

const REFERENCE_TEXT: Record<string, string> = {
  charuco_board: "ChArUco board",
  blank_sheet: "blank reference sheet",
  credit_card: "credit card",
  none: "no reference",
};

export function describeReasonCode(code: string): string {
  return REASON_TEXT[code] ?? code;
}

export interface ReportPageOptions {
  baseUrl: string;
  sessionId: string;
  authToken?: string;
  placementNote?: string;
  now?: () => Date;
  modelViewerOptions?: ModelViewerOptions;
  mountViewer?: (
    root: HTMLElement,
    result: ModelViewerState,
    options?: ModelViewerOptions,
  ) => () => void;
}

export const REPORT_HTML = `
  <article class="report-sheet">
    <header class="report-header">
      <p class="eyebrow">ProofShape condition report</p>
      <h1>Inspection report</h1>
      <dl class="report-meta">
        <div><dt>Session</dt><dd id="report-session-id"></dd></div>
        <div><dt>Report generated</dt><dd id="report-generated-at"></dd></div>
      </dl>
    </header>

    <section aria-labelledby="report-verdict-title">
      <h2 id="report-verdict-title">Verdict</h2>
      <p id="report-verdict" class="report-verdict">Loading…</p>
      <p><strong>What to do:</strong> <span id="report-action">Loading…</span></p>
      <ul id="report-reasons"></ul>
    </section>

    <section aria-labelledby="report-regions-title">
      <h2 id="report-regions-title">Critical regions</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Region</th>
              <th scope="col">Provenance</th>
              <th scope="col">Deviation</th>
              <th scope="col">Counts toward verdict</th>
            </tr>
          </thead>
          <tbody id="report-regions"></tbody>
        </table>
      </div>
    </section>

    <section aria-labelledby="report-orientation-title">
      <h2 id="report-orientation-title">Orientation</h2>
      <p id="report-orientation">Loading…</p>
      <p id="report-placement-note" hidden></p>
    </section>

    <section aria-labelledby="report-reconstruction-title">
      <h2 id="report-reconstruction-title">Reconstruction summary</h2>
      <dl class="report-summary">
        <div><dt>Reference used</dt><dd id="report-reference">Loading…</dd></div>
        <div><dt>Dimensional status</dt><dd id="report-metric">Loading…</dd></div>
        <div><dt>Observed fraction</dt><dd id="report-observed">Loading…</dd></div>
        <div><dt>Scale agreement</dt><dd id="report-scale">Loading…</dd></div>
      </dl>
    </section>

    <section aria-labelledby="report-heatmap-title">
      <h2 id="report-heatmap-title">Deviation heatmap</h2>
      <div id="report-model-viewer"></div>
      <img id="report-print-heatmap" alt="Deviation heatmap snapshot for print" hidden />
      <p id="report-print-legend" class="report-print-legend" hidden></p>
      <p id="report-print-viewer-status" class="report-print-viewer-status" hidden></p>
    </section>

    <section aria-labelledby="report-narrative-title">
      <h2 id="report-narrative-title">Narrative</h2>
      <p id="report-narrative">AI narrative slot (A-09) — not available yet.</p>
    </section>

    <p id="report-status" role="status"></p>
    <div class="report-controls">
      <button id="report-print-button" type="button">Print / Save as PDF</button>
    </div>
  </article>
`;

function requireElement<T extends HTMLElement>(
  root: HTMLElement,
  selector: string,
  constructor: { new (): T },
): T {
  const element = root.querySelector(selector);
  if (!(element instanceof constructor)) {
    throw new Error(`report: expected ${selector}`);
  }
  return element;
}

function formatDeviation(region: Region): string {
  if (region.provenance !== "observed" || region.deviation_mm === undefined) {
    return "—";
  }
  return `${region.deviation_mm.toFixed(2)} mm`;
}

function renderRegions(target: HTMLTableSectionElement, regions: Region[]): void {
  target.replaceChildren();
  for (const region of regions) {
    const row = document.createElement("tr");
    for (const value of [
      region.name,
      region.provenance,
      formatDeviation(region),
      region.contributes_to_verdict ? "Yes" : "No",
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    target.append(row);
  }
}

function renderVerdict(
  root: HTMLElement,
  result: VerdictResult,
  placementNote: string | undefined,
): void {
  requireElement(root, "#report-verdict", HTMLElement).textContent = result.verdict;
  requireElement(root, "#report-action", HTMLElement).textContent = VERDICT_ACTIONS[result.verdict];

  const reasons = requireElement(root, "#report-reasons", HTMLUListElement);
  reasons.replaceChildren();
  for (const code of result.reason_codes ?? []) {
    const item = document.createElement("li");
    item.textContent = describeReasonCode(code);
    reasons.append(item);
  }
  if ((result.reason_codes ?? []).length === 0) {
    const item = document.createElement("li");
    item.textContent = "No additional reasons.";
    reasons.append(item);
  }

  renderRegions(requireElement(root, "#report-regions", HTMLTableSectionElement), result.regions);

  const orientation = requireElement(root, "#report-orientation", HTMLElement);
  orientation.textContent = result.orientation_ambiguous
    ? "Orientation is ambiguous."
    : "Orientation is not ambiguous.";

  const note = requireElement(root, "#report-placement-note", HTMLElement);
  note.hidden = true;
  note.textContent = "";
  if (result.orientation_ambiguous && placementNote !== undefined && placementNote.trim() !== "") {
    note.textContent = `Placement note: ${placementNote}`;
    note.hidden = false;
  }
}

function renderModelOnly(root: HTMLElement): void {
  requireElement(root, "#report-verdict", HTMLElement).textContent = "Model only";
  requireElement(root, "#report-action", HTMLElement).textContent =
    "No CAD on this order: model only";
  const reasons = requireElement(root, "#report-reasons", HTMLUListElement);
  reasons.replaceChildren();
  const item = document.createElement("li");
  item.textContent = "No verdict exists because this order has no CAD.";
  reasons.append(item);
  requireElement(root, "#report-regions", HTMLTableSectionElement).replaceChildren();
  requireElement(root, "#report-orientation", HTMLElement).textContent = "Not checked.";
}

function renderReconstructionSummary(root: HTMLElement, result: ReconstructionResult): void {
  requireElement(root, "#report-reference", HTMLElement).textContent =
    REFERENCE_TEXT[result.reference_tier_used ?? ""] ??
    result.reference_tier_used ??
    "not reported";
  requireElement(root, "#report-metric", HTMLElement).textContent =
    result.metric === true ? "Metric" : "No dimensional claims";
  requireElement(root, "#report-observed", HTMLElement).textContent =
    result.observed_fraction === undefined
      ? "not reported"
      : `${(result.observed_fraction * 100).toFixed(1)}%`;
  const scale = requireElement(root, "#report-scale", HTMLElement);
  scale.textContent =
    result.scale_agreement === undefined
      ? "not checked"
      : result.scale_agreement.flagged
        ? "checked — disagreement flagged"
        : "checked — no disagreement flagged";
}

export function snapshotHeatmapForPrint(root: HTMLElement): boolean {
  const canvas = root.querySelector("#report-model-viewer canvas");
  const image = root.querySelector("#report-print-heatmap");
  const printLegend = root.querySelector("#report-print-legend");
  const printViewerStatus = root.querySelector("#report-print-viewer-status");
  const viewerStatus = root.querySelector('#report-model-viewer [role="status"]');
  const activeLegend = root.querySelector(
    '#report-model-viewer [aria-label="Deviation from CAD legend"], ' +
      '#report-model-viewer [aria-label="3D model provenance legend"]',
  );

  if (printLegend instanceof HTMLElement) {
    if (activeLegend instanceof HTMLElement && activeLegend.textContent.trim() !== "") {
      printLegend.textContent = activeLegend.textContent;
      printLegend.hidden = false;
    } else {
      printLegend.textContent = "";
      printLegend.hidden = true;
    }
  }

  if (printViewerStatus instanceof HTMLElement) {
    if (viewerStatus instanceof HTMLElement && viewerStatus.textContent.trim() !== "") {
      printViewerStatus.textContent = viewerStatus.textContent;
      printViewerStatus.hidden = false;
    } else {
      printViewerStatus.textContent = "";
      printViewerStatus.hidden = true;
    }
  }

  if (!(canvas instanceof HTMLCanvasElement) || !(image instanceof HTMLImageElement)) {
    return false;
  }
  try {
    image.src = canvas.toDataURL("image/png");
    image.hidden = false;
    return image.src !== "";
  } catch {
    image.hidden = true;
    return false;
  }
}

export function mountReportPage(root: HTMLElement, options: ReportPageOptions): () => void {
  root.innerHTML = REPORT_HTML;

  requireElement(root, "#report-session-id", HTMLElement).textContent = options.sessionId;
  requireElement(root, "#report-generated-at", HTMLElement).textContent = (
    options.now?.() ?? new Date()
  ).toISOString();

  const status = requireElement(root, "#report-status", HTMLElement);
  const viewerRoot = requireElement(root, "#report-model-viewer", HTMLElement);
  const printButton = requireElement(root, "#report-print-button", HTMLButtonElement);
  const mountViewer = options.mountViewer ?? mountModelViewer;
  let disposeViewer = (): void => undefined;
  const lifecycle = { disposed: false };

  const beforePrint = (): void => {
    snapshotHeatmapForPrint(root);
  };
  window.addEventListener("beforeprint", beforePrint);
  printButton.addEventListener("click", () => {
    window.print();
  });

  void (async () => {
    try {
      const [reconstruction, verdict] = await Promise.all([
        getReconstruction(options.baseUrl, options.sessionId, options.authToken),
        getVerdict(options.baseUrl, options.sessionId, options.authToken),
      ]);
      if (lifecycle.disposed) return;

      if (reconstruction.ok) {
        renderReconstructionSummary(root, reconstruction.body);
        if (reconstruction.body.status === "complete") {
          disposeViewer = mountViewer(viewerRoot, reconstruction.body, {
            ...options.modelViewerOptions,
            ...(options.authToken !== undefined ? { authToken: options.authToken } : {}),
            initialColourMode: "deviation",
          });
        } else {
          disposeViewer = mountViewer(viewerRoot, reconstruction.body, options.modelViewerOptions);
        }
      } else {
        status.textContent = `Could not load reconstruction — ${reconstruction.error.message}`;
      }

      if (verdict.ok) {
        renderVerdict(root, verdict.body, options.placementNote);
      } else if (verdict.error.code === "no_cad_uploaded") {
        renderModelOnly(root);
      } else {
        requireElement(root, "#report-verdict", HTMLElement).textContent = "Unavailable";
        requireElement(root, "#report-action", HTMLElement).textContent =
          `Could not load verdict — ${verdict.error.message}`;
      }
    } catch (error) {
      status.textContent = `Could not load report — ${
        error instanceof Error ? error.message : String(error)
      }`;
    }
  })();

  return () => {
    lifecycle.disposed = true;
    window.removeEventListener("beforeprint", beforePrint);
    disposeViewer();
  };
}
