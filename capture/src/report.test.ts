// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createSession, finishSession, uploadFrame } from "./api.js";
import {
  REPORT_HTML,
  VERDICT_ACTIONS,
  describeReasonCode,
  mountReportPage,
  snapshotHeatmapForPrint,
} from "./report.js";
import {
  MIN_FINISH_FRAMES,
  startContractStub,
  type ContractStub,
  type VerdictScenario,
} from "../stub/contract-stub.js";
import { PAGE_TEST_STUB_PORT } from "../stub/page-test-port.js";

const FRAME = new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: "image/jpeg" });

describe("C-15 report page", () => {
  let baseUrl: string;
  let stub: ContractStub;
  let close: () => Promise<void>;

  beforeEach(async () => {
    const started = await startContractStub(PAGE_TEST_STUB_PORT);
    baseUrl = started.baseUrl;
    stub = started.stub;
    close = started.close;
    stub.setReconstructionPendingPolls(0);
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await close();
  });

  async function finishedSession(scenario: VerdictScenario): Promise<string> {
    const created = await createSession(baseUrl, "PO-48213-A");
    if (!created.ok) throw new Error("setup: createSession failed");
    for (let index = 0; index < MIN_FINISH_FRAMES; index += 1) {
      const uploaded = await uploadFrame(baseUrl, created.body.session_id, FRAME, {
        alpha: 1,
        beta: 2,
        gamma: 3,
      });
      if (!uploaded.ok) throw new Error("setup: uploadFrame failed");
    }
    const finished = await finishSession(baseUrl, created.body.session_id);
    if (!finished.ok) throw new Error("setup: finishSession failed");
    stub.setVerdictScenario(scenario);
    return created.body.session_id;
  }

  function root(): HTMLElement {
    const element = document.createElement("main");
    document.body.replaceChildren(element);
    return element;
  }

  function fakeViewer() {
    return vi.fn(
      (
        target: HTMLElement,
        result: unknown,
        options?: { initialColourMode?: string },
      ): (() => void) => {
        target.dataset.viewerResultType = typeof result;
        target.dataset.initialColourMode = options?.initialColourMode ?? "";
        const canvas = document.createElement("canvas");
        Object.defineProperty(canvas, "toDataURL", {
          value: vi.fn(() => "data:image/png;base64,heatmap"),
        });
        const legend = document.createElement("div");
        legend.setAttribute("aria-label", "Deviation from CAD legend");
        legend.textContent =
          "Deviation from CAD (mm): −3 mm · 0 mm · +3 mm — scale: largest measured deviation, not a tolerance.";
        const status = document.createElement("p");
        status.setAttribute("role", "status");
        status.textContent =
          "3D model loaded. This model has no provenance colours; observed and unobserved surface cannot be told apart.";
        target.append(canvas, legend, status);
        return () => undefined;
      },
    );
  }

  it("renders the Pass example, reconstruction summary and C-14 heatmap mode", async () => {
    const sessionId = await finishedSession("pass");
    const target = root();
    const mountViewer = fakeViewer();

    mountReportPage(target, {
      baseUrl,
      sessionId,
      now: () => new Date("2026-11-11T08:00:00Z"),
      mountViewer,
    });

    await vi.waitFor(() => {
      if (target.querySelector("#report-verdict")?.textContent !== "Pass") {
        throw new Error("report still loading");
      }
    });

    expect(target.textContent).toContain("Ship it.");
    expect(target.textContent).toContain("mounting_hole_A");
    expect(target.textContent).toContain("0.14 mm");
    expect(target.textContent).toContain("sealing_face");
    expect(target.textContent).toContain("-0.06 mm");
    expect(target.textContent).toContain("ChArUco board");
    expect(target.textContent).toContain("Metric");
    expect(target.textContent).toContain("91.0%");
    expect(target.textContent).toContain("checked — no disagreement flagged");
    expect(target.textContent).toContain(sessionId);
    expect(target.textContent).toContain("2026-11-11T08:00:00.000Z");
    expect(target.textContent).toContain("AI narrative slot (A-09)");
    expect(mountViewer).toHaveBeenCalledOnce();
    expect(mountViewer.mock.calls[0]?.[2]).toMatchObject({ initialColourMode: "deviation" });
  });

  it("renders Rescan reasons, hides unobserved deviation and shows the placement note", async () => {
    const sessionId = await finishedSession("rescan");
    const target = root();

    mountReportPage(target, {
      baseUrl,
      sessionId,
      placementNote: "Keep the sealing face toward the printed arrow.",
      mountViewer: fakeViewer(),
    });

    await vi.waitFor(() => {
      if (target.querySelector("#report-verdict")?.textContent !== "Rescan") {
        throw new Error("report still loading");
      }
    });

    expect(target.textContent).toContain("Shoot it again, covering the named region.");
    expect(target.textContent).toContain("The sealing face was not seen in enough usable views.");
    expect(target.textContent).toContain(
      "More than one CAD orientation is plausible for this reconstruction.",
    );
    expect(target.textContent).toContain(
      "Placement note: Keep the sealing face toward the printed arrow.",
    );

    const rows = [...target.querySelectorAll("#report-regions tr")].map((row) =>
      row.textContent.replace(/\s+/g, " ").trim(),
    );
    const sealingFace = rows.find((row) => row.includes("sealing_face"));
    expect(sealingFace).toContain("unobserved");
    expect(sealingFace).toContain("—");
    expect(sealingFace).toContain("No");
    expect(sealingFace).not.toContain("0.00 mm");
  });

  it("renders the canned Fail example without inventing a tolerance value", async () => {
    const sessionId = await finishedSession("fail");
    const target = root();

    mountReportPage(target, { baseUrl, sessionId, mountViewer: fakeViewer() });

    await vi.waitFor(() => {
      if (target.querySelector("#report-verdict")?.textContent !== "Fail") {
        throw new Error("report still loading");
      }
    });

    expect(target.textContent).toContain(
      "The part is out of spec at a place the camera genuinely saw.",
    );
    expect(target.textContent).toContain(
      "A measured, Observed region is outside the order specification.",
    );
    expect(target.textContent).not.toMatch(/σ_floor|±\s*\d/);
  });

  it("shows No dimensional claims and scale not checked when the fields say so", async () => {
    const sessionId = await finishedSession("no_cad");
    stub.useNoReferenceReconstruction();
    const target = root();

    mountReportPage(target, { baseUrl, sessionId, mountViewer: fakeViewer() });

    await vi.waitFor(() => {
      if (target.querySelector("#report-reference")?.textContent !== "no reference") {
        throw new Error("report still loading");
      }
    });

    expect(target.querySelector("#report-metric")?.textContent).toBe("No dimensional claims");
    expect(target.querySelector("#report-scale")?.textContent).toBe("not checked");
  });

  it("treats no_cad_uploaded as model only, not a report error", async () => {
    const sessionId = await finishedSession("no_cad");
    const target = root();
    const mountViewer = fakeViewer();

    mountReportPage(target, { baseUrl, sessionId, mountViewer });

    await vi.waitFor(() => {
      if (target.querySelector("#report-verdict")?.textContent !== "Model only") {
        throw new Error("report still loading");
      }
    });

    expect(target.textContent).toContain("No CAD on this order: model only");
    expect(target.textContent).toContain("91.0%");
    expect(target.textContent).not.toContain("Could not load verdict");
    expect(mountViewer).toHaveBeenCalledOnce();
  });

  it("shows an unknown reason code verbatim and has an action for Unverifiable", () => {
    expect(describeReasonCode("future_reason_code")).toBe("future_reason_code");
    expect(VERDICT_ACTIONS.Unverifiable).toContain("CMM");
  });

  it("snapshots the heatmap canvas on beforeprint and the print button calls window.print", async () => {
    const sessionId = await finishedSession("pass");
    const target = root();
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => undefined);

    mountReportPage(target, { baseUrl, sessionId, mountViewer: fakeViewer() });

    await vi.waitFor(() => {
      if (target.querySelector("#report-model-viewer canvas") === null) {
        throw new Error("viewer not mounted yet");
      }
    });

    window.dispatchEvent(new Event("beforeprint"));
    const image = target.querySelector("#report-print-heatmap");
    expect(image).toBeInstanceOf(HTMLImageElement);
    expect((image as HTMLImageElement).hidden).toBe(false);
    expect((image as HTMLImageElement).src).toBe("data:image/png;base64,heatmap");

    const printLegend = target.querySelector("#report-print-legend");
    expect(printLegend).toBeInstanceOf(HTMLElement);
    expect((printLegend as HTMLElement).hidden).toBe(false);
    expect(printLegend?.textContent).toContain("largest measured deviation, not a tolerance");

    const printStatus = target.querySelector("#report-print-viewer-status");
    expect(printStatus).toBeInstanceOf(HTMLElement);
    expect((printStatus as HTMLElement).hidden).toBe(false);
    expect(printStatus?.textContent).toContain("has no provenance colours");

    const button = target.querySelector("#report-print-button");
    expect(button).toBeInstanceOf(HTMLButtonElement);
    (button as HTMLButtonElement).click();
    expect(printSpy).toHaveBeenCalledOnce();
  });

  it("returns false when there is no printable canvas", () => {
    const target = root();
    target.innerHTML = REPORT_HTML;
    expect(snapshotHeatmapForPrint(target)).toBe(false);
  });
});
