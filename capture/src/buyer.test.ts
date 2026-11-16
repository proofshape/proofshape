// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from "vitest";
import { buildInspectionSpec, captureLinkFor, mountBuyerPage, validateCadFile } from "./buyer.js";

function required(root: ParentNode, selector: string): Element {
  const element = root.querySelector(selector);
  if (element === null) throw new Error(`test setup missing ${selector}`);
  return element;
}

afterEach(() => {
  vi.restoreAllMocks();
  sessionStorage.clear();
  document.body.innerHTML = "";
});

describe("C-16 buyer setup", () => {
  it("turns edited rows into the proposal section 7.4 JSON shape", () => {
    const result = buildInspectionSpec(
      [
        { name: "mounting hole", kind: "hole", tolerance: "2", critical: true },
        { name: "sealing face", kind: "face", tolerance: "0.5", critical: false },
      ],
      "Buyer says face is cosmetic\nUse CAD orientation",
    );
    expect(result).toEqual({
      ok: true,
      spec: {
        features: [
          { name: "mounting hole", kind: "hole", tolerance_mm: 2, critical: true },
          { name: "sealing face", kind: "face", tolerance_mm: 0.5, critical: false },
        ],
        assumptions: ["Buyer says face is cosmetic", "Use CAD orientation"],
      },
    });
  });

  it.each([
    ["", "1", "Feature 1 needs a name."],
    ["hole A", "0", "Feature 1 needs a tolerance greater than 0 mm."],
    ["hole A", "-1", "Feature 1 needs a tolerance greater than 0 mm."],
    ["hole A", "nope", "Feature 1 needs a tolerance greater than 0 mm."],
  ])("blocks invalid feature input", (name, tolerance, message) => {
    const result = buildInspectionSpec([{ name, kind: "hole", tolerance, critical: true }], "");
    expect(result).toEqual({ ok: false, message });
  });

  it("blocks a wrong CAD extension", () => {
    expect(validateCadFile(new File(["x"], "part.step"))).toBe(
      "CAD file must be STL, OBJ, or GLB.",
    );
    expect(validateCadFile(new File(["x"], "part.STL"))).toBeNull();
  });

  it("builds capture links at root and under the GitHub Pages base path", () => {
    expect(captureLinkFor("https://example.test/buyer.html", "PO-ABC")).toBe(
      "https://example.test/?order=PO-ABC",
    );
    expect(captureLinkFor("https://example.test/proofshape/buyer.html", "PO-ABC")).toBe(
      "https://example.test/proofshape/?order=PO-ABC",
    );
    expect(captureLinkFor("https://example.test/proofshape/buyer", "PO-ABC")).toBe(
      "https://example.test/proofshape/?order=PO-ABC",
    );
  });

  it("requires a buyer to fill every parser null tolerance before createOrder", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          status: "parsed",
          spec: {
            features: [
              { name: "mounting holes", kind: "hole", tolerance_mm: 2, critical: true },
              { name: "sealing face", kind: "face", tolerance_mm: null, critical: true },
            ],
            assumptions: ["No numeric tolerance was given for the sealing face."],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const root = document.createElement("main");
    document.body.append(root);
    mountBuyerPage(root, { baseUrl: "", locationHref: "https://example.test/buyer.html" });

    const key = required(root, "#buyer-key") as HTMLInputElement;
    key.value = "buyer-key";
    key.dispatchEvent(new Event("input"));
    (required(root, "#buyer-prose") as HTMLTextAreaElement).value =
      "Mounting holes within 2 mm; sealing face flat.";
    (required(root, "#parse-spec") as HTMLButtonElement).click();

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".feature-row")).toHaveLength(2);
    });
    const tolerances = root.querySelectorAll<HTMLInputElement>('[data-field="tolerance"]');
    expect(tolerances[1]?.value).toBe("");

    (required(root, "#create-order") as HTMLButtonElement).click();

    expect(root.querySelector("#buyer-status")?.textContent).toBe(
      "Feature 2 needs a tolerance greater than 0 mm.",
    );
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy.mock.calls[0]?.[0]).toBe("/v1/spec/parse");
  });

  it("appends parsed features instead of deleting buyer-entered rows", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          status: "parsed",
          spec: {
            features: [{ name: "parsed hole", kind: "hole", tolerance_mm: 2, critical: true }],
            assumptions: [],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const root = document.createElement("main");
    document.body.append(root);
    mountBuyerPage(root, { baseUrl: "", locationHref: "https://example.test/buyer.html" });

    const key = required(root, "#buyer-key") as HTMLInputElement;
    key.value = "buyer-key";
    key.dispatchEvent(new Event("input"));
    const manualName = required(root, '[data-field="name"]') as HTMLInputElement;
    manualName.value = "manual face";
    manualName.dispatchEvent(new Event("input", { bubbles: true }));
    const manualTolerance = required(root, '[data-field="tolerance"]') as HTMLInputElement;
    manualTolerance.value = "1";
    manualTolerance.dispatchEvent(new Event("input", { bubbles: true }));
    (required(root, "#buyer-prose") as HTMLTextAreaElement).value = "one hole within 2 mm";
    (required(root, "#parse-spec") as HTMLButtonElement).click();

    await vi.waitFor(() => {
      expect(root.querySelectorAll(".feature-row")).toHaveLength(2);
    });
    const names = root.querySelectorAll<HTMLInputElement>('[data-field="name"]');
    expect(names[0]?.value).toBe("manual face");
    expect(names[1]?.value).toBe("parsed hole");
  });

  it("does not expose raw network errors from buyer actions", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    const root = document.createElement("main");
    document.body.append(root);
    mountBuyerPage(root, { baseUrl: "", locationHref: "https://example.test/buyer.html" });

    const key = required(root, "#buyer-key") as HTMLInputElement;
    key.value = "buyer-key";
    key.dispatchEvent(new Event("input"));
    (required(root, "#buyer-prose") as HTMLTextAreaElement).value = "one hole within 2 mm";
    (required(root, "#parse-spec") as HTMLButtonElement).click();

    await vi.waitFor(() => {
      expect(root.querySelector("#buyer-status")?.textContent).toContain(
        "No connection to the service",
      );
    });
    expect(root.querySelector("#buyer-status")?.textContent).not.toContain("Failed to fetch");
  });

  it("shows service failure without clearing the form", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ code: "buyer_auth_required", message: "Wrong buyer key." }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const root = document.createElement("main");
    document.body.append(root);
    mountBuyerPage(root, { baseUrl: "", locationHref: "https://example.test/buyer.html" });

    const key = required(root, "#buyer-key") as HTMLInputElement;
    key.value = "wrong";
    key.dispatchEvent(new Event("input"));
    (required(root, '[data-field="name"]') as HTMLInputElement).value = "hole A";
    (required(root, '[data-field="tolerance"]') as HTMLInputElement).value = "2";
    (required(root, "#create-order") as HTMLButtonElement).click();
    await vi.waitFor(() => {
      expect(root.querySelector("#buyer-status")?.textContent).toBe("Wrong buyer key.");
    });
    expect((required(root, '[data-field="name"]') as HTMLInputElement).value).toBe("hole A");
  });

  it("shows order code, CAD size and capture link on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          order_code: "PO-7K2M9Q4R",
          created_at: "2026-11-15T03:15:00+00:00",
          has_cad: true,
          cad_units: "mm",
          cad_bounding_box_mm: [
            [0, 0, 0],
            [50, 32, 20],
          ],
          spec: {},
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );
    const root = document.createElement("main");
    document.body.append(root);
    mountBuyerPage(root, {
      baseUrl: "",
      locationHref: "https://example.test/proofshape/buyer.html",
    });

    const key = required(root, "#buyer-key") as HTMLInputElement;
    key.value = "buyer-key";
    key.dispatchEvent(new Event("input"));
    (required(root, '[data-field="name"]') as HTMLInputElement).value = "overall";
    (required(root, '[data-field="tolerance"]') as HTMLInputElement).value = "1";
    (required(root, "#create-order") as HTMLButtonElement).click();

    await vi.waitFor(() => {
      expect(root.querySelector("#buyer-result")?.textContent).toContain("PO-7K2M9Q4R");
    });
    expect(root.querySelector("#buyer-result")?.textContent).toContain("50.00 × 32.00 × 20.00 mm");
    expect(root.querySelector<HTMLAnchorElement>("#buyer-result a")?.href).toBe(
      "https://example.test/proofshape/?order=PO-7K2M9Q4R",
    );
  });
});
