import { createOrder, parseInspectionSpec } from "./api.js";
import type { InspectionSpec, SpecFeature, SpecKind } from "./types.js";

const SPEC_KINDS: SpecKind[] = ["hole", "face", "edge", "overall"];
const BUYER_KEY_STORAGE = "proofshape_buyer_key";

export interface BuyerPageOptions {
  baseUrl: string;
  authToken?: string;
  locationHref?: string;
}

export interface FeatureDraft {
  name: string;
  kind: SpecKind;
  tolerance: string;
  critical: boolean;
}

export function buildInspectionSpec(
  rows: FeatureDraft[],
  assumptionsText: string,
): { ok: true; spec: InspectionSpec } | { ok: false; message: string } {
  const features: SpecFeature[] = [];
  for (const [index, row] of rows.entries()) {
    const name = row.name.trim();
    if (!name) {
      return { ok: false, message: `Feature ${String(index + 1)} needs a name.` };
    }
    if (!SPEC_KINDS.includes(row.kind)) {
      return { ok: false, message: `Feature ${String(index + 1)} has an invalid kind.` };
    }
    const tolerance = Number(row.tolerance);
    if (!row.tolerance.trim() || !Number.isFinite(tolerance) || tolerance <= 0) {
      return {
        ok: false,
        message: `Feature ${String(index + 1)} needs a tolerance greater than 0 mm.`,
      };
    }
    features.push({
      name,
      kind: row.kind,
      tolerance_mm: tolerance,
      critical: row.critical,
    });
  }

  if (features.length === 0) {
    return { ok: false, message: "Add at least one inspection feature." };
  }

  const assumptions = assumptionsText
    .split(/\r?\n/)
    .map((value) => value.trim())
    .filter(Boolean);

  return { ok: true, spec: { features, assumptions } };
}

export function validateCadFile(file: File | undefined): string | null {
  if (file === undefined) return null;
  const dot = file.name.lastIndexOf(".");
  const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  return [".stl", ".obj", ".glb"].includes(suffix)
    ? null
    : "CAD file must be STL, OBJ, or GLB.";
}

export function captureLinkFor(locationHref: string, orderCode: string): string {
  const url = new URL(locationHref);
  url.pathname = url.pathname.replace(/buyer\.html$/, "");
  if (!url.pathname.endsWith("/")) url.pathname += "/";
  url.search = "";
  url.hash = "";
  url.searchParams.set("order", orderCode);
  return url.toString();
}

function cadDimensions(box: [number[], number[]] | undefined): string {
  if (box === undefined || box.length !== 2) return "No CAD uploaded";
  const dims = [0, 1, 2].map((index) => (box[1]?.[index] ?? 0) - (box[0]?.[index] ?? 0));
  return `${dims.map((value) => value.toFixed(2)).join(" × ")} mm`;
}

function featureRow(draft: FeatureDraft): HTMLTableRowElement {
  const tr = document.createElement("tr");
  tr.className = "feature-row";

  const nameCell = document.createElement("td");
  const name = document.createElement("input");
  name.type = "text";
  name.value = draft.name;
  name.dataset.field = "name";
  name.setAttribute("aria-label", "Feature name");
  nameCell.append(name);

  const kindCell = document.createElement("td");
  const kind = document.createElement("select");
  kind.dataset.field = "kind";
  kind.setAttribute("aria-label", "Feature kind");
  for (const value of SPEC_KINDS) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === draft.kind;
    kind.append(option);
  }
  kindCell.append(kind);

  const toleranceCell = document.createElement("td");
  const tolerance = document.createElement("input");
  tolerance.type = "number";
  tolerance.min = "0";
  tolerance.step = "any";
  tolerance.value = draft.tolerance;
  tolerance.placeholder = "mm";
  tolerance.dataset.field = "tolerance";
  tolerance.setAttribute("aria-label", "Tolerance mm");
  toleranceCell.append(tolerance);

  const criticalCell = document.createElement("td");
  const critical = document.createElement("input");
  critical.type = "checkbox";
  critical.checked = draft.critical;
  critical.dataset.field = "critical";
  critical.setAttribute("aria-label", "Critical feature");
  criticalCell.append(critical);

  const removeCell = document.createElement("td");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "Remove";
  remove.dataset.action = "remove-feature";
  removeCell.append(remove);

  tr.append(nameCell, kindCell, toleranceCell, criticalCell, removeCell);
  return tr;
}

function readFeatureRows(tbody: HTMLTableSectionElement): FeatureDraft[] {
  return Array.from(tbody.querySelectorAll<HTMLTableRowElement>(".feature-row")).map((row) => ({
    name: row.querySelector<HTMLInputElement>('[data-field="name"]')?.value ?? "",
    kind:
      (row.querySelector<HTMLSelectElement>('[data-field="kind"]')?.value as SpecKind | undefined) ??
      "overall",
    tolerance: row.querySelector<HTMLInputElement>('[data-field="tolerance"]')?.value ?? "",
    critical: row.querySelector<HTMLInputElement>('[data-field="critical"]')?.checked ?? false,
  }));
}

function replaceFeatureRows(tbody: HTMLTableSectionElement, features: SpecFeature[]): void {
  tbody.replaceChildren(
    ...features.map((feature) =>
      featureRow({
        name: feature.name,
        kind: feature.kind,
        tolerance: feature.tolerance_mm === null ? "" : String(feature.tolerance_mm),
        critical: feature.critical,
      }),
    ),
  );
}

export function mountBuyerPage(root: HTMLElement, options: BuyerPageOptions): void {
  root.innerHTML = `
    <section class="buyer-shell">
      <header>
        <p>ProofShape buyer setup</p>
        <h1>Create an inspection order</h1>
        <p class="buyer-warning">
          Tolerances are not yet checked against the method's measured noise floor. Entering a
          tolerance here does not mean ProofShape can verify it.
        </p>
      </header>
      <div class="buyer-grid">
        <label class="buyer-field">
          Buyer key
          <input id="buyer-key" type="password" autocomplete="off" />
          <small>Stored only in this tab's sessionStorage; never built into the app.</small>
        </label>

        <label class="buyer-field">
          CAD file (optional)
          <input id="buyer-cad" type="file" accept=".stl,.obj,.glb" />
          <small>No CAD creates a model-only order with no inspection verdict.</small>
        </label>

        <label class="buyer-field">
          Describe what matters (optional AI assist)
          <textarea id="buyer-prose" rows="3" placeholder="The four mounting holes need to line up within about 2 mm..."></textarea>
        </label>
        <div class="buyer-actions">
          <button id="parse-spec" type="button">Parse into editable table</button>
          <button id="add-feature" type="button">Add feature</button>
        </div>

        <div>
          <table class="feature-table">
            <thead><tr><th>Name</th><th>Kind</th><th>Tolerance (mm)</th><th>Critical</th><th></th></tr></thead>
            <tbody id="feature-rows"></tbody>
          </table>
        </div>

        <label class="buyer-field">
          Assumptions (one per line)
          <textarea id="buyer-assumptions" rows="3"></textarea>
        </label>

        <label class="buyer-field">
          Placement note
          <input id="placement-note" type="text" placeholder="Hole A toward the sheet's arrow" />
        </label>

        <label class="buyer-field">
          Buyer reference
          <input id="buyer-reference" type="text" placeholder="Your PO or internal reference" />
        </label>

        <div class="buyer-actions">
          <button id="create-order" type="button">Create order</button>
        </div>
        <p id="buyer-status" class="buyer-status" role="status">Ready.</p>
        <section id="buyer-result" class="buyer-result" hidden></section>
      </div>
    </section>
  `;

  const keyInput = root.querySelector<HTMLInputElement>("#buyer-key")!;
  const cadInput = root.querySelector<HTMLInputElement>("#buyer-cad")!;
  const proseInput = root.querySelector<HTMLTextAreaElement>("#buyer-prose")!;
  const assumptionsInput = root.querySelector<HTMLTextAreaElement>("#buyer-assumptions")!;
  const placementInput = root.querySelector<HTMLInputElement>("#placement-note")!;
  const referenceInput = root.querySelector<HTMLInputElement>("#buyer-reference")!;
  const tbody = root.querySelector<HTMLTableSectionElement>("#feature-rows")!;
  const status = root.querySelector<HTMLElement>("#buyer-status")!;
  const result = root.querySelector<HTMLElement>("#buyer-result")!;
  const locationHref = options.locationHref ?? window.location.href;

  keyInput.value = sessionStorage.getItem(BUYER_KEY_STORAGE) ?? "";
  keyInput.addEventListener("input", () => {
    if (keyInput.value) sessionStorage.setItem(BUYER_KEY_STORAGE, keyInput.value);
    else sessionStorage.removeItem(BUYER_KEY_STORAGE);
  });

  tbody.append(featureRow({ name: "", kind: "overall", tolerance: "", critical: false }));

  root.querySelector("#add-feature")?.addEventListener("click", () => {
    tbody.append(featureRow({ name: "", kind: "overall", tolerance: "", critical: false }));
  });

  tbody.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.dataset.action === "remove-feature") {
      target.closest("tr")?.remove();
    }
  });

  root.querySelector("#parse-spec")?.addEventListener("click", async () => {
    const buyerKey = keyInput.value.trim();
    const prose = proseInput.value.trim();
    if (!buyerKey) {
      status.textContent = "Enter the buyer key before using the parser.";
      return;
    }
    if (!prose) {
      status.textContent = "Describe what matters before parsing.";
      return;
    }

    status.textContent = "Parsing requirements…";
    try {
      const parsed = await parseInspectionSpec(options.baseUrl, prose, buyerKey, options.authToken);
      if (!parsed.ok) {
        status.textContent = parsed.error.message;
        return;
      }
      if (parsed.body.status === "unparsed") {
        status.textContent = `Could not parse requirements (${parsed.body.reason}). Use the manual table.`;
        return;
      }

      replaceFeatureRows(tbody, parsed.body.spec.features);
      assumptionsInput.value = parsed.body.spec.assumptions.join("\n");
      const hasMissing = parsed.body.spec.features.some((feature) => feature.tolerance_mm === null);
      status.textContent = hasMissing
        ? "Parsed. Fill every blank tolerance before creating the order."
        : "Parsed. Review and edit every field before creating the order.";
    } catch (error) {
      status.textContent = `Parser unavailable: ${String(error)}. Use the manual table.`;
    }
  });

  root.querySelector("#create-order")?.addEventListener("click", async () => {
    const buyerKey = keyInput.value.trim();
    if (!buyerKey) {
      status.textContent = "Enter the buyer key before creating an order.";
      return;
    }

    const file = cadInput.files?.[0];
    const cadError = validateCadFile(file);
    if (cadError !== null) {
      status.textContent = cadError;
      return;
    }

    const specResult = buildInspectionSpec(readFeatureRows(tbody), assumptionsInput.value);
    if (!specResult.ok) {
      status.textContent = specResult.message;
      return;
    }

    status.textContent = "Creating order…";
    result.hidden = true;
    try {
      const created = await createOrder(
        options.baseUrl,
        {
          spec: specResult.spec,
          ...(file !== undefined ? { cad: file } : {}),
          placementNote: placementInput.value,
          buyerReference: referenceInput.value,
        },
        buyerKey,
        options.authToken,
      );
      if (!created.ok) {
        status.textContent = created.error.message;
        return;
      }

      const captureLink = captureLinkFor(locationHref, created.body.order_code);
      result.replaceChildren();

      const heading = document.createElement("h2");
      heading.textContent = "Order created";

      const code = document.createElement("p");
      code.append("Order code: ");
      const codeValue = document.createElement("code");
      codeValue.textContent = created.body.order_code;
      code.append(codeValue);

      const size = document.createElement("p");
      size.textContent = `CAD size: ${cadDimensions(created.body.cad_bounding_box_mm)}`;

      const link = document.createElement("a");
      link.href = captureLink;
      link.textContent = "Open supplier capture link";

      result.append(heading, code, size, link);
      result.hidden = false;
      status.textContent = "Order created. Review the code and capture link below.";
    } catch (error) {
      status.textContent = `Could not create order: ${String(error)}`;
    }
  });
}
