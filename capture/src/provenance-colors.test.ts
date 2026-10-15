// @vitest-environment happy-dom
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  PROVENANCE_DISPLAY_COLORS,
  PROVENANCE_SOURCE_COLORS,
  createProvenanceLegend,
  mapProvenanceRgb,
} from "./provenance-colors.js";

describe("C-12 provenance colors", () => {
  it("maps exact Observed and Unobserved source RGB values to the display palette", () => {
    expect(mapProvenanceRgb(PROVENANCE_SOURCE_COLORS.observed)).toEqual({
      kind: "observed",
      rgb: PROVENANCE_DISPLAY_COLORS.observed,
    });
    expect(mapProvenanceRgb(PROVENANCE_SOURCE_COLORS.unobserved)).toEqual({
      kind: "unobserved",
      rgb: PROVENANCE_DISPLAY_COLORS.unobserved,
    });
  });

  it("maps every other source RGB to a named unknown display case", () => {
    expect(mapProvenanceRgb([255, 128, 0])).toEqual({
      kind: "unknown",
      rgb: PROVENANCE_DISPLAY_COLORS.unknown,
    });
  });

  it("reserves an inferred slot in the visible legend", () => {
    const legend = createProvenanceLegend();
    expect(legend.textContent).toContain("Observed");
    expect(legend.textContent).toContain("Unobserved");
    expect(legend.textContent).toContain("Inferred (reserved)");
  });

  it("stays equal to recon/provenance.py's committed source-color constants", () => {
    const source = readFileSync(new URL("../../recon/provenance.py", import.meta.url), "utf8");
    const observed = source.match(/OBSERVED_COLOR\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)/);
    const unobserved = source.match(/UNOBSERVED_COLOR\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)/);

    expect(observed).not.toBeNull();
    expect(unobserved).not.toBeNull();
    expect(observed?.slice(1).map(Number)).toEqual([...PROVENANCE_SOURCE_COLORS.observed]);
    expect(unobserved?.slice(1).map(Number)).toEqual([...PROVENANCE_SOURCE_COLORS.unobserved]);
  });
});
