import { describe, expect, it } from "vitest";
import { parseGyroInput } from "./gyro.js";

describe("parseGyroInput", () => {
  it("parses three valid numeric strings", () => {
    expect(parseGyroInput({ alpha: "132.4", beta: "-12.1", gamma: "3.7" })).toEqual({
      alpha: 132.4,
      beta: -12.1,
      gamma: 3.7,
    });
  });

  it("trims surrounding whitespace", () => {
    expect(parseGyroInput({ alpha: " 1 ", beta: "2", gamma: "3" })).toEqual({
      alpha: 1,
      beta: 2,
      gamma: 3,
    });
  });

  it("rejects an empty field by name", () => {
    expect(() => parseGyroInput({ alpha: "", beta: "2", gamma: "3" })).toThrow(/"alpha"/);
  });

  it("rejects a non-numeric field by name and value", () => {
    expect(() => parseGyroInput({ alpha: "1", beta: "not a number", gamma: "3" })).toThrow(
      /"beta".*"not a number"/,
    );
  });

  it("rejects NaN produced by whitespace-only input", () => {
    expect(() => parseGyroInput({ alpha: "1", beta: "2", gamma: "   " })).toThrow(/"gamma"/);
  });
});
