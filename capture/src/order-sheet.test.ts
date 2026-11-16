import { readFile, writeFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  BOARD_SHA256,
  ORDER_SHEET_LAYOUT_MM,
  createOrderSheetPdf,
  rectInside,
  rectsOverlap,
  sha256Hex,
} from "./order-sheet.js";

const BOARD_PATH = new URL("../public/charuco_board.pdf", import.meta.url);
const SAMPLE_PATH = new URL("../../fixtures/order_sheet_PO-48213-A.pdf", import.meta.url);
const SAMPLE_ORDER = "PO-48213-A";

describe("C-17 order sheet", () => {
  it("pins the browser board copy to the committed fixture hash", async () => {
    const board = new Uint8Array(await readFile(BOARD_PATH));
    expect(await sha256Hex(board)).toBe(BOARD_SHA256);
  });

  it("keeps every added mark inside the printable area and outside the board pattern", () => {
    const { printable, boardPattern, orderText, qr, arrow, scaleNote } = ORDER_SHEET_LAYOUT_MM;
    for (const mark of [orderText, qr, arrow, scaleNote]) {
      expect(rectInside(mark, printable)).toBe(true);
      expect(rectsOverlap(mark, boardPattern)).toBe(false);
    }
  });

  it("keeps the whole 200 × 150 mm board page at physical 1:1 inside A4", () => {
    const { page, boardPage } = ORDER_SHEET_LAYOUT_MM;
    expect(boardPage).toEqual({ x: 5, y: 30, width: 200, height: 150 });
    expect(rectInside(boardPage, page)).toBe(true);
  });

  it("regenerates the committed sample byte-for-byte", async () => {
    const board = new Uint8Array(await readFile(BOARD_PATH));
    const generated = await createOrderSheetPdf(board, SAMPLE_ORDER);

    if (process.env.PROOFSHAPE_WRITE_C17_SAMPLE === "1") {
      await writeFile(SAMPLE_PATH, generated);
      return;
    }

    const committed = new Uint8Array(await readFile(SAMPLE_PATH));
    expect(generated).toEqual(committed);
  });
});
