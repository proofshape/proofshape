import { PDFDocument, StandardFonts, rgb } from "pdf-lib";
import encodeQR from "qr";

export const BOARD_SHA256 =
  "1b64d0328a7551da20fee6ccf5c6edacd5f48858be4004dffb093a60c9c9a354";

export const MM_TO_PT = 72 / 25.4;
export const mmToPt = (mm: number): number => mm * MM_TO_PT;

export interface RectMm {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const ORDER_SHEET_LAYOUT_MM = {
  page: { x: 0, y: 0, width: 297, height: 210 },
  printable: { x: 5, y: 5, width: 287, height: 200 },
  boardPage: { x: 5, y: 30, width: 200, height: 150 },
  boardPattern: { x: 25, y: 45, width: 160, height: 120 },
  orderText: { x: 214, y: 171, width: 73, height: 24 },
  qr: { x: 224, y: 105, width: 52, height: 52 },
  arrow: { x: 220, y: 72, width: 58, height: 18 },
  scaleNote: { x: 214, y: 18, width: 73, height: 18 },
} as const satisfies Record<string, RectMm>;

const FIXED_PDF_DATE = new Date("2026-11-16T00:00:00.000Z");

export function rectsOverlap(a: RectMm, b: RectMm): boolean {
  return !(
    a.x + a.width <= b.x ||
    b.x + b.width <= a.x ||
    a.y + a.height <= b.y ||
    b.y + b.height <= a.y
  );
}

export function rectInside(inner: RectMm, outer: RectMm): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  );
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join(
    "",
  );
}

export async function verifyBoardFixture(bytes: Uint8Array): Promise<void> {
  const actual = await sha256Hex(bytes);
  if (actual !== BOARD_SHA256) {
    throw new Error(
      "The calibration-board file changed. Refusing to print until its SHA-256 is reviewed.",
    );
  }
}

function drawQr(page: ReturnType<PDFDocument["addPage"]>, orderCode: string): void {
  const matrix = encodeQR(orderCode, "raw", { ecc: "medium", border: 4 });
  const box = ORDER_SHEET_LAYOUT_MM.qr;
  const moduleMm = box.width / matrix.length;

  for (const [rowIndex, row] of matrix.entries()) {
    for (const [columnIndex, dark] of row.entries()) {
      if (!dark) continue;
      page.drawRectangle({
        x: mmToPt(box.x + columnIndex * moduleMm),
        y: mmToPt(box.y + (matrix.length - rowIndex - 1) * moduleMm),
        width: mmToPt(moduleMm),
        height: mmToPt(moduleMm),
        color: rgb(0, 0, 0),
        borderWidth: 0,
      });
    }
  }
}

export async function createOrderSheetPdf(
  boardPdfBytes: Uint8Array,
  orderCode: string,
): Promise<Uint8Array> {
  const normalizedCode = orderCode.trim();
  if (!normalizedCode) throw new Error("An order code is required to print the sheet.");

  await verifyBoardFixture(boardPdfBytes);

  const output = await PDFDocument.create();
  output.setTitle(`ProofShape order sheet ${normalizedCode}`);
  output.setSubject("ProofShape calibration and capture sheet");
  output.setCreator("ProofShape");
  output.setProducer("ProofShape C-17");
  output.setCreationDate(FIXED_PDF_DATE);
  output.setModificationDate(FIXED_PDF_DATE);

  const page = output.addPage([
    mmToPt(ORDER_SHEET_LAYOUT_MM.page.width),
    mmToPt(ORDER_SHEET_LAYOUT_MM.page.height),
  ]);
  const [boardPage] = await output.embedPdf(boardPdfBytes, [0]);
  if (boardPage === undefined) throw new Error("The calibration-board PDF has no first page.");

  const expectedBoardWidth = mmToPt(ORDER_SHEET_LAYOUT_MM.boardPage.width);
  const expectedBoardHeight = mmToPt(ORDER_SHEET_LAYOUT_MM.boardPage.height);
  if (
    Math.abs(boardPage.width - expectedBoardWidth) > 0.05 ||
    Math.abs(boardPage.height - expectedBoardHeight) > 0.05
  ) {
    throw new Error("The calibration-board PDF no longer measures 200 × 150 mm.");
  }

  page.drawPage(boardPage, {
    x: mmToPt(ORDER_SHEET_LAYOUT_MM.boardPage.x),
    y: mmToPt(ORDER_SHEET_LAYOUT_MM.boardPage.y),
    width: boardPage.width,
    height: boardPage.height,
  });

  const regular = await output.embedFont(StandardFonts.Helvetica);
  const bold = await output.embedFont(StandardFonts.HelveticaBold);

  page.drawText("ProofShape order", {
    x: mmToPt(214),
    y: mmToPt(190),
    size: 10,
    font: regular,
    color: rgb(0, 0, 0),
  });
  page.drawText(normalizedCode, {
    x: mmToPt(214),
    y: mmToPt(181),
    size: 15,
    font: bold,
    color: rgb(0, 0, 0),
  });

  drawQr(page, normalizedCode);

  const arrowY = mmToPt(80);
  const arrowStartX = mmToPt(220);
  const arrowEndX = mmToPt(278);
  page.drawText("Board +X", {
    x: mmToPt(220),
    y: mmToPt(87),
    size: 9,
    font: bold,
    color: rgb(0, 0, 0),
  });
  page.drawLine({
    start: { x: arrowStartX, y: arrowY },
    end: { x: arrowEndX, y: arrowY },
    thickness: 2,
    color: rgb(0, 0, 0),
  });
  page.drawLine({
    start: { x: arrowEndX, y: arrowY },
    end: { x: mmToPt(271), y: mmToPt(84) },
    thickness: 2,
    color: rgb(0, 0, 0),
  });
  page.drawLine({
    start: { x: arrowEndX, y: arrowY },
    end: { x: mmToPt(271), y: mmToPt(76) },
    thickness: 2,
    color: rgb(0, 0, 0),
  });
  page.drawText("Arrow points along the board square columns.", {
    x: mmToPt(214),
    y: mmToPt(65),
    size: 7,
    font: regular,
    color: rgb(0, 0, 0),
  });

  page.drawText("Print at Scale: 100%", {
    x: mmToPt(214),
    y: mmToPt(28),
    size: 11,
    font: bold,
    color: rgb(0, 0, 0),
  });
  page.drawText("Do not use Fit to Page / Scale to Fit.", {
    x: mmToPt(214),
    y: mmToPt(21),
    size: 7,
    font: regular,
    color: rgb(0, 0, 0),
  });

  return output.save({ useObjectStreams: false });
}

export async function loadOrderSheetPdf(boardUrl: string, orderCode: string): Promise<Uint8Array> {
  const response = await fetch(boardUrl);
  if (!response.ok) {
    throw new Error(`Could not load the calibration board (HTTP ${String(response.status)}).`);
  }
  return createOrderSheetPdf(new Uint8Array(await response.arrayBuffer()), orderCode);
}

export function downloadPdfBytes(bytes: Uint8Array, filename: string): void {
  const blob = new Blob([bytes], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function downloadOrderSheet(boardUrl: string, orderCode: string): Promise<void> {
  const bytes = await loadOrderSheetPdf(boardUrl, orderCode);
  downloadPdfBytes(bytes, `proofshape-${orderCode}-sheet.pdf`);
}
