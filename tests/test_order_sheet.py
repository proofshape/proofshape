"""C-17 raster checks for the committed per-order printable sheet."""

from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest

from recon.board_pose import (
    BOARD_INTERIOR_CORNERS,
    detect_charuco_corners,
    make_board,
    make_detector,
)

SAMPLE_PDF = (
    Path(__file__).resolve().parents[1] / "fixtures" / "order_sheet_PO-48213-A.pdf"
)
ORDER_CODE = "PO-48213-A"
RASTER_SCALE = 2.0


def render_sample() -> np.ndarray:
    pdf = pdfium.PdfDocument(str(SAMPLE_PDF))
    try:
        page = pdf[0]
        width_pt, height_pt = page.get_size()
        assert width_pt == pytest.approx(297.0 * 72.0 / 25.4, rel=1e-4)
        assert height_pt == pytest.approx(210.0 * 72.0 / 25.4, rel=1e-4)
        return page.render(scale=RASTER_SCALE, grayscale=True).to_numpy()
    finally:
        pdf.close()


def test_order_sheet_preserves_board_scale_ids_and_qr() -> None:
    image = render_sample()
    board = make_board()
    detector = make_detector(board)

    corner_px, corner_id, marker_count = detect_charuco_corners(image, detector)
    assert len(corner_id) == BOARD_INTERIOR_CORNERS
    assert marker_count == len(board.getIds())

    px_per_mm = RASTER_SCALE * 72.0 / 25.4
    corner_by_id = dict(zip(corner_id.tolist(), corner_px))
    pitch_mm = np.linalg.norm(corner_by_id[1] - corner_by_id[0]) / px_per_mm
    assert pitch_mm == pytest.approx(20.0, rel=0.01)

    _charuco, _charuco_ids, _marker_corners, marker_ids = detector.detectBoard(image)
    assert marker_ids is not None
    assert set(marker_ids.reshape(-1).tolist()) == set(
        np.asarray(board.getIds()).reshape(-1).tolist()
    )

    decoded, points, _straight = cv2.QRCodeDetector().detectAndDecode(image)
    assert points is not None
    assert decoded == ORDER_CODE
