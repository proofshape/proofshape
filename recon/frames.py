"""Shared capture-frame listing, used by every recon module that reads a golden-capture folder.

Was duplicated identically in ``board_pose.py`` (R-02) and ``vggt_runner.py`` (R-01) — each
landed with the other not yet merged, so there was nothing to import from. Consolidated once
both were on ``main`` (issue #43).
"""

from __future__ import annotations

from pathlib import Path

FRAME_SUFFIXES = (".jpg", ".jpeg")


def find_capture_frames(capture_dir: Path) -> list[Path]:
    """Return the capture's JPEG frames in a stable, sorted order.

    Sorting matters for reproducibility in more than one way: R-01's VGGT treats the first frame
    as the reference frame, so a filesystem-dependent order would give a different world frame
    from run to run; and R-04 pairs R-01's poses with R-02's board poses by frame name, so a
    stable order keeps every recon module's output rows aligned against the others.
    """
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise FileNotFoundError(
            f"Capture folder not found: {capture_dir}. "
            "Run `bash fixtures/download_golden_capture.sh` first."
        )

    frames: list[Path] = []
    for path in sorted(capture_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES:
            frames.append(path)

    if not frames:
        raise FileNotFoundError(f"No .jpg/.jpeg frames found in {capture_dir}.")
    return frames
