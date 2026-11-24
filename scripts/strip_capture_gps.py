#!/usr/bin/env python3
"""G-14: copy capture photos with their EXIF GPS block emptied, and nothing else changed.

The golden photos (s-01 to s-05) carry the GPS position they were taken at. The pipeline reads
two other EXIF fields itself, the 35 mm-equivalent focal length and the orientation
(`recon/board_pose.py`, `read_exif_fields`), so stripping all EXIF, or re-encoding, would change
calibration. This script empties only the GPS block, in place within each copy's bytes:

- the GPS IFD's entries and every value they point to are overwritten with zero bytes;
- the IFD is left as a valid empty one (entry count 0, no next IFD), still pointed to by IFD0;
- no byte is inserted or removed, so every other offset, the image data, the embedded thumbnail
  and gain-map image, and the file's length are unchanged.

It needs nothing outside the standard library. It never edits its input: copies go to a new
folder. A JPEG with no GPS block is copied byte for byte. Anything that isn't a JPEG is refused,
so a folder of photos can't be half-converted without notice.

Usage:
    python3 scripts/strip_capture_gps.py <source folder> <new destination folder>

It mirrors the source's subfolders. At the end it prints, per folder of JPEGs, the content hash
that `fixtures/download_golden_capture.sh` would compute (`sample_content_hash`) and the byte
total, for the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

JPEG_SUFFIXES = {".jpg", ".jpeg"}
EXIF_HEADER = b"Exif\x00\x00"
GPS_IFD_TAG = 0x8825
APP1 = 0xE1
START_OF_SCAN = 0xDA

# TIFF field type -> bytes per value (TIFF 6.0 and EXIF 2.3).
TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
}


class NotAJpeg(ValueError):
    """The input doesn't start like a JPEG; the script refuses it rather than guess."""


@dataclass(frozen=True)
class StripResult:
    data: bytes
    gps_entries_removed: int
    blanked_ranges: tuple[tuple[int, int], ...]  # [start, end) file offsets set to zero


def _find_exif_tiff_start(data: bytes) -> int | None:
    """File offset of the TIFF header inside the first APP1 Exif segment, or None."""
    if not data.startswith(b"\xff\xd8"):
        raise NotAJpeg("not a JPEG: it doesn't start with the SOI marker")
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            raise ValueError(f"malformed JPEG: expected a marker at byte {position}")
        marker = data[position + 1]
        if marker == START_OF_SCAN:
            return None  # metadata segments all come before the scan
        if marker == 0xFF:  # fill byte
            position += 1
            continue
        length = struct.unpack(">H", data[position + 2 : position + 4])[0]
        body = position + 4
        if marker == APP1 and data[body : body + 6] == EXIF_HEADER:
            return body + 6
        position += 2 + length
    return None


def strip_gps(data: bytes) -> StripResult:
    """Return `data` with its EXIF GPS IFD emptied in place (see the module docstring)."""
    tiff = _find_exif_tiff_start(data)
    if tiff is None:
        return StripResult(data, 0, ())
    order = data[tiff : tiff + 2]
    if order == b"II":
        endian = "<"
    elif order == b"MM":
        endian = ">"
    else:
        raise ValueError("malformed EXIF: unknown TIFF byte order")

    def u16(offset: int) -> int:
        return struct.unpack(endian + "H", data[offset : offset + 2])[0]

    def u32(offset: int) -> int:
        return struct.unpack(endian + "I", data[offset : offset + 4])[0]

    ifd0 = tiff + u32(tiff + 4)
    gps_ifd = None
    for index in range(u16(ifd0)):
        entry = ifd0 + 2 + 12 * index
        if u16(entry) == GPS_IFD_TAG:
            gps_ifd = tiff + u32(entry + 8)
            break
    if gps_ifd is None:
        return StripResult(data, 0, ())

    entry_count = u16(gps_ifd)
    ifd_end = gps_ifd + 2 + 12 * entry_count + 4
    if ifd_end > len(data):
        raise ValueError("malformed EXIF: the GPS IFD runs past the end of the file")
    ranges: list[tuple[int, int]] = []
    for index in range(entry_count):
        entry = gps_ifd + 2 + 12 * index
        field_type, count = u16(entry + 2), u32(entry + 4)
        size = TYPE_SIZES.get(field_type, 1) * count
        if size > 4:  # the value lives elsewhere; the entry holds its offset
            start = tiff + u32(entry + 8)
            if start + size > len(data):
                raise ValueError(
                    "malformed EXIF: a GPS value runs past the end of the file"
                )
            ranges.append((start, start + size))
    ranges.append((gps_ifd, ifd_end))  # count 0 and next-IFD 0: a valid empty IFD

    out = bytearray(data)
    for start, end in ranges:
        out[start:end] = bytes(end - start)
    return StripResult(bytes(out), entry_count, tuple(sorted(ranges)))


def sample_content_hash(folder: Path) -> str:
    """The downloader's `sample_content_hash`: sha256 over "<sha256>  <basename>" lines, in path
    order, for the folder's JPEGs."""
    files = sorted(
        (
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in JPEG_SUFFIXES
        ),
        key=lambda p: str(p).encode(),
    )
    lines = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files
    )
    return hashlib.sha256(lines.encode()).hexdigest()


def strip_folder(source: Path, destination: Path) -> dict:
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f"{source} is not a folder")
    if destination.exists():
        raise ValueError(
            f"{destination} already exists; give a new folder so nothing is overwritten"
        )
    files = sorted(p for p in source.rglob("*") if p.is_file())
    others = [
        p
        for p in files
        if p.suffix.lower() not in JPEG_SUFFIXES and p.name != ".DS_Store"
    ]
    if others:
        listed = ", ".join(str(p.relative_to(source)) for p in others[:5])
        raise NotAJpeg(f"refusing a folder with files that aren't JPEGs: {listed}")

    report: dict = {"files": 0, "had_gps": 0, "folders": {}}
    for path in files:
        if path.name == ".DS_Store":
            continue
        result = strip_gps(path.read_bytes())
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.data)
        report["files"] += 1
        report["had_gps"] += result.gps_entries_removed > 0
    for folder in sorted({p.parent for p in destination.rglob("*") if p.is_file()}):
        jpegs = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in JPEG_SUFFIXES
        ]
        report["folders"][str(folder.relative_to(destination)) or "."] = {
            "frames": len(jpegs),
            "aggregate_bytes": sum(p.stat().st_size for p in jpegs),
            "content_sha256": sample_content_hash(folder),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "destination", type=Path, help="a new folder; it must not exist yet"
    )
    args = parser.parse_args(argv)
    try:
        report = strip_folder(args.source, args.destination)
    except ValueError as exc:
        print(f"G-14 STRIP GPS: FAIL - {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
