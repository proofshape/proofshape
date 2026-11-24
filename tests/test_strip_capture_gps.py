"""G-14: the GPS block goes, and nothing else in the photo changes."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recon.board_pose import read_exif_fields

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "strip_capture_gps.py"
_spec = importlib.util.spec_from_file_location("strip_capture_gps", SCRIPT)
strip = importlib.util.module_from_spec(_spec)
sys.modules["strip_capture_gps"] = strip
_spec.loader.exec_module(strip)

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
FOCAL_35MM = 0xA405
ORIENTATION = 0x0112
# A made-up position in the open Indian Ocean, never any capture's. Each value is exact in
# hundredths, so it reads back unchanged.
LATITUDE = (12.0, 34.0, 56.5)
LONGITUDE = (98.0, 7.0, 6.25)
GPS_DATE = b"2001:02:03"


def _exif_payload(*, gps: bool, big_endian: bool, orientation: int) -> bytes:
    """EXIF bytes built by hand, so both byte orders can be written (iPhones write big-endian;
    Pillow only writes little-endian). IFD0 holds the orientation, a Make string stored outside
    its entry, and pointers to the Exif IFD (35 mm focal length) and, optionally, a GPS IFD whose
    rationals and date string also live outside its entries, as on an iPhone."""
    e = ">" if big_endian else "<"
    ascii_, short, long_, rational = 2, 3, 4, 5

    def ifd(entries, start):
        """entries: (tag, type, count, raw value bytes); returns IFD bytes + value area."""
        body = struct.pack(e + "H", len(entries))
        area = b""
        area_start = start + 2 + 12 * len(entries) + 4
        for tag, kind, count, raw in entries:
            if len(raw) <= 4:
                field = raw.ljust(4, b"\x00")
            else:
                field = struct.pack(e + "I", area_start + len(area))
                area += raw + (b"\x00" if len(raw) % 2 else b"")
            body += struct.pack(e + "HHI", tag, kind, count) + field
        return body + struct.pack(e + "I", 0) + area

    def rationals(values):
        return b"".join(struct.pack(e + "II", int(v * 100), 100) for v in values)

    gps_entries = [
        (1, ascii_, 2, b"S\x00"),
        (2, rational, 3, rationals(LATITUDE)),
        (3, ascii_, 2, b"E\x00"),
        (4, rational, 3, rationals(LONGITUDE)),
        (29, ascii_, 11, GPS_DATE + b"\x00"),
    ]
    exif_entries = [(FOCAL_35MM, short, 1, struct.pack(e + "H", 26))]
    # Lay out: header (8) | IFD0 | Exif IFD | GPS IFD, each followed by its own value area.
    ifd0_count = 4 if gps else 3
    ifd0_size = 2 + 12 * ifd0_count + 4 + 6  # + "Apple\0"
    exif_at = 8 + ifd0_size
    exif_block = ifd(exif_entries, exif_at)
    gps_at = exif_at + len(exif_block)
    ifd0_entries = [
        (0x010F, ascii_, 6, b"Apple\x00"),
        (ORIENTATION, short, 1, struct.pack(e + "H", orientation)),
        (EXIF_IFD, long_, 1, struct.pack(e + "I", exif_at)),
    ]
    if gps:
        ifd0_entries.append((GPS_IFD, long_, 1, struct.pack(e + "I", gps_at)))
    ifd0_block = ifd(ifd0_entries, 8)
    assert len(ifd0_block) == ifd0_size
    tiff = (b"MM\x00\x2a" if big_endian else b"II\x2a\x00") + struct.pack(e + "I", 8)
    tiff += ifd0_block + exif_block + (ifd(gps_entries, gps_at) if gps else b"")
    return b"Exif\x00\x00" + tiff


def _jpeg(*, gps: bool, big_endian: bool = False, orientation: int = 6) -> bytes:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    payload = _exif_payload(gps=gps, big_endian=big_endian, orientation=orientation)
    image.save(buffer, format="JPEG", exif=payload, quality=90)
    return buffer.getvalue()


def _gps(data: bytes) -> dict:
    return dict(Image.open(io.BytesIO(data)).getexif().get_ifd(GPS_IFD))


@pytest.mark.parametrize("big_endian", [False, True])
def test_the_gps_block_comes_out_empty_in_either_byte_order(big_endian):
    original = _jpeg(gps=True, big_endian=big_endian)
    assert _gps(original)[2] == LATITUDE  # the fixture really carries a position
    result = strip.strip_gps(original)
    assert result.gps_entries_removed == 5
    assert _gps(result.data) == {}
    # No trace of the coordinates' bytes is left anywhere in the file.
    endian = ">" if big_endian else "<"
    latitude = b"".join(struct.pack(endian + "II", int(v * 100), 100) for v in LATITUDE)
    for value in (GPS_DATE, latitude):
        assert value in original
        assert value not in result.data


@pytest.mark.parametrize("big_endian", [False, True])
def test_only_the_gps_bytes_change(big_endian):
    original = _jpeg(gps=True, big_endian=big_endian)
    result = strip.strip_gps(original)
    assert len(result.data) == len(original)
    changed = np.flatnonzero(
        np.frombuffer(original, np.uint8) != np.frombuffer(result.data, np.uint8)
    )
    inside = np.zeros(len(original), dtype=bool)
    for start, end in result.blanked_ranges:
        inside[start:end] = True
    assert changed.size > 0
    assert inside[changed].all()
    # So the image decodes to exactly the same pixels.
    before = np.asarray(Image.open(io.BytesIO(original)))
    after = np.asarray(Image.open(io.BytesIO(result.data)))
    np.testing.assert_array_equal(before, after)


def test_the_fields_the_pipeline_reads_are_unchanged(tmp_path):
    original = _jpeg(gps=True, big_endian=True, orientation=6)
    before, after = tmp_path / "before.jpg", tmp_path / "after.jpg"
    before.write_bytes(original)
    after.write_bytes(strip.strip_gps(original).data)
    assert read_exif_fields(before) == read_exif_fields(after) == (26.0, 6)
    assert Image.open(after).getexif()[0x010F] == "Apple"


def test_a_jpeg_without_gps_is_copied_byte_for_byte():
    original = _jpeg(gps=False)
    result = strip.strip_gps(original)
    assert result.data == original
    assert result.gps_entries_removed == 0


def test_a_jpeg_without_exif_is_copied_byte_for_byte():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
    assert strip.strip_gps(buffer.getvalue()).data == buffer.getvalue()


def test_a_non_jpeg_is_refused():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    with pytest.raises(strip.NotAJpeg):
        strip.strip_gps(buffer.getvalue())


def test_a_folder_is_copied_and_its_originals_are_untouched(tmp_path):
    source = tmp_path / "s-01"
    source.mkdir()
    (source / "IMG_1.jpeg").write_bytes(_jpeg(gps=True, big_endian=True))
    (source / "IMG_2.jpeg").write_bytes(_jpeg(gps=False))
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    }

    report = strip.strip_folder(source, tmp_path / "out")

    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    } == before
    assert report["files"] == 2 and report["had_gps"] == 1
    out = tmp_path / "out"
    assert _gps((out / "IMG_1.jpeg").read_bytes()) == {}
    assert (out / "IMG_2.jpeg").read_bytes() == (source / "IMG_2.jpeg").read_bytes()
    folder = report["folders"]["."]
    assert folder["frames"] == 2
    assert folder["content_sha256"] == strip.sample_content_hash(out)


def test_a_folder_with_other_files_is_refused_and_nothing_is_written(tmp_path):
    source = tmp_path / "s-01"
    source.mkdir()
    (source / "IMG_1.jpeg").write_bytes(_jpeg(gps=True))
    (source / "notes.txt").write_text("x")
    with pytest.raises(strip.NotAJpeg):
        strip.strip_folder(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_existing_destination_is_refused(tmp_path):
    source = tmp_path / "s-01"
    source.mkdir()
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        strip.strip_folder(source, tmp_path / "out")


def test_the_content_hash_matches_the_downloaders_definition(tmp_path):
    # sha256 over "<sha256 of file>  <basename>\n" lines in path order (fixtures/README.md).
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8two")
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8one")
    lines = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in (("a.jpg", b"\xff\xd8one"), ("b.jpg", b"\xff\xd8two"))
    )
    assert (
        strip.sample_content_hash(tmp_path)
        == hashlib.sha256(lines.encode()).hexdigest()
    )
