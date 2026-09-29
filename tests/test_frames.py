"""Unit tests for recon/frames.py: the capture-frame listing shared by R-01 and R-02 (issue #43).

Moved here from tests/test_vggt_runner.py, where these were originally written against the
pre-consolidation duplicate. test_vggt_runner.py and test_board_pose.py now each keep a small
re-export sanity check instead of repeating this behavioural coverage.
"""

from __future__ import annotations

import pytest

from recon.frames import find_capture_frames


def test_find_capture_frames_sorts_and_filters(tmp_path):
    for name in ["b.JPG", "a.jpg", "c.jpeg", "notes.txt", "d.png"]:
        (tmp_path / name).write_bytes(b"x")
    (
        tmp_path / "subdir.jpg"
    ).mkdir()  # a directory with a jpg-looking name is not a frame

    names = [path.name for path in find_capture_frames(tmp_path)]

    assert names == ["a.jpg", "b.JPG", "c.jpeg"]


def test_find_capture_frames_missing_folder_points_at_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_golden_capture"):
        find_capture_frames(tmp_path / "nope")


def test_find_capture_frames_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .jpg"):
        find_capture_frames(tmp_path)
