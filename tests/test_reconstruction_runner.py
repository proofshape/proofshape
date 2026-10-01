"""Unit tests for the D-009 reconstruction backend switch."""

from pathlib import Path

import pytest

from recon import reconstruction_runner


def test_resolve_backend_defaults_to_vggt(monkeypatch):
    monkeypatch.delenv(reconstruction_runner.BACKEND_ENV, raising=False)

    assert reconstruction_runner.resolve_backend() == "vggt"


def test_resolve_backend_can_use_one_environment_config(monkeypatch):
    monkeypatch.setenv(reconstruction_runner.BACKEND_ENV, "mast3r")

    assert reconstruction_runner.resolve_backend() == "mast3r"


def test_resolve_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv(reconstruction_runner.BACKEND_ENV, "mystery")

    with pytest.raises(ValueError, match="Unsupported reconstruction backend"):
        reconstruction_runner.resolve_backend()


def test_run_selected_backend_calls_only_mast3r(tmp_path, monkeypatch):
    seen = []

    def fake_mast3r(*args, **kwargs):
        seen.append((args, kwargs))
        return {
            "backend": "mast3r",
            "frame_count": 2,
            "point_count": 4,
            "timings": {"total_s": 1.0},
        }

    def fail_vggt(*args, **kwargs):
        raise AssertionError("VGGT should not run")

    monkeypatch.setattr(
        reconstruction_runner.mast3r_runner,
        "run_on_capture",
        fake_mast3r,
    )
    monkeypatch.setattr(
        reconstruction_runner.vggt_runner,
        "run_on_capture",
        fail_vggt,
    )

    result = reconstruction_runner.run_on_capture(
        Path("capture"),
        tmp_path / "out",
        backend="mast3r",
        device_type="cuda",
    )

    assert result["backend"] == "mast3r"
    assert len(seen) == 1


def test_run_selected_backend_keeps_vggt_default(tmp_path, monkeypatch):
    monkeypatch.delenv(reconstruction_runner.BACKEND_ENV, raising=False)
    seen = []

    def fake_vggt(*args, **kwargs):
        seen.append((args, kwargs))
        return {
            "backend": "vggt",
            "frame_count": 2,
            "point_count": 4,
            "timings": {"total_s": 1.0},
        }

    def fail_mast3r(*args, **kwargs):
        raise AssertionError("MASt3R should not run")

    monkeypatch.setattr(
        reconstruction_runner.vggt_runner,
        "run_on_capture",
        fake_vggt,
    )
    monkeypatch.setattr(
        reconstruction_runner.mast3r_runner,
        "run_on_capture",
        fail_mast3r,
    )

    result = reconstruction_runner.run_on_capture(
        Path("capture"),
        tmp_path / "out",
    )

    assert result["backend"] == "vggt"
    assert len(seen) == 1


def test_main_prints_the_full_per_stage_table_not_just_the_total(
    tmp_path, monkeypatch, capsys
):
    # R-13: before this story, main() printed only `total_s`, discarding the per-stage
    # breakdown each backend's own run_on_capture already returns.
    def fake_run_on_capture(
        capture_dir, out_dir, backend=None, device_type="cuda", min_confidence=0.0
    ):
        return {
            "backend": "vggt",
            "frame_count": 2,
            "point_count": 4,
            "timings": {"find_frames_s": 0.1, "model_load_s": 0.2, "total_s": 0.3},
        }

    monkeypatch.setattr(reconstruction_runner, "run_on_capture", fake_run_on_capture)

    exit_code = reconstruction_runner.main(
        [str(tmp_path / "capture"), "--out-dir", str(tmp_path / "out")]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    for stage in ["find_frames_s", "model_load_s", "total_s"]:
        assert stage in out
    assert "share" in out
