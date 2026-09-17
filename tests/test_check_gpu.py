import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

GPU_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_gpu.py"
gpu_report = runpy.run_path(str(GPU_SCRIPT))["gpu_report"]


class FakeCuda:
    def __init__(self, available: bool, count: int = 0, name: str = "Fake GPU"):
        self._available = available
        self._count = count
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._name


def test_gpu_report_when_cuda_available() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.8.0", cuda=FakeCuda(True, 1, "Tesla T4")
    )

    report = gpu_report(fake_torch)

    assert report == {
        "pytorch_version": "2.8.0",
        "cuda_available": True,
        "gpu_count": 1,
        "gpu_name": "Tesla T4",
    }


def test_gpu_report_when_cuda_unavailable() -> None:
    fake_torch = SimpleNamespace(__version__="2.8.0", cuda=FakeCuda(False))

    report = gpu_report(fake_torch)

    assert report == {
        "pytorch_version": "2.8.0",
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
    }


def test_gpu_report_has_clear_error_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str):
        assert name == "torch"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        gpu_report()
