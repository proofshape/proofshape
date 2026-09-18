import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

GPU_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_gpu.py"
GPU_NAMESPACE = runpy.run_path(str(GPU_SCRIPT))
gpu_report = GPU_NAMESPACE["gpu_report"]
main = GPU_NAMESPACE["main"]
MAIN_GLOBALS = main.__globals__


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
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(importlib, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        gpu_report()


def test_gpu_report_preserves_missing_transitive_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str):
        assert name == "torch"
        raise ModuleNotFoundError("No module named 'sympy'", name="sympy")

    monkeypatch.setattr(importlib, "import_module", missing_import)

    with pytest.raises(ModuleNotFoundError, match="sympy"):
        gpu_report()


def test_main_returns_zero_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(
        MAIN_GLOBALS,
        "gpu_report",
        lambda: {
            "pytorch_version": "2.8.0",
            "cuda_available": True,
            "gpu_count": 1,
            "gpu_name": "Tesla T4",
        },
    )

    assert main() == 0
    assert "GPU CHECK: PASS" in capsys.readouterr().out


def test_main_returns_one_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(
        MAIN_GLOBALS,
        "gpu_report",
        lambda: {
            "pytorch_version": "2.8.0",
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": None,
        },
    )

    assert main() == 1
    assert "PyTorch cannot see CUDA" in capsys.readouterr().out


def test_main_returns_one_when_pytorch_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_torch():
        raise RuntimeError("PyTorch is not installed")

    monkeypatch.setitem(MAIN_GLOBALS, "gpu_report", missing_torch)

    assert main() == 1
    assert "PyTorch is not installed" in capsys.readouterr().out
