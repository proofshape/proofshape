"""Verify that the current Python environment can see an NVIDIA GPU through PyTorch.

F-05 uses this as the shared smoke test. The logic is deliberately tiny: this story is proving
that the shared GPU environment works, not building the project's full Python environment
(which belongs to F-07).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any


def gpu_report(torch_module: Any | None = None) -> dict[str, object]:
    """Return a small, testable GPU report from PyTorch.

    ``torch_module`` is injectable so the unit tests do not require a GPU or even a local
    PyTorch installation. On the real Studio we import the installed PyTorch package.
    """
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is not installed in this environment; F-05 GPU verification cannot run."
            ) from exc

    cuda_available = bool(torch_module.cuda.is_available())
    report: dict[str, object] = {
        "pytorch_version": str(torch_module.__version__),
        "cuda_available": cuda_available,
        "gpu_count": int(torch_module.cuda.device_count()) if cuda_available else 0,
        "gpu_name": None,
    }

    if cuda_available:
        report["gpu_name"] = str(torch_module.cuda.get_device_name(0))

    return report


def main() -> int:
    """Print the shared F-05 smoke-test result and return a process exit code."""
    try:
        report = gpu_report()
    except RuntimeError as exc:
        print(f"GPU CHECK: FAIL - {exc}")
        return 1

    print(f"PyTorch version: {report['pytorch_version']}")
    print(f"CUDA available: {report['cuda_available']}")
    print(f"GPU count: {report['gpu_count']}")
    print(f"GPU: {report['gpu_name'] or 'none'}")

    if not report["cuda_available"]:
        print("GPU CHECK: FAIL - PyTorch cannot see CUDA.")
        return 1

    print("GPU CHECK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
