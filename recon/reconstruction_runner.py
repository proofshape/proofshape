"""Select the reconstruction backend with one flag while keeping one file contract (D-009)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from recon import colmap_runner, mast3r_runner, vggt_runner
from recon.timing import format_timing_table

BACKENDS = ("vggt", "mast3r", "colmap")
DEFAULT_BACKEND = "vggt"
BACKEND_ENV = "PROOFSHAPE_RECON_BACKEND"


def resolve_backend(value: str | None = None) -> str:
    """Return the requested backend; VGGT remains the default required by D-009."""
    backend = value or os.environ.get(BACKEND_ENV, DEFAULT_BACKEND)
    if backend not in BACKENDS:
        choices = ", ".join(BACKENDS)
        raise ValueError(
            f"Unsupported reconstruction backend {backend!r}; choose one of {choices}."
        )
    return backend


def run_on_capture(
    capture_dir: Path,
    out_dir: Path,
    backend: str | None = None,
    device_type: str = "cuda",
    min_confidence: float = 0.0,
    board_pose_path: Path | None = None,
) -> dict[str, Any]:
    """Run one backend without changing the R-01 four-file output contract."""
    selected = resolve_backend(backend)
    if selected == "vggt":
        return vggt_runner.run_on_capture(
            capture_dir,
            out_dir,
            device_type=device_type,
            min_confidence=min_confidence,
        )

    if selected == "mast3r":
        return mast3r_runner.run_on_capture(
            capture_dir,
            out_dir,
            device_type=device_type,
            min_confidence=min_confidence,
        )

    return colmap_runner.run_on_capture(
        capture_dir,
        out_dir,
        board_pose_path=board_pose_path,
        device_type=device_type,
        min_confidence=min_confidence,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ProofShape reconstruction with VGGT, MASt3R or COLMAP behind one backend flag."
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="backend (default: PROOFSHAPE_RECON_BACKEND or vggt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder (default: fixtures/data/recon_runs/<capture folder name>)",
    )
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_dir = args.out_dir or Path("fixtures/data/recon_runs") / args.capture_dir.name
    try:
        backend = resolve_backend(args.backend)
        run_info = run_on_capture(
            args.capture_dir,
            out_dir,
            backend=backend,
            device_type=args.device,
            min_confidence=args.min_confidence,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"RECON RUN: FAIL - {exc}", file=sys.stderr)
        return 1

    if run_info.get("status") == "refused":
        print(
            f"RECON RUN: REFUSED - backend={run_info['backend']}, "
            f"registered={run_info['registered_frame_count']}/"
            f"{run_info['frame_count']}, reason={run_info['reason']}"
        )
        print(format_timing_table(run_info["timings"]))
        print(f"Refusal record written to {out_dir}")
        return 0

    print(
        f"RECON RUN: PASS - backend={run_info['backend']}, "
        f"{run_info['frame_count']} frames, {run_info['point_count']} points"
    )
    print(format_timing_table(run_info["timings"]))
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
