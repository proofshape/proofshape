"""A-04's stub completion runner: the other side of `recon.completion_contract`, with no model.

CI has no GPU and LaS-Comp is CUDA-only (D-057), so the stage's tests run this instead, as a real
subprocess, so the actual process boundary is what gets tested. It does what A-12's real runner
must do -- read the request, honour the budget, write each sample's mesh and rewrite result.json
after each one -- with a box around the partial cloud, jittered by the seed, standing in for the
model's output. Same seed, same bytes; different seed, different surface.

It imports the contract the way A-12's runner will: by putting the repository root on sys.path,
since `pip install -e .` refuses the runner's Python 3.10.

`--mode` makes it misbehave in the specific ways the stage must survive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from recon import completion_contract as contract

MODES = (
    "ok",
    "crash_first",
    "crash_after_first",
    "hang",
    "invalid_result",
    "zero_samples",
    "unit_cube",
    "wrong_seed",
    "huge_stderr",
)
STDERR_SENTINEL = "STUB-RUNNER-STDERR-SENTINEL"

_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ]
)  # fmt: skip


def _box_around(points: np.ndarray, seed: int, unit_cube: bool) -> np.ndarray:
    low = points.min(axis=0)
    high = points.max(axis=0)
    if unit_cube:
        # What a runner that forgot to undo LaS-Comp's normalisation would return.
        low, high = np.full(3, -0.5), np.full(3, 0.5)
    corners = np.array(
        [[x, y, z] for z in (low[2], high[2]) for x, y in
         ((low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1]))]
    )  # fmt: skip
    jitter = np.random.default_rng(seed).normal(0.0, 0.5, size=corners.shape)
    return corners + (0.0 if unit_cube else jitter)


def _write_result(path: Path, status: str, load_s, samples, error=None) -> None:
    contract.write_json_atomic(
        path,
        contract.result_payload(
            contract.RunnerResult(
                status=status,
                model=contract.ExpectedModel().as_dict(),
                load_s=load_s,
                samples=tuple(samples),
                error=error,
            )
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="ok")
    parser.add_argument("--sample-delay-s", type=float, default=0.0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    request = contract.read_request(args.request)
    folder = args.request.parent
    out_dir = folder / contract.RUNNER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / contract.RESULT_FILE
    partial = contract.read_points_ply(folder / contract.PARTIAL_POINTS_PATH)

    if args.mode == "invalid_result":
        result_path.write_text("{not json", encoding="utf-8")
        return 0
    if args.mode == "huge_stderr":
        sys.stderr.write((STDERR_SENTINEL + " ") * 40_000)
        sys.stderr.flush()

    load_s = time.perf_counter() - started
    _write_result(result_path, "running", load_s, [])
    if args.mode == "crash_first":
        return 3
    if args.mode == "zero_samples":
        _write_result(result_path, "complete", load_s, [])
        return 0

    samples: list[contract.SampleRecord] = []
    last_sample_s = 0.0
    for k, seed in enumerate(request.seeds):
        # The budget rule A-12's runner must follow: don't start a sample that would end past it.
        elapsed = time.perf_counter() - started
        if k > 0 and elapsed + last_sample_s > request.budget_s:
            _write_result(result_path, "budget_reached", load_s, samples)
            return 0
        sample_started = time.perf_counter()
        time.sleep(args.sample_delay_s)
        vertices = _box_around(partial, seed, unit_cube=args.mode == "unit_cube")
        contract.write_mesh_ply(
            out_dir / contract.sample_mesh_name(k), vertices, _BOX_FACES
        )
        last_sample_s = time.perf_counter() - sample_started
        reported_seed = seed + 1000 if args.mode == "wrong_seed" else seed
        samples.append(
            contract.SampleRecord(k=k, seed=reported_seed, time_s=last_sample_s)
        )
        _write_result(result_path, "running", load_s, samples)
        if args.mode == "crash_after_first":
            return 3
        if args.mode == "hang":
            time.sleep(600)
    _write_result(result_path, "complete", load_s, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
