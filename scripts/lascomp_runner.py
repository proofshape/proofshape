"""A-12: the real LaS-Comp runner behind A-04's completion contract (D-057, D-058).

The service never imports LaS-Comp. It writes a request folder (`recon/completion_contract.py`)
and runs this script as a separate process, because LaS-Comp needs Python 3.10, torch 2.4 and
CUDA while the service runs Python >=3.11, usually on the M2 Max (D-049). This script runs inside
the environment `scripts/setup_completion_env.sh` builds, and is launched as

    <venv>/bin/python -I <repo>/scripts/lascomp_runner.py --lascomp-root <dir> --request <file>

`-I` keeps the service's PYTHONPATH and user site-packages out of the 3.10 interpreter. It must be
the direct child, with no shell in between, so the service's hard timeout kills this process.

What it does, following the contract:
1. Reads the request strictly, and refuses to touch a `runner/` folder that already has files.
2. Verifies the pins **before** importing torch: LaS-Comp's git commit (with no tracked file
   edited) and the weights revision setup recorded. A result is only worth ingesting if it came
   from the model A-10 measured, so on any mismatch it reports what it actually found and stops.
3. Loads LaS-Comp's pipeline once, then samples one completion per seed while the budget allows,
   with the same budget rule as `tests/completion_stub_runner.py`: never start a sample that
   would end past the budget, judged by the previous sample's time. The clock starts at process
   start, so model load counts against the budget too.
4. Writes each mesh in the contract's layout and rewrites `runner/result.json` atomically after
   every sample, so whatever the service's timeout interrupts, the samples so far survive.

The partial cloud is already z-up (the contract's frame), so it goes in with no Y/Z swap.
LaS-Comp normalises it to a unit cube and this script maps the mesh back with LaS-Comp's own
inverse, so the mesh comes out in the input's millimetres with no axis or winding change.

Everything that needs torch or LaS-Comp lives in `LasCompBackend` and is imported lazily, so CI
(Python 3.11, no GPU) can test the rest with a fake backend. This file stays Python 3.10 syntax
and imports only the standard library, numpy and the contract at module level;
`tests/test_lascomp_runner.py` enforces both.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # `pip install -e .` refuses Python 3.10 (requires-python), so the contract is imported from
    # the repository root, exactly as the stub runner does.
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from recon import completion_contract as contract

LASCOMP_ROOT_ENV = "LASCOMP_ROOT"
PINS_FILE = "proofshape_pins.json"
CKPT_DIR = "ckpt/ckpt_image_large"
METRICS_FILE = "lascomp_metrics.json"
CONDITIONING_IMAGE = "lascomp_conditioning.png"

# A-10's setup fix: TRELLIS's published pipeline.json doesn't list the encoder LaS-Comp needs, and
# the KeyError only surfaces at the first sample, after the ~50 s load.
ENCODER_KEY = "sparse_structure_encoder"
ENCODER_VALUE = "ckpts/ss_enc_conv3d_16l8_fp16"

# DINOv2's code, which TRELLIS loads with torch.hub.load('facebookresearch/dinov2', name) and no
# ref (trellis/pipelines/trellis_image_to_3d.py). torch.hub then uses whatever is cached under
# DINOV2_CACHE_NAME, or downloads main's current code if nothing is. Main has been at this commit
# since 2026-06-03, so it is the code A-12 measured on 2026-11-07. Setup fetches it there as a git
# checkout, and every run checks it, the same way LaS-Comp is checked. It isn't one of the
# contract's four reported pins (D-057 names none for it), so a mismatch fails the run rather
# than changing what the result reports. tests/test_setup_completion_env.py checks that setup uses
# the same commit.
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CACHE_NAME = "facebookresearch_dinov2_main"

# How A-10 called LaS-Comp (stories/A/A-10.md): its CLI defaults, and SPCONV_ALGO=native. Kept
# identical so A-12's timings are comparable with A-10's; changing any of these changes the model
# D-057 measured.
SPCONV_ALGO = "native"
VOXEL_RESOLUTION = 64
LASCOMP_CALL = {
    "alpha_eta": 0.1,
    "optimization_step": 1,
    "lr_sample": 1e-5,
    "rescale_t": 3.0,
}

# LaS-Comp's script sets CUDA_LAUNCH_BLOCKING=1 when imported, so A-10 measured with blocking
# launches. "1" keeps that; "0" is the speed lever A-12 measures.
BLOCKING_CHOICES = ("1", "0")
# Launches count as blocking when queueing a burst of kernels takes nearly as long as finishing
# them. 0.8 leaves room for launch overhead on a fast burst; non-blocking queueing is typically
# well under a tenth of the work.
BLOCKING_QUEUE_FRACTION = 0.8

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


class RunnerError(RuntimeError):
    """A problem this runner found itself, reported as the result's error."""


class Backend(Protocol):
    def load(
        self, image_path: Path, steps: dict[str, dict[str, int]]
    ) -> dict[str, Any]: ...

    def sample(
        self, partial_path: Path, seed: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]: ...


BackendFactory = Callable[[Path, dict[str, Any], str, Path], Backend]


# --- pins -----------------------------------------------------------------------------------------


def read_pins_file(lascomp_root: Path) -> dict[str, Any]:
    """What setup recorded: the weights it downloaded and where it put the model caches."""
    path = Path(lascomp_root) / PINS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerError(
            f"{path} is missing; run scripts/setup_completion_env.sh first"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise RunnerError(f"{path} must hold a JSON object")
    return data


def _git(lascomp_root: Path, *args: str) -> str:
    # safe.directory: setup and the service may run as different users, and git refuses to read a
    # repository another user owns unless told it's safe.
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={lascomp_root}", "-C", str(lascomp_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RunnerError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def edited_tracked_files(porcelain: str) -> list[str]:
    """The `git status --porcelain` lines that mean LaS-Comp's code changed.

    LaS-Comp's repository commits its `__pycache__/*.pyc` files. A fresh clone gives every `.py`
    file the clone's mtime, so none of those caches matches its source any more, and the first
    import (setup's `--check`) recompiles each one from the tracked `.py` and rewrites it. Git then
    reports 33 modified files although no code changed. A-12's first Colab run, 2026-11-06,
    refused all four measurements for exactly that reason. The code that runs is compiled from the
    `.py` files, and those are still checked. So only a *modified* cache inside `__pycache__/` is
    skipped. A deleted or added cache, or any other file, is still reported.
    """
    edited: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:]
        parts = Path(path).parts
        is_rewritten_cache = (
            status == " M"
            and path.endswith(".pyc")
            and len(parts) >= 2
            and parts[-2] == "__pycache__"
        )
        if is_rewritten_cache:
            continue
        edited.append(line.strip())
    return edited


def verify_dinov2(pins: dict[str, Any]) -> list[str]:
    """Problems with the DINOv2 code TRELLIS will load. Empty means it's setup's pinned checkout.

    The checkout must have its own `.git`, because the cache sits inside LaS-Comp's repository and
    git would otherwise answer with LaS-Comp's commit. A plain folder (torch.hub's own zip
    download) has no commit to check, so it's refused, and setup replaces it.
    """
    torch_home = pins.get("caches", {}).get("torch_home")
    if not torch_home:
        return [
            f"{PINS_FILE} names no torch_home cache; run scripts/setup_completion_env.sh"
        ]
    checkout = Path(torch_home) / "hub" / DINOV2_CACHE_NAME
    if not (checkout / ".git").exists():
        return [
            (
                f"DINOv2's code at {checkout} isn't setup's pinned checkout; "
                "run scripts/setup_completion_env.sh"
            )
        ]
    try:
        commit = _git(checkout, "rev-parse", "HEAD").strip()
        modified = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    except (RunnerError, OSError) as exc:
        return [f"DINOv2's checkout at {checkout} can't be read: {exc}"]
    problems = []
    if commit != DINOV2_COMMIT:
        problems.append(f"DINOv2 is at {commit}, expected {DINOV2_COMMIT}")
    edited = edited_tracked_files(modified)
    if edited:
        problems.append(f"DINOv2 has edited tracked files: {'; '.join(edited)}")
    return problems


def verify_pins(
    lascomp_root: Path, expected: contract.ExpectedModel
) -> tuple[dict[str, str], list[str]]:
    """(the pins actually found, problems). An empty problem list means it's the pinned model.

    The found values are what the result reports, so a mismatch reaches the service as a pin
    mismatch, which `parse_result` rejects and records, rather than as the expected pins
    reported blindly.
    """
    problems: list[str] = []
    try:
        commit = _git(lascomp_root, "rev-parse", "HEAD").strip()
    except (RunnerError, OSError) as exc:
        commit = f"unverified: {exc}"
        problems.append(f"LaS-Comp commit can't be read at {lascomp_root}: {exc}")
    else:
        if commit != expected.commit:
            problems.append(f"LaS-Comp is at {commit}, expected {expected.commit}")
        try:
            # Untracked files (the weights, the caches, the pins file) are setup's own and fine;
            # an edited tracked file means the code isn't the pinned code.
            modified = _git(
                lascomp_root, "status", "--porcelain", "--untracked-files=no"
            )
        except (RunnerError, OSError) as exc:
            problems.append(f"LaS-Comp's working tree can't be checked: {exc}")
        else:
            # Unstripped on purpose: a porcelain line's first column is significant (" M").
            edited = edited_tracked_files(modified)
            if edited:
                problems.append(
                    f"LaS-Comp has edited tracked files: {'; '.join(edited)}"
                )

    try:
        pins = read_pins_file(lascomp_root)
    except RunnerError as exc:
        problems.append(str(exc))
        pins = {}
    weights = str(pins.get("weights", "unverified: no pins file"))
    weights_revision = str(pins.get("weights_revision", "unverified: no pins file"))
    if pins and weights != expected.weights:
        problems.append(f"weights are {weights}, expected {expected.weights}")
    if pins and weights_revision != expected.weights_revision:
        problems.append(
            f"weights revision is {weights_revision}, expected {expected.weights_revision}"
        )
    if pins:
        problems.extend(verify_dinov2(pins))

    found = {
        "name": contract.MODEL_NAME,
        "commit": commit,
        "weights": weights,
        "weights_revision": weights_revision,
    }
    return found, problems


# --- the checkpoint folder ------------------------------------------------------------------------


def ensure_sparse_structure_encoder(
    pipeline_json: Path, check_only: bool = False
) -> bool:
    """Make sure pipeline.json lists LaS-Comp's sparse-structure encoder. True if it was added.

    Idempotent, so setup can run it after every download (a re-download may restore the original
    file). With `check_only` it never writes: it returns False when the entry is right and raises
    when it isn't. A different existing value raises rather than being overwritten.
    """
    pipeline_json = Path(pipeline_json)
    config = json.loads(pipeline_json.read_text(encoding="utf-8"))
    models = config.get("args", {}).get("models")
    if not isinstance(models, dict):
        raise RunnerError(f"{pipeline_json} has no args.models table")
    present = models.get(ENCODER_KEY)
    if present == ENCODER_VALUE:
        return False
    if present is not None:
        raise RunnerError(
            f"{pipeline_json} lists {ENCODER_KEY} as {present!r}, not {ENCODER_VALUE!r}"
        )
    if check_only:
        raise RunnerError(
            f"{pipeline_json} doesn't list {ENCODER_KEY}; run scripts/setup_completion_env.sh"
        )
    for suffix in (".json", ".safetensors"):
        weights_file = pipeline_json.parent / f"{ENCODER_VALUE}{suffix}"
        if not weights_file.is_file():
            raise RunnerError(
                f"the encoder's {weights_file} is missing from the download"
            )
    models[ENCODER_KEY] = ENCODER_VALUE
    pipeline_json.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    return True


def missing_model_files(ckpt_dir: Path) -> list[str]:
    """Every model file pipeline.json names that isn't on disk.

    Checked before loading because TRELLIS's loader catches a missing local file with a bare
    `except` and retries the name as a Hugging Face repository, which hides the real problem
    behind an unrelated download error (trellis/pipelines/base.py).
    """
    ckpt_dir = Path(ckpt_dir)
    config = json.loads((ckpt_dir / "pipeline.json").read_text(encoding="utf-8"))
    missing = []
    for name in config["args"]["models"].values():
        for suffix in (".json", ".safetensors"):
            if not (ckpt_dir / f"{name}{suffix}").is_file():
                missing.append(f"{name}{suffix}")
    return missing


def sampler_overrides(
    request: contract.CompletionRequest,
) -> dict[str, dict[str, int]]:
    """Per-call sampler parameters from the request; empty means LaS-Comp's own default.

    LaS-Comp merges these into its defaults on each call (100 sparse-structure and 25 SLAT steps),
    so nothing on the pipeline itself is changed and the steps used are exactly the request's.
    """
    sparse: dict[str, int] = {}
    slat: dict[str, int] = {}
    if request.sparse_structure_steps is not None:
        sparse["steps"] = int(request.sparse_structure_steps)
    if request.slat_steps is not None:
        slat["steps"] = int(request.slat_steps)
    return {"sparse_structure": sparse, "slat": slat}


def launches_are_blocking(queue_s: float, total_s: float) -> bool:
    """True when queueing a kernel burst took nearly as long as running it to completion."""
    if total_s <= 0:
        return True
    return queue_s >= BLOCKING_QUEUE_FRACTION * total_s


# --- the real backend: the only code that touches torch or LaS-Comp -------------------------------


class LasCompBackend:
    """LaS-Comp's pipeline, loaded once in this process, sampled once per seed."""

    def __init__(
        self,
        lascomp_root: Path,
        pins: dict[str, Any],
        cuda_launch_blocking: str,
        runner_dir: Path | None,
    ) -> None:
        self.root = Path(lascomp_root)
        self.pins = pins
        self.blocking = cuda_launch_blocking
        self.runner_dir = runner_dir
        self.torch: Any = None
        self.lascomp: Any = None
        self.pipeline: Any = None
        self.image: Any = None
        self.steps: dict[str, dict[str, int]] = {"sparse_structure": {}, "slat": {}}
        self.stage_times: dict[str, float] = {}

    def _prepare(self) -> dict[str, Any]:
        """Environment, CUDA and LaS-Comp's import, in the one order that keeps the blocking choice.

        LaS-Comp's script sets CUDA_LAUNCH_BLOCKING=1 and imports open3d, which starts CUDA, at
        import time, so clearing the variable afterwards would change nothing. CUDA is therefore
        started here first, with the chosen value, and the import only follows once the context
        exists. Then a kernel burst checks the choice really took effect.
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        os.environ["SPCONV_ALGO"] = SPCONV_ALGO
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.blocking
        caches = self.pins.get("caches", {})
        if caches.get("torch_home"):
            # DINOv2 comes from torch.hub, rembg's u2net from its own cache: both were downloaded
            # by setup, so a sample never waits on a download inside its timed window.
            os.environ["TORCH_HOME"] = str(caches["torch_home"])
        if caches.get("u2net_home"):
            os.environ["U2NET_HOME"] = str(caches["u2net_home"])

        import torch

        if not torch.cuda.is_available():
            raise RunnerError("CUDA is not available; LaS-Comp is CUDA-only (D-057)")
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        timings["cuda_start_s"] = time.perf_counter() - started
        self.torch = torch

        burst = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        queue_started = time.perf_counter()
        for _ in range(20):
            burst = burst @ burst
            burst = burst / burst.norm()
        queued = time.perf_counter()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        queue_s, total_s = queued - queue_started, finished - queue_started
        blocking = launches_are_blocking(queue_s, total_s)
        if blocking != (self.blocking == "1"):
            raise RunnerError(
                f"CUDA_LAUNCH_BLOCKING={self.blocking} was requested but launches "
                f"{'are' if blocking else 'are not'} blocking (queue {queue_s:.3f} s of "
                f"{total_s:.3f} s); the measurement would not be what it claims"
            )

        import_started = time.perf_counter()
        sys.path.insert(0, str(self.root))
        import run_lascomp_image_condition_single as lascomp

        # The import just set it back to "1"; the context already uses our value. Restore the
        # variable so anything that reads it later reports what's actually in effect.
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.blocking
        timings["import_s"] = time.perf_counter() - import_started
        self.lascomp = lascomp
        return {
            "timings_s": timings,
            "cuda_launch_blocking": {
                "requested": self.blocking,
                "effective_blocking": blocking,
                "burst_queue_s": queue_s,
                "burst_total_s": total_s,
            },
            "spconv_algo": SPCONV_ALGO,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory
            / 1e9,
            "torch": torch.__version__,
            "python": sys.version,
        }

    def check(self) -> dict[str, Any]:
        """Setup's last step: imports, CUDA, the blocking check and the checkpoint, no sampling."""
        report = self._prepare()
        ckpt = self.root / CKPT_DIR
        missing = missing_model_files(ckpt)
        if missing:
            raise RunnerError(f"model files missing from {ckpt}: {missing}")
        ensure_sparse_structure_encoder(ckpt / "pipeline.json", check_only=True)
        report["checkpoint"] = str(ckpt)
        return report

    def load(
        self, image_path: Path, steps: dict[str, dict[str, int]]
    ) -> dict[str, Any]:
        report = self._prepare()
        ckpt = self.root / CKPT_DIR
        missing = missing_model_files(ckpt)
        if missing:
            raise RunnerError(f"model files missing from {ckpt}: {missing}")
        ensure_sparse_structure_encoder(ckpt / "pipeline.json", check_only=True)
        self.steps = steps

        torch = self.torch
        started = time.perf_counter()
        self.pipeline = self.lascomp.build_pipeline(str(ckpt))
        torch.cuda.synchronize()
        report["timings_s"]["build_s"] = time.perf_counter() - started
        self._time_stages()

        from PIL import Image

        # Preprocessed once (rembg crop, 518 px) instead of once per sample as A-10 did: the same
        # input every time, without repeating rembg inside each sample's timed window.
        started = time.perf_counter()
        with Image.open(image_path) as opened:
            self.image = self.pipeline.preprocess_image(opened.convert("RGB"))
        report["timings_s"]["preprocess_s"] = time.perf_counter() - started
        if self.runner_dir is not None:
            # The only way to see what the model was actually conditioned on, e.g. that s-02's
            # EXIF-rotated photo reached it upright.
            self.image.save(self.runner_dir / CONDITIONING_IMAGE)
        report["steps"] = steps
        return report

    def _time_stages(self) -> None:
        """Wrap the pipeline's three stages so each sample records where its time went."""
        torch = self.torch
        for name in ("lascomp_sample_sparse_structure", "sample_slat", "decode_slat"):
            original = getattr(self.pipeline, name)

            def timed(
                *args: Any, _name: str = name, _original: Any = original, **kwargs: Any
            ) -> Any:
                torch.cuda.synchronize()
                started = time.perf_counter()
                out = _original(*args, **kwargs)
                torch.cuda.synchronize()
                self.stage_times[_name] = time.perf_counter() - started
                return out

            setattr(self.pipeline, name, timed)

    def sample(
        self, partial_path: Path, seed: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        torch = self.torch
        lascomp = self.lascomp
        from plyfile import PlyData
        from torch.nn import functional

        self.stage_times = {}
        plydata = PlyData.read(str(partial_path))
        # "custom" dataset, no Y/Z swap: the contract's partial cloud is already z-up.
        xyz = lascomp.load_points_from_ply(plydata, "custom", "auto", yz_flip=False)
        xyz, center, scale = lascomp.normalize_point_cloud(xyz)
        torch.cuda.reset_peak_memory_stats()
        ss, kept_points, _ = lascomp.voxelize_unit_cube(
            xyz, resolution=VOXEL_RESOLUTION, open_upper_bound=False
        )
        mask = (
            functional.avg_pool3d(
                ss.float().squeeze(0).unsqueeze(0), kernel_size=4, stride=4
            )
            > 0
        ).float()
        _, decoded = self.pipeline.run_lascomp(
            mask=mask.unsqueeze(0).cuda().float(),
            ss=ss.unsqueeze(0).cuda().float(),
            image=self.image,
            seed=int(seed),
            preprocess_image=False,
            sparse_structure_sampler_params=dict(self.steps["sparse_structure"]),
            slat_sampler_params=dict(self.steps["slat"]),
            **LASCOMP_CALL,
        )
        torch.cuda.synchronize()
        mesh = decoded["mesh"][0]
        # LaS-Comp's own inverse of its normalisation: back to the input's z-up millimetres.
        vertices = (
            lascomp.denormalize_point_cloud(
                mesh.vertices.float(), center.cuda(), scale.cuda()
            )
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        faces = mesh.faces.cpu().numpy().astype(np.int64)
        metrics = {
            "stage_times_s": dict(self.stage_times),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "points_kept": int(kept_points.numel()),
            "mesh_vertex_count": len(vertices),
            "mesh_face_count": len(faces),
        }
        return vertices, faces, metrics


def _real_backend(
    lascomp_root: Path,
    pins: dict[str, Any],
    cuda_launch_blocking: str,
    runner_dir: Path,
) -> Backend:
    return LasCompBackend(lascomp_root, pins, cuda_launch_blocking, runner_dir)


# --- the run --------------------------------------------------------------------------------------


class _Run:
    """The result and metrics files for one run, rewritten atomically as it goes."""

    def __init__(
        self, runner_dir: Path, model: dict[str, str], clock: Callable[[], float]
    ):
        self.runner_dir = runner_dir
        self.model = model
        self.clock = clock
        self.load_s: float | None = None
        self.samples: list[contract.SampleRecord] = []
        self.metrics: dict[str, Any] = {"samples": []}

    def write(self, status: str, error: str | None = None) -> None:
        contract.write_json_atomic(
            self.runner_dir / contract.RESULT_FILE,
            contract.result_payload(
                contract.RunnerResult(
                    status=status,
                    model=self.model,
                    load_s=self.load_s,
                    samples=tuple(self.samples),
                    error=error,
                )
            ),
        )
        contract.write_json_atomic(self.runner_dir / METRICS_FILE, self.metrics)


def run(
    request_path: Path,
    lascomp_root: Path,
    cuda_launch_blocking: str,
    backend_factory: BackendFactory,
    clock: Callable[[], float],
    started: float,
) -> int:
    request_path = Path(request_path).resolve()
    folder = request_path.parent
    try:
        request = contract.read_request(request_path)
    except contract.ContractError as exc:
        print(f"A-12 RUNNER: REFUSED - {exc}", file=sys.stderr)
        return EXIT_REFUSED
    runner_dir = folder / contract.RUNNER_DIR
    if runner_dir.exists() and any(runner_dir.iterdir()):
        # A previous run's result.json would otherwise be ingested as this run's.
        print(f"A-12 RUNNER: REFUSED - {runner_dir} already has files", file=sys.stderr)
        return EXIT_REFUSED
    runner_dir.mkdir(parents=True, exist_ok=True)

    found, problems = verify_pins(lascomp_root, request.expected_model)
    current = _Run(runner_dir, found, clock)
    current.metrics.update(
        {
            "runner_repo_commit": _repo_commit(),
            "lascomp_root": str(lascomp_root),
            "budget_s": request.budget_s,
            "seeds": list(request.seeds),
        }
    )
    if problems:
        error = "; ".join(problems)
        current.write("failed", error)
        print(f"A-12 RUNNER: FAILED - {error}", file=sys.stderr)
        return EXIT_FAILED

    status = "complete"
    try:
        current.write("running")
        pins = read_pins_file(lascomp_root)
        # Setup's record of the DINOv2 commit and weights file hash, kept with the measurement.
        current.metrics["dinov2"] = pins.get("dinov2")
        backend = backend_factory(lascomp_root, pins, cuda_launch_blocking, runner_dir)
        steps = sampler_overrides(request)
        current.metrics["load"] = backend.load(folder / contract.VIEW_IMAGE_PATH, steps)
        current.load_s = clock() - started
        current.write("running")

        last_sample_s = 0.0
        for k, seed in enumerate(request.seeds):
            # The contract's budget rule (the stub's too): don't start a sample that would end
            # past the budget. Sample 0 always runs, even if loading alone used the budget.
            if k > 0 and (clock() - started) + last_sample_s > request.budget_s:
                status = "budget_reached"
                break
            sample_started = clock()
            vertices, faces, sample_metrics = backend.sample(
                folder / contract.PARTIAL_POINTS_PATH, seed
            )
            contract.write_mesh_ply(
                runner_dir / contract.sample_mesh_name(k), vertices, faces
            )
            last_sample_s = clock() - sample_started
            current.samples.append(
                contract.SampleRecord(k=k, seed=seed, time_s=last_sample_s)
            )
            current.metrics["samples"].append(
                {"k": k, "seed": seed, "time_s": last_sample_s, **sample_metrics}
            )
            current.write("running")
    except Exception as exc:  # noqa: BLE001 -- any failure becomes the result's error, samples kept.
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
        current.metrics["total_s"] = clock() - started
        current.write("failed", error)
        print(f"A-12 RUNNER: FAILED - {error}", file=sys.stderr)
        return EXIT_FAILED

    current.metrics["total_s"] = clock() - started
    current.write(status)
    print(f"A-12 RUNNER: {status.upper()} - {len(current.samples)} sample(s)")
    return EXIT_OK


def _repo_commit() -> str | None:
    """This repository's commit, recorded so a measurement says which runner produced it."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


def main(
    argv: list[str] | None = None,
    backend_factory: BackendFactory | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    started = clock()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--request", type=Path, help="the completion folder's request.json"
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="check the environment, without loading weights",
    )
    action.add_argument(
        "--ensure-encoder-entry",
        action="store_true",
        help="add the sparse-structure encoder to pipeline.json if missing (setup runs this)",
    )
    parser.add_argument(
        "--lascomp-root",
        type=Path,
        default=Path(os.environ.get(LASCOMP_ROOT_ENV, str(Path.home() / "LaS-Comp"))),
    )
    parser.add_argument("--cuda-launch-blocking", choices=BLOCKING_CHOICES, default="1")
    args = parser.parse_args(argv)
    lascomp_root = args.lascomp_root.resolve()

    if args.ensure_encoder_entry:
        try:
            changed = ensure_sparse_structure_encoder(
                lascomp_root / CKPT_DIR / "pipeline.json"
            )
        except (RunnerError, OSError, ValueError) as exc:
            print(f"A-12 RUNNER: FAIL - {exc}", file=sys.stderr)
            return EXIT_FAILED
        print(
            f"A-12 RUNNER: PASS - encoder entry {'added' if changed else 'already present'}"
        )
        return EXIT_OK

    if args.check:
        found, problems = verify_pins(lascomp_root, contract.ExpectedModel())
        if problems:
            print(f"A-12 RUNNER: FAIL - {'; '.join(problems)}", file=sys.stderr)
            return EXIT_FAILED
        try:
            report = LasCompBackend(
                lascomp_root,
                read_pins_file(lascomp_root),
                args.cuda_launch_blocking,
                None,
            ).check()
        except Exception as exc:  # noqa: BLE001 -- setup's last step reports any failure plainly.
            traceback.print_exc()
            print(f"A-12 RUNNER: FAIL - {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_FAILED
        # Checked again after the import. A run checks the pins before importing, so anything an
        # import changes would otherwise only surface on the first real run, after setup had
        # already passed. That is how A-12's first Colab run lost all four measurements.
        _, problems = verify_pins(lascomp_root, contract.ExpectedModel())
        if problems:
            print(
                "A-12 RUNNER: FAIL - importing LaS-Comp changed its checkout: "
                f"{'; '.join(problems)}",
                file=sys.stderr,
            )
            return EXIT_FAILED
        print(json.dumps({"pins": found, **report}, indent=2, default=str))
        print("A-12 RUNNER: PASS - environment ready")
        return EXIT_OK

    return run(
        args.request,
        lascomp_root,
        args.cuda_launch_blocking,
        backend_factory or _real_backend,
        clock,
        started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
