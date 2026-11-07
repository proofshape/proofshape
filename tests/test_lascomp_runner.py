"""A-12: the LaS-Comp runner (scripts/lascomp_runner.py), with LaS-Comp itself faked.

CI has no GPU and no torch, and LaS-Comp is CUDA-only (D-057), so everything except
`LasCompBackend` is tested here against a fake backend: the request and pin checks, the budget
loop, the result file the service ingests, the failure paths and the checkpoint helpers. The real
model runs in A-12's written Colab check.
"""

from __future__ import annotations

import ast
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recon import completion
from recon import completion_contract as contract
from recon.vggt_runner import write_ply

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "lascomp_runner.py"
NS = runpy.run_path(str(RUNNER))
main = NS["main"]
RunnerError = NS["RunnerError"]
# Kept before any test patches contract.ExpectedModel, so a patch can still build the real one.
REAL_EXPECTED_MODEL = contract.ExpectedModel

_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ]
)  # fmt: skip


# --- fixtures: a fake LaS-Comp checkout and a prepared completion folder --------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def lascomp_root(tmp_path) -> tuple[Path, str]:
    """A git checkout standing in for LaS-Comp, with the pins file setup would write."""
    root = tmp_path / "LaS-Comp"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "run_lascomp_image_condition_single.py").write_text(
        "# stand-in\n", encoding="utf-8"
    )
    _git(root, "add", ".")
    _git(
        root,
        "-c", "user.email=test@example.com", "-c", "user.name=test",
        "commit", "-q", "-m", "stand-in",
    )  # fmt: skip
    pins = {
        "weights": contract.WEIGHTS_ID,
        "weights_revision": contract.WEIGHTS_REVISION,
        "caches": {},
    }
    (root / NS["PINS_FILE"]).write_text(json.dumps(pins), encoding="utf-8")
    return root, _git(root, "rev-parse", "HEAD")


def _completion_folder(
    tmp_path: Path,
    commit: str,
    seeds=(0, 1, 2),
    budget_s=60.0,
    sparse_structure_steps=None,
    slat_steps=None,
) -> Path:
    """A folder as A-04's prepare leaves it: request, partial cloud (z up, mm) and photo."""
    folder = tmp_path / "completion"
    (folder / "input").mkdir(parents=True)
    grid = np.stack(
        np.meshgrid(np.linspace(60, 100, 9), np.linspace(40, 70, 7), [20.0]), -1
    )
    walls = np.stack(
        np.meshgrid([60.0, 100.0], np.linspace(40, 70, 7), np.linspace(3, 20, 6)), -1
    )
    points = np.vstack([grid.reshape(-1, 3), walls.reshape(-1, 3)]).astype(np.float32)
    write_ply(
        folder / contract.PARTIAL_POINTS_PATH,
        points,
        np.full(points.shape, 255, dtype=np.uint8),
    )
    Image.new("RGB", (64, 64), (200, 30, 30)).save(folder / contract.VIEW_IMAGE_PATH)
    contract.write_request(
        folder,
        contract.CompletionRequest(
            seeds=tuple(seeds),
            budget_s=budget_s,
            sparse_structure_steps=sparse_structure_steps,
            slat_steps=slat_steps,
            expected_model=contract.ExpectedModel(commit=commit),
        ),
    )
    return folder


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeBackend:
    """Stands in for LasCompBackend: a box around the partial cloud, jittered by the seed."""

    def __init__(
        self,
        clock: FakeClock | None = None,
        load_cost_s: float = 0.0,
        sample_cost_s: float = 0.0,
        fail_on_sample: int | None = None,
        fail_on_load: bool = False,
        on_sample=None,
    ) -> None:
        self.clock = clock
        self.load_cost_s = load_cost_s
        self.sample_cost_s = sample_cost_s
        self.fail_on_sample = fail_on_sample
        self.fail_on_load = fail_on_load
        self.on_sample = on_sample
        self.loaded_with = None
        self.load_calls = 0
        self.sample_calls = 0

    def _spend(self, seconds: float) -> None:
        if self.clock is not None:
            self.clock.now += seconds

    def load(self, image_path, steps):
        self.load_calls += 1
        assert Path(image_path).is_file()
        if self.fail_on_load:
            raise RuntimeError("fake load failure")
        self.loaded_with = steps
        self._spend(self.load_cost_s)
        return {"fake": True}

    def sample(self, partial_path, seed):
        k = self.sample_calls
        self.sample_calls += 1
        if self.on_sample is not None:
            self.on_sample(k)
        if self.fail_on_sample == k:
            raise RuntimeError(f"fake failure on sample {k}")
        partial = contract.read_points_ply(partial_path)
        low, high = partial.min(axis=0), partial.max(axis=0)
        corners = np.array(
            [[x, y, z] for z in (low[2], high[2]) for x, y in
             ((low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1]))]
        )  # fmt: skip
        corners = corners + np.random.default_rng(seed).normal(0.0, 0.5, corners.shape)
        self._spend(self.sample_cost_s)
        return corners, _BOX_FACES, {"fake_seed": seed}


def _run(folder: Path, root: Path, backend: FakeBackend, clock=None, extra=()):
    calls = []

    def factory(lascomp_root, pins, blocking, runner_dir):
        calls.append((lascomp_root, pins, blocking, runner_dir))
        return backend

    argv = [
        "--request",
        str(folder / contract.REQUEST_FILE),
        "--lascomp-root",
        str(root),
        *extra,
    ]
    if clock is None:
        exit_code = main(argv, backend_factory=factory)
    else:
        exit_code = main(argv, backend_factory=factory, clock=clock)
    return exit_code, calls


def _result(folder: Path) -> dict:
    path = folder / contract.RUNNER_DIR / contract.RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8"))


# --- what the runner may import -------------------------------------------------------------------


def test_runner_is_python_310_syntax_without_311_only_apis():
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    for token in ("datetime.UTC", "tomllib", "Self", "contextlib.chdir", "file_digest",
                  "add_note", "ExceptionGroup", "except*", "TaskGroup"):  # fmt: skip
        assert token not in source, (
            f"{token} is not available in LaS-Comp's Python 3.10"
        )


def test_module_level_imports_are_stdlib_numpy_and_the_contract_only():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imported = set()
    for node in (
        tree.body
    ):  # module level only; torch and LaS-Comp are imported inside functions
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    stdlib = {"__future__", "argparse", "json", "os", "subprocess", "sys", "time", "traceback",
              "collections", "pathlib", "typing"}  # fmt: skip
    assert imported - stdlib == {"numpy", "recon"}


def test_loading_the_runner_imports_no_torch_and_changes_no_environment():
    probe = (
        "import os, runpy, sys\n"
        "before = dict(os.environ)\n"
        f"runpy.run_path({str(RUNNER)!r})\n"
        "assert 'torch' not in sys.modules, 'torch imported at module load'\n"
        "assert dict(os.environ) == before, 'environment changed at module load'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_recon_package_init_stays_import_free():
    # The runner imports recon.completion_contract in a 3.10 venv without the service's
    # dependencies, so recon/__init__.py must never grow an import.
    tree = ast.parse((REPO_ROOT / "recon" / "__init__.py").read_text(encoding="utf-8"))
    assert not [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


# --- a run ----------------------------------------------------------------------------------------


def test_a_normal_run_writes_meshes_the_service_ingests(tmp_path, lascomp_root):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)

    exit_code, calls = _run(folder, root, FakeBackend())

    assert exit_code == 0
    assert _result(folder)["status"] == "complete"
    assert calls[0][2] == "1"  # A-10's blocking launches unless asked otherwise
    request = contract.read_request(folder / contract.REQUEST_FILE)
    result, samples, rejected = completion.ingest_runner_outputs(folder, request)
    # The service's own strict reader and scale check accept every mesh.
    assert (result.status, len(samples), rejected) == ("complete", 3, [])
    assert [sample["seed"] for sample in samples] == [0, 1, 2]
    metrics = json.loads((folder / "runner" / NS["METRICS_FILE"]).read_text())
    assert [entry["fake_seed"] for entry in metrics["samples"]] == [0, 1, 2]
    assert metrics["load"] == {"fake": True}


def test_the_result_shows_each_sample_as_it_lands(tmp_path, lascomp_root):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)
    seen = []

    def look(k):
        result = _result(folder)
        seen.append((k, result["status"], len(result["samples"])))

    _run(folder, root, FakeBackend(on_sample=look))

    assert seen == [(0, "running", 0), (1, "running", 1), (2, "running", 2)]


def test_the_budget_stops_before_a_sample_that_would_end_past_it(
    tmp_path, lascomp_root
):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head, seeds=(0, 1, 2, 3), budget_s=75.0)
    clock = FakeClock()

    # Load 10 s, samples 30 s: after two the clock reads 70, and a third would end at 100.
    exit_code, _ = _run(
        folder, root, FakeBackend(clock, load_cost_s=10, sample_cost_s=30), clock=clock
    )

    result = _result(folder)
    assert exit_code == 0
    assert result["status"] == "budget_reached"
    assert [sample["k"] for sample in result["samples"]] == [0, 1]
    assert result["load_s"] == pytest.approx(10.0)
    assert [sample["time_s"] for sample in result["samples"]] == [30.0, 30.0]


def test_sample_zero_runs_even_when_loading_used_the_whole_budget(
    tmp_path, lascomp_root
):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head, budget_s=75.0)
    clock = FakeClock()

    _run(
        folder, root, FakeBackend(clock, load_cost_s=100, sample_cost_s=30), clock=clock
    )

    result = _result(folder)
    assert (result["status"], len(result["samples"])) == ("budget_reached", 1)


def test_a_failed_sample_keeps_the_earlier_ones_and_the_service_calls_it_partial(
    tmp_path, lascomp_root
):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)

    exit_code, _ = _run(folder, root, FakeBackend(fail_on_sample=1))

    result = _result(folder)
    assert exit_code == 1
    assert result["status"] == "failed"
    assert "fake failure on sample 1" in result["error"]
    assert len(result["samples"]) == 1
    request = contract.read_request(folder / contract.REQUEST_FILE)
    parsed, samples, rejected = completion.ingest_runner_outputs(folder, request)
    status, _ = completion._classify(parsed, samples, rejected, ended_normally=False)
    assert status == "partial"


def test_a_failed_load_fails_with_no_samples(tmp_path, lascomp_root):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)

    exit_code, _ = _run(folder, root, FakeBackend(fail_on_load=True))

    result = _result(folder)
    assert exit_code == 1
    assert (result["status"], result["samples"], result["load_s"]) == (
        "failed",
        [],
        None,
    )
    # Still a contract-valid result: the service records the failure, it doesn't choke on it.
    contract.parse_result(
        folder / "runner" / "result.json",
        contract.read_request(folder / "request.json"),
    )


def test_a_malformed_request_is_refused_and_nothing_is_written(tmp_path, lascomp_root):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)
    (folder / contract.REQUEST_FILE).write_text("{not json", encoding="utf-8")

    exit_code, calls = _run(folder, root, FakeBackend())

    assert exit_code == NS["EXIT_REFUSED"]
    assert calls == []
    assert not (folder / contract.RUNNER_DIR).exists()


def test_a_runner_folder_with_files_is_left_byte_identical(tmp_path, lascomp_root):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)
    stale = folder / contract.RUNNER_DIR / contract.RESULT_FILE
    stale.parent.mkdir()
    stale.write_text('{"status": "complete"}', encoding="utf-8")

    exit_code, calls = _run(folder, root, FakeBackend())

    assert exit_code == NS["EXIT_REFUSED"]
    assert calls == []
    assert stale.read_text(encoding="utf-8") == '{"status": "complete"}'


@pytest.mark.parametrize(
    ("sparse", "slat", "expected"),
    [
        (None, None, {"sparse_structure": {}, "slat": {}}),
        (50, None, {"sparse_structure": {"steps": 50}, "slat": {}}),
        (None, 12, {"sparse_structure": {}, "slat": {"steps": 12}}),
    ],
)
def test_sampling_steps_come_from_the_request(
    tmp_path, lascomp_root, sparse, slat, expected
):
    root, head = lascomp_root
    folder = _completion_folder(
        tmp_path, head, sparse_structure_steps=sparse, slat_steps=slat
    )
    backend = FakeBackend()

    _run(folder, root, backend)

    assert backend.loaded_with == expected


def test_the_blocking_choice_reaches_the_backend(tmp_path, lascomp_root):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)

    _, calls = _run(folder, root, FakeBackend(), extra=("--cuda-launch-blocking", "0"))

    assert calls[0][2] == "0"


# --- pins -----------------------------------------------------------------------------------------


def _dirty_tracked_file(root: Path) -> None:
    (root / "run_lascomp_image_condition_single.py").write_text(
        "# edited\n", encoding="utf-8"
    )


def _remove_pins_file(root: Path) -> None:
    (root / NS["PINS_FILE"]).unlink()


def _wrong_weights_revision(root: Path) -> None:
    pins = json.loads((root / NS["PINS_FILE"]).read_text())
    pins["weights_revision"] = "0" * 40
    (root / NS["PINS_FILE"]).write_text(json.dumps(pins))


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (_dirty_tracked_file, "edited tracked files"),
        (_remove_pins_file, "missing"),
        (_wrong_weights_revision, "weights revision"),
    ],
)
def test_pin_problems_fail_before_the_model_loads(
    tmp_path, lascomp_root, break_it, message
):
    root, head = lascomp_root
    folder = _completion_folder(tmp_path, head)
    break_it(root)
    backend = FakeBackend()

    exit_code, calls = _run(folder, root, backend)

    result = _result(folder)
    assert exit_code == 1
    assert (calls, backend.load_calls) == ([], 0)
    assert result["status"] == "failed"
    assert message in result["error"]


def _commit_bytecode_cache(root: Path) -> Path:
    """Track a __pycache__ file, as LaS-Comp's repository does, and return its path."""
    cache = root / "trellis" / "__pycache__" / "__init__.cpython-310.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"committed bytecode")
    _git(root, "add", ".")
    _git(
        root,
        "-c", "user.email=test@example.com", "-c", "user.name=test",
        "commit", "-q", "-m", "track bytecode",
    )  # fmt: skip
    return cache


def test_a_rewritten_bytecode_cache_doesnt_stop_a_run(tmp_path, lascomp_root):
    # A-12's first Colab run: setup's --check imported LaS-Comp, Python rewrote its committed
    # caches, and every run after that was refused although no code had changed.
    root, _ = lascomp_root
    _commit_bytecode_cache(root).write_bytes(b"recompiled from the tracked source")
    folder = _completion_folder(tmp_path, _git(root, "rev-parse", "HEAD"))

    exit_code, calls = _run(folder, root, FakeBackend())

    assert exit_code == 0
    assert len(calls) == 1
    assert _result(folder)["status"] == "complete"


def test_an_edited_source_is_refused_even_beside_rewritten_caches(
    tmp_path, lascomp_root
):
    root, _ = lascomp_root
    _commit_bytecode_cache(root).write_bytes(b"recompiled")
    _dirty_tracked_file(root)

    _, problems = NS["verify_pins"](
        root, contract.ExpectedModel(commit=_git(root, "rev-parse", "HEAD"))
    )

    assert len(problems) == 1
    assert "run_lascomp_image_condition_single.py" in problems[0]
    assert ".pyc" not in problems[0]


@pytest.mark.parametrize(
    ("line", "reported"),
    [
        (" M trellis/__pycache__/base.cpython-310.pyc", False),
        (" M trellis/base.py", True),
        (" D trellis/__pycache__/base.cpython-310.pyc", True),
        ("M  trellis/__pycache__/base.cpython-310.pyc", True),
        (" M trellis/base.pyc", True),
        (" M __pycache__.pyc", True),
    ],
)
def test_only_a_modified_cache_inside_pycache_is_skipped(line, reported):
    assert NS["edited_tracked_files"](line + "\n") == (
        [line.strip()] if reported else []
    )


class _CheckOnlyBackend:
    """Stands in for LasCompBackend in --check: its check() does what the import would do."""

    def __init__(self, change) -> None:
        self.change = change

    def __call__(self, lascomp_root, pins, blocking, runner_dir):
        self.root = lascomp_root
        return self

    def check(self):
        self.change(self.root)
        return {"fake": True}


@pytest.mark.parametrize(
    ("change", "exit_code"),
    [(_dirty_tracked_file, 1), (lambda root: None, 0)],
)
def test_check_fails_when_importing_lascomp_edits_its_checkout(
    lascomp_root, monkeypatch, capsys, change, exit_code
):
    root, head = lascomp_root
    monkeypatch.setattr(
        contract, "ExpectedModel", lambda: REAL_EXPECTED_MODEL(commit=head)
    )
    monkeypatch.setitem(main.__globals__, "LasCompBackend", _CheckOnlyBackend(change))

    assert main(["--check", "--lascomp-root", str(root)]) == exit_code
    if exit_code:
        assert "importing LaS-Comp changed its checkout" in capsys.readouterr().err


def test_check_passes_when_the_import_only_rewrites_caches(
    lascomp_root, monkeypatch, capsys
):
    root, _ = lascomp_root
    cache = _commit_bytecode_cache(root)
    head = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(
        contract, "ExpectedModel", lambda: REAL_EXPECTED_MODEL(commit=head)
    )
    monkeypatch.setitem(
        main.__globals__,
        "LasCompBackend",
        _CheckOnlyBackend(lambda _: cache.write_bytes(b"recompiled")),
    )

    assert main(["--check", "--lascomp-root", str(root)]) == 0
    assert "PASS - environment ready" in capsys.readouterr().out


def test_a_different_commit_is_reported_as_found_so_the_service_rejects_it(
    tmp_path, lascomp_root
):
    root, _ = lascomp_root
    folder = _completion_folder(
        tmp_path, contract.MODEL_COMMIT
    )  # the real pin, not this repo's

    exit_code, calls = _run(folder, root, FakeBackend())

    assert (exit_code, calls) == (1, [])
    assert _result(folder)["model"]["commit"] == _git(root, "rev-parse", "HEAD")
    with pytest.raises(contract.ContractError, match="pins"):
        contract.parse_result(
            folder / "runner" / "result.json",
            contract.read_request(folder / "request.json"),
        )


# --- the checkpoint helpers -----------------------------------------------------------------------


def _checkpoint(ckpt: Path, models: dict, encoder_files: bool = True) -> Path:
    (ckpt / "ckpts").mkdir(parents=True)
    names = list(models.values()) + ([NS["ENCODER_VALUE"]] if encoder_files else [])
    for name in names:
        for suffix in (".json", ".safetensors"):
            (ckpt / f"{name}{suffix}").write_text("{}", encoding="utf-8")
    (ckpt / "pipeline.json").write_text(
        json.dumps({"name": "TrellisImageTo3DPipeline", "args": {"models": models}}),
        encoding="utf-8",
    )
    return ckpt


def test_the_encoder_entry_is_added_once_and_then_left_alone(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt", {"slat_decoder_mesh": "ckpts/slat_dec_mesh"})
    pipeline_json = ckpt / "pipeline.json"

    assert NS["ensure_sparse_structure_encoder"](pipeline_json) is True
    patched = pipeline_json.read_bytes()
    assert NS["ensure_sparse_structure_encoder"](pipeline_json) is False
    assert pipeline_json.read_bytes() == patched
    models = json.loads(patched)["args"]["models"]
    assert models[NS["ENCODER_KEY"]] == NS["ENCODER_VALUE"]
    assert models["slat_decoder_mesh"] == "ckpts/slat_dec_mesh"


def test_checking_the_encoder_entry_never_writes(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt", {"slat_decoder_mesh": "ckpts/slat_dec_mesh"})
    before = (ckpt / "pipeline.json").read_bytes()

    with pytest.raises(RunnerError, match="doesn't list"):
        NS["ensure_sparse_structure_encoder"](ckpt / "pipeline.json", check_only=True)

    assert (ckpt / "pipeline.json").read_bytes() == before


def test_a_conflicting_encoder_entry_or_missing_encoder_files_refuse(tmp_path):
    conflicting = _checkpoint(
        tmp_path / "a" / "ckpt", {NS["ENCODER_KEY"]: "ckpts/other"}
    )
    with pytest.raises(RunnerError, match="not"):
        NS["ensure_sparse_structure_encoder"](conflicting / "pipeline.json")

    no_files = _checkpoint(
        tmp_path / "b" / "ckpt", {"slat_decoder_mesh": "x"}, encoder_files=False
    )
    with pytest.raises(RunnerError, match="missing from the download"):
        NS["ensure_sparse_structure_encoder"](no_files / "pipeline.json")


def test_missing_model_files_are_named_before_trellis_hides_them(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt", {"a": "ckpts/a", "b": "ckpts/b"})
    (ckpt / "ckpts" / "b.safetensors").unlink()

    assert NS["missing_model_files"](ckpt) == ["ckpts/b.safetensors"]


def test_ensure_encoder_entry_from_the_command_line(tmp_path):
    root = tmp_path / "LaS-Comp"
    _checkpoint(root / NS["CKPT_DIR"], {"slat_decoder_mesh": "ckpts/slat_dec_mesh"})

    assert main(["--ensure-encoder-entry", "--lascomp-root", str(root)]) == 0
    assert main(["--ensure-encoder-entry", "--lascomp-root", str(root)]) == 0


@pytest.mark.parametrize(
    ("queue_s", "total_s", "blocking"),
    [(0.001, 0.05, False), (0.049, 0.05, True), (0.0, 0.0, True)],
)
def test_launches_count_as_blocking_when_queueing_takes_nearly_as_long_as_running(
    queue_s, total_s, blocking
):
    assert NS["launches_are_blocking"](queue_s, total_s) is blocking


def test_the_runner_calls_lascomp_the_way_a10_measured_it():
    # Changing these changes the model D-057 measured; A-12's lever runs change only what they
    # name (steps through the request, blocking through the flag).
    assert NS["SPCONV_ALGO"] == "native"
    assert NS["VOXEL_RESOLUTION"] == 64
    assert NS["LASCOMP_CALL"] == {
        "alpha_eta": 0.1,
        "optimization_step": 1,
        "lr_sample": 1e-5,
        "rescale_t": 3.0,
    }
