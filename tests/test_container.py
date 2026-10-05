"""R-15 container and workflow invariants that would otherwise only fail on a paid GPU deploy."""

import os
import re
import runpy
from pathlib import Path

import yaml

from recon.vggt_runner import DEFAULT_MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
ENTRYPOINT_PATH = ROOT / "scripts" / "container_entrypoint.sh"
ENTRYPOINT = ENTRYPOINT_PATH.read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
DEPLOY = runpy.run_path(str(ROOT / "scripts" / "deploy_recon.py"))
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    # PyYAML reads the bare `on:` key as boolean True.
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_base_image_is_pinned_by_tag_and_digest():
    base = re.search(r"^FROM (\S+)", DOCKERFILE, re.MULTILINE).group(1)
    assert ":latest" not in base
    assert re.search(r":[\w.-]+@sha256:[0-9a-f]{64}$", base), base


def test_torch_is_never_reinstalled_over_the_base_image():
    for requirements in ("requirements.txt", "requirements-gpu.txt"):
        lines = (ROOT / requirements).read_text(encoding="utf-8").splitlines()
        assert not [line for line in lines if re.match(r"torch(vision)?\b", line)]
    for line in DOCKERFILE.splitlines():
        if "pip install" in line:
            assert "torch" not in line, line


def test_only_the_service_files_are_copied_never_the_whole_context():
    assert not re.search(r"^COPY \. ", DOCKERFILE, re.MULTILINE)
    assert not re.search(r"^ADD ", DOCKERFILE, re.MULTILINE)


def test_secrets_captures_and_local_state_stay_out_of_the_build_context():
    for pattern in (
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        ".git",
        "fixtures/data",
        "proofshape-data/",
    ):
        assert pattern in DOCKERIGNORE, pattern


def test_service_runs_one_worker_on_the_deployed_port_behind_the_proxy():
    serve = ENTRYPOINT[ENTRYPOINT.index("exec uvicorn") :]
    assert "recon.service:app" in serve
    assert "--workers 1" in serve
    assert "--forwarded-allow-ips" in serve
    port = DEPLOY["PORT"]
    assert f"--port {port}" in serve
    assert re.search(rf"^EXPOSE {port}$", DOCKERFILE, re.MULTILINE)


def test_entrypoint_prefetches_the_model_the_runner_actually_loads():
    assert "from recon.vggt_runner import DEFAULT_MODEL_ID" in ENTRYPOINT
    assert "snapshot_download(DEFAULT_MODEL_ID)" in ENTRYPOINT
    assert DEFAULT_MODEL_ID == "facebook/VGGT-1B"
    assert ENTRYPOINT.index("snapshot_download") < ENTRYPOINT.index("exec uvicorn")


def test_entrypoint_is_executable():
    assert os.access(ENTRYPOINT_PATH, os.X_OK)


def test_deployment_is_never_autoscaled_past_one_replica():
    assert DEPLOY["REPLICAS"] == 1


def test_deploy_workflow_is_manual_main_only_and_uses_no_literal_credentials():
    workflow = _workflow("deploy-recon.yml")
    assert set(workflow[True]) == {"workflow_dispatch"}
    job = workflow["jobs"]["deploy"]
    assert job["if"] == "github.ref == 'refs/heads/main'"

    deploy_step = next(s for s in job["steps"] if "deploy_recon.py" in s.get("run", ""))
    for name, value in deploy_step["env"].items():
        assert value.startswith("${{ "), f"{name} must come from secrets/vars/inputs"
    assert deploy_step["env"]["LIGHTNING_API_KEY"] == "${{ secrets.LIGHTNING_API_KEY }}"


def test_deploy_workflow_deploys_the_image_by_digest():
    text = (WORKFLOWS / "deploy-recon.yml").read_text(encoding="utf-8")
    assert "PROOFSHAPE_IMAGE=${IMAGE}@${digest}" in text


def test_image_is_published_only_from_main_and_smoke_tested_first():
    job = _workflow("recon-image.yml")["jobs"]["image"]
    names = [step["name"] for step in job["steps"]]
    publish = next(s for s in job["steps"] if s["name"] == "Publish to GHCR")
    assert (
        publish["if"]
        == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    assert names.index("Smoke-test the running container") < names.index(
        "Publish to GHCR"
    )
