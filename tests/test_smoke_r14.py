"""R-14/R-15 smoke flow: in-process and against a deployed URL."""

import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from recon import service

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_r14.py"
NS = runpy.run_path(str(SCRIPT))
drive_session = NS["drive_session"]


def _frames(tmp_path, count=3):
    capture = tmp_path / "capture"
    capture.mkdir()
    for index in range(count):
        (capture / f"frame{index}.jpg").write_bytes(b"jpeg-data")
    return capture, sorted(capture.glob("*.jpg"))


def _pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    glb_path = artifact_dir / "model.glb"
    glb_path.write_bytes(b"glTF-smoke")
    return {
        "glb_path": str(glb_path),
        "reference_tier_used": "charuco_board",
        "metric": True,
        "stage_timings_s": {"reconstruction": 0.1},
        "warnings": [],
    }


def _failing_pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
    raise service.ReconstructionFailure("board_not_detected", "No board in any frame.")


def test_drives_the_real_service_end_to_end(tmp_path):
    _, frames = _frames(tmp_path)
    app = service.create_app(tmp_path / "data", pipeline_runner=_pipeline)
    with TestClient(app) as client:
        body, glb = drive_session(client, frames)
    assert body["status"] == "complete"
    assert glb == b"glTF-smoke"


def test_a_failed_reconstruction_is_an_error_not_a_pass(tmp_path):
    _, frames = _frames(tmp_path)
    app = service.create_app(tmp_path / "data", pipeline_runner=_failing_pipeline)
    with (
        TestClient(app) as client,
        pytest.raises(RuntimeError, match="board_not_detected"),
    ):
        drive_session(client, frames)


def test_too_few_frames_is_refused_before_any_request(tmp_path):
    _, frames = _frames(tmp_path, count=2)
    with pytest.raises(ValueError, match="at least 3 frames"):
        drive_session(None, frames)


class FakeDeployment:
    """A deployed service as httpx sees it: pending for a while, then complete."""

    def __init__(self, pending_polls: int, glb_scheme: str = "https"):
        self.pending_polls = pending_polls
        self.glb_scheme = glb_scheme
        self.auth_headers: set[str] = set()
        self.uploaded = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth_headers.add(request.headers.get("authorization", ""))
        path = request.url.path
        if path == "/v1/sessions":
            return httpx.Response(
                201,
                json={
                    "session_id": "sess_1",
                    "order_code": "R14-SMOKE",
                    "status": "capturing",
                },
            )
        if path.endswith("/frames"):
            self.uploaded += 1
            return httpx.Response(
                202, json={"accepted": True, "frame_index": self.uploaded - 1}
            )
        if path.endswith("/finish"):
            return httpx.Response(
                202, json={"session_id": "sess_1", "status": "finished"}
            )
        if path.endswith("/reconstruction"):
            if self.pending_polls > 0:
                self.pending_polls -= 1
                return httpx.Response(200, json={"status": "pending"})
            return httpx.Response(
                200,
                json={
                    "status": "complete",
                    "glb_url": f"{self.glb_scheme}://recon.example/v1/sessions/sess_1/model.glb",
                },
            )
        if path.endswith("/model.glb"):
            return httpx.Response(200, content=b"glTF-remote")
        return httpx.Response(404)

    def client_factory(self, **kwargs):
        return httpx.Client(transport=httpx.MockTransport(self.handler), **kwargs)


def test_remote_smoke_polls_until_complete_and_sends_the_token(tmp_path):
    capture, _ = _frames(tmp_path)
    fake = FakeDeployment(pending_polls=2)
    out = tmp_path / "out" / "model.glb"

    result = NS["run_remote_smoke"](
        capture, "https://recon.example", "tok", out, client_factory=fake.client_factory
    )

    assert result["downloaded_glb_bytes"] == len(b"glTF-remote")
    assert out.read_bytes() == b"glTF-remote"
    assert fake.auth_headers == {"Bearer tok"}


def test_remote_smoke_catches_a_proxy_downgrading_the_glb_link(tmp_path):
    capture, _ = _frames(tmp_path)
    fake = FakeDeployment(pending_polls=0, glb_scheme="http")
    with pytest.raises(RuntimeError, match="X-Forwarded-Proto"):
        NS["run_remote_smoke"](
            capture,
            "https://recon.example",
            "tok",
            tmp_path / "m.glb",
            client_factory=fake.client_factory,
        )


def test_pending_forever_times_out(tmp_path):
    _, frames = _frames(tmp_path)
    fake = FakeDeployment(pending_polls=10**6)
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    with (
        fake.client_factory(base_url="https://recon.example") as client,
        pytest.raises(TimeoutError, match="still pending"),
    ):
        drive_session(
            client,
            frames,
            poll_timeout_s=30,
            poll_interval_s=5,
            sleep=sleep,
            clock=lambda: now[0],
        )


@pytest.mark.parametrize(
    "url, ok",
    [
        ("https://recon.example", True),
        ("http://localhost:8000", True),
        ("http://127.0.0.1:8000", True),
        ("http://recon.example", False),
        ("ftp://recon.example", False),
    ],
)
def test_only_https_or_local_http_is_accepted(url, ok):
    if ok:
        NS["require_secure_base_url"](url)
    else:
        with pytest.raises(ValueError, match="not HTTPS"):
            NS["require_secure_base_url"](url)


def test_remote_mode_refuses_to_run_without_the_token(tmp_path, monkeypatch):
    capture, _ = _frames(tmp_path)
    monkeypatch.delenv("PROOFSHAPE_ENDPOINT_TOKEN", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke_r14.py", str(capture), "--base-url", "https://recon.example"],
    )
    with pytest.raises(SystemExit, match="PROOFSHAPE_ENDPOINT_TOKEN is not set"):
        NS["main"]()


def test_in_process_mode_still_saves_the_glb(tmp_path, monkeypatch):
    capture, _ = _frames(tmp_path)
    real_create_app = service.create_app

    def create_app(**kwargs):
        return real_create_app(pipeline_runner=_pipeline, **kwargs)

    monkeypatch.setattr(service, "create_app", create_app)
    result = NS["run_smoke"](capture, tmp_path / "data", "vggt")

    assert json.loads(json.dumps(result))["downloaded_glb_bytes"] == len(b"glTF-smoke")
    assert Path(result["saved_glb"]).read_bytes() == b"glTF-smoke"
