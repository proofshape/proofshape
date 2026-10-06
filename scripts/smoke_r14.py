#!/usr/bin/env python3
"""R-14/R-15 smoke check: real golden frames through the frozen HTTP flow.

In-process (R-14, on the Lightning Studio):
    python scripts/smoke_r14.py fixtures/data/golden_capture/s-04

Against a deployed service (R-15), with the bearer token in an environment variable:
    PROOFSHAPE_ENDPOINT_TOKEN=... python scripts/smoke_r14.py \
        fixtures/data/golden_capture/s-04 --base-url https://<deployment-url>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from recon.frames import find_capture_frames

SMOKE_ORDER_CODE = "R14-SMOKE"
REMOTE_POLL_TIMEOUT_S = 15 * 60
REMOTE_POLL_INTERVAL_S = 5.0
LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def drive_session(
    client: httpx.Client,
    frames: Sequence[Path],
    poll_timeout_s: float = REMOTE_POLL_TIMEOUT_S,
    poll_interval_s: float = REMOTE_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], bytes]:
    """Create, upload, finish, poll and download; require a complete, non-empty GLB."""
    if len(frames) < 3:
        raise ValueError(f"Smoke check needs at least 3 frames; found {len(frames)}.")

    created = client.post("/v1/sessions", json={"order_code": SMOKE_ORDER_CODE})
    created.raise_for_status()
    session_id = created.json()["session_id"]

    gyro = json.dumps({"alpha": 0.0, "beta": 0.0, "gamma": 0.0})
    for expected_index, frame_path in enumerate(frames):
        uploaded = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={
                "frame": (frame_path.name, frame_path.read_bytes(), "image/jpeg"),
                "gyro": ("blob", gyro, "application/json"),
            },
        )
        uploaded.raise_for_status()
        if uploaded.json()["frame_index"] != expected_index:
            raise RuntimeError(
                "Frame index drifted while uploading the golden capture."
            )

    finished = client.post(f"/v1/sessions/{session_id}/finish")
    finished.raise_for_status()

    deadline = clock() + poll_timeout_s
    while True:
        result = client.get(f"/v1/sessions/{session_id}/reconstruction")
        result.raise_for_status()
        result_body = result.json()
        if result_body["status"] != "pending":
            break
        if clock() >= deadline:
            raise TimeoutError(
                f"Reconstruction still pending after {poll_timeout_s:.0f} s "
                f"(session {session_id})."
            )
        sleep(poll_interval_s)

    if result_body["status"] != "complete":
        raise RuntimeError(
            "Reconstruction did not complete: "
            + json.dumps(result_body, sort_keys=True)
        )

    model = client.get(result_body["glb_url"])
    model.raise_for_status()
    if len(model.content) == 0:
        raise RuntimeError("The service returned an empty GLB.")
    return result_body, model.content


def run_smoke(
    capture_dir: Path,
    data_dir: Path,
    backend: str,
) -> dict:
    """Upload a real capture through R-14 in-process and require a downloadable GLB."""
    from fastapi.testclient import TestClient

    from recon.service import create_app

    data_dir = Path(data_dir)
    frames = find_capture_frames(Path(capture_dir))
    if data_dir.exists():
        raise FileExistsError(
            f"{data_dir} already exists. Remove it deliberately or choose --data-dir."
        )

    previous_backend = os.environ.get("PROOFSHAPE_RECON_BACKEND")
    os.environ["PROOFSHAPE_RECON_BACKEND"] = backend
    try:
        app = create_app(data_dir=data_dir, known_order_codes={SMOKE_ORDER_CODE})
        with TestClient(app) as client:
            result_body, glb = drive_session(client, frames)
    finally:
        if previous_backend is None:
            os.environ.pop("PROOFSHAPE_RECON_BACKEND", None)
        else:
            os.environ["PROOFSHAPE_RECON_BACKEND"] = previous_backend

    saved_glb = data_dir / "smoke-result.glb"
    saved_glb.write_bytes(glb)
    result_body["downloaded_glb_bytes"] = len(glb)
    result_body["saved_glb"] = str(saved_glb)
    return result_body


def require_secure_base_url(base_url: str) -> None:
    """A deployed endpoint must be HTTPS (D-036); plain HTTP is only for a local container."""
    parts = urlsplit(base_url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and parts.hostname in LOCAL_HOSTS:
        return
    raise ValueError(
        f"{base_url!r} is not HTTPS. R-15 requires TLS; plain http is accepted only for "
        "localhost."
    )


def require_glb_url_scheme(base_url: str, glb_url: str) -> None:
    """Catch a TLS-terminating proxy downgrading the GLB link the service hands out."""
    expected = urlsplit(base_url).scheme
    actual = urlsplit(glb_url).scheme
    if actual != expected:
        raise RuntimeError(
            f"glb_url {glb_url!r} uses {actual!r} but the service was called over "
            f"{expected!r}; the proxy's X-Forwarded-Proto is not being honoured."
        )


def run_remote_smoke(
    capture_dir: Path,
    base_url: str,
    token: str,
    output_glb: Path,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> dict:
    """Drive a deployed service over HTTP(S) with real frames and save the returned GLB."""
    require_secure_base_url(base_url)
    frames = find_capture_frames(Path(capture_dir))
    output_glb = Path(output_glb)
    with client_factory(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    ) as client:
        result_body, glb = drive_session(client, frames)
    require_glb_url_scheme(base_url, result_body["glb_url"])
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    output_glb.write_bytes(glb)
    result_body["downloaded_glb_bytes"] = len(glb)
    result_body["saved_glb"] = str(output_glb)
    return result_body


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real golden frames through the reconstruction HTTP service."
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/proofshape-r14-smoke"),
        help="in-process mode: where the service stores the session",
    )
    parser.add_argument(
        "--backend",
        choices=("vggt", "mast3r", "colmap"),
        default="vggt",
        help="in-process mode only; a deployed service uses its own backend",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="in-process mode: remove the smoke output after a successful check",
    )
    parser.add_argument(
        "--base-url",
        help="drive a deployed service at this URL instead of running in-process (R-15)",
    )
    parser.add_argument(
        "--token-env",
        default="PROOFSHAPE_ENDPOINT_TOKEN",
        help="environment variable holding the deployed service's bearer token",
    )
    parser.add_argument(
        "--output-glb",
        type=Path,
        default=Path("/tmp/proofshape-r15-smoke/smoke-result.glb"),
        help="remote mode: where to save the returned GLB",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.base_url:
        token = os.environ.get(args.token_env, "").strip()
        if not token:
            raise SystemExit(
                f"{args.token_env} is not set. Export the endpoint token first; it is never "
                "passed on the command line."
            )
        result = run_remote_smoke(
            args.capture_dir, args.base_url, token, args.output_glb
        )
        label = f"R-15 SMOKE ({args.base_url})"
    else:
        result = run_smoke(args.capture_dir, args.data_dir, args.backend)
        label = "R-14 SMOKE"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"{label}: PASS - {result['downloaded_glb_bytes']} GLB bytes via getReconstruction"
    )
    if args.cleanup and not args.base_url:
        shutil.rmtree(args.data_dir)
        print(f"Removed smoke output {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
