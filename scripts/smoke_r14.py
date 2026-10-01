#!/usr/bin/env python3
"""R-14 Lightning smoke check: real golden frames through the frozen HTTP flow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from recon.frames import find_capture_frames
from recon.service import create_app


def run_smoke(
    capture_dir: Path,
    data_dir: Path,
    backend: str,
) -> dict:
    """Upload a real capture through R-14 and require a downloadable non-empty GLB."""
    capture_dir = Path(capture_dir)
    data_dir = Path(data_dir)
    frames = find_capture_frames(capture_dir)
    if len(frames) < 3:
        raise ValueError(f"Smoke check needs at least 3 frames; found {len(frames)}.")

    if data_dir.exists():
        raise FileExistsError(
            f"{data_dir} already exists. Remove it deliberately or choose --data-dir."
        )

    previous_backend = os.environ.get("PROOFSHAPE_RECON_BACKEND")
    os.environ["PROOFSHAPE_RECON_BACKEND"] = backend
    try:
        app = create_app(data_dir=data_dir, known_order_codes={"R14-SMOKE"})
        with TestClient(app) as client:
            created = client.post("/v1/sessions", json={"order_code": "R14-SMOKE"})
            created.raise_for_status()
            session_id = created.json()["session_id"]

            gyro = json.dumps({"alpha": 0.0, "beta": 0.0, "gamma": 0.0})
            for expected_index, frame_path in enumerate(frames):
                with frame_path.open("rb") as handle:
                    uploaded = client.post(
                        f"/v1/sessions/{session_id}/frames",
                        files={
                            "frame": (
                                frame_path.name,
                                handle.read(),
                                "image/jpeg",
                            ),
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

            result = client.get(f"/v1/sessions/{session_id}/reconstruction")
            result.raise_for_status()
            result_body = result.json()
            if result_body["status"] != "complete":
                raise RuntimeError(
                    "R-14 did not complete reconstruction: "
                    + json.dumps(result_body, sort_keys=True)
                )

            model = client.get(result_body["glb_url"])
            model.raise_for_status()
            if len(model.content) == 0:
                raise RuntimeError("R-14 returned an empty GLB.")

            saved_glb = data_dir / "smoke-result.glb"
            saved_glb.write_bytes(model.content)
            result_body["downloaded_glb_bytes"] = len(model.content)
            result_body["saved_glb"] = str(saved_glb)
            return result_body
    except Exception:
        # Keep the failed session directory for diagnosis; only a successful run is
        # eligible for --cleanup below.
        raise
    finally:
        if previous_backend is None:
            os.environ.pop("PROOFSHAPE_RECON_BACKEND", None)
        else:
            os.environ["PROOFSHAPE_RECON_BACKEND"] = previous_backend


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real golden frames through the R-14 HTTP service on Lightning."
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/proofshape-r14-smoke"),
    )
    parser.add_argument(
        "--backend",
        choices=("vggt", "mast3r", "colmap"),
        default="vggt",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove the smoke output after a successful check",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    result = run_smoke(args.capture_dir, args.data_dir, args.backend)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "R-14 SMOKE: PASS - "
        f"{result['downloaded_glb_bytes']} GLB bytes via getReconstruction"
    )
    if args.cleanup:
        shutil.rmtree(args.data_dir)
        print(f"Removed smoke output {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
