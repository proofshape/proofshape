#!/usr/bin/env python3
"""A-01 live evidence check. Sends local photographs to Google; never run by CI."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import threading
from pathlib import Path

from recon.vlm_check import (
    MODEL,
    REQUEST_TIMEOUT_S,
    GeminiInspector,
    VlmSupervisor,
    atomic_json,
    sampled_indices,
)


def run_check(
    capture_dir: Path,
    screen_photo: Path,
    wrong_part: Path,
    output: Path,
    *,
    api_key: str,
    api_tier: str,
    billing_account: str,
    inspector=None,
) -> dict:
    if (
        not api_key.strip()
        or not billing_account.strip()
        or api_tier not in ("free", "paid")
    ):
        raise ValueError(
            "Set GEMINI_API_KEY and explicitly identify API tier/billing account"
        )
    # The golden capture uses .jpeg; other cameras use .jpg or uppercase suffixes.
    # Keep the original names/bytes intact for the manifest's content-hash check.
    frames = []
    for path in sorted(capture_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"):
            frames.append(path)
    if not frames or not screen_photo.is_file() or not wrong_part.is_file():
        raise ValueError(
            "Need golden JPEG frames (.jpg/.jpeg), a screen photograph and a wrong-part photograph"
        )
    inspect = inspector or GeminiInspector(api_key)
    runs = []
    # Staging and reports stay outside the repository: photographs are never fixtures in git.
    with tempfile.TemporaryDirectory(prefix="proofshape-vlm-") as temporary:
        for label, sources in (
            ("golden", frames),
            ("screen_photo", [frames[0], screen_photo]),
            ("wrong_part", [frames[0], wrong_part]),
        ):
            runs.append(_run_batch(label, sources, Path(temporary) / label, inspect))
    evidence = {
        "requested_model": MODEL,
        "api_tier": api_tier,
        "billing_account": billing_account,
        "runs": runs,
    }
    atomic_json(output, evidence)
    return evidence


def _run_batch(label, sources, session_dir, inspect):
    frame_dir = session_dir / "frames"
    frame_dir.mkdir(parents=True)
    for index, source in enumerate(sources):
        shutil.copyfile(source, frame_dir / f"frame_{index:04d}.jpg")
    calls = 0
    lock = threading.Lock()

    def tracked(frame, reference):
        nonlocal calls
        with lock:
            calls += 1
        return inspect(frame, reference)

    count = len(sampled_indices(len(sources)))
    # A manual batch can enqueue more samples at once than a live capture does.
    # The bound and wait are explicit, calculated from this batch, not unbounded.
    supervisor = VlmSupervisor(
        inspector=tracked,
        max_pending=max(16, count),
        join_timeout_s=REQUEST_TIMEOUT_S * math.ceil(count / 2) + 5,
    )
    try:
        supervisor.schedule_finish(session_dir, len(sources))
        report = supervisor.join(session_dir, len(sources))
    finally:
        supervisor.close()
    report["label"] = label
    report["api_requests_started"] = calls
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--screen-photo", type=Path, required=True)
    parser.add_argument("--wrong-part", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-tier", choices=("free", "paid"), required=True)
    parser.add_argument("--billing-account", required=True)
    args = parser.parse_args(argv)
    try:
        evidence = run_check(
            args.capture_dir,
            args.screen_photo,
            args.wrong_part,
            args.output,
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            api_tier=args.api_tier,
            billing_account=args.billing_account,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Wrote VLM evidence to {args.output}; inspect observations before declaring success."
    )
    for run in evidence["runs"]:
        for check in run["checks"]:
            if check["status"] != "checked":
                return 1
    # A complete API response is not proof of semantic accuracy; a person reads the report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
