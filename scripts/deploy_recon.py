#!/usr/bin/env python3
"""R-15: start or stop the reconstruction service on Lightning (D-036).

`start` deploys, waits for a ready replica, then re-sends the same deploy configuration with
PROOFSHAPE_PUBLIC_BASE_URL set to the now-known real URL added (recon/service.py's
public_model_url reads it), and waits again -- Lightning's proxy does not reliably forward a
usable Host/X-Forwarded-Host to the container, confirmed on a real deploy. The second call
resends every flag the first one used, not just --env, since it's unverified whether `lightning
deployment update` patches or replaces the deployment's config.

Run by .github/workflows/deploy-recon.yml on `main` only (D-013/D-014). Needs the Lightning CLI
(`lightning-sdk`) and these environment variables:

    LIGHTNING_USER_ID, LIGHTNING_API_KEY   Lightning credentials (read by the CLI itself)
    LIGHTNING_TEAMSPACE                    owner/teamspace
    PROOFSHAPE_ENDPOINT_TOKEN              bearer token callers must send (start only)
    PROOFSHAPE_IMAGE                       image pinned by digest, repo@sha256:... (start only)
    GITHUB_REF                             must be refs/heads/main
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

DEPLOYMENT_NAME = "proofshape-recon"
MACHINE = "T4"
PORT = 8000
# One replica, never autoscaled: SessionStore keeps sessions on one container's local disk, so a
# second replica would answer session_not_found for sessions created on the first.
REPLICAS = 1
# Cost guard for on-demand use: Lightning releases the machine after this long even if nobody
# runs `stop`.
MAX_RUNTIME_S = 4 * 60 * 60
MAIN_REF = "refs/heads/main"
READY_TIMEOUT_S = 30 * 60
POLL_INTERVAL_S = 20.0
# Must match recon/service.py's PUBLIC_BASE_URL_ENV exactly -- not imported directly, since this
# script runs in the deploy job's minimal environment (just lightning-sdk, not the full
# requirements.txt recon.service needs). tests/test_deploy_recon.py asserts the two stay in sync.
PUBLIC_URL_ENV_NAME = "PROOFSHAPE_PUBLIC_BASE_URL"

Runner = Callable[..., subprocess.CompletedProcess]


class DeployError(RuntimeError):
    """A deploy precondition or outcome that must stop the workflow loudly."""


def require_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise DeployError(
            f"{name} is not set. See recon/README.md, R-15 one-time setup."
        )
    return value


def require_main(ref: str) -> None:
    if ref != MAIN_REF:
        raise DeployError(
            f"Refusing to deploy from {ref or 'an unknown ref'}: deployments run only from "
            f"{MAIN_REF} (D-013, D-014)."
        )


def require_pinned_image(image: str) -> None:
    if "@sha256:" not in image:
        raise DeployError(
            f"Image {image!r} is not pinned by digest. Deploy repo@sha256:..., never a "
            "moving tag, so the running service is traceable to one build."
        )


def start_command(*, exists: bool, teamspace: str, image: str, token: str) -> list[str]:
    """The `lightning deployment create|update` call; update if the deployment already exists."""
    require_pinned_image(image)
    if exists:
        head = ["lightning", "deployment", "update", DEPLOYMENT_NAME]
    else:
        head = ["lightning", "deployment", "create", DEPLOYMENT_NAME]
    return [
        *head,
        "--teamspace",
        teamspace,
        "--image",
        image,
        "--machine",
        MACHINE,
        "--port",
        str(PORT),
        "--replicas",
        str(REPLICAS),
        "--max-runtime",
        str(MAX_RUNTIME_S),
        "--token-auth",
        token,
    ]


def configure_public_url_command(
    *, teamspace: str, image: str, token: str, url: str
) -> list[str]:
    """Tell the running service its own real public URL.

    Lightning's deployment proxy does not reliably forward a Host/X-Forwarded-Host the container
    can build correct absolute URLs from -- confirmed on a real deploy, the container saw
    Host: localhost. Rather than depend on unverified proxy behavior, set it explicitly once the
    real URL is known (which it only is after the first start), via `recon/service.py`'s
    PROOFSHAPE_PUBLIC_BASE_URL override.

    Built from the exact same full flag set as `start_command`'s update branch, plus --env --
    not just --teamspace and --env alone. Whether `lightning deployment update` patches or
    replaces the deployment's config is unverified (dalwalyk's PR #94 review); resending every
    flag makes that distinction irrelevant, so this can never silently drop --token-auth or move
    the deployment off its pinned image digest (which would break D-042's guarantees).
    """
    return [
        *start_command(exists=True, teamspace=teamspace, image=image, token=token),
        "--env",
        f"{PUBLIC_URL_ENV_NAME}={url}",
    ]


def stop_command(*, teamspace: str) -> list[str]:
    return [
        "lightning",
        "deployment",
        "delete",
        DEPLOYMENT_NAME,
        "--teamspace",
        teamspace,
        "--yes",
    ]


def inspect_command(*, teamspace: str) -> list[str]:
    return [
        "lightning",
        "deployment",
        "inspect",
        DEPLOYMENT_NAME,
        "--teamspace",
        teamspace,
    ]


def redact(command: Sequence[str], secret: str) -> str:
    return " ".join("***" if part == secret else part for part in command)


def ready_url(info: Mapping[str, Any]) -> str | None:
    """The endpoint URL once a replica is ready; refuses anything that isn't HTTPS (D-036)."""
    status = info.get("status") or {}
    urls = status.get("urls") or []
    if (status.get("ready_replicas") or 0) < 1 or not urls:
        return None
    url = urls[0]
    if not url.startswith("https://"):
        raise DeployError(
            f"Lightning reported a non-HTTPS endpoint {url!r}; R-15 requires TLS."
        )
    return url


# The exact text `resolve_deployment` in lightning_sdk's CLI raises as a click.ClickException
# when the named deployment doesn't exist; every other failure (auth, network, rate limit) also
# exits 1, so matching on the specific message is the only way to tell "absent" from "broken".
# Confirmed live: `lightning deployment inspect` against an unknown name with no credentials set
# also exits 1, with an unrelated "No Lightning credentials are available" message.
NOT_FOUND_MARKER = "was not found"


def inspect_deployment(
    teamspace: str, run: Runner = subprocess.run
) -> dict[str, Any] | None:
    """The deployment as JSON, None if it genuinely doesn't exist yet.

    Raises DeployError for any other CLI failure, so a transient error never gets silently read
    as "nothing to stop" (see stop(), and D-042: this deployment bills a GPU by the hour).
    """
    completed = run(
        inspect_command(teamspace=teamspace), capture_output=True, text=True
    )
    if completed.returncode == 0:
        return json.loads(completed.stdout)
    if NOT_FOUND_MARKER in completed.stderr:
        return None
    raise DeployError(
        f"`lightning deployment inspect` failed (exit {completed.returncode}): "
        f"{completed.stderr.strip() or 'no error output'}"
    )


def wait_until_ready(
    inspect: Callable[[], Mapping[str, Any] | None],
    timeout_s: float = READY_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    deadline = clock() + timeout_s
    last_message = "no status reported yet"
    while True:
        info = inspect()
        if info is not None:
            url = ready_url(info)
            if url is not None:
                return url
            last_message = (info.get("status") or {}).get("message") or last_message
        if clock() >= deadline:
            raise DeployError(
                f"{DEPLOYMENT_NAME} had no ready replica after {timeout_s:.0f} s "
                f"(last status: {last_message}). Check `lightning deployment logs "
                f"{DEPLOYMENT_NAME}`."
            )
        sleep(poll_interval_s)


def start(env: Mapping[str, str], run: Runner = subprocess.run) -> str:
    require_main(env.get("GITHUB_REF", ""))
    teamspace = require_env(env, "LIGHTNING_TEAMSPACE")
    token = require_env(env, "PROOFSHAPE_ENDPOINT_TOKEN")
    image = require_env(env, "PROOFSHAPE_IMAGE")
    exists = inspect_deployment(teamspace, run) is not None
    command = start_command(
        exists=exists, teamspace=teamspace, image=image, token=token
    )
    print("Running:", redact(command, token), flush=True)
    completed = run(command)
    if completed.returncode != 0:
        raise DeployError(f"`lightning deployment` exited with {completed.returncode}.")
    url = wait_until_ready(lambda: inspect_deployment(teamspace, run))

    # Now that the real public URL is known, configure the service to use it explicitly (see
    # configure_public_url_command) and wait for the resulting restart to come back ready.
    configure = configure_public_url_command(
        teamspace=teamspace, image=image, token=token, url=url
    )
    print("Running:", redact(configure, token), flush=True)
    completed = run(configure)
    if completed.returncode != 0:
        raise DeployError(
            f"Setting {PUBLIC_URL_ENV_NAME} failed (exit {completed.returncode})."
        )
    return wait_until_ready(lambda: inspect_deployment(teamspace, run))


def stop(env: Mapping[str, str], run: Runner = subprocess.run) -> None:
    require_main(env.get("GITHUB_REF", ""))
    teamspace = require_env(env, "LIGHTNING_TEAMSPACE")
    if inspect_deployment(teamspace, run) is None:
        print(f"{DEPLOYMENT_NAME} is not running; nothing to stop.")
        return
    command = stop_command(teamspace=teamspace)
    print("Running:", " ".join(command), flush=True)
    completed = run(command)
    if completed.returncode != 0:
        raise DeployError(
            f"`lightning deployment delete` exited with {completed.returncode}."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("start", "stop"))
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            url = start(os.environ)
            print(f"R-15 DEPLOY: READY at {url}")
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a", encoding="utf-8") as handle:
                    handle.write(f"Reconstruction service ready: {url}\n")
        else:
            stop(os.environ)
            print("R-15 DEPLOY: STOPPED")
    except DeployError as exc:
        print(f"R-15 DEPLOY: FAILED - {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
