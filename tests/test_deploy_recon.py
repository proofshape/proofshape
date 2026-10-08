"""R-15 deploy script: main-only, digest-pinned, one replica, HTTPS, token never printed."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_recon.py"
NS = runpy.run_path(str(SCRIPT))
DeployError = NS["DeployError"]

PINNED = "ghcr.io/proofshape/proofshape-recon@sha256:" + "a" * 64
TOKEN = "s3cret-token-value"
READY = {
    "status": {"ready_replicas": 1, "urls": ["https://recon.example.lightning.ai"]}
}


def _env(**overrides):
    env = {
        "GITHUB_REF": "refs/heads/main",
        "LIGHTNING_TEAMSPACE": "owner/proofshape",
        "PROOFSHAPE_ENDPOINT_TOKEN": TOKEN,
        "PROOFSHAPE_IMAGE": PINNED,
    }
    env.update(overrides)
    return env


# The exact message lightning_sdk's CLI raises for an unknown deployment name (confirmed live
# against the real CLI, see stories/R-15.md). Any other non-zero inspect exit is a different kind
# of failure and must not be read the same way.
LIGHTNING_NOT_FOUND_STDERR = "Error: Deployment 'proofshape-recon' was not found."


class FakeLightning:
    """Stands in for the `lightning` CLI; records every call."""

    def __init__(
        self,
        exists: bool,
        inspect_results=None,
        exit_code: int = 0,
        inspect_failure_stderr: str | None = None,
        configure_exit_code: int = 0,
    ):
        self.exists = exists
        self.inspect_results = list(inspect_results or [READY])
        self.exit_code = exit_code
        # None here means "simulate genuine absence"; any other string means "inspect itself is
        # broken" (auth, network, rate limit) -- the distinction the fix is about.
        self.inspect_failure_stderr = inspect_failure_stderr
        # The configure-public-URL step is a second, distinct `update` call (the first deploy
        # call never has --env); kept separate so a failure there can be tested without also
        # failing the initial create/update.
        self.configure_exit_code = configure_exit_code
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        if command[2] == "inspect":
            if self.inspect_failure_stderr is not None:
                return subprocess.CompletedProcess(
                    command, 1, "", self.inspect_failure_stderr
                )
            if not self.exists:
                return subprocess.CompletedProcess(
                    command, 1, "", LIGHTNING_NOT_FOUND_STDERR
                )
            result = (
                self.inspect_results.pop(0)
                if len(self.inspect_results) > 1
                else self.inspect_results[0]
            )
            return subprocess.CompletedProcess(command, 0, json.dumps(result), "")
        if command[2] == "update" and "--env" in command:
            return subprocess.CompletedProcess(
                command, self.configure_exit_code, "", ""
            )
        if command[2] == "create":
            self.exists = True
        return subprocess.CompletedProcess(command, self.exit_code, "", "")


@pytest.mark.parametrize("ref", ["refs/heads/feature", "refs/pull/1/merge", ""])
def test_refuses_to_deploy_from_anything_but_main(ref):
    with pytest.raises(DeployError, match="only from refs/heads/main"):
        NS["require_main"](ref)


def test_refuses_a_moving_tag():
    with pytest.raises(DeployError, match="not pinned by digest"):
        NS["start_command"](
            exists=False,
            teamspace="o/t",
            image="ghcr.io/proofshape/proofshape-recon:main",
            token=TOKEN,
        )


def test_start_command_pins_one_t4_replica_on_the_service_port_with_token_auth():
    command = NS["start_command"](
        exists=False, teamspace="o/t", image=PINNED, token=TOKEN
    )

    assert command[:4] == ["lightning", "deployment", "create", "proofshape-recon"]

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--image") == PINNED
    assert value("--machine") == "T4"
    assert value("--port") == "8000"
    assert value("--replicas") == "1"
    assert value("--token-auth") == TOKEN
    assert int(value("--max-runtime")) > 0
    assert "--min-replicas" not in command and "--max-replicas" not in command


def test_existing_deployment_is_updated_not_recreated():
    command = NS["start_command"](
        exists=True, teamspace="o/t", image=PINNED, token=TOKEN
    )
    assert command[:4] == ["lightning", "deployment", "update", "proofshape-recon"]


def test_ready_url_waits_for_a_ready_replica_and_requires_https():
    ready_url = NS["ready_url"]
    assert ready_url({"status": {"ready_replicas": 0, "urls": ["https://x"]}}) is None
    assert ready_url({"status": {"ready_replicas": 1, "urls": []}}) is None
    assert ready_url({}) is None
    assert ready_url(READY) == "https://recon.example.lightning.ai"
    with pytest.raises(DeployError, match="non-HTTPS"):
        ready_url({"status": {"ready_replicas": 1, "urls": ["http://recon.example"]}})


def test_wait_until_ready_polls_then_times_out_with_the_last_status():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    pending = {"status": {"ready_replicas": 0, "message": "pulling image"}}
    with pytest.raises(DeployError, match="pulling image"):
        NS["wait_until_ready"](
            lambda: pending,
            timeout_s=60,
            poll_interval_s=20,
            sleep=sleep,
            clock=lambda: now[0],
        )
    assert sleeps == [20, 20, 20]

    answers = iter([None, pending, READY])
    url = NS["wait_until_ready"](
        lambda: next(answers), timeout_s=600, sleep=lambda s: None, clock=lambda: 0.0
    )
    assert url == "https://recon.example.lightning.ai"


def test_start_creates_waits_and_never_prints_the_token(capsys):
    fake = FakeLightning(exists=False)
    url = NS["start"](_env(), run=fake)

    assert url == "https://recon.example.lightning.ai"
    # inspect (absent) -> create -> inspect (ready) -> configure the public URL -> inspect (ready)
    assert [call[2] for call in fake.calls] == [
        "inspect",
        "create",
        "inspect",
        "update",
        "inspect",
    ]
    assert TOKEN not in capsys.readouterr().out


def test_configure_public_url_command_sets_the_env_var_on_the_right_deployment():
    command = NS["configure_public_url_command"](
        teamspace="o/t",
        image=PINNED,
        token=TOKEN,
        url="https://recon.example.lightning.ai",
    )

    assert command[:4] == ["lightning", "deployment", "update", "proofshape-recon"]
    assert command[command.index("--teamspace") + 1] == "o/t"
    assert command[command.index("--env") + 1] == (
        f"{NS['PUBLIC_URL_ENV_NAME']}=https://recon.example.lightning.ai"
    )


def test_configure_public_url_command_resends_the_full_deploy_config():
    """dalwalyk's PR #94 review: this call must not depend on whether `lightning deployment
    update` patches or replaces config -- resending every flag makes that question irrelevant,
    so it can't silently drop --token-auth or move off the pinned image digest (D-042)."""
    command = NS["configure_public_url_command"](
        teamspace="o/t",
        image=PINNED,
        token=TOKEN,
        url="https://recon.example.lightning.ai",
    )

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--image") == PINNED
    assert value("--machine") == "T4"
    assert value("--port") == "8000"
    assert value("--replicas") == "1"
    assert value("--token-auth") == TOKEN
    assert int(value("--max-runtime")) > 0


def test_start_configures_the_service_with_its_own_real_public_url():
    """The actual fix for mbj1994's PR #94 review: don't depend on unverified proxy forwarding
    (X-Forwarded-Host) -- tell the service its real URL explicitly, once it's known."""
    fake = FakeLightning(exists=False)
    NS["start"](_env(), run=fake)

    configure_call = fake.calls[3]
    assert configure_call[2] == "update"
    assert "--env" in configure_call
    env_value = configure_call[configure_call.index("--env") + 1]
    assert (
        env_value == f"{NS['PUBLIC_URL_ENV_NAME']}=https://recon.example.lightning.ai"
    )
    # dalwalyk's PR #94 review: the configure call must also carry the full deploy config, not
    # just --env, so it can't silently drop auth or the pinned image on a real deployment.
    assert configure_call[configure_call.index("--token-auth") + 1] == TOKEN
    assert configure_call[configure_call.index("--image") + 1] == PINNED


def test_start_never_prints_the_token_in_the_configure_call(capsys):
    """The configure call now carries --token-auth too (see above), so it needs the same
    redaction the first deploy call already gets."""
    fake = FakeLightning(exists=False)
    NS["start"](_env(), run=fake)

    assert TOKEN not in capsys.readouterr().out


def test_start_fails_loudly_if_configuring_the_public_url_fails():
    fake = FakeLightning(exists=False, configure_exit_code=3)
    with pytest.raises(DeployError, match="PROOFSHAPE_PUBLIC_BASE_URL"):
        NS["start"](_env(), run=fake)


def test_deploy_script_and_service_agree_on_the_public_url_env_var_name():
    """Not imported directly (the deploy job's environment has no FastAPI/pydantic installed),
    so this is the drift guard: both files must spell the same literal the same way."""
    service_source = (
        Path(__file__).resolve().parents[1] / "recon" / "service.py"
    ).read_text(encoding="utf-8")
    assert f'PUBLIC_BASE_URL_ENV = "{NS["PUBLIC_URL_ENV_NAME"]}"' in service_source


def test_start_reports_which_setting_is_missing():
    for name in (
        "LIGHTNING_TEAMSPACE",
        "PROOFSHAPE_ENDPOINT_TOKEN",
        "PROOFSHAPE_IMAGE",
    ):
        with pytest.raises(DeployError, match=name):
            NS["start"](_env(**{name: ""}), run=FakeLightning(exists=False))


def test_start_fails_loudly_when_the_cli_fails():
    with pytest.raises(DeployError, match="exited with 2"):
        NS["start"](_env(), run=FakeLightning(exists=False, exit_code=2))


def test_inspect_does_not_mistake_a_broken_cli_call_for_absence():
    """dalwalyk's PR #83 review: a transient inspect failure must not read as 'doesn't exist'."""
    broken = FakeLightning(
        exists=False, inspect_failure_stderr="No Lightning credentials are available."
    )
    with pytest.raises(DeployError, match="No Lightning credentials"):
        NS["inspect_deployment"]("o/t", run=broken)


def test_stop_refuses_to_report_success_when_inspect_itself_fails():
    """The exact bug the review caught: stop() must not silently no-op on a broken inspect,
    leaving a billed GPU running while reporting 'nothing to stop'."""
    broken = FakeLightning(
        exists=True, inspect_failure_stderr="rate limited, try again later"
    )
    with pytest.raises(DeployError, match="rate limited"):
        NS["stop"](_env(), run=broken)
    # And it never reached `delete`, which would have been the wrong recovery too.
    assert [call[2] for call in broken.calls] == ["inspect"]


def test_stop_deletes_only_a_running_deployment():
    absent = FakeLightning(exists=False)
    NS["stop"](_env(), run=absent)
    assert [call[2] for call in absent.calls] == ["inspect"]

    running = FakeLightning(exists=True)
    NS["stop"](_env(), run=running)
    assert running.calls[-1][:4] == [
        "lightning",
        "deployment",
        "delete",
        "proofshape-recon",
    ]
    assert "--yes" in running.calls[-1]


@pytest.mark.parametrize("action", ["start", "stop"])
def test_neither_action_touches_lightning_off_main(action):
    fake = FakeLightning(exists=True)
    with pytest.raises(DeployError, match="only from refs/heads/main"):
        NS[action](_env(GITHUB_REF="refs/heads/feature"), run=fake)
    assert fake.calls == []


def test_main_exits_nonzero_off_main(monkeypatch, capsys):
    for name, value in _env(GITHUB_REF="refs/heads/feature").items():
        monkeypatch.setenv(name, value)
    assert NS["main"](["stop"]) == 1
    err = capsys.readouterr().err
    assert "R-15 DEPLOY: FAILED" in err
    assert "only from refs/heads/main" in err
