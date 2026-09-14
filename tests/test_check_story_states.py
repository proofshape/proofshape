"""Tests for scripts/check_story_states.py.

Covers exactly what was flagged in review on PR #5 (dalwalyk):
- consistent states return success
- a file/index mismatch is reported
- Blocked becomes Ready only when every dependency is Done
- --fix updates both the story file and the index
- the script never changes Done or Claimed

The script reads two module-level paths (STORIES_DIR, INDEX) rather than taking a root as a
parameter, so these tests monkeypatch those two attributes to point at a throwaway directory
built fresh for each test, then exercise the real functions against it. That keeps the tests
honest about what the script actually does, without touching the real stories/ directory.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_story_states.py"


def _load_module():
    """Import check_story_states.py fresh, since it lives outside any installed package."""
    spec = importlib.util.spec_from_file_location("check_story_states", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def css(tmp_path, monkeypatch):
    """A fresh check_story_states module pointed at an empty tmp_path/stories/ directory.

    main() reads sys.argv directly rather than accepting an argument list, so this also gives
    every test a clean argv (just the script name, no --fix) unless a test overrides it - without
    this, pytest's own invocation arguments (e.g. "tests/ -v") leak into the script's argparse
    call and it errors out on them.
    """
    module = _load_module()
    stories_dir = tmp_path / "stories"
    stories_dir.mkdir()
    monkeypatch.setattr(module, "STORIES_DIR", stories_dir)
    monkeypatch.setattr(module, "INDEX", stories_dir / "README.md")
    monkeypatch.setattr(sys, "argv", ["check_story_states.py"])
    return module


def _write_story(css, sid, state, deps="nothing"):
    dep_line = "nothing" if deps == "nothing" else deps
    (css.STORIES_DIR / f"{sid}.md").write_text(
        f"# {sid} · a test story\n\n"
        f"**Area:** foundations · **Tier:** 1 · **Estimate:** 1 h\n"
        f"**Depends on:** {dep_line}\n"
        f"**State:** {state}\n"
        f"**Owner:** _unclaimed_\n",
        encoding="utf-8",
    )


def _write_index(css, rows):
    """rows: list of (sid, deps_str, state) tuples."""
    lines = ["| ID | Story | Est | Depends on | State |", "|---|---|---|---|---|"]
    for sid, deps, state in rows:
        lines.append(f"| [{sid}]({sid}.md) | a test story | 1 h | {deps} | {state} |")
    css.INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_consistent_states_report_success(css, capsys):
    """Everything agrees, nothing is stuck Blocked with satisfied deps -> clean exit."""
    _write_story(css, "F-01", "Done", "nothing")
    _write_story(css, "F-03", "Ready", "nothing")
    # F-02 depends on F-03, which is only Ready (not Done) - correctly still Blocked.
    _write_story(css, "F-02", "Blocked", "F-03")
    _write_index(css, [("F-01", "nothing", "Done"), ("F-02", "F-03", "Blocked"), ("F-03", "nothing", "Ready")])

    exit_code = css.main()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "consistent" in out.lower()


def test_file_index_mismatch_is_reported(css, capsys):
    """Index says Ready, the story's own file still says Blocked -> flagged, not silently ignored."""
    _write_story(css, "F-01", "Blocked", "nothing")
    _write_index(css, [("F-01", "nothing", "Ready")])

    exit_code = css.main()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "F-01" in out
    assert "Ready" in out and "Blocked" in out


def test_blocked_becomes_ready_only_when_every_dependency_is_done(css, capsys):
    """Two dependencies: only flag Ready once BOTH are Done, not when just one is."""
    _write_story(css, "F-05", "Done", "nothing")
    _write_story(css, "F-06", "Ready", "nothing")  # not Done yet
    _write_story(css, "F-07", "Blocked", "F-05, F-06")
    _write_index(css, [
        ("F-05", "nothing", "Done"),
        ("F-06", "nothing", "Ready"),
        ("F-07", "F-05, F-06", "Blocked"),
    ])

    exit_code = css.main()
    out = capsys.readouterr().out
    # F-06 isn't Done yet, so F-07 must NOT be flagged as ready-to-unblock.
    assert exit_code == 0
    assert "F-07" not in out

    # Now finish F-06 too - only then should F-07 be flagged.
    _write_story(css, "F-06", "Done", "nothing")
    _write_index(css, [
        ("F-05", "nothing", "Done"),
        ("F-06", "nothing", "Done"),
        ("F-07", "F-05, F-06", "Blocked"),
    ])

    exit_code = css.main()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "F-07" in out
    assert "should be Ready" in out


def test_fix_updates_both_the_story_file_and_the_index(css, monkeypatch):
    """--fix must flip Blocked -> Ready in both places, not just one."""
    _write_story(css, "F-01", "Done", "nothing")
    _write_story(css, "F-02", "Blocked", "F-01")
    _write_index(css, [("F-01", "nothing", "Done"), ("F-02", "F-01", "Blocked")])

    monkeypatch.setattr(sys, "argv", ["check_story_states.py", "--fix"])
    exit_code = css.main()

    assert exit_code == 0
    file_state, _ = css.file_state("F-02")
    assert file_state == "Ready"
    index_text = css.INDEX.read_text(encoding="utf-8")
    assert "| F-01 | Blocked |" not in index_text  # sanity: didn't touch the wrong row
    rows, _ = css.parse_index()
    assert rows["F-02"]["index_state"] == "Ready"


def test_fix_never_changes_done_or_claimed(css, monkeypatch):
    """--fix must never touch a Done or Claimed story, even one with an unrelated mismatch."""
    _write_story(css, "F-01", "Claimed", "nothing")  # index disagrees on purpose, below
    _write_story(css, "F-09", "Done", "nothing")
    _write_index(css, [("F-01", "nothing", "Done"), ("F-09", "nothing", "Done")])

    before_f01 = (css.STORIES_DIR / "F-01.md").read_text(encoding="utf-8")
    before_f09 = (css.STORIES_DIR / "F-09.md").read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["check_story_states.py", "--fix"])
    css.main()

    # The mismatch on F-01 (Claimed vs Done) is real, but --fix only ever produces
    # Blocked -> Ready. It must be reported, never silently resolved either way.
    after_f01 = (css.STORIES_DIR / "F-01.md").read_text(encoding="utf-8")
    after_f09 = (css.STORIES_DIR / "F-09.md").read_text(encoding="utf-8")
    assert after_f01 == before_f01, "a Claimed story must never be rewritten by --fix"
    assert after_f09 == before_f09, "a Done story must never be rewritten by --fix"
