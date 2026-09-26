"""Tests for scripts/check_story_states.py.

Covers exactly what was flagged in review on PR #5 (dalwalyk):
- consistent states return success
- a file/index mismatch is reported
- Blocked becomes Ready only when every dependency is Done
- --fix updates both the story file and the index
- the script never changes Done or Claimed

and, since D-037, the owner checks: the index's Owner cell must match the story file, owned
states need an @github-handle, unowned states must read _unclaimed_, an AI assistant is never an
owner, and a row the parser can't read is reported instead of silently skipped.

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


def _default_owner(state):
    """The owner a well-formed story in this state would have."""
    return "@member" if state in ("Claimed", "In review", "Done") else "_unclaimed_"


def _write_story(css, sid, state, deps="nothing", owner=None):
    dep_line = "nothing" if deps == "nothing" else deps
    owner = _default_owner(state) if owner is None else owner
    (css.STORIES_DIR / f"{sid}.md").write_text(
        f"# {sid} · a test story\n\n"
        f"**Area:** foundations · **Tier:** 1 · **Estimate:** 1 h\n"
        f"**Depends on:** {dep_line}\n"
        f"**State:** {state}\n"
        f"**Owner:** {owner}\n",
        encoding="utf-8",
    )


def _write_index(css, rows):
    """rows: list of (sid, deps_str, state) or (sid, deps_str, state, owner) tuples.

    Without an explicit owner, the row gets the owner a well-formed story in that state has,
    so tests about states don't have to spell out owners they aren't testing.
    """
    lines = [
        "| ID | Story | Est | Depends on | State | Owner |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        sid, deps, state = row[:3]
        owner = row[3] if len(row) > 3 else _default_owner(state)
        lines.append(
            f"| [{sid}]({sid}.md) | a test story | 1 h | {deps} | {state} | {owner} |"
        )
    css.INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_consistent_states_report_success(css, capsys):
    """Everything agrees, nothing is stuck Blocked with satisfied deps -> clean exit."""
    _write_story(css, "F-01", "Done", "nothing")
    _write_story(css, "F-03", "Ready", "nothing")
    # F-02 depends on F-03, which is only Ready (not Done) - correctly still Blocked.
    _write_story(css, "F-02", "Blocked", "F-03")
    _write_index(
        css,
        [
            ("F-01", "nothing", "Done"),
            ("F-02", "F-03", "Blocked"),
            ("F-03", "nothing", "Ready"),
        ],
    )

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
    _write_index(
        css,
        [
            ("F-05", "nothing", "Done"),
            ("F-06", "nothing", "Ready"),
            ("F-07", "F-05, F-06", "Blocked"),
        ],
    )

    exit_code = css.main()
    out = capsys.readouterr().out
    # F-06 isn't Done yet, so F-07 must NOT be flagged as ready-to-unblock.
    assert exit_code == 0
    assert "F-07" not in out

    # Now finish F-06 too - only then should F-07 be flagged.
    _write_story(css, "F-06", "Done", "nothing")
    _write_index(
        css,
        [
            ("F-05", "nothing", "Done"),
            ("F-06", "nothing", "Done"),
            ("F-07", "F-05, F-06", "Blocked"),
        ],
    )

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
    rows, _ = css.parse_index()
    assert rows["F-01"]["index_state"] == "Done"  # didn't touch the wrong row
    assert rows["F-02"]["index_state"] == "Ready"
    # The State cell is no longer the last one; the Owner cell after it must survive the fix.
    assert rows["F-02"]["index_owner"] == "_unclaimed_"


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


# --- owner checks (D-037) --------------------------------------------------------------------


def _issues(css, capsys):
    exit_code = css.main()
    return exit_code, capsys.readouterr().out


def test_well_formed_owners_pass(css, capsys):
    """Every state with the right kind of owner, including two owners, is clean."""
    _write_story(css, "F-01", "Done", owner="@mbj1994; @TabeenRaoof")
    _write_story(css, "F-02", "In review", owner="@dalwalyk")
    _write_story(css, "F-03", "Claimed", owner="@TabeenRaoof")
    _write_story(css, "F-04", "Ready")
    _write_story(css, "F-05", "Blocked", "F-03")
    _write_index(
        css,
        [
            ("F-01", "nothing", "Done", "@mbj1994; @TabeenRaoof"),
            ("F-02", "nothing", "In review", "@dalwalyk"),
            ("F-03", "nothing", "Claimed", "@TabeenRaoof"),
            ("F-04", "nothing", "Ready", "_unclaimed_"),
            ("F-05", "F-03", "Blocked", "_unclaimed_"),
        ],
    )

    exit_code, out = _issues(css, capsys)

    assert exit_code == 0, out


def test_owner_in_file_but_not_in_index_is_reported(css, capsys):
    """The exact gap D-037 closes: the claim was written to the story file only."""
    _write_story(css, "F-01", "Claimed", owner="@TabeenRaoof")
    _write_index(css, [("F-01", "nothing", "Claimed", "_unclaimed_")])

    exit_code, out = _issues(css, capsys)

    assert exit_code == 1
    assert "index owner is '_unclaimed_'" in out
    assert "'@TabeenRaoof'" in out


def test_claimed_story_without_an_owner_is_reported(css, capsys):
    _write_story(css, "F-01", "Claimed", owner="_unclaimed_")
    _write_index(css, [("F-01", "nothing", "Claimed", "_unclaimed_")])

    exit_code, out = _issues(css, capsys)

    assert exit_code == 1
    assert "has no owner" in out


def test_ready_story_with_an_owner_is_reported(css, capsys):
    """An owner on a Ready story means someone claimed it but never set the state."""
    _write_story(css, "F-01", "Ready", owner="@dalwalyk")
    _write_index(css, [("F-01", "nothing", "Ready", "@dalwalyk")])

    exit_code, out = _issues(css, capsys)

    assert exit_code == 1
    assert "must be Claimed" in out


def test_owner_that_is_not_a_github_handle_is_reported(css, capsys):
    _write_story(css, "F-01", "Claimed", owner="Tabeen Raoof")
    _write_index(css, [("F-01", "nothing", "Claimed", "Tabeen Raoof")])

    exit_code, out = _issues(css, capsys)

    assert exit_code == 1
    assert "isn't an @github-handle" in out


@pytest.mark.parametrize("assistant", ["@Claude", "@codex", "@ChatGPT"])
def test_ai_assistant_is_never_an_owner(css, capsys, assistant):
    """An agent claims on behalf of a person; the person's handle goes on the story."""
    _write_story(css, "F-01", "Claimed", owner=assistant)
    _write_index(css, [("F-01", "nothing", "Claimed", assistant)])

    exit_code, out = _issues(css, capsys)

    assert exit_code == 1
    assert "is an AI assistant" in out


def test_row_the_parser_cannot_read_is_reported_not_skipped(css, capsys):
    """A row in the old five-column layout used to vanish from every check silently."""
    _write_story(css, "F-01", "Done")
    _write_story(css, "F-02", "Ready")
    _write_index(css, [("F-01", "nothing", "Done")])
    with css.INDEX.open("a", encoding="utf-8") as index:
        index.write("| [F-02](F-02.md) | a test story | 1 h | nothing | Ready |\n")

    exit_code, out = _issues(css, capsys)

    assert exit_code == 1
    assert "F-02: its row in stories/README.md doesn't match" in out


def test_fix_never_writes_an_owner(css, capsys, monkeypatch):
    """--fix reports owner problems but never resolves them: only a person claims."""
    _write_story(css, "F-01", "Claimed", owner="@TabeenRaoof")
    _write_index(css, [("F-01", "nothing", "Claimed", "_unclaimed_")])
    before_index = css.INDEX.read_text(encoding="utf-8")
    before_story = (css.STORIES_DIR / "F-01.md").read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["check_story_states.py", "--fix"])
    exit_code = css.main()

    assert exit_code == 1  # nothing auto-fixable, so the problem stands
    assert css.INDEX.read_text(encoding="utf-8") == before_index
    assert (css.STORIES_DIR / "F-01.md").read_text(encoding="utf-8") == before_story
