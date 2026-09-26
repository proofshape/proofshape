#!/usr/bin/env python3
"""Check (and optionally fix) story-state consistency in stories/.

Catches these failure modes, the first two of which have both happened already:

1. A story file's own `State:` line disagrees with its row in stories/README.md.
   (F-02 and F-03 said Ready in the index, Blocked in the file itself.)
2. A story is still marked Blocked even though every story it depends on is
   now Done — i.e. completing a story didn't cascade to unblock what waited
   on it. (F-01 went Done; F-02/F-03 needed a manual, separate pass.)
3. The owner is missing, wrong or inconsistent (D-037): the index's Owner cell
   must match the story file's `Owner:` line exactly; a Claimed, In review or
   Done story must name its owner(s) as @github-handles; a Blocked or Ready
   story must still read `_unclaimed_`; and an AI assistant's name is never an
   owner — an agent claims on behalf of the person it's working for.
4. A story row the parser can't read. Before D-037 such a row was silently
   skipped, so it dropped out of every check above without anyone noticing.

This does not decide anything. It never marks a story Done or Claimed, and it
never writes an owner — only a human does that, by actually taking or finishing
the work. All it will change is Blocked -> Ready, and only when every dependency
is verifiably Done.

Usage:
    python3 scripts/check_story_states.py          # report only, exit 1 if issues found
    python3 scripts/check_story_states.py --fix     # also apply the Blocked->Ready fixes

Run this after any merge that sets a story to Done - see AGENTS.md.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR = ROOT / "stories"
INDEX = STORIES_DIR / "README.md"

ROW_RE = re.compile(
    r"^\|\s*\[([A-Z]-\d\d)\]\([A-Z]-\d\d\.md\)\s*\|"  # | [ID](ID.md) |
    r"[^|\n]*\|"  # title
    r"\s*[\d.]+\s*h\s*\|"  # estimate
    r"\s*([^|\n]*?)\s*\|"  # depends-on
    r"\s*(\w[\w ]*?)\s*\|"  # state
    r"\s*([^|\n]*?)\s*\|\s*$",  # owner
    re.MULTILINE,
)
# Any table row that starts with a story link, whether or not ROW_RE can read the rest of it.
# Comparing the two is what stops a malformed row from silently dropping out of every check.
STORY_ROW_START_RE = re.compile(r"^\|\s*\[([A-Z]-\d\d)\]\(", re.MULTILINE)
FILE_STATE_RE = re.compile(r"\*\*State:\*\*\s*(\w[\w ]*?)\s*$", re.MULTILINE)
FILE_OWNER_RE = re.compile(r"\*\*Owner:\*\*\s*(.*?)\s*$", re.MULTILINE)

UNCLAIMED = "_unclaimed_"
OWNED_STATES = {"Claimed", "In review", "Done"}
UNOWNED_STATES = {"Blocked", "Ready"}
# One or more GitHub handles separated by "; " (F-06 has two owners). GitHub handles are
# letters, digits and single hyphens.
HANDLE_RE = re.compile(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
# An agent claims on behalf of a person; the owner is who answers for the work, never the tool
# (D-037). Compared case-insensitively against the handle without its "@".
AI_ASSISTANT_NAMES = {
    "claude",
    "chatgpt",
    "codex",
    "copilot",
    "gpt",
    "openai",
    "assistant",
}


def parse_index():
    text = INDEX.read_text(encoding="utf-8")
    rows = {}
    for sid, deps, state, owner in ROW_RE.findall(text):
        dep_ids = (
            [] if deps.strip().lower() == "nothing" else re.findall(r"[A-Z]-\d\d", deps)
        )
        rows[sid] = {
            "deps": dep_ids,
            "index_state": state.strip(),
            "index_owner": owner.strip(),
        }
    return rows, text


def unparsed_rows(index_text, rows):
    """Story IDs that have a row in the index which ROW_RE could not read."""
    unreadable = []
    for sid in STORY_ROW_START_RE.findall(index_text):
        if sid not in rows:
            unreadable.append(sid)
    return unreadable


def file_state(sid):
    path = STORIES_DIR / f"{sid}.md"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    m = FILE_STATE_RE.search(text)
    return (m.group(1).strip() if m else None), text


def file_owner(story_text):
    m = FILE_OWNER_RE.search(story_text)
    return m.group(1).strip() if m else None


def owner_problems(sid, state, owner):
    """Everything wrong with one story's owner, given its state. Empty list if it's fine."""
    problems = []
    if state in UNOWNED_STATES:
        if owner != UNCLAIMED:
            problems.append(
                f"{sid}: state is {state} but owner is '{owner}' — a story with an owner "
                f"must be Claimed (or later); otherwise set the owner back to {UNCLAIMED}"
            )
        return problems

    if state not in OWNED_STATES:
        return problems  # an unknown state is reported by the state checks, not here

    if not owner or owner == UNCLAIMED:
        problems.append(
            f"{sid}: state is {state} but it has no owner — put the claiming person's "
            "@github-handle in the story file and in its index row"
        )
        return problems

    for handle in owner.split(";"):
        handle = handle.strip()
        if not HANDLE_RE.match(handle):
            problems.append(
                f"{sid}: owner '{handle}' isn't an @github-handle (several owners are "
                "separated by '; ')"
            )
        elif handle[1:].lower() in AI_ASSISTANT_NAMES:
            problems.append(
                f"{sid}: owner '{handle}' is an AI assistant — the owner is the person the "
                "agent is working for, never the agent (D-037)"
            )
    return problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--fix", action="store_true", help="apply Blocked->Ready fixes to disk"
    )
    args = ap.parse_args()

    rows, index_text = parse_index()
    if not rows:
        print(
            "No story rows parsed from stories/README.md — check the table format.",
            file=sys.stderr,
        )
        return 2

    problems = []
    fixes = []  # (sid, old_state, new_state) for stories we will flip Blocked -> Ready

    for sid in unparsed_rows(index_text, rows):
        problems.append(
            f"{sid}: its row in stories/README.md doesn't match the expected "
            "| ID | Story | Est | Depends on | State | Owner | layout, so none of the other "
            "checks can see it"
        )

    for sid, info in rows.items():
        fstate, ftext = file_state(sid)
        if fstate is None:
            problems.append(f"{sid}: no story file, or no **State:** line found")
            continue
        if fstate != info["index_state"]:
            problems.append(
                f"{sid}: index says '{info['index_state']}', story file says '{fstate}' — these must match"
            )

        fowner = file_owner(ftext)
        if fowner is None:
            problems.append(f"{sid}: no **Owner:** line found in the story file")
        else:
            if fowner != info["index_owner"]:
                problems.append(
                    f"{sid}: index owner is '{info['index_owner']}', story file owner is "
                    f"'{fowner}' — these must match"
                )
            # Judge the owner against the file's state and owner; a mismatch with the index is
            # already reported above, and reporting it twice would only add noise.
            problems.extend(owner_problems(sid, fstate, fowner))

        # Cascade check: only meaningful if the story is currently Blocked.
        effective_state = fstate  # trust the story file as the source of truth
        if effective_state == "Blocked":
            dep_states = []
            for dep in info["deps"]:
                dstate, _ = file_state(dep)
                dep_states.append((dep, dstate))
            unresolved = [d for d, s in dep_states if s != "Done"]
            if info["deps"] and not unresolved:
                problems.append(
                    f"{sid}: marked Blocked but every dependency ({', '.join(info['deps'])}) is Done — should be Ready"
                )
                fixes.append(sid)

    if not problems:
        print(
            "All story states consistent; nothing is stuck Blocked with satisfied dependencies."
        )
        return 0

    print(f"{len(problems)} issue(s) found:\n")
    for p in problems:
        print(f"  - {p}")

    if args.fix and fixes:
        print(f"\nApplying {len(fixes)} Blocked -> Ready fix(es)...")
        new_index_text = index_text
        for sid in fixes:
            # story file
            path = STORIES_DIR / f"{sid}.md"
            text = path.read_text(encoding="utf-8")
            text = FILE_STATE_RE.sub("**State:** Ready", text, count=1)
            path.write_text(text, encoding="utf-8")
            # index row: replace only the State cell (the fifth) for this ID's row. It is no
            # longer the last cell, since the Owner column follows it (D-037).
            new_index_text = re.sub(
                rf"^(\|\s*\[{re.escape(sid)}\]\({re.escape(sid)}\.md\)\s*\|(?:[^|\n]*\|){{3}}\s*)"
                r"Blocked(\s*\|)",
                r"\1Ready\2",
                new_index_text,
                count=1,
                flags=re.MULTILINE,
            )
            print(f"  {sid}: Blocked -> Ready (file + index)")
        INDEX.write_text(new_index_text, encoding="utf-8")
        print("\nDone. Review the diff, then commit.")
        return 0

    if fixes and not args.fix:
        print(f"\n{len(fixes)} of these are auto-fixable: re-run with --fix.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
