#!/usr/bin/env python3
"""Check (and optionally fix) story-state consistency in stories/.

Catches two failure modes that have both happened already on this project:

1. A story file's own `State:` line disagrees with its row in stories/README.md.
   (F-02 and F-03 said Ready in the index, Blocked in the file itself.)
2. A story is still marked Blocked even though every story it depends on is
   now Done — i.e. completing a story didn't cascade to unblock what waited
   on it. (F-01 went Done; F-02/F-03 needed a manual, separate pass.)

This does not decide anything. It never marks a story Done or Claimed — only
a human does that, by actually finishing the work. All it will change is
Blocked -> Ready, and only when every dependency is verifiably Done.

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
    r"^\|\s*\[([A-Z]-\d\d)\]\([A-Z]-\d\d\.md\)\s*\|"   # | [ID](ID.md) |
    r".*?\|"                                            # title
    r"\s*[\d.]+\s*h\s*\|"                               # estimate
    r"\s*(.*?)\s*\|"                                    # depends-on
    r"\s*(\w[\w ]*?)\s*\|\s*$",                         # state
    re.MULTILINE,
)
FILE_STATE_RE = re.compile(r"\*\*State:\*\*\s*(\w[\w ]*?)\s*$", re.MULTILINE)


def parse_index():
    text = INDEX.read_text(encoding="utf-8")
    rows = {}
    for sid, deps, state in ROW_RE.findall(text):
        dep_ids = [] if deps.strip().lower() == "nothing" else re.findall(r"[A-Z]-\d\d", deps)
        rows[sid] = {"deps": dep_ids, "index_state": state.strip()}
    return rows, text


def file_state(sid):
    path = STORIES_DIR / f"{sid}.md"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    m = FILE_STATE_RE.search(text)
    return (m.group(1).strip() if m else None), text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="apply Blocked->Ready fixes to disk")
    args = ap.parse_args()

    rows, index_text = parse_index()
    if not rows:
        print("No story rows parsed from stories/README.md — check the table format.", file=sys.stderr)
        return 2

    problems = []
    fixes = []  # (sid, old_state, new_state) for stories we will flip Blocked -> Ready

    for sid, info in rows.items():
        fstate, ftext = file_state(sid)
        if fstate is None:
            problems.append(f"{sid}: no story file, or no **State:** line found")
            continue
        if fstate != info["index_state"]:
            problems.append(
                f"{sid}: index says '{info['index_state']}', story file says '{fstate}' — these must match"
            )

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
        print("All story states consistent; nothing is stuck Blocked with satisfied dependencies.")
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
            # index row: replace only the trailing state cell for this ID's row
            new_index_text = re.sub(
                rf"(\[{re.escape(sid)}\]\({re.escape(sid)}\.md\).*\|\s*)Blocked(\s*\|\s*)$",
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
