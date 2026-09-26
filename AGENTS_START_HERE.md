# Start here — every session, before anything else

The order of operations, not a restatement of the rules. Each step names the file that actually
governs it — go read that file, don't take this file's word for it. Duplicating rule content
here is exactly the drift this project keeps getting burned by (see D-025); pointers only.

## Before touching anything

1. Read `AGENTS.md` in full.
2. Check for a file named after your assistant (`CLAUDE.md`, `CODEX.md`, ...) and read it if it
   exists — it only adds to `AGENTS.md`, never replaces it.
3. Read `stories/README.md`.
4. Skim `docs/decisions.md` — don't reopen it. Think one's wrong? Say so and stop, don't build
   past it.
5. Check `docs/glossary.md` for any unfamiliar term.

## Before starting a specific story

6–10. Follow **"Before starting any story"** in `AGENTS.md`: read the story and its dependencies,
check `docs/decisions.md` for this area, check for conflicts, resolve any ambiguity, plan before
building anything non-trivial, and **claim it before writing any code** (its step 5, with the
mechanics in step 2 of `stories/README.md`). **Claiming means the person's `@github-handle` goes
on the story file *and* its row in the backlog index, not the assistant's name** (D-037). Check
it with `python3 scripts/check_story_states.py`.

## While working

11–12. Follow **"While building"** in `AGENTS.md`: tests alongside the logic as you write it —
never batched, never deferred.

## Before opening the pull request

13. Run the test suite, `ruff check .`, `ruff format --check .`, and
    `python3 scripts/check_story_states.py`. All clean.
14. Tick every box in `.github/PULL_REQUEST_TEMPLATE.md`.

## On merge — whoever merges it, same action as the merge

15–18. Follow the **Definition of Done**'s merge-time steps in `AGENTS.md`: `State: Done` in the
story file and the index, `check_story_states.py --fix` committed, an entry in
`docs/progress-log.md` — **and the completion record's hours must be real and measured, never
the estimate copied in** (this exact gap is why this file exists).

If a step doesn't apply, say so explicitly. Silence reads as "done," not "didn't apply." Missing
something this file should have caught? Flag it and add it — this file is expected to grow.
