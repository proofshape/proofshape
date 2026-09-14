# Start here — every session, before anything else

The order of operations, not a restatement of the rules. Each step names the file that actually
governs it; go read that file when you hit its step, don't just take this file's word for it.

## Before touching anything

1. **Read `AGENTS.md` in full.** Project description, standing rules, definition of done.
   Everything below assumes it.
2. **Check for a file named after your assistant** (`CLAUDE.md`, `CODEX.md`, ...). Read it if it
   exists — it only adds to `AGENTS.md`, never replaces it. None for yours yet? `AGENTS.md`
   alone governs.
3. **Read `stories/README.md`.** How stories work, the state table, the claim-to-merge workflow.
4. **Skim `docs/decisions.md`.** What's already settled — don't reopen it. Think one's wrong?
   Say so and stop; don't build past it either way.
5. **Check `docs/glossary.md`** for any term that isn't obviously plain English here.

## Before starting a specific story

6. **Read the story file in full**, then every story it depends on and every story that depends
   on it.
7. **Check for conflicts**: other Claimed/In-review stories, open branches touching the same
   files or endpoints.
8. Ambiguous acceptance criteria → resolve first. An ambiguous story is not Ready, whatever the
   index says. Ask, don't guess.
9. **Plan before building**, for anything non-trivial — surface real design forks before code.

## While working

10. **Claim it before writing code**: Owner + `State: Claimed`, committed and pushed immediately.
11. **Tests alongside the code, not after** — one test per piece of logic as you write it, same
    pull request. Never batched, never deferred to "a follow-up."

## Before opening the pull request

12. Run the test suite, `ruff check .`, `ruff format --check .`, and
    `python3 scripts/check_story_states.py`. All clean.
13. Tick every box in `.github/PULL_REQUEST_TEMPLATE.md` before requesting review.

## On merge — whoever merges it, author or reviewer, same action as the merge

14. `State: Done` in the story file **and** its row in `stories/README.md`.
15. Completion record filled with a **real, measured** hour count — never the estimate copied
    in. Genuinely don't know it yet? Leave `<n> h actual` visibly unfilled instead of guessing.
16. `python3 scripts/check_story_states.py --fix`, committed.
17. Append one entry to `docs/progress-log.md`.

If a step doesn't apply — nothing to plan, no conflicts found — say that explicitly. Silence
reads as "done," not "didn't apply."
