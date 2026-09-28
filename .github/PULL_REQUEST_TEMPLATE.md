## What this does

<!-- One or two sentences. -->

## Story

<!-- Link the story file, e.g. stories/F-02.md -->

## Definition of done

- [ ] Story's `Owner:` names the person(s) who did the work, as `@github-handle`s, identically in
      the story file and its `stories/README.md` row (D-037)
- [ ] Reviewed and approved by one other team member
- [ ] Runs on a second member's machine from a clean checkout
- [ ] Has one automated test or one written manual check
- [ ] Completion record filled in on the story file (who, date, PR #, actual hours)
- [ ] **Story `State:` set to `Done` in this PR — a commit pushed to this branch, not a follow-up
      — and the matching row in `stories/README.md` updated to `Done` in the same commit.**
      `python3 scripts/check_story_states.py --fix` run and committed too.
      **This has slipped on three consecutive merges already (R-02, R-01, R-03); see D-038.**
      If an AI assistant is finishing this PR, it pushes this commit itself before saying the PR
      is ready to merge — do not merge, or approve merging, while this box is unchecked.

See `AGENTS.md` → *Definition of done* for the full rule.
