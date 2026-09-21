# Progress log

Running chronological journal of what has been decided, built and shipped. Complements — does
not replace — the per-story files in `stories/`, which stay the source of truth for a story's
own detail, and `sprint-plan.md`, which is the schedule.

**This file is the history.** One entry per story completed, pull request reviewed, or
measurement run. Newest at the bottom. Never delete an old entry — if something turned out to
be wrong, append a correction rather than editing the original, because the mistake is usually
more instructive than the fix.

**Why it earns its keep here:** the entries since the last class presentation *are* the "what we
did" slide. Keep this current and the biweekly review takes minutes to prepare instead of an
evening.

---

## Entry template

```
## YYYY-MM-DD — <ID> <short title> (<built | reviewed | merged | measured>)

**Who:** @member
**What changed:** one or two sentences.
**Command / how to reproduce:** the exact command, if there was one.
**Result:** numbers if there are numbers; "works" is not a result.
**Concluded:** what we now believe that we did not before.
**Next:** what this unblocks, or what it forces us to decide.
```

Keep entries short. Three or four lines is a good entry. If it needs more, most of it probably
belongs in the story file instead.

---

## 2026-09-05 — project scaffolding created (built)

**Who:** all three
**What changed:** `AGENTS.md`, `CLAUDE.md`, `docs/decisions.md`, `docs/glossary.md`,
`docs/story-template.md` and this log. The story backlog was split from one file into
`stories/<ID>.md` with `stories/README.md` kept as the index.
**Concluded:** twenty-two stories are written out for foundations and the reconstruction engine,
totalling 93 hours, of which about 42 sit on a serial chain that no amount of parallelism
shortens.
**Next:** F-01, F-05 and F-06 are Ready and have no dependencies — one each, starting day one.

## 2026-09-05 — proposal presented; repository restructured (shipped)

**Who:** all three
**What changed:** the proposal and its deck were presented to the class. Afterwards the folder
was reorganised: coursework deliverables into `submissions/` dated by hand-in, working documents
into `docs/`, the story index into `stories/README.md`, plus a `README.md` and `.gitignore`.
**Concluded:** the presentation happened on 5 September, three days earlier than the M0 date the
plan assumed. Whether the M0 spike ran before it is not recorded here — someone should note it.
**Next:** initialise the git repository and push, which is story F-01.

## 2026-09-05 — F-01: repository created, protected, and moved to an org (built)

**Who:** @TabeenRaoof
**What changed:** repo created public, MIT licence, eight branch protection rules applied, then
transferred from the personal account to the new `proofshape` organisation.
**Result:** both gates proven, not just configured — a merge without approval was refused
("base branch policy prohibits the merge") and a direct push to `main` was rejected (GH006).
All eight rules verified intact after the transfer.
**Concluded:** `enforce_admins` means nobody can bypass, including the owner, so with one
collaborator **no pull request can merge at all**. Correct behaviour, but it makes adding the
other two members the hard blocker on all further work.
**Next:** add collaborators, then PR #1 can be approved. F-02 and F-03 become Ready on merge.

## 2026-09-11 — F-02: repository skeleton and module boundaries (built)

**Who:** @dalwalyk
**What changed:** `/recon`, `/capture`, `/inspect`, `/contracts`, `/fixtures`, `/scripts` created,
each with a `README.md`. Root `pyproject.toml` registers `recon` as an editable package;
`inspect` deliberately left unregistered (D-030 — it collides with the standard library
`inspect` module). Root README gained a "Running each part" section.
**Command / how to reproduce:** `pip install -e .` from a clean venv.
**Result:** installs cleanly; `import recon` works; `import inspect` still resolves to the
standard library, confirmed in a scratch venv.
**Concluded:** the `inspect/` directory name (fixed by D-020) and a top-level Python import of
the same name can't both exist safely — packaging has to route around it, not the directory.
**Next:** F-04 (CI skeleton) and F-07 (reproducible dev environment) become Ready on merge.

## 2026-09-14 — F-02 marked Done; found a merged-but-corrupted story file (build)

**Who:** @TabeenRaoof
**What changed:** PR #6 merged, but `stories/F-02.md` was left at `State: In review` — the third
time this exact gap has hit us — and worse, the file was actually **corrupted**: it had two
`State:`/`Owner:` blocks stacked back to back (`In review`/`@dalwalyk` immediately followed by
`Ready`/`_unclaimed_`, the pre-merge content from `main`). Fixed: de-duplicated the header, set
`State: Done`, filled the completion record (PR #6, merged 2026-09-14, reviewed by
@TabeenRaoof — actual hours left blank, real data). Flipped `stories/README.md` and `F-02` to
`Done`, and `F-04` (its only remaining dependency) to `Ready`. `F-07` stays `Blocked` — it also
needs F-05, which is only `Ready`, not `Done`.
**Result:** verified by reading `git show <merge-commit>^1:stories/F-02.md` and
`^2:stories/F-02.md` directly — both parents made a single-line edit to the same `State:` and
`Owner:` lines, and the merge silently kept both instead of conflicting.
**Concluded:** GitHub's `mergeable: MERGEABLE` only means no conflict *markers* were left — it
does not mean the merged content is *correct*. A clean two-parent line-merge can still silently
duplicate a small, adjacent block instead of flagging it as a conflict, especially when both
sides touch only a line or two. We checked `mergeable` before every approval in this session and
treated it as sufficient; it isn't. This is a new failure mode, distinct from the state-drift
bug `scripts/check_story_states.py` (still unmerged, on PR #5) already guards against — that
script would have caught the *un-updated* state, but not this *duplicated* one, since the file's
first `State:` line was still technically correct.
**Next:** worth a follow-up: `check_story_states.py`'s regex only reads the *first* `**State:**`
match in a file. If a duplicate block like this recurs, the script would silently read the first
(possibly stale) one and miss the corruption entirely. Consider having it also flag a file with
more than one `**State:**` line, once PR #5 merges.

## 2026-09-13 — F-03: interface contract v1, frozen (built)

**Who:** @TabeenRaoof
**What changed:** `contracts/openapi.yaml` (five operations: createSession, uploadFrame,
finishSession, getReconstruction, getVerdict), ten example request/response files under
`contracts/examples/`, `tests/test_contract.py` (six tests, written alongside each schema per
D-029), and a new `CONTRIBUTING.md` recording the contract's change rule. Tier 1 is specified in
full; the verdict result is an envelope only, with tolerance-table internals marked
`x-proofshape-extends-at` for the not-yet-written inspection stories. D-031 reconciles F-03's
"all three approvals" with the sprint plan's "two approvals" (GitHub can't count the author's
own approval, so they're the same rule); D-032 records what v1 freezes and what it deliberately
leaves open.
**Command / how to reproduce:** `pip install -e ".[dev]"` then `pytest tests/ -v`.
**Result:** 13 tests pass (7 pre-existing plus 6 new). Confirmed each new test actually catches
its own defect, not just passes: mutated an unobserved region to `contributes_to_verdict: true`
(caught — D-005 hard rule), added a fifth verdict value (caught), deleted a required example
(caught — this also surfaced a real gap, four documented error responses had no example, fixed
by adding them plus a missing `no_cad_uploaded` error code the getVerdict 404 case needed and
didn't have), and renamed an operationId without renaming its examples (caught — orphan check).
All four mutations reverted after confirming.
**Concluded:** writing the "every operation has an example" test before filling in every example
found a real spec gap (missing error examples, a missing error code) that reading the acceptance
criteria alone did not surface — the same pattern as F-02's review-caught issues, just caught
before review this time.
**Next:** R-14 (reconstruction endpoint) and the S1 mock service can now build against a real
contract. Flagged separately, not fixed here: PR #11 (F-04)'s CI runs only ruff, no pytest job,
so the existing test suites don't gate any merge yet, and `required_status_checks` on `main` is
still empty so no CI result can block a merge — both are F-04's own acceptance criteria to meet.
Also flagged: an untracked, unrelated `capture/test.py` (a LeetCode exercise) is sitting in the
capture lane and should be removed before F-04's ruff job starts linting it.

## 2026-09-14 — F-04 closed out; required status checks wired up (built)

**Who:** @TabeenRaoof
**What changed:** #11 and #12 (F-04's Python and TypeScript CI lanes) merged this morning, but
neither PR set `State: Done`, updated the index, or filled the completion record — the exact
gap `AGENTS.md`'s merge-time DoD steps and `AGENTS_START_HERE.md` exist to catch, caught this
time by the mechanical check (`check_story_states.py`) rather than by either PR itself.
Corrected: `stories/F-04.md` → `State: Done`, `Owner: @dalwalyk` (was "Yashi" — the same
naming-convention nit flagged in review, fixed here since nobody else had), completion record
citing both PR numbers. Index row → `Done`. Separately, registered `ruff` and `typescript` as
required status checks on `main`'s branch protection — an admin-only setting that had been
sitting open since F-01, and the one piece of F-04's acceptance criteria (a lint failure
actually blocks a merge) that no amount of code in either PR could satisfy on its own.
**Command / how to reproduce:** `gh api .../branches/main/protection/required_status_checks
--method PATCH` with `contexts: ["ruff", "typescript"]`; `python3
scripts/check_story_states.py --fix` for the story bookkeeping.
**Result:** `required_status_checks.contexts` now lists both checks (confirmed by reading the
protection settings back, not just assuming the PATCH took). `check_story_states.py` reports
all states consistent afterward.
**Concluded:** actual hours for F-04 are unknown — neither PR recorded them — so the completion
record's hour field stays `<n>` rather than guessed, per D-033.
**Next:** nothing downstream depends on F-04, so this wasn't blocking anyone; it was purely
bookkeeping hygiene.

## 2026-09-15 — F-03 closed out; PR #16's D-031 claim corrected (built)

**Who:** @TabeenRaoof
**What changed:** PR #16 (@mbj1994) proposed keeping F-03 `In review` on the theory that D-031
required a second explicit approval before it could close, since @dalwalyk's review on PR #13
was recorded as `COMMENTED` rather than `APPROVED`. Checked the actual text: both `CONTRIBUTING.md`
and D-031 scope the two-approval rule to changes made **after v1 is frozen** — PR #13 is what
froze v1, so the rule never applied to PR #13's own merge, which needed and got the standard
single approval (from `@mbj1994`, who also merged it). The Codex review bot flagged the same
misreading independently, citing the same two sources. Requested changes on #16 rather than
approving it — it also turned out to be stale, opened before PR #15 merged, so its F-04 content
would have reverted that already-landed fix. Separately, correct the one real, valid point PR #16
surfaced: F-03 genuinely was still stuck `In review` despite a fully valid merge, because nobody
ran the merge-time DoD steps — the same gap class PR #15 fixed for F-04, just for a different
story. Also filled the Notes section, which @dalwalyk's own review on #13 correctly flagged as
still the unfilled template.
**Result:** `stories/F-03.md` → `State: Done`, completion record filled (PR #13, 4.5 h actual
against a 3 h estimate — the honest number I flagged at the time, not the estimate). Index
row → `Done`. `check_story_states.py` reports consistent afterward; nothing further unblocks
since R-14 also needs R-08, still `Blocked`.
**Concluded:** the class of bug PR #15 first caught (a merged story never closed out) isn't
unique to F-04 — `check_story_states.py` can't detect it at all, since it only flags file/index
*disagreement*, and a story stuck `In review` with both file and index agreeing looks
consistent to that script. Worth a genuine follow-up: teach the script to also flag any story
whose state is `In review` or `Claimed` while its own linked PR shows `MERGED`, rather than
relying on someone noticing by hand a second time.
**Next:** none — F-03 has no dependents besides R-14, which also needs R-08.

## 2026-09-17 — progress-review deck built for the 18 Sept review (built)

**Who:** @TabeenRaoof
**What changed:** built `submissions/2026-09-18-progress-review-deck.pptx` — 15 slides covering
how the project was planned (stories, the GitHub gate, the decisions log), what has shipped,
and the change of plan from three per-person lanes to all three of us on foundations and then
the reconstruction engine. Includes a re-planned Gantt reflecting that shape, and six
screenshots taken during the work — a story file, the definition-of-done checklist on a live
PR, a reviewer blocking a merge for missing tests, the approval gate, the GPU smoke test
passing, and a golden-capture frame. Assets live in `submissions/assets/2026-09-18/`, with a
narrowly scoped `.gitignore` exception so they can be committed without ever whitelisting
capture photographs. The generator is
committed at `scripts/build_review_deck.js` so the next six reviews are assembly rather than
invention, which is the only way the 18-person-hour review budget in D-015 survives.
**Command / how to reproduce:** `cd scripts && npm install && node build_review_deck.js
../submissions/2026-09-18-progress-review-deck.pptx`
**Result:** every figure in the deck was read from the repository rather than recalled — 4 of 7
foundation stories Done, 19 pull requests (15 merged, 2 closed, 2 open), 32 decisions logged,
13 tests green, 2 required CI checks, 79 golden-capture frames across 3 sessions. The deck
states plainly that there is no runnable pipeline yet and that M1 (25 Sept) will slip to early
October, because the critical-chain arithmetic in `stories/README.md` has said so since it was
written.
**Concluded:** the lane change is real and documentable, not a retrofit. The archived
pre-proposal draft (`docs/archive/team-proposal-v2.md`) has a "Lanes (to argue about)" section
with one owner per lane, and the submitted proposal still carries a Lane/Owner table with 97
hours budgeted per person. D-019 and D-020 replaced that with backend-first and
lanes-are-places on 4 September; what we have actually *done* since confirms it — all three of
us have worked only in foundations, and nobody has touched `capture/` or `inspect/`.
**Next:** see the marker below.

---

## ► NEXT DECK: covers 18 Sept → next review (~2 Oct 2026)

**Read this before building the next progress deck.** The 18 September deck covers everything up
to and including 17 September. **The next one starts from 18 September** — do not re-present
foundations, the GitHub process, or the lane change; the professor has seen all three. Those
slides exist to be cut, not repeated.

What the next deck needs to answer, using the standing four-slide template:

1. **What we said we would do** — this is already on record, from slide 13 of the 18 Sept deck:
   F-05, F-06, F-07, then R-02, R-01, R-04. The commitment made out loud was *"the golden capture
   going into the reconstruction model and a metric point cloud coming out — a running pipeline,
   not slides."* Hold the next deck to exactly that sentence.
2. **What works now** — a demo or a recording. If R-01 runs, show it running. Per the sprint plan's
   own rule, from R2 onward keep a screen recording of the working path in case the pod or the
   network fails in the room.
3. **What slipped** — M1 (25 Sept) was already flagged as slipping to early October *before* the
   gate. Report what actually happened against that, honestly, including real actual-hours
   numbers now that D-033 requires them.
4. **What we will have by the review after** — the next chain stories, R-06 onward.

Also carry forward, still open as of this entry:
- **O-003 — the review cadence is still unconfirmed.** The plan assumed every second Friday from
  11 September (R1 11 Sept, R2 25 Sept), but the real review is 18 September, which matches
  neither. Re-anchor the review dates in `docs/sprint-plan.md` once the actual schedule is known,
  rather than leaving the Gantt carrying dates nobody has verified.
- **Actual hours are missing on F-01 and F-04.** If anyone can still remember them, fill them in;
  if not, they stay blank, which is what D-033 requires.

---

## 2026-09-21 — F-05 shared GPU workspace (merged)

**Who:** @mbj1994
**What changed:** PR #21 merged the shared Lightning AI GPU workspace, GPU smoke-test tooling,
D-036, and downstream wording updates for F-07/R-15. Story close-out now records the measured
4 h actual and marks F-05 Done.
**Command / how to reproduce:** `bash scripts/start_gpu_workspace.sh` in the shared Lightning
Studio; full local verification on the reviewed branch was `pytest`, `ruff check .`,
`ruff format --check .`, and `python3 scripts/check_story_states.py`.
**Result:** all three members verified the shared T4 workspace; reviewer reported 20 pytest tests
passing on the final reviewed head. F-07 is now Ready because F-02 and F-05 are both Done.
**Concluded:** the remote GPU foundation is complete; Lightning is the primary shared GPU path,
with RunPod retained as fallback per D-036.
**Next:** claim F-07 and finish the reproducible development environment; F-06 is Ready and unclaimed (its own PR is separate and still under review).
