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

## 2026-09-17 — D-033 found silently dropped from main; restored; ChArUco board decided (built)

**Who:** @TabeenRaoof
**What changed:** while running the routine D-028 cross-branch decision-number check before
adding a new decision, the *numeric* maximum on `main` came back as D-032 — one lower than
D-033, a decision known to have merged via PR #14 on 2026-09-14. Traced it commit by commit:
D-033 was present in `17db023` (PR #14's own merge commit), then absent starting at `493164b`
("Merge branch 'main' into foundations/F-03-interface-contract-v1"). That merge combined two
branch tips that had each independently appended different content — D-033 on one side, D-031/
D-032 on the other — at the identical insertion point in `docs/decisions.md` (immediately before
the "Open" section). No line was edited by both sides, so git found no conflict to flag, and the
merge silently kept only one side's addition. `mergeable: MERGEABLE` on PR #13 (which carried
this merge forward) never reflected the loss.
**Command / how to reproduce:** `git show <commit>:docs/decisions.md | grep -c D-033`, walked
across every commit on `main`'s first-parent-plus-merges history in order, to find the exact
commit where the count went from 1 to 0.
**Result:** restored D-033 verbatim (recovered from `33bf3dc`, its last known-good state) back
into `docs/decisions.md`, with a note on the entry itself explaining what happened and when it
was restored. Also recorded the new finding as D-035: a two-parent merge can silently drop one
side's entire independent addition to an append-only file, with zero conflict markers, when
both sides insert different new content at the same location rather than editing the same
lines — a variant of the F-02 corruption class (2026-09-14 entry above), not a duplicate of it.
**Concluded:** D-033 was not merely hidden by file order — it was genuinely absent from every
branch's actual content except the one already-merged source branch, so file-order `tail -1`
would have reported the same (wrong) answer as a true numeric max here; this incident's cause
was the merge, not the check command. But double-checking with the numeric max surfaced a real,
separate latent bug in the check `AGENTS.md` documents: `tail -1` on `grep -oE '^\*\*D-[0-9]+'`
takes the *last matching line in file order*, not the numeric maximum, and this file's entries
are demonstrably not in strict numeric order already (D-027/D-030/D-025/D-028 sit out of
sequence further up). A future branch could have a genuinely higher number sitting earlier in
the file than its own last-in-order entry, and the documented check would silently miss it.
**Next:** `AGENTS.md`'s documented D-028 check command should be corrected to sort numerically
rather than rely on file order — filed as a real follow-up, not fixed in this PR since it's a
distinct, separately-reviewable change to the instructions file itself.

## 2026-09-17 — ChArUco board parameters decided (D-034)

**Who:** @TabeenRaoof
**What changed:** F-06 (claimed by @mbj1994) and R-02/R-03 all need to agree on the same board
parameters, and nothing had fixed them yet. Decided: `DICT_5X5_250`, 8×6 squares at 20 mm,
15 mm markers, generated via `cv2.aruco` and printed at 100% scale. Recorded as D-034, with the
reasoning tied directly to this project's own scale-integrity design (the sheet outline as one
of three scale checks) rather than picked from general best practice alone.
**Next:** whoever builds F-06 (mbj1994, pending a check-in with @TabeenRaoof about possibly
claiming it) prints against these exact numbers; R-02/R-03 implement against the same dictionary
and geometry.
