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

---

## 2026-09-21 — F-06 golden capture set (merged)

**Who:** @mbj1994
**What changed:** PR #22 merged the golden capture set: five capture sets over four rigid shapes
(one reflective metal padlock shot under two lighting conditions), all JPEGs kept outside git behind
`fixtures/download_golden_capture.sh`, a Drive-backed `golden_capture_manifest.json` with per-sample
content checksums, and the s-04 triangular-prism reference CAD committed in-repo under
`fixtures/reference_cad/`.
**Command / how to reproduce:** `bash fixtures/download_golden_capture.sh` (needs `gdown`); the
mocked-Drive tests run offline via `python -m pytest tests/test_download_golden_capture.py`.
**Result:** review over two rounds tightened the real gaps — the gitignore reference-CAD exception,
Drive link sharing (401 for non-owners), per-sample content hashing (a same-count byte swap had
passed silently), macOS `shasum` fallback, and offline downloader tests. Reshooting two objects for
sample quality pushed actual to 6 h against a 5 h estimate.
**Concluded:** the only source of real photographs for the entire backend push is locked and
verifiable; captured bytes never enter git.
**Next:** F-06 satisfied for R-01/R-02; reconstruction can consume the golden capture.

---

## 2026-09-24 — F-07 reproducible dev environment (merged)

**Who:** @dalwalyk (PR #24), @mbj1994 (PR #25)
**What changed:** `scripts/bootstrap_dev_env.sh` is the one command for the dev environment, with
`requirements.txt` pinning every dependency. PR #24 delivered the bootstrap plus the Apple Silicon
clean-checkout verification; PR #25 added detection of Lightning's managed `cloudspace` Conda
environment so the same command works on the shared Studio (which refuses a second `.venv`), and
removed a duplicate editable install.
**Command / how to reproduce:** `bash scripts/bootstrap_dev_env.sh`, then `python -m pytest -q`.
**Result:** verified both halves of the acceptance criterion — Apple Silicon from a clean checkout
(reviewer, @TabeenRaoof) and the shared Lightning Tesla T4 (31 pytest passed, `ruff` clean,
`bash scripts/start_gpu_workspace.sh` → GPU CHECK: PASS, PyTorch 2.8.0+cu128, one Tesla T4).
Actual 3 h against a 3 h estimate.
**Concluded:** one command reproduces the environment on both the laptop and the shared GPU Studio;
Phase 1 foundations (F-01 through F-07) are all Done.
**Next:** R-01 (run the reconstruction model on the golden capture) is now unblocked — F-05, F-06,
and F-07 are all Done.

---

## 2026-09-25 — R-02 board detection and per-frame pose (PR open)

**Who:** @TabeenRaoof (PR #29)
**What changed:** `recon/board_pose.py` detects the D-034 ChArUco board in each frame and solves
a camera-from-board pose in millimetres (SQPnP + LM), reporting every failed frame with a reason.
Intrinsics come from the EXIF 35 mm-equivalent focal until R-03. Corrected D-034: the calib.io
print is OpenCV's `setLegacyPattern(True)` layout, not `False`; with `False` no frame got a pose.
**Command / how to reproduce:** `python -m recon.board_pose <capture folder>`, on each golden
sample after verifying it against the manifest's count and `content_sha256`.
**Result:** 138/142 frames posed across s-01…s-05 (26/26, 29/30, 25/27, 29/30, 29/29), every one
with the part partly hiding the board. Median reprojection RMS was 1.39–2.86 px per sample with
EXIF-only K; that's a self-consistency figure, not an accuracy claim. The 4 failures were checked
by eye (1 blurred, 3 near-table-level grazing views). 3.5 h actual against a 5 h estimate.
**Concluded:** metric board poses exist for the whole golden capture; the board definition in
code is now tested against the committed print file.
**Next:** on merge, R-03 (intrinsics from the board) unblocks, and R-04 needs only R-01.

---

## 2026-09-26 — R-02 merged; close-out

**Who:** @mbj1994 (review, approval and merge of PR #29); @TabeenRaoof (close-out)
**What changed:** R-02 set to `Done` in its story file and the index, and
`check_story_states.py --fix` flipped R-03 `Blocked` → `Ready`. Both were missed at merge time,
the same slip D-025's PR-template checkbox exists to catch, so they're done here in a separate PR.
**Next:** R-03 claimed by @TabeenRaoof. R-04 still waits on R-01.

---

## 2026-09-26 — R-01 first real VGGT runs on the shared T4 (in review)

**Who:** @dalwalyk
**What changed:** `recon/vggt_runner.py` loads a golden-capture folder, runs VGGT-1B and writes
`poses.npz`, `depth.npz`, `points.ply` and `run.json` (per-stage wall-clock timings) per run.
`requirements-gpu.txt` pins VGGT to an upstream commit and leaves the Studio's PyTorch alone.
**Command / how to reproduce:** on the Lightning Studio, `pip install -r requirements-gpu.txt`,
`bash fixtures/download_golden_capture.sh`, then
`python -m recon.vggt_runner fixtures/data/golden_capture/s-04`.
**Result:** PASS on s-04 (yellow prism, 30 frames, 6,091,680 points), s-01 (padlock, 26 frames)
and s-03 (rounded object, 27 frames) on the Tesla T4 in fp16. Single measurements on s-04:
model load 22.14 s, preprocess 14.70 s, inference 16.11 s, total 60.92 s. The s-01 and s-03
figures are in `stories/R-01.md`. The s-04 cloud shows the prism on a single, flat board plane.
Poses and depth are in VGGT's arbitrary scale, not metric. Findings: the `facebook/VGGT-1B`
weights are CC-BY-NC-4.0 (non-commercial); VGGT ignores EXIF orientation, and feeding the raw
sensor buffers gave coherent clouds; Google Drive rate-limited the fixture download once.
**Concluded:** the reconstruction backbone runs end to end on the shared GPU and emits the
output layout that R-11/R-12 should match (D-009).
**Next:** review, plus a clean-checkout run by a second member. After merge, R-04 has both its
dependencies (R-01, R-02) Done, and R-11/R-12 unblock. The weights are not yet on Teamspace
Drive (D-036), because no Drive mount was visible from the Studio.

---

## 2026-09-26 — Claim ownership shown in the backlog index (D-037)

**Who:** @TabeenRaoof
**What changed:** `stories/README.md` gains an **Owner** column. A claim now writes the claimer's
`@github-handle` in both the story file and the index. If an AI assistant claims, it writes the
person's handle, never its own name. The instructions are in `stories/README.md`, `AGENTS.md`,
`AGENTS_START_HERE.md`, `README.md`, the story template, the PR template and the sprint plan.
`check_story_states.py` now reports owner mismatches, missing or stray owners, non-handle and
AI-assistant owners, and any index row it can't parse (such rows used to be skipped silently).
**Command / how to reproduce:** `python3 scripts/check_story_states.py`, and
`python -m pytest tests/test_check_story_states.py`.
**Result:** all 22 current rows filled from their story files and consistent; 15 checker tests
pass, and the owner-match and unparsed-row tests each fail if their check is removed.
**Next:** #31 (R-01) merged first; its index row was carried into the new layout while
resolving #32. The R-03 claim branch already uses the Owner cell. R-01's owner handle
(`@yashidalwala`; the GitHub account is `@dalwalyk`) needs confirming by its owner.

---

## 2026-09-28 — R-01 and R-03 closed out; R-04, R-11, R-12 unblocked

**Who:** @TabeenRaoof (close-out). R-01 was merged by @dalwalyk (#31), R-03 by @mbj1994 (#33).
**What changed:** R-01 and R-03 set to `Done` in their story files and the index, with
completion records naming the owner, date and PR. `check_story_states.py --fix` then flipped
R-04 (needs R-01, R-02), R-11 and R-12 (need R-01) from `Blocked` to `Ready`.
**Result:** both merges had skipped the merge-time state step, three merges in a row now
(R-02, R-01, R-03), so the index showed finished work as `In review` and kept R-04, the
bottleneck story, `Blocked`. The actual hours in both completion records are left as
`<n> h actual` for their owners to fill in; they weren't measured by the person closing out.
**Next:** R-04 is the chain story and is `Ready`. R-11 and R-12 are available in parallel.

---

## 2026-09-28 — Set `Done` before merge, not after (D-038)

**Who:** @TabeenRaoof, with the ordering fix from @dalwalyk's review
**What changed:** the merge-time state change is now required as a commit pushed to a story's
own pull request before it merges, not as a follow-up action taken at or after the merge click.
`AGENTS.md`'s definition of done, its refusal list, `AGENTS_START_HERE.md`, `stories/README.md`
step 6, `README.md`, and the PR template all restate the rule this way. Recorded as D-038.
Also tightened D-037's claim instructions: a claim must reach `main` as its own small pull
request right away, not sit on an unmerged branch, closing the gap that lost a claim earlier
this session.
**Command / how to reproduce:** `python3 scripts/check_story_states.py` and
`python -m pytest tests/test_check_story_states.py`.
**Result:** no code behaviour changes; instructions and one decision entry only. Existing 93
tests pass, `ruff` clean, story states consistent. The PR's first draft said to push the state
commit "once approved, then merge" — @dalwalyk caught in review that `main` dismisses a stale
approval on every new commit (D-013), so a commit pushed after approval always forces a second
approval round. Fixed before merge: the state commit now has to be pushed *before* the approval
the PR merges on, not after.
**Concluded:** the same failure — a story merged without its state flipped — had already
happened three times in a row (R-02, R-01, R-03), each needing its own separate close-out pull
request. The fix follows D-037's pattern: make the artifact (the PR) incomplete without the
step, rather than relying on someone remembering it at the moment they're about to merge. That
this rule's own first draft conflicted with an existing decision (D-013) before anyone had even
approved it is itself a small instance of the same lesson.
**Next:** watch whether this actually holds on R-04's PR, the next one to close.

---

## 2026-09-28 — R-11 MASt3R GPU feasibility and adapter implementation started

**Who:** @mbj1994
**What changed:** claimed R-11 on main, pinned MASt3R upstream commit
`f5209afc300cec36239a7ac992263f36847bbba0`, verified it on the shared Lightning T4, then
added the MASt3R adapter and the single D-009 backend selector. VGGT remains the default.
**Measured result:** two-frame s-04 MASt3R inference passed in 1.96 s with 3.12 GiB peak
allocated GPU memory. Four-frame sparse global alignment passed in 13.72 s with 3.26 GiB peak,
returning four poses, four intrinsics matrices and finite dense depth/confidence at 384×512.
These are smoke tests only, not the story's required same-golden-capture wall-clock result.
**Compatibility finding:** the tested s-04 frame had EXIF orientation 1. The R-11 loader now
ignores EXIF orientation deliberately so other golden captures stay compatible with R-01/R-02's
stored-pixel convention. Camera-to-world poses are inverted to the R-01 camera-from-world
convention, and MASt3R uses the same four output filenames as VGGT.
**Next:** run the repository tests/Ruff on Lightning, then run the completed MASt3R path on all
30 s-04 frames and record its real wall-clock result before moving the story to review.


---

## 2026-09-28 — R-11 full s-04 MASt3R run and downstream compatibility check

**Who:** @mbj1994
**Command / how to reproduce:** run `recon.reconstruction_runner` on the 30-frame s-04 golden
capture with the MASt3R backend, writing to `fixtures/data/recon_runs/s-04-mast3r`, then pass
that output unchanged into `recon.metric_alignment` with the existing s-04 board poses.
**Measured result:** MASt3R PASS, 30 frames, 5,898,240 points, **249.89 s total wall-clock** on
the shared Lightning GPU. R-04 then consumed the MASt3R output without downstream code changes,
paired 29 frames, and PASSed with **12.167 mm RMS camera-centre alignment residual**.
The residual is a reconstruction/board-pose alignment diagnostic, not an object-accuracy or
tolerance result.
**Concluded:** R-11 has now demonstrated its required same-golden-capture wall-clock and the
R-01 output contract is usable by an existing downstream stage. Repository tests/Ruff and the
normal Definition-of-Done checks still need to be clean before R-11 moves to review.

## 2026-09-29 — R-04 closed by splitting its known-size tolerance judgment into R-16

**Who:** @yashidalwala (PR #39), reviewed by @TabeenRaoof.
**What changed:** R-04's known-size acceptance criterion originally asked for a measurement
*and* a stated tolerance. The measurement is now recorded — the s-04 specimen's B-C edge
reconstructs at 60.27 mm against @TabeenRaoof's caliper mean of 60.011 mm (PR #35), a 0.26 mm
(~0.4%) difference — but no tolerance exists anywhere in the project to judge that difference
against; it can only come from the ground-truth calibration study (`G-01` through `G-05`), which
hasn't run yet. A first attempt to close the gap anyway with a derived, provisional tolerance
(D-039, 3x the alignment RMS) was proposed, reviewed, and reverted in the same PR before merge —
@TabeenRaoof caught that it violated the "no promised numbers" rule for the same reason D-032
already declined to set I-01's tolerance-table fields early. Rather than leave R-04 open
indefinitely on a criterion it cannot control, its acceptance criteria were reworded to match
what it actually delivers (transform, board-frame mm output, per-frame residuals, and the raw
known-size comparison), and the tolerance judgment was split into a new story, **R-16**, which
carries the full measurement inline and stays `Blocked` on `R-04, G-03, G-04` until the real
study produces a number to apply. R-04 is now `Done` (9 h actual, est 6 h); `R-05` and `R-06`
cascaded from `Blocked` to `Ready`.
**Also in this PR:** removed an unverifiable "0.50 degrees" orientation-difference claim from
R-04's notes (flagged by @TabeenRaoof on PR #35 — no code in the repo computes that figure).
**Result:** 98 tests pass, `ruff` clean, `check_story_states.py` consistent. `docs/decisions.md`
has no D-039 entry — it was reverted in the same PR that proposed it, before any merge, so the
"a decision number doesn't carry its caveats forward" risk never materialized on `main`.
**Next:** R-05 and R-06 are `Ready` and unclaimed. R-16 stays `Blocked` until `G-01`–`G-05` exist
and produce a real tolerance; those aren't written up yet — `stories/README.md` deliberately
keeps Phase 3 titles-only until the sprint boundary, same reasoning as D-032.

## 2026-09-29 — C-01 minimal upload page (PR #52)

**Who:** @TabeenRaoof (PR #52).
**What changed:** the thin capture thread's first story — a plain-HTML page in `capture/` that
calls `createSession`, `uploadFrame` and `finishSession` against the frozen contract
(`contracts/openapi.yaml`, F-03), proving the wire format end to end before anything builds the
real app shell (C-04) around an assumption about it. Backed by a three-route local stub
(`capture/stub/contract-stub.ts`) that loads its canned bodies from `contracts/examples/*.json`
at startup so it can't drift from the contract — explicitly not the fuller S1 "mock service"
`docs/sprint-plan.md` describes.
**Also in this PR:** the first npm project outside `scripts/`, so it settles `capture/`'s tooling
— Vite + vanilla TypeScript + Vitest, local to `capture/`, Node 24 (**D-040**) — and moves
`.github/workflows/typescript-ci.yml` from a Node-20 root-`package.json` placeholder (which never
actually ran a test) to `capture/package.json` with `npm test` wired in, since Vitest 5 requires
Node ≥22.12. Also records how `gyro` actually travels inside the multipart `uploadFrame` request
(**D-041**), since the contract's own JSON example can't represent a multipart body's parts.
**Result:** 29 capture tests pass, `tsc`/`eslint`/`prettier` clean; repo-wide `pytest` (115),
`ruff` and `check_story_states.py` unaffected. Verified by mutation (sent `gyro` as `{x, y, z}`,
confirmed both `api.test.ts` and `page.test.ts` failed loudly, reverted) and by hand against the
real `npm run dev` server with `curl`, not just the test suite. C-01 is now `Done` (3 h actual,
est 3 h); `C-02` and `C-03` cascaded from `Blocked` to `Ready`.
**Next:** C-02 and C-03 are `Ready` and unclaimed. The thin capture thread is still behind
schedule (three sprints' allocated hours passed before this story started) — worth someone
picking up C-02 or C-03 soon rather than letting it slip further.

## 2026-09-30 — R-06 closed by splitting visual verification into R-17

**Who:** @dalwalyk (PR #54), reviewed by @mbj1994.
**What changed:** `recon/part_segmentation.py` keeps only geometry above the board plane and
inside the printed ChArUco pattern's footprint. Two things in the acceptance criteria needed
resolving against other files before any filtering logic could be written, both recorded in
`stories/R-06.md`'s notes: "sheet footprint" reads as the printed pattern's rectangle (160x120 mm,
`board_pose.py`'s `BOARD_SQUARES_X/Y * BOARD_SQUARE_MM`) rather than the physical, undetected
A4/Letter paper edge — `docs/glossary.md` defines "reference sheet" as "the ChArUco board" for
tier 1, and no code anywhere detects the actual paper outline; and "above the board plane" means
negative z, per `board_pose.py`'s own z-into-the-table convention, pinned by an explicit unit test
rather than left to a comment given this project's history with exactly this kind of sign mistake
(D-034's legacy-pattern flag). `read_ply` was added to `vggt_runner.py` (symmetric to the existing
`write_ply`; no PLY reader existed anywhere before this), and `apply_similarity_transform` was
extracted out of `metric_alignment.py`'s `AlignmentResult.transform_points` for reuse, with no
behaviour change. The story's third criterion — visual confirmation on all five golden
objects — needed real R-01/R-04 output that the implementing session's machine had no GPU access
to produce or check, so rather than leave R-06 open indefinitely on a criterion it couldn't
control, it was split into a new story, **R-17**, the same move R-04 made for its known-size
tolerance judgment (split to R-16). R-06 is now `Done` (4 h actual, est 4 h); `R-07` cascaded from
`Blocked` to `Ready`.
**Result:** 135 tests pass, `ruff` clean, `check_story_states.py` consistent.
**Next:** R-07 (TSDF fusion, the next chain story) and R-17 (the split-out visual check, needs
GPU Studio access) are both `Ready` and unclaimed.

## 2026-09-30 — R-07 closed by splitting real-golden-object mesh production into R-18

**Who:** @dalwalyk (PR #57), reviewed by @mbj1994.
**What changed:** `recon/tsdf_fusion.py` fuses R-01's per-frame depth into a metric TSDF volume
in the board frame (voxel-projection based: project every voxel into every frame's camera,
compare against that frame's measured depth, truncate and accumulate a running weighted average)
and extracts a mesh via marching cubes (`scikit-image`, primary) or Poisson reconstruction
(Open3D, lazily imported fallback). The load-bearing piece, worked out before writing any fusion
code: TSDF fusion needs each frame's camera pose *in the board frame*, which R-04 never produced
(it only gives points there). `recon/metric_alignment.py` gained
`apply_similarity_transform_to_extrinsic` to derive it — `R_board = R_o @ rotation.T`,
`t_board = scale * t_o - R_board @ translation`, with depth pre-scaled by `scale` — checked not
just by inspection but by composition: a cross-check test confirms unprojecting scaled depth
through the derived board extrinsic lands on exactly the same points `apply_similarity_transform`
gives directly. Masking reuses R-06's `segment_part` directly on each frame's unprojected
board-frame points, so board/table/hand pixels never contribute, and `extract_mesh_marching_cubes`
masks never-observed voxels out of `skimage.measure.marching_cubes` so unobserved space can never
fabricate a surface — the hard rule's "unobserved geometry never asserts anything," applied at the
meshing level. `scikit-image` was added to `requirements.txt`/CI (CPU-only, needed by every run);
Open3D was deliberately kept out of it, lazy-imported only inside the Poisson path, the same
pattern R-01 uses for torch/VGGT. Same gap as R-06→R-17: "produces a mesh for every golden object"
needed real R-01/R-04 output this session's machine had no GPU access to produce, so it was split
into a new story, **R-18** (checked for ID collisions across every branch first, same as R-17
was). R-07 is now `Done` (3 h actual, est 6 h); `R-08`, `R-09` and `R-13` all cascaded from
`Blocked` to `Ready`.
**Result:** 165 tests pass (1 skipped — the Open3D Poisson success-path test, since Open3D isn't
installed in CI; the not-installed error-message path is tested instead), `ruff` clean,
`check_story_states.py` consistent.
**Next:** R-08, R-09 and R-13 are all `Ready` and unclaimed — three chain/off-chain stories now
open at once. R-18 (the split-out real-data mesh check, needs GPU Studio access) is also `Ready`.

## 2026-09-30 — R-08 mesh cleanup and GLB export (PR #60)

**Who:** @dalwalyk (PR #60), reviewed by @mbj1994.
**What changed:** `recon/mesh_cleanup.py` takes R-07's raw fused mesh through: keep the largest
connected component (`scipy.sparse.csgraph` on a face-adjacency graph, dropping disconnected TSDF
noise), fill small boundary holes, decimate to a target triangle count, export a GLB. Hole-filling
is hand-rolled: boundary loops are found by walking edges used by exactly one face in the direction
induced by that face's own winding, so a 3-edge loop closes with one triangle directly and a longer
loop gets a correctly-wound centroid fan — loops longer than a configurable threshold are left open
on purpose, on the same "don't assert unobserved geometry" principle as the hard rule, even though
this story runs before R-09's per-vertex provenance exists to label anything. Unlike R-06/R-07, none
of this story's criteria need real captured photos, so it closed end-to-end with synthetic-mesh
tests in one session — no follow-up story.
**Dependency decision:** `fast-simplification` (real quadric decimation) and `pygltflib` (real GLB
read/write) are both core to this story's primary criteria — not a fallback like R-07's Open3D — so
both are real `requirements.txt`/CI dependencies. `scipy` pinned explicitly now that it's imported
directly. "Loads in a browser" is checked by `pygltflib` round-trip (a genuinely malformed GLB won't
parse back either), noted honestly as not a literal browser session rather than claimed as fully
verified — cheap enough to spot-check yourself that it didn't need its own story. No measured
numbers exist for target triangle count, "small holes," or a file-size limit; the defaults
(50,000 triangles, 15 MiB, 8-edge holes) are documented, configurable assumptions, same as
R-06/R-07's precedent. R-08 is now `Done` (3 h actual, est 4 h); `R-14` cascaded from `Blocked` to
`Ready`.
**Result:** 182 tests pass (1 skipped, unrelated — R-07's Open3D Poisson path), `ruff` clean,
`check_story_states.py` consistent.
**Next:** R-14 (Reconstruction endpoint, the last chain story before container deploy) is `Ready`
and unclaimed, alongside R-09, R-13, R-17 and R-18 from earlier entries.

## 2026-09-30 — R-13 per-stage latency table: mechanism done, not yet closed

**Who:** @TabeenRaoof.
**What changed:** `recon/timing.py` (new) renders any runner's `timings` dict as one table —
stage, seconds, share of total — and is now the single thing `vggt_runner.main`,
`mast3r_runner.main` and the unified `reconstruction_runner.main` all call, replacing three
separate copies of the same print loop. The unified CLI was actually the one place dropping the
per-stage detail down to just `total_s`, even though each backend's own entry point already had
it. `find_capture_frames` is now its own timed stage (`find_frames_s`) in both `vggt_runner` and
`mast3r_runner` — the one stage that was running untimed in the gap before `total_s`.
**Not marking R-13 `Done`:** its third acceptance criterion needs all three backbones recorded
on the same capture, and `colmap_runner.py` doesn't exist on `main` yet (R-12 is still open);
even once it lands, that measurement needs the shared Lightning GPU, not available this
session. `State` stays `Claimed`. See `stories/R-13.md`'s Notes for the full reasoning,
including why "backbones" is read as scoped to the three reconstruction methods and not R-07's
fusion stage, and a caught mistake: an early test asserted stage order on the *on-disk* `run.json`,
which is always written with `sort_keys=True` and therefore always alphabetical regardless of
when each stage ran — fixed to check the in-memory dict instead, which is what the table-printing
code path actually reads.
**Result:** 191 tests pass (1 skipped, unrelated), `ruff` clean, `check_story_states.py`
consistent.
**Next:** whoever picks up R-12 should expect a small, normal merge collision in
`reconstruction_runner.py` (both touch the same dispatcher). Once R-12 merges and someone with
GPU access runs VGGT, MASt3R and COLMAP on `s-04` with this code, record the real table in
`stories/R-13.md` and close it out.

## 2026-09-30 — R-09 per-vertex provenance (PR #65)

**Who:** @dalwalyk (PR #65), reviewed by @TabeenRaoof.
**What changed:** `recon/provenance.py` labels every mesh vertex Observed or Unobserved — the
mechanism behind the hard rule ("unobserved geometry can never pass or fail a part"). A frame only
counts as observing a vertex if the vertex also projects within a depth-consistency tolerance of
that frame's *measured* depth at the landed pixel (the occlusion check, reusing R-07's `fuse_tsdf`
logic via a newly-shared helper, `metric_alignment.project_points_to_camera`) — otherwise the back
face of any non-convex part would be miscounted as seen by a frame that's actually looking through
the object at its front. Observed requires two or more such observing frames whose widest pair's
triangulation angle clears a threshold; a vertex seen twice by nearly-coincident cameras stays
Unobserved, pinned by its own test. Two things needed resolving against existing docs before
writing any labelling logic: `docs/glossary.md`'s "Observed" definition matches this story's own
acceptance criteria exactly (view count + triangulation angle only) and deliberately excludes the
older course proposal's "σ below the tier's floor" clause — that's R-10's job, which depends on
R-09, not the reverse, so the story file's own criteria won over the older proposal doc. Vertex-
color GLB encoding (white = Observed, no color asserted yet; grey = Unobserved) follows
`contracts/openapi.yaml`'s "labels travel as vertex colors" spec and the proposal's own "drawn
gray" convention. `mesh_cleanup.write_glb`/`read_glb` gained an optional vertex-colors parameter;
`write_glb` stayed fully backward compatible, though `read_glb`'s return arity did change (2-tuple
to unconditional 3-tuple) — both existing call sites were updated, and the PR description was
corrected after review to state that precisely rather than call it a no-op. `provenance_run`
accepts either R-07's raw `mesh.ply` or R-08's cleaned `mesh_clean.glb` (R-09 depends only on R-07;
R-08 is a sibling, not a prerequisite), and persists the raw per-observation table to
`observations.npz` specifically for R-10 to reuse without recomputing projections. Nothing in this
story's criteria needs real captured photos, so it closed end-to-end with synthetic-mesh tests in
one session — no follow-up story, same as R-08. R-09 is now `Done` (3 h actual, est 5 h); `R-10`
cascaded from `Blocked` to `Ready`.
**Review notes:** two real fixes requested and made — the completion record's PR number was still
a placeholder, and both dated entries used the UTC calendar day instead of the local (Pacific) day
the work was actually done (the same mistake as PR #62's R-08 close-out, now the second occurrence
of this exact class of error).
**Result:** 225 tests pass (1 skipped, unrelated), `ruff` clean, `check_story_states.py`
consistent.
**Next:** R-10 (per-vertex uncertainty, builds directly on `observations.npz`) is `Ready` and
unclaimed, alongside R-13/R-14/R-17/R-18 from earlier entries.

## 2026-09-30 — C-02 camera permissions: mechanism and tests done, device check still open

**Who:** @TabeenRaoof.
**What changed:** `capture/src/camera.ts` (`mountCameraCheck`, `describeCameraError`,
`isGetUserMediaSupported`) adds the camera-permission check to C-01's existing page, as its own
`#camera-app` section rather than touching `page.ts`. Added `@vitejs/plugin-basic-ssl` and a new
`npm run dev:device` script (`HTTPS=true vite --host`) since Vite has had no `--https` CLI flag
since v3 — gated on an env var so C-01's existing plain-HTTP `npm run dev` is untouched. Verified
both modes actually serve (`curl -k` against `localhost` and the printed LAN IP for `dev:device`).
**Not marking C-02 `Done`:** two of its three acceptance criteria are checkable only by a human
tapping the button on a real iPhone over real HTTPS — no device, no network path to one, in this
session. `State` stays `Claimed`; `stories/C-02.md`'s Tests section has a manual-check template
ready to fill in once that happens.
**Verified by mutation, not just written:** the synchronous-`getUserMedia`-call requirement (the
hardest of the three criteria to get right, and the one most prone to silently regressing) has an
automated regression test — temporarily made the click handler `async` with a leading `await
Promise.resolve()`, confirmed it broke two tests for exactly that reason, reverted.
**Result:** 38 capture tests pass (29 from C-01, 9 new); `tsc`/`eslint`/`prettier` clean;
repo-wide `pytest` (225), `ruff` and `check_story_states.py` unaffected.
**Next:** hand-off to run `npm run dev:device` on a real iPhone and record the manual check —
once that's in, C-02 can close and C-03 (already `Ready`, depends on C-01 not C-02) can proceed
independently in the meantime.

## 2026-10-01 — C-02 closed: manual check caught two real bugs no automated test could

**Who:** @TabeenRaoof, manual check performed on a real iPhone 16 (iOS 26.6, Safari 26.6).
**What changed:** the hand-off above led directly to two real fixes, both invisible to the full,
green 38-test suite because happy-dom doesn't render video at all:
- **Grant path:** the permission-granted indicator appeared and `getUserMedia` resolved, but the
  preview stayed blank — `<video>.srcObject` doesn't start playback on its own. Added `autoplay`
  and an explicit `.play()` call, plus a test (a spy on `.play()`) verified by mutation to
  actually fail if the call is removed.
- **Deny path:** the message had the right text but rendered in plain body-text color — present,
  but not actually noticeable, which doesn't meet this story's own "visible" wording. Fixed with
  bold/red error styling (a targeted fix for this one criterion, not the general UI polish still
  deferred to C-04), with tests locking in that error states are styled and the success state
  isn't.
**Result:** both the grant and deny paths re-verified working on the real device after each fix.
38 capture tests still pass (extended with new assertions, not new test cases), `ruff`/
`check_story_states.py` unaffected. C-02 is now `Done` (2 h actual, est 2 h).
**Concluded:** this is exactly the case `AGENTS.md`'s manual-check exception exists for — the
automated suite correctly covered everything it was capable of covering (including a real
regression guard for the hardest criterion, the synchronous `getUserMedia` call), and the manual
check caught the two things that genuinely needed a real device and a real browser.
**Next:** C-03 (gyro permission flow) is `Ready` and unclaimed, independent of C-02.

## 2026-10-01 — C-03 closed: manual check found nothing to fix, by design not luck

**Who:** @dalwalyk, manual check performed on a real iPhone 18 Pro Max (iOS 27.0, Safari).
**What changed:** `capture/src/gyro.ts` gained `isOrientationPermissionRequestNeeded`,
`describeOrientationError`, and `mountGyroCheck`, extending C-01's existing `parseGyroInput`
rather than replacing it. Built the same way @TabeenRaoof built C-02 (PR #72): everything that
doesn't need a real device was built and tested first, with the on-device grant/deny check left
as a written manual-check template; `State` stayed `Claimed`, not `Done`, until the real check
ran. Built directly on review lessons from PR #72, not just read about them — that review
surfaced two real bugs and a reuse problem in C-02, and all three had a direct analog here,
closed from the start rather than repeated: the missing-`autoplay`/`.play()` bug's analog
(requesting permission but never attaching the `deviceorientation` listener) is closed by a test
that dispatches a synthetic orientation event and asserts the DOM actually updates, not just that
status says "granted"; the double-click stream-leak analog is closed by disabling the button
synchronously for the duration of a pending `requestPermission()` call and guarding the listener
against being attached more than once; the `requireElement`/`requireButton` triplication finding
is closed by a new shared `capture/src/dom.ts` instead of a third copy.
**Manual check result:** both grant and deny worked on the first attempt — no fixes needed. Unlike
C-02's own device check, which caught two real bugs invisible to automated tests, this one found
nothing, and that's attributed to the two bug classes C-02's check caught having already been
identified and closed during this story's own build (see above), not to luck. One real platform
behaviour recorded for whoever does the next manual check on this app: once granted, iOS Safari
remembers the permission per site and won't re-prompt on reload — getting back to the deny prompt
required clearing that site's data in Settings → Safari → Advanced → Website Data first, not just
reloading.
**Review note:** the completion record initially shipped with a placeholder PR number (`PR #nn`)
even though `State: Done` was already set — @mbj1994 caught it, requesting the real PR number per
AGENTS.md's carve-out, and it was fixed in a follow-up commit before approval.
**Result:** 58 capture tests pass, `typecheck`/`lint`/`format:check` clean, repo-wide `pytest`/
`ruff`/`check_story_states.py` unaffected (a `capture/`-only change). C-03 is now `Done` (5 h
actual, est 2 h).
**Next:** R-17, R-18 and R-19 (the split-out golden-set/real-data criteria from R-06, R-07 and
R-10) are `Ready` and unclaimed.


## 2026-10-01 — R-14 reconstruction endpoint merged (PR #77)

**Who:** @mbj1994, reviewed and approved by @dalwalyk.
**What changed:** added the real FastAPI reconstruction service for the frozen F-03 `/v1`
contract: session creation, multipart frame + D-041 gyro upload, finish handoff, reconstruction
polling, structured contract errors, and GLB download. The service runs the existing R-02 board
pose → selected reconstruction backend → R-04 metric alignment → R-07 fusion → R-08 cleanup
chain, then applies R-09 provenance and R-10 uncertainty before serving the final labelled GLB.
Session state is file-backed under `PROOFSHAPE_DATA_DIR`; orphaned pending jobs are failed
explicitly after a service restart rather than leaving clients polling forever. Upload/finish race
conditions now return the documented `session_already_finished` envelopes instead of 500s, and
unexpected pipeline exceptions are converted into a terminal failed reconstruction.
**Contract note:** F-03's frozen `ErrorCode` enum has no dedicated malformed-request code. R-14
keeps malformed gyro uploads contract-valid for v1 using an allowed code with a message that
explicitly says it is not a VLM decision; a dedicated request-validation code requires the D-031
sprint-boundary contract process.
**Verification:** final CI passed Python pytest, Ruff lint/format and TypeScript checks. A real
Lightning smoke run on `fixtures/data/golden_capture/s-04` with VGGT returned
`status: complete`, `metric: true`, `reference_tier_used: charuco_board`, and a downloadable
996,488-byte GLB through `getReconstruction`. The measured stage timings for that run were
14.657 s board detection, 68.094 s reconstruction, 0.014 s metric alignment, 3.488 s fusion and
0.860 s export.
**Result:** R-14 is `Done` at 4 h actual (est 5 h); R-15 is now `Ready`.
**Next:** R-15 can be claimed for the Lightning container/deployment story.
