# Decisions log

Settled decisions and the reasoning behind them. **Do not relitigate these.** If one needs
revisiting, raise it with the team and stop there — do not build past it in either direction.

A decision without its reasoning gets reversed by whoever forgets it, so every entry says *why*.
Superseded entries are marked, never deleted.

---

## Product and scope

**D-001 · Three delivery tiers; Tier 1 is never cut.** (2026-09-02)
Tier 1 is photos in, metric 3D model out. Tier 2 is CAD alignment, heatmap and verdict. Tier 3
is stretch.
*Why:* a capstone must produce something that works. Tiering guarantees a working product at
every milestone rather than a half-finished everything at the end.

**D-002 · Batch capture is the guaranteed path.** (2026-09-02)
Capture all frames, reconstruct once. Interactive reconstruction is built only if the M2 latency
table shows it is nearly free.
*Why:* the interactive loop is the most latency-sensitive part of the system and the most likely
to be impossible on available hardware. Batch always works.

**D-003 · No metrology claims.** (2026-09-02)
Coarse geometric inspection, several millimetres on hand-sized parts. Complements CMM
inspection, never replaces it.
*Why:* the method cannot support tighter claims, and a capstone that overclaims loses more
credibility than one that scopes honestly.

**D-004 · No CAD, no verdict.** (2026-09-03)
Tier 1 runs without a CAD file and returns a measured model. Alignment, heatmap and verdict all
require one.
*Why:* inspection is a comparison. Without the design there is nothing to compare against.

---

## Correctness rules

**D-005 · The hard rule: unobserved geometry never passes or fails a part.** (2026-09-02)
Surface the camera did not properly see may only trigger a Rescan. It cannot contribute to a
Pass and cannot cause a Fail.
*Why:* the model's guess must never approve or condemn a component. Without this the report
means nothing.

**D-006 · Uncertainty comes from reconstruction statistics, not generative spread.** (2026-09-02)
σ per vertex is the spread of depth residuals across the frames that observe it. The spread
across generative completion samples is a *guidance* signal only.
*Why:* completion spread measures the completion model's behaviour, not measurement error, and
it collapses wherever the surface was actually seen. Using it as the verdict denominator would
reduce observability scoring to coverage under another name.

**D-007 · Scale has three independent sources that must agree.** (2026-09-02)
The printed board pitch, an optional typed caliper reading, and the CAD registration residual.
Any two disagreeing by more than about 2% flags the session.
*Why:* a home printer can scale a page silently. The typed caliper value is untrusted input and
must never be the only source.

**D-008 · Registration must detect orientation ambiguity.** (2026-09-02)
Robust ICP with a Tukey loss, and if the runner-up alignment residual is within 10% of the best,
the session is marked ambiguous.
*Why:* on a near-symmetric part the feature that breaks the symmetry is often the defect. A
least-squares fit can silently choose the alignment that explains it away.

---

## Technology

**D-009 · VGGT primary, MASt3R then COLMAP behind one config flag.** (2026-09-02)
Chosen for real at M0 by running all three on the golden capture.
*Why:* VGGT is fast enough for guidance and works on glossy featureless parts where COLMAP has
nothing to match. Its failure mode is quiet, so the fallbacks stay one flag away.

**D-010 · Development on the M2 Max MacBook; rented GPU as fallback.** (2026-09-03)
Measured: Metal works, 26.8 GB working set, the whole non-neural stack runs natively.
*Why:* removes pod re-provisioning from daily work and cuts the infrastructure budget. Compute
is the constraint, not memory.

**D-011 · Force fp16 on Apple Silicon.** (2026-09-03)
Measured: bf16 runs at 5.4 TFLOP/s against fp16 at 10.9 — half speed on Metal, the opposite of
a modern NVIDIA card.
*Why:* reconstruction models commonly select bf16 automatically when they detect a capable GPU.
That default is wrong here and must be overridden.

**D-012 · Run natively for local development; containers are for the pod.** (2026-09-03)
*Why:* Docker Desktop on macOS gives containers no Metal access, so the "one image, one command"
story does not survive on a laptop.

---

## Process

**D-013 · Public GitHub repository.** (2026-09-03)
Branch protection on `main`: pull request required, one approval, stale approvals dismissed,
force pushes blocked, bypass disabled. Deploy only from `main`.
*Why:* branch protection on a private repo needs GitHub Pro, and the deploy-time approval gate
needs Enterprise. Public makes all of it free. It also avoids the trap where a team Organization
on the free tier cannot protect private branches even if one member holds Pro.

**D-014 · Deployment is gated by construction, not by a second approval.** (2026-09-03)
*Why:* `main` is unreachable except through an approved pull request, so deploying only from
`main` is simpler and harder to circumvent than an extra gate.

**D-015 · Biweekly progress reviews, seven of them.** (2026-09-03)
Assumed every second Friday from 11 September, using a fixed four-slide template.
*Why:* graded deliverables are planned work, not something absorbed on the day. The fixed
template is what keeps the cost at roughly 18 person-hours.

**D-016 · Capacity raised to 9 hours per person per week, and every added hour goes to AI.** (2026-09-03)
*Why:* adding capacity without spending it on generative AI *lowers* the share, because the
denominator grows faster than the numerator. Spending only the agent's 30 hours across 39 new
hours would have produced 30.4%, worse than building it inside the old budget.

**D-017 · Generative-AI components are never cut; the cut list was inverted.** (2026-09-03)
*Why:* the original ordering dropped the report narrative and frame checks first, which were the
only generative components. Any schedule pressure would have cut precisely what the course grades.

**D-018 · The reconstruction backbone is a learned model, not a generative one.** (2026-09-03)
It is counted separately in the AI share and described that way everywhere.
*Why:* it is deterministic — same photographs, same output. Calling it generative is an argument
we would lose, and losing it in the defence costs more than the hours it claims.

**D-019 · Backend first.** (2026-09-04)
The reconstruction engine is built before the other layers. Exit criterion: the golden capture
goes in and a metric segmented mesh with per-vertex provenance comes out, reproducibly, with
per-stage timings recorded.
*Why:* it is the hardest risk in the project. Retiring it first gives certainty for everything
built on top.

**D-020 · Lanes are places, not people.** (2026-09-04)
`/recon`, `/capture`, `/inspect` are directories. Any member picks up any story in any of them.
*Why:* it is what lets all three push on the engine at once.

**D-021 · One person per story; either of the other two reviews.** (2026-09-04)
A second person joining to unblock someone stuck is expected and gets noted. Two people quietly
working the same story is not.
*Why:* shared backlogs fail when claiming is invisible.

**D-022 · Next-view scoring belongs to the capture lane.** (2026-09-03)
*Why:* it is a guidance feature, capture already owns guidance, and it runs on the processor so
no GPU work crosses the boundary. It also levelled a 20-hour imbalance between lanes.

**D-023 · Per-story files, with the index keeping one row each.** (2026-09-05)
*Why:* stories accumulate implementation notes, decisions and review discussion. One long file
becomes unnavigable and generates merge conflicts on every edit.

**D-024 · The repository lives in the `proofshape` organisation, not a personal account.** (2026-09-05)
`proofshape/proofshape`, public, free org plan.
*Why:* a project-specific home rather than one member's profile, so ownership is shared and the
repository does not depend on a single account. This reverses the earlier recommendation in
D-013, which argued against an organisation — that argument was specific to *private*
repositories, where a free org cannot protect branches. Going public removed the constraint, so
the only remaining cost was one setup step. Free orgs do get protected branches on public repos,
and all eight settings survived the transfer.

**D-027 · Unblocking a dependent story is a mechanical check, not a role.** (2026-09-11)
`scripts/check_story_states.py` reads the dependency graph in `stories/README.md`, flags any
story whose file and index disagree, and (`--fix`) flips `Blocked` → `Ready` once every
dependency is `Done`. Run after every merge that sets a story to `Done`, by whoever does the
merge — not reserved to the approver.
*Why:* the question this answers was "should the approver be the one who unblocks dependent
stories." Making it one specific person's job is the same single-point-of-memory failure that
already caused two bugs (F-01 left `In review` after merging; F-02/F-03 unblocked in the index
but not in their own files). The fix isn't a better-chosen owner, it's removing the step from
memory entirely — a deterministic script gives the same answer regardless of who runs it or
whether they remember to. The script never marks anything `Done` or `Claimed`; only a human
decides work is actually finished.
*Note on numbering:* `D-025` is independently claimed on both `chore/proposal-revisions-2026-09-05`
(module layout) and `chore/pr-template-dod-checklist` (PR template), and `D-026` on the former
(tests ship with code). This entry uses the first number free across *every* open branch at the
time of writing. **When any of these branches merge, reconcile all of D-025 through D-027 in one
pass** rather than resolving them one at a time as each branch lands.
**D-030 · `inspect/` is not an installed Python package; `recon/` is.** (2026-09-11)
The repository-skeleton `pyproject.toml` registers only `recon` as a top-level package.
`inspect/` stays a plain directory with no `__init__.py` until inspection-lane code exists.
*Why:* `inspect` is the name of a Python standard-library module. Installing our `inspect/`
directory as a top-level package of the same name would shadow that module for the whole
environment — anything doing `import inspect`, including tooling like pytest, would silently
get our (empty) package instead. The directory name is fixed by D-020 and the story's
acceptance criteria, so the fix is on the packaging side: whoever starts I-01 picks a real
import name for that lane's code (a prefix, a namespace package, or something else) once there
is actual code to hang it off, rather than guessing now.
*Note on numbering:* originally written as `D-025`, which collided with independent decisions on
`chore/pr-template-dod-checklist` and `chore/proposal-revisions-2026-09-05`. Renumbered to
`D-030` per the check `D-028` describes — as of this update the highest claimed number on any
open branch was `D-029` (`chore/tests-during-implementation-rule`). A branch opened after this
check still won't see it; **reconcile again if D-030 turns out to collide too.**
**D-025 · The definition-of-done checklist ships as the PR template.** (2026-09-11)
`.github/PULL_REQUEST_TEMPLATE.md` mirrors the definition-of-done list in `AGENTS.md`, with the
`State: Done` / index-row item called out specifically.
*Why:* F-01 merged twice without its story state or index row updated — once on the initial
merge, once again after the fix PR itself merged without anyone re-checking it. A rule stated
only in `AGENTS.md` is read once, at onboarding, and forgotten under a real merge. Putting it in
front of every PR as a checkbox costs nothing and is the cheapest thing that could plausibly
catch the specific failure that already happened twice. Deliberately *not* duplicated into
`CLAUDE.md` — that split exists precisely so there is one source of truth; copying the rule into
a second file is the same failure mode as the one being fixed here.
*Note:* `D-025` and `D-026` are also claimed on the unmerged `chore/proposal-revisions-2026-09-05`
branch for unrelated decisions (module layout; tests ship with code). This entry was numbered
from what is actually on `main`. **Renumber one side when that branch merges** — do not let two
different `D-025` entries coexist.

**D-028 · Check every open branch's `docs/decisions.md` before claiming the next number.** (2026-09-12)
Documented in `AGENTS.md` under *Working with the team*, with the one-liner to run.
*Why:* three unrelated decisions independently claimed `D-025`, and one more claimed `D-026`,
each written against only its own branch tip — see D-027 for the first fix and the collision it
was already responding to. This entry's own number was picked using the check it describes: as
of writing, six other branches were at D-024, D-024, D-025, D-025, D-026 and D-027, so D-028 was
the first free everywhere. Doesn't prevent every future collision on its own — a branch opened
after this check still won't see it — but it turns "guess and hope" into "check and know," which
is most of the fix.
*Amended (2026-09-14):* the original one-liner had two bugs, caught in review by dalwalyk —
`git branch -r` lists `origin/HEAD -> origin/main` as a single line, which a naive `sed`/`grep -v
HEAD` mangles into malformed output; and it only checked already-fetched local remote-tracking
branches, silently missing a branch nobody had fetched yet. Fixed in `AGENTS.md` with a
`git fetch --prune` first and a filter on the long refname rather than the short one, since
`refs/remotes/origin/HEAD`'s *short* name is literally `origin`, not `origin/HEAD` — a subtlety
that would have broken a short-name filter too. Verified by reproducing both failures before
landing the fix.

**D-029 · Tests are written during implementation, and it's a joint obligation — engineer and assistant. (2026-09-12)**
Brought forward onto `main` now: unit tests for a story's logic ship in the same pull request as
that logic, written as it's written, not batched before or after and never deferred to a
follow-up. Reflected in `AGENTS.md` (*While building*, the definition of done, and the refusal
list) and in `docs/story-template.md`, which now asks for tests explicitly per story.
*Why:* `main` only had the softer form of this — "one automated test or one written manual
check," phrased as an either-or with no enforcement of *when*. Tests written after code exists
are written to pass, not to find defects; a story that defers tests to a follow-up rarely gets
one. The genuinely new part is naming this a **joint** obligation: an AI assistant doesn't get to
skip tests because a turn didn't ask for them, and the engineer directing it doesn't get to skip
them because the assistant didn't offer. Neither party gets to point at the other afterwards.
*Note:* this substance already exists as `D-026` on the unmerged `chore/proposal-revisions-2026-09-05`
branch (added 2026-09-06, before this project's decision-numbering collisions were even
understood as a pattern). That entry and this one cover the same ground; when that branch
merges, keep one of the two — this one additionally has the joint-accountability framing and the
refusal-list/story-template wiring, so folding D-026's wording into this entry (or vice versa)
is preferable to keeping both. **Also note:** this branch was created before `D-028` (the
cross-branch numbering check this entry's own reasoning refers to) had merged, so `D-028` won't
literally appear in this file until that PR lands too — the reference above is to the practice,
not a broken link.

**D-031 · Changing the frozen contract needs both other members' approval, at a sprint boundary.** (2026-09-13)
`CONTRIBUTING.md` states the rule for changing `contracts/openapi.yaml` after v1: approval from
both other team members — which, with the author, is all three signed off — merged only at a
sprint boundary. This is a written rule, not a branch-protection setting: `main` still requires
one approval repo-wide, left alone rather than raised project-wide to cover one directory.
*Why:* F-03's third acceptance criterion said the contract needs "all three approvals," but
GitHub cannot count the PR author's own approval — the author is barred from approving their own
PR — so on a three-person team the mechanical ceiling is two. `docs/sprint-plan.md` §7 already
says "two approvals" for exactly this reason. This entry reconciles the two: they were never in
tension, "all three agree" and "two reviewer approvals" describe the same outcome once GitHub's
approval-counting limit is accounted for. Recorded so nobody re-opens this as a contradiction
between the story and the sprint plan.
*Checked against D-028 at write time:* D-031 and D-032 were free on every open branch, `main`
included, as of 2026-09-13 — thirteen branches checked, highest claimed anywhere was D-030.

**D-032 · Contract v1 freezes Tier 1 in full; the verdict is an envelope only.** (2026-09-13)
`contracts/openapi.yaml` v1 specifies session creation, frame upload, session finish and the
reconstruction result completely — real stories (R-02, R-04, R-08, R-09, R-10, R-13) already
define those fields. The verdict result freezes only the envelope the existing decisions already
settle: the four verdict values (D-005's hard rule needs exactly these), per-region provenance,
and reason codes. Tolerance-table internals, `k` and `σ_floor` are marked
`x-proofshape-extends-at` in the schema rather than invented.
*Why:* Phase 3 (inspection) is deliberately titles-only in `stories/README.md` — "we elaborate
one sprint ahead... because most of these would be wrong if written in September." Freezing
tolerance-table shapes now would mean designing I-01 through I-10's data model before any of
those stories exist, and would violate the "no promised numbers" rule by giving `k` and
`σ_floor` a schema presence before the October calibration study measures them. The API is
versioned by the `/v1` path prefix specifically so a v2 can be introduced alongside this one,
once inspection is designed, rather than requiring this frozen file to be reopened.
*What this leaves as future work:* whoever picks up I-01 onwards defines the tolerance-table
schema and requests the D-031 approval to add it — this decision does not pre-approve any
particular shape for it.

**D-033 · `AGENTS_START_HERE.md` is the entry point; actual-hours must be measured, never copied from the estimate.** (2026-09-14)
Added `AGENTS_START_HERE.md` at the repo root as the literal first file to open, every session —
a numbered order of operations that points at the file governing each step rather than
restating any of them. `CLAUDE.md` (auto-loaded at the start of every Claude Code session in
this repo) now sends the reader there before `AGENTS.md`; `README.md`'s front door does the
same for a human or another assistant. Also closes a real gap `AGENTS.md`'s definition of done
already covers in spirit but not in words: `stories/F-01.md`'s completion record has sat with
its actual-hours field as the unfilled `<n>` placeholder since 2026-09-11, and nothing said
in writing that copying the *estimate* into that slot would be worse than leaving it unfilled.
`AGENTS.md` now says so explicitly, in both the definition of done and the refusal list.
*Why:* a startup file only works if something forces it to be opened first — a good checklist
nobody's pointed at is no better than no checklist. Wiring `CLAUDE.md`'s auto-loaded content to
name it is what actually makes this session-start-time, not just a file that exists. On the
hours question: the actual-hours field exists specifically so estimates get less wrong over
time (`stories/README.md`'s own words); a completion record with the estimate silently
duplicated into the actual slot is visually indistinguishable from one that was really measured,
which quietly defeats that mechanism while looking complete.
*Checked against D-028 at write time:* D-033 was free on every open branch as of 2026-09-14 —
fourteen branches checked, highest claimed anywhere was D-032 (on the unmerged
`foundations/F-03-interface-contract-v1`).
*Restored 2026-09-17:* this entry merged to `main` via PR #14 on 2026-09-14, then was silently
dropped — not reverted by any commit, just never carried forward — by a later two-parent merge
(`493164b`, "Merge branch 'main' into foundations/F-03-interface-contract-v1") that combined
this branch's tip with the F-03 branch's tip. Both sides had independently appended different
new decisions (this one; D-031/D-032) at the same point in the file, immediately before the
"Open" section — a textbook same-location conflict — and the merge resolved it by keeping only
the F-03 side's addition instead of combining both, with no conflict markers left behind to
flag it. `mergeable: MERGEABLE` on the resulting PR never meant the content was complete; see
the new entry on this exact failure mode below, D-035.

**D-034 · ChArUco golden-capture board: `DICT_5X5_250`, 8×6 squares at 20 mm, 15 mm markers.** (2026-09-17)
Board generated with OpenCV's own `cv2.aruco` module (matching the ChArUco detector R-02/R-03
will use, avoiding any dictionary or version mismatch between what's printed and what gets
detected), exported to PDF and printed at 100% scale — never "fit to page." Physical pattern
size 160 mm × 120 mm, comfortably inside both A4 and Letter with wide margins (~25–28 mm sides,
~60–88 mm top/bottom), leaving room for the per-PO QR code and orientation arrow the proposal's
design already reserves margin space for. 35 interior corners ((8−1)×(6−1)) for a stable
pose/intrinsics solve even with part of the board occluded by a part, per R-02's own acceptance
criterion. Marker size is 0.75× the square size (15 mm in a 20 mm square), a standard detection
margin. ~24 markers needed, well inside the 250-ID dictionary.
*Why:* R-02 (board detection) and R-03 (camera intrinsics from the board) both need to agree
with F-06 on these exact parameters, and nothing had fixed them yet. `DICT_5X5_250` is the most
widely documented ChArUco choice in OpenCV's own tutorials, which matters more than an
optimality argument for a team building this detector from scratch. The margin sizing follows
directly from this project's own scale-integrity design — the proposal already uses "the sheet
outline" (the paper's physical edge) as one of three independent scale checks, which only works
if there's a clean, measurable margin between the pattern and the paper edge in the first place.
*Not yet decided:* whether/when the per-PO QR code and orientation arrow actually get added to
the printed sheet — margin space is reserved for them, but F-06's golden capture set does not
need them (there is no real PO to bind to yet).

**D-035 · A two-parent merge can silently keep only one side's independent addition, with no conflict markers.** (2026-09-17)
Two branches each appended a different new decision at the same location in `docs/decisions.md`
(immediately before the "Open" section) without editing any of the same lines each other had
already written. Git's merge did not treat this as a conflict — nothing overlapped byte-for-byte
— and silently kept only one side's addition. `mergeable: MERGEABLE` on the resulting pull
request never reflected this; the loss surfaced only when a routine cross-branch decision-number
check (`D-028`) computed the true maximum decision number on `main` and found it lower than a
number known to have been merged already.
*Why:* this is the same class of defect the F-02 corruption already demonstrated for a story
file's `State:`/`Owner:` lines (see the 2026-09-14 progress-log entry) — a clean two-parent merge
proves the absence of conflict *markers*, not the presence of all intended *content* — but this
instance shows it is not limited to two sides editing the *same* lines. Two sides *appending
different new content at the same insertion point* can trigger the identical silent-drop
behavior. *Whenever a file is structured as an append-only log (a decisions file, a changelog),
treat a merge that touches it as unverified until someone reads the result and confirms every
independently-added entry from both sides survived — not just that the file merged cleanly.*

---

## Open — not yet decided

**O-001 · Does the reconstruction backbone count toward the generative-AI requirement?**
Worth 28 hours. With it the share is 33.0%; without it, 24.9%. Ask the instructor before
restructuring anything further.

**O-002 · The final presentation date is inherited and unconfirmed.**
December 7 to 18 came from an earlier draft where it was explicitly provisional and anchored to
a different course's calendar. It has since hardened into the milestone table, the sprint plan
and the horizontal axis of both Gantt charts without ever being verified against SFBU's academic
calendar. Confirm it before the schedule is treated as fixed.

**O-003 · Exact biweekly review dates.** Planned for every second Friday from 11 September;
re-anchor if the real dates differ.

**O-004 · Whether a formal user study is expected this semester.** An informal one is planned;
a formal one with IRB approval is future work.
