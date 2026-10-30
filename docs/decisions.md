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
Board generated with [calib.io's Camera Calibration Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)
(free for educational use; picked over generating directly via `cv2.aruco` for print precision —
it targets exact physical dimensions and explicitly warns against printer-driver rescaling, the
same risk this decision's margin reasoning is built around), exported to PDF and printed at 100%
scale — never "fit to page." Physical pattern size 160 mm × 120 mm, comfortably inside both A4
and Letter with wide margins (~25–28 mm sides, ~60–88 mm top/bottom), leaving room for the
per-PO QR code and orientation arrow the proposal's design already reserves margin space for.
35 interior corners ((8−1)×(6−1)) for a stable pose/intrinsics solve even with part of the board
occluded by a part, per R-02's own acceptance criterion. Marker size is 0.75× the square size
(15 mm in a 20 mm square), a standard detection margin. ~24 markers needed, well inside the
250-ID dictionary.
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
*Amended 2026-09-17:* generating the actual print file surfaced a parameter this entry originally
missed — calib.io's "ChArUco Legacy" toggle, which selects OpenCV's older pre-4.6 marker-to-
square layout versus the current default. **Left unchecked (non-legacy),** matching a fresh
build against OpenCV ≥4.7 with no old boards to stay compatible with; R-02/R-03's code must
construct its `cv2.aruco.CharucoBoard` the same way, since legacy and non-legacy boards can fail
to detect against the wrong mode's detector even with dictionary and geometry both correct. The
final print file is committed at `fixtures/charuco_board.pdf` (an exception to fixtures/ normally
holding nothing but a download script — this is the calibration target itself, not captured
data, and small enough to version directly) so the exact parameters, including this one, never
have to be re-derived from a third-party tool's UI.
*Corrected 2026-09-25 (R-02):* the print choice above stands — the board was generated with
calib.io's Legacy box unchecked, and the golden capture was shot on that print. **What was wrong
is the inference about OpenCV:** that print is the layout OpenCV builds with
`setLegacyPattern(True)`, not `False`. Evidence: rasterising the committed
`fixtures/charuco_board.pdf` and detecting it gives 35/35 corners with `True` and 0/35 with
`False` (all 24 markers found either way — the wrong flag only breaks corner interpolation); on
the real `s-01` golden frames the same check gave 26/26 frames posed with `True` and 0/26 with
`False`. The failure mode this amendment warned about — the right dictionary and geometry, the
wrong layout, nothing detected — is exactly what happened; only the mapping between calib.io's
checkbox and OpenCV's flag was backwards. R-02/R-03 code therefore uses `setLegacyPattern(True)`
(`recon/board_pose.py::make_board`), and `tests/test_board_pose.py` now checks the code's board
against the committed PDF itself (all 35 corners, and a 20 mm pitch measured from the PDF's
point units), so the code can't drift from the paper again unnoticed. Marked as a correction
rather than a new decision because nothing about the board or the print changes. Agreed by
@TabeenRaoof while working R-02.

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


**D-036 · Lightning AI is the shared remote GPU platform; Teamspace Drive holds shared large artifacts.** (2026-09-17)
F-05 moves the remote-GPU path from RunPod to Lightning AI. Day-to-day local development remains
on Apple Silicon per D-010, while NVIDIA-only reconstruction work uses the shared Lightning
Teamspace/Studio. Large assets that must survive Studio recreation — model weights, checkpoints
and golden-capture data — live outside git in the Teamspace Drive (or an attached external object
store later if capacity requires it), not only in one Studio's home directory. Studio-local files
and installed packages may use Lightning's native Studio persistence, but they are not the sole
copy of shared project data.

F-07's "pod" therefore means the shared Lightning GPU Studio for reproducible remote environment
setup. R-15 deploys the reconstruction service through Lightning's deployment/public-endpoint path
and must still prove that the resulting service answers over HTTPS/TLS; this decision does not
waive that acceptance criterion. RunPod remains an emergency paid fallback if Lightning capacity,
quotas, or deployment behavior blocks the project.

*Why:* the RunPod network-volume path required a payment method before the team could even finish
F-05, while Lightning allowed all three members to access the same T4 workspace and complete the
GPU smoke test without that upfront billing blocker. The platform choice must preserve the
original engineering intent, not merely rename it: shared durable data, reproducible GPU access,
and a secure deployable endpoint. Lightning documents a Teamspace Drive shared across Studios,
persistent Studio environments/files across sleep, and deployment options with TLS-secured
communication. Those capabilities cover the roles that the original RunPod volume/proxy design
was meant to provide, while keeping RunPod available as fallback rather than deleting it.

*Checked against D-028 at write time:* every open branch visible in the repository was checked
before assigning this number. The highest claimed decision found was D-035, so D-036 was the
first free number.


**D-037 · A claim puts the person's GitHub handle on the story file *and* the backlog index, and the checker enforces it.** (2026-09-26)
The backlog index (`stories/README.md`) gains an **Owner** column next to **State**. Claiming a
story means setting `State: Claimed` and the owner in **both** the story file and its index row,
written identically, in one commit, before any code. The owner is always the person who answers
for the work, as an `@github-handle` (several: `@a; @b`). **When an AI assistant makes the
claim, it writes the handle of the person it is working for, never its own or its tool's
name**; it asks if it can't tell whose handle that is. `scripts/check_story_states.py` now
reports: an index owner that differs from the story file's; a Claimed, In review or Done story
with no owner; a Blocked or Ready story that has one; an owner that isn't an `@github-handle`;
an AI assistant as owner; and any index row it can't parse. Before this, such a row was silently
skipped, so it dropped out of every check. Like the Blocked → Ready cascade (D-027), it only
reports owner problems and never writes an owner itself.
*Why:* until now a claim lived only in the story file, while the index everyone scans to pick
work showed just a state. "Claimed" said *that* someone had a story, not *who*, so finding out
meant opening every file. Agents made this worse: an assistant told to "claim R-03" had no
written rule about whose name to use, so a claim under the assistant's name, or under no name,
was one ambiguous instruction away. Following D-027's reasoning, a rule that lives only in prose
is kept by memory, and memory is what has failed here repeatedly (F-01's state twice, R-02's
state at merge). So the rule is written into `stories/README.md` step 2, `AGENTS.md` step 5, the
refusal list, `AGENTS_START_HERE.md`, `README.md`, the story template and the PR template, and
the script checks it on every run.
*Not decided here:* whether the checker should confirm that a handle is a real member of the
`proofshape` organisation. That needs network access and a token, and the script deliberately
runs offline. R-01's owner, `@yashidalwala`, passes the format check, but the matching GitHub
account appears to be `@dalwalyk`. Left for its owner to confirm rather than edited here.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-036, so D-037 was the first free number.

**D-038 · The `State: Done` commit ships inside the story's own pull request, before it merges — never as a follow-up.** (2026-09-28)
`AGENTS.md`'s definition of done previously said the state change happens "the moment a pull
request merges... in the same action" as the merge. That wording is retired: it describes
merging and updating state as one click, but GitHub's merge button cannot add a commit, so in
practice the state change either rode along in the PR (correct) or became a separate mental step
for whoever clicked merge (what kept failing). The rule is now sequential and mechanical: push a
commit to the pull request that sets `State: Done` in the story file, updates the matching index
row, and runs `check_story_states.py --fix`, with the completion record filled in — **pushed
*before* the approval the PR will merge on, not after**, so it is part of what gets reviewed, not
an addendum once review is finished. Only merge once that commit is in and approved, with
nothing pushed after it. Merging a pull request while its story still shows `In review` is not
allowed; an AI assistant finishing a PR pushes this commit itself, before asking for review,
rather than leaving it for whoever clicks merge. Instructions updated in `AGENTS.md` (definition
of done and the refusal list), `AGENTS_START_HERE.md`, `stories/README.md` step 6, `README.md`,
and the PR template.
*Why:* this exact step — set `State: Done` at merge — has now been missed on three consecutive
merges: R-02 (PR #29, closed out separately in #30), R-01 (PR #31, closed out in #34), and R-03
(PR #33, closed out in the same #34). Each time the fix was a whole extra pull request, and R-04
— the single bottleneck story the sprint plan calls out by name — sat `Blocked` between R-01's
merge and its close-out even though its real dependency was already satisfied. D-025 already put
this checklist item in front of every PR and D-027 made the cascade itself mechanical; neither
fixed the failure, because both still described the state change as something that happens *at*
or *after* merge, which is exactly the moment attention has moved on to the next thing. Moving it
to *before* merge — a required commit on the PR, not a separate action after it — is the same
fix D-037 applied to claiming: don't rely on someone remembering a step at the moment they're
least likely to remember it, make the artifact incomplete without it.
*Amended before this ever reached `main`:* the first draft of this entry said to push the state
commit "once it's approved... then merge." @dalwalyk caught in review that this cannot work —
`main`'s branch protection dismisses a stale approval on every new commit (D-013), so a commit
pushed after approval always needs a second approval round, which is exactly the two-event
problem this decision exists to remove. Fixed by reordering: the state commit is pushed *before*
the approval that will actually be used to merge, as part of the PR's normal content, not after
it. Left as an amendment rather than a silent rewrite so the reasoning that caught it isn't lost.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-037, so D-038 was the first free number.

**D-039 · R-05's three scale sources are compared as ratios to the nominal board pitch.** (2026-09-29)
D-007 states the printed board pitch, the caliper reading, and the CAD registration residual must
agree to within about 2%, but they start out as different kinds of quantities: the board pitch and
the caliper reading are both lengths in millimetres, while the CAD registration residual is
already a fractional scale error, not a length. R-05 (`recon/scale_agreement.py`) converts all
three to a scale ratio relative to the nominal board pitch before comparing them: the board pitch
is always the anchor at ratio 1.0 (it is the assumption the whole reconstruction is built on, not
an independent measurement of itself), the caliper ratio is `caliper_reading_mm / board_pitch_mm`,
and the CAD ratio is `1.0 + cad_registration_residual`. Any two of the ratios that are present and
differ by more than the threshold (default ~2%, configurable) flag the session. Board pitch alone
is rejected — at least one of the caliper reading or the CAD residual must also be supplied,
matching `minItems: 2` on `sources_compared` in the `ScaleAgreement` schema.
*Why:* without a shared unit, "any two disagreeing by more than 2%" is not well-defined between a
length and a fractional error. Ratios relative to the same anchor make every pairwise comparison
meaningful regardless of which two sources are actually present on a given session.
*Note:* the CAD registration residual is accepted as a plain optional float parameter. CAD
registration itself (`inspect/`) is Tier 2 and not built yet (`docs/manual.md` section 7 marks the
check "Planned"); R-05 does not depend on it or stub it out, it only defines the shape the input
takes once that residual exists.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-038, so D-039 was the first free number.

**D-040 · `capture/`'s npm project is Vite + vanilla TypeScript + Vitest, local to `capture/`,
on Node 24.** (2026-09-29)
C-01 is the first real code under `capture/`; nothing before it had picked a build tool, a test
runner, or even where the manifest should live. The npm project (`package.json`,
`node_modules`, configs) lives inside `capture/`, not at the repository root — matching
`capture/README.md`'s existing statement that "the npm project ... arrive[s] with the capture
stories" — so `.github/workflows/typescript-ci.yml` now detects `capture/package.json` and runs
with `working-directory: capture`, instead of the placeholder root-level detection it shipped
with at F-04. No UI framework is chosen here; that choice is left to C-04 (the app shell), since
C-01 only needs a file input and three number inputs. `docs/sprint-plan.md`'s CI table already
named Vitest as the intended TypeScript test runner, so that part was already settled — this
decision is about where the project lives and what else surrounds Vitest (Vite as the bundler,
plain TypeScript, ESLint's `strictTypeChecked`, Prettier).
*Why:* Vitest 5 requires Node `^22.12 || ^24 || >=26`; the CI workflow's Node 20 cannot run it,
so `typescript-ci.yml` moves to Node 24 (current LTS) in the same pull request that first makes
the `typescript` job actually execute something.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-039, so D-040 was the first free number.

**D-041 · `uploadFrame`'s `gyro` field travels as a JSON-encoded multipart part.** (2026-09-29)
`contracts/openapi.yaml`'s `UploadFrameRequest` schema has no `encoding:` block for the `gyro`
property, so OpenAPI's default for an object-typed property inside `multipart/form-data` applies:
it is serialized as its own part with `Content-Type: application/json`, not spread across
multiple form fields and not embedded inside the `frame` part. C-01's client
(`capture/src/api.ts`) builds it as `form.set("gyro", new Blob([JSON.stringify(gyro)], { type:
"application/json" }))`; the `FormData` spec then gives that blob part a synthesized filename
("blob"), which is what makes it arrive as a `File` on the receiving end rather than a plain
string field.
*Why:* recorded so R-14 (the real reconstruction endpoint, `Blocked`) parses `gyro` the same way
the client actually sends it, rather than guessing from the JSON-only example in
`contracts/examples/uploadFrame.request.json`, which — being JSON itself — cannot represent a
multipart request's part boundaries.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-040 (this pull request's own D-040, above), so D-041 was the
first free number.

**D-042 · The deployed reconstruction service is published on every merge, started by hand,
token-gated, on demand, and never more than one replica.** (2026-10-05)
R-15 splits D-014's "deployment runs only from `main`, on push" into two steps. Every push to
`main` that can change the image builds it, smoke-tests the container, and publishes it to GHCR
(`.github/workflows/recon-image.yml`). That step costs no GPU. Starting the service is a separate
manual workflow (`deploy-recon.yml`), which runs only on `main`. It deploys the image by digest,
never by a moving tag, to one Lightning T4 with bearer-token auth and a four-hour
`--max-runtime` cap. The service is started for testing, demos and sprint boundaries, then stopped.
Callers present the token; the phone app cannot hold a secret, so browser-facing access is
decided by whichever story first connects capture to the deployed service, not here.
*Why:* deploying on every push would start a paid T4 on every merge to `recon/`. It would also
restart the service mid-session, losing work in progress. Without auth, R-14's default accepts
any non-empty order code, so anyone holding the URL could spend our GPU credits. Gating by
construction is unchanged: `main` is still unreachable except through an approved pull request,
and the deploy job refuses to run against any other ref. **One replica, one uvicorn worker, is a
correctness constraint, not a cost choice:** R-14's `SessionStore` keeps sessions on one
container's local disk behind an in-process lock. A second replica or worker would answer
`session_not_found` for sessions the other created. Lifting that needs shared session storage,
which is a separate story.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-041, so D-042 was the first free number.

**D-043 · Story files are organised into one subfolder per lane letter.** (2026-10-09)
`stories/` had grown to 39 files in one flat directory (7 F, 19 R, 13 C) with more lanes (I, G,
A, X) still to come. Files now live at `stories/<lane-letter>/<ID>.md` (e.g. `stories/F/F-01.md`,
`stories/C/C-09.md`) instead of directly under `stories/`. This does not relitigate D-023 (one
file per story, index keeping one row each) — it only changes where those files sit.
`stories/README.md`'s own index links are updated to point at the new paths, and
`scripts/check_story_states.py` now resolves each story's file from whatever path the index
actually links to, rather than assuming `<ID>.md` directly under `stories/` — so the checker
does not silently break the next time the layout changes. No story file links to another story
file by markdown path (only the index does), so that was the only link surface to fix; prose
mentions of the old flat paths in `docs/progress-log.md` and code comments are left as-is,
consistent with the progress log's own append-only, never-rewritten history.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-042, so D-043 was the first free number.

**D-044 · R-06's height cut is measured from the reconstructed board surface, not R-04's z=0
assumption; R-04 itself is unchanged.** (2026-10-11)
R-17 measured the real board sitting 1.6-5.4 mm below z=0 after R-04's alignment, with tilt up
to 0.9°, so R-06's "2 mm above z=0" cut removed 3.6-7.4 mm of every part's base (a quarter of
s-04's 30 mm height). R-21's story notes left the owner a choice: fix it at the source (teach
R-04 to land the board on z=0) or measure it downstream (fit the real board plane in R-06 and
cut relative to that). Chosen: measure it in R-06, via a new `measure_board_surface` in
`recon/part_segmentation.py`. R-04 is left alone, for two reasons. First, R-22 (owned separately,
in progress) is investigating a different R-04/R-01 scale discrepancy at the same time; changing
R-04's alignment now would move the ground under that measurement while it's being taken.
Second, the plane fit's own residual noise is what the height threshold needs anyway (see
below), so R-06 has to do this measurement regardless of what R-04 does.

The height threshold itself is no longer a fixed 2 mm: R-17 also found VGGT reconstructing a
weak second board layer 4-6 mm above the real board on two of the five golden objects, which a
fixed margin moved down to the real board would let through. `measure_board_surface` instead
histograms point density above the fitted plane and sets the threshold to the top of the
highest dense bin (never below a 2 mm floor), so a noisier or thicker board layer gets a taller
cut automatically. `recon/tsdf_fusion.py`'s `fuse_run` fits the same surface from its own
valid-depth pixels and uses it for both the mask and the voxel grid's z bounds, so the fused
mesh (and so the deployed service, which calls `fuse_run` directly) keeps the same base R-06
now keeps.

Real-data rerun on the golden set (offset / tilt / threshold, measured vs. R-17's old cut):

| | s-01 | s-02 | s-03 | s-04 | s-05 |
|---|---|---|---|---|---|
| Offset (mm) | 1.9 | 0.9 | 3.2 | 5.4 | 3.2 |
| Tilt (deg) | 0.38 | 0.53 | 0.23 | 0.78 | 0.47 |
| New threshold (mm) | 6.0 | 4.5 | 3.5 | 2.5 | 3.5 |
| Old cut above real board (mm, R-17) | 4.2 | 3.6 | 4.8 | 7.4 | 5.3 |

*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-043, so D-044 was the first free number.

**D-045 · Default confidence threshold is `1.0001`, chosen to exclude only VGGT's exact
confidence floor.** (2026-10-11)
R-17 found 11.5-29.3% of segmented points on the golden set are stray, and that almost all of
them sit at VGGT's confidence floor of 1.0 (R-01's own note: "VGGT's depth confidence bottoms
out at 1.0"). R-20's committed sweep (`recon/confidence_sweep.py`, replacing R-17's one-off
script) measured stray-point fraction and part voxel coverage at a grid of thresholds on all
five real golden-set runs; the full grid is in `stories/R/R-20.md`'s notes. It reproduced R-17's
own numbers as a cross-check first: s-04's baseline stray fraction (29.3%) matches exactly, and
R-17's diagnostic 1.5 cut reproduces R-17's reported 1.1-2.3% band.

The measurement's result was not the one expected going in. Almost all of the achievable
stray-fraction reduction is already captured the instant the threshold clears the *exact* floor
value -- the median pixel confidence is 1.0 on every object, and 50-73% of pixels sit exactly
there. Pushing the threshold any higher past that buys negligible further stray-fraction
reduction while steadily costing part coverage, until a much higher value (~3-5, object-
dependent) where fusion starts fragmenting badly -- R-17's own warning against overshooting:
at confidence >= 5, s-05 kept only 21% of its points and its walls became too sparse to measure.
Baseline vs the chosen value vs R-17's informal 1.5 diagnostic, stray fraction / part voxel
coverage:

| | s-01 | s-02 | s-03 | s-04 | s-05 |
|---|---|---|---|---|---|
| baseline (0.0) | 19.2% / 100% | 18.7% / 100% | 11.4% / 100% | 29.3% / 100% | 17.7% / 100% |
| chosen (1.0001) | 2.5% / 88.2% | 2.5% / 96.6% | 1.8% / 93.5% | 2.0% / 99.8% | 2.5% / 99.1% |
| R-17's diagnostic (1.5) | 2.3% / 68.4% | 1.9% / 75.3% | 1.0% / 86.7% | 1.8% / 96.2% | 1.7% / 92.0% |

1.0001 reaches nearly the same stray-fraction cleanup as 1.5 while preserving meaningfully more
of the part on every object, most visibly on s-01 (88.2% vs 68.4%). s-01 and s-02's baseline
cluster is itself contaminated by R-17's "weaker second board layer" (its baseline x/y extent is
exactly 160x120mm, the board pattern's own size, not the padlock's), so their coverage numbers
are measured against a cluster that is partly that spurious layer, not purely the part -- the
true coverage cost on those two is likely lower than the table shows.

`recon.vggt_runner.DEFAULT_MIN_CONFIDENCE = 1.0001` and the CLI default change with this
decision. `recon.reconstruction_runner`'s dispatcher leaves MASt3R and COLMAP at `0.0`: MASt3R's
confidence is a different, unmeasured scale, and COLMAP's is a binary validity mask, not a
learned confidence, and already refuses any nonzero threshold.

*Why not a simpler round number like 1.5 or 2.0:* the sweep showed those cost real part coverage
for no further stray-fraction benefit; the knee sits essentially at the floor itself, not
partway up the confidence scale.

*Checked against D-028 at write time:* every branch on `origin` was fetched and checked.
`recon/R-21-board-surface-height-cut` (PR #125, open at the time) already held D-044 for an
unrelated decision, so D-045 is the next number free everywhere, not D-044.

**D-046 · `requestCameraPermission` hands back a live `MediaStream`; the caller owns stopping
it.** (2026-10-15)
C-05 changed `requestCameraPermission` (`capture/src/camera.ts`) to stop doing what C-02's
version did — requesting the camera, proving permission, then immediately stopping every track —
and instead hand the granted `MediaStream` back alive, so a caller can keep it running for a real
session. That shifts an obligation onto every caller: nothing stops the stream automatically
anymore, so whoever holds it has to release it on every path that's done with it, not just the
one happy path. `capture/src/shell.ts`'s `releaseCamera()` is the reference implementation —
called before acquiring a new stream (so a second `Start` can't overwrite a live one without
stopping it first), when a granted stream turns out to be unusable (`cameraPreview.play()`
rejecting), and on definitive teardown (a confirmed-successful `finishSession`). Any future caller
of `requestCameraPermission` — C-09's capture loop is the next one — needs the same discipline:
hold the stream only as long as it's wired to something live, and release it on every exit path,
not just success.
*Why:* TabeenRaoof's review of PR #134 (C-05) caught three separate camera-stays-on leaks that
all trace to this one unwritten rule — a second `Start` click overwriting `state.cameraStream`
without stopping the first, a `play()` rejection leaving a granted-but-unusable stream running
with no way to release it, and a successful-finish teardown that stopped the stream's tracks but
left `srcObject` and `cameraGranted` stale. Three independent bugs from the same missing
invariant is exactly the pattern D-025/D-027/D-038 already exist to stop recurring in different
forms — the fix here is writing the rule down where the next caller (C-09) will find it, not just
patching the three call sites.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-045 (on `recon/R-20-confidence-in-fusion`, not yet merged), so
D-046 was the first free number.

**D-047 · Open3D is a pinned `requirements.txt` dependency.** (2026-10-15)
`open3d==0.18.0` is now in `requirements.txt`, the CPU/laptop-and-CI set, so CI, laptops and the
recon image all install it. C-08's next-view scoring uses Open3D's `RaycastingScene`, which is
what the proposal's pinned stack names for "visibility and next view". `recon/next_view.py`
imports Open3D at module level, as R-08 does with its real dependencies. R-07's Poisson fallback
keeps its lazy import, which still works.
*Why:* C-08's story says outright that tests skipping in CI don't count toward the definition of
done, and an optional Open3D is precisely what makes them skip. It is CPU-only use, and
`requirements.txt` exists for CPU dependencies. The separate requirements files (GPU, COLMAP)
hold things that must *not* be installed on CPU, macOS or CI, which is the opposite case.
0.18.0 is the version verified on macOS ARM64 with numpy 1.26.4 (VGGT needs numpy<2):
`RaycastingScene` works there, and `pip check` is clean. R-20's macOS crash is inside Open3D's
Poisson reconstruction, not its raycaster.
*Cost, accepted:* the Linux wheel is ~400 MB and pulls dash, flask, pandas and scikit-learn, so CI
installs are slower and the recon image is bigger. Two side effects: R-07's Poisson success-path
test now runs in CI instead of skipping, and R-07's "Open3D not installed" test now fakes the
missing import rather than skipping whenever Open3D is present.
*Alternatives considered:* a separate `requirements-open3d.txt` installed by CI and the bootstrap
script but not the image. It works, but it's one more file to keep in sync, for an image size
the service will likely need anyway once rescan guidance runs there. `open3d-cpu` (~100 MB) only
ships Linux wheels, so it would need platform markers and two package names for one import. A
pure-numpy raycaster would need its own BVH or z-buffer to be fast enough on golden-size meshes.
That doesn't fit a 4-hour story, and it departs from the pinned stack.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-046 (on `main` and every open branch), so D-047 was the first
free number.

**D-048 · C-13 connects `capture/` to the deployed service with a wildcard-origin CORS policy
and the bearer token embedded in the client build; the order code is the real access control.**
(2026-10-15)
D-042 deliberately deferred this: "the phone app cannot hold a secret, so browser-facing access
is decided by whichever story first connects capture to the deployed service, not here." Traced
concretely before deciding: Lightning's `--token-auth` (R-15, `scripts/deploy_recon.py`) gates
every request at the proxy, including a plain page load (R-15's manual check: POST without the
token → 401). That rules out serving `capture/`'s static files from the recon container itself
— a phone's top-level navigation can't attach an `Authorization` header before any JS runs, so
the page would never load — and it would tie the free static frontend's reachability to the
paid, on-demand GPU container's uptime.

So `capture/` is served from a separate origin (for C-13 itself, the existing `npm run
dev:device` LAN-HTTPS pattern C-02/C-03/C-09 already used for phone checks, pointed at the
deployed base URL instead of the stub; a permanent host like GitHub Pages, named in the original
proposal, is explicitly deferred to its own future story). `recon/service.py`'s `create_app` adds
`CORSMiddleware` with `allow_origins=["*"]` and `allow_credentials=False`. The bearer token is
resolved from a build-time `VITE_RECON_TOKEN` (`capture/src/config.ts`) and attached as
`Authorization: Bearer <token>` on every outgoing `fetch` (`capture/src/api.ts`) — visible to
anyone who opens dev tools on the page, not a real secret.

*Why the wildcard origin is deliberate, not an oversight:* CORS only restricts *browser* JS; a
non-browser caller (`curl`, `scripts/smoke_r14.py`) ignores it entirely. Once the token is
already public-in-the-bundle, restricting the allowed origin adds no real protection and would
otherwise have to track whatever LAN IP a tester's phone session happens to use that day. The
service's real access controls stay what they already were: Lightning's bearer token at the
proxy, and `create_session`'s check against `known_order_codes` — the order code is what
actually gates whether a session does real work, which is why it is named as "the access
control" in C-13's own acceptance criteria.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-047 (on `capture/C-08-next-view-scoring`, not yet merged), so
D-048 was the first free number.
*Superseded in part by D-049 (2026-10-15):* C-13 ended up not using Lightning, so this hosting
choice did not govern C-13's end-to-end check. The code it describes stays, unit-tested, for
whichever story next connects to Lightning. Two of its premises are open questions, not facts,
and are recorded as O-005: whether the browser's preflight gets past `--token-auth`, and the
order code as access control — the deployed `app = create_app()` passes no `known_order_codes`,
so it accepts any non-empty code.

**D-049 · C-13's end-to-end check runs against `recon.service` on the M2 Max, reached through the
dev server's `/v1` proxy; Lightning is out of C-13's scope.** (2026-10-15)
Agreed by the team on 2026-10-15, after C-13's real-phone check hit two walls on Lightning. The
service image only exists for code already on `main` (D-042), so the branch's own changes could
never be tested before merging. And the next start never got a ready replica (run `38838190937`,
no status reported for 30 minutes), with the teamspace's credits in doubt. D-010 already makes
the M2 Max the development machine, and R-17 ran the whole chain on it over Metal.

So for C-13, `recon.service` runs natively on the Mac — `uvicorn` bound to `127.0.0.1`, not in
Docker, which has no Metal access (D-012). `PROOFSHAPE_RECON_DEVICE=mps` selects Metal
(`recon/reconstruction_runner.py`'s `resolve_device`); unset stays `cuda`, so the container is
unchanged. The phone keeps loading `capture/` from `npm run dev:device` over HTTPS, and
`PROOFSHAPE_RECON_PROXY` makes that dev server forward `/v1` to the service instead of mounting
the contract stub (`capture/stub/dev-recon-target.ts`).

*Why a proxy rather than calling the service directly:* the page is HTTPS (Safari only allows the
camera in a secure context), so the browser blocks `http://<mac>:8000` as mixed content, and
serving the service over HTTPS too would mean a second self-signed certificate to accept on the
phone. Through the proxy every request is same-origin: no CORS, no token, one certificate. `xfwd`
makes the service's existing `public_model_url` return a `glb_url` on the dev server's own
origin, which the phone can load. The service itself listens only on `127.0.0.1`; the dev server
is the only way in, and it is on the LAN only while a check is running.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. The
highest decision anywhere was D-048 (on `capture/C-13-connect-deployed-service`, this branch), so
D-049 was the first free number.

**D-050 · When frames carry no EXIF focal length, the service calibrates the camera from the
board (R-03) and poses the frames with that.** (2026-10-15)
C-13's first two real-phone sessions failed with 0 board poses although the board was found in
every frame (16–18 markers each). The capture app takes each frame from the live camera preview
through a canvas, and a canvas JPEG has no EXIF, so R-02 had no focal length to pose with: every
frame was `intrinsics_unavailable`. The golden capture never exercised this, because it is
Camera-app photos with EXIF. `recon/service.py` now runs R-03's `calibrate_from_board` on the
corners R-02 already recorded, then R-02 again with that K for every frame, into
`board_pose_calibrated/`, with the fit recorded in `board_calibration.json`. On the first failed
session's own 19 frames this gave 0.20 px calibration RMS and 18 of 19 frames posed, and the full
pipeline then completed with a metric result.
*Why the board rather than another source:* the board is in every frame by design, R-03 already
calibrates from it with no initial guess, and the frozen contract has no field for the phone to
send a focal length (changing it is a D-031 contract change). Frames that do carry EXIF never take
this path. A calibration failure is reported by name, under the existing
`metric_alignment_failed` code, rather than as the misleading `insufficient_baseline` the phone
showed before.
*Checked against D-028 at write time:* D-049 on this branch was the highest decision anywhere, so
D-050 was the first free number.

**D-051 · Metric alignment is anchored on lifted ChArUco board geometry, not camera centres.** (2026-10-16)
R-23 replaces R-04's camera-centre similarity fit with a direct fit from reconstructed ChArUco
corner observations to the printed board's exact millimetre coordinates. The corners R-02
already detects are rescaled into R-01's processed pixel frame, lifted through R-01 depth and
camera intrinsics/extrinsics, and solved with the existing Umeyama `solve_similarity`. R-22's
board-scale diagnostic reuses the same lifting helper rather than carrying a second copy.

*Why:* R-22 measured every golden reconstruction 4–9% small and found the error tracked the focal
ratio between the board-pose and VGGT cameras. Camera-centre distance depends on the focal length
assumed by the pose source; the physical board dimensions do not. A synthetic regression test
makes the legacy centre fit recover a deliberately wrong 0.9 scale while the board anchor
recovers 1.0 from the same reconstruction. Fewer than 30 usable lifted corners is an explicit
error; there is no fallback to camera centres.

The `metric_alignment.json` transform keys and convention stay unchanged, so R-06, R-07 and
R-09 continue consuming the same scale/rotation/translation representation.

Three implementation constraints were made explicit during review:
- R-23 uses the exact resize/crop geometry recorded by VGGT in `run.json`. Legacy runs are
  accepted only when they are clearly uncropped landscape frames; portrait/ambiguous legacy
  runs are refused and must rerun R-01 rather than guessing a crop offset.
- Only board frames where R-02 wrote `pose_found = true` contribute lifted corners to the fit.
  A frame R-02 distrusted cannot silently affect metric scale.
- R-22's post-R-23 `measure_board_scale` is an acceptance measurement on the same board anchor,
  so it is expected to sit near 1 by construction. The separate legacy camera-centre diagnostic
  is retained to preserve the independent signal that exposed the original focal-length scale
  mechanism; CAD checks on s-04/s-05 provide a part-level cross-check.

The board anchor intentionally samples depth at the detected ChArUco corner locations without
applying R-20's global point-cloud confidence threshold. Those observations are the ruler used to
set scale, not a dense surface estimate; dropping them with a threshold chosen for cloud/fusion
would change the physical ruler opportunistically. Fit quality is therefore reported by residuals
instead of silently substituting the camera-centre fallback.

*Checked against D-028 at merge-sync time:* `main` has D-047 and D-052–D-053; open C-13
holds D-048–D-050. D-051 remains unique and is inserted here before D-052 for readability.

**D-052 · Planning work gets stories too, in a P lane; past work is recorded as it was.**
(2026-10-16)
A new lane, **P — Planning**, holds the work that belongs to no single story: the proposal and
the sprint plan, decisions and process rules, writing and splitting stories, keeping documents
such as `docs/manual.md` in step with `main`, audits and reconciliations across stories, and the
class reviews and reports. Its place is the planning documents (`docs/`, `submissions/`,
`stories/` and the process files at the root), so D-020 still holds: a lane is a place, and anyone
picks up a P story. Branch names use `planning/`. Tier is `—`, because the delivery tiers
(D-001) apply to product features, not to documents.
- **The boundary.** Anything done inside a story's own pull request stays in that story's hours:
  its claim, its `Done` state and completion record, its progress-log entry, and any follow-up
  stories it files. P takes only what no single story owns, plus work on someone else's story
  after its pull request merged (a late close-out, a reconciliation, a record correction),
  because that owner's actual hours never include it.
- **Past work** becomes `Done` stories, owned by the handles git shows did the work; teammates
  confirm, correct or remove their names in review. Where no estimate existed before the work,
  Est is `—`, never a number made up afterwards. `check_story_states.py` accepts `—` there, and
  nothing else that isn't `<number> h`. Actual hours stay `<n> h actual` until the owner gives a
  real figure (D-033), and a figure reconstructed from memory says so. A co-owned story records
  total person-hours, with each person's share in its notes where known. Work pushed straight
  to `main` before branch protection existed (root commit `88d4ad6`) had no pull request. Its
  record says so, and the pull request that adds the story is its first review.
- **Claims that predate the lane** (P-20, P-21, P-22) land with it. They could not reach `main`
  before the lane existed there (D-037).
- **Definition of done for P.** Merged through a pull request and approved by one other member,
  as always. A document can't be unit-tested, so its written check is the reviewer reading it
  against the story's acceptance criteria. Code inside a P story keeps the normal rules: tests
  written alongside it (D-029), and a run on a second member's machine.
- **Budget.** Class review and report stories count against the sprint plan's 18 h `Reviews`
  row. Stand-ups and boundary calls are ceremony overhead, already inside the weekly hours, and
  get no story. Nothing else P covers had a budget line; recording it is how we find out what it
  costs.

*Why:* only code stories carried actual hours. Every hours total therefore left out the
proposal, the plan, the process rules, the backlog write-ups, the documents and the reviews, and
no estimate for that work could ever be checked. Recording it in pull request descriptions
instead would have kept it out of the index, out of the checker and out of the completion records
everyone already reads, so it uses the same story machinery as everything else.
*Not decided here:* whether time spent reviewing others' pull requests is recorded anywhere. The
sprint plan (§8) already counts review time inside each lane's estimates, and a review is not a
deliverable with one owner, so it gets no P story.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. D-047
(`capture/C-08-next-view-scoring`, #144), D-048–D-050 (`capture/C-13-connect-deployed-service`,
#146) and D-051 (`recon/R-23-board-scale-anchor`, #152) are held on open branches, so D-052 was
the first free number. Whichever of these merges after another will need its decisions
reconciled together, not one at a time.
*Rechecked 2026-10-17:* #144 has since merged, so D-047 now sits just above this entry.
D-048–D-050 (#146) and D-051 (#152) are still on open branches, and nothing else on `origin`
goes past D-047. When those two merge after this one, their entries land below D-052, out of
numeric order; that is expected, not a lost or duplicated number.

**D-053 · Inspection code is the package `inspection`, at `inspect/inspection/`; `inspect/` stays
a bare directory.** (2026-10-19)
Settles the import name D-030 left to I-01. The lane directory stays `inspect/` (D-020) with
**no** `__init__.py`. The importable package is `inspection`, living at `inspect/inspection/`,
and `pyproject.toml` replaces `[tool.setuptools.packages.find]` with an explicit
`packages = ["recon", "inspection"]` plus `[tool.setuptools.package-dir] inspection =
"inspect/inspection"`. Code imports it as `from inspection import registration`. I-01 implements
this and strengthens `tests/test_packaging.py` to prove it: the standard library's `inspect` still
resolves under `sysconfig`'s stdlib path, and `inspection.__file__` sits under
`inspect/inspection/`.
*Why:* two simpler layouts fail. An `inspect/__init__.py` at the repository root makes `inspect/`
a regular package, and `python -m pytest` (how CI runs) puts the repository root first on
`sys.path`, so `import inspect` inside pytest itself would get ours; a bare directory is only a
namespace package, which the import system tries last. And mapping `inspection` straight onto
`inspect/` needs that same `__init__.py`, while `packages.find` cannot discover a package under a
name different from its directory, hence the explicit list. Renaming the directory would reverse
part of D-020 and every document citing it; folding the code into `recon/` would make `recon/` the
whole backend, which the lane layout exists to prevent.
*Not decided here:* anything about the lane's modules beyond their package. Their names
(`synthetic`, `cad`, `registration`) are in I-01..I-05's story files.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked. D-048–D-050
(`capture/C-13-connect-deployed-service`, #146) and D-051 (`recon/R-23-board-scale-anchor`, #152)
are held on open branches and D-052 is on `main`, so D-053 was the first free number.

**D-054 · Orbit coverage is recomputed server-side; `check_frame` is `unavailable` until the
frame-gate result is actually reported.** (2026-10-23)
A-06's two phone-side tool sources (`capture/src/frame_gate.ts`, `capture/src/orbit.ts`) reach
the server two different ways. Orbit: every accepted frame's gyro reading is already persisted
at `<session_dir>/frames/frame_NNNN.gyro.json` by `SessionStore.add_frame`, so
`recon.agent_tools.coverage_report` replays `orbit.ts`'s `computeCoverage` logic over those files
directly — no contract change. Frame gate: a rejected frame's pixels and gyro are never
uploaded, so there is nothing to recompute server-side, and nothing has ever reported a
rejection to the server either — only a new contract field could carry that, which needs a
D-031 request at a sprint boundary (Oct 30 or Nov 13), deliberately not added here.
`recon.agent_tools.check_frame` is therefore `unavailable` for every call, the same honesty
`vlm_inspect` already applies to A-01 not existing yet.
*Rejected alternative, caught in review:* an earlier version of this decision had `check_frame`
answer `upload_frame`'s own server-side accept/reject outcome instead — a real signal, but a
different question than the one the acceptance criteria name ("`check_frame` over the gate
result"). @mbj1994 caught that this made the story's own criterion look satisfied while actually
implementing a different tool under the same name. Reverted; no `last_frame_event` state was
kept.
*Why:* A-06 was picked over A-11 specifically to avoid being gated on a D-031 sprint boundary.
Adding frame-gate telemetry now would reintroduce that same gate. Honestly returning
`unavailable` costs nothing and invents nothing; a real, gate-rejection-aware `check_frame` is
explicit future work once the agent (A-07) actually needs it and the team decides to spend a
D-031 slot on it.
*Checked against D-028 at write time:* every branch on `origin` was fetched and checked; nothing
past D-053 exists anywhere, so D-054 was the first free number.

**D-055 · A-01 samples asynchronous Gemini observations outside session.json.** (2026-10-26)
The implementation pins the stable Developer API model code `gemini-3.5-flash-lite` and
`google-genai==2.33.0`, never a `-latest` alias. Google's model documentation identifies
that exact model code as stable, image-input capable and structured-output capable:
https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite . The deprecation schedule
checked on 2026-10-26 announces **no shutdown date** for it:
https://ai.google.dev/gemini-api/docs/deprecations . Record each response's actual model
version and reported token counts; absent counts stay unknown, never zero-filled.

**Owner-confirmed setup (2026-10-26):** Google project name `ProofShape`, **free** API
tier. @mbj1994 reports that the `GEMINI_API_KEY` GitHub Actions secret is saved. The key
has not been read or copied into source code. No paid billing account is used for this
free-tier check; identify the project and tier in the local evidence report. The real
API check subsequently ran from the owner's Windows checkout: 11 checked replies,
correct screen/different-part controls, and returned model/token metadata recorded in
`stories/A/A-01.md`. The key was loaded privately in that PowerShell session. Lightning
key loading and the production rollout remain unverified; the smoke check follows merge.
Private account identifiers and keys stay out of git.

**Data-use limit:** the free tier is permitted here only for team-owned, non-confidential
test parts. Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms), checked
2026-10-26, allow unpaid inputs/outputs to be used for product improvement, including
human review, and prohibit submitting confidential information. Real supplier photographs
require a paid-tier project: Google says paid prompts, images and responses are not used
for product improvement. This is a condition of enabling supplier use, not evidence that
the current free project has been upgraded.

Two app-owned executor workers have at most 16 jobs running/queued across all sessions.
An upload never waits for API work or capacity. Full queues, missing keys, API failures,
invalid/blocked/empty responses, shutdown and missing join results become `unchecked` with
a reason. Each SDK request has a 30 s transport timeout and one attempt (no SDK retries).
The post-pipeline join waits at most 5 s across the entire session; a late reply cannot
replace a terminal timeout file. These are declared operational bounds, not measured
latency or cost claims. D-042's single process/replica remains required.

Per-frame files live at `<session>/vlm/frame_NNNN.json`; the aggregate is
`<session>/vlm_checks.json`. Neither is stored in `session.json`. A-06's `vlm_inspect`
now takes an optional explicit session directory to read this evidence; existing callers
without that path remain unavailable. The caller must not read guessed VLM fields from a
session record. A-02 is still responsible for any veto policy; A-11 owns the future wire
field/UI. There is no endpoint/contract change or Pass/Fail decision here.

*Why:* SDK latency cannot block capture, bounded work cannot grow without limit, and
reconstruction overwrites its saved session record. Separate atomic terminal evidence
prevents both lost worker results and timeout/aggregate disagreements. Failure reasons
and counts are evidence rather than a fabricated pass. Supplier frames, including frame 0
as a reference for later checks, are sent to Google only after key setup.
*Checked against D-028:* fetched all origin branches, inspected their decisions logs;
D-054 was the highest number anywhere. D-048–D-050 remain on C-13's open branch and are
not reused. The owner-run live Gemini evidence is recorded in A-01. Production Lightning
smoke testing follows merge under D-013; repeated --env support was checked in the pinned
SDK source.

---

**D-056 · Contract v1.1: the VLM hint on the upload response, and two error codes.** (2026-10-29)
The first change to the frozen contract, made under D-031 by A-11. `info.version` goes from
1.0.0 to 1.1.0 and the path stays `/v1`: a change that only adds an optional field or an enum
value is a minor bump under the same path, and D-032's `/v2` stays for a breaking change.
An optional field that has nothing to say is left out of the response, never sent as `null`.
Three things change:

- `UploadFrameResult.vlm_feedback` carries the newest *checked* A-01 result for the session:
  its `frame_index`, `hint` and the four checks (`same_part`, null for frame 0, the reference;
  `hands_occluding`; `screen_photo`; `lighting_adequate`). It is read from the per-frame files
  on disk after the upload is queued, so an upload never waits for the model. The model name,
  token counts and `unchecked` reasons stay on the server. The hint is coaching only: it never
  rejects a frame or affects a verdict, and capture shows it as plain text.
- `malformed_request` is for a request that is itself broken: a missing multipart part, an empty
  photo, an unreadable gyro reading. It replaces the borrowed `vlm_veto` and
  `frame_rejected_exposure` (R-14's recorded debt), so `vlm_veto` now means only A-02's semantic
  check. It is described for any route, so another route can use it without a new D-031 round.
- `internal_error` is for a crash unrelated to the photos: an unexpected exception during
  reconstruction, or a session left pending by a restart. Until now these were reported as
  `metric_alignment_failed`, whose advice (lay the sheet flat) was wrong for them. The
  supplier gets a fixed message; the exception text goes to the server log only (#186). Today it
  reaches the phone only inside `ReconstructionResult.warnings` (`internal_error: ...`), because
  a failed reconstruction has no `ErrorDetail` of its own in v1.

The same change widens `Region.contributes_to_verdict`'s wording from `unobserved` to every
provenance other than `observed`, which D-005 already required, and `tests/test_contract.py`
now checks it that way.

*Why:* the hint can't reach the phone without a field, and the app polls nothing during capture,
so the upload response is the only place for it. Sending only the newest checked result keeps
the response small and avoids showing a stale `unchecked` placeholder as advice. The two codes
ride along because each needs the same two approvals, and holding them for separate rounds
would cost a sprint boundary each. They are not generative AI; A-11's completion record notes
their share of the hours.
*Checked against D-028:* fetched every origin branch and read its decisions log on 2026-10-29;
D-055 was the highest number anywhere, so D-056 was free.

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

**O-003 · Exact biweekly review dates. — Closed 2026-10-20.**
~~Planned for every second Friday from 11 September; re-anchor if the real dates differ.~~
The delivered class-review record establishes Progress Review 1 on 18 September 2026 and
Progress Review 2 on 2 October 2026. The sprint plan is therefore re-anchored to every second
Friday from 18 September: 18 Sep, 2 Oct, 16 Oct, 30 Oct, 13 Nov, 27 Nov, 11 Dec. No separate
written instructor-confirmation message is stored in the repo, Gmail or calendar; the delivered
class-review artifacts are the evidence used for this closure, and the plan says so explicitly.

**O-004 · Whether a formal user study is expected this semester.** An informal one is planned;
a formal one with IRB approval is future work.

**O-005 · How a browser reaches the Lightning deployment.** (2026-10-15, from D-049)
The first story that connects `capture/` to Lightning again has to settle three things D-048
assumed. First, whether `--token-auth` refuses the token-less CORS preflight that any
`Authorization` header triggers; one `curl -i -X OPTIONS` with an `Origin` and
`Access-Control-Request-Headers: authorization` against a running deployment answers it. Second,
whether a token visible in the client bundle is acceptable at all. Third, the order code as the
access control: the deployed `app = create_app()` sets no `known_order_codes`, so today any
non-empty code starts a paid GPU run.
