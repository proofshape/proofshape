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

**Amended 2026-09-06.** Batch remains the guaranteed path and now includes **one completion-driven
recapture round**: after the first reconstruction, K completions run over the mesh and high
disagreement over a critical region triggers a request for a few more photographs from named
positions (§6.6). Interactive mode remains optional. This resolves a contradiction — completion
guidance previously had no consumer in batch mode, so cutting interactive mode would have stranded
it. The verdict path is untouched; D-005 and D-006 stand.

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

**D-010 · The rented pod is the canonical environment; local development is optional.** (2026-09-03)
*Why:* every milestone demonstration and every latency measurement must run in one place, or
results are not comparable between people or across weeks.

**Amended 2026-09-06.** Inverted. The original made an Apple Silicon laptop primary and the pod a
fallback, which made the reference environment depend on hardware only one member has. The pod is
now primary and local development on Apple Silicon is a convenience for whoever has it. Measurements
quoted anywhere come from the pod.

**D-011 · If you develop on Apple Silicon, force fp16.** (2026-09-03)
Measured on Apple Silicon: bf16 runs at roughly half the throughput of fp16 on Metal, the opposite
of a modern NVIDIA card.
*Why:* reconstruction models commonly select bf16 automatically when they detect a capable GPU.
That default is wrong on Metal and must be overridden.

**Amended 2026-09-06.** Demoted from a project-wide decision to a documented gotcha for the optional
local path, following D-010's inversion. It does not apply on the pod.

**D-012 · The container image is the unit of deployment; local native runs are the exception.** (2026-09-03)
*Why:* one image, one command, on the pod. Docker Desktop on macOS gives containers no Metal access,
so anyone developing locally on Apple Silicon runs natively instead — that is the exception, not the
rule.

**Amended 2026-09-06.** Reframed to match D-010. The container path is canonical; the native path is
the local-only workaround.

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

**D-016 · Capacity raised to 9 hours per person per week.** (2026-09-03)
*Why:* the added hours fund the components the project's differentiation depends on — guidance,
gating, and communicating with a non-expert supplier.

**Amended 2026-09-06.** The original reasoning was ratio arithmetic about a share target. Component
sizing is now on merit, so that reasoning no longer governs and has been moved to private notes
(see `docs/needs-human-decision.md`). The capacity figure itself is unchanged.

**D-017 · Generative-AI components are never cut; the cut list was inverted.** (2026-09-03)
*Why:* the original ordering dropped the report narrative and frame checks first, which were the
only generative components. Any schedule pressure would have cut precisely what the course grades.

**Amended 2026-09-06.** The protection stands, but the reasoning above is superseded. These
components are protected because after D-002's amendment they are **structurally load-bearing** —
completion decides where the recapture round photographs, the VLM gates whether a session can pass,
the spec parser produces the tolerances the pre-check runs on. Cutting them removes function, not
polish. The note that the earlier cut list was inverted is retained as history.

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

**D-025 · Module layout, and AI code grouped by technology.** (2026-09-06)
`recon/` reconstruction, `service/` API and sessions, `inspect/` registration and verdicts,
`capture/` front end, `ai/` model clients and prompts, `common/` shared types, plus `contracts/`,
`fixtures/` and `scripts/`. Tests live inside their module: `<module>/tests/` for Python,
`*.test.ts` beside the source for TypeScript.
*Why:* thirteen A-prefix stories had no home and would have scattered prompts and API-key handling
across three modules. Grouping model calls by technology keeps prompts, keys, retries and
structured-output parsing in one place; the *policy* about when to call a model stays with the
module that owns the decision. `service/` is split from `recon/` so reconstruction can run from a
script or a test without starting a server, and the service can be tested without a GPU.
`common/` imports nothing local, which is what prevents an import cycle.

**D-026 · Unit tests ship in the same pull request as the code they cover.** (2026-09-06)
Applies to everything that reaches production, without exception for schedule.
*Why:* tests written afterwards are written to pass, not to find defects, and tests deferred to a
follow-up story are not written at all. Writing them alongside the logic is also the only point at
which the author still remembers the edge cases. The narrow exception is behaviour that cannot be
economically automated — device permissions, a gyro overlay on real hardware — which takes a written
manual check, and the story must state why automation was not possible.

---

## Open — not yet decided

**O-001 · Does the reconstruction backbone count toward the generative-AI requirement?**
D-018 states our position: it is a large learned model, deterministic, and we do not count it.

**Updated 2026-09-06.** Not raised with the instructor, and not escalated. D-018 states our position;
no ruling is needed unless the question is asked of us. Components are sized on merit rather than to
reach a share, so the answer no longer changes what we build. It was never listed in §16.

**O-002 · The final presentation date is inherited and unconfirmed.**
December 7 to 18 came from an earlier draft where it was explicitly provisional and anchored to
a different course's calendar. It has since hardened into the milestone table, the sprint plan
and the horizontal axis of both Gantt charts without ever being verified against SFBU's academic
calendar. Confirm it before the schedule is treated as fixed.

**O-003 · Exact biweekly review dates.** Planned for every second Friday from 11 September;
re-anchor if the real dates differ.

**O-004 · Whether a formal user study is expected this semester.** An informal one is planned;
a formal one with IRB approval is future work.
