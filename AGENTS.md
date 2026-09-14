# ProofShape — instructions for AI assistants

Read this before doing any work on this project. It applies to every assistant the team uses,
Claude and ChatGPT alike. The point is that all three of us get the same behaviour out of our
tools, and that the tools do not quietly undo decisions we have already made.

---

## What ProofShape is

A web application that turns a smartphone into a coarse 3D inspection tool for manufactured
parts. A supplier places a part on a printed reference sheet, photographs it with guidance,
and gets back a metric 3D model in the browser. If the buyer has uploaded a CAD file, the
system aligns the model to it and returns a **Pass / Fail / Rescan / Unverifiable** report
with a deviation heatmap, so gross defects are caught before the part ships.

CS595 capstone, Fall 2026, three team members, roughly 349 person-hours total.

**Delivery tiers.** Tier 1 is the core: photos in, metric 3D model out. Tier 2 is inspection:
CAD alignment, heatmap, verdict. Tier 3 is stretch. Tier 1 is never cut.

---

## Standing rules — enforce in every session

### The hard rule
**Geometry that was not observed can never pass or fail a part.** It may only trigger a Rescan.
A model's guess must never approve or condemn a component. This governs the verdict logic, the
provenance labelling, and how the heatmap is coloured. Do not weaken it for convenience.

### Accuracy honesty
This is coarse geometric inspection, several millimetres on hand-sized parts. It complements
CMM inspection and does not replace it. **Never write or imply that the system achieves
metrology-grade accuracy, and never state a tolerance the calibration study has not measured.**

### No promised numbers
Every figure in the planning documents is an estimate until it is measured. Accuracy, latency,
noise floor, hours: all provisional. **Do not invent a measured value, and do not restate a
planning estimate as a result.** If a number is needed and does not exist yet, say it does not
exist yet and name the milestone that will produce it.

### The generative-AI requirement
The course requires roughly one third of the project to be generative AI. Five components
carry that: the capture agent, generative shape completion, VLM capture supervision, the
prose-to-spec parser, and the report narrative. **These are never cut**, and they carry the
same protection as Tier 1. If a schedule conversation proposes dropping one, flag it rather
than agreeing.

The test for whether something counts: *if you run it twice, can it give two different answers?*
The reconstruction backbone cannot, so it is described as a large learned model, never as
generative.

### Do not relitigate settled decisions
`docs/decisions.md` records what has been decided and why. Do not reopen or quietly reverse
D-001 onwards. If you believe one needs revisiting, say so explicitly and stop there — do not
build past it in either direction.

---

## Before starting any story

1. **Read the story file in full**, in `stories/` — the index is `stories/README.md`. Then read the stories it depends on, and any
   story that depends on it. Confirm compatibility before writing code.
2. **Check `docs/decisions.md`** for anything that governs this area.
3. **Check for conflicts** with stories currently in progress, and with existing code touching
   the same modules, endpoints or data.
4. **Plan before building** for anything non-trivial. Explore the real code first, surface any
   genuine design fork to the team, agree the approach, then implement. Do not go straight to
   code on a story you have not read the surrounding context for.

If the story's acceptance criteria are ambiguous, resolve that first. An ambiguous story is not
Ready, whatever the index says.

---

## While building

- **Tests alongside the code — during implementation, not after the story is "done."** Not
  written first as a batch, not bolted on afterwards as a batch, and never deferred to a
  follow-up story — add the test for each piece of logic as that logic is written. **This is
  the joint responsibility of the engineer and whatever AI assistant they're working with: the
  assistant does not get to skip it because it wasn't asked for that turn, and the engineer does
  not get to skip it because the assistant didn't offer.** Tests written after the fact are
  written to pass, not to find defects — they document the path you built, not the paths you
  missed — and a "we'll add tests later" story almost never gets that later. The narrow
  exception is behaviour that genuinely cannot be economically automated (a device permission
  prompt, a sensor overlay on real hardware); that takes a written manual check instead, with
  the story stating *why* automation wasn't possible. "It was quicker" is not a reason.
- **Verbose inline comments that explain *why*, not what.** Someone returning to this in
  November needs the reasoning, not a restatement of the line below.
- **Progressive logic.** Build up incrementally before abstracting. Explicit loops are
  preferred over comprehensions or lambdas wherever clarity matters more than brevity.
- **No silent failures.** An explicit error message always beats a bare `except: pass` or a
  swallowed return. In an inspection system, a quiet wrong answer is worse than a loud refusal.
- **Meaningful names.** `inlier_ratio`, `sigma_floor`, `observed_fraction` — not `score`,
  `thresh`, `val`.
- **One story is one pull request.** If the change is growing past one or two work sessions,
  stop and split it.

### Never commit
- **Photographs or fixture sets.** The golden capture lives outside git behind a download
  script. Git never forgets: one large file bloats the repository permanently.
- **Model weights.** Downloaded at setup, never committed.
- **Any key, token or credential.** The repository is public. Secrets go in Actions secrets
  and a gitignored local env file.

---

## Definition of done

Every story, without exception:

- Merged to `main` through a pull request.
- Reviewed and approved by one other team member.
- Runs on a second member's machine from a clean checkout.
- **Unit tests for this story's logic, written while the logic was written and passing in the
  same pull request** — or, only where automation was genuinely impossible, a written manual
  check with the reason stated. Not "one or the other, whichever's easier": the manual check is
  the fallback for what can't be automated, not an alternative to trying.
- Demonstrated at the sprint boundary.
- **Completion recorded on the story file**: who, the date, the pull request, and actual hours
  against the estimate.
- **The moment a pull request merges, whoever merges it sets `State: Done` on the story file
  and updates the matching row in `stories/README.md` to `Done`, in the same action.** Not a
  follow-up, not the next person's problem — the merge and the state change happen together.
  A story left `In review` after merging is exactly what makes the index untrustworthy, which
  is what then leads someone to work a story that is actually already blocked or already done.

Acceptance criteria are per story. The definition of done is not — it is the same every time.

Every pull request carries this list as a checklist (`.github/PULL_REQUEST_TEMPLATE.md`). Tick
each item before merging — the state/index item specifically has been missed twice already.

---

## Repo map

```
README.md              — repository front door
AGENTS.md              — this file; instructions for every AI assistant
CLAUDE.md              — Claude-specific notes; points here for everything else

docs/
  decisions.md           — settled decisions, D-001 onwards; do not relitigate
  glossary.md            — project vocabulary
  sprint-plan.md         — schedule, capacity, lanes, reviews, repository process
  progress-log.md        — running chronological journal, newest at the bottom
  story-template.md      — template for a new story file
  archive/               — superseded drafts, kept for their reasoning

stories/               — one file per story; README.md is the index
submissions/           — what was handed to the course, dated

recon/                 — reconstruction engine (backend)
capture/               — progressive web app (front end)
inspect/               — CAD registration, verdict logic, ground truth
contracts/openapi.yaml — the frozen interface between capture and recon
fixtures/              — download script and checksums only; never the images
```

**Lanes are places, not people.** Those three directories are code locations. Any team member
picks up any story in any of them.

---

## Working with the team

- **One person per story.** If you are asked to work on a story someone else has claimed, say
  so rather than proceeding.
- **Record decisions as they are made.** If a conversation settles something that will govern
  later work, add it to `docs/decisions.md` with a number and the reasoning. A decision without
  its reasoning gets reversed by whoever forgets it.
- **Append to the progress log** after any story is completed, any pull request is reviewed, or
  any measurement run is done. Newest entry at the bottom, never delete an old one. This is what
  makes the biweekly review slides almost write themselves.
- **Superseded instructions are marked, not deleted.** Note what replaced them and why; the
  original reasoning usually still matters.

---

## Things to refuse or flag rather than do

- Weakening the hard rule about unobserved geometry.
- Cutting a generative-AI component to save schedule.
- Writing a specific accuracy, latency or tolerance figure that has not been measured.
- Committing fixtures, weights or secrets.
- Opening a pull request whose code has no tests, or agreeing to "add tests in a follow-up."
- Reversing a decision in `docs/decisions.md` without the team agreeing first.
- Expanding scope beyond the current tier because it "would be easy to add."
