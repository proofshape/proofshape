# ProofShape

A web application that turns a smartphone into a coarse 3D inspection tool for manufactured
parts. A supplier places a part on a printed reference sheet, photographs it with on-screen
guidance, and gets back a metric 3D model in the browser. If the buyer has uploaded a CAD file,
the system aligns the model to it and returns a **Pass / Fail / Rescan / Unverifiable** report
with a deviation heatmap, so gross defects are caught before the part ships.

Coarse geometric inspection, several millimetres on hand-sized parts. It complements CMM
inspection; it does not replace it.

**CS595 capstone · Fall 2026 · San Francisco Bay University**

https://github.com/proofshape/proofshape

---

## Start here

| If you want to | Read |
|---|---|
| Understand what we are building | [the proposal](submissions/2026-09-05-proposal.md) |
| Work on the project with an AI assistant | **[AGENTS.md](AGENTS.md)** — read this first |
| Pick up a task | [the story index](stories/README.md) |
| Know why something is the way it is | [docs/decisions.md](docs/decisions.md) |
| Know what a term means | [docs/glossary.md](docs/glossary.md) |
| See the schedule | [docs/sprint-plan.md](docs/sprint-plan.md) |
| See what has happened so far | [docs/progress-log.md](docs/progress-log.md) |

## Layout

```
AGENTS.md          instructions for every AI assistant — the rules that govern how we work
CLAUDE.md          Claude-specific notes; points to AGENTS.md for everything else

docs/              internal working documents
  decisions.md       settled decisions, D-001 onwards; do not relitigate
  glossary.md        project vocabulary
  sprint-plan.md     schedule, capacity, lanes, reviews, repository process
  progress-log.md    running journal, newest at the bottom
  story-template.md  template for a new story
  archive/           superseded drafts, kept for their reasoning

stories/           one file per story; README.md is the index
submissions/       what was handed to the course, dated

recon/             reconstruction engine (backend)
capture/           progressive web app (front end)
inspect/           CAD registration, verdict logic, ground truth
contracts/         the frozen interface between capture and recon
fixtures/          download script and checksums only — never the images
scripts/           one-command helpers
```

`recon/`, `capture/` and `inspect/` are **places, not people**. Any team member picks up any
story in any of them.

## Working agreements, in brief

- Every change reaches `main` through a pull request approved by one other member.
- One person per story, claimed visibly before work starts.
- Tests alongside the code, not batched before or after it.
- Never commit fixtures, model weights, or secrets. This repository is public.

Full detail in [AGENTS.md](AGENTS.md) and [docs/sprint-plan.md](docs/sprint-plan.md).

## Licence

MIT covers **this repository's own code only**. Third-party components, their licences, and
confirmation that we redistribute no model weights and vendor no third-party code, are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Note that the reconstruction model weights carry **non-commercial research licences**. That
constrains commercial reuse of a deployed system, not publication of this repository or the
coursework.
