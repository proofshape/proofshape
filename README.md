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
| **Work on this project at all, human or AI** | **[AGENTS_START_HERE.md](AGENTS_START_HERE.md)** — read this first, every session |
| Understand what we are building | [the proposal](submissions/2026-09-05-proposal.md) |
| Actually use the system, or print the reference sheet | [docs/manual.md](docs/manual.md) |
| Work on the project with an AI assistant | [AGENTS.md](AGENTS.md) |
| Pick up a task | [the story index](stories/README.md) |
| Know why something is the way it is | [docs/decisions.md](docs/decisions.md) |
| Know what a term means | [docs/glossary.md](docs/glossary.md) |
| See the schedule | [docs/sprint-plan.md](docs/sprint-plan.md) |
| See what has happened so far | [docs/progress-log.md](docs/progress-log.md) |

## Layout

```
AGENTS_START_HERE.md  read this first, every session — the step-by-step order of operations
AGENTS.md             instructions for every AI assistant — the rules that govern how we work
CLAUDE.md             Claude-specific notes; points to AGENTS.md for everything else

docs/                 internal working documents
  decisions.md          settled decisions, D-001 onwards; do not relitigate
  glossary.md           project vocabulary
  sprint-plan.md        schedule, capacity, lanes, reviews, repository process
  progress-log.md       running journal, newest at the bottom
  story-template.md     template for a new story
  archive/              superseded drafts, kept for their reasoning

stories/              one file per story; README.md is the index
submissions/          what was handed to the course, dated

recon/                reconstruction engine (backend)
capture/              progressive web app (front end)
inspect/              CAD registration, verdict logic, ground truth
contracts/            the frozen interface between capture and recon
fixtures/             download script and checksums only — never the images
scripts/              one-command helpers
```

`recon/`, `capture/` and `inspect/` are **places, not people**. Any team member picks up any
story in any of them.

## Running each part

This is the repository skeleton (F-02) — none of the lanes have working code yet. What exists
today:

| Part | How to run it |
|---|---|
| Python project | `./scripts/bootstrap_dev_env.sh` from the repository root, one command. Creates `.venv` with Python 3.11+, installs `recon` in editable mode plus every pinned dependency in `requirements.txt` (F-07). |
| `recon/` | Nothing runnable yet. First piece lands with R-01. |
| `capture/` | No npm project yet. Lands with the capture stories, starting C-01. |
| `inspect/` | No runnable code yet, and not registered as a Python package — see `inspect/README.md`. Lands with the I-0x stories. |
| `contracts/` | `openapi.yaml`, frozen at v1 (F-03). Changing it needs `CONTRIBUTING.md`'s approval rule. |
| `fixtures/` | Empty until F-06 adds the download script. |
| `scripts/` | `bootstrap_dev_env.sh` (F-07). Otherwise empty until a story needs a one-command helper. |

Each directory has its own `README.md` with more detail. The bootstrap script works the same way
on Apple Silicon and on the shared GPU environment (F-05) — same command, same pinned versions.

## Working agreements, in brief

- Every change reaches `main` through a pull request approved by one other member.
- One person per story, claimed visibly before work starts: the claimer's GitHub handle goes
  on the story file and on its row in the [backlog index](stories/README.md), even when an AI
  assistant makes the claim for them.
- Tests alongside the code, not batched before or after it.
- Never commit fixtures, model weights, or secrets. This repository is public.

Full detail in [AGENTS.md](AGENTS.md) and [docs/sprint-plan.md](docs/sprint-plan.md).

## Licence

Our own code is MIT (see `LICENSE`). Note separately that the reconstruction model weights we
build on carry **non-commercial research licences**. That constrains commercial reuse of a
deployed system, not publication of this repository or the coursework.
