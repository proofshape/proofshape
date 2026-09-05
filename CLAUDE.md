# ProofShape — CLAUDE.md

**Read `AGENTS.md` first.** It holds the project description, the standing rules, the workflow
before and during a story, the definition of done, and the repo map. Everything there applies
to Claude. This file only adds what is specific to working with Claude on this project.

---

## Claude-specific notes

**Plan before building.** For any story beyond a trivial change, use plan mode: explore the real
code, read the story and its dependencies, surface genuine design forks to the team, get the
plan agreed, then implement. Do not go straight from a story title to code.

**Model choice.** Use the stronger reasoning model for planning, exploration and review; a
faster model is fine once a plan is agreed and the work is mechanical implementation. Claude
cannot switch its own model — say so at the transition point rather than continuing on the
wrong one.

**Memory and this repository disagree sometimes.** If something recalled from a previous session
conflicts with `docs/decisions.md` or a story file, the repository wins. Say that the conflict
exists rather than silently picking one.

**Verify before asserting.** This project has a lot of numbers that look settled but are not.
Before stating an hour figure, a date, a tolerance or a percentage, read it from the current
file rather than recalling it. The planning documents have been revised many times.

---

## Quick reference

| Need | File |
|---|---|
| What we are building, and the rules | `AGENTS.md`, then `submissions/` for the proposal |
| Why something is the way it is | `docs/decisions.md` |
| What a term means | `docs/glossary.md` |
| What to work on | `stories/README.md`, then `stories/<ID>.md` |
| Schedule, capacity, process | `docs/sprint-plan.md` |
| What happened already | `docs/progress-log.md` |
