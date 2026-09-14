# Contributing

For the project description, the standing rules, the story workflow and the definition of done,
read `AGENTS.md` first, then `stories/README.md`. This file adds the one rule that governs
changing the frozen interface contract, since that rule is more restrictive than the ordinary
one-approval process everything else in the repo uses.

## Changing `contracts/openapi.yaml`

`contracts/` is the shared surface all three lanes build against (`docs/sprint-plan.md` §4, §7).
Changing it after v1 is frozen requires:

- **Approval from both other team members**, not the usual single approval — which, together
  with the author, means all three have signed off.
- **Merging only at a sprint boundary**, never mid-sprint. Mid-sprint contract drift turns three
  independent lanes back into one tangled one (`docs/sprint-plan.md` §7).

This is a **written rule, enforced by review discipline, not by branch protection** — `main`
currently requires one approval repo-wide, and that is intentionally left alone rather than
raised for the whole repository just to cover one directory. See D-031 for why "all three
approvals" (as F-03 first stated it) and "two approvals" (as the sprint plan states it) are the
same rule, once GitHub's inability to count the author's own approval is accounted for.
