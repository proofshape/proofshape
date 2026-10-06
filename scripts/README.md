# scripts

One-command helpers. Each is runnable on its own; none is part of the shipped system.

| Script | What it does |
|---|---|
| `check_story_states.py` | Reads the dependency graph in `stories/`, flags any story whose file and index disagree, and reports (or with `--fix`, corrects) stories stuck `Blocked` when every dependency is `Done`. Run after any merge that sets a story to `Done` — see `AGENTS.md`. |
| `build_review_deck.js` | Builds the biweekly progress-review deck as a `.pptx` into `submissions/`. See the header comment for why its `package.json` lives here rather than at the repository root. |
| `smoke_r14.py` | Runs real golden frames through the reconstruction HTTP flow: in-process (R-14), or against a deployed URL with `--base-url` (R-15). |
| `deploy_recon.py` | Starts or stops the deployed reconstruction service on Lightning. Run by `.github/workflows/deploy-recon.yml` on `main` only; see `recon/README.md`, R-15. |
| `container_entrypoint.sh` | **The one exception to "not shipped":** the R-15 image's entrypoint. It fetches the model weights, then serves `recon.service`. |

## Node helpers

```
cd scripts && npm install
node build_review_deck.js ../submissions/YYYY-MM-DD-progress-review-deck.pptx
```

**Do not move `package.json` to the repository root.** `.github/workflows/typescript-ci.yml`
detects a TypeScript project with `[ -f package.json ]` from the workspace root, so a root
manifest switches on its `tsc`/eslint job — which then fails, because there is no `tsconfig.json`
and no eslint config. Whoever starts C-01 needs to add both in the *same* pull request as the
root manifest.
