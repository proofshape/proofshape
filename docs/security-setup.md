# Security setup — human checklist

Settings that cannot be configured from a script or by an assistant. Someone with admin rights on
the repository has to click these. All are free on public repositories.

Tick them off and note the date, so the next person does not re-check blindly.

## GitHub settings

- [ ] **Secret scanning: on.** Settings → Code security → Secret scanning. Detects committed
      credentials in the repository and its history.
- [ ] **Push protection: on.** Same page. This is the one that matters most — it blocks a
      credential *before* it reaches the remote, rather than telling you afterwards that it is
      permanently in the history.
- [ ] **Dependabot alerts: on.** Settings → Code security. Flags vulnerable dependencies.
- [ ] **Dependabot security updates: on.** Opens the fix as a pull request, which still goes
      through review like anything else.
- [ ] **Branch protection matches D-013.** Settings → Branches, or read it back with
      `gh api repos/proofshape/proofshape/branches/main/protection`. All eight settings, and
      confirm "do not allow bypassing" is still on.

## Verify, do not assume

Branch protection was applied and both gates tested when the repository was created: a merge
without approval was refused, and a direct push to `main` was rejected. Re-test after any settings
change, because a rule that is configured but not tested is a rule you are guessing about.

## If a credential is ever committed

1. **Revoke it first.** Rotating the secret is the fix; removing it from git is cleanup.
2. Then remove it from history and force-push, which needs branch protection temporarily relaxed.
3. Record what happened in `docs/progress-log.md`. A near-miss is worth more written down than a
   success.

Assume anything ever pushed to a public repository has been seen, even if it was removed within
minutes.

## Scan status

- **2026-09-06** — full-history scan for credential patterns (`api_key`, `secret`, `token`, `AIza`,
  `sk-`, `ghp_`, `gho_`, private keys). **No credentials found.** All matches were prose about
  handling secrets. gitleaks was not installed; the scan used pattern matching over
  `git log -p --all`. Installing gitleaks would give a stronger check.
