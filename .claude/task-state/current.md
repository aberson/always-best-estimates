# Task State

**Task:** always-best-estimates — post-Track-2 operator handoff (Track 2 fully built + soaked; only UAT + old V1 items remain)
**Status:** Track 2 COMPLETE incl. Step 29 soak (PASS, #35 closed). Launcher + automated soak harness shipped this window. Remaining work is OPERATOR-only.
**Last written:** 2026-07-15T17:59:00Z
**Session SHA:** 3312bd7

## Completed (this window, after the Track 2 build)
- One-click launcher `scripts/launch-abe.ps1` (#38 closed): starts backend (:8140) + Vite (:5174) + opens Chrome; idempotent. Surfaced as dev-observatory's `run` button via a `launch=` pointer in `dev/.claude/observatory/registry.toml`. Committed fb31c08 + docs ce3b7d8; pushed.
- Macro backfill (#18 data half): 6 FRED series populated (unlocks the `fracdiff_macro` feature set). Degraded-mode *check* still owed.
- Automated soak harness `scripts/soak.py` (committed 7f99f61): hands-off; drives on-demand load + samples DB/WAL/stuck-rows/RSS + writes a PASS/ATTENTION verdict.
- **Step 29 soak PASS (#35 closed):** 4h, 79 on-demand + 48 loop runs, 0 writer errors, 40/40 force=false cache hits, 0 stuck/error rows, WAL bounded at 0 MB, RSS steady ~73 MB. Findings `docs/soak/track2-soak-2026-07-11.md` committed 3312bd7; independently db-verified. Backend has since run ~2 more days / 1331 runs total, still 0 stuck/error.

## Next Action (OPERATOR — not agent-completable)
- **Step 30 UAT (#36):** compare + scenario UI acceptance — pick impls per stage, author a counterfactual, run a comparison, confirm the central scenario stays unambiguous, sanity-check the AGG floor (a `min_weight` config shows non-zero AGG). Agent-assist available: `/user-uat --ui` (mechanical + vision-judged tiers) or `/user-walkthrough` (attended). Launch first: `scripts/launch-abe.ps1`.
- **V1 leftovers:** Step 15 soak (#16, `Type: wait` — `scripts/soak.py` can drive it); M2 degraded-mode check (#18 — FRED key set + macro now backfilled).

## Critical Gotchas
- **dev repo is a parallel-session zone.** `dev/` is on branch `switchboard-offload-plan` (tip 1173d62, advanced past my registry commit 69b454a) with active worktrees (skill-iterate, switchboard-endpoint-launcher). This window left `dev.code-workspace` modified (observatory-sync regen, derived) + the registry launch-pointer commit 69b454a — DO NOT push the dev repo; leave it for that session.
- `data/abe.db` is schema v3; backup at `data/abe.db.pre-track2-backup`. The `min-var-demo` config (config_id 2) is the non-central config the soak harness drives — keep it.
- Backend/frontend/Chrome from the soak may still be running (:8140 / :5174).
- README / CLAUDE.md / plan.md "Remaining (operator)" lines still list the Step 29 soak as pending — now stale (#35 done). A `/repo-update` after the UAT should sweep them.

## Key Files
- `scripts/{launch-abe.ps1, soak.py, smoke.py}`; `docs/soak/track2-soak-2026-07-11.md`; `docs/track2-scenario-engine-plan.md` (Step 29 carries a `Harness:` line).
- Backend: `backend/abe/{config,registry,pipeline,scheduler,api}.py`, `blend/views.py`, `optimize/{mvu,min_variance}.py`.
