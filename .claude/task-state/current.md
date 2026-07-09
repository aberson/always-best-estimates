# Task State

**Task:** Track 2 (scenario/compare engine) — /build-phase over docs/track2-scenario-engine-plan.md
**Status:** COMPLETE — automated Steps 16–28 (sub-phases 2A/2B/2C) all shipped; operator handoff for Steps 29–30
**Last written:** 2026-07-09T02:05:00Z
**Session SHA:** 6bd6c62

## Completed
- All 13 automated steps 16–28 DONE + committed (11 checkpoint commits 10ef099..6bd6c62);
  issues #22–#34 all CLOSED. Goal gates green: `uv run pytest` 497 passed, `uv run mypy backend`
  clean (32 files), `uv run ruff check .` clean.
- 2A foundation: migration framework + schema v2/v3 (configs/view_scenarios/runs.config_id +
  configs.updated_at_utc), Config/ViewScenario CRUD, stage registries, Config-driven run_pipeline
  (byte-identical parity golden), config-pipeline smoke gate.
- 2B pluggable stages: on-demand config runs (cached/tagged, single-writer), view sources
  (forecast/historical/counterfactual + seeded library), optimizer variants (mvu min_weight floor +
  min_variance), selectable feature sets (fracdiff_macro).
- 2C UI + API: config/scenario/compare API (one-writer via scheduler.run_write), react-router SPA
  (HashRouter) — DashboardView, StageDetailTab, CompareView, ScenarioEditor. `npm run build` clean.
- Real db migrated v1→v3 (backup at data/abe.db.pre-track2-backup); runtime-verified end-to-end on
  :8140 (min-variance holds AGG=0.60; delete guard 409; library browsable).

## Next Action (OPERATOR — not agent-completable; the /build-phase goal STOPPED here by design)
Two operator/wait steps remain (out of the automated goal scope):
- **Step 29 (#35, Type: wait):** >=4h soak of the always-on loop + on-demand comparisons; capture
  `docs/soak/track2-soak-<date>.md`. Resume a fresh session after the wait.
- **Step 30 (#36, Type: operator):** UAT of the compare + scenario UI (pick impls per stage, author a
  counterfactual, run a comparison, confirm central stays unambiguous, sanity-check the AGG floor).
Run `/repo-update` when ready to update README/docs + push the Track 2 work.

## Critical Gotchas
- data/abe.db is now schema v3; backup at data/abe.db.pre-track2-backup (delete once satisfied).
- A stray `min-var-demo` config (config_id 2) + run 104 exist on the real db from runtime verification
  (a valid min-variance config; delete its run then the config if unwanted).
- Steps 26–28 landed as ONE cohesive frontend commit (router SPA must build atomically).

## Key Files
- Plan: docs/track2-scenario-engine-plan.md (Steps 16–30; 16–28 Status: DONE).
- backend/abe/{migrations,config,registry,pipeline,scheduler,api}.py; backend/abe/blend/views.py;
  backend/abe/optimize/{mvu,min_variance}.py; frontend/src/components/*.
