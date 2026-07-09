# Task State

**Task:** Track 2 planning → /plan-expedite (build-phase prep)
**Status:** READY for /build-phase — Steps 16–30 synced to issues #22–#36 (umbrella #21)
**Last written:** 2026-07-08T17:28:00Z
**Session SHA:** 7657d99

## Completed
- V1 build (Steps 1–14) + Track 1 transparency pass (commits df4b278..c63b33b) shipped + pushed;
  docs synced (7657d99); Track 1 issue #20 + M1 #17 closed.
- Track 2 feature plan authored: `docs/track2-scenario-engine-plan.md` (full arc, Steps 16–30,
  sub-phases 2A/2B/2C). Decisions: full-arc-in-one-plan; SQLite evolved w/ real migrations;
  central config on the 5-min loop, others on-demand + cached.
- /plan-expedite ran: plan-review READY (0 autofix) → plan-wrap READY (4 autofixes: shape tables,
  min-variance optimizer, comparison=derived, API/compare shapes) → repo-sync (umbrella #21 +
  15 step issues #22–#36; plan Issue: lines backfilled) → task-handoff (this).

## Current State
- Track 2 plan committed + pushed. GitHub: umbrella #21 open; step issues #22–#36 open.
  Still open from V1: #16 soak, #18 M2. #1 V1 umbrella.
- Server may still be running (scratchpad/uvicorn.pid) at 127.0.0.1:8140 on the V1+Track1 build.

## Next Action
Run `/build-phase --plan docs/track2-scenario-engine-plan.md` to build sub-phase 2A (Steps 16–20:
migration framework + schema v2 → Config/ViewScenario model → registries → Config-driven
run_pipeline w/ parity golden → smoke gate). Arm the Stop hook with the /goal over the
agent-completable span (Steps 16–28; stop before the Step 29 wait + Step 30 operator UAT).

## Critical Gotchas
- Step 19 (#25) is the risky refactor — the byte-identical **parity golden** (central Config ==
  pre-refactor run_stages) is the gating regression test; keep the `model` seam back-compat shim.
- Steps 22/23/24 (#28/#29/#30) are parallel-safe (blend/views vs optimize vs features).
- Steps 26–28 (#32–#34) are `--reviewers full --ui`; their `:8140` bind collides with a running
  dev server — free the port before their runtime review.
- Steps 29 (#35, Type: wait ≥4h soak) and 30 (#36, Type: operator UAT) are NOT agent-completable —
  /build-phase halts there; they are an operator handoff, not part of the automated /goal.
- SQLite kept (no Postgres); back up `data/abe.db` before the first real v1→v2 migrate (Step 16).

## Key Files
- `docs/track2-scenario-engine-plan.md` — the Track 2 plan (Steps 16–30, §5 data shapes, §7 steps).
- Umbrella issue #21; step issues #22–#36.
- memory `project-v2-scenario-harness-vision` — the operator decisions.
