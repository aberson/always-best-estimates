# Task State

**Task:** Track 1 (transparency pass) for always-best-estimates + repo-update
**Status:** COMPLETE (pushed to origin/master)
**Last written:** 2026-07-08T22:08:47Z

## Completed
- V1 build (Steps 1-14) shipped 2026-07-08 (was HEAD 92ae37d).
- Track 1 transparency pass — 5 commits df4b278..c63b33b:
  - backend: calc.py relocation (basic.py + confidence.py deleted) + EXPLANATIONS + /api/explain
    + additive stage-detail enrichments (blend prior/view/covariance_window, ingest price_provider,
    optimize objective, features windows). Built via build-step (4-reviewer gauntlet).
  - frontend: render enrichments + inline "how is this computed?" expander; freshness folded into
    the top bar (5 cards); optimize consolidated; BL blend 3-component view; covariance note.
  - tweaks reviewed by a 2-agent adversarial pass (both findings fixed).
- repo-update: README + plan.md §14 + CLAUDE.md refreshed; drift check clean; memory updated;
  record issue filed + closed; pushed.
- 412 tests, mypy strict clean, ruff clean, real smoke green.

## Current State
- master pushed to origin. Server running (scratchpad/uvicorn.pid) at 127.0.0.1:8140, macro MACRO_OK.
- FRED key in gitignored .env (32-char, validated); .env.example is the empty placeholder.
- Open issues: #1 umbrella, #16 soak (Type: wait), #18 M2. (#17 M1 closed — accepted.)

## Next Action
Operator's choice: (a) Track 2 — pluggable-per-stage scenario/compare engine + run-harness/DB;
needs /plan-feature (decisions in memory project-v2-scenario-harness-vision); (b) M2 macro backfill
now that the FRED key works (`uv run pytest -m network` then `uv run python -m abe.ingest.macro
--backfill`); (c) the >=4h soak (#16).

## Critical Gotchas
- calc.py is the single home for the simple calcs; basic.py + confidence.py were DELETED.
- pytest default deselects smoke/network/realdb (intentional); 412 is the baseline.
- AGG=0% is unchanged by Track 1 (display-only); it's a Track 2 concern (crude EWMA + no floor).

## Key Files
- `backend/abe/calc.py`, `backend/abe/pipeline.py` (stage-detail enrichments), `backend/abe/api.py`.
- `frontend/src/components/StageCard.tsx` (card rendering + expander + hiddenExtra).
- `plan.md` §14 (Track 1 record + Track 2 scope); memory project-v2-scenario-harness-vision.
