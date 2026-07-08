# Task State

**Task:** /build-phase --plan plan.md (always-best-estimates V1, Steps 1-14)
**Status:** IN PROGRESS
**Last written:** 2026-07-08T00:30:00Z

## WIP
**Current:** Step 9: Smoke gate — one real end-to-end cycle (#10)
**Approach:** scripts/smoke.py — ~60s real end-to-end, NO mocks: boot app, trigger one real run against cached data (real data/abe.db), assert no exception, all run_stages ok, target_weights persisted; pytest smoke marker

## Next Action
/build-phase --plan plan.md --resume 3

## Completed
- [814954d] Step 1 Scaffold + constants + observatory registration: PASS iter 2/3 (17 tests; #2 closed)
- [0a4ea36] Step 2 SQLite storage module: PASS iter 2/3 (40 tests total; #3 closed)
- Step 3 Price ingest yfinance+cache: PASS iter 2/3 (67 tests total; #4 closed; real backfill in data/abe.db: SPY 8415/ACWI 4597/AGG 5728 rows; AGG 10y guard 1.37%>1%)
- Step 4 FRED macro ingest: PASS iter 2/3 (89 tests total; #5 closed; degraded mode live-verified exit 2; real fetch = keyed self-skip test, NO KEY on machine)
- Step 5 WorldModel + EWMA: PASS iter 2/3 (145 tests total; #6 closed; SIGMA = H-day PREDICTIVE forecast std — decision recorded in plan Step 5 Status; contract fn frozen in tests/test_model_base.py)
- Step 6 Blend cov+confidence+BL: PASS iter 1+orch fixes (204 tests total; #7 closed; Idzorek Table-6 golden pins; confidence from RAW H-day pair; rf must be exactly 0.0)
- Step 7 cvxpy MVU optimizer: PASS iter 1+orch fixes (244 tests total; #8 closed; γ_tc=0.002 band anchored both directions; MVUResult(weights,prev_weights,turnover,relaxed_turnover,status))
- Step 8 Pipeline+API+ledger: PASS iter 2/3 (274 tests total; #9; dual-watermark freshness gate; two-phase txn; V1 sync trigger blocks loop — Step 11 swaps to executor + owns stale-running sweep)

## Dead Ends
(none yet)

## Critical Gotchas
- Goal armed: Steps 1-14 DONE + issues #2-#15 closed + pytest/mypy/ruff green; STOP before Step 15 soak (#16) and M1/M2 (#17-#18)
- Baseline test count: 40 (after Step 2)
- mypy STRICT; new modules fully annotated
- storage.coerce_scalar REJECTS NaN (ValueError) — missing values must be explicit None; macro ingest (Step 4) converts NaN parses to None
- storage API: open_writer / open_read_only / insert_row / upsert_row / latest_ok_run_id / wal_checkpoint_truncate (must run on connection-owning thread)
- .item() coercion only for 0-dim (ndim==0); 1-element arrays rejected TypeError
- pytest -m smoke exits 5 until Step 9 adds a marked test

## Key Files
- `plan.md`: §3 schema spec; §12 constants; per-step Done-when
- `backend/abe/storage.py`: PRAGMAs+DDL+coercion boundary; asset writes validated against UNIVERSE
- `pyproject.toml`: hatchling packages=["backend/abe"]; mypy strict, mypy_path=backend
