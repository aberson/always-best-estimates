# Task State

**Task:** /build-phase --plan plan.md (always-best-estimates V1, Steps 1-14)
**Status:** IN PROGRESS
**Last written:** 2026-07-08T00:30:00Z

## WIP
**Current:** Step 3: Price ingest — yfinance adapter + cache (#4)
**Approach:** SourceAdapter protocol; YFinanceAdapter (auto_adjust=True, multi_level_index=False, progress=False + column-set assertion); CacheAdapter; incremental upsert with 429 backoff; backfill entrypoint

## Next Action
/build-phase --plan plan.md --resume 3

## Completed
- [814954d] Step 1 Scaffold + constants + observatory registration: PASS iter 2/3 (17 tests; #2 closed)
- Step 2 SQLite storage module: PASS iter 2/3 (40 tests total; #3)

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
