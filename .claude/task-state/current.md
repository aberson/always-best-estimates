# Task State

**Task:** /build-phase --plan plan.md (always-best-estimates V1, Steps 1-14)
**Status:** IN PROGRESS
**Last written:** 2026-07-07T23:40:00Z

## WIP
**Current:** Step 2: SQLite storage module (#3)
**Approach:** storage.py — connection + PRAGMAs (WAL/synchronous=NORMAL/busy_timeout/foreign_keys), full schema DDL (plan §3), numpy/torch scalar coercion insert boundary, one-writer connection

## Next Action
/build-phase --plan plan.md --resume 2

## Completed
- Step 1 Scaffold + constants + observatory registration: PASS iter 2/3 (17 tests; issue #2)

## Dead Ends
(none yet)

## Critical Gotchas
- Goal armed: Steps 1-14 DONE + issues #2-#15 closed + pytest/mypy/ruff green; STOP before Step 15 soak (#16) and M1/M2 (#17-#18)
- Baseline test count: 17 (after Step 1)
- mypy is STRICT mode via pyproject; new modules must be fully annotated
- pytest -m smoke exits 5 until Step 9 adds a marked test — do not wire smoke gates before Step 9
- Observatory registration already done (registry.toml has always-best-estimates, owned=true)

## Key Files
- `plan.md`: 15 automated steps + M1/M2 manual; §12 Appendix has canonical constants; §3 schema DDL spec
- `pyproject.toml`: hatchling packages=["backend/abe"] → import name `abe`; mypy strict, mypy_path=backend
