"""Smoke gate: one REAL end-to-end pipeline cycle against cached data — NO mocks.

Run it: ``uv run python scripts/smoke.py [--db path]`` (plan.md Step 9; the
``uv run pytest -m smoke`` gate in CLAUDE.md section 3 drives the same
:func:`run_smoke` core via ``tests/test_smoke.py``).

What this gate proves — and deliberately nothing more
=====================================================

One real cycle through the production stack completes without crashing:
``POST /api/runs/trigger {"force": true}`` -> all six ``pipeline.STAGES``
land ``status='ok'`` rows -> ``target_weights`` persisted for the whole
``UNIVERSE`` and summing to 1 -> ``/api/runs/latest`` serves the run.
Business-logic QUALITY is out of scope: the smoke does not judge whether the
weights are sensible, only that the producer->consumer chain holds together
end-to-end. Its purpose is surfacing the drift class that mocked unit tests
cannot see (code-quality rule: tests with mocks can't see producer-consumer
drift).

Stop the production app first — the smoke opens its own writer connection,
and a concurrent scheduler run can otherwise contend for the write lock or
land a newer run between our trigger and the latest-run check.

Why in-process (TestClient) instead of a uvicorn subprocess
===========================================================

The app is booted IN-PROCESS via ``fastapi.testclient.TestClient`` over
``abe.api.create_app(db_path)`` **with the lifespan context entered** — this
exercises the production startup path (the FRED macro key-probe, THE writer
connection) plus the production route handlers, exactly the code uvicorn
would serve. In-process is deterministic (no port juggling, no readiness
polling, no orphaned child process) and cross-platform (Windows subprocess
signal handling never enters the picture), while still running the production
factory + routes end-to-end. A subprocess would only additionally test
uvicorn itself, which is not our code.

Network use for prices — asserted two ways:

- **Provenance label:** the ingest card's ``source`` must be ``'cache'``
  (``CacheAdapter``'s label; a network price path stamps ``'yfinance'``).
  Self-reported by the code path it checks, so alone it is circumstantial.
- **Structural:** after the run, ``yfinance`` must be ABSENT from
  ``sys.modules`` — it is imported only inside ``YFinanceAdapter.fetch``, so
  its presence proves a network price fetch executed. Likewise ``fredapi``
  when the run recorded the no-key degraded mode (the probe returns early
  without touching the lazy import). Either module already being loaded
  BEFORE the smoke boots makes its half of the structural check inconclusive,
  so that half is skipped (the label check still applies).

The startup macro probe makes one FRED request iff a ``FRED_API_KEY`` is
configured (that IS the production startup path); with no key it makes no
request and the run proceeds in the explicit macro-disabled degraded mode.

Exit codes
==========

- ``0`` — smoke PASS.
- ``1`` — smoke FAILURE: an assertion failed, a stage errored (the failing
  stage's ``detail_json`` and the run's ``error_text`` are printed), the
  watchdog timed out (``SMOKE TIMEOUT``), or the smoke itself crashed
  (``SMOKE INTERNAL ERROR`` + traceback).
- ``3`` — PRECONDITION missing (no db / no price rows for some asset): the
  operator must run the one-time backfill first. Distinct from failure so a
  fresh clone is not mistaken for a broken pipeline.

Runtime budget: one pipeline run against a warm cache is seconds. The
watchdog is a hard one: the whole booted cycle (lifespan enter -> trigger ->
verify) runs on a worker thread that the main thread joins with a
:data:`WATCHDOG_SECONDS` timeout — a hung solver or a stalled keyed FRED
probe at lifespan-enter therefore still produces a classified
``SMOKE TIMEOUT`` + exit 1 instead of blocking forever. On timeout the worker
is a daemon thread and is simply abandoned (the process is about to exit
anyway). ``monotonic()`` bookkeeping is kept for the elapsed report.
"""

import argparse
import json
import logging
import math
import sqlite3
import sys
import threading
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Module-level binding (not ``import time``) so tests can monkeypatch the
# smoke's clock without touching the shared ``time`` module.
from time import monotonic
from typing import Any, Final

from fastapi.testclient import TestClient

from abe import storage
from abe.api import create_app
from abe.constants import UNIVERSE
from abe.ingest.macro import MACRO_DISABLED_NO_KEY
from abe.pipeline import STAGES

__all__ = [
    "DEFAULT_DB_PATH",
    "EXIT_FAILURE",
    "EXIT_OK",
    "EXIT_PRECONDITION",
    "WATCHDOG_SECONDS",
    "SmokeFailureError",
    "SmokePreconditionError",
    "SmokeReport",
    "SmokeTimeoutError",
    "StageCard",
    "main",
    "run_smoke",
]

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
"""Resolved from THIS FILE's location (scripts/ is one level below the root),
never from the process cwd — the smoke must find the same db no matter where
it is launched from."""

DEFAULT_DB_PATH: Final[Path] = PROJECT_ROOT / storage.DEFAULT_DB_PATH
"""The REAL production db (``<project root>/data/abe.db``)."""

WATCHDOG_SECONDS: Final[float] = 120.0
"""Hard wall-clock budget for the whole booted cycle (~60s target, 2x headroom)."""

EXIT_OK: Final[int] = 0
"""Smoke PASS: one real cycle completed and every assertion held."""

EXIT_FAILURE: Final[int] = 1
"""Smoke FAILURE: assertion failed / stage errored / watchdog timeout /
unclassified internal error."""

EXIT_PRECONDITION: Final[int] = 3
"""Precondition missing (no db / no backfill) — distinct from failure so a
fresh clone reads as "backfill first", not "pipeline broken" (mirrors
macro.py's ``EXIT_MACRO_DISABLED`` degraded-vs-crashed distinction)."""

_NETWORK_MODULES: Final[tuple[str, ...]] = ("yfinance", "fredapi")
"""Lazily-imported network-client modules the structural check watches."""


class SmokePreconditionError(RuntimeError):
    """A prerequisite is missing (no db / no backfill) — CLI exit 3.

    Deliberately distinct from :class:`SmokeFailureError`: a fresh clone that
    never ran the backfill is not a broken pipeline.
    """


class SmokeFailureError(RuntimeError):
    """The smoke itself failed (assertion / stage error) — CLI exit 1."""


class SmokeTimeoutError(SmokeFailureError):
    """The watchdog fired: the booted cycle exceeded the wall-clock budget.

    A subclass of :class:`SmokeFailureError` so every caller's exit-code
    contract (1) is preserved; the CLI prints it as its own classification.
    """


@dataclass(frozen=True)
class StageCard:
    """One-line summary of one ``run_stages`` row."""

    stage: str
    status: str
    duration_s: float | None

    def line(self) -> str:
        duration = "duration n/a" if self.duration_s is None else f"{self.duration_s:.0f}s"
        return f"  [{self.status}] {self.stage:<10} ({duration})"


@dataclass(frozen=True)
class SmokeReport:
    """What a PASSING smoke observed (failures raise instead)."""

    db_path: Path
    run_id: int
    cards: tuple[StageCard, ...]
    weights: dict[str, float]
    weights_sum: float
    ingest_source: str
    elapsed_s: float


def _check_preconditions(db_path: Path) -> None:
    """Exit-3 gate: the db must exist and hold price rows for EVERY asset.

    Per-asset, not just total: a partially-backfilled db would otherwise
    surface as a confusing downstream stage failure instead of the clear
    "backfill first" message.
    """
    backfill_hint = (
        "run the one-time backfill first:\n"
        "  uv run python -m abe.ingest.prices --backfill\n"
        "  uv run python -m abe.ingest.macro --backfill   (optional; needs FRED_API_KEY)"
    )
    if not db_path.is_file():
        raise SmokePreconditionError(f"db not found at {db_path} - {backfill_hint}")
    try:
        conn = storage.open_read_only(db_path)
    except sqlite3.Error as exc:
        raise SmokePreconditionError(
            f"db at {db_path} is not openable ({exc}) - {backfill_hint}"
        ) from exc
    try:
        try:
            rows = conn.execute("SELECT asset, COUNT(*) FROM prices GROUP BY asset").fetchall()
        except sqlite3.Error as exc:
            raise SmokePreconditionError(
                f"db at {db_path} has no readable prices table ({exc}) - {backfill_hint}"
            ) from exc
        per_asset = {str(asset): int(count) for asset, count in rows}
    finally:
        conn.close()
    empty_assets = [asset for asset in UNIVERSE if per_asset.get(asset, 0) == 0]
    if len(empty_assets) == len(UNIVERSE):
        raise SmokePreconditionError(f"db at {db_path} has no price rows - {backfill_hint}")
    if empty_assets:
        raise SmokePreconditionError(
            f"db at {db_path} has no price rows for asset(s) {', '.join(empty_assets)} - "
            f"{backfill_hint}"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailureError(message)


def _duration_s(started: object, finished: object) -> float | None:
    """Seconds between two stored ``*_at_utc`` strings, or None if unparseable."""
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    except ValueError:
        return None
    return delta.total_seconds()


def _stage_failure_message(
    run_id: int,
    problem: str,
    stage_rows: list[tuple[Any, ...]],
    error_text: object,
) -> str:
    """Failure detail the operator can act on: the run's ``error_text`` plus
    every non-ok stage's ``detail_json`` verbatim."""
    lines = [f"run {run_id}: {problem}", f"  runs.error_text: {error_text!r}"]
    for stage, status, _started, _finished, detail_json in stage_rows:
        if str(status) != "ok":
            lines.append(f"  stage {stage!r} status={status!r} detail_json: {detail_json}")
    return "\n".join(lines)


def _parsed_stage_detail(run_id: int, stage: str, raw: object) -> dict[str, Any]:
    """Defensively parse one stage's stored ``detail_json`` into a dict.

    NULL, invalid JSON, or a non-object payload is a classified
    :class:`SmokeFailureError` naming the stage — never a raw traceback.
    """
    if raw is None:
        raise SmokeFailureError(f"run {run_id}: stage {stage!r} has no detail_json")
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise SmokeFailureError(
            f"run {run_id}: stage {stage!r} detail_json is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise SmokeFailureError(
            f"run {run_id}: stage {stage!r} detail_json is not a JSON object: {parsed!r}"
        )
    return parsed


def _verify_run(
    conn: sqlite3.Connection, run_id: int
) -> tuple[tuple[StageCard, ...], dict[str, Any]]:
    """Assert the run and EVERY stage landed ``ok``; return (cards, ingest detail)."""
    run_row = conn.execute(
        "SELECT status, error_text FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        raise SmokeFailureError(f"run {run_id}: no runs row was persisted")
    run_status, error_text = str(run_row[0]), run_row[1]

    stage_rows: list[tuple[Any, ...]] = conn.execute(
        "SELECT stage, status, started_at_utc, finished_at_utc, detail_json "
        "FROM run_stages WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()

    if run_status != "ok":
        raise SmokeFailureError(
            _stage_failure_message(
                run_id, f"runs.status is {run_status!r}, expected 'ok'", stage_rows, error_text
            )
        )

    by_stage = {str(row[0]): row for row in stage_rows}
    missing = [stage for stage in STAGES if stage not in by_stage]
    if missing:
        raise SmokeFailureError(
            _stage_failure_message(
                run_id, f"missing run_stages row(s) for {missing}", stage_rows, error_text
            )
        )
    not_ok = [stage for stage in STAGES if str(by_stage[stage][1]) != "ok"]
    if not_ok:
        raise SmokeFailureError(
            _stage_failure_message(
                run_id, f"stage(s) {not_ok} did not land status='ok'", stage_rows, error_text
            )
        )

    cards = tuple(
        StageCard(
            stage=stage,
            status=str(by_stage[stage][1]),
            duration_s=_duration_s(by_stage[stage][2], by_stage[stage][3]),
        )
        for stage in STAGES
    )

    # Provenance-label half of the zero-network-for-prices assertion; the
    # structural sys.modules half runs in _verify_no_network_modules.
    ingest_detail = _parsed_stage_detail(run_id, "ingest", by_stage["ingest"][4])
    ingest_source = str(ingest_detail.get("source"))
    _require(
        ingest_source == "cache",
        f"run {run_id}: ingest source is {ingest_source!r}, expected 'cache' "
        "(the smoke must run against cached data with zero network use for prices)",
    )
    return cards, ingest_detail


def _verify_weights(conn: sqlite3.Connection, run_id: int) -> tuple[dict[str, float], float]:
    """Assert ``target_weights`` covers the UNIVERSE and sums to 1."""
    rows = conn.execute(
        "SELECT asset, weight FROM target_weights WHERE run_id = ?", (run_id,)
    ).fetchall()
    weights = {str(asset): float(weight) for asset, weight in rows}
    _require(
        set(weights) == set(UNIVERSE),
        f"run {run_id}: target_weights assets {sorted(weights)} != UNIVERSE {sorted(UNIVERSE)}",
    )
    total = sum(weights.values())
    _require(
        math.isclose(total, 1.0, abs_tol=1e-9),
        f"run {run_id}: weights sum to {total!r}, expected 1.0 within 1e-9 ({weights})",
    )
    return weights, total


def _verify_no_network_modules(preloaded: frozenset[str], macro_code: str, run_id: int) -> None:
    """Structural zero-network check via ``sys.modules`` (module docstring).

    ``yfinance`` is imported only inside ``YFinanceAdapter.fetch``, so its
    appearance during the smoke proves a network price path executed.
    ``fredapi`` is checked only when the run recorded the no-key degraded mode
    (a configured key legitimately loads it for the startup probe). A module
    in ``preloaded`` (already imported before the smoke booted) makes its half
    inconclusive and is skipped — the ingest 'cache' label check still holds.
    """
    if "yfinance" not in preloaded:
        _require(
            "yfinance" not in sys.modules,
            f"run {run_id}: yfinance was imported during the smoke - a network price "
            "path executed (it is imported only inside YFinanceAdapter.fetch); the "
            "smoke must serve prices from the SQLite cache only",
        )
    if macro_code == MACRO_DISABLED_NO_KEY and "fredapi" not in preloaded:
        _require(
            "fredapi" not in sys.modules,
            f"run {run_id}: fredapi was imported during the smoke despite the run "
            "recording the no-key degraded mode - no FRED path should have executed",
        )


def _run_cycle(path: Path, start: float) -> SmokeReport:
    """The whole booted cycle (lifespan enter -> trigger -> verify).

    Runs on the watchdog worker thread (:func:`run_smoke` joins it with the
    budget); everything here, including the SQLite reads, stays on this one
    thread.
    """
    preloaded = frozenset(name for name in _NETWORK_MODULES if name in sys.modules)
    # Production factory + lifespan: real macro probe, THE writer connection.
    with TestClient(create_app(path)) as client:
        # force=True bypasses the freshness gate so the smoke ALWAYS exercises
        # all six stages (an unforced second run of the day would skip).
        response = client.post("/api/runs/trigger", json={"force": True})
        _require(
            response.status_code == 202,
            f"POST /api/runs/trigger returned {response.status_code}, expected 202 "
            f"(body: {response.text})",
        )
        run_id = int(response.json()["run_id"])

        # Verify straight from the db file (the same rows the UI reads).
        conn = storage.open_read_only(path)
        try:
            cards, ingest_detail = _verify_run(conn, run_id)
            weights, weights_sum = _verify_weights(conn, run_id)
        finally:
            conn.close()

        # And through the API: /api/runs/latest must serve an ok run at least
        # as new as ours (>=, not ==: a concurrent run may have landed after).
        latest = client.get("/api/runs/latest")
        _require(
            latest.status_code == 200,
            f"GET /api/runs/latest returned {latest.status_code}, expected 200",
        )
        latest_run_id = int(latest.json()["run"]["run_id"])
        _require(
            latest_run_id >= run_id,
            f"GET /api/runs/latest serves run {latest_run_id}, older than the smoke run {run_id}",
        )

    macro = ingest_detail.get("macro")
    macro_code = str(macro.get("code")) if isinstance(macro, dict) else ""
    _verify_no_network_modules(preloaded, macro_code, run_id)

    return SmokeReport(
        db_path=path,
        run_id=run_id,
        cards=cards,
        weights=weights,
        weights_sum=weights_sum,
        ingest_source=str(ingest_detail.get("source")),
        elapsed_s=monotonic() - start,
    )


def run_smoke(db_path: str | Path = DEFAULT_DB_PATH) -> SmokeReport:
    """The importable smoke core (``tests/test_smoke.py`` drives this too).

    Raises :class:`SmokePreconditionError` (no db / no price rows),
    :class:`SmokeTimeoutError` (watchdog), or :class:`SmokeFailureError` (any
    assertion failed); returns a :class:`SmokeReport` only on a full PASS.
    """
    path = Path(db_path)
    _check_preconditions(path)

    start = monotonic()
    result: list[SmokeReport] = []
    failure: list[Exception] = []

    def _worker() -> None:
        try:
            result.append(_run_cycle(path, start))
        except Exception as exc:  # re-raised on the main thread below
            failure.append(exc)

    # daemon=True: on watchdog timeout the hung thread is abandoned — the
    # process is about to exit with the failure, and a non-daemon thread
    # would block interpreter shutdown forever.
    thread = threading.Thread(target=_worker, name="smoke-cycle", daemon=True)
    thread.start()
    thread.join(WATCHDOG_SECONDS)
    if thread.is_alive():
        raise SmokeTimeoutError(
            f"SMOKE TIMEOUT after {monotonic() - start:.1f}s "
            f"(budget {WATCHDOG_SECONDS:.0f}s): the booted cycle (lifespan enter -> "
            "trigger -> verify) is hung - abandoning the daemon worker thread"
        )
    if failure:
        raise failure[0]
    if not result:  # pragma: no cover — the worker always appends one or the other
        raise SmokeFailureError("smoke worker produced neither a report nor an error")
    return result[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Thin CLI over :func:`run_smoke`; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python scripts/smoke.py",
        description="Smoke gate: one real end-to-end pipeline cycle against cached data.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite db to run against (default: the real production db at {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        report = run_smoke(args.db)
    except SmokePreconditionError as exc:
        print(f"SMOKE PRECONDITION MISSING (exit {EXIT_PRECONDITION}):\n{exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    except SmokeTimeoutError as exc:
        print(f"{exc} (exit {EXIT_FAILURE})", file=sys.stderr)
        return EXIT_FAILURE
    except SmokeFailureError as exc:
        print(f"SMOKE FAILURE (exit {EXIT_FAILURE}):\n{exc}", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:
        # Unclassified crash of the smoke itself: keep the exit-code contract
        # (1) but make it visually distinct from a classified failure.
        traceback.print_exc()
        print(f"SMOKE INTERNAL ERROR (unclassified): {exc!r}", file=sys.stderr)
        return EXIT_FAILURE

    print(f"smoke db : {report.db_path}")
    print(f"run_id   : {report.run_id}  (ingest source: {report.ingest_source})")
    print("stages:")
    for card in report.cards:
        print(card.line())
    print("target weights:")
    for asset in UNIVERSE:
        print(f"  {asset:<5} {report.weights[asset]:+.6f}")
    print(f"weights sum: {report.weights_sum:.12f}")
    print(f"elapsed: {report.elapsed_s:.1f}s (budget {WATCHDOG_SECONDS:.0f}s)")
    print("SMOKE PASS")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
