"""FastAPI app: the plan section 6 route contract over the run ledger.

uvicorn target: ``abe.api:app`` (127.0.0.1:8140, single worker, no --reload).

Routes (plan section 6):

- ``GET /health`` -> ``{"status": "ok"}`` (liveness).
- ``GET /api/runs/latest`` -> the latest ok run + its stages (plan section 3's
  "Latest" definition: ``MAX(run_id) WHERE status='ok'`` via
  ``storage.latest_ok_run_id``); 404 JSON (``{"detail": ...}``) when no run
  has succeeded yet.
- ``GET /api/runs/{run_id}/stages`` -> that run's ``run_stages`` rows (404 for
  an unknown run). Rows come back in rowid order, which the pipeline
  guarantees is stage-execution order on both the success and error paths.
- ``GET /api/history?limit=N`` -> recent runs (newest first) each with its
  ``target_weights`` rows. ``limit`` defaults to
  :data:`DEFAULT_HISTORY_LIMIT`; values outside ``[1, MAX_HISTORY_LIMIT]``
  are REJECTED with 422 (FastAPI query validation), not silently clamped —
  the only clients are our own UI and the operator, and a loud contract
  beats a quiet one.
- ``POST /api/runs/trigger`` ``{force?: bool}`` -> ``202 {run_id,
  already_running: false}``.

Response shapes are plain dicts mirroring the stored rows; dates/timestamps
are returned as the stored TEXT strings. ``detail_json`` columns are parsed
and exposed as a structured ``detail`` member (the UI cards consume objects,
not JSON strings); ``relaxed_turnover`` comes back as a bool.

Connection discipline (plan section 3, one-writer):

- All reads go through short-lived ``storage.open_read_only`` connections,
  opened and closed inside the request.
- The trigger endpoint uses THE writer connection owned by the app
  (``app.state.writer_conn``, created in lifespan). In Step 11 this same
  connection becomes the pipeline thread's connection (handed to the
  single-worker executor); until then the endpoint is ``async`` so it runs on
  the lifespan's event-loop thread — sqlite3's ``check_same_thread`` guard
  stays intact, and the single event loop serializes concurrent triggers.

Trigger design note (V1 pre-scheduler): the endpoint RUNS the pipeline
synchronously inside the request and returns ``202 {run_id,
already_running: false}`` with the finished run's id. Known V1 consequence:
while that synchronous run executes, the single event loop is blocked, so ALL
routes — including ``/health`` — stall until it finishes (seconds against a
warm cache). Step 11 fixes this by swapping ONLY the body of ``trigger_run``
— set the scheduler's ``asyncio.Event`` and return immediately (coalescing to
``already_running: true`` during an active run), with the pipeline body moved
onto the single-worker executor — the route shape, status code, and response
keys are already final.

Lifespan also resolves the macro status ONCE via
``probe_fred_key(load_fred_api_key())`` (plan Step 4's startup key-probe):
with no key configured this makes no network request and yields the explicit
``MACRO_DISABLED_NO_KEY`` degraded mode, which every run surfaces on its
ingest card.
"""

import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from abe import storage
from abe.ingest.macro import load_fred_api_key, probe_fred_key
from abe.pipeline import run_pipeline

__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "MAX_HISTORY_LIMIT",
    "TriggerRequest",
    "app",
    "create_app",
]

DEFAULT_HISTORY_LIMIT: Final[int] = 20
"""History page size when the client sends no ``limit``."""

MAX_HISTORY_LIMIT: Final[int] = 200
"""Upper bound on ``limit`` (the ledger grows forever; the API page must
not). Enforced as a 422 REJECTION by query validation, not a silent clamp."""

_RUN_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "started_at_utc",
    "finished_at_utc",
    "status",
    "trigger",
    "error_text",
)

_RUN_SELECT: Final[str] = (
    'SELECT run_id, started_at_utc, finished_at_utc, status, "trigger", error_text FROM runs'
)


class TriggerRequest(BaseModel):
    """``POST /api/runs/trigger`` body; omitted body means ``force=False``."""

    force: bool = False


@contextmanager
def _read_conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """A short-lived read-only connection scoped to one request."""
    conn = storage.open_read_only(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _run_dict(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute(f"{_RUN_SELECT} WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return dict(zip(_RUN_COLUMNS, row, strict=True))


def _stage_dicts(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT stage, status, started_at_utc, finished_at_utc, detail_json "
        "FROM run_stages WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()
    return [
        {
            "stage": stage,
            "status": status,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "detail": _parsed_detail(detail_json),
        }
        for stage, status, started, finished, detail_json in rows
    ]


def _parsed_detail(detail_json: object) -> object:
    """Parse a stored ``detail_json`` string, defensively.

    A corrupt row must degrade to a ``{"parse_error": ...}`` marker on that
    one card — never a permanent 500 on the UI poll target (mirrors
    pipeline.py's own defensive read of the freshness detail).
    """
    if not detail_json:
        return None
    try:
        return json.loads(str(detail_json))
    except json.JSONDecodeError as exc:
        return {"parse_error": f"stored detail_json is not valid JSON: {exc}"}


def _weight_dicts(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT asset, weight, prev_weight, turnover, relaxed_turnover "
        "FROM target_weights WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()
    return [
        {
            "asset": asset,
            "weight": weight,
            "prev_weight": prev_weight,
            "turnover": turnover,
            "relaxed_turnover": bool(relaxed),
        }
        for asset, weight, prev_weight, turnover, relaxed in rows
    ]


def create_app(db_path: str | Path = storage.DEFAULT_DB_PATH) -> FastAPI:
    """Build the app against ``db_path`` (configurable for tests).

    The module-level ``app`` uses ``storage.DEFAULT_DB_PATH`` — that is the
    uvicorn production target.
    """
    resolved_path = Path(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup key-probe FIRST, once (plan Step 4): no key -> degraded
        # mode, no network touched. Ordering is load-bearing: the probe needs
        # no db and MAY raise (key set + FRED unreachable propagates by
        # macro.py's contract), so it runs BEFORE any writer connection
        # exists — a failed startup can never leak open WAL handles (Windows
        # file locks would otherwise pin the db file).
        app.state.macro_status = probe_fred_key(load_fred_api_key())
        # THE writer connection (one-writer discipline). Step 11 hands this
        # same connection to the pipeline executor thread.
        app.state.writer_conn = storage.open_writer(resolved_path)
        try:
            yield
        finally:
            app.state.writer_conn.close()

    app = FastAPI(title="always-best-estimates", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/api/runs/latest")
    def runs_latest() -> dict[str, Any]:
        """The latest ok run + its stages (the UI poll target)."""
        with _read_conn(resolved_path) as conn:
            run_id = storage.latest_ok_run_id(conn)
            if run_id is None:
                raise HTTPException(status_code=404, detail="no successful run yet")
            run = _run_dict(conn, run_id)
            stages = _stage_dicts(conn, run_id)
        return {"run": run, "stages": stages}

    @app.get("/api/runs/{run_id}/stages")
    def run_stages(run_id: int) -> list[dict[str, Any]]:
        """One card per stage for any run (ok, error, or skipped)."""
        with _read_conn(resolved_path) as conn:
            if _run_dict(conn, run_id) is None:
                raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
            return _stage_dicts(conn, run_id)

    @app.get("/api/history")
    def history(
        limit: Annotated[
            int, Query(ge=1, le=MAX_HISTORY_LIMIT, description="max runs returned (newest first)")
        ] = DEFAULT_HISTORY_LIMIT,
    ) -> dict[str, Any]:
        """Recent runs (newest first), each with its target_weights rows."""
        with _read_conn(resolved_path) as conn:
            run_rows = conn.execute(
                f"{_RUN_SELECT} ORDER BY run_id DESC LIMIT ?", (limit,)
            ).fetchall()
            runs: list[dict[str, Any]] = []
            for row in run_rows:
                entry = dict(zip(_RUN_COLUMNS, row, strict=True))
                entry["target_weights"] = _weight_dicts(conn, int(row[0]))
                runs.append(entry)
        return {"runs": runs}

    @app.post("/api/runs/trigger", status_code=202)
    async def trigger_run(request: Request, body: TriggerRequest | None = None) -> dict[str, Any]:
        """Run the pipeline now (V1: synchronously in the request).

        Step 11 swap point: replace this body with "set the scheduler event,
        return immediately" (coalescing to ``already_running: true`` while a
        run is active). The response contract is already final.
        """
        payload = body if body is not None else TriggerRequest()
        conn: sqlite3.Connection = request.app.state.writer_conn
        run_id = run_pipeline(
            conn,
            trigger="manual",
            force=payload.force,
            macro_status=request.app.state.macro_status,
        )
        return {"run_id": run_id, "already_running": False}

    return app


app = create_app()
