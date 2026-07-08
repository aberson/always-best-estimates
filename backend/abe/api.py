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
  already_running}`` — answered at the run's START, not its completion
  (Step 11 trigger contract below).

Response shapes are plain dicts mirroring the stored rows; dates/timestamps
are returned as the stored TEXT strings. ``detail_json`` columns are parsed
and exposed as a structured ``detail`` member (the UI cards consume objects,
not JSON strings); ``relaxed_turnover`` comes back as a bool.

Connection discipline (plan section 3, one-writer):

- All reads go through short-lived ``storage.open_read_only`` connections,
  opened and closed inside the request.
- THE writer connection is owned by the :class:`abe.scheduler.Scheduler`
  (created in lifespan, stored on ``app.state.scheduler``), which opens it ON
  its single-worker executor thread and runs every pipeline body there —
  sqlite3's ``check_same_thread`` guard stays intact, and the single executor
  thread serializes all writes. The API process never touches the writer
  connection directly.

Trigger contract (Step 11): the endpoint calls
``scheduler.request_run(force)`` and answers ``202`` when the run has STARTED
(its ``run_id`` exists as a committed ``'running'`` row) but not finished —
clients poll ``/api/runs/latest`` (the UI already does) for the result. While
a run is executing, a concurrent trigger COALESCES: no second run is queued;
the response carries the ACTIVE run's id with ``already_running: true`` (and
the coalesced request's ``force`` flag is dropped — plan section 6). Because
the pipeline executes on the executor thread, the event loop stays free: ALL
routes, including ``/health``, respond during an active run.

Model toggle (plan Step 13): lifespan resolves the forecaster from the
environment via :func:`resolve_startup_model` — ``ABE_MODEL`` (``ewma`` default,
or ``jepa``) and ``ABE_JEPA_CHECKPOINT`` (the checkpoint path for ``jepa``). No
env => EWMA (the DEFAULT stays EWMA). An invalid ``jepa`` config (missing or
malformed checkpoint) raises at startup rather than silently falling back to
EWMA — the plan's fail-loud posture. An explicit ``scheduler_config`` (test
seam) bypasses the env read and is used verbatim.

Lifespan resolves the macro status ONCE via
``probe_fred_key(load_fred_api_key())`` (plan Step 4's startup key-probe):
with no key configured this makes no network request and yields the explicit
``MACRO_DISABLED_NO_KEY`` degraded mode, which every run surfaces on its
ingest card. The scheduler then starts (sweeping orphaned ``'running'`` rows)
and fires an immediate ``trigger='startup'`` run (plan section 10).

Production static serving (plan Step 10, section 2 one-process mode): IF the
built frontend exists (``frontend/dist``, produced by ``npm run build
--prefix frontend``), it is mounted at ``/`` via ``StaticFiles(html=True)``.
The mount is registered LAST, so every API route (``/api/*``, ``/health``)
matches first; the mount is CONDITIONAL on the directory existing so dev
(Vite on :5174 proxying ``/api`` here) and tests without a build keep
working — without a dist, ``GET /`` is a plain 404 while the API is
untouched. The check runs at ``create_app`` time: build the frontend before
starting uvicorn (a dist created afterwards needs a restart).
"""

import json
import os
import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from abe import storage
from abe.ingest.macro import load_fred_api_key, probe_fred_key
from abe.model import load_model
from abe.model.base import WorldModel
from abe.scheduler import Scheduler, SchedulerConfig

__all__ = [
    "ABE_JEPA_CHECKPOINT_ENV",
    "ABE_MODEL_ENV",
    "DEFAULT_HISTORY_LIMIT",
    "FRONTEND_DIST",
    "MAX_HISTORY_LIMIT",
    "TriggerRequest",
    "app",
    "create_app",
    "resolve_startup_model",
]

ABE_MODEL_ENV: Final[str] = "ABE_MODEL"
"""Env var selecting the forecaster at startup: ``ewma`` (default) or ``jepa``."""

ABE_JEPA_CHECKPOINT_ENV: Final[str] = "ABE_JEPA_CHECKPOINT"
"""Env var giving the JEPA checkpoint path when ``ABE_MODEL=jepa`` (plan Step 13)."""

DEFAULT_HISTORY_LIMIT: Final[int] = 20
"""History page size when the client sends no ``limit``."""

MAX_HISTORY_LIMIT: Final[int] = 200
"""Upper bound on ``limit`` (the ledger grows forever; the API page must
not). Enforced as a 422 REJECTION by query validation, not a silent clamp."""

FRONTEND_DIST: Final[Path] = Path(__file__).resolve().parents[2] / "frontend" / "dist"
"""The built frontend (``npm run build --prefix frontend``), resolved from
this file's location (``backend/abe/`` -> repo root) so it is cwd-independent.
Mounted at ``/`` by :func:`create_app` IF it exists (module docstring)."""

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


def resolve_startup_model(env: Mapping[str, str] | None = None) -> WorldModel:
    """Resolve the startup forecaster from the environment (plan Step 13 toggle).

    Reads :data:`ABE_MODEL_ENV` (default ``"ewma"``) and, for ``jepa``,
    :data:`ABE_JEPA_CHECKPOINT_ENV`, then delegates to
    ``abe.model.load_model``. With NO env set this returns an
    :class:`~abe.model.base.EWMABaseline` — the DEFAULT stays EWMA (pinned by
    test). An invalid ``jepa`` config (missing/bad checkpoint) raises loudly at
    startup rather than silently falling back to EWMA — the plan's fail-loud
    posture. Directly testable (pass an ``env`` mapping).
    """
    environ = os.environ if env is None else env
    spec = environ.get(ABE_MODEL_ENV, "ewma")
    checkpoint = environ.get(ABE_JEPA_CHECKPOINT_ENV)
    return load_model(spec, Path(checkpoint) if checkpoint else None)


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


def create_app(
    db_path: str | Path = storage.DEFAULT_DB_PATH,
    *,
    static_dir: str | Path = FRONTEND_DIST,
    scheduler_config: SchedulerConfig | None = None,
) -> FastAPI:
    """Build the app against ``db_path`` (configurable for tests).

    The module-level ``app`` uses ``storage.DEFAULT_DB_PATH`` — that is the
    uvicorn production target.

    ``static_dir`` is the built frontend to mount at ``/`` (default
    :data:`FRONTEND_DIST`; configurable for tests). A missing directory means
    NO mount — the conditional-serving contract in the module docstring.

    ``scheduler_config`` tunes the lifespan scheduler; ``None`` (production)
    uses ``SchedulerConfig()`` defaults. Overrides are the documented test
    seams (offline suites disable the daily network fetch and/or inject a
    ``pipeline_fn`` — see ``abe.scheduler``'s module docstring).
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
        # Model toggle (plan Step 13): with no explicit scheduler_config
        # (production), resolve the forecaster from ABE_MODEL/ABE_JEPA_CHECKPOINT
        # — default EWMA, fail-loud on an invalid jepa config (no silent EWMA
        # fallback). An explicit scheduler_config (tests) is respected as-is.
        resolved_config = (
            SchedulerConfig(model=resolve_startup_model())
            if scheduler_config is None
            else scheduler_config
        )
        # The scheduler owns THE writer connection + the single-worker
        # executor thread (Step 11): start() opens the connection ON that
        # thread, sweeps orphaned 'running' rows, then fires the immediate
        # startup run; stop() drains the executor and closes the connection
        # on its owning thread.
        scheduler = Scheduler(
            resolved_path,
            macro_status=app.state.macro_status,
            config=resolved_config,
        )
        await scheduler.start()
        app.state.scheduler = scheduler
        try:
            yield
        finally:
            await scheduler.stop()

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
        """Request a run NOW; the 202 answers at the run's START, not its end.

        Idle: wakes the scheduler and awaits the new run's START (a committed
        ``'running'`` row with an allocated id) -> ``{run_id,
        already_running: false}``; poll ``/api/runs/latest`` for completion.
        Run already executing: coalesces (single-flight, no second run
        queued) -> the ACTIVE run's ``{run_id, already_running: true}``
        immediately; a coalesced request's ``force`` flag is dropped by
        design (module docstring / plan section 6).
        """
        payload = body if body is not None else TriggerRequest()
        scheduler: Scheduler = request.app.state.scheduler
        pending = await scheduler.request_run(force=payload.force)
        return {"run_id": pending.run_id, "already_running": pending.already_running}

    # Production static serving (module docstring): mounted LAST so every API
    # route above matches first; conditional so dev/tests without a build
    # (and the no-dist 404 contract) keep working. html=True serves
    # index.html at ``/``.
    resolved_static = Path(static_dir)
    if resolved_static.is_dir():
        app.mount("/", StaticFiles(directory=resolved_static, html=True), name="frontend")

    return app


app = create_app()
