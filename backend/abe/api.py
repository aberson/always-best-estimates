"""FastAPI app: the plan section 6 route contract over the run ledger.

uvicorn target: ``abe.api:app`` (127.0.0.1:8140, single worker, no --reload).

Routes (plan section 6):

- ``GET /health`` -> ``{"status": "ok"}`` (liveness).
- ``GET /api/explain`` -> ``{"explanations": {key: {label, formula,
  description, example, unit, window}}}`` — the static calculation registry
  (``abe.calc.EXPLANATIONS``); read-only, no DB, so the UI can annotate cards.
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

from abe import config as config_module
from abe import storage
from abe.calc import EXPLANATIONS
from abe.config import Config, ViewScenario
from abe.ingest.macro import load_fred_api_key, probe_fred_key
from abe.model import load_model
from abe.model.base import WorldModel
from abe.pipeline import latest_ok_central_run_id, latest_ok_run_id_for_config
from abe.registry import registries_manifest
from abe.scheduler import CentralConfigRunError, Scheduler, SchedulerConfig

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


class ConfigCreate(BaseModel):
    """``POST /api/configs`` body (mirrors the ``configs`` shape, plan §5)."""

    name: str
    feature_set: str
    forecaster: str
    view_scenario_id: int
    optimizer: str
    params: dict[str, Any] = {}


class ConfigUpdate(BaseModel):
    """``PATCH /api/configs/{id}`` body — every field optional (partial update)."""

    name: str | None = None
    feature_set: str | None = None
    forecaster: str | None = None
    view_scenario_id: int | None = None
    optimizer: str | None = None
    params: dict[str, Any] | None = None


class ScenarioCreate(BaseModel):
    """``POST /api/scenarios`` body (mirrors the ``view_scenarios`` shape)."""

    name: str
    kind: str
    payload: dict[str, Any] = {}


class ScenarioUpdate(BaseModel):
    """``PATCH /api/scenarios/{id}`` body — ``kind`` is immutable (author anew)."""

    name: str | None = None
    payload: dict[str, Any] | None = None


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


def _config_payload(config: Config) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "name": config.name,
        "feature_set": config.feature_set,
        "forecaster": config.forecaster,
        "view_scenario_id": config.view_scenario_id,
        "optimizer": config.optimizer,
        "params": config.params,
        "is_central": config.is_central,
        "created_at_utc": config.created_at_utc,
        "updated_at_utc": config.updated_at_utc,
    }


def _scenario_payload(scenario: ViewScenario) -> dict[str, Any]:
    return {
        "view_scenario_id": scenario.view_scenario_id,
        "name": scenario.name,
        "kind": scenario.kind,
        "payload": scenario.payload,
        "created_at_utc": scenario.created_at_utc,
    }


def _compare_entry(conn: sqlite3.Connection, config_id: int) -> dict[str, Any] | None:
    """One config's comparison row: its latest ok run's weights + objective, or
    empty facts when it has no run yet. ``None`` when the config doesn't exist."""
    config = config_module.get_config(conn, config_id)
    if config is None:
        return None
    run_id = latest_ok_run_id_for_config(conn, config_id)
    weights: dict[str, float] = {}
    objective: Any = None
    finished_at_utc: Any = None
    if run_id is not None:
        weights = {
            str(asset): float(weight)
            for asset, weight in conn.execute(
                "SELECT asset, weight FROM target_weights WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        }
        opt_row = conn.execute(
            "SELECT detail_json FROM run_stages WHERE run_id = ? AND stage = 'optimize'",
            (run_id,),
        ).fetchone()
        if opt_row is not None and opt_row[0]:
            parsed = _parsed_detail(opt_row[0])
            if isinstance(parsed, dict):
                objective = parsed.get("objective")
        fin_row = conn.execute(
            "SELECT finished_at_utc FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        finished_at_utc = None if fin_row is None else fin_row[0]
    return {
        "config_id": config.config_id,
        "name": config.name,
        "is_central": config.is_central,
        "weights": weights,
        "objective": objective,
        "run_id": run_id,
        "finished_at_utc": finished_at_utc,
    }


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

    @app.get("/api/explain")
    def explain() -> dict[str, Any]:
        """The calculation registry (Track 1 transparency pass): a legible
        formula + worked example per quantity the stage cards display, built
        from :data:`abe.calc.EXPLANATIONS`. Read-only and static — no DB, no
        run state — so the UI can annotate any card without a per-run round
        trip."""
        return {"explanations": {key: exp.payload() for key, exp in EXPLANATIONS.items()}}

    @app.get("/api/runs/latest")
    def runs_latest() -> dict[str, Any]:
        """The latest ok CENTRAL run + its stages (the UI poll target).

        Scoped to the central config so an on-demand non-central run never
        hijacks the displayed "portfolio you'd buy" (Track 2 Step 21)."""
        with _read_conn(resolved_path) as conn:
            run_id = latest_ok_central_run_id(conn)
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

    @app.post("/api/configs/{config_id}/run")
    async def run_config(
        config_id: int, request: Request, body: TriggerRequest | None = None
    ) -> dict[str, Any]:
        """Compute a run for a Config on demand (Track 2 Step 21).

        Dispatched through the scheduler's single executor (one-writer / FIFO
        preserved). ``force=false`` (default) serves a cached run when the data
        is unchanged since this config's last ok run; otherwise a fresh run is
        computed and tagged with ``config_id``. Blocks until the run completes.
        Returns ``{run_id, config_id, cached}``; 404 on an unknown config.
        """
        payload = body if body is not None else TriggerRequest()
        scheduler: Scheduler = request.app.state.scheduler
        try:
            result = await scheduler.request_config_run(config_id, force=payload.force)
        except CentralConfigRunError as exc:
            # central belongs to the always-on loop, not the on-demand path
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": result.run_id, "config_id": result.config_id, "cached": result.cached}

    # ------------------------------------------------------------------ #
    # Config / scenario / compare API (Track 2 Step 25). Writes go through the
    # scheduler's single writer (one-writer discipline); reads use ro conns.
    # ------------------------------------------------------------------ #

    @app.get("/api/registries")
    def registries() -> dict[str, Any]:
        """Stage-registry manifest (keys + param schemas) for the UI dropdowns."""
        return {"registries": registries_manifest()}

    @app.get("/api/configs")
    def list_configs() -> dict[str, Any]:
        with _read_conn(resolved_path) as conn:
            return {"configs": [_config_payload(c) for c in config_module.list_configs(conn)]}

    @app.post("/api/configs", status_code=201)
    async def create_config(body: ConfigCreate, request: Request) -> dict[str, Any]:
        scheduler: Scheduler = request.app.state.scheduler

        def _write(conn: sqlite3.Connection) -> Config:
            return config_module.create_config(
                conn,
                name=body.name,
                feature_set=body.feature_set,
                forecaster=body.forecaster,
                view_scenario_id=body.view_scenario_id,
                optimizer=body.optimizer,
                params=body.params,
            )

        try:
            created = await scheduler.run_write(_write)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail=f"config create conflict: {exc}") from exc
        return _config_payload(created)

    @app.get("/api/configs/{config_id}")
    def get_config(config_id: int) -> dict[str, Any]:
        with _read_conn(resolved_path) as conn:
            config = config_module.get_config(conn, config_id)
        if config is None:
            raise HTTPException(status_code=404, detail=f"unknown config_id {config_id}")
        return _config_payload(config)

    @app.patch("/api/configs/{config_id}")
    async def update_config(
        config_id: int, body: ConfigUpdate, request: Request
    ) -> dict[str, Any]:
        scheduler: Scheduler = request.app.state.scheduler

        def _write(conn: sqlite3.Connection) -> Config:
            return config_module.update_config(
                conn,
                config_id,
                name=body.name,
                feature_set=body.feature_set,
                forecaster=body.forecaster,
                view_scenario_id=body.view_scenario_id,
                optimizer=body.optimizer,
                params=body.params,
            )

        try:
            updated = await scheduler.run_write(_write)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _config_payload(updated)

    @app.delete("/api/configs/{config_id}", status_code=204)
    async def delete_config(config_id: int, request: Request) -> None:
        scheduler: Scheduler = request.app.state.scheduler
        with _read_conn(resolved_path) as conn:
            if config_module.get_config(conn, config_id) is None:
                raise HTTPException(status_code=404, detail=f"unknown config_id {config_id}")

        def _write(conn: sqlite3.Connection) -> None:
            config_module.delete_config(conn, config_id)

        try:
            await scheduler.run_write(_write)
        except ValueError as exc:  # central / referenced by runs
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/configs/{config_id}/central")
    async def set_central(config_id: int, request: Request) -> dict[str, Any]:
        """Promote a config to central — the deliberate, guarded operator action."""
        scheduler: Scheduler = request.app.state.scheduler

        def _write(conn: sqlite3.Connection) -> Config:
            return config_module.set_central(conn, config_id)

        try:
            promoted = await scheduler.run_write(_write)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _config_payload(promoted)

    @app.get("/api/scenarios")
    def list_scenarios() -> dict[str, Any]:
        with _read_conn(resolved_path) as conn:
            scenarios = config_module.list_view_scenarios(conn)
        return {"scenarios": [_scenario_payload(s) for s in scenarios]}

    @app.post("/api/scenarios", status_code=201)
    async def create_scenario(body: ScenarioCreate, request: Request) -> dict[str, Any]:
        scheduler: Scheduler = request.app.state.scheduler

        def _write(conn: sqlite3.Connection) -> ViewScenario:
            return config_module.create_view_scenario(
                conn, name=body.name, kind=body.kind, payload=body.payload
            )

        try:
            created = await scheduler.run_write(_write)
        except ValueError as exc:  # bad kind
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _scenario_payload(created)

    @app.patch("/api/scenarios/{scenario_id}")
    async def update_scenario(
        scenario_id: int, body: ScenarioUpdate, request: Request
    ) -> dict[str, Any]:
        scheduler: Scheduler = request.app.state.scheduler

        def _write(conn: sqlite3.Connection) -> ViewScenario:
            return config_module.update_view_scenario(
                conn, scenario_id, name=body.name, payload=body.payload
            )

        try:
            updated = await scheduler.run_write(_write)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _scenario_payload(updated)

    @app.delete("/api/scenarios/{scenario_id}", status_code=204)
    async def delete_scenario(scenario_id: int, request: Request) -> None:
        scheduler: Scheduler = request.app.state.scheduler
        with _read_conn(resolved_path) as conn:
            if config_module.get_view_scenario(conn, scenario_id) is None:
                raise HTTPException(
                    status_code=404, detail=f"unknown view_scenario_id {scenario_id}"
                )

        def _write(conn: sqlite3.Connection) -> None:
            config_module.delete_view_scenario(conn, scenario_id)

        try:
            await scheduler.run_write(_write)
        except ValueError as exc:  # referenced by a config
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/compare")
    def compare(
        config_ids: Annotated[str, Query(description="comma-separated config ids")],
    ) -> dict[str, Any]:
        """N configs' latest allocations side by side; the central id is flagged.

        Unknown config_ids are silently skipped (a compare view tolerates a
        stale/deleted id); a config with no run yet returns empty weights + null
        objective. Weights reflect the config's LATEST run, which may lag a
        recipe edit until it is re-run via ``POST /api/configs/{id}/run``."""
        try:
            ids = [int(token) for token in config_ids.split(",") if token.strip()]
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="config_ids must be comma-separated integers"
            ) from exc
        if not ids:
            raise HTTPException(status_code=422, detail="config_ids must not be empty")
        with _read_conn(resolved_path) as conn:
            central = config_module.get_central_config(conn)
            entries = [
                entry for cid in ids if (entry := _compare_entry(conn, cid)) is not None
            ]
        return {"central_config_id": central.config_id, "configs": entries}

    # Production static serving (module docstring): mounted LAST so every API
    # route above matches first; conditional so dev/tests without a build
    # (and the no-dist 404 contract) keep working. html=True serves
    # index.html at ``/``.
    resolved_static = Path(static_dir)
    if resolved_static.is_dir():
        app.mount("/", StaticFiles(directory=resolved_static, html=True), name="frontend")

    return app


app = create_app()
