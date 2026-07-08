"""Pipeline orchestrator: one run = six stages, each writing a ``run_stages`` row.

plan.md sections 3 (schema + "Latest" + per-run transaction), 4 (stages), 5
(module description) and 8 (fetch-split, excess-return convention) are the
authoritative spec. :func:`run_pipeline` is SYNC and runs on whatever thread
owns the writer connection (Step 11 moves it onto the single-worker executor);
it wires

    freshness -> ingest(cache) -> features -> forecast -> blend -> optimize

for the fixed ``UNIVERSE``, persisting one ledger row per stage plus the
derived-table rows (``features``, ``forecasts``, ``bl_posteriors``,
``target_weights``).

Freshness predicate (stage 0)
=============================

Daily data changes once per day, so the gate asks "has anything about the
stored prices changed since the last ok run?" — which is TWO watermarks, not
one:

- ``data_max_date`` = ``SELECT MAX(date) FROM prices`` — a new trading day
  arrived.
- ``data_fetched_at`` = ``SELECT MAX(fetched_at_utc) FROM prices`` — the
  stored history was REWRITTEN without a new date. Step 3's ingest stamps a
  fresh ``fetched_at_utc`` on every upsert (the overlap-window re-upserts and
  the backward-adjustment full refresh included), so a post-dividend rebase
  that rewrites the whole ``adj_close`` history — same ``MAX(date)`` — still
  moves this watermark. Without it the gate would keep serving pre-rebase
  weights until the next new trading day (up to 3 days over a weekend).

Every run that PROCEEDS records both watermarks in its own freshness stage
``detail_json``. The predicate (:func:`should_skip_run`) then skips iff ALL of:

- ``force`` is False, AND
- a last ok run exists (``storage.latest_ok_run_id``), AND
- the current stored ``MAX(date)`` is non-NULL and equals the
  ``data_max_date`` recorded by that last ok run's freshness row, AND
- the current ``MAX(fetched_at_utc)`` is non-NULL and equals the recorded
  ``data_fetched_at``.

Anything else proceeds: ``force=True`` always recomputes; a cold DB (no ok run
yet) always tries; an empty prices table proceeds and fails LOUD downstream
(features raises on <2 closes) rather than silently skipping forever; a last ok
run with missing/unparseable freshness detail (either watermark) fails open
toward recompute. On skip the run writes ``runs.status='skipped'`` + a single
``'skipped'`` freshness stage row and stops — no data writes.

Transaction discipline (two-phase design)
=========================================

The writer connection is in autocommit mode (``storage.open_writer``); this
module owns the per-run transaction (the storage docstring delegates it here).

- **Phase 0 (autocommit):** insert the ``runs`` row (``status='running'``).
  Committed immediately so the run is observable while executing. An
  AUTOINCREMENT ``run_id`` allocated inside the data transaction would be
  reusable after rollback — allocating it outside keeps run ids stable.
- **Phase 1 (``BEGIN IMMEDIATE`` ... ``COMMIT``):** all six stages. Each stage
  writes its ``run_stages`` row and its data rows inside the transaction, and
  every stage record is ALSO kept in memory. On success (or skip) the ``runs``
  row is updated to ``ok``/``skipped`` inside the same transaction and
  committed atomically.
- **Phase 2 (error path):** on ANY stage exception the whole transaction is
  rolled back — no partial data rows, no partial stage rows — then a FRESH
  ``BEGIN IMMEDIATE`` re-inserts the in-memory stage records (the completed
  ``ok`` rows plus the failing stage's ``error`` row, whose ``detail_json``
  carries the exception message) and updates ``runs`` to ``status='error'`` +
  ``error_text``. These ledger writes survive precisely BECAUSE they happen
  after the rollback in their own transaction: the run's story is inspectable,
  but none of its data rows exist (plan section 3's "a crash leaves an
  inspectable ``status='error'`` row, never orphans").

Stage exceptions NEVER propagate out of :func:`run_pipeline` (the Step 11
scheduler must survive every run); the function returns the ``run_id`` and the
ledger tells the story. Only non-stage infrastructure failures raise: invalid
arguments, a non-file-backed connection, or the phase-2 ledger write itself
failing (nothing left to record to). A ``'running'`` row can therefore outlive
its run in two ways — a hard process kill
(``KeyboardInterrupt``/``SystemExit`` propagate), or an infrastructure failure
that prevented even the phase-2 ledger write (e.g. disk full). Step 11's
scheduler startup owns the sweep for such stale ``'running'`` rows (mark them
``'error'`` before the first scheduled run); this module deliberately does not
guess about rows it did not write.

Stage payloads (``detail_json`` — the UI card contract)
=======================================================

- ``freshness``: ``data_max_date``, ``data_fetched_at``, ``last_ok_run_id``,
  ``last_ok_data_max_date``, ``last_ok_data_fetched_at``, ``force``.
- ``ingest``: per-asset row counts + date ranges (served from SQLite via
  ``CacheAdapter`` — NO network on this path; the daily fetch job is Step 11),
  plus the macro block ``{enabled, code, message}`` — degraded mode
  (``MACRO_DISABLED_*``) surfaces here as a card fact, NOT an error.
- ``features``: feature names + per-asset latest date/values (rows persisted
  for the LATEST bar only in V1 — the card shows current features).
- ``forecast``: ``model_version``, ``horizon_days``, per-asset ``(mu, sigma)``.
- ``blend``: posterior mu AND posterior sigma per asset + confidences + tilt
  (the card can show mu ± sigma). Each ``bl_posteriors`` row stores the
  per-asset slice of ``BLResult.diagnostics`` in its own ``detail_json``;
  ``posterior_sigma`` is ``sqrt(sigma_post[asset, asset])`` — the annualized
  posterior return std.
- ``optimize``: weights, prev_weights, turnover, ``relaxed_turnover``, solver
  status, cold_start. ``w_prev`` is the last persisted allocation of the
  latest ok run (:func:`load_last_weights`); ``None`` = cold start.

Insufficient history (< ``MIN_LW_ROWS`` daily returns for Ledoit-Wolf, or
< ``MIN_HISTORY_BARS`` for the EWMA) is an ordinary stage error: the stage
raises loudly with the clear message and the pipeline records it.
"""

import json
import logging
import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

from abe import storage
from abe.blend.black_litterman import BLResult, bl_blend
from abe.blend.covariance import ledoit_wolf_sigma
from abe.constants import HORIZON_BARS, UNIVERSE
from abe.features.basic import (
    LOG_RETURN_COLUMN,
    REALIZED_VOL_COLUMN,
    log_returns,
    realized_vol,
)
from abe.ingest.macro import MacroStatus, load_fred_api_key, probe_fred_key
from abe.ingest.sources import CacheAdapter, utc_now_iso
from abe.model.base import EWMABaseline, Forecast, WorldModel
from abe.optimize.mvu import optimize_weights

__all__ = [
    "STAGES",
    "TRIGGERS",
    "last_ok_data_fetched_at",
    "last_ok_data_max_date",
    "load_last_weights",
    "run_pipeline",
    "should_skip_run",
    "stored_data_fetched_at",
    "stored_data_max_date",
]

logger = logging.getLogger(__name__)

STAGES: Final[tuple[str, ...]] = (
    "freshness",
    "ingest",
    "features",
    "forecast",
    "blend",
    "optimize",
)
"""The six stage names in execution order (= ``run_stages`` insert order —
consumers may rely on rowid order matching pipeline order)."""

TRIGGERS: Final[tuple[str, ...]] = ("schedule", "manual", "startup")
"""Valid ``runs.trigger`` values (mirrors the table CHECK constraint)."""


# --------------------------------------------------------------------------- #
# Freshness predicate (unit-testable pieces)
# --------------------------------------------------------------------------- #


def stored_data_max_date(conn: sqlite3.Connection) -> str | None:
    """Current ``MAX(date)`` across the whole prices table (one scalar for the
    fixed universe), or ``None`` on an empty table."""
    row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def stored_data_fetched_at(conn: sqlite3.Connection) -> str | None:
    """Current ``MAX(fetched_at_utc)`` across the prices table, or ``None``.

    The revision watermark: Step 3's ingest stamps a fresh ``fetched_at_utc``
    on EVERY upsert (overlap re-upserts and backward-adjustment full refreshes
    included), so any rewrite of stored history moves this value even when
    ``MAX(date)`` does not (module docstring's post-dividend-rebase scenario).
    """
    row = conn.execute("SELECT MAX(fetched_at_utc) FROM prices").fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _last_ok_freshness_value(
    conn: sqlite3.Connection, last_ok_run_id: int | None, key: str
) -> str | None:
    """One string value from the last ok run's freshness ``detail_json``.

    ``None`` when there is no ok run, the freshness row is missing, or the
    detail is unparseable / lacks the key — all of which make the gate fail
    OPEN (recompute).
    """
    if last_ok_run_id is None:
        return None
    row = conn.execute(
        "SELECT detail_json FROM run_stages WHERE run_id = ? AND stage = 'freshness'",
        (last_ok_run_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def last_ok_data_max_date(conn: sqlite3.Connection, last_ok_run_id: int | None) -> str | None:
    """The ``data_max_date`` the last ok run recorded in its freshness detail."""
    return _last_ok_freshness_value(conn, last_ok_run_id, "data_max_date")


def last_ok_data_fetched_at(conn: sqlite3.Connection, last_ok_run_id: int | None) -> str | None:
    """The ``data_fetched_at`` the last ok run recorded in its freshness detail."""
    return _last_ok_freshness_value(conn, last_ok_run_id, "data_fetched_at")


def should_skip_run(
    *,
    force: bool,
    data_max_date: str | None,
    data_fetched_at: str | None,
    last_ok_run_id: int | None,
    last_ok_data_max_date: str | None,
    last_ok_data_fetched_at: str | None,
) -> bool:
    """The freshness gate predicate (module docstring): pure and unit-testable.

    True (skip) iff not forced, an ok run exists, and BOTH watermarks — the
    stored ``MAX(date)`` and the stored ``MAX(fetched_at_utc)`` — are non-NULL
    and equal to the values that run already used. A ``None`` on either side
    of either comparison proceeds (fail-open toward recompute).
    """
    if force:
        return False
    if last_ok_run_id is None:
        return False
    if data_max_date is None or data_max_date != last_ok_data_max_date:
        return False
    if data_fetched_at is None or data_fetched_at != last_ok_data_fetched_at:
        return False
    return True


def load_last_weights(conn: sqlite3.Connection) -> dict[str, float] | None:
    """The last persisted allocation: ``target_weights`` of the latest ok run.

    ``None`` when no ok run exists (or, defensively, when an ok run has no
    weight rows) — the optimizer's cold start, which DROPS the turnover term.
    """
    last_ok = storage.latest_ok_run_id(conn)
    if last_ok is None:
        return None
    rows = conn.execute(
        "SELECT asset, weight FROM target_weights WHERE run_id = ?", (last_ok,)
    ).fetchall()
    if not rows:
        return None
    return {str(row[0]): float(row[1]) for row in rows}


# --------------------------------------------------------------------------- #
# Run machinery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _StageRecord:
    """One stage's ledger row, kept in memory so the error path can replay it
    after the rollback (phase 2 of the transaction design)."""

    stage: str
    status: str  # ok | error | skipped
    started_at_utc: str
    finished_at_utc: str
    detail: dict[str, object] | None


class _StageFailure(Exception):
    """Internal: wraps a stage exception with its stage name for the ledger."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"stage {stage!r} failed: {type(cause).__name__}: {cause}")


@dataclass(slots=True)
class _RunContext:
    """Mutable state threaded through the stages of one run."""

    conn: sqlite3.Connection
    run_id: int
    db_path: Path
    force: bool
    model: WorldModel
    macro_status: MacroStatus
    skip: bool = False
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    returns: dict[str, pd.Series] = field(default_factory=dict)
    features_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    forecasts: dict[str, Forecast] = field(default_factory=dict)
    bl: BLResult | None = None


_StageFn = Callable[[_RunContext], tuple[str, dict[str, object]]]


def _database_file(conn: sqlite3.Connection) -> Path:
    """The on-disk path of the connection's ``main`` database.

    The pipeline derives the path from the connection itself (rather than
    taking a second parameter that could drift from it) because the ingest
    stage's ``CacheAdapter`` re-opens the SAME file read-only.
    """
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) == "main" and row[2]:
            return Path(str(row[2]))
    raise ValueError(
        "writer connection has no on-disk 'main' database (in-memory db?) — "
        "run_pipeline needs a file-backed db so CacheAdapter can re-open it read-only"
    )


def _write_stage_row(conn: sqlite3.Connection, run_id: int, record: _StageRecord) -> None:
    storage.insert_row(
        conn,
        "run_stages",
        {
            "run_id": run_id,
            "stage": record.stage,
            "status": record.status,
            "started_at_utc": record.started_at_utc,
            "finished_at_utc": record.finished_at_utc,
            "detail_json": None if record.detail is None else json.dumps(record.detail),
        },
    )


def _execute_stage(ctx: _RunContext, records: list[_StageRecord], stage: str, fn: _StageFn) -> None:
    """Run one stage: record timestamps, write its ledger row in-transaction,
    and convert any exception into a :class:`_StageFailure` after recording an
    ``error`` stage record (written to the DB by the phase-2 handler)."""
    started = utc_now_iso()
    try:
        status, detail = fn(ctx)
    except Exception as exc:
        records.append(
            _StageRecord(
                stage=stage,
                status="error",
                started_at_utc=started,
                finished_at_utc=utc_now_iso(),
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        raise _StageFailure(stage, exc) from exc
    record = _StageRecord(
        stage=stage,
        status=status,
        started_at_utc=started,
        finished_at_utc=utc_now_iso(),
        detail=detail,
    )
    records.append(record)
    _write_stage_row(ctx.conn, ctx.run_id, record)


def _record_failure(
    conn: sqlite3.Connection, run_id: int, records: list[_StageRecord], failure: _StageFailure
) -> None:
    """Phase 2: after the rollback, persist the ledger in a fresh transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        for record in records:
            _write_stage_row(conn, run_id, record)
        conn.execute(
            "UPDATE runs SET status = 'error', error_text = ?, finished_at_utc = ? "
            "WHERE run_id = ?",
            (str(failure), utc_now_iso(), run_id),
        )
        conn.execute("COMMIT")
    except BaseException:
        # A failed COMMIT (SQLITE_FULL/IOERR) auto-rolls-back and leaves NO
        # open transaction — an unconditional ROLLBACK would then raise "no
        # transaction is active" and mask the original error.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


# --------------------------------------------------------------------------- #
# The six stages
# --------------------------------------------------------------------------- #


def _stage_freshness(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    current_date = stored_data_max_date(ctx.conn)
    current_fetched = stored_data_fetched_at(ctx.conn)
    last_ok = storage.latest_ok_run_id(ctx.conn)
    last_date = last_ok_data_max_date(ctx.conn, last_ok)
    last_fetched = last_ok_data_fetched_at(ctx.conn, last_ok)
    ctx.skip = should_skip_run(
        force=ctx.force,
        data_max_date=current_date,
        data_fetched_at=current_fetched,
        last_ok_run_id=last_ok,
        last_ok_data_max_date=last_date,
        last_ok_data_fetched_at=last_fetched,
    )
    detail: dict[str, object] = {
        "data_max_date": current_date,
        "data_fetched_at": current_fetched,
        "last_ok_run_id": last_ok,
        "last_ok_data_max_date": last_date,
        "last_ok_data_fetched_at": last_fetched,
        "force": ctx.force,
    }
    return ("skipped" if ctx.skip else "ok", detail)


def _stage_ingest(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    adapter = CacheAdapter(ctx.db_path)
    per_asset: dict[str, dict[str, object]] = {}
    for asset in UNIVERSE:
        frame = adapter.fetch(asset)
        ctx.frames[asset] = frame
        per_asset[asset] = {
            "rows": int(len(frame)),
            "first_date": str(frame.index[0]) if len(frame) else None,
            "last_date": str(frame.index[-1]) if len(frame) else None,
        }
    detail: dict[str, object] = {
        "source": adapter.source,
        "prices": per_asset,
        "macro": {
            "enabled": ctx.macro_status.enabled,
            "code": ctx.macro_status.code,
            "message": ctx.macro_status.message,
        },
    }
    return ("ok", detail)


def _stage_features(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for asset in UNIVERSE:
        returns = log_returns(ctx.frames[asset]["adj_close"])
        vol = realized_vol(returns)
        ctx.returns[asset] = returns
        ctx.features_frames[asset] = returns.to_frame()
        values = {
            LOG_RETURN_COLUMN: float(returns.iloc[-1]),
            REALIZED_VOL_COLUMN: float(vol.iloc[-1]),
        }
        for name, value in values.items():
            storage.insert_row(
                ctx.conn,
                "features",
                {"run_id": ctx.run_id, "asset": asset, "name": name, "value": value},
            )
        latest[asset] = {"date": str(returns.index[-1]), **values}
    detail: dict[str, object] = {
        "features": [LOG_RETURN_COLUMN, REALIZED_VOL_COLUMN],
        "latest": latest,
    }
    return ("ok", detail)


def _stage_forecast(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    ctx.forecasts = ctx.model.forecast(ctx.features_frames)
    per_asset: dict[str, dict[str, float]] = {}
    for asset in UNIVERSE:
        forecast = ctx.forecasts[asset]
        storage.insert_row(
            ctx.conn,
            "forecasts",
            {
                "run_id": ctx.run_id,
                "asset": asset,
                "horizon_days": HORIZON_BARS,
                "mu": forecast.mu,
                "sigma": forecast.sigma,
                "model_version": ctx.model.model_version,
            },
        )
        per_asset[asset] = {"mu": forecast.mu, "sigma": forecast.sigma}
    detail: dict[str, object] = {
        "model_version": ctx.model.model_version,
        "horizon_days": HORIZON_BARS,
        "forecasts": per_asset,
    }
    return ("ok", detail)


def _stage_blend(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    returns_frame = pd.concat(
        [ctx.returns[asset].rename(asset) for asset in UNIVERSE], axis=1, join="inner"
    ).sort_index()
    sigma_annual = ledoit_wolf_sigma(returns_frame)
    bl = bl_blend(sigma_annual, ctx.forecasts)
    ctx.bl = bl
    diag = bl.diagnostics
    for asset in UNIVERSE:
        row_detail = {
            "pi": diag["pi"][asset],
            "Q": diag["Q"][asset],
            "confidence": diag["confidences"][asset],
            "omega_diag": diag["omega_diag"][asset],
            "tilt": diag["tilt"][asset],
        }
        storage.insert_row(
            ctx.conn,
            "bl_posteriors",
            {
                "run_id": ctx.run_id,
                "asset": asset,
                "prior_mu": diag["pi"][asset],
                "view_mu": diag["Q"][asset],
                "view_confidence": diag["confidences"][asset],
                "posterior_mu": float(bl.mu_post[asset]),
                # Annualized posterior return std (sqrt of the posterior
                # covariance diagonal) — the scalar the card displays.
                "posterior_sigma": math.sqrt(float(bl.sigma_post.loc[asset, asset])),
                "detail_json": json.dumps(row_detail),
            },
        )
    detail: dict[str, object] = {
        "posterior_mu": {asset: float(bl.mu_post[asset]) for asset in UNIVERSE},
        "posterior_sigma": {
            asset: math.sqrt(float(bl.sigma_post.loc[asset, asset])) for asset in UNIVERSE
        },
        "confidences": diag["confidences"],
        "tilt": diag["tilt"],
    }
    return ("ok", detail)


def _stage_optimize(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    bl = ctx.bl
    if bl is None:  # pragma: no cover — stages run in order by construction
        raise RuntimeError("optimize stage reached without a blend result")
    w_prev = load_last_weights(ctx.conn)
    result = optimize_weights(bl.mu_post, bl.sigma_post, w_prev)
    for asset in UNIVERSE:
        storage.insert_row(
            ctx.conn,
            "target_weights",
            {
                "run_id": ctx.run_id,
                "asset": asset,
                "weight": result.weights[asset],
                "prev_weight": None if result.prev_weights is None else result.prev_weights[asset],
                "turnover": result.turnover[asset],
                "relaxed_turnover": int(result.relaxed_turnover),
            },
        )
    detail: dict[str, object] = {
        "weights": result.weights,
        "prev_weights": result.prev_weights,
        "turnover": result.turnover,
        "relaxed_turnover": result.relaxed_turnover,
        "solver_status": result.status,
        "cold_start": result.prev_weights is None,
    }
    return ("ok", detail)


_STAGE_FNS: Final[dict[str, _StageFn]] = {
    "freshness": _stage_freshness,
    "ingest": _stage_ingest,
    "features": _stage_features,
    "forecast": _stage_forecast,
    "blend": _stage_blend,
    "optimize": _stage_optimize,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_pipeline(
    conn: sqlite3.Connection,
    *,
    trigger: str = "manual",
    force: bool = False,
    model: WorldModel | None = None,
    macro_status: MacroStatus | None = None,
    on_run_started: Callable[[int], None] | None = None,
) -> int:
    """Execute one pipeline run on the writer connection; return its ``run_id``.

    Parameters
    ----------
    conn:
        THE single writer connection (``storage.open_writer``, autocommit
        mode; must be file-backed). This function owns the per-run
        ``BEGIN IMMEDIATE`` transaction — see the module docstring's two-phase
        design.
    trigger:
        Who started the run: one of :data:`TRIGGERS` (stored in ``runs``).
    force:
        Bypass the freshness gate (stage 0) and recompute unconditionally.
    model:
        The forecaster; ``None`` builds the default :class:`EWMABaseline`.
        Structurally asserted against the ``WorldModel`` protocol.
    macro_status:
        The startup macro probe result (Step 11's scheduler probes ONCE at
        startup and passes it in; the API lifespan does the same). ``None``
        resolves it here via ``probe_fred_key(load_fred_api_key())`` — with no
        key configured that makes NO network request; with a key it costs one
        FRED probe per call, which is why long-lived callers pass the cached
        status.
    on_run_started:
        Invoked with the ``run_id`` the moment the run has STARTED — the
        phase-0 ``runs`` row is committed (``status='running'``) and the id
        exists, but no stage has executed yet. Step 11's scheduler resolves
        its ``request_run`` handle here (the trigger endpoint returns at run
        START, not completion). The callback runs on the pipeline thread and
        must not raise: a raising callback is an infrastructure failure that
        propagates and leaves this run's ``'running'`` row for the startup
        sweep.

    Stage exceptions are recorded (``runs.status='error'`` + the failing
    stage's ``error`` row) and NEVER propagate; the ledger tells the story.
    Infrastructure failures (bad ``trigger``, in-memory db, the error-ledger
    write itself failing) raise.
    """
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger must be one of {TRIGGERS}, got {trigger!r}")
    resolved_model: WorldModel = model if model is not None else EWMABaseline()
    if not isinstance(resolved_model, WorldModel):
        raise TypeError(
            f"model {type(resolved_model).__qualname__} does not satisfy the "
            "WorldModel protocol (model_version + forecast)"
        )
    resolved_status = (
        macro_status if macro_status is not None else probe_fred_key(load_fred_api_key())
    )
    db_path = _database_file(conn)

    # Phase 0 (autocommit): open the run ledger row before the transaction so
    # the AUTOINCREMENT run_id survives any rollback.
    run_id_raw = storage.insert_row(
        conn,
        "runs",
        {"started_at_utc": utc_now_iso(), "status": "running", "trigger": trigger},
    )
    if run_id_raw is None:  # pragma: no cover — INSERT always yields a rowid
        raise RuntimeError("runs insert returned no run_id")
    run_id = int(run_id_raw)
    if on_run_started is not None:
        # The run has started: id allocated, 'running' row committed. The
        # Step 11 scheduler unblocks request_run awaiters here.
        on_run_started(run_id)

    ctx = _RunContext(
        conn=conn,
        run_id=run_id,
        db_path=db_path,
        force=force,
        model=resolved_model,
        macro_status=resolved_status,
    )
    records: list[_StageRecord] = []

    # Phase 1: the per-run transaction (storage docstring delegates it here).
    conn.execute("BEGIN IMMEDIATE")
    try:
        _execute_stage(ctx, records, "freshness", _STAGE_FNS["freshness"])
        if not ctx.skip:
            for stage in STAGES[1:]:
                _execute_stage(ctx, records, stage, _STAGE_FNS[stage])
        final_status = "skipped" if ctx.skip else "ok"
        conn.execute(
            "UPDATE runs SET status = ?, finished_at_utc = ? WHERE run_id = ?",
            (final_status, utc_now_iso(), run_id),
        )
        conn.execute("COMMIT")
    except _StageFailure as failure:
        # Phase 2: rollback data writes, then persist the ledger afresh. The
        # in_transaction guard matters when the failure WAS the COMMIT: sqlite
        # auto-rolls-back on SQLITE_FULL/IOERR, and an unconditional ROLLBACK
        # would raise "no transaction is active", masking the original error
        # and skipping phase 2 entirely.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        logger.error("run %d failed: %s", run_id, failure)
        _record_failure(conn, run_id, records, failure)
    except BaseException:
        # Non-stage failure (or a hard interrupt): leave no open transaction.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return run_id
