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
  ``source`` (who served this run) + ``price_provider`` (the human-readable
  ultimate upstream, ``PRICE_PROVIDER_LABEL``), plus the macro block
  ``{enabled, code, message}`` — degraded mode (``MACRO_DISABLED_*``) surfaces
  here as a card fact, NOT an error.
- ``features``: feature names + a ``windows`` map (per-feature lookback, e.g.
  ``realized_vol`` = ``{HORIZON_BARS} days, annualized``) + per-asset latest
  date/values (rows persisted for the LATEST bar only in V1 — the card shows
  current features).
- ``forecast``: ``model_version``, ``horizon_days``, per-asset ``(mu, sigma)``.
- ``blend``: the equilibrium ``prior`` (pi, all assets) and forecast ``view``
  (Q, viewed assets) surfaced from ``BLResult.diagnostics`` (no new math),
  then posterior mu AND posterior sigma per asset + confidences + tilt (the
  card shows prior -> view -> posterior, mu ± sigma). Each ``bl_posteriors``
  row stores the per-asset slice of ``BLResult.diagnostics`` in its own
  ``detail_json``; ``posterior_sigma`` is ``sqrt(sigma_post[asset, asset])`` —
  the annualized posterior return std.
- ``optimize``: weights, prev_weights, turnover, ``relaxed_turnover``, solver
  status, cold_start, plus an ``objective`` block (the solved form + the real
  ``delta``/``gamma_tc``/``w_max`` constants + constraints) for the card.
  ``w_prev`` is the last persisted allocation of the latest ok run
  (:func:`load_last_weights`); ``None`` = cold start.

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

from abe import config as config_module
from abe import registry, storage
from abe.blend.black_litterman import BLResult, bl_blend
from abe.blend.covariance import ledoit_wolf_sigma
from abe.constants import HORIZON_BARS, UNIVERSE
from abe.ingest.macro import MacroStatus, load_fred_api_key, probe_fred_key
from abe.ingest.sources import PRICE_PROVIDER_LABEL, CacheAdapter, utc_now_iso
from abe.model.base import Forecast, WorldModel

__all__ = [
    "STAGES",
    "TRIGGERS",
    "cached_config_run",
    "last_ok_data_fetched_at",
    "last_ok_data_max_date",
    "latest_ok_central_run_id",
    "latest_ok_run_id_for_config",
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


def latest_ok_run_id_for_config(conn: sqlite3.Connection, config_id: int) -> int | None:
    """``MAX(run_id) WHERE status='ok' AND config_id=?`` — the latest cached run
    for one Config (the per-config analogue of ``storage.latest_ok_run_id``)."""
    row = conn.execute(
        "SELECT MAX(run_id) FROM runs WHERE status = 'ok' AND config_id = ?", (config_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def latest_ok_central_run_id(conn: sqlite3.Connection) -> int | None:
    """The CENTRAL config's latest ok run — the UI's central-answer poll target.

    Scoped to the central config so an on-demand non-central run (which lands a
    higher ``run_id``) can NEVER hijack ``/api/runs/latest`` (Step 21; the plan's
    "the central answer stays unambiguous" invariant)."""
    central = config_module.get_central_config(conn)
    return latest_ok_run_id_for_config(conn, central.config_id)


def cached_config_run(conn: sqlite3.Connection, config_id: int) -> int | None:
    """The on-demand cache gate (Step 21): the cached ok run_id for ``config_id``
    IFF the current stored data matches the watermarks that run recorded, else
    ``None`` (recompute needed).

    Reuses the freshness watermarks (``data_max_date`` + ``data_fetched_at``)
    keyed to THIS config's latest ok run — so a second on-demand request on
    unchanged data is served from cache, while new/revised prices force a
    recompute. Mirrors :func:`should_skip_run`'s two-watermark rule, per-config.

    Recipe freshness (Step 25): a config edited AFTER its last cached run makes
    that run stale (it was computed under the OLD recipe), so ``updated_at_utc >
    run.started_at_utc`` forces a recompute even on unchanged data. Residual gap:
    both stamps are 1-second-resolution, so an edit within the SAME second as the
    last run's start is not detected (``>`` is strict — ``>=`` would thrash a run
    that legitimately used the new recipe that second); negligible for a human
    operator since the compute itself spans seconds."""
    last = latest_ok_run_id_for_config(conn, config_id)
    if last is None:
        return None
    run_row = conn.execute(
        "SELECT started_at_utc FROM runs WHERE run_id = ?", (last,)
    ).fetchone()
    cfg_row = conn.execute(
        "SELECT updated_at_utc FROM configs WHERE config_id = ?", (config_id,)
    ).fetchone()
    run_started = None if run_row is None or run_row[0] is None else str(run_row[0])
    updated_at = None if cfg_row is None or cfg_row[0] is None else str(cfg_row[0])
    if run_started is not None and updated_at is not None and updated_at > run_started:
        return None  # recipe edited since the cached run — stale
    current_date = stored_data_max_date(conn)
    current_fetched = stored_data_fetched_at(conn)
    cached_date = last_ok_data_max_date(conn, last)
    cached_fetched = last_ok_data_fetched_at(conn, last)
    if (
        current_date is not None
        and current_date == cached_date
        and current_fetched is not None
        and current_fetched == cached_fetched
    ):
        return last
    return None


def load_last_weights(conn: sqlite3.Connection, config_id: int) -> dict[str, float] | None:
    """The last persisted allocation FOR THIS CONFIG: ``target_weights`` of the
    config's latest ok run (the turnover baseline).

    ``None`` when this config has no ok run yet (or, defensively, when it has no
    weight rows) — the optimizer's cold start, which DROPS the turnover term.
    Scoped by ``config_id`` so an on-demand alt run never shifts the central
    config's turnover baseline (Step 21)."""
    last_ok = latest_ok_run_id_for_config(conn, config_id)
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
    config_id: int
    db_path: Path
    force: bool
    stack: registry.ResolvedStack
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
    # Per-config freshness: compare against THIS config's own last ok run so an
    # interleaved on-demand run of a DIFFERENT config never makes the central
    # loop skip its recompute (Step 21).
    last_ok = latest_ok_run_id_for_config(ctx.conn, ctx.config_id)
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
        # Human-readable provenance for the card: `source` records WHO served
        # this run (cache), `price_provider` names the ultimate upstream (Yahoo)
        # regardless of the cache hop. One source of truth (ingest.sources).
        "price_provider": PRICE_PROVIDER_LABEL,
        "prices": per_asset,
        "macro": {
            "enabled": ctx.macro_status.enabled,
            "code": ctx.macro_status.code,
            "message": ctx.macro_status.message,
        },
    }
    return ("ok", detail)


def _load_macro_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Long-form macro table for the feature builder's lookahead-free join.

    Columns match ``build_features``'s join keys (``series_id``, ``obs_date``,
    ``value``, ``available_date``). An empty frame (no macro stored — degraded
    mode or pre-backfill) makes ``build_features`` omit macro columns, never
    error. The ``basic`` builder ignores the frame entirely (parity)."""
    rows = conn.execute(
        "SELECT series_id, obs_date, value, available_date FROM macro"
    ).fetchall()
    return pd.DataFrame(rows, columns=["series_id", "obs_date", "value", "available_date"])


def _stage_features(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    # The resolved feature builder owns the math + the card detail; the pipeline
    # owns persistence (adding run_id) and threading returns/frames downstream.
    # Macro is loaded only when enabled; `basic` ignores it (byte-identical V1).
    macro = _load_macro_frame(ctx.conn) if ctx.macro_status.enabled else None
    bundle = ctx.stack.feature_builder.build(ctx.frames, macro)
    ctx.returns = dict(bundle.returns)
    ctx.features_frames = dict(bundle.features_frames)
    for row in bundle.rows:
        storage.insert_row(ctx.conn, "features", {"run_id": ctx.run_id, **row})
    return ("ok", bundle.detail)


def _stage_forecast(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    forecaster = ctx.stack.forecaster
    ctx.forecasts = forecaster.forecast(ctx.features_frames)
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
                "model_version": forecaster.model_version,
            },
        )
        per_asset[asset] = {"mu": forecast.mu, "sigma": forecast.sigma}
    detail: dict[str, object] = {
        "model_version": forecaster.model_version,
        "horizon_days": HORIZON_BARS,
        "forecasts": per_asset,
    }
    return ("ok", detail)


def _stage_blend(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    returns_frame = pd.concat(
        [ctx.returns[asset].rename(asset) for asset in UNIVERSE], axis=1, join="inner"
    ).sort_index()
    sigma_annual = ledoit_wolf_sigma(returns_frame)
    # The resolved view source produces {asset: Forecast} views; bl_blend
    # consumes them identically regardless of origin. For the `forecast` source
    # (the V1 central) this is the model's forecasts unchanged (parity).
    views = ctx.stack.view_source.provide(
        registry.ViewContext(forecasts=ctx.forecasts, returns=ctx.returns)
    )
    bl = bl_blend(sigma_annual, views)
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
        # Prior (pi, all assets) and view (Q, viewed assets) come straight from
        # the diagnostics the blend already computes — surfacing them (not new
        # math) so the card shows prior -> view -> posterior, the BL story.
        "prior": diag["pi"],
        "view": diag["Q"],
        "posterior_mu": {asset: float(bl.mu_post[asset]) for asset in UNIVERSE},
        "posterior_sigma": {
            asset: math.sqrt(float(bl.sigma_post.loc[asset, asset])) for asset in UNIVERSE
        },
        "confidences": diag["confidences"],
        "tilt": diag["tilt"],
        # Covariance is fit on the COMMON history: the inner join above aligns
        # the varying-length per-asset series to the window all assets share,
        # so THIS is where the truncation happens (ACWI's 2008 inception is the
        # binding start). Surfaced so the card can say so.
        "covariance_window": {
            "start": str(returns_frame.index[0]),
            "end": str(returns_frame.index[-1]),
            "bars": int(len(returns_frame)),
        },
    }
    return ("ok", detail)


def _stage_optimize(ctx: _RunContext) -> tuple[str, dict[str, object]]:
    bl = ctx.bl
    if bl is None:  # pragma: no cover — stages run in order by construction
        raise RuntimeError("optimize stage reached without a blend result")
    w_prev = load_last_weights(ctx.conn, ctx.config_id)
    result = ctx.stack.optimizer.optimize(bl.mu_post, bl.sigma_post, w_prev)
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
        # The RESOLVED optimizer owns its objective card detail — so a min-variance
        # config shows "minimize w^T Sigma w", not the MVU form. For the central
        # (mvu, min_weight=0) this is byte-identical to V1 (Step 19 parity golden).
        "objective": ctx.stack.optimizer.objective(),
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
    config: config_module.Config | None = None,
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
    config:
        The :class:`~abe.config.Config` to run (resolved into the concrete stage
        stack via :mod:`abe.registry`; the run is tagged with its ``config_id``).
        ``None`` re-reads the db's central Config each run (so a ``set_central``
        takes effect on the very next tick — the always-on loop runs central).
    trigger:
        Who started the run: one of :data:`TRIGGERS` (stored in ``runs``).
    force:
        Bypass the freshness gate (stage 0) and recompute unconditionally.
    model:
        Back-compat forecaster OVERRIDE (the seam tests inject through and the
        ``ABE_MODEL`` env toggle sets). When given, it replaces the resolved
        Config's forecaster (structurally asserted against ``WorldModel``);
        ``None`` uses the Config's own forecaster. The other three stages always
        come from the resolved Config.
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

    # Resolve the Config (explicit, else the db's central) + its ViewScenario
    # into the concrete stage stack. Reading central per-run (not freezing it at
    # startup) means set_central takes effect on the next tick without a restart.
    resolved_config = config if config is not None else config_module.get_central_config(conn)
    view_scenario = config_module.get_view_scenario(conn, resolved_config.view_scenario_id)
    if view_scenario is None:
        raise ValueError(
            f"config {resolved_config.config_id!r} references missing "
            f"view_scenario {resolved_config.view_scenario_id!r}"
        )
    if model is not None and not isinstance(model, WorldModel):
        raise TypeError(
            f"model {type(model).__qualname__} does not satisfy the "
            "WorldModel protocol (model_version + forecast)"
        )
    # A model override supersedes the config's forecaster: resolve() then skips
    # building it entirely (so a broken central forecaster can't break an
    # override run, and no checkpoint is loaded only to be discarded).
    stack = registry.resolve(resolved_config, view_scenario, forecaster_override=model)

    resolved_status = (
        macro_status if macro_status is not None else probe_fred_key(load_fred_api_key())
    )
    db_path = _database_file(conn)

    # Phase 0 (autocommit): open the run ledger row before the transaction so
    # the AUTOINCREMENT run_id survives any rollback; tag it with config_id.
    run_id_raw = storage.insert_row(
        conn,
        "runs",
        {
            "started_at_utc": utc_now_iso(),
            "status": "running",
            "trigger": trigger,
            "config_id": resolved_config.config_id,
        },
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
        config_id=resolved_config.config_id,
        db_path=db_path,
        force=force,
        stack=stack,
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
