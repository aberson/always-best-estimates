"""Always-on scheduler: the asyncio lifespan task that owns the pipeline.

plan.md Step 11, section 5 (module description), section 8 (fetch-split
decision) and section 9 (silent-task-death + laptop-sleep risks) are the
authoritative spec. :class:`Scheduler` is created and started by ``api.py``'s
lifespan; it owns THE writer connection and the single-worker executor for the
whole process lifetime.

Concurrency design (one loop, one executor thread, one writer connection)
==========================================================================

- **One asyncio task** (:meth:`Scheduler._run_loop`) drives everything:
  the fixed-delay 5-minute recompute, on-demand triggers, and the daily fetch.
  A single sequential task means the writer connection can never be used by
  two jobs at once — single-flight holds by construction, not by locking.
- **One executor thread** (``ThreadPoolExecutor(max_workers=1)``) executes
  every DB-touching body: the startup sweep, each pipeline run, the WAL
  checkpoint, the daily fetch, and the final connection close. The writer
  connection is CREATED on that thread (:meth:`Scheduler.start` submits
  ``_init_on_executor``) so sqlite3's default ``check_same_thread=True`` guard
  stays intact for its entire life — every later use is dispatched through
  the same single-worker executor, and the FIFO queue guarantees ordering.
- **Fixed-DELAY loop**, not fixed-rate: each iteration waits
  ``recompute_interval_s`` measured FROM THE PREVIOUS RUN'S COMPLETION
  (``wait_for(trigger_event.wait(), timeout=...)``). After a laptop-sleep gap
  the next wait simply starts when the machine wakes — no backlog of missed
  ticks to drain, no drifting schedule to reconcile (plan section 9's
  laptop-sleep row). A timeout means a scheduled recompute; the event firing
  means an on-demand run.

``request_run`` semantics (the section 6 trigger contract)
==========================================================

``request_run(force) -> PendingRun`` awaits the run's START — the pipeline's
phase-0 ``runs`` row is committed and the ``run_id`` exists — but NOT its
completion (``run_pipeline``'s ``on_run_started`` callback resolves the
handle via ``call_soon_threadsafe``). Coalescing (single-flight):

- **Idle:** the trigger event is set, the loop wakes immediately, and the
  caller gets ``PendingRun(run_id, already_running=False)`` as soon as the
  new run's id exists. Multiple triggers arriving before the run starts
  coalesce into that ONE run (each caller's request IS that run, so each gets
  ``already_running=False`` with the same id); their force flags OR together.
- **A run already executing** (or its loop iteration is mid-daily-fetch): NO
  second run is queued. The caller gets the ACTIVE iteration's run handle
  with ``already_running=True``; a coalesced request's ``force`` flag is
  DROPPED by design — plan section 6's "coalesces if a run is active".
- An infrastructure failure that prevents the run from starting (broken
  executor, locked db at phase 0) raises out of ``request_run`` — the
  endpoint answers 500, loudly. The loop itself logs and CONTINUES.

Startup / shutdown
==================

:meth:`Scheduler.start` (in lifespan startup) opens the writer connection ON
the executor thread, then sweeps orphaned ``status='running'`` runs to
``status='error'`` (:data:`SWEEP_ERROR_TEXT`) — the sweep owed from Step 8:
a hard process kill or an infra failure that defeated even the phase-2 error
ledger leaves a ``'running'`` row this process did not write, and only
startup (when no run can be executing) can safely reclassify it. The loop's
FIRST iteration runs immediately with ``trigger='startup'`` (plan section 10:
"a run fires immediately at startup"); subsequent iterations use
``'schedule'`` (timeout) or ``'manual'`` (event).

:meth:`Scheduler.stop` (lifespan exit) cancels the loop task, then submits
the connection close TO THE EXECUTOR: with one FIFO worker the close queues
behind any still-running pipeline body, so the drain happens by construction
and the close runs on the thread that owns the connection. Finally the
executor itself shuts down.

Error resilience (plan section 9's "silent asyncio task death")
===============================================================

``run_pipeline`` already converts stage exceptions into ``status='error'``
ledger rows and returns. The loop body additionally wraps EVERYTHING in a
per-iteration try/except: an infrastructure exception (locked db, executor
failure) is logged — and delivered to any ``request_run`` awaiter — and the
loop CONTINUES to the next scheduled tick. The loop exits only via
cancellation at shutdown.

Daily fetch (plan section 8 fetch-split)
========================================

The 5-minute recompute reads SQLite only. Network ingest is a separate DAILY
job (once per UTC day, after ``daily_fetch_hour_utc``, default 22:00 UTC ≈
after US close) that this same loop dispatches at the top of an iteration —
BEFORE the recompute, so the very same iteration's run picks the new bars up
(the freshness watermarks move). Running it inside the one sequential loop
(rather than a second task) keeps the single-flight/one-writer property
structural. Details:

- **Persistent already-ran marker:** ``MAX(fetched_at_utc)`` in ``prices`` —
  if its UTC date >= today's, the fetch already happened (Step 3's ingest
  stamps every upsert). An EMPTY prices table means the fetch never ran, so
  it fires (incremental ingest on an empty table IS the full backfill).
- **In-memory attempt marker:** a failing fetch is logged and retried NEXT
  UTC day, never on the next 5-minute tick — a Yahoo outage must not become
  a hammering retry storm. The marker is process-local; a restart re-arms the
  attempt, and the persistent marker still prevents double-fetching after a
  success.
- **Partial failure semantics:** prices and macro run sequentially in one
  attempt; if prices land but macro raises, the attempt marker (set before
  the fetch) still defers the WHOLE retry to the next UTC day. In the interim
  prices are fresh (their ``fetched_at_utc`` watermark moved) while macro lags
  by up to one UTC day — an accepted staleness bound, not a silent gap. That
  next-day retry is safe and cheap: the price half is an idempotent
  incremental no-op (the persistent marker only skips the attempt when its
  date is already today's), and macro then gets its turn.
- Macro (``ingest_macro``) runs only when the startup probe enabled it
  (``macro_status.enabled``); without a FRED key no client is ever built.
- The fetch is NOT a pipeline run: it records nothing in ``runs`` (the ledger
  stays meaningful, plan section 8) but logs loudly either way.

Stale-'running' sweeps
======================

Two sweeps share :data:`SWEEP_ERROR_TEXT` (one stable ledger contract):

- **Startup sweep** (in :meth:`Scheduler.start`, before the loop exists) —
  reclassifies rows orphaned by a previous process.
- **Iteration sweep** (at the top of every dispatch, on the executor thread)
  — closes the always-on gap: an infra failure that defeats even phase 2's
  error ledger (disk full mid-run) would otherwise leave a phantom
  ``'running'`` row until the next process restart. Provably safe: the
  single sequential loop + single executor worker guarantee NO run can be
  executing when a dispatch begins, so any ``'running'`` row seen there is
  orphaned by definition.

Test seams (documented, test-only — like ``optimize.mvu._solve_once_fn``)
=========================================================================

:class:`SchedulerConfig` is the complete seam list: ``pipeline_fn`` (stand-in
for ``run_pipeline``), ``price_adapter_factory`` / ``fred_client_factory``
(fake network sources), ``clock`` (UTC now), ``recompute_interval_s``,
``daily_fetch_hour_utc``, ``daily_fetch_enabled``, ``model``, and
``startup_sweep_fn`` (force the startup sweep to raise — proves a failed sweep
cannot leak the writer connection). Production code never passes a config
override; the defaults ARE the production wiring. ``model`` is the one field
that doubles as a FUTURE production knob — Step 13's EWMA<->JEPA toggle will
inject the ``WorldModel`` here; until then only tests set it.
"""

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

from abe import config as config_module
from abe import storage
from abe.ingest.macro import (
    FredApiClient,
    FredClient,
    MacroStatus,
    ingest_macro,
    load_fred_api_key,
)
from abe.ingest.prices import ingest_prices
from abe.ingest.sources import SourceAdapter, YFinanceAdapter, utc_now_iso
from abe.model.base import WorldModel
from abe.pipeline import cached_config_run, run_pipeline, stored_data_fetched_at

__all__ = [
    "DAILY_FETCH_HOUR_UTC",
    "RECOMPUTE_INTERVAL_S",
    "SWEEP_ERROR_TEXT",
    "CentralConfigRunError",
    "ConfigRun",
    "PendingRun",
    "PipelineFn",
    "Scheduler",
    "SchedulerConfig",
]


class CentralConfigRunError(ValueError):
    """The on-demand path was asked to run the CENTRAL config. The central config
    is the portfolio you'd buy — the always-on loop owns it (use ``request_run``
    / ``POST /api/runs/trigger``); running it on the on-demand path would bypass
    the loop's model override and land a redundant central run."""

logger = logging.getLogger(__name__)

RECOMPUTE_INTERVAL_S: Final[float] = 300.0
"""Fixed DELAY between run completion and the next scheduled recompute."""

DAILY_FETCH_HOUR_UTC: Final[int] = 22
"""Earliest UTC hour for the daily network fetch (~after US market close)."""

SWEEP_ERROR_TEXT: Final[str] = "orphaned by process kill/infra failure; swept at startup"
"""``runs.error_text`` stamped on stale ``'running'`` rows by BOTH sweeps
(startup + iteration — module docstring); one stable ledger contract."""


class PipelineFn(Protocol):
    """The shape of ``run_pipeline`` (the scheduler's injectable pipeline seam).

    Test-only injection point (module docstring); production always uses
    ``abe.pipeline.run_pipeline``. Implementations run ON the executor thread
    with THE writer connection and must return the run's id after allocating
    it through the ``runs`` table (and must invoke ``on_run_started`` with it
    as soon as it exists).
    """

    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        trigger: str,
        force: bool,
        model: WorldModel | None,
        macro_status: MacroStatus | None,
        on_run_started: Callable[[int], None] | None,
    ) -> int: ...


@dataclass(frozen=True)
class PendingRun:
    """What ``request_run`` resolves to: the run's id at its START.

    ``already_running=True`` means the request coalesced into an
    already-executing iteration (no second run was queued) — the section 6
    ``202 {run_id, already_running: true}`` response.
    """

    run_id: int
    already_running: bool


@dataclass(frozen=True)
class ConfigRun:
    """What :meth:`Scheduler.request_config_run` resolves to (Step 21).

    ``cached=True`` means the current data matched the config's last ok run's
    watermarks, so that cached ``run_id`` is returned WITHOUT a recompute;
    ``cached=False`` means a fresh run was computed and tagged with ``config_id``.
    """

    run_id: int
    config_id: int
    cached: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_fred_client() -> FredClient:
    """Production FRED client factory (daily macro fetch).

    Only ever invoked when ``macro_status.enabled`` — reaching it without a
    key is a bug, not a degraded mode, so it raises.
    """
    key = load_fred_api_key()
    if key is None:
        raise RuntimeError(
            "daily macro fetch dispatched without a FRED_API_KEY — "
            "macro_status.enabled should have prevented this"
        )
    return FredApiClient(key)


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler knobs. The defaults ARE the production wiring; any field an
    instantiation sets explicitly is a documented test seam (module
    docstring) — except ``model``, which Step 13's EWMA<->JEPA toggle will
    also set in production."""

    recompute_interval_s: float = RECOMPUTE_INTERVAL_S
    daily_fetch_enabled: bool = True
    daily_fetch_hour_utc: int = DAILY_FETCH_HOUR_UTC
    model: WorldModel | None = None
    pipeline_fn: PipelineFn | None = None
    price_adapter_factory: Callable[[], SourceAdapter] | None = None
    fred_client_factory: Callable[[], FredClient] | None = None
    clock: Callable[[], datetime] | None = None
    startup_sweep_fn: Callable[[sqlite3.Connection], int] | None = None


def _resolve_started(future: "asyncio.Future[int]", run_id: int) -> None:
    """Resolve a request_run handle exactly once (idempotent)."""
    if not future.done():
        future.set_result(run_id)


class Scheduler:
    """The lifespan-owned always-on engine (module docstring for the design)."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        macro_status: MacroStatus,
        config: SchedulerConfig | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._macro_status = macro_status
        self._config = config if config is not None else SchedulerConfig()
        cfg = self._config
        self._pipeline_fn: PipelineFn = (
            cfg.pipeline_fn if cfg.pipeline_fn is not None else run_pipeline
        )
        self._price_adapter_factory: Callable[[], SourceAdapter] = (
            cfg.price_adapter_factory if cfg.price_adapter_factory is not None else YFinanceAdapter
        )
        self._fred_client_factory: Callable[[], FredClient] = (
            cfg.fred_client_factory if cfg.fred_client_factory is not None else _default_fred_client
        )
        self._now_utc: Callable[[], datetime] = cfg.clock if cfg.clock is not None else _utc_now
        # Startup-sweep seam (module docstring): defaults to the production
        # sweep, so every other test exercises the real path. Only the
        # leak-safety test injects a raising stand-in here.
        self._startup_sweep_fn: Callable[[sqlite3.Connection], int] = (
            cfg.startup_sweep_fn if cfg.startup_sweep_fn is not None else self._sweep_stale_running
        )
        self._wake_event = asyncio.Event()
        self._pending_force = False
        self._run_in_progress = False
        self._last_fetch_attempt_date: str | None = None
        self._conn: sqlite3.Connection | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._started_future: asyncio.Future[int] | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Open the writer conn ON the executor thread, sweep, start the loop.

        Returns after the sweep completed and the loop task exists; the
        startup run executes asynchronously (the loop's first iteration).
        """
        if self._task is not None:
            raise RuntimeError("scheduler already started")
        self._loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="abe-writer")
        try:
            swept = await self._loop.run_in_executor(self._executor, self._init_on_executor)
        except BaseException:
            # Mirror storage.open_writer's discipline: never leak a
            # half-started scheduler (the executor thread would outlive us).
            self._executor.shutdown(wait=True)
            self._executor = None
            self._loop = None
            raise
        if swept:
            logger.warning(
                "startup sweep: marked %d orphaned 'running' run(s) as 'error' (%r)",
                swept,
                SWEEP_ERROR_TEXT,
            )
        self._started_future = self._loop.create_future()
        self._task = self._loop.create_task(self._run_loop(), name="abe-scheduler")

    async def stop(self) -> None:
        """Cancel the loop, drain the executor, close the conn on ITS thread."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        started = self._started_future
        if started is not None and not started.done():
            # Unblock any request_run awaiter left behind by the cancellation.
            started.set_exception(RuntimeError("scheduler stopped before the run started"))
            started.exception()  # mark retrieved: no GC warning when nobody awaits
        self._started_future = None
        executor = self._executor
        conn = self._conn
        if conn is not None and executor is not None:
            # max_workers=1 + FIFO: this queues BEHIND any pipeline body
            # the cancelled task left running, so the close both drains
            # the executor and runs on the thread that owns the conn.
            await self._require_loop().run_in_executor(executor, conn.close)
            self._conn = None
        elif conn is not None:
            # Shouldn't be reachable (_init_on_executor never publishes a conn
            # without a live executor), but a leaked WAL handle pins the db
            # file on Windows — attempt a direct close as defense in depth.
            # check_same_thread may veto it (ProgrammingError); log either way.
            logger.warning("closing writer connection without an executor (unexpected state)")
            try:
                conn.close()
            except sqlite3.ProgrammingError:
                logger.exception("cross-thread writer close failed; handle may leak")
            self._conn = None
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None
        self._loop = None

    @property
    def run_in_progress(self) -> bool:
        """True while a loop iteration (daily fetch + run dispatch) is active.

        Read-only observability — tests (and the smoke gate) use it to wait
        for the startup run to fully settle before triggering.
        """
        return self._run_in_progress

    # ------------------------------------------------------------------ #
    # The on-demand trigger (section 6 contract)
    # ------------------------------------------------------------------ #

    async def request_run(self, force: bool = False) -> PendingRun:
        """Request a run NOW; resolve at the run's START (module docstring).

        Coalesces during an active iteration (``already_running=True``, force
        dropped); otherwise wakes the loop and awaits the new run's id.
        Raises if the scheduler is not running, or if an infrastructure
        failure prevented the run from starting.
        """
        task = self._task
        started = self._started_future
        if task is None or task.done() or started is None:
            raise RuntimeError("scheduler is not running (start() not called, or stopped)")
        if self._run_in_progress:
            run_id = await asyncio.shield(started)
            return PendingRun(run_id=run_id, already_running=True)
        self._pending_force = self._pending_force or force
        self._wake_event.set()
        run_id = await asyncio.shield(started)
        return PendingRun(run_id=run_id, already_running=False)

    async def request_config_run(self, config_id: int, *, force: bool = False) -> ConfigRun:
        """Compute a run for a NON-central Config on demand (Step 21).

        Dispatched through the SAME single-worker executor as the always-on loop
        (FIFO), so it serializes behind / ahead of the central recompute — the
        one-writer, single-flight discipline is preserved by construction (no new
        connection, no second writer thread). ``force=False`` first consults the
        per-config cache (:func:`abe.pipeline.cached_config_run`): unchanged data
        since this config's last ok run returns that run WITHOUT recomputing.
        Otherwise a fresh run is computed and tagged with ``config_id``.

        Raises ``ValueError`` (unknown config) or ``RuntimeError`` (scheduler not
        running). Blocks until the run completes (or the cache hit resolves) —
        an on-demand comparison is a deliberate, awaited action.
        """
        task = self._task
        if task is None or task.done():
            raise RuntimeError("scheduler is not running (start() not called, or stopped)")
        executor = self._require_executor()
        loop = self._require_loop()

        def _config_job() -> ConfigRun:
            conn = self._require_conn()
            cfg = config_module.get_config(conn, config_id)
            if cfg is None:
                raise ValueError(f"no config with id {config_id}")
            if cfg.is_central:
                # Central belongs to the loop (which applies the model override);
                # running it here would bypass that and land a redundant run.
                raise CentralConfigRunError(
                    f"config {config_id} is central; trigger it via the always-on loop "
                    "(POST /api/runs/trigger), not the on-demand config-run path"
                )
            if not force:
                cached = cached_config_run(conn, config_id)
                if cached is not None:
                    return ConfigRun(run_id=cached, config_id=config_id, cached=True)
            # force=True bypasses the pipeline's GLOBAL freshness gate (which is
            # central-oriented); the per-config cache check above is the gate for
            # on-demand runs.
            run_id = run_pipeline(
                conn, config=cfg, trigger="manual", force=True, macro_status=self._macro_status
            )
            busy, log_frames, checkpointed = storage.wal_checkpoint_truncate(conn)
            if busy:
                logger.warning(
                    "wal_checkpoint(TRUNCATE) after config run %d was blocked (busy=%d): "
                    "%d of %d WAL frames checkpointed",
                    run_id,
                    busy,
                    checkpointed,
                    log_frames,
                )
            return ConfigRun(run_id=int(run_id), config_id=config_id, cached=False)

        return await asyncio.shield(loop.run_in_executor(executor, _config_job))

    async def run_write[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run a write ``fn(conn)`` on THE single writer thread (Step 25).

        Config/scenario CRUD + set-central dispatch through here so the API never
        opens a SECOND writer connection that would contend with the always-on
        loop for the WAL write lock — one-writer discipline preserved by
        construction. Returns ``fn``'s result; ``fn``'s exceptions (IntegrityError,
        ValueError, ...) propagate to the caller for HTTP mapping."""
        task = self._task
        if task is None or task.done():
            raise RuntimeError("scheduler is not running (start() not called, or stopped)")
        executor = self._require_executor()
        loop = self._require_loop()

        def _job() -> T:
            return fn(self._require_conn())

        return await asyncio.shield(loop.run_in_executor(executor, _job))

    # ------------------------------------------------------------------ #
    # Executor-thread bodies
    # ------------------------------------------------------------------ #

    def _init_on_executor(self) -> int:
        """Runs ON the executor thread: create THE writer connection there
        (sqlite3 ``check_same_thread`` ownership) and sweep orphaned
        ``'running'`` rows. Returns the number of rows swept.

        A raising sweep closes the just-opened connection before re-raising
        (mirroring ``storage.open_writer``'s own idiom) — api.py's "a failed
        startup can never leak open WAL handles" invariant must hold here
        too, or the leaked handle pins the db file on Windows.
        """
        conn = storage.open_writer(self._db_path)
        try:
            swept = self._startup_sweep_fn(conn)
        except BaseException:
            conn.close()
            raise
        self._conn = conn
        return swept

    @staticmethod
    def _sweep_stale_running(conn: sqlite3.Connection) -> int:
        """Mark every ``'running'`` row ``'error'`` (:data:`SWEEP_ERROR_TEXT`).

        Executor-thread only. Safe at startup (no loop yet) AND at the top of
        each dispatch (single-flight: nothing can be running then) — the two
        sweeps in the module docstring.
        """
        cursor = conn.execute(
            "UPDATE runs SET status = 'error', error_text = ?, finished_at_utc = ? "
            "WHERE status = 'running'",
            (SWEEP_ERROR_TEXT, utc_now_iso()),
        )
        return int(cursor.rowcount)

    def _require_conn(self) -> sqlite3.Connection:
        conn = self._conn
        if conn is None:  # pragma: no cover — init runs before any job by FIFO order
            raise RuntimeError("scheduler writer connection is not open")
        return conn

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None:  # pragma: no cover — set in start() before any use
            raise RuntimeError("scheduler event loop is not set (start() not called)")
        return loop

    def _require_executor(self) -> ThreadPoolExecutor:
        executor = self._executor
        if executor is None:  # pragma: no cover — set in start() before any use
            raise RuntimeError("scheduler executor is not running (start() not called)")
        return executor

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self) -> None:
        """Fixed-delay recompute loop + daily fetch; exits only on cancellation."""
        loop = self._require_loop()
        first = True
        while True:
            if first:
                trigger = "startup"  # plan section 10: run immediately at startup
                first = False
            else:
                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(), timeout=self._config.recompute_interval_s
                    )
                    trigger = "manual"
                except TimeoutError:
                    trigger = "schedule"
            # Consume the wake state and mark the WHOLE iteration busy before
            # the first await, so a trigger can never slip into the gap
            # between consume and dispatch — it coalesces instead.
            self._wake_event.clear()
            force = self._pending_force
            self._pending_force = False
            self._run_in_progress = True
            try:
                try:
                    await self._daily_fetch_if_due()
                except Exception:
                    # Fetch failures never stop the recompute; the in-memory
                    # attempt marker defers the retry to the next UTC day.
                    logger.exception("daily fetch failed; retrying next UTC day")
                await self._dispatch_run(trigger, force)
            except asyncio.CancelledError:
                raise  # shutdown: the ONLY way out of the loop
            except Exception as exc:
                # Infrastructure failure (locked db, executor breakage): the
                # run never started or died outside run_pipeline's own
                # error-ledger handling. Log, deliver to any awaiter, and
                # CONTINUE — the next scheduled run must still fire (plan
                # section 9 silent-task-death row).
                logger.exception("scheduler iteration failed (infrastructure): %s", exc)
                started = self._started_future
                if started is not None and not started.done():
                    started.set_exception(RuntimeError(f"pipeline run failed to start: {exc}"))
                    started.exception()  # mark retrieved: no GC warning sans awaiters
            finally:
                self._run_in_progress = False
                started = self._started_future
                if started is not None and started.done():
                    self._started_future = loop.create_future()

    async def _dispatch_run(self, trigger: str, force: bool) -> None:
        """One pipeline run + WAL checkpoint, both on the executor thread."""
        loop = self._require_loop()
        started = self._started_future
        if started is None:  # pragma: no cover — created in start() before the task
            raise RuntimeError("scheduler started-future missing")

        def _notify_started(run_id: int) -> None:
            # Executor thread -> event loop: resolve the awaited handle the
            # instant the run's id exists (phase-0 row committed).
            loop.call_soon_threadsafe(_resolve_started, started, run_id)

        def _job() -> int:
            conn = self._require_conn()
            # Iteration sweep (module docstring): single-flight means nothing
            # can be running now, so any 'running' row is a phantom left by an
            # infra failure that defeated even the phase-2 error ledger.
            swept = self._sweep_stale_running(conn)
            if swept:
                logger.warning(
                    "iteration sweep: marked %d phantom 'running' run(s) as 'error' (%r)",
                    swept,
                    SWEEP_ERROR_TEXT,
                )
            run_id = self._pipeline_fn(
                conn,
                trigger=trigger,
                force=force,
                model=self._config.model,
                macro_status=self._macro_status,
                on_run_started=_notify_started,
            )
            # After EVERY run, on the conn-owning thread (storage docstring).
            busy, log_frames, checkpointed = storage.wal_checkpoint_truncate(conn)
            if busy:
                logger.warning(
                    "wal_checkpoint(TRUNCATE) after run %d was blocked (busy=%d): "
                    "%d of %d WAL frames checkpointed; the -wal file was not truncated",
                    run_id,
                    busy,
                    checkpointed,
                    log_frames,
                )
            return int(run_id)

        run_id = await loop.run_in_executor(self._require_executor(), _job)
        # Defensive: guaranteed resolution even if a pipeline_fn seam never
        # invoked on_run_started (idempotent when the callback already fired).
        _resolve_started(started, run_id)
        logger.info("run %d finished (trigger=%r, force=%r)", run_id, trigger, force)

    async def _daily_fetch_if_due(self) -> None:
        """The once-per-UTC-day network ingest (module docstring for the rules)."""
        cfg = self._config
        if not cfg.daily_fetch_enabled:
            return
        now = self._now_utc()
        if now.hour < cfg.daily_fetch_hour_utc:
            return
        today = now.date().isoformat()
        if self._last_fetch_attempt_date == today:
            return  # already attempted today (success OR failure): next day
        loop = self._require_loop()
        executor = self._require_executor()

        def _already_fetched_today() -> bool:
            last = stored_data_fetched_at(self._require_conn())
            return last is not None and last[:10] >= today

        if await loop.run_in_executor(executor, _already_fetched_today):
            self._last_fetch_attempt_date = today
            return
        # Mark the attempt BEFORE fetching: a failure is retried next UTC day,
        # never on the next 5-minute tick.
        self._last_fetch_attempt_date = today
        logger.info("daily fetch: starting (UTC day %s)", today)

        def _fetch() -> tuple[dict[str, int], dict[str, int] | None]:
            conn = self._require_conn()
            price_counts = ingest_prices(conn, self._price_adapter_factory())
            macro_counts: dict[str, int] | None = None
            if self._macro_status.enabled:
                macro_counts = ingest_macro(conn, self._fred_client_factory())
            return price_counts, macro_counts

        price_counts, macro_counts = await loop.run_in_executor(executor, _fetch)
        logger.info(
            "daily fetch done: prices %s; macro %s",
            price_counts,
            "skipped (macro disabled)" if macro_counts is None else macro_counts,
        )
