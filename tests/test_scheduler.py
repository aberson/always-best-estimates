"""Step 11 scheduler tests: loop, coalescing, sweep, daily fetch, liveness.

Real tmp SQLite dbs + the real ``run_pipeline`` wherever a run executes; the
ONLY stand-ins are the scheduler's documented test seams (``SchedulerConfig``:
``pipeline_fn``, ``price_adapter_factory``, ``fred_client_factory``,
``clock``, ``recompute_interval_s``, ``daily_fetch_enabled``,
``daily_fetch_hour_utc``, ``startup_sweep_fn``) — no mocks of
our own code beyond those. Async scenarios run under ``asyncio.run`` (no
pytest-asyncio dependency); route-level tests go through the production
FastAPI app via TestClient with lifespan. No fixed sleeps — every wait is a
bounded poll; the recompute interval is injected small (0.05s) where a
scheduled tick must fire.
"""

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from seeding import seed_prices

from abe import storage
from abe.api import create_app
from abe.constants import FRED_DAILY, UNIVERSE
from abe.ingest.macro import MACRO_DISABLED_NO_KEY, MACRO_OK, MacroStatus
from abe.ingest.sources import PRICE_COLUMNS, utc_now_iso
from abe.model.base import WorldModel
from abe.scheduler import SWEEP_ERROR_TEXT, PendingRun, Scheduler, SchedulerConfig

MACRO_DISABLED = MacroStatus(enabled=False, code=MACRO_DISABLED_NO_KEY, message="no key (test)")
MACRO_ENABLED = MacroStatus(enabled=True, code=MACRO_OK, message="key accepted (test)")

NO_FETCH = SchedulerConfig(daily_fetch_enabled=False)
"""Production defaults minus the daily network fetch (offline discipline)."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rows(db: Path, sql: str, *params: object) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _run_status(db: Path, run_id: int) -> str | None:
    rows = _rows(db, "SELECT status FROM runs WHERE run_id = ?", run_id)
    return str(rows[0][0]) if rows else None


def _terminal(db: Path, run_id: int) -> bool:
    status = _run_status(db, run_id)
    return status is not None and status not in ("running", "queued")


async def _wait_for(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    """Bounded async poll — never a fixed sleep."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not reached within {timeout}s")
        await asyncio.sleep(0.01)


class BlockingPipeline:
    """Documented ``pipeline_fn`` seam stand-in (like optimize's
    ``_solve_once_fn``): allocates a REAL ``runs`` row on the writer
    connection (executor thread), notifies ``on_run_started``, then BLOCKS
    until the test releases it — a deterministically slow run, no timing
    races. Marks the row ``'ok'`` on release."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        trigger: str,
        force: bool,
        model: WorldModel | None,
        macro_status: MacroStatus | None,
        on_run_started: Callable[[int], None] | None,
    ) -> int:
        self.calls += 1
        run_id_raw = storage.insert_row(
            conn,
            "runs",
            {"started_at_utc": utc_now_iso(), "status": "running", "trigger": trigger},
        )
        assert run_id_raw is not None
        run_id = int(run_id_raw)
        if on_run_started is not None:
            on_run_started(run_id)
        self.started.set()
        assert self.release.wait(timeout=10.0), "test forgot to release the blocking run"
        conn.execute(
            "UPDATE runs SET status = 'ok', finished_at_utc = ? WHERE run_id = ?",
            (utc_now_iso(), run_id),
        )
        return run_id


class FlakyPipeline:
    """Documented ``pipeline_fn`` seam: raises RuntimeError BEFORE any row
    insert on the calls whose 1-based index is in ``raise_on_calls`` (an infra
    failure that prevents the run from ever starting — the run_pipeline error
    ledger never gets a chance to record it); every other call allocates a REAL
    ``runs`` row, notifies ``on_run_started``, and marks it 'ok'. Drives the
    loop's infra-failure branch deterministically: the failing call's error
    must reach request_run's awaiter, and the NEXT call must still run."""

    def __init__(self, raise_on_calls: set[int]) -> None:
        self.raise_on_calls = raise_on_calls
        self.calls = 0

    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        trigger: str,
        force: bool,
        model: WorldModel | None,
        macro_status: MacroStatus | None,
        on_run_started: Callable[[int], None] | None,
    ) -> int:
        self.calls += 1
        if self.calls in self.raise_on_calls:
            raise RuntimeError(f"infra failure before row insert (call {self.calls}, test)")
        run_id_raw = storage.insert_row(
            conn,
            "runs",
            {"started_at_utc": utc_now_iso(), "status": "running", "trigger": trigger},
        )
        assert run_id_raw is not None
        run_id = int(run_id_raw)
        if on_run_started is not None:
            on_run_started(run_id)
        conn.execute(
            "UPDATE runs SET status = 'ok', finished_at_utc = ? WHERE run_id = ?",
            (utc_now_iso(), run_id),
        )
        return run_id


class FakeAdapter:
    """``price_adapter_factory`` seam: serves ONE new bar per asset, counting
    calls. The new date sits past the seeded history, and no overlap dates are
    returned, so ingest_prices takes the plain incremental path."""

    source = "yfinance"

    def __init__(self, new_date: str, price: float = 101.0) -> None:
        self.new_date = new_date
        self.price = price
        self.calls: list[tuple[str, str | None]] = []

    def fetch(self, asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        self.calls.append((asset, start))
        data = {column: [self.price] for column in PRICE_COLUMNS}
        return pd.DataFrame(data, index=pd.Index([self.new_date], name="date"))


class ExplodingAdapter:
    """``price_adapter_factory`` seam: the provider is down."""

    source = "yfinance"

    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        self.calls += 1
        raise RuntimeError("provider down (test)")


class FakeFredClient:
    """``fred_client_factory`` seam: one observation per requested series."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
        self.calls.append((series_id, observation_start))
        return pd.Series([1.5], index=["2026-07-30"])


# --------------------------------------------------------------------------- #
# Recompute loop: scheduled skipped run + error resilience
# --------------------------------------------------------------------------- #


def test_scheduled_tick_writes_skipped_run(tmp_path: Path) -> None:
    """With the freshness gate armed by the ok startup run, a scheduled tick
    writes a RECORDED 'skipped' run (trigger='schedule'), not silence."""
    db = tmp_path / "abe.db"
    seed_prices(db)
    cfg = SchedulerConfig(recompute_interval_s=0.05, daily_fetch_enabled=False)

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=cfg)
        await scheduler.start()
        try:
            await _wait_for(lambda: _terminal(db, 1))
            await _wait_for(lambda: _terminal(db, 2))  # the next scheduled tick
        finally:
            await scheduler.stop()

    asyncio.run(scenario())
    rows = _rows(db, 'SELECT status, "trigger" FROM runs WHERE run_id IN (1, 2) ORDER BY run_id')
    assert [(str(status), str(trigger)) for status, trigger in rows] == [
        ("ok", "startup"),
        ("skipped", "schedule"),
    ]


def test_error_run_recorded_and_loop_continues(tmp_path: Path) -> None:
    """A raising stage (25 seeded days -> forecast fails on MIN_HISTORY_BARS)
    lands an error row AND the loop survives: the next scheduled tick still
    fires and writes another run (plan section 9 silent-task-death row)."""
    db = tmp_path / "abe.db"
    seed_prices(db, days=25)
    cfg = SchedulerConfig(recompute_interval_s=0.05, daily_fetch_enabled=False)

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=cfg)
        await scheduler.start()
        try:
            await _wait_for(lambda: _terminal(db, 1))
            await _wait_for(lambda: _terminal(db, 2))  # the loop is still alive
        finally:
            await scheduler.stop()

    asyncio.run(scenario())
    assert _run_status(db, 1) == "error"
    assert _run_status(db, 2) == "error"
    (error_text,) = _rows(db, "SELECT error_text FROM runs WHERE run_id = 1")[0]
    assert "MIN_HISTORY_BARS" in str(error_text)
    stage_rows = _rows(db, "SELECT stage, status FROM run_stages WHERE run_id = 1 ORDER BY rowid")
    assert (str(stage_rows[-1][0]), str(stage_rows[-1][1])) == ("forecast", "error")


# --------------------------------------------------------------------------- #
# request_run: coalescing + force
# --------------------------------------------------------------------------- #


def test_request_run_coalesces_during_active_run(tmp_path: Path) -> None:
    """Single-flight: triggers during an executing run do NOT queue a second
    run — they return the ACTIVE run's handle with already_running=True."""
    db = tmp_path / "abe.db"
    fake = BlockingPipeline()
    cfg = SchedulerConfig(daily_fetch_enabled=False, pipeline_fn=fake)

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=cfg)
        await scheduler.start()
        try:
            # The startup run is EXECUTING (blocked inside the seam).
            await _wait_for(fake.started.is_set)
            first = await scheduler.request_run(force=True)
            second = await scheduler.request_run()
            assert first == PendingRun(run_id=1, already_running=True)
            assert second == PendingRun(run_id=1, already_running=True)
        finally:
            fake.release.set()
            await scheduler.stop()

    asyncio.run(scenario())
    # Exactly ONE run row: the concurrent triggers (their force flags
    # included) coalesced into the active run instead of queuing.
    assert fake.calls == 1
    assert _rows(db, "SELECT COUNT(*) FROM runs") == [(1,)]
    assert _run_status(db, 1) == "ok"


def test_request_run_force_bypasses_freshness(tmp_path: Path) -> None:
    """force=True through request_run recomputes on unchanged data; an
    unforced request lands a recorded 'skipped' run."""
    db = tmp_path / "abe.db"
    seed_prices(db)

    async def scenario() -> tuple[PendingRun, PendingRun]:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=NO_FETCH)
        await scheduler.start()
        try:
            # Settle the startup run AND the loop iteration (run_in_progress),
            # so the requests below never coalesce into it.
            await _wait_for(lambda: _terminal(db, 1) and not scheduler.run_in_progress)
            unforced = await scheduler.request_run()
            await _wait_for(lambda: _terminal(db, unforced.run_id))
            await _wait_for(lambda: not scheduler.run_in_progress)
            forced = await scheduler.request_run(force=True)
            await _wait_for(lambda: _terminal(db, forced.run_id))
            # WAL checkpoint wiring (asserted while the writer conn is still
            # open, so the -wal sidecar exists): the post-run
            # wal_checkpoint_truncate zeroed it. Gate on run_in_progress=False
            # so the checkpoint (end of _job, after the run status commits) has
            # definitely completed. Deleting the scheduler's
            # wal_checkpoint_truncate call regresses this to a non-empty -wal.
            await _wait_for(lambda: not scheduler.run_in_progress)
            wal = db.with_name(db.name + "-wal")
            assert wal.exists(), "the -wal sidecar must exist while the writer conn is open"
            assert wal.stat().st_size == 0, (
                "wal_checkpoint(TRUNCATE) after the run must have zeroed the -wal file"
            )
            return unforced, forced
        finally:
            await scheduler.stop()

    unforced, forced = asyncio.run(scenario())
    assert unforced == PendingRun(run_id=2, already_running=False)
    assert forced == PendingRun(run_id=3, already_running=False)
    assert _run_status(db, 1) == "ok"  # startup
    assert _run_status(db, 2) == "skipped"  # gate armed, not forced
    assert _run_status(db, 3) == "ok"  # force bypassed the gate
    triggers = _rows(db, 'SELECT "trigger" FROM runs ORDER BY run_id')
    assert [str(trigger) for (trigger,) in triggers] == ["startup", "manual", "manual"]


def test_pre_start_force_or_coalescing(tmp_path: Path) -> None:
    """Pre-start coalescing ORs force flags together
    (``self._pending_force = self._pending_force or force``): two request_run
    calls that reach the parked loop TOGETHER — force=True then force=False —
    must produce a FORCED run, not a last-wins unforced skip. A regression to
    ``self._pending_force = force`` would let the second call's force=False
    win, and the armed freshness gate would record a 'skipped' run instead."""
    db = tmp_path / "abe.db"
    seed_prices(db)  # the startup run arms the gate: an unforced run -> 'skipped'

    async def scenario() -> tuple[PendingRun, PendingRun]:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=NO_FETCH)
        await scheduler.start()
        try:
            # Startup run settled and the loop is parked on the wake event.
            await _wait_for(lambda: _terminal(db, 1) and not scheduler.run_in_progress)
            # Fire both BEFORE the parked loop consumes _pending_force: gather
            # schedules the two request_run tasks in order, so both synchronous
            # preludes (set _pending_force, set the wake event) run before the
            # loop's next step — the OR must combine their flags into force.
            first, second = await asyncio.gather(
                scheduler.request_run(force=True),
                scheduler.request_run(force=False),
            )
            await _wait_for(lambda: _terminal(db, first.run_id))
            return first, second
        finally:
            await scheduler.stop()

    first, second = asyncio.run(scenario())
    # Both callers coalesced into the SAME new run, each resolved at its start.
    assert first == PendingRun(run_id=2, already_running=False)
    assert second == PendingRun(run_id=2, already_running=False)
    # Exactly one NEW run row and force won: it is NOT skipped.
    assert _rows(db, "SELECT COUNT(*) FROM runs") == [(2,)]
    assert _run_status(db, 2) == "ok"


def test_infra_failure_reaches_awaiter_and_loop_survives(tmp_path: Path) -> None:
    """An infra failure that prevents the run from starting (pipeline_fn raises
    BEFORE any row insert, so run_pipeline's own error ledger never records it)
    is delivered to request_run's awaiter AND the loop survives: the next
    trigger executes and writes a run row (plan section 9 silent-task-death
    row). No 5-minute cadence here — only explicit triggers run, so the failing
    call is deterministically the request_run-triggered one."""
    db = tmp_path / "abe.db"
    fake = FlakyPipeline(raise_on_calls={2})  # startup ok; the triggered run explodes
    cfg = SchedulerConfig(daily_fetch_enabled=False, pipeline_fn=fake)

    async def scenario() -> PendingRun:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=cfg)
        await scheduler.start()
        try:
            await _wait_for(lambda: _terminal(db, 1) and not scheduler.run_in_progress)
            # Call 2 raises before inserting a row -> the loop wraps it and
            # delivers it to this awaiter (the endpoint would answer 500).
            with pytest.raises(RuntimeError, match="pipeline run failed to start"):
                await scheduler.request_run()
            await _wait_for(lambda: not scheduler.run_in_progress)
            # Call 3 proves the loop kept running after the infra failure.
            survivor = await scheduler.request_run()
            await _wait_for(lambda: _terminal(db, survivor.run_id))
            return survivor
        finally:
            await scheduler.stop()

    survivor = asyncio.run(scenario())
    assert survivor == PendingRun(run_id=2, already_running=False)
    assert fake.calls == 3  # startup, the failed trigger (no row), the survivor
    # The failed call inserted NO row; the ledger holds only the startup run
    # and the survivor, both 'ok'.
    assert _rows(db, "SELECT COUNT(*) FROM runs") == [(2,)]
    assert _run_status(db, 1) == "ok"
    assert _run_status(db, 2) == "ok"


# --------------------------------------------------------------------------- #
# Startup sweep
# --------------------------------------------------------------------------- #


def test_startup_sweep_marks_orphaned_running_runs(tmp_path: Path) -> None:
    """A stale 'running' row (process kill / infra failure) is swept to
    'error' with the stable text BEFORE the first run — owed from Step 8."""
    db = tmp_path / "abe.db"
    seed_prices(db)
    conn = storage.open_writer(db)
    try:
        orphan_raw = storage.insert_row(
            conn,
            "runs",
            {"started_at_utc": "2026-07-06T00:00:00Z", "status": "running", "trigger": "schedule"},
        )
        assert orphan_raw is not None
        orphan = int(orphan_raw)
    finally:
        conn.close()

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=NO_FETCH)
        await scheduler.start()  # returns only after the sweep completed
        try:
            row = _rows(
                db, "SELECT status, error_text, finished_at_utc FROM runs WHERE run_id = ?", orphan
            )[0]
            assert str(row[0]) == "error"
            assert str(row[1]) == SWEEP_ERROR_TEXT
            assert row[2] is not None
        finally:
            await scheduler.stop()

    asyncio.run(scenario())


def test_startup_sweep_failure_closes_writer_conn(tmp_path: Path) -> None:
    """A failing startup sweep cannot leak the writer connection: start()
    re-raises the error AND its executor teardown closes the conn, so the db
    file is unlocked afterwards. The Windows lock check is the proof — a leaked
    WAL handle would pin abe.db and the unlink below would raise
    PermissionError. Uses the startup_sweep_fn seam (which defaults to the real
    sweep that every other test exercises)."""
    db = tmp_path / "abe.db"
    seed_prices(db)  # a real db for the sweep to (fail to) touch

    def _boom(conn: sqlite3.Connection) -> int:
        raise RuntimeError("startup sweep exploded (test)")

    cfg = SchedulerConfig(daily_fetch_enabled=False, startup_sweep_fn=_boom)

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=cfg)
        with pytest.raises(RuntimeError, match="startup sweep exploded"):
            await scheduler.start()

    asyncio.run(scenario())
    # The conn closed: on Windows a live WAL handle pins the file, so a clean
    # remove proves no connection (and no -wal handle) leaked.
    db.unlink()
    assert not db.exists()


# --------------------------------------------------------------------------- #
# Event-loop liveness + route-level coalescing (production route)
# --------------------------------------------------------------------------- #


def test_health_responds_during_active_run_and_trigger_coalesces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE point of Step 11: the pipeline body runs on the executor thread,
    so the event loop answers /health DURING a run — and the production
    trigger route coalesces instead of queuing a second run."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "abe.db"
    fake = BlockingPipeline()
    cfg = SchedulerConfig(daily_fetch_enabled=False, pipeline_fn=fake)
    with TestClient(create_app(db, scheduler_config=cfg)) as client:
        try:
            # The startup run is EXECUTING (deterministically blocked).
            assert fake.started.wait(timeout=10.0)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            # Route-level coalescing: 202 with the ACTIVE run's id.
            trig = client.post("/api/runs/trigger", json={"force": True})
            assert trig.status_code == 202
            assert trig.json() == {"run_id": 1, "already_running": True}
        finally:
            fake.release.set()
    assert fake.calls == 1
    assert _rows(db, "SELECT COUNT(*) FROM runs") == [(1,)]


# --------------------------------------------------------------------------- #
# Daily fetch (plan section 8 fetch-split)
# --------------------------------------------------------------------------- #

_FETCH_DUE_CLOCK = datetime(2026, 8, 1, 23, 30, tzinfo=UTC)
"""Injected 'now': past the default fetch hour, on a day AFTER the seeded
``fetched_at_utc`` (2026-07-07) — the persistent marker says fetch is due."""


def _drive_one_iteration(db: Path, macro_status: MacroStatus, cfg: SchedulerConfig) -> None:
    """Start the scheduler, let the first loop iteration (fetch-if-due +
    startup run) settle, and stop."""

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=macro_status, config=cfg)
        await scheduler.start()
        try:
            await _wait_for(lambda: _terminal(db, 1) and not scheduler.run_in_progress)
        finally:
            await scheduler.stop()

    asyncio.run(scenario())


def test_daily_fetch_runs_when_due_and_recompute_sees_new_data(tmp_path: Path) -> None:
    db = tmp_path / "abe.db"
    last_seeded = seed_prices(db)
    adapter = FakeAdapter(new_date="2026-07-31")
    fred = FakeFredClient()
    cfg = SchedulerConfig(
        daily_fetch_hour_utc=22,
        price_adapter_factory=lambda: adapter,
        fred_client_factory=lambda: fred,
        clock=lambda: _FETCH_DUE_CLOCK,
    )
    _drive_one_iteration(db, MACRO_ENABLED, cfg)

    # Prices fetched once per asset, incrementally (start > None).
    assert {asset for asset, _ in adapter.calls} == set(UNIVERSE)
    assert all(start is not None for _, start in adapter.calls)
    # Macro fetched for the whole daily set (key enabled).
    assert [series for series, _ in fred.calls] == list(FRED_DAILY)
    assert _rows(db, "SELECT COUNT(*) FROM macro") == [(len(FRED_DAILY),)]
    # The new bar landed for every asset...
    assert _rows(db, "SELECT COUNT(*) FROM prices WHERE date = '2026-07-31'") == [(len(UNIVERSE),)]
    # ...and the SAME iteration's startup recompute picked it up (fetch runs
    # BEFORE the run, so the freshness watermark already reflects the fetch).
    (detail_json,) = _rows(
        db, "SELECT detail_json FROM run_stages WHERE run_id = 1 AND stage = 'freshness'"
    )[0]
    detail = json.loads(str(detail_json))
    assert detail["data_max_date"] == "2026-07-31"
    assert detail["data_max_date"] > last_seeded
    # The fetch is NOT a pipeline run: exactly one runs row (the startup run).
    assert _rows(db, "SELECT COUNT(*) FROM runs") == [(1,)]
    assert _run_status(db, 1) == "ok"


def test_daily_fetch_skips_when_already_fetched_today(tmp_path: Path) -> None:
    """Persistent marker: MAX(fetched_at_utc) on the injected 'today' means
    the fetch already ran — the adapter is never constructed-for/called."""
    db = tmp_path / "abe.db"
    seed_prices(db)
    conn = storage.open_writer(db)
    try:
        conn.execute("UPDATE prices SET fetched_at_utc = '2026-08-01T05:00:00Z'")
    finally:
        conn.close()
    adapter = FakeAdapter(new_date="2026-08-02")
    fred = FakeFredClient()
    cfg = SchedulerConfig(
        price_adapter_factory=lambda: adapter,
        fred_client_factory=lambda: fred,
        clock=lambda: _FETCH_DUE_CLOCK,
    )
    _drive_one_iteration(db, MACRO_ENABLED, cfg)
    assert adapter.calls == []
    assert fred.calls == []


def test_daily_fetch_waits_for_configured_hour(tmp_path: Path) -> None:
    db = tmp_path / "abe.db"
    seed_prices(db)
    adapter = FakeAdapter(new_date="2026-08-02")
    cfg = SchedulerConfig(
        price_adapter_factory=lambda: adapter,
        clock=lambda: datetime(2026, 8, 1, 10, 0, tzinfo=UTC),  # before 22:00 UTC
    )
    _drive_one_iteration(db, MACRO_DISABLED, cfg)
    assert adapter.calls == []


def test_daily_fetch_macro_skipped_without_key(tmp_path: Path) -> None:
    """macro_status disabled (no FRED key) -> prices still fetch, but no FRED
    client is ever BUILT (not merely never called)."""
    db = tmp_path / "abe.db"
    seed_prices(db)
    adapter = FakeAdapter(new_date="2026-07-31")

    def _forbidden_fred() -> FakeFredClient:
        raise AssertionError("fred_client_factory must not be invoked without a key")

    cfg = SchedulerConfig(
        price_adapter_factory=lambda: adapter,
        fred_client_factory=_forbidden_fred,
        clock=lambda: _FETCH_DUE_CLOCK,
    )
    _drive_one_iteration(db, MACRO_DISABLED, cfg)
    assert {asset for asset, _ in adapter.calls} == set(UNIVERSE)
    assert _rows(db, "SELECT COUNT(*) FROM macro") == [(0,)]
    assert _run_status(db, 1) == "ok"


def test_daily_fetch_failure_keeps_loop_alive_and_defers_retry(tmp_path: Path) -> None:
    """A failing fetch never crashes the loop (the recompute still runs and
    the next tick still fires) and is NOT retried on the 5-minute cadence —
    the in-memory attempt marker defers the retry to the next UTC day."""
    db = tmp_path / "abe.db"
    seed_prices(db)
    adapter = ExplodingAdapter()
    cfg = SchedulerConfig(
        recompute_interval_s=0.05,
        price_adapter_factory=lambda: adapter,
        clock=lambda: _FETCH_DUE_CLOCK,
    )

    async def scenario() -> None:
        scheduler = Scheduler(db, macro_status=MACRO_DISABLED, config=cfg)
        await scheduler.start()
        try:
            await _wait_for(lambda: _terminal(db, 1))
            await _wait_for(lambda: _terminal(db, 2))  # loop survived the fetch failure
        finally:
            await scheduler.stop()

    asyncio.run(scenario())
    assert adapter.calls == 1  # attempted ONCE today, not on every tick
    assert _run_status(db, 1) == "ok"  # the same iteration's recompute still ran
    assert _run_status(db, 2) == "skipped"  # and the loop kept ticking
