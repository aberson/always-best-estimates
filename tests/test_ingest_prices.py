"""Price-ingest tests: normalization, overlap-checked incremental fetch, NaN boundary,
cache, backoff, and the yfinance call contract (via a fake yfinance module).

The default suite runs OFFLINE on a fresh clone: tmp_path DBs, canned DataFrames,
fake ``SourceAdapter`` implementations (no unittest.mock of our own code).
Real-Yahoo checks are marked ``@pytest.mark.network`` and the real-backfilled-db
check is marked ``@pytest.mark.realdb``; both are deselected by default (pytest
``addopts``) and each self-skips with an explicit reason when its prerequisite
(connectivity / the backfilled db) is absent. Real db location: ``ABE_REAL_DB``
env var, else ``storage.DEFAULT_DB_PATH`` resolved against the project root.
"""

import logging
import os
import re
import socket
import sqlite3
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abe import constants, storage
from abe.ingest.prices import (
    OVERLAP_DAYS,
    RATE_LIMIT_SLEEPS,
    ingest_prices,
    main,
    stored_max_date,
)
from abe.ingest.sources import (
    PRICE_COLUMNS,
    CacheAdapter,
    YFinanceAdapter,
    normalize_yfinance_frame,
)

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _real_db_path() -> Path:
    override = os.environ.get("ABE_REAL_DB")
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parent.parent
    return project_root / storage.DEFAULT_DB_PATH


REAL_DB = _real_db_path()
"""The backfilled production db (gitignored). Override via ABE_REAL_DB."""


# --------------------------------------------------------------------------- #
# Helpers: canned frames + fake adapters (no mocks of our own code)
# --------------------------------------------------------------------------- #


def _raw_yfinance_frame(dates: Sequence[str], tz: str | None = "America/New_York") -> pd.DataFrame:
    """A canned raw ``Ticker.history``-shaped frame (auto_adjust=True, actions=False)."""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=index,
    )


def _normalized_frame(dates: Sequence[str], base: float = 100.0) -> pd.DataFrame:
    """A canned frame already in the adapter-contract shape."""
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [base + i for i in range(n)],
            "high": [base + 1.0 + i for i in range(n)],
            "low": [base - 1.0 + i for i in range(n)],
            "close": [base + 0.5 + i for i in range(n)],
            "adj_close": [base + 0.5 + i for i in range(n)],
            "volume": [1_000_000 + i for i in range(n)],
        },
        index=pd.Index(list(dates), name="date"),
    )


class FakeAdapter:
    """SourceAdapter impl serving a canned normalized frame; records fetch calls."""

    def __init__(self, frame: pd.DataFrame, source: str = "yfinance") -> None:
        self.frame = frame
        self.source = source
        self.calls: list[tuple[str, str | None, str | None]] = []

    def fetch(self, asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        self.calls.append((asset, start, end))
        if start is None:
            return self.frame
        return self.frame.loc[self.frame.index >= start]


class RateLimitedThenOk:
    """Raises a rate-limit-shaped error ``failures`` times, then delegates."""

    source = "yfinance"

    def __init__(self, inner: FakeAdapter, failures: int, exc: Exception) -> None:
        self.inner = inner
        self.remaining_failures = failures
        self.exc = exc

    def fetch(self, asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.exc
        return self.inner.fetch(asset, start=start, end=end)


class YFRateLimitError(Exception):
    """Same class NAME as yfinance's — exercises the no-yfinance-import detection."""


def _recording_sleep(sleeps: list[float]) -> Callable[[float], None]:
    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return _sleep


def _overlap_start(max_date: str) -> str:
    return (date.fromisoformat(max_date) - timedelta(days=OVERLAP_DAYS)).isoformat()


@pytest.fixture()
def writer(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = storage.open_writer(tmp_path / "abe.db")
    yield conn
    conn.close()


# --------------------------------------------------------------------------- #
# Fake yfinance module (tests the exact call contract of YFinanceAdapter.fetch)
# --------------------------------------------------------------------------- #


class _FakeYFinance:
    """A sys.modules['yfinance'] stand-in recording Ticker.history calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.frame = _raw_yfinance_frame(["2026-07-01", "2026-07-02", "2026-07-03"])
        outer = self
        module = types.ModuleType("yfinance")

        class Ticker:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, **kwargs: object) -> pd.DataFrame:
                outer.calls.append((self.symbol, dict(kwargs)))
                return outer.frame

        module.Ticker = Ticker  # type: ignore[attr-defined]
        module.config = types.SimpleNamespace(  # type: ignore[attr-defined]
            debug=types.SimpleNamespace(hide_exceptions=True)
        )
        self.module = module


@pytest.fixture()
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> _FakeYFinance:
    fake = _FakeYFinance()
    monkeypatch.setitem(sys.modules, "yfinance", fake.module)
    return fake


def test_full_history_uses_period_max_and_sets_fail_loud_config(
    fake_yfinance: _FakeYFinance,
) -> None:
    """start=None must fetch period='max' — yfinance defaults to a 1mo window."""
    frame = YFinanceAdapter().fetch("SPY")
    symbol, kwargs = fake_yfinance.calls[0]
    assert symbol == "SPY"
    assert kwargs["period"] == "max"
    assert "start" not in kwargs and "end" not in kwargs
    assert kwargs["interval"] == "1d"
    assert kwargs["auto_adjust"] is True
    assert kwargs["actions"] is False
    # Raising semantics genuinely enabled (backoff depends on this being live).
    assert fake_yfinance.module.config.debug.hide_exceptions is False  # type: ignore[attr-defined]
    assert len(frame) == 3


def test_full_history_with_end_bound_still_fetches_max_then_filters(
    fake_yfinance: _FakeYFinance,
) -> None:
    """start=None + end set must NOT fall into yfinance's 1mo-before-end default."""
    frame = YFinanceAdapter().fetch("SPY", end="2026-07-03")
    _, kwargs = fake_yfinance.calls[0]
    assert kwargs["period"] == "max"
    assert "start" not in kwargs and "end" not in kwargs
    assert list(frame.index) == ["2026-07-01", "2026-07-02"]  # end exclusive, post-fetch


def test_incremental_fetch_passes_start_through(fake_yfinance: _FakeYFinance) -> None:
    YFinanceAdapter().fetch("AGG", start="2026-07-01")
    symbol, kwargs = fake_yfinance.calls[0]
    assert symbol == "AGG"
    assert kwargs["start"] == "2026-07-01"
    assert kwargs["end"] is None
    assert "period" not in kwargs


def test_full_history_empty_frame_raises_loudly(fake_yfinance: _FakeYFinance) -> None:
    """An empty full-history result is a failed download, never no-data."""
    fake_yfinance.frame = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex([])
    )
    with pytest.raises(RuntimeError, match="full-history fetch for 'SPY'"):
        YFinanceAdapter().fetch("SPY")


def test_cli_main_backfills_through_production_entry_point(
    fake_yfinance: _FakeYFinance, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`python -m abe.ingest.prices --backfill --db <path>` end-to-end (fake yfinance)."""
    db_path = tmp_path / "abe.db"
    exit_code = main(["--backfill", "--db", str(db_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    for asset in constants.UNIVERSE:
        assert f"{asset}: +3 rows (total 3, 2026-07-01 .. 2026-07-03)" in out
    conn = storage.open_read_only(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    finally:
        conn.close()
    assert total == 3 * len(constants.UNIVERSE)


# --------------------------------------------------------------------------- #
# Normalization (offline: canned raw frames)
# --------------------------------------------------------------------------- #


def test_normalize_maps_close_to_both_and_strips_tz() -> None:
    raw = _raw_yfinance_frame(["2026-07-01", "2026-07-02"], tz="America/New_York")
    normalized = normalize_yfinance_frame(raw)
    assert list(normalized.columns) == list(PRICE_COLUMNS)
    assert list(normalized.index) == ["2026-07-01", "2026-07-02"]
    assert all(isinstance(d, str) and DATE_RE.fullmatch(d) for d in normalized.index)
    # auto_adjust=True: Close IS adjusted → mapped to BOTH close and adj_close.
    assert normalized["adj_close"].tolist() == normalized["close"].tolist()
    assert normalized["close"].tolist() == raw["Close"].tolist()


def test_normalize_accepts_tz_naive_index_and_sorts() -> None:
    raw = _raw_yfinance_frame(["2026-07-02", "2026-07-01"], tz=None)
    normalized = normalize_yfinance_frame(raw)
    assert list(normalized.index) == ["2026-07-01", "2026-07-02"]


def test_normalize_rejects_unexpected_column_set() -> None:
    raw = _raw_yfinance_frame(["2026-07-01"])
    with_adj = raw.assign(**{"Adj Close": raw["Close"]})  # old-default drift shape
    with pytest.raises(ValueError, match="column set drifted"):
        normalize_yfinance_frame(with_adj)
    with pytest.raises(ValueError, match="column set drifted"):
        normalize_yfinance_frame(raw.drop(columns=["Volume"]))
    with_actions = raw.assign(Dividends=0.0)  # actions=False drift shape
    with pytest.raises(ValueError, match="column set drifted"):
        normalize_yfinance_frame(with_actions)


def test_normalize_rejects_multiindex_columns() -> None:
    raw = _raw_yfinance_frame(["2026-07-01"])
    raw.columns = pd.MultiIndex.from_product([list(raw.columns), ["SPY"]])
    with pytest.raises(ValueError, match="MultiIndex"):
        normalize_yfinance_frame(raw)


def test_normalize_rejects_non_datetime_index() -> None:
    raw = _raw_yfinance_frame(["2026-07-01"])
    raw.index = pd.Index([0])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        normalize_yfinance_frame(raw)


def test_normalize_empty_frames_are_empty_not_drift() -> None:
    """Zero rows = nothing to write, so the pre-write drift assertion does not
    apply — including yfinance's failed-download placeholder shape (zero rows
    but LEGACY columns incl. 'Adj Close'), seen live on an up-to-date fetch."""
    for empty in (
        pd.DataFrame(),
        pd.DataFrame(
            columns=["Adj Close", "Close", "High", "Low", "Open", "Volume"],
            index=pd.DatetimeIndex([]),
        ),
    ):
        normalized = normalize_yfinance_frame(empty)
        assert list(normalized.columns) == list(PRICE_COLUMNS)
        assert normalized.empty


# --------------------------------------------------------------------------- #
# Incremental ingest with overlap-consistency (fake adapter, real tmp db)
# --------------------------------------------------------------------------- #


def test_first_ingest_full_history_then_incremental_noop(writer: sqlite3.Connection) -> None:
    adapter = FakeAdapter(_normalized_frame(["2026-07-01", "2026-07-02", "2026-07-03"]))
    counts = ingest_prices(writer, adapter, assets=("SPY",))
    assert counts == {"SPY": 3}
    assert adapter.calls[0] == ("SPY", None, None)  # empty table → full history

    counts_again = ingest_prices(writer, adapter, assets=("SPY",))
    # Second run fetches from stored max MINUS the overlap window (inclusive)…
    assert adapter.calls[1] == ("SPY", _overlap_start("2026-07-03"), None)
    # …and with a consistent overlap it is a no-op row-count-wise (idempotency).
    assert counts_again == {"SPY": 0}
    total = writer.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert total == 3


def test_matching_overlap_adds_only_new_rows(writer: sqlite3.Connection) -> None:
    adapter = FakeAdapter(_normalized_frame(["2026-07-01", "2026-07-02"]))
    ingest_prices(writer, adapter, assets=("SPY",))
    # Same history basis + one new bar: overlap matches → incremental append only.
    adapter.frame = _normalized_frame(["2026-07-01", "2026-07-02", "2026-07-03"])
    counts = ingest_prices(writer, adapter, assets=("SPY",))
    assert adapter.calls[1] == ("SPY", _overlap_start("2026-07-02"), None)
    assert len(adapter.calls) == 2  # no full-refresh fetch
    assert counts == {"SPY": 1}
    rows = writer.execute("SELECT date FROM prices ORDER BY date").fetchall()
    assert [r[0] for r in rows] == ["2026-07-01", "2026-07-02", "2026-07-03"]


def test_mismatched_overlap_triggers_full_refresh(
    writer: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """A backward adjustment (dividend rescales ALL history) must be detected in
    the overlap window and answered with a full refresh, not a silent append."""
    dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
    adapter = FakeAdapter(_normalized_frame(dates))
    ingest_prices(writer, adapter, assets=("SPY",))

    # Provider re-based: every historical adj_close rescaled by ~1%, plus a new bar.
    rebased = _normalized_frame([*dates, "2026-07-06"])
    rebased["adj_close"] = rebased["adj_close"] * 0.99
    adapter.frame = rebased
    with caplog.at_level(logging.WARNING):
        counts = ingest_prices(writer, adapter, assets=("SPY",))

    assert [c[1] for c in adapter.calls] == [None, _overlap_start("2026-07-03"), None]
    assert adapter.calls[2] == ("SPY", None, None)  # the full refresh
    assert "full refresh" in caplog.text
    assert counts == {"SPY": 4}  # full-refresh counts every upserted row
    stored = dict(writer.execute("SELECT date, adj_close FROM prices ORDER BY date").fetchall())
    assert stored["2026-07-01"] == pytest.approx(100.5 * 0.99)  # old basis replaced
    assert stored["2026-07-06"] == pytest.approx(103.5 * 0.99)
    assert len(stored) == 4


def test_changed_last_bar_self_heals_without_full_refresh(
    writer: sqlite3.Connection,
) -> None:
    """The stored MAX(date) bar may be intraday-partial; a changed value there is
    upserted (self-heal) but must NOT vote as an adjustment mismatch."""
    dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
    adapter = FakeAdapter(_normalized_frame(dates))
    ingest_prices(writer, adapter, assets=("SPY",))

    finalized = _normalized_frame(dates)
    finalized.loc["2026-07-03", "adj_close"] = 999.0  # last bar finalized ≠ stored
    finalized.loc["2026-07-03", "close"] = 999.0
    adapter.frame = finalized
    counts = ingest_prices(writer, adapter, assets=("SPY",))

    assert len(adapter.calls) == 2  # overlap fetch only — no full refresh
    assert counts == {"SPY": 0}  # nothing NEW, just the heal
    healed = writer.execute("SELECT adj_close FROM prices WHERE date = '2026-07-03'").fetchone()[0]
    assert healed == pytest.approx(999.0)


def test_ingest_stamps_source_from_adapter_and_fetched_at(writer: sqlite3.Connection) -> None:
    ingest_prices(writer, FakeAdapter(_normalized_frame(["2026-07-01"])), assets=("AGG",))
    source, fetched_at = writer.execute(
        "SELECT source, fetched_at_utc FROM prices WHERE asset = 'AGG'"
    ).fetchone()
    assert source == "yfinance"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", fetched_at)
    # The label comes from adapter.source, not a hardcode:
    cache_labeled = FakeAdapter(_normalized_frame(["2026-07-01"]), source="cache")
    ingest_prices(writer, cache_labeled, assets=("SPY",))
    assert writer.execute("SELECT source FROM prices WHERE asset = 'SPY'").fetchone()[0] == "cache"


def test_ingest_covers_all_universe_assets(writer: sqlite3.Connection) -> None:
    adapter = FakeAdapter(_normalized_frame(["2026-07-01", "2026-07-02"]))
    counts = ingest_prices(writer, adapter)  # default: constants.UNIVERSE
    assert counts == {asset: 2 for asset in constants.UNIVERSE}
    assert [c[0] for c in adapter.calls] == list(constants.UNIVERSE)


def test_ingest_rejects_frame_missing_contract_columns(writer: sqlite3.Connection) -> None:
    bad = _normalized_frame(["2026-07-01"]).drop(columns=["adj_close"])
    with pytest.raises(ValueError, match="missing column"):
        ingest_prices(writer, FakeAdapter(bad), assets=("SPY",))


def test_ingest_rejects_non_iso_date_keys_at_write_boundary(
    writer: sqlite3.Connection,
) -> None:
    bad = _normalized_frame(["2026-07-01"])
    bad.index = pd.Index(["07/01/2026"], name="date")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        ingest_prices(writer, FakeAdapter(bad), assets=("SPY",))


def test_ingest_sorts_unordered_adapter_frames(writer: sqlite3.Connection) -> None:
    descending = _normalized_frame(["2026-07-01", "2026-07-02", "2026-07-03"]).iloc[::-1]
    counts = ingest_prices(writer, FakeAdapter(descending), assets=("SPY",))
    assert counts == {"SPY": 3}
    rows = writer.execute("SELECT date FROM prices ORDER BY date").fetchall()
    assert [r[0] for r in rows] == ["2026-07-01", "2026-07-02", "2026-07-03"]


def test_stored_max_date_empty_and_populated(writer: sqlite3.Connection) -> None:
    assert stored_max_date(writer, "SPY") is None
    ingest_prices(writer, FakeAdapter(_normalized_frame(["2026-07-31"])), assets=("SPY",))
    assert stored_max_date(writer, "SPY") == "2026-07-31"
    assert stored_max_date(writer, "AGG") is None  # per-asset independence


# --------------------------------------------------------------------------- #
# NaN → None conversion, verified through to storage
# --------------------------------------------------------------------------- #


def test_nan_cells_become_sql_null(writer: sqlite3.Connection) -> None:
    frame = _normalized_frame(["2026-07-01", "2026-07-02"])
    frame.loc["2026-07-01", "open"] = np.nan
    frame.loc["2026-07-01", "volume"] = np.nan
    ingest_prices(writer, FakeAdapter(frame), assets=("SPY",))
    row = writer.execute(
        "SELECT typeof(open), typeof(volume), typeof(close) FROM prices WHERE date = '2026-07-01'"
    ).fetchone()
    assert row == ("null", "null", "real")
    # The non-NaN sibling row is untouched.
    intact = writer.execute(
        "SELECT typeof(open), typeof(volume) FROM prices WHERE date = '2026-07-02'"
    ).fetchone()
    assert intact[0] == "real"
    assert intact[1] in ("integer", "real")


# --------------------------------------------------------------------------- #
# Cache path: full history from SQLite, zero network, zero yfinance import
# --------------------------------------------------------------------------- #


class _ForbidYFinanceImport:
    """Meta-path hook that fails the test if anything imports yfinance."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "yfinance" or fullname.startswith("yfinance."):
            raise AssertionError(f"cache-only code path imported {fullname!r}")
        return None


def test_cache_adapter_serves_history_without_yfinance(tmp_path: Path) -> None:
    db_path = tmp_path / "abe.db"
    conn = storage.open_writer(db_path)
    try:
        ingest_prices(
            conn,
            FakeAdapter(_normalized_frame(["2026-07-01", "2026-07-02", "2026-07-03"])),
            assets=("SPY",),
        )
    finally:
        conn.close()

    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if "yfinance" in name}
    guard = _ForbidYFinanceImport()
    sys.meta_path.insert(0, guard)
    try:
        frame = CacheAdapter(db_path).fetch("SPY")
    finally:
        sys.meta_path.remove(guard)
        sys.modules.update(saved)

    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert list(frame.index) == ["2026-07-01", "2026-07-02", "2026-07-03"]
    assert frame["adj_close"].tolist() == [100.5, 101.5, 102.5]
    assert CacheAdapter(db_path).source == "cache"


def test_cache_adapter_date_range_and_unknown_asset(tmp_path: Path) -> None:
    db_path = tmp_path / "abe.db"
    conn = storage.open_writer(db_path)
    try:
        ingest_prices(
            conn,
            FakeAdapter(_normalized_frame(["2026-07-01", "2026-07-02", "2026-07-03"])),
            assets=("SPY",),
        )
    finally:
        conn.close()
    cache = CacheAdapter(db_path)
    window = cache.fetch("SPY", start="2026-07-02", end="2026-07-03")  # end exclusive
    assert list(window.index) == ["2026-07-02"]
    assert cache.fetch("ACWI").empty  # nothing stored for this asset


# --------------------------------------------------------------------------- #
# 429 backoff
# --------------------------------------------------------------------------- #


def test_backoff_retries_then_succeeds(writer: sqlite3.Connection) -> None:
    inner = FakeAdapter(_normalized_frame(["2026-07-01", "2026-07-02"]))
    flaky = RateLimitedThenOk(inner, failures=2, exc=YFRateLimitError("Too Many Requests"))
    sleeps: list[float] = []
    counts = ingest_prices(writer, flaky, assets=("SPY",), sleep_fn=_recording_sleep(sleeps))
    assert counts == {"SPY": 2}
    assert sleeps == [2.0, 8.0]  # exponential, injected sleep — no real waiting


def test_backoff_budget_exhausted_raises_loudly(writer: sqlite3.Connection) -> None:
    inner = FakeAdapter(_normalized_frame(["2026-07-01"]))
    always_429 = RateLimitedThenOk(
        inner, failures=99, exc=ConnectionError("HTTP Error 429: rate limited")
    )
    sleeps: list[float] = []
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        ingest_prices(writer, always_429, assets=("SPY",), sleep_fn=_recording_sleep(sleeps))
    assert sleeps == list(RATE_LIMIT_SLEEPS)
    assert writer.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 0


def test_non_rate_limit_error_propagates_without_retry(writer: sqlite3.Connection) -> None:
    inner = FakeAdapter(_normalized_frame(["2026-07-01"]))
    broken = RateLimitedThenOk(inner, failures=99, exc=ValueError("delisted ticker"))
    sleeps: list[float] = []
    with pytest.raises(ValueError, match="delisted"):
        ingest_prices(writer, broken, assets=("SPY",), sleep_fn=_recording_sleep(sleeps))
    assert sleeps == []


# --------------------------------------------------------------------------- #
# Real-network / real-db tests (deselected by default via addopts; self-skipping)
# --------------------------------------------------------------------------- #


def _online() -> bool:
    try:
        with socket.create_connection(("query1.finance.yahoo.com", 443), timeout=5):
            return True
    except OSError:
        return False


@pytest.mark.network
def test_real_yfinance_tiny_fetch_passes_column_assertion() -> None:
    if not _online():
        pytest.skip("offline: cannot reach query1.finance.yahoo.com:443")
    start = (date.today() - timedelta(days=20)).isoformat()
    frame = YFinanceAdapter().fetch("SPY", start=start)
    assert not frame.empty  # ~10+ trading days in a 20-day window
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert all(isinstance(d, str) and DATE_RE.fullmatch(d) for d in frame.index)
    assert (frame["adj_close"] > 0).all()
    assert frame["adj_close"].tolist() == frame["close"].tolist()  # auto_adjust contract


@pytest.mark.realdb
def test_agg_trailing_10y_total_return_guard() -> None:
    """AGG adj_close must show TOTAL return (>1%/yr trailing 10y annualized mean
    log-return). Price-only AGG is ~flat-to-negative over any 10y window — this
    guards the price-only-vs-total-return bond bug (plan Step 3 Done-when).
    Needs no network, only the backfilled db."""
    if not REAL_DB.exists():
        pytest.skip(f"real backfilled db absent: {REAL_DB} (set ABE_REAL_DB to override)")
    frame = CacheAdapter(REAL_DB).fetch("AGG")
    if frame.empty:
        pytest.skip("real db exists but holds no AGG rows (backfill not run yet)")
    last = date.fromisoformat(str(frame.index[-1]))
    cutoff = (last - timedelta(days=3653)).isoformat()
    tail = frame.loc[frame.index >= cutoff, "adj_close"].astype(float)
    assert len(tail) > 2000  # ~252 × 10 trading days
    log_returns = np.diff(np.log(tail.to_numpy()))
    annualized = float(log_returns.mean() * constants.TRADING_DAYS)
    assert annualized > 0.01, (
        f"AGG trailing-10y annualized mean log-return {annualized:.4f} <= 1%: "
        "adj_close looks price-only, not total-return"
    )
