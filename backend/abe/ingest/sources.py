"""Price-source adapters: the ``SourceAdapter`` protocol + yfinance / SQLite-cache impls.

plan.md section 3 (date-key rule) and section 8 (yfinance pins) are the authoritative
spec. Every adapter returns the SAME normalized shape:

- columns: exactly :data:`PRICE_COLUMNS` (``open, high, low, close, adj_close, volume``)
- index: ISO-8601 ``YYYY-MM-DD`` date strings, tz stripped at this boundary, ascending
- values: floats (volume may be integral); missing cells stay NaN here — the *ingest*
  layer converts NaN → ``None`` before the storage boundary (storage rejects NaN).

yfinance pins (plan section 8, re-verified against the installed 1.5.1 source):
``auto_adjust=True`` means there is NO ``Adj Close`` column — ``Close`` IS the adjusted
series, so it is mapped to BOTH ``close`` and ``adj_close`` explicitly. ``actions=False``
keeps Dividends / Stock Splits / Capital Gains columns out. The raw column set is
asserted before any rows are returned (fail loud on Yahoo schema drift).

The adapter calls ``Ticker.history`` — NOT ``yfinance.download`` — because ``download``
swallows every per-ticker exception (1.5.1 ``multi.py``: ``except Exception`` → empty
placeholder + ``logger.error``), which makes a failed download indistinguishable from
no-new-data and turns rate-limit backoff into dead code. ``Ticker.history`` raises for
real once ``yfinance.config.debug.hide_exceptions = False`` is set (the sanctioned
replacement for the deprecated ``raise_errors=True``); ``YFRateLimitError`` propagates
unconditionally either way.

``yfinance`` is imported lazily inside :meth:`YFinanceAdapter.fetch` only — the
``CacheAdapter`` path must work with zero network and zero yfinance import.

This module also owns the shared write-boundary trio :data:`DATE_FORMAT` /
:data:`DATE_KEY_RE` / :func:`utc_now_iso` — both ingest modules (prices, macro)
import them from here so the date-key shape and the ``*_at_utc`` timestamp
format have ONE source of truth (code-quality rule).
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

import pandas as pd

from abe import storage

__all__ = [
    "DATE_FORMAT",
    "DATE_KEY_RE",
    "PRICE_COLUMNS",
    "CacheAdapter",
    "SourceAdapter",
    "YFinanceAdapter",
    "normalize_yfinance_frame",
    "utc_now_iso",
]

PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "adj_close", "volume")
"""The fixed normalized column set every adapter returns (matches the prices table)."""

_RAW_YFINANCE_COLUMNS: Final[frozenset[str]] = frozenset({"Open", "High", "Low", "Close", "Volume"})
"""Exact raw column set ``Ticker.history(auto_adjust=True, actions=False)`` returns."""

DATE_FORMAT: Final[str] = "%Y-%m-%d"
"""The ISO-8601 date-key strftime format (plan section 3 date-key rule)."""

DATE_KEY_RE: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}")
"""Write-boundary shape check for date keys — shared by the prices AND macro
ingest boundaries; never re-declare it (one-source-of-truth rule)."""


def utc_now_iso() -> str:
    """UTC now as ISO-8601 with a ``Z`` suffix (the ``*_at_utc`` column format)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SourceAdapter(Protocol):
    """Fetch daily adjusted OHLCV for one asset, normalized to the fixed shape.

    ``start`` is inclusive, ``end`` exclusive (yfinance convention); both are
    ISO-8601 ``YYYY-MM-DD`` strings. ``start=None`` means full available history.
    ``source`` is the provenance label the ingest layer stamps into the prices
    table's ``source`` column (must satisfy its CHECK constraint).
    """

    source: str

    def fetch(
        self, asset: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame: ...


def _empty_normalized_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype=float) for column in PRICE_COLUMNS},
        index=pd.Index([], dtype=str, name="date"),
    )


def normalize_yfinance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw yfinance OHLCV frame to the fixed adapter shape.

    Factored out of :meth:`YFinanceAdapter.fetch` so the assertion + mapping logic
    is testable offline with canned frames. Raises ``ValueError`` loudly on any
    schema drift (unexpected/missing columns, MultiIndex columns, non-datetime
    index) instead of writing garbage. A zero-row frame normalizes to an empty
    frame regardless of its columns — with no rows there is nothing to write, so
    the pre-write schema assertion does not apply.
    """
    if raw.empty:
        return _empty_normalized_frame()
    if isinstance(raw.columns, pd.MultiIndex):
        raise ValueError(
            f"yfinance returned MultiIndex columns (schema drift): {list(raw.columns)!r}"
        )
    got = {str(column) for column in raw.columns}
    if got != set(_RAW_YFINANCE_COLUMNS):
        raise ValueError(
            "yfinance column set drifted: expected exactly "
            f"{sorted(_RAW_YFINANCE_COLUMNS)} (auto_adjust=True has no 'Adj Close', "
            f"actions=False has no Dividends/Stock Splits), got {sorted(got)}"
        )
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError(f"yfinance index is not a DatetimeIndex: {type(raw.index).__qualname__}")
    index = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    dates = pd.Index(index.strftime(DATE_FORMAT), name="date")
    normalized = pd.DataFrame(
        {
            "open": raw["Open"].to_numpy(),
            "high": raw["High"].to_numpy(),
            "low": raw["Low"].to_numpy(),
            "close": raw["Close"].to_numpy(),
            # auto_adjust=True: "Close" IS the adjusted close — map it to BOTH.
            "adj_close": raw["Close"].to_numpy(),
            # Note: volume > 2**53 would round silently through float64 — not
            # reachable at ETF share magnitudes.
            "volume": raw["Volume"].to_numpy(),
        },
        index=dates,
    )
    return normalized.sort_index()


class YFinanceAdapter:
    """Network source: yfinance daily bars via ``Ticker.history`` (raising semantics).

    See the module docstring for why this is NOT ``yfinance.download``: download
    swallows per-ticker exceptions, so failures would masquerade as no-new-data
    and the ingest layer's rate-limit backoff would never fire.
    """

    source: str = "yfinance"

    def fetch(self, asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        import yfinance  # lazy: the cache path must never touch this import

        # Fail loud, the sanctioned way (raise_errors=True is deprecated). This
        # makes YFPricesMissingError / YFTzMissingError / request errors raise
        # instead of logging; YFRateLimitError raises unconditionally.
        yfinance.config.debug.hide_exceptions = False
        kwargs: dict[str, Any] = {
            "interval": "1d",
            "auto_adjust": True,  # "Close" IS adjusted; there is no "Adj Close"
            "actions": False,  # no Dividends / Stock Splits / Capital Gains columns
        }
        ticker = yfinance.Ticker(asset)
        if start is None:
            # Full history needs an explicit period="max": yfinance defaults to
            # a 1mo window when start is absent (even with only end set), so any
            # end bound is applied post-fetch instead of passed through.
            raw = ticker.history(period="max", **kwargs)
        else:
            raw = ticker.history(start=start, end=end, **kwargs)
        if raw is None:
            raise ValueError(f"yfinance Ticker.history returned None for {asset!r}")
        frame = normalize_yfinance_frame(raw)
        if start is None:
            if frame.empty:
                raise RuntimeError(
                    f"full-history fetch for {asset!r} returned an empty frame — never "
                    "legitimate for the fixed universe; treating as a failed download"
                )
            if end is not None:
                frame = frame.loc[frame.index < end]
        return frame


class CacheAdapter:
    """Offline source: serve stored history from the SQLite ``prices`` table.

    Read-only (``storage.open_read_only``), zero network, zero yfinance import.
    Reads return whatever is stored, whoever stored it — the ``source`` column
    records the original writer; this adapter never writes.
    """

    source: str = "cache"

    def __init__(self, db_path: str | Path = storage.DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)

    def fetch(self, asset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        sql = "SELECT date, open, high, low, close, adj_close, volume FROM prices WHERE asset = ?"
        params: list[str] = [asset]
        if start is not None:
            sql += " AND date >= ?"
            params.append(start)
        if end is not None:
            sql += " AND date < ?"
            params.append(end)
        sql += " ORDER BY date"
        conn = storage.open_read_only(self._db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        if not rows:
            return _empty_normalized_frame()
        frame = pd.DataFrame(rows, columns=["date", *PRICE_COLUMNS])
        return frame.set_index("date")
