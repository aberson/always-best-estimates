"""Incremental price ingest with an overlap-consistency check, through storage.

For each asset in ``constants.UNIVERSE``:

1. Empty table → full-history fetch (``start=None``), upsert everything.
2. Otherwise fetch from ``stored MAX(date) - OVERLAP_DAYS`` (INCLUSIVE — Yahoo
   clamps a pre-inception start to the series start) and compare the fetched
   ``adj_close`` values against the stored rows over the overlap window.

   - Overlap consistent → upsert the whole fetched frame. Overlap rows are no-op
     updates by primary key; this also self-heals a last bar that was stored
     intraday-partial by an earlier fetch.
   - Overlap MISMATCH (relative tolerance :data:`ADJ_REL_TOL`) → a distribution
     rescaled history (``auto_adjust=True`` adjustments are backward-looking:
     AGG pays monthly, so appended new-basis rows would silently degrade the
     stored series toward price-only — the exact bond bug the Step 3 Done-when
     guards). Log it and run a FULL refresh: refetch ``start=None`` and upsert
     everything (idempotent by primary key).

   The last stored bar is upserted but does NOT vote in the mismatch comparison:
   it legitimately changes when the prior fetch happened intraday, which is not
   evidence of an adjustment-basis change.

Rows are stamped ``source=<adapter.source>`` + ``fetched_at_utc`` (UTC ISO-8601),
date keys are asserted ``YYYY-MM-DD`` at the write boundary (plan section 3
date-key rule), and NaN cells are converted to explicit ``None`` HERE — the
storage coercion boundary rejects NaN by contract.

Rate limits: yfinance 429s (``YFRateLimitError`` / HTTP-429-shaped errors) get
exponential backoff (sleeps :data:`RATE_LIMIT_SLEEPS` = 2/8/32s between tries),
then a loud ``RuntimeError``. Detection is duck-typed on class name + message so
this module never imports yfinance (the offline test suite stays yfinance-free).

CLI (one-time backfill, or the daily incremental fetch job)::

    uv run python -m abe.ingest.prices --backfill [--db <path>]

Incremental ingest on an empty table IS the full-history backfill, so ``--backfill``
and a bare invocation share one code path; the flag documents first-run intent.
Any universe asset left with zero stored rows after a run exits non-zero.
"""

import argparse
import logging
import math
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import pandas as pd

from abe import constants, storage
from abe.ingest.sources import (
    DATE_KEY_RE,
    PRICE_COLUMNS,
    SourceAdapter,
    YFinanceAdapter,
    utc_now_iso,
)

__all__ = [
    "ADJ_REL_TOL",
    "OVERLAP_DAYS",
    "RATE_LIMIT_SLEEPS",
    "ingest_prices",
    "main",
    "stored_max_date",
]

logger = logging.getLogger(__name__)

RATE_LIMIT_SLEEPS: Final[tuple[float, ...]] = (2.0, 8.0, 32.0)
"""Backoff sleeps between rate-limited attempts (budget: 1 try + 3 retries)."""

OVERLAP_DAYS: Final[int] = 10
"""Calendar days of stored history re-fetched each incremental run for the
adjustment-consistency check (and last-bar self-heal)."""

ADJ_REL_TOL: Final[float] = 1e-4
"""Relative tolerance for the overlap adj_close comparison. Real adjustment
steps (dividends) are ~0.2%+, comfortably above this; float noise is below."""


def stored_max_date(conn: sqlite3.Connection, asset: str) -> str | None:
    """Latest stored trading date for ``asset``, or ``None`` when nothing is stored.

    ``None`` tells the caller to run a full-history fetch instead of an
    incremental one.
    """
    row = conn.execute("SELECT MAX(date) FROM prices WHERE asset = ?", (asset,)).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when ``exc`` looks like a Yahoo rate-limit (HTTP 429) error.

    Duck-typed: yfinance's ``YFRateLimitError`` is matched by class name so this
    module never imports yfinance; other shapes match on the message text.
    """
    if any(cls.__name__ == "YFRateLimitError" for cls in type(exc).__mro__):
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _fetch_with_backoff(
    adapter: SourceAdapter,
    asset: str,
    start: str | None,
    sleep_fn: Callable[[float], None],
) -> pd.DataFrame:
    remaining = list(RATE_LIMIT_SLEEPS)
    while True:
        try:
            return adapter.fetch(asset, start=start)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            if not remaining:
                raise RuntimeError(
                    f"rate-limited fetching {asset!r}: retry budget exhausted after "
                    f"{len(RATE_LIMIT_SLEEPS) + 1} attempts (sleeps {RATE_LIMIT_SLEEPS})"
                ) from exc
            sleep_fn(remaining.pop(0))


def _none_if_nan(value: object) -> object:
    """The NaN → explicit-None conversion required by storage's insert boundary."""
    return None if pd.isna(value) else value


def _validated_sorted(asset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Assert the adapter frame's contract (columns + date-key shape), sort ascending.

    Sorting here means resume-after-crash correctness (MAX(date) reflects a
    contiguous prefix) never depends on adapter goodwill.
    """
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"adapter frame for {asset!r} is missing column(s) {missing!r}; "
            f"expected exactly {list(PRICE_COLUMNS)}"
        )
    for date_key in frame.index:
        if not isinstance(date_key, str) or not DATE_KEY_RE.fullmatch(date_key):
            raise ValueError(
                f"date key {date_key!r} for {asset!r} is not ISO-8601 YYYY-MM-DD "
                "(plan section 3 date-key rule)"
            )
    return frame.sort_index()


def _upsert_frame(
    conn: sqlite3.Connection,
    asset: str,
    frame: pd.DataFrame,
    source: str,
    new_after: str | None = None,
) -> int:
    """Upsert every row of a validated frame; count rows with date > ``new_after``.

    ``new_after=None`` counts everything (full-history / full-refresh runs).
    Overlap rows re-upserted at or below ``new_after`` are updates by primary
    key and deliberately not counted as new.
    """
    fetched_at = utc_now_iso()
    new_rows = 0
    for date_key, bar in frame.iterrows():
        row: dict[str, object] = {
            "asset": asset,
            "date": str(date_key),
            "source": source,
            "fetched_at_utc": fetched_at,
        }
        for column in PRICE_COLUMNS:
            row[column] = _none_if_nan(bar[column])
        storage.upsert_row(conn, "prices", row)
        if new_after is None or str(date_key) > new_after:
            new_rows += 1
    return new_rows


def _overlap_mismatches(
    conn: sqlite3.Connection,
    asset: str,
    frame: pd.DataFrame,
    overlap_start: str,
    max_date: str,
) -> list[str]:
    """Dates in ``[overlap_start, max_date)`` where fetched adj_close != stored.

    Any hit means the provider re-based history (backward adjustment) since the
    stored rows were written. The ``max_date`` row itself is excluded: it changes
    legitimately when the previous fetch was intraday.
    """
    stored: dict[str, float | None] = {
        str(row[0]): row[1]
        for row in conn.execute(
            "SELECT date, adj_close FROM prices WHERE asset = ? AND date >= ? AND date < ?",
            (asset, overlap_start, max_date),
        )
    }
    mismatched: list[str] = []
    for date_key, fetched in frame["adj_close"].items():
        day = str(date_key)
        if day >= max_date or day not in stored:
            continue
        stored_value = stored[day]
        fetched_is_missing = bool(pd.isna(fetched))
        if fetched_is_missing and stored_value is None:
            continue
        if fetched_is_missing != (stored_value is None):
            mismatched.append(day)
        elif not math.isclose(float(fetched), float(stored_value or 0.0), rel_tol=ADJ_REL_TOL):
            mismatched.append(day)
    return mismatched


def ingest_prices(
    conn: sqlite3.Connection,
    adapter: SourceAdapter,
    *,
    assets: Sequence[str] = constants.UNIVERSE,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Ingest daily bars for ``assets``; return NEW rows (date > stored max) per asset.

    A full-history or full-refresh run counts every upserted row instead (there
    is no meaningful old/new split when the whole series is (re)written).
    ``sleep_fn`` is injectable so backoff tests never really sleep.
    """
    counts: dict[str, int] = {}
    for asset in assets:
        max_date = stored_max_date(conn, asset)
        if max_date is None:
            frame = _validated_sorted(asset, _fetch_with_backoff(adapter, asset, None, sleep_fn))
            counts[asset] = _upsert_frame(conn, asset, frame, adapter.source)
            continue
        overlap_start = (date.fromisoformat(max_date) - timedelta(days=OVERLAP_DAYS)).isoformat()
        frame = _validated_sorted(
            asset, _fetch_with_backoff(adapter, asset, overlap_start, sleep_fn)
        )
        mismatched = _overlap_mismatches(conn, asset, frame, overlap_start, max_date)
        if mismatched:
            logger.warning(
                "adj_close overlap mismatch for %s on %d date(s) (e.g. %s): backward "
                "adjustment detected — running full refresh",
                asset,
                len(mismatched),
                mismatched[0],
            )
            frame = _validated_sorted(asset, _fetch_with_backoff(adapter, asset, None, sleep_fn))
            counts[asset] = _upsert_frame(conn, asset, frame, adapter.source)
        else:
            counts[asset] = _upsert_frame(conn, asset, frame, adapter.source, new_after=max_date)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m abe.ingest.prices --backfill [--db <path>]``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m abe.ingest.prices",
        description=(
            "Incremental price ingest for the fixed universe "
            f"{constants.UNIVERSE}: fetches from stored MAX(date) - {OVERLAP_DAYS} days "
            "(adjustment-consistency overlap); on an empty table this is the "
            "full-history backfill."
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="explicit first-run intent (full history lands whenever the table is empty)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=storage.DEFAULT_DB_PATH,
        help=f"SQLite db path (default: {storage.DEFAULT_DB_PATH})",
    )
    args = parser.parse_args(argv)
    conn = storage.open_writer(args.db)
    try:
        counts = ingest_prices(conn, YFinanceAdapter())
        empty_assets: list[str] = []
        for asset, written in counts.items():
            total, first, last = conn.execute(
                "SELECT COUNT(*), MIN(date), MAX(date) FROM prices WHERE asset = ?",
                (asset,),
            ).fetchone()
            print(f"{asset}: +{written} rows (total {total}, {first} .. {last})")
            if int(total) == 0:
                empty_assets.append(asset)
        if empty_assets:
            print(
                f"ERROR: no rows stored for {empty_assets} after ingest — treating as a failed run",
                file=sys.stderr,
            )
            return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
