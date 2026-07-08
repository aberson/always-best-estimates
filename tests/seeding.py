"""Synthetic-price seeding helpers shared by the Step 8 pipeline/API tests.

Real tmp SQLite dbs, seeded through the production ``storage.upsert_row``
boundary — no mocks. 70 business days is the standard seed: 69 daily returns
covers both ``MIN_LW_ROWS`` (60, Ledoit-Wolf) and ``MIN_HISTORY_BARS`` (52,
EWMA).

Also home of :data:`OFFLINE_SCHEDULER` — the ONE shared offline scheduler
config every suite that boots the lifespan imports (one source of truth;
code-quality rule).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from abe import storage
from abe.constants import UNIVERSE
from abe.scheduler import SchedulerConfig

DRIFTS = {"SPY": 0.0005, "ACWI": 0.0004, "AGG": 0.0001}
VOLS = {"SPY": 0.010, "ACWI": 0.011, "AGG": 0.003}

FETCHED_AT = "2026-07-07T00:00:00Z"

OFFLINE_SCHEDULER = SchedulerConfig(daily_fetch_enabled=False)
"""Production scheduler defaults minus the daily NETWORK fetch. Every offline
test that enters the app lifespan (or starts a Scheduler against the real
run_pipeline) must use this — the default config would construct a real
YFinanceAdapter once the fetch window is due. The fetch job itself is covered
with fake adapters in tests/test_scheduler.py."""


def _upsert_bar(conn: sqlite3.Connection, asset: str, date_key: str, price: float) -> None:
    storage.upsert_row(
        conn,
        "prices",
        {
            "asset": asset,
            "date": date_key,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 1000,
            "source": "cache",
            "fetched_at_utc": FETCHED_AT,
        },
    )


def seed_prices(db_path: Path, *, days: int = 70, start: str = "2026-01-02", seed: int = 7) -> str:
    """Seed a geometric-random-walk daily series per universe asset.

    Returns the last (max) seeded ISO date. Deterministic for a given seed.
    """
    dates = [stamp.strftime("%Y-%m-%d") for stamp in pd.bdate_range(start, periods=days)]
    rng = np.random.default_rng(seed)
    conn = storage.open_writer(db_path)
    try:
        for asset in UNIVERSE:
            daily = rng.normal(DRIFTS[asset], VOLS[asset], size=days - 1)
            prices = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(daily))))
            for date_key, price in zip(dates, prices, strict=True):
                _upsert_bar(conn, asset, date_key, float(price))
    finally:
        conn.close()
    return dates[-1]


def append_price_day(db_path: Path, date_key: str, *, price: float = 100.0) -> None:
    """Append one new trading day for every asset (advances MAX(date))."""
    conn = storage.open_writer(db_path)
    try:
        for asset in UNIVERSE:
            _upsert_bar(conn, asset, date_key, price)
    finally:
        conn.close()


def revise_prices(db_path: Path, *, fetched_at: str, factor: float = 0.7) -> None:
    """Rewrite every stored adj_close in place (same dates, scaled values).

    Mirrors Step 3's backward-adjustment full refresh: every row is
    re-upserted through the production ``storage.upsert_row`` path with a
    FRESH ``fetched_at_utc`` stamp — ``MAX(date)`` is unchanged, but the
    ``MAX(fetched_at_utc)`` revision watermark moves.
    """
    conn = storage.open_writer(db_path)
    try:
        rows = conn.execute(
            "SELECT asset, date, open, high, low, close, adj_close, volume, source FROM prices"
        ).fetchall()
        for asset, date_key, open_, high, low, close, adj_close, volume, source in rows:
            storage.upsert_row(
                conn,
                "prices",
                {
                    "asset": str(asset),
                    "date": str(date_key),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": float(adj_close) * factor,
                    "volume": volume,
                    "source": str(source),
                    "fetched_at_utc": fetched_at,
                },
            )
    finally:
        conn.close()
