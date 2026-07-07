"""SQLite storage layer: connections, PRAGMAs, schema DDL (v1), coercion boundary.

plan.md section 3 (Data Store) is the authoritative spec for everything here.

One-writer discipline: the pipeline thread owns THE single writer connection
(``open_writer``; ownership lands in Step 8). The API layer reads through
short-lived read-only connections (``open_read_only``, ``file:...?mode=ro``).
The scheduler calls ``wal_checkpoint_truncate`` after each run (Step 11).

All inserts flow through ONE coercion boundary (``coerce_scalar``, applied by
``insert_row``/``upsert_row``): numpy scalars (``np.float64``, ``np.int64``, ...)
and 0-dim ``torch.Tensor`` values are converted to native Python types before the
write — otherwise SQLite persists them as BLOBs or refuses the bind.

Missing values are explicit ``None``, never NaN: the boundary REJECTS NaN floats
with ``ValueError`` (SQLite would otherwise bind NaN as a silent NULL). Callers
that parse NaN-bearing sources (e.g. Step 4's macro ingest) must convert
NaN → ``None`` deliberately before insert.

Connections are opened in autocommit mode (``isolation_level=None``): every
statement commits immediately. Per-run transactions (``BEGIN IMMEDIATE`` keyed by
``run_id``) are the pipeline's responsibility (Step 8).
"""

import math
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from abe.constants import UNIVERSE

__all__ = [
    "DEFAULT_DB_PATH",
    "SCHEMA_VERSION",
    "TABLES",
    "SQLiteScalar",
    "coerce_scalar",
    "ensure_schema",
    "insert_row",
    "latest_ok_run_id",
    "open_read_only",
    "open_writer",
    "upsert_row",
    "wal_checkpoint_truncate",
]

SQLiteScalar = int | float | str | bytes | None
"""The only value types that may cross the insert boundary into SQLite."""

DEFAULT_DB_PATH: Final[Path] = Path("data") / "abe.db"
"""Default DB location, relative to the project root (the process cwd)."""

SCHEMA_VERSION: Final[int] = 1
"""Stamped into ``PRAGMA user_version`` by :func:`ensure_schema`."""

# --------------------------------------------------------------------------- #
# Schema DDL (v1) — plan.md section 3. One source of truth: table + index DDL
# live ONLY here; insert helpers derive column/PK facts from the live schema
# via PRAGMA table_info, so nothing below is duplicated in Python constants.
# --------------------------------------------------------------------------- #

_SCHEMA_DDL: Final[dict[str, str]] = {
    "runs": """
        CREATE TABLE IF NOT EXISTS runs (
            run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_utc  TEXT NOT NULL,
            finished_at_utc TEXT,
            status          TEXT NOT NULL
                CHECK (status IN ('queued', 'running', 'ok', 'error', 'skipped')),
            "trigger"       TEXT NOT NULL
                CHECK ("trigger" IN ('schedule', 'manual', 'startup')),
            error_text      TEXT
        )
    """,
    "run_stages": """
        CREATE TABLE IF NOT EXISTS run_stages (
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            stage           TEXT NOT NULL,
            status          TEXT NOT NULL,
            started_at_utc  TEXT,
            finished_at_utc TEXT,
            detail_json     TEXT,
            PRIMARY KEY (run_id, stage)
        )
    """,
    "prices": """
        CREATE TABLE IF NOT EXISTS prices (
            asset          TEXT NOT NULL,
            date           TEXT NOT NULL,
            open           REAL,
            high           REAL,
            low            REAL,
            close          REAL,
            adj_close      REAL,
            volume         INTEGER,
            source         TEXT NOT NULL CHECK (source IN ('yfinance', 'cache')),
            fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY (asset, date)
        )
    """,
    "macro": """
        CREATE TABLE IF NOT EXISTS macro (
            series_id       TEXT NOT NULL,
            obs_date        TEXT NOT NULL,
            value           REAL,
            available_date  TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            PRIMARY KEY (series_id, obs_date)
        )
    """,
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            run_id INTEGER NOT NULL REFERENCES runs(run_id),
            asset  TEXT NOT NULL,
            name   TEXT NOT NULL,
            value  REAL,
            PRIMARY KEY (run_id, asset, name)
        )
    """,
    "forecasts": """
        CREATE TABLE IF NOT EXISTS forecasts (
            run_id        INTEGER NOT NULL REFERENCES runs(run_id),
            asset         TEXT NOT NULL,
            horizon_days  INTEGER NOT NULL,
            mu            REAL NOT NULL,
            sigma         REAL NOT NULL,
            model_version TEXT NOT NULL,
            PRIMARY KEY (run_id, asset)
        )
    """,
    "bl_posteriors": """
        CREATE TABLE IF NOT EXISTS bl_posteriors (
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            asset           TEXT NOT NULL,
            prior_mu        REAL NOT NULL,
            view_mu         REAL NOT NULL,
            view_confidence REAL NOT NULL,
            posterior_mu    REAL NOT NULL,
            posterior_sigma REAL NOT NULL,
            detail_json     TEXT,
            PRIMARY KEY (run_id, asset)
        )
    """,
    "target_weights": """
        CREATE TABLE IF NOT EXISTS target_weights (
            run_id           INTEGER NOT NULL REFERENCES runs(run_id),
            asset            TEXT NOT NULL,
            weight           REAL NOT NULL,
            prev_weight      REAL,
            turnover         REAL,
            relaxed_turnover INTEGER NOT NULL DEFAULT 0
                CHECK (relaxed_turnover IN (0, 1)),
            PRIMARY KEY (run_id, asset)
        )
    """,
}

_INDEX_DDL: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_runs_status_run_id ON runs (status, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_features_asset_run_id ON features (asset, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_forecasts_asset_run_id ON forecasts (asset, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_bl_posteriors_asset_run_id ON bl_posteriors (asset, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_target_weights_asset_run_id ON target_weights (asset, run_id)",
)

TABLES: Final[frozenset[str]] = frozenset(_SCHEMA_DDL)
"""All v1 table names (derived from the DDL — do not redefine elsewhere)."""


# --------------------------------------------------------------------------- #
# Connections + PRAGMAs
# --------------------------------------------------------------------------- #


def _apply_writer_pragmas(conn: sqlite3.Connection) -> None:
    """Set the plan section-3 PRAGMAs once on a fresh writer connection."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")


def open_writer(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open THE single writer connection (one-writer discipline).

    Exactly one of these should exist per process; the pipeline thread owns it
    (Step 8). Creates the parent directory if missing, applies the WAL/PRAGMA
    set, and ensures the v1 schema exists. ``db_path`` is configurable for
    tests; production uses :data:`DEFAULT_DB_PATH`.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        _apply_writer_pragmas(conn)
        ensure_schema(conn)
    except BaseException:
        # Never leak a half-configured connection (it would keep the file
        # locked on Windows, e.g. after a DatabaseError on a corrupt db).
        conn.close()
        raise
    return conn


def open_read_only(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a short-lived read-only connection (``file:...?mode=ro`` URI).

    This is the API layer's read path: it can never write (SQLite enforces
    ``mode=ro`` at open; ``query_only`` is set as defense in depth).
    """
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA query_only = ON")
    except BaseException:
        conn.close()
        raise
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the v1 schema from scratch. Idempotent (CREATE ... IF NOT EXISTS)."""
    for table_ddl in _SCHEMA_DDL.values():
        conn.execute(table_ddl)
    for index_ddl in _INDEX_DDL:
        conn.execute(index_ddl)
    row = conn.execute("PRAGMA user_version").fetchone()
    if int(row[0]) < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION:d}")


def wal_checkpoint_truncate(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` (scheduler calls this after each run).

    Returns SQLite's ``(busy, log_frames, checkpointed_frames)`` triple;
    ``busy == 0`` means the checkpoint completed. Must be invoked on the thread
    that owns ``conn`` (sqlite3 defaults to ``check_same_thread=True``) — Step
    11's scheduler must dispatch it through the pipeline's executor thread.
    """
    row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if row is None:  # pragma: no cover — SQLite always returns one row
        raise RuntimeError("PRAGMA wal_checkpoint(TRUNCATE) returned no row")
    return (int(row[0]), int(row[1]), int(row[2]))


# --------------------------------------------------------------------------- #
# Coercion boundary — ALL inserts flow through coerce_scalar via the helpers
# --------------------------------------------------------------------------- #


def coerce_scalar(value: object) -> SQLiteScalar:
    """Coerce a value to a native Python scalar SQLite can store natively.

    Handles numpy scalars (``np.float64``, ``np.int64``, ``np.bool_``, ...),
    0-dim numpy arrays, and 0-dim ``torch.Tensor`` via their ``.item()`` —
    genuinely 0-dim only; 1-element containers are rejected, not unwrapped.
    Native ``None | bool | int | float | str | bytes`` pass through (bool as
    int). NaN floats raise ``ValueError`` (missing values must be explicit
    ``None`` — SQLite would bind NaN as a silent NULL). Anything else raises
    ``TypeError`` — better a loud failure at the boundary than a silent BLOB
    in the DB.
    """
    return _coerce_scalar(value, allow_item=True)


def _checked_float(value: float) -> float:
    if math.isnan(value):
        raise ValueError(
            "NaN cannot cross the insert boundary; pass None explicitly for missing values"
        )
    return value


def _coerce_scalar(value: object, allow_item: bool) -> SQLiteScalar:
    if value is None:
        return None
    if type(value) is bool:
        return int(value)
    if type(value) is int:
        return value
    if type(value) is float:
        return _checked_float(value)
    if type(value) is str:
        return value
    if type(value) is bytes:
        return value
    if allow_item and getattr(value, "ndim", None) == 0:
        # Genuinely 0-dim only: numpy scalar / 0-dim ndarray / 0-dim
        # torch.Tensor → native scalar. 1-element arrays/Series are NOT
        # unwrapped — that would mask an indexing slip upstream.
        item_method = getattr(value, "item", None)
        if callable(item_method):
            return _coerce_scalar(item_method(), allow_item=False)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _checked_float(float(value))
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bytes):
        return bytes(value)
    raise TypeError(f"cannot coerce {type(value).__qualname__} to a SQLite scalar")


def _validated_columns(conn: sqlite3.Connection, table: str) -> dict[str, bool]:
    """Map column name → is-primary-key for a known table, from the live schema.

    The membership check against the DDL registry (``_SCHEMA_DDL``) also makes
    the interpolated identifiers injection-proof.
    """
    if table not in _SCHEMA_DDL:
        raise ValueError(f"unknown table {table!r}; known tables: {sorted(_SCHEMA_DDL)}")
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not rows:
        raise ValueError(f"table {table!r} does not exist; call ensure_schema() first")
    return {str(row[1]): bool(row[5]) for row in rows}


def _prepare_row(
    conn: sqlite3.Connection, table: str, row: Mapping[str, object]
) -> tuple[list[str], list[SQLiteScalar], dict[str, bool]]:
    """Validate identifiers + apply the coercion boundary to every value."""
    if not row:
        raise ValueError("row must contain at least one column")
    columns = _validated_columns(conn, table)
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ValueError(f"unknown column(s) {unknown!r} for table {table!r}")
    if "asset" in row and row["asset"] not in UNIVERSE:
        raise ValueError(f"asset {row['asset']!r} not in UNIVERSE {UNIVERSE}")
    names = list(row)
    params = [coerce_scalar(row[name]) for name in names]
    return names, params, columns


def insert_row(conn: sqlite3.Connection, table: str, row: Mapping[str, object]) -> int | None:
    """Insert one row through the coercion boundary.

    Returns the cursor's ``lastrowid`` (for ``runs`` this is the new ``run_id``).
    """
    names, params, _ = _prepare_row(conn, table, row)
    column_sql = ", ".join(f'"{name}"' for name in names)
    placeholder_sql = ", ".join("?" for _ in names)
    cursor = conn.execute(
        f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholder_sql})',
        params,
    )
    return cursor.lastrowid


def upsert_row(conn: sqlite3.Connection, table: str, row: Mapping[str, object]) -> None:
    """Upsert one row through the coercion boundary.

    ``INSERT ... ON CONFLICT (<pk>) DO UPDATE`` keyed on the table's primary
    key (derived from the live schema); re-ingesting an existing prices/macro
    row is therefore a no-op update. ``row`` must contain every PK column.
    """
    names, params, columns = _prepare_row(conn, table, row)
    pk = [name for name, is_pk in columns.items() if is_pk]
    missing_pk = [name for name in pk if name not in row]
    if missing_pk:
        raise ValueError(f"upsert into {table!r} missing primary-key column(s) {missing_pk!r}")
    column_sql = ", ".join(f'"{name}"' for name in names)
    placeholder_sql = ", ".join("?" for _ in names)
    conflict_sql = ", ".join(f'"{name}"' for name in pk)
    non_pk = [name for name in names if name not in pk]
    if non_pk:
        update_sql = ", ".join(f'"{name}" = excluded."{name}"' for name in non_pk)
        action = f"DO UPDATE SET {update_sql}"
    else:
        action = "DO NOTHING"
    conn.execute(
        f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholder_sql}) '
        f"ON CONFLICT ({conflict_sql}) {action}",
        params,
    )


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def latest_ok_run_id(conn: sqlite3.Connection) -> int | None:
    """The plan's "latest" definition: ``MAX(run_id) WHERE status='ok'``, or None."""
    row = conn.execute("SELECT MAX(run_id) FROM runs WHERE status = 'ok'").fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])
