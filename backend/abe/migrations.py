"""Forward-only versioned schema migrations for the SQLite store (Track 2 Step 16).

Replaces the pre-Track-2 ``PRAGMA user_version`` stamp with an ordered list of
``(from_version, fn)`` migrations applied by :func:`abe.storage.ensure_schema`.

One source of truth
===================
Table DDL lives HERE, frozen at the version that introduced it — never
re-derived from the live schema. :data:`abe.storage.ALL_TABLES` owns the set of
current table *names* (the insert-boundary allow-list); this module owns the DDL
that *creates* them. Splitting names from DDL keeps the frozen migrations from
drifting when a later version changes a table.

Fresh vs. existing db converge
==============================
A brand-new db is built by running EVERY migration in order (v0 -> v1 -> v2 ->
...); an existing db runs only the migrations past its stamped ``user_version``.
Both paths therefore land on the identical schema — asserted by
``tests/test_storage.py::test_fresh_and_migrated_schema_equivalent`` (the
measurement-validity anchor that guards against fresh/migrated drift).

Transactionality
================
:func:`apply` runs each migration inside a transaction (SQLite DDL *and*
``PRAGMA user_version`` are both transactional), so the schema change and the
version bump commit atomically: a crash mid-migration rolls back with no partial
schema and no advanced version. Each migration is additionally written to be
individually idempotent (``CREATE ... IF NOT EXISTS``, a column-exists guard
before ``ALTER TABLE``, a seed-exists guard) as belt-and-suspenders.

Self-containment
================
Migrations use raw SQL (not ``storage.insert_row``) and a local timestamp helper
rather than importing live library code: a frozen historical migration must keep
producing the same result even if the coercion boundary or the ``utc_now_iso``
formatter changes in a future version. This is deliberate — do not "DRY" the
seed through the live insert path.
"""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

__all__ = [
    "CENTRAL_CONFIG_NAME",
    "DEFAULT_VIEW_SCENARIO_NAME",
    "MIGRATIONS",
    "TARGET_VERSION",
    "Migration",
    "apply",
]

Migration = tuple[int, Callable[[sqlite3.Connection], None]]
"""``(from_version, fn)``: :func:`apply` runs ``fn`` when the live
``user_version`` equals ``from_version``, then stamps ``from_version + 1``."""

# Registry keys the seeded central Config names (defined for real in the Step 18
# registries). Kept as literals in this frozen migration on purpose — a v1->v2
# migration must not depend on a live registry constant that a later step could
# rename; the seed row is a historical fact.
CENTRAL_CONFIG_NAME: Final[str] = "central"
DEFAULT_VIEW_SCENARIO_NAME: Final[str] = "Forecast (default)"
_CENTRAL_FEATURE_SET: Final[str] = "basic"
_CENTRAL_FORECASTER: Final[str] = "ewma"
_CENTRAL_OPTIMIZER: Final[str] = "mvu"
_FORECAST_VIEW_KIND: Final[str] = "forecast"


def _utc_now_iso() -> str:
    """UTC now, ISO-8601 with a ``Z`` suffix (the ``*_at_utc`` column format).

    Local to this module by design (see the module docstring's Self-containment
    note) — a frozen migration owns its own timestamp formatter.
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in conn.execute(f'PRAGMA table_info("{table}")'))


# --------------------------------------------------------------------------- #
# Migration 0 -> 1: the original single-config schema (frozen v1 snapshot).
# These DDL strings are the byte-for-byte v1 tables/indexes that lived in
# storage.py before Track 2 — moved here, unchanged, so v0->v1 reproduces the
# exact pre-Track-2 schema.
# --------------------------------------------------------------------------- #

_V1_TABLE_DDL: Final[tuple[str, ...]] = (
    """
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
    """
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
    """
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
    """
        CREATE TABLE IF NOT EXISTS macro (
            series_id       TEXT NOT NULL,
            obs_date        TEXT NOT NULL,
            value           REAL,
            available_date  TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            PRIMARY KEY (series_id, obs_date)
        )
    """,
    """
        CREATE TABLE IF NOT EXISTS features (
            run_id INTEGER NOT NULL REFERENCES runs(run_id),
            asset  TEXT NOT NULL,
            name   TEXT NOT NULL,
            value  REAL,
            PRIMARY KEY (run_id, asset, name)
        )
    """,
    """
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
    """
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
    """
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
)

_V1_INDEX_DDL: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_runs_status_run_id ON runs (status, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_features_asset_run_id ON features (asset, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_forecasts_asset_run_id ON forecasts (asset, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_bl_posteriors_asset_run_id ON bl_posteriors (asset, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_target_weights_asset_run_id ON target_weights (asset, run_id)",
)


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """Create the original single-config schema (the pre-Track-2 v1 tables)."""
    for table_ddl in _V1_TABLE_DDL:
        conn.execute(table_ddl)
    for index_ddl in _V1_INDEX_DDL:
        conn.execute(index_ddl)


# --------------------------------------------------------------------------- #
# Migration 1 -> 2: pluggable-engine schema (Track 2). Adds configs +
# view_scenarios, a nullable runs.config_id FK, seeds the central Config + the
# default forecast ViewScenario, and backfills existing runs to the central id.
# --------------------------------------------------------------------------- #

_V2_VIEW_SCENARIOS_DDL: Final[str] = """
    CREATE TABLE IF NOT EXISTS view_scenarios (
        view_scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name             TEXT NOT NULL,
        kind             TEXT NOT NULL
            CHECK (kind IN ('forecast', 'historical', 'counterfactual')),
        payload_json     TEXT,
        created_at_utc   TEXT NOT NULL
    )
"""

_V2_CONFIGS_DDL: Final[str] = """
    CREATE TABLE IF NOT EXISTS configs (
        config_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name             TEXT NOT NULL UNIQUE,
        feature_set      TEXT NOT NULL,
        forecaster       TEXT NOT NULL,
        view_scenario_id INTEGER NOT NULL REFERENCES view_scenarios(view_scenario_id),
        optimizer        TEXT NOT NULL,
        params_json      TEXT,
        is_central       INTEGER NOT NULL DEFAULT 0 CHECK (is_central IN (0, 1)),
        created_at_utc   TEXT NOT NULL
    )
"""

# At most one central Config: a partial unique index over the rows where
# is_central = 1. The Step 25 set-central guard rides on this DB invariant.
_V2_SINGLE_CENTRAL_INDEX: Final[str] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_configs_single_central "
    "ON configs (is_central) WHERE is_central = 1"
)

_V2_CONFIG_INDEXES: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_runs_config_id ON runs (config_id, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_configs_view_scenario ON configs (view_scenario_id)",
)


def _seed_central(conn: sqlite3.Connection) -> int:
    """Insert the default forecast ViewScenario + the central Config; return its id.

    The central Config reproduces the pre-Track-2 recipe: ``basic`` features +
    ``ewma`` forecaster + ``forecast`` views + ``mvu`` optimizer (Step 19's
    parity golden asserts this reproduces the V1 pipeline byte-for-byte).
    """
    now = _utc_now_iso()
    view_cursor = conn.execute(
        "INSERT INTO view_scenarios (name, kind, payload_json, created_at_utc) "
        "VALUES (?, ?, ?, ?)",
        (DEFAULT_VIEW_SCENARIO_NAME, _FORECAST_VIEW_KIND, "{}", now),
    )
    view_scenario_id = view_cursor.lastrowid
    if view_scenario_id is None:  # pragma: no cover — INSERT always yields a rowid
        raise RuntimeError("default view_scenario insert returned no view_scenario_id")
    config_cursor = conn.execute(
        "INSERT INTO configs "
        "(name, feature_set, forecaster, view_scenario_id, optimizer, "
        " params_json, is_central, created_at_utc) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            CENTRAL_CONFIG_NAME,
            _CENTRAL_FEATURE_SET,
            _CENTRAL_FORECASTER,
            view_scenario_id,
            _CENTRAL_OPTIMIZER,
            None,
            1,
            now,
        ),
    )
    config_id = config_cursor.lastrowid
    if config_id is None:  # pragma: no cover — INSERT always yields a rowid
        raise RuntimeError("central config insert returned no config_id")
    return int(config_id)


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    conn.execute(_V2_VIEW_SCENARIOS_DDL)
    conn.execute(_V2_CONFIGS_DDL)
    conn.execute(_V2_SINGLE_CENTRAL_INDEX)
    if not _has_column(conn, "runs", "config_id"):
        # ADD COLUMN with a NULL-default REFERENCES clause is legal in SQLite;
        # existing runs get config_id = NULL until the backfill below. Must
        # precede idx_runs_config_id (which indexes the new column).
        conn.execute("ALTER TABLE runs ADD COLUMN config_id INTEGER REFERENCES configs(config_id)")
    for index_ddl in _V2_CONFIG_INDEXES:
        conn.execute(index_ddl)
    central_row = conn.execute("SELECT config_id FROM configs WHERE is_central = 1").fetchone()
    central_id = int(central_row[0]) if central_row is not None else _seed_central(conn)
    # Backfill: attribute every pre-Track-2 run to the central Config. No-op on a
    # fresh db (no runs yet).
    conn.execute("UPDATE runs SET config_id = ? WHERE config_id IS NULL", (central_id,))


# --------------------------------------------------------------------------- #
# Migration 2 -> 3: configs.updated_at_utc (Track 2 Step 25). Lets the on-demand
# cache detect a recipe edit — a config edited after its last cached run is stale
# (abe.pipeline.cached_config_run compares this against the run's started_at_utc).
# --------------------------------------------------------------------------- #


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    if not _has_column(conn, "configs", "updated_at_utc"):
        conn.execute("ALTER TABLE configs ADD COLUMN updated_at_utc TEXT")
    # Backfill: an un-edited config's "updated" is its creation time.
    conn.execute(
        "UPDATE configs SET updated_at_utc = created_at_utc WHERE updated_at_utc IS NULL"
    )


MIGRATIONS: Final[tuple[Migration, ...]] = (
    (0, _migrate_v0_to_v1),
    (1, _migrate_v1_to_v2),
    (2, _migrate_v2_to_v3),
)

TARGET_VERSION: Final[int] = len(MIGRATIONS)
"""The schema version a fully-migrated db lands on (one per migration)."""


def _apply_one(conn: sqlite3.Connection, fn: Callable[[sqlite3.Connection], None], to: int) -> None:
    """Run one migration + stamp its version atomically (own the txn if none open).

    Atomicity holds ONLY when this function owns the transaction — which it does
    for the sole production caller (``storage.open_writer`` runs ``ensure_schema``
    on an autocommit connection, so ``manage_txn`` is always True). If a future
    caller wraps ``ensure_schema`` in its own transaction, that caller becomes
    responsible for rolling back on error: a swallowed exception would otherwise
    leave a version bump + partial DDL uncommitted-but-pending in the caller's
    txn. Prefer calling ``ensure_schema`` in autocommit mode.
    """
    manage_txn = not conn.in_transaction
    if manage_txn:
        conn.execute("BEGIN IMMEDIATE")
    try:
        fn(conn)
        conn.execute(f"PRAGMA user_version = {to:d}")
        if manage_txn:
            conn.execute("COMMIT")
    except BaseException:
        if manage_txn and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def apply(conn: sqlite3.Connection) -> None:
    """Bring ``conn`` up to :data:`TARGET_VERSION`, running each pending migration.

    Idempotent: a db already at the target is untouched (each migration runs only
    when the live ``user_version`` equals its ``from_version``). Re-reads
    ``user_version`` per step so a migration's own version bump gates the next.
    """
    for from_version, fn in MIGRATIONS:
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if current != from_version:
            continue
        _apply_one(conn, fn, from_version + 1)
