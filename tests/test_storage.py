"""Storage-layer tests: schema, PRAGMAs, coercion boundary, FK/CHECK, ro reads.

All tests run against a real tmp_path SQLite db — no mocks. Integration through
the production entry point is N/A this step (pipeline arrives Step 8); storage
is the leaf, so direct tests are correct here.
"""

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from abe import constants, migrations, storage

EXPECTED_TABLES = {
    "runs",
    "run_stages",
    "prices",
    "macro",
    "features",
    "forecasts",
    "bl_posteriors",
    "target_weights",
    # v2 (Track 2) tables
    "configs",
    "view_scenarios",
}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    # Parent dir deliberately missing: open_writer must create it (data/ handling).
    return tmp_path / "data" / "abe.db"


@pytest.fixture()
def writer(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = storage.open_writer(db_path)
    yield conn
    conn.close()


def _start_run(conn: sqlite3.Connection, status: str = "ok") -> int:
    run_id = storage.insert_row(
        conn,
        "runs",
        {"started_at_utc": "2026-07-07T00:00:00Z", "status": status, "trigger": "manual"},
    )
    assert run_id is not None
    return run_id


# --------------------------------------------------------------------------- #
# Schema + PRAGMAs
# --------------------------------------------------------------------------- #


def test_schema_creates_all_tables(writer: sqlite3.Connection) -> None:
    rows = writer.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert {row[0] for row in rows} == EXPECTED_TABLES
    assert storage.TABLES == frozenset(EXPECTED_TABLES)


def test_schema_is_idempotent(db_path: Path) -> None:
    first = storage.open_writer(db_path)
    _start_run(first)
    storage.ensure_schema(first)  # re-run on a live connection: no error
    first.close()
    second = storage.open_writer(db_path)  # re-open on an existing db: no error
    assert storage.latest_ok_run_id(second) == 1  # existing data untouched
    second.close()


def test_schema_version_stamped(writer: sqlite3.Connection) -> None:
    assert writer.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION


def test_writer_pragmas(writer: sqlite3.Connection) -> None:
    assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert writer.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# --------------------------------------------------------------------------- #
# Coercion boundary round-trips (real numpy + torch, no mocks)
# --------------------------------------------------------------------------- #


def test_numpy_float64_round_trip(writer: sqlite3.Connection) -> None:
    run_id = _start_run(writer)
    storage.insert_row(
        writer,
        "features",
        {"run_id": run_id, "asset": "SPY", "name": "vol_21d", "value": np.float64(0.1234)},
    )
    value, typeof = writer.execute(
        "SELECT value, typeof(value) FROM features WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert typeof == "real"  # not 'blob'
    assert type(value) is float
    assert value == pytest.approx(0.1234)


def test_torch_scalar_and_numpy_int_round_trip(writer: sqlite3.Connection) -> None:
    run_id = _start_run(writer)
    storage.insert_row(
        writer,
        "forecasts",
        {
            "run_id": run_id,
            "asset": "AGG",
            "horizon_days": np.int64(constants.HORIZON_BARS),
            "mu": torch.tensor(0.0217),  # 0-dim tensor
            "sigma": np.float64(0.015),
            "model_version": "ewma",
        },
    )
    row = writer.execute(
        "SELECT horizon_days, typeof(horizon_days), mu, typeof(mu), sigma, typeof(sigma) "
        "FROM forecasts WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    horizon, horizon_type, mu, mu_type, sigma, sigma_type = row
    assert (horizon_type, mu_type, sigma_type) == ("integer", "real", "real")
    assert type(horizon) is int
    assert horizon == constants.HORIZON_BARS
    assert type(mu) is float
    assert mu == pytest.approx(0.0217, abs=1e-6)
    assert type(sigma) is float


def test_coerce_scalar_units() -> None:
    coerced_np_float = storage.coerce_scalar(np.float64(1.5))
    assert type(coerced_np_float) is float
    assert coerced_np_float == 1.5

    coerced_np_int = storage.coerce_scalar(np.int64(7))
    assert type(coerced_np_int) is int
    assert coerced_np_int == 7

    coerced_tensor = storage.coerce_scalar(torch.tensor(2.5))
    assert type(coerced_tensor) is float
    assert coerced_tensor == 2.5

    coerced_np_bool = storage.coerce_scalar(np.bool_(True))
    assert type(coerced_np_bool) is int
    assert coerced_np_bool == 1

    coerced_zero_dim = storage.coerce_scalar(np.array(1.5))  # genuinely 0-dim: unwrapped
    assert type(coerced_zero_dim) is float
    assert coerced_zero_dim == 1.5

    assert storage.coerce_scalar(None) is None
    assert storage.coerce_scalar("ewma") == "ewma"
    assert storage.coerce_scalar(21) == 21
    assert storage.coerce_scalar(True) == 1

    with pytest.raises(TypeError):
        storage.coerce_scalar(object())
    with pytest.raises(TypeError):
        storage.coerce_scalar([1.0, 2.0])


def test_coerce_scalar_rejects_one_element_containers() -> None:
    """1-element arrays are an upstream indexing slip, not a scalar — reject, don't unwrap."""
    with pytest.raises(TypeError):
        storage.coerce_scalar(np.array([1.5]))
    with pytest.raises(TypeError):
        storage.coerce_scalar(torch.tensor([1.5]))


def test_coerce_scalar_rejects_nan() -> None:
    """Missing values must be explicit None; NaN would bind as a silent NULL."""
    with pytest.raises(ValueError, match="NaN"):
        storage.coerce_scalar(float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        storage.coerce_scalar(np.float64("nan"))
    assert storage.coerce_scalar(None) is None  # the sanctioned missing-value path


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def test_fk_violation_rejects_orphan_run_stage(writer: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_row(
            writer,
            "run_stages",
            {"run_id": 9999, "stage": "ingest", "status": "ok"},
        )


@pytest.mark.parametrize(
    ("table", "row"),
    [
        pytest.param(
            "runs",
            {"started_at_utc": "2026-07-07T00:00:00Z", "status": "bogus", "trigger": "manual"},
            id="bad-run-status",
        ),
        pytest.param(
            "runs",
            {"started_at_utc": "2026-07-07T00:00:00Z", "status": "ok", "trigger": "cron"},
            id="bad-run-trigger",
        ),
        pytest.param(
            "prices",
            {
                "asset": "SPY",
                "date": "2026-07-06",
                "adj_close": 500.0,
                "source": "manual",
                "fetched_at_utc": "2026-07-07T00:00:00Z",
            },
            id="bad-price-source",
        ),
    ],
)
def test_check_constraints_reject_bad_domain_values(
    writer: sqlite3.Connection, table: str, row: dict[str, object]
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_row(writer, table, row)


def test_insert_rejects_unknown_table_and_column(writer: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        storage.insert_row(writer, "positions", {"asset": "SPY"})
    with pytest.raises(ValueError, match="unknown column"):
        storage.insert_row(writer, "macro", {"series_id": "DGS10", "nope": 1.0})
    with pytest.raises(ValueError, match="at least one column"):
        storage.insert_row(writer, "macro", {})


def test_insert_rejects_asset_outside_universe(writer: sqlite3.Connection) -> None:
    run_id = _start_run(writer)
    with pytest.raises(ValueError, match="not in UNIVERSE"):
        storage.insert_row(
            writer,
            "features",
            {"run_id": run_id, "asset": "TSLA", "name": "vol_21d", "value": 0.5},
        )


# --------------------------------------------------------------------------- #
# Upsert (prices/macro dedup: re-ingest is a no-op)
# --------------------------------------------------------------------------- #


def test_upsert_price_reingest_updates_in_place(writer: sqlite3.Connection) -> None:
    base = {
        "asset": "ACWI",
        "date": "2026-07-06",
        "adj_close": np.float64(130.25),
        "volume": np.int64(1_000_000),
        "source": "yfinance",
        "fetched_at_utc": "2026-07-07T00:00:00Z",
    }
    storage.upsert_row(writer, "prices", base)
    storage.upsert_row(writer, "prices", {**base, "adj_close": np.float64(131.00)})
    rows = writer.execute(
        "SELECT adj_close, typeof(adj_close), volume, typeof(volume) FROM prices"
    ).fetchall()
    assert len(rows) == 1
    adj_close, adj_type, volume, vol_type = rows[0]
    assert (adj_type, vol_type) == ("real", "integer")
    assert adj_close == pytest.approx(131.00)
    assert type(volume) is int


def test_upsert_requires_primary_key_columns(writer: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="primary-key"):
        storage.upsert_row(writer, "macro", {"series_id": "DGS10", "value": 4.2})


# --------------------------------------------------------------------------- #
# latest_ok_run_id
# --------------------------------------------------------------------------- #


def test_latest_ok_run_id_none_on_empty(writer: sqlite3.Connection) -> None:
    assert storage.latest_ok_run_id(writer) is None


def test_latest_ok_run_id_max_ok_among_mixed(writer: sqlite3.Connection) -> None:
    _start_run(writer, status="ok")
    _start_run(writer, status="error")
    ok_id = _start_run(writer, status="ok")
    _start_run(writer, status="running")  # newer, but not ok
    latest = storage.latest_ok_run_id(writer)
    assert latest == ok_id
    assert type(latest) is int


# --------------------------------------------------------------------------- #
# One-writer discipline: read-only connections
# --------------------------------------------------------------------------- #


def test_read_only_connection_rejects_writes(writer: sqlite3.Connection, db_path: Path) -> None:
    _start_run(writer)
    ro = storage.open_read_only(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute(
                'INSERT INTO runs (started_at_utc, status, "trigger") '
                "VALUES ('2026-07-07T00:00:00Z', 'ok', 'manual')"
            )
    finally:
        ro.close()


def test_read_only_connection_sees_writer_data(writer: sqlite3.Connection, db_path: Path) -> None:
    run_id = _start_run(writer)
    ro = storage.open_read_only(db_path)
    try:
        assert storage.latest_ok_run_id(ro) == run_id
    finally:
        ro.close()


# --------------------------------------------------------------------------- #
# WAL checkpoint
# --------------------------------------------------------------------------- #


def test_wal_checkpoint_truncate(writer: sqlite3.Connection) -> None:
    _start_run(writer)
    busy, log_frames, checkpointed = storage.wal_checkpoint_truncate(writer)
    assert busy == 0
    # Successful TRUNCATE in WAL mode reports (0, 0) — the log is zeroed before
    # SQLite reports its size. Non-WAL mode returns the (-1, -1) sentinel, so
    # this is NOT vacuous: it proves the db really is in WAL mode. (Verified
    # empirically: WAL → (0, 0, 0); non-WAL → (0, -1, -1).)
    assert (log_frames, checkpointed) == (0, 0)


# --------------------------------------------------------------------------- #
# One-source-of-truth: first cross-module consumer of constants
# --------------------------------------------------------------------------- #


def test_storage_universe_is_constants_universe() -> None:
    """`is`-identity regression: storage must reference THE constants object,
    so any future re-duplication of the universe fails CI."""
    assert storage.UNIVERSE is constants.UNIVERSE


# --------------------------------------------------------------------------- #
# Migration framework (Track 2 Step 16): v1 -> v2, fresh-at-v2, drift guard
# --------------------------------------------------------------------------- #


def _build_v1_db(db_path: Path) -> sqlite3.Connection:
    """Open a raw connection at exactly the pre-Track-2 v1 schema — the state an
    OLD ``open_writer`` left on disk: v1 tables, ``user_version = 1``, NO
    configs/view_scenarios, NO ``runs.config_id``."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    migrations.MIGRATIONS[0][1](conn)  # the v0 -> v1 delta only
    conn.execute("PRAGMA user_version = 1")
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _table_layout(conn: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
    # (name, type, notnull, pk) — the normalized column facts, order-stable.
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [(row[1], row[2], row[3], row[5]) for row in rows]


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _foreign_keys(conn: sqlite3.Connection, table: str) -> set[tuple[object, ...]]:
    # (referenced_table, from_column, to_column) per foreign_key_list row.
    return {
        (row[2], row[3], row[4]) for row in conn.execute(f'PRAGMA foreign_key_list("{table}")')
    }


def test_target_version() -> None:
    assert storage.SCHEMA_VERSION == 3
    assert migrations.TARGET_VERSION == 3


def test_all_tables_match_built_schema(writer: sqlite3.Connection) -> None:
    """The hand-maintained ALL_TABLES name registry must equal what the
    migrations actually build — a drift guard between names and DDL."""
    assert set(storage.ALL_TABLES) == _tables(writer)
    assert storage.TABLES == frozenset(storage.ALL_TABLES)


def test_fresh_db_opens_at_current_version(writer: sqlite3.Connection) -> None:
    assert writer.execute("PRAGMA user_version").fetchone()[0] == 3
    assert writer.execute("SELECT COUNT(*) FROM configs WHERE is_central = 1").fetchone()[0] == 1
    # updated_at_utc backfilled to created_at_utc on a fresh migrate (v2->v3)
    created, updated = writer.execute(
        "SELECT created_at_utc, updated_at_utc FROM configs WHERE is_central = 1"
    ).fetchone()
    assert updated == created
    kinds = {str(row[0]) for row in writer.execute("SELECT kind FROM view_scenarios")}
    assert kinds == {"forecast"}
    name, feature_set, forecaster, optimizer = writer.execute(
        "SELECT name, feature_set, forecaster, optimizer FROM configs WHERE is_central = 1"
    ).fetchone()
    assert (name, feature_set, forecaster, optimizer) == (
        migrations.CENTRAL_CONFIG_NAME,
        "basic",
        "ewma",
        "mvu",
    )


def test_migrate_v1_to_v2_preserves_rows_and_backfills(db_path: Path) -> None:
    conn = _build_v1_db(db_path)
    try:
        assert "config_id" not in _columns(conn, "runs")  # v1 has no config_id
        assert "configs" not in _tables(conn)
        conn.execute(
            'INSERT INTO runs (started_at_utc, status, "trigger") VALUES (?, ?, ?)',
            ("2026-07-07T00:00:00Z", "ok", "manual"),
        )
        conn.execute(
            'INSERT INTO runs (started_at_utc, status, "trigger") VALUES (?, ?, ?)',
            ("2026-07-07T00:05:00Z", "ok", "schedule"),
        )
        conn.execute("INSERT INTO run_stages (run_id, stage, status) VALUES (1, 'freshness', 'ok')")
        before = conn.execute(
            "SELECT run_id, started_at_utc, status FROM runs ORDER BY run_id"
        ).fetchall()

        storage.ensure_schema(conn)  # v1 -> v2

        assert conn.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION
        after = conn.execute(
            "SELECT run_id, started_at_utc, status FROM runs ORDER BY run_id"
        ).fetchall()
        assert after == before  # zero row loss
        assert conn.execute("SELECT COUNT(*) FROM run_stages").fetchone()[0] == 1
        central = conn.execute(
            "SELECT config_id FROM configs WHERE is_central = 1"
        ).fetchall()
        assert len(central) == 1
        central_id = central[0][0]
        # every pre-existing run backfilled to the central config
        backfilled = {row[0] for row in conn.execute("SELECT config_id FROM runs")}
        assert backfilled == {central_id}
    finally:
        conn.close()


# The v1 schema is FROZEN history: real on-disk dbs at user_version 1 carry
# exactly this shape, so _migrate_v0_to_v1 must reproduce it forever. This
# fingerprint is a committed golden captured INDEPENDENTLY of migrations.py's
# source — editing the frozen _V1_TABLE_DDL/_V1_INDEX_DDL changes the computed
# hash and trips test_frozen_v1_schema_is_golden, forcing a deliberate
# acknowledgement that historical schema (which live migrations must handle) is
# being changed. This is the real measurement-validity anchor: it CAN fail on
# the drift it names (unlike test_fresh_and_migrated_schema_equivalent, whose
# two paths run the same functions and so agree by construction).
_V1_SCHEMA_SHA256 = "eeb0f46dc9aa525db448d459cd955139e65192943dc9cecdd260f4de6ee8a303"


def _v1_schema_fingerprint(conn: sqlite3.Connection) -> str:
    tables = sorted(_tables(conn))
    parts = [
        (table, tuple(_table_layout(conn, table)), tuple(sorted(_foreign_keys(conn, table))))
        for table in tables
    ]
    canon = repr((tuple(parts), tuple(sorted(_indexes(conn)))))
    return hashlib.sha256(canon.encode()).hexdigest()


def test_frozen_v1_schema_is_golden(db_path: Path) -> None:
    conn = _build_v1_db(db_path)
    try:
        assert _v1_schema_fingerprint(conn) == _V1_SCHEMA_SHA256, (
            "frozen v1 schema changed — real on-disk v1 dbs carry the OLD shape; "
            "if intentional, migrations must handle the difference and this golden "
            "must be updated deliberately"
        )
    finally:
        conn.close()


def test_fresh_and_migrated_schema_equivalent(tmp_path: Path) -> None:
    """Path-independence consistency check: a db built fresh and a db migrated
    from v1 land on identical schemas (tables, columns, FKs, indexes, version) at
    the current TARGET_VERSION. Both paths run the same migration functions, so
    this AGREES by construction — the teeth against frozen-v1 drift live in
    test_frozen_v1_schema_is_golden; this guards apply()'s path-independence and
    that config_id's FK really lands via ALTER."""
    fresh = storage.open_writer(tmp_path / "fresh" / "abe.db")
    migrated = _build_v1_db(tmp_path / "migrated" / "abe.db")
    try:
        storage.ensure_schema(migrated)  # v1 -> v2
        assert _tables(fresh) == _tables(migrated)
        assert _indexes(fresh) == _indexes(migrated)
        assert (
            fresh.execute("PRAGMA user_version").fetchone()[0]
            == migrated.execute("PRAGMA user_version").fetchone()[0]
        )
        for table in sorted(_tables(fresh)):
            assert _table_layout(fresh, table) == _table_layout(migrated, table), table
            assert _foreign_keys(fresh, table) == _foreign_keys(migrated, table), table
        # config_id's FK really landed (both inline-free paths ALTER it in)
        assert ("configs", "config_id", "config_id") in _foreign_keys(fresh, "runs")
    finally:
        fresh.close()
        migrated.close()


def test_migrate_idempotent_on_v2(writer: sqlite3.Connection) -> None:
    storage.ensure_schema(writer)  # already v2 → no-op
    storage.ensure_schema(writer)
    assert writer.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION
    assert writer.execute("SELECT COUNT(*) FROM configs WHERE is_central = 1").fetchone()[0] == 1
    assert writer.execute("SELECT COUNT(*) FROM view_scenarios").fetchone()[0] == 1


def test_single_central_invariant(writer: sqlite3.Connection) -> None:
    """The partial unique index enforces at most one is_central = 1 row."""
    with pytest.raises(sqlite3.IntegrityError):
        writer.execute(
            "INSERT INTO configs (name, feature_set, forecaster, view_scenario_id, "
            "optimizer, is_central, created_at_utc) "
            "VALUES ('rival', 'basic', 'ewma', 1, 'mvu', 1, '2026-07-08T00:00:00Z')"
        )


def test_config_id_fk_rejects_orphan_run(writer: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_row(
            writer,
            "runs",
            {
                "started_at_utc": "2026-07-08T00:00:00Z",
                "status": "ok",
                "trigger": "manual",
                "config_id": 9999,
            },
        )


def test_run_tagged_with_central_config_id(writer: sqlite3.Connection) -> None:
    central_id = writer.execute("SELECT config_id FROM configs WHERE is_central = 1").fetchone()[0]
    run_id = storage.insert_row(
        writer,
        "runs",
        {
            "started_at_utc": "2026-07-08T00:00:00Z",
            "status": "ok",
            "trigger": "manual",
            "config_id": central_id,
        },
    )
    assert (
        writer.execute("SELECT config_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
        == central_id
    )


def test_configs_insert_through_coercion_boundary(writer: sqlite3.Connection) -> None:
    """The new tables are accepted by the insert boundary (ALL_TABLES allow-list)."""
    vs_id = writer.execute("SELECT view_scenario_id FROM view_scenarios LIMIT 1").fetchone()[0]
    config_id = storage.insert_row(
        writer,
        "configs",
        {
            "name": "alt",
            "feature_set": "basic",
            "forecaster": "jepa",
            "view_scenario_id": vs_id,
            "optimizer": "mvu",
            "params_json": None,
            "is_central": 0,
            "created_at_utc": "2026-07-08T00:00:00Z",
        },
    )
    assert config_id is not None
    row = writer.execute(
        "SELECT name, forecaster, is_central FROM configs WHERE config_id = ?", (config_id,)
    ).fetchone()
    assert row == ("alt", "jepa", 0)
