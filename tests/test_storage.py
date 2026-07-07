"""Storage-layer tests: schema, PRAGMAs, coercion boundary, FK/CHECK, ro reads.

All tests run against a real tmp_path SQLite db — no mocks. Integration through
the production entry point is N/A this step (pipeline arrives Step 8); storage
is the leaf, so direct tests are correct here.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from abe import constants, storage

EXPECTED_TABLES = {
    "runs",
    "run_stages",
    "prices",
    "macro",
    "features",
    "forecasts",
    "bl_posteriors",
    "target_weights",
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
