"""Pipeline orchestrator tests: freshness predicate, ledger, rollback, w_prev.

Real tmp SQLite + the real EWMABaseline against synthetic seeded prices — no
mocks of our own code. The headline integration through the production FastAPI
route lives in tests/test_api.py; here run_pipeline is exercised directly for
the transaction/ledger contracts that need row-level assertions.
"""

import json
import math
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from seeding import append_price_day, revise_prices, seed_prices

from abe import storage
from abe.constants import DELTA, HORIZON_BARS, UNIVERSE, W_MAX
from abe.ingest.macro import MACRO_DISABLED_NO_KEY, MACRO_OK, MacroStatus
from abe.ingest.sources import PRICE_PROVIDER_LABEL
from abe.optimize.mvu import GAMMA_TC
from abe.pipeline import (
    STAGES,
    last_ok_data_fetched_at,
    last_ok_data_max_date,
    load_last_weights,
    run_pipeline,
    should_skip_run,
    stored_data_fetched_at,
    stored_data_max_date,
)

OK_STATUS = MacroStatus(enabled=True, code=MACRO_OK, message="test key accepted")

_NETWORK_LIBS = ("yfinance", "fredapi")


class _ForbidNetworkLibImport:
    """Meta-path hook: fail the test if the recompute path imports a network lib.

    The fetch-split (plan section 8) says the 5-min recompute path serves from
    SQLite only — importing yfinance or fredapi during run_pipeline would mean
    a network dependency leaked into it. Same idiom as
    test_ingest_prices._ForbidYFinanceImport.
    """

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        root = fullname.partition(".")[0]
        if root in _NETWORK_LIBS:
            raise AssertionError(
                f"recompute pipeline imported {fullname!r} — fetch-split violation "
                "(plan section 8: fetch is a separate daily job)"
            )
        return None


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "abe.db"


@pytest.fixture()
def seeded_db(db_path: Path) -> Path:
    seed_prices(db_path)
    return db_path


@pytest.fixture()
def writer(seeded_db: Path) -> Iterator[sqlite3.Connection]:
    conn = storage.open_writer(seeded_db)
    yield conn
    conn.close()


def _run_row(conn: sqlite3.Connection, run_id: int) -> tuple[str, str | None, str | None]:
    row = conn.execute(
        "SELECT status, finished_at_utc, error_text FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return (str(row[0]), row[1], row[2])


def _stage_rows(conn: sqlite3.Connection, run_id: int) -> list[tuple[str, str, object]]:
    return [
        (str(row[0]), str(row[1]), json.loads(row[2]) if row[2] else None)
        for row in conn.execute(
            "SELECT stage, status, detail_json FROM run_stages WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
    ]


def _count(conn: sqlite3.Connection, table: str, run_id: int) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()
    return int(row[0])


# --------------------------------------------------------------------------- #
# Freshness predicate — unit tests
# --------------------------------------------------------------------------- #


_D0, _D1 = "2026-07-07", "2026-07-08"  # data dates
_F0, _F1 = "2026-07-07T21:00:00Z", "2026-07-08T21:00:00Z"  # fetched_at watermarks


@pytest.mark.parametrize(
    ("force", "data_max", "fetched", "last_ok", "used_date", "used_fetched", "expected"),
    [
        # force always recomputes, even on identical watermarks
        (True, _D0, _F0, 3, _D0, _F0, False),
        # no ok run yet -> always try
        (False, _D0, _F0, None, None, None, False),
        # empty prices table -> proceed (fail loud downstream, never skip forever)
        (False, None, None, 3, _D0, _F0, False),
        # BOTH watermarks unchanged since the last ok run -> skip (the gate's point)
        (False, _D0, _F0, 3, _D0, _F0, True),
        # a new date arrived -> recompute
        (False, _D1, _F1, 3, _D0, _F0, False),
        # SAME date but history was rewritten (backward-adjustment rebase:
        # fetched_at watermark moved) -> recompute, do NOT skip
        (False, _D0, _F1, 3, _D0, _F0, False),
        # last ok run's freshness detail missing/unparseable -> fail open
        (False, _D0, _F0, 3, None, None, False),
        # older ok run recorded a date but no fetched watermark -> fail open
        (False, _D0, _F0, 3, _D0, None, False),
        # defensive: rows exist but fetched watermark unreadable -> fail open
        (False, _D0, None, 3, _D0, _F0, False),
    ],
)
def test_should_skip_run_predicate(
    force: bool,
    data_max: str | None,
    fetched: str | None,
    last_ok: int | None,
    used_date: str | None,
    used_fetched: str | None,
    expected: bool,
) -> None:
    assert (
        should_skip_run(
            force=force,
            data_max_date=data_max,
            data_fetched_at=fetched,
            last_ok_run_id=last_ok,
            last_ok_data_max_date=used_date,
            last_ok_data_fetched_at=used_fetched,
        )
        is expected
    )


def test_stored_data_max_date(tmp_path: Path) -> None:
    db = tmp_path / "abe.db"
    last_seeded = seed_prices(db)
    conn = storage.open_writer(db)
    try:
        assert stored_data_max_date(conn) == last_seeded
    finally:
        conn.close()


def test_stored_watermarks_empty_table(tmp_path: Path) -> None:
    conn = storage.open_writer(tmp_path / "abe.db")
    try:
        assert stored_data_max_date(conn) is None
        assert stored_data_fetched_at(conn) is None
    finally:
        conn.close()


def test_last_ok_watermarks_read_freshness_detail(writer: sqlite3.Connection) -> None:
    run_id = run_pipeline(writer, trigger="startup", macro_status=OK_STATUS)
    assert _run_row(writer, run_id)[0] == "ok"
    assert last_ok_data_max_date(writer, run_id) == stored_data_max_date(writer)
    assert last_ok_data_fetched_at(writer, run_id) == stored_data_fetched_at(writer)
    assert last_ok_data_fetched_at(writer, run_id) is not None
    assert last_ok_data_max_date(writer, None) is None
    assert last_ok_data_fetched_at(writer, None) is None


# --------------------------------------------------------------------------- #
# Full run: ledger + derived rows
# --------------------------------------------------------------------------- #


def test_full_run_writes_all_tables(writer: sqlite3.Connection) -> None:
    # The whole run executes under a meta-path guard forbidding yfinance AND
    # fredapi imports: the recompute path must be network-library-free.
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.partition(".")[0] in _NETWORK_LIBS
    }
    guard = _ForbidNetworkLibImport()
    sys.meta_path.insert(0, guard)
    try:
        run_id = run_pipeline(writer, trigger="startup", macro_status=OK_STATUS)
    finally:
        sys.meta_path.remove(guard)
        sys.modules.update(saved)

    status, finished_at, error_text = _run_row(writer, run_id)
    assert status == "ok"
    assert finished_at is not None
    assert error_text is None

    stages = _stage_rows(writer, run_id)
    assert [stage for stage, _, _ in stages] == list(STAGES)
    assert all(stage_status == "ok" for _, stage_status, _ in stages)

    # The blend card carries mu AND sigma per asset (Step 10 shows mu +/- sigma).
    blend_detail = stages[4][2]
    assert isinstance(blend_detail, dict)
    assert set(blend_detail["posterior_mu"]) == set(UNIVERSE)
    assert set(blend_detail["posterior_sigma"]) == set(UNIVERSE)
    assert all(value > 0.0 for value in blend_detail["posterior_sigma"].values())

    # features: 2 rows per asset (latest bar only)
    assert _count(writer, "features", run_id) == 2 * len(UNIVERSE)

    forecasts = writer.execute(
        "SELECT asset, horizon_days, mu, sigma, model_version FROM forecasts WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    assert {row[0] for row in forecasts} == set(UNIVERSE)
    assert all(int(row[1]) == HORIZON_BARS for row in forecasts)
    assert all(float(row[3]) > 0.0 for row in forecasts)
    assert all(str(row[4]) == "ewma" for row in forecasts)

    posteriors = writer.execute(
        "SELECT asset, posterior_sigma, detail_json FROM bl_posteriors WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    assert {row[0] for row in posteriors} == set(UNIVERSE)
    assert all(float(row[1]) > 0.0 for row in posteriors)
    for _, _, detail_json in posteriors:
        detail = json.loads(detail_json)
        assert set(detail) == {"pi", "Q", "confidence", "omega_diag", "tilt"}

    weights = writer.execute(
        "SELECT asset, weight, prev_weight, turnover, relaxed_turnover "
        "FROM target_weights WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    assert {row[0] for row in weights} == set(UNIVERSE)
    assert math.isclose(sum(float(row[1]) for row in weights), 1.0, abs_tol=1e-9)
    assert all(row[2] is None for row in weights)  # cold start
    assert all(float(row[3]) == 0.0 for row in weights)
    assert all(int(row[4]) == 0 for row in weights)


def test_transparency_fields_land_in_stage_details(writer: sqlite3.Connection) -> None:
    """Track 1 producer -> consumer round trip through the PRODUCTION pipeline.

    Run the real pipeline, read the persisted ``run_stages`` rows back (the
    exact path the API serves the UI cards from), and assert the added
    transparency fields are present with the right values — the bug class
    (a card-payload key that never actually lands) lives in this relationship,
    not in either endpoint alone.
    """
    run_id = run_pipeline(writer, trigger="manual", macro_status=OK_STATUS)
    assert _run_row(writer, run_id)[0] == "ok"
    by_stage = {stage: detail for stage, _, detail in _stage_rows(writer, run_id)}

    # ingest: price_provider added alongside the existing `source`.
    ingest = by_stage["ingest"]
    assert isinstance(ingest, dict)
    assert ingest["source"] == "cache"  # existing field untouched
    assert ingest["price_provider"] == PRICE_PROVIDER_LABEL

    # features: per-feature windows added alongside the existing feature list.
    features = by_stage["features"]
    assert isinstance(features, dict)
    assert features["features"] == ["log_return", "realized_vol"]  # existing key
    assert features["windows"] == {
        "log_return": "1 day",
        "realized_vol": f"{HORIZON_BARS} days, annualized",
    }

    # blend: prior (pi, all assets) and view (Q, viewed assets) now surfaced,
    # per asset, in addition to the existing posterior keys.
    blend = by_stage["blend"]
    assert isinstance(blend, dict)
    assert set(blend["prior"]) == set(UNIVERSE)
    assert set(blend["view"]) == set(UNIVERSE)  # EWMA forecasts every asset
    assert all(isinstance(v, float) for v in blend["prior"].values())
    assert set(blend["posterior_mu"]) == set(UNIVERSE)  # existing key intact
    # covariance window: the common (inner-join) history the covariance is fit on.
    cov_window = blend["covariance_window"]
    assert set(cov_window) == {"start", "end", "bars"}
    assert isinstance(cov_window["bars"], int) and cov_window["bars"] > 0
    assert cov_window["start"] <= cov_window["end"]

    # optimize: objective block sourced from the REAL constants (not hardcoded).
    optimize = by_stage["optimize"]
    assert isinstance(optimize, dict)
    objective = optimize["objective"]
    assert objective["delta"] == DELTA
    assert objective["gamma_tc"] == GAMMA_TC
    assert objective["w_max"] == W_MAX
    assert objective["constraints"] == ["sum(w) = 1", "0 <= w <= w_max"]
    assert "maximize" in objective["form"]
    assert set(optimize["weights"]) == set(UNIVERSE)  # existing key intact


def test_invalid_trigger_raises(writer: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="trigger"):
        run_pipeline(writer, trigger="cron", macro_status=OK_STATUS)


def test_on_run_started_fires_at_run_start_through_production_pipeline(
    writer: sqlite3.Connection, seeded_db: Path
) -> None:
    """run_pipeline ITSELF must invoke on_run_started at the run's START —
    the phase-0 'running' row is already COMMITTED (asserted through a SECOND
    connection, so in-transaction visibility can't fake it) and NO stage has
    executed yet. This pins the Step 11 trigger contract (202 at run START)
    at the production producer: the scheduler's defensive
    resolve-at-completion fallback would otherwise mask a dropped call and
    silently regress the endpoint to 202-at-completion."""
    observed: list[tuple[int, str, int]] = []

    def probe(run_id: int) -> None:
        other = sqlite3.connect(seeded_db)
        try:
            status = other.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            stage_count = other.execute(
                "SELECT COUNT(*) FROM run_stages WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        finally:
            other.close()
        observed.append((run_id, str(status), int(stage_count)))

    returned = run_pipeline(writer, trigger="manual", macro_status=OK_STATUS, on_run_started=probe)

    # Exactly one callback, carrying the returned id, observed while the run
    # was 'running' with ZERO stage rows (at START, not at completion).
    assert observed == [(returned, "running", 0)]
    # And the run then completed normally.
    assert _run_row(writer, returned)[0] == "ok"


# --------------------------------------------------------------------------- #
# Freshness gate through run_pipeline
# --------------------------------------------------------------------------- #


def test_second_run_skipped_force_bypasses_new_data_unblocks(
    writer: sqlite3.Connection, seeded_db: Path
) -> None:
    first = run_pipeline(writer, trigger="startup", macro_status=OK_STATUS)
    assert _run_row(writer, first)[0] == "ok"

    # Same data -> skipped: one 'skipped' freshness stage row, no data rows.
    second = run_pipeline(writer, trigger="schedule", macro_status=OK_STATUS)
    assert _run_row(writer, second)[0] == "skipped"
    stages = _stage_rows(writer, second)
    assert stages == [("freshness", "skipped", stages[0][2])]
    assert _count(writer, "target_weights", second) == 0
    assert storage.latest_ok_run_id(writer) == first

    # force=True bypasses the gate on identical data.
    third = run_pipeline(writer, trigger="manual", force=True, macro_status=OK_STATUS)
    assert _run_row(writer, third)[0] == "ok"

    # A new price date arriving unblocks an unforced run.
    append_price_day(seeded_db, "2026-04-13")
    fourth = run_pipeline(writer, trigger="schedule", macro_status=OK_STATUS)
    assert _run_row(writer, fourth)[0] == "ok"
    freshness_detail = _stage_rows(writer, fourth)[0][2]
    assert isinstance(freshness_detail, dict)
    assert freshness_detail["data_max_date"] == "2026-04-13"


def test_in_place_revision_unblocks_unforced_run(
    writer: sqlite3.Connection, seeded_db: Path
) -> None:
    """A backward-adjustment rebase (same MAX(date), rewritten history) must
    NOT be skipped — the live-verified review scenario: post-dividend rebase
    lands via the daily fetch, every 5-min recompute would otherwise skip for
    up to 3 days while the UI shows pre-rebase weights."""
    first = run_pipeline(writer, trigger="startup", macro_status=OK_STATUS)
    assert _run_row(writer, first)[0] == "ok"
    skipped = run_pipeline(writer, trigger="schedule", macro_status=OK_STATUS)
    assert _run_row(writer, skipped)[0] == "skipped"

    # Rewrite the whole adj_close history in place: dates unchanged, but every
    # re-upsert stamps a fresh fetched_at_utc (exactly what Step 3's full
    # refresh does through the same storage.upsert_row path).
    revise_prices(seeded_db, fetched_at="2026-07-08T21:05:00Z")

    revised = run_pipeline(writer, trigger="schedule", macro_status=OK_STATUS)
    assert _run_row(writer, revised)[0] == "ok"
    detail = _stage_rows(writer, revised)[0][2]
    assert isinstance(detail, dict)
    # Same trading date as the last ok run -- the OLD one-watermark gate would
    # have skipped here; the fetched_at watermark is what unblocked the run.
    assert detail["data_max_date"] == detail["last_ok_data_max_date"]
    assert detail["data_fetched_at"] == "2026-07-08T21:05:00Z"
    assert detail["data_fetched_at"] != detail["last_ok_data_fetched_at"]

    # And the gate re-arms: an unchanged db skips again afterwards.
    resettled = run_pipeline(writer, trigger="schedule", macro_status=OK_STATUS)
    assert _run_row(writer, resettled)[0] == "skipped"


# --------------------------------------------------------------------------- #
# Error path: rollback + surviving ledger
# --------------------------------------------------------------------------- #


def test_error_path_rolls_back_data_but_keeps_ledger(db_path: Path) -> None:
    # 25 bars -> 24 returns: enough for features (window 21) but far short of
    # the EWMA's MIN_HISTORY_BARS=52, so the FORECAST stage fails loud.
    seed_prices(db_path, days=25)
    conn = storage.open_writer(db_path)
    try:
        run_id = run_pipeline(conn, trigger="manual", macro_status=OK_STATUS)  # must NOT raise

        status, finished_at, error_text = _run_row(conn, run_id)
        assert status == "error"
        assert finished_at is not None
        assert error_text is not None and "forecast" in error_text
        assert "MIN_HISTORY_BARS" in error_text

        stages = _stage_rows(conn, run_id)
        assert [(stage, stage_status) for stage, stage_status, _ in stages] == [
            ("freshness", "ok"),
            ("ingest", "ok"),
            ("features", "ok"),
            ("forecast", "error"),
        ]  # later stages absent
        error_detail = stages[-1][2]
        assert isinstance(error_detail, dict)
        assert "MIN_HISTORY_BARS" in str(error_detail["error"])

        # Rollback verified: NO partial data rows from the failed run — the
        # features stage succeeded and inserted rows, but they were rolled back.
        for table in ("features", "forecasts", "bl_posteriors", "target_weights"):
            assert _count(conn, table, run_id) == 0

        assert storage.latest_ok_run_id(conn) is None
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# w_prev round trip
# --------------------------------------------------------------------------- #


def test_w_prev_round_trip(writer: sqlite3.Connection) -> None:
    first = run_pipeline(writer, trigger="manual", force=True, macro_status=OK_STATUS)
    second = run_pipeline(writer, trigger="manual", force=True, macro_status=OK_STATUS)
    assert _run_row(writer, first)[0] == "ok"
    assert _run_row(writer, second)[0] == "ok"

    first_weights = {
        str(row[0]): float(row[1])
        for row in writer.execute(
            "SELECT asset, weight FROM target_weights WHERE run_id = ?", (first,)
        ).fetchall()
    }
    second_rows = writer.execute(
        "SELECT asset, weight, prev_weight, turnover FROM target_weights WHERE run_id = ?",
        (second,),
    ).fetchall()
    assert len(second_rows) == len(UNIVERSE)
    for asset, weight, prev_weight, turnover in second_rows:
        # SQLite REAL round-trips float64 exactly: prev_weight IS run 1's weight.
        assert prev_weight == first_weights[str(asset)]
        assert float(turnover) == pytest.approx(abs(float(weight) - float(prev_weight)), abs=1e-15)

    # load_last_weights now reads run 2 (the central config's new latest ok run).
    from abe import config as config_module

    central_id = config_module.get_central_config(writer).config_id
    assert load_last_weights(writer, central_id) == {
        str(row[0]): float(row[1]) for row in second_rows
    }


# --------------------------------------------------------------------------- #
# Macro-disabled degraded mode is a card fact, not an error
# --------------------------------------------------------------------------- #


def test_macro_disabled_surfaces_on_ingest_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No FRED key anywhere: env cleared + cwd moved to an .env-less tmp dir,
    # so macro_status=None resolves through the REAL probe path (no network).
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "abe.db"
    seed_prices(db)
    conn = storage.open_writer(db)
    try:
        run_id = run_pipeline(conn, trigger="startup", macro_status=None)
        assert _run_row(conn, run_id)[0] == "ok"
        ingest_detail = _stage_rows(conn, run_id)[1][2]
        assert isinstance(ingest_detail, dict)
        macro = ingest_detail["macro"]
        assert isinstance(macro, dict)
        assert macro["enabled"] is False
        assert macro["code"] == MACRO_DISABLED_NO_KEY
    finally:
        conn.close()
