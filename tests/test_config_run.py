"""On-demand config-run path tests (Track 2 Step 21).

Covers the API trigger (`POST /api/configs/{id}/run`) end-to-end through the real
scheduler + executor, the per-config cache gate, and the single-writer invariant
(no phantom 'running' rows after interleaving central + on-demand runs).
"""

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from seeding import OFFLINE_SCHEDULER, seed_prices

from abe import config as config_module
from abe import storage
from abe.api import create_app
from abe.constants import UNIVERSE
from abe.ingest.macro import MACRO_DISABLED_NO_KEY, MacroStatus
from abe.pipeline import cached_config_run, run_pipeline

DISABLED_MACRO = MacroStatus(enabled=False, code=MACRO_DISABLED_NO_KEY, message="offline")
STARTUP_RUN_ID = 1


def _query(db_path: Path, sql: str, *params: object) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _wait_terminal(db_path: Path, run_id: int, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _query(db_path, "SELECT status FROM runs WHERE run_id = ?", run_id)
        if rows and str(rows[0][0]) not in ("running", "queued"):
            return str(rows[0][0])
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} not terminal within {timeout}s")


def _wait_idle(app: FastAPI, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while app.state.scheduler.run_in_progress:
        if time.monotonic() > deadline:
            raise AssertionError("scheduler did not go idle")
        time.sleep(0.01)


def _make_alt_config(db_path: Path) -> int:
    """Create a NON-central config (basic/ewma/mvu, central's view scenario)."""
    conn = storage.open_writer(db_path)
    try:
        central = config_module.get_central_config(conn)
        alt = config_module.create_config(
            conn,
            name="alt",
            feature_set="basic",
            forecaster="ewma",
            view_scenario_id=central.view_scenario_id,
            optimizer="mvu",
        )
        return alt.config_id
    finally:
        conn.close()


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, int]:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "data" / "abe.db"
    seed_prices(db_path)
    alt_id = _make_alt_config(db_path)
    return db_path, alt_id


@pytest.fixture()
def client(setup: tuple[Path, int]) -> Iterator[tuple[TestClient, Path, int]]:
    db_path, alt_id = setup
    app = create_app(db_path, scheduler_config=OFFLINE_SCHEDULER)
    with TestClient(app) as test_client:
        assert _wait_terminal(db_path, STARTUP_RUN_ID) == "ok"  # central startup run
        _wait_idle(app)
        yield test_client, db_path, alt_id


# --------------------------------------------------------------------------- #
# API: on-demand config run is tagged + cached
# --------------------------------------------------------------------------- #


def test_config_run_tagged_then_cached(client: tuple[TestClient, Path, int]) -> None:
    test_client, db_path, alt_id = client

    first = test_client.post(f"/api/configs/{alt_id}/run")
    assert first.status_code == 200
    body = first.json()
    assert body["config_id"] == alt_id
    assert body["cached"] is False
    run1 = body["run_id"]
    assert run1 > STARTUP_RUN_ID
    # the run is tagged by config_id and landed ok
    rows = _query(db_path, "SELECT config_id, status FROM runs WHERE run_id = ?", run1)
    assert rows[0] == (alt_id, "ok")

    # second request on UNCHANGED data → served from cache, same run
    second = test_client.post(f"/api/configs/{alt_id}/run")
    body2 = second.json()
    assert body2["cached"] is True
    assert body2["run_id"] == run1


def test_config_run_unknown_config_404(client: tuple[TestClient, Path, int]) -> None:
    test_client, _db_path, _alt_id = client
    resp = test_client.post("/api/configs/9999/run")
    assert resp.status_code == 404
    assert "no config" in resp.json()["detail"]


def test_config_run_central_id_rejected_409(client: tuple[TestClient, Path, int]) -> None:
    """The central config belongs to the always-on loop, not the on-demand path."""
    test_client, db_path, _alt_id = client
    central_id = _query(db_path, "SELECT config_id FROM configs WHERE is_central = 1")[0][0]
    resp = test_client.post(f"/api/configs/{central_id}/run")
    assert resp.status_code == 409
    assert "central" in resp.json()["detail"]


def test_alt_run_does_not_hijack_central_latest(client: tuple[TestClient, Path, int]) -> None:
    """The BLOCKER regression: an on-demand alt run lands a higher run_id but must
    NOT become the central answer that /api/runs/latest serves, and must not
    freeze the central loop's recompute (per-config freshness)."""
    test_client, db_path, alt_id = client
    central_id = _query(db_path, "SELECT config_id FROM configs WHERE is_central = 1")[0][0]
    assert test_client.get("/api/runs/latest").json()["run"]["run_id"] == STARTUP_RUN_ID

    alt = test_client.post(f"/api/configs/{alt_id}/run").json()
    assert alt["run_id"] > STARTUP_RUN_ID and alt["config_id"] == alt_id

    # /api/runs/latest STILL serves the CENTRAL run, not the newer alt run
    latest_id = test_client.get("/api/runs/latest").json()["run"]["run_id"]
    assert latest_id == STARTUP_RUN_ID
    served = _query(db_path, "SELECT config_id FROM runs WHERE run_id = ?", latest_id)
    assert served[0][0] == central_id

    # the central loop is NOT frozen by the alt run's watermarks: a forced central
    # recompute still produces a new ok central run that becomes latest
    forced = test_client.post("/api/runs/trigger", json={"force": True}).json()
    assert _wait_terminal(db_path, forced["run_id"]) == "ok"
    _wait_idle(test_client.app)
    final_id = test_client.get("/api/runs/latest").json()["run"]["run_id"]
    assert final_id == forced["run_id"]
    served_final = _query(db_path, "SELECT config_id FROM runs WHERE run_id = ?", final_id)
    assert served_final[0][0] == central_id


def test_single_writer_no_phantom_running_after_interleave(
    client: tuple[TestClient, Path, int],
) -> None:
    """Interleave a central trigger + an on-demand config run; the shared single
    executor serializes them, so no 'running' row is left behind and both runs
    are tagged with their own config_id."""
    test_client, db_path, alt_id = client
    central_id = _query(db_path, "SELECT config_id FROM configs WHERE is_central = 1")[0][0]

    forced = test_client.post("/api/runs/trigger", json={"force": True})
    assert forced.status_code == 202
    central_run = forced.json()["run_id"]
    assert _wait_terminal(db_path, central_run) == "ok"
    _wait_idle(test_client.app)

    cfg = test_client.post(f"/api/configs/{alt_id}/run", json={"force": True})
    config_run = cfg.json()["run_id"]

    running = _query(db_path, "SELECT COUNT(*) FROM runs WHERE status = 'running'")[0][0]
    assert running == 0  # single-flight: no phantom running rows
    tags = dict(_query(db_path, "SELECT run_id, config_id FROM runs WHERE run_id IN (?, ?)",
                       central_run, config_run))
    assert tags[central_run] == central_id
    assert tags[config_run] == alt_id


# --------------------------------------------------------------------------- #
# Cache logic: unchanged data caches, new data recomputes (direct, no app)
# --------------------------------------------------------------------------- #


def test_cached_config_run_recomputes_on_new_data(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "abe.db"
    last_date = seed_prices(db_path)
    conn = storage.open_writer(db_path)
    try:
        central = config_module.get_central_config(conn)
        alt = config_module.create_config(
            conn,
            name="alt",
            feature_set="basic",
            forecaster="ewma",
            view_scenario_id=central.view_scenario_id,
            optimizer="mvu",
        )
        assert cached_config_run(conn, alt.config_id) is None  # no run yet

        run_id = run_pipeline(
            conn, config=alt, trigger="manual", force=True, macro_status=DISABLED_MACRO
        )
        # unchanged data → the cache gate returns this run
        assert cached_config_run(conn, alt.config_id) == run_id

        # append a new trading day (moves both watermarks) → recompute needed
        next_day = "2026-12-31"
        assert next_day > last_date
        for asset in UNIVERSE:
            storage.upsert_row(
                conn,
                "prices",
                {
                    "asset": asset,
                    "date": next_day,
                    "adj_close": 101.0,
                    "source": "cache",
                    "fetched_at_utc": "2026-12-31T00:00:00Z",
                },
            )
        assert cached_config_run(conn, alt.config_id) is None
    finally:
        conn.close()


def test_cache_invalidates_after_recipe_edit(tmp_path: Path) -> None:
    """A config edited after its last cached run is stale (Step 25 recipe-freshness)."""
    db_path = tmp_path / "data" / "abe.db"
    seed_prices(db_path)
    conn = storage.open_writer(db_path)
    try:
        central = config_module.get_central_config(conn)
        alt = config_module.create_config(
            conn,
            name="alt",
            feature_set="basic",
            forecaster="ewma",
            view_scenario_id=central.view_scenario_id,
            optimizer="mvu",
        )
        run_id = run_pipeline(
            conn, config=alt, trigger="manual", force=True, macro_status=DISABLED_MACRO
        )
        assert cached_config_run(conn, alt.config_id) == run_id  # cached on unchanged data
        # a recipe edit AFTER the run (deterministic future stamp) invalidates it
        conn.execute(
            "UPDATE configs SET updated_at_utc = '2099-01-01T00:00:00Z' WHERE config_id = ?",
            (alt.config_id,),
        )
        assert cached_config_run(conn, alt.config_id) is None
    finally:
        conn.close()
