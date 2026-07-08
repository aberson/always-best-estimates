"""API tests — THE Step 8 integration requirement.

POST /api/runs/trigger is driven end-to-end through the PRODUCTION FastAPI
route (TestClient with lifespan context: real writer connection + real macro
probe) against a tmp SQLite db seeded with >= 70 rows/asset of synthetic
prices, with the real EWMABaseline — no mocks of our own code. Row-level
assertions read the same db file directly.

Every test clears FRED_API_KEY and chdirs to an .env-less tmp dir, so the
lifespan macro probe resolves to MACRO_DISABLED_NO_KEY without any network.
"""

import json
import math
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from seeding import seed_prices

from abe import api, storage
from abe.api import create_app
from abe.constants import UNIVERSE
from abe.ingest.macro import MACRO_DISABLED_NO_KEY
from abe.pipeline import STAGES


@pytest.fixture()
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in cwd -> the startup probe stays offline
    return tmp_path / "data" / "abe.db"


@pytest.fixture()
def client(db_path: Path) -> Iterator[TestClient]:
    seed_prices(db_path)
    with TestClient(create_app(db_path)) as test_client:
        yield test_client


def _query(db_path: Path, sql: str, *params: object) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# The end-to-end integration requirement
# --------------------------------------------------------------------------- #


def test_trigger_end_to_end_through_production_route(client: TestClient, db_path: Path) -> None:
    response = client.post("/api/runs/trigger", json={"force": False})
    assert response.status_code == 202
    body = response.json()
    run_id = body["run_id"]
    assert isinstance(run_id, int)
    assert body["already_running"] is False

    # Rows landed in every table, straight from the db file.
    (run_row,) = _query(db_path, "SELECT status, error_text FROM runs WHERE run_id = ?", run_id)
    assert run_row == ("ok", None)

    stage_rows = _query(
        db_path,
        "SELECT stage, status FROM run_stages WHERE run_id = ? ORDER BY rowid",
        run_id,
    )
    assert [(str(stage), str(status)) for stage, status in stage_rows] == [
        (stage, "ok") for stage in STAGES
    ]

    assert len(_query(db_path, "SELECT 1 FROM forecasts WHERE run_id = ?", run_id)) == len(UNIVERSE)
    assert len(_query(db_path, "SELECT 1 FROM bl_posteriors WHERE run_id = ?", run_id)) == len(
        UNIVERSE
    )
    weight_rows = _query(
        db_path, "SELECT asset, weight FROM target_weights WHERE run_id = ?", run_id
    )
    assert {str(asset) for asset, _ in weight_rows} == set(UNIVERSE)
    assert math.isclose(sum(float(weight) for _, weight in weight_rows), 1.0, abs_tol=1e-9)

    # /api/runs/latest returns the run + its stages.
    latest = client.get("/api/runs/latest")
    assert latest.status_code == 200
    payload = latest.json()
    assert payload["run"]["run_id"] == run_id
    assert payload["run"]["status"] == "ok"
    assert [stage["stage"] for stage in payload["stages"]] == list(STAGES)
    assert all(stage["status"] == "ok" for stage in payload["stages"])
    # detail_json is exposed as parsed objects for the UI cards.
    optimize_detail = payload["stages"][-1]["detail"]
    assert optimize_detail["relaxed_turnover"] is False
    assert set(optimize_detail["weights"]) == set(UNIVERSE)
    assert optimize_detail["solver_status"] in ("optimal", "optimal_inaccurate")

    # /api/history returns the run with its weights.
    history = client.get("/api/history", params={"limit": 5})
    assert history.status_code == 200
    runs = history.json()["runs"]
    assert runs[0]["run_id"] == run_id
    assert {row["asset"] for row in runs[0]["target_weights"]} == set(UNIVERSE)

    # /api/runs/{id}/stages returns the same six cards.
    stages = client.get(f"/api/runs/{run_id}/stages")
    assert stages.status_code == 200
    assert [stage["stage"] for stage in stages.json()] == list(STAGES)


def test_freshness_gate_and_force_through_route(client: TestClient) -> None:
    first = client.post("/api/runs/trigger", json={}).json()["run_id"]

    # Second trigger without force: a skipped run is RECORDED (not silent).
    second_response = client.post("/api/runs/trigger")  # no body at all -> force=False
    assert second_response.status_code == 202
    second = second_response.json()["run_id"]
    assert second != first
    second_stages = client.get(f"/api/runs/{second}/stages").json()
    assert [(stage["stage"], stage["status"]) for stage in second_stages] == [
        ("freshness", "skipped")
    ]
    # Latest still points at the first (ok) run — skipped is not 'ok'.
    assert client.get("/api/runs/latest").json()["run"]["run_id"] == first

    # force=true bypasses the gate on identical data.
    third = client.post("/api/runs/trigger", json={"force": True}).json()["run_id"]
    assert client.get("/api/runs/latest").json()["run"]["run_id"] == third

    # History shows all three runs, newest first.
    history = client.get("/api/history").json()["runs"]
    assert [run["run_id"] for run in history] == [third, second, first]
    assert [run["status"] for run in history] == ["ok", "skipped", "ok"]
    assert history[1]["target_weights"] == []


def test_macro_disabled_mode_surfaces_on_ingest_card(client: TestClient) -> None:
    run_id = client.post("/api/runs/trigger", json={}).json()["run_id"]
    stages = client.get(f"/api/runs/{run_id}/stages").json()
    ingest = next(stage for stage in stages if stage["stage"] == "ingest")
    assert ingest["status"] == "ok"  # degraded macro is a card fact, NOT an error
    assert ingest["detail"]["macro"]["enabled"] is False
    assert ingest["detail"]["macro"]["code"] == MACRO_DISABLED_NO_KEY


def test_error_run_through_route_returns_202_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage errors must NOT propagate out of the route (the scheduler survives)."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "short.db"
    seed_prices(db, days=25)  # features ok, forecast fails (24 < MIN_HISTORY_BARS)
    with TestClient(create_app(db)) as client:
        response = client.post("/api/runs/trigger", json={})
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        stages = client.get(f"/api/runs/{run_id}/stages").json()
        assert [(stage["stage"], stage["status"]) for stage in stages] == [
            ("freshness", "ok"),
            ("ingest", "ok"),
            ("features", "ok"),
            ("forecast", "error"),
        ]
        assert "MIN_HISTORY_BARS" in stages[-1]["detail"]["error"]

        # No ok run yet -> latest is a 404 with a JSON body.
        latest = client.get("/api/runs/latest")
        assert latest.status_code == 404
        assert "detail" in latest.json()

        # Rollback verified through the file: no data rows for the failed run.
        for table in ("features", "forecasts", "bl_posteriors", "target_weights"):
            assert _query(db, f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", run_id)[0][0] == 0


# --------------------------------------------------------------------------- #
# Contract edges
# --------------------------------------------------------------------------- #


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_latest_404_before_any_run(client: TestClient) -> None:
    response = client.get("/api/runs/latest")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_unknown_run_stages_404(client: TestClient) -> None:
    response = client.get("/api/runs/9999/stages")
    assert response.status_code == 404


def test_history_empty_and_limit_validation(client: TestClient) -> None:
    assert client.get("/api/history").json() == {"runs": []}
    assert client.get("/api/history", params={"limit": 0}).status_code == 422
    assert client.get("/api/history", params={"limit": 100000}).status_code == 422


def test_stage_detail_json_stored_as_valid_json(client: TestClient, db_path: Path) -> None:
    """The stored detail_json strings parse back to the dicts the API serves."""
    run_id = client.post("/api/runs/trigger", json={}).json()["run_id"]
    stored = {
        str(stage): json.loads(str(detail_json))
        for stage, detail_json in _query(
            db_path, "SELECT stage, detail_json FROM run_stages WHERE run_id = ?", run_id
        )
    }
    served = {
        stage["stage"]: stage["detail"] for stage in client.get(f"/api/runs/{run_id}/stages").json()
    }
    assert stored == served


def test_corrupt_detail_json_degrades_to_parse_error_marker(
    client: TestClient, db_path: Path
) -> None:
    """A corrupt stored detail_json must degrade to a parse_error marker on
    that one card, never a permanent 500 on the UI poll target."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            'INSERT INTO runs (started_at_utc, status, "trigger") '
            "VALUES ('2026-07-07T00:00:00Z', 'error', 'manual')"
        )
        run_id = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()[0]
        conn.execute(
            "INSERT INTO run_stages (run_id, stage, status, detail_json) "
            "VALUES (?, 'freshness', 'ok', '{not valid json')",
            (run_id,),
        )
        conn.commit()
    finally:
        conn.close()

    response = client.get(f"/api/runs/{run_id}/stages")
    assert response.status_code == 200
    (stage,) = response.json()
    assert stage["stage"] == "freshness"
    assert "parse_error" in stage["detail"]


def test_failed_startup_probe_leaks_no_writer_connection(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lifespan ordering is load-bearing: the macro probe runs BEFORE
    open_writer, so a raising probe (key set + FRED unreachable propagates by
    macro.py's contract) can never leak an open WAL handle. Fault injection at
    the exact seam api.py calls; the db file's non-existence afterwards proves
    structurally that no writer connection was ever opened."""

    def _raising_probe(api_key: str | None, client: object = None) -> object:
        raise ConnectionError("FRED unreachable during startup probe")

    monkeypatch.setattr(api, "probe_fred_key", _raising_probe)
    with pytest.raises(ConnectionError, match="FRED unreachable"):
        with TestClient(create_app(db_path)):
            pass  # pragma: no cover — startup must fail before entry

    # Probe-before-open, proven: open_writer would have created the file.
    assert not db_path.exists()

    # And the path is fully usable afterwards (no lingering lock/handle).
    writer = storage.open_writer(db_path)
    try:
        writer.execute("SELECT 1").fetchone()
    finally:
        writer.close()
