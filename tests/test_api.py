"""API tests — the Step 8 integration requirement under the Step 11 scheduler.

POST /api/runs/trigger is driven end-to-end through the PRODUCTION FastAPI
route (TestClient with lifespan context: real macro probe + the real
scheduler owning the writer connection) against a tmp SQLite db seeded with
>= 70 rows/asset of synthetic prices, with the real EWMABaseline — no mocks
of our own code. Row-level assertions read the same db file directly.

Step 11 contract exercised here:

- entering the lifespan fires an immediate ``trigger='startup'`` run (the
  ``client`` fixture waits for it to settle, so every test starts idle and
  deterministic);
- the trigger endpoint answers 202 at the run's START — tests poll the
  ``runs`` row to a terminal status before asserting on its ledger.

The scheduler's daily NETWORK fetch is disabled via the documented
``SchedulerConfig`` seam (the offline suite must never construct a real
``YFinanceAdapter``); the fetch job's own behavior is covered with fake
adapters in tests/test_scheduler.py.

Every test clears FRED_API_KEY and chdirs to an .env-less tmp dir, so the
lifespan macro probe resolves to MACRO_DISABLED_NO_KEY without any network.
"""

import json
import math
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from seeding import OFFLINE_SCHEDULER, seed_prices

from abe import api, storage
from abe.api import create_app
from abe.constants import UNIVERSE
from abe.ingest.macro import MACRO_DISABLED_NO_KEY
from abe.pipeline import STAGES

STARTUP_RUN_ID = 1
"""The lifespan startup run is always run 1 against a fresh db."""


@pytest.fixture()
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in cwd -> the startup probe stays offline
    return tmp_path / "data" / "abe.db"


def _query(db_path: Path, sql: str, *params: object) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _wait_run_terminal(db_path: Path, run_id: int, timeout: float = 30.0) -> str:
    """Poll a run to a terminal status (the 202 answers at run START)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _query(db_path, "SELECT status FROM runs WHERE run_id = ?", run_id)
        if rows:
            status = str(rows[0][0])
            if status not in ("running", "queued"):
                return status
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach a terminal status within {timeout}s")


def _wait_idle(app: FastAPI, timeout: float = 30.0) -> None:
    """Wait for the scheduler's loop iteration to fully settle, so the next
    trigger cannot coalesce (already_running) into a finishing run."""
    deadline = time.monotonic() + timeout
    while app.state.scheduler.run_in_progress:
        if time.monotonic() > deadline:
            raise AssertionError(f"scheduler did not go idle within {timeout}s")
        time.sleep(0.01)


@pytest.fixture()
def client(db_path: Path) -> Iterator[TestClient]:
    seed_prices(db_path)
    application = create_app(db_path, scheduler_config=OFFLINE_SCHEDULER)
    with TestClient(application) as test_client:
        # The scheduler fires an immediate startup run at lifespan enter
        # (plan section 10); settle it so every test starts deterministic.
        assert _wait_run_terminal(db_path, STARTUP_RUN_ID) == "ok"
        _wait_idle(application)
        yield test_client


# --------------------------------------------------------------------------- #
# The end-to-end integration requirement (production route, real scheduler)
# --------------------------------------------------------------------------- #


def test_trigger_end_to_end_through_production_route(client: TestClient, db_path: Path) -> None:
    # The settled startup run armed the freshness gate; force=True makes this
    # manual run recompute all six stages instead of skipping.
    response = client.post("/api/runs/trigger", json={"force": True})
    assert response.status_code == 202
    body = response.json()
    run_id = body["run_id"]
    assert isinstance(run_id, int)
    assert run_id == STARTUP_RUN_ID + 1
    assert body["already_running"] is False

    # The 202 answered at run START; poll it to completion, then assert rows
    # landed in every table, straight from the db file.
    assert _wait_run_terminal(db_path, run_id) == "ok"
    (run_row,) = _query(
        db_path, 'SELECT status, error_text, "trigger" FROM runs WHERE run_id = ?', run_id
    )
    assert run_row == ("ok", None, "manual")

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

    # /api/history returns the runs (newest first) with their weights.
    history = client.get("/api/history", params={"limit": 5})
    assert history.status_code == 200
    runs = history.json()["runs"]
    assert [run["run_id"] for run in runs] == [run_id, STARTUP_RUN_ID]
    assert {row["asset"] for row in runs[0]["target_weights"]} == set(UNIVERSE)

    # /api/runs/{id}/stages returns the same six cards.
    stages = client.get(f"/api/runs/{run_id}/stages")
    assert stages.status_code == 200
    assert [stage["stage"] for stage in stages.json()] == list(STAGES)


def test_freshness_gate_and_force_through_route(client: TestClient, db_path: Path) -> None:
    first = STARTUP_RUN_ID  # the settled startup run ('ok') armed the gate

    # Trigger without force: a skipped run is RECORDED (not silent).
    second_response = client.post("/api/runs/trigger")  # no body at all -> force=False
    assert second_response.status_code == 202
    assert second_response.json()["already_running"] is False
    second = second_response.json()["run_id"]
    assert second != first
    assert _wait_run_terminal(db_path, second) == "skipped"
    second_stages = client.get(f"/api/runs/{second}/stages").json()
    assert [(stage["stage"], stage["status"]) for stage in second_stages] == [
        ("freshness", "skipped")
    ]
    # Latest still points at the startup (ok) run — skipped is not 'ok'.
    assert client.get("/api/runs/latest").json()["run"]["run_id"] == first

    # force=true bypasses the gate on identical data.
    third = client.post("/api/runs/trigger", json={"force": True}).json()["run_id"]
    assert _wait_run_terminal(db_path, third) == "ok"
    assert client.get("/api/runs/latest").json()["run"]["run_id"] == third

    # History shows all three runs, newest first.
    history = client.get("/api/history").json()["runs"]
    assert [run["run_id"] for run in history] == [third, second, first]
    assert [run["status"] for run in history] == ["ok", "skipped", "ok"]
    assert [run["trigger"] for run in history] == ["manual", "manual", "startup"]
    assert history[1]["target_weights"] == []


def test_macro_disabled_mode_surfaces_on_ingest_card(client: TestClient) -> None:
    # The startup run itself carries the probe result on its ingest card.
    stages = client.get(f"/api/runs/{STARTUP_RUN_ID}/stages").json()
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
    application = create_app(db, scheduler_config=OFFLINE_SCHEDULER)
    with TestClient(application) as client:
        # The startup run errors too (same short history); settle it first so
        # the trigger below starts a FRESH run instead of coalescing.
        assert _wait_run_terminal(db, STARTUP_RUN_ID) == "error"
        _wait_idle(application)

        response = client.post("/api/runs/trigger", json={})
        assert response.status_code == 202
        assert response.json()["already_running"] is False
        run_id = response.json()["run_id"]
        assert _wait_run_terminal(db, run_id) == "error"

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


def test_startup_run_fires_and_latest_serves_it(client: TestClient) -> None:
    """Plan section 10: 'a run fires immediately at startup' — through the
    production route, without any POST."""
    latest = client.get("/api/runs/latest")
    assert latest.status_code == 200
    run = latest.json()["run"]
    assert run["run_id"] == STARTUP_RUN_ID
    assert run["status"] == "ok"
    assert run["trigger"] == "startup"


def test_unknown_run_stages_404(client: TestClient) -> None:
    response = client.get("/api/runs/9999/stages")
    assert response.status_code == 404


def test_explain_endpoint_serves_calculation_registry(client: TestClient) -> None:
    """GET /api/explain returns the static calculation registry (Track 1):
    a legible formula + worked example per quantity the stage cards read.
    Read-only, no run state needed."""
    response = client.get("/api/explain")
    assert response.status_code == 200
    payload = response.json()
    assert "explanations" in payload
    explanations = payload["explanations"]
    # Every quantity a card displays has an entry.
    for key in (
        "log_return",
        "realized_vol",
        "ewma_mu",
        "forecast_sigma",
        "bl_prior",
        "bl_view",
        "bl_confidence",
        "bl_posterior",
        "mvu_objective",
    ):
        assert key in explanations, key
    # Each entry carries exactly the documented fields.
    entry = explanations["bl_posterior"]
    assert set(entry) == {"label", "formula", "description", "example", "unit", "window"}
    assert entry["label"] and entry["formula"] and entry["example"]


def test_history_startup_run_and_limit_validation(client: TestClient) -> None:
    history = client.get("/api/history").json()["runs"]
    assert [run["run_id"] for run in history] == [STARTUP_RUN_ID]
    assert client.get("/api/history", params={"limit": 0}).status_code == 422
    assert client.get("/api/history", params={"limit": 100000}).status_code == 422


def test_stage_detail_json_stored_as_valid_json(client: TestClient, db_path: Path) -> None:
    """The stored detail_json strings parse back to the dicts the API serves
    (asserted on the startup run — a full six-stage ok run)."""
    stored = {
        str(stage): json.loads(str(detail_json))
        for stage, detail_json in _query(
            db_path,
            "SELECT stage, detail_json FROM run_stages WHERE run_id = ?",
            STARTUP_RUN_ID,
        )
    }
    served = {
        stage["stage"]: stage["detail"]
        for stage in client.get(f"/api/runs/{STARTUP_RUN_ID}/stages").json()
    }
    assert set(stored) == set(STAGES)
    assert stored == served


def test_corrupt_detail_json_degrades_to_parse_error_marker(
    client: TestClient, db_path: Path
) -> None:
    """A corrupt stored detail_json must degrade to a parse_error marker on
    that one card, never a permanent 500 on the UI poll target."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
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
    """Lifespan ordering is load-bearing: the macro probe runs BEFORE the
    scheduler opens the writer connection, so a raising probe (key set + FRED
    unreachable propagates by macro.py's contract) can never leak an open WAL
    handle. Fault injection at the exact seam api.py calls; the db file's
    non-existence afterwards proves structurally that no writer connection
    (and no scheduler) was ever opened."""

    def _raising_probe(api_key: str | None, client: object = None) -> object:
        raise ConnectionError("FRED unreachable during startup probe")

    monkeypatch.setattr(api, "probe_fred_key", _raising_probe)
    with pytest.raises(ConnectionError, match="FRED unreachable"):
        with TestClient(create_app(db_path, scheduler_config=OFFLINE_SCHEDULER)):
            pass  # pragma: no cover — startup must fail before entry

    # Probe-before-scheduler, proven: the scheduler's open_writer would have
    # created the file.
    assert not db_path.exists()

    # And the path is fully usable afterwards (no lingering lock/handle).
    writer = storage.open_writer(db_path)
    try:
        writer.execute("SELECT 1").fetchone()
    finally:
        writer.close()
