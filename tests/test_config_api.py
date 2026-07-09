"""Config / scenario / compare API tests (Track 2 Step 25).

Through the production FastAPI routes (TestClient + lifespan → real scheduler,
single writer). Covers CRUD, the set-central invariant (exactly one central), and
the /api/compare payload shape.
"""

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from seeding import OFFLINE_SCHEDULER, seed_prices

from abe.api import create_app
from abe.constants import UNIVERSE

STARTUP_RUN_ID = 1


def _wait_terminal(app: FastAPI, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while app.state.scheduler.run_in_progress:
        if time.monotonic() > deadline:
            raise AssertionError("scheduler did not go idle")
        time.sleep(0.01)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "data" / "abe.db"
    seed_prices(db_path, days=90)
    app = create_app(db_path, scheduler_config=OFFLINE_SCHEDULER)
    with TestClient(app) as test_client:
        deadline = time.monotonic() + 30.0
        while test_client.get("/api/runs/latest").status_code != 200:
            if time.monotonic() > deadline:
                raise AssertionError("startup run never settled")
            time.sleep(0.02)
        _wait_terminal(app)
        yield test_client


def _default_view_id(client: TestClient) -> int:
    return client.get("/api/scenarios").json()["scenarios"][0]["view_scenario_id"]


def _make_config(client: TestClient, name: str, **overrides: object) -> dict:
    body = {
        "name": name,
        "feature_set": "basic",
        "forecaster": "ewma",
        "view_scenario_id": _default_view_id(client),
        "optimizer": "mvu",
        **overrides,
    }
    resp = client.post("/api/configs", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Registries + config CRUD
# --------------------------------------------------------------------------- #


def test_registries_endpoint(client: TestClient) -> None:
    manifest = client.get("/api/registries").json()["registries"]
    assert set(manifest) == {"feature_set", "forecaster", "view_source", "optimizer"}
    assert "min_variance" in manifest["optimizer"]


def test_config_crud_round_trip(client: TestClient) -> None:
    created = _make_config(client, "alt", optimizer="min_variance")
    cid = created["config_id"]
    assert created["is_central"] is False
    assert created["optimizer"] == "min_variance"
    # list includes it
    names = {c["name"] for c in client.get("/api/configs").json()["configs"]}
    assert "alt" in names
    # get
    assert client.get(f"/api/configs/{cid}").json()["config_id"] == cid
    # patch
    patched = client.patch(f"/api/configs/{cid}", json={"forecaster": "jepa"}).json()
    assert patched["forecaster"] == "jepa"
    # delete → 204, then 404
    assert client.delete(f"/api/configs/{cid}").status_code == 204
    assert client.get(f"/api/configs/{cid}").status_code == 404


def test_config_create_duplicate_name_conflicts(client: TestClient) -> None:
    _make_config(client, "dupe")
    body = {
        "name": "dupe",
        "feature_set": "basic",
        "forecaster": "ewma",
        "view_scenario_id": _default_view_id(client),
        "optimizer": "mvu",
    }
    assert client.post("/api/configs", json=body).status_code == 409


def test_get_unknown_config_404(client: TestClient) -> None:
    assert client.get("/api/configs/9999").status_code == 404


# --------------------------------------------------------------------------- #
# set-central invariant
# --------------------------------------------------------------------------- #


def test_set_central_moves_the_flag_exactly_one(client: TestClient) -> None:
    alt = _make_config(client, "challenger")
    promoted = client.post(f"/api/configs/{alt['config_id']}/central")
    assert promoted.status_code == 200
    assert promoted.json()["is_central"] is True
    configs = client.get("/api/configs").json()["configs"]
    centrals = [c for c in configs if c["is_central"]]
    assert len(centrals) == 1
    assert centrals[0]["config_id"] == alt["config_id"]


def test_delete_central_conflicts(client: TestClient) -> None:
    central = next(c for c in client.get("/api/configs").json()["configs"] if c["is_central"])
    assert client.delete(f"/api/configs/{central['config_id']}").status_code == 409


# --------------------------------------------------------------------------- #
# scenario CRUD
# --------------------------------------------------------------------------- #


def test_scenario_crud_and_bad_kind(client: TestClient) -> None:
    created = client.post(
        "/api/scenarios",
        json={"name": "SPY +10%", "kind": "counterfactual", "payload": {"SPY": {"mu": 0.1}}},
    )
    assert created.status_code == 201
    sid = created.json()["view_scenario_id"]
    renamed = client.patch(f"/api/scenarios/{sid}", json={"name": "renamed"})
    assert renamed.json()["name"] == "renamed"
    assert client.delete(f"/api/scenarios/{sid}").status_code == 204
    # bad kind → 400
    bad = client.post("/api/scenarios", json={"name": "x", "kind": "nonsense"})
    assert bad.status_code == 400


def test_delete_referenced_scenario_conflicts(client: TestClient) -> None:
    # the default forecast scenario is referenced by the central config
    sid = _default_view_id(client)
    assert client.delete(f"/api/scenarios/{sid}").status_code == 409


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def test_compare_payload_shape(client: TestClient) -> None:
    alt = _make_config(client, "alt-run")
    # run the alt config so it has weights
    run = client.post(f"/api/configs/{alt['config_id']}/run")
    assert run.status_code == 200
    central_id = next(
        c["config_id"] for c in client.get("/api/configs").json()["configs"] if c["is_central"]
    )
    resp = client.get(f"/api/compare?config_ids={central_id},{alt['config_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["central_config_id"] == central_id
    by_id = {c["config_id"]: c for c in body["configs"]}
    assert set(by_id) == {central_id, alt["config_id"]}
    assert by_id[central_id]["is_central"] is True
    assert by_id[alt["config_id"]]["is_central"] is False
    for entry in body["configs"]:
        assert set(entry) == {
            "config_id",
            "name",
            "is_central",
            "weights",
            "objective",
            "run_id",
            "finished_at_utc",
        }
        assert set(entry["weights"]) == set(UNIVERSE)
        assert entry["objective"]["form"]  # the optimizer's objective is surfaced


def test_compare_empty_config_ids_422(client: TestClient) -> None:
    assert client.get("/api/compare?config_ids=").status_code == 422
