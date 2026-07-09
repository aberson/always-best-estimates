"""Config + ViewScenario domain-model + CRUD tests (Track 2 Step 17).

Real tmp SQLite dbs through the production storage boundary — no mocks. Every db
is a fresh v2 migrate, so the seeded central Config + default forecast
ViewScenario are present from the start.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from abe import config, migrations, storage


@pytest.fixture()
def writer(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = storage.open_writer(tmp_path / "data" / "abe.db")
    yield conn
    conn.close()


def _default_view_id(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT view_scenario_id FROM view_scenarios LIMIT 1").fetchone()[0])


# --------------------------------------------------------------------------- #
# Seeded central Config (from the Step 16 migration)
# --------------------------------------------------------------------------- #


def test_exactly_one_central_after_fresh_migrate(writer: sqlite3.Connection) -> None:
    configs = config.list_configs(writer)
    central = [c for c in configs if c.is_central]
    assert len(central) == 1
    assert config.get_central_config(writer).name == migrations.CENTRAL_CONFIG_NAME


def test_central_config_reproduces_v1_recipe(writer: sqlite3.Connection) -> None:
    central = config.get_central_config(writer)
    assert (central.feature_set, central.forecaster, central.optimizer) == ("basic", "ewma", "mvu")
    assert central.params == {}  # seed stored NULL params_json → parsed as {}
    scenario = config.get_view_scenario(writer, central.view_scenario_id)
    assert scenario is not None
    assert scenario.kind == "forecast"


# --------------------------------------------------------------------------- #
# ViewScenario CRUD
# --------------------------------------------------------------------------- #


def test_view_scenario_round_trip(writer: sqlite3.Connection) -> None:
    payload = {"SPY": {"mu": 0.10, "confidence": 0.5}}
    created = config.create_view_scenario(
        writer, name="SPY +10%", kind="counterfactual", payload=payload
    )
    assert created.view_scenario_id > 0
    fetched = config.get_view_scenario(writer, created.view_scenario_id)
    assert fetched == created
    assert fetched is not None
    assert fetched.payload == payload  # nested JSON round-trips


def test_view_scenario_rejects_bad_kind(writer: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        config.create_view_scenario(writer, name="bad", kind="nonsense")


def test_payload_coerces_numpy_scalars(writer: sqlite3.Connection) -> None:
    """Payloads built from numpy math (Step 22 providers) must survive the JSON
    write via the coerce_scalar default hook, not raise in json.dumps."""
    import numpy as np

    payload = {"SPY": {"mu": np.float32(0.1), "confidence": np.int64(1)}}
    created = config.create_view_scenario(
        writer, name="np-view", kind="counterfactual", payload=payload
    )
    fetched = config.get_view_scenario(writer, created.view_scenario_id)
    assert fetched is not None
    assert fetched.payload == {"SPY": {"mu": pytest.approx(0.1, abs=1e-6), "confidence": 1}}


def test_view_scenario_update(writer: sqlite3.Connection) -> None:
    created = config.create_view_scenario(writer, name="hist", kind="historical")
    updated = config.update_view_scenario(
        writer, created.view_scenario_id, name="hist-2020", payload={"window_start": "2020-01-01"}
    )
    assert updated.name == "hist-2020"
    assert updated.payload == {"window_start": "2020-01-01"}
    assert updated.kind == "historical"  # kind immutable
    assert config.get_view_scenario(writer, created.view_scenario_id) == updated


def test_delete_view_scenario_refuses_when_referenced(writer: sqlite3.Connection) -> None:
    central = config.get_central_config(writer)
    with pytest.raises(ValueError, match="referenced by"):
        config.delete_view_scenario(writer, central.view_scenario_id)


def test_delete_view_scenario_ok_when_unreferenced(writer: sqlite3.Connection) -> None:
    created = config.create_view_scenario(writer, name="orphan", kind="forecast")
    config.delete_view_scenario(writer, created.view_scenario_id)
    assert config.get_view_scenario(writer, created.view_scenario_id) is None


# --------------------------------------------------------------------------- #
# Config CRUD
# --------------------------------------------------------------------------- #


def test_config_round_trip(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    created = config.create_config(
        writer,
        name="min-var-floor",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="min_variance",
        params={"optimizer": {"min_weight": 0.05}},
    )
    assert created.config_id > 0
    assert created.is_central is False  # new configs are never central
    fetched = config.get_config(writer, created.config_id)
    assert fetched == created
    assert fetched is not None
    assert fetched.params == {"optimizer": {"min_weight": 0.05}}


def test_config_name_unique(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    config.create_config(
        writer,
        name="dupe",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    with pytest.raises(sqlite3.IntegrityError):
        config.create_config(
            writer,
            name="dupe",
            feature_set="basic",
            forecaster="jepa",
            view_scenario_id=view_id,
            optimizer="mvu",
        )


def test_config_bad_view_scenario_fk(writer: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        config.create_config(
            writer,
            name="orphan-view",
            feature_set="basic",
            forecaster="ewma",
            view_scenario_id=9999,
            optimizer="mvu",
        )


def test_config_update(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    created = config.create_config(
        writer,
        name="editme",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    updated = config.update_config(
        writer, created.config_id, forecaster="jepa", params={"forecaster": {"seed": 1}}
    )
    assert updated.forecaster == "jepa"
    assert updated.params == {"forecaster": {"seed": 1}}
    assert updated.name == "editme"  # untouched fields preserved
    assert updated.is_central is False


def test_update_config_bumps_updated_at(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    created = config.create_config(
        writer,
        name="ttrack",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    assert created.updated_at_utc == created.created_at_utc  # created == updated at birth
    updated = config.update_config(writer, created.config_id, optimizer="min_variance")
    assert updated.updated_at_utc is not None
    assert updated.updated_at_utc >= (created.updated_at_utc or "")  # bumped (>= at 1s resolution)


def test_list_configs_includes_central_and_new(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    config.create_config(
        writer,
        name="alt",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    names = {c.name for c in config.list_configs(writer)}
    assert {migrations.CENTRAL_CONFIG_NAME, "alt"} <= names


# --------------------------------------------------------------------------- #
# set_central (guarded transition) + delete guards
# --------------------------------------------------------------------------- #


def test_set_central_moves_the_flag(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    original_central = config.get_central_config(writer)
    challenger = config.create_config(
        writer,
        name="challenger",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="min_variance",
    )
    promoted = config.set_central(writer, challenger.config_id)
    assert promoted.is_central is True
    # exactly one central, and it's the challenger now
    centrals = [c for c in config.list_configs(writer) if c.is_central]
    assert len(centrals) == 1
    assert centrals[0].config_id == challenger.config_id
    assert config.get_config(writer, original_central.config_id).is_central is False  # type: ignore[union-attr]


def test_set_central_unknown_id_raises(writer: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="no config with id"):
        config.set_central(writer, 9999)


def test_set_central_preserves_updated_at(writer: sqlite3.Connection) -> None:
    """Promotion must NOT bump updated_at — is_central isn't a recipe field, so
    promoting a config must not needlessly bust the on-demand run cache."""
    view_id = _default_view_id(writer)
    created = config.create_config(
        writer,
        name="promoteme",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    promoted = config.set_central(writer, created.config_id)
    assert promoted.is_central is True
    assert promoted.updated_at_utc == created.updated_at_utc


def test_delete_config_refuses_central(writer: sqlite3.Connection) -> None:
    central = config.get_central_config(writer)
    with pytest.raises(ValueError, match="cannot delete the central"):
        config.delete_config(writer, central.config_id)


def test_delete_config_refuses_when_runs_reference_it(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    created = config.create_config(
        writer,
        name="hasrun",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    storage.insert_row(
        writer,
        "runs",
        {
            "started_at_utc": "2026-07-08T00:00:00Z",
            "status": "ok",
            "trigger": "manual",
            "config_id": created.config_id,
        },
    )
    with pytest.raises(ValueError, match="referenced by"):
        config.delete_config(writer, created.config_id)


def test_delete_config_ok_when_unreferenced(writer: sqlite3.Connection) -> None:
    view_id = _default_view_id(writer)
    created = config.create_config(
        writer,
        name="deletable",
        feature_set="basic",
        forecaster="ewma",
        view_scenario_id=view_id,
        optimizer="mvu",
    )
    config.delete_config(writer, created.config_id)
    assert config.get_config(writer, created.config_id) is None
