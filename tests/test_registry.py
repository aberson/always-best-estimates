"""Stage-registry tests (Track 2 Step 18).

Assert every registered key resolves to a working callable + its param schema,
that ``resolve(central_config, forecast_scenario)`` yields the V1 stack, and that
unknown keys fail loud at resolve time.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abe import config, registry, storage
from abe.constants import UNIVERSE
from abe.model.base import EWMABaseline, Forecast


@pytest.fixture()
def writer(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = storage.open_writer(tmp_path / "data" / "abe.db")
    yield conn
    conn.close()


def _price_frames(n: int = 60) -> dict[str, pd.DataFrame]:
    dates = pd.Index(
        [stamp.strftime("%Y-%m-%d") for stamp in pd.bdate_range("2026-01-01", periods=n)],
        name="date",
    )
    rng = np.random.default_rng(3)
    frames: dict[str, pd.DataFrame] = {}
    for asset in UNIVERSE:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=n)))
        frames[asset] = pd.DataFrame({"adj_close": prices}, index=dates)
    return frames


def _forecasts() -> dict[str, Forecast]:
    return {asset: Forecast(mu=0.01, sigma=0.02) for asset in UNIVERSE}


def _bl_posterior() -> tuple[pd.Series, pd.DataFrame]:
    mu_post = pd.Series([0.05, 0.05, 0.02], index=list(UNIVERSE))
    sigma_post = pd.DataFrame(
        np.diag([0.04, 0.045, 0.01]), index=list(UNIVERSE), columns=list(UNIVERSE)
    )
    return mu_post, sigma_post


# --------------------------------------------------------------------------- #
# Each key resolves to a working callable
# --------------------------------------------------------------------------- #


def test_basic_feature_builder_works() -> None:
    builder = registry.FEATURE_BUILDERS["basic"].factory({})
    assert builder.key == "basic"
    bundle = builder.build(_price_frames())
    assert set(bundle.returns) == set(UNIVERSE)
    assert set(bundle.features_frames) == set(UNIVERSE)
    # two rows per asset (log_return, realized_vol) in that order
    assert len(bundle.rows) == 2 * len(UNIVERSE)
    assert {row["name"] for row in bundle.rows} == {"log_return", "realized_vol"}
    # detail is byte-identical in SHAPE to V1 (no extra keys — Step 19 parity)
    assert set(bundle.detail) == {"features", "windows", "latest"}
    assert bundle.detail["features"] == ["log_return", "realized_vol"]


def test_ewma_forecaster_works_and_honors_halflife() -> None:
    model = registry.FORECASTERS["ewma"].factory({})
    assert isinstance(model, EWMABaseline)
    assert model.model_version == "ewma"
    tuned = registry.FORECASTERS["ewma"].factory({"halflife": 30.0})
    assert isinstance(tuned, EWMABaseline)
    assert tuned._halflife == 30.0  # type: ignore[attr-defined]


def test_ewma_factory_rejects_non_numeric_halflife() -> None:
    with pytest.raises(ValueError, match="halflife must be numeric"):
        registry.FORECASTERS["ewma"].factory({"halflife": "fast"})


def test_jepa_factory_fails_loud_without_checkpoint() -> None:
    """jepa is wired to the real loader — no checkpoint means a loud ValueError,
    never a silent EWMA fallback."""
    assert "jepa" in registry.FORECASTERS
    with pytest.raises(ValueError, match="checkpoint"):
        registry.FORECASTERS["jepa"].factory({})


def test_forecast_view_source_is_identity() -> None:
    source = registry.VIEW_SOURCES["forecast"].factory({})
    assert source.kind == "forecast"
    forecasts = _forecasts()
    ctx = registry.ViewContext(forecasts=forecasts)
    provided = source.provide(ctx)
    assert provided == forecasts
    assert provided is not forecasts  # a copy, not the same object


def test_mvu_optimizer_works() -> None:
    optimizer = registry.OPTIMIZERS["mvu"].factory({})
    assert optimizer.key == "mvu"
    mu_post, sigma_post = _bl_posterior()
    result = optimizer.optimize(mu_post, sigma_post, None)
    assert set(result.weights) == set(UNIVERSE)
    assert result.weights == pytest.approx(result.weights)  # finite
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_mvu_optimizer_min_weight_param_and_objective() -> None:
    tuned = registry.OPTIMIZERS["mvu"].factory({"min_weight": 0.05})
    assert tuned.objective()["min_weight"] == 0.05
    # the default (no param) objective has NO min_weight key — byte-identical to V1
    assert "min_weight" not in registry.OPTIMIZERS["mvu"].factory({}).objective()


def test_min_variance_optimizer_resolves() -> None:
    optimizer = registry.OPTIMIZERS["min_variance"].factory({})
    assert optimizer.key == "min_variance"
    mu_post, sigma_post = _bl_posterior()
    result = optimizer.optimize(mu_post, sigma_post, None)
    assert set(result.weights) == set(UNIVERSE)
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert "minimize" in str(optimizer.objective()["form"])


# --------------------------------------------------------------------------- #
# resolve() — the central config yields the V1 stack; unknown keys fail loud
# --------------------------------------------------------------------------- #


def test_resolve_central_config_is_v1_stack(writer: sqlite3.Connection) -> None:
    central = config.get_central_config(writer)
    scenario = config.get_view_scenario(writer, central.view_scenario_id)
    assert scenario is not None
    stack = registry.resolve(central, scenario)
    assert stack.feature_builder.key == "basic"
    assert isinstance(stack.forecaster, EWMABaseline)
    assert stack.forecaster.model_version == "ewma"
    assert stack.view_source.kind == "forecast"
    assert stack.optimizer.key == "mvu"


def test_resolve_unknown_keys_fail_loud(writer: sqlite3.Connection) -> None:
    central = config.get_central_config(writer)
    scenario = config.get_view_scenario(writer, central.view_scenario_id)
    assert scenario is not None
    from dataclasses import replace

    with pytest.raises(ValueError, match="unknown feature_set"):
        registry.resolve(replace(central, feature_set="nope"), scenario)
    with pytest.raises(ValueError, match="unknown forecaster"):
        registry.resolve(replace(central, forecaster="nope"), scenario)
    with pytest.raises(ValueError, match="unknown optimizer"):
        registry.resolve(replace(central, optimizer="nope"), scenario)
    with pytest.raises(ValueError, match="unknown view kind"):
        registry.resolve(central, replace(scenario, kind="nope"))


def test_forecaster_override_supersedes_broken_config_forecaster(
    writer: sqlite3.Connection,
) -> None:
    """A model override must NOT resolve the config's forecaster — so a broken
    central forecaster can't break an override run (the ABE_MODEL rescue path)."""
    central = config.get_central_config(writer)
    scenario = config.get_view_scenario(writer, central.view_scenario_id)
    assert scenario is not None
    from dataclasses import replace

    broken = replace(central, forecaster="does-not-exist")
    override = EWMABaseline()
    # Without an override this would raise; with one, the config forecaster is skipped.
    stack = registry.resolve(broken, scenario, forecaster_override=override)
    assert stack.forecaster is override
    assert stack.feature_builder.key == "basic"  # other stages still resolved


# --------------------------------------------------------------------------- #
# Param-schema manifest (for the UI dropdowns)
# --------------------------------------------------------------------------- #


def test_registries_manifest_shape() -> None:
    manifest = registry.registries_manifest()
    assert set(manifest) == {"feature_set", "forecaster", "view_source", "optimizer"}
    assert set(manifest["feature_set"]) == {"basic", "fracdiff_macro"}
    assert set(manifest["forecaster"]) == {"ewma", "jepa"}
    assert set(manifest["view_source"]) == {"forecast", "historical", "counterfactual"}
    assert set(manifest["optimizer"]) == {"mvu", "min_variance"}
    ewma_params = {p["name"] for p in manifest["forecaster"]["ewma"]["params"]}  # type: ignore[index]
    assert "halflife" in ewma_params
    jepa_params = {p["name"] for p in manifest["forecaster"]["jepa"]["params"]}  # type: ignore[index]
    assert "checkpoint_path" in jepa_params
