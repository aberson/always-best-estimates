"""View-source provider tests (Track 2 Step 22).

Goldens (plan §7 Step 22 done-when): forecast reproduces today's views; a
counterfactual view materially moves the posterior toward it AND round-trips its
target through bl_blend's derivation; a historical window yields the expected
sign. Plus the pre-programmed library seed.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from seeding import seed_prices

from abe import config as config_module
from abe import storage
from abe.blend.black_litterman import bl_blend
from abe.blend.covariance import ledoit_wolf_sigma
from abe.blend.views import (
    LIBRARY,
    CounterfactualViewSource,
    ForecastViewSource,
    HistoricalViewSource,
    ViewContext,
    seed_library,
)
from abe.constants import UNIVERSE
from abe.model.base import Forecast


def _returns_frame(n: int = 90) -> pd.DataFrame:
    dates = pd.Index(
        [stamp.strftime("%Y-%m-%d") for stamp in pd.bdate_range("2026-01-01", periods=n)],
        name="date",
    )
    rng = np.random.default_rng(11)
    cols = {asset: rng.normal(0.0002, 0.01, size=n) for asset in UNIVERSE}
    return pd.DataFrame(cols, index=dates)[list(UNIVERSE)]


def _sigma() -> pd.DataFrame:
    return ledoit_wolf_sigma(_returns_frame())


# --------------------------------------------------------------------------- #
# forecast: identity on the model's forecasts
# --------------------------------------------------------------------------- #


def test_forecast_source_is_identity() -> None:
    forecasts = {asset: Forecast(mu=0.01, sigma=0.02) for asset in UNIVERSE}
    provided = ForecastViewSource().provide(ViewContext(forecasts=forecasts))
    assert provided == forecasts
    assert provided is not forecasts


# --------------------------------------------------------------------------- #
# counterfactual: round-trips the target view + moves the posterior toward it
# --------------------------------------------------------------------------- #


def test_counterfactual_round_trips_view_and_confidence() -> None:
    """bl_blend re-derives the counterfactual's target annual view + confidence."""
    source = CounterfactualViewSource({"SPY": {"mu": 0.10, "confidence": 0.5}})
    views = source.provide(ViewContext(forecasts={}))
    bl = bl_blend(_sigma(), views)
    assert bl.diagnostics["Q"]["SPY"] == pytest.approx(0.10, abs=1e-9)
    assert bl.diagnostics["confidences"]["SPY"] == pytest.approx(0.5, abs=1e-9)


def test_counterfactual_moves_posterior_toward_the_view() -> None:
    sigma = _sigma()
    prior = bl_blend(sigma, {}).mu_post  # empty views → the equilibrium prior
    bullish = CounterfactualViewSource({"SPY": {"mu": 0.40, "confidence": 0.9}})
    post = bl_blend(sigma, bullish.provide(ViewContext(forecasts={}))).mu_post
    # a strong bullish SPY view drags SPY's posterior up above the prior
    assert post["SPY"] > prior["SPY"]


def test_counterfactual_rejects_unknown_asset_and_missing_mu() -> None:
    with pytest.raises(ValueError, match="not in UNIVERSE"):
        CounterfactualViewSource({"TSLA": {"mu": 0.1}})
    with pytest.raises(ValueError, match="'mu'"):
        CounterfactualViewSource({"SPY": {"confidence": 0.5}})


# --------------------------------------------------------------------------- #
# historical: window realized returns → expected sign
# --------------------------------------------------------------------------- #


def _series(values: list[float]) -> pd.Series:
    stamps = pd.bdate_range("2026-01-01", periods=len(values))
    dates = [stamp.strftime("%Y-%m-%d") for stamp in stamps]
    return pd.Series(values, index=pd.Index(dates, name="date"))


def test_historical_view_sign_matches_window_mean() -> None:
    up = {asset: _series([0.01] * 40) for asset in UNIVERSE}
    down = {asset: _series([-0.01] * 40) for asset in UNIVERSE}
    up_views = HistoricalViewSource({}).provide(ViewContext(forecasts={}, returns=up))
    down_views = HistoricalViewSource({}).provide(ViewContext(forecasts={}, returns=down))
    assert all(up_views[asset].mu > 0 for asset in UNIVERSE)
    assert all(down_views[asset].mu < 0 for asset in UNIVERSE)


def test_historical_window_slices_the_series() -> None:
    # negative early, positive late; a late window yields a positive view
    values = [-0.02] * 30 + [0.02] * 30
    series = {"SPY": _series(values)}
    late_start = _series(values).index[35]
    late = HistoricalViewSource({"window_start": late_start}).provide(
        ViewContext(forecasts={}, returns=series)
    )
    assert late["SPY"].mu > 0


def test_historical_skips_assets_without_enough_history() -> None:
    views = HistoricalViewSource({}).provide(
        ViewContext(forecasts={}, returns={"SPY": _series([0.01])})  # 1 point < 2
    )
    assert "SPY" not in views


# --------------------------------------------------------------------------- #
# pre-programmed library
# --------------------------------------------------------------------------- #


@pytest.fixture()
def writer(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    seed_prices(tmp_path / "data" / "abe.db")
    conn = storage.open_writer(tmp_path / "data" / "abe.db")
    yield conn
    conn.close()


def test_seed_library_is_idempotent(writer: sqlite3.Connection) -> None:
    first = seed_library(writer)
    assert len(first) == len(LIBRARY)
    names = {vs.name for vs in config_module.list_view_scenarios(writer)}
    assert {str(entry["name"]) for entry in LIBRARY} <= names
    # re-seeding returns the SAME ids (matched by name), creates no duplicates
    second = seed_library(writer)
    assert second == first
    counts = writer.execute(
        "SELECT name, COUNT(*) FROM view_scenarios GROUP BY name HAVING COUNT(*) > 1"
    ).fetchall()
    assert counts == []


def test_library_scenarios_resolve_and_run(writer: sqlite3.Connection) -> None:
    """Every library scenario resolves to a working view source (Step 18 registry)."""
    from abe import registry

    seed_library(writer)
    for scenario in config_module.list_view_scenarios(writer):
        source = registry.VIEW_SOURCES[scenario.kind].factory(scenario.payload)
        assert source.kind == scenario.kind
