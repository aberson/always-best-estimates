"""WorldModel contract + EWMA baseline tests — THE interface freeze (plan Step 5).

``assert_worldmodel_contract`` is the reusable contract check every WorldModel
implementation must pass. Step 13's JEPA imports and runs this SAME function
(``from test_model_base import assert_worldmodel_contract``) — do not fork it.

No mocks; synthetic pandas series only; fast.
"""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from abe import calc, constants
from abe.calc import LOG_RETURN_COLUMN, log_returns
from abe.model import base
from abe.model.base import (
    MIN_HISTORY_BARS,
    EWMABaseline,
    Forecast,
    WorldModel,
)

# --------------------------------------------------------------------------- #
# Synthetic feature builders (deterministic; ISO-date-string index like the
# CacheAdapter shape)
# --------------------------------------------------------------------------- #


def _dates(n: int) -> pd.Index:
    return pd.Index(pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"), name="date")


def _returns_frame(values: list[float] | np.ndarray) -> pd.DataFrame:
    array = np.asarray(values, dtype=float)
    return pd.DataFrame({LOG_RETURN_COLUMN: array}, index=_dates(len(array)))


def _noisy_frame(seed: int, n: int = 300, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _returns_frame(drift + 0.01 * rng.standard_normal(n))


def _near_constant_frame(daily_r: float = 0.001, n: int = 200, eps: float = 1e-6) -> pd.DataFrame:
    # Constant drift + tiny alternating noise: non-degenerate (sigma > 0) yet
    # the EWMA mean is pinned to daily_r within ~eps — a KNOWN input for the
    # H-day mu-scaling assertion.
    noise = np.where(np.arange(n) % 2 == 0, eps, -eps)
    return _returns_frame(daily_r + noise)


# --------------------------------------------------------------------------- #
# THE WorldModel contract (frozen here; plan Step 5 / section 8).
# Step 13's JEPA must pass this exact function unchanged.
# --------------------------------------------------------------------------- #


def assert_worldmodel_contract(
    model: WorldModel,
    features: dict[str, pd.DataFrame],
    *,
    expected_daily_mu: float | None = None,
    mu_rel_tol: float = 0.05,
) -> None:
    """Assert ``model`` honors the frozen WorldModel contract on ``features``.

    Checks: runtime Protocol membership; non-empty ``model_version``; output
    keys == input keys (a UNIVERSE subset); every value a ``Forecast`` with
    finite native-float ``mu`` and finite ``sigma > 0`` (the H-day PREDICTIVE
    forecast std — full semantics on ``Forecast``). When ``expected_daily_mu``
    is given (a known-input drift), also asserts the documented H-day horizon
    semantic: ``mu ~= expected_daily_mu * HORIZON_BARS`` — i.e. mu is an H-day
    quantity, not daily or annualized.

    REQUIRED of every implementation's test suite (EWMA here, Step 13's JEPA):
    at least one invocation MUST arm ``expected_daily_mu`` on a known-drift
    input — the horizon pin is mandatory coverage, not opt-in — and the sigma
    SCALE must be pinned on iid noise (sigma ~ s*sqrt(H); see
    ``test_ewma_sigma_is_h_day_scaled_on_iid_noise``).
    """
    assert isinstance(model, WorldModel), "model must satisfy the runtime-checkable WorldModel"
    assert isinstance(model.model_version, str), "model_version must be a str"
    assert model.model_version, "model_version must be non-empty"
    assert features, "contract check needs at least one asset frame"
    assert set(features) <= set(constants.UNIVERSE), (
        f"input keys must be a UNIVERSE subset, got {sorted(features)}"
    )
    result = model.forecast(features)
    assert set(result) == set(features), (
        f"output keys {sorted(result)} != input keys {sorted(features)}"
    )
    for asset, forecast in result.items():
        assert isinstance(forecast, Forecast), f"{asset}: expected Forecast, got {type(forecast)}"
        assert type(forecast.mu) is float, f"{asset}: mu must be a native float"
        assert math.isfinite(forecast.mu), f"{asset}: mu must be finite, got {forecast.mu!r}"
        assert type(forecast.sigma) is float, f"{asset}: sigma must be a native float"
        assert math.isfinite(forecast.sigma), f"{asset}: sigma must be finite"
        assert forecast.sigma > 0.0, (
            f"{asset}: sigma must be strictly positive (predictive H-day std)"
        )
        if expected_daily_mu is not None:
            expected_h_day = expected_daily_mu * constants.HORIZON_BARS
            assert forecast.mu == pytest.approx(expected_h_day, rel=mu_rel_tol), (
                f"{asset}: mu={forecast.mu} is not H-day-scaled — expected "
                f"~{expected_h_day} (= daily {expected_daily_mu} x HORIZON_BARS="
                f"{constants.HORIZON_BARS}); daily or annualized mu violates the contract"
            )


def test_ewma_passes_contract_on_full_universe() -> None:
    features = {asset: _noisy_frame(seed) for seed, asset in enumerate(constants.UNIVERSE, start=1)}
    assert_worldmodel_contract(EWMABaseline(), features)


def test_ewma_passes_contract_on_universe_subset() -> None:
    assert_worldmodel_contract(EWMABaseline(), {"AGG": _noisy_frame(9)})


def test_contract_asserts_h_day_mu_scaling_on_known_input() -> None:
    # THE mandatory known-drift horizon pin: ~0.001 daily drift => mu must land
    # near 0.001 * 21 within 0.5% — a daily-scale mu (21x too small) or an
    # annualized one (12x too big) fails by orders of magnitude.
    features = {"SPY": _near_constant_frame(daily_r=0.001)}
    assert_worldmodel_contract(EWMABaseline(), features, expected_daily_mu=0.001, mu_rel_tol=0.005)


def test_non_model_fails_runtime_worldmodel_check() -> None:
    class NotAModel:
        model_version = "nope"  # has the attr but no forecast()

    assert not isinstance(object(), WorldModel)
    assert not isinstance(NotAModel(), WorldModel)


# --------------------------------------------------------------------------- #
# Garbage anchors: the contract function must FAIL wrong output (a gate that
# cannot fail garbage cannot gate — workspace measurement-validity rule).
# --------------------------------------------------------------------------- #


class _DailyScaleGarbageModel:
    """Emits mu on the DAILY scale (forgets the xH) — must fail the horizon pin."""

    model_version: str = "garbage-daily-mu"

    def __init__(self, daily_mu: float) -> None:
        self._daily_mu = daily_mu

    def forecast(self, features: dict[str, pd.DataFrame]) -> dict[str, Forecast]:
        return {asset: Forecast(mu=self._daily_mu, sigma=0.05) for asset in features}


class _WrongKeysGarbageModel:
    """Emits a hardcoded asset key regardless of input — must fail the key check."""

    model_version: str = "garbage-keys"

    def forecast(self, features: dict[str, pd.DataFrame]) -> dict[str, Forecast]:
        return {"AGG": Forecast(mu=0.01, sigma=0.05)}


def test_contract_fails_daily_scale_mu_garbage_anchor() -> None:
    features = {"SPY": _near_constant_frame(daily_r=0.001)}
    with pytest.raises(AssertionError, match="H-day"):
        assert_worldmodel_contract(
            _DailyScaleGarbageModel(0.001), features, expected_daily_mu=0.001
        )


def test_contract_fails_wrong_output_keys_garbage_anchor() -> None:
    with pytest.raises(AssertionError, match="keys"):
        assert_worldmodel_contract(_WrongKeysGarbageModel(), {"SPY": _noisy_frame(1)})


# --------------------------------------------------------------------------- #
# EWMA math on known synthetic series
# --------------------------------------------------------------------------- #


def test_ewma_sigma_is_h_day_scaled_on_iid_noise() -> None:
    # Pins sigma's SCALE, not just its sign: on iid daily noise with std s, the
    # H-day predictive forecast std must land near s*sqrt(H) (independent H-day
    # sums have std s*sqrt(21) ~ 4.6*s; the EWMA adds a small forecast term).
    # The [0.5x, 2x] band excludes a daily-scale sigma (~s, 4.6x too small) and
    # an annualized one (~s*sqrt(252), 3.5x too big) with margin — either bug
    # would silently distort every Idzorek confidence downstream.
    s = 0.01
    rng = np.random.default_rng(23)
    features = {"SPY": _returns_frame(s * rng.standard_normal(600))}
    forecast = EWMABaseline().forecast(features)["SPY"]
    expected = s * math.sqrt(constants.HORIZON_BARS)
    assert 0.5 * expected < forecast.sigma < 2.0 * expected


def test_ewma_alternating_series_sanity() -> None:
    # Alternating +/-1%: the EWMA mean is ~0 (mu small), while realized 21-day
    # sums alternate +/-1% against a ~0 forecast => sigma lands near 0.01.
    values = np.where(np.arange(300) % 2 == 0, 0.01, -0.01)
    forecast = EWMABaseline().forecast({"SPY": _returns_frame(values)})["SPY"]
    assert abs(forecast.mu) < 0.005
    assert 0.001 < forecast.sigma < 0.05


def test_ewma_rejects_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        EWMABaseline().forecast({"SPY": _returns_frame([0.001] * 200)})


def test_ewma_rejects_too_short_history() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="too short"):
        EWMABaseline().forecast(
            {"SPY": _returns_frame(0.01 * rng.standard_normal(MIN_HISTORY_BARS - 1))}
        )


def test_ewma_accepts_minimum_history_boundary() -> None:
    rng = np.random.default_rng(4)
    features = {"SPY": _returns_frame(0.01 * rng.standard_normal(MIN_HISTORY_BARS))}
    assert_worldmodel_contract(EWMABaseline(), features)


@pytest.mark.parametrize(
    ("bad_value", "match"),
    [
        pytest.param(float("nan"), "NaN/non-finite", id="nan"),
        pytest.param(float("inf"), "NaN/non-finite", id="inf"),
    ],
)
def test_ewma_rejects_non_finite_returns(bad_value: float, match: str) -> None:
    rng = np.random.default_rng(6)
    values = 0.01 * rng.standard_normal(100)
    values[50] = bad_value
    with pytest.raises(ValueError, match=match):
        EWMABaseline().forecast({"SPY": _returns_frame(values)})


def test_ewma_rejects_missing_log_return_column() -> None:
    frame = pd.DataFrame({"ret": [0.01] * 100}, index=_dates(100))
    with pytest.raises(ValueError, match="log_return"):
        EWMABaseline().forecast({"SPY": frame})


def test_ewma_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        EWMABaseline().forecast({})


def test_ewma_rejects_disordered_index() -> None:
    # The EWMA walk is order-dependent; a disordered frame must fail loud, not
    # produce a silently wrong forecast.
    disordered = _noisy_frame(8).iloc[::-1]
    with pytest.raises(ValueError, match="ascending"):
        EWMABaseline().forecast({"SPY": disordered})


@pytest.mark.parametrize("halflife", [0.0, -5.0, float("nan"), float("inf")])
def test_ewma_rejects_bad_halflife(halflife: float) -> None:
    with pytest.raises(ValueError, match="halflife"):
        EWMABaseline(halflife=halflife)


def test_ewma_is_deterministic() -> None:
    features = {asset: _noisy_frame(seed) for seed, asset in enumerate(constants.UNIVERSE)}
    first = EWMABaseline().forecast(features)
    second = EWMABaseline().forecast(features)
    assert first == second  # dataclass equality => bitwise-identical mu/sigma


def test_ewma_tolerates_extra_feature_columns() -> None:
    # Contract: frames carry AT LEAST log_return; extra columns (Step 12's
    # richer matrix) must not change the EWMA's answer.
    minimal = _noisy_frame(31)
    enriched = minimal.assign(realized_vol=0.15, dgs10=4.2)
    baseline = EWMABaseline()
    assert baseline.forecast({"SPY": minimal}) == baseline.forecast({"SPY": enriched})


def test_ewma_model_version() -> None:
    assert EWMABaseline().model_version == "ewma"


# --------------------------------------------------------------------------- #
# Forecast dataclass invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mu", "sigma", "match"),
    [
        pytest.param(0.01, 0.0, r"Forecast\.sigma", id="sigma-zero"),
        pytest.param(0.01, -0.1, r"Forecast\.sigma", id="sigma-negative"),
        pytest.param(0.01, float("nan"), r"Forecast\.sigma", id="sigma-nan"),
        pytest.param(0.01, float("inf"), r"Forecast\.sigma", id="sigma-inf"),
        pytest.param(float("nan"), 0.1, r"Forecast\.mu", id="mu-nan"),
        pytest.param(float("inf"), 0.1, r"Forecast\.mu", id="mu-inf"),
    ],
)
def test_forecast_rejects_degenerate_values(mu: float, sigma: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Forecast(mu=mu, sigma=sigma)


def test_forecast_is_frozen_and_coerces_to_native_float() -> None:
    forecast = Forecast(mu=np.float64(0.02), sigma=np.float64(0.01))
    assert type(forecast.mu) is float
    assert type(forecast.sigma) is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        forecast.mu = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Producer -> consumer round trip + one-source-of-truth
# --------------------------------------------------------------------------- #


def test_features_to_model_round_trip() -> None:
    # Full producer->consumer path: synthetic prices -> log_returns() ->
    # .to_frame() (named LOG_RETURN_COLUMN by the producer) -> EWMABaseline.
    # The bug class this guards lives in the relationship, not the endpoints.
    rng = np.random.default_rng(17)
    daily = 0.0003 + 0.008 * rng.standard_normal(400)
    prices = pd.Series(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(daily)])), index=_dates(401))
    features = {"SPY": log_returns(prices).to_frame()}
    assert_worldmodel_contract(EWMABaseline(), features)


def test_log_return_column_has_one_source_of_truth() -> None:
    # The consumer must reference THE producer's constant (imported, not
    # re-declared) so any future re-duplication fails CI.
    assert base.LOG_RETURN_COLUMN is calc.LOG_RETURN_COLUMN
