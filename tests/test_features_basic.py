"""features.basic tests: hand-computed log-returns + realized vol; no mocks.

Synthetic pandas series only, indexed by ISO date strings (the CacheAdapter
shape). The realized-vol expectations are cross-checked against
``statistics.stdev`` — an independent implementation, not pandas re-asserting
itself.
"""

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from abe import constants
from abe.features.basic import (
    LOG_RETURN_COLUMN,
    REALIZED_VOL_COLUMN,
    log_returns,
    non_finite_summary,
    realized_vol,
)


def _dates(n: int) -> pd.Index:
    return pd.Index(pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"), name="date")


def _series(values: list[float] | np.ndarray) -> pd.Series:
    array = np.asarray(values, dtype=float)
    return pd.Series(array, index=_dates(len(array)))


# --------------------------------------------------------------------------- #
# log_returns
# --------------------------------------------------------------------------- #


def test_log_returns_hand_computed() -> None:
    prices = _series([100.0, 105.0, 99.75])
    returns = log_returns(prices)
    assert returns.name == LOG_RETURN_COLUMN
    assert list(returns.index) == list(prices.index[1:])
    assert returns.iloc[0] == pytest.approx(math.log(105.0 / 100.0))
    assert returns.iloc[1] == pytest.approx(math.log(99.75 / 105.0))


def test_log_returns_recovers_synthetic_returns() -> None:
    rng = np.random.default_rng(3)
    r = 0.0003 + 0.01 * rng.standard_normal(250)
    prices = _series(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)])))
    returns = log_returns(prices)
    np.testing.assert_allclose(returns.to_numpy(), r, rtol=0, atol=1e-10)


def test_log_returns_drops_first_row_and_has_no_nan() -> None:
    prices = _series([100.0, 101.0, 102.0, 101.5])
    returns = log_returns(prices)
    assert len(returns) == len(prices) - 1  # first value dropped, not NaN
    assert not returns.isna().any()
    assert np.isfinite(returns.to_numpy()).all()


@pytest.mark.parametrize(
    ("values", "match"),
    [
        pytest.param([100.0], "at least 2", id="too-short"),
        pytest.param([100.0, float("nan"), 101.0], "NaN/non-finite", id="nan"),
        pytest.param([100.0, float("inf")], "NaN/non-finite", id="inf"),
        pytest.param([100.0, 0.0], "strictly positive", id="zero-price"),
        pytest.param([100.0, -5.0], "strictly positive", id="negative-price"),
    ],
)
def test_log_returns_rejects_bad_input(values: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        log_returns(_series(values))


def test_log_returns_deterministic() -> None:
    prices = _series([100.0, 101.0, 99.0, 103.0, 102.5])
    assert log_returns(prices).equals(log_returns(prices))  # bitwise-identical


def test_disordered_index_rejected_by_both_functions() -> None:
    # Both functions are order-dependent: a disordered index must fail loud,
    # not compute silently wrong values.
    dates = _dates(10)
    shuffled = pd.Index([dates[1], dates[0], *dates[2:]], name="date")
    with pytest.raises(ValueError, match="ascending"):
        log_returns(pd.Series(np.linspace(100.0, 110.0, 10), index=shuffled))
    with pytest.raises(ValueError, match="ascending"):
        realized_vol(pd.Series(np.full(10, 0.01), index=shuffled), window=3)


def test_nan_rejection_names_offending_rows() -> None:
    # The error must NAME the bad rows (count + date labels) so the operator
    # never has to hand-query SQLite to find a NULL adj_close.
    prices = _series([100.0, float("nan"), 101.0])
    with pytest.raises(ValueError, match="2020-01-02") as exc_info:
        log_returns(prices)
    assert "1 NaN/non-finite value(s)" in str(exc_info.value)


def test_non_finite_summary_counts_and_truncates() -> None:
    index = _dates(8)
    values = np.array([1.0, np.nan, np.inf, np.nan, np.nan, np.nan, np.nan, 1.0])
    summary = non_finite_summary(index, values)
    assert summary.startswith("6 NaN/non-finite value(s)")
    assert index[1] in summary  # first bad label named
    assert "+1 more" in summary  # only the first 5 labels are spelled out


def test_feature_column_names_are_pinned_literals() -> None:
    # Literal pin: consumers store/join on these names; a rename is a schema
    # change, not a refactor — it must fail CI loudly.
    assert LOG_RETURN_COLUMN == "log_return"
    assert REALIZED_VOL_COLUMN == "realized_vol"


# --------------------------------------------------------------------------- #
# realized_vol
# --------------------------------------------------------------------------- #


def test_realized_vol_alternating_hand_computed() -> None:
    # Alternating +/-1% daily returns, window=2: every window is (+0.01, -0.01)
    # up to sign, so every output value is the SAME hand-computable constant.
    returns = _series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    vol = realized_vol(returns, window=2)
    expected = statistics.stdev([0.01, -0.01]) * math.sqrt(constants.TRADING_DAYS)
    assert expected == pytest.approx(0.01 * math.sqrt(2) * math.sqrt(constants.TRADING_DAYS))
    assert vol.name == REALIZED_VOL_COLUMN
    assert len(vol) == len(returns) - 1
    assert list(vol.index) == list(returns.index[1:])
    for value in vol.to_numpy():
        assert value == pytest.approx(expected)


def test_realized_vol_matches_independent_stdev() -> None:
    rng = np.random.default_rng(11)
    values = 0.01 * rng.standard_normal(30)
    window = 5
    vol = realized_vol(_series(values), window=window)
    assert len(vol) == len(values) - window + 1
    for position, got in enumerate(vol.to_numpy()):
        expected = statistics.stdev(values[position : position + window]) * math.sqrt(
            constants.TRADING_DAYS
        )
        assert got == pytest.approx(expected, rel=1e-12)


def test_realized_vol_constant_returns_is_exactly_zero() -> None:
    # Documented behavior: a constant-return window has zero sample std, so
    # vol is exactly 0.0 — values are >= 0, never NaN, never negative.
    vol = realized_vol(_series([0.004] * 30), window=constants.HORIZON_BARS)
    assert (vol.to_numpy() == 0.0).all()
    assert (vol.to_numpy() >= 0.0).all()


def test_realized_vol_default_window_is_horizon_bars() -> None:
    rng = np.random.default_rng(5)
    returns = _series(0.01 * rng.standard_normal(30))
    vol = realized_vol(returns)  # default window
    assert len(vol) == len(returns) - constants.HORIZON_BARS + 1
    assert list(vol.index) == list(returns.index[constants.HORIZON_BARS - 1 :])


def test_realized_vol_drops_warmup_and_has_no_nan() -> None:
    rng = np.random.default_rng(7)
    returns = _series(0.01 * rng.standard_normal(40))
    vol = realized_vol(returns, window=10)
    assert not vol.isna().any()  # warm-up rows dropped, not NaN
    assert list(vol.index) == list(returns.index[9:])


@pytest.mark.parametrize(
    ("values", "window", "match"),
    [
        pytest.param([0.01] * 10, 1, "window must be >= 2", id="window-1"),
        pytest.param([0.01] * 10, 0, "window must be >= 2", id="window-0"),
        pytest.param([0.01] * 4, 5, "at least window=5", id="too-short"),
        pytest.param([0.01, float("nan"), 0.01], 2, "NaN/non-finite", id="nan"),
        pytest.param([0.01, float("inf"), 0.01], 2, "NaN/non-finite", id="inf"),
    ],
)
def test_realized_vol_rejects_bad_input(values: list[float], window: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        realized_vol(_series(values), window=window)


def test_realized_vol_deterministic() -> None:
    rng = np.random.default_rng(13)
    returns = _series(0.01 * rng.standard_normal(25))
    assert realized_vol(returns, window=4).equals(realized_vol(returns, window=4))
