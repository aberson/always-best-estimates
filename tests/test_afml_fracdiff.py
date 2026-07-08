"""afml.fracdiff tests: FFD weights, fixed-width transform, min-d ADF search.

Synthetic seeded series only — no network, no db. The plan section 12 REQUIRED
garbage anchors live here: white noise (already stationary) -> min_d_search
picks d ~ 0; a pure random walk needs substantial differencing -> d high, and
the returned series passes ADF. ADF results are cross-checked with a direct
``statsmodels.adfuller`` call, not by trusting the stored p-value.
"""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from abe.afml.fracdiff import (
    DEFAULT_THRESHOLD,
    FRACDIFF_COLUMN,
    MIN_ADF_SAMPLES,
    FracDiffParams,
    ffd_weights,
    fracdiff_series,
    min_d_search,
)
from abe.calc import log_returns


def _dates(n: int) -> pd.Index:
    return pd.Index(pd.bdate_range("2000-01-03", periods=n).strftime("%Y-%m-%d"), name="date")


def _series(values: np.ndarray) -> pd.Series:
    array = np.asarray(values, dtype=float)
    return pd.Series(array, index=_dates(len(array)))


# --------------------------------------------------------------------------- #
# ffd_weights
# --------------------------------------------------------------------------- #


def test_ffd_weights_d_one_is_first_difference() -> None:
    # d=1 truncates to exactly [1, -1] (w_2 = 0 < threshold) — the first-diff op.
    np.testing.assert_allclose(ffd_weights(1.0), np.array([1.0, -1.0]))


def test_ffd_weights_d_zero_is_identity() -> None:
    # d=0 truncates to [1] (w_1 = 0) — the identity transform.
    np.testing.assert_allclose(ffd_weights(0.0), np.array([1.0]))


def test_ffd_weights_recurrence_hand_computed() -> None:
    # w_0=1; w_1=-d; w_2=-w_1*(d-1)/2; hand-checked for d=0.4.
    weights = ffd_weights(0.4)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(-0.4)
    assert weights[2] == pytest.approx(0.4 * (0.4 - 1.0) / 2.0)  # -w_1*(d-1)/2 = -0.12


def test_ffd_weights_fractional_sign_and_decay() -> None:
    # For 0<d<1 only w_0 is positive; every later weight is negative (the
    # (-1)^k*C(d,k) generalized-binomial terms), with strictly decreasing
    # magnitude.
    weights = ffd_weights(0.4)
    assert weights[0] > 0.0
    assert np.all(weights[1:] < 0.0)
    magnitudes = np.abs(weights)
    assert np.all(np.diff(magnitudes) < 0.0)  # strictly decaying


def test_ffd_weights_threshold_controls_length() -> None:
    # A looser threshold truncates sooner -> a shorter window.
    assert len(ffd_weights(0.4, threshold=1e-3)) < len(ffd_weights(0.4, threshold=1e-6))


def test_ffd_weights_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be finite and > 0"):
        ffd_weights(0.4, threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be finite and > 0"):
        ffd_weights(0.4, threshold=-1e-5)


# --------------------------------------------------------------------------- #
# fracdiff_series
# --------------------------------------------------------------------------- #


def test_fracdiff_d_one_recovers_log_returns() -> None:
    # FFD at d=1 of a log-price series is its first difference == the log-return.
    rng = np.random.default_rng(4)
    prices = _series(100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(200))))
    log_price = pd.Series(np.log(prices.to_numpy()), index=prices.index)
    fd = fracdiff_series(log_price, 1.0)
    returns = log_returns(prices)
    np.testing.assert_allclose(fd.to_numpy(), returns.to_numpy(), atol=1e-12)
    assert list(fd.index) == list(returns.index)


def test_fracdiff_warmup_dropped_and_named() -> None:
    # d=0.9 -> a ~125-term window (small enough for a short synthetic series).
    series = _series(np.linspace(1.0, 5.0, 300))
    fd = fracdiff_series(series, 0.9)
    window = len(ffd_weights(0.9))
    assert fd.name == FRACDIFF_COLUMN
    assert len(fd) == len(series) - window + 1  # warm-up dropped, not NaN
    assert list(fd.index) == list(series.index[window - 1 :])
    assert not fd.isna().any()
    assert np.isfinite(fd.to_numpy()).all()


def test_fracdiff_deterministic() -> None:
    rng = np.random.default_rng(9)
    series = _series(np.cumsum(rng.standard_normal(400)))
    assert fracdiff_series(series, 0.9).equals(fracdiff_series(series, 0.9))


def test_fracdiff_rejects_disordered_index() -> None:
    dates = _dates(300)
    shuffled = pd.Index([dates[1], dates[0], *dates[2:]], name="date")
    series = pd.Series(np.linspace(1.0, 3.0, 300), index=shuffled)
    with pytest.raises(ValueError, match="ascending"):
        fracdiff_series(series, 0.4)


def test_fracdiff_rejects_non_finite() -> None:
    values = np.linspace(1.0, 3.0, 300)
    values[10] = np.nan
    with pytest.raises(ValueError, match="NaN/non-finite"):
        fracdiff_series(_series(values), 0.4)


def test_fracdiff_rejects_series_shorter_than_window() -> None:
    # d=0.3 needs a ~2000-term window; a 10-point series is far too short.
    with pytest.raises(ValueError, match="shorter than the FFD window"):
        fracdiff_series(_series(np.linspace(1.0, 2.0, 10)), 0.3)


# --------------------------------------------------------------------------- #
# min_d_search — garbage anchors (plan section 12 REQUIRED)
# --------------------------------------------------------------------------- #


def test_min_d_search_white_noise_picks_d_near_zero() -> None:
    # White noise is already stationary -> the smallest grid d (0) passes ADF.
    rng = np.random.default_rng(0)
    noise = _series(rng.standard_normal(2000))
    params = min_d_search(noise)
    assert params.d <= 0.1  # d ~ 0 (plan anchor: noise -> d ~ 0)
    assert params.adf_pvalue_achieved < 0.05
    assert params.corr_with_original == pytest.approx(1.0)  # d=0 is the identity
    assert params.window_len == len(ffd_weights(params.d, params.threshold))


def test_min_d_search_random_walk_needs_substantial_differencing() -> None:
    # A pure random walk needs real differencing. At threshold 1e-5 every grid d
    # in (0, 0.35) has a window wider than 2000 samples (untestable, skipped),
    # and d=0 (the raw walk) is non-stationary -> the search must land at d>=0.35
    # AND the returned series must genuinely pass ADF. (Full d=1 recovers returns.)
    rng = np.random.default_rng(1)
    walk = _series(np.cumsum(rng.standard_normal(2000)))

    # Anchor half 1: the raw walk (FFD at d=0) genuinely fails ADF — the search
    # cannot trivially return 0 for non-stationary input.
    raw = fracdiff_series(walk, 0.0)
    assert float(adfuller(raw.to_numpy(dtype=float))[1]) >= 0.05

    params = min_d_search(walk)
    assert params.d >= 0.35  # RW needs substantial differencing
    assert params.adf_pvalue_achieved < 0.05
    assert 0.0 < params.corr_with_original < 1.0  # retains some, not all, memory

    # Independently re-verify the chosen series is stationary (don't trust the
    # stored p-value): recompute the FFD and run adfuller directly.
    reconstructed = fracdiff_series(walk, params.d, params.threshold)
    assert float(adfuller(reconstructed.to_numpy(dtype=float))[1]) < 0.05


def test_min_d_search_respects_custom_grid_and_returns_smallest() -> None:
    rng = np.random.default_rng(2)
    noise = _series(rng.standard_normal(1500))
    params = min_d_search(noise, d_grid=np.array([0.0, 0.5, 1.0]))
    assert params.d == 0.0  # smallest passing d on the supplied grid


def test_min_d_search_raises_when_series_too_short() -> None:
    # Below MIN_ADF_SAMPLES no grid d is even testable (even d=1's window leaves
    # too few points) -> an explicit error, not a silent empty result.
    rng = np.random.default_rng(3)
    short = _series(rng.standard_normal(MIN_ADF_SAMPLES - 10))
    with pytest.raises(ValueError, match="too short"):
        min_d_search(short)


def test_min_d_search_raises_when_no_d_passes() -> None:
    # An I(2) series (double-integrated noise) is still non-stationary after one
    # full difference (d=1 leaves a random walk), so no grid d reaches p<0.05.
    rng = np.random.default_rng(5)
    i2 = _series(np.cumsum(np.cumsum(rng.standard_normal(2000))))
    with pytest.raises(ValueError, match="no testable d"):
        min_d_search(i2)


def test_min_d_search_rejects_disordered_index() -> None:
    dates = _dates(2000)
    shuffled = pd.Index([dates[1], dates[0], *dates[2:]], name="date")
    rng = np.random.default_rng(6)
    series = pd.Series(rng.standard_normal(2000), index=shuffled)
    with pytest.raises(ValueError, match="ascending"):
        min_d_search(series)


# --------------------------------------------------------------------------- #
# FracDiffParams
# --------------------------------------------------------------------------- #


def test_fracdiff_params_is_frozen() -> None:
    params = FracDiffParams(
        d=0.4,
        threshold=DEFAULT_THRESHOLD,
        window_len=100,
        adf_pvalue_achieved=0.01,
        corr_with_original=0.8,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.d = 0.5  # type: ignore[misc]


def test_default_threshold_is_pinned() -> None:
    assert DEFAULT_THRESHOLD == 1e-5
    assert math.isclose(DEFAULT_THRESHOLD, 1e-5)
