"""Minimum-variance optimizer tests (Track 2 Step 23).

Done-when (plan §7 Step 23): the alternative optimizer produces a valid long-only
Σ=1 allocation and — being variance-only — holds the low-vol asset (AGG), the
V1 AGG=0% realism fix.
"""

import numpy as np
import pandas as pd
import pytest

from abe.blend.covariance import ledoit_wolf_sigma
from abe.constants import UNIVERSE, W_MAX
from abe.optimize.min_variance import min_variance_weights


def _low_agg_vol_sigma() -> pd.DataFrame:
    dates = pd.Index(
        [stamp.strftime("%Y-%m-%d") for stamp in pd.bdate_range("2026-01-01", periods=90)],
        name="date",
    )
    rng = np.random.default_rng(5)
    vols = {"SPY": 0.012, "ACWI": 0.013, "AGG": 0.003}  # AGG is the low-vol asset
    cols = {asset: rng.normal(0.0, vols[asset], size=90) for asset in UNIVERSE}
    return ledoit_wolf_sigma(pd.DataFrame(cols, index=dates)[list(UNIVERSE)])


def test_min_variance_valid_and_overweights_low_vol_agg() -> None:
    result = min_variance_weights(_low_agg_vol_sigma())
    assert set(result.weights) == set(UNIVERSE)
    assert all(0.0 <= w <= W_MAX + 1e-6 for w in result.weights.values())
    assert sum(result.weights.values()) == pytest.approx(1.0)
    # variance-only → the lowest-vol asset dominates (AGG > equities), NOT 0%
    assert result.weights["AGG"] > result.weights["SPY"]
    assert result.weights["AGG"] > result.weights["ACWI"]
    assert result.weights["AGG"] > 0.0


def test_min_variance_is_deterministic_and_ignores_mu() -> None:
    sigma = _low_agg_vol_sigma()
    # takes only sigma — no mu path exists, so repeated solves match exactly
    assert min_variance_weights(sigma).weights == min_variance_weights(sigma).weights


def test_min_variance_min_weight_floor() -> None:
    result = min_variance_weights(_low_agg_vol_sigma(), min_weight=0.1)
    for asset in UNIVERSE:
        assert result.weights[asset] >= 0.1 - 1e-9
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_min_variance_reports_turnover_vs_w_prev() -> None:
    sigma = _low_agg_vol_sigma()
    w_prev = {"SPY": 0.33, "ACWI": 0.33, "AGG": 0.34}
    result = min_variance_weights(sigma, w_prev)
    assert result.prev_weights == w_prev
    assert result.relaxed_turnover is False  # no turnover term to relax
    for asset in UNIVERSE:
        assert result.turnover[asset] == pytest.approx(
            abs(result.weights[asset] - w_prev[asset]), abs=1e-12
        )


def test_min_variance_infeasible_min_weight_raises() -> None:
    with pytest.raises(ValueError, match="min_weight"):
        min_variance_weights(_low_agg_vol_sigma(), min_weight=0.5)  # 0.5 * 3 > 1
