"""Ledoit-Wolf covariance tests (plan Step 6). No mocks; synthetic frames only.

The load-bearing checks: PSD on the near-singular SPY/ACWI-like frame this
universe actually produces, the annualized-vol units band (plan section 9's
producer/consumer unit-drift risk), and loud NaN/short/disorder rejection.
"""

import numpy as np
import pandas as pd
import pytest

from abe import constants
from abe.blend import covariance
from abe.blend.covariance import MIN_LW_ROWS, PSD_EIGENVALUE_TOL, ledoit_wolf_sigma

# --------------------------------------------------------------------------- #
# Synthetic daily log-return frames (deterministic; ISO-date-string index)
# --------------------------------------------------------------------------- #


def _dates(n: int) -> pd.Index:
    return pd.Index(pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"), name="date")


def _near_singular_frame(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Three assets, two of them ~0.95+ correlated (the SPY-in-ACWI shape).

    SPY and ACWI share a common factor with only a sliver of idiosyncratic
    noise; AGG is independent low-vol. Daily stds ~1% (equity) / ~0.3% (bonds)
    so annualized vols land near 0.16 / 0.05 — realistic units.
    """
    rng = np.random.default_rng(seed)
    factor = 0.01 * rng.standard_normal(n)
    spy = factor + 0.002 * rng.standard_normal(n)
    acwi = 0.95 * factor + 0.002 * rng.standard_normal(n)
    agg = 0.003 * rng.standard_normal(n)
    return pd.DataFrame({"SPY": spy, "ACWI": acwi, "AGG": agg}, index=_dates(n))


# --------------------------------------------------------------------------- #
# PSD + labels + units
# --------------------------------------------------------------------------- #


def test_psd_on_near_singular_frame() -> None:
    frame = _near_singular_frame()
    assert float(frame[["SPY", "ACWI"]].corr().iloc[0, 1]) > 0.9  # the frame IS near-singular
    sigma = ledoit_wolf_sigma(frame)
    eigenvalues = np.linalg.eigvalsh(sigma.to_numpy())
    assert float(eigenvalues.min()) > 0.0  # strictly PD after shrinkage
    assert float(eigenvalues.min()) > PSD_EIGENVALUE_TOL


def test_result_is_exactly_symmetric() -> None:
    sigma = ledoit_wolf_sigma(_near_singular_frame()).to_numpy()
    assert np.array_equal(sigma, sigma.T)


def test_labels_preserved_in_input_order() -> None:
    frame = _near_singular_frame()
    sigma = ledoit_wolf_sigma(frame)
    assert list(sigma.index) == ["SPY", "ACWI", "AGG"]
    assert list(sigma.columns) == ["SPY", "ACWI", "AGG"]


def test_annualized_equity_vol_in_realistic_band() -> None:
    # ~1%/day equity std -> ~0.16 annualized; the 0.05-0.60 band is the plan
    # section 9 unit-drift tripwire (daily-not-annualized Sigma would land
    # near 0.01; a variance-vs-std mixup near 0.026).
    sigma = ledoit_wolf_sigma(_near_singular_frame())
    spy_vol = float(np.sqrt(sigma.loc["SPY", "SPY"]))
    assert 0.05 < spy_vol < 0.60
    assert spy_vol == pytest.approx(0.01 * np.sqrt(constants.TRADING_DAYS), rel=0.35)


def test_annualization_uses_the_shared_constant() -> None:
    # One-source-of-truth guard (code-quality rule): the module must consume
    # constants.TRADING_DAYS itself, not a re-declared 252.
    assert covariance.TRADING_DAYS is constants.TRADING_DAYS


def test_covariance_sign_structure_survives_shrinkage() -> None:
    sigma = ledoit_wolf_sigma(_near_singular_frame())
    assert float(sigma.loc["SPY", "ACWI"]) > 0.0  # the 0.95-correlated pair stays coupled
    assert float(sigma.loc["SPY", "ACWI"]) < float(sigma.loc["SPY", "SPY"])


# --------------------------------------------------------------------------- #
# Loud rejection
# --------------------------------------------------------------------------- #


def test_nan_rejected_loudly_with_location() -> None:
    frame = _near_singular_frame(n=120)
    frame.iloc[5, frame.columns.get_loc("ACWI")] = np.nan
    with pytest.raises(ValueError, match="ACWI") as excinfo:
        ledoit_wolf_sigma(frame)
    assert str(frame.index[5]) in str(excinfo.value)  # the bad row's date is named


def test_short_input_rejected_at_boundary() -> None:
    ok = _near_singular_frame(n=MIN_LW_ROWS)
    ledoit_wolf_sigma(ok)  # exactly MIN_LW_ROWS rows is accepted
    with pytest.raises(ValueError, match="MIN_LW_ROWS"):
        ledoit_wolf_sigma(_near_singular_frame(n=MIN_LW_ROWS - 1))


def test_non_ascending_index_rejected() -> None:
    frame = _near_singular_frame(n=120).iloc[::-1]
    with pytest.raises(ValueError, match="ascending"):
        ledoit_wolf_sigma(frame)


def test_empty_frame_rejected() -> None:
    with pytest.raises(ValueError, match="no asset columns"):
        ledoit_wolf_sigma(pd.DataFrame(index=_dates(100)))
