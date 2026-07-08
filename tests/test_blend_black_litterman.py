"""``bl_blend`` + Idzorek golden-value tests (plan Step 6). No mocks.

Golden fixtures are transcribed from Idzorek (2005), "A STEP-BY-STEP GUIDE TO
THE BLACK-LITTERMAN MODEL Incorporating user-specified confidence levels"
(workspace-level ``C:/Users/abero/dev/papers/Idzorek_onBL.pdf`` — not in this repo):

- ``IDZOREK_SIGMA``: Table 5, the 8-asset covariance matrix of excess returns.
- ``IDZOREK_PI``: Table 6 column 3, the Implied Equilibrium Return Vector
  (paper values, rounded to 1bp — the tolerance floor for everything below).
- Views + stated confidences: section 2.2 (View 1 absolute: Int'l Dev Equity
  excess return 5.25%, confidence 25%; Views 2-3 relative). P: Formula 7
  (market-capitalization method); Q = [5.25%, 0.25%, 2%].
- He-Litterman omega calibration: page 15, tau = 0.025, omega_k = tau *
  p_k Sigma p_k' -> diag(0.000709, 0.000141, 0.000866); view-portfolio
  variances: Table 4 (2.836%, 0.563%, 3.462%).
- GOLDEN PIN #1: Table 6 column 2, the New Combined Return Vector E[R] =
  (0.07%, 0.50%, 6.50%, 4.32%, 7.59%, 3.94%, 4.93%, 6.84%), reproduced
  through the pinned pypfopt 1.6.0 ``bl_returns()`` within 1e-4 (the paper's
  own 1bp rounding; observed max error ~6.4e-5).
- GOLDEN PIN #2: the Idzorek-omega confidence semantics on View 1 (the
  paper's only absolute view — the exact shape ``bl_blend`` emits). pypfopt
  implements Idzorek via Walters' closed form omega = tau*(1-c)/c * p Sigma
  p', under which the posterior tilt is EXACTLY c times the 100%-confidence
  tilt — Idzorek's Formulas 10/15 intent (tilt ~= C * 100%-tilt), exact for a
  single view. Pinned: omega_1(c=0.25) = 3 * 0.025 * 0.028355 = 0.002126625,
  the E[R]_100% view-asset return == Q (Formula 12 property), and the
  per-asset tilt ratio == 0.25. (Table 7's implied-confidence column — 32.94%
  for View 1 — belongs to the SCALED-VARIANCE omega of pin #1, not the
  Idzorek method, and is therefore not pinned here.)

The paper example needs 8 assets and relative views, while ``bl_blend`` is
deliberately locked to the 3-asset ``UNIVERSE`` with absolute views — so the
golden pins exercise the exact pypfopt code paths ``bl_blend`` calls into
(``bl_returns``/idzorek omega), and the remaining tests pin ``bl_blend``'s own
wiring (canonical ordering, the H-day -> annual boundary, raw-H-day
confidence, graceful no-view degradation) on the production universe.
"""

import importlib.metadata
import math

import numpy as np
import pandas as pd
import pytest

from abe import constants
from abe.blend.black_litterman import BLResult, bl_blend
from abe.blend.covariance import ledoit_wolf_sigma
from abe.calc import H_TO_ANNUAL, LOG_RETURN_COLUMN, idzorek_confidence
from abe.constants import DELTA, HORIZON_BARS, TAU, TRADING_DAYS, UNIVERSE, W_MKT
from abe.model.base import EWMABaseline, Forecast

# --------------------------------------------------------------------------- #
# Idzorek (2005) paper fixture — see module docstring for table provenance
# --------------------------------------------------------------------------- #

IDZOREK_ASSETS = [
    "US Bonds",
    "Intl Bonds",
    "US Large Growth",
    "US Large Value",
    "US Small Growth",
    "US Small Value",
    "Intl Dev Equity",
    "Intl Emerg Equity",
]

# Table 5 — covariance matrix of excess returns
IDZOREK_SIGMA = np.array(
    [
        [0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437],
        [0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535],
        [-0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039],
        [-0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854],
        [0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994],
        [0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235],
        [-0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064],
        [-0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.079958],
    ]
)

# Table 6 column 3 — Implied Equilibrium Return Vector (Pi)
IDZOREK_PI = np.array([0.0008, 0.0067, 0.0641, 0.0408, 0.0743, 0.0370, 0.0480, 0.0660])

# Section 2.2 views; P rows per Formula 7 (market-capitalization method)
IDZOREK_Q = np.array([0.0525, 0.0025, 0.02])
IDZOREK_P = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9, -0.9, 0.1, -0.1, 0.0, 0.0],
    ]
)
IDZOREK_TAU = 0.025

# Table 4 — variances of the view portfolios (p_k Sigma p_k')
TABLE4_VIEW_VARIANCES = np.array([0.02836, 0.00563, 0.03462])

# Page 15 — He-Litterman omega diagonal (tau * view variance)
PAGE15_OMEGA_DIAG = np.array([0.000709, 0.000141, 0.000866])

# Table 6 column 2 — the New Combined Return Vector E[R] (GOLDEN PIN #1)
TABLE6_POSTERIOR = np.array([0.0007, 0.0050, 0.0650, 0.0432, 0.0759, 0.0394, 0.0493, 0.0684])


def _idzorek_sigma_df() -> pd.DataFrame:
    return pd.DataFrame(IDZOREK_SIGMA, index=IDZOREK_ASSETS, columns=IDZOREK_ASSETS)


# --------------------------------------------------------------------------- #
# Pinned-dependency smoke
# --------------------------------------------------------------------------- #


def test_pyportfolioopt_pinned_version() -> None:
    import pypfopt

    assert importlib.metadata.version("pyportfolioopt") == "1.6.0"
    assert hasattr(pypfopt, "black_litterman")


# --------------------------------------------------------------------------- #
# GOLDEN PIN #1 — Idzorek Table 6 through pypfopt bl_returns()
# --------------------------------------------------------------------------- #


def test_idzorek_table6_new_combined_return_vector() -> None:
    """Reproduce Table 6's E[R] with the paper's own He-Litterman omega."""
    from pypfopt.black_litterman import BlackLittermanModel

    view_variances = np.array([p @ IDZOREK_SIGMA @ p for p in IDZOREK_P])
    # Table 4 pins (paper rounds to 0.001%):
    np.testing.assert_allclose(view_variances, TABLE4_VIEW_VARIANCES, atol=1e-5)
    omega = np.diag(IDZOREK_TAU * view_variances)
    # Page 15 omega pins:
    np.testing.assert_allclose(np.diag(omega), PAGE15_OMEGA_DIAG, atol=5e-7)

    bl = BlackLittermanModel(
        _idzorek_sigma_df(),
        pi=IDZOREK_PI,
        Q=IDZOREK_Q,
        P=IDZOREK_P,
        omega=omega,
        tau=IDZOREK_TAU,
    )
    posterior = bl.bl_returns().to_numpy()
    # 1e-4 = the paper's 1bp rounding of both Pi (input) and E[R] (output);
    # observed max error ~6.4e-5.
    np.testing.assert_allclose(posterior, TABLE6_POSTERIOR, atol=1e-4)


# --------------------------------------------------------------------------- #
# GOLDEN PIN #2 — Idzorek-omega confidence semantics on the paper's View 1
# --------------------------------------------------------------------------- #


def test_idzorek_omega_single_view_tilt_is_confidence_times_full_tilt() -> None:
    """View 1 (Int'l Dev Equity 5.25% absolute, c=25%) through omega='idzorek'.

    Pins: (a) the closed-form omega value; (b) Formula 12's property that the
    100%-confidence posterior hits Q exactly on the viewed asset; (c) the
    Formula 10/15 intent — posterior tilt == c * 100%-confidence tilt, exact
    for a single view under the closed form, on EVERY asset.
    """
    from pypfopt.black_litterman import BlackLittermanModel

    sigma_df = _idzorek_sigma_df()
    view = {"Intl Dev Equity": 0.0525}
    confidence = 0.25

    bl_c = BlackLittermanModel(
        sigma_df,
        pi=IDZOREK_PI,
        absolute_views=view,
        omega="idzorek",
        view_confidences=np.array([confidence]),
        tau=IDZOREK_TAU,
    )
    bl_full = BlackLittermanModel(
        sigma_df,
        pi=IDZOREK_PI,
        absolute_views=view,
        omega="idzorek",
        view_confidences=np.array([1.0]),
        tau=IDZOREK_TAU,
    )

    # (a) omega = tau * (1-c)/c * p Sigma p' = 0.025 * 3 * 0.028355
    assert float(bl_c.omega[0, 0]) == pytest.approx(0.002126625, rel=1e-9)
    assert float(bl_full.omega[0, 0]) == 0.0  # c=1 -> certainty -> omega=0

    tilt_c = bl_c.bl_returns().to_numpy() - IDZOREK_PI
    tilt_full = bl_full.bl_returns().to_numpy() - IDZOREK_PI

    # (b) 100% confidence drives the viewed asset's posterior exactly to Q
    viewed_idx = IDZOREK_ASSETS.index("Intl Dev Equity")
    assert IDZOREK_PI[viewed_idx] + tilt_full[viewed_idx] == pytest.approx(0.0525, abs=1e-12)

    # (c) tilt(c) == c * tilt(100%) on every asset the view touches at all
    np.testing.assert_allclose(tilt_c, confidence * tilt_full, rtol=1e-9, atol=1e-15)


# --------------------------------------------------------------------------- #
# 3-asset production-universe fixtures for bl_blend
# --------------------------------------------------------------------------- #

_VOLS = np.array([0.17, 0.16, 0.05])  # realistic annualized SPY/ACWI/AGG vols
_CORR = np.array(
    [
        [1.00, 0.95, -0.10],
        [0.95, 1.00, -0.05],
        [-0.10, -0.05, 1.00],
    ]
)


def _sigma3() -> pd.DataFrame:
    return pd.DataFrame(
        np.outer(_VOLS, _VOLS) * _CORR, index=list(UNIVERSE), columns=list(UNIVERSE)
    )


def _prior(sigma: pd.DataFrame) -> np.ndarray:
    w = np.array([W_MKT[asset] for asset in UNIVERSE])
    result: np.ndarray = DELTA * (sigma.to_numpy() @ w)
    return result


# --------------------------------------------------------------------------- #
# bl_blend — graceful no-view degradation
# --------------------------------------------------------------------------- #


def test_no_views_returns_market_prior_exactly() -> None:
    """Empty forecasts -> mu_post == pi (plan section 4's degradation path)."""
    sigma = _sigma3()
    result = bl_blend(sigma, {})
    assert isinstance(result, BLResult)
    np.testing.assert_allclose(result.mu_post.to_numpy(), _prior(sigma), rtol=0, atol=1e-14)
    assert list(result.mu_post.index) == list(UNIVERSE)
    # sigma_post is the K=0 limit of the He-Litterman posterior: (1+tau)*Sigma
    np.testing.assert_allclose(
        result.sigma_post.to_numpy(), (1.0 + TAU) * sigma.to_numpy(), rtol=0, atol=1e-14
    )
    assert result.diagnostics["Q"] == {}
    assert result.diagnostics["confidences"] == {}
    assert result.diagnostics["omega_diag"] == {}
    assert all(tilt == 0.0 for tilt in result.diagnostics["tilt"].values())


# --------------------------------------------------------------------------- #
# bl_blend — the H-day -> annual boundary
# --------------------------------------------------------------------------- #


def test_h_to_annual_factor_derived_from_shared_constants() -> None:
    assert H_TO_ANNUAL == TRADING_DAYS / HORIZON_BARS
    assert H_TO_ANNUAL == 12.0  # 252 / 21 — the literal factor for H=21
    assert constants.TRADING_DAYS is TRADING_DAYS
    assert constants.HORIZON_BARS is HORIZON_BARS


def test_q_is_h_day_mu_annualized() -> None:
    """mu = 0.01 H-day -> Q = 0.12 annualized (x 252/21 = 12) at THIS boundary."""
    result = bl_blend(_sigma3(), {"SPY": Forecast(mu=0.01, sigma=0.05)})
    assert result.diagnostics["Q"] == {"SPY": pytest.approx(0.12, abs=1e-15)}


def test_confidence_computed_from_raw_h_day_pair() -> None:
    """The documented DECISION: confidence uses (mu, sigma) as emitted, unscaled.

    Annualizing mu (x12) without touching sigma — or scaling sigma by sqrt(12)
    — would give a different value; the pin is the exact raw-pair map output.
    """
    mu_h, sigma_h = 0.01, 0.02
    result = bl_blend(_sigma3(), {"SPY": Forecast(mu=mu_h, sigma=sigma_h)})
    expected = idzorek_confidence(mu_h, sigma_h)
    assert result.diagnostics["confidences"]["SPY"] == expected
    # And it is genuinely different from the mis-scaled alternatives:
    assert expected != idzorek_confidence(mu_h * 12.0, sigma_h)
    assert expected != idzorek_confidence(mu_h * 12.0, sigma_h * math.sqrt(12.0))


# --------------------------------------------------------------------------- #
# bl_blend — confidence steers the posterior (both endpoints, both directions)
# --------------------------------------------------------------------------- #


def test_near_zero_confidence_view_stays_at_prior() -> None:
    sigma = _sigma3()
    pi = _prior(sigma)
    spy = list(UNIVERSE).index("SPY")
    # mu ~ 0 relative to sigma -> confidence clamps to the 0.02 floor
    result = bl_blend(sigma, {"SPY": Forecast(mu=1e-6, sigma=0.05)})
    assert result.diagnostics["confidences"]["SPY"] == 0.02
    q = result.diagnostics["Q"]["SPY"]
    # Single-view closed form: tilt on the viewed asset == c * (Q - pi_spy)
    expected_tilt = 0.02 * (q - pi[spy])
    assert result.diagnostics["tilt"]["SPY"] == pytest.approx(expected_tilt, rel=1e-9)
    # ...which is ~2% of the view's pull: the posterior is prior-dominated.
    assert abs(result.mu_post.iloc[spy] - pi[spy]) <= 0.021 * abs(q - pi[spy])


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_high_confidence_absurd_view_moves_posterior_materially(direction: float) -> None:
    sigma = _sigma3()
    pi = _prior(sigma)
    spy = list(UNIVERSE).index("SPY")
    # |mu/sigma| = 10 -> ceiling confidence 0.95; Q = +/-0.60 annualized (absurd)
    result = bl_blend(sigma, {"SPY": Forecast(mu=direction * 0.05, sigma=0.005)})
    assert result.diagnostics["confidences"]["SPY"] == 0.95
    q = result.diagnostics["Q"]["SPY"]
    assert q == pytest.approx(direction * 0.60, abs=1e-15)
    tilt = float(result.mu_post.iloc[spy] - pi[spy])
    # Exact single-view property: tilt == 0.95 * (Q - pi)
    assert tilt == pytest.approx(0.95 * (q - pi[spy]), rel=1e-9)
    # Material move toward Q, in the view's direction
    assert math.copysign(1.0, tilt) == math.copysign(1.0, q - pi[spy])
    assert abs(tilt) > 0.5 * abs(q - pi[spy])


# --------------------------------------------------------------------------- #
# bl_blend — multi-view shape, ordering, diagnostics
# --------------------------------------------------------------------------- #


def test_multi_view_result_shape_and_diagnostics() -> None:
    sigma = _sigma3()
    forecasts = {
        # Deliberately inserted in NON-canonical order: output must be UNIVERSE order
        "AGG": Forecast(mu=0.002, sigma=0.01),
        "SPY": Forecast(mu=0.01, sigma=0.03),
        "ACWI": Forecast(mu=-0.005, sigma=0.04),
    }
    result = bl_blend(sigma, forecasts)
    assert list(result.mu_post.index) == list(UNIVERSE)
    assert list(result.sigma_post.index) == list(UNIVERSE)
    assert list(result.sigma_post.columns) == list(UNIVERSE)
    assert np.isfinite(result.mu_post.to_numpy()).all()
    # Views serialized in canonical UNIVERSE order
    assert list(result.diagnostics["Q"]) == list(UNIVERSE)
    assert list(result.diagnostics["omega_diag"]) == list(UNIVERSE)
    # Posterior covariance: symmetric, PD, and inflated vs the prior Sigma
    post = result.sigma_post.to_numpy()
    assert np.array_equal(post, post.T)
    assert float(np.linalg.eigvalsh(post).min()) > 0.0
    assert np.trace(post) > np.trace(sigma.to_numpy())
    # Diagnostics are JSON-friendly native floats, tilt == mu_post - pi
    for section in result.diagnostics.values():
        for value in section.values():
            assert isinstance(value, float)
    for i, asset in enumerate(UNIVERSE):
        expected = float(result.mu_post.iloc[i] - result.diagnostics["pi"][asset])
        assert result.diagnostics["tilt"][asset] == pytest.approx(expected, abs=1e-15)
    # All confidences respect the clamp
    assert all(0.02 <= c <= 0.95 for c in result.diagnostics["confidences"].values())


# --------------------------------------------------------------------------- #
# bl_blend — producer -> consumer round trip (features -> model -> LW -> BL)
# --------------------------------------------------------------------------- #


def test_round_trip_ledoit_wolf_and_ewma_forecasts_into_bl_blend() -> None:
    """The full Step-6 relationship: real LW Sigma + real EWMA Forecasts blend."""
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.Index(pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"), name="date")
    factor = 0.01 * rng.standard_normal(n)
    returns = pd.DataFrame(
        {
            "SPY": 0.0004 + factor + 0.002 * rng.standard_normal(n),
            "ACWI": 0.0003 + 0.95 * factor + 0.002 * rng.standard_normal(n),
            "AGG": 0.0001 + 0.003 * rng.standard_normal(n),
        },
        index=dates,
    )[list(UNIVERSE)]

    sigma = ledoit_wolf_sigma(returns)
    features = {asset: returns[asset].rename(LOG_RETURN_COLUMN).to_frame() for asset in UNIVERSE}
    forecasts = EWMABaseline().forecast(features)
    result = bl_blend(sigma, forecasts)

    assert list(result.mu_post.index) == list(UNIVERSE)
    assert np.isfinite(result.mu_post.to_numpy()).all()
    assert float(np.linalg.eigvalsh(result.sigma_post.to_numpy()).min()) > 0.0
    # Finite-and-not-absurd sanity bound only. This is NOT a unit tripwire:
    # an unconverted H-day mu (~0.03) or a daily Sigma (~0.07) stays inside
    # this band — those regressions are caught by test_q_is_h_day_mu_annualized
    # and test_annualized_equity_vol_in_realistic_band respectively.
    assert (np.abs(result.mu_post.to_numpy()) < 1.5).all()
    assert all(0.02 <= c <= 0.95 for c in result.diagnostics["confidences"].values())


# --------------------------------------------------------------------------- #
# bl_blend — fail-loud validation
# --------------------------------------------------------------------------- #


def test_sigma_in_wrong_asset_order_rejected() -> None:
    wrong_order = ["ACWI", "SPY", "AGG"]
    sigma = _sigma3()
    shuffled = sigma.loc[wrong_order, wrong_order]
    with pytest.raises(ValueError, match="UNIVERSE order"):
        bl_blend(shuffled, {})


def test_sigma_with_wrong_assets_rejected() -> None:
    bad = pd.DataFrame(
        np.eye(3) * 0.04, index=["SPY", "ACWI", "TLT"], columns=["SPY", "ACWI", "TLT"]
    )
    with pytest.raises(ValueError, match="UNIVERSE order"):
        bl_blend(bad, {})


def test_sigma_with_nan_rejected() -> None:
    sigma = _sigma3()
    sigma.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bl_blend(sigma, {})


def test_asymmetric_sigma_rejected() -> None:
    sigma = _sigma3()
    sigma.iloc[0, 1] += 1e-3
    with pytest.raises(ValueError, match="symmetric"):
        bl_blend(sigma, {})


def test_forecast_for_unknown_asset_rejected() -> None:
    with pytest.raises(ValueError, match="TLT"):
        bl_blend(_sigma3(), {"TLT": Forecast(mu=0.01, sigma=0.05)})


def test_w_mkt_with_missing_asset_rejected() -> None:
    with pytest.raises(ValueError, match="w_mkt keys"):
        bl_blend(_sigma3(), {}, w_mkt={"SPY": 0.5, "ACWI": 0.5})


def test_w_mkt_not_summing_to_one_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        bl_blend(_sigma3(), {}, w_mkt={"SPY": 0.3, "ACWI": 0.3, "AGG": 0.3})


@pytest.mark.parametrize("delta", [0.0, -2.5, float("nan")])
def test_bad_delta_rejected(delta: float) -> None:
    with pytest.raises(ValueError, match="delta"):
        bl_blend(_sigma3(), {}, delta=delta)


@pytest.mark.parametrize("tau", [0.0, -0.05, 1.5, float("nan")])
def test_bad_tau_rejected(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        bl_blend(_sigma3(), {}, tau=tau)


@pytest.mark.parametrize("rf", [float("nan"), 0.045, -0.01])
def test_nonzero_or_non_finite_rf_rejected(rf: float) -> None:
    # V1 excess-return convention: nonzero rf would mix a total-return pi
    # with excess-return Q (every view residual biased by -rf).
    with pytest.raises(ValueError, match="rf"):
        bl_blend(_sigma3(), {}, rf=rf)


def test_default_w_mkt_is_the_shared_constant() -> None:
    # One-source-of-truth guard: the default prior weights ARE constants.W_MKT
    # (the no-view anchor test above then certifies the plan section 8 vector).
    assert bl_blend.__defaults__ is not None
    assert bl_blend.__defaults__[0] is W_MKT
    assert constants.W_MKT is W_MKT
