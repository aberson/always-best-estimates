"""abe.calc tests — the transparent calculation home (Track 1 relocation).

Ported verbatim from the pre-relocation ``test_features_basic.py`` (hand-computed
log-returns + realized vol) and ``test_blend_confidence.py`` (the sigma ->
Idzorek confidence map), now targeting ``abe.calc``; behavior is byte-identical,
so the assertions are unchanged. Plus coverage for the two new annualization
helpers, the constant ``is``-identity (producer <-> consumer share ONE object),
and the :data:`abe.calc.EXPLANATIONS` registry completeness.

No mocks; synthetic pandas series only. The realized-vol expectations are
cross-checked against ``statistics.stdev`` — an independent implementation, not
pandas re-asserting itself. The clamp endpoints (0.02 / 0.95) are LITERAL pins
(plan-section-4 contract values, not derived numbers).
"""

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from abe import calc, constants
from abe.calc import (
    CONFIDENCE_CEIL,
    CONFIDENCE_FLOOR,
    EXPLANATIONS,
    H_TO_ANNUAL,
    LOG_RETURN_COLUMN,
    REALIZED_VOL_COLUMN,
    annualize_mean,
    annualize_vol,
    idzorek_confidence,
    log_returns,
    non_finite_summary,
    realized_vol,
)
from abe.features import build
from abe.model import base


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


def test_log_returns_docstring_worked_example() -> None:
    # The docstring's worked example: 100 -> 101 => ln(1.01) = +0.995%.
    returns = log_returns(_series([100.0, 101.0]))
    assert returns.iloc[0] == pytest.approx(math.log(1.01))
    assert returns.iloc[0] == pytest.approx(0.00995033, abs=1e-8)


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


def test_feature_column_constants_have_one_source_of_truth() -> None:
    # is-identity: a production consumer (features.build) references THE
    # abe.calc object, not a re-declared copy — any future re-duplication
    # (== but not is) fails here (code-quality one-source-of-truth rule).
    assert build.LOG_RETURN_COLUMN is calc.LOG_RETURN_COLUMN
    assert build.REALIZED_VOL_COLUMN is calc.REALIZED_VOL_COLUMN
    assert base.LOG_RETURN_COLUMN is calc.LOG_RETURN_COLUMN


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


# --------------------------------------------------------------------------- #
# annualize_mean / annualize_vol (new Track 1 helpers)
# --------------------------------------------------------------------------- #


def test_h_to_annual_is_derived_and_equals_twelve() -> None:
    assert H_TO_ANNUAL == constants.TRADING_DAYS / constants.HORIZON_BARS
    assert H_TO_ANNUAL == 12.0  # 252 / 21 for H=21


def test_annualize_mean_worked_example() -> None:
    # Docstring worked example: mu_H = +1% over 21 days => +12%/yr (x 252/21).
    assert annualize_mean(0.01) == pytest.approx(0.12, abs=1e-15)
    # And it IS mu * H_TO_ANNUAL (the byte-identical relocation of bl_blend's
    # old `mu * H_TO_ANNUAL` view derivation — one source of truth for the scale).
    assert annualize_mean(0.01) == 0.01 * H_TO_ANNUAL
    assert annualize_mean(0.0) == 0.0
    assert annualize_mean(-0.02) == pytest.approx(-0.24, abs=1e-15)


def test_annualize_vol_worked_example() -> None:
    # Docstring worked example: 0.8% daily std => 0.008 * sqrt(252) = 12.7%.
    assert annualize_vol(0.008) == pytest.approx(0.008 * math.sqrt(252.0))
    assert annualize_vol(0.008) == pytest.approx(0.12700, abs=1e-5)
    assert annualize_vol(0.0) == 0.0


def test_annualize_vol_matches_realized_vol_scaling() -> None:
    # annualize_vol is the SAME sqrt-of-time scaling realized_vol applies:
    # a constant-window daily std annualized by hand equals realized_vol's out.
    returns = _series([0.01, -0.01] * 6)  # window=2 std = 0.01*sqrt(2)
    vol = realized_vol(returns, window=2)
    hand = annualize_vol(statistics.stdev([0.01, -0.01]))
    for value in vol.to_numpy():
        assert value == pytest.approx(hand)


# --------------------------------------------------------------------------- #
# idzorek_confidence — literal pins + boundaries
# --------------------------------------------------------------------------- #


def test_floor_and_ceil_literal_values() -> None:
    assert CONFIDENCE_FLOOR == 0.02
    assert CONFIDENCE_CEIL == 0.95


def test_sigma_to_zero_plus_clamps_to_ceiling() -> None:
    # A vanishing predictive std means near-certainty; the ceiling caps it.
    assert idzorek_confidence(0.01, 1e-12) == 0.95
    # Even a denormal-scale sigma (mu/sigma overflows to inf) must not crash:
    # erf(inf) == 1.0 -> ceiling.
    assert idzorek_confidence(0.01, 1e-300) == 0.95


def test_sigma_huge_falls_to_floor() -> None:
    assert idzorek_confidence(0.01, 1e6) == 0.02


def test_mu_zero_is_exactly_the_floor() -> None:
    # No-information forecast: |2*Phi(0) - 1| == 0 -> clamped to exactly 0.02,
    # the graceful-degradation path where BL returns ~ the market prior.
    assert idzorek_confidence(0.0, 0.05) == 0.02


@pytest.mark.parametrize(
    ("mu", "sigma"),
    [
        pytest.param(0.01, 0.05, id="mid-band"),
        pytest.param(3.0, 0.01, id="clamped-ceiling"),
    ],
)
def test_symmetric_in_mu_sign(mu: float, sigma: float) -> None:
    # Evenness is by construction (abs(mu / sigma)); two cases suffice.
    assert idzorek_confidence(mu, sigma) == idzorek_confidence(-mu, sigma)


def test_percent_instead_of_decimal_misuse_clamps_not_crashes() -> None:
    # The "25 instead of 0.25" misuse (plan section 9 risk table): a caller
    # passing percent-scale mu must hit the ceiling, never overflow or raise.
    assert idzorek_confidence(25.0, 1.0) == 0.95


def test_one_sigma_forecast_gives_68_percent_confidence() -> None:
    # |mu| == sigma -> |2*Phi(1) - 1| = erf(1/sqrt(2)) = 0.682689... (the
    # one-sigma normal mass) — a known-good mid-range anchor between clamps.
    assert idzorek_confidence(0.02, 0.02) == pytest.approx(0.6826894921370859, abs=1e-12)


def test_monotonic_in_signal_to_noise() -> None:
    sigma = 0.05
    ratios = [0.1, 0.5, 1.0, 1.5, 2.0]
    values = [idzorek_confidence(r * sigma, sigma) for r in ratios]
    assert values == sorted(values)
    assert all(0.02 <= value <= 0.95 for value in values)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_bad_sigma_raises(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        idzorek_confidence(0.01, sigma)


@pytest.mark.parametrize("mu", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_mu_raises(mu: float) -> None:
    with pytest.raises(ValueError, match="mu"):
        idzorek_confidence(mu, 0.05)


def test_confidence_is_native_float() -> None:
    assert isinstance(idzorek_confidence(0.01, 0.05), float)


def test_erf_identity_matches_two_phi_minus_one() -> None:
    # The implementation uses erf(|x|/sqrt(2)); the contract formula is
    # |2*Phi(x) - 1| with Phi the standard normal CDF. Assert the identity on
    # a grid inside the unclamped band (raw value in (0.02, 0.95)) so a future
    # "simplification" cannot silently change the map.
    for x in [0.1, 0.7, 1.3, 1.9]:
        phi = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert idzorek_confidence(x, 1.0) == pytest.approx(abs(2.0 * phi - 1.0), abs=1e-15)


# --------------------------------------------------------------------------- #
# EXPLANATIONS registry (the UI-servable formula + worked-example catalog)
# --------------------------------------------------------------------------- #

# Every stage-detail quantity the UI cards read must have a legible explanation.
_REQUIRED_EXPLANATION_KEYS = frozenset(
    {
        "log_return",  # features card + per-asset latest
        "realized_vol",  # features card + per-asset latest
        "ewma_mu",  # forecast card per-asset mu
        "forecast_sigma",  # forecast card per-asset sigma
        "bl_prior",  # blend card `prior` (pi)
        "bl_view",  # blend card `view` (Q)
        "bl_confidence",  # blend card `confidences`
        "bl_posterior",  # blend card `posterior_mu`
        "mvu_objective",  # optimize card `objective`
    }
)


def test_every_required_ui_quantity_has_an_explanation() -> None:
    assert _REQUIRED_EXPLANATION_KEYS <= set(EXPLANATIONS)


def test_registry_is_self_keyed_and_complete_per_entry() -> None:
    for key, explanation in EXPLANATIONS.items():
        assert explanation.key == key  # dict key == entry.key (no drift)
        # Every entry carries a real formula, description, and worked example.
        assert explanation.label
        assert explanation.formula
        assert explanation.description
        assert explanation.example


def test_payload_shape_is_the_api_contract() -> None:
    # payload() is exactly what GET /api/explain serves per key: the six
    # documented fields, no `key` (the map already keys by it).
    payload = EXPLANATIONS["mvu_objective"].payload()
    assert set(payload) == {"label", "formula", "description", "example", "unit", "window"}
    assert payload["formula"].startswith("maximize")
