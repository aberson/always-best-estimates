"""Boundary + property tests for the sigma -> Idzorek confidence map (plan Step 6).

No mocks; pure-function checks. The clamp endpoints are LITERAL pins (0.02 /
0.95 are plan-section-4 contract values, not derived numbers).
"""

import math

import pytest

from abe.blend.confidence import CONFIDENCE_CEIL, CONFIDENCE_FLOOR, idzorek_confidence

# --------------------------------------------------------------------------- #
# Literal pins for the clamp constants (plan section 4 contract values)
# --------------------------------------------------------------------------- #


def test_floor_and_ceil_literal_values() -> None:
    assert CONFIDENCE_FLOOR == 0.02
    assert CONFIDENCE_CEIL == 0.95


# --------------------------------------------------------------------------- #
# Boundary behavior
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Mid-range values + monotonicity
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Loud rejection
# --------------------------------------------------------------------------- #


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
