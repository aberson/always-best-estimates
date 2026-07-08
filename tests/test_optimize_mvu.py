"""``optimize_weights`` tests (plan Step 7). No mocks — one documented fake.

Fixture provenance and declared bounds:

- ``_sigma_near_singular``: SPY/ACWI correlation 0.999 with EQUAL AGG
  correlations (-0.075 both). The equal-x structure keeps the matrix PSD for
  any equity correlation < 1 (eigenvector (1,-1,0) has eigenvalue ``1-rho``
  in correlation space; the symmetric 2x2 block is PD since
  ``1+rho > 2*x**2``), so it is genuinely NEAR-singular (min eigenvalue
  ~2.7e-5), not accidentally indefinite. Mismatched AGG correlations at
  rho=0.999 produce a genuinely INDEFINITE matrix (min eigenvalue ~-6.9e-6) —
  kept as the fail-loud fixture, discovered the hard way in development.
- STABILITY BOUND (cold start, LW-realistic corr 0.95): declared 0.25 —
  strictly below the 0.30 amplitude of a base-to-box corner flip (0.30 ->
  0.60 under W_MAX=0.6 from the 0.30/0.30/0.40 equilibrium base), with ~60%
  headroom over the measured worst drift (0.158). The test guards
  CORNER-FLIPPING, not tiny drift: a flip fails, smooth drift passes.
- STABILITY BOUND (production shape, near-singular 0.999 + w_prev + default
  GAMMA_TC): declared 0.01 — measured <= 2e-7. GAMMA_TC's no-trade band
  (0.002 of marginal annual return) is wider than the ~0.001 gradient shift a
  +/-10bp mu perturbation produces, so noise-scale drift does not trade AT
  ALL; the same perturbation without the penalty moves weights by ~0.30
  (asserted in the same test, proving the penalty is load-bearing).
- RETRY-PATH TESTING CHOICE (documented per the step spec): the box+budget
  constraint set is feasible for any valid ``w_max``, so real INFEASIBLE
  cannot be constructed from valid inputs. The retry policy is therefore
  exercised through ``optimize_weights``'s own ``_solve_once_fn`` seam with a
  hand-written fake (class with a call log, not unittest.mock) that fails the
  first call and delegates to the REAL ``_solve_once`` afterwards — the outer
  path (validation, retry, flag, post-processing) runs unmodified.
"""

import logging
import math

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from abe import constants
from abe.blend.black_litterman import bl_blend
from abe.blend.covariance import ledoit_wolf_sigma
from abe.calc import LOG_RETURN_COLUMN
from abe.constants import DELTA, UNIVERSE, W_MAX, W_MKT
from abe.model.base import EWMABaseline
from abe.optimize.mvu import (
    GAMMA_TC,
    SOLVER,
    MVUResult,
    _solve_once,
    optimize_weights,
)

# Box-constraint tolerance for assertions: CLARABEL's feasibility tolerance
# plus the renormalization epsilon (observed overshoot ~5e-9).
BOX_TOL = 1e-6

# Realistic annualized SPY/ACWI/AGG vols (same as the Step 6 test fixtures).
_VOLS = np.array([0.17, 0.16, 0.05])


def _sigma_from_corr(corr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.outer(_VOLS, _VOLS) * corr, index=list(UNIVERSE), columns=list(UNIVERSE))


def _sigma_well_conditioned() -> pd.DataFrame:
    return _sigma_from_corr(
        np.array(
            [
                [1.00, 0.60, -0.10],
                [0.60, 1.00, -0.05],
                [-0.10, -0.05, 1.00],
            ]
        )
    )


def _sigma_lw_realistic() -> pd.DataFrame:
    """Correlation ~0.95 — what Ledoit-Wolf shrinkage of this universe emits."""
    return _sigma_from_corr(
        np.array(
            [
                [1.00, 0.95, -0.075],
                [0.95, 1.00, -0.075],
                [-0.075, -0.075, 1.00],
            ]
        )
    )


def _sigma_near_singular() -> pd.DataFrame:
    """PSD with min eigenvalue ~2.7e-5 — THE plan section 8 risk case."""
    return _sigma_from_corr(
        np.array(
            [
                [1.000, 0.999, -0.075],
                [0.999, 1.000, -0.075],
                [-0.075, -0.075, 1.000],
            ]
        )
    )


def _sigma_rank1() -> pd.DataFrame:
    """Exactly singular PSD (rank 1: all correlations 1.0) — the jitter path."""
    return pd.DataFrame(np.outer(_VOLS, _VOLS), index=list(UNIVERSE), columns=list(UNIVERSE))


def _mu(values: list[float] | np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=list(UNIVERSE))


def _prior(sigma: pd.DataFrame) -> np.ndarray:
    w = np.array([W_MKT[asset] for asset in UNIVERSE])
    result: np.ndarray = DELTA * (sigma.to_numpy() @ w)
    return result


def _assert_valid_allocation(result: MVUResult) -> None:
    assert list(result.weights) == list(UNIVERSE)
    values = list(result.weights.values())
    assert all(isinstance(v, float) and math.isfinite(v) for v in values)
    assert sum(result.weights.values()) == 1.0  # float-EXACT (renormalization contract)
    assert all(0.0 <= v <= W_MAX + BOX_TOL for v in values)


# --------------------------------------------------------------------------- #
# Core contract: sum, box, ordering
# --------------------------------------------------------------------------- #


def test_equilibrium_mu_reproduces_market_weights() -> None:
    """mu = delta*Sigma*w_mkt inverts to w_mkt — the delta-sharing anchor.

    The BL prior and the optimizer share DELTA (plan section 8), so feeding
    the unblended equilibrium prior back through the QP must reproduce the
    benchmark weights. Also certifies sum_squares(chol.T @ w) == w'Sigma*w
    numerically (a wrong quadratic form would not invert to w_mkt).
    """
    sigma = _sigma_lw_realistic()
    result = optimize_weights(_mu(_prior(sigma)), sigma)
    for asset in UNIVERSE:
        assert result.weights[asset] == pytest.approx(W_MKT[asset], abs=1e-5)


def test_w_max_binds_on_strongly_favored_asset() -> None:
    result = optimize_weights(_mu([0.50, 0.01, 0.005]), _sigma_well_conditioned())
    _assert_valid_allocation(result)
    assert result.weights["SPY"] == pytest.approx(W_MAX, abs=BOX_TOL)


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #


def test_cold_start_drops_turnover_and_reports_zero() -> None:
    result = optimize_weights(_mu([0.05, 0.04, 0.02]), _sigma_well_conditioned(), w_prev=None)
    _assert_valid_allocation(result)
    assert result.prev_weights is None
    assert result.turnover == {asset: 0.0 for asset in UNIVERSE}
    assert result.relaxed_turnover is False
    assert result.status == str(cp.OPTIMAL)


# --------------------------------------------------------------------------- #
# Near-singular / singular / indefinite Sigma (THE plan section 8 risk)
# --------------------------------------------------------------------------- #


def test_near_singular_sigma_flows_through() -> None:
    sigma = _sigma_near_singular()
    min_eig = float(np.linalg.eigvalsh(sigma.to_numpy()).min())
    assert 0.0 < min_eig < 1e-4  # genuinely near-singular, genuinely PSD
    result = optimize_weights(_mu(_prior(sigma) + [0.005, -0.002, 0.001]), sigma)
    _assert_valid_allocation(result)


def test_exactly_singular_psd_sigma_takes_jitter_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sigma = _sigma_rank1()
    # Guard: this fixture genuinely defeats plain Cholesky, so success below
    # proves the jitter path (not luck).
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(sigma.to_numpy())
    with caplog.at_level(logging.WARNING, logger="abe.optimize.mvu"):
        result = optimize_weights(_mu([0.06, 0.05, 0.02]), sigma)
    _assert_valid_allocation(result)
    assert any("PSD boundary" in record.message for record in caplog.records)


def test_indefinite_sigma_fails_loud() -> None:
    # Mismatched AGG correlations at rho=0.999: symmetric but INDEFINITE
    # (min eigenvalue ~-6.9e-6, far beyond any legitimate jitter).
    sigma = _sigma_from_corr(
        np.array(
            [
                [1.000, 0.999, -0.10],
                [0.999, 1.000, -0.05],
                [-0.10, -0.05, 1.000],
            ]
        )
    )
    assert float(np.linalg.eigvalsh(sigma.to_numpy()).min()) < -1e-7
    with pytest.raises(ValueError, match="not decomposable"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), sigma)


# --------------------------------------------------------------------------- #
# +/-10bp mu-perturbation stability (bounds documented in module docstring)
# --------------------------------------------------------------------------- #


def test_mu_perturbation_stability_cold_start() -> None:
    """Cold start, LW-realistic Sigma: max weight change < 0.25 (no corner flip)."""
    sigma = _sigma_lw_realistic()
    mu0 = _prior(sigma)
    base = np.array(list(optimize_weights(_mu(mu0), sigma).weights.values()))
    worst = 0.0
    for i in range(len(UNIVERSE)):
        for sign in (+1.0, -1.0):
            mu_p = mu0.copy()
            mu_p[i] += sign * 0.001  # +/-10bp, annualized units
            perturbed = np.array(list(optimize_weights(_mu(mu_p), sigma).weights.values()))
            worst = max(worst, float(np.abs(perturbed - base).max()))
    assert worst < 0.25, f"corner-flip guard: max weight change {worst:.4f} >= 0.25"


def test_mu_perturbation_stability_production_shape() -> None:
    """Near-singular Sigma + w_prev + default GAMMA_TC: +/-10bp moves < 0.01.

    Also asserts the penalty is LOAD-BEARING: the same perturbation with
    gamma_tc=0 moves weights by more than the cold-start bound (a corner
    flip), so the production stability is the turnover band's doing, not an
    accident of the fixture.
    """
    sigma = _sigma_near_singular()
    mu0 = _prior(sigma)
    base_result = optimize_weights(_mu(mu0), sigma)
    base = np.array(list(base_result.weights.values()))

    worst_prod = 0.0
    worst_free = 0.0
    for i in range(len(UNIVERSE)):
        for sign in (+1.0, -1.0):
            mu_p = mu0.copy()
            mu_p[i] += sign * 0.001
            prod = optimize_weights(_mu(mu_p), sigma, w_prev=base_result.weights)
            free = optimize_weights(_mu(mu_p), sigma, w_prev=base_result.weights, gamma_tc=0.0)
            worst_prod = max(
                worst_prod,
                float(np.abs(np.array(list(prod.weights.values())) - base).max()),
            )
            worst_free = max(
                worst_free,
                float(np.abs(np.array(list(free.weights.values())) - base).max()),
            )
    assert worst_prod < 0.01, f"no-trade band failed: max change {worst_prod:.6f}"
    assert worst_free > 0.25, "fixture no longer stresses the near-singular flip"


# --------------------------------------------------------------------------- #
# Turnover penalty semantics
# --------------------------------------------------------------------------- #


def test_turnover_penalty_binds() -> None:
    """Large gamma_tc keeps weights closer to w_prev than gamma_tc=0 (L1 order)."""
    sigma = _sigma_lw_realistic()
    mu = _mu([0.08, 0.02, 0.01])  # strongly favors SPY, far from w_prev
    w_prev = {"SPY": 0.10, "ACWI": 0.30, "AGG": 0.60}
    free = optimize_weights(mu, sigma, w_prev=w_prev, gamma_tc=0.0)
    tight = optimize_weights(mu, sigma, w_prev=w_prev, gamma_tc=0.05)
    l1_free = sum(abs(free.weights[a] - w_prev[a]) for a in UNIVERSE)
    l1_tight = sum(abs(tight.weights[a] - w_prev[a]) for a in UNIVERSE)
    assert l1_tight < l1_free
    assert l1_free > 0.5  # unpenalized optimum really is far from w_prev
    assert l1_tight < 0.01  # 5%-per-unit-turnover freezes the book
    # Turnover diagnostics match |w - w_prev| on the FINAL weights
    for asset in UNIVERSE:
        assert free.turnover[asset] == pytest.approx(
            abs(free.weights[asset] - w_prev[asset]), abs=1e-12
        )


def test_default_gamma_still_trades_on_genuine_tilt() -> None:
    """Lower anchor of the no-trade band: a percent-scale view shift MUST move
    the book at the DEFAULT gamma_tc. A silent GAMMA_TC bump (0.002 -> 0.05
    provably freezes the book per test_turnover_penalty_binds) would pass
    every other test while the engine never rebalances again."""
    sigma = _sigma_lw_realistic()
    mu = _mu([0.06, 0.02, 0.01])  # percent-scale tilt toward SPY vs balanced book
    w_prev = dict(constants.W_MKT)
    result = optimize_weights(mu, sigma, w_prev=w_prev)  # default gamma_tc
    l1 = sum(abs(result.weights[a] - w_prev[a]) for a in UNIVERSE)
    assert l1 > 0.05  # genuine views clear the band and trade


def test_w_prev_subset_keys_fill_zero() -> None:
    """A persisted book may omit clip-zeroed assets; missing keys mean 0.0."""
    result = optimize_weights(
        _mu([0.05, 0.04, 0.02]),
        _sigma_well_conditioned(),
        w_prev={"SPY": 0.4, "AGG": 0.6},
    )
    assert result.prev_weights == {"SPY": 0.4, "ACWI": 0.0, "AGG": 0.6}
    assert result.turnover["ACWI"] == pytest.approx(result.weights["ACWI"], abs=1e-12)


# --------------------------------------------------------------------------- #
# Retry policy through the _solve_once_fn seam (choice documented on module)
# --------------------------------------------------------------------------- #


class _FailFirstThenReal:
    """Fake solve seam: fail call #1, delegate to the REAL _solve_once after."""

    def __init__(self) -> None:
        self.w_prev_per_call: list[np.ndarray | None] = []

    def __call__(
        self,
        mu: np.ndarray,
        chol: np.ndarray,
        w_prev: np.ndarray | None,
        delta: float,
        gamma_tc: float,
        w_max: float,
    ) -> tuple[str, np.ndarray | None]:
        self.w_prev_per_call.append(w_prev)
        if len(self.w_prev_per_call) == 1:
            return ("solver_error: injected first-call failure", None)
        return _solve_once(mu, chol, w_prev, delta, gamma_tc, w_max)


class _AlwaysFail:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        mu: np.ndarray,
        chol: np.ndarray,
        w_prev: np.ndarray | None,
        delta: float,
        gamma_tc: float,
        w_max: float,
    ) -> tuple[str, np.ndarray | None]:
        self.calls += 1
        return (str(cp.INFEASIBLE), None)


def test_solver_failure_retries_without_turnover_and_flags_relaxed() -> None:
    fake = _FailFirstThenReal()
    w_prev = {"SPY": 0.10, "ACWI": 0.30, "AGG": 0.60}
    result = optimize_weights(
        _mu([0.05, 0.04, 0.02]),
        _sigma_well_conditioned(),
        w_prev=w_prev,
        _solve_once_fn=fake,
    )
    _assert_valid_allocation(result)
    assert result.relaxed_turnover is True
    assert result.status == str(cp.OPTIMAL)
    # First call carried the turnover anchor; the retry dropped it.
    assert len(fake.w_prev_per_call) == 2
    assert fake.w_prev_per_call[0] is not None
    assert fake.w_prev_per_call[1] is None
    # prev_weights and turnover still reflect the REAL previous book.
    assert result.prev_weights == w_prev
    for asset in UNIVERSE:
        assert result.turnover[asset] == pytest.approx(
            abs(result.weights[asset] - w_prev[asset]), abs=1e-12
        )


def test_both_solves_failing_raises_with_both_statuses() -> None:
    fake = _AlwaysFail()
    with pytest.raises(RuntimeError, match="relaxed retry"):
        optimize_weights(
            _mu([0.05, 0.04, 0.02]),
            _sigma_well_conditioned(),
            w_prev={"SPY": 0.10, "ACWI": 0.30, "AGG": 0.60},
            _solve_once_fn=fake,
        )
    assert fake.calls == 2


def test_cold_start_failure_raises_without_retry() -> None:
    fake = _AlwaysFail()
    with pytest.raises(RuntimeError, match="cold start"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), _sigma_well_conditioned(), _solve_once_fn=fake)
    assert fake.calls == 1  # nothing to relax without a turnover term


def test_optimal_inaccurate_accepted_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def inaccurate_solve(
        mu: np.ndarray,
        chol: np.ndarray,
        w_prev: np.ndarray | None,
        delta: float,
        gamma_tc: float,
        w_max: float,
    ) -> tuple[str, np.ndarray | None]:
        status, weights = _solve_once(mu, chol, w_prev, delta, gamma_tc, w_max)
        assert status == str(cp.OPTIMAL)
        return (str(cp.OPTIMAL_INACCURATE), weights)

    with caplog.at_level(logging.WARNING, logger="abe.optimize.mvu"):
        result = optimize_weights(
            _mu([0.05, 0.04, 0.02]),
            _sigma_well_conditioned(),
            _solve_once_fn=inaccurate_solve,
        )
    _assert_valid_allocation(result)
    assert result.status == str(cp.OPTIMAL_INACCURATE)
    assert result.relaxed_turnover is False
    assert any("OPTIMAL_INACCURATE" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# Producer -> consumer: a real bl_blend output feeds optimize_weights
# --------------------------------------------------------------------------- #


def test_blresult_round_trip() -> None:
    """features -> EWMA -> LW Sigma -> bl_blend -> optimize_weights, no mocks."""
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
    blend = bl_blend(sigma, forecasts)

    result = optimize_weights(blend.mu_post, blend.sigma_post, w_prev=W_MKT)
    _assert_valid_allocation(result)
    assert result.relaxed_turnover is False
    assert result.prev_weights == {asset: float(W_MKT[asset]) for asset in UNIVERSE}
    assert list(result.turnover) == list(UNIVERSE)


def test_exact_sum_holds_across_many_solves() -> None:
    """Stress the renormalization contract on the near-singular fixture."""
    sigma = _sigma_near_singular()
    mu0 = _prior(sigma)
    rng = np.random.default_rng(7)
    for i in range(60):
        w_prev = {"SPY": 0.2, "ACWI": 0.2, "AGG": 0.6} if i % 3 == 0 else None
        result = optimize_weights(_mu(mu0 + rng.normal(0.0, 0.01, 3)), sigma, w_prev=w_prev)
        _assert_valid_allocation(result)


# --------------------------------------------------------------------------- #
# Input rejection matrix (fail-loud validation)
# --------------------------------------------------------------------------- #


def test_mu_in_wrong_asset_order_rejected() -> None:
    mu = pd.Series([0.05, 0.04, 0.02], index=["ACWI", "SPY", "AGG"])
    with pytest.raises(ValueError, match="UNIVERSE order"):
        optimize_weights(mu, _sigma_well_conditioned())


def test_mu_with_nan_names_the_cell() -> None:
    mu = _mu([0.05, np.nan, 0.02])
    with pytest.raises(ValueError, match="ACWI"):
        optimize_weights(mu, _sigma_well_conditioned())


def test_sigma_in_wrong_asset_order_rejected() -> None:
    wrong = ["ACWI", "SPY", "AGG"]
    sigma = _sigma_well_conditioned().loc[wrong, wrong]
    with pytest.raises(ValueError, match="UNIVERSE order"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), sigma)


def test_sigma_with_nan_names_the_cell() -> None:
    sigma = _sigma_well_conditioned()
    sigma.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"\(SPY, SPY\)"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), sigma)


def test_asymmetric_sigma_rejected() -> None:
    sigma = _sigma_well_conditioned()
    sigma.iloc[0, 1] += 1e-3
    with pytest.raises(ValueError, match="symmetric"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), sigma)


def test_w_prev_with_unknown_asset_rejected() -> None:
    with pytest.raises(ValueError, match="TLT"):
        optimize_weights(
            _mu([0.05, 0.04, 0.02]),
            _sigma_well_conditioned(),
            w_prev={"TLT": 1.0},
        )


def test_w_prev_not_summing_to_one_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        optimize_weights(
            _mu([0.05, 0.04, 0.02]),
            _sigma_well_conditioned(),
            w_prev={"SPY": 0.5, "ACWI": 0.3, "AGG": 0.1},
        )


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5, float("nan")])
def test_w_prev_weight_outside_unit_interval_rejected(bad_weight: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        optimize_weights(
            _mu([0.05, 0.04, 0.02]),
            _sigma_well_conditioned(),
            w_prev={"SPY": bad_weight, "ACWI": 0.5, "AGG": 0.5},
        )


@pytest.mark.parametrize("delta", [0.0, -2.5, float("nan")])
def test_bad_delta_rejected(delta: float) -> None:
    with pytest.raises(ValueError, match="delta"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), _sigma_well_conditioned(), delta=delta)


@pytest.mark.parametrize("gamma_tc", [-0.001, float("nan"), float("inf")])
def test_bad_gamma_tc_rejected(gamma_tc: float) -> None:
    with pytest.raises(ValueError, match="gamma_tc"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), _sigma_well_conditioned(), gamma_tc=gamma_tc)


@pytest.mark.parametrize("w_max", [0.0, -0.5, 1.5, float("nan")])
def test_bad_w_max_rejected(w_max: float) -> None:
    with pytest.raises(ValueError, match="w_max"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), _sigma_well_conditioned(), w_max=w_max)


def test_structurally_infeasible_w_max_rejected_at_validation() -> None:
    # 3 assets capped at 0.2 cannot sum to 1 — a config bug, not a solver run.
    with pytest.raises(ValueError, match="infeasible"):
        optimize_weights(_mu([0.05, 0.04, 0.02]), _sigma_well_conditioned(), w_max=0.2)


# --------------------------------------------------------------------------- #
# Pins: solver + shared-constant defaults
# --------------------------------------------------------------------------- #


def test_clarabel_pinned_and_installed() -> None:
    assert SOLVER == "CLARABEL"
    assert SOLVER == str(cp.CLARABEL)
    assert "CLARABEL" in cp.installed_solvers()


def test_defaults_are_the_shared_constants() -> None:
    # One-source-of-truth guard (code-quality rule): the QP's risk aversion IS
    # the BL prior's DELTA and the box IS the shared W_MAX. `is` (not ==) on
    # the signature defaults so a re-typed literal (delta: float = 2.5) —
    # equal but a distinct object — fails CI.
    assert optimize_weights.__defaults__ is not None
    assert optimize_weights.__defaults__[0] is None
    assert optimize_weights.__defaults__[1] is constants.DELTA
    assert optimize_weights.__defaults__[2] is GAMMA_TC
    assert optimize_weights.__defaults__[3] is constants.W_MAX
