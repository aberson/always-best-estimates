"""``bl_blend`` — Black-Litterman posterior from forecasts (plan Step 6).

The blend stage: prior equilibrium excess returns ``pi = delta * Sigma *
w_mkt`` (Idzorek Formula 1; fixed benchmark vector ``W_MKT`` per plan section
8, sanctioned by Idzorek fn.4's "presumed efficient benchmark"), one ordered
list of ABSOLUTE views derived from the ``WorldModel`` forecasts (P = identity
rows; relative views are out of scope for V1), Idzorek-omega posterior via the
pinned ``pypfopt==1.6.0`` ``BlackLittermanModel``.

Unit conventions (plan section 8 — "everything internal in annualized excess
returns"):

- ``Sigma`` arrives ANNUALIZED (``covariance.ledoit_wolf_sigma``).
- ``Forecast.mu`` is an H-day log-return; it is annualized HERE — and only
  here — via ``Q = annualize_mean(mu)`` (= ``mu * TRADING_DAYS / HORIZON_BARS``,
  x12 for H=21; the shared ``abe.calc`` helper is the one source of truth for
  the scale). This function is THE H-day -> annual boundary; nothing upstream
  or downstream converts again.
- Everything is EXCESS return with ``rf = 0.0`` explicit: ``risk_free_rate``
  is passed at every pypfopt call site that accepts it
  (``market_implied_prior_returns``, ``BlackLittermanModel``). V1 always calls
  with ``rf=0.0`` — the risk-free subtraction happens once at the ingest
  adapter, never here.

Confidence-scale DECISION (documented per plan Step 6): Idzorek confidence is
computed from the RAW H-day ``(mu, sigma)`` exactly as the model emits them —
``sigma`` is NOT rescaled. The map depends only on the ratio ``mu/sigma``,
which is invariant only when both members scale identically; annualization
scales ``mu`` by ``252/H`` (=12) but an independent-draws ``sigma`` by
``sqrt(252/H)`` (~3.46), so "annualize first" would silently multiply the
z-score by ~3.46 and saturate every confidence at the 0.95 ceiling. The H-day
predictive pair is the scale plan section 4's calibration gate certifies, so
it is the honest input.

Constructor contract verified against the installed 1.6.0 source
(``pypfopt/black_litterman.py``): ``pi`` takes the prior Series directly
(``market_prior``/``pi="market"`` is only for the library's own cap-weighted
path); ``absolute_views`` is a ``{ticker: annual_excess_return}`` mapping
parsed into Q (Kx1) and identity-row P in mapping order; ``omega="idzorek"``
requires ``view_confidences`` and builds a diagonal omega via Walters'
closed form ``omega_k = tau * (1-c)/c * p_k Sigma p_k'`` (for a single view
this makes the posterior tilt EXACTLY ``c`` times the 100%-confidence tilt —
Idzorek's Formula 10/15 intent, pinned by the golden tests).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

from abe.calc import annualize_mean, idzorek_confidence
from abe.constants import DELTA, TAU, UNIVERSE, W_MKT
from abe.model.base import Forecast

__all__ = [
    "BLResult",
    "bl_blend",
]

_W_MKT_SUM_TOL: Final[float] = 1e-6
"""Absolute tolerance for ``sum(w_mkt) == 1`` — loose enough for hand-typed
config vectors (0.30/0.30/0.40), tight enough to catch a forgotten asset."""

_SYMMETRY_ATOL: Final[float] = 1e-8
"""Absolute symmetry tolerance for the input covariance — ledoit_wolf_sigma
symmetrizes exactly, so anything beyond float noise means a foreign matrix."""


@dataclass(frozen=True, slots=True)
class BLResult:
    """The blend stage's output (consumed by the ``bl_posteriors`` table + UI card).

    - ``mu_post``: posterior expected excess returns, ANNUALIZED, indexed in
      canonical ``UNIVERSE`` order.
    - ``sigma_post``: posterior covariance (He-Litterman ``Sigma + M`` via
      ``bl_cov()``), annualized, ``UNIVERSE``-labeled, symmetrized.
    - ``diagnostics``: JSON-friendly ``{section: {asset: float}}`` with
      sections ``pi`` (prior, all assets), ``Q`` (annualized view returns,
      viewed assets only), ``confidences`` (Idzorek c per viewed asset),
      ``omega_diag`` (view uncertainty per viewed asset), ``tilt``
      (``mu_post - pi``, all assets) — exactly what ``detail_json`` stores.

    ``frozen=True`` prevents attribute REBINDING only: the pandas members are
    themselves mutable (``result.mu_post.iloc[0] = ...`` succeeds). Treat the
    result as read-only; each ``bl_blend`` call returns freshly built objects.
    """

    mu_post: pd.Series
    sigma_post: pd.DataFrame
    diagnostics: dict[str, dict[str, float]]


def _validate_sigma(sigma_annual: pd.DataFrame) -> None:
    canonical = list(UNIVERSE)
    if list(sigma_annual.index) != canonical or list(sigma_annual.columns) != canonical:
        raise ValueError(
            "sigma_annual must be labeled exactly UNIVERSE order "
            f"{canonical!r} on both axes; got index={list(sigma_annual.index)!r}, "
            f"columns={list(sigma_annual.columns)!r}"
        )
    array = sigma_annual.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        bad_rows, bad_cols = np.nonzero(~np.isfinite(array))
        cells = [
            f"({sigma_annual.index[r]}, {sigma_annual.columns[c]})"
            for r, c in zip(bad_rows[:5], bad_cols[:5], strict=True)
        ]
        more = "" if len(bad_rows) <= 5 else f" (+{len(bad_rows) - 5} more)"
        raise ValueError(
            f"sigma_annual contains {len(bad_rows)} NaN/non-finite cell(s) at "
            f"{', '.join(cells)}{more}"
        )
    if not np.allclose(array, array.T, rtol=0.0, atol=_SYMMETRY_ATOL):
        raise ValueError("sigma_annual is not symmetric — it did not come from ledoit_wolf_sigma")


def _validate_w_mkt(w_mkt: Mapping[str, float]) -> pd.Series:
    if set(w_mkt) != set(UNIVERSE):
        raise ValueError(
            f"w_mkt keys must be exactly UNIVERSE {sorted(UNIVERSE)!r}, got {sorted(w_mkt)!r}"
        )
    ordered = pd.Series([float(w_mkt[asset]) for asset in UNIVERSE], index=list(UNIVERSE))
    values = ordered.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"w_mkt contains non-finite weights: {dict(w_mkt)!r}")
    total = float(values.sum())
    if abs(total - 1.0) > _W_MKT_SUM_TOL:
        # pypfopt renormalizes market "caps" silently; a wrong-sum weight
        # vector here is an upstream bug, not something to paper over.
        raise ValueError(f"w_mkt must sum to 1.0 (benchmark weights), got {total!r}")
    return ordered


def _validate_forecasts(forecasts: Mapping[str, Forecast]) -> list[str]:
    unknown = sorted(set(forecasts) - set(UNIVERSE))
    if unknown:
        raise ValueError(
            f"forecasts contain assets outside UNIVERSE {sorted(UNIVERSE)!r}: {unknown!r}"
        )
    # ONE canonical view order: constants.UNIVERSE, always.
    return [asset for asset in UNIVERSE if asset in forecasts]


def bl_blend(
    sigma_annual: pd.DataFrame,
    forecasts: Mapping[str, Forecast],
    w_mkt: Mapping[str, float] = W_MKT,
    delta: float = DELTA,
    tau: float = TAU,
    rf: float = 0.0,
) -> BLResult:
    """Blend the equilibrium prior with forecast views into a BL posterior.

    ``pi = delta * sigma_annual * w_mkt + rf`` (annualized excess; ``rf=0.0``
    in V1), computed through pypfopt's ``market_implied_prior_returns`` with
    ``risk_free_rate`` explicit. Each forecast becomes one absolute view in
    canonical ``UNIVERSE`` order: ``Q[i] = annualize_mean(forecasts[asset].mu)``
    (the H-day -> annual boundary; module docstring) with
    ``confidence[i] = idzorek_confidence(mu, sigma)`` computed from the RAW
    H-day pair (decision in the module docstring). Posterior mu/Sigma via
    ``BlackLittermanModel(omega="idzorek")``.

    Graceful degradation (plan section 4): an EMPTY ``forecasts`` mapping is
    the documented no-view path — ``mu_post == pi`` exactly (the market
    prior) and ``sigma_post == (1 + tau) * sigma_annual``, the K=0 limit of
    the He-Litterman posterior covariance ``Sigma + M`` (``M = tau * Sigma``
    with no views), so the no-view and one-view paths agree in the limit.

    Validation (all fail loud): ``sigma_annual`` labeled exactly ``UNIVERSE``
    on both axes, finite, symmetric; ``w_mkt`` keys exactly ``UNIVERSE``,
    finite, summing to 1; ``forecasts`` keys a subset of ``UNIVERSE`` (a
    subset is allowed — pypfopt handles K < N absolute views); ``delta > 0``,
    ``0 < tau <= 1``; ``rf`` must be exactly ``0.0`` in V1 — a nonzero rf
    would make ``pi`` a TOTAL-return vector while Q stays excess (unit mix,
    every view residual biased by -rf). The rf subtraction happens once at
    the ingest adapter (plan section 8), never here.
    """
    _validate_sigma(sigma_annual)
    w_mkt_series = _validate_w_mkt(w_mkt)
    viewed = _validate_forecasts(forecasts)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be finite and > 0, got {delta!r}")
    if not math.isfinite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau!r}")
    if rf != 0.0:
        raise ValueError(
            f"rf must be exactly 0.0 in V1 (excess-return convention; nonzero rf "
            f"mixes total-return pi with excess-return Q), got {rf!r}"
        )

    pi_raw = market_implied_prior_returns(w_mkt_series, delta, sigma_annual, risk_free_rate=rf)
    pi = pd.Series(pi_raw.to_numpy(dtype=float), index=list(UNIVERSE))

    if not viewed:
        mu_post = pi.copy()
        sigma_post = sigma_annual.astype(float) * (1.0 + tau)
        return BLResult(
            mu_post=mu_post,
            sigma_post=sigma_post,
            diagnostics=_diagnostics(pi, mu_post, {}, {}, {}),
        )

    absolute_views = {asset: annualize_mean(float(forecasts[asset].mu)) for asset in viewed}
    confidences = {
        asset: idzorek_confidence(forecasts[asset].mu, forecasts[asset].sigma) for asset in viewed
    }

    bl = BlackLittermanModel(
        sigma_annual,
        pi=pi,
        absolute_views=absolute_views,
        omega="idzorek",
        view_confidences=np.asarray([confidences[asset] for asset in viewed], dtype=float),
        tau=tau,
        risk_aversion=delta,
        risk_free_rate=rf,
    )
    mu_post_raw = bl.bl_returns()
    mu_post = pd.Series(mu_post_raw.to_numpy(dtype=float), index=list(UNIVERSE))
    posterior_cov = bl.bl_cov().to_numpy(dtype=float)
    posterior_cov = (posterior_cov + posterior_cov.T) / 2.0
    sigma_post = pd.DataFrame(posterior_cov, index=list(UNIVERSE), columns=list(UNIVERSE))
    omega_diag_values = np.diag(np.asarray(bl.omega, dtype=float))
    omega_diag = {asset: float(omega_diag_values[i]) for i, asset in enumerate(viewed)}

    return BLResult(
        mu_post=mu_post,
        sigma_post=sigma_post,
        diagnostics=_diagnostics(pi, mu_post, absolute_views, confidences, omega_diag),
    )


def _diagnostics(
    pi: pd.Series,
    mu_post: pd.Series,
    q: dict[str, float],
    confidences: dict[str, float],
    omega_diag: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Native-float diagnostics in the exact shape ``detail_json`` persists."""
    return {
        "pi": {asset: float(pi[asset]) for asset in UNIVERSE},
        "Q": {asset: float(value) for asset, value in q.items()},
        "confidences": {asset: float(value) for asset, value in confidences.items()},
        "omega_diag": {asset: float(value) for asset, value in omega_diag.items()},
        "tilt": {asset: float(mu_post[asset] - pi[asset]) for asset in UNIVERSE},
    }
