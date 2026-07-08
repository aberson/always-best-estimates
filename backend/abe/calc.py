"""The transparent home for the project's simple, hand-checkable calculations.

Track 1 (transparency pass) collects here — in ONE leaf module — every small,
self-contained calculation the pipeline performs, so the six stage cards can be
made legible without hunting the math across ``features/``, ``blend/`` and
``optimize/``. Nothing here adds new pipeline math: the functions were RELOCATED
verbatim from ``features/basic.py`` (``log_returns``, ``realized_vol``,
``non_finite_summary`` + the column-name constants) and ``blend/confidence.py``
(``idzorek_confidence`` + its clamp constants), plus the ``H_TO_ANNUAL`` scale
previously defined in ``blend/black_litterman.py`` and two tiny annualization
helpers. Behavior is byte-identical to the pre-relocation code.

Design: a LEAF module (imports only :mod:`abe.constants`), so producer and
consumer share ONE definition of every constant and function — the
one-source-of-truth rule (``LOG_RETURN_COLUMN``/``REALIZED_VOL_COLUMN`` are
imported, never re-declared; ``is``-identity regression tests pin this).

Every calculation carries a docstring with its plain formula AND an intuitive
worked numeric example. The :class:`Explanation` registry (:data:`EXPLANATIONS`)
lifts those same formulas/examples into a structured, UI-servable form (the
``GET /api/explain`` endpoint) so each stage card can show what it computed and
how.

NaN policy (relocated verbatim): outputs NEVER contain NaN. Undefined rows (the
first return, the rolling warm-up) are DROPPED, not emitted as NaN — storage and
every downstream consumer reject NaN by contract. Inputs containing
NaN/non-finite values raise ``ValueError`` loudly.
"""

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from abe.constants import HORIZON_BARS, TRADING_DAYS

__all__ = [
    "CONFIDENCE_CEIL",
    "CONFIDENCE_FLOOR",
    "EXPLANATIONS",
    "H_TO_ANNUAL",
    "LOG_RETURN_COLUMN",
    "REALIZED_VOL_COLUMN",
    "Explanation",
    "annualize_mean",
    "annualize_vol",
    "idzorek_confidence",
    "log_returns",
    "non_finite_summary",
    "realized_vol",
]

# --------------------------------------------------------------------------- #
# Data-shape constants (one source of truth; consumers import, never redeclare)
# --------------------------------------------------------------------------- #

LOG_RETURN_COLUMN: Final[str] = "log_return"
"""Feature-frame column name for daily log-returns (consumers import this)."""

REALIZED_VOL_COLUMN: Final[str] = "realized_vol"
"""Feature-frame column name for annualized realized vol (consumers import this)."""

CONFIDENCE_FLOOR: Final[float] = 0.02
"""Lower clamp: keeps the Idzorek omega finite (c=0 would mean omega=1e6 in
pypfopt — an arbitrary sentinel, not math) while still yielding a posterior
that is prior-dominated to ~2 decimal places."""

CONFIDENCE_CEIL: Final[float] = 0.95
"""Upper clamp: c=1 would mean omega=0 (a view treated as certain truth); 0.95
caps any single forecast's pull on the posterior."""

H_TO_ANNUAL: Final[float] = TRADING_DAYS / HORIZON_BARS
"""H-day -> annual scaling for a MEAN log-return (x12 for H=21). Derived from
the shared constants; :func:`annualize_mean` is the one conversion helper that
applies it (``bl_blend`` routes its view derivation through that helper)."""


# --------------------------------------------------------------------------- #
# NaN-rejection helper
# --------------------------------------------------------------------------- #


def non_finite_summary(index: pd.Index, values: npt.NDArray[np.float64]) -> str:
    """Locate NaN/non-finite rows for an error message: count + first few labels.

    Shared by every NaN-rejection boundary (both functions here, plus
    ``model/base.py``) so the operator sees WHICH rows are bad — e.g. the date
    keys of NULL ``adj_close`` rows — instead of hand-querying SQLite.
    """
    bad_positions = np.flatnonzero(~np.isfinite(values))
    labels = ", ".join(repr(index[int(position)]) for position in bad_positions[:5])
    suffix = "" if len(bad_positions) <= 5 else f", ... (+{len(bad_positions) - 5} more)"
    return f"{len(bad_positions)} NaN/non-finite value(s) at rows [{labels}{suffix}]"


# --------------------------------------------------------------------------- #
# Features: daily log-returns + annualized realized volatility
# --------------------------------------------------------------------------- #


def log_returns(adj_close: pd.Series) -> pd.Series:
    """Natural-log daily returns of an adjusted-close series.

    Formula: ``r_t = ln(P_t / P_{t-1})``. Worked example: a close moving
    ``100 -> 101`` gives ``ln(1.01) = +0.995%``. The first observation has no
    predecessor and is DROPPED (never NaN), so the result has
    ``len(adj_close) - 1`` rows, indexed by ``adj_close.index[1:]`` and named
    :data:`LOG_RETURN_COLUMN`.

    Raises ``ValueError`` on fewer than 2 observations, on a non-ascending
    index (order-dependence must fail loud), on NaN/non-finite prices, and on
    non-positive prices (log-return undefined) — never emits NaN downstream.
    """
    values = adj_close.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError(f"need at least 2 adjusted closes to form returns, got {len(values)}")
    if not adj_close.index.is_monotonic_increasing:
        raise ValueError(
            "adjusted-close index must be sorted ascending (chronological order); "
            "log-returns over a disordered series would be silently wrong"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "adjusted closes contain NaN/non-finite values: "
            f"{non_finite_summary(adj_close.index, values)}; clean the series first"
        )
    if not (values > 0.0).all():
        raise ValueError("adjusted closes must be strictly positive (log-return undefined)")
    returns = np.log(values[1:] / values[:-1])
    return pd.Series(returns, index=adj_close.index[1:], name=LOG_RETURN_COLUMN)


def realized_vol(returns: pd.Series, window: int = HORIZON_BARS) -> pd.Series:
    """Rolling realized volatility of daily log-returns, annualized.

    Formula: sample std (``ddof=1``) over a trailing ``window`` of daily
    log-returns (default ``HORIZON_BARS`` = 21), annualized by
    ``sqrt(TRADING_DAYS)``. Worked example: a window whose daily std is 0.8%
    annualizes to ``0.008 * sqrt(252) = 12.7%``. Warm-up rows (the first
    ``window - 1``) are DROPPED, not NaN, so the result has
    ``len(returns) - window + 1`` rows, indexed by ``returns.index[window - 1:]``
    and named :data:`REALIZED_VOL_COLUMN`.

    Values are always >= 0; a constant-return window yields exactly 0.0 — a
    real property of the estimator (asserted in tests), so downstream
    consumers must not assume vol > 0.

    Raises ``ValueError`` when ``window < 2`` (a ``ddof=1`` sample std needs
    two points), when the series is shorter than one full window (the result
    would be silently empty), on a non-ascending index (rolling windows are
    order-dependent), or on NaN/non-finite returns.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 for a sample std (ddof=1), got {window}")
    values = returns.to_numpy(dtype=float)
    if not returns.index.is_monotonic_increasing:
        raise ValueError(
            "returns index must be sorted ascending (chronological order); "
            "a rolling window over a disordered series would be silently wrong"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "returns contain NaN/non-finite values: "
            f"{non_finite_summary(returns.index, values)}; clean the series first"
        )
    if len(values) < window:
        raise ValueError(
            f"need at least window={window} returns for one full window, got {len(values)}"
        )
    rolling_std = pd.Series(values, index=returns.index).rolling(window=window).std(ddof=1)
    annualized = rolling_std.iloc[window - 1 :] * math.sqrt(TRADING_DAYS)
    return annualized.rename(REALIZED_VOL_COLUMN)


# --------------------------------------------------------------------------- #
# Blend: the sigma -> Idzorek confidence map
# --------------------------------------------------------------------------- #


def idzorek_confidence(mu: float, sigma: float) -> float:
    """Map an H-day forecast ``(mu, sigma)`` to an Idzorek view confidence.

    Formula: ``c = clamp(|2*Phi(mu/sigma) - 1|, CONFIDENCE_FLOOR,
    CONFIDENCE_CEIL)``, computed as ``erf(|mu/sigma| / sqrt(2))`` (identical by
    symmetry of the normal CDF, so the map is even in ``mu`` by construction).
    ``|2*Phi(x) - 1|`` is the probability mass of a standard normal within
    ``|x|`` of zero — how sure the model is the H-day return shares ``mu``'s
    sign given predictive uncertainty ``sigma``. Worked example: a one-sigma
    signal ``|mu| == sigma`` gives ``erf(1/sqrt(2)) = 0.68`` — a moderately
    confident view; a no-information forecast (``mu ~ 0``) lands on the 0.02
    floor and Black-Litterman degrades to the market prior.

    Scale note: ``c`` depends only on the RATIO ``mu/sigma``, so it is
    invariant under any rescaling applied to BOTH inputs identically — but the
    H-day -> annual conversion is NOT such a rescaling (``mu`` scales by
    252/H, an independent-draws ``sigma`` by sqrt(252/H)), which is why
    ``bl_blend`` feeds this map the raw H-day pair (decision documented
    there).

    Properties (all pinned by tests):
    - ``mu = 0`` -> exactly ``CONFIDENCE_FLOOR`` (no-information forecast).
    - ``sigma -> 0+`` (or huge ``|mu|``, e.g. the "25 instead of 0.25" misuse)
      -> saturates at ``CONFIDENCE_CEIL``; never overflows or crashes
      (``erf(inf) == 1``).
    - ``sigma`` huge -> floor.
    - Symmetric in the sign of ``mu``.

    Raises ``ValueError`` on non-finite ``mu`` and on ``sigma <= 0`` or
    non-finite ``sigma`` (a ``Forecast`` can never carry those, but this map
    is also callable on raw floats and must fail loud on its own).
    """
    if not math.isfinite(mu):
        raise ValueError(f"mu must be finite, got {mu!r}")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and > 0 (H-day predictive std), got {sigma!r}")
    raw = math.erf(abs(mu / sigma) / math.sqrt(2.0))
    return min(max(raw, CONFIDENCE_FLOOR), CONFIDENCE_CEIL)


# --------------------------------------------------------------------------- #
# Annualization helpers (the H-day -> annual boundary, one source of truth)
# --------------------------------------------------------------------------- #


def annualize_mean(mu_h: float) -> float:
    """Annualize an H-day mean log-return.

    Formula: ``mu_h * TRADING_DAYS / HORIZON_BARS`` (= ``mu_h * H_TO_ANNUAL``,
    x12 for H=21). Worked example: ``mu_h = +1%`` over 21 days annualizes to
    ``0.01 * 252/21 = +12%``. This is THE H-day -> annual boundary for a mean
    (``bl_blend`` derives its absolute view Q through it); nothing else
    converts a mean's horizon.
    """
    return mu_h * H_TO_ANNUAL


def annualize_vol(daily_std: float) -> float:
    """Annualize a daily return standard deviation.

    Formula: ``daily_std * sqrt(TRADING_DAYS)`` (the sqrt-of-time rule, x~15.87
    for 252 trading days). Worked example: a 0.8% daily std annualizes to
    ``0.008 * sqrt(252) = 12.7%`` — the same scaling :func:`realized_vol`
    applies to its rolling window std.
    """
    return daily_std * math.sqrt(TRADING_DAYS)


# --------------------------------------------------------------------------- #
# Explanation registry: structured, UI-servable formula + worked example
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Explanation:
    """One legible explanation of a quantity a stage card displays.

    ``formula`` is the plain math (a composite quantity's real linear algebra
    stays in pypfopt/cvxpy — the string is descriptive); ``example`` is an
    intuitive worked number. ``unit`` and ``window`` are optional context
    (e.g. "annualized excess return", "21-day horizon"). ``frozen=True`` +
    ``slots=True`` keep entries immutable and cheap; :meth:`payload` renders
    the JSON shape the ``GET /api/explain`` route serves.
    """

    key: str
    label: str
    formula: str
    description: str
    example: str
    unit: str | None = None
    window: str | None = None

    def payload(self) -> dict[str, str | None]:
        """The JSON member ``GET /api/explain`` serves for this key (no ``key``
        — the registry dict already keys by it)."""
        return {
            "label": self.label,
            "formula": self.formula,
            "description": self.description,
            "example": self.example,
            "unit": self.unit,
            "window": self.window,
        }


def _registry(*entries: Explanation) -> dict[str, Explanation]:
    """Build the key -> Explanation map, failing loud on a duplicate key."""
    registry: dict[str, Explanation] = {}
    for entry in entries:
        if entry.key in registry:
            raise ValueError(f"duplicate Explanation key {entry.key!r}")
        registry[entry.key] = entry
    return registry


EXPLANATIONS: Final[dict[str, Explanation]] = _registry(
    Explanation(
        key="log_return",
        label="Daily log return",
        formula="r_t = ln(P_t / P_{t-1})",
        description=(
            "Natural log of the ratio of consecutive adjusted closes. Additive across "
            "time and symmetric around zero, so it is the clean input to every "
            "downstream estimator."
        ),
        example="100 -> 101 => ln(1.01) = +0.995%",
        unit="log-return (fraction)",
        window="1 day",
    ),
    Explanation(
        key="realized_vol",
        label="Realized volatility",
        formula="sigma_ann = stdev(r over trailing 21d, ddof=1) * sqrt(252)",
        description=(
            "Sample standard deviation of the trailing 21-day daily log-returns, "
            "annualized by sqrt(252). Measures recent dispersion of what actually "
            "happened — NOT a forecast."
        ),
        example="daily std 0.8% => 0.008 * sqrt(252) = 12.7% annualized",
        unit="annualized volatility (fraction)",
        window="21 days, annualized",
    ),
    Explanation(
        key="ewma_mu",
        label="EWMA forecast mean (H-day)",
        formula="mu_H = EWMA(daily r; halflife=63) * 21",
        description=(
            "Expanding exponentially-weighted mean of daily log-returns (about a "
            "63-day halflife) evaluated at the latest bar, then scaled to the 21-day "
            "horizon. This is the EWMA baseline's mu forecast."
        ),
        example="daily EWMA +0.05%/day => mu_H = 0.0005 * 21 = +1.05% over 21 days",
        unit="H-day log-return",
        window="21-day horizon",
    ),
    Explanation(
        key="forecast_sigma",
        label="Forecast uncertainty (H-day predictive std)",
        formula="sigma_H = stdev(forecast_t - realized_t over history, ddof=1)",
        description=(
            "Standard deviation of the model's trailing H-day forecast errors — the "
            "scale at which mu +/- 1.64*sigma covers ~90% of realized 21-day returns. "
            "This is the uncertainty of the forecast itself, not realized volatility."
        ),
        example="forecast errors std 3% => sigma_H = 0.03; a +1% forecast spans ~ -4% to +6%",
        unit="H-day log-return std",
        window="21-day horizon",
    ),
    Explanation(
        key="bl_prior",
        label="Equilibrium prior (pi)",
        formula="pi = delta * Sigma * w_mkt",
        description=(
            "Black-Litterman implied-equilibrium excess returns: the returns the "
            "market must expect in order to hold the fixed 30/30/40 SPY/ACWI/AGG "
            "benchmark, given risk aversion delta and covariance Sigma. The neutral "
            "starting point before any forecast view is applied."
        ),
        example="delta=2.5, benchmark 30/30/40 => e.g. SPY prior ~ +5%/yr, AGG ~ +1%/yr",
        unit="annualized excess return",
    ),
    Explanation(
        key="bl_view",
        label="Forecast view (Q)",
        formula="Q = mu_H * 252 / 21",
        description=(
            "The model's forecast expressed as an annualized absolute view on each "
            "asset — the 21-day mu scaled to a yearly rate. One absolute BL view is "
            "formed per forecast asset."
        ),
        example="mu_H = +1% over 21 days => Q = 0.01 * 252/21 = +12%/yr",
        unit="annualized excess return",
    ),
    Explanation(
        key="bl_confidence",
        label="View confidence (Idzorek c)",
        formula="c = clamp(|2*Phi(mu/sigma) - 1|, 0.02, 0.95)",
        description=(
            "How strongly each forecast view pulls the posterior away from the prior, "
            "read from the raw H-day signal-to-noise ratio mu/sigma. 0.02 = essentially "
            "ignored (posterior ~ prior); 0.95 = treated as near-certain. A "
            "no-information forecast (mu ~ 0) lands on the floor."
        ),
        example="mu == sigma (one-sigma signal) => c = erf(1/sqrt(2)) = 0.68",
    ),
    Explanation(
        key="bl_posterior",
        label="Black-Litterman posterior",
        formula="mu_post = BL(prior pi, views Q, Omega(c))",
        description=(
            "The confidence-weighted blend of the equilibrium prior pi and the forecast "
            "views Q. Each view tilts the posterior toward Q by an amount proportional "
            "to its confidence c (via the Idzorek omega); the linear algebra runs in "
            "PyPortfolioOpt."
        ),
        example="prior SPY +5%/yr, view +12%/yr at c=0.2 => posterior ~ +6.4%/yr",
        unit="annualized excess return",
    ),
    Explanation(
        key="mvu_objective",
        label="Mean-variance-utility optimization",
        formula=(
            "maximize w^T mu - (delta/2) w^T Sigma w - gamma*||w - w_prev||_1  "
            "s.t. sum(w)=1, 0<=w<=w_max"
        ),
        description=(
            "The final allocation maximizes expected posterior return minus a risk "
            "penalty (delta) and an L1 turnover penalty (gamma) against the last "
            "weights, long-only and capped at w_max per asset. Solved with "
            "cvxpy/CLARABEL."
        ),
        example="delta=2.5, gamma=0.002, w_max=0.60 => weights sum to 1, no asset above 60%",
    ),
)
