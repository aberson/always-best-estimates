"""The frozen ``WorldModel`` interface + ``Forecast`` + the EWMA baseline.

plan.md section 4 (Core Pipeline — "The WorldModel contract") is the
authoritative spec. This step (plan Step 5) FREEZES the interface with a
contract test (``tests/test_model_base.py::assert_worldmodel_contract``) that
every implementation must pass — the EWMA baseline now, Step 13's JEPA later,
through the SAME function.

The contract, for every implementation:

- ``forecast(features)`` maps ``{asset: features frame}`` to
  ``{asset: Forecast}`` for EXACTLY the input assets. Each input frame carries
  at least the ``log_return`` column (richer frames arrive with Step 12's
  ``build.py``); implementations must tolerate extra columns.
- ``Forecast.mu`` is the H-day log-return mean forecast over
  H = ``HORIZON_BARS`` = 21 trading bars — NOT daily, NOT annualized. The
  blend layer converts units through the shared constants; unit/horizon drift
  is the top silent-bug class here (plan section 8).
- ``Forecast.sigma`` is the H-DAY PREDICTIVE FORECAST STD — the uncertainty
  scale at which ``mu ± 1.64 * sigma`` should cover ~90% of realized H-day
  log-returns (plan section 4's pre-registered calibration gate, a
  predictive-interval property). The full semantic — including the scales an
  implementation must NOT emit — is defined once on :class:`Forecast`.
  ``sigma`` must be strictly positive and finite: degenerate inputs raise
  ``ValueError``, never emit sigma=0/NaN.
- ``model_version`` is a non-empty label stored in the ``forecasts`` table
  (``"ewma"``, ``"jepa:<hash>"``).
"""

import math
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from abe.constants import HORIZON_BARS
from abe.features.basic import LOG_RETURN_COLUMN, non_finite_summary

__all__ = [
    "DEFAULT_HALFLIFE",
    "ERROR_WARMUP_BARS",
    "MIN_ERROR_SAMPLES",
    "MIN_HISTORY_BARS",
    "EWMABaseline",
    "Forecast",
    "WorldModel",
]

DEFAULT_HALFLIFE: Final[float] = 63.0
"""Default EWMA halflife in trading days (~one quarter): slow enough to be a
stable mean estimate, fast enough to track regime drift."""

ERROR_WARMUP_BARS: Final[int] = HORIZON_BARS
"""Bars skipped at the start of the forecast-error walk: the expanding EWMA
there has seen less than one forecast horizon of data, so its errors would
measure initialization noise, not the model."""

MIN_ERROR_SAMPLES: Final[int] = 10
"""Minimum forecast-error samples for a meaningful std — a 2-sample ``ddof=1``
std is technically defined but statistically garbage."""

MIN_HISTORY_BARS: Final[int] = ERROR_WARMUP_BARS + HORIZON_BARS + MIN_ERROR_SAMPLES
"""Minimum daily log-return bars ``EWMABaseline.forecast`` accepts (= 52):
warm-up + one unobservable trailing horizon + ``MIN_ERROR_SAMPLES`` observable
errors."""


@dataclass(frozen=True, slots=True)
class Forecast:
    """One asset's H-day forecast: ``(mu, sigma)`` over H = ``HORIZON_BARS`` bars.

    ``mu`` — mean H-day log-return forecast (H-day scale, NOT daily/annual).

    ``sigma`` — the H-DAY PREDICTIVE FORECAST STD: the uncertainty scale at
    which ``mu ± 1.64 * sigma`` should cover ~90% of realized H-day
    log-returns. That coverage property IS plan section 4's pre-registered
    calibration gate, so this predictive-interval definition is the operative
    contract semantic. Every implementation must emit sigma on THIS scale:

    - Step 13's JEPA (ensemble disagreement + purged walk-forward residual
      variance composite) MUST land its composite on the same predictive
      H-day scale, or Step 14's promotion comparison is invalid — the eval
      compares sigma calibration across models on identical windows.
    - Classic failure modes to avoid: emitting raw return VARIANCE (sigma^2 —
      ~20-40x too large in units) or a daily/annualized std. Both break the
      Idzorek confidence map ``c = clamp(|2*Phi(mu/sigma) - 1|, 0.02, 0.95)``.
    - With predictive sigma, realistic EWMA drift yields modest Idzorek
      confidence (c ~ 0.1-0.2). That is the intended graceful
      weakly-informative behavior, not a bug.

    Always strictly positive and finite — construction rejects anything else,
    so a ``Forecast`` that exists is safe to feed the Step 6 confidence map.

    Values are coerced to native ``float`` at construction (numpy scalars in,
    Python floats out — the storage boundary and the API want natives).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu", float(self.mu))
        object.__setattr__(self, "sigma", float(self.sigma))
        if not math.isfinite(self.mu):
            raise ValueError(f"Forecast.mu must be finite, got {self.mu!r}")
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError(
                f"Forecast.sigma must be finite and > 0 (predictive H-day std), got {self.sigma!r}"
            )


@runtime_checkable
class WorldModel(Protocol):
    """The pluggable forecaster seam (plan sections 4-5), frozen in Step 5.

    Implementations: :class:`EWMABaseline` (the default), Step 13's JEPA
    (behind a config toggle). Both must pass
    ``tests/test_model_base.py::assert_worldmodel_contract`` unchanged.
    ``runtime_checkable`` so the pipeline can structurally assert its wired
    forecaster at startup (attribute/method presence only — signatures are
    enforced by mypy and the contract test).

    ``model_version`` is the non-empty forecaster label stored in the
    ``forecasts`` table. ``forecast`` maps ``{asset: features frame}`` to
    ``{asset: Forecast}`` with the SAME keys; ``Forecast.sigma`` must be the
    H-day PREDICTIVE forecast std — the exact scale defined on
    :class:`Forecast` (emitting a variance or a daily/annualized std breaks
    the Idzorek confidence map and invalidates the Step 14 comparison).
    """

    model_version: str

    def forecast(self, features: dict[str, pd.DataFrame]) -> dict[str, Forecast]:
        """Map ``{asset: features frame}`` -> ``{asset: Forecast}`` (same keys)."""
        ...


class EWMABaseline:
    """EWMA-of-returns baseline: the default ``WorldModel`` (plan Step 5).

    ``mu_H``: expanding EWMA of daily log-returns (halflife ``halflife``
    trading days; ``adjust=True`` so the estimate is a proper weighted mean
    over the full history) evaluated at the last bar, scaled by
    ``HORIZON_BARS`` — a daily mean forecast scaled to the H-day horizon.

    ``sigma_H`` — trailing forecast-error std: the plan-sanctioned baseline
    estimator of the H-day PREDICTIVE forecast std (the semantic frozen on
    :class:`Forecast`; Step 13's JEPA must land its ensemble-disagreement +
    walk-forward-residual composite on the SAME predictive scale):

    1. Walk the available history. At each past bar t compute the forecast the
       model WOULD have issued — EWMA over ``returns[0..t]`` times H
       (``ewm().mean()`` at position t weights only data <= t, so the walk has
       no lookahead by construction) — and the realized H-day log-return over
       ``(t, t+H]`` (``rolling(H).sum()`` shifted back H bars).
    2. Collect errors ``e_t = forecast_t - realized_t`` over the whole
       available history (expanding collection), skipping the first
       :data:`ERROR_WARMUP_BARS` bars (initialization noise) and the last
       ``HORIZON_BARS`` bars (their realized return is not yet observable).
    3. ``sigma = std(e_t, ddof=1)``. The errors are H-day quantities, so
       ``sigma`` is already on ``mu``'s H-day scale — no further scaling. By
       construction sigma is the historical dispersion of (forecast -
       realized), which is exactly the scale plan section 4's calibration
       gate checks (``mu ± 1.64 * sigma`` covering ~90% of realized H-day
       returns). On iid daily noise with std s it lands near ``s * sqrt(H)``
       (pinned by test).

    Degenerate inputs fail LOUD (never sigma=0/NaN): a missing ``log_return``
    column, a non-ascending index, NaN/non-finite returns, history shorter
    than :data:`MIN_HISTORY_BARS`, or a constant series (all forecast errors
    identical up to float noise, so no honest std exists) raise ``ValueError``.
    """

    model_version: str = "ewma"

    def __init__(self, halflife: float = DEFAULT_HALFLIFE) -> None:
        if not math.isfinite(halflife) or halflife <= 0.0:
            raise ValueError(f"halflife must be a finite positive bar count, got {halflife!r}")
        self._halflife = float(halflife)

    def forecast(self, features: dict[str, pd.DataFrame]) -> dict[str, Forecast]:
        """See :meth:`WorldModel.forecast`; per-asset math in the class docstring."""
        if not features:
            raise ValueError("features must contain at least one asset frame")
        return {asset: self._forecast_asset(asset, frame) for asset, frame in features.items()}

    def _forecast_asset(self, asset: str, frame: pd.DataFrame) -> Forecast:
        if LOG_RETURN_COLUMN not in frame.columns:
            raise ValueError(
                f"features frame for {asset!r} lacks required column {LOG_RETURN_COLUMN!r}; "
                f"got columns {list(frame.columns)!r}"
            )
        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                f"features frame index for {asset!r} must be sorted ascending "
                "(chronological order); the EWMA walk over a disordered series would "
                "be silently wrong"
            )
        values = frame[LOG_RETURN_COLUMN].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"log-returns for {asset!r} contain NaN/non-finite values: "
                f"{non_finite_summary(frame.index, values)}"
            )
        n = len(values)
        if n < MIN_HISTORY_BARS:
            raise ValueError(
                f"history for {asset!r} too short: {n} bars < MIN_HISTORY_BARS="
                f"{MIN_HISTORY_BARS} (warm-up {ERROR_WARMUP_BARS} + horizon {HORIZON_BARS} "
                f"+ {MIN_ERROR_SAMPLES} error samples)"
            )
        if float(np.ptp(values)) == 0.0:
            raise ValueError(
                f"log-returns for {asset!r} are constant — forecast errors carry no "
                "information, so no strictly positive predictive sigma can be estimated"
            )
        returns = pd.Series(values)
        forecast_h = returns.ewm(halflife=self._halflife, adjust=True).mean() * HORIZON_BARS
        realized_h = returns.rolling(window=HORIZON_BARS).sum().shift(-HORIZON_BARS)
        errors = (forecast_h - realized_h).iloc[ERROR_WARMUP_BARS : n - HORIZON_BARS]
        sigma = float(errors.std(ddof=1))
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"degenerate history for {asset!r}: trailing forecast-error std is "
                f"{sigma!r} — a Forecast must carry strictly positive predictive sigma"
            )
        return Forecast(mu=float(forecast_h.iloc[-1]), sigma=sigma)
