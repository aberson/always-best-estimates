"""Pluggable Black-Litterman view sources (Track 2 Step 22).

A view source turns a run's context into ``{asset: Forecast}`` — the SAME shape
``bl_blend`` already consumes (it derives the absolute view ``Q`` via
``annualize_mean(mu)`` and the Idzorek confidence via ``idzorek_confidence(mu,
sigma)``). So the blend math is UNTOUCHED regardless of where the views come from:

- **forecast** (V1): the model's forecasts, unchanged — identity on
  ``ctx.forecasts``. The central config uses this; Step 19's parity golden pins it.
- **historical**: absolute views from a chosen past window's realized returns.
  ``mu_H = mean(daily) * H``; ``sigma_H = std(daily) * sqrt(H)`` (the iid H-day
  dispersion). A window with clear positive drift yields a positive, higher-
  confidence view; the view's SIGN matches the window's mean return.
- **counterfactual**: hand-authored absolute views. The payload gives a TARGET
  annual return + confidence per asset; the source INVERTS the two boundary maps
  so ``bl_blend`` re-derives EXACTLY those:
  ``mu_H = annual * H / TRADING_DAYS`` (inverse of ``annualize_mean``) and
  ``sigma_H = |mu_H| / (sqrt(2) * erfinv(clamp(c)))`` (inverse of
  ``idzorek_confidence`` = ``erf(|mu/sigma| / sqrt(2))``). Verified to round-trip.

Partial views (K < N assets) are fine — ``bl_blend`` blends whatever subset it
gets, and the plan's Idzorek clamp bounds each view's influence.
"""

import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

import pandas as pd
from scipy.special import erfinv

from abe.calc import CONFIDENCE_CEIL, CONFIDENCE_FLOOR
from abe.constants import HORIZON_BARS, TRADING_DAYS, UNIVERSE
from abe.model.base import Forecast

__all__ = [
    "LIBRARY",
    "CounterfactualViewSource",
    "ForecastViewSource",
    "HistoricalViewSource",
    "ViewContext",
    "ViewSource",
    "counterfactual_view_factory",
    "forecast_view_factory",
    "historical_view_factory",
    "seed_library",
]

# Forecast.sigma must be strictly positive; floor a degenerate (zero-dispersion
# / zero-mu) view's sigma so a Forecast can still be constructed.
_SIGMA_FLOOR: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class ViewContext:
    """The per-run data a view source needs. ``forecasts`` are the run's model
    forecasts (used by ``forecast``); ``returns`` are per-asset log-return Series
    (used by ``historical``). A source captures its OWN scenario payload at
    construction (the factory takes it), not here."""

    forecasts: dict[str, Forecast]
    returns: dict[str, pd.Series] = field(default_factory=dict)


class ViewSource(Protocol):
    kind: str

    def provide(self, ctx: ViewContext) -> dict[str, Forecast]:
        """Produce ``{asset: Forecast}`` (H-day pairs) for ``bl_blend``."""
        ...


class ForecastViewSource:
    """``forecast``: views ARE the model's forecasts (the V1 behavior)."""

    kind = "forecast"

    def provide(self, ctx: ViewContext) -> dict[str, Forecast]:
        return dict(ctx.forecasts)


class HistoricalViewSource:
    """``historical``: absolute views from a past window's realized returns.

    ``payload``: ``{"window_start": <ISO date>, "window_end": <ISO date>}`` (both
    optional; missing bound = open-ended, so ``{}`` uses all available history).
    ISO date strings sort chronologically, so the slice is a plain string range.
    """

    kind = "historical"

    def __init__(self, payload: Mapping[str, object]) -> None:
        start = payload.get("window_start")
        end = payload.get("window_end")
        self._start = str(start) if start is not None else None
        self._end = str(end) if end is not None else None

    def _window(self, series: pd.Series) -> pd.Series:
        sliced = series
        if self._start is not None:
            sliced = sliced[sliced.index >= self._start]
        if self._end is not None:
            sliced = sliced[sliced.index <= self._end]
        return sliced

    def provide(self, ctx: ViewContext) -> dict[str, Forecast]:
        views: dict[str, Forecast] = {}
        for asset in UNIVERSE:
            series = ctx.returns.get(asset)
            if series is None:
                continue
            window = self._window(series)
            if len(window) < 2:  # need ≥2 points for a std
                continue
            mean_daily = float(window.mean())
            std_daily = float(window.std(ddof=1))
            mu_h = mean_daily * HORIZON_BARS
            sigma_h = max(std_daily * math.sqrt(HORIZON_BARS), _SIGMA_FLOOR)
            if not math.isfinite(mu_h) or not math.isfinite(sigma_h):
                continue
            views[asset] = Forecast(mu=mu_h, sigma=sigma_h)
        return views


class CounterfactualViewSource:
    """``counterfactual``: hand-authored absolute views (e.g. "SPY +10%").

    ``payload``: ``{asset: {"mu": <annual return>, "confidence": <0..1>}}``.
    Confidence defaults to 0.5 and is clamped to the Idzorek bounds. Only the
    listed assets get a view (partial views allowed)."""

    kind = "counterfactual"

    def __init__(self, payload: Mapping[str, object]) -> None:
        specs: dict[str, tuple[float, float]] = {}
        for asset, raw in payload.items():
            if asset not in UNIVERSE:
                raise ValueError(f"counterfactual view for {asset!r} not in UNIVERSE {UNIVERSE}")
            if not isinstance(raw, Mapping) or "mu" not in raw:
                raise ValueError(
                    f"counterfactual view for {asset!r} must be a mapping with a 'mu' "
                    f"(annual return); got {raw!r}"
                )
            mu_annual = float(raw["mu"])
            confidence = float(raw.get("confidence", 0.5))
            specs[asset] = (mu_annual, confidence)
        self._specs = specs

    def provide(self, ctx: ViewContext) -> dict[str, Forecast]:
        views: dict[str, Forecast] = {}
        for asset, (mu_annual, confidence) in self._specs.items():
            mu_h = mu_annual * HORIZON_BARS / TRADING_DAYS  # inverse of annualize_mean
            clamped = min(max(confidence, CONFIDENCE_FLOOR), CONFIDENCE_CEIL)
            # inverse of idzorek_confidence: c = erf(|mu/sigma|/sqrt(2))
            z = math.sqrt(2.0) * float(erfinv(clamped))
            sigma_h = max(abs(mu_h) / z, _SIGMA_FLOOR) if z > 0.0 else _SIGMA_FLOOR
            views[asset] = Forecast(mu=mu_h, sigma=sigma_h)
        return views


def forecast_view_factory(_payload: Mapping[str, object]) -> ViewSource:
    return ForecastViewSource()


def historical_view_factory(payload: Mapping[str, object]) -> ViewSource:
    return HistoricalViewSource(payload)


def counterfactual_view_factory(payload: Mapping[str, object]) -> ViewSource:
    return CounterfactualViewSource(payload)


# --------------------------------------------------------------------------- #
# Pre-programmed library — ready-to-apply example scenarios (plan §5 / §7 S22)
# --------------------------------------------------------------------------- #

LIBRARY: Final[tuple[dict[str, object], ...]] = (
    {
        "name": "SPY +10% (bullish equity)",
        "kind": "counterfactual",
        "payload": {"SPY": {"mu": 0.10, "confidence": 0.5}},
    },
    {
        "name": "AGG +3% (bond rally)",
        "kind": "counterfactual",
        "payload": {"AGG": {"mu": 0.03, "confidence": 0.4}},
    },
    {
        "name": "Risk-off (equities down, bonds up)",
        "kind": "counterfactual",
        "payload": {
            "SPY": {"mu": -0.15, "confidence": 0.5},
            "ACWI": {"mu": -0.12, "confidence": 0.5},
            "AGG": {"mu": 0.04, "confidence": 0.4},
        },
    },
    {
        "name": "Trailing full history (realized)",
        "kind": "historical",
        "payload": {},
    },
)
"""Ready-to-apply example view scenarios (counterfactual + historical). Seeded on
demand via :func:`seed_library`; the Step 28 authoring UI browses these."""


def seed_library(conn: sqlite3.Connection) -> list[int]:
    """Create any LIBRARY scenarios not already present (matched by name); return
    the resulting view_scenario_ids in LIBRARY order. Idempotent.

    ``config`` is imported lazily so this module has no import-time dependency on
    the storage layer (it is imported by the registry, which config does not
    depend on)."""
    from abe import config as config_module

    existing = {vs.name: vs.view_scenario_id for vs in config_module.list_view_scenarios(conn)}
    ids: list[int] = []
    for entry in LIBRARY:
        name = str(entry["name"])
        if name in existing:
            ids.append(existing[name])
            continue
        payload = entry["payload"]
        assert isinstance(payload, Mapping)
        created = config_module.create_view_scenario(
            conn, name=name, kind=str(entry["kind"]), payload=payload
        )
        ids.append(created.view_scenario_id)
    return ids
