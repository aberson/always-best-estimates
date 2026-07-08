"""Pre-registered JEPA-vs-EWMA purged walk-forward evaluation gate (plan.md Step 14).

============================================================================
PRE-REGISTRATION (written BEFORE any result was looked at)
============================================================================

These numbers — per-asset and pooled forecast **MSE** (mu vs the realized H-day
log-return), **information coefficient** (Spearman rank correlation of mu vs the
realized return across time, per asset), and **sigma calibration coverage**
(fraction of ``|realized - mu| <= 1.64 * sigma``; target ~= 0.90) — are computed
from ``(mu, sigma)`` produced **through the production ``WorldModel.forecast``
interface** (never a bench-only path) on **identical purged walk-forward windows**
over the cached joint SPY/ACWI/AGG panel. They decide whether the JEPA replaces
the EWMA baseline as the live forecaster.

**Promotion rule (mechanical — no judgment call at read time):** the JEPA is
promoted **ONLY IF**

1. JEPA's pooled MSE is strictly lower than EWMA's pooled MSE, **AND**
2. JEPA's pooled coverage is within +-5 percentage points of 0.90
   (``|coverage - 0.90| <= 0.05``), **AND**
3. JEPA's coverage is no worse than EWMA's distance from 0.90
   (``|coverage_jepa - 0.90| <= |coverage_ewma - 0.90|``).

Otherwise **EWMA remains the default**. A tie on MSE (not strictly lower) leaves
EWMA the default — the baseline wins ties by construction (plan section 1: V1 is
never gated on the JEPA beating the baseline; ~4,600 joint bars make parity the
realistic best case). The exact same rule text drives :func:`decide_promotion`
and the rendered report, so the decision is reproducible from the numbers with no
human judgment.

Pooled-metric semantics: pooled MSE (decision clause 1) is EQUAL-POINT-WEIGHTED
across assets — the higher-variance SPY/ACWI errors dominate its magnitude and
AGG contributes little. Pooled IC is Spearman over the concatenated cross-asset
series — a scale artifact reported for context ONLY; it is never used by the
promotion rule.

============================================================================
How the measurement stays valid (dev-workspace measurement-validity rule)
============================================================================

- **Production code path, not a proxy.** Both models are scored through their real
  ``.forecast()`` (the exact interface the pipeline calls); the JEPA is trained via
  the production ``train_jepa`` on each fold's training slice only. Nothing here
  re-implements a forecaster or hand-builds a prompt — the eval measures the system
  that ships.
- **Production-shaped inputs.** At every evaluation time ``t`` both models receive
  the SAME per-asset ``{asset: log_return frame}`` dict, where each asset's frame is
  built from that asset's OWN FULL cached history (mirroring the production
  ``pipeline._stage_features``, which feeds each asset its complete ``adj_close``
  series — SPY from 1993, not the ACWI-limited joint panel) truncated to dates
  ``<= t``. The joint (inner-join) panel is used ONLY for fold geometry and
  realized-return alignment. This matters for EWMA: its sigma is the forecast-error
  std over the WHOLE input frame, so a panel-truncated frame would emit a sigma the
  live EWMA never produces — and EWMA's coverage gates promotion clauses 2 and 3.
- **No leakage.** Folds come from :func:`~abe.afml.purged_cv.purged_walk_forward_splits`
  (an H-bar purge) and are re-checked by
  :func:`~abe.afml.purged_cv.validate_no_leakage`; the JEPA is trained only on prices
  ``<= last_train_date`` so no training label's horizon reaches a test bar.
- **The gate is the eval EXISTING, not JEPA winning.** The CLI writes the report and
  exits 0 regardless of who wins.

============================================================================
Promotion is a MANUAL operator action
============================================================================

This module only RECORDS the decision in the committed report. It never touches the
scheduler or the API. Promoting the JEPA is a deliberate operator step: set
``ABE_MODEL=jepa`` + ``ABE_JEPA_CHECKPOINT=<path>`` (plan Step 13 toggle;
``resolve_startup_model``). The default stays EWMA until an operator flips it.

CLI::

    uv run python -m abe.eval.walk_forward --db data/abe.db \\
        --out docs/eval/jepa-vs-ewma-2026-07-08.md [--splits N] [--quick]

``--quick`` uses a smaller (ci-like) JEPA config + fewer splits for bounded runtime;
the default aims at the full joint panel with the production ``JEPAConfig`` and must
stay CPU-feasible (the net is ~41.8k params; expect a few minutes on ~4,600 bars).
"""

import argparse
import dataclasses
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

from abe.afml.purged_cv import purged_walk_forward_splits, validate_no_leakage
from abe.calc import LOG_RETURN_COLUMN, log_returns
from abe.constants import HORIZON_BARS, UNIVERSE
from abe.features.build import build_features
from abe.ingest.sources import CacheAdapter
from abe.model.base import EWMABaseline, Forecast, WorldModel
from abe.model.jepa import JEPAConfig
from abe.model.train import train_jepa

__all__ = [
    "CALIBRATION_Z",
    "COVERAGE_TARGET",
    "COVERAGE_TOLERANCE_PP",
    "EWMA_REMAINS_DEFAULT",
    "JEPA_PROMOTED",
    "EvalReport",
    "EwmaFactory",
    "FoldInfo",
    "JepaFactory",
    "ModelMetrics",
    "coverage",
    "decide_promotion",
    "eval_positions",
    "forecast_mse",
    "information_coefficient",
    "load_prices_from_cache",
    "main",
    "production_ewma_factory",
    "production_jepa_factory",
    "quick_config",
    "realized_horizon_return",
    "render_report_markdown",
    "run_walk_forward_eval",
]

logger = logging.getLogger(__name__)

CALIBRATION_Z: Final[float] = 1.64
"""The multiplier in the calibration gate ``mu +- z * sigma`` — plan section 4's
stated ~90% two-sided coverage multiplier (the true two-sided 90% normal quantile
is 1.6449; the plan rounds it to 1.64 and this eval uses the plan's value)."""

COVERAGE_TARGET: Final[float] = 0.90
"""The calibration coverage the gate targets (plan section 4)."""

COVERAGE_TOLERANCE_PP: Final[float] = 0.05
"""Half-width of the coverage band around :data:`COVERAGE_TARGET` (+-5 percentage
points) inside which the JEPA's coverage must land to be promotable."""

JEPA_PROMOTED: Final[str] = "JEPA promoted"
"""Decision string when the mechanical rule promotes the JEPA — one of exactly two
outcomes (plan Step 14 done-when); its counterpart is :data:`EWMA_REMAINS_DEFAULT`."""

EWMA_REMAINS_DEFAULT: Final[str] = "EWMA remains default"
"""Decision string when EWMA stays the live forecaster — the outcome for any failed
clause AND for every tie (the baseline's default bias)."""

JepaFactory = Callable[[dict[str, pd.Series], JEPAConfig], WorldModel]
"""Train a JEPA on a fold's training-price slice and return it as a WorldModel.
The production impl (:func:`production_jepa_factory`) routes through ``train_jepa``;
tests inject a fake to exercise the loop mechanics without real training."""

EwmaFactory = Callable[[dict[str, pd.Series]], WorldModel]
"""Return the EWMA WorldModel for a fold (EWMA needs no training — the production
impl just constructs a fresh :class:`~abe.model.base.EWMABaseline`)."""


# --------------------------------------------------------------------------- #
# Pre-registered metrics (pure, unit-tested on hand-computed values)
# --------------------------------------------------------------------------- #


def forecast_mse(mu: Sequence[float] | npt.NDArray[np.float64], realized: Sequence[float]) -> float:
    """Mean squared error of the H-day mean forecast vs the realized H-day return."""
    mu_a = np.asarray(mu, dtype=np.float64)
    realized_a = np.asarray(realized, dtype=np.float64)
    if mu_a.size == 0:
        return float("nan")
    errors = mu_a - realized_a
    return float(np.mean(errors * errors))


def information_coefficient(
    mu: Sequence[float] | npt.NDArray[np.float64], realized: Sequence[float]
) -> float:
    """Spearman rank correlation of ``mu`` vs ``realized`` (the information coefficient).

    Returns NaN when fewer than 2 points or either series is constant (rank
    correlation is undefined there) — never raises, so a degenerate asset does
    not sink the whole report.
    """
    mu_a = np.asarray(mu, dtype=np.float64)
    realized_a = np.asarray(realized, dtype=np.float64)
    if mu_a.size < 2 or np.ptp(mu_a) == 0.0 or np.ptp(realized_a) == 0.0:
        return float("nan")
    return float(spearmanr(mu_a, realized_a).statistic)


def coverage(
    mu: Sequence[float] | npt.NDArray[np.float64],
    sigma: Sequence[float] | npt.NDArray[np.float64],
    realized: Sequence[float],
    z: float = CALIBRATION_Z,
) -> float:
    """Fraction of points with ``|realized - mu| <= z * sigma`` (the calibration gate)."""
    mu_a = np.asarray(mu, dtype=np.float64)
    sigma_a = np.asarray(sigma, dtype=np.float64)
    realized_a = np.asarray(realized, dtype=np.float64)
    if mu_a.size == 0:
        return float("nan")
    within = np.abs(realized_a - mu_a) <= z * sigma_a
    return float(np.mean(within))


# --------------------------------------------------------------------------- #
# Walk-forward geometry (pure, unit-tested)
# --------------------------------------------------------------------------- #


def eval_positions(
    test_idx: npt.NDArray[np.int_], horizon: int, stride: int, n_panel: int
) -> list[int]:
    """Strided evaluation context-end positions inside one test window.

    Walk the test window from its first bar in ``stride`` steps; keep only bars
    ``t`` whose full future horizon ``(t, t+horizon]`` still fits in the panel
    (``t + horizon <= n_panel - 1``). ``stride == horizon`` (the default) gives
    NON-OVERLAPPING realized windows WITHIN a fold — consecutive eval points in
    one test window are H bars apart, so their ``(t, t+H]`` spans do not overlap.
    ACROSS a fold boundary the last window of fold ``i`` and the first of fold
    ``i+1`` can overlap by up to ``H - 1`` bars (test windows tile contiguously
    and each fold's stride restarts at its own first test bar). That is a
    negligible weighting effect on point estimates, not leakage — realized
    returns involve no training data.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if test_idx.size == 0:
        return []
    first = int(test_idx[0])
    last = int(test_idx[-1])
    return [t for t in range(first, last + 1, stride) if t + horizon <= n_panel - 1]


def realized_horizon_return(daily_returns: npt.NDArray[np.float64], t: int, horizon: int) -> float:
    """Realized H-day log-return over the span ``(t, t+horizon]`` == positions ``t+1..t+H``.

    A forecast issued at bar ``t`` is scored against exactly the future the model
    could not see: the sum of daily log-returns at positions ``t+1`` through
    ``t+horizon`` inclusive.
    """
    return float(daily_returns[t + 1 : t + horizon + 1].sum())


# --------------------------------------------------------------------------- #
# Mechanical promotion rule (table-tested across all branches)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """One model's pre-registered metrics: per-asset + pooled MSE / IC / coverage.

    ``pooled_mse`` is equal-point-weighted across assets (SPY/ACWI's larger return
    variance dominates the magnitude); ``pooled_ic`` is a concatenated-cross-asset
    scale artifact — reporting-only, never decision-gating (module docstring).
    """

    model_version: str
    per_asset_mse: dict[str, float]
    per_asset_ic: dict[str, float]
    per_asset_coverage: dict[str, float]
    pooled_mse: float
    pooled_ic: float
    pooled_coverage: float
    n_points: int


def decide_promotion(
    ewma: ModelMetrics,
    jepa: ModelMetrics,
    *,
    target: float = COVERAGE_TARGET,
    tolerance: float = COVERAGE_TOLERANCE_PP,
) -> tuple[str, str]:
    """Apply the pre-registered promotion rule; return ``(decision, rationale)``.

    ``decision`` is exactly :data:`JEPA_PROMOTED` or :data:`EWMA_REMAINS_DEFAULT`.
    Purely mechanical: the three clauses (strictly-lower pooled MSE, coverage within
    +-tolerance of target — INCLUSIVE at the band edge, coverage distance no worse
    than EWMA's — INCLUSIVE on equality) with EWMA winning every tie.
    """
    mse_lower = jepa.pooled_mse < ewma.pooled_mse
    jepa_distance = abs(jepa.pooled_coverage - target)
    ewma_distance = abs(ewma.pooled_coverage - target)
    within_band = jepa_distance <= tolerance
    no_worse = jepa_distance <= ewma_distance

    if mse_lower and within_band and no_worse:
        return (
            JEPA_PROMOTED,
            f"pooled MSE strictly lower ({jepa.pooled_mse:.4e} < {ewma.pooled_mse:.4e}), "
            f"coverage {jepa.pooled_coverage:.3f} within +-{tolerance:.2f} of {target:.2f}, "
            f"and no worse than EWMA (|{jepa.pooled_coverage:.3f}-{target:.2f}|="
            f"{jepa_distance:.3f} <= {ewma_distance:.3f})",
        )

    failed: list[str] = []
    if not mse_lower:
        failed.append(
            f"pooled MSE not strictly lower (JEPA {jepa.pooled_mse:.4e} vs "
            f"EWMA {ewma.pooled_mse:.4e})"
        )
    if not within_band:
        failed.append(
            f"coverage {jepa.pooled_coverage:.3f} outside +-{tolerance:.2f} of {target:.2f} "
            f"(distance {jepa_distance:.3f})"
        )
    if not no_worse:
        failed.append(
            f"coverage distance {jepa_distance:.3f} worse than EWMA's {ewma_distance:.3f}"
        )
    return EWMA_REMAINS_DEFAULT, "; ".join(failed)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FoldInfo:
    """Per-fold metadata: split sizes + the strided eval-point span."""

    index: int
    train_bars: int
    test_bars: int
    n_eval_points: int
    first_eval_date: str
    last_eval_date: str


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The committed comparison: fold metadata, both models' metrics, the decision.

    ``config_fingerprint`` / ``data_fingerprint`` pin what was measured (the FULL
    JEPA config + eval knobs, and the panel row counts + max dates) so the report is
    reproducible and auditable; ``config_hash`` is a short digest of the fingerprint.
    """

    universe: tuple[str, ...]
    horizon: int
    n_splits: int
    stride: int
    calibration_z: float
    coverage_target: float
    coverage_tolerance_pp: float
    folds: tuple[FoldInfo, ...]
    ewma: ModelMetrics
    jepa: ModelMetrics
    decision: str
    decision_rationale: str
    config_fingerprint: dict[str, object]
    config_hash: str
    data_fingerprint: dict[str, object]
    runtime_seconds: float
    generated_at_utc: str

    @property
    def jepa_promoted(self) -> bool:
        """True iff the mechanical rule promoted the JEPA."""
        return self.decision == JEPA_PROMOTED


# --------------------------------------------------------------------------- #
# Metric assembly
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Accum:
    """Per-asset accumulator of ``(mu, sigma, realized)`` across every eval point."""

    mu: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    realized: list[float] = field(default_factory=list)

    def add(self, mu: float, sigma: float, realized: float) -> None:
        self.mu.append(mu)
        self.sigma.append(sigma)
        self.realized.append(realized)


def _model_metrics(model_version: str, accum: dict[str, _Accum]) -> ModelMetrics:
    """Fold every asset's accumulated triples into per-asset + pooled metrics."""
    per_asset_mse: dict[str, float] = {}
    per_asset_ic: dict[str, float] = {}
    per_asset_coverage: dict[str, float] = {}
    pooled_mu: list[float] = []
    pooled_sigma: list[float] = []
    pooled_realized: list[float] = []
    for asset, acc in accum.items():
        per_asset_mse[asset] = forecast_mse(acc.mu, acc.realized)
        per_asset_ic[asset] = information_coefficient(acc.mu, acc.realized)
        per_asset_coverage[asset] = coverage(acc.mu, acc.sigma, acc.realized)
        pooled_mu.extend(acc.mu)
        pooled_sigma.extend(acc.sigma)
        pooled_realized.extend(acc.realized)
    return ModelMetrics(
        model_version=model_version,
        per_asset_mse=per_asset_mse,
        per_asset_ic=per_asset_ic,
        per_asset_coverage=per_asset_coverage,
        pooled_mse=forecast_mse(pooled_mu, pooled_realized),
        pooled_ic=information_coefficient(pooled_mu, pooled_realized),
        pooled_coverage=coverage(pooled_mu, pooled_sigma, pooled_realized),
        n_points=len(pooled_mu),
    )


def _jepa_version_label(fold_versions: list[str]) -> str:
    """The pooled-metrics label for the JEPA: one hash, or all per-fold hashes.

    Each fold trains its OWN ensemble, so pooled JEPA numbers come from several
    checkpoints — labeling them with the last fold's hash alone would misattribute
    every earlier fold's points. All distinct versions are listed.
    """
    unique = list(dict.fromkeys(fold_versions))
    if len(unique) == 1:
        return unique[0]
    return f"per-fold ensembles: {', '.join(fold_versions)}"


# --------------------------------------------------------------------------- #
# Production factories (route JEPA through train_jepa; EWMA needs no training)
# --------------------------------------------------------------------------- #


def production_jepa_factory(train_prices: dict[str, pd.Series], config: JEPAConfig) -> WorldModel:
    """Train the JEPA on a fold's training slice via the production ``train_jepa``.

    Mirrors ``model.train.load_training_data``'s assembly (build_features matrix +
    per-asset daily log-returns) so the eval trains the SAME way production does, then
    returns the checkpoint as an in-memory :class:`~abe.model.jepa.JEPAModel`.
    """
    features_matrix = build_features(train_prices, macro=None)
    returns = pd.DataFrame({asset: log_returns(price) for asset, price in train_prices.items()})
    return train_jepa(features_matrix, returns, config).to_model()


def production_ewma_factory(train_prices: dict[str, pd.Series]) -> WorldModel:
    """The EWMA baseline for a fold. EWMA is stateless — ``train_prices`` is unused;
    the forecast recomputes its expanding EWMA + forecast-error std from each eval
    point's truncated frame."""
    del train_prices  # EWMA needs no training slice.
    return EWMABaseline()


# --------------------------------------------------------------------------- #
# The walk-forward eval loop (the production interface, the production truncation)
# --------------------------------------------------------------------------- #


def run_walk_forward_eval(
    prices: dict[str, pd.Series],
    *,
    n_splits: int,
    model_factory_jepa: JepaFactory,
    config: JEPAConfig,
    model_factory_ewma: EwmaFactory = production_ewma_factory,
    stride: int = HORIZON_BARS,
    horizon: int = HORIZON_BARS,
) -> EvalReport:
    """Score JEPA vs EWMA on identical purged walk-forward windows; return an :class:`EvalReport`.

    ``prices`` maps each asset to its adjusted-close series (CacheAdapter shape). The
    JOINT panel — the inner join of every asset's daily log-returns — defines fold
    geometry and realized-return alignment ONLY: folds and eval points are positions
    on this one shared index, so every model is scored on the same dates against the
    same realized returns.

    The frames each model *forecasts on* are PRODUCTION-SHAPED (module docstring):
    per fold, the JEPA is trained on prices ``<= last_train_date`` only (EWMA needs
    no training); then at each strided eval point ``t`` in the test window, EACH
    model's ``.forecast()`` receives ``{asset: log_return frame}`` where every frame
    is that asset's OWN FULL history truncated to dates ``<= panel_dates[t]`` —
    exactly what ``pipeline._stage_features`` feeds the live forecaster, so EWMA's
    whole-frame forecast-error sigma matches what production emits. Recorded
    ``(mu, sigma)`` are scored against the realized ``(t, t+H]`` panel return.

    ``config.horizon`` MUST equal ``horizon`` — the JEPA forecasts an H-day return and
    the realized target is an H-day return; a mismatch would compare different horizons
    (a silent validity break), so it raises.
    """
    started = datetime.now(UTC)
    if not prices:
        raise ValueError("prices must contain at least one asset series")
    if config.horizon != horizon:
        raise ValueError(
            f"config.horizon={config.horizon} != eval horizon={horizon}: the JEPA forecasts an "
            "H-day return and realized targets span H bars; they must match or the comparison "
            "measures different horizons"
        )

    assets = list(prices)
    # Full per-asset return series (production shape) — the frames models forecast on.
    asset_returns = {asset: log_returns(price) for asset, price in prices.items()}
    # The joint panel (inner join) — fold geometry + realized alignment ONLY.
    panel_returns = pd.concat(asset_returns, axis=1, join="inner")
    if not panel_returns.index.is_monotonic_increasing:
        panel_returns = panel_returns.sort_index()
    n_panel = len(panel_returns)
    panel_dates = [str(date) for date in panel_returns.index]
    panel_arrays = {asset: panel_returns[asset].to_numpy(dtype=np.float64) for asset in assets}

    splits = purged_walk_forward_splits(n_panel, n_splits, horizon=horizon)
    validate_no_leakage(splits, horizon)

    ewma_accum = {asset: _Accum() for asset in assets}
    jepa_accum = {asset: _Accum() for asset in assets}
    folds: list[FoldInfo] = []
    jepa_fold_versions: list[str] = []

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        last_train_date = panel_dates[int(train_idx[-1])]
        train_prices = {
            asset: price[price.index <= last_train_date] for asset, price in prices.items()
        }
        jepa_model = model_factory_jepa(train_prices, config)
        ewma_model = model_factory_ewma(train_prices)
        jepa_fold_versions.append(jepa_model.model_version)

        points = eval_positions(test_idx, horizon, stride, n_panel)
        for t in points:
            date_t = panel_dates[t]
            # Production-shaped frames: each asset's FULL history up to date_t
            # (never the panel-truncated series — see the module docstring).
            frame_upto_t = {
                asset: series.loc[series.index <= date_t].to_frame(name=LOG_RETURN_COLUMN)
                for asset, series in asset_returns.items()
            }
            ewma_forecast = ewma_model.forecast(frame_upto_t)
            jepa_forecast = jepa_model.forecast(frame_upto_t)
            for asset in assets:
                realized = realized_horizon_return(panel_arrays[asset], t, horizon)
                _record(ewma_accum[asset], ewma_forecast[asset], realized)
                _record(jepa_accum[asset], jepa_forecast[asset], realized)

        folds.append(
            FoldInfo(
                index=fold_index,
                train_bars=int(train_idx.size),
                test_bars=int(test_idx.size),
                n_eval_points=len(points),
                first_eval_date=panel_dates[points[0]] if points else "",
                last_eval_date=panel_dates[points[-1]] if points else "",
            )
        )

    ewma_metrics = _model_metrics("ewma", ewma_accum)
    jepa_metrics = _model_metrics(_jepa_version_label(jepa_fold_versions), jepa_accum)
    decision, rationale = decide_promotion(ewma_metrics, jepa_metrics)

    config_fingerprint = _config_fingerprint(config, n_splits, stride, horizon)
    data_fingerprint = _data_fingerprint(prices, panel_dates)
    runtime = (datetime.now(UTC) - started).total_seconds()

    return EvalReport(
        universe=tuple(assets),
        horizon=horizon,
        n_splits=n_splits,
        stride=stride,
        calibration_z=CALIBRATION_Z,
        coverage_target=COVERAGE_TARGET,
        coverage_tolerance_pp=COVERAGE_TOLERANCE_PP,
        folds=tuple(folds),
        ewma=ewma_metrics,
        jepa=jepa_metrics,
        decision=decision,
        decision_rationale=rationale,
        config_fingerprint=config_fingerprint,
        config_hash=_fingerprint_hash(config_fingerprint),
        data_fingerprint=data_fingerprint,
        runtime_seconds=runtime,
        generated_at_utc=started.replace(microsecond=0).isoformat(),
    )


def _record(accum: _Accum, forecast: Forecast, realized: float) -> None:
    accum.add(forecast.mu, forecast.sigma, realized)


def _config_fingerprint(
    config: JEPAConfig, n_splits: int, stride: int, horizon: int
) -> dict[str, object]:
    """Everything that defines what was measured — ANY config drift changes the hash.

    The FULL ``JEPAConfig`` (every field, via ``dataclasses.asdict`` — seed, lr,
    ema_momentum, the VICReg weights, holdout_fraction, sigma_floor, feature_names,
    ...) plus the eval knobs. Cherry-picking fields here would let a determinism-
    relevant knob drift without moving ``config_hash``.
    """
    jepa_config = dataclasses.asdict(config)
    jepa_config["feature_names"] = list(config.feature_names)  # tuple -> JSON array
    return {
        "jepa_config": jepa_config,
        "horizon": horizon,
        "n_splits": n_splits,
        "stride": stride,
        "calibration_z": CALIBRATION_Z,
        "coverage_target": COVERAGE_TARGET,
        "coverage_tolerance_pp": COVERAGE_TOLERANCE_PP,
    }


def _data_fingerprint(prices: dict[str, pd.Series], panel_dates: list[str]) -> dict[str, object]:
    """Panel size + per-asset row counts and max dates (the data actually scored)."""
    return {
        "panel_bars": len(panel_dates),
        "panel_start": panel_dates[0] if panel_dates else "",
        "panel_end": panel_dates[-1] if panel_dates else "",
        "assets": {
            asset: {"rows": int(len(price)), "max_date": str(price.index[-1])}
            for asset, price in prices.items()
        },
    }


def _fingerprint_hash(fingerprint: dict[str, object]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Markdown report (the committed doc)
# --------------------------------------------------------------------------- #


def _fmt(value: float) -> str:
    """Format a metric; NaN renders as ``n/a``."""
    return "n/a" if not np.isfinite(value) else f"{value:.4e}"


def _fmt_cov(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _fmt_ic(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:+.3f}"


def render_report_markdown(report: EvalReport) -> str:
    """Render the committed markdown report from an :class:`EvalReport`.

    Contains, by contract (the report's readers depend on these being present): the
    pre-registered rule text, both models' per-asset + pooled numbers, the decision
    line (:data:`JEPA_PROMOTED` or :data:`EWMA_REMAINS_DEFAULT`) with its mechanical
    rationale, and the config + data fingerprints.
    """
    z = report.calibration_z
    target = report.coverage_target
    tol = report.coverage_tolerance_pp
    lines: list[str] = []
    lines.append(f"# JEPA vs EWMA walk-forward evaluation ({report.generated_at_utc})")
    lines.append("")
    lines.append(
        f"Universe **{', '.join(report.universe)}** | horizon **H = {report.horizon}** bars | "
        f"**{report.n_splits}** purged walk-forward split(s) | stride **{report.stride}** "
        f"(non-overlapping within a fold) | runtime **{report.runtime_seconds:.1f}s**."
    )
    lines.append("")

    lines.append("## Pre-registered rule (written before results)")
    lines.append("")
    lines.append(
        f"Metrics, computed from `(mu, sigma)` produced **through the production "
        f"`WorldModel.forecast` interface** on identical purged walk-forward windows (each asset's "
        f"frame is its own full cached history truncated to the eval date — the production "
        f"`_stage_features` shape): per-asset + pooled forecast **MSE** (mu vs realized H-day "
        f"log-return), **information coefficient** (Spearman rank corr of mu vs realized, per "
        f"asset), and **sigma calibration coverage** (fraction with `|realized - mu| <= {z} * "
        f"sigma`; target ~= {target:.2f})."
    )
    lines.append("")
    lines.append(
        "**Promotion rule (mechanical):** the JEPA is **promoted ONLY IF** "
        "(1) its pooled MSE is strictly lower than EWMA's, **AND** "
        f"(2) its pooled coverage is within +-{tol:.2f} of {target:.2f}, **AND** "
        "(3) its coverage is no worse than EWMA's distance from "
        f"{target:.2f}. Otherwise **EWMA remains the default** (EWMA wins ties)."
    )
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(f"### {report.decision}")
    lines.append("")
    lines.append(f"Rationale: {report.decision_rationale}")
    lines.append("")

    lines.append("## Pooled metrics")
    lines.append("")
    lines.append("| Model | Version | Points | Pooled MSE | Pooled IC | Coverage |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for metrics in (report.ewma, report.jepa):
        lines.append(
            f"| {'EWMA' if metrics is report.ewma else 'JEPA'} | `{metrics.model_version}` | "
            f"{metrics.n_points} | {_fmt(metrics.pooled_mse)} | {_fmt_ic(metrics.pooled_ic)} | "
            f"{_fmt_cov(metrics.pooled_coverage)} |"
        )
    lines.append("")
    lines.append(
        "Pooled MSE is equal-point-weighted across assets (higher-variance SPY/ACWI dominate its "
        "magnitude; AGG contributes little). Pooled IC is Spearman over the concatenated "
        "cross-asset series — a scale artifact reported for context only; the promotion rule "
        "never uses it."
    )
    lines.append("")

    lines.append("## Per-asset metrics")
    lines.append("")
    lines.append("| Asset | Model | MSE | IC | Coverage |")
    lines.append("|---|---|---:|---:|---:|")
    for asset in report.universe:
        for label, metrics in (("EWMA", report.ewma), ("JEPA", report.jepa)):
            lines.append(
                f"| {asset} | {label} | {_fmt(metrics.per_asset_mse.get(asset, float('nan')))} | "
                f"{_fmt_ic(metrics.per_asset_ic.get(asset, float('nan')))} | "
                f"{_fmt_cov(metrics.per_asset_coverage.get(asset, float('nan')))} |"
            )
    lines.append("")

    lines.append("## Folds")
    lines.append("")
    lines.append("| Fold | Train bars | Test bars | Eval points | First eval | Last eval |")
    lines.append("|---:|---:|---:|---:|---|---|")
    for fold in report.folds:
        lines.append(
            f"| {fold.index} | {fold.train_bars} | {fold.test_bars} | {fold.n_eval_points} | "
            f"{fold.first_eval_date} | {fold.last_eval_date} |"
        )
    lines.append("")

    lines.append("## Fingerprints")
    lines.append("")
    lines.append(f"- **Config** (`{report.config_hash}`):")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.config_fingerprint, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("- **Data:**")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.data_fingerprint, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    lines.append("## How promotion happens (manual operator action)")
    lines.append("")
    lines.append(
        "This report only **records** the decision. Promotion is a deliberate operator step, "
        "never automatic: the scheduler and API default to EWMA and this eval never touches them. "
        "To promote the JEPA, set `ABE_MODEL=jepa` + `ABE_JEPA_CHECKPOINT=<path>` at startup (the "
        "Step 13 toggle via `resolve_startup_model`). Until an operator flips it, EWMA stays the "
        "live forecaster — consistent with plan section 1 (V1 is never gated on the JEPA beating "
        "the baseline; on ~4,600 joint bars, parity is the realistic best case)."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def quick_config() -> JEPAConfig:
    """A smaller (ci-like) JEPA config for a bounded ``--quick`` run.

    Shrinks size/speed knobs only (dims, K=2, epochs); the loss weights — notably the
    deliberately small production ``return_weight`` — stay at the ``JEPAConfig``
    defaults, so ``--quick`` still trains the production objective, just faster.
    """
    return JEPAConfig(latent_dim=16, hidden_dim=32, n_seeds=2, epochs=60, context_len=16)


def load_prices_from_cache(db_path: str | Path) -> dict[str, pd.Series]:
    """Load ``UNIVERSE`` adjusted closes from the SQLite cache (zero network)."""
    adapter = CacheAdapter(db_path)
    prices: dict[str, pd.Series] = {}
    for asset in UNIVERSE:
        frame = adapter.fetch(asset)
        if frame.empty:
            raise ValueError(
                f"no cached prices for {asset!r} in {db_path} — run the price backfill first "
                "(uv run python -m abe.ingest.prices --backfill)"
            )
        prices[asset] = frame["adj_close"]
    return prices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m abe.eval.walk_forward",
        description="Pre-registered JEPA-vs-EWMA purged walk-forward eval; writes a report.",
    )
    parser.add_argument("--db", required=True, help="SQLite price cache path (read-only).")
    parser.add_argument("--out", required=True, help="Markdown report path to write.")
    parser.add_argument(
        "--splits",
        type=int,
        default=None,
        help="Number of purged walk-forward splits (default: 4 full / 3 quick).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller (ci-like) JEPA config + fewer splits for bounded runtime.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: run the eval against a cache db and write the report. Returns 0 on success.

    Exit 0 regardless of who wins — the gate is the eval EXISTING, not the JEPA winning
    (plan: never gated on the JEPA beating the baseline).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    config = quick_config() if args.quick else JEPAConfig()
    n_splits = args.splits if args.splits is not None else (3 if args.quick else 4)

    prices = load_prices_from_cache(args.db)
    logger.info(
        "walk-forward eval: mode=%s, n_splits=%d, config(latent=%d,hidden=%d,K=%d,epochs=%d)",
        "quick" if args.quick else "full",
        n_splits,
        config.latent_dim,
        config.hidden_dim,
        config.n_seeds,
        config.epochs,
    )
    report = run_walk_forward_eval(
        prices,
        n_splits=n_splits,
        model_factory_jepa=production_jepa_factory,
        config=config,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report_markdown(report), encoding="utf-8")

    logger.info("wrote report %s (runtime %.1fs)", out_path, report.runtime_seconds)
    logger.info(
        "EWMA pooled: MSE=%.4e coverage=%.3f | JEPA pooled: MSE=%.4e coverage=%.3f",
        report.ewma.pooled_mse,
        report.ewma.pooled_coverage,
        report.jepa.pooled_mse,
        report.jepa.pooled_coverage,
    )
    logger.info("DECISION: %s — %s", report.decision, report.decision_rationale)
    return 0


if __name__ == "__main__":  # pragma: no cover — module CLI entrypoint
    raise SystemExit(main())
