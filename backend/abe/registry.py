"""Stage registries — the pluggable-per-stage seam (Track 2 Step 18).

Four string-keyed registries, one per pipeline stage, generalizing the existing
EWMA/JEPA ``WorldModel`` toggle and the Track 1 ``EXPLANATIONS`` registry:

- **feature builders** (``feature_set`` key): ``basic`` today; Step 24 adds
  ``fracdiff_macro``. A builder turns per-asset price frames into the returns +
  feature matrices the forecaster consumes, plus the ``features`` rows to persist
  and the stage-card ``detail``.
- **forecasters** (``forecaster`` key): ``ewma`` (default) + ``jepa`` (checkpoint).
  Resolves to a :class:`~abe.model.base.WorldModel`.
- **view sources** (keyed by the ViewScenario ``kind``): ``forecast`` today
  (Step 22 adds ``historical`` + ``counterfactual``). A view source produces
  ``{asset: Forecast}`` — the SAME shape ``bl_blend`` already consumes, so the
  blend math is untouched no matter where the views come from.
- **optimizers** (``optimizer`` key): ``mvu`` today; Step 23 adds ``min_variance``
  + the MVU ``min_weight`` floor. Resolves to something producing an
  :class:`~abe.optimize.mvu.MVUResult`.

Each registry entry carries a ``factory`` (params -> concrete impl), a declared
``param_schema`` (surfaced to the UI dropdowns), and a human ``description``.
:func:`resolve` turns a :class:`~abe.config.Config` + its
:class:`~abe.config.ViewScenario` into a :class:`ResolvedStack` of concrete stage
impls; an unknown key fails loud HERE (the resolve-time validation
``config.create_config`` defers to). :func:`registries_manifest` renders the whole
set JSON-friendly for the Step 25 API.

Parity note (Step 19): ``resolve(central_config, forecast_scenario)`` must yield
the exact V1 stack — ``basic`` features whose ``detail`` is byte-identical to the
pre-Track-2 ``_stage_features``, an ``EWMABaseline``, the ``forecast`` view source
(identity on the model's forecasts), and ``optimize_weights`` with V1 defaults.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import pandas as pd

from abe.blend.views import (
    ViewContext,
    ViewSource,
    counterfactual_view_factory,
    forecast_view_factory,
    historical_view_factory,
)
from abe.calc import (
    LOG_RETURN_COLUMN,
    REALIZED_VOL_COLUMN,
    log_returns,
    realized_vol,
)
from abe.config import Config, ViewScenario
from abe.constants import HORIZON_BARS, UNIVERSE
from abe.model import load_model
from abe.model.base import DEFAULT_HALFLIFE, EWMABaseline, WorldModel
from abe.optimize.mvu import MVUResult, optimize_weights

__all__ = [
    "FEATURE_BUILDERS",
    "FORECASTERS",
    "OPTIMIZERS",
    "VIEW_SOURCES",
    "FeatureBuilder",
    "FeatureBundle",
    "Optimizer",
    "ParamSpec",
    "RegistryEntry",
    "ResolvedStack",
    "ViewContext",
    "ViewSource",
    "registries_manifest",
    "resolve",
]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable parameter of a stage impl (surfaced to the UI param form)."""

    name: str
    type: str  # "float" | "int" | "str" | "bool"
    default: object
    description: str


@dataclass(frozen=True, slots=True)
class RegistryEntry[T]:
    """A registered stage impl: its ``factory`` (params -> impl), declared
    ``param_schema``, and human ``description``."""

    factory: Callable[[Mapping[str, object]], T]
    param_schema: tuple[ParamSpec, ...]
    description: str


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """A feature builder's output. ``returns`` feeds the blend covariance;
    ``features_frames`` feeds the forecaster; ``rows`` are persisted into the
    ``features`` table; ``detail`` is the stage card payload."""

    returns: dict[str, pd.Series]
    features_frames: dict[str, pd.DataFrame]
    rows: list[dict[str, object]]
    detail: dict[str, object]


class FeatureBuilder(Protocol):
    key: str

    def build(self, frames: Mapping[str, pd.DataFrame]) -> FeatureBundle:
        """Turn per-asset price frames (each with an ``adj_close`` column) into a
        :class:`FeatureBundle` over the fixed ``UNIVERSE``."""
        ...


class BasicFeatureBuilder:
    """``basic``: log-return + ``HORIZON_BARS``-day realized vol (the V1 stack).

    Reproduces the pre-Track-2 ``_stage_features`` exactly — the returns, the
    persisted rows (log_return then realized_vol per asset), and a ``detail``
    that is byte-identical to V1 (Step 19's parity golden pins this)."""

    key = "basic"

    def build(self, frames: Mapping[str, pd.DataFrame]) -> FeatureBundle:
        returns: dict[str, pd.Series] = {}
        features_frames: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, object]] = []
        latest: dict[str, dict[str, object]] = {}
        for asset in UNIVERSE:
            asset_returns = log_returns(frames[asset]["adj_close"])
            vol = realized_vol(asset_returns)
            returns[asset] = asset_returns
            features_frames[asset] = asset_returns.to_frame()
            values = {
                LOG_RETURN_COLUMN: float(asset_returns.iloc[-1]),
                REALIZED_VOL_COLUMN: float(vol.iloc[-1]),
            }
            for name, value in values.items():
                rows.append({"asset": asset, "name": name, "value": value})
            latest[asset] = {"date": str(asset_returns.index[-1]), **values}
        detail: dict[str, object] = {
            "features": [LOG_RETURN_COLUMN, REALIZED_VOL_COLUMN],
            "windows": {
                LOG_RETURN_COLUMN: "1 day",
                REALIZED_VOL_COLUMN: f"{HORIZON_BARS} days, annualized",
            },
            "latest": latest,
        }
        return FeatureBundle(returns, features_frames, rows, detail)


def _basic_feature_factory(_params: Mapping[str, object]) -> FeatureBuilder:
    return BasicFeatureBuilder()


FEATURE_BUILDERS: Final[dict[str, RegistryEntry[FeatureBuilder]]] = {
    "basic": RegistryEntry(
        factory=_basic_feature_factory,
        param_schema=(),
        description="Log-return + realized-vol (the V1 feature set); no parameters.",
    ),
}


# --------------------------------------------------------------------------- #
# Forecasters (WorldModel)
# --------------------------------------------------------------------------- #


def _ewma_factory(params: Mapping[str, object]) -> WorldModel:
    raw = params.get("halflife", DEFAULT_HALFLIFE)
    if not isinstance(raw, (int, float)):
        raise ValueError(f"ewma halflife must be numeric, got {type(raw).__name__}")
    return EWMABaseline(halflife=float(raw))


def _jepa_factory(params: Mapping[str, object]) -> WorldModel:
    from pathlib import Path

    raw = params.get("checkpoint_path")
    checkpoint = Path(str(raw)) if raw else None
    # Delegates to the same loader resolve_startup_model uses; a missing/bad
    # checkpoint raises loudly (never a silent EWMA fallback).
    return load_model("jepa", checkpoint)


FORECASTERS: Final[dict[str, RegistryEntry[WorldModel]]] = {
    "ewma": RegistryEntry(
        factory=_ewma_factory,
        param_schema=(
            ParamSpec(
                name="halflife",
                type="float",
                default=DEFAULT_HALFLIFE,
                description="EWMA halflife in trading days (~one quarter default).",
            ),
        ),
        description="EWMA-of-returns baseline (the V1 default forecaster).",
    ),
    "jepa": RegistryEntry(
        factory=_jepa_factory,
        param_schema=(
            ParamSpec(
                name="checkpoint_path",
                type="str",
                default=None,
                description="Path to a trained JEPA checkpoint (required for jepa).",
            ),
        ),
        description="Minimal JEPA forecaster loaded from a checkpoint (behind a toggle).",
    ),
}


# --------------------------------------------------------------------------- #
# View sources (produce {asset: Forecast} for bl_blend). The impls live in
# abe.blend.views; ViewContext / ViewSource are re-exported here so the pipeline
# keeps using registry.ViewContext / registry.resolve(...).view_source.
# --------------------------------------------------------------------------- #


VIEW_SOURCES: Final[dict[str, RegistryEntry[ViewSource]]] = {
    "forecast": RegistryEntry(
        factory=forecast_view_factory,
        param_schema=(),
        description="Views derived from the run's forecaster (the V1 view source).",
    ),
    "historical": RegistryEntry(
        factory=historical_view_factory,
        param_schema=(
            ParamSpec(
                name="window_start",
                type="str",
                default=None,
                description="ISO start date of the realized-return window (optional).",
            ),
            ParamSpec(
                name="window_end",
                type="str",
                default=None,
                description="ISO end date of the realized-return window (optional).",
            ),
        ),
        description="Absolute views from a past window's realized returns.",
    ),
    "counterfactual": RegistryEntry(
        factory=counterfactual_view_factory,
        param_schema=(
            ParamSpec(
                name="views",
                type="str",
                default=None,
                description="Per-asset {mu: annual return, confidence: 0..1} hand-authored views.",
            ),
        ),
        description="Hand-authored absolute views (e.g. SPY +10%).",
    ),
}


# --------------------------------------------------------------------------- #
# Optimizers (produce MVUResult)
# --------------------------------------------------------------------------- #


class Optimizer(Protocol):
    key: str

    def optimize(
        self,
        mu_post: pd.Series,
        sigma_post: pd.DataFrame,
        w_prev: Mapping[str, float] | None,
    ) -> MVUResult:
        """Solve for target weights from the BL posterior."""
        ...


class MVUOptimizer:
    """``mvu``: the V1 mean-variance-utility QP. Step 23 adds a ``min_weight``
    box floor via the param schema."""

    key = "mvu"

    def optimize(
        self,
        mu_post: pd.Series,
        sigma_post: pd.DataFrame,
        w_prev: Mapping[str, float] | None,
    ) -> MVUResult:
        return optimize_weights(mu_post, sigma_post, w_prev)


def _mvu_factory(_params: Mapping[str, object]) -> Optimizer:
    return MVUOptimizer()


OPTIMIZERS: Final[dict[str, RegistryEntry[Optimizer]]] = {
    "mvu": RegistryEntry(
        factory=_mvu_factory,
        param_schema=(),
        description="Mean-variance-utility QP with L1 turnover (the V1 optimizer).",
    ),
}


# --------------------------------------------------------------------------- #
# Resolve a Config into concrete stage impls
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResolvedStack:
    """The four concrete stage impls a Config resolves to."""

    feature_builder: FeatureBuilder
    forecaster: WorldModel
    view_source: ViewSource
    optimizer: Optimizer


def _resolve_one[T](
    registry: Mapping[str, RegistryEntry[T]],
    key: str,
    label: str,
    params: Mapping[str, object],
) -> T:
    entry = registry.get(key)
    if entry is None:
        raise ValueError(
            f"unknown {label} {key!r}; registered: {sorted(registry)}"
        )
    return entry.factory(params)


def _stage_params(config: Config, stage: str) -> Mapping[str, object]:
    raw = config.params.get(stage, {})
    return raw if isinstance(raw, Mapping) else {}


def resolve(
    config: Config,
    view_scenario: ViewScenario,
    *,
    forecaster_override: WorldModel | None = None,
) -> ResolvedStack:
    """Turn a Config + its ViewScenario into concrete stage impls.

    Per-stage params come from ``config.params[<stage>]`` (``features`` /
    ``forecaster`` / ``optimizer``); the view source is keyed by the scenario's
    ``kind`` and parameterized by its ``payload``. An unknown key at any stage
    raises ``ValueError`` (the loud resolve-time validation).

    ``forecaster_override`` (the back-compat ``ABE_MODEL`` / test seam) SUPERSEDES
    the config's forecaster: when given, the config's ``forecaster`` key is NOT
    resolved at all — so a broken central forecaster (e.g. a missing JEPA
    checkpoint) does not break a run the override would otherwise rescue, and no
    checkpoint is loaded only to be discarded."""
    forecaster = (
        forecaster_override
        if forecaster_override is not None
        else _resolve_one(
            FORECASTERS, config.forecaster, "forecaster", _stage_params(config, "forecaster")
        )
    )
    return ResolvedStack(
        feature_builder=_resolve_one(
            FEATURE_BUILDERS, config.feature_set, "feature_set", _stage_params(config, "features")
        ),
        forecaster=forecaster,
        view_source=_resolve_one(
            VIEW_SOURCES, view_scenario.kind, "view kind", view_scenario.payload
        ),
        optimizer=_resolve_one(
            OPTIMIZERS, config.optimizer, "optimizer", _stage_params(config, "optimizer")
        ),
    )


def _manifest_one[T](registry: Mapping[str, RegistryEntry[T]]) -> dict[str, object]:
    return {
        key: {
            "description": entry.description,
            "params": [
                {
                    "name": spec.name,
                    "type": spec.type,
                    "default": spec.default,
                    "description": spec.description,
                }
                for spec in entry.param_schema
            ],
        }
        for key, entry in registry.items()
    }


def registries_manifest() -> dict[str, dict[str, object]]:
    """JSON-friendly view of every registry (keys + param schemas + descriptions)
    for the UI dropdowns / param forms (Step 25 serves this)."""
    return {
        "feature_set": _manifest_one(FEATURE_BUILDERS),
        "forecaster": _manifest_one(FORECASTERS),
        "view_source": _manifest_one(VIEW_SOURCES),
        "optimizer": _manifest_one(OPTIMIZERS),
    }
