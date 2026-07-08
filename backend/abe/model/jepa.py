"""Minimal JEPA world-model (plan.md Step 13) — behind a config toggle; EWMA stays default.

This module owns the *architecture + inference + checkpoint* half of Step 13;
``model/train.py`` owns the offline *training* half (``train_jepa``,
``shuffled_target_control``, the CLI). :class:`JEPAModel` is a drop-in
:class:`~abe.model.base.WorldModel` — it passes the SAME frozen contract test
(``tests/test_model_base.py::assert_worldmodel_contract``) EWMA passes, through
the SAME per-asset ``{asset: frame}`` signature the pipeline hands every
forecaster.

Architecture (a small VICReg-JEPA, <500k params — see :func:`count_parameters`)
================================================================================

Per seed, one :class:`JEPANet` holds four MLPs:

- **context encoder** ``f_ctx``: flattened trailing window ``x_t`` (the last
  ``context_len`` bars x ``F`` features, row-major) -> latent ``z_t``.
- **EMA target encoder** ``f_tgt``: same architecture, weights are an
  exponential moving average of ``f_ctx`` (no gradients) — the JEPA target.
- **predictor** ``h``: ``z_t`` -> a prediction of the FUTURE window's target
  embedding.
- **return head** ``g``: ``z_t`` -> a (standardized) H-day log-return scalar,
  joint-trained at a small loss weight (LeCun §4.5.2 sanctions the auxiliary
  head).

JEPA objective (in train.py): predict ``f_tgt(x_future)`` from ``h(f_ctx(x_t))``
via MSE, plus VICReg variance/covariance penalties on the embeddings
(anti-collapse), plus ``lambda_ret`` * return-head MSE.

No-lookahead batch construction (:func:`window_indices`)
========================================================

For a context-end bar ``t`` the triple is:

- context window ``[t-C+1, t]`` (ends AT ``t`` — the forecast never sees the
  future),
- target window ``[t+H-C+1, t+H]`` (ends at the horizon ``t+H``; strictly in the
  future because :class:`JEPAConfig` enforces ``context_len <= horizon``, so
  every target index ``> t``),
- return span ``(t, t+H]`` == ``[t+1, t+H]`` — the realized H-day log-return.

The ``context_len <= horizon`` invariant is what makes "every target/return
index is strictly after the context end ``t``" a validated property, exercised
directly by the batch-builder test.

Ensemble sigma (the predictive H-day std — the Step 5 frozen scale)
===================================================================

``Forecast.sigma`` must be the H-DAY PREDICTIVE forecast std (the scale at which
``mu +- 1.64*sigma`` covers ~90% of realized H-day returns — the semantic frozen
on :class:`~abe.model.base.Forecast`; Step 14's promotion eval compares sigma
calibration across models on identical windows, so a mis-scaled sigma silently
invalidates it). The composite, per asset:

    sigma_H = sqrt(residual_var + ensemble_var)

- ``residual_var`` — variance of ``(realized_H - ensemble_mean_mu)`` on a PURGED
  walk-forward holdout (an expanding-window split with an H-bar gap, computed in
  train.py via :func:`~abe.afml.purged_cv.purged_walk_forward_splits`, so no
  training label's horizon reaches into the holdout). This is the DOMINANT term
  and sets the predictive scale: on iid daily noise with std ``s`` the model
  cannot predict the noise, so residuals are ~the realized H-day sums and
  ``sqrt(residual_var) ~ s*sqrt(H)`` — exactly the Step 5 scale (pinned by
  test). Stored per asset in the checkpoint (pooled fallback for an unseen
  asset).
- ``ensemble_var`` — the LIVE K-seed disagreement: sample variance (``ddof=1``)
  of the K models' mu predictions on the CURRENT window (epistemic uncertainty
  of the mean). Zero when ``K == 1``.

Collapse instrumentation (hard-fail)
====================================

Training evaluates each seed's context embeddings on a validation batch and
computes :func:`collapse_metrics`: the median per-dim embedding std and the
effective rank (participation ratio of the embedding-covariance eigenvalues,
``(sum lambda)^2 / sum(lambda^2)`` — scale-invariant, 1.0 == collapsed to a
line, up to ``latent_dim`` == full rank). ``train_jepa`` raises
:class:`CollapseError` if the ensemble's worst median std is below
``std_median_threshold`` OR its worst effective rank is below
``effective_rank_threshold``; a passing :class:`CollapseReport` is stored in the
checkpoint. The thresholds are TOTAL-COLLAPSE floors (point/line degeneration),
not quality bars — Step 14's walk-forward eval is the quality gate.

Checkpoint format (safe load)
=============================

``<path>`` is a ``torch.save`` of ONLY tensors (``{"ensemble": [state_dict,
...]}``) — loaded with ``weights_only=True``. All non-tensor metadata (config,
feature names + standardization stats, per-asset residual variances, collapse
report, frozen frac-diff params, the content hash) lives in a JSON sidecar
``<path>.meta.json``. ``model_version`` is ``f"jepa:{checkpoint_hash[:8]}"``
(plan §3 ``forecasts.model_version`` shape).
"""

import dataclasses
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor, nn

from abe.afml.fracdiff import FracDiffParams
from abe.calc import LOG_RETURN_COLUMN, non_finite_summary
from abe.constants import HORIZON_BARS
from abe.model.base import Forecast

__all__ = [
    "META_SUFFIX",
    "CollapseError",
    "CollapseReport",
    "JEPACheckpoint",
    "JEPAConfig",
    "JEPAModel",
    "JEPANet",
    "WindowIndices",
    "collapse_metrics",
    "count_parameters",
    "ensemble_predictions",
    "hash_state_dicts",
    "standardize_windows",
    "window_indices",
]

META_SUFFIX: Final[str] = ".meta.json"
"""Appended to the ``.pt`` path for the non-tensor metadata sidecar."""

_CHECKPOINT_FORMAT_VERSION: Final[int] = 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class JEPAConfig:
    """Hyperparameters for the minimal JEPA (architecture + training + sigma).

    ``feature_names`` are the BARE per-asset feature columns the model consumes
    (``"log_return"`` by default — the only column the production pipeline's
    per-asset frames carry). Training reads them from the flat
    ``"{asset}_{feature}"`` matrix (``features/build.py``); inference reads them
    by bare name from each per-asset frame. A checkpoint that lists a richer set
    (e.g. ``fracdiff``) raises a clear ``ValueError`` at forecast time if an
    input frame lacks the column — the documented V1 resolution.

    ``context_len <= horizon`` is enforced (the target window ends at ``t+H`` and
    must be strictly in the future — see the module docstring's batch-builder
    contract). The defaults keep the K=3 production ensemble well under 500k
    parameters (see :func:`count_parameters`).

    ``return_weight`` defaults to **0.05** — plan §4 / LeCun §4.5.2 sanction the
    return head only as an auxiliary joint-trained at a SMALL loss weight: the
    shared encoder must be shaped primarily by the self-supervised JEPA
    objective (invariance + VICReg), with the return regression riding along.
    A co-equal weight (1.0) would turn the model into a supervised return MLP
    wearing a ``jepa:`` label, invalidating what Step 14 compares. CI tests may
    override it upward purely as a trainability/speed knob (documented there);
    the production default stays small.

    ``std_median_threshold`` / ``effective_rank_threshold`` are TOTAL-COLLAPSE
    floors, not quality bars: they catch embeddings degenerating to a point or a
    line (the failure VICReg exists to prevent). Model *quality* is judged by
    Step 14's pre-registered walk-forward eval, never by these thresholds.
    """

    context_len: int = 16
    latent_dim: int = 48
    hidden_dim: int = 64
    return_head_hidden: int = 16
    horizon: int = HORIZON_BARS
    n_seeds: int = 3
    feature_names: tuple[str, ...] = (LOG_RETURN_COLUMN,)

    epochs: int = 200
    lr: float = 1e-3
    ema_momentum: float = 0.99

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.1
    return_weight: float = 0.05  # SMALL by design (class docstring / plan §4)
    vicreg_gamma: float = 1.0

    std_median_threshold: float = 0.1
    effective_rank_threshold: float = 1.5

    holdout_fraction: float = 0.25
    seed: int = 0
    sigma_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.context_len < 2:
            raise ValueError(f"context_len must be >= 2, got {self.context_len}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.context_len > self.horizon:
            raise ValueError(
                f"context_len={self.context_len} must be <= horizon={self.horizon}: the target "
                "window ends at t+H and must be strictly in the future (every target index > t)"
            )
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in (0, 1), got {self.holdout_fraction}")

    @property
    def n_features(self) -> int:
        """Number of per-asset feature columns (``F`` in the ``C x F`` window)."""
        return len(self.feature_names)

    @property
    def input_dim(self) -> int:
        """Flattened context-window dimension fed to the encoders (``C * F``)."""
        return self.context_len * self.n_features


# --------------------------------------------------------------------------- #
# No-lookahead batch construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WindowIndices:
    """Bar indices for one training sample (no-lookahead by construction).

    ``t`` is the context-end bar. ``context_idx`` ends at ``t``; ``target_idx``
    ends at ``t + horizon`` and is strictly in the future; ``return_idx`` is the
    span ``(t, t+H]``. See the module docstring's batch-builder contract.
    """

    t: int
    context_idx: npt.NDArray[np.int_]
    target_idx: npt.NDArray[np.int_]
    return_idx: npt.NDArray[np.int_]


def window_indices(n_bars: int, context_len: int, horizon: int) -> list[WindowIndices]:
    """Every valid ``(context, target, return)`` index triple over ``n_bars``.

    ``context_len`` must be ``<= horizon`` (so the target window, length
    ``context_len`` ending at ``t+H``, is strictly future). Valid context-end
    bars are ``t in [context_len - 1, n_bars - 1 - horizon]`` — the context
    window must fit at the start and the target/return horizon at the end.
    Returns an empty list when no triple fits (never raises on a short series;
    the caller decides whether that is an error).
    """
    if context_len < 2:
        raise ValueError(f"context_len must be >= 2, got {context_len}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if context_len > horizon:
        raise ValueError(
            f"context_len={context_len} must be <= horizon={horizon} "
            "(the target window must be strictly in the future)"
        )
    first_t = context_len - 1
    last_t = n_bars - 1 - horizon
    triples: list[WindowIndices] = []
    for t in range(first_t, last_t + 1):
        triples.append(
            WindowIndices(
                t=t,
                context_idx=np.arange(t - context_len + 1, t + 1, dtype=np.int_),
                target_idx=np.arange(t + horizon - context_len + 1, t + horizon + 1, dtype=np.int_),
                return_idx=np.arange(t + 1, t + horizon + 1, dtype=np.int_),
            )
        )
    return triples


# --------------------------------------------------------------------------- #
# Collapse instrumentation
# --------------------------------------------------------------------------- #


class CollapseError(RuntimeError):
    """Raised by training when embeddings collapse (VICReg failed its job).

    Carries the failing :class:`CollapseReport` so the operator sees which
    metric tripped and by how much.
    """

    def __init__(self, report: "CollapseReport") -> None:
        self.report = report
        super().__init__(
            "JEPA embeddings collapsed: "
            f"median per-dim std {report.per_dim_std_median:.4g} "
            f"(threshold {report.std_median_threshold:.4g}), "
            f"effective rank {report.effective_rank:.4g} "
            f"(threshold {report.effective_rank_threshold:.4g}); "
            "raise var/cov weights, add data, or train longer"
        )


@dataclass(frozen=True, slots=True)
class CollapseReport:
    """Anti-collapse evidence stored in a passing checkpoint.

    ``per_dim_std_median`` / ``effective_rank`` are the ENSEMBLE-WORST values
    (min across seeds) — a checkpoint passes only if every seed passes.
    ``effective_rank`` is the participation ratio of the embedding-covariance
    eigenvalues (scale-invariant; 1.0 == a line, up to ``latent_dim``).
    """

    per_dim_std_median: float
    effective_rank: float
    latent_dim: int
    std_median_threshold: float
    effective_rank_threshold: float
    n_samples: int

    @property
    def passed(self) -> bool:
        """True iff both metrics clear their thresholds."""
        return (
            self.per_dim_std_median >= self.std_median_threshold
            and self.effective_rank >= self.effective_rank_threshold
        )


def collapse_metrics(embeddings: Tensor) -> tuple[float, float]:
    """``(median per-dim std, effective rank)`` of an ``(N, D)`` embedding batch.

    Effective rank == ``(sum lambda)^2 / sum(lambda^2)`` over the covariance
    eigenvalues (the participation ratio): scale-invariant, so it flags collapse
    independent of the embedding's overall magnitude. Requires ``N >= 2``.
    """
    z = embeddings.detach()
    if z.dim() != 2:
        raise ValueError(f"embeddings must be 2-D (N, D), got shape {tuple(z.shape)}")
    n_samples = int(z.shape[0])
    if n_samples < 2:
        raise ValueError(f"need >= 2 samples for collapse metrics, got {n_samples}")
    per_dim_std = z.std(dim=0, unbiased=True)
    std_median = float(per_dim_std.median().item())
    centered = z - z.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (n_samples - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    total = float(eigvals.sum().item())
    total_sq = float((eigvals * eigvals).sum().item())
    effective_rank = (total * total) / total_sq if total_sq > 0.0 else 0.0
    return std_median, effective_rank


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """A 1-hidden-layer GELU MLP (``in -> hidden -> out``)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class JEPANet(nn.Module):
    """One seed's JEPA: context encoder + EMA target encoder + predictor + return head."""

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.input_dim
        self.context_encoder = _build_mlp(input_dim, config.hidden_dim, config.latent_dim)
        self.target_encoder = _build_mlp(input_dim, config.hidden_dim, config.latent_dim)
        self.predictor = _build_mlp(config.latent_dim, config.latent_dim, config.latent_dim)
        self.return_head = _build_mlp(config.latent_dim, config.return_head_hidden, 1)
        # Target encoder starts equal to the context encoder, then trails it by
        # EMA — and never receives gradients (it is the JEPA target).
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    def encode_context(self, x: Tensor) -> Tensor:
        """Context embedding ``z_t = f_ctx(x_t)`` (gradients flow)."""
        out: Tensor = self.context_encoder(x)
        return out

    def encode_target(self, x: Tensor) -> Tensor:
        """Target embedding ``f_tgt(x_future)`` — no gradients (EMA branch)."""
        with torch.no_grad():
            out: Tensor = self.target_encoder(x)
        return out

    def predict_latent(self, context_embedding: Tensor) -> Tensor:
        """Predicted future target embedding ``h(z_t)``."""
        out: Tensor = self.predictor(context_embedding)
        return out

    def predict_return_std(self, context_embedding: Tensor) -> Tensor:
        """Standardized H-day return head ``g(z_t)`` (shape ``(N,)``)."""
        out: Tensor = self.return_head(context_embedding)
        return out.squeeze(-1)

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        """Move the target encoder toward the context encoder by EMA."""
        for target_param, context_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target_param.mul_(momentum).add_(context_param, alpha=1.0 - momentum)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Total (or trainable-only) parameter count of ``module``."""
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------- #
# Shared inference primitives (one source of truth for train.py + JEPAModel)
# --------------------------------------------------------------------------- #


def standardize_windows(
    windows: npt.NDArray[np.float64],
    feature_mean: npt.NDArray[np.float64],
    feature_std: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-feature z-score an ``(N, C, F)`` window batch, then flatten to ``(N, C*F)``.

    ``feature_mean`` / ``feature_std`` are length-``F`` (TRAIN stats — no
    lookahead). The flatten is row-major ``(C, F) -> C*F`` — the SAME order the
    encoders were trained on (producer/consumer share this one function).
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must be 3-D (N, C, F), got shape {windows.shape}")
    standardized = (windows - feature_mean) / feature_std
    return standardized.reshape(windows.shape[0], -1).astype(np.float64, copy=False)


def ensemble_predictions(
    nets: list[JEPANet],
    flat_windows: Tensor,
    target_mean: float,
    target_std: float,
) -> Tensor:
    """Per-seed un-standardized H-day return predictions, shape ``(N, K)``.

    ``mean(dim=1)`` is ``mu``; ``var(dim=1, unbiased=True)`` is the live ensemble
    disagreement. Runs in ``eval`` + ``no_grad`` (deterministic — no dropout).
    """
    per_seed: list[Tensor] = []
    with torch.no_grad():
        for net in nets:
            net.eval()
            embedding = net.encode_context(flat_windows)
            standardized = net.predict_return_std(embedding)
            per_seed.append(standardized * target_std + target_mean)
    return torch.stack(per_seed, dim=1)


# --------------------------------------------------------------------------- #
# Checkpoint (data + save)
# --------------------------------------------------------------------------- #


def hash_state_dicts(state_dicts: list[dict[str, Tensor]]) -> str:
    """sha256 of the ``torch.save``-serialized ensemble tensors (content hash).

    Stable for a given torch build (same tensors -> same bytes -> same hash on
    the machine that trained them); ``torch.save`` bytes are NOT a cross-
    platform/cross-version contract, so treat the hash as a local identity
    label (the ``model_version`` suffix), not a portable digest.
    """
    buffer = io.BytesIO()
    torch.save({"ensemble": state_dicts}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _config_to_json(config: JEPAConfig) -> dict[str, object]:
    payload = dataclasses.asdict(config)
    payload["feature_names"] = list(config.feature_names)  # tuple -> JSON list
    return payload


def _config_from_json(payload: dict[str, object]) -> JEPAConfig:
    fields = {f.name for f in dataclasses.fields(JEPAConfig)}
    kwargs: dict[str, object] = {key: value for key, value in payload.items() if key in fields}
    raw_names = kwargs.get("feature_names")
    if isinstance(raw_names, list):
        kwargs["feature_names"] = tuple(str(name) for name in raw_names)
    return JEPAConfig(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class JEPACheckpoint:
    """Everything needed to reconstruct a :class:`JEPAModel` (train_jepa's output).

    ``ensemble_state_dicts`` are the K trained :class:`JEPANet` weights;
    everything else is the metadata JSON sidecar (:meth:`save`). ``residual_var``
    is per-asset, ``pooled_residual_var`` the unseen-asset fallback — both on the
    predictive H-day scale (module docstring's sigma composite).
    """

    ensemble_state_dicts: list[dict[str, Tensor]]
    config: JEPAConfig
    feature_mean: npt.NDArray[np.float64]
    feature_std: npt.NDArray[np.float64]
    target_mean: float
    target_std: float
    residual_var: dict[str, float]
    pooled_residual_var: float
    collapse_report: CollapseReport
    checkpoint_hash: str
    fracdiff_params: dict[str, FracDiffParams] | None = None

    @property
    def model_version(self) -> str:
        """``jepa:<8 hex>`` (plan §3 ``forecasts.model_version`` shape)."""
        return f"jepa:{self.checkpoint_hash[:8]}"

    def save(self, path: str | Path) -> Path:
        """Write the ``.pt`` (tensors only) + ``.meta.json`` sidecar; return the path.

        The ``.pt`` re-serializes the same tensors hashed at construction, so
        on the torch build that trained them the on-disk bytes match
        ``checkpoint_hash`` (see :func:`hash_state_dicts` for the scope of that
        stability claim).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ensemble": self.ensemble_state_dicts}, path)
        meta: dict[str, object] = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "checkpoint_hash": self.checkpoint_hash,
            "config": _config_to_json(self.config),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "residual_var": self.residual_var,
            "pooled_residual_var": self.pooled_residual_var,
            "collapse_report": dataclasses.asdict(self.collapse_report),
            "fracdiff_params": (
                None
                if self.fracdiff_params is None
                else {
                    asset: dataclasses.asdict(params)
                    for asset, params in self.fracdiff_params.items()
                }
            ),
        }
        _meta_path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def to_model(self) -> "JEPAModel":
        """Build an in-memory :class:`JEPAModel` without a round-trip to disk."""
        return JEPAModel._from_parts(
            state_dicts=self.ensemble_state_dicts,
            config=self.config,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            residual_var=self.residual_var,
            pooled_residual_var=self.pooled_residual_var,
            checkpoint_hash=self.checkpoint_hash,
        )


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + META_SUFFIX)


# --------------------------------------------------------------------------- #
# The WorldModel implementation
# --------------------------------------------------------------------------- #


class JEPAModel:
    """A trained JEPA ensemble as a :class:`~abe.model.base.WorldModel`.

    ``forecast`` consumes the SAME ``{asset: frame}`` dict EWMA does — each frame
    carries at least the checkpoint's ``feature_names`` columns (by bare name;
    ``"log_return"`` in the V1 default). A frame missing a required column raises
    a clear ``ValueError`` (the documented enrichment-resolution rule). Per asset
    it z-scores the last ``context_len`` bars with the frozen train stats, runs
    the K-seed ensemble, and emits ``mu`` (ensemble-mean H-day log-return) and
    ``sigma = sqrt(residual_var + ensemble_var)`` on the frozen predictive scale.
    """

    def __init__(
        self,
        nets: list[JEPANet],
        config: JEPAConfig,
        *,
        feature_mean: npt.NDArray[np.float64],
        feature_std: npt.NDArray[np.float64],
        target_mean: float,
        target_std: float,
        residual_var: dict[str, float],
        pooled_residual_var: float,
        checkpoint_hash: str,
    ) -> None:
        self._nets = nets
        self._config = config
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self._target_mean = target_mean
        self._target_std = target_std
        self._residual_var = residual_var
        self._pooled_residual_var = pooled_residual_var
        self.model_version = f"jepa:{checkpoint_hash[:8]}"
        for net in self._nets:
            net.eval()

    # -- construction ---------------------------------------------------- #

    @classmethod
    def _from_parts(
        cls,
        *,
        state_dicts: list[dict[str, Tensor]],
        config: JEPAConfig,
        feature_mean: npt.NDArray[np.float64],
        feature_std: npt.NDArray[np.float64],
        target_mean: float,
        target_std: float,
        residual_var: dict[str, float],
        pooled_residual_var: float,
        checkpoint_hash: str,
    ) -> "JEPAModel":
        nets: list[JEPANet] = []
        for state_dict in state_dicts:
            net = JEPANet(config)
            net.load_state_dict(state_dict)
            nets.append(net)
        return cls(
            nets,
            config,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            residual_var=residual_var,
            pooled_residual_var=pooled_residual_var,
            checkpoint_hash=checkpoint_hash,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "JEPAModel":
        """Load a checkpoint written by :meth:`JEPACheckpoint.save`.

        Reads the ``.pt`` with ``weights_only=True`` (only tensors are stored,
        so the safe loader suffices) and the ``.meta.json`` sidecar. Raises a
        clear ``ValueError`` when either file is missing or malformed — the
        fail-loud posture; never a silent EWMA fallback.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"JEPA checkpoint not found: {path}")
        meta_path = _meta_path(path)
        if not meta_path.is_file():
            raise ValueError(
                f"JEPA checkpoint metadata sidecar not found: {meta_path} "
                "(a checkpoint is the .pt PLUS its .meta.json)"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JEPA checkpoint sidecar {meta_path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(meta, dict):
            raise ValueError(f"JEPA checkpoint sidecar {meta_path} must be a JSON object")
        # Format-version gate: a sidecar written by a different checkpoint
        # format must be rejected loudly, never half-parsed (a missing key is
        # treated as version 0 — pre-versioned == unknown == rejected).
        found_version = meta.get("format_version", 0)
        if found_version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"JEPA checkpoint format_version mismatch in {meta_path}: found "
                f"{found_version!r}, this build reads version {_CHECKPOINT_FORMAT_VERSION} — "
                "retrain (python -m abe.model.train) to regenerate the checkpoint"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — surface any torch load failure as a load error
            raise ValueError(f"failed to load JEPA tensors from {path}: {exc}") from exc
        state_dicts = payload.get("ensemble") if isinstance(payload, dict) else None
        if not isinstance(state_dicts, list) or not state_dicts:
            raise ValueError(f"JEPA checkpoint {path} has no 'ensemble' state dicts")
        try:
            config = _config_from_json(meta["config"])
            feature_mean = np.asarray(meta["feature_mean"], dtype=np.float64)
            feature_std = np.asarray(meta["feature_std"], dtype=np.float64)
            residual_var = {str(k): float(v) for k, v in meta["residual_var"].items()}
            return cls._from_parts(
                state_dicts=state_dicts,
                config=config,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_mean=float(meta["target_mean"]),
                target_std=float(meta["target_std"]),
                residual_var=residual_var,
                pooled_residual_var=float(meta["pooled_residual_var"]),
                checkpoint_hash=str(meta["checkpoint_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"JEPA checkpoint sidecar {meta_path} is malformed: {exc}") from exc

    # -- forecast (the WorldModel contract) ------------------------------ #

    def forecast(self, features: dict[str, pd.DataFrame]) -> dict[str, Forecast]:
        """See :meth:`~abe.model.base.WorldModel.forecast`; per-asset math below."""
        if not features:
            raise ValueError("features must contain at least one asset frame")
        return {asset: self._forecast_asset(asset, frame) for asset, frame in features.items()}

    def _forecast_asset(self, asset: str, frame: pd.DataFrame) -> Forecast:
        window = self._extract_window(asset, frame)
        flat = torch.from_numpy(
            standardize_windows(window[None, :, :], self._feature_mean, self._feature_std)
        ).to(torch.float32)
        preds = ensemble_predictions(self._nets, flat, self._target_mean, self._target_std)[0]
        mu = float(preds.mean().item())
        ensemble_var = float(preds.var(unbiased=True).item()) if preds.numel() > 1 else 0.0
        residual_var = self._residual_var.get(asset, self._pooled_residual_var)
        sigma_sq = max(residual_var + ensemble_var, self._config.sigma_floor**2)
        sigma = math.sqrt(sigma_sq)
        if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"JEPA produced a non-finite/degenerate forecast for {asset!r}: "
                f"mu={mu!r}, sigma={sigma!r}"
            )
        return Forecast(mu=mu, sigma=sigma)

    def _extract_window(self, asset: str, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The last ``context_len`` bars x ``F`` features for ``asset`` (guarded)."""
        missing = [name for name in self._config.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"features frame for {asset!r} lacks JEPA feature column(s) {missing}; the "
                f"checkpoint requires {list(self._config.feature_names)} (by bare name). Got "
                f"{list(frame.columns)!r} — enrich the frame or retrain with fewer features"
            )
        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                f"features frame index for {asset!r} must be sorted ascending (chronological); "
                "a window over a disordered series would be silently wrong"
            )
        matrix: npt.NDArray[np.float64] = frame.loc[:, list(self._config.feature_names)].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(matrix).all():
            flat_index = frame.index[np.flatnonzero(~np.isfinite(matrix).all(axis=1))]
            raise ValueError(
                f"features for {asset!r} contain NaN/non-finite values: "
                f"{non_finite_summary(flat_index, np.zeros(len(flat_index)))}"
            )
        context_len = self._config.context_len
        if len(matrix) < context_len:
            raise ValueError(
                f"history for {asset!r} too short for the JEPA window: {len(matrix)} bars < "
                f"context_len={context_len}"
            )
        return matrix[-context_len:, :]
