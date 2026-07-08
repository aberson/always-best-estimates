"""Offline JEPA training (plan.md Step 13) — the ``train_jepa`` entrypoint + CLI.

This module NEVER runs on the hot recompute path. It is invoked offline by an
operator to produce a checkpoint that ``JEPAModel.from_checkpoint`` later serves
through the frozen ``WorldModel`` interface:

    uv run python -m abe.model.train --db data/abe.db --out checkpoints/jepa.pt

Flow: load real prices via ``CacheAdapter`` -> ``build_features`` -> assemble
no-lookahead ``(context, target, return)`` windows -> a PURGED walk-forward split
(expanding window + H-bar gap, so no training label's horizon reaches the
holdout) -> train the K-seed ensemble -> hard-fail on embedding collapse ->
estimate the per-asset residual variance on the holdout -> write the checkpoint
(``model/jepa.py`` owns the file format).

Two measurement-validity anchors (plan §12) live here:

- **Collapse hard-fail** — :func:`train_jepa` raises
  :class:`~abe.model.jepa.CollapseError` when VICReg fails to keep the
  embeddings from collapsing (a garbage-anchor test drives this).
- **Shuffled-target control** — :func:`shuffled_target_control` trains one model
  on the real feature->return relationship and one on TARGET-SHUFFLED pairs, and
  returns both held-out prediction losses. The CI test asserts real < shuffled
  (a genuine relationship must beat a destroyed one); a small model on tiny data
  is enough to show the gap.

Standardization stats (per-feature z-score, H-day-return z-score) are computed on
TRAIN windows only (no lookahead) and frozen into the checkpoint; the residual
variance and every forecast are un-standardized back to the real H-day
log-return scale (so ``sigma`` lands on the Step 5 predictive scale).
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 — the conventional torch alias

from abe.afml.fracdiff import FracDiffParams
from abe.afml.purged_cv import purged_walk_forward_splits, validate_no_leakage
from abe.constants import UNIVERSE
from abe.features.basic import log_returns
from abe.features.build import build_features, feature_column
from abe.ingest.sources import CacheAdapter
from abe.model.jepa import (
    CollapseError,
    CollapseReport,
    JEPACheckpoint,
    JEPAConfig,
    JEPANet,
    collapse_metrics,
    count_parameters,
    ensemble_predictions,
    hash_state_dicts,
    standardize_windows,
)

__all__ = [
    "build_config_from_args",
    "load_training_data",
    "main",
    "shuffled_target_control",
    "train_jepa",
]

logger = logging.getLogger(__name__)

_STD_FLOOR: Final[float] = 1e-8
"""Below this, a standardization std is treated as degenerate and replaced by
1.0 (a constant feature/target carries no scale information to divide by)."""


# --------------------------------------------------------------------------- #
# VICReg loss terms + the JEPA objective
# --------------------------------------------------------------------------- #


def _variance_term(z: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance hinge: penalize per-dim std below ``gamma`` (anti-collapse)."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))


def _covariance_term(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance penalty: squared off-diagonal covariance per dim (decorrelate)."""
    n_samples, dim = z.shape
    centered = z - z.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(n_samples - 1, 1)
    off_diagonal = cov - torch.diag(torch.diagonal(cov))
    return off_diagonal.pow(2).sum() / dim


def _jepa_loss(
    net: JEPANet,
    context_x: torch.Tensor,
    target_x: torch.Tensor,
    return_std: torch.Tensor,
    config: JEPAConfig,
) -> torch.Tensor:
    """The joint JEPA objective: prediction MSE + VICReg var/cov + lambda_ret * return MSE."""
    context_embedding = net.encode_context(context_x)
    target_embedding = net.encode_target(target_x)  # EMA branch, no grad
    predicted_embedding = net.predict_latent(context_embedding)

    invariance = F.mse_loss(predicted_embedding, target_embedding)
    variance = _variance_term(context_embedding, config.vicreg_gamma) + _variance_term(
        predicted_embedding, config.vicreg_gamma
    )
    covariance = _covariance_term(context_embedding) + _covariance_term(predicted_embedding)
    return_pred = net.predict_return_std(context_embedding)
    return_loss = F.mse_loss(return_pred, return_std)

    return (
        config.inv_weight * invariance
        + config.var_weight * variance
        + config.cov_weight * covariance
        + config.return_weight * return_loss
    )


def _train_single(
    context_x: torch.Tensor,
    target_x: torch.Tensor,
    return_std: torch.Tensor,
    config: JEPAConfig,
    *,
    seed: int,
) -> JEPANet:
    """Train one seed to convergence (full-batch, deterministic given ``seed``)."""
    torch.manual_seed(seed)
    net = JEPANet(config)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.lr)
    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = _jepa_loss(net, context_x, target_x, return_std, config)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        net.ema_update(config.ema_momentum)
    net.eval()
    return net


# --------------------------------------------------------------------------- #
# Sample assembly (no-lookahead windows + purged holdout)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Prepared:
    """Assembled train/holdout window batches + frozen standardization stats."""

    train_context: npt.NDArray[np.float64]  # (Ntr, C, F)
    train_target: npt.NDArray[np.float64]
    train_return: npt.NDArray[np.float64]  # (Ntr,) realized H-day log-returns
    hold_context: npt.NDArray[np.float64]  # (Nho, C, F)
    hold_return: npt.NDArray[np.float64]
    hold_assets: list[str]
    feature_mean: npt.NDArray[np.float64]  # (F,)
    feature_std: npt.NDArray[np.float64]  # (F,)
    target_mean: float
    target_std: float
    assets: list[str]


def _asset_arrays(
    features_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    config: JEPAConfig,
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]], list[str]]:
    """Per-asset ``(T, F)`` feature array + aligned ``(T,)`` daily-return array."""
    if not features_matrix.index.is_monotonic_increasing:
        raise ValueError("features_matrix index must be sorted ascending (chronological order)")
    assets = [str(asset) for asset in returns.columns]
    if not assets:
        raise ValueError("returns must have at least one asset column")
    aligned_returns = returns.reindex(features_matrix.index)
    feature_arrays: dict[str, npt.NDArray[np.float64]] = {}
    return_arrays: dict[str, npt.NDArray[np.float64]] = {}
    for asset in assets:
        columns = [feature_column(asset, name) for name in config.feature_names]
        missing = [column for column in columns if column not in features_matrix.columns]
        if missing:
            raise ValueError(
                f"features_matrix lacks column(s) {missing} for asset {asset!r}; expected the flat "
                f"'{{asset}}_{{feature}}' columns from build_features for features "
                f"{list(config.feature_names)}"
            )
        matrix = features_matrix.loc[:, columns].to_numpy(dtype=np.float64)
        asset_returns = aligned_returns[asset].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all() or not np.isfinite(asset_returns).all():
            raise ValueError(
                f"features/returns for {asset!r} contain NaN/non-finite values after aligning to "
                "the feature-matrix index (it must be a subset of returns.index)"
            )
        feature_arrays[asset] = matrix
        return_arrays[asset] = asset_returns
    return feature_arrays, return_arrays, assets


def _assemble(
    feature_arrays: dict[str, npt.NDArray[np.float64]],
    return_arrays: dict[str, npt.NDArray[np.float64]],
    ts: npt.NDArray[np.int_],
    config: JEPAConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
    """Gather ``(context, target, return, asset)`` for every ``(asset, t)`` in ``ts``."""
    c = config.context_len
    h = config.horizon
    contexts: list[npt.NDArray[np.float64]] = []
    targets: list[npt.NDArray[np.float64]] = []
    returns_list: list[float] = []
    sample_assets: list[str] = []
    for asset, matrix in feature_arrays.items():
        daily = return_arrays[asset]
        for raw_t in ts:
            t = int(raw_t)
            contexts.append(matrix[t - c + 1 : t + 1, :])
            targets.append(matrix[t + h - c + 1 : t + h + 1, :])
            returns_list.append(float(daily[t + 1 : t + h + 1].sum()))
            sample_assets.append(asset)
    context_array = np.stack(contexts).astype(np.float64)
    target_array = np.stack(targets).astype(np.float64)
    return_array = np.asarray(returns_list, dtype=np.float64)
    return context_array, target_array, return_array, sample_assets


def _purged_holdout_ts(
    valid_ts: npt.NDArray[np.int_], config: JEPAConfig
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Split the valid context-end bars into ``(train_ts, holdout_ts)`` with an H-bar gap.

    The holdout is the trailing ``holdout_fraction`` of the valid bars; training
    bars whose H-day label reaches into the first holdout bar are PURGED. The
    split is handed to :func:`~abe.afml.purged_cv.validate_no_leakage` (the tested
    leakage guard) so ``max(train)+H < min(holdout)`` is enforced, not just
    intended.
    """
    n_valid = len(valid_ts)
    n_holdout = max(2, int(round(config.holdout_fraction * n_valid)))
    holdout_start = n_valid - n_holdout
    if holdout_start < 1:
        raise ValueError(
            f"too few windows ({n_valid}) for a train/holdout split at holdout_fraction="
            f"{config.holdout_fraction}; supply a longer price history"
        )
    holdout_ts = valid_ts[holdout_start:]
    first_holdout_t = int(holdout_ts[0])
    train_mask = valid_ts[:holdout_start] + config.horizon < first_holdout_t
    train_ts = valid_ts[:holdout_start][train_mask]
    if train_ts.size == 0:
        raise ValueError(
            "the H-bar purge emptied the training set — supply a longer history or a smaller "
            "horizon/holdout_fraction"
        )
    validate_no_leakage(
        [(train_ts.astype(np.int_), holdout_ts.astype(np.int_))], horizon=config.horizon
    )
    return train_ts.astype(np.int_), holdout_ts.astype(np.int_)


def _prepare(features_matrix: pd.DataFrame, returns: pd.DataFrame, config: JEPAConfig) -> _Prepared:
    """Assemble train/holdout batches + TRAIN-only standardization stats."""
    feature_arrays, return_arrays, assets = _asset_arrays(features_matrix, returns, config)
    n_bars = len(features_matrix)
    first_t = config.context_len - 1
    last_t = n_bars - 1 - config.horizon
    if last_t < first_t:
        raise ValueError(
            f"price history too short: {n_bars} bars yield no valid JEPA windows for "
            f"context_len={config.context_len} + horizon={config.horizon}"
        )
    valid_ts = np.arange(first_t, last_t + 1, dtype=np.int_)
    # Reuse the tested purge geometry as a sanity check that the config even admits a split.
    purged_walk_forward_splits(len(valid_ts), n_splits=1, horizon=config.horizon)
    train_ts, holdout_ts = _purged_holdout_ts(valid_ts, config)

    train_context, train_target, train_return, _ = _assemble(
        feature_arrays, return_arrays, train_ts, config
    )
    hold_context, _, hold_return, hold_assets = _assemble(
        feature_arrays, return_arrays, holdout_ts, config
    )

    feature_flat = train_context.reshape(-1, config.n_features)
    feature_mean = feature_flat.mean(axis=0)
    feature_std = feature_flat.std(axis=0, ddof=0)
    feature_std = np.where(feature_std < _STD_FLOOR, 1.0, feature_std)
    target_mean = float(train_return.mean())
    raw_target_std = float(train_return.std(ddof=0))
    target_std = raw_target_std if raw_target_std > _STD_FLOOR else 1.0

    return _Prepared(
        train_context=train_context,
        train_target=train_target,
        train_return=train_return,
        hold_context=hold_context,
        hold_return=hold_return,
        hold_assets=hold_assets,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        assets=assets,
    )


def _to_tensor(array: npt.NDArray[np.float64]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float64)).to(torch.float32)


# --------------------------------------------------------------------------- #
# The training entrypoint
# --------------------------------------------------------------------------- #


def train_jepa(
    features_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    config: JEPAConfig,
    fracdiff_params: dict[str, FracDiffParams] | None = None,
) -> JEPACheckpoint:
    """Train the K-seed JEPA ensemble and return a :class:`JEPACheckpoint`.

    ``features_matrix`` is the flat ``build_features`` matrix (``{asset}_{feature}``
    columns); ``returns`` is per-asset daily log-returns (columns == assets),
    aligned to (a superset of) the matrix index. Raises
    :class:`~abe.model.jepa.CollapseError` on embedding collapse, and
    ``ValueError`` on a history too short for a purged split.

    ``fracdiff_params`` (frozen per-fold FracDiffParams) is passed straight into
    the checkpoint for provenance when the feature set includes ``fracdiff``; it
    does not affect training here.
    """
    prepared = _prepare(features_matrix, returns, config)

    context_x = _to_tensor(
        standardize_windows(prepared.train_context, prepared.feature_mean, prepared.feature_std)
    )
    target_x = _to_tensor(
        standardize_windows(prepared.train_target, prepared.feature_mean, prepared.feature_std)
    )
    return_std = _to_tensor((prepared.train_return - prepared.target_mean) / prepared.target_std)

    nets: list[JEPANet] = [
        _train_single(context_x, target_x, return_std, config, seed=config.seed + k)
        for k in range(config.n_seeds)
    ]

    hold_x = _to_tensor(
        standardize_windows(prepared.hold_context, prepared.feature_mean, prepared.feature_std)
    )
    report = _collapse_report(nets, hold_x, config)
    if not report.passed:
        raise CollapseError(report)

    residual_var, pooled_residual_var = _residual_variances(nets, hold_x, prepared, config)

    state_dicts = [_cpu_state_dict(net) for net in nets]
    return JEPACheckpoint(
        ensemble_state_dicts=state_dicts,
        config=config,
        feature_mean=prepared.feature_mean,
        feature_std=prepared.feature_std,
        target_mean=prepared.target_mean,
        target_std=prepared.target_std,
        residual_var=residual_var,
        pooled_residual_var=pooled_residual_var,
        collapse_report=report,
        checkpoint_hash=hash_state_dicts(state_dicts),
        fracdiff_params=fracdiff_params,
    )


def _collapse_report(
    nets: list[JEPANet], hold_x: torch.Tensor, config: JEPAConfig
) -> CollapseReport:
    """Ensemble-worst collapse metrics on the holdout context embeddings."""
    worst_std = math.inf
    worst_rank = math.inf
    for net in nets:
        with torch.no_grad():
            embedding = net.encode_context(hold_x)
        std_median, effective_rank = collapse_metrics(embedding)
        worst_std = min(worst_std, std_median)
        worst_rank = min(worst_rank, effective_rank)
    return CollapseReport(
        per_dim_std_median=worst_std,
        effective_rank=worst_rank,
        latent_dim=config.latent_dim,
        std_median_threshold=config.std_median_threshold,
        effective_rank_threshold=config.effective_rank_threshold,
        n_samples=int(hold_x.shape[0]),
    )


def _residual_variances(
    nets: list[JEPANet],
    hold_x: torch.Tensor,
    prepared: _Prepared,
    config: JEPAConfig,
) -> tuple[dict[str, float], float]:
    """Per-asset (+ pooled) purged-holdout residual variance — the sigma scale term."""
    predictions = ensemble_predictions(nets, hold_x, prepared.target_mean, prepared.target_std)
    mu = predictions.mean(dim=1).numpy()
    residuals = prepared.hold_return - mu
    floor = config.sigma_floor**2
    pooled = max(float(np.var(residuals, ddof=1)) if residuals.size >= 2 else 0.0, floor)
    hold_assets = np.asarray(prepared.hold_assets)
    residual_var: dict[str, float] = {}
    for asset in prepared.assets:
        asset_residuals = residuals[hold_assets == asset]
        if asset_residuals.size >= 2:
            residual_var[asset] = max(float(np.var(asset_residuals, ddof=1)), floor)
        else:
            residual_var[asset] = pooled
    return residual_var, pooled


def _cpu_state_dict(net: JEPANet) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


# --------------------------------------------------------------------------- #
# Shuffled-target control (plan §12 measurement-validity anchor)
# --------------------------------------------------------------------------- #


def _holdout_return_mse(
    net: JEPANet, hold_context: torch.Tensor, hold_return_std: torch.Tensor
) -> float:
    """Held-out standardized return-head MSE (the control's comparison metric)."""
    with torch.no_grad():
        embedding = net.encode_context(hold_context)
        prediction = net.predict_return_std(embedding)
        mse = F.mse_loss(prediction, hold_return_std)
    return float(mse.item())


def shuffled_target_control(
    features_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    config: JEPAConfig,
    *,
    seed: int = 0,
) -> tuple[float, float]:
    """Train real vs target-shuffled models; return ``(real_mse, shuffled_mse)``.

    Both are evaluated on the SAME correctly-paired holdout (standardized return
    MSE). The shuffled model is trained on permuted ``(context, return)`` pairs —
    the temporal feature->return relationship is destroyed — so on a genuine
    signal it can do no better than predicting the mean, while the real model
    exploits the relationship. The CI test asserts ``real_mse < shuffled_mse``.
    Uses one seed each (fast); ``config.n_seeds`` is ignored here.
    """
    prepared = _prepare(features_matrix, returns, config)
    context_x = _to_tensor(
        standardize_windows(prepared.train_context, prepared.feature_mean, prepared.feature_std)
    )
    target_x = _to_tensor(
        standardize_windows(prepared.train_target, prepared.feature_mean, prepared.feature_std)
    )
    return_std = (prepared.train_return - prepared.target_mean) / prepared.target_std
    return_std_t = _to_tensor(return_std)

    hold_x = _to_tensor(
        standardize_windows(prepared.hold_context, prepared.feature_mean, prepared.feature_std)
    )
    hold_return_std = _to_tensor(
        (prepared.hold_return - prepared.target_mean) / prepared.target_std
    )

    real_net = _train_single(context_x, target_x, return_std_t, config, seed=seed)
    real_mse = _holdout_return_mse(real_net, hold_x, hold_return_std)

    permutation = np.random.default_rng(seed).permutation(len(return_std))
    shuffled_return_t = _to_tensor(return_std[permutation])
    shuffled_net = _train_single(context_x, target_x, shuffled_return_t, config, seed=seed)
    shuffled_mse = _holdout_return_mse(shuffled_net, hold_x, hold_return_std)

    return real_mse, shuffled_mse


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_training_data(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``UNIVERSE`` prices from the cache and build ``(features_matrix, returns)``.

    Reads adjusted closes via ``CacheAdapter`` (zero network), builds the default
    log-return + realized-vol feature matrix (no frac-diff/macro in V1), and the
    per-asset daily log-returns frame. Raises ``ValueError`` if any asset has no
    cached history.
    """
    adapter = CacheAdapter(db_path)
    prices: dict[str, pd.Series] = {}
    returns_columns: dict[str, pd.Series] = {}
    for asset in UNIVERSE:
        frame = adapter.fetch(asset)
        if frame.empty:
            raise ValueError(
                f"no cached prices for {asset!r} in {db_path} — run the price backfill first"
            )
        close = frame["adj_close"]
        prices[asset] = close
        returns_columns[asset] = log_returns(close)
    features_matrix = build_features(prices, macro=None, params=None)
    returns = pd.DataFrame(returns_columns)
    return features_matrix, returns


def build_config_from_args(args: argparse.Namespace) -> JEPAConfig:
    """Build a :class:`JEPAConfig` from parsed CLI args (defaults == JEPAConfig defaults)."""
    defaults = JEPAConfig()
    return JEPAConfig(
        context_len=args.context_len if args.context_len is not None else defaults.context_len,
        latent_dim=args.latent_dim if args.latent_dim is not None else defaults.latent_dim,
        hidden_dim=args.hidden_dim if args.hidden_dim is not None else defaults.hidden_dim,
        n_seeds=args.seeds if args.seeds is not None else defaults.n_seeds,
        epochs=args.epochs if args.epochs is not None else defaults.epochs,
        seed=args.seed if args.seed is not None else defaults.seed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m abe.model.train",
        description="Train the minimal JEPA ensemble offline and write a checkpoint.",
    )
    parser.add_argument("--db", required=True, help="SQLite price cache path (read-only).")
    parser.add_argument("--out", required=True, help="Checkpoint .pt path to write.")
    parser.add_argument("--seeds", type=int, default=None, help="Ensemble size K (default 3).")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per seed.")
    parser.add_argument("--context-len", type=int, default=None, dest="context_len")
    parser.add_argument("--latent-dim", type=int, default=None, dest="latent_dim")
    parser.add_argument("--hidden-dim", type=int, default=None, dest="hidden_dim")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: train from a cache db and write a checkpoint. Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    config = build_config_from_args(args)
    features_matrix, returns = load_training_data(args.db)
    logger.info(
        "training JEPA: %d bars, %d assets, K=%d, context_len=%d, latent_dim=%d, epochs=%d",
        len(features_matrix),
        len(returns.columns),
        config.n_seeds,
        config.context_len,
        config.latent_dim,
        config.epochs,
    )
    checkpoint = train_jepa(features_matrix, returns, config)
    out_path = checkpoint.save(args.out)
    per_net = count_parameters(JEPANet(config))
    logger.info("wrote checkpoint %s (sidecar %s.meta.json)", out_path, out_path)
    logger.info("model_version=%s", checkpoint.model_version)
    logger.info("params/seed=%d, total(K=%d)=%d", per_net, config.n_seeds, per_net * config.n_seeds)
    logger.info(
        "collapse: per-dim std median=%.4g, effective rank=%.4g (passed=%s)",
        checkpoint.collapse_report.per_dim_std_median,
        checkpoint.collapse_report.effective_rank,
        checkpoint.collapse_report.passed,
    )
    logger.info(
        "residual_var=%s (pooled=%.4g)",
        checkpoint.residual_var,
        checkpoint.pooled_residual_var,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — module CLI entrypoint
    raise SystemExit(main())
