"""Deterministic feature matrix for the JEPA path (plan.md Step 12).

Assembles the per-asset feature matrix the minimal JEPA (Step 13) consumes:
daily ``log_return`` + annualized ``realized_vol`` (both from
``abe.calc``, the EWMA path's building blocks) + an optional
frac-differenced log-price column per asset, joined with FRED macro columns
**without lookahead**.

**Column shape (one source of truth).** Flat ``"ASSET_feature"`` names
(:func:`feature_column`), chosen over a MultiIndex for JEPA friendliness — the
model reads a plain 2-D float matrix and a flat column list needs no
``get_level_values`` plumbing. Per-asset feature names reuse the abe.calc /
fracdiff.py constants (``log_return``, ``realized_vol``, ``fracdiff``); macro
columns are the FRED ``series_id`` strings verbatim. Universe tickers carry no
``_`` so ``ASSET_feature`` stays unambiguous.

**No macro lookahead (plan section 9 mitigation).** Macro columns are joined
with ``pd.merge_asof(direction="backward")`` on the FRED ``available_date``
(``obs_date + release lag``), so a trading date ``t`` sees only observations
whose ``available_date <= t`` — never a value published after ``t``. Missing
macro (``None`` / empty — the degraded mode) yields the matrix WITHOUT macro
columns, never an error.

**Determinism.** Same ``(prices, macro, params)`` in → byte-identical matrix out
(pinned by a hash-stability test). Every step is a pure transform; asset column
order follows the ``prices`` dict order, macro columns are sorted by
``series_id``.

**Frac-diff columns are opt-in.** When ``params`` is provided, each asset's
log-price is frac-differenced with that asset's FROZEN
:class:`~abe.afml.fracdiff.FracDiffParams` (computed once per training fold —
see fracdiff.py's leakage warning). When ``params is None`` the frac-diff
columns are omitted entirely (EWMA-path compatibility).
"""

from typing import Final

import numpy as np
import pandas as pd

from abe.afml.fracdiff import FRACDIFF_COLUMN, FracDiffParams, fracdiff_series
from abe.calc import LOG_RETURN_COLUMN, REALIZED_VOL_COLUMN, log_returns, realized_vol

__all__ = [
    "FEATURE_NAME_SEP",
    "MACRO_AVAILABLE_DATE_COLUMN",
    "MACRO_OBS_DATE_COLUMN",
    "MACRO_SERIES_ID_COLUMN",
    "MACRO_VALUE_COLUMN",
    "build_features",
    "feature_column",
]

FEATURE_NAME_SEP: Final[str] = "_"
"""Separator in the flat ``"ASSET<sep>feature"`` column name (see
:func:`feature_column`)."""

MACRO_SERIES_ID_COLUMN: Final[str] = "series_id"
MACRO_OBS_DATE_COLUMN: Final[str] = "obs_date"
MACRO_VALUE_COLUMN: Final[str] = "value"
MACRO_AVAILABLE_DATE_COLUMN: Final[str] = "available_date"
"""Long-form macro-table column names (the ``macro`` table shape, plan section
3) that :func:`build_features` reads. One source of truth for the join keys."""


def feature_column(asset: str, feature: str) -> str:
    """Flat feature-matrix column name for one asset's feature.

    ``feature_column("SPY", "log_return") -> "SPY_log_return"``. The inverse
    (splitting on :data:`FEATURE_NAME_SEP`) is unambiguous only because universe
    tickers contain no separator.
    """
    return f"{asset}{FEATURE_NAME_SEP}{feature}"


def build_features(
    prices: dict[str, pd.Series],
    macro: pd.DataFrame | None,
    params: dict[str, FracDiffParams] | None = None,
) -> pd.DataFrame:
    """Deterministic per-asset feature matrix + lookahead-free macro columns.

    ``prices`` maps each asset to its adjusted-close series (ascending ISO-8601
    ``YYYY-MM-DD`` index — the CacheAdapter shape). For every asset the matrix
    carries ``log_return`` and ``realized_vol``; when ``params`` is provided it
    also carries a ``fracdiff`` column (fixed-width FFD of the log-price under
    that asset's frozen params). Per-asset warm-up rows (the return/vol/FFD
    windows) are dropped via an inner join, then assets are inner-joined on their
    common dates — the result has no per-asset NaN and a dense, JEPA-ready shape.

    ``macro`` is the long-form macro table (columns ``series_id``, ``obs_date``,
    ``value``, ``available_date``). Each ``series_id`` becomes a column matched
    to each trading date ``t`` by ``merge_asof`` backward on ``available_date``
    (no lookahead). ``macro is None`` or an empty frame → no macro columns
    (degraded mode), never an error. Macro columns may carry leading NaN for
    dates before a series' first ``available_date`` — a genuine point-in-time
    gap, left for the consumer to handle.

    Raises ``ValueError`` on empty ``prices``, or (when ``params`` is provided)
    on any asset missing from ``params``. Per-asset guards (ascending index,
    NaN/non-finite, positivity, minimum length) are enforced by the underlying
    abe.calc + fracdiff.py functions.
    """
    if not prices:
        raise ValueError("prices must contain at least one asset series")
    if params is not None:
        missing = [asset for asset in prices if asset not in params]
        if missing:
            raise ValueError(
                f"params provided but missing frac-diff params for assets {missing}; "
                "supply one FracDiffParams per asset (frozen per training fold) or pass "
                "params=None to omit frac-diff columns"
            )

    per_asset_frames: list[pd.DataFrame] = []
    for asset, price in prices.items():
        columns: dict[str, pd.Series] = {}
        returns = log_returns(price)
        columns[feature_column(asset, LOG_RETURN_COLUMN)] = returns
        columns[feature_column(asset, REALIZED_VOL_COLUMN)] = realized_vol(returns)
        if params is not None:
            log_price = pd.Series(
                np.log(price.to_numpy(dtype=float)), index=price.index, name=asset
            )
            asset_params = params[asset]
            fracdiffed = fracdiff_series(log_price, asset_params.d, asset_params.threshold)
            columns[feature_column(asset, FRACDIFF_COLUMN)] = fracdiffed
        # Inner join drops each asset's warm-up rows (the longest feature window wins).
        per_asset_frames.append(pd.concat(columns, axis=1, join="inner"))

    matrix = pd.concat(per_asset_frames, axis=1, join="inner")
    matrix.index.name = prices[next(iter(prices))].index.name

    if macro is not None and not macro.empty:
        macro_frame = _join_macro_no_lookahead(matrix.index, macro)
        matrix = pd.concat([matrix, macro_frame], axis=1)
    return matrix


def _join_macro_no_lookahead(trading_dates: pd.Index, macro: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time macro columns for ``trading_dates`` (merge_asof backward).

    For each ``series_id`` (columns sorted for determinism) a trading date ``t``
    gets the value of the latest observation with ``available_date <= t``. Right
    rows are sorted by ``(available_date, obs_date)`` so ties on
    ``available_date`` resolve to the most recent ``obs_date`` (the freshest
    observation available at ``t``). Returns a frame indexed like
    ``trading_dates`` (original ISO strings preserved).
    """
    left = pd.DataFrame({"_trading_date": pd.to_datetime(trading_dates)})
    columns: dict[str, np.ndarray] = {}
    for series_id in sorted(macro[MACRO_SERIES_ID_COLUMN].unique()):
        right = macro.loc[
            macro[MACRO_SERIES_ID_COLUMN] == series_id,
            [MACRO_AVAILABLE_DATE_COLUMN, MACRO_OBS_DATE_COLUMN, MACRO_VALUE_COLUMN],
        ].copy()
        right["_available"] = pd.to_datetime(right[MACRO_AVAILABLE_DATE_COLUMN])
        right["_obs"] = pd.to_datetime(right[MACRO_OBS_DATE_COLUMN])
        right = right.sort_values(["_available", "_obs"]).reset_index(drop=True)
        merged = pd.merge_asof(
            left,
            right[["_available", MACRO_VALUE_COLUMN]],
            left_on="_trading_date",
            right_on="_available",
            direction="backward",
        )
        columns[str(series_id)] = merged[MACRO_VALUE_COLUMN].to_numpy()
    return pd.DataFrame(columns, index=trading_dates)
