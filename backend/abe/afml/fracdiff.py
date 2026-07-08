"""Fixed-width fractional differentiation (FFD) — Lopez de Prado, AFML ch. 5.

Fractional differentiation makes a series stationary (so an ADF test passes)
while retaining as much *memory* (correlation with the original level series) as
integer differencing throws away. Integer d=1 on a log-price series is the
ordinary log-return: stationary, but memoryless. FFD finds the SMALLEST d that
buys stationarity, keeping the most memory for the downstream model.

Three public pieces (plan.md Step 12, module ``afml/fracdiff.py``):

- :func:`ffd_weights` — the fixed-width weight sequence of ``(1 - B)**d``,
  truncated where a weight falls below ``threshold``. ``w_0 = 1``;
  ``w_k = -w_{k-1} * (d - k + 1) / k``.
- :func:`fracdiff_series` — apply those weights as a fixed-width convolution to
  an ascending ISO-indexed (log-price) series. Warm-up rows shorter than the
  weight window are DROPPED, never emitted as NaN (the abe.calc NaN
  contract). Index-order and non-finite guards mirror abe.calc.
- :func:`min_d_search` — smallest d on a grid whose FFD series passes an ADF
  stationarity test, returning a frozen :class:`FracDiffParams`.

**Determinism / frozen params.** :func:`fracdiff_series` is a pure convolution:
the same series + the same ``(d, threshold)`` give a byte-identical result.
:func:`min_d_search` is meant to be run ONCE per training fold; the resulting
:class:`FracDiffParams` are FROZEN and reused verbatim at inference time
(features/build.py applies frozen params via :func:`fracdiff_series`). The
search never runs on the hot 5-minute recompute path — only offline, on a
training fold.

**Leakage warning (loud, on purpose).** :func:`min_d_search` chooses ``d`` by
looking at the WHOLE series it is handed. Calling it on the full price history
and then reusing that ``d`` to build features over the same history LEAKS the
test window into the transform choice. The caller MUST pass TRAINING-fold data
only (Step 13/14 call it per purged walk-forward fold); the frozen ``d`` is then
applied to later folds through :func:`fracdiff_series`.

**Small-d window blow-up (why the search skips some grid points).** At a fixed
``threshold``, the FFD window grows as ``d`` shrinks (the weights of a nearly-
undifferenced series decay very slowly): at ``threshold=1e-5`` a ``d=0.10``
window is ~4000 terms. A grid ``d`` whose window exceeds the available series
(leaving fewer than :data:`MIN_ADF_SAMPLES` points) is not *testable* on that
fold, so :func:`min_d_search` SKIPS it rather than crashing — the search reports
the smallest *feasible* passing ``d``.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from abe.calc import non_finite_summary

__all__ = [
    "DEFAULT_D_GRID",
    "DEFAULT_THRESHOLD",
    "FRACDIFF_COLUMN",
    "MIN_ADF_SAMPLES",
    "FracDiffParams",
    "ffd_weights",
    "fracdiff_series",
    "min_d_search",
]

DEFAULT_THRESHOLD: Final[float] = 1e-5
"""Default weight-truncation threshold: drop the tail of the FFD weight
sequence once ``|w_k|`` falls below this (Lopez de Prado's fixed-width rule)."""

FRACDIFF_COLUMN: Final[str] = "fracdiff"
"""Series name for a standalone FFD output. features/build.py composes the
per-asset flat column name as ``f"{asset}_{FRACDIFF_COLUMN}"``."""

MIN_ADF_SAMPLES: Final[int] = 100
"""Minimum FFD-series length for an ADF test to be meaningful. A grid ``d``
whose window leaves fewer than this many points on the given series is skipped
by :func:`min_d_search` (untestable on this fold), not an error."""

DEFAULT_D_GRID: Final[npt.NDArray[np.float64]] = np.round(np.linspace(0.0, 1.0, 21), 2)
"""Default differencing grid ``[0.00, 0.05, ..., 1.00]`` (step 0.05), ascending
so :func:`min_d_search` returns the smallest passing ``d`` first."""


@dataclass(frozen=True, slots=True)
class FracDiffParams:
    """Frozen result of a per-training-fold :func:`min_d_search`.

    Computed ONCE on a training fold and reused verbatim at inference time via
    :func:`fracdiff_series` (``d`` + ``threshold`` fully determine the
    transform). Persisting these makes feature regeneration deterministic and
    lookahead-free.

    - ``d`` — the chosen fractional differencing order.
    - ``threshold`` — the weight-truncation threshold used (fixes ``window_len``).
    - ``window_len`` — number of FFD weights (the fixed convolution width); the
      first ``window_len - 1`` rows of any transformed series are warm-up.
    - ``adf_pvalue_achieved`` — the ADF p-value of the FFD series at ``d`` (< the
      requested ``adf_pvalue``; this is the stationarity evidence).
    - ``corr_with_original`` — Pearson correlation of the FFD series with the
      original level series over the shared (post-warm-up) index: the memory
      retained (1.0 at ``d=0``, small near ``d=1``).
    """

    d: float
    threshold: float
    window_len: int
    adf_pvalue_achieved: float
    corr_with_original: float


def ffd_weights(d: float, threshold: float = DEFAULT_THRESHOLD) -> npt.NDArray[np.float64]:
    """Fixed-width FFD weight sequence of ``(1 - B)**d``, in natural (lag) order.

    ``w[0] = 1`` multiplies the current observation; ``w[k]`` multiplies the
    ``k``-th lag, via the recurrence ``w_k = -w_{k-1} * (d - k + 1) / k``. The
    sequence is truncated at the first ``k`` with ``|w_k| < threshold`` (the
    fixed-width rule), so the returned length is the convolution window.

    Sign/decay shape: ``w[0] = 1 > 0``; for a fractional ``0 < d < 1`` every
    later weight is negative (``w_1 = -d``, and the ``(-1)^k * C(d, k)``
    generalized-binomial terms stay negative) with strictly decreasing
    magnitude. Integer ``d`` alternates and terminates exactly: ``d=1`` gives
    ``[1, -1]``, ``d=0`` gives ``[1]``.

    Raises ``ValueError`` on a non-positive or non-finite ``threshold``, or a
    negative/non-finite ``d`` (negative d integrates — weights stop decaying
    and the truncation loop would never terminate).
    """
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be finite and > 0, got {threshold!r}")
    if not np.isfinite(d) or d < 0.0:
        raise ValueError(f"d must be finite and >= 0, got {d!r}")
    weights: list[float] = [1.0]
    k = 1
    while True:
        next_weight = -weights[-1] * (d - k + 1) / k
        if abs(next_weight) < threshold:
            break
        weights.append(next_weight)
        k += 1
    return np.asarray(weights, dtype=np.float64)


def fracdiff_series(
    series: pd.Series,
    d: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.Series:
    """Fixed-width fractional differentiation of an ascending ISO-indexed series.

    ``out[t] = sum_k w_k * series[t - k]`` over the fixed weight window
    (:func:`ffd_weights`), computed as ``np.convolve(values, weights,
    'valid')``. The first ``window_len - 1`` rows have an incomplete window and
    are DROPPED (never NaN); the result is indexed by
    ``series.index[window_len - 1:]`` and named :data:`FRACDIFF_COLUMN`.

    Applying a persisted ``(d, threshold)`` here is fully deterministic — this
    is the frozen-params inference path.

    Guards mirror abe.calc (order-dependence and NaN must fail loud,
    never compute silently wrong): raises ``ValueError`` on a non-ascending
    index, on NaN/non-finite values (naming the offending rows), and when the
    series is shorter than the weight window (the result would be silently
    empty).
    """
    weights = ffd_weights(d, threshold)
    window_len = len(weights)
    values = series.to_numpy(dtype=float)
    if not series.index.is_monotonic_increasing:
        raise ValueError(
            "series index must be sorted ascending (chronological order); "
            "fixed-width FFD over a disordered series would be silently wrong"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "series contains NaN/non-finite values: "
            f"{non_finite_summary(series.index, values)}; clean the series first"
        )
    if len(values) < window_len:
        raise ValueError(
            f"series length {len(values)} is shorter than the FFD window {window_len} "
            f"(d={d!r}, threshold={threshold!r}); the FFD series would be empty"
        )
    fracdiffed = np.convolve(values, weights, mode="valid")
    return pd.Series(fracdiffed, index=series.index[window_len - 1 :], name=FRACDIFF_COLUMN)


def min_d_search(
    series: pd.Series,
    d_grid: npt.NDArray[np.float64] | None = None,
    adf_pvalue: float = 0.05,
    threshold: float = DEFAULT_THRESHOLD,
) -> FracDiffParams:
    """Smallest grid ``d`` whose FFD series passes an ADF stationarity test.

    Walks ``d_grid`` ascending (default :data:`DEFAULT_D_GRID`) and returns the
    first :class:`FracDiffParams` whose FFD series has an ADF p-value strictly
    below ``adf_pvalue`` — the smallest differencing that buys stationarity,
    keeping the most memory.

    **Compute the search on TRAINING data only.** It inspects the whole series
    to choose ``d``; running it on full history and reusing ``d`` over that same
    history leaks the test window into the transform (see the module docstring).

    A grid ``d`` whose FFD window leaves fewer than :data:`MIN_ADF_SAMPLES`
    points on ``series`` is SKIPPED (untestable on this fold — small ``d`` at a
    tight ``threshold`` needs thousands of terms). ADF via
    ``statsmodels.tsa.stattools.adfuller`` is imported lazily and runs only here
    (offline / training-fold work), never on the hot recompute path.

    Raises ``ValueError`` if the series index is non-ascending or non-finite, if
    no grid ``d`` is even testable (series too short for the smallest feasible
    window), or if no testable ``d`` reaches ``adf_pvalue``.
    """
    from statsmodels.tsa.stattools import adfuller

    grid = DEFAULT_D_GRID if d_grid is None else np.asarray(d_grid, dtype=np.float64)
    values = series.to_numpy(dtype=float)
    if not series.index.is_monotonic_increasing:
        raise ValueError(
            "series index must be sorted ascending (chronological order) for min_d_search"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "series contains NaN/non-finite values: "
            f"{non_finite_summary(series.index, values)}; clean the series first"
        )
    tested_any = False
    for raw_d in grid:
        d = float(raw_d)
        window_len = len(ffd_weights(d, threshold))
        if len(values) - window_len + 1 < MIN_ADF_SAMPLES:
            continue  # window too wide for this fold — untestable, not an error
        tested_any = True
        fracdiffed = fracdiff_series(series, d, threshold)
        pvalue = float(adfuller(fracdiffed.to_numpy(dtype=float))[1])
        if pvalue < adf_pvalue:
            aligned_original = series.loc[fracdiffed.index]
            corr = float(fracdiffed.corr(aligned_original))
            return FracDiffParams(
                d=d,
                threshold=float(threshold),
                window_len=window_len,
                adf_pvalue_achieved=pvalue,
                corr_with_original=corr,
            )
    if not tested_any:
        raise ValueError(
            f"series length {len(values)} too short: no grid d has an FFD window leaving "
            f"MIN_ADF_SAMPLES={MIN_ADF_SAMPLES} points (smallest feasible window is d=1); "
            "supply a longer training series"
        )
    raise ValueError(
        f"no testable d on the grid reached ADF p < {adf_pvalue} — the series may be "
        "non-stationary even fully differenced, or the grid may need a wider range"
    )
