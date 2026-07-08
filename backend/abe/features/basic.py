"""Minimal features for the EWMA path: daily log-returns + realized volatility.

plan.md section 5 (``features/basic.py``) is the spec. Both functions are pure
and deterministic: same input series -> identical output series; no state, no
I/O, no randomness.

Input shape: a ``pd.Series`` of adjusted closes indexed by ISO-8601
``YYYY-MM-DD`` date strings ascending — exactly the ``adj_close`` column of the
frame ``CacheAdapter.fetch`` returns (plan section 3 date-key rule). Both
functions are order-dependent, so a non-ascending index raises ``ValueError``
instead of being silently wrong.

NaN policy: outputs NEVER contain NaN. Undefined rows (the first return, the
rolling-window warm-up) are DROPPED, not emitted as NaN — storage and every
downstream consumer reject NaN by contract (``storage.coerce_scalar``). Inputs
containing NaN/non-finite values raise ``ValueError`` loudly instead of
silently propagating garbage.

The column-name constants (:data:`LOG_RETURN_COLUMN`,
:data:`REALIZED_VOL_COLUMN`) live HERE and are imported by every consumer
(``model/base.py`` now, Step 12's ``build.py`` later) — one source of truth for
the feature-name shape (code-quality rule). :func:`log_returns` names its
output series :data:`LOG_RETURN_COLUMN` so ``.to_frame()`` yields exactly the
column the ``WorldModel`` consumers expect.
"""

import math
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from abe.constants import HORIZON_BARS, TRADING_DAYS

__all__ = [
    "LOG_RETURN_COLUMN",
    "REALIZED_VOL_COLUMN",
    "log_returns",
    "non_finite_summary",
    "realized_vol",
]

LOG_RETURN_COLUMN: Final[str] = "log_return"
"""Feature-frame column name for daily log-returns (consumers import this)."""

REALIZED_VOL_COLUMN: Final[str] = "realized_vol"
"""Feature-frame column name for annualized realized vol (consumers import this)."""


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


def log_returns(adj_close: pd.Series) -> pd.Series:
    """Natural-log daily returns of an adjusted-close series.

    ``r_t = ln(P_t / P_{t-1})``. The first observation has no predecessor and
    is DROPPED (never NaN), so the result has ``len(adj_close) - 1`` rows,
    indexed by ``adj_close.index[1:]`` and named :data:`LOG_RETURN_COLUMN`.

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

    Sample std (``ddof=1``) over a trailing ``window`` of daily log-returns
    (default ``HORIZON_BARS`` = 21), annualized by ``sqrt(TRADING_DAYS)``.
    Warm-up rows (the first ``window - 1``) are DROPPED, not NaN, so the
    result has ``len(returns) - window + 1`` rows, indexed by
    ``returns.index[window - 1:]`` and named :data:`REALIZED_VOL_COLUMN`.

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
