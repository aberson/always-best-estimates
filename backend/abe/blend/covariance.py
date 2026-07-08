"""Ledoit-Wolf shrinkage — the ONLY covariance path (plan section 8).

SPY/ACWI correlation is ~0.95 (SPY is ~60% of ACWI), so the sample covariance
of this universe is near-singular and an unshrunk mean-variance optimizer
flips the entire equity sleeve on basis-point mu changes. Ledoit-Wolf shrinks
toward a scaled identity, lifting the smallest eigenvalue and guaranteeing a
well-conditioned PSD Sigma. Marchenko-Pastur / RMT denoising is explicitly
excluded (meaningless at N=3).

Implementation choice — pypfopt over raw sklearn: plan section 2 frames the
dependency as "PyPortfolioOpt ==1.6.0 (Black-Litterman + Ledoit-Wolf shrinkage
only)", so routing Sigma through ``pypfopt.risk_models.CovarianceShrinkage``
keeps the pinned dependency load-bearing and keeps annualization + labeling in
ONE library code path shared with the BL model. Verified against the installed
1.6.0 source (``risk_models.py``): with ``returns_data=True`` the frame is
used directly as returns (``self.X = prices.dropna(how="all")``, NO
price->return conversion), ``ledoit_wolf()`` delegates to
``sklearn.covariance.ledoit_wolf`` (constant-variance target), and
``_format_and_annualize`` multiplies by ``frequency`` and applies a spectral
PSD fix — so the sklearn estimator is under the hood either way, reached
through the production library. Caveat the loud validation below exists for:
1.6.0 silently maps NaN through ``dropna(how="all")`` + ``np.nan_to_num``
(NaN -> 0.0), which would fabricate zero-return days — hence NaN is rejected
HERE, before the library ever sees the frame.
"""

from typing import Final

import numpy as np
import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage

from abe.calc import non_finite_summary
from abe.constants import TRADING_DAYS

__all__ = [
    "MIN_LW_ROWS",
    "PSD_EIGENVALUE_TOL",
    "ledoit_wolf_sigma",
]

MIN_LW_ROWS: Final[int] = 60
"""Minimum daily log-return rows (~one quarter). The Ledoit-Wolf shrinkage
intensity is itself estimated from the sample; on shorter windows both the
sample covariance and the intensity are noise, so short input fails loud
instead of returning a plausible-looking garbage Sigma."""

PSD_EIGENVALUE_TOL: Final[float] = -1e-10
"""Minimum eigenvalue accepted as "PSD up to float noise" after defensive
symmetrization."""


def ledoit_wolf_sigma(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized Ledoit-Wolf covariance of a daily log-returns frame.

    Input: ``returns`` with one column per asset and an ascending ISO-date
    index — the shape ``abe.calc.log_returns`` emits per asset, joined
    column-wise. Output: the shrunk covariance annualized by ``TRADING_DAYS``
    (the ``CovarianceShrinkage(frequency=...)`` path), as a DataFrame labeled
    exactly like ``returns.columns`` (order preserved), defensively
    symmetrized and validated PSD (min eigenvalue > ``PSD_EIGENVALUE_TOL``).

    Raises ``ValueError`` on: an empty/column-less frame, fewer than
    ``MIN_LW_ROWS`` rows, a non-ascending index (contract hygiene — the shared
    input convention is chronological; a disordered frame usually means a bad
    join upstream), NaN/non-finite cells (per-column locations in the
    message — 1.6.0 would silently coerce NaN to 0.0 otherwise), and a
    non-PSD result (cannot happen for constant-variance Ledoit-Wolf, which is
    a convex blend of two PSD matrices, but the guarantee is validated rather
    than assumed).
    """
    if returns.shape[1] == 0:
        raise ValueError("returns frame has no asset columns")
    if len(returns) < MIN_LW_ROWS:
        raise ValueError(
            f"need at least MIN_LW_ROWS={MIN_LW_ROWS} daily return rows for a meaningful "
            f"Ledoit-Wolf estimate, got {len(returns)}"
        )
    if not returns.index.is_monotonic_increasing:
        raise ValueError(
            "returns index must be sorted ascending (chronological order); a disordered "
            "frame signals a broken upstream join"
        )
    for column in returns.columns:
        values = returns[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"returns for {column!r} contain NaN/non-finite values: "
                f"{non_finite_summary(returns.index, values)}; pypfopt would silently "
                "coerce these to 0.0 — clean the frame first"
            )

    shrunk = CovarianceShrinkage(returns, returns_data=True, frequency=TRADING_DAYS).ledoit_wolf()

    # Defensive symmetrization: the estimator is symmetric in exact arithmetic;
    # this removes float-noise asymmetry so eigvalsh (and cholesky downstream)
    # see a genuinely symmetric matrix.
    array = shrunk.to_numpy(dtype=float)
    array = (array + array.T) / 2.0
    min_eigenvalue = float(np.linalg.eigvalsh(array).min())
    if min_eigenvalue <= PSD_EIGENVALUE_TOL:
        raise ValueError(
            f"Ledoit-Wolf covariance is not PSD (min eigenvalue {min_eigenvalue:.3e} <= "
            f"{PSD_EIGENVALUE_TOL:.0e}) — refusing to hand a broken Sigma downstream"
        )
    return pd.DataFrame(array, index=returns.columns, columns=returns.columns)
