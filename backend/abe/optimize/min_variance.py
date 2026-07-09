"""Minimum-variance optimizer (Track 2 Step 23) — the AGG=0% realism alternative.

Minimizes ``w' Sigma w`` subject to the SAME constraints as the MVU QP
(long-only, ``sum(w) = 1``, box ``[min_weight, w_max]``) but IGNORES ``mu``
entirely. With no return term it loads the lowest-variance asset — bonds/AGG in
this universe — so it naturally holds the position the V1 mean-variance QP drove
to the 0% corner (plan §6 "AGG=0% addressed via the optimizer seam").

One source of truth: reuses ``mvu.py``'s numerical hardening verbatim — the
``sum_squares(chol.T @ w)`` convex form (never ``quad_form``), the
jitter-escalating Cholesky, the exact-sum clip+renormalize, and the sigma /
w_prev / box validators. The result is an :class:`~abe.optimize.mvu.MVUResult`
(``relaxed_turnover`` is always False — there is no turnover term to relax; the
per-asset ``turnover`` vs ``w_prev`` is still reported for the card).
"""

import logging
from collections.abc import Mapping

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd
from cvxpy.atoms.affine.sum import sum as cvx_sum
from cvxpy.atoms.sum_squares import sum_squares

from abe.constants import UNIVERSE, W_MAX
from abe.optimize.mvu import (
    _ACCEPTED_STATUSES,
    SOLVER,
    MVUResult,
    _cholesky_with_jitter,
    _clip_and_renormalize,
    _validate_sigma,
    _validate_w_prev,
    validate_box,
)

__all__ = ["min_variance_weights"]

logger = logging.getLogger(__name__)


def min_variance_weights(
    sigma_post: pd.DataFrame,
    w_prev: Mapping[str, float] | None = None,
    w_max: float = W_MAX,
    min_weight: float = 0.0,
) -> MVUResult:
    """Solve the global-minimum-variance portfolio (Step 23).

    ``minimize w' Sigma w  s.t.  sum(w) == 1, min_weight <= w <= w_max``, with
    ``sigma_post`` the annualized posterior covariance labeled ``UNIVERSE`` on
    both axes (exactly what ``bl_blend`` emits — ``mu`` is accepted by the
    optimizer seam but unused here). Validation, Cholesky, and post-processing
    match :func:`~abe.optimize.mvu.optimize_weights`. Raises ``RuntimeError`` if
    the (feasible) box+budget solve is not accepted."""
    sigma = _validate_sigma(sigma_post)
    w_prev_vector = None if w_prev is None else _validate_w_prev(w_prev)
    validate_box(w_max, min_weight)

    chol = _cholesky_with_jitter(sigma)
    w = cp.Variable(len(UNIVERSE))
    problem = cp.Problem(
        cp.Minimize(sum_squares(chol.T @ w)),
        [cvx_sum(w) == 1.0, w >= min_weight, w <= w_max],
    )
    try:
        problem.solve(solver=SOLVER)  # type: ignore[no-untyped-call]  # cvxpy ships solve untyped
    except cp.SolverError as exc:  # pragma: no cover — box+budget is always feasible
        raise RuntimeError(f"min-variance solve raised: {exc}") from exc
    status = str(problem.status)
    if status not in _ACCEPTED_STATUSES or w.value is None:
        raise RuntimeError(
            f"min-variance optimization failed: status {status!r} on a feasible box+budget set"
        )
    if status == str(cp.OPTIMAL_INACCURATE):
        logger.warning("CLARABEL returned OPTIMAL_INACCURATE for min-variance; accepting")

    final: npt.NDArray[np.float64] = _clip_and_renormalize(np.asarray(w.value, dtype=float))
    weights = {asset: float(final[i]) for i, asset in enumerate(UNIVERSE)}
    if w_prev_vector is None:
        prev_weights = None
        turnover = {asset: 0.0 for asset in UNIVERSE}
    else:
        prev_weights = {asset: float(w_prev_vector[i]) for i, asset in enumerate(UNIVERSE)}
        turnover = {
            asset: float(abs(final[i] - w_prev_vector[i])) for i, asset in enumerate(UNIVERSE)
        }
    return MVUResult(
        weights=weights,
        prev_weights=prev_weights,
        turnover=turnover,
        relaxed_turnover=False,
        status=status,
    )
