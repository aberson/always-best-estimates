"""``optimize_weights`` — the hand-rolled cvxpy mean-variance-utility QP (plan Step 7).

::

    maximize    mu_post @ w  -  0.5 * delta * sum_squares(chol.T @ w)
                             -  gamma_tc * norm1(w - w_prev)
    subject to  sum(w) == 1,   0 <= w <= w_max

with ``chol = cholesky(sigma_post)`` (annualized BL posterior covariance).
Everything is in ANNUALIZED EXCESS RETURN units — the exact scale ``bl_blend``
emits (plan section 8's one-unit-convention rule), so ``delta`` here is the
same He-Litterman risk aversion that built the BL prior.

Design decisions (plan section 8):

- **``sum_squares(chol.T @ w)``, NEVER ``quad_form``.** ``quad_form``'s ARPACK
  PSD check is known to fail on exactly the near-singular matrices this
  universe produces (SPY/ACWI correlation ~0.95). With the Cholesky form,
  PSD-ness is established ONCE at factorization time by numpy, and cvxpy sees
  a plain sum of squares — convex by construction, no spectral check.
- **Solver pinned CLARABEL** (:data:`SOLVER`). Accepted statuses: ``OPTIMAL``
  and ``OPTIMAL_INACCURATE`` (the latter logged as a warning but accepted —
  a slightly loose interior-point exit is still a valid allocation).
- **Turnover is stateful with a cold-start DROP.** ``w_prev`` is the last
  persisted allocation (the ``target_weights`` table). ``w_prev=None`` (first
  ever run — nothing persisted) removes the turnover TERM from the objective
  entirely; it does NOT pretend the previous book was all-zero. (Under
  ``w >= 0`` and ``sum(w) == 1`` a zero-vector ``w_prev`` would make the
  penalty the constant ``gamma_tc``, so dropping is also the exact algebraic
  limit — but dropping keeps the diagnostics honest: cold-start turnover is
  reported as 0.0, not 1.0.)
- **Solver-failure retry.** The box + budget constraints are feasible for any
  sane ``w_max``, so a non-optimal exit with the turnover term present is a
  numerical failure, not a modeling one: retry ONCE without the turnover term
  and flag ``relaxed_turnover=True`` on the result. If the retry also fails,
  ``RuntimeError`` with both solver statuses.
- **Post-processing.** Solver output carries interior-point fuzz (weights like
  ``-3e-13`` or ``1e-11``): every weight below :data:`WEIGHT_CLIP` is clipped
  to exactly 0.0, then the vector is renormalized to sum EXACTLY 1.0 (float
  ``==``) — division by the post-clip total (a relative adjustment of at most
  ~``N * WEIGHT_CLIP`` plus CLARABEL's feasibility tolerance, i.e. parts per
  billion), followed by a verified single-ulp nudge of one weight until
  ``float(sum(w)) == 1.0`` holds exactly (:func:`_clip_and_renormalize`). The
  persisted budget invariant is therefore float-exact; a weight pinned at the
  box may exceed ``w_max`` by the renormalization epsilon. That epsilon scales
  with the accepted solve's feasibility tolerance: ppb-scale (~5e-9 observed)
  on ``OPTIMAL`` exits, but up to ~1e-4 relative on an accepted
  ``OPTIMAL_INACCURATE`` exit (CLARABEL's reduced tolerances) — downstream
  validation gates must key any box-cap check to the persisted status.

Near-singularity handling (THE plan section 8 risk): ``sigma_post`` is PSD by
construction (Ledoit-Wolf + BL posterior, defensively symmetrized upstream)
but can sit numerically ON the PSD boundary, and ``np.linalg.cholesky``
requires strictly PD. :func:`_cholesky_with_jitter` escalates through
:data:`_CHOL_JITTER_SCALES` — adding ``scale * mean(diag(sigma)) * I`` before
factorizing. The largest jitter (``1e-8 * mean(diag)``) perturbs any
portfolio variance by a relative ~1e-8, many orders below the Ledoit-Wolf
shrinkage already applied; a matrix that still fails (a genuinely negative
eigenvalue beyond float noise) is not a covariance and fails LOUD.
"""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd

# Atom-module imports: cvxpy's top-level re-exports (cp.sum_squares, cp.norm1,
# cp.sum) are star-imported and invisible to mypy strict; the defining modules
# export them explicitly.
from cvxpy.atoms.affine.sum import sum as cvx_sum
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.sum_squares import sum_squares

from abe.constants import DELTA, UNIVERSE, W_MAX

__all__ = [
    "GAMMA_TC",
    "SOLVER",
    "WEIGHT_CLIP",
    "MVUResult",
    "optimize_weights",
    "validate_box",
]

logger = logging.getLogger(__name__)

SOLVER: Final[str] = str(cp.CLARABEL)
"""The pinned solver (plan section 8). ``cp.CLARABEL`` is the literal string
``"CLARABEL"``; the pin is asserted against ``cp.installed_solvers()`` by
test, so an environment where the wheel lost its CLARABEL extra fails loud."""

GAMMA_TC: Final[float] = 0.002
"""Default L1 turnover penalty, in annualized-excess-return units per unit of
one-way turnover (20 bp per 100% of the book traded).

An L1 penalty creates a per-asset NO-TRADE BAND: at the optimum an asset only
moves off ``w_prev`` when its marginal expected-utility gain per unit weight
(same units as ``mu_post``) exceeds ``gamma_tc``. 0.002 is chosen so that:

- the basis-point-scale ``mu_post`` drift the 5-minute recompute loop emits
  (features barely change between bars) falls INSIDE the band — a +/-10 bp
  perturbation moves the utility gradient by ~0.001 < 0.002, so pure noise
  does not trade (the plan section 9 churn mitigation);
- a genuine BL view shift still trades: with EWMA-scale Idzorek confidences
  (c ~ 0.1-0.2) on percent-scale views, posterior tilts run tens of bp to
  percents — comfortably outside the band;
- it dominates the honest round-trip cost of these three highly liquid ETFs
  (~1-2 bp spread), so the penalty is a churn damper first and a cost model
  only second — deliberately conservative for an advisory-display engine.
"""

WEIGHT_CLIP: Final[float] = 1e-8
"""Post-solve clip threshold (plan section 8): any weight below this is
interior-point fuzz, set to exactly 0.0 before renormalization so persisted
weights are clean natives (no ``-3e-13`` AGG rows in ``target_weights``)."""

_CHOL_JITTER_SCALES: Final[tuple[float, ...]] = (0.0, 1e-10, 1e-8)
"""Escalating diagonal jitter for :func:`_cholesky_with_jitter`, as multiples
of ``mean(diag(sigma_post))``: try the raw matrix first (the common
well-conditioned case), then two jitter levels for PSD-boundary matrices.
1e-8 is the loud-failure line — see the module docstring."""

_W_PREV_SUM_TOL: Final[float] = 1e-6
"""Absolute tolerance for ``sum(w_prev) == 1``. ``w_prev`` is a PERSISTED
allocation (this module renormalizes exactly before persisting), so anything
beyond float-and-storage noise means the caller read the wrong rows."""

_ACCEPTED_STATUSES: Final[frozenset[str]] = frozenset({str(cp.OPTIMAL), str(cp.OPTIMAL_INACCURATE)})
"""Solver exits accepted as an allocation (plan Step 7). Everything else
(INFEASIBLE / UNBOUNDED / solver error) goes through the retry-then-raise
path."""

_MAX_RENORM_ULPS: Final[int] = 8
"""Search radius (in ulps, per weight) for the exact-sum adjustment in
:func:`_clip_and_renormalize`. The post-division sum sits within 1-2 ulps of
1.0, so a hit is found at offset 1-2 in practice; 8 is loud-failure headroom."""

_SolveOnceFn = Callable[
    [
        npt.NDArray[np.float64],  # mu, UNIVERSE order
        npt.NDArray[np.float64],  # chol (lower-triangular factor of sigma_post)
        npt.NDArray[np.float64] | None,  # w_prev vector, None = drop turnover
        float,  # delta
        float,  # gamma_tc
        float,  # w_max
        float,  # min_weight (box floor; 0.0 = long-only w >= 0)
    ],
    tuple[str, npt.NDArray[np.float64] | None],
]


@dataclass(frozen=True, slots=True)
class MVUResult:
    """The optimize stage's output — one ``target_weights`` row per asset (plan section 3).

    - ``weights``: final post-processed target weights, native floats, keyed
      in canonical ``UNIVERSE`` order; sums to EXACTLY 1.0 (float ``==``).
    - ``prev_weights``: the validated previous allocation expanded to the full
      universe (assets absent from the caller's mapping filled with 0.0), or
      ``None`` on cold start — the ``prev_weight`` column.
    - ``turnover``: per-asset one-way turnover ``|w - w_prev|`` against the
      FINAL post-processed weights; all 0.0 on cold start. Computed from real
      ``prev_weights`` even when ``relaxed_turnover`` is True (the penalty was
      dropped from the objective; the trade distance is still real).
    - ``relaxed_turnover``: True when the first solve failed and the accepted
      solution came from the retry WITHOUT the turnover term.
    - ``status``: the cvxpy status string of the ACCEPTED solve (``"optimal"``
      or ``"optimal_inaccurate"``).

    ``frozen=True`` prevents attribute rebinding only; treat the dict members
    as read-only (each ``optimize_weights`` call returns fresh objects).
    """

    weights: dict[str, float]
    prev_weights: dict[str, float] | None
    turnover: dict[str, float]
    relaxed_turnover: bool
    status: str


def _validate_mu(mu_post: pd.Series) -> npt.NDArray[np.float64]:
    canonical = list(UNIVERSE)
    if list(mu_post.index) != canonical:
        raise ValueError(
            f"mu_post must be indexed exactly UNIVERSE order {canonical!r}, "
            f"got {list(mu_post.index)!r}"
        )
    values: npt.NDArray[np.float64] = mu_post.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [str(mu_post.index[int(i)]) for i in np.flatnonzero(~np.isfinite(values))]
        raise ValueError(f"mu_post contains NaN/non-finite value(s) at {', '.join(bad)}")
    return values


def _validate_sigma(sigma_post: pd.DataFrame) -> npt.NDArray[np.float64]:
    canonical = list(UNIVERSE)
    if list(sigma_post.index) != canonical or list(sigma_post.columns) != canonical:
        raise ValueError(
            "sigma_post must be labeled exactly UNIVERSE order "
            f"{canonical!r} on both axes; got index={list(sigma_post.index)!r}, "
            f"columns={list(sigma_post.columns)!r}"
        )
    array: npt.NDArray[np.float64] = sigma_post.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        bad_rows, bad_cols = np.nonzero(~np.isfinite(array))
        cells = [
            f"({sigma_post.index[r]}, {sigma_post.columns[c]})"
            for r, c in zip(bad_rows[:5], bad_cols[:5], strict=True)
        ]
        more = "" if len(bad_rows) <= 5 else f" (+{len(bad_rows) - 5} more)"
        raise ValueError(
            f"sigma_post contains {len(bad_rows)} NaN/non-finite cell(s) at "
            f"{', '.join(cells)}{more}"
        )
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-8):
        raise ValueError(
            "sigma_post is not symmetric — bl_blend symmetrizes its posterior, so a "
            "foreign matrix reached the optimizer"
        )
    return array


def _validate_w_prev(w_prev: Mapping[str, float]) -> npt.NDArray[np.float64]:
    """Validate a persisted allocation; expand to the full universe (0.0 fill).

    Keys must be a SUBSET of ``UNIVERSE`` (a persisted book may omit assets the
    clip zeroed out); each weight must be finite and in ``[0, 1]`` (long-only
    budget — NOT re-checked against ``w_max``, which may legitimately have
    changed since the allocation was persisted); the total must be 1 within
    :data:`_W_PREV_SUM_TOL`.
    """
    unknown = sorted(set(w_prev) - set(UNIVERSE))
    if unknown:
        raise ValueError(
            f"w_prev contains assets outside UNIVERSE {sorted(UNIVERSE)!r}: {unknown!r}"
        )
    for asset in UNIVERSE:
        if asset not in w_prev:
            continue
        value = float(w_prev[asset])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"w_prev weight for {asset!r} must be finite and in [0, 1] "
                f"(a persisted long-only allocation), got {w_prev[asset]!r}"
            )
    vector = np.array([float(w_prev.get(asset, 0.0)) for asset in UNIVERSE], dtype=float)
    total = float(vector.sum())
    if abs(total - 1.0) > _W_PREV_SUM_TOL:
        raise ValueError(
            f"w_prev must sum to 1.0 (a persisted allocation; renormalization happens "
            f"at persist time), got {total!r}"
        )
    return vector


def validate_box(w_max: float, min_weight: float) -> None:
    """Validate the box constraint bounds (shared by MVU + min-variance).

    ``w_max`` finite, in ``(0, 1]``, and large enough that ``w_max * N >= 1``
    (else the budget is infeasible). ``min_weight`` finite, in ``[0, w_max]``,
    and small enough that ``min_weight * N <= 1`` (else the floors overflow the
    budget). Config bugs surface HERE, not as a solver status to retry."""
    n = len(UNIVERSE)
    if not math.isfinite(w_max) or not 0.0 < w_max <= 1.0:
        raise ValueError(f"w_max must be finite and in (0, 1], got {w_max!r}")
    if w_max * n < 1.0:
        raise ValueError(
            f"w_max={w_max!r} makes the budget constraint infeasible: "
            f"{n} assets capped at {w_max!r} cannot sum to 1"
        )
    if not math.isfinite(min_weight) or min_weight < 0.0:
        raise ValueError(f"min_weight must be finite and >= 0, got {min_weight!r}")
    if min_weight > w_max:
        raise ValueError(f"min_weight={min_weight!r} exceeds w_max={w_max!r}")
    if min_weight * n > 1.0:
        raise ValueError(
            f"min_weight={min_weight!r} makes the budget infeasible: "
            f"{n} assets floored at {min_weight!r} exceed sum 1"
        )


def _cholesky_with_jitter(sigma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lower-triangular Cholesky factor of a PSD-but-possibly-boundary matrix.

    Escalates through :data:`_CHOL_JITTER_SCALES` (relative diagonal jitter);
    fails LOUD with the offending eigenvalue if the matrix is genuinely
    indefinite. Rationale in the module docstring.
    """
    mean_diag = float(np.mean(np.diag(sigma)))
    identity = np.eye(sigma.shape[0])
    for scale in _CHOL_JITTER_SCALES:
        try:
            factor: npt.NDArray[np.float64] = np.linalg.cholesky(
                sigma + scale * mean_diag * identity
            )
        except np.linalg.LinAlgError:
            continue
        if scale > 0.0:
            logger.warning(
                "sigma_post sits on the PSD boundary; Cholesky succeeded with "
                "diagonal jitter %.0e * mean(diag) (relative variance perturbation ~%.0e)",
                scale,
                scale,
            )
        return factor
    min_eigenvalue = float(np.linalg.eigvalsh((sigma + sigma.T) / 2.0).min())
    raise ValueError(
        f"sigma_post is not decomposable even with {_CHOL_JITTER_SCALES[-1]:.0e} * mean(diag) "
        f"jitter (min eigenvalue {min_eigenvalue:.3e}) — a genuinely indefinite matrix is "
        "not a covariance; refusing to optimize against it"
    )


def _solve_once(
    mu: npt.NDArray[np.float64],
    chol: npt.NDArray[np.float64],
    w_prev: npt.NDArray[np.float64] | None,
    delta: float,
    gamma_tc: float,
    w_max: float,
    min_weight: float,
) -> tuple[str, npt.NDArray[np.float64] | None]:
    """One CLARABEL solve. Returns ``(status, weights)``; ``weights=None`` = not accepted.

    ``w_prev=None`` means the turnover term is DROPPED from the objective
    (cold start, or the relaxed retry). ``min_weight`` is the box floor
    (``w >= min_weight``; 0.0 reproduces the V1 long-only ``w >= 0`` exactly). A
    raising solver is folded into the status string (``"solver_error: ..."``) so
    the retry policy in :func:`optimize_weights` sees every failure mode through
    one shape — this function is also the injection seam that policy is
    unit-tested through.
    """
    w = cp.Variable(len(UNIVERSE))
    # sum_squares(chol.T @ w) == w' Sigma w, convex WITHOUT quad_form's
    # spectral PSD check (plan section 8 — the check fails on exactly the
    # near-singular matrices this universe produces).
    utility = mu @ w - 0.5 * delta * sum_squares(chol.T @ w)
    if w_prev is not None:
        utility = utility - gamma_tc * norm1(w - w_prev)
    problem = cp.Problem(
        cp.Maximize(utility),
        [cvx_sum(w) == 1.0, w >= min_weight, w <= w_max],
    )
    try:
        problem.solve(solver=SOLVER)  # type: ignore[no-untyped-call]  # cvxpy ships solve untyped
    except cp.SolverError as exc:
        return (f"solver_error: {exc}", None)
    status = str(problem.status)
    if status not in _ACCEPTED_STATUSES or w.value is None:
        return (status, None)
    return (status, np.asarray(w.value, dtype=float))


def _budget_sum(weights: npt.NDArray[np.float64]) -> float:
    """The budget invariant's DEFINING summation: Python's built-in ``sum``.

    "Sums to exactly 1.0" is only meaningful relative to a summation
    algorithm, so the invariant is verified with the exact expression
    consumers and tests write — ``sum(result.weights.values())``. On CPython
    3.12+ the built-in uses Neumaier-COMPENSATED summation for floats, which
    can differ by one ulp from both a naive left-to-right fold and numpy's
    pairwise ``np.sum`` on the same vector (all three disagreed on real
    solver output during development). Using anything but the consumer's own
    algorithm here would make the invariant true in the producer and false in
    the consumer.
    """
    return float(sum(float(value) for value in weights))


def _clip_and_renormalize(raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Clip interior-point fuzz to 0.0 and renormalize to sum EXACTLY 1.0.

    Two documented stages (module docstring): (1) proportional division by the
    post-clip total — a relative adjustment bounded by CLARABEL's feasibility
    tolerance plus ``N * WEIGHT_CLIP``; (2) an exact-sum fix-up: the division
    leaves the budget sum within 1-2 ulps of 1.0, and stage 2 searches
    single-ulp nudges of one weight at a time (largest first, up to
    :data:`_MAX_RENORM_ULPS` ulps each way) for a vector whose
    :func:`_budget_sum` (the consumer's own fold order) ``== 1.0`` EXACTLY,
    verifying each candidate by direct evaluation. Explicit
    search-plus-verification is used instead of a
    closed-form residual push because intermediate rounding (ties-to-even at
    the running-sum boundaries) makes naive pushes oscillate across 1.0
    without ever landing on it — observed in development. A nudge is at most
    ``_MAX_RENORM_ULPS`` ulps of one weight (~1e-15 absolute), far below any
    financial meaning; candidates that would push a weight negative are
    rejected, so long-only survives stage 2.
    """
    clipped = np.where(raw < WEIGHT_CLIP, 0.0, raw)
    total = _budget_sum(clipped)
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            f"solver returned a degenerate weight vector (post-clip total {total!r}); "
            "cannot renormalize"
        )
    normalized: npt.NDArray[np.float64] = clipped / total
    if _budget_sum(normalized) == 1.0:
        return normalized
    # Largest-first: an ulp of the largest weight is the coarsest lever, and
    # nudging it is the smallest RELATIVE perturbation.
    for index in np.argsort(normalized)[::-1]:
        for direction in (math.inf, -math.inf):
            candidate = normalized.copy()
            for _ in range(_MAX_RENORM_ULPS):
                candidate[index] = np.nextafter(candidate[index], direction)
                if candidate[index] < 0.0:
                    break
                if _budget_sum(candidate) == 1.0:
                    return candidate
    raise RuntimeError(
        f"weight renormalization found no exact-sum vector within {_MAX_RENORM_ULPS} "
        f"ulps of any weight (sum {_budget_sum(normalized)!r}) — pathological float "
        "pattern"
    )


def optimize_weights(
    mu_post: pd.Series,
    sigma_post: pd.DataFrame,
    w_prev: Mapping[str, float] | None = None,
    delta: float = DELTA,
    gamma_tc: float = GAMMA_TC,
    w_max: float = W_MAX,
    min_weight: float = 0.0,
    *,
    _solve_once_fn: _SolveOnceFn = _solve_once,
) -> MVUResult:
    """Solve the mean-variance-utility QP for target weights (plan Step 7).

    Inputs are the BL posterior exactly as ``bl_blend`` emits it: ``mu_post``
    a ``UNIVERSE``-ordered Series of annualized excess returns, ``sigma_post``
    the annualized posterior covariance labeled ``UNIVERSE`` on both axes.
    ``w_prev`` is the last persisted allocation (``None`` on the first ever
    run — the cold start, which DROPS the turnover term). ``delta`` defaults
    to the shared He-Litterman constant; ``gamma_tc`` to :data:`GAMMA_TC`
    (rationale on the constant); ``w_max`` to the shared box cap.

    Solve policy, post-processing, and near-singularity handling are the
    module docstring's contract. ``_solve_once_fn`` is a TEST-ONLY injection
    seam for the retry policy (a fake that fails the first call) — production
    callers never pass it.

    Validation (all fail loud): labels/finiteness/symmetry as above (NaN
    rejection names the offending cells, the blend idiom); ``w_prev`` keys a
    subset of ``UNIVERSE`` with finite weights in ``[0, 1]`` summing to 1;
    ``delta`` finite and > 0; ``gamma_tc`` finite and >= 0 (0 disables the
    penalty explicitly); ``w_max`` finite, in ``(0, 1]``, and large enough
    that ``w_max * len(UNIVERSE) >= 1`` (otherwise the budget constraint is
    structurally infeasible — that is a config bug to surface at validation
    time, not a solver status to retry).

    Raises ``RuntimeError`` when the solve AND the relaxed retry both fail,
    with both solver statuses in the message.
    """
    mu = _validate_mu(mu_post)
    sigma = _validate_sigma(sigma_post)
    w_prev_vector = None if w_prev is None else _validate_w_prev(w_prev)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be finite and > 0, got {delta!r}")
    if not math.isfinite(gamma_tc) or gamma_tc < 0.0:
        raise ValueError(f"gamma_tc must be finite and >= 0, got {gamma_tc!r}")
    validate_box(w_max, min_weight)

    chol = _cholesky_with_jitter(sigma)

    status, raw = _solve_once_fn(mu, chol, w_prev_vector, delta, gamma_tc, w_max, min_weight)
    relaxed_turnover = False
    if raw is None and w_prev_vector is not None:
        # Retry ONCE without the turnover term (plan section 8's
        # INFEASIBLE-retry guard; the box+budget set is feasible, so the
        # failure is numerical — the turnover kink is the moving part).
        logger.warning(
            "MVU solve failed with turnover term (status %r); retrying without it", status
        )
        retry_status, raw = _solve_once_fn(mu, chol, None, delta, gamma_tc, w_max, min_weight)
        if raw is None:
            raise RuntimeError(
                f"MVU optimization failed: status {status!r} with turnover term, "
                f"{retry_status!r} on the relaxed retry"
            )
        status = retry_status
        relaxed_turnover = True
    elif raw is None:
        raise RuntimeError(f"MVU optimization failed on cold start: status {status!r}")
    if status == str(cp.OPTIMAL_INACCURATE):
        logger.warning("CLARABEL returned OPTIMAL_INACCURATE; accepting the allocation")

    final = _clip_and_renormalize(raw)

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
        relaxed_turnover=relaxed_turnover,
        status=status,
    )
