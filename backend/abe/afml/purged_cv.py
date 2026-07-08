"""Purged + embargoed chronological walk-forward splits (Lopez de Prado ch. 7).

A forecast label at bar ``t`` spans the horizon ``(t, t + H]`` (an H-day forward
return). Ordinary K-fold — or even a naive expanding walk-forward that lets the
train set run right up to the test window — leaks: a training label whose
horizon reaches INTO the test window shares information with the test bars. The
fix is a **purge**: drop training bars whose label horizon would overlap the
test window (train must end at least ``horizon`` bars before the test starts),
plus an optional **embargo** of extra bars for good measure.

Public API (plan.md Step 12, module ``afml/purged_cv.py``):

- :func:`purged_walk_forward_splits` — expanding-window walk-forward splits as a
  list of ``(train_idx, test_idx)`` plain integer arrays. Test windows are
  chronological, non-overlapping, and tile the tail of the series.
- :func:`validate_no_leakage` — asserts every split obeys the purge
  (``max(train) + horizon < min(test)``) plus internal ordering; the tests run
  it on produced splits AND assert it FAILS on a deliberately-leaky split.

**Deliberate simplicity vs sklearn.** We return plain ``np.ndarray`` index pairs
rather than implementing sklearn's ``BaseCrossValidator`` (a ``split(X, y,
groups)`` generator, ``get_n_splits``, estimator-clone plumbing). There is one
consumer (the JEPA train/eval folds), no ``Pipeline``/``GridSearchCV``
integration, and the purge/embargo logic is the whole point — a 40-line explicit
function is more auditable here than a sklearn subclass, and adds no dependency.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

from abe.constants import HORIZON_BARS

__all__ = [
    "MIN_TRAIN_BARS",
    "purged_walk_forward_splits",
    "validate_no_leakage",
]

IndexArray = npt.NDArray[np.int_]
Split = tuple[IndexArray, IndexArray]

MIN_TRAIN_BARS: Final[int] = 1
"""Smallest train set any split may have. The first (smallest) split's train
runs ``[0, first_test_start - horizon - embargo)``; if that is shorter than
this, the configuration is rejected up front rather than yielding an empty
train fold."""


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    horizon: int = HORIZON_BARS,
    embargo: int = 0,
) -> list[Split]:
    """Expanding-window walk-forward splits with an H-bar purge (+ embargo).

    The last ``n_splits`` contiguous, equal-size test windows tile the tail of
    ``[0, n_samples)`` (the final window absorbs the remainder). Split ``i``'s
    train set is the expanding prefix ``[0, test_start_i - horizon - embargo)``:
    it ends at least ``horizon`` bars before its test window so no training
    label's ``(t, t + horizon]`` horizon reaches into the test window (the
    purge), widened by ``embargo`` extra bars.

    Returns a list of ``(train_idx, test_idx)`` ``int`` arrays. Test windows are
    chronological and non-overlapping; train sets grow with ``i``.

    Raises ``ValueError`` on non-sane geometry: ``n_splits < 1``,
    ``horizon < 0``, ``embargo < 0``, a test size of 0 (``n_samples`` too small
    to give each split a non-empty test window), or a first-split train shorter
    than :data:`MIN_TRAIN_BARS` (the purge + embargo would empty it).
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    test_size = n_samples // (n_splits + 1)
    if test_size < 1:
        raise ValueError(
            f"n_samples={n_samples} too small for n_splits={n_splits}: "
            f"test_size = n_samples // (n_splits + 1) = {test_size} < 1 "
            "(need at least one bar per test window plus an initial train block)"
        )
    first_test_start = n_samples - n_splits * test_size
    first_train_end = first_test_start - horizon - embargo
    if first_train_end < MIN_TRAIN_BARS:
        raise ValueError(
            f"first split's train would be [0, {first_train_end}) — shorter than "
            f"MIN_TRAIN_BARS={MIN_TRAIN_BARS}. horizon={horizon} + embargo={embargo} "
            f"purge exceeds the initial train block ({first_test_start} bars); "
            "use more samples, fewer splits, or a smaller horizon/embargo"
        )

    splits: list[Split] = []
    for i in range(n_splits):
        test_start = first_test_start + i * test_size
        test_end = n_samples if i == n_splits - 1 else test_start + test_size
        train_end = test_start - horizon - embargo
        train_idx = np.arange(0, train_end, dtype=np.int_)
        test_idx = np.arange(test_start, test_end, dtype=np.int_)
        splits.append((train_idx, test_idx))
    return splits


def validate_no_leakage(splits: list[Split], horizon: int) -> None:
    """Assert every ``(train, test)`` split is purge-clean and well-ordered.

    For each split, requires (raising ``AssertionError`` on the first
    violation):

    - non-empty train and test, each a sorted ascending index array;
    - the purge: ``max(train) + horizon < min(test)`` — no training label's
      ``(t, t + horizon]`` horizon reaches into the test window;
    - chronology across splits: test windows in ascending order and
      non-overlapping.

    ``raise AssertionError`` (not a bare ``assert``) so the guard survives
    ``python -O`` and stays a real leakage check. The tests call this on
    produced splits (must pass) and on a hand-built leaky split (must fail).
    """
    if horizon < 0:
        raise AssertionError(f"horizon must be >= 0, got {horizon}")
    previous_test_max: int | None = None
    for split_position, (train_idx, test_idx) in enumerate(splits):
        if train_idx.size == 0:
            raise AssertionError(f"split {split_position}: train index is empty")
        if test_idx.size == 0:
            raise AssertionError(f"split {split_position}: test index is empty")
        if not _is_sorted_ascending(train_idx):
            raise AssertionError(f"split {split_position}: train index is not sorted ascending")
        if not _is_sorted_ascending(test_idx):
            raise AssertionError(f"split {split_position}: test index is not sorted ascending")
        train_max = int(train_idx.max())
        test_min = int(test_idx.min())
        if not (train_max + horizon < test_min):
            raise AssertionError(
                f"split {split_position}: leakage — max(train)={train_max} + horizon="
                f"{horizon} = {train_max + horizon} is not < min(test)={test_min}; a "
                "training label's horizon overlaps the test window"
            )
        if previous_test_max is not None and test_min <= previous_test_max:
            raise AssertionError(
                f"split {split_position}: test window starts at {test_min}, not after the "
                f"previous test window's last index {previous_test_max} (windows must be "
                "chronological and non-overlapping)"
            )
        previous_test_max = int(test_idx.max())


def _is_sorted_ascending(index_array: IndexArray) -> bool:
    """True when ``index_array`` is strictly ascending (contiguous or not)."""
    return bool(np.all(np.diff(index_array) > 0)) if index_array.size > 1 else True
