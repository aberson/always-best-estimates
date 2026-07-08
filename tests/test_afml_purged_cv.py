"""afml.purged_cv tests: hand-verified split boundaries + leakage assertions.

Tiny integer cases are hand-verified to exact index boundaries; the built-in
``validate_no_leakage`` is exercised BOTH as a pass on produced splits AND as a
failure on a deliberately-leaky hand-built split (the garbage anchor).
"""

import numpy as np
import pytest

from abe.afml.purged_cv import (
    MIN_TRAIN_BARS,
    purged_walk_forward_splits,
    validate_no_leakage,
)
from abe.constants import HORIZON_BARS

# --------------------------------------------------------------------------- #
# hand-verified boundaries
# --------------------------------------------------------------------------- #


def test_splits_exact_boundaries_n100_3splits_h5() -> None:
    # n=100, n_splits=3, horizon=5, embargo=0 -> test_size = 100//4 = 25,
    # first_test_start = 100 - 3*25 = 25. Hand-verified:
    #   split 0: train [0,20)  test [25,50)
    #   split 1: train [0,45)  test [50,75)
    #   split 2: train [0,70)  test [75,100)
    splits = purged_walk_forward_splits(n_samples=100, n_splits=3, horizon=5)
    assert len(splits) == 3

    expected = [
        (np.arange(0, 20), np.arange(25, 50)),
        (np.arange(0, 45), np.arange(50, 75)),
        (np.arange(0, 70), np.arange(75, 100)),
    ]
    for (train, test), (exp_train, exp_test) in zip(splits, expected, strict=True):
        np.testing.assert_array_equal(train, exp_train)
        np.testing.assert_array_equal(test, exp_test)


def test_test_windows_tile_the_tail_disjoint_and_ordered() -> None:
    splits = purged_walk_forward_splits(n_samples=100, n_splits=3, horizon=5)
    tests = [test for _, test in splits]
    # Chronological + non-overlapping (pairwise: intentionally one shorter).
    for earlier, later in zip(tests, tests[1:], strict=False):
        assert int(earlier.max()) < int(later.min())
    # Together they cover exactly the tail [25, 100).
    covered = np.concatenate(tests)
    np.testing.assert_array_equal(covered, np.arange(25, 100))


def test_train_is_expanding() -> None:
    splits = purged_walk_forward_splits(n_samples=200, n_splits=4, horizon=5)
    train_sizes = [train.size for train, _ in splits]
    assert train_sizes == sorted(train_sizes)  # non-decreasing
    assert train_sizes[0] < train_sizes[-1]  # strictly grows overall


def test_default_horizon_is_horizon_bars() -> None:
    # Default purge is HORIZON_BARS (the label horizon), the whole point of the
    # purge — verify by comparing to an explicit-horizon call.
    default_splits = purged_walk_forward_splits(n_samples=400, n_splits=3)
    explicit = purged_walk_forward_splits(n_samples=400, n_splits=3, horizon=HORIZON_BARS)
    for (dt, dte), (et, ete) in zip(default_splits, explicit, strict=True):
        np.testing.assert_array_equal(dt, et)
        np.testing.assert_array_equal(dte, ete)


# --------------------------------------------------------------------------- #
# purge + embargo geometry
# --------------------------------------------------------------------------- #


def test_purge_leaves_horizon_gap() -> None:
    # With embargo=0 the gap is exactly the horizon: max(train) + horizon is the
    # last index strictly before the test window.
    horizon = 5
    splits = purged_walk_forward_splits(n_samples=100, n_splits=3, horizon=horizon)
    for train, test in splits:
        assert int(train.max()) + horizon == int(test.min()) - 1


def test_embargo_widens_the_gap() -> None:
    horizon = 5
    no_embargo = purged_walk_forward_splits(n_samples=100, n_splits=3, horizon=horizon, embargo=0)
    with_embargo = purged_walk_forward_splits(n_samples=100, n_splits=3, horizon=horizon, embargo=3)
    for (train0, test0), (train3, test3) in zip(no_embargo, with_embargo, strict=True):
        # Same test windows; embargo pulls the train end back by 3 bars.
        np.testing.assert_array_equal(test0, test3)
        assert int(train0.max()) - int(train3.max()) == 3
        # Gap grows: train_max + horizon is further below the test start.
        gap0 = int(test0.min()) - (int(train0.max()) + horizon)
        gap3 = int(test3.min()) - (int(train3.max()) + horizon)
        assert gap3 == gap0 + 3


# --------------------------------------------------------------------------- #
# validate_no_leakage
# --------------------------------------------------------------------------- #


def test_validate_no_leakage_passes_on_produced_splits() -> None:
    for n, k, h, e in [(100, 3, 5, 0), (100, 3, 5, 3), (500, 5, 21, 0), (1000, 4, 21, 10)]:
        splits = purged_walk_forward_splits(n_samples=n, n_splits=k, horizon=h, embargo=e)
        validate_no_leakage(splits, horizon=h)  # must not raise


def test_validate_no_leakage_fails_on_leaky_split() -> None:
    # Garbage anchor: train runs to 29, its horizon reaches to 34, the test
    # window starts at 25 -> the label interval overlaps the test window.
    leaky = [(np.arange(0, 30), np.arange(25, 40))]
    with pytest.raises(AssertionError, match="leakage"):
        validate_no_leakage(leaky, horizon=5)


def test_validate_no_leakage_fails_on_wrong_horizon() -> None:
    # Splits built with horizon=5 are clean at horizon=5 but leak if the label
    # horizon is actually 10 (horizon-consistency check).
    splits = purged_walk_forward_splits(n_samples=100, n_splits=3, horizon=5)
    validate_no_leakage(splits, horizon=5)
    with pytest.raises(AssertionError, match="leakage"):
        validate_no_leakage(splits, horizon=10)


def test_validate_no_leakage_fails_on_unordered_test_windows() -> None:
    # Second test window starts inside/at the first -> not chronological.
    bad = [
        (np.arange(0, 10), np.arange(30, 50)),
        (np.arange(0, 20), np.arange(40, 60)),
    ]
    with pytest.raises(AssertionError, match="chronological"):
        validate_no_leakage(bad, horizon=5)


def test_validate_no_leakage_fails_on_empty_train() -> None:
    empty_train = [(np.array([], dtype=np.int_), np.arange(25, 50))]
    with pytest.raises(AssertionError, match="train index is empty"):
        validate_no_leakage(empty_train, horizon=5)


# --------------------------------------------------------------------------- #
# validation of bad geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"n_samples": 100, "n_splits": 0}, "n_splits must be >= 1", id="zero-splits"),
        pytest.param(
            {"n_samples": 100, "n_splits": 3, "horizon": -1}, "horizon must be >= 0", id="neg-h"
        ),
        pytest.param(
            {"n_samples": 100, "n_splits": 3, "embargo": -1}, "embargo must be >= 0", id="neg-e"
        ),
        pytest.param({"n_samples": 3, "n_splits": 5}, "too small", id="test-size-zero"),
        pytest.param(
            {"n_samples": 30, "n_splits": 3, "horizon": 21},
            "shorter than",
            id="purge-empties-train",
        ),
    ],
)
def test_rejects_bad_geometry(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        purged_walk_forward_splits(**kwargs)


def test_min_train_bars_pinned() -> None:
    assert MIN_TRAIN_BARS == 1
