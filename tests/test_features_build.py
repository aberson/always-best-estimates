"""features.build tests: deterministic matrix, no macro lookahead, column pins.

Synthetic prices + inline macro frames only — no network, no db. The load-bearing
checks: (1) macro is joined WITHOUT lookahead (a value published after date t
must not appear at t — the poison test), and (2) the matrix is byte-deterministic
(same inputs twice -> identical frame).
"""

import numpy as np
import pandas as pd
import pytest

from abe.afml.fracdiff import FracDiffParams
from abe.features.basic import LOG_RETURN_COLUMN, REALIZED_VOL_COLUMN, log_returns, realized_vol
from abe.features.build import (
    FEATURE_NAME_SEP,
    build_features,
    feature_column,
)

ASSETS = ("SPY", "ACWI", "AGG")


def _dates(n: int, start: str = "2020-01-01") -> pd.Index:
    return pd.Index(pd.bdate_range(start, periods=n).strftime("%Y-%m-%d"), name="date")


def _price_series(n: int, seed: int, drift: float = 0.0004, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = drift + vol * rng.standard_normal(n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    return pd.Series(prices, index=_dates(n))


def _prices(n: int = 60) -> dict[str, pd.Series]:
    return {asset: _price_series(n, seed=i + 1) for i, asset in enumerate(ASSETS)}


def _d1_params() -> dict[str, FracDiffParams]:
    # d=1 -> a 2-term window (tiny warm-up) so frac-diff tests stay short; the
    # frac-diff of a log-price at d=1 is just its first difference.
    return {
        asset: FracDiffParams(
            d=1.0, threshold=1e-5, window_len=2, adf_pvalue_achieved=0.0, corr_with_original=0.1
        )
        for asset in ASSETS
    }


def _dense_macro(trading_dates: pd.Index, series_ids: tuple[str, ...]) -> pd.DataFrame:
    # One obs per business day per series, starting well BEFORE the price range
    # so every trading date has a prior available obs. available_date =
    # obs_date + 1 business day (available strictly after the obs date).
    obs_dates = pd.bdate_range(pd.Timestamp(trading_dates[0]) - pd.Timedelta(days=15), periods=90)
    rows = []
    for series_id in series_ids:
        for i, obs in enumerate(obs_dates):
            rows.append(
                {
                    "series_id": series_id,
                    "obs_date": obs.strftime("%Y-%m-%d"),
                    "value": float(i) + (0.0 if series_id == series_ids[0] else 1000.0),
                    "available_date": (obs + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
                }
            )
    return pd.DataFrame(rows)


def _expected_columns(assets: tuple[str, ...], *, fracdiff: bool) -> list[str]:
    features = [LOG_RETURN_COLUMN, REALIZED_VOL_COLUMN]
    if fracdiff:
        features.append("fracdiff")
    return [feature_column(asset, feature) for asset in assets for feature in features]


# --------------------------------------------------------------------------- #
# column shape / names
# --------------------------------------------------------------------------- #


def test_feature_column_flat_name() -> None:
    assert feature_column("SPY", "log_return") == "SPY_log_return"
    assert FEATURE_NAME_SEP == "_"


def test_columns_without_params_omit_fracdiff() -> None:
    matrix = build_features(_prices(), macro=None, params=None)
    assert list(matrix.columns) == _expected_columns(ASSETS, fracdiff=False)


def test_columns_with_params_include_fracdiff() -> None:
    matrix = build_features(_prices(), macro=None, params=_d1_params())
    assert list(matrix.columns) == _expected_columns(ASSETS, fracdiff=True)


def test_macro_columns_are_series_ids() -> None:
    prices = _prices()
    macro = _dense_macro(_dates(60), ("DGS10", "VIXCLS"))
    matrix = build_features(prices, macro=macro, params=None)
    # Per-asset columns, then macro series-id columns sorted.
    assert list(matrix.columns) == [
        *_expected_columns(ASSETS, fracdiff=False),
        "DGS10",
        "VIXCLS",
    ]


# --------------------------------------------------------------------------- #
# feature values wire through basic.py
# --------------------------------------------------------------------------- #


def test_log_return_and_vol_match_basic() -> None:
    prices = _prices()
    matrix = build_features(prices, macro=None, params=None)
    for asset in ASSETS:
        expected_ret = log_returns(prices[asset]).reindex(matrix.index)
        expected_vol = realized_vol(log_returns(prices[asset])).reindex(matrix.index)
        np.testing.assert_allclose(
            matrix[feature_column(asset, LOG_RETURN_COLUMN)].to_numpy(),
            expected_ret.to_numpy(),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            matrix[feature_column(asset, REALIZED_VOL_COLUMN)].to_numpy(),
            expected_vol.to_numpy(),
            atol=1e-15,
        )
    assert not matrix.isna().any().any()  # dense: no per-asset NaN


# --------------------------------------------------------------------------- #
# no macro lookahead (plan section 9)
# --------------------------------------------------------------------------- #


def test_macro_no_lookahead_poison() -> None:
    # POISON: an observation dated t but only AVAILABLE at t+1 must NOT appear at
    # t. At t the matrix must show the earlier, already-released value.
    prices = _prices()
    base = build_features(prices, macro=None, params=None)
    t = str(base.index[10])
    t_plus_1 = (pd.Timestamp(t) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t_minus_3 = (pd.Timestamp(t) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    macro = pd.DataFrame(
        [
            # Released on/before t -> visible at t.
            {"series_id": "DGS10", "obs_date": t_minus_3, "value": 1.0, "available_date": t},
            # Dated t but only available t+1 -> the poison; must be invisible at t.
            {"series_id": "DGS10", "obs_date": t, "value": 99.0, "available_date": t_plus_1},
        ]
    )
    matrix = build_features(prices, macro=macro, params=None)
    assert matrix.loc[t, "DGS10"] == 1.0  # NOT 99.0 — no lookahead


def test_macro_no_lookahead_property_over_all_dates() -> None:
    # Independent (pure-Python) point-in-time reconstruction vs merge_asof: for
    # every trading date, the matched value is the latest obs with
    # available_date <= t (ties resolved by obs_date).
    prices = _prices()
    macro = _dense_macro(_dates(60), ("DGS10",))
    matrix = build_features(prices, macro=macro, params=None)

    obs = sorted(
        (row["available_date"], row["obs_date"], row["value"]) for _, row in macro.iterrows()
    )
    for t in matrix.index:
        tt = pd.Timestamp(t)
        visible = [value for avail, _, value in obs if pd.Timestamp(avail) <= tt]
        expected = visible[-1] if visible else float("nan")
        got = matrix.loc[t, "DGS10"]
        if visible:
            assert got == expected
        else:
            assert pd.isna(got)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_hash_stable_same_inputs_identical_frame() -> None:
    prices = _prices()
    macro = _dense_macro(_dates(60), ("DGS10", "VIXCLS"))
    params = _d1_params()
    first = build_features(prices, macro=macro, params=params)
    second = build_features(prices, macro=macro, params=params)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    # Wire-shape pin for the Step 13 JEPA consumer: a plain all-float64 matrix
    # (an object-dtype macro column would pass equality checks but break torch).
    assert (first.dtypes == np.float64).all()


def test_deterministic_across_fresh_price_dicts() -> None:
    # Rebuilding the identical inputs from scratch yields the identical matrix.
    first = build_features(_prices(), macro=_dense_macro(_dates(60), ("DGS10",)), params=None)
    second = build_features(_prices(), macro=_dense_macro(_dates(60), ("DGS10",)), params=None)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


# --------------------------------------------------------------------------- #
# degraded / error paths
# --------------------------------------------------------------------------- #


def test_macro_none_omits_macro_columns() -> None:
    matrix = build_features(_prices(), macro=None, params=None)
    assert not any(col in ("DGS10", "VIXCLS") for col in matrix.columns)
    assert not matrix.empty


def test_macro_empty_frame_omits_macro_columns() -> None:
    empty = pd.DataFrame(columns=["series_id", "obs_date", "value", "available_date"])
    matrix = build_features(_prices(), macro=empty, params=None)
    assert list(matrix.columns) == _expected_columns(ASSETS, fracdiff=False)


def test_empty_prices_raises() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        build_features({}, macro=None, params=None)


def test_params_missing_asset_raises() -> None:
    prices = _prices()
    partial = {"SPY": _d1_params()["SPY"]}  # ACWI, AGG missing
    with pytest.raises(ValueError, match="missing frac-diff params"):
        build_features(prices, macro=None, params=partial)


def test_fracdiff_column_is_finite_and_dense() -> None:
    matrix = build_features(_prices(), macro=None, params=_d1_params())
    for asset in ASSETS:
        column = matrix[feature_column(asset, "fracdiff")]
        assert np.isfinite(column.to_numpy()).all()
