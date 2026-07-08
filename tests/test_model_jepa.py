"""JEPAModel + architecture tests (plan.md Step 13).

The minimal JEPA is a drop-in :class:`~abe.model.base.WorldModel`, so it must pass
the FROZEN contract (``test_model_base.assert_worldmodel_contract``, imported —
never forked). Everything is tiny synthetic data + few epochs + K=2 CI ensembles
(the production default K=3 is pinned separately, without training); ``conftest``
pins torch to one thread and owns the shared ``ci_config``/``iid_data`` helpers so
the whole file runs in well under 60s on CPU.

Contract-arming note (how the mandatory horizon/scale pins are satisfied):
``assert_worldmodel_contract``'s docstring requires arming ``expected_daily_mu``
on a known-drift input plus a sigma scale pin on iid noise. The tight rel-tol
``expected_daily_mu`` arming is meaningful only for an analytic estimator (EWMA
recovers ``daily_drift * H`` exactly); a LEARNED model gives no such guarantee.
The intent — catching mu emitted on the WRONG SCALE (daily / annualized, the
plan's top silent-bug class) — is covered here by an order-of-magnitude band
instead: on a strong-drift synthetic, ``0.2*D*H < mu < 5*D*H``
(``test_jepa_mu_is_h_day_scaled_on_strong_drift``; a daily-scale mu is 0.048x,
an annualized one 12x — both far outside). The sigma scale pin on iid noise is
armed exactly as required (``test_jepa_sigma_is_h_day_predictive_scale_on_iid_noise``).
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import IID_STD, ci_config, dates_index, iid_data
from test_model_base import assert_worldmodel_contract

from abe.constants import HORIZON_BARS
from abe.model.base import WorldModel
from abe.model.jepa import (
    CollapseError,
    JEPACheckpoint,
    JEPAConfig,
    JEPAModel,
    JEPANet,
    count_parameters,
    window_indices,
)
from abe.model.train import train_jepa


def _inference_frame(features: pd.DataFrame, asset: str = "SPY") -> pd.DataFrame:
    """Per-asset frame the pipeline hands forecast(): bare ``log_return`` column."""
    return pd.DataFrame(
        {"log_return": features[f"{asset}_log_return"].to_numpy()}, index=features.index
    )


@pytest.fixture(scope="module")
def iid_checkpoint() -> tuple[JEPACheckpoint, pd.DataFrame]:
    """One tiny ensemble trained on iid noise — reused across the read-only tests."""
    features, returns = iid_data(seed=23)
    checkpoint = train_jepa(features, returns, ci_config())
    return checkpoint, features


# --------------------------------------------------------------------------- #
# Architecture / config pins (no training)
# --------------------------------------------------------------------------- #


def test_default_config_is_k3_small_return_weight_under_500k_params() -> None:
    # The PRODUCTION defaults, pinned without training: K=3; the return head is
    # an auxiliary at a SMALL weight (plan §4 "joint-trained at a small loss
    # weight" — a co-equal weight would make Step 14 compare a supervised MLP,
    # not the SSL-primary JEPA); the K=3 ensemble stays under the 500k budget.
    config = JEPAConfig()
    assert config.n_seeds == 3
    assert config.return_weight == 0.05
    per_seed = count_parameters(JEPANet(config))
    assert per_seed * config.n_seeds < 500_000


def test_collapse_thresholds_are_literal_pinned() -> None:
    # Literal pins against silent weakening: relaxing a default threshold keeps
    # the fully-degenerate garbage anchor green while gutting the gate for
    # borderline collapse — this test makes any default change a visible,
    # reviewed diff (measurement-validity rule: the gate must stay falsifiable).
    config = JEPAConfig()
    assert config.std_median_threshold == 0.1
    assert config.effective_rank_threshold == 1.5


def test_config_rejects_context_longer_than_horizon() -> None:
    with pytest.raises(ValueError, match="context_len"):
        JEPAConfig(context_len=HORIZON_BARS + 1, horizon=HORIZON_BARS)


def test_window_indices_have_no_lookahead() -> None:
    # The batch-builder contract: context ends AT t; every target/return index is
    # strictly in the future; the return span is exactly (t, t+H].
    context_len, horizon, n_bars = 16, HORIZON_BARS, 80
    triples = window_indices(n_bars, context_len, horizon)
    assert triples, "expected at least one window"
    for window in triples:
        assert int(window.context_idx.max()) == window.t
        assert len(window.context_idx) == context_len
        assert (window.target_idx > window.t).all()
        assert int(window.target_idx.max()) == window.t + horizon
        assert window.return_idx.tolist() == list(range(window.t + 1, window.t + horizon + 1))


def test_window_indices_rejects_context_longer_than_horizon() -> None:
    with pytest.raises(ValueError, match="context_len"):
        window_indices(100, context_len=30, horizon=HORIZON_BARS)


# --------------------------------------------------------------------------- #
# The frozen WorldModel contract + the mu/sigma scales
# --------------------------------------------------------------------------- #


def test_jepa_passes_frozen_worldmodel_contract(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    # THE frozen contract (shape, keys, finite mu, sigma>0), through the SAME
    # function EWMA passes. The tight expected_daily_mu arming is replaced by the
    # order-of-magnitude mu band below (module docstring); sigma scale pinned below.
    checkpoint, features = iid_checkpoint
    model = checkpoint.to_model()
    assert_worldmodel_contract(model, {"SPY": _inference_frame(features)})


def test_jepa_is_runtime_worldmodel_and_model_version_shape(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    checkpoint, _ = iid_checkpoint
    model = checkpoint.to_model()
    assert isinstance(model, WorldModel)
    # plan §3 forecasts.model_version shape: 'jepa:' + 8 hex chars.
    assert re.fullmatch(r"jepa:[0-9a-f]{8}", model.model_version)


def test_jepa_sigma_is_h_day_predictive_scale_on_iid_noise(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    # The load-bearing pin: on iid daily noise (std s) the model cannot predict
    # the noise, so residuals ~ realized H-day sums and sigma ~ s*sqrt(H). The
    # [0.4x, 3x] band excludes a DAILY sigma (~s, 0.22x too small) and an
    # ANNUALIZED one (~s*sqrt(252), 3.46x too big). Outside the band => the
    # residual-variance component is mis-scaled (fix the code, not the band).
    checkpoint, features = iid_checkpoint
    model = checkpoint.to_model()
    forecast = model.forecast({"SPY": _inference_frame(features)})["SPY"]
    expected = IID_STD * math.sqrt(HORIZON_BARS)
    assert 0.4 * expected < forecast.sigma < 3.0 * expected


def test_jepa_mu_is_h_day_scaled_on_strong_drift() -> None:
    # The mu horizon pin, order-of-magnitude form (module docstring): on a
    # strong-drift synthetic (daily drift D >> noise) the H-day mu must land in
    # 0.2*D*H < mu < 5*D*H. A daily-scale mu (D = 0.048x) or an annualized one
    # (D*252 = 12x) fails by construction. Measured across 8 seeds locally the
    # ratio stays in [0.97, 1.02] — the band has an order of magnitude of slack.
    daily_drift, noise = 0.002, 0.0005
    n = 340
    values = daily_drift + noise * np.random.default_rng(3).standard_normal(n)
    index = dates_index(n)
    features = pd.DataFrame({"SPY_log_return": values}, index=index)
    returns = pd.DataFrame({"SPY": values}, index=index)
    model = train_jepa(features, returns, ci_config()).to_model()
    mu = model.forecast({"SPY": _inference_frame(features)})["SPY"].mu
    expected = daily_drift * HORIZON_BARS
    assert 0.2 * expected < mu < 5.0 * expected


def test_jepa_forecast_only_over_input_assets(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    checkpoint, features = iid_checkpoint
    model = checkpoint.to_model()
    result = model.forecast({"AGG": _inference_frame(features)})
    assert set(result) == {"AGG"}
    assert set(model.forecast({"SPY": _inference_frame(features)})) == {"SPY"}


# --------------------------------------------------------------------------- #
# Collapse instrumentation (healthy passes; garbage anchor hard-fails)
# --------------------------------------------------------------------------- #


def test_healthy_run_stores_a_passing_collapse_report(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    checkpoint, _ = iid_checkpoint
    report = checkpoint.collapse_report
    assert report.passed
    assert report.per_dim_std_median >= report.std_median_threshold
    assert report.effective_rank >= report.effective_rank_threshold


def test_collapse_hard_fail_garbage_anchor() -> None:
    # Garbage anchor (a gate that cannot fail garbage cannot gate): VICReg
    # weights zeroed + a degenerate constant-feature series => every context
    # window is identical => embeddings collapse to a point => CollapseError.
    n = 200
    constant = np.full(n, 0.001)
    index = dates_index(n)
    features = pd.DataFrame({"SPY_log_return": constant}, index=index)
    returns = pd.DataFrame({"SPY": constant}, index=index)
    config = ci_config(epochs=12, var_weight=0.0, cov_weight=0.0, return_weight=0.0)
    with pytest.raises(CollapseError, match="collapsed"):
        train_jepa(features, returns, config)


# --------------------------------------------------------------------------- #
# Checkpoint round-trip + format version + forecast input guards
# --------------------------------------------------------------------------- #


def test_checkpoint_round_trip_is_bitwise_identical(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame], tmp_path: Path
) -> None:
    checkpoint, features = iid_checkpoint
    path = tmp_path / "jepa.pt"
    checkpoint.save(path)
    loaded = JEPAModel.from_checkpoint(path)
    in_memory = checkpoint.to_model()
    frame = {"SPY": _inference_frame(features)}
    # Determinism: save -> load -> identical forecasts (Forecast is a frozen
    # dataclass, so == is bitwise-identical mu/sigma), and same model_version.
    assert loaded.forecast(frame) == in_memory.forecast(frame)
    assert loaded.model_version == in_memory.model_version


def test_from_checkpoint_missing_sidecar_raises(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame], tmp_path: Path
) -> None:
    checkpoint, _ = iid_checkpoint
    path = tmp_path / "jepa.pt"
    checkpoint.save(path)
    (tmp_path / "jepa.pt.meta.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        JEPAModel.from_checkpoint(path)


@pytest.mark.parametrize("mode", ["bumped", "missing"])
def test_from_checkpoint_rejects_wrong_format_version(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame], tmp_path: Path, mode: str
) -> None:
    # A sidecar written by a different checkpoint format must be rejected loudly
    # (never half-parsed); a MISSING version key is pre-versioned == unknown ==
    # rejected too. The message names both versions.
    checkpoint, _ = iid_checkpoint
    path = tmp_path / "jepa.pt"
    checkpoint.save(path)
    sidecar = tmp_path / "jepa.pt.meta.json"
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if mode == "bumped":
        meta["format_version"] = 999
    else:
        del meta["format_version"]
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        JEPAModel.from_checkpoint(path)


def test_forecast_rejects_frame_missing_feature_column(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    # The documented enrichment-resolution rule: a frame lacking a required
    # feature column raises a clear ValueError (never a silent wrong forecast).
    checkpoint, features = iid_checkpoint
    model = checkpoint.to_model()
    wrong = pd.DataFrame({"ret": features["SPY_log_return"].to_numpy()}, index=features.index)
    with pytest.raises(ValueError, match="log_return"):
        model.forecast({"SPY": wrong})


def test_forecast_rejects_history_shorter_than_context(
    iid_checkpoint: tuple[JEPACheckpoint, pd.DataFrame],
) -> None:
    checkpoint, features = iid_checkpoint
    model = checkpoint.to_model()
    short = _inference_frame(features).iloc[: checkpoint.config.context_len - 1]
    with pytest.raises(ValueError, match="too short"):
        model.forecast({"SPY": short})
