"""Tests for the pre-registered JEPA-vs-EWMA walk-forward eval gate (plan.md Step 14).

Layers, fast to slow:

- **Metrics** (MSE / IC / coverage) on hand-computed values — literal asserts.
- **Promotion rule** table-driven across every branch (promote / bad-calibration /
  worse-MSE / tie / coverage-no-worse-fails / equal-distance-inclusive), EWMA winning
  ties, plus an exact-binary-fraction band-edge test pinning ``<=`` inclusivity.
- **The walk-forward loop** with a FAKE deterministic model pair (our seam) on synthetic
  prices where SPY carries EXTRA pre-panel history: fold boundaries are purge-clean,
  eval points stride correctly, realized returns are over exactly ``(t, t+H]``, and a
  POISON fake proves each forecast frame is the asset's FULL OWN history (production
  ``_stage_features`` shape) truncated to dates ``<= panel_dates[t]`` — never a future
  bar, never panel-truncated.
- **Report render**: the markdown carries the pre-registered rule, both models' numbers,
  the decision line, the fingerprints, and renders NaN metrics as ``n/a``.
- **CLI**: ``main()`` end-to-end against a seeded tmp cache db (``--quick --splits 2``)
  writes a report file containing a decision + the rule text and exits 0.
- **One tiny end-to-end** with the real ``EWMABaseline`` + a ci_config-trained JEPA on
  ~380 synthetic bars, 2 splits — completes with finite metrics under the CPU budget.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import ci_config
from seeding import seed_prices

from abe.afml.purged_cv import purged_walk_forward_splits, validate_no_leakage
from abe.calc import log_returns
from abe.constants import HORIZON_BARS, UNIVERSE
from abe.eval.walk_forward import (
    CALIBRATION_Z,
    EWMA_REMAINS_DEFAULT,
    JEPA_PROMOTED,
    EvalReport,
    ModelMetrics,
    coverage,
    decide_promotion,
    eval_positions,
    forecast_mse,
    information_coefficient,
    main,
    production_ewma_factory,
    realized_horizon_return,
    render_report_markdown,
    run_walk_forward_eval,
)
from abe.features.build import build_features
from abe.model.base import EWMABaseline, Forecast, WorldModel
from abe.model.jepa import JEPAConfig
from abe.model.train import train_jepa

# --------------------------------------------------------------------------- #
# Synthetic data + a recording fake WorldModel (our seam)
# --------------------------------------------------------------------------- #


def _synthetic_prices(
    assets: tuple[str, ...],
    n: int,
    seed: int,
    mu: float = 0.0004,
    vol: float = 0.01,
    extra: dict[str, int] | None = None,
) -> dict[str, pd.Series]:
    """A geometric-random-walk adjusted-close series per asset (CacheAdapter shape).

    ``extra`` grants named assets that many EXTRA leading bars (a longer own history
    ending on the same last date) — models the real SPY-from-1993 vs ACWI-from-2008
    asymmetry the production frames carry.
    """
    extra = extra or {}
    max_extra = max([0, *extra.values()])
    all_dates = pd.Index(
        pd.bdate_range("2015-01-01", periods=n + max_extra).strftime("%Y-%m-%d"), name="date"
    )
    rng = np.random.default_rng(seed)
    prices: dict[str, pd.Series] = {}
    for offset, asset in enumerate(assets):
        n_asset = n + extra.get(asset, 0)
        dates = all_dates[len(all_dates) - n_asset :]
        daily = rng.normal(mu, vol, size=n_asset - 1)
        levels = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(daily))))
        prices[asset] = pd.Series(levels + offset, index=dates, name=asset)
    return prices


class _RecordingFake:
    """A deterministic fake WorldModel recording every frame it forecasts on.

    ``mu`` fixed (default 0.0) makes the realized-return math checkable end-to-end.
    ``calls`` is the poison trail: per forecast, each asset's ``(rows, max_date)`` —
    the loop must feed each asset its FULL own history truncated to ``<= t``'s date,
    so ``max_date`` must equal the eval date exactly (no future bar, no gap) and
    ``rows`` must reflect the asset's own (possibly longer-than-panel) history.
    """

    def __init__(self, version: str, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.model_version = version
        self._mu = mu
        self._sigma = sigma
        self.calls: list[dict[str, tuple[int, str]]] = []

    def forecast(self, features: dict[str, pd.DataFrame]) -> dict[str, Forecast]:
        self.calls.append(
            {asset: (len(frame), str(frame.index.max())) for asset, frame in features.items()}
        )
        return {asset: Forecast(mu=self._mu, sigma=self._sigma) for asset in features}


# --------------------------------------------------------------------------- #
# Metrics on hand-computed values
# --------------------------------------------------------------------------- #


def test_forecast_mse_hand_computed() -> None:
    # errors [0, 0, -1] -> squared [0, 0, 1] -> mean 1/3.
    assert forecast_mse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(1.0 / 3.0)


def test_information_coefficient_perfect_and_inverse() -> None:
    # Monotone-up pairs -> Spearman +1; monotone-down -> -1.
    assert information_coefficient([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(
        1.0
    )
    assert information_coefficient([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(
        -1.0
    )


def test_information_coefficient_constant_is_nan() -> None:
    # A constant forecast has no rank information -> NaN, not a crash.
    assert math.isnan(information_coefficient([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_coverage_exactly_0_9_and_0_5() -> None:
    # mu=0, sigma=1: within iff |realized| <= CALIBRATION_Z. 9 inside + 1 outside -> 0.9.
    mu = [0.0] * 10
    sigma = [1.0] * 10
    inside = CALIBRATION_Z / 2.0  # comfortably within
    outside = CALIBRATION_Z + 1.0  # comfortably outside
    realized_90 = [inside] * 9 + [outside]
    assert coverage(mu, sigma, realized_90) == pytest.approx(0.9)
    realized_50 = [inside] * 5 + [outside] * 5
    assert coverage(mu, sigma, realized_50) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Promotion rule — every branch, EWMA wins ties, inclusive boundaries
# --------------------------------------------------------------------------- #


def _metrics(mse: float, cov: float) -> ModelMetrics:
    return ModelMetrics("m", {}, {}, {}, mse, float("nan"), cov, 100)


@pytest.mark.parametrize(
    ("ewma", "jepa", "expected"),
    [
        # JEPA wins both: strictly lower MSE + well-calibrated + no worse than EWMA.
        (_metrics(1e-3, 0.88), _metrics(5e-4, 0.90), JEPA_PROMOTED),
        # Better MSE but bad calibration (coverage far outside the +-5pp band) -> stay.
        (_metrics(1e-3, 0.90), _metrics(5e-4, 0.70), EWMA_REMAINS_DEFAULT),
        # Worse MSE -> stay regardless of calibration.
        (_metrics(5e-4, 0.90), _metrics(1e-3, 0.90), EWMA_REMAINS_DEFAULT),
        # Tie on MSE (not strictly lower) -> EWMA default bias.
        (_metrics(1e-3, 0.90), _metrics(1e-3, 0.90), EWMA_REMAINS_DEFAULT),
        # Lower MSE, coverage within band, but WORSE than EWMA's distance -> stay.
        (_metrics(1e-3, 0.89), _metrics(5e-4, 0.86), EWMA_REMAINS_DEFAULT),
        # EQUAL coverage distances (identical coverage) + lower MSE -> PROMOTED:
        # no_worse is INCLUSIVE (<=); a '<' mutation would flip this row to stay.
        (_metrics(1e-3, 0.88), _metrics(5e-4, 0.88), JEPA_PROMOTED),
    ],
)
def test_decide_promotion_branches(ewma: ModelMetrics, jepa: ModelMetrics, expected: str) -> None:
    decision, rationale = decide_promotion(ewma, jepa)
    assert decision == expected
    assert rationale  # never empty


def test_decide_promotion_band_edge_is_inclusive() -> None:
    # Exact binary fractions so distance == tolerance EXACTLY (0.875 - 0.75 = 0.125,
    # all exactly representable): within_band is INCLUSIVE (<=), so the edge promotes;
    # a '<' mutation would flip this to stay.
    ewma = _metrics(1e-3, 0.5)  # distance 0.375 — further from target than JEPA
    jepa = _metrics(5e-4, 0.75)  # distance exactly == tolerance
    decision, _ = decide_promotion(ewma, jepa, target=0.875, tolerance=0.125)
    assert decision == JEPA_PROMOTED


# --------------------------------------------------------------------------- #
# Walk-forward geometry helpers
# --------------------------------------------------------------------------- #


def test_realized_horizon_return_is_exactly_t_plus_1_to_t_plus_h() -> None:
    daily = np.arange(10, dtype=np.float64)  # [0,1,2,...,9]
    # (t=2, H=3] -> positions 3,4,5 -> 3+4+5 = 12.
    assert realized_horizon_return(daily, t=2, horizon=3) == pytest.approx(12.0)


def test_eval_positions_stride_and_horizon_fit() -> None:
    test_idx = np.arange(50, 100, dtype=np.int_)
    positions = eval_positions(test_idx, horizon=21, stride=21, n_panel=120)
    # Strided by 21 from 50; drop any t whose t+21 would exceed n_panel-1 = 119.
    assert positions == [50, 71, 92]
    assert all(np.diff(positions) == 21)


# --------------------------------------------------------------------------- #
# The loop with a FAKE model pair (fold boundaries, stride, realized, poison)
# --------------------------------------------------------------------------- #


def test_walk_forward_loop_geometry_and_no_future_leak() -> None:
    assets = ("SPY", "ACWI", "AGG")
    extra = 63  # SPY gets 63 extra leading bars — a longer own history than the panel
    prices = _synthetic_prices(assets, n=260, seed=1, extra={"SPY": extra})
    horizon = HORIZON_BARS
    n_splits = 3

    # Shared fake instances that accumulate the poison trail across ALL folds (a fake is
    # our seam; reusing one instance lets `calls` span the whole run's eval points).
    jepa_fake = _RecordingFake("jepa:fake")
    ewma_fake = _RecordingFake("ewma")

    def fake_jepa_factory(train_prices: dict[str, pd.Series], config: JEPAConfig) -> WorldModel:
        return jepa_fake

    def fake_ewma_factory(train_prices: dict[str, pd.Series]) -> WorldModel:
        return ewma_fake

    report = run_walk_forward_eval(
        prices,
        n_splits=n_splits,
        model_factory_jepa=fake_jepa_factory,
        config=JEPAConfig(),
        model_factory_ewma=fake_ewma_factory,
    )

    # Rebuild the joint panel + splits the loop used and confirm they are purge-clean.
    # Panel = inner join of per-asset returns = the SHORT assets' return dates (n-1).
    panel = pd.concat({a: log_returns(prices[a]) for a in assets}, axis=1, join="inner")
    panel_dates = [str(d) for d in panel.index]
    n_panel = len(panel)
    assert n_panel == 260 - 1  # SPY's extra history must NOT stretch the panel
    splits = purged_walk_forward_splits(n_panel, n_splits, horizon=horizon)
    validate_no_leakage(splits, horizon)  # the loop calls this too; assert it holds here

    # Fold metadata matches the split geometry.
    assert len(report.folds) == n_splits
    for fold, (train_idx, test_idx) in zip(report.folds, splits, strict=True):
        assert fold.train_bars == int(train_idx.size)
        assert fold.test_bars == int(test_idx.size)

    # Eval points: the exact strided set the loop should have produced (fold order).
    expected_positions = [
        t for _, test_idx in splits for t in eval_positions(test_idx, horizon, horizon, n_panel)
    ]

    # POISON + production shape: for eval point t, every asset's frame must end at
    # EXACTLY panel_dates[t] (data <= t, no future bar) and carry the asset's FULL own
    # history up to that date — SPY's frame is `extra` bars LONGER than the panel
    # prefix (the production `_stage_features` shape), the short assets' is t+1 rows.
    for fake in (jepa_fake, ewma_fake):
        assert len(fake.calls) == len(expected_positions)
        for call, t in zip(fake.calls, expected_positions, strict=True):
            for _asset, (_rows, max_date) in call.items():
                assert max_date == panel_dates[t]
            assert call["SPY"][0] == t + 1 + extra
            assert call["ACWI"][0] == t + 1
            assert call["AGG"][0] == t + 1

    # Realized-return math end-to-end: with fake mu=0, pooled MSE = mean(realized^2)
    # over the panel-aligned daily returns.
    daily = {a: panel[a].to_numpy(dtype=float) for a in assets}
    realized_all = [
        realized_horizon_return(daily[a], t, horizon) for t in expected_positions for a in assets
    ]
    expected_mse = float(np.mean(np.square(realized_all)))
    assert report.ewma.pooled_mse == pytest.approx(expected_mse)
    assert report.jepa.pooled_mse == pytest.approx(expected_mse)
    assert 0.0 <= report.ewma.pooled_coverage <= 1.0
    # Fake mu=0 -> equal MSE -> tie -> EWMA default bias.
    assert report.decision == EWMA_REMAINS_DEFAULT


def test_run_rejects_horizon_config_mismatch() -> None:
    prices = _synthetic_prices(("SPY", "ACWI", "AGG"), n=120, seed=2)
    with pytest.raises(ValueError, match="must match|different horizons"):
        run_walk_forward_eval(
            prices,
            n_splits=2,
            model_factory_jepa=lambda p, c: _RecordingFake("jepa:fake"),
            config=JEPAConfig(horizon=10, context_len=8),  # != eval horizon (default 21)
        )


# --------------------------------------------------------------------------- #
# Report render
# --------------------------------------------------------------------------- #


def _sample_report(decision: str) -> EvalReport:
    ewma = ModelMetrics(
        "ewma",
        {"SPY": 1e-3},
        {"SPY": 0.05},
        {"SPY": 0.88},
        pooled_mse=1.1e-3,
        pooled_ic=0.04,
        pooled_coverage=0.88,
        n_points=120,
    )
    jepa = ModelMetrics(
        "jepa:abcd1234",
        {"SPY": 9e-4},
        {"SPY": 0.06},
        {"SPY": 0.90},
        pooled_mse=9.5e-4,
        pooled_ic=0.06,
        pooled_coverage=0.90,
        n_points=120,
    )
    return EvalReport(
        universe=("SPY", "ACWI", "AGG"),
        horizon=HORIZON_BARS,
        n_splits=3,
        stride=HORIZON_BARS,
        calibration_z=CALIBRATION_Z,
        coverage_target=0.90,
        coverage_tolerance_pp=0.05,
        folds=(),
        ewma=ewma,
        jepa=jepa,
        decision=decision,
        decision_rationale="test rationale",
        config_fingerprint={"n_splits": 3},
        config_hash="deadbeef0000",
        data_fingerprint={"panel_bars": 4596},
        runtime_seconds=12.3,
        generated_at_utc="2026-07-08T00:00:00+00:00",
    )


def test_render_report_contains_rule_numbers_and_decision() -> None:
    markdown = render_report_markdown(_sample_report(EWMA_REMAINS_DEFAULT))
    # Pre-registered mechanical rule text.
    assert "promoted ONLY IF" in markdown
    assert "EWMA remains the default" in markdown
    # Both models' versions + numbers.
    assert "ewma" in markdown
    assert "jepa:abcd1234" in markdown
    assert "9.5000e-04" in markdown  # JEPA pooled MSE
    # Decision line + fingerprints + manual-promotion footer.
    assert EWMA_REMAINS_DEFAULT in markdown
    assert "deadbeef0000" in markdown
    assert "ABE_MODEL" in markdown


def test_render_report_promoted_branch() -> None:
    markdown = render_report_markdown(_sample_report(JEPA_PROMOTED))
    assert JEPA_PROMOTED in markdown


def test_render_report_nan_metrics_render_as_na() -> None:
    # A constant-mu asset yields NaN IC (a reachable state) — the render must show
    # "n/a", never crash or print "nan".
    base = _sample_report(EWMA_REMAINS_DEFAULT)
    nan_jepa = dataclasses.replace(
        base.jepa, pooled_ic=float("nan"), per_asset_ic={"SPY": float("nan")}
    )
    markdown = render_report_markdown(dataclasses.replace(base, jepa=nan_jepa))
    assert "n/a" in markdown
    assert "| nan |" not in markdown


# --------------------------------------------------------------------------- #
# CLI: main() end-to-end against a seeded tmp cache db
# --------------------------------------------------------------------------- #


def test_cli_main_writes_report_with_decision(tmp_path: Path) -> None:
    db_path = tmp_path / "abe.db"
    seed_prices(db_path, days=400, seed=9)
    out_path = tmp_path / "eval" / "report.md"  # exercises the parent-mkdir path too
    exit_code = main(["--db", str(db_path), "--out", str(out_path), "--quick", "--splits", "2"])
    # Exit 0 regardless of who wins — the gate is the eval existing.
    assert exit_code == 0
    assert out_path.is_file()
    text = out_path.read_text(encoding="utf-8")
    assert (JEPA_PROMOTED in text) or (EWMA_REMAINS_DEFAULT in text)
    assert "promoted ONLY IF" in text  # the pre-registered rule text
    assert "jepa_config" in text  # the full config fingerprint landed


# --------------------------------------------------------------------------- #
# Tiny end-to-end: real EWMABaseline + a ci_config-trained JEPA
# --------------------------------------------------------------------------- #


def test_end_to_end_real_models_finite_metrics() -> None:
    prices = _synthetic_prices(UNIVERSE, n=380, seed=7)
    config = ci_config()  # horizon stays HORIZON_BARS=21

    def jepa_factory(train_prices: dict[str, pd.Series], cfg: JEPAConfig) -> WorldModel:
        # Route through the SAME production assembly the CLI uses (build_features +
        # per-asset log-returns -> train_jepa), but with the tiny ci config.
        features_matrix = build_features(train_prices, macro=None)
        returns = pd.DataFrame({a: log_returns(p) for a, p in train_prices.items()})
        return train_jepa(features_matrix, returns, cfg).to_model()

    report = run_walk_forward_eval(
        prices,
        n_splits=2,
        model_factory_jepa=jepa_factory,
        config=config,
        model_factory_ewma=production_ewma_factory,
    )

    for metrics in (report.ewma, report.jepa):
        assert metrics.n_points > 0
        assert math.isfinite(metrics.pooled_mse)
        assert 0.0 <= metrics.pooled_coverage <= 1.0
    # Two folds train two distinct ensembles -> the pooled label lists BOTH hashes.
    assert report.jepa.model_version.startswith(("jepa:", "per-fold ensembles:"))
    assert report.ewma.model_version == "ewma"
    assert isinstance(EWMABaseline(), WorldModel)  # the eval scores a real WorldModel
    assert report.decision in {JEPA_PROMOTED, EWMA_REMAINS_DEFAULT}
    # The report renders without error and records the decision.
    assert report.decision in render_report_markdown(report)
