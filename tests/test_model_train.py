"""Training + config-toggle tests (plan.md Step 13).

Covers ``train_jepa`` end-to-end (usable checkpoint, positive residual variance),
the ``shuffled_target_control`` measurement-validity anchor (real < shuffled), the
``load_model`` EWMA<->JEPA toggle + the api ``resolve_startup_model`` env helper
(DEFAULT stays EWMA; invalid jepa config fails loud), and the ``python -m
abe.model.train`` CLI against a seeded cache db. Tiny nets + few epochs +
single-thread torch (conftest) keep it well under the 60s CPU budget.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import ci_config, dates_index, iid_data
from seeding import seed_prices

from abe import storage
from abe.api import resolve_startup_model
from abe.constants import UNIVERSE
from abe.ingest.macro import MACRO_OK, MacroStatus
from abe.model import load_model
from abe.model.base import EWMABaseline
from abe.model.jepa import JEPAModel
from abe.model.train import load_training_data, main, shuffled_target_control, train_jepa
from abe.pipeline import run_pipeline

_OK_STATUS = MacroStatus(enabled=True, code=MACRO_OK, message="test key accepted")


def _sinusoidal_data(
    seed: int, n: int = 500, period: int = 40, amp: float = 0.008, noise: float = 0.002
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A learnable temporal signal: the H-day forward sum is predictable from context."""
    bars = np.arange(n)
    values = amp * np.sin(2 * np.pi * bars / period) + noise * np.random.default_rng(
        seed
    ).standard_normal(n)
    index = dates_index(n)
    return (
        pd.DataFrame({"SPY_log_return": values}, index=index),
        pd.DataFrame({"SPY": values}, index=index),
    )


@pytest.fixture(scope="module")
def saved_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny trained checkpoint saved to disk — reused by the toggle tests."""
    features, returns = iid_data(seed=5)
    checkpoint = train_jepa(features, returns, ci_config())
    path = tmp_path_factory.mktemp("jepa") / "model.pt"
    checkpoint.save(path)
    return path


# --------------------------------------------------------------------------- #
# train_jepa produces a usable checkpoint
# --------------------------------------------------------------------------- #


def test_train_jepa_produces_usable_checkpoint() -> None:
    features, returns = iid_data(seed=11)
    config = ci_config()
    checkpoint = train_jepa(features, returns, config)
    assert len(checkpoint.ensemble_state_dicts) == config.n_seeds
    assert checkpoint.collapse_report.passed
    # Residual variance (the sigma scale term) is strictly positive per asset.
    assert checkpoint.residual_var["SPY"] > 0.0
    assert checkpoint.pooled_residual_var > 0.0
    # And it round-trips through an in-memory model as a real forecaster.
    model = checkpoint.to_model()
    forecast = model.forecast(
        {
            "SPY": pd.DataFrame(
                {"log_return": features["SPY_log_return"].to_numpy()}, index=features.index
            )
        }
    )["SPY"]
    assert forecast.sigma > 0.0


# --------------------------------------------------------------------------- #
# Shuffled-target control (plan §12 anchor): real < shuffled
# --------------------------------------------------------------------------- #


def test_shuffled_target_control_real_beats_shuffled() -> None:
    # Breaking the temporal feature->return relationship (shuffling the target)
    # must score measurably WORSE on held-out prediction loss than the real run.
    features, returns = _sinusoidal_data(seed=0)
    config = ci_config(n_seeds=1, epochs=40)
    real_mse, shuffled_mse = shuffled_target_control(features, returns, config, seed=0)
    assert real_mse < shuffled_mse


# --------------------------------------------------------------------------- #
# The EWMA<->JEPA config toggle (load_model)
# --------------------------------------------------------------------------- #


def test_load_model_ewma_is_baseline() -> None:
    model = load_model("ewma")
    assert isinstance(model, EWMABaseline)


def test_load_model_jepa_from_checkpoint(saved_checkpoint: Path) -> None:
    model = load_model("jepa", saved_checkpoint)
    assert isinstance(model, JEPAModel)
    assert model.model_version.startswith("jepa:")


def test_load_model_jepa_without_path_raises() -> None:
    with pytest.raises(ValueError, match="requires a checkpoint_path"):
        load_model("jepa", None)


def test_load_model_jepa_bad_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_model("jepa", tmp_path / "missing.pt")


def test_load_model_unknown_spec_raises() -> None:
    with pytest.raises(ValueError, match="unknown model_spec"):
        load_model("transformer")


# --------------------------------------------------------------------------- #
# The api startup env helper (DEFAULT stays EWMA; fail-loud on bad jepa config)
# --------------------------------------------------------------------------- #


def test_resolve_startup_model_default_is_ewma() -> None:
    # DEFAULT-stays-EWMA pin: no env => EWMABaseline (never the JEPA).
    assert isinstance(resolve_startup_model({}), EWMABaseline)


def test_resolve_startup_model_jepa_from_env(saved_checkpoint: Path) -> None:
    model = resolve_startup_model(
        {"ABE_MODEL": "jepa", "ABE_JEPA_CHECKPOINT": str(saved_checkpoint)}
    )
    assert isinstance(model, JEPAModel)


def test_resolve_startup_model_jepa_without_checkpoint_fails_loud() -> None:
    # Fail-loud posture: an invalid jepa config raises rather than silently EWMA.
    with pytest.raises(ValueError, match="requires a checkpoint_path"):
        resolve_startup_model({"ABE_MODEL": "jepa"})


# --------------------------------------------------------------------------- #
# The CLI (python -m abe.model.train) against a seeded cache db
# --------------------------------------------------------------------------- #


def test_cli_trains_and_writes_loadable_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "abe.db"
    seed_prices(db_path, days=240, seed=3)
    out_path = tmp_path / "cli-jepa.pt"
    exit_code = main(
        [
            "--db",
            str(db_path),
            "--out",
            str(out_path),
            "--seeds",
            "2",
            "--epochs",
            "20",
            "--context-len",
            "16",
            "--latent-dim",
            "8",
            "--hidden-dim",
            "16",
        ]
    )
    assert exit_code == 0
    assert out_path.is_file()
    assert (tmp_path / "cli-jepa.pt.meta.json").is_file()
    model = JEPAModel.from_checkpoint(out_path)
    assert model.model_version.startswith("jepa:")


def test_load_training_data_shapes(tmp_path: Path) -> None:
    db_path = tmp_path / "abe.db"
    seed_prices(db_path, days=180, seed=4)
    features_matrix, returns = load_training_data(db_path)
    # Flat {asset}_log_return columns for every universe asset; returns per asset.
    assert "SPY_log_return" in features_matrix.columns
    assert set(returns.columns) == {"SPY", "ACWI", "AGG"}
    assert len(features_matrix) > 0


# --------------------------------------------------------------------------- #
# Integration: JEPAModel THROUGH the production caller (run_pipeline)
# --------------------------------------------------------------------------- #


def test_jepa_forecasts_through_production_pipeline(tmp_path: Path) -> None:
    # The code-quality rule: a new component must be exercised through its
    # production entry point, not just a hand-built frame. run_pipeline hands
    # JEPAModel the SAME per-asset {asset: log_return frame} dict it hands EWMA;
    # this asserts the forecast stage lands jepa: rows for the whole universe.
    db_path = tmp_path / "abe.db"
    seed_prices(db_path, days=240, seed=3)
    features_matrix, returns = load_training_data(db_path)
    model = train_jepa(features_matrix, returns, ci_config()).to_model()

    conn = storage.open_writer(db_path)
    try:
        run_id = run_pipeline(conn, trigger="manual", model=model, macro_status=_OK_STATUS)
        # An 'ok' run => every stage (incl. forecast -> blend -> optimize) succeeded.
        assert storage.latest_ok_run_id(conn) == run_id
        rows = conn.execute(
            "SELECT asset, model_version, sigma FROM forecasts WHERE run_id = ?", (run_id,)
        ).fetchall()
        assert {str(row[0]) for row in rows} == set(UNIVERSE)
        assert all(str(row[1]).startswith("jepa:") for row in rows)
        assert all(float(row[2]) > 0.0 for row in rows)
    finally:
        conn.close()
