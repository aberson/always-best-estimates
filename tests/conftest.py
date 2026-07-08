"""Shared pytest setup + the Step 13 JEPA test helpers.

torch thread pin
================
torch defaults to a multi-thread CPU pool; for the Step 13 JEPA tests (tiny nets,
tiny data) that pool's coordination overhead DOMINATES and a single training run
balloons to ~12s. Pinning one thread makes each tiny train ~0.2s, keeping the
whole JEPA suite well under its 60s CPU budget. This is a test-only knob (the
offline production trainer benefits from many threads on the real ~4,600-bar
panel), so it lives here, not in library code.

Shared JEPA helpers (one source of truth)
=========================================
``dates_index`` / ``iid_data`` / ``ci_config`` are imported by BOTH
``test_model_jepa.py`` and ``test_model_train.py`` (``from conftest import ...``
— ``tests/`` is on ``pythonpath``). They live here so the tested CI config
cannot silently fork between the two suites (code-quality one-source-of-truth
rule). ``ci_config`` shrinks ONLY size/speed knobs (latent/hidden dims, K=2,
epochs); loss weights — in particular the deliberately SMALL production
``return_weight`` — are inherited from ``JEPAConfig`` defaults, so the tests
exercise the production objective.
"""

import numpy as np
import pandas as pd
import torch

from abe.model.jepa import JEPAConfig

torch.set_num_threads(1)

IID_STD = 0.01  # daily iid-noise std for the sigma-scale pin


def dates_index(n: int) -> pd.Index:
    """Deterministic ISO-date-string index (the CacheAdapter shape)."""
    return pd.Index(pd.bdate_range("2015-01-01", periods=n).strftime("%Y-%m-%d"), name="date")


def iid_data(seed: int, n: int = 340, s: float = IID_STD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A single-asset (SPY) iid-noise ``(features_matrix, returns)`` pair."""
    values = s * np.random.default_rng(seed).standard_normal(n)
    index = dates_index(n)
    features = pd.DataFrame({"SPY_log_return": values}, index=index)
    returns = pd.DataFrame({"SPY": values}, index=index)
    return features, returns


def ci_config(**overrides: object) -> JEPAConfig:
    """The tiny-but-production-objective JEPA config the CI suites train with."""
    base: dict[str, object] = {
        "context_len": 16,
        "latent_dim": 8,
        "hidden_dim": 16,
        "n_seeds": 2,
        "epochs": 40,
        "feature_names": ("log_return",),
    }
    base.update(overrides)
    return JEPAConfig(**base)  # type: ignore[arg-type]
