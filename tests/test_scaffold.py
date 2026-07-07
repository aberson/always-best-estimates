"""Scaffold tests: constants integrity, /health liveness, pinned-import smoke."""

import importlib
import math

import pytest


def test_constants_plan_conformance() -> None:
    """Scalar pins + universe from plan.md section 12 Appendix."""
    from abe import constants

    assert constants.HORIZON_BARS == 21
    assert constants.TRADING_DAYS == 252
    assert constants.UNIVERSE == ("SPY", "ACWI", "AGG")
    assert constants.DELTA == 2.5
    assert constants.TAU == 0.05
    assert constants.W_MAX == 0.60


def test_w_mkt_sums_to_one() -> None:
    from abe import constants

    assert math.isclose(sum(constants.W_MKT.values()), 1.0, rel_tol=0, abs_tol=1e-12)


def test_w_mkt_keys_match_universe() -> None:
    from abe import constants

    assert tuple(constants.W_MKT) == constants.UNIVERSE


def test_fred_release_lag_keys_match_fred_daily() -> None:
    from abe import constants

    assert set(constants.FRED_RELEASE_LAG) == set(constants.FRED_DAILY)


def test_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    from abe.api import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "module_name",
    [
        "fastapi",
        "uvicorn",
        "pypfopt",
        "cvxpy",
        "yfinance",
        "fredapi",
        "torch",
        "pandas",
        "numpy",
        "sklearn",
        "dotenv",
    ],
)
def test_runtime_dependency_imports(module_name: str) -> None:
    """Every pinned runtime dep actually imports.

    A green ``uv sync`` alone doesn't prove torch's DLLs load on
    Windows/py312 — importing is the real check.
    """
    importlib.import_module(module_name)


def test_clarabel_solver_available() -> None:
    import cvxpy

    assert "CLARABEL" in cvxpy.installed_solvers()
