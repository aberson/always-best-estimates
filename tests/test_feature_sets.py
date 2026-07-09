"""Selectable-feature-set tests (Track 2 Step 24).

Done-when (plan §7 Step 24): the pipeline runs under both feature sets and the
feature card/detail reflects the chosen set; ``basic`` remains byte-identical to
V1 (the Step 19 parity golden pins that; here we assert the discriminating marker).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from seeding import seed_prices

from abe import config as config_module
from abe import storage
from abe.constants import UNIVERSE
from abe.ingest.macro import MACRO_OK, MacroStatus
from abe.pipeline import run_pipeline
from abe.registry import BasicFeatureBuilder, FracDiffMacroFeatureBuilder

ENABLED_MACRO = MacroStatus(enabled=True, code=MACRO_OK, message="test key")


def _price_frames(n: int = 120) -> dict[str, pd.DataFrame]:
    dates = pd.Index(
        [stamp.strftime("%Y-%m-%d") for stamp in pd.bdate_range("2026-01-01", periods=n)],
        name="date",
    )
    rng = np.random.default_rng(9)
    frames: dict[str, pd.DataFrame] = {}
    for asset in UNIVERSE:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=n)))
        frames[asset] = pd.DataFrame({"adj_close": prices}, index=dates)
    return frames


def _macro_frame() -> pd.DataFrame:
    # two series, available well before the price window → every trading date sees them
    return pd.DataFrame(
        {
            "series_id": ["DGS10", "VIXCLS"],
            "obs_date": ["2025-12-01", "2025-12-01"],
            "value": [4.2, 18.0],
            "available_date": ["2025-12-02", "2025-12-02"],
        }
    )


def _seed_macro(conn: sqlite3.Connection) -> None:
    for series_id, value in (("DGS10", 4.2), ("VIXCLS", 18.0)):
        storage.upsert_row(
            conn,
            "macro",
            {
                "series_id": series_id,
                "obs_date": "2025-12-01",
                "value": value,
                "available_date": "2025-12-02",
                "ingested_at_utc": "2025-12-02T00:00:00Z",
            },
        )


# --------------------------------------------------------------------------- #
# The builders directly
# --------------------------------------------------------------------------- #


def test_basic_builder_has_no_feature_set_marker() -> None:
    """basic's detail is byte-identical V1 — no fracdiff_macro marker."""
    bundle = BasicFeatureBuilder().build(_price_frames())
    assert "feature_set" not in bundle.detail
    assert set(bundle.detail) == {"features", "windows", "latest"}


def test_fracdiff_macro_builder_reflects_the_set_and_feeds_forecaster() -> None:
    bundle = FracDiffMacroFeatureBuilder().build(_price_frames(), _macro_frame())
    assert bundle.detail["feature_set"] == "fracdiff_macro"
    macro_cols = bundle.detail["macro_columns"]
    assert isinstance(macro_cols, list)
    assert {"DGS10", "VIXCLS"} <= set(macro_cols)  # macro columns present
    # each per-asset frame still carries log_return so the forecaster works
    for asset in UNIVERSE:
        assert "log_return" in bundle.features_frames[asset].columns
        assert "DGS10" in bundle.features_frames[asset].columns


def test_fracdiff_macro_without_macro_omits_columns() -> None:
    bundle = FracDiffMacroFeatureBuilder().build(_price_frames(), None)
    assert bundle.detail["feature_set"] == "fracdiff_macro"
    assert bundle.detail["macro_columns"] == []  # no macro → no columns, no error


# --------------------------------------------------------------------------- #
# Through the pipeline (both feature sets)
# --------------------------------------------------------------------------- #


def _features_detail(conn: sqlite3.Connection, run_id: int) -> dict[str, object]:
    row = conn.execute(
        "SELECT detail_json FROM run_stages WHERE run_id = ? AND stage = 'features'", (run_id,)
    ).fetchone()
    return json.loads(row[0])


def test_pipeline_runs_under_both_feature_sets(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "abe.db"
    seed_prices(db_path, days=120)
    conn = storage.open_writer(db_path)
    try:
        _seed_macro(conn)
        central = config_module.get_central_config(conn)  # feature_set=basic
        fracdiff_cfg = config_module.create_config(
            conn,
            name="fracdiff-macro",
            feature_set="fracdiff_macro",
            forecaster="ewma",
            view_scenario_id=central.view_scenario_id,
            optimizer="mvu",
        )
        basic_run = run_pipeline(conn, config=central, force=True, macro_status=ENABLED_MACRO)
        fracdiff_run = run_pipeline(
            conn, config=fracdiff_cfg, force=True, macro_status=ENABLED_MACRO
        )
        basic_detail = _features_detail(conn, basic_run)
        fracdiff_detail = _features_detail(conn, fracdiff_run)

        # the card reflects the chosen set: basic has no marker, fracdiff_macro does
        assert "feature_set" not in basic_detail
        assert fracdiff_detail["feature_set"] == "fracdiff_macro"
        assert {"DGS10", "VIXCLS"} <= set(fracdiff_detail["macro_columns"])  # type: ignore[arg-type]
        # both runs completed ok with the full six stages
        for run_id in (basic_run, fracdiff_run):
            statuses = [
                row[0]
                for row in conn.execute(
                    "SELECT status FROM run_stages WHERE run_id = ?", (run_id,)
                ).fetchall()
            ]
            assert all(status == "ok" for status in statuses)
    finally:
        conn.close()
