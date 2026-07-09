"""Step 19 parity golden: the central Config reproduces V1 byte-for-byte.

``tests/golden/parity_central_v1.json`` was captured from the PRE-refactor
pipeline (``scripts/_capture_parity_golden.py``, since deleted) on a deterministic
seeded db. This test regenerates the same seeded run through the refactored
Config-driven pipeline and asserts byte-identical ``run_stages`` (status + parsed
detail) for every stage — the load-bearing regression gate for the Step 19
refactor (plan §7 Step 19, §9 Testing Strategy).
"""

import json
import sqlite3
from pathlib import Path

import pytest
from seeding import seed_prices

from abe import config as config_module
from abe import storage
from abe.ingest.macro import MACRO_DISABLED_NO_KEY, MacroStatus
from abe.model.base import EWMABaseline
from abe.pipeline import run_pipeline

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "parity_central_v1.json"
DISABLED_MACRO = MacroStatus(enabled=False, code=MACRO_DISABLED_NO_KEY, message="offline (no key)")


def _golden() -> dict[str, dict[str, object]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _run_stages(conn: sqlite3.Connection, run_id: int) -> dict[str, dict[str, object]]:
    rows = conn.execute(
        "SELECT stage, status, detail_json FROM run_stages WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()
    return {
        str(stage): {"status": str(status), "detail": json.loads(detail) if detail else None}
        for stage, status, detail in rows
    }


def _seeded_writer(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "data" / "abe.db"
    seed_prices(db_path)  # deterministic seed=7, matches the golden capture
    return storage.open_writer(db_path)


def test_central_config_matches_v1_golden(tmp_path: Path) -> None:
    """The default (central) Config path reproduces the pre-refactor run_stages."""
    conn = _seeded_writer(tmp_path)
    try:
        run_id = run_pipeline(conn, trigger="manual", force=True, macro_status=DISABLED_MACRO)
        actual = _run_stages(conn, run_id)
        golden = _golden()
        assert set(actual) == set(golden)  # same six stages present
        for stage in golden:
            assert actual[stage] == golden[stage], f"stage {stage!r} drifted from V1 golden"
        # the run is tagged with the central config_id
        central = config_module.get_central_config(conn)
        row = conn.execute("SELECT config_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert row[0] == central.config_id
    finally:
        conn.close()


def test_model_override_matches_v1_golden(tmp_path: Path) -> None:
    """The back-compat model=EWMABaseline() override (the seam tests use) also
    reproduces V1 — an injected default EWMA equals the central's own forecaster."""
    conn = _seeded_writer(tmp_path)
    try:
        run_id = run_pipeline(
            conn, trigger="manual", force=True, model=EWMABaseline(), macro_status=DISABLED_MACRO
        )
        actual = _run_stages(conn, run_id)
        golden = _golden()
        for stage in golden:
            assert actual[stage] == golden[stage], f"stage {stage!r} drifted under model override"
    finally:
        conn.close()


def test_explicit_central_config_matches_golden(tmp_path: Path) -> None:
    """Passing the central Config explicitly is identical to the default path."""
    conn = _seeded_writer(tmp_path)
    try:
        central = config_module.get_central_config(conn)
        run_id = run_pipeline(
            conn, config=central, trigger="manual", force=True, macro_status=DISABLED_MACRO
        )
        actual = _run_stages(conn, run_id)
        golden = _golden()
        for stage in golden:
            assert actual[stage] == golden[stage], f"stage {stage!r} drifted for explicit central"
    finally:
        conn.close()


def test_golden_is_present_and_complete() -> None:
    """Guard against a vacuously-passing test if the golden went missing/empty."""
    golden = _golden()
    assert set(golden) == {"freshness", "ingest", "features", "forecast", "blend", "optimize"}
    assert golden["optimize"]["detail"]["weights"]  # type: ignore[index]
    assert pytest.approx(sum(golden["optimize"]["detail"]["weights"].values())) == 1.0  # type: ignore[index]
