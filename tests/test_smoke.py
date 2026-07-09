"""Step 9 smoke-gate tests: the ``pytest -m smoke`` integration + portable variants.

Two tiers:

- ``@pytest.mark.smoke`` — drives :func:`smoke.run_smoke` against the REAL
  backfilled db (``ABE_REAL_DB`` env override, else ``<root>/data/abe.db`` —
  the same location idiom as the ``realdb``-marked tests). Deselected by
  default (pytest ``addopts``); ``uv run pytest -m smoke`` selects it (a CLI
  ``-m`` overrides addopts). It NEVER skips: whoever runs ``-m smoke`` is
  asking for the gate, and a skip would exit 0 — a vacuous pass
  indistinguishable from a real one (measurement-validity rule: fail loud on
  fallback config). A missing or empty db FAILS with the backfill hint.
  NOTE: like ``scripts/smoke.py`` itself, it WRITES one real run to that db —
  that is the gate's job.
- Default-suite tests — the same smoke core against a seeded tmp db
  (``tests/seeding.py``), so the smoke logic is proven on fresh clones that
  have no real db, plus the precondition/failure/watchdog exit paths, the CLI
  exit-code contract, and the addopts meta-guard.
"""

import math
import os
import threading
import tomllib
from pathlib import Path

import pytest
from seeding import seed_prices
from smoke import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_PRECONDITION,
    SmokePreconditionError,
    SmokeReport,
    SmokeTimeoutError,
    main,
    run_smoke,
)

from abe import storage
from abe.constants import UNIVERSE
from abe.pipeline import STAGES


def _real_db_path() -> Path:
    override = os.environ.get("ABE_REAL_DB")
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parent.parent
    return project_root / storage.DEFAULT_DB_PATH


REAL_DB = _real_db_path()
"""The backfilled production db (gitignored). Override via ABE_REAL_DB."""


@pytest.fixture()
def offline_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No FRED key + an .env-less cwd -> the startup macro probe stays offline."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _assert_passing_report(report: SmokeReport, db_path: Path) -> None:
    """The SmokeReport contract a PASSING smoke must satisfy."""
    assert report.db_path == db_path
    assert report.run_id >= 1
    assert [card.stage for card in report.cards] == list(STAGES)
    assert all(card.status == "ok" for card in report.cards)
    assert set(report.weights) == set(UNIVERSE)
    assert math.isclose(report.weights_sum, 1.0, abs_tol=1e-9)
    assert report.ingest_source == "cache"
    assert report.config_id >= 1  # Track 2: the run is Config-tagged (central)
    assert 0.0 <= report.elapsed_s <= 120.0


# --------------------------------------------------------------------------- #
# THE smoke gate: `uv run pytest -m smoke` (real db)
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_smoke_real_db() -> None:
    """One REAL end-to-end cycle against the backfilled production db.

    Never skips (module docstring): a missing/empty real db is a FAIL with
    the backfill hint, not a green skip.
    """
    try:
        report = run_smoke(REAL_DB)
    except SmokePreconditionError as exc:
        pytest.fail(
            f"smoke gate cannot pass vacuously - real db precondition missing at {REAL_DB} "
            f"(run `uv run python -m abe.ingest.prices --backfill` first, or set "
            f"ABE_REAL_DB): {exc}"
        )
    _assert_passing_report(report, REAL_DB)


@pytest.mark.smoke
def test_smoke_real_db_config_tagged() -> None:
    """Track 2 Step 20 gate: one REAL central-Config cycle end-to-end on the real
    db lands all six stages ok AND tags the run with the central config_id.

    Independently cross-checks the report's config_id against the db's central
    config (the smoke core already asserts central-match; this names the gate and
    proves the id really is the ``is_central`` row on the real db)."""
    from abe import config as config_module

    try:
        report = run_smoke(REAL_DB)
    except SmokePreconditionError as exc:
        pytest.fail(
            f"Step 20 smoke gate cannot pass vacuously - real db precondition missing at "
            f"{REAL_DB}: {exc}"
        )
    _assert_passing_report(report, REAL_DB)
    conn = storage.open_read_only(REAL_DB)
    try:
        central = config_module.get_central_config(conn)
    finally:
        conn.close()
    assert report.config_id == central.config_id
    assert [card.stage for card in report.cards] == list(STAGES)
    assert all(card.status == "ok" for card in report.cards)


# --------------------------------------------------------------------------- #
# Default suite: the smoke logic itself, portable (seeded tmp db)
# --------------------------------------------------------------------------- #


def test_run_smoke_tmp_db(offline_cwd: Path) -> None:
    """run_smoke passes against a seeded tmp db — proves the smoke logic works
    everywhere (fresh clones without the real db)."""
    db_path = offline_cwd / "data" / "abe.db"
    seed_prices(db_path)
    report = run_smoke(db_path)
    _assert_passing_report(report, db_path)
    # The card lines the CLI prints are renderable for every stage.
    for card in report.cards:
        assert card.stage in card.line()


def test_run_smoke_is_repeatable_because_forced(offline_cwd: Path) -> None:
    """force=True bypasses the freshness gate: a second smoke on unchanged data
    still exercises all six stages (never lands a 'skipped' run)."""
    db_path = offline_cwd / "data" / "abe.db"
    seed_prices(db_path)
    first = run_smoke(db_path)
    second = run_smoke(db_path)
    assert second.run_id > first.run_id
    _assert_passing_report(second, db_path)


def test_run_smoke_empty_db_is_precondition_not_failure(tmp_path: Path) -> None:
    """Schema-exists/zero-rows discrimination: an empty db is 'backfill first'
    (exit-3 class), not a confusing downstream stage failure."""
    db_path = tmp_path / "empty.db"
    storage.open_writer(db_path).close()  # schema exists, zero price rows
    with pytest.raises(SmokePreconditionError, match="no price rows"):
        run_smoke(db_path)


def test_watchdog_times_out_hung_cycle_via_thread_join(
    offline_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watchdog joins the worker thread with the budget, so a HUNG cycle
    (stalled solver, stalled keyed startup probe) still yields a classified
    timeout instead of blocking forever. Deterministic: the stubbed cycle
    blocks on an Event; the budget is 50ms; no real sleeps beyond that."""
    db_path = offline_cwd / "data" / "abe.db"
    seed_prices(db_path)
    release = threading.Event()

    def hung_cycle(path: Path, start: float) -> SmokeReport:
        release.wait(10.0)  # hangs far past the tiny budget; released in finally
        raise RuntimeError("released")  # ignored — the main thread already raised

    monkeypatch.setattr("smoke._run_cycle", hung_cycle)
    monkeypatch.setattr("smoke.WATCHDOG_SECONDS", 0.05)
    try:
        with pytest.raises(SmokeTimeoutError, match="SMOKE TIMEOUT"):
            run_smoke(db_path)
    finally:
        release.set()  # unblock the abandoned daemon thread promptly


# --------------------------------------------------------------------------- #
# CLI exit-code contract (0 pass / 1 failure / 3 precondition)
# --------------------------------------------------------------------------- #


def test_main_pass_prints_cards_and_weights(
    offline_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = offline_cwd / "data" / "abe.db"
    seed_prices(db_path)
    assert main(["--db", str(db_path)]) == EXIT_OK
    out = capsys.readouterr().out
    for stage in STAGES:
        assert stage in out  # one card line per stage
    for asset in UNIVERSE:
        assert asset in out  # weights are printed
    assert "weights sum" in out
    assert "SMOKE PASS" in out


def test_main_precondition_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", str(tmp_path / "missing.db")]) == EXIT_PRECONDITION
    err = capsys.readouterr().err
    assert "SMOKE PRECONDITION MISSING" in err
    assert "backfill" in err  # tells the operator what to run first


def test_main_failure_exit_1_with_stage_detail(
    offline_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stage error is a classified FAILURE (exit 1) whose stderr names the
    failing stage and carries its detail_json + runs.error_text (25 seeded
    days -> the forecast stage raises on < MIN_HISTORY_BARS)."""
    db_path = offline_cwd / "short.db"
    seed_prices(db_path, days=25)
    assert main(["--db", str(db_path)]) == EXIT_FAILURE
    err = capsys.readouterr().err
    assert "SMOKE FAILURE" in err
    assert "forecast" in err  # the failing stage is named
    assert "MIN_HISTORY_BARS" in err  # its detail_json reaches the operator
    assert "error_text" in err  # and so does runs.error_text


# --------------------------------------------------------------------------- #
# Meta-guard: the addopts deselection itself
# --------------------------------------------------------------------------- #


def test_addopts_keeps_all_side_effect_markers_deselected() -> None:
    """A dropped addopts term would silently put real-db/network tests into
    every default run. Parses the committed pyproject.toml — deliberately NOT
    the live markexpr, so ad-hoc `-m smoke` / `-m network` runs don't
    false-alarm on this guard."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    addopts = str(config["tool"]["pytest"]["ini_options"]["addopts"])
    for term in ("not network", "not realdb", "not smoke"):
        assert term in addopts, f"pyproject addopts must keep {term!r} deselected: {addopts!r}"
