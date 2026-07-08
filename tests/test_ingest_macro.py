"""Macro-ingest tests: key probe / degraded modes, '.'/''-parse, release-lag
shift, incremental fetch, and the fredapi call contract (via a fake fredapi module).

The default suite runs OFFLINE on a fresh clone: tmp_path DBs, canned pandas
Series, fake ``FredClient`` implementations (no unittest.mock of our own code).
The real-FRED check is marked ``@pytest.mark.network`` (deselected by default
via pytest ``addopts``) and self-skips with an explicit reason when no
FRED_API_KEY is discoverable — there is no key on the build machine, so the
keyed path is exercised through fakes here and for real only where a key exists.
The missing-key degraded path needs no key and IS asserted for real (CLI exit
code 2 + the stable code on stderr). Tests that let dotenv touch os.environ use
the ``scrubbed_fred_key_env`` fixture — dotenv writes OUTSIDE monkeypatch's
ledger, and a leaked phantom key would send later "offline" tests to the real
network.
"""

import os
import re
import socket
import sqlite3
import sys
import types
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abe import constants, storage
from abe.ingest.macro import (
    EXIT_MACRO_DISABLED,
    FRED_TIMEOUT_SECONDS,
    MACRO_DISABLED_BAD_KEY,
    MACRO_DISABLED_NO_KEY,
    MACRO_OK,
    FredApiClient,
    available_date,
    ingest_macro,
    load_fred_api_key,
    main,
    parse_fred_value,
    probe_fred_key,
    stored_max_obs_date,
)

UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

# fredapi's exact auth-rejection shapes: fred.py __fetch_data catches the HTTP
# 400 and re-raises ValueError(<FRED XML 'message'>) — these are those messages.
FRED_BAD_KEY_MESSAGE = (
    "Bad Request. The value for variable api_key is not registered. "
    "Read https://fred.stlouisfed.org/docs/api/api_key.html for more information."
)
FRED_MALFORMED_KEY_MESSAGE = (
    "Bad Request. Variable api_key is not a 32 character alpha-numeric lower-case string. "
    "Read https://fred.stlouisfed.org/docs/api/api_key.html for more information."
)

# A well-formed (32 lowercase hex) but unregistered key shape.
WELL_FORMED_KEY = "0123456789abcdef0123456789abcdef"


# --------------------------------------------------------------------------- #
# Helpers: canned series + fake clients + fixtures (no mocks of our own code)
# --------------------------------------------------------------------------- #


def _series(dates: list[str], values: list[object]) -> pd.Series:
    """A canned fredapi-shaped Series: DatetimeIndex → value (float/NaN/str)."""
    return pd.Series(values, index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))


class FakeFredClient:
    """FredClient impl serving canned per-series data; records get_series calls."""

    def __init__(self, series: dict[str, pd.Series]) -> None:
        self.series = series
        self.calls: list[tuple[str, str | None]] = []

    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
        self.calls.append((series_id, observation_start))
        data = self.series[series_id]
        if observation_start is None:
            return data
        return data.loc[data.index >= pd.Timestamp(observation_start)]


class SloppyFullHistoryClient:
    """FredClient impl that IGNORES observation_start and returns everything —
    models a manual full refetch hitting already-stored observations."""

    def __init__(self, series: pd.Series) -> None:
        self.series = series

    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
        return self.series


class RaisingFredClient:
    """FredClient impl that always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
        raise self.exc


@pytest.fixture()
def writer(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = storage.open_writer(tmp_path / "abe.db")
    yield conn
    conn.close()


@pytest.fixture()
def scrubbed_fred_key_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """FRED_API_KEY absent for the test body, with leak-proof teardown.

    ``monkeypatch.setenv`` FIRST records the machine's true prior state in the
    undo ledger (``delenv`` on an absent var records nothing). The test body may
    then let ``dotenv.load_dotenv`` write the var OUTSIDE the ledger, so
    teardown scrubs explicitly and ASSERTS no phantom key survives — a leaked
    key would send later "offline" tests to the real network.
    """
    monkeypatch.setenv("FRED_API_KEY", "sentinel-recorded-in-ledger")
    monkeypatch.delenv("FRED_API_KEY")
    yield
    os.environ.pop("FRED_API_KEY", None)
    # Ordering-independent leak check (monkeypatch undo then restores the
    # machine's true original state, recorded by the setenv above).
    assert "FRED_API_KEY" not in os.environ


def _six_series_client(dates: list[str]) -> FakeFredClient:
    return FakeFredClient(
        {
            series_id: _series(dates, [float(i * 10 + j) for j in range(len(dates))])
            for i, series_id in enumerate(constants.FRED_DAILY)
        }
    )


# --------------------------------------------------------------------------- #
# Fetch + store through a fake client into real tmp SQLite
# --------------------------------------------------------------------------- #


def test_six_series_fetch_and_store(writer: sqlite3.Connection) -> None:
    client = _six_series_client(["2026-07-01", "2026-07-02"])
    counts = ingest_macro(writer, client)

    assert counts == {series_id: 2 for series_id in constants.FRED_DAILY}
    assert [c[0] for c in client.calls] == list(constants.FRED_DAILY)
    assert all(start is None for _, start in client.calls)  # empty table → full history
    for i, series_id in enumerate(constants.FRED_DAILY):
        rows = writer.execute(
            "SELECT obs_date, value, typeof(value), typeof(obs_date), ingested_at_utc "
            "FROM macro WHERE series_id = ? ORDER BY obs_date",
            (series_id,),
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [
            ("2026-07-01", float(i * 10)),
            ("2026-07-02", float(i * 10 + 1)),
        ]
        for _, value, value_type, obs_type, ingested_at in rows:
            assert value_type == "real"
            assert type(value) is float
            assert obs_type == "text"
            assert UTC_RE.fullmatch(ingested_at)


# --------------------------------------------------------------------------- #
# available_date: release lag in BUSINESS days (weekend skips + weekend obs)
# --------------------------------------------------------------------------- #


def test_available_date_business_day_shift() -> None:
    # DGS10 lag 1: obs Friday → available Monday (weekend skipped).
    assert available_date("DGS10", "2026-07-03") == "2026-07-06"
    # DTWEXBGS lag 3: obs Monday → available Thursday.
    assert available_date("DTWEXBGS", "2026-06-29") == "2026-07-02"
    # DTWEXBGS lag 3: obs Friday → available Wednesday (weekend skipped).
    assert available_date("DTWEXBGS", "2026-07-03") == "2026-07-08"


def test_available_date_weekend_observations_roll_forward() -> None:
    """DFF publishes 7 days/week, so weekend obs_dates ARE a production input:
    Saturday and Sunday obs deliberately collapse to the SAME Monday anchor."""
    assert available_date("DFF", "2026-07-04") == "2026-07-06"  # Sat + 1 BDay → Mon
    assert available_date("DFF", "2026-07-05") == "2026-07-06"  # Sun + 1 BDay → same Mon


def test_available_date_rejects_unknown_series() -> None:
    with pytest.raises(ValueError, match="unknown FRED series"):
        available_date("NOPE", "2026-07-03")


def test_available_date_stored_through_ingest(writer: sqlite3.Connection) -> None:
    client = FakeFredClient(
        {
            "DGS10": _series(["2026-07-03"], [4.2]),
            "DTWEXBGS": _series(["2026-06-29", "2026-07-03"], [120.0, 121.0]),
        }
    )
    ingest_macro(writer, client, series_ids=("DGS10", "DTWEXBGS"))
    stored = dict(
        writer.execute("SELECT series_id || '/' || obs_date, available_date FROM macro").fetchall()
    )
    assert stored == {
        "DGS10/2026-07-03": "2026-07-06",
        "DTWEXBGS/2026-06-29": "2026-07-02",
        "DTWEXBGS/2026-07-03": "2026-07-08",
    }


# --------------------------------------------------------------------------- #
# '.'/'' parse → None → SQL NULL
# --------------------------------------------------------------------------- #


def test_parse_fred_value_units() -> None:
    assert parse_fred_value(".") is None
    assert parse_fred_value("") is None
    assert parse_fred_value(" . ") is None  # whitespace-tolerant
    assert parse_fred_value(float("nan")) is None
    assert parse_fred_value(np.float64("nan")) is None
    assert parse_fred_value(None) is None
    assert parse_fred_value(4.25) == 4.25
    assert parse_fred_value("4.25") == 4.25
    assert parse_fred_value(np.float64(1.5)) == 1.5
    with pytest.raises(ValueError, match="unparseable FRED value"):
        parse_fred_value("N/A")
    with pytest.raises(TypeError, match="cannot parse FRED value"):
        parse_fred_value(object())


def test_dot_and_empty_values_become_sql_null(writer: sqlite3.Connection) -> None:
    client = FakeFredClient(
        {
            "VIXCLS": _series(
                ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06"],
                [17.5, ".", "", np.nan],
            )
        }
    )
    counts = ingest_macro(writer, client, series_ids=("VIXCLS",))
    assert counts == {"VIXCLS": 4}
    rows = writer.execute(
        "SELECT obs_date, typeof(value) FROM macro WHERE series_id = 'VIXCLS' ORDER BY obs_date"
    ).fetchall()
    assert rows == [
        ("2026-07-01", "real"),
        ("2026-07-02", "null"),
        ("2026-07-03", "null"),
        ("2026-07-06", "null"),
    ]


# --------------------------------------------------------------------------- #
# Key probe: missing / invalid / valid — the explicit degraded-mode contract
# --------------------------------------------------------------------------- #


def test_stable_code_literals_are_pinned() -> None:
    """The code VALUES are an operator/pipeline contract (grepped on stderr,
    branched on by Step 8/11) — pin the literals so a rename fails CI."""
    assert MACRO_OK == "MACRO_OK"
    assert MACRO_DISABLED_NO_KEY == "MACRO_DISABLED_NO_KEY"
    assert MACRO_DISABLED_BAD_KEY == "MACRO_DISABLED_BAD_KEY"
    assert EXIT_MACRO_DISABLED == 2


def test_missing_key_probe_returns_stable_no_key_status() -> None:
    status = probe_fred_key(None)
    assert status.enabled is False
    assert status.code == MACRO_DISABLED_NO_KEY == "MACRO_DISABLED_NO_KEY"
    assert "FRED_API_KEY" in status.message


def test_missing_key_cli_exits_2_with_stable_code_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scrubbed_fred_key_env: None,
) -> None:
    """The live degraded path: no key in env, no .env findable → exit 2 + code."""
    monkeypatch.chdir(tmp_path)  # .env is cwd-relative; tmp has none
    exit_code = main(["--backfill", "--db", str(tmp_path / "abe.db")])
    assert exit_code == EXIT_MACRO_DISABLED == 2
    err = capsys.readouterr().err
    assert MACRO_DISABLED_NO_KEY in err
    assert not (tmp_path / "abe.db").exists()  # degraded run never opens the db


def test_load_fred_api_key_env_and_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scrubbed_fred_key_env: None,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_fred_api_key() is None
    # load_dotenv writes os.environ OUTSIDE monkeypatch's ledger — the
    # scrubbed_fred_key_env fixture guarantees this cannot leak past the test.
    (tmp_path / ".env").write_text(f"FRED_API_KEY={WELL_FORMED_KEY}\n", encoding="ascii")
    assert load_fred_api_key() == WELL_FORMED_KEY
    monkeypatch.setenv("FRED_API_KEY", "  ")  # blank → None, never an empty fetch
    assert load_fred_api_key() is None


def test_invalid_key_probe_maps_fredapi_auth_error_to_bad_key() -> None:
    """fredapi surfaces the FRED 400 as ValueError(<message>) — probe must map
    BOTH real rejection phrasings (unregistered / malformed) to the stable
    BAD_KEY code, not crash and not silently pass."""
    for message in (FRED_BAD_KEY_MESSAGE, FRED_MALFORMED_KEY_MESSAGE):
        status = probe_fred_key(WELL_FORMED_KEY, client=RaisingFredClient(ValueError(message)))
        assert status.enabled is False
        assert status.code == MACRO_DISABLED_BAD_KEY == "MACRO_DISABLED_BAD_KEY"
        assert "api_key" in status.message


def test_probe_non_auth_errors_propagate() -> None:
    """Crashes must stay distinguishable from degraded: network failures,
    timeouts, and non-key FRED 400s all propagate — only genuine key
    rejections may enter the degraded mode."""
    with pytest.raises(ConnectionError, match="getaddrinfo"):
        probe_fred_key(
            WELL_FORMED_KEY, client=RaisingFredClient(ConnectionError("getaddrinfo failed"))
        )
    # The bounded-socket path raises TimeoutError on a stalled connection.
    with pytest.raises(TimeoutError, match="timed out"):
        probe_fred_key(WELL_FORMED_KEY, client=RaisingFredClient(TimeoutError("timed out")))
    # A ValueError that is NOT a key rejection (e.g. bad series id) propagates.
    with pytest.raises(ValueError, match="series does not exist"):
        probe_fred_key(
            WELL_FORMED_KEY,
            client=RaisingFredClient(ValueError("Bad Request. The series does not exist.")),
        )


def test_valid_key_probe_is_cheap_and_returns_ok() -> None:
    client = FakeFredClient({"DGS10": _series(["2026-07-02", "2026-07-06"], [4.2, 4.3])})
    status = probe_fred_key(WELL_FORMED_KEY, client=client)
    assert status.enabled is True
    assert status.code == MACRO_OK == "MACRO_OK"
    (series_id, start) = client.calls[0]
    assert series_id == "DGS10"
    assert start is not None  # a small recent window, never full history


# --------------------------------------------------------------------------- #
# Incremental fetch + idempotency + revision upsert (append-only semantics)
# --------------------------------------------------------------------------- #


def test_incremental_fetch_asks_only_after_stored_max(writer: sqlite3.Connection) -> None:
    client = FakeFredClient(
        {"DGS10": _series(["2026-06-29", "2026-06-30", "2026-07-01"], [1.0, 2.0, 3.0])}
    )
    counts = ingest_macro(writer, client, series_ids=("DGS10",))
    assert counts == {"DGS10": 3}
    assert client.calls == [("DGS10", None)]
    assert stored_max_obs_date(writer, "DGS10") == "2026-07-01"

    # Two new obs appear upstream: the second run must ask only for obs > MAX
    # (observation_start is FRED-inclusive, so max + 1 day).
    client.series["DGS10"] = _series(
        ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    counts = ingest_macro(writer, client, series_ids=("DGS10",))
    assert client.calls[1] == ("DGS10", "2026-07-02")
    assert counts == {"DGS10": 2}

    # Nothing new → the fetch window is empty and the run is a no-op.
    counts = ingest_macro(writer, client, series_ids=("DGS10",))
    assert client.calls[2] == ("DGS10", "2026-07-04")
    assert counts == {"DGS10": 0}
    assert writer.execute("SELECT COUNT(*) FROM macro").fetchone()[0] == 5


def test_reingest_upserts_revisions_never_duplicates_or_deletes(
    writer: sqlite3.Connection,
) -> None:
    """A manual full refetch (client ignoring observation_start) re-ingests
    existing obs: PK upsert → no duplicate rows, no deletes; a FRED revision on
    an existing obs lands as an updated value (the documented manual-refetch
    path — production incremental runs never re-fetch stored obs)."""
    client = SloppyFullHistoryClient(_series(["2026-07-01", "2026-07-02"], [4.2, 4.3]))
    ingest_macro(writer, client, series_ids=("DGS10",))

    # FRED revised the 07-02 value; the refetch simply upserts it.
    client.series = _series(["2026-07-01", "2026-07-02"], [4.2, 4.35])
    ingest_macro(writer, client, series_ids=("DGS10",))

    rows = writer.execute(
        "SELECT obs_date, value FROM macro WHERE series_id = 'DGS10' ORDER BY obs_date"
    ).fetchall()
    assert rows == [("2026-07-01", 4.2), ("2026-07-02", 4.35)]  # updated, not duplicated


def test_ingest_rejects_non_iso_obs_dates_at_write_boundary(
    writer: sqlite3.Connection,
) -> None:
    bad = pd.Series([4.2], index=pd.Index(["07/01/2026"]))
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        ingest_macro(writer, FakeFredClient({"DGS10": bad}), series_ids=("DGS10",))


# --------------------------------------------------------------------------- #
# Identity: the default series set IS the constants object (no re-duplication)
# --------------------------------------------------------------------------- #


def test_ingest_default_series_is_constants_fred_daily() -> None:
    """`is`-identity regression (test_storage idiom): the default must reference
    THE constants object, so any future re-duplication fails CI."""
    kwdefaults = ingest_macro.__kwdefaults__
    assert kwdefaults is not None
    assert kwdefaults["series_ids"] is constants.FRED_DAILY


# --------------------------------------------------------------------------- #
# CLI through the production entry point (fake fredapi module in sys.modules)
# --------------------------------------------------------------------------- #


class _FakeFredApi:
    """A sys.modules['fredapi'] stand-in recording Fred(...).get_series calls
    and the socket default timeout active DURING each call (fredapi 0.5.2's
    urlopen inherits exactly that)."""

    def __init__(self, accepted_key: str) -> None:
        self.calls: list[tuple[str | None, str, str | None]] = []
        self.timeouts: list[float | None] = []
        self.series: dict[str, pd.Series] = {
            series_id: _series(["2026-07-01", "2026-07-02"], [float(i), float(i) + 0.5])
            for i, series_id in enumerate(constants.FRED_DAILY)
        }
        outer = self
        module = types.ModuleType("fredapi")

        class Fred:
            def __init__(self, api_key: str | None = None) -> None:
                self.api_key = api_key

            def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
                outer.calls.append((self.api_key, series_id, observation_start))
                outer.timeouts.append(socket.getdefaulttimeout())
                if self.api_key != accepted_key:
                    raise ValueError(FRED_BAD_KEY_MESSAGE)
                data = outer.series[series_id]
                if observation_start is None:
                    return data
                return data.loc[data.index >= pd.Timestamp(observation_start)]

        module.Fred = Fred  # type: ignore[attr-defined]
        self.module = module


@pytest.fixture()
def fake_fredapi(monkeypatch: pytest.MonkeyPatch) -> _FakeFredApi:
    fake = _FakeFredApi(accepted_key=WELL_FORMED_KEY)
    monkeypatch.setitem(sys.modules, "fredapi", fake.module)
    return fake


def test_fred_client_bounds_socket_timeout_and_restores(fake_fredapi: _FakeFredApi) -> None:
    """fredapi 0.5.2 calls urlopen() with NO timeout, so FredApiClient must
    bound the socket for the call's duration (a stalled connection then raises
    instead of hanging the startup probe / daily fetch forever) and restore the
    process default afterwards — on success AND on failure."""
    before = socket.getdefaulttimeout()
    try:
        FredApiClient(WELL_FORMED_KEY).get_series("DGS10")
        assert fake_fredapi.timeouts == [FRED_TIMEOUT_SECONDS]  # bound active during urlopen
        assert socket.getdefaulttimeout() == before  # restored on success
        with pytest.raises(ValueError, match="api_key is not registered"):
            FredApiClient("wrong-key").get_series("DGS10")
        assert socket.getdefaulttimeout() == before  # restored on failure too
    finally:
        socket.setdefaulttimeout(before)  # defensive: never leak into other tests


def test_cli_main_backfills_through_production_entry_point(
    fake_fredapi: _FakeFredApi,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m abe.ingest.macro --backfill --db <path>` end-to-end: env key →
    probe (cheap real request through the seam) → 6-series ingest → summary."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FRED_API_KEY", WELL_FORMED_KEY)
    db_path = tmp_path / "abe.db"
    exit_code = main(["--backfill", "--db", str(db_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    for series_id in constants.FRED_DAILY:
        assert f"{series_id}: +2 rows (total 2, 2026-07-01 .. 2026-07-02)" in out
    # First call is the startup probe: keyed, DGS10, small recent window.
    probe_key, probe_series, probe_start = fake_fredapi.calls[0]
    assert probe_key == WELL_FORMED_KEY
    assert probe_series == "DGS10"
    assert probe_start is not None
    conn = storage.open_read_only(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM macro").fetchone()[0]
    finally:
        conn.close()
    assert total == 2 * len(constants.FRED_DAILY)


def test_cli_bad_key_exits_2_with_stable_code(
    fake_fredapi: _FakeFredApi,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FRED_API_KEY", "not-a-registered-key")
    exit_code = main(["--backfill", "--db", str(tmp_path / "abe.db")])
    assert exit_code == EXIT_MACRO_DISABLED
    err = capsys.readouterr().err
    assert MACRO_DISABLED_BAD_KEY in err
    assert not (tmp_path / "abe.db").exists()


def test_cli_keyed_run_with_zero_rows_exits_1(
    fake_fredapi: _FakeFredApi,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1 (failed run) stays distinct from exit 2 (degraded): key accepted
    but nothing stored is a failure, mirroring prices."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FRED_API_KEY", WELL_FORMED_KEY)
    for series_id in constants.FRED_DAILY:
        fake_fredapi.series[series_id] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    exit_code = main(["--backfill", "--db", str(tmp_path / "abe.db")])
    assert exit_code == 1
    assert "no rows stored" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Real-network keyed test (deselected by default; self-skips without a key)
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_real_fred_small_window_stores_all_six_series(tmp_path: Path) -> None:
    api_key = load_fred_api_key()
    if api_key is None:
        pytest.skip("FRED_API_KEY not set (env or .env): cannot run the real-FRED keyed test")
    status = probe_fred_key(api_key)
    assert status.enabled, f"probe failed: {status.code}: {status.message}"

    conn = storage.open_writer(tmp_path / "abe.db")
    try:
        # Seed a sentinel obs per series so the incremental path fetches only a
        # small recent window instead of full history (keeps the test cheap).
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        for series_id in constants.FRED_DAILY:
            storage.upsert_row(
                conn,
                "macro",
                {
                    "series_id": series_id,
                    "obs_date": cutoff,
                    "value": None,
                    "available_date": available_date(series_id, cutoff),
                    "ingested_at_utc": "2026-01-01T00:00:00Z",
                },
            )
        counts = ingest_macro(conn, FredApiClient(api_key))
        for series_id in constants.FRED_DAILY:
            assert counts[series_id] > 0, f"no new observations stored for {series_id}"
            non_null = conn.execute(
                "SELECT COUNT(*) FROM macro WHERE series_id = ? AND value IS NOT NULL",
                (series_id,),
            ).fetchone()[0]
            assert non_null > 0, f"all stored values NULL for {series_id}"
            shifted_ok = conn.execute(
                "SELECT COUNT(*) FROM macro WHERE series_id = ? AND available_date <= obs_date",
                (series_id,),
            ).fetchone()[0]
            assert shifted_ok == 0  # every available_date strictly after its obs_date
    finally:
        conn.close()
