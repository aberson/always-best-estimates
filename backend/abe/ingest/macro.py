"""FRED macro ingest for the fixed daily set, through storage (append-only).

For each series in ``constants.FRED_DAILY`` (DGS10, T10Y2Y, VIXCLS, DFF,
BAMLH0A0HYM2, DTWEXBGS) rows land in the ``macro`` table as
``(series_id, obs_date, value, available_date, ingested_at_utc)``:

- ``available_date = obs_date + FRED_RELEASE_LAG[series_id]`` in **business
  days** (pandas ``BDay``): an obs on Friday with lag 1 becomes available
  Monday; weekend observations (DFF publishes 7 days/week) roll forward past
  the weekend the same way. The declared lag unit is business days — that is
  the ``constants.FRED_RELEASE_LAG`` contract. Caveat: ``BDay`` carries no
  holiday calendar, so an obs whose lag window spans a US federal holiday gets
  an ``available_date`` ~1 real day early a couple of times a year — an
  accepted V1 simplification.
- Append-only semantics: every write is a ``storage.upsert_row`` on the
  ``(series_id, obs_date)`` primary key — re-ingesting an existing obs is a
  no-op update; rows are never deleted.
- FRED revisions: in production, stored observations are NEVER re-fetched —
  the incremental fetch starts strictly after stored ``MAX(obs_date)``, so a
  value first published as ``'.'`` and later filled in (or revised) keeps its
  first-print (possibly NULL) stored value. First-print storage IS the
  deliberate V1 point-in-time semantics (ALFRED vintages are a V2 concern).
  The upsert-by-PK write path only re-touches an existing obs on a MANUAL full
  refetch, where a revised value gracefully upserts — no duplicates, no deletes.
- Incremental fetch starts strictly after stored ``MAX(obs_date)`` per series
  (``observation_start`` is inclusive in FRED semantics, so max + 1 day). No
  overlap window is re-fetched — unlike prices, FRED daily series don't rebase
  whole histories the way backward-adjusted closes do.

Missing values: FRED encodes them as ``'.'`` (fredapi maps those to NaN before
we see them) and occasionally ``''`` in raw payloads. Both — plus NaN floats —
become explicit ``None`` at the write boundary (:func:`parse_fred_value`);
storage's coercion boundary rejects NaN by contract.

Key handling (the app boundary): :func:`load_fred_api_key` loads the project
``.env`` (cwd-relative, same convention as ``storage.DEFAULT_DB_PATH``) via
python-dotenv, then reads ``FRED_API_KEY`` from the environment — no key means
``None``, never an empty fetch. :func:`probe_fred_key` turns that into an
explicit :class:`MacroStatus` with a STABLE code (never silent-empty):

- key present + accepted → ``MACRO_OK`` (probed with one cheap real request)
- key missing → ``MACRO_DISABLED_NO_KEY`` (degraded mode, macro disabled)
- key present but rejected → ``MACRO_DISABLED_BAD_KEY``

Step 8's pipeline and Step 11's degraded-mode card consume ``MacroStatus``.

``fredapi`` is imported lazily inside :meth:`FredApiClient.get_series` only
(mirror of sources.py's lazy yfinance import) — the offline test suite uses
fake :class:`FredClient` implementations and never touches the import.

Network bound: fredapi 0.5.2 calls ``urlopen(url)`` with NO timeout, so a
stalled connection would hang the startup probe (and the daily fetch job)
forever with zero output. :class:`FredApiClient` bounds every request with
``socket.setdefaulttimeout(FRED_TIMEOUT_SECONDS)`` (saved/restored around the
call); a timeout raises and PROPAGATES — a crash, deliberately NOT the
degraded mode, so hangs and key problems stay distinguishable.

CLI (one-time backfill, or the daily incremental fetch job)::

    uv run python -m abe.ingest.macro --backfill [--db <path>]

Exit codes: 0 = keyed run stored rows for every series; 1 = a keyed run left a
series with zero stored rows (failed run, mirrors prices); 2 =
:data:`EXIT_MACRO_DISABLED` — no/rejected key, the explicit degraded mode
(distinct from 1 so the operator can tell degraded from crashed; the stable
code is printed to stderr).
"""

import argparse
import logging
import math
import os
import socket
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Protocol

import dotenv
import pandas as pd
from pandas.tseries.offsets import BDay

from abe import constants, storage
from abe.ingest.sources import DATE_FORMAT, DATE_KEY_RE, utc_now_iso

__all__ = [
    "EXIT_MACRO_DISABLED",
    "FRED_TIMEOUT_SECONDS",
    "MACRO_DISABLED_BAD_KEY",
    "MACRO_DISABLED_NO_KEY",
    "MACRO_OK",
    "FredApiClient",
    "FredClient",
    "MacroStatus",
    "available_date",
    "ingest_macro",
    "load_fred_api_key",
    "main",
    "parse_fred_value",
    "probe_fred_key",
    "stored_max_obs_date",
]

MACRO_OK: Final[str] = "MACRO_OK"
"""Stable status code: key present and accepted by FRED."""

MACRO_DISABLED_NO_KEY: Final[str] = "MACRO_DISABLED_NO_KEY"
"""Stable status code: no FRED_API_KEY configured — explicit degraded mode."""

MACRO_DISABLED_BAD_KEY: Final[str] = "MACRO_DISABLED_BAD_KEY"
"""Stable status code: FRED rejected the configured key — explicit degraded mode."""

EXIT_MACRO_DISABLED: Final[int] = 2
"""CLI exit code for the key-degraded modes — distinct from 1 (keyed run that
stored nothing) so the operator can tell degraded from crashed."""

FRED_TIMEOUT_SECONDS: Final[float] = 15.0
"""Socket timeout bounding every fredapi request (fredapi 0.5.2 passes NO
timeout to urlopen) — a stalled connection raises instead of hanging forever."""

_ENV_VAR: Final[str] = "FRED_API_KEY"
_ENV_FILE: Final[Path] = Path(".env")
"""Project .env, relative to the process cwd (the project root by convention —
the same convention as ``storage.DEFAULT_DB_PATH``)."""

_PROBE_SERIES_ID: Final[str] = "DGS10"
_PROBE_WINDOW_DAYS: Final[int] = 14


@dataclass(frozen=True)
class MacroStatus:
    """Startup macro-ingest status (stable ``code`` + human ``message``).

    ``enabled=False`` is the explicit macro-disabled degraded mode; consumers
    (Step 8 pipeline, Step 11 degraded-mode card) branch on ``enabled`` and
    surface ``code``/``message`` verbatim — the codes are a stable contract.
    """

    enabled: bool
    code: str
    message: str


class FredClient(Protocol):
    """Thin seam over fredapi's ``Fred.get_series`` so tests can use fakes.

    Returns a pandas Series indexed by observation date (datetime-like, or an
    ISO-8601 ``YYYY-MM-DD`` string) whose values may be floats, NaN, or raw
    FRED strings (``'.'``/``''``) — the ingest layer parses them at the write
    boundary. ``observation_start`` is an inclusive ISO-8601 lower bound (FRED
    semantics); ``None`` means full available history.
    """

    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series: ...


class FredApiClient:
    """Real ``FredClient``: fredapi's ``Fred.get_series`` behind the thin seam.

    ``fredapi`` is imported lazily inside :meth:`get_series` only — fake-client
    test paths must work with zero network and zero fredapi import (mirror of
    sources.py's lazy yfinance pattern).

    Every request is bounded by :data:`FRED_TIMEOUT_SECONDS` via
    ``socket.setdefaulttimeout`` (saved/restored in ``try``/``finally``) because
    fredapi 0.5.2 calls ``urlopen(url)`` with no timeout — without the bound a
    stalled connection hangs the startup probe and the daily fetch job forever.
    The default timeout is process-global, which is acceptable in this
    single-worker app; a timeout raises (``TimeoutError``/``URLError``) and
    propagates as a crash, deliberately NOT the degraded mode.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
        from fredapi import Fred  # lazy: fake-client paths never touch this import

        fred = Fred(api_key=self._api_key)
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(FRED_TIMEOUT_SECONDS)
        try:
            if observation_start is None:
                return fred.get_series(series_id)
            return fred.get_series(series_id, observation_start=observation_start)
        finally:
            socket.setdefaulttimeout(previous_timeout)


def load_fred_api_key() -> str | None:
    """``FRED_API_KEY`` from the environment, after loading the project ``.env``.

    python-dotenv loads :data:`_ENV_FILE` (cwd-relative; a missing file is a
    silent no-op) without overriding real environment variables (its default).
    A missing or blank key returns ``None`` — callers turn that into the
    explicit ``MACRO_DISABLED_NO_KEY`` degraded mode via :func:`probe_fred_key`,
    never a silent empty fetch.
    """
    dotenv.load_dotenv(_ENV_FILE)
    key = os.environ.get(_ENV_VAR, "").strip()
    return key or None


def _is_auth_error(exc: BaseException) -> bool:
    """True when ``exc`` is FRED rejecting the API key, and only that.

    fredapi surfaces HTTP errors as ``ValueError(<FRED XML 'message'>)``
    (fred.py ``__fetch_data``), so a rejection must be (a) a ``ValueError``
    AND (b) carry FRED's actual key-rejection phrasing: ``"Bad Request. The
    value for variable api_key is not registered."`` or ``"...api_key is not a
    32 character alpha-numeric lower-case string."``. Requiring both the
    ``api_key`` token and a rejection phrase keeps network outages and other
    FRED 400s (e.g. a bad series id) OUT of the degraded mode — they propagate
    as crashes. Duck-typed on the message (like prices._is_rate_limit_error)
    so this module never imports fredapi.
    """
    if not isinstance(exc, ValueError):
        return False
    text = str(exc).lower()
    if "api_key" not in text and "api key" not in text:
        return False
    return "not registered" in text or "not a 32 character" in text


def probe_fred_key(api_key: str | None, client: FredClient | None = None) -> MacroStatus:
    """Startup probe: distinguish valid / missing / rejected FRED keys explicitly.

    - ``api_key is None`` → ``MACRO_DISABLED_NO_KEY`` (no request made).
    - Key present → one cheap real request (a :data:`_PROBE_WINDOW_DAYS`-day
      window of :data:`_PROBE_SERIES_ID`); an auth-shaped failure →
      ``MACRO_DISABLED_BAD_KEY``.
    - Any NON-auth failure (network down, FRED outage) propagates — a crashed
      probe must stay distinguishable from the explicit degraded modes.

    ``client`` is injectable for tests; ``None`` builds the real
    :class:`FredApiClient` from ``api_key``.
    """
    if api_key is None:
        return MacroStatus(
            enabled=False,
            code=MACRO_DISABLED_NO_KEY,
            message=(
                "FRED_API_KEY is not set (environment or .env) — macro ingest disabled; "
                "running in explicit macro-disabled degraded mode"
            ),
        )
    probe_client: FredClient = client if client is not None else FredApiClient(api_key)
    start = (date.today() - timedelta(days=_PROBE_WINDOW_DAYS)).isoformat()
    try:
        probe_client.get_series(_PROBE_SERIES_ID, observation_start=start)
    except Exception as exc:
        if _is_auth_error(exc):
            return MacroStatus(
                enabled=False,
                code=MACRO_DISABLED_BAD_KEY,
                message=(
                    f"FRED rejected the configured FRED_API_KEY: {exc} — macro ingest "
                    "disabled; running in explicit macro-disabled degraded mode"
                ),
            )
        raise
    return MacroStatus(enabled=True, code=MACRO_OK, message="FRED API key accepted")


def parse_fred_value(value: object) -> float | None:
    """Parse one raw FRED observation value; missing becomes explicit ``None``.

    FRED's missing-value encodings ``'.'`` and ``''`` (whitespace tolerated)
    and NaN floats (fredapi pre-converts ``'.'`` → NaN) all return ``None`` —
    storage's insert boundary rejects NaN by contract, so the conversion is
    deliberate and happens HERE. Numeric strings parse to float; anything else
    raises loudly rather than storing garbage.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in ("", "."):
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"unparseable FRED value {value!r}") from exc
    if isinstance(value, int | float):
        number = float(value)
        return None if math.isnan(number) else number
    raise TypeError(f"cannot parse FRED value of type {type(value).__qualname__}: {value!r}")


def available_date(series_id: str, obs_date: str) -> str:
    """``obs_date`` + the series' declared release lag in BUSINESS days.

    ``constants.FRED_RELEASE_LAG`` counts business days (pandas ``BDay``): a
    Friday obs with lag 1 is available Monday; a Monday obs with lag 3 is
    available Thursday; weekend observations (DFF publishes 7 days/week) roll
    forward, so Saturday and Sunday obs collapse to the same next-Monday anchor.
    Caveat: ``BDay`` carries no holiday calendar — a lag window spanning a US
    federal holiday lands ~1 real day early a couple of times a year (accepted
    V1 simplification; see the module docstring).
    """
    if series_id not in constants.FRED_RELEASE_LAG:
        raise ValueError(
            f"unknown FRED series {series_id!r}; known: {sorted(constants.FRED_RELEASE_LAG)}"
        )
    lag = constants.FRED_RELEASE_LAG[series_id]
    shifted = pd.Timestamp(obs_date) + BDay(lag)
    return str(shifted.strftime(DATE_FORMAT))


def stored_max_obs_date(conn: sqlite3.Connection, series_id: str) -> str | None:
    """Latest stored obs_date for ``series_id``, or ``None`` when nothing is stored.

    ``None`` tells the caller to run a full-history fetch instead of an
    incremental one.
    """
    row = conn.execute(
        "SELECT MAX(obs_date) FROM macro WHERE series_id = ?", (series_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _iso_obs_date(series_id: str, key: object) -> str:
    """Normalize a fetched observation index entry to the ISO date-key shape."""
    text = key if isinstance(key, str) else str(pd.Timestamp(key).strftime(DATE_FORMAT))
    if not DATE_KEY_RE.fullmatch(text):
        raise ValueError(
            f"obs date {key!r} for {series_id!r} is not ISO-8601 YYYY-MM-DD "
            "(plan section 3 date-key rule)"
        )
    return text


def ingest_macro(
    conn: sqlite3.Connection,
    client: FredClient,
    *,
    series_ids: Sequence[str] = constants.FRED_DAILY,
) -> dict[str, int]:
    """Ingest FRED daily observations for ``series_ids``; return upserted rows per series.

    Empty table for a series → full-history fetch (``observation_start=None``);
    otherwise fetch only observations strictly after stored ``MAX(obs_date)``
    (FRED's ``observation_start`` is inclusive, so max + 1 calendar day). All
    writes are PK upserts (append-only by date — see the module docstring for
    the revision contract). Fetched frames are sorted by obs date before
    writing so a crash mid-series leaves ``MAX(obs_date)`` over a contiguous
    prefix (resume-after-crash correctness, mirroring prices).
    """
    counts: dict[str, int] = {}
    ingested_at = utc_now_iso()
    for series_id in series_ids:
        max_obs = stored_max_obs_date(conn, series_id)
        start = (
            None
            if max_obs is None
            else (date.fromisoformat(max_obs) + timedelta(days=1)).isoformat()
        )
        fetched = client.get_series(series_id, observation_start=start).sort_index()
        upserted = 0
        for key, raw in fetched.items():
            obs_date = _iso_obs_date(series_id, key)
            storage.upsert_row(
                conn,
                "macro",
                {
                    "series_id": series_id,
                    "obs_date": obs_date,
                    "value": parse_fred_value(raw),
                    "available_date": available_date(series_id, obs_date),
                    "ingested_at_utc": ingested_at,
                },
            )
            upserted += 1
        counts[series_id] = upserted
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m abe.ingest.macro --backfill [--db <path>]``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m abe.ingest.macro",
        description=(
            "Incremental FRED macro ingest for the daily set "
            f"{constants.FRED_DAILY}: fetches only observations after stored "
            "MAX(obs_date) per series; on an empty table this is the full-history "
            "backfill. Requires FRED_API_KEY (environment or .env); a missing or "
            f"rejected key exits {EXIT_MACRO_DISABLED} (degraded, not crashed)."
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="explicit first-run intent (full history lands whenever a series is empty)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=storage.DEFAULT_DB_PATH,
        help=f"SQLite db path (default: {storage.DEFAULT_DB_PATH})",
    )
    args = parser.parse_args(argv)
    api_key = load_fred_api_key()
    status = probe_fred_key(api_key)
    if api_key is None or not status.enabled:
        print(f"{status.code}: {status.message}", file=sys.stderr)
        return EXIT_MACRO_DISABLED
    conn = storage.open_writer(args.db)
    try:
        counts = ingest_macro(conn, FredApiClient(api_key))
        empty_series: list[str] = []
        for series_id, written in counts.items():
            total, first, last = conn.execute(
                "SELECT COUNT(*), MIN(obs_date), MAX(obs_date) FROM macro WHERE series_id = ?",
                (series_id,),
            ).fetchone()
            print(f"{series_id}: +{written} rows (total {total}, {first} .. {last})")
            if int(total) == 0:
                empty_series.append(series_id)
        if empty_series:
            print(
                f"ERROR: no rows stored for {empty_series} after a keyed run — "
                "treating as a failed run",
                file=sys.stderr,
            )
            return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
