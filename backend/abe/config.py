"""Config + ViewScenario domain entities and storage CRUD (Track 2 Step 17).

A :class:`Config` is one recipe — a named implementation per pipeline stage
(feature-builder / forecaster / view-source / optimizer, keyed by the Step 18
registries) plus per-stage param overrides. Exactly one Config is ``is_central``
(the "portfolio you'd actually buy"; the 5-minute loop runs only it). A
:class:`ViewScenario` is a named Black-Litterman view set with a ``kind``:
``forecast`` (views derived from the run's forecaster — today's behavior),
``historical`` (views from a past window's realized returns), or
``counterfactual`` (hand-authored absolute views). A Config's blend stage
references one ViewScenario.

The schema (``configs`` / ``view_scenarios`` tables + the ``runs.config_id`` FK)
and the seeded central Config land in the v1->v2 migration (:mod:`abe.migrations`
Step 16). This module is the typed access layer over that schema: entities +
CRUD through the storage coercion boundary, plus the guarded ``set_central``
transition. All writes take THE writer connection (one-writer discipline); reads
work on any connection.

JSON columns
============
``configs.params_json`` and ``view_scenarios.payload_json`` hold JSON dicts. The
entities carry parsed ``dict`` fields (``params`` / ``payload``); the row mappers
serialize on write and tolerate a NULL column on read (the seed row stores NULL
params) by treating it as ``{}``.
"""

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from abe import storage
from abe.ingest.sources import utc_now_iso

__all__ = [
    "VIEW_SCENARIO_KINDS",
    "Config",
    "ViewScenario",
    "create_config",
    "create_view_scenario",
    "delete_config",
    "delete_view_scenario",
    "get_central_config",
    "get_config",
    "get_view_scenario",
    "list_configs",
    "list_view_scenarios",
    "set_central",
    "update_config",
    "update_view_scenario",
]

VIEW_SCENARIO_KINDS: Final[frozenset[str]] = frozenset(
    {"forecast", "historical", "counterfactual"}
)
"""The valid ``view_scenarios.kind`` values (mirrors the table CHECK constraint)."""

_CONFIG_COLUMNS: Final[str] = (
    "config_id, name, feature_set, forecaster, view_scenario_id, optimizer, "
    "params_json, is_central, created_at_utc"
)
_VIEW_SCENARIO_COLUMNS: Final[str] = "view_scenario_id, name, kind, payload_json, created_at_utc"


@dataclass(frozen=True, slots=True)
class ViewScenario:
    """A named Black-Litterman view set. ``payload`` shape depends on ``kind``:
    ``forecast`` -> ``{}``; ``historical`` -> ``{"window_start", "window_end"}``;
    ``counterfactual`` -> ``{asset: {"mu", "confidence"}}`` (plan §5)."""

    view_scenario_id: int
    name: str
    kind: str
    payload: dict[str, object] = field(default_factory=dict)
    created_at_utc: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    """One pipeline recipe. ``feature_set`` / ``forecaster`` / ``optimizer`` are
    Step 18 registry keys; ``view_scenario_id`` references a :class:`ViewScenario`;
    ``params`` holds per-stage overrides (e.g. ``{"optimizer": {"min_weight": 0.05}}``)."""

    config_id: int
    name: str
    feature_set: str
    forecaster: str
    view_scenario_id: int
    optimizer: str
    params: dict[str, object] = field(default_factory=dict)
    is_central: bool = False
    created_at_utc: str | None = None


def _loads(text: str | None) -> dict[str, object]:
    """Parse a JSON-object column, tolerating NULL (the seed's params_json)."""
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _dumps(payload: Mapping[str, object]) -> str:
    """Serialize a params/payload dict for a JSON column.

    Routes any leaf value json can't natively handle through the SAME coercion
    the insert boundary uses (``storage.coerce_scalar``), so a payload built
    programmatically from numpy/torch scalars (Step 22's historical/counterfactual
    providers) survives the write instead of raising in ``json.dumps``. Genuinely
    non-scalar leaves still raise loudly (correct — a NaN or an array in a view is
    a bug)."""
    return json.dumps(dict(payload), default=storage.coerce_scalar)


def _view_scenario_from_row(row: Sequence[Any]) -> ViewScenario:
    return ViewScenario(
        view_scenario_id=int(row[0]),
        name=str(row[1]),
        kind=str(row[2]),
        payload=_loads(row[3]),
        created_at_utc=None if row[4] is None else str(row[4]),
    )


def _config_from_row(row: Sequence[Any]) -> Config:
    return Config(
        config_id=int(row[0]),
        name=str(row[1]),
        feature_set=str(row[2]),
        forecaster=str(row[3]),
        view_scenario_id=int(row[4]),
        optimizer=str(row[5]),
        params=_loads(row[6]),
        is_central=bool(row[7]),
        created_at_utc=None if row[8] is None else str(row[8]),
    )


# --------------------------------------------------------------------------- #
# ViewScenario CRUD
# --------------------------------------------------------------------------- #


def create_view_scenario(
    conn: sqlite3.Connection,
    *,
    name: str,
    kind: str,
    payload: Mapping[str, object] | None = None,
) -> ViewScenario:
    """Insert a view scenario; return the persisted entity (with id + timestamp)."""
    if kind not in VIEW_SCENARIO_KINDS:
        raise ValueError(f"kind must be one of {sorted(VIEW_SCENARIO_KINDS)}, got {kind!r}")
    created = utc_now_iso()
    view_scenario_id = storage.insert_row(
        conn,
        "view_scenarios",
        {
            "name": name,
            "kind": kind,
            "payload_json": _dumps(payload or {}),
            "created_at_utc": created,
        },
    )
    if view_scenario_id is None:  # pragma: no cover — INSERT always yields a rowid
        raise RuntimeError("view_scenario insert returned no id")
    return ViewScenario(
        view_scenario_id=int(view_scenario_id),
        name=name,
        kind=kind,
        payload=dict(payload or {}),
        created_at_utc=created,
    )


def get_view_scenario(conn: sqlite3.Connection, view_scenario_id: int) -> ViewScenario | None:
    row = conn.execute(
        f"SELECT {_VIEW_SCENARIO_COLUMNS} FROM view_scenarios WHERE view_scenario_id = ?",
        (view_scenario_id,),
    ).fetchone()
    return None if row is None else _view_scenario_from_row(row)


def list_view_scenarios(conn: sqlite3.Connection) -> list[ViewScenario]:
    rows = conn.execute(
        f"SELECT {_VIEW_SCENARIO_COLUMNS} FROM view_scenarios ORDER BY view_scenario_id"
    ).fetchall()
    return [_view_scenario_from_row(row) for row in rows]


def update_view_scenario(
    conn: sqlite3.Connection,
    view_scenario_id: int,
    *,
    name: str | None = None,
    payload: Mapping[str, object] | None = None,
) -> ViewScenario:
    """Update a view scenario's ``name`` and/or ``payload`` (``kind`` is immutable —
    it changes the payload contract, so author a new scenario instead)."""
    existing = get_view_scenario(conn, view_scenario_id)
    if existing is None:
        raise ValueError(f"no view_scenario with id {view_scenario_id}")
    new_name = existing.name if name is None else name
    new_payload = existing.payload if payload is None else dict(payload)
    conn.execute(
        "UPDATE view_scenarios SET name = ?, payload_json = ? WHERE view_scenario_id = ?",
        (new_name, _dumps(new_payload), view_scenario_id),
    )
    return ViewScenario(
        view_scenario_id=view_scenario_id,
        name=new_name,
        kind=existing.kind,
        payload=new_payload,
        created_at_utc=existing.created_at_utc,
    )


def delete_view_scenario(conn: sqlite3.Connection, view_scenario_id: int) -> None:
    """Delete a view scenario. Refuses if any Config still references it (a clear
    message ahead of the raw FK violation)."""
    referencing = conn.execute(
        "SELECT COUNT(*) FROM configs WHERE view_scenario_id = ?", (view_scenario_id,)
    ).fetchone()[0]
    if referencing:
        raise ValueError(
            f"view_scenario {view_scenario_id} is referenced by {referencing} config(s); "
            "reassign or delete them first"
        )
    conn.execute("DELETE FROM view_scenarios WHERE view_scenario_id = ?", (view_scenario_id,))


# --------------------------------------------------------------------------- #
# Config CRUD
# --------------------------------------------------------------------------- #


def create_config(
    conn: sqlite3.Connection,
    *,
    name: str,
    feature_set: str,
    forecaster: str,
    view_scenario_id: int,
    optimizer: str,
    params: Mapping[str, object] | None = None,
) -> Config:
    """Insert a NON-central config; return the persisted entity.

    New configs are always ``is_central = 0`` — promotion is the deliberate
    :func:`set_central` action (plan §6). ``name`` must be unique (DB constraint);
    ``view_scenario_id`` must exist (FK). The stage keys (``feature_set`` /
    ``forecaster`` / ``optimizer``) are NOT validated against the Step 18 registry
    here — an unknown key surfaces loudly at resolve time (``registry.resolve``),
    not silently at run time."""
    created = utc_now_iso()
    config_id = storage.insert_row(
        conn,
        "configs",
        {
            "name": name,
            "feature_set": feature_set,
            "forecaster": forecaster,
            "view_scenario_id": view_scenario_id,
            "optimizer": optimizer,
            "params_json": _dumps(params or {}),
            "is_central": 0,
            "created_at_utc": created,
        },
    )
    if config_id is None:  # pragma: no cover — INSERT always yields a rowid
        raise RuntimeError("config insert returned no id")
    return Config(
        config_id=int(config_id),
        name=name,
        feature_set=feature_set,
        forecaster=forecaster,
        view_scenario_id=view_scenario_id,
        optimizer=optimizer,
        params=dict(params or {}),
        is_central=False,
        created_at_utc=created,
    )


def get_config(conn: sqlite3.Connection, config_id: int) -> Config | None:
    row = conn.execute(
        f"SELECT {_CONFIG_COLUMNS} FROM configs WHERE config_id = ?", (config_id,)
    ).fetchone()
    return None if row is None else _config_from_row(row)


def list_configs(conn: sqlite3.Connection) -> list[Config]:
    rows = conn.execute(f"SELECT {_CONFIG_COLUMNS} FROM configs ORDER BY config_id").fetchall()
    return [_config_from_row(row) for row in rows]


def get_central_config(conn: sqlite3.Connection) -> Config:
    """The single central Config. Raises if the invariant is violated (there
    should always be exactly one after a migrate)."""
    row = conn.execute(
        f"SELECT {_CONFIG_COLUMNS} FROM configs WHERE is_central = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("no central config exists — the schema invariant is violated")
    return _config_from_row(row)


def update_config(
    conn: sqlite3.Connection,
    config_id: int,
    *,
    name: str | None = None,
    feature_set: str | None = None,
    forecaster: str | None = None,
    view_scenario_id: int | None = None,
    optimizer: str | None = None,
    params: Mapping[str, object] | None = None,
) -> Config:
    """Update a config's recipe fields. ``is_central`` is NOT mutable here — use
    :func:`set_central` (the guarded transition)."""
    existing = get_config(conn, config_id)
    if existing is None:
        raise ValueError(f"no config with id {config_id}")
    updates: dict[str, object] = {}
    if name is not None:
        updates["name"] = name
    if feature_set is not None:
        updates["feature_set"] = feature_set
    if forecaster is not None:
        updates["forecaster"] = forecaster
    if view_scenario_id is not None:
        updates["view_scenario_id"] = view_scenario_id
    if optimizer is not None:
        updates["optimizer"] = optimizer
    if params is not None:
        updates["params_json"] = _dumps(params)
    if updates:
        assignments = ", ".join(f'"{col}" = ?' for col in updates)
        values = [storage.coerce_scalar(value) for value in updates.values()]
        conn.execute(
            f"UPDATE configs SET {assignments} WHERE config_id = ?", (*values, config_id)
        )
    refreshed = get_config(conn, config_id)
    if refreshed is None:  # pragma: no cover — row exists (checked above)
        raise RuntimeError(f"config {config_id} vanished during update")
    return refreshed


def delete_config(conn: sqlite3.Connection, config_id: int) -> None:
    """Delete a NON-central config. Refuses the central Config (would leave the
    invariant unsatisfiable) and any config referenced by existing runs (a clear
    message ahead of the raw FK violation)."""
    existing = get_config(conn, config_id)
    if existing is None:
        raise ValueError(f"no config with id {config_id}")
    if existing.is_central:
        raise ValueError("cannot delete the central config; set a different config central first")
    referencing = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE config_id = ?", (config_id,)
    ).fetchone()[0]
    if referencing:
        raise ValueError(
            f"config {config_id} is referenced by {referencing} run(s); its history pins it"
        )
    conn.execute("DELETE FROM configs WHERE config_id = ?", (config_id,))


def set_central(conn: sqlite3.Connection, config_id: int) -> Config:
    """Promote ``config_id`` to the central Config (the deliberate operator action).

    Atomic + single-writer safe: unsets the current central, then sets the new one
    (the partial unique index rejects any transient two-central state, so the
    unset MUST precede the set). Wrapped in its own transaction unless the caller
    already owns one — on the caller-owned path, rolling back on error is the
    CALLER's responsibility (a swallowed inner failure could leave zero central
    rows). Prefer calling this on an autocommit connection."""
    target = get_config(conn, config_id)
    if target is None:
        raise ValueError(f"no config with id {config_id}")
    manage_txn = not conn.in_transaction
    if manage_txn:
        conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("UPDATE configs SET is_central = 0 WHERE is_central = 1")
        conn.execute("UPDATE configs SET is_central = 1 WHERE config_id = ?", (config_id,))
        if manage_txn:
            conn.execute("COMMIT")
    except BaseException:
        if manage_txn and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    refreshed = get_config(conn, config_id)
    if refreshed is None:  # pragma: no cover — row exists (checked above)
        raise RuntimeError(f"config {config_id} vanished during set_central")
    return refreshed
