"""Automated Track 2 soak harness (plan Step 29 / issue #35).

Hands-off: start it once and walk away. While the backend's central Config runs
on its own 5-minute loop, this driver exercises the OTHER Track 2 write path --
periodic on-demand config runs -- so the single-writer contention between the
loop and on-demand runs actually happens, and samples the metrics that only a
wall-clock soak can expose into a findings file.

What it does on two independent timers until the deadline:
  * poke   (default every 180s): POST /api/configs/{id}/run on a NON-central
    config, alternating force=false (must serve a cache hit on unchanged data)
    and force=true (forces a fresh run -> contends with the loop). Every 3rd
    poke also GETs /api/compare (the read path).
  * sample (default every 300s): db + WAL file sizes, runs/run_stages counts,
    stuck-in-running/queued count, error-run count, backend RSS (best-effort).

It NEVER creates throwaway configs (a config that has runs can't be deleted, so
that would leave cruft): it drives whatever non-central configs already exist.
On completion OR Ctrl-C it writes docs/soak/track2-soak-<date>.md.

Usage (from the project root, backend already running -- click the run button
or scripts/launch-abe.ps1 first):

    uv run python scripts/soak.py                 # 4h soak, defaults
    uv run python scripts/soak.py --hours 6       # longer
    uv run python scripts/soak.py --minutes 2     # quick self-test

Exit codes: 0 = ran + wrote findings; 3 = precondition failed (backend down /
no non-central config).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime

_DEFAULT_BASE_URL = "http://127.0.0.1:8140"
_DEFAULT_DB = "data/abe.db"
_HTTP_TIMEOUT = 120.0  # an on-demand run blocks until it completes
_LOOP_GRANULARITY_S = 5.0  # how often the scheduler wakes to check the two timers


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# HTTP (stdlib only)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HttpResult:
    ok: bool
    status: int | None
    body: dict | list | None
    error: str | None
    elapsed_ms: float


def _request(method: str, url: str, payload: dict | None = None) -> HttpResult:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = (time.perf_counter() - start) * 1000.0
            body = json.loads(raw) if raw else None
            return HttpResult(True, resp.status, body, None, elapsed)
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:200]
        except Exception:
            pass
        return HttpResult(False, exc.code, None, f"HTTP {exc.code}: {detail}", elapsed)
    except Exception as exc:  # timeout, connection refused, decode error...
        elapsed = (time.perf_counter() - start) * 1000.0
        return HttpResult(False, None, None, f"{type(exc).__name__}: {exc}", elapsed)


# --------------------------------------------------------------------------- #
# Best-effort backend RSS via PowerShell (Windows workspace; degrades to None)
# --------------------------------------------------------------------------- #


def _ps(command: str) -> str | None:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _discover_backend_pid(port: int) -> int | None:
    val = _ps(
        f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
        f"-ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess"
    )
    if val and val.isdigit():
        return int(val)
    return None


def _sample_rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    val = _ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64")
    if val and val.isdigit():
        return int(val)
    return None


# --------------------------------------------------------------------------- #
# DB metrics (read-only)
# --------------------------------------------------------------------------- #


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


@dataclass(frozen=True)
class DbSample:
    db_bytes: int
    wal_bytes: int
    shm_bytes: int
    runs_total: int
    stages_total: int
    running_or_queued: int
    error_runs: int


def _db_sample(db_path: str) -> DbSample:
    db_b = _file_size(db_path)
    wal_b = _file_size(db_path + "-wal")
    shm_b = _file_size(db_path + "-shm")
    runs = stages = stuck = errs = -1
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            conn.execute("PRAGMA query_only = ON")
            runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            stages = conn.execute("SELECT COUNT(*) FROM run_stages").fetchone()[0]
            stuck = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status IN ('running','queued')"
            ).fetchone()[0]
            errs = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'error'"
            ).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        pass  # a momentary lock -> record the file sizes, leave counts at -1
    return DbSample(db_b, wal_b, shm_b, runs, stages, stuck, errs)


def _runs_by_trigger(db_path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            for trig, n in conn.execute(
                'SELECT "trigger", COUNT(*) FROM runs GROUP BY "trigger"'
            ).fetchall():
                out[str(trig)] = int(n)
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    return out


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #


@dataclass
class PokeStats:
    total: int = 0
    force_true: int = 0
    force_false: int = 0
    cached_hits: int = 0  # cached=true responses (only expected on force=false)
    fresh: int = 0  # cached=false responses
    errors: int = 0
    locked_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    compare_calls: int = 0
    compare_errors: int = 0
    error_examples: list[str] = field(default_factory=list)


@dataclass
class Sample:
    ts: str
    minute: float
    db: DbSample
    rss_bytes: int | None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _mb(n: int | None) -> str:
    if n is None or n < 0:
        return "n/a"
    return f"{n / 1_048_576:.1f} MB"


def _write_report(
    path: str,
    *,
    started: datetime,
    ended: datetime,
    args: argparse.Namespace,
    config_ids: list[int],
    baseline: DbSample,
    samples: list[Sample],
    pokes: PokeStats,
    triggers_before: dict[str, int],
    triggers_after: dict[str, int],
    interrupted: bool,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dur_min = (ended - started).total_seconds() / 60.0
    last = samples[-1].db if samples else baseline
    max_wal = max((s.db.wal_bytes for s in samples), default=baseline.wal_bytes)
    max_stuck = max((s.db.running_or_queued for s in samples), default=0)
    rss_values = [s.rss_bytes for s in samples if s.rss_bytes]
    rss_start = rss_values[0] if rss_values else None
    rss_max = max(rss_values) if rss_values else None
    on_demand_added = triggers_after.get("manual", 0) - triggers_before.get("manual", 0)
    loop_added = triggers_after.get("schedule", 0) - triggers_before.get("schedule", 0)

    # Verdict checks
    checks: list[tuple[str, bool, str]] = []
    checks.append(("no writer errors on on-demand runs", pokes.errors == 0,
                   f"{pokes.errors} errored poke(s), {pokes.locked_errors} 'database is locked'"))
    final_stuck = last.running_or_queued
    checks.append(("no runs stuck in running/queued at end", final_stuck == 0,
                   f"final running/queued = {final_stuck} (max seen {max_stuck})"))
    err_delta = (last.error_runs - baseline.error_runs) if last.error_runs >= 0 else 0
    checks.append(("no new error-status runs", err_delta <= 0,
                   f"error-status runs delta = {err_delta}"))
    checks.append(("cache hits observed on force=false (caching works)",
                   pokes.cached_hits > 0 or pokes.force_false == 0,
                   f"{pokes.cached_hits} hit(s) of {pokes.force_false} force=false poke(s)"))
    passed = all(ok for _, ok, _ in checks)
    verdict = "PASS" if passed else "ATTENTION"
    if interrupted:
        verdict += " (interrupted before the planned duration)"

    lines: list[str] = []
    lines.append(f"# Track 2 soak -- {started.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("Plan Step 29 / issue #35. Generated by `scripts/soak.py` (automated, hands-off).")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Started (UTC): {_iso(started)}")
    lines.append(f"- Ended (UTC): {_iso(ended)}")
    lines.append(f"- Duration: {dur_min:.1f} min (target {args.target_minutes:.0f} min)")
    lines.append(f"- Poke interval: {args.poke_interval:.0f}s; "
                 f"sample interval: {args.sample_interval:.0f}s")
    lines.append(f"- Base URL: {args.base_url}; DB: {args.db}")
    lines.append(f"- Non-central configs driven on-demand: {config_ids or '(none)'}")
    lines.append(f"- Interrupted early: {'yes' if interrupted else 'no'}")
    lines.append("")
    lines.append("## Verdict checks")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")
    lines.append("## On-demand load (the contention path)")
    lines.append("")
    lines.append(f"- Pokes: {pokes.total} "
                 f"(force=false {pokes.force_false}, force=true {pokes.force_true})")
    lines.append(f"- Cache hits (force=false serving unchanged data): {pokes.cached_hits}")
    lines.append(f"- Fresh runs computed: {pokes.fresh}")
    lines.append(f"- Errored pokes: {pokes.errors} "
                 f"(of which 'database is locked': {pokes.locked_errors})")
    lines.append(f"- /api/compare calls: {pokes.compare_calls} ({pokes.compare_errors} errored)")
    lines.append(f"- Poke latency p50 / p95: {_pct(pokes.latencies_ms, 0.5):.0f} / "
                 f"{_pct(pokes.latencies_ms, 0.95):.0f} ms")
    lines.append(f"- Runs added -- loop (schedule): {loop_added}, "
                 f"on-demand (manual): {on_demand_added}")
    if pokes.error_examples:
        lines.append("")
        lines.append("Error examples:")
        for e in pokes.error_examples[:8]:
            lines.append(f"- `{e}`")
    lines.append("")
    lines.append("## Growth + memory")
    lines.append("")
    lines.append(f"- DB size: {_mb(baseline.db_bytes)} -> {_mb(last.db_bytes)}")
    lines.append(f"- WAL size: max {_mb(max_wal)}, final {_mb(last.wal_bytes)} "
                 "(bounded = checkpointing works)")
    lines.append(f"- runs rows: {baseline.runs_total} -> {last.runs_total}; "
                 f"run_stages rows: {baseline.stages_total} -> {last.stages_total}")
    lines.append(f"- Backend RSS: start {_mb(rss_start)}, max {_mb(rss_max)}")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    lines.append("| min | db | wal | runs | stages | run/queued | err | rss |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in samples:
        lines.append(
            f"| {s.minute:.0f} | {_mb(s.db.db_bytes)} | {_mb(s.db.wal_bytes)} | "
            f"{s.db.runs_total} | {s.db.stages_total} | {s.db.running_or_queued} | "
            f"{s.db.error_runs} | {_mb(s.rss_bytes)} |"
        )
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automated Track 2 soak harness (issue #35).")
    dur = p.add_mutually_exclusive_group()
    dur.add_argument("--hours", type=float, help="soak duration in hours (default 4.0)")
    dur.add_argument("--minutes", type=float, help="soak duration in minutes (overrides --hours)")
    p.add_argument("--poke-interval", type=float, default=180.0,
                   help="seconds between on-demand pokes (default 180)")
    p.add_argument("--sample-interval", type=float, default=300.0,
                   help="seconds between metric samples (default 300)")
    p.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="backend base URL")
    p.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite db")
    p.add_argument("--out", default=None,
                   help="findings path (default docs/soak/track2-soak-<date>.md)")
    args = p.parse_args(argv)
    if args.minutes is not None:
        args.target_minutes = args.minutes
    elif args.hours is not None:
        args.target_minutes = args.hours * 60.0
    else:
        args.target_minutes = 240.0
    if args.out is None:
        args.out = f"docs/soak/track2-soak-{_utcnow().strftime('%Y-%m-%d')}.md"
    return args


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    # Preflight: backend up?
    health = _request("GET", f"{args.base_url}/health")
    if not health.ok:
        print(f"PRECONDITION: backend not reachable at {args.base_url} ({health.error}).",
              file=sys.stderr)
        print("Start it first: run scripts/launch-abe.ps1 (or the dev-observatory run button).",
              file=sys.stderr)
        return 3

    # Discover NON-central configs to drive on-demand (never create cruft).
    cfg = _request("GET", f"{args.base_url}/api/configs")
    configs = cfg.body.get("configs", []) if cfg.ok and isinstance(cfg.body, dict) else []
    non_central = [c["config_id"] for c in configs if not c.get("is_central")]
    compare_ids = ",".join(str(c["config_id"]) for c in configs)  # central + non-central
    if not non_central:
        print("PRECONDITION: no non-central config to drive on-demand.", file=sys.stderr)
        print("Create one in the compare UI (or POST /api/configs), then re-run.",
              file=sys.stderr)
        return 3

    port = 8140
    try:
        port = int(args.base_url.rsplit(":", 1)[1].split("/")[0])
    except (IndexError, ValueError):
        pass
    backend_pid = _discover_backend_pid(port)

    started = _utcnow()
    deadline = started.timestamp() + args.target_minutes * 60.0
    baseline = _db_sample(args.db)
    triggers_before = _runs_by_trigger(args.db)
    pokes = PokeStats()
    samples: list[Sample] = []

    print(f"soak: driving configs {non_central} for {args.target_minutes:.0f} min; "
          f"poke {args.poke_interval:.0f}s, sample {args.sample_interval:.0f}s -> {args.out}")
    print("soak: hands-off now -- Ctrl-C writes the findings early.")

    # Take an initial sample immediately so the report has a t=0 row.
    samples.append(Sample(_iso(started), 0.0, baseline, _sample_rss_bytes(backend_pid)))
    next_poke = started.timestamp() + args.poke_interval
    next_sample = started.timestamp() + args.sample_interval
    cfg_cursor = 0
    poke_index = 0
    interrupted = False

    try:
        while time.time() < deadline:
            now = time.time()

            if now >= next_poke:
                config_id = non_central[cfg_cursor % len(non_central)]
                cfg_cursor += 1
                force = poke_index % 2 == 1  # alternate: even=false (cache), odd=true (fresh)
                poke_index += 1
                res = _request("POST", f"{args.base_url}/api/configs/{config_id}/run",
                               {"force": force})
                pokes.total += 1
                pokes.force_true += int(force)
                pokes.force_false += int(not force)
                if res.ok and isinstance(res.body, dict):
                    pokes.latencies_ms.append(res.elapsed_ms)
                    if res.body.get("cached"):
                        pokes.cached_hits += 1
                    else:
                        pokes.fresh += 1
                else:
                    pokes.errors += 1
                    msg = (res.error or "unknown")
                    if "lock" in msg.lower():
                        pokes.locked_errors += 1
                    if len(pokes.error_examples) < 12:
                        pokes.error_examples.append(f"cfg {config_id} force={force}: {msg}")
                # Every 3rd poke also exercise the read path.
                if poke_index % 3 == 0:
                    comp = _request(
                        "GET", f"{args.base_url}/api/compare?config_ids={compare_ids}"
                    )
                    pokes.compare_calls += 1
                    if not comp.ok:
                        pokes.compare_errors += 1
                next_poke = time.time() + args.poke_interval

            if now >= next_sample:
                minute = (time.time() - started.timestamp()) / 60.0
                samples.append(Sample(_iso(_utcnow()), minute, _db_sample(args.db),
                                      _sample_rss_bytes(backend_pid)))
                print(f"soak: sample @ {minute:.0f} min -- pokes {pokes.total}, "
                      f"errors {pokes.errors}, wal {_mb(samples[-1].db.wal_bytes)}")
                next_sample = time.time() + args.sample_interval

            time.sleep(_LOOP_GRANULARITY_S)
    except KeyboardInterrupt:
        interrupted = True
        print("\nsoak: interrupted -- writing findings so far.")

    ended = _utcnow()
    samples.append(Sample(_iso(ended), (ended - started).total_seconds() / 60.0,
                          _db_sample(args.db), _sample_rss_bytes(backend_pid)))
    triggers_after = _runs_by_trigger(args.db)

    _write_report(
        args.out,
        started=started,
        ended=ended,
        args=args,
        config_ids=non_central,
        baseline=baseline,
        samples=samples,
        pokes=pokes,
        triggers_before=triggers_before,
        triggers_after=triggers_after,
        interrupted=interrupted,
    )
    print(f"soak: wrote {args.out} ({pokes.total} pokes, {pokes.errors} errors, "
          f"{len(samples)} samples).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
