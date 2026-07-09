# always-best-estimates

A **local, single-user, always-on portfolio engine**. Every ~5 minutes (or on demand) it runs a
pipeline — **ingest → features → world-model forecast → Black-Litterman blend → constrained
optimization → UI** — over a fixed 3-asset universe (**SPY**, **ACWI**, **AGG**) and shows a short
card for each stage in a React frontend. It is an **advisory display only**: no live trading, no
broker, no auth; it runs on `127.0.0.1`.

The forecaster sits behind a pluggable `WorldModel` interface. V1 ships an **EWMA baseline** that
drives the whole pipeline end-to-end; a **minimal JEPA** (Joint-Embedding Predictive Architecture)
is added later behind a config toggle and is promoted to the live forecaster **only if** it wins a
pre-registered walk-forward evaluation against the baseline.

## Stack

| Layer | Tool |
|---|---|
| Runtime | Python 3.12 (uv-managed) |
| Backend | FastAPI + uvicorn (single worker, no `--reload`) |
| Scheduler | asyncio lifespan task + `ThreadPoolExecutor(max_workers=1)` |
| ML | PyTorch (minimal JEPA) |
| Portfolio math | PyPortfolioOpt `==1.6.0` (Black-Litterman + Ledoit-Wolf), scipy (`erfinv`, view-source math) |
| Optimizer | cvxpy (solver `CLARABEL`) |
| Data | yfinance `>=1.5` (prices), fredapi (macro), SQLite (WAL) cache |
| Frontend | React + Vite (TypeScript) + react-router-dom (HashRouter) |

Ports: backend `127.0.0.1:8140`, Vite dev `127.0.0.1:5174`.

## Prerequisites

- Windows 11 + PowerShell (dev default); uv-managed Python 3.12; Node (for the frontend).
- A free **FRED API key** → gitignored `.env` (`FRED_API_KEY=…`); `.env.example` is committed.
  Missing key → the app starts in an explicit macro-disabled degraded mode (never silent-empty).
- Network for the daily price/macro fetch; the SQLite cache serves history offline.

## Setup

```powershell
uv sync                                                    # backend deps (never pip)
Copy-Item .env.example .env                                # then edit: FRED_API_KEY=<key>
uv run python -m abe.ingest.prices --backfill              # one-time price backfill
uv run python -m abe.ingest.macro --backfill               # one-time macro backfill
npm install --prefix frontend                              # frontend deps

# Dev (two processes): API on :8140, Vite on :5174 (proxies /api)
uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140
npm run dev --prefix frontend

# Production (one process): build the UI, FastAPI serves it
npm run build --prefix frontend
uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140
```

Open `http://127.0.0.1:5174` (dev) or `http://127.0.0.1:8140` (prod). The scheduler starts with the
app; a run fires at startup, then every 5 minutes / on the refresh button.

**One-click (dev):** `.\scripts\launch-abe.ps1` starts the backend + Vite each in its own window and
opens the app in Chrome (falls back to the default browser if Chrome is absent). It's also surfaced
as dev-observatory's one-click **`run`** button.

## Key design decisions

- **Skeleton-first.** The EWMA baseline drives BL → optimizer → UI → scheduler end-to-end before any
  JEPA or full AFML work. V1 "done" = pipeline correct + JEPA *honestly evaluated* behind a toggle —
  never gated on the JEPA beating the baseline (joint history is only ~4,600 daily bars).
- **BL prior** = fixed benchmark SPY 0.30 / ACWI 0.30 / AGG 0.40 (SPY ⊂ ACWI, so market-cap weights
  would double-count US large-cap). **Ledoit-Wolf is the only covariance path** (SPY/ACWI ≈ 0.95
  correlated → near-singular Σ). Fixed risk-aversion δ = 2.5, shared by prior and optimizer.
- **Fetch is split from recompute** — the 5-min loop recomputes from SQLite; a separate daily job
  fetches incrementally (yfinance is an unofficial scraper; a 5-min fetch loop invites IP bans).
- **One source of truth** (`constants.py`, `is`-asserted) for HORIZON_BARS, TRADING_DAYS, the
  universe, and FRED release lags — the top silent-bug class here is producer/consumer unit drift.

## Project layout

```
backend/abe/   constants.py, calc.py (simple calcs + explain registry), storage.py, ingest/,
               features/, afml/, model/, blend/, optimize/, eval/, pipeline.py, scheduler.py, api.py
frontend/      React + Vite (per-stage cards + compare / scenario-authoring views, HashRouter)
data/          SQLite db (gitignored)
docs/          seed-hardening research + docs/eval/ (committed walk-forward eval reports)
scripts/       smoke.py (real end-to-end gate, exit 0/1/3), launch-abe.ps1 (one-click backend+Vite+Chrome launcher)
plan.md        full build plan (15 automated steps + M1/M2 operator UAT; per-step Status records)
```

## Status

**V1 automated build complete (Steps 1–14)** — issues #2–#15 closed. Full six-stage pipeline live
end-to-end (EWMA default): scheduler + degraded modes, React stage-card UI served by FastAPI,
AFML feature layer, minimal JEPA (41.8k params) behind the `ABE_MODEL` toggle, and the
pre-registered walk-forward eval committed at
[`docs/eval/jepa-vs-ewma-2026-07-08.md`](docs/eval/jepa-vs-ewma-2026-07-08.md) (mechanical verdict
"JEPA promoted" on a thin margin, honestly read as parity — the live default remains EWMA;
promotion is a manual operator action).

**Track 1 — Transparency pass (post-V1)** — made the stage cards self-explaining **without
changing any pipeline math**: relocated the simple calculations into a single transparent
`backend/abe/calc.py` (formula + worked example per quantity), added a `GET /api/explain` endpoint
+ an inline "how is this computed?" expander per card, and surfaced already-computed detail
(Black-Litterman prior/view/posterior, price provenance, feature windows, the optimizer objective,
the covariance common-window). M1 UI walkthrough accepted (#17 closed).

**Track 2 — Pluggable-per-stage scenario + compare engine (post-Track-1)** — automated Steps
16–28 (issues #22–#34 closed). A forward-only schema migration framework (`backend/abe/migrations.py`,
v1→v3) backs new `configs` / `view_scenarios` tables; `run_pipeline` now executes a resolved
**Config** (byte-identical V1 parity golden) drawn from four string-keyed stage registries
(`backend/abe/registry.py`: feature-builder / forecaster / view-source / optimizer). New stage impls
land additively — historical + counterfactual Black-Litterman view sources (`backend/abe/blend/views.py`),
a `min_variance` optimizer + an MVU `min_weight` floor (fixes the V1 AGG=0% corner), and a
`fracdiff_macro` feature set. A new API route group (`/api/registries`, `/api/configs`,
`/api/scenarios`, `/api/compare`) drives new React views — a **compare** grid, a **scenario-authoring**
editor, and **per-stage detail tabs** — routed via `react-router-dom` (HashRouter). The central
Config runs the 5-min loop; non-central Configs run on-demand, cached by `config_id` and serialized
through the single writer.

497 tests passing, 0 type errors (32 source files), 0 lint violations; real end-to-end smoke green
(`uv run pytest -m smoke`).

**Remaining (operator):** Track 2 Step 29 soak (#35, ≥4h wait) + Step 30 UAT (#36); V1 Step 15 soak
(#16, ≥4h wait); M2 degraded-mode check (#18 — the FRED key is now configured, so run the macro
backfill). Track 2 detail lives in [`docs/track2-scenario-engine-plan.md`](docs/track2-scenario-engine-plan.md)
(Steps 16–30; 16–28 marked DONE); V1 per-step records in [`plan.md`](plan.md) §13–§14.
