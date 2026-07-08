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
| Portfolio math | PyPortfolioOpt `==1.6.0` (Black-Litterman + Ledoit-Wolf) |
| Optimizer | cvxpy (solver `CLARABEL`) |
| Data | yfinance `>=1.5` (prices), fredapi (macro), SQLite (WAL) cache |
| Frontend | React + Vite (TypeScript) |

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
backend/abe/   constants.py, storage.py, ingest/, features/, afml/, model/, blend/, optimize/,
               eval/, pipeline.py, scheduler.py, api.py
frontend/      React + Vite (one card per pipeline stage)
data/          SQLite db (gitignored)
docs/          seed-hardening research + docs/eval/ (committed walk-forward eval reports)
scripts/       smoke.py (real end-to-end gate, exit 0/1/3)
plan.md        full build plan (15 automated steps + M1/M2 operator UAT; per-step Status records)
```

## Status

**V1 automated build complete (Steps 1–14)** — issues #2–#15 closed. Full six-stage pipeline live
end-to-end (EWMA default): scheduler + degraded modes, React stage-card UI served by FastAPI,
AFML feature layer, minimal JEPA (41.8k params) behind the `ABE_MODEL` toggle, and the
pre-registered walk-forward eval committed at
[`docs/eval/jepa-vs-ewma-2026-07-08.md`](docs/eval/jepa-vs-ewma-2026-07-08.md) (mechanical verdict
"JEPA promoted" on a thin margin, honestly read as parity — the live default remains EWMA;
promotion is a manual operator action). 401 tests passing, 0 type errors, 0 lint violations;
real end-to-end smoke green (`uv run pytest -m smoke`).

**Remaining (operator):** Step 15 soak (#16, ≥4h wait), M1 UI walkthrough (#17), M2 degraded-mode
check (#18 — needs a FRED key in `.env` first). Details in [`plan.md`](plan.md) §Manual Steps.
