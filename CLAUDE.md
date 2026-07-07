# always-best-estimates — Project instructions

## 1. Overview

A local, single-user, **always-on portfolio engine**: every ~5 minutes (or on demand) it runs
ingest → features → world-model forecast → Black-Litterman blend → constrained optimization and
displays a short card per stage in a React UI, for a fixed 3-asset universe (SPY, ACWI, AGG).
Advisory display only — no trading, no auth, `127.0.0.1`. Forecaster sits behind a pluggable
`WorldModel` interface: an EWMA baseline ships first; a minimal JEPA is added later behind a toggle
and promoted only if it wins a walk-forward eval. Full plan: [`plan.md`](plan.md).

## 2. Stack

| Layer | Tool |
|---|---|
| Runtime | Python 3.12 (uv-managed) |
| Backend | FastAPI + uvicorn (single worker, no `--reload`) |
| Scheduler | asyncio lifespan task + `ThreadPoolExecutor(max_workers=1)` |
| ML | PyTorch (minimal JEPA) |
| Portfolio math | PyPortfolioOpt `==1.6.0` (Black-Litterman + Ledoit-Wolf only) |
| Optimizer | cvxpy (solver `CLARABEL`) |
| Data | yfinance `>=1.5` (prices), fredapi (macro), SQLite (WAL) cache |
| Frontend | React + Vite (TypeScript) |
| Tests | pytest (unit + integration + smoke + soak) |

Ports: backend `127.0.0.1:8140`, Vite dev `127.0.0.1:5174`.

## 3. Commands

```
uv sync                                                    # install backend deps (never pip)
npm install --prefix frontend                              # install frontend deps
uv run python -m abe.ingest.prices --backfill              # one-time price backfill
uv run python -m abe.ingest.macro --backfill               # one-time macro backfill
uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140    # run backend (starts scheduler)
npm run dev --prefix frontend                              # run Vite dev server on 127.0.0.1:5174
npm run build --prefix frontend                            # build UI (prod: FastAPI serves it)
uv run pytest                                              # run tests
uv run pytest -m smoke                                     # 60s real end-to-end smoke gate
uv run ruff check .                                        # lint
uv run mypy backend                                        # typecheck
```

## 4. Directory layout

```
always-best-estimates/
├── plan.md, CLAUDE.md, pyproject.toml, .env.example, .gitignore
├── data/                        # SQLite db (gitignored)
├── backend/abe/
│   ├── constants.py             # one source of truth (HORIZON_BARS, TRADING_DAYS, UNIVERSE, W_MKT, DELTA…)
│   ├── storage.py               # SQLite conn/PRAGMAs/schema/coercion boundary
│   ├── ingest/ {sources,prices,macro}.py
│   ├── features/ {basic,build}.py
│   ├── afml/ {fracdiff,purged_cv}.py
│   ├── model/ {base,jepa,train}.py
│   ├── blend/ {covariance,confidence,black_litterman}.py
│   ├── optimize/ mvu.py
│   ├── eval/ walk_forward.py
│   ├── pipeline.py, scheduler.py, api.py
├── frontend/ (vite.config.ts, src/{App.tsx,api.ts,components/})
├── scripts/ (smoke.py, soak.py)
└── tests/
```

## 5. Architecture

- **Pipeline (`pipeline.py`)** — one sync `run_pipeline(force)` wiring six stages; each writes a
  `run_stages` row (the one-card-per-stage API). Runs off the event loop in a single-worker threadpool.
- **Scheduler (`scheduler.py`)** — asyncio lifespan loop; fixed-delay 5-min timer + on-demand event
  trigger + single-flight lock. **Fetch is split from recompute**: the 5-min loop recomputes from
  SQLite only; a separate daily job fetches prices/macro incrementally (prevents Yahoo IP bans).
- **WorldModel (`model/`)** — Protocol `forecast(features) → {asset:(mu_H,sigma_H)}` over H=21.
  EWMA baseline default; JEPA optional behind a toggle. σ is **epistemic** (uncertainty of the mean),
  mapped to Idzorek confidence `c = clamp(|2Φ(μ/σ)−1|, 0.02, 0.95)`.
- **Blend (`blend/`)** — Ledoit-Wolf Σ (only covariance path; SPY⊂ACWI ⇒ near-singular), π = δ·Σ·w_mkt,
  BL posterior via PyPortfolioOpt idzorek mode. Everything in annualized excess returns (rf=0.0 explicit).
- **Optimize (`optimize/mvu.py`)** — hand-rolled cvxpy mean-variance-utility (`sum_squares(chol.T@w)`,
  not `quad_form`), long-only, box W_MAX=0.6, L1 turnover vs last persisted weights.

## 6. Current state

**Plan written, no code yet.** Build via `/build-phase --plan plan.md` (15 automated steps +
M1/M2 manual). Update this section at each phase end via `/repo-update`.

## 7. Environment requirements

- Windows 11 + PowerShell (workspace default); uv-managed Python 3.12; Node for the frontend.
- **FRED API key** required (free) → gitignored `.env` (`FRED_API_KEY=…`); `.env.example` committed.
  Missing key → app starts in explicit macro-disabled degraded mode (never silent-empty).
- Network access for the daily price/macro fetch; the SQLite cache serves history offline.
- No Docker, no cloud, no GPU required (the minimal JEPA is <500k params, CPU-trainable).
