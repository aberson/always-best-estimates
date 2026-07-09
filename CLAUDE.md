# always-best-estimates — Project instructions

## 1. Overview

A local, single-user, **always-on portfolio engine**: every ~5 minutes (or on demand) it runs
ingest → features → world-model forecast → Black-Litterman blend → constrained optimization and
displays a short card per stage in a React UI, for a fixed 3-asset universe (SPY, ACWI, AGG).
Advisory display only — no trading, no auth, `127.0.0.1`. Forecaster sits behind a pluggable
`WorldModel` interface: an EWMA baseline ships first; a minimal JEPA is added later behind a toggle
and promoted only if it wins a walk-forward eval. **Every stage is now pluggable** (Track 2:
feature-builder / forecaster / view-source / optimizer registries) — the app runs a central `Config`
on the 5-min loop and any number of alternate `Config`s on-demand, with a compare view and saved
view-scenarios for side-by-side what-ifs. Full plan: [`plan.md`](plan.md); Track 2 detail:
[`docs/track2-scenario-engine-plan.md`](docs/track2-scenario-engine-plan.md).

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
| Stationarity | statsmodels (ADF test in the offline min-d search only) |
| Special functions | scipy (`erfinv` for the counterfactual view-source confidence in `blend/views.py`; `spearmanr` in eval) — a declared direct dep |
| Frontend | React + Vite (TypeScript) + react-router-dom (HashRouter) |
| Tests | pytest (default suite + `smoke`/`network`/`realdb` markers, deselected by addopts) |

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
uv run pytest                                              # run tests (smoke/network/realdb deselected)
uv run pytest -m smoke                                     # real end-to-end smoke gate (needs data/abe.db; NEVER skips)
uv run ruff check .                                        # lint
uv run mypy backend                                        # typecheck (strict)
uv run python scripts/smoke.py                             # same smoke via CLI (exit 0/1/3 = pass/failure/precondition)
uv run python -m abe.model.train --db data/abe.db --out data/jepa.pt      # offline JEPA training -> checkpoint
uv run python -m abe.eval.walk_forward --db data/abe.db --out docs/eval/jepa-vs-ewma-<date>.md   # promotion eval
```

## 4. Directory layout

```
always-best-estimates/
├── plan.md, CLAUDE.md, pyproject.toml, .env.example, .gitignore
├── data/                        # SQLite db (gitignored)
├── backend/abe/
│   ├── constants.py             # one source of truth (HORIZON_BARS, TRADING_DAYS, UNIVERSE, W_MKT, DELTA…)
│   ├── calc.py                  # simple calcs (log-return, realized-vol, annualize, idzorek) + Explanation registry
│   ├── storage.py               # SQLite conn/PRAGMAs/coercion boundary + table registry
│   ├── migrations.py            # forward-only schema migration framework (v1→v3, auto-applied)
│   ├── config.py                # Config/ViewScenario entities + CRUD + guarded set_central
│   ├── registry.py              # four stage registries (feature/forecaster/view-source/optimizer) + resolve()
│   ├── ingest/ {sources,prices,macro}.py
│   ├── features/ build.py       # (basic.py relocated into calc.py)
│   ├── afml/ {fracdiff,purged_cv}.py
│   ├── model/ {base,jepa,train}.py
│   ├── blend/ {covariance,black_litterman,views}.py   # views = forecast/historical/counterfactual view sources
│   ├── optimize/ {mvu,min_variance}.py
│   ├── eval/ walk_forward.py
│   ├── pipeline.py, scheduler.py, api.py
├── frontend/ (vite.config.ts, src/{App.tsx,api.ts,components/{DashboardView,StageDetailTab,CompareView,ScenarioEditor,StageCard,RunHeader}.tsx})
├── scripts/ (smoke.py)
├── docs/eval/                   # committed walk-forward eval reports
└── tests/ (+ tests/seeding.py + tests/conftest.py shared helpers)
```

## 5. Architecture

- **Pipeline (`pipeline.py`)** — one sync `run_pipeline(conn, *, config, …)` runs a **resolved
  `Config`** through six stages; each writes a `run_stages` row (the one-card-per-stage API). A
  byte-identical parity golden pins V1 reproduction. Runs off the event loop in a single-worker threadpool.
- **Scheduler (`scheduler.py`)** — asyncio lifespan loop; fixed-delay 5-min timer + on-demand event
  trigger + single-flight lock. **Fetch is split from recompute**: the 5-min loop recomputes from
  SQLite only; a separate daily job fetches prices/macro incrementally (prevents Yahoo IP bans).
- **Config + registries (`config.py`, `registry.py`, `migrations.py`)** — a `Config` names one impl
  per stage from four string-keyed registries (feature-builder / forecaster / view-source /
  optimizer); `registry.resolve()` maps keys to impls. The **central `Config`** runs on the 5-min
  loop; **non-central `Config`s run on-demand** (cached by `config_id`, tagged), and every DB write
  funnels through `scheduler.run_write` (the one executor = single writer). Schema is versioned by a
  forward-only migration framework (v1→v3). View sources (`blend/views.py`): forecast (V1),
  historical, counterfactual.
- **WorldModel (`model/`)** — Protocol `forecast(features) → {asset:(mu_H,sigma_H)}` over H=21.
  EWMA baseline default; JEPA behind the `ABE_MODEL`/`ABE_JEPA_CHECKPOINT` env toggle (invalid
  config fails loud at startup). σ is the **H-day PREDICTIVE forecast std** (the scale at which
  μ±1.64σ covers ~90% of realized returns), mapped to Idzorek confidence
  `c = clamp(|2Φ(μ/σ)−1|, 0.02, 0.95)` computed from the RAW H-day pair.
- **Blend (`blend/`)** — Ledoit-Wolf Σ (only covariance path; SPY⊂ACWI ⇒ near-singular), π = δ·Σ·w_mkt,
  BL posterior via PyPortfolioOpt idzorek mode. Everything in annualized excess returns (rf=0.0 explicit).
- **Optimize (`optimize/{mvu,min_variance}.py`)** — hand-rolled cvxpy mean-variance-utility
  (`sum_squares(chol.T@w)`, not `quad_form`), long-only, box W_MAX=0.6, L1 turnover vs last persisted
  weights, plus an optional `min_weight` box floor (fixes the V1 AGG=0% corner). A sibling
  min-variance optimizer is selectable per `Config`.
- **Transparency (`calc.py` + `/api/explain`)** — the simple calcs live once in `calc.py`
  (formula + worked-example docstrings) with an `EXPLANATIONS` registry (formula/description/example
  per quantity); `GET /api/explain` serves it to the UI's per-card "how is this computed?" expanders.
  Stage details carry additive explanatory fields (BL prior/view, covariance common-window, price
  provenance, feature windows, optimizer objective) — surfacing computed data, not new math.

## 6. Current state

**V1 build complete (Steps 1–14) + Track 1 transparency pass (2026-07-08).** V1: issues #2–#15
closed; six-stage pipeline end-to-end on the EWMA default; JEPA behind the `ABE_MODEL` toggle; the
pre-registered eval ([`docs/eval/jepa-vs-ewma-2026-07-08.md`](docs/eval/jepa-vs-ewma-2026-07-08.md))
reads "JEPA promoted" on a thin margin (honestly: parity) — **live default remains EWMA**, promotion
is a manual operator action. **Track 1 (post-V1):** stage cards made self-explaining without changing
any pipeline math — simple calcs relocated to `backend/abe/calc.py` + an `EXPLANATIONS` registry,
`GET /api/explain` + an inline per-card "how is this computed?" expander, and additive stage-detail
enrichments (BL prior/view/posterior, price provenance, feature windows, optimizer objective,
covariance common-window); M1 (#17) accepted/closed. **Track 2 (pluggable-per-stage scenario +
compare engine) then landed:** automated Steps 16–28 (sub-phases 2A/2B/2C; issues #22–#34 all
closed) shipped the forward-only migration framework (v1→v3; real db backed up at
`data/abe.db.pre-track2-backup`), stage registries + `Config`/`ViewScenario` entities, the
config/scenario/compare API route group, on-demand config runs (single-writer via `run_write`),
historical/counterfactual view sources, a min-variance optimizer + `min_weight` floor, and the React
compare/scenario UI (HashRouter). 497 tests (5 deselected: smoke/network/realdb), mypy strict clean
(32 source files), ruff clean, real smoke green. **Remaining operator (all still open):** Track 2
Step 29 soak (#35, `Type: wait`) + Step 30 UAT (#36); plus the V1 Step 15 soak (#16) and M2 macro
backfill (#18). Per-step decisions live in `plan.md`'s `**Status:**` lines + §13/§14 and
[`docs/track2-scenario-engine-plan.md`](docs/track2-scenario-engine-plan.md) (Steps 16–30; 16–28 DONE).

## 7. Environment requirements

- Windows 11 + PowerShell (workspace default); uv-managed Python 3.12; Node for the frontend.
- **FRED API key** required (free) → gitignored `.env` (`FRED_API_KEY=…`); `.env.example` committed.
  Missing key → app starts in explicit macro-disabled degraded mode (never silent-empty).
- Network access for the daily price/macro fetch; the SQLite cache serves history offline.
- No Docker, no cloud, no GPU required (the minimal JEPA is <500k params, CPU-trainable).
