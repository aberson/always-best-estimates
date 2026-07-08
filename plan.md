# always-best-estimates — Project Plan

## 1. What This Is

`always-best-estimates` is a **local, single-user, always-on portfolio engine**. It continuously
(every ~5 minutes, or on demand) produces a target allocation for a fixed 3-asset universe —
**SPY** (S&P 500), **ACWI** (MSCI ACWI), **AGG** (US Aggregate Bond) — by running a pipeline:
**ingest → features → world-model forecast → Black-Litterman blend → constrained optimization →
UI**. Each stage is shown as a short card in a React frontend. It is an **advisory display only**:
no live trading, broker, or order execution; runs on `127.0.0.1` with no auth.

The forecaster sits behind a pluggable `WorldModel` interface. V1 ships an **EWMA baseline** that
drives the whole pipeline end-to-end; a **minimal JEPA** (Joint-Embedding Predictive Architecture,
per LeCun's world-model paper) is built later behind a config toggle and is promoted to the live
forecaster **only if** it wins a pre-registered walk-forward evaluation against the baseline. V1 is
"done" when the pipeline is correct and the JEPA has been *honestly evaluated* — never gated on the
JEPA beating the baseline (the joint history is only ~4,600 daily bars, so parity is the realistic
best case).

This plan was produced from a 7-dimension research pass (verified live 2026-07-07). The exhaustive
landmine catalog with file:line anchors lives at
[`docs/investigations/seed-hardening-research-2026-07-07.md`](docs/investigations/seed-hardening-research-2026-07-07.md).

**Out of scope (V1) / deferred to V2.** No live trading, broker, or order execution (advisory
display only). No auth (local `127.0.0.1`). Single portfolio, single user. Daily bars only (no
intraday). No full historical backtester beyond the walk-forward eval. **Deferred to V2:**
triple-barrier labeling, meta-labeling, sample-weights, sequential bootstrap, CUSUM event sampling
(these have **no V1 consumer** — the JEPA is self-supervised and Black-Litterman consumes returns,
not classification labels); ALFRED point-in-time vintages (V1 uses per-series release-lag constants);
a second live price source (Stooq and Google Finance are cut as verified non-functional; a keyed
Tiingo/Alpha Vantage adapter is a later add); additional assets; scheduled JEPA retraining;
multi-horizon forecasts; relative (P ≠ I) Black-Litterman views.

## 2. Stack

| Layer | Tool | Why |
|---|---|---|
| Language / runtime | Python 3.12 (uv-managed) | Workspace default; scientific stack |
| Web backend | FastAPI + uvicorn (single worker, no `--reload`) | Async API + lifespan scheduler; `--reload` drops PATH on Windows |
| Scheduler | Plain `asyncio` background task in FastAPI lifespan | Timer + on-demand + single-flight in ~20 lines, zero deps (APScheduler 4.x is alpha vaporware) |
| Compute isolation | `ThreadPoolExecutor(max_workers=1)` | CPU-bound pipeline off the event loop; doubles as single-flight lock |
| ML | PyTorch | Minimal JEPA (encoder + EMA target + predictor + VICReg variance-covariance regularization) |
| Portfolio math | PyPortfolioOpt `==1.6.0` | Black-Litterman (Idzorek) + Ledoit-Wolf shrinkage only |
| Optimization | cvxpy (solver pinned `CLARABEL`) | Hand-rolled mean-variance-utility QP with turnover |
| Feature transforms | in-project `backend/afml/` | Fractional differentiation + purged CV (charlesrambo repo is unlicensed / pandas-2-broken — reference only) |
| Data — prices | yfinance `>=1.5` | Free daily adjusted OHLCV; primary at ≤1 fetch/day |
| Data — macro | fredapi (FRED API key) | US macro daily series |
| Storage | SQLite (WAL mode) | Local, single-writer, timestamped history |
| Frontend | React + Vite (TypeScript) | One card per pipeline stage; polls the API |
| Tests | pytest (unit + integration + smoke + soak) | Integration through the production FastAPI route is load-bearing |

**Pinned ports:** backend `127.0.0.1:8140`, Vite dev `127.0.0.1:5174` (5173/8000/3000 are taken in
the workspace — confirm via `uv run --project dev-observatory observatory ports`). Production serves
the built frontend from FastAPI `StaticFiles` (one process).

## 3. Data Store

**SQLite** at `data/abe.db` (WAL mode; `data/` gitignored). PRAGMAs set once by `storage.py`:
`journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, `foreign_keys=ON`. **One writer
connection** owned by the pipeline thread; the API uses short-lived read-only connections
(`file:...?mode=ro`). `wal_checkpoint(TRUNCATE)` after each run. A single insert-boundary helper
coerces `float()/int()/.item()` on every numpy/torch scalar before write (else values persist as
BLOBs).

**Date key = one source of truth:** trading date stored as TEXT ISO-8601 `YYYY-MM-DD`, tz stripped
at the ingestion boundary. All event timestamps (`asof_utc`, `*_at_utc`) are UTC ISO-8601.

### Schema (v1)

| Table | Primary key | Columns (beyond PK) | Notes |
|---|---|---|---|
| `runs` | `run_id` INTEGER AUTOINCREMENT | `started_at_utc`, `finished_at_utc`, `status`, `trigger`, `error_text` | `status ∈ {queued,running,ok,error,skipped}`; `trigger ∈ {schedule,manual,startup}` |
| `run_stages` | (`run_id`, `stage`) | `status`, `started_at_utc`, `finished_at_utc`, `detail_json` | This table **is** the one-card-per-stage API |
| `prices` | (`asset`, `date`) | `open`,`high`,`low`,`close`,`adj_close`,`volume`,`source`,`fetched_at_utc` | `adj_close` is the modeling series; `source ∈ {yfinance,cache}` |
| `macro` | (`series_id`, `obs_date`) | `value`,`available_date`,`ingested_at_utc` | **append-only** → point-in-time from day one; `available_date = obs_date + release_lag` |
| `features` | (`run_id`, `asset`, `name`) | `value` | Long form; keyed for history charts |
| `forecasts` | (`run_id`, `asset`) | `horizon_days`,`mu`,`sigma`,`model_version` | `horizon_days = HORIZON_BARS` (21); `model_version ∈ {ewma,jepa:<hash>}` |
| `bl_posteriors` | (`run_id`, `asset`) | `prior_mu`,`view_mu`,`view_confidence`,`posterior_mu`,`posterior_sigma`,`detail_json` | `detail_json` holds Ω, π, tilt vectors |
| `target_weights` | (`run_id`, `asset`) | `weight`,`prev_weight`,`turnover`,`relaxed_turnover` | `relaxed_turnover` true when the turnover constraint was dropped |

**"Latest" = `MAX(run_id) WHERE status='ok'`.** Indices: `runs(status, run_id)`, and `(asset, run_id)`
on each derived table. **Deduplication:** prices/macro upsert (`INSERT … ON CONFLICT … DO UPDATE`);
re-ingesting an existing date is a no-op. **Corruption protection:** WAL + one-writer + per-run
transaction keyed by `run_id` so a crash leaves an inspectable `status='error'` row, never orphans;
`*.db*` gitignored; DB lives under `data/`, never OneDrive.

## 4. Core Pipeline (domain)

One **run** = one pass of six stages, each writing a `run_stages` row (→ one UI card):

0. **Freshness gate** — if the provider's latest bar date == stored `MAX(date)` and `force=false`,
   write `status='skipped'` and stop (daily data only changes once/day).
1. **Ingest** — serve prices/macro from SQLite cache (fetch is a *separate* daily job, not part of
   the 5-min recompute).
2. **Features** — log-returns + realized vol (EWMA path); frac-diff + macro join (JEPA path).
3. **Forecast** — `WorldModel.forecast(features) → {asset: (mu_H, sigma_H)}` over H=21 trading bars.
4. **Blend** — Ledoit-Wolf Σ; π = δ·Σ·w_mkt; forecasts → Idzorek views (confidence from σ);
   BL posterior (μ, Σ).
5. **Optimize** — cvxpy mean-variance-utility → target weights.

**The WorldModel contract (the highest-risk joint):** a JEPA emits latent embeddings, **not**
returns or uncertainty, so the full chain is named explicitly:
- **μ:** a return head `g(latent) → H-day log-return`, joint-trained at a small loss weight
  (sanctioned by LeCun §4.5.2).
- **σ (epistemic — uncertainty of the *mean*, NOT raw return variance):** deep-ensemble (K=3–5
  seeds) disagreement + rolling purged-walk-forward residual variance. If σ were the raw return
  variance (~20–40× larger), every Idzorek confidence would collapse to ~5% and views would never
  move the portfolio.
- **Calibration gate:** μ ± 1.64σ must cover ~90% of realized H-day returns on the walk-forward
  window.
- **Idzorek confidence:** `c = clamp(|2·Φ(μ/σ) − 1|, 0.02, 0.95)` — a no-information forecast → c≈0
  → BL returns the market prior (graceful degradation by construction).
- **Anti-collapse instrumentation** (per-dim embedding std, effective rank) is a build deliverable
  with a hard-fail threshold, plus a shuffled-target control asserted to score *worse* than the real
  run.

## 5. Modules

### `backend/abe/` (Python package)
- `constants.py` — **one source of truth**: `HORIZON_BARS=21`, `TRADING_DAYS=252`,
  `UNIVERSE=("SPY","ACWI","AGG")`, `W_MKT={"SPY":0.30,"ACWI":0.30,"AGG":0.40}`, `DELTA=2.5`,
  `TAU=0.05`, `W_MAX=0.60`, `FRED_DAILY` series list + `FRED_RELEASE_LAG` (business-day) table.
  Imported by every producer AND consumer; regression tests assert `is` identity.
- `storage.py` — connection/PRAGMAs, schema DDL, numpy-coercion insert boundary, one-writer conn.
- `ingest/sources.py` — `SourceAdapter` protocol; `YFinanceAdapter` (`auto_adjust=True,
  multi_level_index=False, progress=False`, column-set assertion); `CacheAdapter` (serve from DB).
- `ingest/prices.py` — incremental backfill + upsert (missing dates only, 429 backoff).
- `ingest/macro.py` — fredapi daily set, startup key-probe, `'.'/''→NaN`, `available_date` shift.
- `features/basic.py` — log-returns, realized vol (the EWMA path's minimal features).
- `afml/fracdiff.py` — fixed-width FFD (fractional differentiation), min-d ADF (Augmented Dickey-Fuller) search (training folds only), frozen weights.
- `afml/purged_cv.py` — purged + embargoed (≥H) chronological splits; leakage assertions.
- `features/build.py` — deterministic feature matrix (frac-diff + vol + macro merge_asof-backward).
- `model/base.py` — `WorldModel` Protocol + `EWMABaseline`.
- `model/jepa.py` — context encoder + EMA target encoder + predictor + VICReg + return head +
  ensemble σ; offline `train.py` → checkpoint; collapse instrumentation.
- `blend/covariance.py` — Ledoit-Wolf shrinkage (the only Σ path), annualization.
- `blend/confidence.py` — σ → Idzorek confidence map (leaf module, boundary-tested).
- `blend/black_litterman.py` — `bl_blend(...) → (mu_post, S_post, diagnostics)`; excess-return
  convention (`risk_free_rate=0.0` explicit); one ordered `View` list drives P/Q/confidences.
- `optimize/mvu.py` — cvxpy mean-variance-utility QP + constraints + guards.
- `pipeline.py` — the sync `run_pipeline(force)` function wiring all stages + ledger writes.
- `scheduler.py` — asyncio lifespan loop (fixed-delay + event trigger + single-flight).
- `api.py` — FastAPI routes.
- `eval/walk_forward.py` — pre-registered JEPA-vs-EWMA purged walk-forward eval + report.

### `frontend/` (React + Vite + TS)
- `src/App.tsx`, `src/api.ts` (typed client, polls `/api/runs/latest`), `src/components/StageCard.tsx`,
  `src/components/RunHeader.tsx` ("last successful run: <age>", refresh button).

## 6. API Route Contract

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| GET | `/health` | — | `{status:"ok"}` | Liveness |
| GET | `/api/runs/latest` | — | latest `ok` run + its stages | Poll target (5–10s) |
| GET | `/api/runs/{id}/stages` | — | `run_stages` rows | One card per stage |
| GET | `/api/history?limit=N` | `limit` | recent runs + target_weights | History view |
| POST | `/api/runs/trigger` | `{force?:bool}` | `202 {run_id}` (or `{run_id, already_running:true}`) | Sets the scheduler event; coalesces if a run is active |

All writes flow through the pipeline thread; the trigger endpoint sets an `asyncio.Event`, it does
**not** write to SQLite directly (single-writer discipline).

## 7. Project Structure

```
always-best-estimates/
├── plan.md                      # this file
├── CLAUDE.md                    # session bootstrap
├── README.md                    # (created at first /repo-update)
├── pyproject.toml               # uv-managed; pinned deps
├── .env.example                 # FRED_API_KEY= (real .env is gitignored)
├── .gitignore                   # .env, data/, *.db, *.db-wal, *.db-shm, node_modules/
├── data/                        # SQLite db (gitignored)
├── backend/
│   └── abe/
│       ├── constants.py         # one source of truth
│       ├── storage.py
│       ├── ingest/  {sources,prices,macro}.py
│       ├── features/ {basic,build}.py
│       ├── afml/    {fracdiff,purged_cv}.py
│       ├── model/   {base,jepa,train}.py
│       ├── blend/   {covariance,confidence,black_litterman}.py
│       ├── optimize/ mvu.py
│       ├── eval/    walk_forward.py
│       ├── pipeline.py
│       ├── scheduler.py
│       └── api.py
├── frontend/
│   ├── vite.config.ts           # port 5174, strictPort, host 127.0.0.1, /api proxy
│   └── src/ {App.tsx, api.ts, components/*}
├── scripts/                     # smoke.py, soak.py, backfill entrypoint
└── tests/                       # unit + integration + anchors
```

## 8. Key Design Decisions

- **Skeleton-first ordering.** The EWMA baseline drives BL → optimizer → UI → scheduler end-to-end
  *before* any JEPA or full AFML work. The `WorldModel` interface (frozen with a contract test in
  Step 5) is the lever; every downstream stage is exercised against the baseline, so the
  research-grade ML never blocks the shippable product.
- **BL equilibrium prior = fixed config vector SPY=0.30 / ACWI=0.30 / AGG=0.40** (60/40 with the
  equity leg split). Chosen over market-cap weights because SPY ⊂ ACWI (US ≈ 60% of ACWI) double-
  counts US large-cap, and ETF AUM is the wrong unit. Idzorek fn.4 sanctions a "presumed efficient
  benchmark." The no-view anchor test asserts the pipeline reproduces this vector.
- **Ledoit-Wolf is the only covariance path.** SPY/ACWI correlation ≈ 0.95 → near-singular Σ → an
  unshrunk mean-variance optimizer flips the entire equity sleeve on basis-point μ changes. LW lifts
  the smallest eigenvalue and guarantees PSD. **Marchenko-Pastur / RMT denoising is explicitly
  excluded** (meaningless at N=3, despite the AFML framing).
- **Fixed δ = 2.5** (He-Litterman), shared by the BL prior and the optimizer objective. Avoids the
  market-implied δ going negative in a drawdown window and inverting the QP.
- **cvxpy uses `sum_squares(chol.T @ w)`, not `quad_form`.** `quad_form`'s ARPACK PSD check is known
  to fail on exactly the near-singular matrices this universe produces. Solver pinned to `CLARABEL`;
  weights clipped `<1e-8 → 0` and renormalized; turnover is stateful (`w_prev` = last persisted
  allocation) with a cold-start drop and INFEASIBLE-retry guard.
- **Everything internal in annualized excess returns.** `risk_free_rate=0.0` passed explicitly at
  every pypfopt call site; rf (from FRED) subtracted at ONE adapter; H-day → annual via `×252/H`
  through the shared constants. Units/horizon drift is the top silent-bug class here.
- **Fetch is split from recompute.** The 5-min timer recomputes from SQLite only; a separate daily
  post-close job fetches incrementally. This prevents Yahoo IP bans (yfinance is an unofficial
  scraper) and keeps the run ledger meaningful (`skipped` vs real runs).
- **Cache is the real fallback.** Stooq (PoW/API-key wall since ~2026-03) and Google Finance (no API
  since 2012) are both cut — verified non-functional from this machine. The SQLite last-known-good
  cache + a "stale since <date>" UI banner is the guaranteed fallback.
- **Data-source facts pinned from live verification**, not memory: yfinance 1.5.x defaults changed
  (`auto_adjust=True` → no "Adj Close" column; `multi_level_index=True`), so the ingest wrapper
  passes explicit flags and asserts the column set before any write.

## 9. Open Questions / Risks

| Item | Risk | Mitigation |
|---|---|---|
| JEPA on ~4,600 daily bars | Overfits / collapses; may not beat EWMA | Non-blocking behind toggle; collapse instrumentation + shuffled-target anchor; promotion only via pre-registered walk-forward eval |
| Producer/consumer unit drift (μ vs Σ, H vs annual) | Plausible-looking wrong weights | One constants module (`is`-asserted); integration test asserts annualized SPY vol in 0.05–0.60 band |
| SPY/ACWI near-singular Σ | Unstable corner weights | Ledoit-Wolf + box caps (W_MAX=0.6) + turnover; ±10bp μ-perturbation stability test |
| yfinance breakage window (Yahoo endpoint change) | Ingest fails | Cache serves history offline; a real-network ingest smoke test surfaces breakage at ingest, not as garbage output |
| FRED lookahead (revision + release lag) | Inflated backtest skill | `available_date` join (merge_asof backward); append-only macro; daily-only set keeps lags ~1 day |
| Idzorek confidence endpoints (0 → Ω=1e6, 1 → Ω=0) | Crash or extreme tilt | Clamp `c ∈ [0.02, 0.95]` at the adapter; unit-test both edges + the "25 instead of 0.25" case |
| PyPortfolioOpt version/numpy pin | Resolver failure mid-build | Pin `==1.6.0` + import smoke test; hand-rolled BL posterior (~30 lines) named as fallback |
| Windows laptop sleep | Loop clock gaps | Fixed-delay loop self-heals on wake; UI shows "last successful run: <age>", no keep-awake hacks |
| Silent asyncio task death | "Always-on" silently stops | Per-iteration try/except → `error` row + loop restart; raising-stage test asserts next run still fires |

## 10. How to Run

```powershell
# 1. Backend deps (uv-managed; never pip)
uv sync

# 2. Secrets — copy the example and paste your FRED key
Copy-Item .env.example .env
# then edit .env: FRED_API_KEY=<your key>

# 3. One-time price/macro backfill into SQLite
uv run python -m abe.ingest.prices --backfill
uv run python -m abe.ingest.macro --backfill

# 4. Frontend deps
npm install --prefix frontend

# 5a. Dev (two processes): API on :8140, Vite on :5174 (proxies /api)
uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140
npm run dev --prefix frontend

# 5b. Production (one process): build the UI, FastAPI serves it
npm run build --prefix frontend
uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140
```

Open `http://127.0.0.1:5174` (dev) or `http://127.0.0.1:8140` (prod). The scheduler starts with the
app; a run fires immediately at startup, then every 5 minutes / on the refresh button.

## 11. Development Process

Build with **`/build-phase`** over the ordered steps below. Default `--reviewers code`
(4-agent gauntlet + typecheck/lint/test gates); the real UI step gets `--reviewers full`
(+ `--start-cmd` + `--url`); the soak is `Type: wait`. Isolation: `worktree` default. Steps are sized
as one vertical slice each. The plan has a `Type: wait` step + operator M-steps, so Build Steps split
into Automated + Manual subsections.

### Automated Steps
(These run unattended via `/build-phase`.)

### Step 1: Scaffold + constants + observatory registration
- **Problem:** Create the uv project (FastAPI `/health`, React+Vite stub, `pyproject.toml` with all pinned deps), `.gitignore` (`.env`, `data/`, `*.db*`, `node_modules/`), `.env.example`, `backend/abe/constants.py` (the one-source-of-truth values), and register the owned project with dev-observatory.
- **Type:** code
- **Issue:** #2
- **Flags:** --reviewers code
- **Produces:** project skeleton, `constants.py`, `vite.config.ts` (port 5174, strictPort, host 127.0.0.1, `/api` proxy), registry entry in `dev/.claude/observatory/registry.toml`
- **Done when:** `uv run pytest` green on scaffold tests; `uvicorn` serves `/health` 200; the Vite stub renders on 127.0.0.1:5174; `uv run --project dev-observatory observatory status` lists `always-best-estimates`; `observatory ports` shows 8140/5174 with no collision.
- **Depends on:** none
- **Status:** DONE (2026-07-07)

### Step 2: SQLite storage module
- **Problem:** `storage.py` — connection + PRAGMAs (WAL/synchronous=NORMAL/busy_timeout/foreign_keys), full schema DDL (§3), the numpy/torch scalar coercion insert boundary, one-writer connection.
- **Type:** code
- **Issue:** #3
- **Flags:** --reviewers code
- **Produces:** `backend/abe/storage.py`, schema migration, `data/` dir handling
- **Done when:** schema creates from scratch; `journal_mode=WAL` asserted; a round-trip test writes a real `np.float64` and a `torch.Tensor` scalar and reads back Python `float`; foreign-key violation test rejects an orphan `run_stages` row.
- **Depends on:** 1
- **Status:** DONE (2026-07-07)

### Step 3: Price ingest — yfinance adapter + cache
- **Problem:** `SourceAdapter` protocol; `YFinanceAdapter` (`auto_adjust=True, multi_level_index=False, progress=False` + column-set assertion before write); `CacheAdapter`; incremental upsert (missing dates only, 429 backoff); backfill entrypoint.
- **Type:** code
- **Issue:** #4
- **Flags:** --reviewers code
- **Produces:** `backend/abe/ingest/{sources,prices}.py`, `scripts` backfill hook
- **Done when:** backfill loads SPY/ACWI/AGG adjusted daily closes; a second run is a no-op (idempotency test); a network-off test serves full history from cache; a test asserts AGG trailing-10y annualized mean return > 1% (guards the price-only-vs-total-return bond bug).
- **Depends on:** 2
- **Status:** DONE (2026-07-07) — note: adapter uses `yf.Ticker().history(auto_adjust=True, actions=False, interval="1d")` + fail-loud config instead of `download()` (which swallows all exceptions incl. rate limits); `multi_level_index`/`progress` are download()-only kwargs. Incremental fetch uses a 10-day inclusive overlap window with adj_close consistency check → full refresh on backward-adjustment rebase.

### Step 4: FRED macro ingest (daily set)
- **Problem:** `macro.py` — fredapi adapter for the daily set (DGS10, T10Y2Y, VIXCLS, DFF, BAMLH0A0HYM2, DTWEXBGS); startup key-probe (missing/invalid key → explicit degraded mode with a stable error code, never silent-empty); parse `'.'/''` → NaN; store `(series_id, obs_date, value, available_date, ingested_at)` append-only with per-series release-lag shift.
- **Type:** code
- **Issue:** #5
- **Flags:** --reviewers code
- **Produces:** `backend/abe/ingest/macro.py`, `FRED_RELEASE_LAG` usage
- **Done when:** the 6 daily series fetch and store; `available_date = obs_date + declared lag` asserted; a missing-key test starts the app in macro-disabled degraded mode with the stable code; empty-string FRED values parse as NaN (test).
- **Depends on:** 2
- **Status:** DONE (2026-07-07) — no FRED key on this machine: the real 6-series fetch ships as a keyed self-skipping `network` test (operator runs it after adding the key, pre-M2); degraded mode (MACRO_DISABLED_NO_KEY, exit 2) live-verified. Stable codes: MACRO_OK / MACRO_DISABLED_NO_KEY / MACRO_DISABLED_BAD_KEY; FRED calls bounded by 15s socket timeout.

### Step 5: WorldModel interface + EWMA baseline
- **Problem:** `model/base.py` — the `WorldModel` Protocol `forecast(features) → {asset: (mu_H, sigma_H)}` over H=21, and `EWMABaseline` (μ = EWMA of returns; σ = trailing forecast-error std, a genuine positive uncertainty). Freeze the interface with a contract test every implementation must pass.
- **Type:** code
- **Issue:** #6
- **Flags:** --reviewers code
- **Produces:** `backend/abe/model/base.py`, `backend/abe/features/basic.py`
- **Done when:** EWMA emits `(mu, sigma)` per asset with σ>0; the contract test asserts shape, horizon, and non-degenerate σ; μ/σ are per-horizon (H-day) returns.
- **Depends on:** 2
- **Status:** DONE (2026-07-08) — SEMANTIC DECISION recorded: `Forecast.sigma` is the **H-day PREDICTIVE forecast std** (the scale at which μ±1.64σ covers ~90% of realized H-day returns — §4's calibration gate is the definition). EWMA's trailing forecast-error std is predictive by construction; Step 13's JEPA composite must land on this same scale or Step 14's comparison is invalid. §4's "epistemic" phrasing is superseded for V1; the failure mode to avoid is raw-VARIANCE/daily/annualized units. Contract test (assert_worldmodel_contract in tests/test_model_base.py) is frozen; implementations must arm expected_daily_mu on a known-drift input + pin σ scale on iid noise.

### Step 6: Blend module — covariance + confidence + Black-Litterman
- **Problem:** `blend/covariance.py` (Ledoit-Wolf as the only Σ path, annualized), `blend/confidence.py` (σ → Idzorek confidence `c = clamp(|2Φ(μ/σ)−1|, 0.02, 0.95)`, leaf module), `blend/black_litterman.py` (`bl_blend` pure fn: π = δ·Σ·w_mkt with `risk_free_rate=0.0`; `BlackLittermanModel(omega='idzorek', view_confidences, tau=0.05)`; ONE ordered `View` list → P/Q/confidences; diagnostics).
- **Type:** code
- **Issue:** #7
- **Flags:** --reviewers code
- **Produces:** `backend/abe/blend/{covariance,confidence,black_litterman}.py`
- **Done when:** confidence boundary tests (σ→0 ⇒ c clamps to 0.95, σ large ⇒ c→~0.02); LW returns PSD; annualized SPY vol lands in 0.05–0.60 (units test); a golden-value test reproduces an Idzorek-paper Table-6/7 example within tolerance; `pyportfolioopt==1.6.0` import smoke test.
- **Depends on:** 5
- **Status:** DONE (2026-07-08) — golden pins: Idzorek Table 6 col 2 posterior (atol 1e-4, max err 6.4e-5) + Table 4 view variances + p.15 Ω diag; idzorek-ω closed form + tilt∝c pinned separately. Decisions: confidence computed from RAW H-day (μ,σ) pair (annualize-first would inflate z by √12); Q annualized ×12 at the bl_blend boundary only; rf hard-rejected unless exactly 0.0 (V1 excess-return convention).

### Step 7: cvxpy mean-variance-utility optimizer
- **Problem:** `optimize/mvu.py` — `maximize(mu@w − 0.5·δ·sum_squares(chol.T@w) − γ_tc·norm1(w − w_prev))` s.t. `sum(w)==1, 0≤w≤W_MAX`; `chol = cholesky(annualized Σ_post)`; solver `CLARABEL`; require status ∈ {OPTIMAL, OPTIMAL_INACCURATE}; clip `w<1e-8→0` + renormalize; cold-start drops turnover; INFEASIBLE retries without turnover and flags `relaxed_turnover`.
- **Type:** code
- **Issue:** #8
- **Flags:** --reviewers code
- **Produces:** `backend/abe/optimize/mvu.py`
- **Done when:** weights sum to 1 with all constraints test-asserted; cold-start (no `w_prev`) test passes; a deliberately near-singular Σ flows through without crashing; a ±10bp μ-perturbation moves max weight below a declared bound (stability test).
- **Depends on:** 6
- **Status:** DONE (2026-07-08) — γ_tc default 0.002 (no-trade band: ±10bp noise never trades — measured zero trades at corr 0.999; percent-scale views clear the band — anchored both directions). Stability bounds: 0.25 cold-start (measured 0.158; corner-flip is 0.30) / 0.01 with w_prev+γ (measured ≤2e-7). W_MKT inversion anchor certifies δ-sharing + quadratic-form orientation. Renormalization is float-exact under builtin sum; box overshoot scales with solver status (ppb on OPTIMAL, ~1e-4 on OPTIMAL_INACCURATE — downstream gates must key to status).

### Step 8: Pipeline orchestrator + JSON API + run ledger
- **Problem:** `pipeline.py` (sync `run_pipeline(force)` wiring freshness-gate → ingest(cache) → features → WorldModel → blend → optimize → persist, each writing a `run_stages` row) and `api.py` (routes in §6, reads via short-lived read-only connections).
- **Type:** code
- **Issue:** #9
- **Flags:** --reviewers code
- **Produces:** `backend/abe/{pipeline,api}.py`
- **Done when:** an integration test drives `POST /api/runs/trigger` end-to-end **through the production FastAPI route** (TestClient with lifespan) and asserts rows land in `runs`, `run_stages`, and `target_weights` with weights summing to 1; `/api/runs/latest` returns the run + stages.
- **Depends on:** 3, 4, 5, 6, 7
- **Status:** DONE (2026-07-08) — freshness gate uses DUAL watermarks (MAX(date) + MAX(fetched_at_utc)) so in-place adj_close rebases un-skip recomputes. Two-phase transaction: phase-0 autocommit runs row, phase-1 BEGIN IMMEDIATE data writes (rollback-verified), phase-2 error-ledger replay. V1 trigger runs the pipeline synchronously on the event loop (ALL routes stall during a run — Step 11's executor swap fixes; Step 11 also owns the stale-'running' startup sweep). probe_fred_key runs BEFORE open_writer (leak-proof ordering).

### Step 9: Smoke gate — one real end-to-end cycle
- **Problem:** `scripts/smoke.py` — a ~60-second real end-to-end run with NO mocks: boot the app, trigger one real run against cached data, assert no exception, all `run_stages` `status='ok'`, and `target_weights` persisted. Surfaces producer/consumer drift that mocked unit tests miss.
- **Type:** code
- **Issue:** #10
- **Flags:** --reviewers code
- **Produces:** `scripts/smoke.py`, a `pytest` smoke marker
- **Done when:** the smoke script exits 0; every stage card is `ok`; weights sum to 1. (Business-logic quality is out of scope — this gate only proves the pipeline completes one real cycle without crashing.)
- **Depends on:** 8

### Step 10: React UI — one card per stage
- **Problem:** React cards for each pipeline stage (latest prices, features, per-asset forecast, BL posterior, final weights) reading `run_stages`; a refresh button (`POST /api/runs/trigger`); `error`/`skipped` first-class card states; a header showing "last successful run: <age>". The optimizer card states the SPY/ACWI overlap caveat.
- **Type:** code
- **Issue:** #11
- **Flags:** --reviewers full --isolation worktree --start-cmd "uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140" --url http://127.0.0.1:8140 --ui
- **Produces:** `frontend/src/**`
- **Done when:** the UI renders all stage cards from a live run; the refresh button triggers a run and the cards update; error/skipped states render (not blank); runtime reviewers pass (no auth gate — the URL is open).
- **Depends on:** 8

### Step 11: Scheduler + degraded modes + error resilience
- **Problem:** `scheduler.py` — asyncio lifespan task: fixed-delay loop (`wait_for(event, timeout=300)`) + on-demand trigger + `asyncio.Lock` single-flight; pipeline body via `ThreadPoolExecutor(max_workers=1)`; stage-0 freshness gate (skip on unchanged); separate daily-fetch vs 5-min-recompute paths; per-iteration try/except → `status='error'` row + loop restart; `wal_checkpoint(TRUNCATE)` after each run.
- **Type:** code
- **Issue:** #12
- **Flags:** --reviewers code
- **Produces:** `backend/abe/scheduler.py`, lifespan wiring in `api.py`
- **Done when:** an unchanged-data test writes `status='skipped'`; a raising-stage test writes an `error` row AND the next scheduled run still fires; the trigger endpoint coalesces during an active run; a `force=true` trigger bypasses the freshness gate.
- **Depends on:** 8

### Step 12: Feature layer — frac-diff + purged CV + macro join
- **Problem:** `afml/fracdiff.py` (fixed-width FFD on log-prices; min-d ADF grid search on TRAINING folds only; persist frozen `{d, tau, window_len, adf_p, corr}`), `afml/purged_cv.py` (purged + embargoed ≥H splits), `features/build.py` (returns/vol + frac-diff + FRED `merge_asof` backward on `available_date`; deterministic matrix). In-project, against current pandas/sklearn.
- **Type:** code
- **Issue:** #13
- **Flags:** --reviewers code
- **Produces:** `backend/abe/afml/{fracdiff,purged_cv}.py`, `backend/abe/features/build.py`
- **Done when:** feature regeneration is hash-stable (deterministic); a no-lookahead test (feature at t uses only data available at t); ADF confirms FFD stationarity; a leakage test asserts no train label-interval overlaps any test interval; garbage anchors (white-noise → d≈0, random-walk → d≈1).
- **Depends on:** 5

### Step 13: Minimal JEPA behind a toggle
- **Problem:** `model/jepa.py` (context encoder + EMA target encoder + predictor + VICReg anti-collapse + return head + ensemble σ, <100–500k params), `model/train.py` (offline training → checkpoint), an EWMA↔JEPA config toggle wiring `JEPAModel` through the `WorldModel` interface, and collapse instrumentation (per-dim embedding std, effective rank, hard-fail threshold).
- **Type:** code
- **Issue:** #14
- **Flags:** --reviewers code
- **Produces:** `backend/abe/model/{jepa,train}.py`, checkpoint loader
- **Done when:** training produces a checkpoint; the no-collapse check passes; a shuffled-target control run scores measurably worse than the real run (asserted in CI); the toggle routes `JEPAModel` through the production interface; DEFAULT stays EWMA.
- **Depends on:** 12

### Step 14: JEPA walk-forward evaluation gate
- **Problem:** `eval/walk_forward.py` — a pre-registered purged walk-forward eval of `(μ, σ)` produced **through the production `WorldModel` interface**, JEPA vs EWMA on identical windows (forecast MSE/IC + σ calibration coverage); writes a committed eval report and records the promotion decision.
- **Type:** code
- **Issue:** #15
- **Flags:** --reviewers code
- **Produces:** `backend/abe/eval/walk_forward.py`, `docs/eval/jepa-vs-ewma-<date>.md`
- **Done when:** the eval report exists comparing JEPA vs EWMA on identical purged walk-forward windows; calibration coverage is computed; the report records "EWMA remains default unless JEPA wins" with the measured numbers.
- **Depends on:** 13

### Step 15: Soak test
- **Problem:** Run the full always-on engine for a target duration (≥4 hours) against cached + live-daily data; capture findings (run cadence, skipped-vs-real distribution, memory, any error rows, laptop-sleep gap handling).
- **Type:** wait
- **Issue:** #16
- **Produces:** `docs/soak/soak-<date>.md` findings log
- **Done when:** the engine runs ≥4h; the run ledger shows periodic runs with `skipped` vs real correctly marked; no unhandled crash; findings captured. (Resume in a fresh session via `--resume 16` after the wait.)
- **Depends on:** 11

### Manual Steps
(These run after `/build-phase` completes. Operator drives.)

### Step M1: UI acceptance walkthrough
- **Source step:** Steps 10, 11 (§ Build Steps)
- **Issue:** #17
- **Commands:**
  ```powershell
  npm run build --prefix frontend
  uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140
  # open http://127.0.0.1:8140
  ```
- **What to look for:**
  | Check | Expected outcome |
  |---|---|
  | Startup run | A run fires immediately; all five stage cards populate |
  | Refresh button | Triggers a new run; cards + "last successful run" age update |
  | Weights card | SPY/ACWI/AGG weights sum to 1; overlap caveat text is present |
  | Skipped run | Refreshing with no new daily data shows a `skipped` state, not an error |

### Step M2: Degraded-mode check
- **Source step:** Steps 3, 4, 11 (§ Build Steps)
- **Issue:** #18
- **Commands:**
  ```powershell
  # (a) no macro key
  Rename-Item .env .env.bak; uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140
  # observe, then: Rename-Item .env.bak .env
  # (b) network off: disable the NIC, then hit the refresh button in the UI
  ```
- **What to look for:**
  | Check | Expected outcome |
  |---|---|
  | Missing FRED key | App starts in macro-disabled degraded mode with the stable error code on the macro card — not a crash, not silent-empty |
  | Network off | Ingest card shows "stale since <date>"; the pipeline still completes from cache and produces weights |
  | Recovery | Restoring `.env` / network clears the degraded banners on the next run |

Please run **M1** next after the automated steps complete.

## 12. Appendix

**Constants (one source of truth — `backend/abe/constants.py`):**
```python
HORIZON_BARS = 21          # trading days; the single forecast horizon
TRADING_DAYS = 252         # annualization factor
UNIVERSE = ("SPY", "ACWI", "AGG")
W_MKT = {"SPY": 0.30, "ACWI": 0.30, "AGG": 0.40}   # BL equilibrium prior (60/40, equity split)
DELTA = 2.5                # He-Litterman risk aversion; shared by BL prior + optimizer
TAU = 0.05                 # BL prior scalar (cancels in idzorek omega mode)
W_MAX = 0.60               # per-asset box cap
FRED_DAILY = ("DGS10", "T10Y2Y", "VIXCLS", "DFF", "BAMLH0A0HYM2", "DTWEXBGS")
FRED_RELEASE_LAG = {       # business days from obs_date to available_date
    "DGS10": 1, "T10Y2Y": 1, "VIXCLS": 1, "DFF": 1, "BAMLH0A0HYM2": 1, "DTWEXBGS": 3,
}
```

**Verified data facts (2026-07-07):** yfinance 1.5.x serves SPY (from 1993, 8,415 rows), ACWI (from
2008-03-28, 4,597 rows), AGG (from 2003-09-29, 5,728 rows). **Joint panel ≈ 4,600 daily bars from
ACWI's 2008 inception** → ~220 non-overlapping H=21 windows per asset. yfinance 1.5.x defaults:
`auto_adjust=True` (no "Adj Close" column), `multi_level_index=True`. Stooq CSV: dead (PoW + API-key
wall since ~2026-03). Google Finance: no API since 2012.

**Measurement-validity anchors (must exist in CI):** (1) no-view ⇒ weights = W_MKT within 1e-6
through the production entry point; (2) absurd Q at c=0.99 moves weights materially; (3) shuffled-
target JEPA scores worse than the real run; (4) frac-diff garbage anchors (noise→d≈0, RW→d≈1).

**Reference material** (the two PDFs are external third-party papers kept in the workspace `papers/`
dir, referenced by absolute path — not committed to this repo): JEPA —
`C:/Users/abero/dev/papers/10356_a_path_towards_autonomous_mach.pdf` (LeCun §4.4–4.8). Black-Litterman/
Idzorek — `C:/Users/abero/dev/papers/Idzorek_onBL.pdf` (Formulas 1, 12–19; Tables 6–7). Full landmine
catalog (in-repo) — [`docs/investigations/seed-hardening-research-2026-07-07.md`](docs/investigations/seed-hardening-research-2026-07-07.md).

---

**Next:** run `/plan-expedite --plan plan.md` to auto-prep (plan-review → plan-wrap → repo-sync →
session-wrap) for `/build-phase`, or `/plan-review` first if you want a manual review pass.
