# Track 2 — Pluggable-per-stage scenario + compare engine

> Feature plan (sub-plan of the canonical [`plan.md`](../plan.md) — see §14 there for the Track 1/2
> split and the operator decisions this plan implements). Continues the step numbering from V1
> (Steps 1–15); Track 2 is **Steps 16–30**, grouped into sub-phases **2A / 2B / 2C**.

## 1. What This Feature Does

Turns the fixed six-stage pipeline into a **pluggable-per-stage engine** where each stage
(features → forecast → blend-views → optimize) has selectable implementations, bundled into a named
**Config**. The always-on 5-minute loop keeps computing exactly one **central Config** — the
portfolio you'd actually buy — while any number of other Configs and what-if **view scenarios**
(forecast-derived, historical, or hand-authored counterfactuals like "SPY +10%") are computed
**on demand** and cached, so you can **compare** allocations side by side. Everything (Configs,
scenarios, per-Config runs) is tracked in an evolved SQLite schema with a real migration path. It
is built because V1 hard-wired one recipe (basic features → EWMA → forecast views → MVU); the
operator wants to explore alternative forecasters, feature sets, view scenarios, and optimizers —
and to see the data → features → forecast → blend → optimize flow for whichever recipe is active —
without losing the single, unambiguous "central" answer.

## 2. Existing Context

- **Pipeline (`backend/abe/pipeline.py`).** `run_pipeline(conn, *, trigger, force, model,
  macro_status, on_run_started)` wires six stage functions through a `_RunContext`. Today only the
  **forecaster** is pluggable (the `WorldModel` Protocol in `model/base.py`; EWMA default, JEPA via
  `ABE_MODEL`). `_stage_features` calls `calc.log_returns`/`calc.realized_vol` directly;
  `_stage_blend` calls `bl_blend`; `_stage_optimize` calls `optimize_weights` — all hardcoded.
- **Storage (`backend/abe/storage.py`).** Single-config schema v1: `runs → run_stages / features /
  forecasts / bl_posteriors / target_weights`, all keyed by `run_id`. `ensure_schema` is
  `CREATE TABLE IF NOT EXISTS` + `PRAGMA user_version` stamp — **no migration path**. One-writer
  discipline; a `coerce_scalar` insert boundary; `insert_row`/`upsert_row` derive columns from the
  live schema.
- **Scheduler (`backend/abe/scheduler.py`).** One asyncio loop + one `ThreadPoolExecutor(1)` + one
  writer connection runs exactly one config (`SchedulerConfig.model`), fixed-delay 5-min recompute +
  daily fetch, single-flight by construction. `request_run(force)` resolves at run START.
- **API (`backend/abe/api.py`).** `/health`, `/api/runs/latest`, `/api/runs/{id}/stages`,
  `/api/history`, `POST /api/runs/trigger`, `GET /api/explain` (Track 1), static frontend mount.
  Startup builds `SchedulerConfig(model=resolve_startup_model())`.
- **Frontend (`frontend/src/`).** Bare React + Vite (`react`, `react-dom` only — **no router**); a
  single polling dashboard (`App.tsx`) rendering `StageCard`s + the Track 1 explain expanders.
- **Transparency (`backend/abe/calc.py`, Track 1).** `EXPLANATIONS` registry + `GET /api/explain`
  already provide formula/example metadata per quantity — the model the per-stage registries extend.

## 3. Scope

**In scope (Steps 16–30):**
- A **migration framework** (versioned, forward-only) and a schema v2 adding `configs`,
  `view_scenarios`, and `config_id` on `runs`.
- **Config** + **ViewScenario** domain models, storage CRUD, and a seeded **central Config** that
  reproduces today's behavior byte-for-byte.
- **Stage registries** (feature-builder / forecaster / view-source / optimizer) keyed by string with
  a declared param schema, extending the Track 1 `EXPLANATIONS` pattern.
- `run_pipeline` refactored to run a resolved **Config**; the scheduler runs the **central** Config.
- **On-demand** runs for non-central Configs (cached, tagged `config_id`), via the one executor.
- New stage implementations: **historical** + **counterfactual** view sources (with a
  pre-programmed library + on-the-fly authoring); a **min-weight floor** on MVU and **one
  alternative optimizer** (addresses the V1 AGG=0% realism gap); the **frac-diff+macro** feature set
  exposed as selectable.
- **Frontend**: a central-scenario dashboard (default), per-stage **detail tabs** with dropdowns to
  pick impl + params, a **compare** view, and **view-scenario authoring**.
- A **soak/observation** step (the always-on loop + on-demand runs under load) and an operator UAT.

**Out of scope (deferred):**
- Postgres / any non-SQLite engine (explicitly kept SQLite — operator decision).
- Live trading, brokerage, order execution (still advisory-only, `127.0.0.1`, no auth).
- Automatic promotion of a non-central Config to central (always a deliberate operator action).
- Scheduled retraining of JEPA / multi-horizon forecasts / relative BL views (P≠I) — V2+ items
  already deferred in `plan.md`.
- Running every Config on the 5-minute loop (operator decision: central-only on the loop).

## 4. Impact Analysis

| File | Change Type | Reason | Verified |
|---|---|---|---|
| `backend/abe/storage.py` | extend | Add migration framework + schema v2 (`configs`, `view_scenarios`, `runs.config_id`); bump `SCHEMA_VERSION`. | grep'd `ensure_schema`/`SCHEMA_VERSION`: consumers are `storage.py` internals + `open_writer` (1 call) + `tests/test_storage.py:69`. No other module imports the DDL — inserts derive columns from live schema. |
| `backend/abe/pipeline.py` | refactor | `run_pipeline` takes a resolved `Config`; stages resolve impls from registries; persist `config_id`. | grep'd `run_pipeline` callers: `scheduler.py:545` (`_job`) + `tests/test_pipeline.py`. `optimize_weights` caller: `pipeline.py:590` only. `bl_blend` caller: `pipeline.py:_stage_blend` only. |
| `backend/abe/scheduler.py` | modify | `SchedulerConfig` carries the central `Config` (supersedes the `model` field); loop runs the central Config; new on-demand config-run dispatch. | grep'd `SchedulerConfig(`/`config.model`: `api.py:270` (startup), `scheduler.py:549`, `tests/seeding.py:28`, `tests/test_scheduler.py` (5 seams). All updated to the Config-based field with a back-compat shim. |
| `backend/abe/api.py` | extend | New routes: configs + view-scenarios CRUD, set-central, trigger-config-run, comparison; startup builds the central Config. | grep'd routes in `api.py` (7 existing); startup at `api.py:270` builds `SchedulerConfig(model=…)`. |
| `backend/abe/model/base.py` | modify | Register EWMA/JEPA in the forecaster registry (keep the `WorldModel` Protocol as-is). | `WorldModel` Protocol read; EWMA/JEPA are its only impls. |
| `backend/abe/blend/black_litterman.py` | extend | `bl_blend` gains a pluggable view source (forecast/historical/counterfactual) feeding `absolute_views`. | `bl_blend` caller: `pipeline._stage_blend` only (grep above). |
| `backend/abe/optimize/mvu.py` | extend | Add a `min_weight` box-floor param; register MVU + a second optimizer. | `optimize_weights` caller: `pipeline.py:590` only. |
| `backend/abe/features/build.py` | extend | Expose `build_features` (frac-diff+macro) as a selectable feature set. | `build_features` callers: `eval/walk_forward.py:444`, `model/train.py`. `_stage_features` currently bypasses it (uses `calc` directly). |
| `frontend/src/**` | extend | Router + dashboard/detail-tabs/compare/scenario-authoring views; config/scenario API client. | `package.json` has no router; `App.tsx` is a single dashboard. New dep: `react-router-dom`. |
| `tests/**` | extend | Migration golden, Config-parity golden, registry, on-demand-run, view-source, optimizer-floor, API, and a soak observation log. | `tests/test_storage.py`, `test_pipeline.py`, `test_scheduler.py`, `test_optimize_mvu.py`, `test_blend_black_litterman.py`, `test_api.py` all touch the changed modules. |

## 5. New Components

**Data shapes (illustrative v2 additions — Step 16 finalizes the exact DDL; ids follow the existing
`runs.run_id` convention: `INTEGER PRIMARY KEY AUTOINCREMENT`):**

`configs` shape:
| field | type | note |
|---|---|---|
| `config_id` | INTEGER PK AUTOINCREMENT | id convention matches `runs.run_id` |
| `name` | TEXT NOT NULL | unique human label |
| `feature_set` | TEXT NOT NULL | feature-builder registry key (`basic`, `fracdiff_macro`) |
| `forecaster` | TEXT NOT NULL | forecaster registry key (`ewma`, `jepa`) |
| `view_scenario_id` | INTEGER NOT NULL → `view_scenarios` | the blend's view source |
| `optimizer` | TEXT NOT NULL | optimizer registry key (`mvu`, `min_variance`) |
| `params_json` | TEXT | JSON of per-stage param overrides (e.g. `{"optimizer":{"min_weight":0.05}}`) |
| `is_central` | INTEGER 0/1 NOT NULL | exactly one row = 1 (guarded set-central) |
| `created_at_utc` | TEXT NOT NULL | |

`view_scenarios` shape:
| field | type | note |
|---|---|---|
| `view_scenario_id` | INTEGER PK AUTOINCREMENT | |
| `name` | TEXT NOT NULL | |
| `kind` | TEXT NOT NULL CHECK IN (`forecast`,`historical`,`counterfactual`) | |
| `payload_json` | TEXT | `forecast`: `{}` (views derived from the run's forecaster); `historical`: `{"window_start":…,"window_end":…}`; `counterfactual`: `{asset:{"mu":…,"confidence":…}}` |
| `created_at_utc` | TEXT NOT NULL | |

`runs` additions:
| column | type | note |
|---|---|---|
| `config_id` | INTEGER → `configs`, nullable | the Config that produced the run; backfilled to the central Config on the v1→v2 migrate |

- **`backend/abe/config.py`** — `Config` (feature_set, forecaster, view_scenario, optimizer, params,
  `is_central`) + `ViewScenario` (`kind ∈ {forecast, historical, counterfactual}`, payload) typed
  entities; storage CRUD; the seeded default central Config.
- **`backend/abe/registry.py`** — four registries (feature-builder / forecaster / view-source /
  optimizer): `key → (factory, param_schema)`; a resolver that turns a `Config` into concrete stage
  callables; param-schema surfaced to the UI (extends the Track 1 `EXPLANATIONS` idea).
- **`backend/abe/migrations.py`** (or a `storage.migrations` submodule) — ordered forward-only
  migrations applied by `ensure_schema`; each migration is `(from_version, fn(conn))`.
- **`backend/abe/blend/views.py`** — the three view providers (forecast/historical/counterfactual)
  behind a common interface producing `{asset: absolute_view}` + confidences for `bl_blend`.
- **`backend/abe/optimize/`** — a second optimizer module (`min_variance.py`) + the MVU `min_weight`
  floor.
- **Frontend**: `router` wiring; `DashboardView` (central), `StageDetailTab` (per stage, with
  impl+param dropdowns), `CompareView`, `ScenarioEditor`; a `configs`/`scenarios` API client.

## 6. Design Decisions

- **SQLite, evolved (not Postgres).** SQLite WAL already fits local single-user and preserves the
  zero-setup / no-Docker / `127.0.0.1` property. Track 2 adds a **forward-only versioned migration
  framework** (today there is none) rather than a new engine. Alternative (Postgres) rejected: it
  needs a running service and a connection-pool story the single-writer model doesn't want.
- **Central config on the loop; others on-demand + cached.** The 5-minute always-on loop recomputes
  only the central Config (the "portfolio you'd buy"), keeping the single-writer loop cheap and the
  central answer unmistakable. Non-central Configs and what-if scenarios are computed when the user
  opens a comparison / triggers a scenario, dispatched through the **same one executor** (so
  single-flight + one-writer discipline is preserved) and cached in the DB by `config_id`.
  Alternative (run all configs every tick) rejected: N× compute on a deliberately single-threaded
  writer, and it blurs which allocation is "the" one.
- **Config = a recipe; ViewScenario = a view set.** A `Config` names one implementation per stage +
  params; exactly one Config is `is_central`. A `ViewScenario` is a named set of Black-Litterman
  views with a `kind`: `forecast` (today's behavior — views derived from the forecaster),
  `historical` (views seeded from a past window's realized returns), `counterfactual` (hand-authored
  absolute views, e.g. SPY +10% at chosen confidence). A Config's blend stage references a
  ViewScenario. "Compare" = the latest run of N Configs, side by side, central marked.
- **Registry mirrors the WorldModel seam.** Each stage gets a string-keyed registry with a declared
  param schema — the exact generalization of the existing EWMA/JEPA `WorldModel` pattern and the
  Track 1 `EXPLANATIONS` registry, so the UI dropdowns + param forms are generic.
- **Backward-compatibility is a golden test, not a promise.** The seeded central Config (basic
  features + EWMA + forecast views + MVU with today's params) must produce **byte-identical**
  `run_stages` to the pre-refactor pipeline on a seeded db — asserted before the refactor merges.
- **AGG=0% addressed via the optimizer seam.** The pluggable optimizer makes the realism fix
  natural: a `min_weight` box floor on MVU and a **min-variance** alternative that won't zero bonds.
  Left as a selectable choice, not forced on the central Config.
- **Frontend gains `react-router-dom`.** A dashboard + per-stage detail tabs + compare + scenario
  editor is 5+ views; hand-rolled tab state is worse than the standard router. It's a small,
  well-understood dep and the only new frontend dependency.

## 7. Build Steps

### Sub-phase 2A — Foundation (data model, migrations, Config-driven pipeline)

### Step 16: Versioned migration framework + schema v2
- **Problem:** Add a forward-only migration framework to `storage.py` (ordered `(from_version, fn)` migrations applied by `ensure_schema`, replacing the bare `user_version` stamp) and schema v2: new `configs` and `view_scenarios` tables + a nullable `config_id` column on `runs`. A migration must upgrade an existing v1 db (backfilling `runs.config_id` to the seeded central Config) AND a fresh db must build v2 directly.
- **Type:** code
- **Issue:** #22
- **Flags:** --reviewers code
- **Produces:** `backend/abe/migrations.py` (or `storage` submodule), v2 DDL, `SCHEMA_VERSION=2`.
- **Done when:** a test migrates a seeded v1 db to v2 with zero row loss and `config_id` backfilled; a fresh db opens at v2; `uv run pytest tests/test_storage.py` green.
- **Depends on:** none
- **Status:** DONE (2026-07-08)

### Step 17: Config + ViewScenario domain model + storage CRUD + seeded central Config
- **Problem:** Add `backend/abe/config.py` with typed `Config` and `ViewScenario` entities + storage CRUD (through the existing coercion boundary). Seed the default **central** Config (basic features + EWMA + forecast views + MVU, today's params) and a default `forecast` ViewScenario on first migration.
- **Type:** code
- **Issue:** #23
- **Flags:** --reviewers code
- **Produces:** `backend/abe/config.py`, CRUD helpers, seed logic wired into the Step 16 migration.
- **Done when:** round-trip CRUD tests pass; exactly one `is_central` Config exists after a fresh migrate; `uv run pytest` green.
- **Depends on:** 16
- **Status:** DONE (2026-07-08)

### Step 18: Stage registries + param schemas
- **Problem:** Add `backend/abe/registry.py` with four string-keyed registries (feature-builder, forecaster, view-source, optimizer), each entry declaring a factory + a param schema. Register the current impls: `basic` features, `ewma`+`jepa` forecasters, `forecast` view source, `mvu` optimizer. A resolver turns a `Config` into concrete stage callables.
- **Type:** code
- **Issue:** #24
- **Flags:** --reviewers code
- **Produces:** `backend/abe/registry.py`, registrations for existing impls, a `resolve(config)` function.
- **Done when:** registry tests assert each key resolves to a working callable and its param schema; `resolve(central_config)` returns the V1 stack; `uv run pytest` green.
- **Depends on:** 17
- **Status:** DONE (2026-07-08)

### Step 19: Refactor `run_pipeline` to run a resolved Config (parity golden)
- **Problem:** Refactor `run_pipeline` to accept a `Config` (resolve stage impls via the registry) and persist `config_id` on the `runs` row. Update the scheduler to run the **central** Config (`SchedulerConfig` carries the Config; keep a back-compat shim for the `model` seam used by tests). The central Config must reproduce V1 behavior exactly.
- **Type:** code
- **Issue:** #25
- **Flags:** --reviewers code
- **Produces:** refactored `pipeline.py`, `scheduler.py`, `api.py` startup wiring.
- **Done when:** a **parity golden test** asserts the central Config produces byte-identical `run_stages` detail to the pre-refactor pipeline on a seeded db; `uv run pytest`, `uv run mypy backend`, `uv run ruff check .` green.
- **Depends on:** 18
- **Status:** DONE (2026-07-08)

### Step 20: Config pipeline smoke gate
- **Problem:** End-to-end smoke: run the real `run_pipeline` with the seeded central Config against the real `data/abe.db` and assert one full cycle completes with all six stages `ok` and a `config_id`-tagged run row — the producer→consumer gate before any observation work.
- **Type:** code
- **Issue:** #26
- **Flags:** --reviewers code
- **Produces:** a `-m smoke`-marked end-to-end test (extends `scripts/smoke.py` / the smoke suite).
- **Done when:** `uv run pytest -m smoke` green on the real db (never skips vacuously).
- **Depends on:** 19
- **Status:** DONE (2026-07-08)

### Sub-phase 2B — Pluggable stages + new implementations

### Step 21: On-demand config-run path (cached, tagged)
- **Problem:** Add the ability to compute a run for a **non-central** `config_id` on demand: an API trigger + a scheduler dispatch that runs through the **same single executor** (preserving single-flight/one-writer), tags the run with `config_id`, and caches it. Reuse the freshness gate so an unchanged-data re-request returns the cached run.
- **Type:** code
- **Issue:** #27
- **Flags:** --reviewers code
- **Produces:** `POST /api/configs/{id}/run` (or equivalent) + scheduler dispatch; cache/read helpers.
- **Done when:** a test triggers a non-central config run, reads it back tagged by `config_id`, and a second same-data request is served from cache; single-writer invariant asserted; `uv run pytest` green.
- **Depends on:** 19
- **Status:** DONE (2026-07-08)
- **Note (for Step 25):** on-demand runs reject the central config id (409 — central runs via the loop). The per-config cache (`cached_config_run`) is keyed on data watermarks only, NOT the recipe — Step 25's `update_config` API MUST invalidate the cache on a recipe edit (documented on `cached_config_run`).

### Step 22: View scenarios — historical + counterfactual providers + library
- **Problem:** Make the blend's view source pluggable behind `backend/abe/blend/views.py`: `forecast` (current), `historical` (absolute views from a chosen past window's realized returns), `counterfactual` (hand-authored absolute views + confidences, e.g. SPY +10%). Add a pre-programmed library + CRUD for on-the-fly scenarios. `bl_blend` consumes the provider's `{asset: view}` + confidences.
- **Type:** code
- **Issue:** #28
- **Flags:** --reviewers code
- **Produces:** `backend/abe/blend/views.py`, three providers, library seeds, ViewScenario CRUD wiring.
- **Done when:** golden tests — `forecast` reproduces today's views; a counterfactual view materially moves the posterior toward it; a historical window yields the expected sign; `uv run pytest` green.
- **Depends on:** 18, 21
- **Status:** DONE (2026-07-08)

<!-- autofix-applied: 2026-07-08 -->
### Step 23: Optimizer variants — MVU min-weight floor + min-variance
- **Problem:** Add a `min_weight` box-floor parameter to `optimize_weights` (addresses V1 AGG=0% realism) and register a second optimizer — **min-variance** (minimizes wᵀΣw subject to the same long-only/Σ=1/box constraints; it naturally holds low-volatility bonds, directly countering the V1 AGG=0%; risk-parity is a later add) — behind the optimizer registry. Both selectable per Config; the central Config's optimizer stays MVU unless the operator changes it.
- **Type:** code
- **Issue:** #29
- **Flags:** --reviewers code
- **Produces:** `min_weight` support in `optimize/mvu.py`, a new optimizer module, registrations.
- **Done when:** a test shows `min_weight>0` yields a non-zero AGG; the alternative optimizer produces a valid long-only Σ=1 allocation; anchor tests (no-view ⇒ prior, garbage scored low) still hold; `uv run pytest` green.
- **Depends on:** 18
- **Status:** DONE (2026-07-08)

### Step 24: Selectable feature sets
- **Problem:** Generalize `_stage_features` to resolve its feature builder from the registry, exposing the existing `build_features` (frac-diff + macro) path as a selectable feature set alongside `basic` (log_return + realized_vol). The central Config keeps `basic`.
- **Type:** code
- **Issue:** #30
- **Flags:** --reviewers code
- **Produces:** feature-builder registrations; `_stage_features` reads the resolved builder.
- **Done when:** a test runs the pipeline under both feature sets and asserts the feature card/detail reflects the chosen set; `basic` remains byte-identical to V1; `uv run pytest` green.
- **Depends on:** 18
- **Status:** DONE (2026-07-08)

### Sub-phase 2C — Compare + scenario UI

### Step 25: Config / scenario / compare API
- **Problem:** Backend API for the UI: list/create/update/delete Configs and ViewScenarios; set-central (a deliberate, guarded action); trigger a config run (Step 21); a **comparison** endpoint returning N Configs' latest runs (weights + key stage facts) with the central clearly flagged. Request/response shapes follow the existing `api.py` typed-dict convention (e.g. `/api/runs/latest`, `/api/explain`); the CRUD payloads mirror the `configs`/`view_scenarios` shapes in §5. The one new aggregate shape — `GET /api/compare?config_ids=…` — returns `{"central_config_id": <int>, "configs": [{"config_id", "name", "is_central", "weights": {asset: w}, "objective": {...}, "run_id", "finished_at_utc"}, …]}`.
- **Type:** code
- **Issue:** #31
- **Flags:** --reviewers code
- **Produces:** the `/api/configs`, `/api/scenarios`, `/api/compare` route group.
- **Done when:** API tests cover CRUD, set-central invariant (exactly one central), and a comparison payload shape; `uv run pytest` green.
- **Depends on:** 21, 22, 23, 24

### Step 26: Frontend shell — router, central dashboard, per-stage detail tabs
- **Problem:** Introduce `react-router-dom`. Keep the central-scenario dashboard as the default route; add per-stage detail tabs (features / forecast / blend / optimize) with dropdowns to pick the implementation + edit its params (param schema from the registry), previewing the effect on the active Config.
- **Type:** code
- **Issue:** #32
- **Flags:** --reviewers full --start-cmd "npm run build --prefix frontend && uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140" --url http://127.0.0.1:8140
- **Produces:** router wiring, `DashboardView`, `StageDetailTab`, config/scenario API client.
- **Done when:** runtime reviewers confirm the dashboard still renders the central allocation and each detail tab lists impls + editable params; tsc + vite build clean.
- **Depends on:** 25

### Step 27: Compare view
- **Problem:** A view that shows N Configs' latest allocations side by side (weights + μ/σ + objective), the **central** Config unmistakably marked, with a control to run/refresh a non-central Config on demand.
- **Type:** code
- **Issue:** #33
- **Flags:** --reviewers full --start-cmd "npm run build --prefix frontend && uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140" --url http://127.0.0.1:8140
- **Produces:** `CompareView` + its route.
- **Done when:** runtime reviewers confirm ≥2 Configs render side by side with the central flagged and an on-demand refresh works; tsc + vite clean.
- **Depends on:** 26

### Step 28: View-scenario authoring UI
- **Problem:** UI to build/edit view scenarios: author counterfactual absolute views (asset, return, confidence), pick a historical window, and browse/apply the pre-programmed library; attach a scenario to a Config's blend stage.
- **Type:** code
- **Issue:** #34
- **Flags:** --reviewers full --start-cmd "npm run build --prefix frontend && uv run uvicorn abe.api:app --host 127.0.0.1 --port 8140" --url http://127.0.0.1:8140
- **Produces:** `ScenarioEditor` + its route.
- **Done when:** runtime reviewers confirm authoring a "SPY +10%" counterfactual and attaching it changes that Config's blend/optimize output; tsc + vite clean.
- **Depends on:** 27

### Step 29: Soak / observation (always-on + on-demand under load)
- **Problem:** Run the engine with the central Config on the 5-minute loop for ≥4h while periodically exercising on-demand comparisons and scenario runs. Observe: DB growth + WAL checkpointing, single-writer contention between the loop and on-demand runs, cache correctness (config runs served from cache when data is unchanged), no phantom `running` rows, memory. Capture a findings log.
- **Type:** wait
- **Issue:** #35
- **Produces:** `docs/soak/track2-soak-<date>.md`.
- **Done when:** ≥4h continuous; the loop keeps the central Config fresh; on-demand runs interleave without writer errors; no unhandled crash; findings captured. (Resume in a fresh session after the wait.)
- **Depends on:** 25

### Step 30: Operator UAT — compare + scenario walkthrough
- **Problem:** Operator-driven acceptance of the compare + scenario UI: pick alternative impls per stage, author a counterfactual, run a comparison, confirm the central scenario stays unambiguous, and sanity-check the AGG=0% fix (a `min_weight` Config shows a non-zero bond weight).
- **Type:** operator
- **Issue:** #36
- **Done when:** operator confirms the four checks; any defects filed as follow-ups.
- **Depends on:** 26, 27, 28

## 8. Risks and Open Questions

| Item | Risk | Mitigation |
|---|---|---|
| Migration on a live db | A bad migration corrupts the real `data/abe.db` (price backfill). | Forward-only migrations tested on a copied seeded v1 db first; back up `data/abe.db` before first real migrate; migration is idempotent + wrapped in a transaction. |
| `run_pipeline` refactor drift | The Config-driven refactor silently changes V1 output. | Byte-identical parity golden (Step 19) gating the merge; grep confirmed the single call site + tests. |
| Single-writer contention | On-demand runs + the 5-min loop compete for the one writer. | All runs dispatched through the SAME one executor (FIFO), preserving single-flight; on-demand runs queue behind the loop — asserted in Step 21 + observed in Step 29. |
| On-demand compute cost | Comparisons could trigger many expensive JEPA/frac-diff runs. | Cache by `config_id` + freshness gate (only recompute on new data); the loop still only runs the central Config. |
| Counterfactual view abuse | An extreme hand-authored view produces a nonsensical allocation. | Views flow through the existing BL Idzorek confidence clamp; the box + Σ=1 constraints bound the optimizer; the caveat UI already flags overlap. |
| UI scope creep | Detail tabs + compare + authoring is a lot of frontend. | Sub-phase 2C is last; each view is its own step with runtime review; the central dashboard is unchanged as the default. |
| Comparison persistence | A saved/named-comparison feature might later want its own table. | Decided: comparisons are **derived on the fly** (the latest run per Config) — no `comparisons` table in Track 2. Add one only if saved/named comparisons are requested later. |

## 9. Testing Strategy

- **Migration**: golden test upgrading a seeded v1 db → v2 (zero row loss, `config_id` backfilled)
  and a fresh-db-at-v2 test. Back up the real db before the first production migrate.
- **Parity**: the central Config produces byte-identical `run_stages` to the pre-refactor pipeline
  (Step 19) — the load-bearing regression gate for the refactor.
- **Registries**: every registered key resolves to a working callable with its declared param
  schema; `resolve(central_config)` yields the V1 stack.
- **Stage impls**: view-source goldens (forecast reproduces today; counterfactual moves the
  posterior; historical sign); optimizer goldens (`min_weight>0` ⇒ non-zero AGG; alternative
  optimizer valid); feature-set goldens (`basic` unchanged; frac-diff+macro selectable). Keep the
  V1 measurement-validity anchors (no-view ⇒ prior within 1e-6; garbage scored low).
- **On-demand + single-writer**: a config run interleaves with the loop through one executor without
  writer errors; cache hit on unchanged data.
- **API**: CRUD, set-central invariant (exactly one central), comparison payload shape.
- **Smoke**: `-m smoke` runs one real central-Config cycle end-to-end on `data/abe.db` (Step 20).
- **Soak (Step 29)**: the only way to expose loop-vs-on-demand contention, WAL growth, and
  cache correctness over wall-clock time — component tests cannot substitute for it.
- **Existing tests that will move/break**: `test_pipeline.py` (Config-based `run_pipeline`
  signature), `test_scheduler.py` + `tests/seeding.py` (`SchedulerConfig` Config field),
  `test_storage.py` (schema v2 + migrations), `test_optimize_mvu.py` (`min_weight` param),
  `test_blend_black_litterman.py` (pluggable view source). Update in the step that changes each
  contract; keep the `is`-identity constant tests.
