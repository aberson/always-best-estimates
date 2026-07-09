/**
 * Typed client for the plan section 6 API route contract — the ONLY module
 * that talks to the backend.
 *
 * All URLs are relative: the Vite dev server (127.0.0.1:5174) proxies
 * `/api` -> 127.0.0.1:8140 (vite.config.ts), and in production FastAPI
 * serves the BUILT frontend from the same origin, so relative paths work in
 * both modes with zero configuration.
 *
 * Types mirror the actual api.py response construction (plain dicts over the
 * stored rows; `detail_json` is parsed server-side and exposed as a
 * structured `detail` member, which we type as `unknown` — StageCard narrows
 * it defensively and falls back to raw JSON on any unexpected shape).
 */

/** `runs.status` values (storage schema CHECK constraint). */
export type RunStatus = "running" | "ok" | "error" | "skipped";

/** `run_stages.status` values. */
export type StageStatus = "ok" | "error" | "skipped";

/** One `runs` row as served by api.py (`_run_dict`). */
export interface RunInfo {
  run_id: number;
  started_at_utc: string;
  finished_at_utc: string | null;
  status: RunStatus;
  trigger: string;
  error_text: string | null;
}

/** One `run_stages` row as served by api.py (`_stage_dicts`). */
export interface StageRow {
  stage: string;
  status: StageStatus;
  started_at_utc: string | null;
  finished_at_utc: string | null;
  /** Parsed `detail_json` (object), `null` when absent, or a
   *  `{parse_error: ...}` marker for a corrupt stored row. */
  detail: unknown;
}

/** `GET /api/runs/latest` — the latest ok run + its stages (poll target). */
export interface LatestRunResponse {
  run: RunInfo;
  stages: StageRow[];
}

/** One `target_weights` row inside a history run (`_weight_dicts`). */
export interface WeightRow {
  asset: string;
  weight: number;
  prev_weight: number | null;
  turnover: number;
  relaxed_turnover: boolean;
}

/** One run entry in `GET /api/history` (run row + its weights). */
export interface HistoryRun extends RunInfo {
  target_weights: WeightRow[];
}

/** `GET /api/history?limit=N` — recent runs, newest first. */
export interface HistoryResponse {
  runs: HistoryRun[];
}

/** `POST /api/runs/trigger` -> `202 {run_id, already_running}`. */
export interface TriggerResponse {
  run_id: number;
  already_running: boolean;
}

/** One entry served by `GET /api/explain` (calc.py `Explanation.payload()`;
 *  the registry keys by quantity, so `key` is the map key, not a field). */
export interface Explanation {
  label: string;
  formula: string;
  description: string;
  example: string;
  unit: string | null;
  window: string | null;
}

/** `GET /api/explain` — the calc.py explanation registry, keyed by quantity. */
export interface ExplainResponse {
  explanations: Record<string, Explanation>;
}

/** Non-OK HTTP response (other than the latest-run 404, which is a state). */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestJson(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  return response;
}

/**
 * The UI poll target. Resolves `null` on 404 (no successful run yet — the
 * empty state, NOT an error); throws `ApiError`/`TypeError` on real failures
 * (server error, backend unreachable) so the caller can show the banner.
 */
export async function getLatestRun(): Promise<LatestRunResponse | null> {
  const response = await requestJson("/api/runs/latest");
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new ApiError(response.status, `GET /api/runs/latest failed: HTTP ${response.status}`);
  }
  return (await response.json()) as LatestRunResponse;
}

/** Recent runs (newest first), each with its target_weights rows. */
export async function getHistory(limit = 20): Promise<HistoryResponse> {
  const response = await requestJson(`/api/history?limit=${encodeURIComponent(limit)}`);
  if (!response.ok) {
    throw new ApiError(response.status, `GET /api/history failed: HTTP ${response.status}`);
  }
  return (await response.json()) as HistoryResponse;
}

/**
 * Run the pipeline now. `force=true` bypasses the freshness gate. V1 runs
 * the pipeline synchronously inside the request, so the returned `run_id`
 * refers to a FINISHED run; Step 11 keeps the same shape but may coalesce to
 * `already_running: true`.
 */
export async function triggerRun(force = false): Promise<TriggerResponse> {
  const response = await requestJson("/api/runs/trigger", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
  if (response.status !== 202) {
    throw new ApiError(response.status, `POST /api/runs/trigger failed: HTTP ${response.status}`);
  }
  return (await response.json()) as TriggerResponse;
}

/**
 * The calc.py explanation registry (formula + worked example per quantity),
 * keyed by quantity. Static content — fetched once by the App and passed to
 * the cards for the inline "how is this computed?" expanders. A failure here
 * is non-fatal: the cards simply render without explanations.
 */
export async function getExplanations(): Promise<Record<string, Explanation>> {
  const response = await requestJson("/api/explain");
  if (!response.ok) {
    throw new ApiError(response.status, `GET /api/explain failed: HTTP ${response.status}`);
  }
  return ((await response.json()) as ExplainResponse).explanations;
}

// ==========================================================================
// Track 2: pluggable-per-stage config / scenario / compare API (Step 25).
// ==========================================================================

/** The fixed 3-asset universe (mirrors backend `abe.constants.UNIVERSE`). */
export const UNIVERSE = ["SPY", "ACWI", "AGG"] as const;

/** One tunable parameter of a stage impl (registry `param_schema` entry). */
export interface RegistryParam {
  name: string;
  type: string; // "float" | "int" | "str" | "bool"
  default: unknown;
  description: string;
}

/** One registered stage impl: its human description + declared param schema. */
export interface RegistryEntry {
  description: string;
  params: RegistryParam[];
}

/** A single registry: impl key -> entry (registries_manifest per-stage value). */
export type RegistryMap = Record<string, RegistryEntry>;

/** The four stage registries served by `GET /api/registries`. */
export interface Registries {
  feature_set: RegistryMap;
  forecaster: RegistryMap;
  view_source: RegistryMap;
  optimizer: RegistryMap;
}

/** `GET /api/registries` envelope. */
export interface RegistriesResponse {
  registries: Registries;
}

/** One `configs` row (a named per-stage pipeline recipe; plan §5). */
export interface Config {
  config_id: number;
  name: string;
  feature_set: string;
  forecaster: string;
  view_scenario_id: number;
  optimizer: string;
  /** Per-stage param overrides, structurally `{stage: {param: value}}`. */
  params: Record<string, unknown>;
  is_central: boolean;
  created_at_utc: string | null;
  updated_at_utc: string | null;
}

/** `GET /api/configs` envelope. */
export interface ConfigsResponse {
  configs: Config[];
}

/** `POST /api/configs` body. */
export interface ConfigCreate {
  name: string;
  feature_set: string;
  forecaster: string;
  view_scenario_id: number;
  optimizer: string;
  params: Record<string, unknown>;
}

/** `PATCH /api/configs/{id}` body — any subset of the recipe fields. */
export type ConfigPatch = Partial<ConfigCreate>;

/** `POST /api/configs/{id}/run` -> `{run_id, config_id, cached}`. */
export interface ConfigRunResponse {
  run_id: number;
  config_id: number;
  cached: boolean;
}

/** The valid `view_scenarios.kind` values (backend CHECK constraint). */
export type ScenarioKind = "forecast" | "historical" | "counterfactual";

/** One `view_scenarios` row (a named Black-Litterman view set; plan §5). */
export interface Scenario {
  view_scenario_id: number;
  name: string;
  kind: ScenarioKind;
  /** kind-dependent: forecast -> `{}`; historical -> `{window_start?, window_end?}`;
   *  counterfactual -> `{asset: {mu, confidence}}`. */
  payload: Record<string, unknown>;
  created_at_utc: string | null;
}

/** `GET /api/scenarios` envelope. */
export interface ScenariosResponse {
  scenarios: Scenario[];
}

/** `POST /api/scenarios` body. */
export interface ScenarioCreate {
  name: string;
  kind: ScenarioKind;
  payload: Record<string, unknown>;
}

/** `PATCH /api/scenarios/{id}` body (`kind` is immutable — author anew). */
export interface ScenarioPatch {
  name?: string;
  payload?: Record<string, unknown>;
}

/** One config's row in `GET /api/compare` (its latest ok run's allocation). */
export interface CompareConfig {
  config_id: number;
  name: string;
  is_central: boolean;
  weights: Record<string, number>;
  objective: Record<string, unknown> | null;
  run_id: number | null;
  finished_at_utc: string | null;
}

/** `GET /api/compare?config_ids=…` — N configs side by side + the central id. */
export interface CompareResponse {
  central_config_id: number;
  configs: CompareConfig[];
}

/** Build a JSON-body `RequestInit` for POST/PATCH (Content-Type merged by
 *  `requestJson`, which keeps the Accept header). */
function jsonBody(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

/** Pull FastAPI's `{detail: ...}` message out of a non-OK response, defensively
 *  (the body may be absent or not JSON). */
async function detailOf(response: Response): Promise<string | null> {
  try {
    const body = (await response.json()) as unknown;
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as Record<string, unknown>)["detail"];
      if (typeof detail === "string") {
        return detail;
      }
      if (detail !== undefined) {
        return JSON.stringify(detail);
      }
    }
  } catch {
    /* body absent or not JSON — fall back to the bare status */
  }
  return null;
}

/** Map a non-OK response to an `ApiError`, surfacing the server's `detail`
 *  (e.g. the 409 "config is central" / "referenced by N config(s)" reasons). */
async function apiErrorFrom(response: Response, label: string): Promise<ApiError> {
  const detail = await detailOf(response);
  return new ApiError(
    response.status,
    detail !== null
      ? `${label}: ${detail} (HTTP ${response.status})`
      : `${label} failed: HTTP ${response.status}`,
  );
}

/** The four stage registries (keys + param schemas) for the UI dropdowns. */
export async function getRegistries(): Promise<Registries> {
  const response = await requestJson("/api/registries");
  if (!response.ok) {
    throw await apiErrorFrom(response, "GET /api/registries");
  }
  return ((await response.json()) as RegistriesResponse).registries;
}

/** All configs (recipes), ordered by id. */
export async function getConfigs(): Promise<Config[]> {
  const response = await requestJson("/api/configs");
  if (!response.ok) {
    throw await apiErrorFrom(response, "GET /api/configs");
  }
  return ((await response.json()) as ConfigsResponse).configs;
}

/** One config by id (404 -> ApiError). */
export async function getConfig(configId: number): Promise<Config> {
  const response = await requestJson(`/api/configs/${configId}`);
  if (!response.ok) {
    throw await apiErrorFrom(response, `GET /api/configs/${configId}`);
  }
  return (await response.json()) as Config;
}

/** Create a non-central config (201). Name must be unique (409 on conflict). */
export async function createConfig(body: ConfigCreate): Promise<Config> {
  const response = await requestJson("/api/configs", jsonBody("POST", body));
  if (response.status !== 201) {
    throw await apiErrorFrom(response, "POST /api/configs");
  }
  return (await response.json()) as Config;
}

/** Partial-update a config's recipe fields. */
export async function updateConfig(configId: number, body: ConfigPatch): Promise<Config> {
  const response = await requestJson(`/api/configs/${configId}`, jsonBody("PATCH", body));
  if (!response.ok) {
    throw await apiErrorFrom(response, `PATCH /api/configs/${configId}`);
  }
  return (await response.json()) as Config;
}

/** Delete a config (204). 409 if it is central or referenced by runs. */
export async function deleteConfig(configId: number): Promise<void> {
  const response = await requestJson(`/api/configs/${configId}`, { method: "DELETE" });
  if (response.status !== 204) {
    throw await apiErrorFrom(response, `DELETE /api/configs/${configId}`);
  }
}

/** Promote a config to central (the deliberate operator action). */
export async function setConfigCentral(configId: number): Promise<Config> {
  const response = await requestJson(`/api/configs/${configId}/central`, { method: "POST" });
  if (!response.ok) {
    throw await apiErrorFrom(response, `POST /api/configs/${configId}/central`);
  }
  return (await response.json()) as Config;
}

/** Run a config on demand (blocks until done). 409 for the central config. */
export async function runConfig(configId: number, force = false): Promise<ConfigRunResponse> {
  const response = await requestJson(
    `/api/configs/${configId}/run`,
    jsonBody("POST", { force }),
  );
  if (!response.ok) {
    throw await apiErrorFrom(response, `POST /api/configs/${configId}/run`);
  }
  return (await response.json()) as ConfigRunResponse;
}

/** All view scenarios (seeded library + operator-authored), ordered by id. */
export async function getScenarios(): Promise<Scenario[]> {
  const response = await requestJson("/api/scenarios");
  if (!response.ok) {
    throw await apiErrorFrom(response, "GET /api/scenarios");
  }
  return ((await response.json()) as ScenariosResponse).scenarios;
}

/** Author a new view scenario (201). 400 on an invalid `kind`. */
export async function createScenario(body: ScenarioCreate): Promise<Scenario> {
  const response = await requestJson("/api/scenarios", jsonBody("POST", body));
  if (response.status !== 201) {
    throw await apiErrorFrom(response, "POST /api/scenarios");
  }
  return (await response.json()) as Scenario;
}

/** Update a scenario's name and/or payload (`kind` immutable). */
export async function updateScenario(
  scenarioId: number,
  body: ScenarioPatch,
): Promise<Scenario> {
  const response = await requestJson(
    `/api/scenarios/${scenarioId}`,
    jsonBody("PATCH", body),
  );
  if (!response.ok) {
    throw await apiErrorFrom(response, `PATCH /api/scenarios/${scenarioId}`);
  }
  return (await response.json()) as Scenario;
}

/** Delete a scenario (204). 409 if a config still references it. */
export async function deleteScenario(scenarioId: number): Promise<void> {
  const response = await requestJson(`/api/scenarios/${scenarioId}`, { method: "DELETE" });
  if (response.status !== 204) {
    throw await apiErrorFrom(response, `DELETE /api/scenarios/${scenarioId}`);
  }
}

/** Compare N configs' latest allocations side by side (central id flagged). */
export async function getCompare(configIds: readonly number[]): Promise<CompareResponse> {
  const query = configIds.join(",");
  const response = await requestJson(`/api/compare?config_ids=${encodeURIComponent(query)}`);
  if (!response.ok) {
    throw await apiErrorFrom(response, "GET /api/compare");
  }
  return (await response.json()) as CompareResponse;
}
