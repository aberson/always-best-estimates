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
