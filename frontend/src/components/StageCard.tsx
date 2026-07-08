/**
 * ONE component rendering any pipeline stage card from its `run_stages` row.
 *
 * The card head shows the stage name, a status pill (ok green / error red /
 * skipped amber) and the stage duration. The body dispatches on the stage
 * name to a per-stage renderer of the pipeline.py `detail_json` payloads
 * (the "UI card contract" section of pipeline.py's docstring). Every
 * renderer narrows the untyped `detail` defensively and returns `null` when
 * the payload does not have the expected shape, in which case the card falls
 * back to raw pretty-printed JSON — a card is NEVER blank.
 *
 * `error` stages are first-class: `detail.error` ("ExcType: message") is
 * rendered prominently with the exception class pulled out.
 */

import { useState, type ReactNode } from "react";
import type { Explanation, StageRow } from "../api";

type JsonRecord = Record<string, unknown>;

const STAGE_TITLES: Record<string, string> = {
  freshness: "Freshness",
  ingest: "Ingest",
  features: "Features",
  forecast: "Forecast",
  blend: "BL Blend",
  optimize: "Optimize",
};

/** The optimizer card's mandated caveat (plan Step 10). */
const OVERLAP_CAVEAT =
  "Note: SPY overlaps ACWI (~60% US); combined equity exposure reads through both.";

/** Which calc.py explanation keys each card reveals in its inline
 *  "how is this computed?" expander (fed by GET /api/explain). */
const STAGE_EXPLAIN_KEYS: Record<string, string[]> = {
  features: ["log_return", "realized_vol"],
  forecast: ["ewma_mu", "forecast_sigma"],
  blend: ["bl_prior", "bl_view", "bl_confidence", "bl_posterior"],
  optimize: ["mvu_objective"],
};

// --------------------------------------------------------------------------
// Narrowing + formatting helpers (detail payloads arrive as unknown JSON)
// --------------------------------------------------------------------------

function asRecord(value: unknown): JsonRecord | null {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as JsonRecord;
  }
  return null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/** Generic scalar display: null/undefined -> em dash. */
function scalar(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (typeof value === "number" || typeof value === "string") {
    return String(value);
  }
  return JSON.stringify(value);
}

/** 0.3412 -> "34.1%" (weights, turnover). */
function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** 0.0123 -> "+1.23%" (returns keep their sign). */
function signedPct(value: number, digits = 2): string {
  const text = `${(value * 100).toFixed(digits)}%`;
  return value >= 0 ? `+${text}` : text;
}

function durationLabel(started: string | null, finished: string | null): string | null {
  if (started === null || finished === null) {
    return null;
  }
  const ms = Date.parse(finished) - Date.parse(started);
  if (!Number.isFinite(ms) || ms < 0) {
    return null;
  }
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function rawJsonFallback(detail: unknown): ReactNode {
  if (detail === null || detail === undefined) {
    return <p className="muted">no detail recorded</p>;
  }
  return <pre className="raw-json">{JSON.stringify(detail, null, 2)}</pre>;
}

// --------------------------------------------------------------------------
// Per-stage detail renderers (return null on unexpected shape -> raw JSON)
// --------------------------------------------------------------------------

function renderError(detail: unknown): ReactNode {
  const record = asRecord(detail);
  const message = record !== null ? asString(record["error"]) : null;
  if (message === null) {
    // An error stage without the expected {error: ...} payload still renders.
    return (
      <div className="error-body">
        <span className="error-class">error</span>
        {rawJsonFallback(detail)}
      </div>
    );
  }
  const colon = message.indexOf(":");
  const errorClass = colon > 0 ? message.slice(0, colon) : "Error";
  const errorText = colon > 0 ? message.slice(colon + 1).trim() : message;
  return (
    <div className="error-body">
      <span className="error-class">{errorClass}</span>
      <pre className="error-message">{errorText}</pre>
    </div>
  );
}

function renderFreshness(detail: JsonRecord, status: string): ReactNode {
  if (!("data_max_date" in detail)) {
    return null;
  }
  const lastOkRun = asNumber(detail["last_ok_run_id"]);
  return (
    <>
      {status === "skipped" ? (
        <p className="skip-reason">
          skipped &mdash; stored data unchanged since last ok run
          {lastOkRun !== null ? ` #${lastOkRun}` : ""}
        </p>
      ) : null}
      <table className="detail-table">
        <tbody>
          <tr>
            <th>data max date</th>
            <td>{scalar(detail["data_max_date"])}</td>
          </tr>
          <tr>
            <th>fetched watermark</th>
            <td>{scalar(detail["data_fetched_at"])}</td>
          </tr>
          <tr>
            <th>last ok run</th>
            <td>{lastOkRun !== null ? `#${lastOkRun}` : "—"}</td>
          </tr>
          <tr>
            <th>last ok max date</th>
            <td>{scalar(detail["last_ok_data_max_date"])}</td>
          </tr>
          <tr>
            <th>last ok fetched</th>
            <td>{scalar(detail["last_ok_data_fetched_at"])}</td>
          </tr>
          <tr>
            <th>forced</th>
            <td>{scalar(detail["force"])}</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}

function renderMacroBadge(macro: JsonRecord): ReactNode {
  const enabled = macro["enabled"] === true;
  const code = asString(macro["code"]) ?? "unknown";
  const message = asString(macro["message"]) ?? undefined;
  if (enabled) {
    return (
      <span className="badge badge-ok" title={message}>
        Macro (FRED): enabled
      </span>
    );
  }
  // Degraded macro is a card FACT, not an error (pipeline.py contract).
  const label =
    code === "MACRO_DISABLED_NO_KEY"
      ? "Macro (FRED) disabled: no key"
      : `Macro (FRED) disabled: ${code}`;
  return (
    <span className="badge badge-amber" title={message}>
      {label}
    </span>
  );
}

function renderIngest(detail: JsonRecord): ReactNode {
  const prices = asRecord(detail["prices"]);
  const macro = asRecord(detail["macro"]);
  if (prices === null || macro === null) {
    return null;
  }
  return (
    <>
      <table className="detail-table">
        <thead>
          <tr>
            <th>asset</th>
            <th>rows</th>
            <th>range</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(prices).map(([asset, info]) => {
            const record = asRecord(info) ?? {};
            return (
              <tr key={asset}>
                <th>{asset}</th>
                <td>{scalar(record["rows"])}</td>
                <td>
                  {scalar(record["first_date"])} → {scalar(record["last_date"])}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="badge-row">
        {asString(detail["price_provider"]) !== null ? (
          <span className="badge">Prices: {asString(detail["price_provider"])}</span>
        ) : null}
        <span className="badge">served from: {scalar(detail["source"])}</span>
        {renderMacroBadge(macro)}
      </div>
    </>
  );
}

function renderFeatures(detail: JsonRecord): ReactNode {
  const latest = asRecord(detail["latest"]);
  const windows = asRecord(detail["windows"]) ?? {};
  const names = Array.isArray(detail["features"])
    ? detail["features"].filter((name): name is string => typeof name === "string")
    : null;
  if (latest === null || names === null) {
    return null;
  }
  return (
    <table className="detail-table">
      <thead>
        <tr>
          <th>asset</th>
          <th>date</th>
          {names.map((name) => {
            const win = asString(windows[name]);
            return (
              <th key={name}>
                {name}
                {win !== null ? ` (${win})` : ""}
              </th>
            );
          })}
        </tr>
      </thead>
      <tbody>
        {Object.entries(latest).map(([asset, info]) => {
          const record = asRecord(info) ?? {};
          return (
            <tr key={asset}>
              <th>{asset}</th>
              <td>{scalar(record["date"])}</td>
              {names.map((name) => {
                const value = asNumber(record[name]);
                return <td key={name}>{value !== null ? value.toFixed(5) : scalar(record[name])}</td>;
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function renderForecast(detail: JsonRecord): ReactNode {
  const forecasts = asRecord(detail["forecasts"]);
  if (forecasts === null) {
    return null;
  }
  const horizon = asNumber(detail["horizon_days"]);
  return (
    <>
      <table className="detail-table">
        <thead>
          <tr>
            <th>asset</th>
            <th>{horizon !== null ? `μ (${horizon}d)` : "μ"}</th>
            <th>{horizon !== null ? `σ (${horizon}d)` : "σ"}</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(forecasts).map(([asset, info]) => {
            const record = asRecord(info) ?? {};
            const mu = asNumber(record["mu"]);
            const sigma = asNumber(record["sigma"]);
            return (
              <tr key={asset}>
                <th>{asset}</th>
                <td>{mu !== null ? signedPct(mu) : scalar(record["mu"])}</td>
                <td>{sigma !== null ? pct(sigma, 2) : scalar(record["sigma"])}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="badge-row">
        <span className="badge">model: {scalar(detail["model_version"])}</span>
      </div>
    </>
  );
}

function renderBlend(detail: JsonRecord): ReactNode {
  const posteriorMu = asRecord(detail["posterior_mu"]);
  const confidences = asRecord(detail["confidences"]);
  if (posteriorMu === null || confidences === null) {
    return null;
  }
  // The three Black-Litterman pieces: equilibrium prior (pi), the forecast
  // view (Q), and the blended posterior. prior/view are additive fields that
  // may be absent on rows written before this stage was enriched.
  const prior = asRecord(detail["prior"]);
  const view = asRecord(detail["view"]);
  return (
    <>
      <table className="detail-table">
        <thead>
          <tr>
            <th>asset</th>
            <th>prior &pi;</th>
            <th>view Q</th>
            <th>posterior &mu;</th>
            <th>confidence</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(posteriorMu).map(([asset, rawMu]) => {
            const mu = asNumber(rawMu);
            const pi = prior !== null ? asNumber(prior[asset]) : null;
            const q = view !== null ? asNumber(view[asset]) : null;
            const confidence = asNumber(confidences[asset]);
            return (
              <tr key={asset}>
                <th>{asset}</th>
                <td>{pi !== null ? signedPct(pi) : "—"}</td>
                <td>{q !== null ? signedPct(q) : "—"}</td>
                <td>{mu !== null ? signedPct(mu) : scalar(rawMu)}</td>
                <td>{confidence !== null ? pct(confidence, 0) : scalar(confidences[asset])}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="caveat">
        All annualized. View Q is the Forecast annualized; the posterior blends the
        market-equilibrium prior &pi; toward Q, weighted by each view&rsquo;s confidence.
      </p>
    </>
  );
}

function renderOptimize(detail: JsonRecord): ReactNode {
  const weights = asRecord(detail["weights"]);
  if (weights === null) {
    return null;
  }
  const objective = asRecord(detail["objective"]);
  const objWMax = objective !== null ? asNumber(objective["w_max"]) : null;
  const constraints = objective !== null && Array.isArray(objective["constraints"])
    ? (objective["constraints"] as unknown[]).map(String)
    : [];
  const turnover = asRecord(detail["turnover"]) ?? {};
  const prevWeights = asRecord(detail["prev_weights"]);
  const relaxed = detail["relaxed_turnover"] === true;
  const coldStart = detail["cold_start"] === true;
  const totalTurnover = Object.values(turnover)
    .map(asNumber)
    .filter((value): value is number => value !== null)
    .reduce((sum, value) => sum + value, 0);
  return (
    <>
      {/* THE WEIGHTS: the product of the whole pipeline, big and prominent. */}
      <div className="weights-row">
        {Object.entries(weights).map(([asset, rawWeight]) => {
          const weight = asNumber(rawWeight);
          return (
            <div className="weight-block" key={asset}>
              <span className="weight-value">
                {weight !== null ? pct(weight) : scalar(rawWeight)}
              </span>
              <span className="weight-asset">{asset}</span>
            </div>
          );
        })}
      </div>
      <table className="detail-table">
        <thead>
          <tr>
            <th>asset</th>
            <th>prev</th>
            <th>turnover</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(weights).map((asset) => {
            const prev = prevWeights !== null ? asNumber(prevWeights[asset]) : null;
            const assetTurnover = asNumber(turnover[asset]);
            return (
              <tr key={asset}>
                <th>{asset}</th>
                <td>{prev !== null ? pct(prev) : "—"}</td>
                <td>{assetTurnover !== null ? pct(assetTurnover, 2) : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {objective !== null ? (
        <div className="objective">
          <code className="objective-form">{scalar(objective["form"])}</code>
          <p className="muted">
            risk aversion &delta; = {scalar(objective["delta"])}, turnover penalty &gamma; ={" "}
            {scalar(objective["gamma_tc"])}, max weight ={" "}
            {objWMax !== null ? pct(objWMax, 0) : scalar(objective["w_max"])}
          </p>
          {constraints.length > 0 ? (
            <p className="muted">subject to: {constraints.join("; ")}</p>
          ) : null}
        </div>
      ) : null}
      <div className="badge-row">
        <span className="badge">total turnover: {pct(totalTurnover, 2)}</span>
        <span className="badge">solver: {scalar(detail["solver_status"])}</span>
        {relaxed ? <span className="badge badge-amber">turnover constraint relaxed</span> : null}
        {coldStart ? <span className="badge badge-amber">cold start (no previous weights)</span> : null}
      </div>
      <p className="caveat">{OVERLAP_CAVEAT}</p>
    </>
  );
}

function detailBody(stage: StageRow): ReactNode {
  if (stage.status === "error") {
    return renderError(stage.detail);
  }
  const record = asRecord(stage.detail);
  if (record !== null) {
    let body: ReactNode = null;
    switch (stage.stage) {
      case "freshness":
        body = renderFreshness(record, stage.status);
        break;
      case "ingest":
        body = renderIngest(record);
        break;
      case "features":
        body = renderFeatures(record);
        break;
      case "forecast":
        body = renderForecast(record);
        break;
      case "blend":
        body = renderBlend(record);
        break;
      case "optimize":
        body = renderOptimize(record);
        break;
      default:
        body = null;
    }
    if (body !== null) {
      return body;
    }
  }
  // Unknown stage, missing detail, or unexpected payload shape (including
  // api.py's {parse_error: ...} marker): raw JSON, never blank.
  return rawJsonFallback(stage.detail);
}

export default function StageCard({
  stage,
  explanations,
}: {
  stage: StageRow;
  explanations?: Record<string, Explanation>;
}) {
  const [showExplain, setShowExplain] = useState(false);
  const duration = durationLabel(stage.started_at_utc, stage.finished_at_utc);
  const pillClass =
    stage.status === "ok"
      ? "pill pill-ok"
      : stage.status === "error"
        ? "pill pill-error"
        : stage.status === "skipped"
          ? "pill pill-skipped"
          : "pill";
  // Inline "how is this computed?" entries for this stage (calc.py registry).
  const explainEntries = (STAGE_EXPLAIN_KEYS[stage.stage] ?? [])
    .map((key) => (explanations ? explanations[key] : undefined))
    .filter((entry): entry is Explanation => entry !== undefined);
  return (
    <section className={`card card-${stage.status}`}>
      <header className="card-head">
        <h2>{STAGE_TITLES[stage.stage] ?? stage.stage}</h2>
        <span className="card-head-right">
          {duration !== null ? <span className="duration">{duration}</span> : null}
          <span className={pillClass}>{stage.status}</span>
        </span>
      </header>
      <div className="card-body">{detailBody(stage)}</div>
      {explainEntries.length > 0 ? (
        <div className="explain">
          <button
            type="button"
            className="explain-toggle"
            aria-expanded={showExplain}
            onClick={() => setShowExplain((open) => !open)}
          >
            {showExplain ? "▾ hide how this is computed" : "▸ how is this computed?"}
          </button>
          {showExplain ? (
            <dl className="explain-list">
              {explainEntries.map((entry) => (
                <div key={entry.label} className="explain-item">
                  <dt>{entry.label}</dt>
                  <dd>
                    <code>{entry.formula}</code>
                    <span className="explain-desc">{entry.description}</span>
                    <span className="explain-example">e.g. {entry.example}</span>
                  </dd>
                </div>
              ))}
            </dl>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
