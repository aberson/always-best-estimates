/**
 * Scenario editor (route `/scenarios`, Track 2 Step 28).
 *
 * Lists the view-scenario library (seeded + operator-authored), authors NEW
 * counterfactual (per-asset {mu, confidence}) and historical (date window)
 * scenarios, deletes scenarios (409-aware when one is referenced by a config),
 * and attaches a scenario to a config's blend stage (PATCH view_scenario_id).
 */

import { useEffect, useState } from "react";
import type { Config, Scenario } from "../api";
import {
  UNIVERSE,
  createScenario,
  deleteScenario,
  getConfigs,
  getScenarios,
  updateConfig,
} from "../api";

type LoadState = "loading" | "ready" | "error";
type Msg = { kind: "ok" | "error"; text: string } | null;

interface AssetView {
  include: boolean;
  mu: string; // annual %, e.g. "10" -> 0.10
  confidence: string; // 0..1
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function emptyViews(): Record<string, AssetView> {
  return Object.fromEntries(
    UNIVERSE.map((asset) => [asset, { include: false, mu: "", confidence: "0.5" }])
  );
}

/** A one-line human summary of a scenario's payload, per kind. */
function summarize(scenario: Scenario): string {
  if (scenario.kind === "counterfactual") {
    const entries = Object.entries(scenario.payload).map(([asset, raw]) => {
      if (isRecord(raw)) {
        const mu = raw["mu"];
        const confidence = raw["confidence"];
        const muStr = typeof mu === "number" ? `${(mu * 100).toFixed(1)}%` : String(mu);
        const confStr = typeof confidence === "number" ? confidence.toFixed(2) : "0.50";
        return `${asset} ${muStr} @ conf ${confStr}`;
      }
      return `${asset}: ${JSON.stringify(raw)}`;
    });
    return entries.length > 0 ? entries.join(", ") : "no views";
  }
  if (scenario.kind === "historical") {
    const start = scenario.payload["window_start"];
    const end = scenario.payload["window_end"];
    const startStr = typeof start === "string" ? start : "…";
    const endStr = typeof end === "string" ? end : "…";
    return typeof start === "string" || typeof end === "string"
      ? `window ${startStr} → ${endStr}`
      : "full history";
  }
  return "derived from the run's forecaster";
}

export default function ScenarioEditor() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [configs, setConfigs] = useState<Config[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [msg, setMsg] = useState<Msg>(null);

  // Counterfactual authoring form.
  const [cfName, setCfName] = useState("");
  const [cfViews, setCfViews] = useState<Record<string, AssetView>>(emptyViews);
  // Historical authoring form.
  const [histName, setHistName] = useState("");
  const [windowStart, setWindowStart] = useState("");
  const [windowEnd, setWindowEnd] = useState("");
  // Attach-to-config form.
  const [attachConfigId, setAttachConfigId] = useState<number | null>(null);
  const [attachScenarioId, setAttachScenarioId] = useState<number | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function reload(): Promise<void> {
    const [scns, cfgs] = await Promise.all([getScenarios(), getConfigs()]);
    setScenarios(scns);
    setConfigs(cfgs);
    setAttachConfigId((prev) => prev ?? cfgs.find((c) => !c.is_central)?.config_id ?? cfgs[0]?.config_id ?? null);
    setAttachScenarioId((prev) => prev ?? scns[0]?.view_scenario_id ?? null);
  }

  useEffect(() => {
    let cancelled = false;
    reload()
      .then(() => {
        if (!cancelled) {
          setLoadState("ready");
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : String(error));
          setLoadState("error");
        }
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setView(asset: string, patch: Partial<AssetView>): void {
    setCfViews((prev) => ({ ...prev, [asset]: { ...prev[asset], ...patch } }));
  }

  async function handleCreateCounterfactual(): Promise<void> {
    if (cfName.trim() === "") {
      setMsg({ kind: "error", text: "name is required." });
      return;
    }
    const payload: Record<string, { mu: number; confidence: number }> = {};
    for (const asset of UNIVERSE) {
      const view = cfViews[asset];
      if (!view.include) {
        continue;
      }
      const muPct = Number(view.mu);
      if (view.mu.trim() === "" || !Number.isFinite(muPct)) {
        setMsg({ kind: "error", text: `${asset}: mu must be a number (annual %).` });
        return;
      }
      const confidence = Number(view.confidence);
      payload[asset] = {
        mu: muPct / 100,
        confidence: Number.isFinite(confidence) ? confidence : 0.5,
      };
    }
    if (Object.keys(payload).length === 0) {
      setMsg({ kind: "error", text: "include at least one asset view." });
      return;
    }
    setSubmitting(true);
    setMsg(null);
    try {
      const created = await createScenario({ name: cfName, kind: "counterfactual", payload });
      await reload();
      setCfName("");
      setCfViews(emptyViews());
      setMsg({ kind: "ok", text: `created "${created.name}" (#${created.view_scenario_id}).` });
    } catch (error) {
      setMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCreateHistorical(): Promise<void> {
    if (histName.trim() === "") {
      setMsg({ kind: "error", text: "name is required." });
      return;
    }
    const payload: Record<string, string> = {};
    if (windowStart !== "") {
      payload["window_start"] = windowStart;
    }
    if (windowEnd !== "") {
      payload["window_end"] = windowEnd;
    }
    setSubmitting(true);
    setMsg(null);
    try {
      const created = await createScenario({ name: histName, kind: "historical", payload });
      await reload();
      setHistName("");
      setWindowStart("");
      setWindowEnd("");
      setMsg({ kind: "ok", text: `created "${created.name}" (#${created.view_scenario_id}).` });
    } catch (error) {
      setMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(scenario: Scenario): Promise<void> {
    setDeletingId(scenario.view_scenario_id);
    setMsg(null);
    try {
      await deleteScenario(scenario.view_scenario_id);
      await reload();
      setMsg({ kind: "ok", text: `deleted "${scenario.name}".` });
    } catch (error) {
      // 409 = referenced by a config — surface it, do not crash.
      setMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setDeletingId(null);
    }
  }

  async function handleAttach(): Promise<void> {
    if (attachConfigId === null || attachScenarioId === null) {
      return;
    }
    setSubmitting(true);
    setMsg(null);
    try {
      const updated = await updateConfig(attachConfigId, { view_scenario_id: attachScenarioId });
      await reload();
      setMsg({
        kind: "ok",
        text: `attached scenario #${attachScenarioId} to config "${updated.name}".`,
      });
    } catch (error) {
      setMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setSubmitting(false);
    }
  }

  if (loadState === "loading") {
    return (
      <section className="view">
        <p className="muted">loading scenarios…</p>
      </section>
    );
  }
  if (loadState === "error") {
    return (
      <section className="view">
        <div className="banner banner-error">
          could not load scenarios{loadError !== null ? ` — ${loadError}` : ""}
        </div>
      </section>
    );
  }

  return (
    <section className="view">
      <div className="view-head">
        <h2>View scenarios</h2>
        <p className="muted">
          A scenario is a named Black-Litterman view set. The library below (seeded + authored)
          feeds any config's blend stage.
        </p>
      </div>

      {msg !== null ? (
        <div className={`form-msg ${msg.kind === "ok" ? "form-msg-ok" : "form-msg-error"}`}>
          {msg.text}
        </div>
      ) : null}

      <div className="scenario-list">
        {scenarios.length === 0 ? (
          <p className="empty-state">no scenarios yet — author one below.</p>
        ) : (
          <table className="detail-table">
            <thead>
              <tr>
                <th>id</th>
                <th>name</th>
                <th>kind</th>
                <th>views</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map((scenario) => (
                <tr key={scenario.view_scenario_id}>
                  <th>#{scenario.view_scenario_id}</th>
                  <td>{scenario.name}</td>
                  <td>
                    <span className={`kind-badge kind-${scenario.kind}`}>{scenario.kind}</span>
                  </td>
                  <td className="scenario-summary">{summarize(scenario)}</td>
                  <td>
                    <button
                      type="button"
                      className="btn-small btn-danger"
                      onClick={() => void handleDelete(scenario)}
                      disabled={deletingId !== null}
                    >
                      {deletingId === scenario.view_scenario_id ? "Deleting…" : "Delete"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="form-grid">
        <section className="form-card">
          <h3>New counterfactual scenario</h3>
          <p className="muted stage-note">
            Hand-authored absolute views. Include any subset of assets; mu is an annual return %.
          </p>
          <label className="field">
            <span>name</span>
            <input
              type="text"
              value={cfName}
              onChange={(event) => setCfName(event.target.value)}
              placeholder="e.g. My bullish equity"
            />
          </label>
          <table className="detail-table cf-table">
            <thead>
              <tr>
                <th>use</th>
                <th>asset</th>
                <th>mu (annual %)</th>
                <th>confidence (0..1)</th>
              </tr>
            </thead>
            <tbody>
              {UNIVERSE.map((asset) => {
                const view = cfViews[asset];
                return (
                  <tr key={asset}>
                    <td>
                      <input
                        type="checkbox"
                        checked={view.include}
                        onChange={(event) => setView(asset, { include: event.target.checked })}
                      />
                    </td>
                    <th>{asset}</th>
                    <td>
                      <input
                        type="number"
                        className="cf-input"
                        value={view.mu}
                        disabled={!view.include}
                        onChange={(event) => setView(asset, { mu: event.target.value })}
                        placeholder="10"
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        className="cf-input"
                        step="0.05"
                        min="0"
                        max="1"
                        value={view.confidence}
                        disabled={!view.include}
                        onChange={(event) => setView(asset, { confidence: event.target.value })}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="form-actions">
            <button
              type="button"
              onClick={() => void handleCreateCounterfactual()}
              disabled={submitting}
            >
              Create counterfactual
            </button>
          </div>
        </section>

        <section className="form-card">
          <h3>New historical scenario</h3>
          <p className="muted stage-note">
            Absolute views from a past window's realized returns. Leave a bound blank for open-ended.
          </p>
          <label className="field">
            <span>name</span>
            <input
              type="text"
              value={histName}
              onChange={(event) => setHistName(event.target.value)}
              placeholder="e.g. 2020 window"
            />
          </label>
          <label className="field">
            <span>window start</span>
            <input
              type="date"
              value={windowStart}
              onChange={(event) => setWindowStart(event.target.value)}
            />
          </label>
          <label className="field">
            <span>window end</span>
            <input
              type="date"
              value={windowEnd}
              onChange={(event) => setWindowEnd(event.target.value)}
            />
          </label>
          <div className="form-actions">
            <button
              type="button"
              onClick={() => void handleCreateHistorical()}
              disabled={submitting}
            >
              Create historical
            </button>
          </div>
        </section>

        <section className="form-card">
          <h3>Attach scenario to a config</h3>
          <p className="muted stage-note">
            Point a config's blend stage at a scenario (PATCH its view_scenario_id).
          </p>
          <label className="field">
            <span>config</span>
            <select
              value={attachConfigId ?? ""}
              onChange={(event) => setAttachConfigId(Number(event.target.value))}
            >
              {configs.map((config) => (
                <option key={config.config_id} value={config.config_id}>
                  #{config.config_id} {config.name}
                  {config.is_central ? " ★ central" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>scenario</span>
            <select
              value={attachScenarioId ?? ""}
              onChange={(event) => setAttachScenarioId(Number(event.target.value))}
            >
              {scenarios.map((scenario) => (
                <option key={scenario.view_scenario_id} value={scenario.view_scenario_id}>
                  #{scenario.view_scenario_id} {scenario.name} ({scenario.kind})
                </option>
              ))}
            </select>
          </label>
          <div className="form-actions">
            <button
              type="button"
              className="btn-secondary"
              onClick={() => void handleAttach()}
              disabled={submitting || attachConfigId === null || attachScenarioId === null}
            >
              Attach
            </button>
          </div>
        </section>
      </div>
    </section>
  );
}
