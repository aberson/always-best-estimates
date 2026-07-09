/**
 * Config / per-stage detail tab (route `/config`, Track 2 Step 26).
 *
 * Edits a selected config's four stage choices as a working draft:
 * - features   -> `feature_set` registry impl + params (config.params.features)
 * - forecast   -> `forecaster`  registry impl + params (config.params.forecaster)
 * - blend      -> the attached scenario (its `kind` selects the `view_source`
 *                 impl; the scenario's payload is its params — read-only here,
 *                 authored on the Scenarios tab)
 * - optimize   -> `optimizer`   registry impl + params (config.params.optimizer)
 *
 * The draft loads from the central config by default; "Save changes" PATCHes the
 * selected config, "Save as new config" POSTs a fresh one. Non-central configs
 * can then be run/compared on the Compare tab. Nothing is applied until saved.
 */

import { useEffect, useState } from "react";
import type { Config, RegistryMap, RegistryParam, Registries, Scenario } from "../api";
import {
  createConfig,
  getConfigs,
  getRegistries,
  getScenarios,
  updateConfig,
} from "../api";

type LoadState = "loading" | "ready" | "error";

/** The three stages whose impl+params live directly on the config. */
const PARAM_STAGES = ["features", "forecaster", "optimizer"] as const;
type ParamStage = (typeof PARAM_STAGES)[number];

type ParamText = Record<ParamStage, Record<string, string>>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A stage's stored param overrides as display strings (for the text inputs). */
function stageTextParams(params: Record<string, unknown>, stage: ParamStage): Record<string, string> {
  const raw = params[stage];
  if (!isRecord(raw)) {
    return {};
  }
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) {
    out[key] = value === null || value === undefined ? "" : String(value);
  }
  return out;
}

function emptyParamText(): ParamText {
  return { features: {}, forecaster: {}, optimizer: {} };
}

/** Coerce a param input string to the type the impl declares (numbers become
 *  JSON numbers; a non-numeric entry is kept verbatim so the server can reject
 *  it loudly rather than the UI swallowing it). */
function coerceParam(spec: RegistryParam, raw: string): unknown {
  if (spec.type === "float" || spec.type === "int") {
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : raw;
  }
  if (spec.type === "bool") {
    return raw === "true";
  }
  return raw;
}

function defaultText(spec: RegistryParam): string {
  return spec.default === null || spec.default === undefined ? "" : String(spec.default);
}

export default function StageDetailTab() {
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [registries, setRegistries] = useState<Registries | null>(null);
  const [configs, setConfigs] = useState<Config[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);

  // The working draft.
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [featureSet, setFeatureSet] = useState("");
  const [forecaster, setForecaster] = useState("");
  const [optimizer, setOptimizer] = useState("");
  const [viewScenarioId, setViewScenarioId] = useState<number | null>(null);
  const [paramText, setParamText] = useState<ParamText>(emptyParamText);

  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  function loadConfigIntoForm(config: Config): void {
    setSelectedConfigId(config.config_id);
    setName(config.name);
    setFeatureSet(config.feature_set);
    setForecaster(config.forecaster);
    setOptimizer(config.optimizer);
    setViewScenarioId(config.view_scenario_id);
    setParamText({
      features: stageTextParams(config.params, "features"),
      forecaster: stageTextParams(config.params, "forecaster"),
      optimizer: stageTextParams(config.params, "optimizer"),
    });
    setSaveMsg(null);
  }

  useEffect(() => {
    let cancelled = false;
    async function load(): Promise<void> {
      try {
        const [regs, cfgs, scns] = await Promise.all([
          getRegistries(),
          getConfigs(),
          getScenarios(),
        ]);
        if (cancelled) {
          return;
        }
        setRegistries(regs);
        setConfigs(cfgs);
        setScenarios(scns);
        const central = cfgs.find((c) => c.is_central) ?? cfgs[0];
        if (central !== undefined) {
          loadConfigIntoForm(central);
        }
        setLoadState("ready");
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : String(error));
          setLoadState("error");
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  function onSelectConfig(configId: number): void {
    const config = configs.find((c) => c.config_id === configId);
    if (config !== undefined) {
      loadConfigIntoForm(config);
    }
  }

  function onParamChange(stage: ParamStage, param: string, value: string): void {
    setParamText((prev) => ({ ...prev, [stage]: { ...prev[stage], [param]: value } }));
  }

  /** Build the params payload from the current impls' schemas (drops empties and
   *  keys that don't belong to the selected impl). */
  function buildParams(): Record<string, unknown> {
    if (registries === null) {
      return {};
    }
    const stageSpecs: [ParamStage, RegistryMap, string][] = [
      ["features", registries.feature_set, featureSet],
      ["forecaster", registries.forecaster, forecaster],
      ["optimizer", registries.optimizer, optimizer],
    ];
    const out: Record<string, unknown> = {};
    for (const [stage, registry, key] of stageSpecs) {
      const entry = registry[key];
      if (entry === undefined) {
        continue;
      }
      const stageOut: Record<string, unknown> = {};
      for (const spec of entry.params) {
        const raw = paramText[stage][spec.name];
        if (raw === undefined || raw.trim() === "") {
          continue;
        }
        stageOut[spec.name] = coerceParam(spec, raw);
      }
      if (Object.keys(stageOut).length > 0) {
        out[stage] = stageOut;
      }
    }
    return out;
  }

  async function refreshConfigs(): Promise<Config[]> {
    const cfgs = await getConfigs();
    setConfigs(cfgs);
    return cfgs;
  }

  async function handleSaveExisting(): Promise<void> {
    if (selectedConfigId === null || viewScenarioId === null) {
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      const updated = await updateConfig(selectedConfigId, {
        name,
        feature_set: featureSet,
        forecaster,
        view_scenario_id: viewScenarioId,
        optimizer,
        params: buildParams(),
      });
      await refreshConfigs();
      loadConfigIntoForm(updated);
      setSaveMsg({ kind: "ok", text: `saved changes to "${updated.name}" (config #${updated.config_id}).` });
    } catch (error) {
      setSaveMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setSaving(false);
    }
  }

  async function handleSaveAsNew(): Promise<void> {
    if (viewScenarioId === null || name.trim() === "") {
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      const created = await createConfig({
        name,
        feature_set: featureSet,
        forecaster,
        view_scenario_id: viewScenarioId,
        optimizer,
        params: buildParams(),
      });
      await refreshConfigs();
      loadConfigIntoForm(created);
      setSaveMsg({
        kind: "ok",
        text: `created config "${created.name}" (#${created.config_id}) — run or compare it on the Compare tab.`,
      });
    } catch (error) {
      setSaveMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setSaving(false);
    }
  }

  function handleReset(): void {
    const config = configs.find((c) => c.config_id === selectedConfigId);
    if (config !== undefined) {
      loadConfigIntoForm(config);
    }
  }

  if (loadState === "loading") {
    return (
      <section className="view">
        <p className="muted">loading registries + configs…</p>
      </section>
    );
  }
  if (loadState === "error" || registries === null) {
    return (
      <section className="view">
        <div className="banner banner-error">
          could not load config editor{loadError !== null ? ` — ${loadError}` : ""}
        </div>
      </section>
    );
  }

  const selectedConfig = configs.find((c) => c.config_id === selectedConfigId) ?? null;
  const editingCentral = selectedConfig?.is_central === true;
  const selectedScenario = scenarios.find((s) => s.view_scenario_id === viewScenarioId) ?? null;
  const viewSourceEntry =
    selectedScenario !== null ? registries.view_source[selectedScenario.kind] : undefined;

  return (
    <section className="view">
      <div className="view-head">
        <h2>Config recipe</h2>
        <p className="muted">
          One config = a per-stage implementation choice (plus params). The central config is what
          the always-on 5-minute loop runs; edit a draft and save it as a new config to compare.
        </p>
      </div>

      <div className="config-picker">
        <label className="field">
          <span>editing config</span>
          <select
            value={selectedConfigId ?? ""}
            onChange={(event) => onSelectConfig(Number(event.target.value))}
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
          <span>config name</span>
          <input type="text" value={name} onChange={(event) => setName(event.target.value)} />
        </label>
      </div>

      {editingCentral ? (
        <div className="banner banner-note">
          You are editing the CENTRAL config — "Save changes" changes the always-on portfolio.
          Prefer "Save as new config" (rename first), then promote it on the Compare tab.
        </div>
      ) : null}

      <div className="stage-grid">
        <RegistryStage
          title="Features"
          note="Turns prices into the returns + feature matrix the forecaster consumes."
          registry={registries.feature_set}
          selected={featureSet}
          onSelect={setFeatureSet}
          params={paramText.features}
          onParamChange={(param, value) => onParamChange("features", param, value)}
        />
        <RegistryStage
          title="Forecast"
          note="The WorldModel that produces per-asset (mu, sigma) over the horizon."
          registry={registries.forecaster}
          selected={forecaster}
          onSelect={setForecaster}
          params={paramText.forecaster}
          onParamChange={(param, value) => onParamChange("forecaster", param, value)}
        />

        <section className="stage-section">
          <h3>Blend — view source</h3>
          <p className="muted stage-note">
            The Black-Litterman views come from the attached scenario; its kind selects the
            view-source impl. Author scenarios on the Scenarios tab.
          </p>
          <label className="field">
            <span>scenario</span>
            <select
              value={viewScenarioId ?? ""}
              onChange={(event) => setViewScenarioId(Number(event.target.value))}
            >
              {scenarios.length === 0 ? <option value="">no scenarios</option> : null}
              {scenarios.map((scenario) => (
                <option key={scenario.view_scenario_id} value={scenario.view_scenario_id}>
                  #{scenario.view_scenario_id} {scenario.name} ({scenario.kind})
                </option>
              ))}
            </select>
          </label>
          {selectedScenario !== null ? (
            <>
              <p className="stage-desc">
                view source <strong>{selectedScenario.kind}</strong>
                {viewSourceEntry !== undefined ? ` — ${viewSourceEntry.description}` : ""}
              </p>
              <p className="param-desc muted">
                payload: <code>{JSON.stringify(selectedScenario.payload)}</code>
              </p>
            </>
          ) : (
            <p className="stage-desc muted">no scenario selected.</p>
          )}
        </section>

        <RegistryStage
          title="Optimize"
          note="Solves target weights from the BL posterior (long-only, boxed)."
          registry={registries.optimizer}
          selected={optimizer}
          onSelect={setOptimizer}
          params={paramText.optimizer}
          onParamChange={(param, value) => onParamChange("optimizer", param, value)}
        />
      </div>

      {saveMsg !== null ? (
        <div className={`form-msg ${saveMsg.kind === "ok" ? "form-msg-ok" : "form-msg-error"}`}>
          {saveMsg.text}
        </div>
      ) : null}

      <div className="form-actions">
        <button
          type="button"
          onClick={() => void handleSaveExisting()}
          disabled={saving || selectedConfigId === null}
        >
          {saving ? "Saving…" : "Save changes"}
        </button>
        <button
          type="button"
          className="btn-secondary"
          onClick={() => void handleSaveAsNew()}
          disabled={saving || name.trim() === ""}
        >
          Save as new config
        </button>
        <button type="button" className="btn-secondary" onClick={handleReset} disabled={saving}>
          Reset
        </button>
      </div>
    </section>
  );
}

/** One editable stage whose impl+params live directly on the config. */
function RegistryStage({
  title,
  note,
  registry,
  selected,
  onSelect,
  params,
  onParamChange,
}: {
  title: string;
  note: string;
  registry: RegistryMap;
  selected: string;
  onSelect: (key: string) => void;
  params: Record<string, string>;
  onParamChange: (param: string, value: string) => void;
}) {
  const entry = registry[selected];
  return (
    <section className="stage-section">
      <h3>{title}</h3>
      <p className="muted stage-note">{note}</p>
      <label className="field">
        <span>implementation</span>
        <select value={selected} onChange={(event) => onSelect(event.target.value)}>
          {registry[selected] === undefined ? <option value={selected}>{selected}</option> : null}
          {Object.keys(registry).map((key) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
      </label>
      {entry !== undefined ? (
        <p className="stage-desc">{entry.description}</p>
      ) : (
        <p className="stage-desc muted">unknown impl "{selected}" (would fail at resolve time).</p>
      )}
      {entry !== undefined && entry.params.length > 0 ? (
        <div className="param-form">
          {entry.params.map((spec) => (
            <label className="field" key={spec.name}>
              <span>
                {spec.name} <em className="param-type">({spec.type})</em>
              </span>
              <input
                type={spec.type === "float" || spec.type === "int" ? "number" : "text"}
                value={params[spec.name] ?? defaultText(spec)}
                placeholder={defaultText(spec)}
                onChange={(event) => onParamChange(spec.name, event.target.value)}
              />
              <span className="param-desc">{spec.description}</span>
            </label>
          ))}
        </div>
      ) : entry !== undefined ? (
        <p className="param-desc muted">no parameters.</p>
      ) : null}
    </section>
  );
}
