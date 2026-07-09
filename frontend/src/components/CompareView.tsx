/**
 * Compare view (route `/compare`, Track 2 Step 27).
 *
 * Pick >=2 configs; `GET /api/compare` returns each one's latest ok allocation.
 * Renders the weights side by side (rows = SPY/ACWI/AGG, one column per config),
 * each config's objective + run_id/finished, and marks the CENTRAL config. A
 * per-config "Run / refresh" button computes a fresh run then re-fetches — hidden
 * for the central config (the always-on loop owns it; its run endpoint 409s).
 */

import { useEffect, useState } from "react";
import type { CompareConfig, CompareResponse, Config } from "../api";
import { UNIVERSE, getCompare, getConfigs, runConfig } from "../api";

function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** "2026-07-08T00:02:06Z" -> "2026-07-08 00:02Z" (compact). */
function fmtFinished(iso: string | null): string {
  if (iso === null) {
    return "—";
  }
  const [date, time] = iso.split("T");
  return time !== undefined ? `${date} ${time.slice(0, 5)}Z` : date;
}

function objectiveForm(objective: Record<string, unknown> | null): string | null {
  if (objective === null) {
    return null;
  }
  const form = objective["form"];
  return typeof form === "string" ? form : JSON.stringify(objective);
}

type LoadState = "loading" | "ready" | "error";

export default function CompareView() {
  const [configs, setConfigs] = useState<Config[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  const [compare, setCompare] = useState<CompareResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareNonce, setCompareNonce] = useState(0);

  const [runningId, setRunningId] = useState<number | null>(null);
  const [runMsg, setRunMsg] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  // Load the configs once; preselect all so the comparison is populated on entry.
  useEffect(() => {
    let cancelled = false;
    getConfigs()
      .then((cfgs) => {
        if (cancelled) {
          return;
        }
        setConfigs(cfgs);
        setSelectedIds(cfgs.map((c) => c.config_id));
        setLoadState("ready");
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
  }, []);

  const idsKey = selectedIds.join(",");

  // Re-fetch the comparison whenever the selection (or the run nonce) changes.
  useEffect(() => {
    if (selectedIds.length < 1) {
      setCompare(null);
      setCompareError(null);
      return;
    }
    let cancelled = false;
    setCompareLoading(true);
    getCompare(selectedIds)
      .then((data) => {
        if (!cancelled) {
          setCompare(data);
          setCompareError(null);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setCompareError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCompareLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
    // idsKey encodes selectedIds; compareNonce forces a manual refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey, compareNonce]);

  function toggle(configId: number): void {
    setSelectedIds((prev) =>
      prev.includes(configId) ? prev.filter((id) => id !== configId) : [...prev, configId]
    );
  }

  async function handleRun(configId: number): Promise<void> {
    setRunningId(configId);
    setRunMsg(null);
    try {
      const result = await runConfig(configId);
      setRunMsg({
        kind: "ok",
        text: `config #${configId}: ${result.cached ? "served cached" : "computed"} run #${result.run_id}.`,
      });
      setCompareNonce((nonce) => nonce + 1);
    } catch (error) {
      setRunMsg({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    } finally {
      setRunningId(null);
    }
  }

  if (loadState === "loading") {
    return (
      <section className="view">
        <p className="muted">loading configs…</p>
      </section>
    );
  }
  if (loadState === "error") {
    return (
      <section className="view">
        <div className="banner banner-error">
          could not load configs{loadError !== null ? ` — ${loadError}` : ""}
        </div>
      </section>
    );
  }

  const centralId = compare?.central_config_id ?? null;
  const columns: CompareConfig[] = compare?.configs ?? [];

  return (
    <section className="view">
      <div className="view-head">
        <h2>Compare configs</h2>
        <p className="muted">
          Each column is a config's LATEST allocation; a recipe edit only shows after you re-run it.
          The central config is marked and always-on (no run button).
        </p>
      </div>

      {configs.length < 2 ? (
        <div className="banner banner-note">
          only one config exists — create another on the Config tab to compare.
        </div>
      ) : null}

      <div className="config-picker">
        <span className="picker-label">configs to compare:</span>
        <div className="checkbox-row">
          {configs.map((config) => (
            <label key={config.config_id} className="checkbox">
              <input
                type="checkbox"
                checked={selectedIds.includes(config.config_id)}
                onChange={() => toggle(config.config_id)}
              />
              <span>
                #{config.config_id} {config.name}
                {config.is_central ? " ★" : ""}
              </span>
            </label>
          ))}
        </div>
      </div>

      {selectedIds.length < 2 ? (
        <p className="muted">select at least 2 configs to compare.</p>
      ) : null}

      {compareError !== null ? (
        <div className="banner banner-error">compare failed — {compareError}</div>
      ) : null}
      {runMsg !== null ? (
        <div className={`form-msg ${runMsg.kind === "ok" ? "form-msg-ok" : "form-msg-error"}`}>
          {runMsg.text}
        </div>
      ) : null}
      {compareLoading ? <p className="muted">loading comparison…</p> : null}

      {columns.length > 0 ? (
        <>
          <div className="table-scroll">
            <table className="detail-table compare-table">
              <thead>
                <tr>
                  <th>asset</th>
                  {columns.map((column) => (
                    <th key={column.config_id}>
                      <span className="compare-col-head">
                        {column.name}
                        {column.config_id === centralId ? (
                          <span className="central-badge">central</span>
                        ) : null}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {UNIVERSE.map((asset) => (
                  <tr key={asset}>
                    <th>{asset}</th>
                    {columns.map((column) => {
                      const weight = column.weights[asset];
                      return (
                        <td key={column.config_id}>
                          {typeof weight === "number" ? pct(weight) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="compare-meta">
            {columns.map((column) => {
              const isCentral = column.config_id === centralId;
              const form = objectiveForm(column.objective);
              return (
                <div
                  key={column.config_id}
                  className={`compare-card${isCentral ? " compare-card-central" : ""}`}
                >
                  <div className="compare-card-head">
                    <strong>{column.name}</strong>
                    {isCentral ? <span className="central-badge">central</span> : null}
                  </div>
                  <p className="compare-run">
                    {column.run_id !== null
                      ? `run #${column.run_id} · ${fmtFinished(column.finished_at_utc)}`
                      : "no run yet"}
                  </p>
                  {form !== null ? (
                    <code className="objective-form">{form}</code>
                  ) : (
                    <p className="muted">no objective (config not run).</p>
                  )}
                  {isCentral ? (
                    <p className="muted compare-central-note">always-on (5-min loop).</p>
                  ) : (
                    <button
                      type="button"
                      className="btn-small"
                      onClick={() => void handleRun(column.config_id)}
                      disabled={runningId !== null}
                    >
                      {runningId === column.config_id ? "Running…" : "Run / refresh"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </>
      ) : !compareLoading && selectedIds.length >= 2 ? (
        <p className="empty-state">no comparison data.</p>
      ) : null}
    </section>
  );
}
