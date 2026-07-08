/**
 * The always-best-estimates UI: RunHeader + one card per pipeline stage,
 * fed by polling `GET /api/runs/latest` every POLL_INTERVAL_MS.
 *
 * States (all first-class, never blank):
 * - loading: first poll not resolved yet.
 * - empty: 404 from the poll target (no successful run yet) — shows
 *   "no runs yet — trigger one"; the RunHeader's Refresh button IS the trigger.
 * - ready: render the six stage cards in pipeline order (the API serves
 *   `run_stages` rows in rowid order, which pipeline.py guarantees is
 *   execution order).
 * - unreachable: a banner over whatever data we last had; the poll keeps
 *   retrying on its interval, and the banner clears on the next success.
 */

import { useEffect, useState } from "react";
import type { LatestRunResponse, TriggerResponse } from "./api";
import { getLatestRun } from "./api";
import RunHeader from "./components/RunHeader";
import StageCard from "./components/StageCard";
import "./App.css";

const POLL_INTERVAL_MS = 7_000; // plan section 6: poll target at 5-10s

type FetchState = "loading" | "empty" | "ready";

export default function App() {
  const [latest, setLatest] = useState<LatestRunResponse | null>(null);
  const [state, setState] = useState<FetchState>("loading");
  const [unreachable, setUnreachable] = useState<string | null>(null);
  const [lastTrigger, setLastTrigger] = useState<TriggerResponse | null>(null);
  // Bumped after a trigger: re-runs the poll effect (immediate tick + fresh interval).
  const [pollNonce, setPollNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const tick = async (): Promise<void> => {
      try {
        const data = await getLatestRun();
        if (cancelled) {
          return;
        }
        if (data === null) {
          setLatest(null);
          setState("empty");
        } else {
          setLatest(data);
          setState("ready");
        }
        setUnreachable(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        // Keep the last-known data on screen; the interval keeps retrying.
        setUnreachable(error instanceof Error ? error.message : String(error));
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollNonce]);

  function handleTriggered(result: TriggerResponse): void {
    setLastTrigger(result);
    setPollNonce((nonce) => nonce + 1); // re-poll now
  }

  // A forced trigger that did not become the latest ok run failed (force
  // bypasses the freshness gate, so it cannot have been skipped). Surface it.
  const triggerNote =
    lastTrigger !== null && state !== "loading" && latest?.run.run_id !== lastTrigger.run_id
      ? lastTrigger.already_running
        ? `a run was already active (run #${lastTrigger.run_id})`
        : `triggered run #${lastTrigger.run_id} did not finish ok — check the run ledger ` +
          `(GET /api/runs/${lastTrigger.run_id}/stages)`
      : null;

  return (
    <main className="app">
      <header className="app-head">
        <h1>always-best-estimates</h1>
        <RunHeader run={latest?.run ?? null} onTriggered={handleTriggered} />
      </header>

      {unreachable !== null ? (
        <div className="banner banner-error">API unreachable — retrying… ({unreachable})</div>
      ) : null}
      {triggerNote !== null ? <div className="banner banner-note">{triggerNote}</div> : null}

      {state === "loading" ? <p className="muted">loading latest run…</p> : null}
      {state === "empty" ? (
        <p className="empty-state">no runs yet — trigger one with the Refresh button above.</p>
      ) : null}
      {latest !== null ? (
        <div className="cards">
          {latest.stages.map((stage) => (
            <StageCard key={stage.stage} stage={stage} />
          ))}
        </div>
      ) : null}
    </main>
  );
}
