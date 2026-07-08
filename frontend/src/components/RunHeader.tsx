/**
 * Run header: "last successful run: <age>" (re-derived every ~10s), the
 * latest run's id / status / trigger, and a Refresh button that POSTs
 * `/api/runs/trigger {force: true}` and then asks the App to re-poll.
 *
 * The button is disabled while the POST is in flight. Step 11's trigger
 * answers 202 at the run's START (not completion), so the button re-enables
 * quickly; a repeat click during the run coalesces server-side
 * (`already_running: true`) instead of queuing a second run.
 */

import { useEffect, useState } from "react";
import type { RunInfo, TriggerResponse } from "../api";
import { triggerRun } from "../api";

const AGE_TICK_MS = 10_000;

function ageLabel(finishedAtUtc: string | null, nowMs: number): string {
  if (finishedAtUtc === null) {
    return "—";
  }
  const finished = Date.parse(finishedAtUtc);
  if (Number.isNaN(finished)) {
    return "—";
  }
  const seconds = Math.max(0, Math.floor((nowMs - finished) / 1000));
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ${minutes % 60}m ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h ago`;
}

interface RunHeaderProps {
  /** The latest ok run, or null when none exists yet (empty state). */
  run: RunInfo | null;
  /** Called with the trigger response so the App re-polls immediately. */
  onTriggered: (result: TriggerResponse) => void;
}

export default function RunHeader({ run, onTriggered }: RunHeaderProps) {
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const [busy, setBusy] = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);

  // Re-derive the age display every ~10s (the run itself only changes when
  // the App's poll delivers new data; this tick keeps the age honest).
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), AGE_TICK_MS);
    return () => window.clearInterval(id);
  }, []);

  async function handleRefresh(): Promise<void> {
    setBusy(true);
    setTriggerError(null);
    try {
      const result = await triggerRun(true);
      onTriggered(result);
    } catch (error) {
      setTriggerError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="run-header">
      <div className="run-header-facts">
        <span className="run-age">
          last successful run: <strong>{ageLabel(run?.finished_at_utc ?? null, nowMs)}</strong>
        </span>
        {run !== null ? (
          <span className="run-meta">
            run #{run.run_id} · {run.status} · {run.trigger}
          </span>
        ) : (
          <span className="run-meta">no successful run yet</span>
        )}
      </div>
      <div className="run-header-actions">
        {triggerError !== null ? <span className="trigger-error">{triggerError}</span> : null}
        <button type="button" onClick={() => void handleRefresh()} disabled={busy}>
          {busy ? "Running…" : "Refresh"}
        </button>
      </div>
    </div>
  );
}
