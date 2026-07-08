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
import type { Explanation, LatestRunResponse, TriggerResponse } from "./api";
import { getExplanations, getLatestRun } from "./api";
import RunHeader from "./components/RunHeader";
import StageCard from "./components/StageCard";
import "./App.css";

const POLL_INTERVAL_MS = 7_000; // plan section 6: poll target at 5-10s

// Resolved polls to wait before a still-not-latest triggered run is called a
// failure (grace for Step 11's answer-at-run-START trigger: poll 1 fires
// immediately after the 202, while the run is usually still executing).
const FAILURE_NOTE_POLLS = 2;

type FetchState = "loading" | "empty" | "ready";

export default function App() {
  const [latest, setLatest] = useState<LatestRunResponse | null>(null);
  const [state, setState] = useState<FetchState>("loading");
  const [unreachable, setUnreachable] = useState<string | null>(null);
  const [lastTrigger, setLastTrigger] = useState<TriggerResponse | null>(null);
  // Polls RESOLVED since the most recent trigger. Step 11's trigger answers
  // 202 at the run's START (not completion), so the first post-trigger poll
  // legitimately still serves the PREVIOUS latest run while the new one
  // executes — the failure note may only render after the grace below.
  const [pollsSinceTrigger, setPollsSinceTrigger] = useState(FAILURE_NOTE_POLLS);
  // Bumped after a trigger: re-runs the poll effect (immediate tick + fresh interval).
  const [pollNonce, setPollNonce] = useState(0);
  // Static calc.py explanation registry, fetched once. A failure is non-fatal:
  // the cards render without the "how is this computed?" expanders.
  const [explanations, setExplanations] = useState<Record<string, Explanation>>({});

  useEffect(() => {
    let cancelled = false;
    void getExplanations()
      .then((data) => {
        if (!cancelled) {
          setExplanations(data);
        }
      })
      .catch(() => {
        /* non-fatal: cards render without explanations */
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
          // Supersession clears the note: once ANY run at or past the
          // triggered id is the latest ok run, the note is stale (a newer
          // scheduled/manual ok run must not resurrect an old failure note).
          setLastTrigger((prev) =>
            prev !== null && data.run.run_id >= prev.run_id ? null : prev
          );
        }
        setUnreachable(null);
        setPollsSinceTrigger((count) => count + 1);
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
    setPollsSinceTrigger(0); // arm the grace window (no note flash)
    setPollNonce((nonce) => nonce + 1); // re-poll now
  }

  // A forced trigger that did not become the latest ok run after the grace
  // window failed (force bypasses the freshness gate, so it cannot have been
  // skipped). Step 11's 202 answers at run START, so within the grace the run
  // is presumed still executing (a neutral in-progress note); the note is
  // cleared forever once any run_id >= the triggered id is latest (a newer
  // scheduled/manual ok run must not resurrect an old failure note).
  const triggerNote =
    lastTrigger !== null &&
    state !== "loading" &&
    (latest === null || latest.run.run_id < lastTrigger.run_id)
      ? lastTrigger.already_running
        ? `a run was already active (run #${lastTrigger.run_id})`
        : pollsSinceTrigger >= FAILURE_NOTE_POLLS
          ? `triggered run #${lastTrigger.run_id} did not finish ok — check the run ledger ` +
            `(GET /api/runs/${lastTrigger.run_id}/stages)`
          : pollsSinceTrigger >= 1
            ? `run #${lastTrigger.run_id} started — waiting for it to finish…`
            : null
      : null;

  // Freshness data-recency, folded into the top bar (no separate card).
  const freshnessStage = latest?.stages.find((stage) => stage.stage === "freshness");
  const freshnessDetail =
    freshnessStage && typeof freshnessStage.detail === "object" && freshnessStage.detail !== null
      ? (freshnessStage.detail as Record<string, unknown>)
      : null;
  const dataMaxDate =
    freshnessDetail && typeof freshnessDetail["data_max_date"] === "string"
      ? freshnessDetail["data_max_date"]
      : null;
  const dataFetchedAt =
    freshnessDetail && typeof freshnessDetail["data_fetched_at"] === "string"
      ? freshnessDetail["data_fetched_at"]
      : null;

  return (
    <main className="app">
      <header className="app-head">
        <h1>always-best-estimates</h1>
        <RunHeader
          run={latest?.run ?? null}
          onTriggered={handleTriggered}
          dataMaxDate={dataMaxDate}
          dataFetchedAt={dataFetchedAt}
        />
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
          {latest.stages
            .filter((stage) => stage.stage !== "freshness")
            .map((stage) => (
              <StageCard key={stage.stage} stage={stage} explanations={explanations} />
            ))}
        </div>
      ) : null}
    </main>
  );
}
