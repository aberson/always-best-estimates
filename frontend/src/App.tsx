/**
 * The always-best-estimates shell (Track 2 Step 26): a persistent title + nav
 * bar over the routed views. The V1 dashboard moved into `DashboardView` (the
 * default `/` route); Track 2 adds Config, Compare, and Scenarios.
 */

import { NavLink, Route, Routes } from "react-router-dom";
import CompareView from "./components/CompareView";
import DashboardView from "./components/DashboardView";
import ScenarioEditor from "./components/ScenarioEditor";
import StageDetailTab from "./components/StageDetailTab";
import "./App.css";

const NAV_LINKS: { to: string; label: string; end?: boolean }[] = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/config", label: "Config" },
  { to: "/compare", label: "Compare" },
  { to: "/scenarios", label: "Scenarios" },
];

export default function App() {
  return (
    <main className="app">
      <header className="app-head">
        <h1>always-best-estimates</h1>
        <nav className="app-nav">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => (isActive ? "nav-link nav-link-active" : "nav-link")}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<DashboardView />} />
        <Route path="/config" element={<StageDetailTab />} />
        <Route path="/compare" element={<CompareView />} />
        <Route path="/scenarios" element={<ScenarioEditor />} />
        <Route
          path="*"
          element={<p className="empty-state">unknown route — pick a tab above.</p>}
        />
      </Routes>
    </main>
  );
}
