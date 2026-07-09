import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";

// HashRouter (not BrowserRouter): the built SPA is served by FastAPI's
// StaticFiles(html=True) with NO server-side catch-all, so hash routes
// (`/#/compare`) keep deep links + hard refreshes working without a backend
// change — every route resolves from index.html client-side.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
);
