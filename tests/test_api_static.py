"""Step 10 production static-serving tests (api.py's conditional mount).

Contract: IF a built frontend directory exists, ``create_app`` mounts it at
``/`` via ``StaticFiles(html=True)`` — registered LAST, so the API routes
(``/api/*``, ``/health``) always win; WITHOUT it, ``/`` is a plain 404 and
the API is untouched. Exercised through a fake dist dir (no real npm build
needed) via the injectable ``static_dir`` parameter.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from seeding import OFFLINE_SCHEDULER

from abe.api import FRONTEND_DIST, create_app

MARKER = "step10 fake built frontend"

# These tests boot the lifespan (and its scheduler) against EMPTY tmp dbs, so
# they use the shared OFFLINE_SCHEDULER (seeding.py) to keep the daily network
# fetch off. The startup run errors on the empty db — recorded, harmless, and
# irrelevant to the static-serving contract under test.


@pytest.fixture(autouse=True)
def _offline_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No FRED key + .env-less cwd -> the lifespan macro probe stays offline."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_dist_present_serves_index_and_api_routes_win(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(f"<!doctype html><p>{MARKER}</p>", encoding="utf-8")
    app = create_app(tmp_path / "abe.db", static_dir=dist, scheduler_config=OFFLINE_SCHEDULER)
    with TestClient(app) as client:
        # index.html served at / (html=True).
        root = client.get("/")
        assert root.status_code == 200
        assert MARKER in root.text
        # API routes are registered BEFORE the mount and still win.
        assert client.get("/health").json() == {"status": "ok"}
        latest = client.get("/api/runs/latest")
        assert latest.status_code == 404
        # Route-specific message uniquely proves the API handler executed
        # (a static-mount 404 would be a generic {"detail": "Not Found"}).
        assert latest.json()["detail"] == "no successful run yet"


def test_dist_absent_root_404_while_api_works(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "abe.db", static_dir=tmp_path / "no-dist", scheduler_config=OFFLINE_SCHEDULER
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/health").json() == {"status": "ok"}


def test_default_dist_path_points_at_frontend_dist() -> None:
    """The default resolves to <repo>/frontend/dist, cwd-independent."""
    assert FRONTEND_DIST.parts[-2:] == ("frontend", "dist")
    # Repo-root anchoring: the resolved parent must contain backend/abe.
    assert (FRONTEND_DIST.parent.parent / "backend" / "abe" / "api.py").is_file()
    # `is`-identity: create_app's default IS the module constant (a re-typed
    # cwd-relative literal in the signature would leave all other tests green).
    assert create_app.__kwdefaults__ is not None
    assert create_app.__kwdefaults__["static_dir"] is FRONTEND_DIST
