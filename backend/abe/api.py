"""FastAPI app. Minimal for the scaffold; real routes arrive in Step 8.

uvicorn target: ``abe.api:app`` (127.0.0.1:8140, single worker, no --reload).
"""

from fastapi import FastAPI

app = FastAPI(title="always-best-estimates")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
