"""HTTP service exposing the fraud-scoring model."""

from fastapi import FastAPI
from pydantic import BaseModel

from txn_sentinel import __version__

app = FastAPI(title="txn-sentinel", version=__version__)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe for the container platform and uptime monitoring."""
    return HealthResponse(status="ok", version=__version__)
