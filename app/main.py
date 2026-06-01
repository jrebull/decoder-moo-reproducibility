"""FastAPI app — Visa Predict AI backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.api import scenarios, pareto, convergence, optimize, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the problem singleton on startup
    scenarios.get_problem()
    yield


app = FastAPI(
    title="Visa Predict AI",
    description="MOHHO optimizer for US EB visa allocation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(scenarios.router)
app.include_router(pareto.router)
app.include_router(convergence.router)
app.include_router(optimize.router)
app.include_router(ws.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
