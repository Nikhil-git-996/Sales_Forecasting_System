"""
api/main.py
-----------
FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docker:
    docker build -t sales-forecaster .
    docker run -p 8000:8000 sales-forecaster
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routes import router, set_system
from src.forecaster import SalesForecastingSystem

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config from env vars (with sensible defaults) ──────────────────────────────
DATA_PATH     = os.getenv("DATA_PATH",     "data/Forecasting_Case-_Study.xlsx")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
N_FORECAST    = int(os.getenv("N_FORECAST", "8"))


# ── Lifespan: runs on startup & shutdown ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Sales Forecasting API …")

    system = SalesForecastingSystem(
        data_path     = DATA_PATH,
        artifacts_dir = ARTIFACTS_DIR,
        n_forecast    = N_FORECAST,
    )
    system.load_data()   # pre-load dataset at startup
    set_system(system)

    logger.info(f"✅ Dataset loaded. {len(system.states)} states available.")
    logger.info("📡 API ready. Visit http://localhost:8000/docs for Swagger UI.")

    yield  # ← application runs here

    logger.info("🛑 Shutting down …")


# ── App factory ────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Sales Forecasting API",
        description = (
            "End-to-end time series forecasting system.\n\n"
            "Trains **SARIMA, Prophet, XGBoost, LSTM** per US state and "
            "automatically selects the best model by validation RMSE.\n\n"
            "Use `POST /train/{state}` to train, then `GET /forecast/{state}` "
            "to retrieve the next 8 weeks of predicted sales."
        ),
        version     = "1.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # CORS – allow all origins for development (restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # Mount all routes under /api/v1
    app.include_router(router, prefix="/api/v1")

    # Root redirect
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    return app


app = create_app()
