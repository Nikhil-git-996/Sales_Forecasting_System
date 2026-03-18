"""
api/routes.py
-------------
IMPORTANT – route ordering rule in FastAPI:
  Static paths (/train/all, /forecast/all) MUST be registered BEFORE
  dynamic paths (/train/{state}, /forecast/{state}).
  FastAPI evaluates routes in declaration order; if the dynamic route
  comes first it captures the literal string "all" as the path parameter
  and the static route is never reached → 404.
"""

import logging
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks

from api.schemas import (
    ForecastResponse, ForecastPoint,
    ModelMetricsResponse, MetricsEntry,
    TrainResponse, AllForecastsResponse, HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_system = None

def set_system(system):
    global _system
    _system = system


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        trained_states=len(_system._forecasts) if _system else 0,
    )


# ── States ────────────────────────────────────────────────────────────────────

@router.get("/states", tags=["Data"])
def list_states():
    """Return all states available in the dataset."""
    if not _system._state_data:
        _system.load_data()
    return {"states": _system.states, "count": len(_system.states)}


# ── Training  (static route BEFORE dynamic) ───────────────────────────────────

@router.post("/train/all", tags=["Training"])
def train_all_states(background_tasks: BackgroundTasks):
    """Kick off training for all 43 states in the background."""
    if not _system._state_data:
        _system.load_data()

    def _run():
        _system.train_all()
        logger.info("All states trained successfully.")

    background_tasks.add_task(_run)
    return {
        "message": "Training started for all states in the background.",
        "states":  _system.states,
    }


@router.post("/train/{state}", response_model=TrainResponse, tags=["Training"])
def train_state(state: str):
    """Train all 4 models for a single state and auto-select the best by RMSE."""
    if not _system._state_data:
        _system.load_data()

    available = _system.states
    matched   = next((s for s in available if s.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. Available: {available}",
        )

    try:
        result = _system.train_state(matched)
    except Exception as e:
        logger.error(f"Training error for {matched}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return TrainResponse(
        state      = matched,
        best_model = result.best_model,
        message    = f"Training complete. Best model: {result.best_model}",
        metrics    = {k: MetricsEntry(**v) for k, v in result.all_metrics.items()},
    )


# ── Forecasting  (static route BEFORE dynamic) ────────────────────────────────

@router.get("/forecast/all", response_model=AllForecastsResponse, tags=["Forecast"])
def get_all_forecasts():
    """Return forecasts for every state that has been trained so far."""
    forecasts = _system.get_all_forecasts()
    return AllForecastsResponse(
        total_states = len(forecasts),
        forecasts    = [
            ForecastResponse(
                state      = f["state"],
                best_model = f["best_model"],
                forecast   = [ForecastPoint(**p) for p in f["forecast"]],
            )
            for f in forecasts
        ],
    )


@router.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(
    state: str,
    weeks: int = Query(default=8, ge=1, le=52, description="Number of weeks to forecast"),
):
    """
    Return the next `weeks` weeks of sales forecast for the given state.
    If the state has not been trained yet, training is triggered automatically.
    """
    if not _system._state_data:
        _system.load_data()

    available = _system.states
    matched   = next((s for s in available if s.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    try:
        fc_dict = _system.get_forecast(matched, n_weeks=weeks)
    except Exception as e:
        logger.error(f"Forecast error for {matched}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ForecastResponse(
        state      = fc_dict["state"],
        best_model = fc_dict["best_model"],
        forecast   = [ForecastPoint(**p) for p in fc_dict["forecast"]],
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

@router.get("/metrics/{state}", response_model=ModelMetricsResponse, tags=["Evaluation"])
def get_metrics(state: str):
    """Return validation-set RMSE / MAE / MAPE / R² for all 4 models."""
    available = _system.states
    matched   = next((s for s in available if s.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    try:
        m = _system.get_metrics(matched)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ModelMetricsResponse(
        state       = m["state"],
        best_model  = m["best_model"],
        all_metrics = {k: MetricsEntry(**v) for k, v in m["all_metrics"].items()},
    )


# ── Summary ───────────────────────────────────────────────────────────────────

@router.get("/summary", tags=["Evaluation"])
def get_summary():
    """Leaderboard: best model + RMSE for every trained state."""
    if not _system._results:
        return {"message": "No states trained yet. Use POST /train/{state} to start."}

    df = _system.summary()
    return {
        "summary":       df.to_dict(orient="records"),
        "total_trained": len(df),
    }