"""
api/schemas.py
--------------
Pydantic v2 request / response models for the FastAPI layer.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ForecastPoint(BaseModel):
    date:  str   = Field(..., description="ISO date string (YYYY-MM-DD)")
    sales: float = Field(..., description="Forecasted sales value", ge=0)


class ForecastResponse(BaseModel):
    state:      str
    best_model: str
    forecast:   list[ForecastPoint]

    class Config:
        json_schema_extra = {
            "example": {
                "state": "California",
                "best_model": "XGBoost",
                "forecast": [
                    {"date": "2024-01-06", "sales": 450000000.00},
                    {"date": "2024-01-13", "sales": 460000000.00},
                ],
            }
        }


class MetricsEntry(BaseModel):
    rmse: float
    mae:  float
    mape: float
    r2:   float


class ModelMetricsResponse(BaseModel):
    state:       str
    best_model:  str
    all_metrics: dict[str, MetricsEntry]

    class Config:
        json_schema_extra = {
            "example": {
                "state": "California",
                "best_model": "XGBoost",
                "all_metrics": {
                    "SARIMA":  {"rmse": 10000, "mae": 8000, "mape": 2.1, "r2": 0.95},
                    "Prophet": {"rmse": 11000, "mae": 9000, "mape": 2.4, "r2": 0.94},
                    "XGBoost": {"rmse":  9000, "mae": 7500, "mape": 1.9, "r2": 0.96},
                    "LSTM":    {"rmse": 12000, "mae": 9500, "mape": 2.7, "r2": 0.93},
                },
            }
        }


class TrainResponse(BaseModel):
    state:      str
    best_model: str
    message:    str
    metrics:    dict[str, MetricsEntry]


class AllForecastsResponse(BaseModel):
    total_states: int
    forecasts:    list[ForecastResponse]


class HealthResponse(BaseModel):
    status:         str
    trained_states: int
    version:        str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
