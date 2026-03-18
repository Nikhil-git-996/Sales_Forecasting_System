"""
model_selector.py
-----------------
Trains all four models on the training split, evaluates them on the
validation split, and selects the best model per state by RMSE.

Metrics computed
----------------
  RMSE  – Root Mean Squared Error      (primary ranking metric)
  MAE   – Mean Absolute Error
  MAPE  – Mean Absolute Percentage Error
  R²    – Coefficient of Determination
"""

import numpy as np
import pandas as pd
import logging
import time
from dataclasses import dataclass, field

from src.models.arima_model   import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model    import LSTMForecaster

logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "rmse": rmse(actual, predicted),
        "mae":  mae(actual, predicted),
        "mape": mape(actual, predicted),
        "r2":   r2(actual, predicted),
    }


# ── Data class for results ────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_name:  str
    metrics:     dict
    fit_seconds: float
    forecaster:  object   # fitted model instance


@dataclass
class SelectionResult:
    state:       str
    best_model:  str
    all_metrics: dict        # model_name → metrics dict
    results:     list[ModelResult] = field(default_factory=list)
    val_forecasts: dict = field(default_factory=dict)   # model_name → np.ndarray


# ── Selector class ────────────────────────────────────────────────────────────

class ModelSelector:
    """
    Trains and compares SARIMA, Prophet, XGBoost, and LSTM for a given state.

    Usage
    -----
    selector = ModelSelector()
    result   = selector.run(state_name, train_series, val_series)
    """

    def __init__(self, n_forecast: int = 8, freq: str = "W-SAT"):
        self.n_forecast = n_forecast
        self.freq       = freq

    def _run_model(
        self,
        forecaster,
        train: pd.Series,
        val_len: int,
    ) -> tuple[np.ndarray, float]:
        """Fit a model and return (val_predictions, fit_duration_seconds)."""
        t0 = time.time()
        forecaster.fit(train)
        preds = forecaster.predict(val_len)
        elapsed = time.time() - t0
        return np.array(preds[:val_len]), elapsed

    def run(
        self,
        state:  str,
        train:  pd.Series,
        val:    pd.Series,
    ) -> SelectionResult:
        """
        Train all models, evaluate on val, return a SelectionResult.
        """
        val_len  = len(val)
        actual   = val.values

        models = {
            "SARIMA":  ARIMAForecaster(seasonal=True, m=52),
            "Prophet": ProphetForecaster(),
            "XGBoost": XGBoostForecaster(),
            "LSTM":    LSTMForecaster(lookback=13, epochs=80, patience=10),
        }

        results: list[ModelResult] = []
        val_forecasts: dict        = {}

        for model_name, forecaster in models.items():
            logger.info(f"[{state}] Training {model_name} …")
            try:
                preds, elapsed = self._run_model(forecaster, train, val_len)
                metrics = compute_metrics(actual, preds)
                results.append(ModelResult(
                    model_name  = model_name,
                    metrics     = metrics,
                    fit_seconds = elapsed,
                    forecaster  = forecaster,
                ))
                val_forecasts[model_name] = preds
                logger.info(
                    f"[{state}] {model_name} → RMSE={metrics['rmse']:.2f}, "
                    f"MAPE={metrics['mape']:.2f}%, time={elapsed:.1f}s"
                )
            except Exception as e:
                logger.error(f"[{state}] {model_name} failed: {e}", exc_info=True)

        if not results:
            raise RuntimeError(f"All models failed for state '{state}'.")

        # Select best by RMSE
        best = min(results, key=lambda r: r.metrics["rmse"])
        all_metrics = {r.model_name: r.metrics for r in results}

        logger.info(f"[{state}] ✅ Best model: {best.model_name} "
                    f"(RMSE={best.metrics['rmse']:.2f})")

        return SelectionResult(
            state         = state,
            best_model    = best.model_name,
            all_metrics   = all_metrics,
            results       = results,
            val_forecasts = val_forecasts,
        )
