"""
forecaster.py
-------------
High-level orchestrator that ties together:
  data loading  →  model selection  →  final 8-week forecast
  →  persistence (save/load state to disk)

This is the object the FastAPI layer calls.
"""

import json
import logging
import pickle
from pathlib import Path
from unittest import result

import numpy as np
import pandas as pd
from panel import state

from src.data_loader    import load_all_states, prepare_state_series, load_raw
from src.model_selector import ModelSelector, SelectionResult
from src.models.arima_model   import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model    import LSTMForecaster

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {
    "SARIMA":  ARIMAForecaster,
    "Prophet": ProphetForecaster,
    "XGBoost": XGBoostForecaster,
    "LSTM":    LSTMForecaster,
}


class SalesForecastingSystem:
    """
    End-to-end sales forecasting system.

    Parameters
    ----------
    data_path   : path to the Excel dataset
    artifacts_dir : directory to store pickled models / results
    n_forecast  : number of future weeks to predict
    val_weeks   : validation window size (same as n_forecast)
    freq        : pandas offset alias for weekly frequency
    """

    def __init__(
        self,
        data_path:     str | Path = "data/Forecasting_Case-_Study.xlsx",
        artifacts_dir: str | Path = "artifacts",
        n_forecast:    int  = 8,
        val_weeks:     int  = 8,
        freq:          str  = "W-SAT",
    ):
        self.data_path     = Path(data_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.n_forecast    = n_forecast
        self.val_weeks     = val_weeks
        self.freq          = freq

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state
        self._state_data:    dict[str, dict] = {}   # loaded from Excel
        self._results:       dict[str, SelectionResult] = {}
        self._final_models:  dict[str, object] = {}   # best fitted model per state
        self._forecasts:     dict[str, pd.Series] = {}

    # ── Data loading ───────────────────────────────────────────────────────────

    def load_data(self) -> "SalesForecastingSystem":
        logger.info(f"Loading data from {self.data_path} …")
        self._state_data = load_all_states(
            self.data_path,
            freq      = self.freq,
            val_weeks = self.val_weeks,
        )
        logger.info(f"Data loaded for {len(self._state_data)} states.")

        # Auto-load all previously trained states from artifacts/ on startup
        loaded = 0
        for state in self.states:
            if self._load_state(state):
                loaded += 1
        if loaded > 0:
            logger.info(f"✅ Auto-loaded {loaded} pre-trained states from artifacts/")
        else:
            logger.info("No pre-trained artifacts found. Use POST /train/all to train.")

        return self

    @property
    def states(self) -> list[str]:
        return sorted(self._state_data.keys())

    # ── Training ───────────────────────────────────────────────────────────────

    def train_state(self, state: str) -> SelectionResult:
        """
        Run model selection for a single state and store the best model.
        Then re-fit the best model on the FULL series for final forecasting.
        """

        if not self._state_data:
            self.load_data()

        # Avoid retraining if already trained
        if state in self._results or self._load_state(state):
            logger.info(f"[{state}] Already trained. Using cached model.")
            return self._results[state]

        data = self._state_data.get(state)
        if data is None:
            raise ValueError(f"State '{state}' not found.")

        logger.info(f"[{state}] Starting training...")

        selector = ModelSelector(n_forecast=self.n_forecast, freq=self.freq)
        result   = selector.run(state, data["train"], data["val"])
        self._results[state] = result

        # Re-train best model on FULL series (no val holdout)
        best_cls   = _MODEL_CLASSES[result.best_model]
        best_model = best_cls()
        best_model.fit(data["full_series"])
        self._final_models[state] = best_model

        # Generate forecast
        values = best_model.predict(self.n_forecast)
        future_idx = pd.date_range(
            start=data["full_series"].index[-1],
            periods=self.n_forecast + 1,
            freq=self.freq,
        )[1:]

        self._forecasts[state] = pd.Series(
            np.clip(values, 0, None),
            index=future_idx,
            name=f"{state}_forecast",
        )

        # Save artifacts
        self._save_state(state)

        logger.info(f"[{state}] Training completed. Best model: {result.best_model}")

        return result

    def train_all(self) -> dict[str, SelectionResult]:
        """Train all states sequentially (skip already trained states)."""

        if not self._state_data:
            self.load_data()

        for state in self.states:
            try:
                # Skip already trained states
                if state in self._results or self._load_state(state):
                    logger.info(f"[{state}] Skipping (already trained)")
                    continue

                self.train_state(state)

            except Exception as e:
                logger.error(f"Training failed for {state}: {e}", exc_info=True)

        logger.info("✅ Training completed for all states.")

        return self._results

    # ── Forecasting ────────────────────────────────────────────────────────────

    def get_forecast(self, state: str, n_weeks: int | None = None) -> dict:
        """
        Return the forecast for a state.
        If the state hasn't been trained, train it on-the-fly.
        """
        if state not in self._forecasts:
            self._load_state(state)   # try loading from disk
        if state not in self._forecasts:
            self.train_state(state)

        fc = self._forecasts[state]

        # If caller wants a different horizon, re-forecast
        if n_weeks and n_weeks != self.n_forecast:
            model  = self._final_models[state]
            values = model.predict(n_weeks)
            data   = self._state_data[state]["full_series"]
            future_idx = pd.date_range(
                start=data.index[-1],
                periods=n_weeks + 1,
                freq=self.freq,
            )[1:]
            fc = pd.Series(np.clip(values, 0, None), index=future_idx)

        return {
            "state":      state,
            "best_model": self._results[state].best_model,
            "forecast": [
                {"date": str(d.date()), "sales": round(float(v), 2)}
                for d, v in zip(fc.index, fc.values)
            ],
        }

    def get_metrics(self, state: str) -> dict:
        """Return validation metrics for all models for a given state."""
        if state not in self._results:
            self._load_state(state)
        if state not in self._results:
            raise ValueError(f"State '{state}' has not been trained yet.")
        return {
            "state":       state,
            "best_model":  self._results[state].best_model,
            "all_metrics": self._results[state].all_metrics,
        }

    def get_all_forecasts(self) -> list[dict]:
        """Return forecasts for all trained states."""
        return [self.get_forecast(s) for s in sorted(self._forecasts.keys())]

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_state(self, state: str) -> None:
        safe_name = state.replace(" ", "_")
        result_path = self.artifacts_dir / f"{safe_name}_result.pkl"
        model_path  = self.artifacts_dir / f"{safe_name}_model.pkl"

        with open(result_path, "wb") as f:
            # Don't pickle the heavy model object inside result
            r = self._results[state]
            lightweight = {
                "state":       r.state,
                "best_model":  r.best_model,
                "all_metrics": r.all_metrics,
            }
            pickle.dump(lightweight, f)

        with open(model_path, "wb") as f:
            pickle.dump({
                "model":    self._final_models[state],
                "forecast": self._forecasts[state],
            }, f)

        logger.info(f"[{state}] Artifacts saved.")

    def _load_state(self, state: str) -> bool:
        safe_name   = state.replace(" ", "_")
        result_path = self.artifacts_dir / f"{safe_name}_result.pkl"
        model_path  = self.artifacts_dir / f"{safe_name}_model.pkl"

        if not result_path.exists() or not model_path.exists():
            return False

        with open(result_path, "rb") as f:
            lightweight = pickle.load(f)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Reconstruct a minimal SelectionResult
        from src.model_selector import SelectionResult
        self._results[state] = SelectionResult(
            state       = lightweight["state"],
            best_model  = lightweight["best_model"],
            all_metrics = lightweight["all_metrics"],
        )
        self._final_models[state] = model_data["model"]
        self._forecasts[state]    = model_data["forecast"]
        logger.info(f"[{state}] Artifacts loaded from disk.")
        return True

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising best model & RMSE per state."""
        rows = []
        for state, result in self._results.items():
            best = result.best_model
            rows.append({
                "state":      state,
                "best_model": best,
                "rmse":       result.all_metrics[best]["rmse"],
                "mape":       result.all_metrics[best]["mape"],
                "mae":        result.all_metrics[best]["mae"],
            })
        return pd.DataFrame(rows).sort_values("rmse")
