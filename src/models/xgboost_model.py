"""
models/xgboost_model.py
-----------------------
XGBoost regressor trained on lag + rolling + calendar features.
Uses iterative (recursive) multi-step forecasting.
"""

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

from src.feature_engineering import build_features, get_feature_columns

logger = logging.getLogger(__name__)

FEATURE_COLS = get_feature_columns()


class XGBoostForecaster:
    """
    Multi-step XGBoost forecaster using recursive prediction.

    For each future week, the model predicts one step ahead and
    appends the prediction to the series before computing the next
    step's features (recursive / iterated strategy).
    """

    name = "XGBoost"

    def __init__(
        self,
        n_estimators:    int   = 500,
        max_depth:       int   = 5,
        learning_rate:   float = 0.05,
        subsample:       float = 0.85,
        colsample_bytree:float = 0.85,
        reg_alpha:       float = 0.1,
        reg_lambda:      float = 1.0,
        random_state:    int   = 42,
    ):
        from xgboost import XGBRegressor
        self._model = XGBRegressor(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            reg_alpha         = reg_alpha,
            reg_lambda        = reg_lambda,
            random_state      = random_state,
            objective         = "reg:squarederror",
            n_jobs            = -1,
            verbosity         = 0,
        )
        self._train_series: pd.Series | None = None
        self.fitted_ = False

    def fit(self, train: pd.Series) -> "XGBoostForecaster":
        """Build feature matrix from `train` and fit XGBoost."""
        logger.info(f"[XGBoost] Building features from {len(train)} observations …")

        feat_df  = build_features(train, dropna=True)
        X        = feat_df[FEATURE_COLS]
        y        = feat_df["sales"]

        self._model.fit(X, y)
        self._train_series = train.copy()
        self.fitted_       = True
        logger.info(f"[XGBoost] Fitting complete. Feature importances computed.")
        return self

    def predict(self, n_periods: int, freq: str = "W-SAT") -> np.ndarray:
        """
        Recursive multi-step forecast.

        Each future step uses the predicted value of the previous step
        to compute lag features, preventing data leakage.
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        ts = self._train_series.copy()
        predictions = []

        for _ in range(n_periods):
            # Build feature matrix for extended series (no dropna so future rows exist)
            feat_df = build_features(ts, dropna=False)

            # Take the last row (the upcoming week)
            last_row = feat_df[FEATURE_COLS].iloc[[-1]]
            # Fill any remaining NaNs with column medians from training
            last_row = last_row.fillna(feat_df[FEATURE_COLS].median())

            pred = float(self._model.predict(last_row)[0])
            pred = max(pred, 0)   # non-negative sales
            predictions.append(pred)

            # Append prediction to series for next iteration
            next_date = ts.index[-1] + pd.tseries.frequencies.to_offset(freq)
            ts = pd.concat([ts, pd.Series([pred], index=[next_date])])

        return np.array(predictions)

    def forecast_series(
        self,
        train: pd.Series,
        n_periods: int,
        freq: str = "W-SAT",
    ) -> pd.Series:
        """Fit + predict; return a dated pd.Series."""
        self.fit(train)
        values    = self.predict(n_periods, freq=freq)
        future_idx = pd.date_range(
            start=train.index[-1],
            periods=n_periods + 1,
            freq=freq,
        )[1:]
        return pd.Series(values, index=future_idx, name="xgboost_forecast")

    def feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted pd.Series."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted.")
        return pd.Series(
            self._model.feature_importances_,
            index=FEATURE_COLS,
        ).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        joblib.dump(self._model, path)
        logger.info(f"[XGBoost] Model saved to {path}")

    def load(self, path: str | Path) -> "XGBoostForecaster":
        self._model = joblib.load(path)
        self.fitted_ = True
        return self
