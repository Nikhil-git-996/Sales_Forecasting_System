"""
models/arima_model.py
---------------------
SARIMA model using pmdarima's auto_arima for automatic order selection.
Handles seasonality (period=52 for yearly weekly seasonality).
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    Wrapper around pmdarima.auto_arima with SARIMA support.

    Auto-selects (p,d,q)(P,D,Q,m) using AIC minimisation.
    Falls back to a simpler ARIMA if seasonal fitting is too slow.
    """

    name = "SARIMA"

    def __init__(
        self,
        seasonal: bool = True,
        m: int = 52,          # weekly yearly seasonality
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 1,
        max_Q: int = 1,
        stepwise: bool = True,
        information_criterion: str = "aic",
    ):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.stepwise = stepwise
        self.information_criterion = information_criterion
        self._model = None
        self.fitted_ = False

    def fit(self, train: pd.Series) -> "ARIMAForecaster":
        """Fit auto_arima on the training series."""
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("pmdarima is required: pip install pmdarima")

        logger.info(f"[SARIMA] Fitting on {len(train)} observations …")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model = pm.auto_arima(
                    train.values,
                    seasonal=self.seasonal,
                    m=self.m,
                    max_p=self.max_p,
                    max_q=self.max_q,
                    max_P=self.max_P,
                    max_Q=self.max_Q,
                    stepwise=self.stepwise,
                    information_criterion=self.information_criterion,
                    error_action="ignore",
                    suppress_warnings=True,
                )
            except Exception as e:
                logger.warning(f"[SARIMA] Seasonal fit failed ({e}). Retrying non-seasonal …")
                self._model = pm.auto_arima(
                    train.values,
                    seasonal=False,
                    max_p=self.max_p,
                    max_q=self.max_q,
                    stepwise=self.stepwise,
                    error_action="ignore",
                    suppress_warnings=True,
                )

        self.fitted_ = True
        order = self._model.order
        sorder = getattr(self._model, "seasonal_order", None)
        logger.info(f"[SARIMA] Order: {order}, Seasonal: {sorder}")
        return self

    def predict(self, n_periods: int, return_conf_int: bool = False):
        """
        Forecast `n_periods` steps ahead.

        Returns
        -------
        np.ndarray of shape (n_periods,)  – or tuple (forecast, conf_int)
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        forecast, conf_int = self._model.predict(
            n_periods=n_periods,
            return_conf_int=True,
        )
        forecast = np.clip(forecast, 0, None)   # sales cannot be negative
        if return_conf_int:
            return forecast, conf_int
        return forecast

    def forecast_series(
        self,
        train: pd.Series,
        n_periods: int,
        freq: str = "W-SAT",
    ) -> pd.Series:
        """Fit + predict; return a dated pd.Series."""
        self.fit(train)
        values = self.predict(n_periods)
        future_idx = pd.date_range(
            start=train.index[-1],
            periods=n_periods + 1,
            freq=freq,
        )[1:]
        return pd.Series(values, index=future_idx, name="sarima_forecast")
