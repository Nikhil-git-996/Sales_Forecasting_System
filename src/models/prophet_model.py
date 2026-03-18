"""
models/prophet_model.py
------------------------
Facebook Prophet wrapper for weekly sales forecasting.
Handles yearly seasonality, US holidays, and weekly patterns automatically.
"""

import numpy as np
import pandas as pd
import logging
import warnings

logger = logging.getLogger(__name__)

# ADD THIS — before the class definition
def _ensure_cmdstan():
    """Ensures CmdStan backend is ready. Fixes Windows/Python 3.10 stan_backend error."""
    try:
        import cmdstanpy
        cmdstanpy.install_cmdstan(quiet=True, overwrite=False)
    except Exception:
        pass
    
class ProphetForecaster:
    """
    Thin wrapper around Facebook Prophet.

    Prophet requires a DataFrame with columns ['ds', 'y'].
    We add US country holidays and let Prophet handle seasonality.
    """

    name = "Prophet"

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        country_holidays: str = "US",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
    ):
        self.yearly_seasonality    = yearly_seasonality
        self.weekly_seasonality    = weekly_seasonality
        self.daily_seasonality     = daily_seasonality
        self.country_holidays      = country_holidays
        self.changepoint_prior_scale  = changepoint_prior_scale
        self.seasonality_prior_scale  = seasonality_prior_scale
        self.seasonality_mode         = seasonality_mode
        self._model  = None
        self.fitted_ = False

    def _to_prophet_df(self, ts: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"ds": ts.index, "y": ts.values})

    def fit(self, train: pd.Series) -> "ProphetForecaster":
        """Fit Prophet on the training series."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required: pip install prophet")

        logger.info(f"[Prophet] Fitting on {len(train)} observations …")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ensure_cmdstan()
            self._model = Prophet(
                yearly_seasonality    = self.yearly_seasonality,
                weekly_seasonality    = self.weekly_seasonality,
                daily_seasonality     = self.daily_seasonality,
                changepoint_prior_scale  = self.changepoint_prior_scale,
                seasonality_prior_scale  = self.seasonality_prior_scale,
                seasonality_mode         = self.seasonality_mode,
            )
            if self.country_holidays:
                self._model.add_country_holidays(country_name=self.country_holidays)

            prophet_df = self._to_prophet_df(train)
            self._model.fit(prophet_df)

        self.fitted_ = True
        logger.info("[Prophet] Fitting complete.")
        return self

    def predict(self, n_periods: int, freq: str = "W-SAT") -> np.ndarray:
        """Forecast `n_periods` weeks ahead. Returns ndarray of predictions."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        future = self._model.make_future_dataframe(
            periods=n_periods, freq=freq, include_history=False
        )
        forecast = self._model.predict(future)
        values = np.clip(forecast["yhat"].values, 0, None)
        return values

    def forecast_series(
        self,
        train: pd.Series,
        n_periods: int,
        freq: str = "W-SAT",
    ) -> pd.Series:
        """Fit + predict; return a dated pd.Series."""
        self.fit(train)
        values = self.predict(n_periods, freq=freq)
        future_idx = pd.date_range(
            start=train.index[-1],
            periods=n_periods + 1,
            freq=freq,
        )[1:]
        # Align lengths (Prophet may return ± 1 row depending on freq)
        values = values[: len(future_idx)]
        return pd.Series(values, index=future_idx, name="prophet_forecast")
