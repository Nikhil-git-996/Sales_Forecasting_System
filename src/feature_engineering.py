"""
feature_engineering.py
-----------------------
Creates a rich feature matrix from a weekly time series.

Features created
----------------
  Lag features  : t-1, t-7 (weeks), t-30 (weeks proxied as t-4)
  Rolling stats : mean/std over 4-week, 8-week, 13-week windows
  Calendar      : week-of-year, month, quarter, year
  Seasonal      : sin/cos encoding of week-of-year
  Holiday flag  : US federal holidays in the same week as the date
  Target        : sales (original)

Note: "t-7 weeks" and "t-30 weeks" are adapted to a weekly series
      as lag-7 and lag-4 respectively, matching the spirit of the
      assignment (which was designed for daily data).  We also add
      lag-2 and lag-13 for richer context.
"""

import pandas as pd
import numpy as np
import holidays
import logging

logger = logging.getLogger(__name__)

# US federal holiday calendar
_US_HOLIDAYS = holidays.country_holidays("US")


def _has_holiday(dt: pd.Timestamp) -> int:
    """Return 1 if the iso-week that contains `dt` has a US federal holiday."""
    week_start = dt - pd.Timedelta(days=dt.weekday())
    for offset in range(7):
        if (week_start + pd.Timedelta(days=offset)) in _US_HOLIDAYS:
            return 1
    return 0


def build_features(ts: pd.Series, dropna: bool = True) -> pd.DataFrame:
    """
    Build feature matrix from a weekly pd.Series (index = DatetimeIndex).

    Parameters
    ----------
    ts      : weekly time series with DatetimeIndex
    dropna  : drop rows that have NaN in any lag/rolling feature (default True)

    Returns
    -------
    pd.DataFrame with columns [lag_*, roll_*, calendar_*, holiday_flag, sales]
    """
    df = pd.DataFrame({"sales": ts})
    idx = df.index

    # ── Lag features ──────────────────────────────────────────────────────────
    # t-1 week  (most recent week)
    df["lag_1"]  = df["sales"].shift(1)
    # t-7 weeks (quarter-behind)
    df["lag_7"]  = df["sales"].shift(7)
    # t-4 weeks (~monthly)  – proxy for "t-30 days" in weekly data
    df["lag_4"]  = df["sales"].shift(4)
    # t-13 weeks (quarterly)
    df["lag_13"] = df["sales"].shift(13)
    # t-52 weeks (annual same-week)
    df["lag_52"] = df["sales"].shift(52)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for window in [4, 8, 13]:
        rolled = df["sales"].shift(1).rolling(window)
        df[f"roll_mean_{window}w"] = rolled.mean()
        df[f"roll_std_{window}w"]  = rolled.std()

    # ── Calendar features ─────────────────────────────────────────────────────
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["month"]        = idx.month
    df["quarter"]      = idx.quarter
    df["year"]         = idx.year

    # Cyclical (sin/cos) encoding to capture periodicity
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Day-of-week of the period-end date (constant for W-SAT = 5, kept for generality)
    df["day_of_week"] = idx.dayofweek

    # ── Holiday flag ──────────────────────────────────────────────────────────
    df["holiday_flag"] = [_has_holiday(d) for d in idx]

    if dropna:
        before = len(df)
        df = df.dropna()
        logger.debug(f"Dropped {before - len(df)} rows due to NaN in lag/rolling features.")

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns (excluding target 'sales')."""
    lags    = ["lag_1", "lag_4", "lag_7", "lag_13", "lag_52"]
    rolling = [f"roll_{stat}_{w}w" for w in [4, 8, 13] for stat in ["mean", "std"]]
    cal     = ["week_of_year", "month", "quarter", "year",
               "sin_week", "cos_week", "sin_month", "cos_month", "day_of_week"]
    return lags + rolling + cal + ["holiday_flag"]


def prepare_future_features(
    ts: pd.Series,
    n_weeks: int = 8,
) -> pd.DataFrame:
    """
    Extend the feature matrix into the future for forecasting.

    Strategy
    --------
    Append `n_weeks` NaN rows to `ts`, then call `build_features`.
    Lag and rolling features for future rows will propagate from
    the last observed values (where possible).  For multi-step
    XGBoost / LSTM, the caller is responsible for iterative prediction.

    Returns a DataFrame containing ONLY the future rows (n_weeks rows).
    """
    freq = ts.index.freq or pd.tseries.frequencies.to_offset("W-SAT")
    future_dates = pd.date_range(
        start=ts.index[-1] + freq,
        periods=n_weeks,
        freq=freq,
    )
    future_ts = pd.Series(np.nan, index=future_dates, name="sales")
    extended  = pd.concat([ts, future_ts])
    features  = build_features(extended, dropna=False)
    return features.loc[future_dates]
