"""
data_loader.py
--------------
Loads and preprocesses the raw Excel file into clean, per-state weekly time series.
Handles:
  - Mixed date formats
  - Irregular / missing dates  →  resampled to weekly (W-SAT) with linear interpolation
  - Per-state splitting
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_raw(filepath: str | Path) -> pd.DataFrame:
    """Read Excel file and return a tidy DataFrame with proper dtypes."""
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = df.columns.str.strip()

    # Normalise column names to lowercase
    df = df.rename(columns={
        "State": "state",
        "Date":  "date",
        "Total": "sales",
        "Category": "category",
    })

    # Parse dates robustly
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])               # drop any un-parseable dates
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["state"] = df["state"].str.strip()

    logger.info(f"Raw data loaded: {len(df)} rows, "
                f"{df['state'].nunique()} states, "
                f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def get_states(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique state names."""
    return sorted(df["state"].unique().tolist())


def prepare_state_series(
    df: pd.DataFrame,
    state: str,
    freq: str = "W-SAT",
) -> pd.Series:
    """
    Extract a clean weekly time series for a single state.

    Steps
    -----
    1. Filter state rows
    2. Aggregate duplicate dates (sum)
    3. Set date as index & sort
    4. Resample to `freq` – fills missing weeks with NaN
    5. Interpolate (linear) then back-fill edge NaNs
    """
    state_df = df[df["state"] == state][["date", "sales"]].copy()

    if state_df.empty:
        raise ValueError(f"State '{state}' not found in dataset.")

    # Aggregate duplicates
    state_df = (
        state_df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
        .set_index("date")
    )

    # Resample to regular weekly frequency
    ts = state_df["sales"].resample(freq).sum()

    # Replace zeros introduced by resample for genuinely missing weeks with NaN
    # (zeros that existed in raw data are preserved only if adjacent values confirm them)
    ts = ts.replace(0, np.nan)
    ts = ts.interpolate(method="linear").bfill().ffill()

    logger.info(f"[{state}] Series length after resampling: {len(ts)}")
    return ts


def train_val_split(
    ts: pd.Series,
    val_weeks: int = 8,
) -> tuple[pd.Series, pd.Series]:
    """
    Temporal split: last `val_weeks` for validation, rest for training.
    No data leakage – validation is always strictly after training.
    """
    train = ts.iloc[:-val_weeks]
    val   = ts.iloc[-val_weeks:]
    return train, val


def load_all_states(
    filepath: str | Path,
    freq: str = "W-SAT",
    val_weeks: int = 8,
) -> dict[str, dict]:
    """
    Convenience loader: returns a dict keyed by state name.

    Each value is:
        {
            "full_series": pd.Series,
            "train":       pd.Series,
            "val":         pd.Series,
        }
    """
    df = load_raw(filepath)
    states = get_states(df)
    result = {}

    for state in states:
        try:
            ts = prepare_state_series(df, state, freq=freq)
            train, val = train_val_split(ts, val_weeks=val_weeks)
            result[state] = {
                "full_series": ts,
                "train": train,
                "val":   val,
            }
        except Exception as e:
            logger.warning(f"Skipping state '{state}': {e}")

    logger.info(f"Prepared {len(result)} states successfully.")
    return result
