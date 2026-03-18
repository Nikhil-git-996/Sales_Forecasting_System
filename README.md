# 🏬 End-to-End Sales Forecasting System

> **Assignment – Data Science | MicroGCC Apprenticeship**  
> Forecast the **next 8 weeks of beverage sales** for each US state using four ML models and serve predictions via a production-ready **FastAPI** REST service.

---

## Project Structure

```
forecasting_system/
├── data/
│   └── Forecasting_Case-_Study.xlsx   # Raw dataset (43 states, 2019–2023)
├── src/
│   ├── data_loader.py                 # Load, clean, resample, split
│   ├── feature_engineering.py         # Lag / rolling / calendar / holiday features
│   ├── model_selector.py              # Train all models, compare by RMSE
│   ├── forecaster.py                  # Orchestrator: train → select → forecast → persist
│   └── models/
│       ├── arima_model.py             # SARIMA via pmdarima auto_arima
│       ├── prophet_model.py           # Facebook Prophet + US holidays
│       ├── xgboost_model.py           # XGBoost with recursive multi-step forecast
│       └── lstm_model.py              # Bidirectional LSTM (Keras / TensorFlow)
├── api/
│   ├── main.py                        # FastAPI app factory + lifespan
│   ├── routes.py                      # All HTTP endpoint handlers
│   └── schemas.py                     # Pydantic v2 request/response models
├── notebooks/
│   └── forecasting_walkthrough.ipynb  # EDA, training, visualisation demo
├── artifacts/                         # Auto-created: saved models + plots
├── train.py                           # Standalone CLI training script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1 · Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · (Optional) Pre-train models
```bash
python train.py                                  # all 43 states
python train.py --states "California" "Texas"    # specific states
```

### 3 · Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Dataset Overview

| Property | Value |
|---|---|
| Category | Beverages |
| States | 43 US states |
| Date range | Jan 2019 – Dec 2023 |
| Frequency | Weekly (irregular → resampled to **W-SAT**) |
| Missing values | 0 |
| Missing weeks | Linear interpolation after resampling |

---

## Architecture

```
Excel Dataset
     │
     ▼
┌─────────────────┐
│   data_loader   │  load_raw → prepare_state_series → train_val_split
└────────┬────────┘
         │  pd.Series per state (weekly, W-SAT)
         ▼
┌─────────────────────┐
│ feature_engineering │  lag, rolling, calendar, holiday
└────────┬────────────┘
         │  feature matrix (X) + target (y)
         ▼
┌───────────────────────────────────────────────────┐
│                 model_selector                    │
│   SARIMA  │  Prophet  │  XGBoost  │  LSTM         │
│           └──── compare RMSE on val ─────────────►│ best model
└────────┬──────────────────────────────────────────┘
         │  re-fit best model on full series
         ▼
┌──────────────┐     persist to disk
│  forecaster  │─────────────────────► artifacts/
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   FastAPI    │  /forecast/{state}  →  JSON
└──────────────┘
```

---

## Feature Engineering

All features are derived from values **strictly before** the target date (no leakage).

| Feature | Description |
|---|---|
| `lag_1` | Sales 1 week ago |
| `lag_4` | Sales 4 weeks ago (~monthly) |
| `lag_7` | Sales 7 weeks ago |
| `lag_13` | Sales 13 weeks ago (quarterly) |
| `lag_52` | Sales 52 weeks ago (same-week last year) |
| `roll_mean_4w/8w/13w` | Rolling mean over 4, 8, 13 weeks (shifted by 1) |
| `roll_std_4w/8w/13w` | Rolling standard deviation |
| `week_of_year` | ISO week number (1–52) |
| `month / quarter / year` | Calendar components |
| `sin_week / cos_week` | Cyclical encoding of week (avoids week 52 ↔ week 1 discontinuity) |
| `sin_month / cos_month` | Cyclical encoding of month |
| `day_of_week` | Day-of-week of period-end date |
| `holiday_flag` | 1 if any US federal holiday falls in that week |

**Train / validation split:**
```
|──────── TRAIN (all but last 8 weeks) ────────|── VAL (8w) ──|
```

---

## Models Implemented

### 1 · SARIMA
- `pmdarima.auto_arima` auto-selects (p,d,q)(P,D,Q,52) by AIC
- Falls back to non-seasonal ARIMA if seasonal fitting is slow
- Handles trend via differencing

### 2 · Facebook Prophet
- Additive seasonality with yearly + weekly components
- US federal holidays added as regressors
- `changepoint_prior_scale=0.05` prevents over-fitting on sparse data

### 3 · XGBoost
- Trained on full feature matrix (lag + rolling + calendar + holiday)
- **Recursive** multi-step forecasting: each predicted week feeds back as a lag feature
- L1 + L2 regularisation; `n_estimators=500`, `max_depth=5`
- Exposes feature importances

### 4 · Bidirectional LSTM
- `BiLSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1)`
- Sliding window of 13 weeks; MinMaxScaler; Huber loss
- EarlyStopping (patience=10) on 10% validation split
- Recursive multi-step forecasting

---

## Model Selection Strategy

```
For each state:
  1. Train all 4 models on training split
  2. Predict 8 validation weeks
  3. Compute RMSE, MAE, MAPE, R²
  4. Select winner by lowest RMSE
  5. Re-train winner on FULL series
  6. Generate 8-week ahead forecast
  7. Persist model + forecast to artifacts/
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/states` | List all 43 available states |
| `POST` | `/train/{state}` | Train all 4 models for one state |
| `POST` | `/train/all` | Train all states (background task) |
| `GET` | `/forecast/{state}?weeks=8` | 8-week sales forecast |
| `GET` | `/forecast/all` | Forecasts for all trained states |
| `GET` | `/metrics/{state}` | Validation metrics for all 4 models |
| `GET` | `/summary` | Leaderboard: best model + RMSE per state |

### Example

```bash
# Train + forecast California
curl -X POST http://localhost:8000/api/v1/train/California
curl      http://localhost:8000/api/v1/forecast/California
```

```json
{
  "state": "California",
  "best_model": "XGBoost",
  "forecast": [
    { "date": "2024-01-06", "sales": 452341000.00 },
    { "date": "2024-01-13", "sales": 461820000.00 },
    ...
  ]
}
```

---

## Docker

```bash
docker build -t sales-forecaster .
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  sales-forecaster
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/Forecasting_Case-_Study.xlsx` | Path to dataset |
| `ARTIFACTS_DIR` | `artifacts` | Model storage directory |
| `N_FORECAST` | `8` | Forecast horizon (weeks) |

---

*Built for the MicroGCC Data Science Apprenticeship case study.*
