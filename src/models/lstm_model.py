"""
models/lstm_model.py
--------------------
Bidirectional LSTM for weekly sales forecasting.

Architecture
------------
  Input  → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1)

Uses a sliding-window approach: trains on sequences of `lookback` weeks
to predict the next week.  Multi-step forecasting is done recursively.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    LSTM forecaster wrapping a Keras model.

    Parameters
    ----------
    lookback   : number of past weeks used as input sequence
    epochs     : training epochs
    batch_size : mini-batch size
    units      : LSTM hidden units (first layer)
    dropout    : dropout rate between LSTM layers
    """

    name = "LSTM"

    def __init__(
        self,
        lookback:   int   = 13,   # 13 weeks ≈ 1 quarter
        epochs:     int   = 80,
        batch_size: int   = 16,
        units:      int   = 64,
        dropout:    float = 0.2,
        patience:   int   = 10,
        random_state: int = 42,
    ):
        self.lookback     = lookback
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.units        = units
        self.dropout      = dropout
        self.patience     = patience
        self.random_state = random_state

        self._model   = None
        self._scaler  = MinMaxScaler(feature_range=(0, 1))
        self._last_window: np.ndarray | None = None
        self.fitted_  = False

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback : i, 0])
            y.append(scaled[i, 0])
        return np.array(X)[..., np.newaxis], np.array(y)

    def _build_model(self):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        # Functional API with explicit Input() 
        inputs = tf.keras.Input(shape=(self.lookback, 1))
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.units, return_sequences=True)
        )(inputs)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.LSTM(self.units // 2)(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="huber")
        return model

    # ── Public interface ───────────────────────────────────────────────────────

    def fit(self, train: pd.Series) -> "LSTMForecaster":
        """Scale, build sequences, and train LSTM."""
        import tensorflow as tf

        logger.info(f"[LSTM] Fitting on {len(train)} observations …")

        values = train.values.reshape(-1, 1).astype(float)
        scaled = self._scaler.fit_transform(values)

        X, y = self._make_sequences(scaled)

        self._model = self._build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True, verbose=0,
            )
        ]

        self._model.fit(
            X, y,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            validation_split= 0.1,
            callbacks       = callbacks,
            verbose         = 0,
            shuffle         = False,
        )

        # Store the last window for recursive prediction
        self._last_window = scaled[-self.lookback :].copy()
        self.fitted_      = True
        logger.info("[LSTM] Fitting complete.")
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        """
        Recursive multi-step forecast.

        Each predicted value is appended to the sliding window before
        predicting the next step.
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        window = self._last_window.copy()       # shape (lookback, 1)
        predictions_scaled = []

        for _ in range(n_periods):
            x_input = window.reshape(1, self.lookback, 1)
            pred_scaled = float(self._model.predict(x_input, verbose=0)[0, 0])
            predictions_scaled.append(pred_scaled)
            window = np.append(window[1:], [[pred_scaled]], axis=0)

        predictions = self._scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()
        return np.clip(predictions, 0, None)

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
        return pd.Series(values, index=future_idx, name="lstm_forecast")

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path)
            logger.info(f"[LSTM] Model saved to {path}")

    def load(self, path: str) -> "LSTMForecaster":
        import tensorflow as tf
        self._model  = tf.keras.models.load_model(path)
        self.fitted_ = True
        return self
