import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Any

import keras
from keras import layers, callbacks

from config import (
    LSTM_MODEL_PATH,
    SARIMAX_MODEL_PATH,
    SCALER_PATH,
    TARGET_COL,
    ALL_EXOG_FEATURES
)
from src.preprocessing import clean_dataset, split_train_test
from src.feature_engineering import add_engineered_features, load_scalers
from src.dataset_generator import generate_nasa_power_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_lstm_model(input_shape: Tuple[int, int]) -> keras.Model:
    """
    Build a stacked LSTM model for residual error correction.

    Architecture:
        - LSTM layer (128 units) with dropout for sequence modelling
        - LSTM layer (64 units) with dropout for deeper temporal abstraction
        - Dense(32) with ReLU for non-linear feature mixing
        - Dense(1) linear output for residual scalar prediction
    """
    inputs = keras.Input(shape=input_shape, name="sequence_input")
    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="hybrid_lstm_residuals")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def create_residual_sequences(
    residuals: np.ndarray,
    exog_features: np.ndarray,
    seq_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences of past residuals + exogenous weather features
    to predict the target residual error at the next timestep.
    """
    X, y = [], []
    num_samples = len(residuals)
    combined = np.column_stack([exog_features, residuals])

    for i in range(seq_length, num_samples):
        X.append(combined[i - seq_length:i, :])
        y.append(residuals[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm_residuals_model(
    residuals: pd.Series,
    train_df: pd.DataFrame,
    seq_length: int = 24,
    epochs: int = 50,
    batch_size: int = 32,
    model_path: str = LSTM_MODEL_PATH
) -> keras.Model:
    """
    Train a stacked LSTM model on SARIMAX residuals and save the Keras model artifact.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scalers = load_scalers(SCALER_PATH)

    feature_scaler = scalers["feature_scaler"]
    scaled_exog = feature_scaler.transform(train_df[ALL_EXOG_FEATURES])

    res_vals = residuals.values.reshape(-1, 1)
    res_std = float(res_vals.std()) if res_vals.std() > 0 else 1.0
    scaled_res = res_vals / (3.0 * res_std)

    # Persist res_std so inference can correctly invert: res_kw = lstm_out * 3.0 * res_std
    scalers["res_std"] = res_std
    joblib.dump(scalers, SCALER_PATH)
    logger.info(f"Saved res_std={res_std:.4f} kW into scalers.pkl for inference inverse-scaling.")

    X_seq, y_seq = create_residual_sequences(scaled_res.flatten(), scaled_exog, seq_length=seq_length)

    logger.info(f"LSTM Training dataset shape -> X: {X_seq.shape}, y: {y_seq.shape}")
    logger.info(f"Input shape per sample: (seq_length={seq_length}, features={X_seq.shape[2]})")

    input_shape = (X_seq.shape[1], X_seq.shape[2])
    model = build_lstm_model(input_shape)
    model.summary(print_fn=logger.info)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    logger.info("Fitting LSTM Residual Model...")
    model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    model.save(model_path)
    logger.info(f"Successfully saved LSTM model to '{model_path}'")
    return model


def run_lstm_pipeline() -> None:
    """Execute end-to-end LSTM residual correction training pipeline using pre-calculated SARIMAX residuals."""
    from src.train_sarimax import LinearExogBaseline
    logger.info("--- Starting LSTM Residual Training Pipeline ---")

    if not os.path.exists(SARIMAX_MODEL_PATH):
        raise FileNotFoundError(f"SARIMAX model missing: '{SARIMAX_MODEL_PATH}'. Run train_sarimax.py first!")

    sarimax_model = joblib.load(SARIMAX_MODEL_PATH)

    df_raw = generate_nasa_power_dataset()
    df_clean = clean_dataset(df_raw)
    df_feat = add_engineered_features(df_clean)
    train_df, _ = split_train_test(df_feat)

    exog_train = train_df[ALL_EXOG_FEATURES]
    y_train = train_df[TARGET_COL]
    sarimax_preds = sarimax_model.predict(start=0, end=len(y_train)-1, exog=exog_train)
    if isinstance(sarimax_preds, pd.Series):
        sarimax_preds = sarimax_preds.values
    residuals = pd.Series(y_train.values - sarimax_preds)

    train_lstm_residuals_model(residuals, train_df)

    logger.info("Evaluating hybrid pipeline on test dataset and writing model_metrics.json...")
    try:
        from src.evaluation import get_test_dataset_evaluation
        metrics_df, acc = get_test_dataset_evaluation()
        logger.info(f"Hybrid Forecast Accuracy on test set: {acc}%")
    except Exception as e:
        logger.warning(f"Could not compute test metrics: {e}")

    logger.info("--- LSTM Residual Training Pipeline Completed ---")


if __name__ == "__main__":
    run_lstm_pipeline()
