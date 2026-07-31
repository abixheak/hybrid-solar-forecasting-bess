import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Dict, Any

# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

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

def create_residual_sequences(
    residuals: np.ndarray,
    exog_features: np.ndarray,
    seq_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence sliding windows of past residuals + exogenous weather features
    to predict the target residual error at the next timestep.
    """
    X, y = [], []
    num_samples = len(residuals)
    
    # Combined feature matrix: [exog_features, residual]
    combined = np.column_stack([exog_features, residuals])
    
    for i in range(seq_length, num_samples):
        X.append(combined[i - seq_length:i, :])
        y.append(residuals[i])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_residual_lstm_model(input_shape: Tuple[int, int]) -> keras.Model:
    """
    Build a 2-Layer LSTM neural network specialized in predicting SARIMAX residual corrections.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_Residual_Predictor")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

def train_lstm_residuals_model(
    residuals: pd.Series,
    train_df: pd.DataFrame,
    seq_length: int = 24,
    epochs: int = 15,
    batch_size: int = 32,
    model_path: str = LSTM_MODEL_PATH
) -> keras.Model:
    """
    Train LSTM model exclusively on SARIMAX residuals and save artifact to `hybrid_lstm_residuals.keras`.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scalers = load_scalers(SCALER_PATH)
    
    feature_scaler = scalers["feature_scaler"]
    scaled_exog = feature_scaler.transform(train_df[ALL_EXOG_FEATURES])
    
    # Scale residuals
    res_vals = residuals.values.reshape(-1, 1)
    res_std = res_vals.std() if res_vals.std() > 0 else 1.0
    scaled_res = res_vals / (3.0 * res_std) # Normalized residual bounds
    
    X_seq, y_seq = create_residual_sequences(scaled_res.flatten(), scaled_exog, seq_length=seq_length)
    
    logger.info(f"LSTM Training dataset shape -> X: {X_seq.shape}, y: {y_seq.shape}")
    
    model = build_residual_lstm_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
    
    logger.info("Training LSTM residual predictor...")
    model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=1
    )
    
    # Save model in .keras native format
    model.save(model_path)
    logger.info(f"Successfully trained & saved residual LSTM model to '{model_path}'")
    return model

def run_lstm_pipeline() -> None:
    """Execute end-to-end LSTM pipeline using pre-calculated SARIMAX residuals."""
    logger.info("--- Starting LSTM Residual Training Pipeline ---")
    
    if not os.path.exists(SARIMAX_MODEL_PATH):
        raise FileNotFoundError("SARIMAX model pickle not found. Run train_sarimax.py first!")
        
    sarimax_model = joblib.load(SARIMAX_MODEL_PATH)
    
    df_raw = generate_nasa_power_dataset()
    df_clean = clean_dataset(df_raw)
    df_feat = add_engineered_features(df_clean)
    train_df, _ = split_train_test(df_feat)
    
    # Extract in-sample SARIMAX predictions and calculate residuals
    exog_train = train_df[ALL_EXOG_FEATURES]
    y_train = train_df[TARGET_COL]
    sarimax_preds = sarimax_model.predict(start=0, end=len(y_train)-1, exog=exog_train)
    if isinstance(sarimax_preds, pd.Series):
        sarimax_preds = sarimax_preds.values
    residuals = pd.Series(y_train.values - sarimax_preds)
    
    train_lstm_residuals_model(residuals, train_df)
    logger.info("--- LSTM Residual Training Pipeline Completed ---")

if __name__ == "__main__":
    run_lstm_pipeline()
