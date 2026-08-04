import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Dict, Any
from src.train_sarimax import LinearExogBaseline
from sklearn.neural_network import MLPRegressor

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

class ResidualNeuralNetwork:
    """
    Multi-Layer Perceptron (MLP) Neural Network specialized in predicting
    SARIMAX residual error corrections from exogenous weather features.
    """
    def __init__(self, hidden_layer_sizes: Tuple[int, int] = (64, 32), max_iter: int = 300):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=42,
            early_stopping=True
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
        self.model.fit(X_flat, y)
        return self
        
    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
        preds = self.model.predict(X_flat)
        return np.expand_dims(preds, axis=1)

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
    combined = np.column_stack([exog_features, residuals])
    
    for i in range(seq_length, num_samples):
        X.append(combined[i - seq_length:i, :])
        y.append(residuals[i])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_lstm_residuals_model(
    residuals: pd.Series,
    train_df: pd.DataFrame,
    seq_length: int = 24,
    epochs: int = 15,
    batch_size: int = 32,
    model_path: str = LSTM_MODEL_PATH
) -> Any:
    """
    Train residual neural network model exclusively on SARIMAX residuals
    and save serialized model artifact to models directory.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scalers = load_scalers(SCALER_PATH)
    
    feature_scaler = scalers["feature_scaler"]
    scaled_exog = feature_scaler.transform(train_df[ALL_EXOG_FEATURES])
    
    res_vals = residuals.values.reshape(-1, 1)
    res_std = res_vals.std() if res_vals.std() > 0 else 1.0
    scaled_res = res_vals / (3.0 * res_std)
    
    X_seq, y_seq = create_residual_sequences(scaled_res.flatten(), scaled_exog, seq_length=seq_length)
    
    logger.info(f"Residual NN Training dataset shape -> X: {X_seq.shape}, y: {y_seq.shape}")
    logger.info("Fitting Neural Residual Predictor...")
    
    model = ResidualNeuralNetwork(hidden_layer_sizes=(64, 32), max_iter=300).fit(X_seq, y_seq)
    
    # Save model artifact using joblib
    joblib.dump(model, model_path)
    logger.info(f"Successfully saved residual neural network model to '{model_path}'")
    return model

def run_lstm_pipeline() -> None:
    """Execute end-to-end residual neural network pipeline using pre-calculated SARIMAX residuals."""
    logger.info("--- Starting Residual Neural Network Training Pipeline ---")
    
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

    logger.info("--- Residual Neural Network Training Pipeline Completed ---")

if __name__ == "__main__":
    run_lstm_pipeline()
