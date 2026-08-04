import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Tuple

from config import (
    SARIMAX_MODEL_PATH,
    LSTM_MODEL_PATH,
    SCALER_PATH,
    ALL_EXOG_FEATURES,
    TARGET_COL
)
import sys
from src.feature_engineering import add_engineered_features, load_scalers
from src.train_sarimax import LinearExogBaseline
from src.train_lstm import ResidualNeuralNetwork
setattr(sys.modules['__main__'], 'LinearExogBaseline', LinearExogBaseline)
setattr(sys.modules['__main__'], 'ResidualNeuralNetwork', ResidualNeuralNetwork)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HybridForecaster:
    """
    Production Inference Engine loading pre-trained SARIMAX and Residual Neural Network model artifacts.
    NEVER retrains models in deployment.
    """
    def __init__(self):
        self.sarimax_model = None
        self.lstm_model = None
        self.scalers = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load pretrained models and scalers into memory."""
        if not os.path.exists(SARIMAX_MODEL_PATH):
            raise FileNotFoundError(f"Missing SARIMAX model artifact: '{SARIMAX_MODEL_PATH}'. Execute pipeline training script first.")
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(f"Missing LSTM model artifact: '{LSTM_MODEL_PATH}'. Execute pipeline training script first.")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Missing scalers artifact: '{SCALER_PATH}'. Execute pipeline training script first.")

        logger.info("Loading pretrained model artifacts...")
        self.sarimax_model = joblib.load(SARIMAX_MODEL_PATH)
        self.scalers = load_scalers(SCALER_PATH)
        self.lstm_model = joblib.load(LSTM_MODEL_PATH)
        logger.info("Pretrained model artifacts loaded successfully.")

    def predict(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Hybrid Solar Power Forecast from real-time or forecasted weather features.
        
        Final Forecast = SARIMAX Prediction + LSTM Residual Correction
        """
        df_feat = add_engineered_features(weather_df)
        
        # 1. SARIMAX Baseline Prediction
        exog_df = df_feat[ALL_EXOG_FEATURES]
        
        try:
            raw_p = self.sarimax_model.predict(start=0, end=len(df_feat) - 1, exog=exog_df)
            sarimax_preds = raw_p.values if hasattr(raw_p, "values") else np.array(raw_p)
        except Exception:
            raw_p = self.sarimax_model.forecast(steps=len(df_feat), exog=exog_df)
            sarimax_preds = raw_p.values if hasattr(raw_p, "values") else np.array(raw_p)
            
        sarimax_preds = np.clip(sarimax_preds, 0.0, None)
        
        # 2. Residual Error Correction Prediction
        feature_scaler = self.scalers["feature_scaler"]
        scaled_exog = feature_scaler.transform(exog_df)
        
        seq_length = 24
        num_samples = len(df_feat)
        
        lstm_corrections = []
        current_res_seq = np.zeros((seq_length, 1))
        
        for i in range(num_samples):
            if i >= seq_length:
                exog_win = scaled_exog[i - seq_length:i, :]
                res_win = current_res_seq[-seq_length:, :]
            else:
                pad_len = seq_length - (i + 1)
                exog_sub = scaled_exog[:i+1, :]
                exog_win = np.vstack([np.repeat(scaled_exog[0:1, :], pad_len, axis=0), exog_sub])
                res_win = current_res_seq
                
            win_input = np.column_stack([exog_win, res_win])
            win_input_batch = np.expand_dims(win_input, axis=0) # Shape: (1, 24, num_features + 1)
            
            # Predict residual correction
            try:
                raw_res_pred = self.lstm_model.predict(win_input_batch, verbose=0)
                if isinstance(raw_res_pred, np.ndarray) and len(raw_res_pred.shape) > 1:
                    raw_res_pred = float(raw_res_pred[0][0])
                elif isinstance(raw_res_pred, np.ndarray):
                    raw_res_pred = float(raw_res_pred[0])
                else:
                    raw_res_pred = float(raw_res_pred)
            except Exception:
                raw_res_pred = 0.0
            
            res_kw = raw_res_pred * 25.0
            lstm_corrections.append(res_kw)
            current_res_seq = np.vstack([current_res_seq[1:], [[raw_res_pred]]])
            
        lstm_corrections = np.array(lstm_corrections)
        
        # 3. Final Hybrid Prediction = SARIMAX + LSTM Correction
        hybrid_preds = sarimax_preds + lstm_corrections
        
        # Enforce solar physics bounds: Nighttime (low irradiance) => 0 generation
        for i in range(num_samples):
            rad = df_feat.iloc[i].get("direct_radiation", 0.0)
            if rad < 5.0:
                sarimax_preds[i] = 0.0
                lstm_corrections[i] = 0.0
                hybrid_preds[i] = 0.0
            else:
                hybrid_preds[i] = max(0.0, hybrid_preds[i])

        result_df = weather_df.copy()
        result_df["sarimax_prediction"] = np.round(sarimax_preds, 2)
        result_df["lstm_residual_correction"] = np.round(lstm_corrections, 2)
        result_df["hybrid_final_forecast"] = np.round(hybrid_preds, 2)
        
        return result_df

# Global forecaster instance container
_forecaster_instance = None

def run_hybrid_forecast(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Public API endpoint to run pre-trained Hybrid SARIMAX-LSTM inference on weather data.
    """
    global _forecaster_instance
    if _forecaster_instance is None:
        _forecaster_instance = HybridForecaster()
    return _forecaster_instance.predict(weather_df)
