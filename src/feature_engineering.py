import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from config import (
    ALL_EXOG_FEATURES,
    TARGET_COL,
    SCALER_PATH
)

logger = logging.getLogger(__name__)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features for SARIMAX and LSTM models:
    - Cyclical hour and month sine/cosine features
    - Cloud Attenuation Index
    - Solar lag features (1h, 24h)
    - Rolling temperature and irradiance statistics
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Temporal cyclical features
    hours = df["timestamp"].dt.hour
    months = df["timestamp"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * months / 12.0)

    # 2. Cloud Attenuation Index
    if "clear_sky_irradiance" in df.columns and "direct_radiation" in df.columns:
        clear_sky = df["clear_sky_irradiance"].values
        direct_rad = df["direct_radiation"].values
        attenuation = np.where(clear_sky > 10.0, 1.0 - (direct_rad / clear_sky), 0.0)
        df["cloud_attenuation_index"] = np.clip(attenuation, 0.0, 1.0)
    elif "cloud_cover" in df.columns:
        df["cloud_attenuation_index"] = df["cloud_cover"] / 100.0
    else:
        df["cloud_attenuation_index"] = 0.2

    # 3. Lag features (for target solar power if present, or zero fill)
    if TARGET_COL in df.columns:
        df["solar_lag_1"] = df[TARGET_COL].shift(1).fillna(0.0)
        df["solar_lag_24"] = df[TARGET_COL].shift(24).fillna(0.0)
    else:
        df["solar_lag_1"] = 0.0
        df["solar_lag_24"] = 0.0

    # 4. Rolling statistics
    if "temperature_2m" in df.columns:
        df["temp_rolling_6h"] = df["temperature_2m"].rolling(window=6, min_periods=1).mean()
    else:
        df["temp_rolling_6h"] = 25.0

    if "direct_radiation" in df.columns:
        df["rad_rolling_6h"] = df["direct_radiation"].rolling(window=6, min_periods=1).mean()
    else:
        df["rad_rolling_6h"] = 0.0

    # Ensure all expected exogenous columns exist
    for col in ALL_EXOG_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df

def fit_and_save_scalers(df: pd.DataFrame, scaler_path: str = SCALER_PATH) -> Dict[str, MinMaxScaler]:
    """Fit MinMaxScalers for features and residual targets and save to file."""
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    residual_scaler = MinMaxScaler(feature_range=(-1, 1))

    feature_scaler.fit(df[ALL_EXOG_FEATURES])
    
    scalers = {
        "feature_scaler": feature_scaler,
        "residual_scaler": residual_scaler,
        "features": ALL_EXOG_FEATURES
    }
    
    joblib.dump(scalers, scaler_path)
    logger.info(f"Saved feature scalers to {scaler_path}")
    return scalers

def load_scalers(scaler_path: str = SCALER_PATH) -> Dict[str, Any]:
    """Load pre-trained scalers from file."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at '{scaler_path}'. Run training pipeline first.")
    return joblib.load(scaler_path)
