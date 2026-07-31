import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Tuple, Dict, Any

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    from sklearn.linear_model import Ridge

from config import (
    SARIMAX_MODEL_PATH,
    TARGET_COL,
    ALL_EXOG_FEATURES
)
from src.dataset_generator import generate_nasa_power_dataset
from src.preprocessing import clean_dataset, split_train_test
from src.feature_engineering import add_engineered_features, fit_and_save_scalers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LinearExogBaseline:
    """Fallback linear exogenous regression baseline when statsmodels is uninstalled."""
    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        
    def fit(self, y, exog):
        self.model.fit(exog, y)
        return self
        
    def predict(self, start=0, end=0, exog=None):
        if exog is None:
            return np.zeros(end - start + 1)
        preds = self.model.predict(exog)
        return pd.Series(preds)

def check_stationarity(series: pd.Series) -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller (ADF) test for stationarity."""
    if HAS_STATSMODELS:
        result = adfuller(series.dropna())
        adf_stat = result[0]
        p_value = result[1]
        is_stationary = p_value < 0.05
        logger.info(f"ADF Statistic: {adf_stat:.4f}, p-value: {p_value:.4e} -> Stationary: {is_stationary}")
        return {"adf_stat": adf_stat, "p_value": p_value, "is_stationary": is_stationary}
    else:
        logger.info("Statsmodels not available; skipping ADF stationarity check.")
        return {"adf_stat": 0.0, "p_value": 0.0, "is_stationary": True}

def train_sarimax_model(train_df: pd.DataFrame, model_path: str = SARIMAX_MODEL_PATH) -> Tuple[Any, pd.Series]:
    """
    Train SARIMAX model using target solar power generation and exogenous weather features.
    Saves serialized model object to `models/sarimax.pkl`.
    Returns (fitted_model, train_residuals).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    y_train = train_df[TARGET_COL]
    exog_train = train_df[ALL_EXOG_FEATURES]

    logger.info("Performing ADF Stationarity Check on Target Series...")
    check_stationarity(y_train)

    if HAS_STATSMODELS:
        logger.info("Fitting SARIMAX(1, 0, 1)x(1, 0, 1, 24) model with exogenous weather features...")
        model = SARIMAX(
            endog=y_train,
            exog=exog_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarimax_results = model.fit(disp=False, maxiter=200)
    else:
        logger.info("Fitting Ridge Linear Exogenous Baseline model...")
        sarimax_results = LinearExogBaseline().fit(y_train, exog_train)
    
    # Calculate training residuals
    in_sample_preds = sarimax_results.predict(start=0, end=len(y_train)-1, exog=exog_train)
    if isinstance(in_sample_preds, pd.Series):
        in_sample_preds = in_sample_preds.values
    residuals = y_train.values - in_sample_preds
    
    # Save model artifact
    joblib.dump(sarimax_results, model_path)
    logger.info(f"SARIMAX model successfully saved to '{model_path}'")
    logger.info(f"SARIMAX Residual Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
    
    return sarimax_results, residuals

def run_sarimax_pipeline() -> None:
    """End-to-end pipeline execution for SARIMAX model training."""
    logger.info("--- Starting SARIMAX Training Pipeline ---")
    df_raw = generate_nasa_power_dataset()
    df_clean = clean_dataset(df_raw)
    df_feat = add_engineered_features(df_clean)
    
    train_df, test_df = split_train_test(df_feat)
    
    # Fit scalers
    fit_and_save_scalers(train_df)
    
    # Train SARIMAX
    train_sarimax_model(train_df)
    logger.info("--- SARIMAX Training Pipeline Completed ---")

if __name__ == "__main__":
    run_sarimax_pipeline()
