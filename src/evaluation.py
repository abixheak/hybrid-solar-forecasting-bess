import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) avoiding zero division."""
    # Filter daylight hours where y_true > epsilon to avoid near-zero distortions
    mask = y_true > epsilon
    if not np.any(mask):
        return 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return float(mape)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Compute rigorous empirical regression metrics: MAE, RMSE, MAPE, R-squared.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE (kW)": round(float(mae), 3),
        "RMSE (kW)": round(float(rmse), 3),
        "MAPE (%)": round(float(mape), 2),
        "R2 Score": round(float(r2), 4)
    }

def generate_evaluation_report(df: pd.DataFrame, target_col: str, sarimax_col: str, hybrid_col: str) -> pd.DataFrame:
    """
    Generate model comparison report comparing baseline SARIMAX vs Hybrid SARIMAX-LSTM.
    """
    y_true = df[target_col].values
    y_sarimax = df[sarimax_col].values
    y_hybrid = df[hybrid_col].values

    sarimax_metrics = evaluate_predictions(y_true, y_sarimax, model_name="SARIMAX Baseline")
    hybrid_metrics = evaluate_predictions(y_true, y_hybrid, model_name="Hybrid SARIMAX-LSTM")

    # Compute percentage improvement
    mae_imp = ((sarimax_metrics["MAE (kW)"] - hybrid_metrics["MAE (kW)"]) / sarimax_metrics["MAE (kW)"]) * 100.0 if sarimax_metrics["MAE (kW)"] > 0 else 0
    rmse_imp = ((sarimax_metrics["RMSE (kW)"] - hybrid_metrics["RMSE (kW)"]) / sarimax_metrics["RMSE (kW)"]) * 100.0 if sarimax_metrics["RMSE (kW)"] > 0 else 0
    
    metrics_df = pd.DataFrame([sarimax_metrics, hybrid_metrics])
    return metrics_df
