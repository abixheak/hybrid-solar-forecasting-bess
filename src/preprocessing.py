import pandas as pd
import numpy as np
import logging
from typing import Tuple
from config import RAW_DATASET_PATH, TARGET_COL, WEATHER_FEATURES

logger = logging.getLogger(__name__)

def load_raw_data(file_path: str = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded raw dataset from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading raw dataset from {file_path}: {e}")
        raise

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    1. Datetime construction
    2. Duplicate removal
    3. Missing value imputation
    4. Outlier detection and clipping
    """
    df = df.copy()

    # 1. Datetime index construction if not present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif all(col in df.columns for col in ["YEAR", "MONTH", "DAY", "HOUR"]):
        df["timestamp"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY", "HOUR"]].assign(MINUTE=0, SECOND=0))
    else:
        raise ValueError("Dataset lacks valid timestamp columns")

    # Sort and set timestamp index
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # 2. Missing value handling (forward fill + backward fill)
    df = df.ffill().bfill()

    # 3. Outlier handling (clip extreme values to physical domain bounds)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0.0)
        
    if "temperature_2m" in df.columns:
        df["temperature_2m"] = df["temperature_2m"].clip(lower=-10.0, upper=60.0)
        
    if "relative_humidity_2m" in df.columns:
        df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(lower=0.0, upper=100.0)
        
    if "direct_radiation" in df.columns:
        df["direct_radiation"] = df["direct_radiation"].clip(lower=0.0, upper=1400.0)
        
    if "cloud_cover" in df.columns:
        df["cloud_cover"] = df["cloud_cover"].clip(lower=0.0, upper=100.0)

    logger.info(f"Dataset cleaned successfully. Final row count: {len(df)}")
    return df

def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and test sets."""
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    logger.info(f"Chronological Split -> Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    return train_df, test_df
