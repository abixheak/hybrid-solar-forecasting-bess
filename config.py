import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure required directories exist
for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, DATABASE_DIR, ASSETS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Database
DB_PATH = os.path.join(DATABASE_DIR, "solar_data_fleet.db")

# Admin Credentials
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Saved Model Paths
SARIMAX_MODEL_PATH = os.path.join(MODEL_DIR, "sarimax.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_lstm_residuals.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scalers.pkl")
RAW_DATASET_PATH = os.path.join(DATA_RAW_DIR, "nasa_power_solar_data.csv")

# Supported Cities with Coordinates & Base Demand (kW)
CITIES = {
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "base_demand_kw": 450.0, "peak_demand_kw": 850.0},
    "New Delhi": {"lat": 28.6139, "lon": 77.2090, "base_demand_kw": 550.0, "peak_demand_kw": 1100.0},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "base_demand_kw": 500.0, "peak_demand_kw": 950.0},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "base_demand_kw": 400.0, "peak_demand_kw": 750.0},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "base_demand_kw": 420.0, "peak_demand_kw": 800.0},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "base_demand_kw": 430.0, "peak_demand_kw": 820.0},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "base_demand_kw": 480.0, "peak_demand_kw": 900.0},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "base_demand_kw": 380.0, "peak_demand_kw": 720.0},
}

# Feature definitions
TARGET_COL = "solar_power_kw"
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "direct_radiation",
    "cloud_cover"
]

ENGINEERED_FEATURES = [
    "cloud_attenuation_index",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "solar_lag_1",
    "solar_lag_24",
    "temp_rolling_6h",
    "rad_rolling_6h"
]

ALL_EXOG_FEATURES = WEATHER_FEATURES + ENGINEERED_FEATURES

# BESS Defaults
BESS_DEFAULTS = {
    "capacity_kwh": 1000.0,      # 1 MWh nominal capacity
    "initial_soc_kwh": 500.0,     # 50% initial SOC
    "max_charge_kw": 250.0,       # 0.25C charge rate
    "max_discharge_kw": 250.0,    # 0.25C discharge rate
    "charge_efficiency": 0.95,    # 95% charge efficiency
    "discharge_efficiency": 0.95, # 95% discharge efficiency
    "min_soc_pct": 10.0,          # 10% minimum SOC
    "max_soc_pct": 90.0,          # 90% maximum SOC
}

# UI Theme Color Palette
COLORS = {
    "background": "#0b0f19",
    "card_bg": "rgba(22, 31, 49, 0.75)",
    "accent_cyan": "#00f2fe",
    "accent_blue": "#4facfe",
    "accent_amber": "#ffb703",
    "accent_green": "#06d6a0",
    "accent_red": "#ef476f",
    "text_main": "#f8fafc",
    "text_muted": "#94a3b8",
    "border_glow": "rgba(0, 242, 254, 0.25)"
}
