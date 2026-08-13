import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from config import CITIES, DB_PATH
from src.database import insert_weather_records, fetch_weather_records

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def compute_clear_sky_irradiance(lat: float, timestamps: pd.Series) -> np.ndarray:
    """
    Compute theoretical clear-sky direct irradiance (W/m²) from latitude and timestamps
    using the same cosine-of-zenith physics formula as dataset_generator.py.
    Matches the feature distribution the SARIMAX was trained on.
    """
    clear_sky_vals = []
    for ts in pd.to_datetime(timestamps):
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
        hour_angle = (hour - 12) * 15.0
        lat_rad = np.radians(lat)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)
        cos_zenith = (np.sin(lat_rad) * np.sin(dec_rad)
                      + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
        if cos_zenith > 0 and 6 <= hour <= 18:
            cs = 1000.0 * cos_zenith * (1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365)))
            clear_sky_vals.append(float(np.clip(cs, 0.0, 1150.0)))
        else:
            clear_sky_vals.append(0.0)
    return np.array(clear_sky_vals, dtype=np.float32)

def fetch_open_meteo_forecast(city_name: str, forecast_days: int = 7) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetch hourly forecast from Open-Meteo API for specified city.
    Returns (records, source) where source is "live" or "synthetic".
    Callers should surface a visible warning when source == "synthetic".
    """
    if city_name not in CITIES:
        raise ValueError(f"City '{city_name}' is not in supported list: {list(CITIES.keys())}")

    coords = CITIES[city_name]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,direct_radiation,cloud_cover",
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        timestamps = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        rads = hourly.get("direct_radiation", [])
        clouds = hourly.get("cloud_cover", [])

        records = []
        for i in range(len(timestamps)):
            records.append({
                "location": city_name,
                "timestamp": timestamps[i],
                "temperature": float(temps[i]) if temps[i] is not None else 28.0,
                "humidity": float(humidity[i]) if humidity[i] is not None else 60.0,
                "cloud_cover": float(clouds[i]) if clouds[i] is not None else 20.0,
                "irradiance": float(rads[i]) if rads[i] is not None else 0.0,
                "wind_speed": float(winds[i]) if winds[i] is not None else 3.5
            })

        logger.info(f"Successfully fetched {len(records)} hours of live weather for {city_name} from Open-Meteo.")
        return records, "live"

    except Exception as e:
        logger.warning(f"Failed to fetch Open-Meteo forecast for {city_name}: {e}. Falling back to synthetic weather generation.")
        return generate_synthetic_weather_records(city_name, days=forecast_days), "synthetic"

def generate_synthetic_weather_records(city_name: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Generate realistic synthetic hourly weather telemetry if offline.
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    records = []
    
    base_temp = 25.0 + (CITIES[city_name]["lat"] % 5)
    
    for h in range(days * 24):
        dt = now + timedelta(hours=h)
        hour = dt.hour

        # Diurnal temperature cycle (peaks at 14:00, drops at 04:00)
        temp = base_temp + 6.0 * np.sin((hour - 8) * np.pi / 12) + np.random.normal(0, 0.5)
        humidity = 60.0 - 15.0 * np.sin((hour - 8) * np.pi / 12) + np.random.normal(0, 2.0)
        humidity = float(np.clip(humidity, 20.0, 95.0))
        
        # Diurnal direct solar radiation (W/m2)
        if 6 <= hour <= 18:
            solar_peak = 750.0 + np.random.normal(0, 30.0)
            irradiance = solar_peak * np.sin((hour - 6) * np.pi / 12)
            irradiance = max(0.0, float(irradiance))
        else:
            irradiance = 0.0
            
        cloud_cover = float(np.clip(25.0 + 10.0 * np.sin(h * 0.1) + np.random.normal(0, 5.0), 0.0, 100.0))
        wind_speed = float(np.clip(3.5 + 1.5 * np.sin(h * 0.2) + np.random.normal(0, 0.5), 0.5, 12.0))

        records.append({
            "location": city_name,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "cloud_cover": round(cloud_cover, 2),
            "irradiance": round(irradiance, 2),
            "wind_speed": round(wind_speed, 2),
            "clear_sky_irradiance": round(
                compute_clear_sky_irradiance(
                    CITIES[city_name]["lat"],
                    pd.Series([dt.strftime("%Y-%m-%d %H:%M:%S")])
                )[0], 2
            )
        })
        
    return records

def sync_city_weather_to_sqlite(city_name: str, forecast_days: int = 7) -> Tuple[pd.DataFrame, str]:
    """
    Fetch weather from Open-Meteo, store into SQLite database, and read back from SQLite.
    Returns (DataFrame, source) where source is "live" or "synthetic".
    The source flag should be surfaced in the UI when synthetic data is in use.
    Also injects clear_sky_irradiance from physics formula to match training data distribution.
    """
    records, source = fetch_open_meteo_forecast(city_name, forecast_days=forecast_days)
    insert_weather_records(records, db_path=DB_PATH)
    df_sqlite = fetch_weather_records(city_name, limit=forecast_days * 24, db_path=DB_PATH)
    
    # Map column aliases so ML models receive expected exogenous feature names
    if "temperature" in df_sqlite.columns:
        df_sqlite["temperature_2m"] = df_sqlite["temperature"]
    if "humidity" in df_sqlite.columns:
        df_sqlite["relative_humidity_2m"] = df_sqlite["humidity"]
    if "wind_speed" in df_sqlite.columns:
        df_sqlite["wind_speed_10m"] = df_sqlite["wind_speed"]
    if "irradiance" in df_sqlite.columns:
        df_sqlite["direct_radiation"] = df_sqlite["irradiance"]

    # Inject physics-based clear_sky_irradiance so cloud_attenuation_index
    # is computed correctly in feature_engineering (matches training distribution)
    lat = CITIES[city_name]["lat"]
    if "clear_sky_irradiance" not in df_sqlite.columns or df_sqlite["clear_sky_irradiance"].isna().all():
        df_sqlite["clear_sky_irradiance"] = compute_clear_sky_irradiance(
            lat, df_sqlite["timestamp"]
        )
        logger.info(f"Injected clear_sky_irradiance for {city_name} (lat={lat}).")

    return df_sqlite, source
