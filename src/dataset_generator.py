import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from config import RAW_DATASET_PATH, TARGET_COL

logger = logging.getLogger(__name__)

def generate_nasa_power_dataset(output_path: str = RAW_DATASET_PATH, num_hours: int = 8760) -> pd.DataFrame:
    """
    Generate realistic NASA POWER solar irradiance and weather dataset covering 1 full year (8760 hours).
    Includes clear sky irradiance, all sky irradiance, temperature, humidity, wind speed,
    and physics-derived solar power generation target with noise and cloud attenuation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_hours)]
    
    records = []
    for dt in timestamps:
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        day_of_year = dt.timetuple().tm_yday

        # Solar zenith angle dynamics
        # Solar declination angle approximation
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
        # Hour angle (0 at solar noon 12:00)
        hour_angle = (hour - 12) * 15.0
        
        # Latitude approximation for solar farm (e.g. 20 degrees North)
        lat_rad = np.radians(20.5)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)
        
        # Cosine of zenith angle
        cos_zenith = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        
        if cos_zenith > 0 and 6 <= hour <= 18:
            # Clear Sky Irradiance (W/m2)
            clear_sky = 1000.0 * cos_zenith * (1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365)))
            clear_sky = float(np.clip(clear_sky, 0, 1150.0))
        else:
            clear_sky = 0.0
            
        # Cloud cover dynamics (stochastic weather pattern)
        cloud_factor = 0.15 + 0.25 * np.sin(day_of_year * 0.05) + np.random.uniform(-0.1, 0.25)
        cloud_factor = np.clip(cloud_factor, 0.0, 0.85)
        
        # All Sky Irradiance (attenuated by cloud cover)
        all_sky = clear_sky * (1.0 - 0.75 * cloud_factor)
        all_sky = float(np.clip(all_sky, 0, clear_sky if clear_sky > 0 else 0))
        
        # Temperature (C) with seasonal & diurnal fluctuations
        seasonal_temp = 22.0 + 8.0 * np.sin(np.radians((day_of_year - 100) * 360 / 365))
        diurnal_temp = 5.0 * np.sin(np.radians((hour - 9) * 15))
        temperature = seasonal_temp + diurnal_temp + np.random.normal(0, 0.8)
        
        # Relative Humidity (%) inversely correlated with temp
        humidity = 65.0 - 1.2 * diurnal_temp + np.random.normal(0, 3.0)
        humidity = float(np.clip(humidity, 15.0, 95.0))
        
        # Wind speed (m/s)
        wind_speed = float(np.clip(3.5 + 1.2 * np.sin(hour * np.pi / 12) + np.random.normal(0, 0.8), 0.5, 14.0))

        # Solar Power Generation (kW) for a 500 kW Nominal Solar Farm
        # Photovoltaic efficiency drops slightly with high cell temperature: P = Irradiance * Area * Eff * Temp_coeff
        pv_capacity_kw = 500.0
        temp_coeff = 1.0 - 0.004 * (temperature - 25.0)
        inverter_efficiency = 0.96
        
        power_gen = (all_sky / 1000.0) * pv_capacity_kw * temp_coeff * inverter_efficiency
        power_gen += np.random.normal(0, 3.5) # Sensor noise
        power_gen = max(0.0, float(power_gen)) if all_sky > 5.0 else 0.0

        records.append({
            "YEAR": year,
            "MONTH": month,
            "DAY": day,
            "HOUR": hour,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "clear_sky_irradiance": round(clear_sky, 2),
            "direct_radiation": round(all_sky, 2), # All sky irradiance
            "temperature_2m": round(temperature, 2),
            "wind_speed_10m": round(wind_speed, 2),
            "relative_humidity_2m": round(humidity, 2),
            "cloud_cover": round(cloud_factor * 100, 2),
            TARGET_COL: round(power_gen, 2)
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated NASA POWER historical dataset with {len(df)} rows at '{output_path}'")
    return df

if __name__ == "__main__":
    generate_nasa_power_dataset()
