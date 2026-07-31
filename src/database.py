import sqlite3
import pandas as pd
import os
import logging
from typing import Optional, List, Dict, Any
from config import DB_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite database and create real_time_weather table if not exists."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS real_time_weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            cloud_cover REAL NOT NULL,
            irradiance REAL NOT NULL,
            wind_speed REAL NOT NULL,
            UNIQUE(location, timestamp) ON CONFLICT REPLACE
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")

def insert_weather_records(records: List[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """
    Insert a list of weather telemetry records into the real_time_weather table.
    Overwrites records if location + timestamp already exists.
    """
    if not records:
        return 0
    
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    inserted_count = 0
    for record in records:
        cursor.execute("""
            INSERT OR REPLACE INTO real_time_weather 
            (location, timestamp, temperature, humidity, cloud_cover, irradiance, wind_speed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record["location"],
            record["timestamp"],
            record["temperature"],
            record["humidity"],
            record["cloud_cover"],
            record["irradiance"],
            record["wind_speed"]
        ))
        inserted_count += 1
        
    conn.commit()
    conn.close()
    logger.info(f"Inserted/Updated {inserted_count} weather records in SQLite for location '{records[0].get('location')}'")
    return inserted_count

def fetch_weather_records(location: str, limit: int = 168, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Fetch weather records from SQLite for a given location, ordered by timestamp ascending.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT location, timestamp, temperature, humidity, cloud_cover, irradiance, wind_speed
        FROM real_time_weather
        WHERE location = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(location, limit))
    conn.close()
    return df

def get_fleet_stats(db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch total count of telemetry records per city stored in SQLite."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    query = """
        SELECT location, COUNT(*) as record_count, 
               MIN(timestamp) as start_time, MAX(timestamp) as end_time
        FROM real_time_weather
        GROUP BY location
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
