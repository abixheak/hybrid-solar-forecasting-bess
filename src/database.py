import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
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

def prune_old_weather_records(location: str, cutoff_days: int = 30, db_path: str = DB_PATH) -> int:
    """
    Delete rows older than cutoff_days for a given location to prevent unbounded table growth.
    Returns the number of rows deleted.
    """
    cutoff_ts = (datetime.utcnow() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM real_time_weather WHERE location = ? AND timestamp < ?",
        (location, cutoff_ts)
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(f"Pruned {deleted} stale weather records older than {cutoff_days}d for '{location}'.")
    return deleted


def insert_weather_records(records: List[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """
    Insert a list of weather telemetry records into the real_time_weather table.
    Overwrites records if location + timestamp already exists.
    Automatically prunes rows older than 30 days to prevent unbounded table growth.
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

    # Prune stale rows so the table doesn't grow unbounded
    prune_old_weather_records(records[0].get("location", ""), cutoff_days=30, db_path=db_path)
    return inserted_count

def fetch_weather_records(location: str, limit: int = 168, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Fetch the most recent `limit` weather records from SQLite for a given location,
    returned in ascending timestamp order (newest window, oldest-first).

    Uses DESC LIMIT to grab the latest rows, then reverses to restore chronological order.
    This prevents returning stale historical rows accumulated from past syncs.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)

    query = """
        SELECT location, timestamp, temperature, humidity, cloud_cover, irradiance, wind_speed
        FROM real_time_weather
        WHERE location = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(location, limit))
    conn.close()
    # Restore chronological (ascending) order after DESC fetch
    df = df.iloc[::-1].reset_index(drop=True)
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
