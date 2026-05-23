#!/usr/bin/env python3
"""Build activity cache for Qazi from CSV export"""

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

def main():
    print("="*80)
    print("Building Qazi's Activity Cache from CSV")
    print("="*80)

    csv_path = Path.home() / "Downloads" / "export_40747977_qazi" / "activities.csv"
    cache_dir = Path.home() / ".strava_guru_cache" / "qazi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "activities.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE,
            activity_date TEXT,
            activity_type TEXT,
            distance_miles REAL,
            duration_minutes REAL,
            avg_hr REAL,
            max_hr REAL,
            elevation_gain REAL,
            avg_pace_min_per_mile REAL,
            calories REAL,
            name TEXT,
            weather_temp REAL,
            apparent_temp REAL,
            humidity REAL,
            created_at TEXT
        )
    ''')
    conn.commit()

    runs_added = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Activity Type') != 'Run':
                continue
            try:
                distance_m = float(row.get('Distance', 0) or 0)
                distance_miles = distance_m / 1609.34
                if distance_miles < 0.5:
                    continue

                moving_sec = float(row.get('Moving Time', 0) or 0)
                elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
                duration_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

                date_str = row.get('Activity Date', '')
                try:
                    activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                except:
                    try:
                        activity_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                    except:
                        continue

                avg_hr = float(row.get('Average Heart Rate', 0) or 0) or None
                max_hr = float(row.get('Max Heart Rate', 0) or 0) or None
                elevation_gain = float(row.get('Elevation Gain', 0) or 0)
                avg_pace = duration_min / distance_miles if distance_miles > 0 else None

                weather_temp = apparent_temp = humidity = None
                try:
                    weather_temp = float(row.get('Weather Temperature', '') or 0) or None
                    apparent_temp = float(row.get('Apparent Temperature', '') or 0) or None
                    humidity = float(row.get('Humidity', '') or 0) or None
                except:
                    pass

                file_path = row.get('Filename', '') or f"activity_{activity_date.isoformat()}"

                cursor.execute('''
                    INSERT OR REPLACE INTO activities
                    (file_path, activity_date, activity_type, distance_miles, duration_minutes,
                     avg_hr, max_hr, elevation_gain, avg_pace_min_per_mile, calories, name,
                     weather_temp, apparent_temp, humidity, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (file_path, activity_date.isoformat(), 'Run', distance_miles, duration_min,
                      avg_hr, max_hr, elevation_gain, avg_pace, float(row.get('Calories', 0) or 0),
                      row.get('Activity Name', ''), weather_temp, apparent_temp, humidity,
                      datetime.now().isoformat()))
                runs_added += 1
            except:
                continue

    conn.commit()
    cursor.execute("SELECT COUNT(*), MIN(activity_date), MAX(activity_date) FROM activities")
    total, min_date, max_date = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM activities WHERE avg_hr IS NOT NULL")
    with_hr = cursor.fetchone()[0]
    conn.close()

    print(f"\nCache built: {total} runs ({with_hr} with HR)")
    print(f"Date range: {min_date[:10]} to {max_date[:10]}")
    print(f"Database: {db_path}")

if __name__ == '__main__':
    main()
