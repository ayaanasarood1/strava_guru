#!/usr/bin/env python3
"""
Build activity cache for Sara from CSV export
"""

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

def main():
    print("="*80)
    print("Building Sara's Activity Cache from CSV")
    print("="*80)

    csv_path = Path.home() / "Downloads" / "export_108527851_sara" / "activities.csv"
    cache_dir = Path.home() / ".strava_guru_cache" / "sara"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "activities.db"

    print(f"\nSource: {csv_path}")
    print(f"Cache: {db_path}")

    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS track_point_summary (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER,
            bucket_start_time REAL,
            avg_hr REAL,
            avg_speed REAL,
            avg_cadence REAL,
            distance REAL,
            elevation REAL,
            FOREIGN KEY (activity_id) REFERENCES activities(id)
        )
    ''')

    conn.commit()

    # Parse CSV
    runs_added = 0
    runs_with_hr = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            activity_type = row.get('Activity Type', '')
            if activity_type != 'Run':
                continue

            try:
                # Parse distance (meters to miles)
                distance_m = float(row.get('Distance', 0) or 0)
                distance_miles = distance_m / 1609.34

                if distance_miles < 0.5:  # Skip very short runs
                    continue

                # Parse time
                moving_sec = float(row.get('Moving Time', 0) or 0)
                elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
                duration_sec = moving_sec if moving_sec > 0 else elapsed_sec
                duration_min = duration_sec / 60

                # Parse date
                date_str = row.get('Activity Date', '')
                try:
                    # Format: "Apr 26, 2026, 9:00:00 AM"
                    activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                except:
                    try:
                        activity_date = datetime.strptime(date_str[:12].strip(), "%b %d, %Y")
                    except:
                        continue

                # Parse HR
                avg_hr = float(row.get('Average Heart Rate', 0) or 0) or None
                max_hr = float(row.get('Max Heart Rate', 0) or 0) or None

                # Parse other fields
                elevation_gain = float(row.get('Elevation Gain', 0) or 0)
                calories = float(row.get('Calories', 0) or 0)
                name = row.get('Activity Name', '')

                # Calculate pace
                avg_pace = duration_min / distance_miles if distance_miles > 0 else None

                # Weather
                weather_temp = None
                apparent_temp = None
                humidity = None

                try:
                    weather_temp = float(row.get('Weather Temperature', '') or 0) or None
                    apparent_temp = float(row.get('Apparent Temperature', '') or 0) or None
                    humidity = float(row.get('Humidity', '') or 0) or None
                except:
                    pass

                # Use filename as unique identifier
                file_path = row.get('Filename', '') or f"activity_{activity_date.isoformat()}"

                # Insert activity
                cursor.execute('''
                    INSERT OR REPLACE INTO activities
                    (file_path, activity_date, activity_type, distance_miles, duration_minutes,
                     avg_hr, max_hr, elevation_gain, avg_pace_min_per_mile, calories, name,
                     weather_temp, apparent_temp, humidity, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_path,
                    activity_date.isoformat(),
                    activity_type,
                    distance_miles,
                    duration_min,
                    avg_hr,
                    max_hr,
                    elevation_gain,
                    avg_pace,
                    calories,
                    name,
                    weather_temp,
                    apparent_temp,
                    humidity,
                    datetime.now().isoformat()
                ))

                runs_added += 1
                if avg_hr:
                    runs_with_hr += 1

            except Exception as e:
                continue

    conn.commit()

    # Get stats
    cursor.execute("SELECT COUNT(*) FROM activities")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(activity_date), MAX(activity_date) FROM activities")
    date_range = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM activities WHERE avg_hr IS NOT NULL")
    with_hr = cursor.fetchone()[0]

    conn.close()

    print(f"\n{'='*80}")
    print("Cache Build Complete!")
    print(f"{'='*80}")
    print(f"  Total runs: {total}")
    print(f"  With HR data: {with_hr}")
    print(f"  Date range: {date_range[0][:10]} to {date_range[1][:10]}")
    print(f"  Database: {db_path}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
