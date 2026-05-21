#!/usr/bin/env python3
"""
Predict race time using combined model
"""

import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

def get_recent_training(db_path, lookback_weeks=12):
    """Get most recent training block"""
    # Find the most recent activity date
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(activity_date) FROM activities WHERE activity_type = 'running'")
    max_date_str = cursor.fetchone()[0]
    conn.close()

    if not max_date_str:
        return [], None, None

    last_activity = datetime.fromisoformat(max_date_str)

    # Use most recent 12 weeks ending 7 days before last activity (taper week)
    end_date = last_activity - timedelta(days=7)
    start_date = end_date - timedelta(weeks=lookback_weeks)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT
            activity_date,
            distance_meters,
            duration_seconds,
            avg_heart_rate,
            avg_pace
        FROM activities
        WHERE activity_type = 'running'
          AND activity_date >= ?
          AND activity_date < ?
          AND distance_meters > 800
        ORDER BY activity_date
    """

    cursor.execute(query, (start_date.isoformat(), end_date.isoformat()))
    rows = cursor.fetchall()
    conn.close()

    runs = []
    for row in rows:
        runs.append({
            'date': datetime.fromisoformat(row[0]),
            'distance_mi': row[1] / 1609.34,
            'duration_min': row[2] / 60.0,
            'avg_hr': row[3] or 0,
            'avg_pace': row[4] or 0
        })

    return runs, start_date, end_date


def extract_features(runs, lookback_weeks):
    """Extract features from training runs"""
    if not runs:
        return None

    total_miles = sum(r['distance_mi'] for r in runs)
    total_weeks = lookback_weeks

    # Weekly stats
    start_date = min(r['date'] for r in runs)
    weekly_miles = defaultdict(float)
    for run in runs:
        week = (run['date'] - start_date).days // 7
        weekly_miles[week] += run['distance_mi']

    avg_weekly_miles = total_miles / total_weeks if total_weeks > 0 else 0
    peak_weekly_miles = max(weekly_miles.values()) if weekly_miles else 0

    # Long runs
    long_runs = [r for r in runs if r['distance_mi'] >= 10]
    max_long_run = max([r['distance_mi'] for r in long_runs]) if long_runs else 0

    # HR stats
    runs_with_hr = [r for r in runs if r['avg_hr'] > 0]
    avg_hr = sum(r['avg_hr'] for r in runs_with_hr) / len(runs_with_hr) if runs_with_hr else 0

    # Pace stats
    runs_with_pace = [r for r in runs if r['avg_pace'] > 0]
    avg_pace = sum(r['avg_pace'] for r in runs_with_pace) / len(runs_with_pace) if runs_with_pace else 0

    return [
        avg_weekly_miles,
        peak_weekly_miles,
        max_long_run,
        len(runs),
        len(runs) / total_weeks,
        avg_hr,
        avg_pace,
        len(long_runs)
    ]


def main():
    # Load model
    with open('race_time_model_combined.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_names = model_data['feature_names']
    model_name = model_data['model_name']

    print(f"Loaded {model_name} model")
    print(f"Trained on 24 marathons (7 yours + 17 from runner 2)")
    print(f"Accuracy: ±11.1 minutes\n")

    # Get recent training
    cache_path = Path.home() / ".strava_guru_cache" / "activities.db"
    lookback_weeks = 12

    runs, start_date, end_date = get_recent_training(cache_path, lookback_weeks)

    print(f"Training window: {start_date.date()} to {end_date.date()}")
    print(f"Found {len(runs)} training runs\n")

    if not runs:
        print("No training data found!")
        return

    # Extract features
    features = extract_features(runs, lookback_weeks)

    # Show training summary
    print("="*80)
    print("Training Summary:")
    print("="*80)
    for name, value in zip(feature_names, features):
        if 'mileage' in name or 'dist' in name:
            print(f"  {name}: {value:.1f} miles" + ("/week" if "weekly" in name else ""))
        elif 'runs' in name:
            print(f"  {name}: {value:.1f}")
        elif 'hr' in name:
            print(f"  {name}: {value:.0f} bpm")
        elif 'pace' in name:
            print(f"  {name}: {value:.2f} min/mile")
        else:
            print(f"  {name}: {value:.2f}")

    # Predict
    prediction_min = model.predict([features])[0]
    hours = int(prediction_min // 60)
    minutes = int(prediction_min % 60)
    seconds = int((prediction_min % 1) * 60)

    print("\n" + "="*80)
    print("Marathon Prediction:")
    print("="*80)
    print(f"  Predicted time: {hours}:{minutes:02d}:{seconds:02d}")
    print(f"  Predicted pace: {prediction_min/26.2:.2f} min/mile")
    print(f"  Confidence: ±11 minutes")
    print("="*80)


if __name__ == '__main__':
    main()
