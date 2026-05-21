#!/usr/bin/env python3
"""
Predict runner 2's next marathon time
"""

import pickle
import csv
from datetime import datetime, timedelta
from collections import defaultdict

def load_runner2_activities(csv_path):
    """Load runner 2's running activities from CSV"""
    activities = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                activity_type = row.get('Activity Type', '')
                if activity_type != 'Run':
                    continue

                date_str = row.get('Activity Date', '')
                if not date_str:
                    continue

                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                distance_m = float(row.get('Distance', 0) or 0)
                distance_mi = distance_m / 1609.34
                duration_s = float(row.get('Moving Time', 0) or 0)
                duration_min = duration_s / 60.0
                avg_hr = int(float(row.get('Average Heart Rate', 0) or 0))

                if distance_mi > 0.5 and duration_min > 5:
                    activities.append({
                        'date': activity_date,
                        'distance_mi': distance_mi,
                        'duration_min': duration_min,
                        'avg_hr': avg_hr,
                        'pace': duration_min / distance_mi if distance_mi > 0 else 0
                    })

            except Exception as e:
                continue

    return sorted(activities, key=lambda x: x['date'])


def get_recent_training(activities, lookback_weeks=12):
    """Get most recent training block"""
    if not activities:
        return [], None, None

    last_activity = max(activities, key=lambda x: x['date'])
    end_date = last_activity['date'] - timedelta(days=7)
    start_date = end_date - timedelta(weeks=lookback_weeks)

    training_runs = [a for a in activities if start_date <= a['date'] < end_date]

    return training_runs, start_date, end_date


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
    avg_pace = sum(r['pace'] for r in runs) / len(runs) if runs else 0

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
    print(f"Trained on 24 marathons (7 from runner 1 + 17 from runner 2)")
    print(f"Accuracy: ±11.1 minutes\n")

    # Load runner 2's activities
    csv_path = '/Users/osman/Downloads/export_1884062/activities.csv'
    print("Loading runner 2's activities...")
    activities = load_runner2_activities(csv_path)
    print(f"Loaded {len(activities)} running activities")

    # Get most recent activity
    last_activity = max(activities, key=lambda x: x['date'])
    print(f"Most recent activity: {last_activity['date'].date()}\n")

    # Get recent training
    lookback_weeks = 12
    runs, start_date, end_date = get_recent_training(activities, lookback_weeks)

    if not runs:
        print("No recent training data found!")
        return

    print(f"Training window: {start_date.date()} to {end_date.date()}")
    print(f"Found {len(runs)} training runs\n")

    # Extract features
    features = extract_features(runs, lookback_weeks)

    # Show training summary
    print("="*80)
    print("Runner 2's Training Summary:")
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
    print("Runner 2's Marathon Prediction:")
    print("="*80)
    print(f"  Predicted time: {hours}:{minutes:02d}:{seconds:02d}")
    print(f"  Predicted pace: {prediction_min/26.2:.2f} min/mile")
    print(f"  Confidence: ±11 minutes")

    # Show recent race history
    print("\n" + "="*80)
    print("Runner 2's Recent Marathon History:")
    print("="*80)
    recent_marathons = [
        ("Chicago 2025", "2:56:42", "Oct 12, 2025"),
        ("Jack & Jill 2025", "2:55:26", "Jul 27, 2025"),
        ("Boston 2025", "3:19:18", "Apr 21, 2025"),
        ("CIM 2024", "2:55:51 (PR)", "Dec 8, 2024"),
        ("Jack & Jill 2024", "2:57:56", "Jul 27, 2024")
    ]
    for name, time, date in recent_marathons:
        print(f"  {name}: {time} ({date})")

    print("="*80)


if __name__ == '__main__':
    main()
