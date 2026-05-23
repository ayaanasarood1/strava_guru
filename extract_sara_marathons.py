#!/usr/bin/env python3
"""
Extract marathon features for Sara
"""

import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

def get_activities_in_window(db_path, end_date, lookback_weeks=12):
    """Get activities in training window before race"""
    start_date = end_date - timedelta(weeks=lookback_weeks)
    taper_start = end_date - timedelta(days=7)  # Exclude taper week

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM activities
        WHERE activity_date >= ? AND activity_date < ?
        ORDER BY activity_date
    ''', (start_date.isoformat(), taper_start.isoformat()))

    activities = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return activities


def extract_features(activities, race_date, race_distance, race_time_min, runner_context):
    """Extract training features from activities"""
    features = {}

    if not activities:
        return features

    # Training volume
    total_distance = sum(a['distance_miles'] or 0 for a in activities)
    total_runs = len(activities)

    # Calculate weeks
    dates = [datetime.fromisoformat(a['activity_date']) for a in activities]
    if dates:
        weeks = max(1, (max(dates) - min(dates)).days / 7)
    else:
        weeks = 1

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    # Long runs (>= 15 miles)
    long_runs = [a for a in activities if (a['distance_miles'] or 0) >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)
    long_run_count = len(long_runs)

    # Peak week
    weekly_distances = {}
    for a in activities:
        week_num = datetime.fromisoformat(a['activity_date']).isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + (a['distance_miles'] or 0)
    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    # HR data
    hr_activities = [a for a in activities if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else None

    # Training consistency
    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1:
        import statistics
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if statistics.mean(mileage_values) > 0 else 0
    else:
        mileage_consistency = 1.0

    # Populate features
    features['race_distance_miles'] = race_distance
    features['total_weekly_mileage'] = round(weekly_mileage, 2)
    features['peak_weekly_mileage'] = round(peak_weekly_mileage, 2)
    features['long_run_distance'] = round(long_run_distance, 2)
    features['long_run_count'] = long_run_count
    features['total_runs'] = total_runs
    features['runs_per_week'] = round(runs_per_week, 2)
    features['mileage_consistency'] = round(mileage_consistency, 3)

    # Runner profile
    features['age_normalized'] = runner_context.get('age_normalized', 0.9)
    features['sex_encoded'] = runner_context.get('sex_encoded', 0)  # 0 = Female
    features['max_hr_normalized'] = runner_context.get('max_hr_normalized', 0.9)
    features['experience_years'] = runner_context.get('experience_years', 5)
    features['training_consistency_score'] = round(mileage_consistency, 3)

    # Intensity (simplified without track points)
    features['zone1_percent'] = 0
    features['zone2_percent'] = 0
    features['zone3_percent'] = 0
    features['zone4_percent'] = 0
    features['zone5_percent'] = 0
    features['tempo_workout_count'] = 0
    features['interval_workout_count'] = 0
    features['quality_workout_percent'] = 0

    # Efficiency
    features['hr_at_easy_pace'] = avg_hr
    features['hr_at_marathon_pace'] = avg_hr
    features['cardiac_drift'] = 0
    features['aerobic_decoupling'] = 0
    features['hr_variability_coefficient'] = 0

    # Terrain
    features['hr_per_grade_uphill'] = 0
    features['hr_per_grade_downhill'] = 0
    features['hill_recovery_rate'] = 0
    features['elevation_tolerance'] = 1.0

    # Race context
    features['taper_quality_score'] = 0.5
    features['days_since_last_hard_effort'] = 7

    # LT features (placeholder)
    features['lt_heart_rate'] = None
    features['lt_pace'] = None
    features['lt_percent_max_hr'] = None
    features['aet_heart_rate'] = None
    features['aet_pace'] = None

    # Long run percent
    features['long_run_percent_weekly'] = round(long_run_distance / weekly_mileage * 100, 1) if weekly_mileage > 0 else 0

    # Reserved
    features['reserved_1'] = None
    features['reserved_2'] = None
    features['reserved_3'] = None

    return features


def main():
    print("="*80)
    print("Extracting Marathon Features for Sara")
    print("="*80)

    csv_path = Path.home() / "Downloads" / "export_108527851_sara" / "activities.csv"
    db_path = Path.home() / ".strava_guru_cache" / "sara" / "activities.db"

    # Sara's runner context (female runner)
    runner_context = {
        'age_normalized': 0.9,  # Assuming ~30-35 years old
        'sex_encoded': 0,  # Female
        'max_hr_normalized': 0.9,
        'experience_years': 5
    }

    # Find marathons from CSV
    marathons = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Activity Type') != 'Run':
                continue

            try:
                distance_m = float(row.get('Distance', 0) or 0)
                distance_miles = distance_m / 1609.34
            except:
                continue

            if 25.0 <= distance_miles <= 27.5:
                # Parse date
                date_str = row.get('Activity Date', '')
                try:
                    race_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                except:
                    try:
                        race_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                    except:
                        continue

                # Parse time
                moving_sec = float(row.get('Moving Time', 0) or 0)
                elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
                time_sec = moving_sec if moving_sec > 0 else elapsed_sec
                time_min = time_sec / 60

                # Weather
                apparent_temp = None
                try:
                    temp_c = float(row.get('Apparent Temperature', '') or 0)
                    if temp_c:
                        apparent_temp = temp_c * 9/5 + 32  # Convert to F
                except:
                    pass

                humidity = None
                try:
                    humidity = float(row.get('Humidity', '') or 0) or None
                except:
                    pass

                marathons.append({
                    'date': race_date,
                    'name': row.get('Activity Name', ''),
                    'distance_miles': distance_miles,
                    'time_minutes': time_min,
                    'apparent_temp': apparent_temp,
                    'humidity': humidity
                })

    # Sort by date
    marathons.sort(key=lambda x: x['date'])

    print(f"\nFound {len(marathons)} marathons")

    # Extract features for each marathon
    race_results = []

    for i, m in enumerate(marathons, 1):
        race_date = m['date']
        race_name = m['name']
        time_min = m['time_minutes']

        hours = int(time_min // 60)
        mins = int(time_min % 60)

        print(f"\n{i}. {race_date.strftime('%Y-%m-%d')} - {race_name}")
        print(f"   Time: {hours}:{mins:02d}")

        # Get training activities
        activities = get_activities_in_window(db_path, race_date, lookback_weeks=12)
        print(f"   Training activities: {len(activities)}")

        # Extract features
        features = extract_features(activities, race_date, 26.2, time_min, runner_context)

        # Add weather features
        features['race_temperature'] = m['apparent_temp']
        features['race_humidity'] = m['humidity']
        features['race_wind_speed'] = None
        features['race_apparent_temp'] = m['apparent_temp']

        print(f"   Weekly mileage: {features.get('total_weekly_mileage', 0):.1f} mi")
        print(f"   Long run: {features.get('long_run_distance', 0):.1f} mi")

        # Create race result entry
        race_id = f"sara_marathon_{race_date.strftime('%Y%m%d')}"

        race_results.append({
            'race_id': race_id,
            'runner_id': 'runner_sara',
            'race_date': race_date.isoformat(),
            'race_name': race_name,
            'race_distance_miles': 26.2,
            'actual_time_minutes': time_min,
            'features': features
        })

    # Save results
    output_path = Path('/Users/osman/PycharmProjects/strava_guru/race_data/sara_marathons.json')
    with open(output_path, 'w') as f:
        json.dump(race_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Saved {len(race_results)} marathons to {output_path}")
    print(f"{'='*80}")

    # Summary
    print("\nSara's Marathon Summary:")
    print(f"{'Date':<12} {'Name':<25} {'Time':<6} {'Weekly Mi':<10} {'Temp':<8}")
    print("-" * 70)
    for r in race_results:
        time_min = r['actual_time_minutes']
        hours = int(time_min // 60)
        mins = int(time_min % 60)
        temp = r['features'].get('race_apparent_temp')
        temp_str = f"{temp:.0f}°F" if temp else "N/A"

        print(f"{r['race_date'][:10]:<12} {r['race_name'][:25]:<25} {hours}:{mins:02d}  {r['features']['total_weekly_mileage']:<10.1f} {temp_str:<8}")


if __name__ == '__main__':
    main()
