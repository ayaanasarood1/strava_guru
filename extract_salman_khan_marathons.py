#!/usr/bin/env python3
"""Extract marathon features for Salman Khan"""

import csv
import json
import sqlite3
import statistics
from datetime import datetime, timedelta
from pathlib import Path

def get_activities_in_window(db_path, end_date, lookback_weeks=16):
    """Get training activities in the window before race"""
    start_date = end_date - timedelta(weeks=lookback_weeks)
    taper_start = end_date - timedelta(days=7)

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

def extract_features(activities, race_date, race_distance, runner_context):
    """Extract all features from training activities"""
    features = {}
    if not activities:
        return features

    total_distance = sum(a['distance_miles'] or 0 for a in activities)
    total_runs = len(activities)

    dates = [datetime.fromisoformat(a['activity_date']) for a in activities]
    weeks = max(1, (max(dates) - min(dates)).days / 7) if dates else 1
    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    # Long runs (15+ miles)
    long_runs = [a for a in activities if (a['distance_miles'] or 0) >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    # Weekly breakdown
    weekly_distances = {}
    for a in activities:
        week_num = datetime.fromisoformat(a['activity_date']).isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + (a['distance_miles'] or 0)
    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    # HR features
    hr_activities = [a for a in activities if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else None

    # Mileage consistency
    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1 and statistics.mean(mileage_values) > 0:
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values))
    else:
        mileage_consistency = 1.0
    mileage_consistency = max(0, min(1, mileage_consistency))

    # Quality workouts (tempo < 8:00, fast < 7:30)
    tempo_runs = [a for a in activities if a['avg_pace_min_per_mile'] and 7.0 <= a['avg_pace_min_per_mile'] < 8.0]
    fast_runs = [a for a in activities if a['avg_pace_min_per_mile'] and a['avg_pace_min_per_mile'] < 7.5]
    quality_runs = [a for a in activities if a['avg_pace_min_per_mile'] and a['avg_pace_min_per_mile'] < 8.0]

    # Populate features
    features['race_distance_miles'] = race_distance
    features['total_weekly_mileage'] = round(weekly_mileage, 2)
    features['peak_weekly_mileage'] = round(peak_weekly_mileage, 2)
    features['long_run_distance'] = round(long_run_distance, 2)
    features['long_run_count'] = len(long_runs)
    features['total_runs'] = total_runs
    features['runs_per_week'] = round(runs_per_week, 2)
    features['mileage_consistency'] = round(mileage_consistency, 3)
    features['long_run_percent_weekly'] = round(long_run_distance / weekly_mileage * 100, 1) if weekly_mileage > 0 else 0

    # Runner context
    features['age_normalized'] = runner_context.get('age_normalized', 0.85)
    features['sex_encoded'] = runner_context.get('sex_encoded', 1)  # Male
    features['max_hr_normalized'] = runner_context.get('max_hr_normalized', 0.9)
    features['experience_years'] = runner_context.get('experience_years', 8)
    features['training_consistency_score'] = round(mileage_consistency, 3)

    # Quality workouts
    features['tempo_workout_count'] = len(tempo_runs)
    features['fast_workout_count'] = len(fast_runs)
    features['quality_workout_percent'] = round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0
    features['interval_workout_count'] = 0  # Not available from CSV

    # HR features
    features['avg_hr'] = avg_hr
    features['hr_at_easy_pace'] = avg_hr
    features['hr_at_marathon_pace'] = avg_hr

    # Zone features (not available from CSV)
    for f in ['zone1_percent', 'zone2_percent', 'zone3_percent', 'zone4_percent', 'zone5_percent',
              'cardiac_drift', 'aerobic_decoupling', 'hr_variability_coefficient',
              'hr_per_grade_uphill', 'hr_per_grade_downhill', 'hill_recovery_rate']:
        features[f] = 0

    features['elevation_tolerance'] = 1.0
    features['taper_quality_score'] = 0.5
    features['days_since_last_hard_effort'] = 7

    # Reserved features
    for f in ['lt_heart_rate', 'lt_pace', 'lt_percent_max_hr', 'aet_heart_rate', 'aet_pace',
              'reserved_1', 'reserved_2', 'reserved_3', 'recent_injury_flag']:
        features[f] = None

    return features

def main():
    print("="*80)
    print("Extracting Marathon Features for Salman Khan")
    print("="*80)

    csv_path = Path.home() / "Downloads" / "salman_khan" / "activities.csv"
    db_path = Path.home() / ".strava_guru_cache" / "salman_khan" / "activities.db"

    # Runner context (experienced marathoner)
    runner_context = {
        'age_normalized': 0.85,  # Experienced runner
        'sex_encoded': 1,  # Male
        'max_hr_normalized': 0.9,
        'experience_years': 8
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
                date_str = row.get('Activity Date', '')
                try:
                    race_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                except:
                    try:
                        race_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                    except:
                        continue

                moving_sec = float(row.get('Moving Time', 0) or 0)
                elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
                time_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

                # Filter to actual races (2:30 - 5:00)
                if not (150 <= time_min <= 300):
                    continue

                apparent_temp = humidity = None
                try:
                    temp_c = float(row.get('Apparent Temperature', '') or 0)
                    if temp_c:
                        apparent_temp = temp_c * 9/5 + 32  # Convert to F
                    humidity = float(row.get('Humidity', '') or 0) or None
                    if humidity and humidity > 1:
                        humidity = humidity / 100  # Normalize
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

    marathons.sort(key=lambda x: x['date'])
    print(f"\nFound {len(marathons)} marathons")

    # Extract features for each marathon
    race_results = []
    historical_pr = None  # Track PR for historical_pr_minutes feature

    for i, m in enumerate(marathons, 1):
        race_date = m['date']
        time_min = m['time_minutes']
        hours, mins = int(time_min // 60), int(time_min % 60)

        print(f"\n{i}. {race_date.strftime('%Y-%m-%d')} - {m['name'][:50]}")
        print(f"   Time: {hours}:{mins:02d}")

        activities = get_activities_in_window(db_path, race_date)
        print(f"   Training activities: {len(activities)}")

        features = extract_features(activities, race_date, 26.2, runner_context)

        # Add weather
        features['race_temperature'] = m['apparent_temp']
        features['race_humidity'] = m['humidity']
        features['race_wind_speed'] = None
        features['race_apparent_temp'] = m['apparent_temp']

        # Add historical PR
        features['historical_pr_minutes'] = historical_pr

        # Update PR if this is a new best
        if historical_pr is None or time_min < historical_pr:
            historical_pr = time_min

        print(f"   Weekly mileage: {features.get('total_weekly_mileage', 0):.1f} mi")
        print(f"   Tempo workouts: {features.get('tempo_workout_count', 0)}")

        race_results.append({
            'race_id': f"salman_khan_marathon_{race_date.strftime('%Y%m%d')}",
            'runner_id': 'runner_salman_khan',
            'race_date': race_date.isoformat(),
            'race_name': m['name'],
            'race_distance_miles': 26.2,
            'actual_time_minutes': time_min,
            'features': features
        })

    # Save results
    output_path = Path('/Users/osman/PycharmProjects/strava_guru/race_data/salman_khan_marathons.json')
    with open(output_path, 'w') as f:
        json.dump(race_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Saved {len(race_results)} marathons to {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    times = [r['actual_time_minutes'] for r in race_results]
    pr = min(times)
    print(f"Marathons: {len(race_results)}")
    print(f"PR: {int(pr//60)}:{int(pr%60):02d}")
    print(f"Date range: {race_results[0]['race_date'][:10]} to {race_results[-1]['race_date'][:10]}")

if __name__ == '__main__':
    main()
