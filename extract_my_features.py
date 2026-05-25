#!/usr/bin/env python3
"""
Extract features from your Strava export for the Marathon Predictor app.

Usage:
    python extract_my_features.py /path/to/strava_export.zip --race-date 2026-10-15

This creates a features.json file that you can upload to the app.
"""

import argparse
import json
import zipfile
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import statistics
import sys

# Try to import fitparse
try:
    from fitparse import FitFile
    FIT_AVAILABLE = True
except ImportError:
    FIT_AVAILABLE = False
    print("Warning: fitparse not installed. Install with: pip install fitparse")
    print("Continuing with CSV-only features...")


def parse_strava_csv(csv_content):
    """Parse Strava activities.csv"""
    df = pd.read_csv(StringIO(csv_content))

    activities = []
    for _, row in df.iterrows():
        if row.get('Activity Type') != 'Run':
            continue

        try:
            # Handle duplicate columns
            distance_m = float(row.get('Distance.1', 0) or 0)
            if distance_m > 0:
                distance_miles = distance_m / 1609.34
            else:
                distance_val = float(row.get('Distance', 0) or 0)
                distance_miles = distance_val / 1609.34 if distance_val > 100 else distance_val

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
            pace = duration_min / distance_miles if distance_miles > 0 else None

            activities.append({
                'date': activity_date,
                'distance_miles': distance_miles,
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'pace': pace,
                'source': 'csv'
            })
        except:
            continue

    return activities


def parse_fit_file(fit_content):
    """Parse a single FIT file"""
    if not FIT_AVAILABLE:
        return None

    try:
        fitfile = FitFile(BytesIO(fit_content))

        records = []
        for record in fitfile.get_messages('record'):
            record_data = {}
            for field in record:
                record_data[field.name] = field.value
            records.append(record_data)

        if not records:
            return None

        distances = [r.get('distance', 0) for r in records if r.get('distance')]
        heart_rates = [r.get('heart_rate', 0) for r in records if r.get('heart_rate')]
        timestamps = [r.get('timestamp') for r in records if r.get('timestamp')]

        if not distances or not timestamps:
            return None

        total_distance_m = max(distances) if distances else 0
        distance_miles = total_distance_m / 1609.34

        if timestamps and len(timestamps) > 1:
            duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60
        else:
            duration = 0

        avg_hr = sum(heart_rates) / len(heart_rates) if heart_rates else None
        pace = duration / distance_miles if distance_miles > 0.5 else None
        activity_date = timestamps[0] if timestamps else None

        return {
            'date': activity_date,
            'distance_miles': distance_miles,
            'duration_min': duration,
            'avg_hr': avg_hr,
            'pace': pace,
            'source': 'fit'
        }
    except:
        return None


def extract_features(activities, race_date):
    """Extract all features from activities"""
    lookback_start = race_date - timedelta(weeks=16)
    taper_start = race_date - timedelta(days=7)

    training = [a for a in activities if lookback_start <= a['date'] < taper_start]

    if not training:
        return None, lookback_start, taper_start

    # Volume features
    total_distance = sum(a['distance_miles'] for a in training)
    total_runs = len(training)

    dates = [a['date'] for a in training]
    weeks = max(1, (max(dates) - min(dates)).days / 7)

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    # Long runs
    long_runs = [a for a in training if a['distance_miles'] >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    # Weekly breakdown
    weekly_distances = {}
    for a in training:
        week_num = a['date'].isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + a['distance_miles']

    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    # Consistency
    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1:
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if statistics.mean(mileage_values) > 0 else 0
    else:
        mileage_consistency = 1.0

    # Quality workouts
    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    fast_runs = [a for a in training if a['pace'] and a['pace'] < 7.5]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    # HR features
    hr_activities = [a for a in training if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else 0

    # Easy run HR
    easy_runs = [a for a in hr_activities if a['pace'] and a['pace'] > 9.0]
    hr_at_easy = sum(a['avg_hr'] for a in easy_runs) / len(easy_runs) if easy_runs else avg_hr

    # MP run HR
    mp_runs = [a for a in hr_activities if a['pace'] and 7.0 <= a['pace'] <= 8.5]
    hr_at_mp = sum(a['avg_hr'] for a in mp_runs) / len(mp_runs) if mp_runs else avg_hr

    # Count sources
    csv_count = sum(1 for a in training if a.get('source') == 'csv')
    fit_count = sum(1 for a in training if a.get('source') == 'fit')

    features = {
        'total_weekly_mileage': round(weekly_mileage, 2),
        'peak_weekly_mileage': round(peak_weekly_mileage, 2),
        'runs_per_week': round(runs_per_week, 2),
        'total_runs': total_runs,
        'long_run_distance': round(long_run_distance, 2),
        'long_run_count': len(long_runs),
        'long_run_percent_weekly': round(long_run_distance / weekly_mileage * 100, 1) if weekly_mileage > 0 else 0,
        'mileage_consistency': round(max(0, min(1, mileage_consistency)), 3),
        'training_consistency_score': round(max(0, min(1, mileage_consistency)), 3),
        'tempo_workout_count': len(tempo_runs),
        'fast_workout_count': len(fast_runs),
        'quality_workout_percent': round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0,
        'avg_hr': round(avg_hr, 1) if avg_hr else 0,
        'hr_at_easy_pace': round(hr_at_easy, 1) if hr_at_easy else 0,
        'hr_at_marathon_pace': round(hr_at_mp, 1) if hr_at_mp else 0,
        'csv_activities': csv_count,
        'fit_activities': fit_count,
    }

    return features, lookback_start, taper_start


def main():
    parser = argparse.ArgumentParser(description='Extract features from Strava export')
    parser.add_argument('zip_file', help='Path to Strava export zip file')
    parser.add_argument('--race-date', required=True, help='Target race date (YYYY-MM-DD)')
    parser.add_argument('--output', default='features.json', help='Output file (default: features.json)')
    args = parser.parse_args()

    try:
        race_date = datetime.strptime(args.race_date, "%Y-%m-%d")
    except:
        print("Error: Invalid race date format. Use YYYY-MM-DD")
        sys.exit(1)

    print(f"Strava Export: {args.zip_file}")
    print(f"Race Date: {args.race_date}")
    print(f"Training Window: {(race_date - timedelta(weeks=16)).date()} to {(race_date - timedelta(days=7)).date()}")
    print()

    all_activities = []
    fit_parsed = 0
    fit_failed = 0

    with zipfile.ZipFile(args.zip_file, 'r') as z:
        # Parse CSV
        print("Parsing activities.csv...")
        csv_file = None
        for name in z.namelist():
            if name.endswith('activities.csv'):
                csv_file = name
                break

        if not csv_file:
            print("Error: No activities.csv found in zip")
            sys.exit(1)

        with z.open(csv_file) as f:
            csv_content = f.read().decode('utf-8')
            csv_activities = parse_strava_csv(csv_content)
            all_activities.extend(csv_activities)
            print(f"  Found {len(csv_activities)} runs in CSV")

        # Parse FIT files
        if FIT_AVAILABLE:
            fit_files = [f for f in z.namelist()
                        if (f.endswith('.fit') or f.endswith('.fit.gz'))
                        and not f.startswith('__MACOSX')]

            print(f"\nParsing {len(fit_files)} FIT files...")

            lookback_start = race_date - timedelta(weeks=16)
            taper_start = race_date - timedelta(days=7)

            for i, fit_path in enumerate(fit_files):
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(fit_files)} files...")

                try:
                    with z.open(fit_path) as f:
                        content = f.read()

                        if fit_path.endswith('.gz'):
                            content = gzip.decompress(content)

                        activity = parse_fit_file(content)

                        if activity and activity['date']:
                            if lookback_start <= activity['date'] < taper_start:
                                all_activities.append(activity)
                                fit_parsed += 1
                except:
                    fit_failed += 1

            print(f"  Parsed {fit_parsed} FIT files in training window")
            if fit_failed > 0:
                print(f"  Failed to parse {fit_failed} files (corrupted)")

    # Extract features
    print("\nExtracting features...")
    features, window_start, window_end = extract_features(all_activities, race_date)

    if not features:
        print("Error: No training activities found in the window")
        sys.exit(1)

    # Create output
    output = {
        'race_date': args.race_date,
        'training_window': {
            'start': window_start.strftime('%Y-%m-%d'),
            'end': window_end.strftime('%Y-%m-%d'),
        },
        'features': features,
        'extraction_info': {
            'csv_activities': features['csv_activities'],
            'fit_activities': features['fit_activities'],
            'fit_failed': fit_failed,
            'extracted_at': datetime.now().isoformat(),
        }
    }

    # Save
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {args.output}")
    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Weekly Mileage:     {features['total_weekly_mileage']:.1f} miles")
    print(f"Peak Week:          {features['peak_weekly_mileage']:.1f} miles")
    print(f"Runs per Week:      {features['runs_per_week']:.1f}")
    print(f"Total Runs:         {features['total_runs']}")
    print(f"Long Runs (15+ mi): {features['long_run_count']}")
    print(f"Longest Run:        {features['long_run_distance']:.1f} miles")
    print(f"Tempo Workouts:     {features['tempo_workout_count']}")
    print(f"Quality Run %:      {features['quality_workout_percent']:.1f}%")
    print(f"Avg Heart Rate:     {features['avg_hr']:.0f} bpm")
    print()
    print(f"Upload {args.output} to the Marathon Predictor app!")


if __name__ == '__main__':
    main()
