#!/usr/bin/env python3
"""
Show Salman's training stats across all three feature sources: Cache, CSV, Hybrid
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import statistics

SALMAN_CSV = Path('/Users/osman/Downloads/export_1884062_salman/activities.csv')


def format_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"


def parse_strava_csv(csv_path):
    df = pd.read_csv(csv_path)
    activities = []
    for _, row in df.iterrows():
        if row.get('Activity Type') != 'Run':
            continue
        try:
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
            })
        except:
            continue
    return activities


def extract_csv_features(activities, race_date):
    if not activities:
        return {}, None, None

    race_dt = datetime.strptime(race_date[:10], "%Y-%m-%d") if isinstance(race_date, str) else race_date
    lookback_start = race_dt - timedelta(weeks=16)
    taper_start = race_dt - timedelta(days=7)

    training = [a for a in activities if lookback_start <= a['date'] < taper_start]
    if not training:
        return {}, lookback_start, taper_start

    total_distance = sum(a['distance_miles'] for a in training)
    total_runs = len(training)
    dates = [a['date'] for a in training]
    weeks = max(1, (max(dates) - min(dates)).days / 7)

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    long_runs = [a for a in training if a['distance_miles'] >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    weekly_distances = {}
    for a in training:
        week_num = a['date'].isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + a['distance_miles']

    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    return {
        'total_weekly_mileage': round(weekly_mileage, 1),
        'peak_weekly_mileage': round(peak_weekly_mileage, 1),
        'runs_per_week': round(runs_per_week, 1),
        'total_runs': total_runs,
        'long_run_distance': round(long_run_distance, 1),
        'long_run_count': len(long_runs),
        'tempo_workout_count': len(tempo_runs),
        'quality_workout_percent': round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0,
    }, lookback_start, taper_start


def create_hybrid_features(cache_f, csv_f):
    return {
        'total_weekly_mileage': csv_f.get('total_weekly_mileage') or cache_f.get('total_weekly_mileage', 0),
        'peak_weekly_mileage': max(csv_f.get('peak_weekly_mileage', 0) or 0, cache_f.get('peak_weekly_mileage', 0) or 0),
        'runs_per_week': csv_f.get('runs_per_week') or cache_f.get('runs_per_week', 0),
        'total_runs': csv_f.get('total_runs') or cache_f.get('total_runs', 0),
        'long_run_distance': max(csv_f.get('long_run_distance', 0) or 0, cache_f.get('long_run_distance', 0) or 0),
        'long_run_count': max(csv_f.get('long_run_count', 0) or 0, cache_f.get('long_run_count', 0) or 0),
        'tempo_workout_count': max(csv_f.get('tempo_workout_count', 0) or 0, cache_f.get('tempo_workout_count', 0) or 0),
        'quality_workout_percent': max(csv_f.get('quality_workout_percent', 0) or 0, cache_f.get('quality_workout_percent', 0) or 0),
    }


def main():
    print("=" * 140)
    print("SALMAN'S TRAINING STATS: Cache vs CSV vs Hybrid")
    print("=" * 140)

    # Load cache data
    with open('race_data/combined_41_features.json', 'r') as f:
        all_races = json.load(f)

    # Filter to Salman's races
    salman_races = [r for r in all_races if r['runner_id'] == 'runner_2']
    salman_races.sort(key=lambda x: x['race_date'])

    # Load CSV activities
    csv_activities = parse_strava_csv(SALMAN_CSV)
    print(f"Total CSV activities for Salman: {len(csv_activities)}")

    # Key features to compare
    features = [
        ('total_weekly_mileage', 'Wkly Miles'),
        ('peak_weekly_mileage', 'Peak Miles'),
        ('runs_per_week', 'Runs/Wk'),
        ('total_runs', 'Total Runs'),
        ('long_run_distance', 'Long Run'),
        ('tempo_workout_count', 'Tempo Cnt'),
        ('quality_workout_percent', 'Quality %'),
    ]

    print(f"\nTraining window: 16 weeks before race, excluding last 7 days (taper)")
    print("\n" + "=" * 140)

    # Header
    print(f"\n{'Race Date':<12} {'Actual':<8} {'Training Window':<25} ", end="")
    for _, label in features:
        print(f"{label:<12}", end="")
    print()
    print("-" * 140)

    for race in salman_races:
        race_date = race['race_date'][:10]
        actual = race['actual_time_minutes']
        cache_f = race.get('features', {})

        csv_f, window_start, window_end = extract_csv_features(csv_activities, race_date)
        hybrid_f = create_hybrid_features(cache_f, csv_f)

        window_str = f"{window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}" if window_start else "N/A"

        # Print Cache row
        print(f"\n{race_date:<12} {format_time(actual):<8} {window_str:<25} ", end="")
        for feat, _ in features:
            val = cache_f.get(feat, 0) or 0
            print(f"{val:<12.1f}", end="")
        print(" [CACHE]")

        # Print CSV row
        print(f"{'':<12} {'':<8} {'':<25} ", end="")
        for feat, _ in features:
            val = csv_f.get(feat, 0) or 0
            cache_val = cache_f.get(feat, 0) or 0
            diff = val - cache_val
            diff_str = f"({diff:+.1f})" if abs(diff) > 0.5 else ""
            print(f"{val:<6.1f}{diff_str:<6}", end="")
        print(" [CSV]")

        # Print Hybrid row
        print(f"{'':<12} {'':<8} {'':<25} ", end="")
        for feat, _ in features:
            val = hybrid_f.get(feat, 0) or 0
            print(f"{val:<12.1f}", end="")
        print(" [HYBRID]")

        print("-" * 140)

    # Summary statistics
    print("\n" + "=" * 140)
    print("SUMMARY: Average feature values across all Salman's races")
    print("=" * 140)

    cache_avgs = {f: [] for f, _ in features}
    csv_avgs = {f: [] for f, _ in features}
    hybrid_avgs = {f: [] for f, _ in features}

    for race in salman_races:
        cache_f = race.get('features', {})
        csv_f, _, _ = extract_csv_features(csv_activities, race['race_date'])
        hybrid_f = create_hybrid_features(cache_f, csv_f)

        for feat, _ in features:
            cache_avgs[feat].append(cache_f.get(feat, 0) or 0)
            csv_avgs[feat].append(csv_f.get(feat, 0) or 0)
            hybrid_avgs[feat].append(hybrid_f.get(feat, 0) or 0)

    print(f"\n{'Feature':<25} {'Cache Avg':<15} {'CSV Avg':<15} {'Hybrid Avg':<15} {'Diff (CSV-Cache)':<15}")
    print("-" * 90)

    for feat, label in features:
        cache_avg = sum(cache_avgs[feat]) / len(cache_avgs[feat])
        csv_avg = sum(csv_avgs[feat]) / len(csv_avgs[feat])
        hybrid_avg = sum(hybrid_avgs[feat]) / len(hybrid_avgs[feat])
        diff = csv_avg - cache_avg

        print(f"{label:<25} {cache_avg:<15.1f} {csv_avg:<15.1f} {hybrid_avg:<15.1f} {diff:+.1f}")

    # Breakdown by performance tier
    print("\n" + "=" * 140)
    print("BREAKDOWN BY PERFORMANCE (Sub-3:00 vs 3:00-3:30 vs 3:30+)")
    print("=" * 140)

    tiers = {
        'Sub-3:00': [r for r in salman_races if r['actual_time_minutes'] < 180],
        '3:00-3:30': [r for r in salman_races if 180 <= r['actual_time_minutes'] < 210],
        '3:30+': [r for r in salman_races if r['actual_time_minutes'] >= 210],
    }

    for tier_name, tier_races in tiers.items():
        if not tier_races:
            continue

        print(f"\n{tier_name} ({len(tier_races)} races):")
        print(f"  {'Feature':<20} {'Cache':<12} {'CSV':<12} {'Hybrid':<12}")
        print("  " + "-" * 60)

        for feat, label in features[:5]:  # Top 5 features
            cache_avg = sum((r.get('features', {}).get(feat, 0) or 0) for r in tier_races) / len(tier_races)
            csv_vals = []
            hybrid_vals = []
            for r in tier_races:
                csv_f, _, _ = extract_csv_features(csv_activities, r['race_date'])
                hybrid_f = create_hybrid_features(r.get('features', {}), csv_f)
                csv_vals.append(csv_f.get(feat, 0) or 0)
                hybrid_vals.append(hybrid_f.get(feat, 0) or 0)

            csv_avg = sum(csv_vals) / len(csv_vals)
            hybrid_avg = sum(hybrid_vals) / len(hybrid_vals)

            print(f"  {label:<20} {cache_avg:<12.1f} {csv_avg:<12.1f} {hybrid_avg:<12.1f}")


if __name__ == '__main__':
    main()
