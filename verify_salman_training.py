#!/usr/bin/env python3
"""
Verify Salman's actual training from activities.csv
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict

def main():
    csv_path = '/Users/osman/Downloads/export_1884062_salman/activities.csv'

    # Jack & Jill 2025 marathon
    race_date = datetime(2025, 7, 27)

    # Training window: 12 weeks before race, excluding taper week
    taper_start = race_date - timedelta(days=7)
    training_start = taper_start - timedelta(weeks=12)

    print("="*80)
    print(f"Verifying Salman's Training for Jack & Jill 2025 (2:55 marathon)")
    print("="*80)
    print(f"\nRace date: {race_date.date()}")
    print(f"Training window: {training_start.date()} to {taper_start.date()}")
    print(f"Duration: 12 weeks\n")

    # Load activities
    runs = []

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

                # Check if in training window
                if training_start <= activity_date < taper_start:
                    distance_m = float(row.get('Distance', 0) or 0)
                    distance_mi = distance_m / 1609.34
                    duration_s = float(row.get('Moving Time', 0) or 0)
                    duration_min = duration_s / 60.0

                    runs.append({
                        'date': activity_date,
                        'distance_mi': distance_mi,
                        'duration_min': duration_min,
                        'pace': duration_min / distance_mi if distance_mi > 0 else 0
                    })

            except Exception as e:
                continue

    runs.sort(key=lambda x: x['date'])

    print(f"Found {len(runs)} runs in training window\n")

    if not runs:
        print("⚠️ NO RUNS FOUND! This explains the 8.4 mi/week feature extraction.")
        print("The feature extraction found no data, which is correct.")
        return

    # Calculate weekly totals
    weekly_miles = defaultdict(float)
    weekly_runs = defaultdict(int)

    for run in runs:
        week_num = (run['date'] - training_start).days // 7
        weekly_miles[week_num] += run['distance_mi']
        weekly_runs[week_num] += 1

    total_miles = sum(run['distance_mi'] for run in runs)
    avg_weekly_miles = total_miles / 12  # 12 weeks

    print("="*80)
    print("Weekly Breakdown:")
    print("="*80)

    for week in range(12):
        week_start = training_start + timedelta(weeks=week)
        week_miles = weekly_miles[week]
        week_run_count = weekly_runs[week]
        print(f"Week {week+1:2d} ({week_start.date()}): {week_miles:5.1f} miles, {week_run_count} runs")

    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    print(f"Total runs: {len(runs)}")
    print(f"Total miles: {total_miles:.1f} miles")
    print(f"Average weekly mileage: {avg_weekly_miles:.1f} mi/week")
    print(f"Peak weekly mileage: {max(weekly_miles.values()):.1f} miles")
    print(f"Runs per week: {len(runs) / 12:.1f}")

    # Compare to extracted feature
    print("\n" + "="*80)
    print("Comparison to Extracted Features:")
    print("="*80)
    print(f"Extracted feature: 8.4 mi/week")
    print(f"Actual from CSV: {avg_weekly_miles:.1f} mi/week")

    if abs(avg_weekly_miles - 8.4) < 1.0:
        print("\n✓ Feature extraction is CORRECT!")
        print("  Salman really did have very low mileage during this training block.")
        print("  This suggests he ran 2:55 on residual fitness from previous training.")
    else:
        print(f"\n⚠️ Mismatch! Off by {abs(avg_weekly_miles - 8.4):.1f} mi/week")
        print("  This indicates a feature extraction issue.")

    # Show recent runs
    print("\n" + "="*80)
    print("Last 10 runs before taper:")
    print("="*80)
    for run in runs[-10:]:
        print(f"{run['date'].date()}: {run['distance_mi']:5.2f} mi @ {run['pace']:.2f} min/mile")

    # Check if there are runs right before the window that we missed
    print("\n" + "="*80)
    print("Checking for runs BEFORE training window:")
    print("="*80)

    early_runs = []
    check_start = training_start - timedelta(weeks=4)

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

                if check_start <= activity_date < training_start:
                    distance_m = float(row.get('Distance', 0) or 0)
                    distance_mi = distance_m / 1609.34
                    early_runs.append({
                        'date': activity_date,
                        'distance_mi': distance_mi
                    })

            except:
                continue

    if early_runs:
        early_runs.sort(key=lambda x: x['date'])
        early_total = sum(r['distance_mi'] for r in early_runs)
        print(f"Found {len(early_runs)} runs in 4 weeks before training window")
        print(f"Total: {early_total:.1f} miles ({early_total/4:.1f} mi/week)")
        print("\nLast 5 runs before training window:")
        for run in early_runs[-5:]:
            print(f"  {run['date'].date()}: {run['distance_mi']:.2f} mi")
    else:
        print("No runs found in 4 weeks before training window")

if __name__ == '__main__':
    main()
