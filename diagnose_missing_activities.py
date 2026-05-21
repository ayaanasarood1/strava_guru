#!/usr/bin/env python3
"""
Diagnose why April-July 2025 activities aren't being cached
"""

import csv
from pathlib import Path
from datetime import datetime
from activity_analyzer import ActivityAnalyzer

def main():
    print("="*80)
    print("Diagnosing Missing Activities in Cache")
    print("="*80)

    # Load activities.csv to see what should be there
    csv_path = '/Users/osman/Downloads/export_1884062_salman/activities.csv'
    activities_dir = Path('/Users/osman/Downloads/export_1884062_salman/activities')

    # Training window for Jack & Jill 2025
    race_date = datetime(2025, 7, 27)
    training_start = datetime(2025, 4, 27)
    taper_start = datetime(2025, 7, 20)

    print(f"\nTraining window: {training_start.date()} to {taper_start.date()}")
    print(f"Activities directory: {activities_dir}")
    print()

    # Parse CSV to find runs in training window
    training_runs = []

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

                if training_start <= activity_date < taper_start:
                    # Get filename - strip 'activities/' prefix if present
                    filename = row.get('Filename', '')
                    if filename.startswith('activities/'):
                        filename = filename[11:]  # Remove 'activities/' prefix

                    training_runs.append({
                        'date': activity_date,
                        'filename': filename,
                        'distance_mi': float(row.get('Distance', 0) or 0) / 1609.34,
                        'duration_min': float(row.get('Moving Time', 0) or 0) / 60.0
                    })
            except Exception as e:
                continue

    training_runs.sort(key=lambda x: x['date'])

    print(f"Found {len(training_runs)} runs in CSV for training window\n")

    # Check which files exist and can be parsed
    print("="*80)
    print("Checking File Existence and Parseability")
    print("="*80)

    analyzer = ActivityAnalyzer()

    file_exists_count = 0
    file_missing_count = 0
    parse_success_count = 0
    parse_fail_count = 0

    failed_files = []
    missing_files = []

    for i, run in enumerate(training_runs, 1):
        filename = run['filename']
        file_path = activities_dir / filename

        # Check if file exists
        if not file_path.exists():
            file_missing_count += 1
            missing_files.append(run)
            if i <= 10:  # Show first 10
                print(f"{i:3d}. MISSING: {filename}")
                print(f"     Date: {run['date'].date()}, Distance: {run['distance_mi']:.2f} mi")
            continue

        file_exists_count += 1

        # Try to parse
        try:
            stats = analyzer.analyze_file(file_path)
            if stats and stats.start_time:
                parse_success_count += 1
                if i <= 10:
                    print(f"{i:3d}. ✓ PARSED: {filename}")
                    print(f"     Date: {stats.start_time.date()}, Distance: {stats.total_distance/1609.34:.2f} mi")
            else:
                parse_fail_count += 1
                failed_files.append({**run, 'reason': 'analyze_file returned None'})
                if i <= 10:
                    print(f"{i:3d}. ✗ FAILED: {filename} (returned None)")
        except Exception as e:
            parse_fail_count += 1
            failed_files.append({**run, 'reason': str(e)})
            if i <= 10:
                print(f"{i:3d}. ✗ FAILED: {filename}")
                print(f"     Error: {type(e).__name__}: {str(e)[:80]}")

        print()

    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Total runs in CSV: {len(training_runs)}")
    print(f"Files exist: {file_exists_count}")
    print(f"Files missing: {file_missing_count}")
    print(f"Parse successful: {parse_success_count}")
    print(f"Parse failed: {parse_fail_count}")
    print()

    # Root cause analysis
    print("="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    if file_missing_count > 0:
        print(f"\n❌ PROBLEM 1: {file_missing_count} files are MISSING from disk!")
        print("   These files are in activities.csv but not in the activities/ directory")
        print("   This could mean:")
        print("   - Files were deleted after export")
        print("   - Export was incomplete")
        print("   - Filenames in CSV don't match actual files")

        if missing_files:
            print(f"\n   First 5 missing files:")
            for run in missing_files[:5]:
                print(f"   - {run['filename']}")
                print(f"     {run['date'].date()}: {run['distance_mi']:.2f} mi")

    if parse_fail_count > 0:
        print(f"\n❌ PROBLEM 2: {parse_fail_count} files EXIST but FAIL to parse!")
        print("   These files are corrupted or have parsing issues")

        if failed_files:
            print(f"\n   First 5 failed files:")
            for run in failed_files[:5]:
                print(f"   - {run['filename']}")
                print(f"     {run['date'].date()}: {run['distance_mi']:.2f} mi")
                print(f"     Reason: {run['reason'][:100]}")

    if parse_success_count == len(training_runs):
        print("\n✓ All files exist and parse successfully!")
        print("  Issue must be in:")
        print("  - Cache storage logic")
        print("  - Date filtering during cache build")
        print("  - Activity type filtering")

    # Check a few specific dates
    print("\n" + "="*80)
    print("Spot Check: Specific Dates")
    print("="*80)

    # Check May 15, 2025 (should have activity)
    may_15_runs = [r for r in training_runs if r['date'].date() == datetime(2025, 5, 15).date()]
    print(f"\nMay 15, 2025: {len(may_15_runs)} runs in CSV")
    for run in may_15_runs:
        print(f"  - {run['filename']}: {run['distance_mi']:.2f} mi")

    # Check June 1, 2025
    june_1_runs = [r for r in training_runs if r['date'].date() == datetime(2025, 6, 1).date()]
    print(f"\nJune 1, 2025: {len(june_1_runs)} runs in CSV")
    for run in june_1_runs:
        print(f"  - {run['filename']}: {run['distance_mi']:.2f} mi")

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)

    if file_missing_count > parse_fail_count:
        print("PRIMARY ISSUE: Files are missing from disk")
        print("→ Need to check if CSV has wrong filenames or files were deleted")
    elif parse_fail_count > 0:
        print("PRIMARY ISSUE: Files fail to parse")
        print("→ Need to investigate ActivityAnalyzer parsing logic")
    else:
        print("FILES ARE FINE: Issue is in cache storage or filtering")
        print("→ Need to check cache.cache_activity() and date filtering")

if __name__ == '__main__':
    main()
