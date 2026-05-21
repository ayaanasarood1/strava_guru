#!/usr/bin/env python3
"""
Test parsing 10 FIT files from Salman's data
"""

from pathlib import Path
from activity_analyzer import ActivityAnalyzer

def main():
    activities_dir = Path('/Users/osman/Downloads/export_1884062_salman/activities')

    if not activities_dir.exists():
        print(f"Error: Directory not found: {activities_dir}")
        return

    # Get all FIT files
    fit_files = sorted(activities_dir.glob("*.fit.gz"))
    print(f"Found {len(fit_files)} total FIT files in {activities_dir}")

    # Test with first 10 files
    test_files = fit_files[:10]
    print(f"\nTesting with first 10 files...\n")

    analyzer = ActivityAnalyzer()

    successful = 0
    failed = 0

    for i, file_path in enumerate(test_files, 1):
        print(f"{i}/10: {file_path.name}")

        try:
            stats = analyzer.analyze_file(file_path)

            if stats:
                print(f"  ✓ Success!")
                print(f"    Date: {stats.start_time}")
                print(f"    Distance: {stats.total_distance/1609.34:.2f} miles")
                print(f"    Duration: {stats.total_time.total_seconds()/60:.1f} minutes")
                print(f"    Avg HR: {stats.avg_heart_rate or 'N/A'}")
                print(f"    Track points: {len(stats.track_points) if stats.track_points else 0}")
                successful += 1
            else:
                print(f"  ✗ Failed to parse (returned None)")
                failed += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            failed += 1

        print()

    print("="*80)
    print(f"Results: {successful} successful, {failed} failed")
    print("="*80)

    if successful > 0:
        print("\n✓ FIT parsing is working! Ready to process more files.")
    else:
        print("\n✗ All files failed. Need to investigate further.")

if __name__ == '__main__':
    main()
