#!/usr/bin/env python3
"""
Batch visualize multiple Strava activities
"""

from pathlib import Path
import sys

from activity_analyzer import ActivityAnalyzer
from visualizer import ActivityVisualizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_all.py <activities_directory> [output_directory] [--limit N]")
        print("\nExample:")
        print("  python visualize_all.py ~/Downloads/export_40402578/activities/ ./charts/")
        print("  python visualize_all.py ~/Downloads/export_40402578/activities/ ./charts/ --limit 10")
        sys.exit(1)

    activities_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('./charts')

    # Parse limit
    limit = None
    if '--limit' in sys.argv:
        limit_idx = sys.argv.index('--limit')
        if limit_idx + 1 < len(sys.argv):
            limit = int(sys.argv[limit_idx + 1])

    if not activities_dir.exists():
        print(f"Error: Directory not found: {activities_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all activity files
    activity_files = sorted(activities_dir.glob("*.fit.gz"))
    if not activity_files:
        activity_files = sorted(activities_dir.glob("*.fit"))
    if not activity_files:
        activity_files = sorted(activities_dir.glob("*.gpx*"))

    if limit:
        activity_files = activity_files[:limit]

    print(f"Found {len(activity_files)} activity files")
    print(f"Output directory: {output_dir}")
    print()

    successful = 0
    failed = 0

    for i, file_path in enumerate(activity_files, 1):
        print(f"[{i}/{len(activity_files)}] Processing {file_path.name}...", end=' ')

        try:
            # Analyze
            analyzer = ActivityAnalyzer()
            stats = analyzer.analyze_file(file_path)

            if not stats:
                print("❌ Failed to analyze")
                failed += 1
                continue

            # Visualize
            viz = ActivityVisualizer(stats)
            base_name = file_path.stem.replace('.fit', '').replace('.gpx', '')

            # Create full report only (skip individual charts to save time)
            output_path = output_dir / f"{base_name}_report.png"
            viz.create_full_report(output_path)

            print(f"✅ Saved")
            successful += 1

        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
            continue

    print()
    print("="*60)
    print(f"Complete! Successful: {successful}, Failed: {failed}")
    print(f"Charts saved to: {output_dir}")


if __name__ == '__main__':
    main()
