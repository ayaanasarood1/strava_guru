#!/usr/bin/env python3
"""
Quick test script to analyze a sample activity
"""

from pathlib import Path
from activity_analyzer import ActivityAnalyzer, print_activity_summary

# Path to your Strava export
STRAVA_EXPORT = Path.home() / "Downloads" / "export_40402578"

# Get the first few activities
activities_dir = STRAVA_EXPORT / "activities"
activity_files = sorted(activities_dir.glob("*.fit.gz"))[:5]

print(f"Found {len(activity_files)} sample activities to analyze\n")

for activity_file in activity_files:
    print(f"\n{'='*80}")
    print(f"Analyzing: {activity_file.name}")
    print('='*80)

    analyzer = ActivityAnalyzer()
    stats = analyzer.analyze_file(activity_file)

    if stats:
        print_activity_summary(stats)
    else:
        print(f"Failed to analyze {activity_file.name}")

    # Pause between activities
    input("\nPress Enter to continue to next activity...")
