#!/usr/bin/env python3
"""
Recent Lactate Threshold Analysis - Last 3 months only
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys

from lactate_threshold_analyzer import LactateThresholdAnalyzer
from activity_analyzer import ActivityAnalyzer

def main():
    activities_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "export_40402578" / "activities"

    # Calculate 3 months ago
    months_ago = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cutoff_date = datetime.now() - timedelta(days=months_ago * 30)

    print(f"Analyzing activities from last {months_ago} months (since {cutoff_date.strftime('%Y-%m-%d')})...")

    # Load and filter activities
    analyzer = LactateThresholdAnalyzer()
    activity_files = sorted(activities_dir.glob("*.fit.gz"))

    loaded = 0
    skipped_old = 0

    for file_path in activity_files:
        try:
            stats = ActivityAnalyzer().analyze_file(file_path)
            if stats:
                if stats.start_time >= cutoff_date:
                    if stats.avg_heart_rate and stats.avg_pace > 0:
                        analyzer.activities.append(stats)
                        loaded += 1
                else:
                    skipped_old += 1
        except:
            continue

    print(f"Loaded {loaded} recent activities (skipped {skipped_old} older activities)")

    if loaded < 5:
        print("Error: Need at least 5 activities for reliable LT estimation")
        sys.exit(1)

    # Extract HR-pace data
    analyzer._extract_hr_pace_data()

    # Estimate LT
    lt_estimate = analyzer.estimate_lactate_threshold()

    # Visualize
    output_path = Path(f"./charts/lactate_threshold_last_{months_ago}_months.png")
    print(f"\nGenerating visualization...")
    analyzer.visualize_analysis(lt_estimate, output_path)

    print(f"\n✅ Analysis complete!")
    print(f"\nTo compare with full history:")
    print(f"  Recent ({months_ago}mo):  LT HR={lt_estimate.lt_heart_rate:.0f} bpm, Pace={lt_estimate.lt_pace:.2f} min/mi")
    print(f"  Full data:      LT HR=157 bpm, Pace=7.67 min/mi")

if __name__ == '__main__':
    main()
