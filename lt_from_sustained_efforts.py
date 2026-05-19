#!/usr/bin/env python3
"""
Find lactate threshold from sustained hard efforts in continuous runs
(Doesn't require manual laps)
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

from activity_cache import ActivityCache
from activity_analyzer import ActivityAnalyzer, format_pace


@dataclass
class SustainedEffort:
    """A sustained threshold effort found in a continuous run"""
    date: datetime
    activity_name: str
    start_time: float  # seconds into run
    duration: timedelta
    distance_mi: float
    avg_hr: int
    avg_pace: float  # min/mile


def find_sustained_threshold_efforts(activity, min_hr: int = 150, min_duration_min: int = 10):
    """
    Find sustained threshold efforts within an activity

    Looks for periods of 10+ minutes where:
    - HR stays above threshold (e.g., 150 bpm)
    - Pace is consistent
    - Effort is sustained (not intervals with rest)
    """
    # Parse the activity file to get track points
    activities_dir = Path.home() / "Downloads" / "export_40402578" / "activities"
    file_path = activities_dir / activity['file_name']

    if not file_path.exists():
        return []

    analyzer = ActivityAnalyzer()
    stats = analyzer.analyze_file(file_path)

    if not stats or not stats.track_points:
        return []

    # Find sustained periods above threshold HR
    efforts = []
    in_effort = False
    effort_start_idx = None

    for i, point in enumerate(stats.track_points):
        if not point.heart_rate:
            continue

        if point.heart_rate >= min_hr and not in_effort:
            # Start of potential threshold effort
            in_effort = True
            effort_start_idx = i

        elif (point.heart_rate < min_hr - 5 or i == len(stats.track_points) - 1) and in_effort:
            # End of effort (HR dropped or end of run)
            in_effort = False

            if effort_start_idx is not None:
                effort_points = stats.track_points[effort_start_idx:i]

                if len(effort_points) < 60:  # Need at least 60 seconds
                    continue

                # Calculate effort stats
                duration = effort_points[-1].timestamp - effort_points[0].timestamp
                duration_min = duration.total_seconds() / 60

                if duration_min < min_duration_min:
                    continue

                # Distance
                if effort_points[0].distance and effort_points[-1].distance:
                    distance_m = effort_points[-1].distance - effort_points[0].distance
                    distance_mi = distance_m * 0.000621371
                else:
                    continue

                # Average HR
                hrs = [p.heart_rate for p in effort_points if p.heart_rate]
                if not hrs:
                    continue
                avg_hr = int(np.mean(hrs))

                # Average pace
                pace_sec_per_mile = duration.total_seconds() / distance_mi if distance_mi > 0 else 0
                avg_pace = pace_sec_per_mile / 60  # min/mile

                # Check pace is reasonable (5-9 min/mile)
                if not (5 < avg_pace < 9):
                    continue

                # Check pace consistency (CV < 20%)
                paces = []
                for j in range(len(effort_points) - 1):
                    if effort_points[j].speed and effort_points[j].speed > 0:
                        pace = 26.8224 / effort_points[j].speed
                        if 5 < pace < 9:
                            paces.append(pace)

                if paces:
                    pace_cv = np.std(paces) / np.mean(paces)
                    if pace_cv > 0.20:  # Too variable
                        continue

                effort = SustainedEffort(
                    date=stats.start_time,
                    activity_name=activity['file_name'],
                    start_time=effort_start_idx,
                    duration=duration,
                    distance_mi=distance_mi,
                    avg_hr=avg_hr,
                    avg_pace=avg_pace
                )
                efforts.append(effort)

    return efforts


def main():
    """Main entry point"""
    import sys

    cache = ActivityCache()
    stats = cache.get_stats()

    if stats['total_activities'] == 0:
        print("Error: Cache is empty! Run activity_cache.py first.")
        sys.exit(1)

    # Parse arguments
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    min_hr = 150

    if '--min-hr' in sys.argv:
        hr_idx = sys.argv.index('--min-hr')
        if hr_idx + 1 < len(sys.argv):
            min_hr = int(sys.argv[hr_idx + 1])

    print(f"Finding sustained threshold efforts (last {months} months)")
    print(f"Criteria: HR >= {min_hr} bpm, 10+ min duration, consistent pace")
    print()

    # Get recent activities with HR
    start_date = datetime.now() - timedelta(days=months * 30)
    activities = cache.get_activities_by_date_range(start_date, datetime.now())
    activities = [a for a in activities if a['avg_heart_rate']]

    print(f"Analyzing {len(activities)} activities...")

    all_efforts = []
    for i, activity in enumerate(activities):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(activities)}")

        efforts = find_sustained_threshold_efforts(activity, min_hr=min_hr)
        all_efforts.extend(efforts)

    print(f"\nFound {len(all_efforts)} sustained threshold efforts!")
    print()

    if len(all_efforts) == 0:
        print("No threshold efforts found. Try lowering --min-hr")
        sys.exit(1)

    # Calculate LT
    hrs = [e.avg_hr for e in all_efforts]
    paces = [e.avg_pace for e in all_efforts]

    # Weight recent efforts more
    sorted_efforts = sorted(all_efforts, key=lambda e: e.date)
    weights = np.linspace(0.5, 1.0, len(sorted_efforts))

    lt_hr = int(np.average(hrs, weights=weights))
    lt_pace = np.average(paces, weights=weights)

    print("="*60)
    print("🎯 LACTATE THRESHOLD (from sustained efforts)")
    print("="*60)
    print(f"  LT Heart Rate: {lt_hr} bpm")
    print(f"  LT Pace:       {format_pace(lt_pace)}")
    print(f"  Based on:      {len(all_efforts)} efforts")
    print("="*60)
    print()

    # Show recent efforts
    print("Recent threshold efforts:")
    print("-" * 80)
    for effort in sorted_efforts[-10:]:
        print(f"{effort.date.strftime('%Y-%m-%d')}: {effort.duration.seconds//60}min @ {format_pace(effort.avg_pace)}, {effort.avg_hr} bpm")


if __name__ == '__main__':
    main()
