#!/usr/bin/env python3
"""
Lactate Threshold from Actual Workouts
Uses your actual tempo/threshold workouts (laps) instead of statistical guessing
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from activity_cache import ActivityCache
from activity_analyzer import format_pace, format_time


@dataclass
class ThresholdEffort:
    """A tempo/threshold lap or workout"""
    date: datetime
    activity_name: str
    lap_number: int
    distance_mi: float
    duration: timedelta
    pace: float  # min/mile
    gap: float  # grade adjusted pace
    avg_hr: int
    max_hr: int


class WorkoutBasedLTAnalyzer:
    """Analyze LT from actual tempo/threshold workouts"""

    def __init__(self, cache: ActivityCache):
        self.cache = cache
        self.threshold_efforts: List[ThresholdEffort] = []

    def find_threshold_workouts(self, months_back: Optional[int] = None, min_hr: int = 150):
        """
        Find activities with tempo/threshold efforts

        Criteria for threshold effort:
        - Laps between 1-4 miles (not too short, not full race distance)
        - Consistent pace (CV < 15%)
        - Hard effort (HR >= 150 bpm - true threshold effort)
        - Duration 6-30 minutes per lap
        - Pace between 5-9 min/mile (not easy pace)
        """
        print("Searching for tempo/threshold workouts...")

        # Get activities with laps
        if months_back:
            start_date = datetime.now() - timedelta(days=months_back * 30)
            activities = self.cache.get_activities_by_date_range(start_date, datetime.now())
        else:
            activities = self.cache.get_activities_with_hr()

        activities_with_laps = [a for a in activities if a['laps_json']]

        print(f"Found {len(activities_with_laps)} activities with laps")

        for activity in activities_with_laps:
            laps = json.loads(activity['laps_json'])

            # Filter for threshold-effort laps
            tempo_laps = []
            activity_hr = activity['avg_heart_rate']  # Fall back to activity HR if lap HR missing

            for lap in laps:
                # Check criteria
                distance_mi = lap['distance'] * 0.000621371
                duration_sec = lap['duration_seconds']
                duration_min = duration_sec / 60

                # Must be 1-4 miles
                if not (1.0 <= distance_mi <= 4.0):
                    continue

                # Must be 6-30 minutes
                if not (6 <= duration_min <= 30):
                    continue

                # Use lap HR if available, otherwise use activity HR
                lap_hr = lap['avg_hr'] if lap['avg_hr'] else activity_hr

                # Must have HR data and be hard effort
                if not lap_hr or lap_hr < min_hr:
                    continue

                # Pace must be reasonable threshold pace (5-9 min/mile, not slow easy runs)
                if not (5 < lap['pace'] < 9):
                    continue

                # Store with HR (lap or activity level)
                lap_copy = lap.copy()
                if not lap_copy['avg_hr']:
                    lap_copy['avg_hr'] = activity_hr
                    lap_copy['max_hr'] = activity['max_heart_rate']

                tempo_laps.append(lap_copy)

            # If we have 2+ consistent tempo laps, it's likely a threshold workout
            if len(tempo_laps) >= 2:
                # Check pace consistency
                paces = [lap['pace'] for lap in tempo_laps]
                pace_cv = np.std(paces) / np.mean(paces)

                # If pace is consistent (< 15% variation), these are threshold efforts
                if pace_cv < 0.15:
                    for lap in tempo_laps:
                        effort = ThresholdEffort(
                            date=datetime.fromisoformat(activity['activity_date']),
                            activity_name=activity['file_name'],
                            lap_number=lap['number'],
                            distance_mi=lap['distance'] * 0.000621371,
                            duration=timedelta(seconds=lap['duration_seconds']),
                            pace=lap['pace'],
                            gap=lap['gap'],
                            avg_hr=lap['avg_hr'],
                            max_hr=lap['max_hr']
                        )
                        self.threshold_efforts.append(effort)

        print(f"Identified {len(self.threshold_efforts)} threshold laps from {len(set(e.activity_name for e in self.threshold_efforts))} workouts")

        return self.threshold_efforts

    def estimate_lactate_threshold(self) -> Tuple[float, float]:
        """
        Estimate LT from threshold efforts
        Returns: (LT HR, LT Pace)
        """
        if len(self.threshold_efforts) < 5:
            raise ValueError(f"Need at least 5 threshold efforts, found {len(self.threshold_efforts)}")

        # Recent efforts are more relevant - weight by recency
        sorted_efforts = sorted(self.threshold_efforts, key=lambda e: e.date)
        weights = np.linspace(0.5, 1.0, len(sorted_efforts))  # Recent efforts weighted more

        # Calculate weighted averages
        hrs = np.array([e.avg_hr for e in sorted_efforts])
        paces = np.array([e.gap if e.gap > 0 else e.pace for e in sorted_efforts])

        lt_hr = np.average(hrs, weights=weights)
        lt_pace = np.average(paces, weights=weights)

        return lt_hr, lt_pace

    def analyze_trends(self):
        """Analyze trends over time"""
        if len(self.threshold_efforts) < 10:
            print("Need at least 10 threshold efforts to analyze trends")
            return

        sorted_efforts = sorted(self.threshold_efforts, key=lambda e: e.date)

        # Group by month
        monthly_data = {}
        for effort in sorted_efforts:
            month_key = effort.date.strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(effort)

        print("\nThreshold Trends by Month:")
        print("-" * 60)
        print(f"{'Month':<10} {'Workouts':<10} {'Avg HR':<10} {'Avg Pace':<10}")
        print("-" * 60)

        for month, efforts in sorted(monthly_data.items()):
            avg_hr = np.mean([e.avg_hr for e in efforts])
            avg_pace = np.mean([e.gap if e.gap > 0 else e.pace for e in efforts])

            print(f"{month:<10} {len(efforts):<10} {avg_hr:.0f} bpm    {format_pace(avg_pace)}")

    def plot_threshold_progression(self, output_path: Optional[Path] = None):
        """Visualize threshold progression over time"""
        if len(self.threshold_efforts) < 5:
            print("Not enough data for visualization")
            return

        sorted_efforts = sorted(self.threshold_efforts, key=lambda e: e.date)

        dates = [e.date for e in sorted_efforts]
        hrs = [e.avg_hr for e in sorted_efforts]
        paces = [e.gap if e.gap > 0 else e.pace for e in sorted_efforts]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # HR over time
        ax1.scatter(dates, hrs, alpha=0.6, s=50)
        ax1.plot(dates, hrs, alpha=0.3, linestyle='-', linewidth=1)

        # Add trend line
        from sklearn.linear_model import LinearRegression
        X = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
        y_hr = np.array(hrs)
        model = LinearRegression()
        model.fit(X, y_hr)
        trend_hr = model.predict(X)
        ax1.plot(dates, trend_hr, 'r--', linewidth=2, label='Trend')

        ax1.set_ylabel('Heart Rate (bpm)', fontweight='bold')
        ax1.set_title('Threshold Heart Rate Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Pace over time
        ax2.scatter(dates, paces, alpha=0.6, s=50, color='orange')
        ax2.plot(dates, paces, alpha=0.3, linestyle='-', linewidth=1, color='orange')

        # Add trend line
        y_pace = np.array(paces)
        model.fit(X, y_pace)
        trend_pace = model.predict(X)
        ax2.plot(dates, trend_pace, 'r--', linewidth=2, label='Trend')

        ax2.set_ylabel('Pace (min/mile)', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.set_title('Threshold Pace Over Time', fontweight='bold')
        ax2.invert_yaxis()  # Faster pace = lower on chart
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main entry point"""
    import sys

    cache = ActivityCache()
    stats = cache.get_stats()

    if stats['total_activities'] == 0:
        print("Error: Cache is empty! Run activity_cache.py first to build the cache.")
        sys.exit(1)

    print(f"Using cached data: {stats['total_activities']} activities")

    # Parse arguments
    months = None
    min_hr = 150  # Default threshold HR

    if len(sys.argv) > 1:
        try:
            months = int(sys.argv[1])
            print(f"Analyzing last {months} months")
        except ValueError:
            print(f"Usage: python lt_from_workouts.py [months] [--min-hr HR]")
            sys.exit(1)

    if '--min-hr' in sys.argv:
        hr_idx = sys.argv.index('--min-hr')
        if hr_idx + 1 < len(sys.argv):
            min_hr = int(sys.argv[hr_idx + 1])

    print(f"Minimum HR for threshold efforts: {min_hr} bpm")

    analyzer = WorkoutBasedLTAnalyzer(cache)
    analyzer.find_threshold_workouts(months_back=months, min_hr=min_hr)

    if len(analyzer.threshold_efforts) < 5:
        print("\n⚠️  Not enough threshold workouts found!")
        print("Make sure you're creating laps during tempo/threshold runs.")
        print("The analyzer looks for:")
        print("  - Laps between 1-4 miles")
        print("  - Duration 6-30 minutes per lap")
        print("  - Consistent pace across laps")
        print("  - HR >= 150 bpm (hard effort)")
        print("  - Pace 5-9 min/mile")
        sys.exit(1)

    # Estimate LT
    lt_hr, lt_pace = analyzer.estimate_lactate_threshold()

    print("\n" + "="*60)
    print("🎯 LACTATE THRESHOLD ESTIMATE (from actual workouts)")
    print("="*60)
    print(f"  LT Heart Rate: {lt_hr:.0f} bpm")
    print(f"  LT Pace (GAP):  {format_pace(lt_pace)}")
    print(f"  Based on:      {len(analyzer.threshold_efforts)} threshold laps")
    print(f"  From:          {len(set(e.activity_name for e in analyzer.threshold_efforts))} workouts")
    print("="*60)

    # Show trends
    analyzer.analyze_trends()

    # Visualize
    output_path = Path("./charts/threshold_workout_progression.png")
    analyzer.plot_threshold_progression(output_path)

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
