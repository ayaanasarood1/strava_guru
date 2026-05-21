#!/usr/bin/env python3
"""
Create Sample Race Results for Training

This creates synthetic race results from your activity history
for demonstration purposes. In production, use actual race results.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from activity_cache import ActivityCache


def create_synthetic_race_results():
    """Create synthetic race results from activity history"""

    cache = ActivityCache()
    stats = cache.get_stats()

    print("=" * 60)
    print("Creating Synthetic Race Results")
    print("=" * 60)

    if stats['total_activities'] == 0:
        print("ERROR: No activities in cache")
        return

    print(f"\nActivity cache: {stats['total_activities']} activities")
    print(f"Date range: {stats['date_range']}")

    # Get date range
    if stats['date_range'][0] and stats['date_range'][1]:
        start_date = datetime.fromisoformat(stats['date_range'][0])
        end_date = datetime.fromisoformat(stats['date_range'][1])
    else:
        print("ERROR: No date range in cache")
        return

    # Create race results at different time points
    # We'll use the activity history to create plausible race results

    race_results = []

    # Helper function to estimate race time based on distance
    def estimate_time(distance_miles, pace_min_per_mile):
        """Estimate race time from distance and pace"""
        # Add race day slowdown factor
        race_pace = pace_min_per_mile * 1.05  # 5% slower than training pace
        return distance_miles * race_pace

    # Get some activities to base estimates on
    activities = cache.get_activities_with_hr(limit=100)

    if len(activities) < 10:
        print(f"WARNING: Only {len(activities)} activities found")

    # Find activities at different time points
    # We'll create race results after periods of training
    time_span = (end_date - start_date).days

    if time_span < 365:
        print(f"WARNING: Activity history only spans {time_span} days")
        print("For best results, use at least 1 year of training data")

    # Create races at quarterly intervals
    num_races = min(10, max(3, time_span // 90))  # 3-10 races depending on history

    print(f"\nCreating {num_races} synthetic race results...")

    race_distances = [
        (26.2, "Marathon"),
        (13.1, "Half Marathon"),
        (6.2, "10K"),
        (3.1, "5K"),
    ]

    runner_profile = {
        "age": 35,
        "sex": "M",
        "max_hr": 185,
        "experience_years": 8,
        "resting_hr": 52
    }

    for i in range(num_races):
        # Spread races across timeline
        race_date = start_date + timedelta(days=(time_span // num_races) * (i + 1))

        # Vary race distance
        distance, distance_name = race_distances[i % len(race_distances)]

        # Find activities around this time to estimate pace
        race_date_cutoff = race_date - timedelta(days=90)
        recent_activities = [
            a for a in activities
            if race_date_cutoff <= datetime.fromisoformat(a['activity_date']) < race_date
        ]

        if recent_activities:
            # Estimate race time from recent training pace
            avg_pace = sum(a.get('avg_pace', 8.0) for a in recent_activities) / len(recent_activities)
            race_time = estimate_time(distance, avg_pace)
        else:
            # Default estimates
            if distance == 26.2:
                race_time = 210  # 3:30:00 marathon
            elif distance == 13.1:
                race_time = 95  # 1:35:00 half
            elif distance == 6.2:
                race_time = 42  # 42:00 10K
            else:
                race_time = 20  # 20:00 5K

        # Add some variation
        import random
        race_time *= random.uniform(0.95, 1.05)

        race_result = {
            "race_id": f"synthetic_{distance_name.lower().replace(' ', '_')}_{race_date.strftime('%Y%m%d')}",
            "runner_id": "synthetic_runner",
            "race_date": race_date.strftime('%Y-%m-%d'),
            "race_distance_miles": distance,
            "actual_time_minutes": round(race_time, 1),
            **runner_profile,
            "lookback_weeks": 12 if distance >= 13.1 else 8
        }

        race_results.append(race_result)

        # Format time for display
        hours = int(race_time // 60)
        mins = int(race_time % 60)
        time_str = f"{hours}:{mins:02d}" if hours > 0 else f"{mins}:00"

        print(f"  {i+1}. {distance_name:<15} on {race_date.strftime('%Y-%m-%d')}  ({time_str})")

    # Save to JSON
    output_file = Path("synthetic_race_results.json")

    with open(output_file, 'w') as f:
        json.dump(race_results, f, indent=2)

    print(f"\n✓ Saved {len(race_results)} race results to {output_file}")
    print(f"\nNext step: python train_race_model.py {output_file}")

    return output_file


if __name__ == '__main__':
    create_synthetic_race_results()
