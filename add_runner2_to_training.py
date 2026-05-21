#!/usr/bin/env python3
"""
Add runner 2's marathons to the training dataset
"""

import json
from pathlib import Path
from datetime import datetime
from activity_cache import ActivityCache, build_cache
from race_prediction.data_collector import RaceDataCollector
from feature_engineering import RunnerContext

def main():
    print("Setting up activity cache for runner 2...")
    # Point to runner 2's activities directory
    activities_dir = Path('/Users/osman/Downloads/export_1884062/activities')
    runner2_cache = ActivityCache(
        cache_dir=activities_dir
    )

    # Build cache from activities
    print("\nScanning and caching activities...")
    build_cache(activities_dir, runner2_cache)

    stats = runner2_cache.get_stats()
    print(f"\nCached {stats.get('total_activities', 0)} activities for runner 2")

    # Load runner 2's marathon data
    print("\nLoading runner 2's marathon results...")
    with open('runner2_marathons.json', 'r') as f:
        runner2_races = json.load(f)

    print(f"Found {len(runner2_races)} marathons for runner 2")

    # Initialize data collector with runner 2's cache
    print("\nExtracting features for runner 2's marathons...")
    collector = RaceDataCollector(cache=runner2_cache)

    # Add each race
    for i, race in enumerate(runner2_races, 1):
        print(f"\n{i}/{len(runner2_races)}: {race['_race_name']} ({race['race_date']})")

        runner_context = RunnerContext(
            age=race['age'],
            sex=race['sex'],
            max_hr=race['max_hr'],
            experience_years=race.get('experience_years'),
            resting_hr=race.get('resting_hr')
        )

        try:
            race_date = datetime.fromisoformat(race['race_date'])

            collector.add_race_result(
                race_id=race['race_id'],
                runner_id=race['runner_id'],
                race_date=race_date,
                race_distance_miles=race['race_distance_miles'],
                actual_time_minutes=race['actual_time_minutes'],
                runner_context=runner_context,
                lookback_weeks=race['lookback_weeks']
            )

            print(f"  ✓ Features extracted successfully")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Save runner 2's dataset
    runner2_dataset_path = 'race_data/runner2_dataset.json'
    collector.save_dataset(runner2_dataset_path)
    print(f"\n✓ Saved runner 2's dataset to {runner2_dataset_path}")

    # Now combine with your dataset
    print("\n" + "="*80)
    print("Combining datasets...")

    # Load your dataset
    with open('race_data/race_dataset.json', 'r') as f:
        your_dataset = json.load(f)

    print(f"Your marathons: {len(your_dataset)}")
    print(f"Runner 2's marathons: {len(collector.race_results)}")

    # Combine
    combined_dataset = your_dataset + collector.race_results
    print(f"Combined total: {len(combined_dataset)} marathons")

    # Save combined dataset
    combined_path = 'race_data/combined_dataset.json'
    with open(combined_path, 'w') as f:
        json.dump(combined_dataset, f, indent=2)

    print(f"\n✓ Saved combined dataset to {combined_path}")

    # Show summary
    print("\n" + "="*80)
    print("Dataset Summary:")
    print(f"  Total races: {len(combined_dataset)}")

    # Calculate time range
    times = [r['actual_time_minutes'] for r in combined_dataset]
    print(f"  Time range: {min(times):.1f} - {max(times):.1f} minutes")
    print(f"              ({min(times)//60}:{int(min(times)%60):02d} - {max(times)//60}:{int(max(times)%60):02d})")

    # Count by runner
    runner_counts = {}
    for r in combined_dataset:
        runner_id = r['runner_id']
        runner_counts[runner_id] = runner_counts.get(runner_id, 0) + 1

    print(f"\n  By runner:")
    for runner_id, count in sorted(runner_counts.items()):
        print(f"    {runner_id}: {count} races")

    print("\nReady to retrain the model with:")
    print(f"  python train_race_model.py --dataset {combined_path}")

if __name__ == '__main__':
    main()
