#!/usr/bin/env python3
"""
Extract full 41 features for Salman's marathons
"""

import json
from pathlib import Path
from datetime import datetime
from activity_cache import ActivityCache
from race_prediction.data_collector import RaceDataCollector
from feature_engineering import RunnerContext

def main():
    print("="*80)
    print("Extracting Full 41 Features for Salman's Marathons")
    print("="*80)

    # Setup Salman's cache
    salman_cache_dir = Path.home() / ".strava_guru_cache" / "salman"
    salman_cache = ActivityCache(cache_dir=salman_cache_dir)

    # Load Salman's marathon data
    with open('runner2_marathons.json', 'r') as f:
        marathons = json.load(f)

    print(f"\nFound {len(marathons)} marathons to process")
    print(f"Cache has {salman_cache.get_stats().get('total_activities', 0)} activities\n")

    # Initialize data collector
    collector = RaceDataCollector(cache=salman_cache)

    # Process each marathon
    for i, marathon in enumerate(marathons, 1):
        print(f"\n{i}/{len(marathons)}: {marathon.get('_race_name', marathon['race_date'])}")
        print(f"  Date: {marathon['race_date']}")
        print(f"  Time: {int(marathon['actual_time_minutes']//60)}:{int(marathon['actual_time_minutes']%60):02d}")

        runner_context = RunnerContext(
            age=marathon['age'],
            sex=marathon['sex'],
            max_hr=marathon['max_hr'],
            experience_years=marathon.get('experience_years'),
            resting_hr=marathon.get('resting_hr')
        )

        try:
            race_date = datetime.fromisoformat(marathon['race_date'])

            collector.add_race_result(
                race_id=marathon['race_id'],
                runner_id=marathon['runner_id'],
                race_date=race_date,
                race_distance_miles=marathon['race_distance_miles'],
                actual_time_minutes=marathon['actual_time_minutes'],
                runner_context=runner_context,
                lookback_weeks=marathon['lookback_weeks']
            )

            # Count features extracted
            if collector.race_results:
                features = collector.race_results[-1].get('features', {})
                non_null_features = sum(1 for v in features.values() if v is not None and v != 0)
                print(f"  ✓ Extracted {non_null_features} features")

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")

    # Save Salman's dataset
    output_path = 'salman_full_features_dataset.json'
    collector.save_dataset(output_path)

    print("\n" + "="*80)
    print(f"✓ Successfully extracted features for {len(collector.race_results)} marathons")
    print(f"✓ Saved to: {output_path}")
    print("="*80)

    # Show summary
    if collector.race_results:
        times = [r['actual_time_minutes'] for r in collector.race_results]
        print(f"\nTime range: {min(times)//60}:{int(min(times)%60):02d} - {max(times)//60}:{int(max(times)%60):02d}")

if __name__ == '__main__':
    main()
