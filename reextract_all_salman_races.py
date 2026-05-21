#!/usr/bin/env python3
"""
Re-extract ALL Salman marathons with CSV-based cache
"""

import json
from pathlib import Path
from datetime import datetime
from feature_engineering.feature_extractor import TrainingFeatureExtractor
from feature_engineering.runner_context import RunnerContext
from activity_cache import ActivityCache

def main():
    print("="*80)
    print("Re-extracting ALL Salman Marathons with Corrected Cache")
    print("="*80)

    # Load dataset
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'

    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    # Load Salman's cache
    salman_cache = ActivityCache(
        cache_dir=Path.home() / ".strava_guru_cache" / "salman"
    )

    # Runner context
    runner_context = RunnerContext(
        age=30,
        sex='M',
        max_hr=190,
        resting_hr=50,
        experience_years=10
    )

    # Find all Salman races
    salman_races = [r for r in all_races if r['runner_id'] == 'runner_2']
    salman_races.sort(key=lambda x: x['race_date'])

    print(f"\nFound {len(salman_races)} Salman marathons")
    print()

    extractor = TrainingFeatureExtractor(salman_cache)
    updated_count = 0

    for i, race in enumerate(salman_races, 1):
        race_date_str = race['race_date']
        race_date = datetime.fromisoformat(race_date_str.replace('T00:00:00', ''))
        race_name = race.get('_race_name', race_date_str[:10])
        actual_time = race['actual_time_minutes']

        old_mileage = race['features'].get('total_weekly_mileage', 0)

        print(f"{i:2d}. {race_name:30s} ({race_date.date()})")
        print(f"    Time: {int(actual_time//60)}:{int(actual_time%60):02d}")
        print(f"    Old features: {old_mileage:.1f} mi/week")

        try:
            # Extract fresh features
            features = extractor.extract_features(
                runner_id="runner_2",
                race_date=race_date,
                lookback_weeks=12,
                race_distance_miles=26.2,
                runner_context=runner_context
            )

            new_mileage = features.total_weekly_mileage

            # Update
            race['features'] = features.to_dict()
            updated_count += 1

            print(f"    New features: {new_mileage:.1f} mi/week")

            if abs(new_mileage - old_mileage) > 5.0:
                print(f"    ✓ UPDATED (difference: {new_mileage - old_mileage:+.1f} mi/week)")
            else:
                print(f"    (no significant change)")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")

        print()

    # Save
    print("="*80)
    print(f"Saving updated dataset...")
    with open(dataset_path, 'w') as f:
        json.dump(all_races, f, indent=2)

    print(f"✓ Updated {updated_count}/{len(salman_races)} races")
    print(f"✓ Saved to {dataset_path}")
    print("="*80)

if __name__ == '__main__':
    main()
