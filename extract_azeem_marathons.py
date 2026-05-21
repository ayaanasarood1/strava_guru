#!/usr/bin/env python3
"""
Extract features for all Azeem marathons
"""

import json
from pathlib import Path
from datetime import datetime
from feature_engineering.feature_extractor import TrainingFeatureExtractor
from feature_engineering.runner_context import RunnerContext
from activity_cache import ActivityCache

def main():
    print("="*80)
    print("Extracting Features for Azeem's Marathons")
    print("="*80)

    # Load Azeem's cache
    azeem_cache = ActivityCache(
        cache_dir=Path.home() / ".strava_guru_cache" / "azeem"
    )

    # Runner context for Azeem
    runner_context = RunnerContext(
        age=32,  # Estimate
        sex='M',
        max_hr=185,  # Estimate
        resting_hr=50,
        experience_years=6  # Based on first marathon in 2022
    )

    # Azeem's marathons
    marathons = [
        {
            'race_date': datetime(2022, 11, 6),
            'race_name': 'Stoney Creek Marathon',
            'actual_time_minutes': 207.0,  # 3:27
            'distance_miles': 26.34
        },
        {
            'race_date': datetime(2023, 10, 8),
            'race_name': 'Chicago Marathon 2023',
            'actual_time_minutes': 218.0,  # 3:38
            'distance_miles': 26.38
        },
        {
            'race_date': datetime(2024, 2, 10),
            'race_name': 'Mesa Marathon',
            'actual_time_minutes': 203.0,  # 3:23
            'distance_miles': 26.43
        },
        {
            'race_date': datetime(2025, 10, 4),
            'race_name': 'Chicago Marathon 2025',
            'actual_time_minutes': 206.0,  # 3:26
            'distance_miles': 26.31
        },
        {
            'race_date': datetime(2026, 1, 11),
            'race_name': 'Houston Marathon 2026',
            'actual_time_minutes': 202.0,  # 3:22
            'distance_miles': 26.35
        }
    ]

    print(f"\nExtracting features for {len(marathons)} marathons")
    print()

    # Create extractor
    extractor = TrainingFeatureExtractor(azeem_cache)

    # Extract features for each marathon
    azeem_races = []

    for i, marathon in enumerate(marathons, 1):
        race_date = marathon['race_date']
        race_name = marathon['race_name']
        actual_time = marathon['actual_time_minutes']

        print(f"{i}. {race_name} ({race_date.date()})")
        print(f"   Time: {int(actual_time//60)}:{int(actual_time%60):02d}")

        try:
            # Extract features
            features = extractor.extract_features(
                runner_id="runner_3",  # Azeem
                race_date=race_date,
                lookback_weeks=12,
                race_distance_miles=marathon['distance_miles'],
                runner_context=runner_context
            )

            # Show key features
            print(f"   Training: {features.total_weekly_mileage:.1f} mi/week")
            print(f"   Peak week: {features.peak_weekly_mileage:.1f} miles")
            print(f"   Long run: {features.long_run_distance:.1f} miles")

            # Create race record
            race_id = f"marathon_{race_date.strftime('%Y%m%d')}"

            race_record = {
                'race_id': race_id,
                'runner_id': 'runner_3',
                'race_date': race_date.isoformat(),
                'actual_time_minutes': actual_time,
                'features': features.to_dict(),
                '_race_name': race_name
            }

            azeem_races.append(race_record)
            print(f"   ✓ Extracted {len([k for k, v in features.to_dict().items() if v is not None and v != 0])} features")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        print()

    # Load existing dataset
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'

    if dataset_path.exists():
        with open(dataset_path, 'r') as f:
            all_races = json.load(f)
        print(f"Loaded existing dataset: {len(all_races)} races")
    else:
        all_races = []
        print("Creating new dataset")

    # Add Azeem's races
    all_races.extend(azeem_races)

    # Save
    with open(dataset_path, 'w') as f:
        json.dump(all_races, f, indent=2)

    print(f"✓ Saved {len(all_races)} total races to {dataset_path}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Azeem's marathons: {len(azeem_races)}")
    print(f"Total dataset: {len(all_races)} races")

    # Count by runner
    user_count = len([r for r in all_races if r['runner_id'] == 'my_runner'])
    salman_count = len([r for r in all_races if r['runner_id'] == 'runner_2'])
    azeem_count = len([r for r in all_races if r['runner_id'] == 'runner_3'])

    print(f"\nBreakdown:")
    print(f"  You: {user_count} marathons")
    print(f"  Salman: {salman_count} marathons")
    print(f"  Azeem: {azeem_count} marathons")
    print("="*80)

if __name__ == '__main__':
    main()
