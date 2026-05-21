#!/usr/bin/env python3
"""
Check Azeem's marathons for bonking
"""

from pathlib import Path
from datetime import datetime
from activity_cache import ActivityCache
from extract_all_marathons import detect_bonking

def main():
    print("="*80)
    print("Checking Azeem's Marathons for Bonking")
    print("="*80)

    # Load Azeem's cache
    azeem_cache = ActivityCache(
        cache_dir=Path.home() / ".strava_guru_cache" / "azeem"
    )

    # Azeem's activities directory
    azeem_activities_dir = Path('/Users/osman/Downloads/export_52983191_azeem/activities')

    # Azeem's marathons
    marathons = [
        {'date': datetime(2022, 11, 6), 'name': 'Stoney Creek Marathon', 'time': '3:27'},
        {'date': datetime(2023, 10, 8), 'name': 'Chicago Marathon 2023', 'time': '3:38'},
        {'date': datetime(2024, 2, 10), 'name': 'Mesa Marathon', 'time': '3:23'},
        {'date': datetime(2025, 10, 4), 'name': 'Chicago Marathon 2025', 'time': '3:26'},
        {'date': datetime(2026, 1, 11), 'name': 'Houston Marathon 2026', 'time': '3:22'}
    ]

    print(f"\nAnalyzing {len(marathons)} marathons...\n")

    bonked_races = []
    clean_races = []

    for i, marathon in enumerate(marathons, 1):
        race_name = marathon['name']
        race_date = marathon['date']
        time = marathon['time']

        print(f"{i}. {race_name:30s} ({race_date.date()}) - {time}")

        # Detect bonking
        analysis = detect_bonking(
            azeem_cache,
            race_date,
            threshold_percent=15.0,
            activities_dir=azeem_activities_dir
        )

        if analysis.first_half_pace == 0:
            print(f"   ⚠ {analysis.explanation}")
        elif analysis.is_bonk:
            print(f"   🚨 BONK DETECTED: {analysis.explanation}")
            bonked_races.append(marathon)
        else:
            print(f"   ✓ Clean: {analysis.explanation}")
            clean_races.append(marathon)
        print()

    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"  Clean races: {len(clean_races)}")
    print(f"  Bonked races: {len(bonked_races)}")

    if bonked_races:
        print(f"\n{'='*80}")
        print("Bonked Races (Should Filter Out)")
        print(f"{'='*80}")
        for m in bonked_races:
            print(f"  • {m['name']} ({m['date'].date()}) - {m['time']}")
    else:
        print(f"\n✓ All races look clean!")

if __name__ == '__main__':
    main()
