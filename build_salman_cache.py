#!/usr/bin/env python3
"""
Build activity cache for Salman's data with full feature extraction
"""

from pathlib import Path
from activity_cache import ActivityCache, build_cache

def main():
    print("Building activity cache for Salman...")
    print("="*80)

    # Setup paths
    activities_dir = Path('/Users/osman/Downloads/export_1884062_salman/activities')
    cache_dir = Path.home() / ".strava_guru_cache" / "salman"

    if not activities_dir.exists():
        print(f"Error: Activities directory not found: {activities_dir}")
        return

    print(f"Activities directory: {activities_dir}")
    print(f"Cache directory: {cache_dir}")
    print()

    # Create cache
    cache = ActivityCache(cache_dir=cache_dir)

    # Build cache from all FIT files
    print("Processing all FIT files...")
    print("This will take a while - there are 2,748 files to process")
    print("(Corrupted files will be skipped automatically)")
    print()

    build_cache(activities_dir, cache)

    # Show stats
    stats = cache.get_stats()
    print("\n" + "="*80)
    print("Cache Build Complete!")
    print("="*80)
    print(f"Total activities cached: {stats.get('total_activities', 0)}")
    print(f"Activities with HR data: {stats.get('activities_with_hr', 0)}")
    print(f"Cache location: {cache_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
