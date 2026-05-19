#!/usr/bin/env python3
"""
Activity Cache System
Parse activities once, store in SQLite for fast queries
"""

import sqlite3
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import hashlib

from activity_analyzer import ActivityAnalyzer, ActivityStats


class ActivityCache:
    """Cache processed activity data for fast queries"""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".strava_guru_cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.db_path = self.cache_dir / "activities.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main activities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activities (
                file_name TEXT PRIMARY KEY,
                file_hash TEXT,
                activity_date TIMESTAMP,
                activity_type TEXT,
                distance_meters REAL,
                duration_seconds REAL,
                moving_time_seconds REAL,
                avg_pace REAL,
                avg_gap REAL,
                elevation_gain_meters REAL,
                elevation_loss_meters REAL,
                avg_heart_rate INTEGER,
                max_heart_rate INTEGER,
                calories INTEGER,
                processed_at TIMESTAMP,
                track_points_file TEXT,
                laps_json TEXT
            )
        """)

        # Track points summary table (for fast queries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_point_summary (
                file_name TEXT,
                time_bucket INTEGER,
                avg_hr REAL,
                avg_pace REAL,
                avg_gap REAL,
                avg_elevation REAL,
                point_count INTEGER,
                PRIMARY KEY (file_name, time_bucket)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON activities(activity_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON activities(activity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hr ON activities(avg_heart_rate)")

        conn.commit()
        conn.close()

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def is_cached(self, file_path: Path) -> bool:
        """Check if activity is already cached and unchanged"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        file_name = file_path.name
        file_hash = self._get_file_hash(file_path)

        cursor.execute(
            "SELECT file_hash FROM activities WHERE file_name = ?",
            (file_name,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0] == file_hash

        return False

    def cache_activity(self, file_path: Path, stats: ActivityStats):
        """Cache processed activity data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        file_name = file_path.name
        file_hash = self._get_file_hash(file_path)

        # Store track points separately (pickled for efficiency)
        track_points_file = None
        if stats.track_points:
            track_points_file = f"{file_name}.pkl"
            track_points_path = self.cache_dir / track_points_file
            with open(track_points_path, 'wb') as f:
                pickle.dump(stats.track_points, f)

        # Store laps as JSON
        laps_json = None
        if stats.laps:
            laps_json = json.dumps([{
                'number': lap.number,
                'distance': lap.distance,
                'duration_seconds': lap.duration.total_seconds(),
                'pace': lap.pace,
                'gap': lap.gap,
                'elevation_gain': lap.elevation_gain,
                'avg_hr': lap.avg_heart_rate,
                'max_hr': lap.max_heart_rate
            } for lap in stats.laps])

        # Insert/update main record
        cursor.execute("""
            INSERT OR REPLACE INTO activities (
                file_name, file_hash, activity_date, activity_type,
                distance_meters, duration_seconds, moving_time_seconds,
                avg_pace, avg_gap, elevation_gain_meters, elevation_loss_meters,
                avg_heart_rate, max_heart_rate, calories,
                processed_at, track_points_file, laps_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_name, file_hash, stats.start_time, stats.activity_type,
            stats.total_distance, stats.total_time.total_seconds(),
            stats.moving_time.total_seconds(),
            stats.avg_pace, stats.avg_gap,
            stats.elevation_gain, stats.elevation_loss,
            stats.avg_heart_rate, stats.max_heart_rate, stats.calories,
            datetime.now(), track_points_file, laps_json
        ))

        # Store track point summaries (bucketed by 10-second intervals)
        if stats.track_points:
            self._cache_track_point_summary(cursor, file_name, stats.track_points)

        conn.commit()
        conn.close()

    def _cache_track_point_summary(self, cursor, file_name: str, track_points):
        """Store aggregated track point data for fast queries"""
        # Delete existing summaries
        cursor.execute("DELETE FROM track_point_summary WHERE file_name = ?", (file_name,))

        # Bucket points by 10-second intervals
        buckets = {}
        for point in track_points:
            if not point.timestamp:
                continue

            # Get 10-second bucket
            bucket = int((point.timestamp - track_points[0].timestamp).total_seconds() / 10)

            if bucket not in buckets:
                buckets[bucket] = []

            buckets[bucket].append(point)

        # Insert aggregated data
        for bucket, points in buckets.items():
            hr_values = [p.heart_rate for p in points if p.heart_rate]

            # Calculate pace from speed
            pace_values = []
            for p in points:
                if p.speed and p.speed > 0:
                    pace_values.append(26.8224 / p.speed)  # m/s to min/mile

            cursor.execute("""
                INSERT INTO track_point_summary (
                    file_name, time_bucket, avg_hr, avg_pace, avg_gap,
                    avg_elevation, point_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_name,
                bucket,
                sum(hr_values) / len(hr_values) if hr_values else None,
                sum(pace_values) / len(pace_values) if pace_values else None,
                None,  # GAP calculation would go here
                sum(p.elevation for p in points if p.elevation) / len(points) if points else None,
                len(points)
            ))

    def get_activity(self, file_name: str) -> Optional[Dict]:
        """Get cached activity data"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM activities WHERE file_name = ?", (file_name,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return dict(result)
        return None

    def get_activities_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get activities within date range"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM activities
            WHERE activity_date BETWEEN ? AND ?
            ORDER BY activity_date
        """, (start_date, end_date))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_activities_with_hr(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all activities with heart rate data"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM activities WHERE avg_heart_rate IS NOT NULL ORDER BY activity_date DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_hr_pace_data(self, start_date: Optional[datetime] = None) -> List[tuple]:
        """Get all HR-pace pairs for LT analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if start_date:
            cursor.execute("""
                SELECT avg_hr, avg_pace
                FROM track_point_summary tps
                JOIN activities a ON tps.file_name = a.file_name
                WHERE a.activity_date >= ? AND avg_hr IS NOT NULL AND avg_pace IS NOT NULL
            """, (start_date,))
        else:
            cursor.execute("""
                SELECT avg_hr, avg_pace
                FROM track_point_summary
                WHERE avg_hr IS NOT NULL AND avg_pace IS NOT NULL
            """)

        results = cursor.fetchall()
        conn.close()
        return results

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM activities")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM activities WHERE avg_heart_rate IS NOT NULL")
        with_hr = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(activity_date), MAX(activity_date) FROM activities")
        date_range = cursor.fetchone()

        conn.close()

        return {
            'total_activities': total,
            'with_hr': with_hr,
            'date_range': date_range
        }


def build_cache(activities_dir: Path, cache: ActivityCache, force_rebuild: bool = False):
    """Build or update cache from activities directory"""
    activity_files = sorted(activities_dir.glob("*.fit.gz"))

    print(f"Found {len(activity_files)} activity files")
    print(f"Cache location: {cache.cache_dir}")

    if force_rebuild:
        print("Force rebuild: clearing existing cache...")
        # Clear database
        cache.db_path.unlink(missing_ok=True)
        cache._init_database()

    analyzer = ActivityAnalyzer()
    processed = 0
    skipped = 0
    errors = 0

    for i, file_path in enumerate(activity_files, 1):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(activity_files)} ({i/len(activity_files)*100:.1f}%)")

        # Check if already cached
        if not force_rebuild and cache.is_cached(file_path):
            skipped += 1
            continue

        try:
            stats = analyzer.analyze_file(file_path)
            if stats:
                cache.cache_activity(file_path, stats)
                processed += 1
        except Exception as e:
            errors += 1
            continue

    print(f"\nCache build complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already cached): {skipped}")
    print(f"  Errors: {errors}")

    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Total activities: {stats['total_activities']}")
    print(f"  With HR data: {stats['with_hr']}")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python activity_cache.py <activities_directory> [--rebuild]")
        print("\nExample:")
        print("  python activity_cache.py ~/Downloads/export_40402578/activities/")
        print("  python activity_cache.py ~/Downloads/export_40402578/activities/ --rebuild")
        sys.exit(1)

    activities_dir = Path(sys.argv[1])
    force_rebuild = '--rebuild' in sys.argv

    cache = ActivityCache()
    build_cache(activities_dir, cache, force_rebuild)


if __name__ == '__main__':
    main()
