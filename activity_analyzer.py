#!/usr/bin/env python3
"""
Strava Activity Analyzer
Reads .fit and .gpx files and computes detailed activity statistics
including pace analysis, splits, laps, elevation, and more.
"""

import gzip
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import math
import json

try:
    from fitparse import FitFile
    import gpxpy
    import gpxpy.gpx
except ImportError:
    print("Required packages not installed. Install with:")
    print("pip install fitparse gpxpy")
    exit(1)


@dataclass
class TrackPoint:
    """Represents a single GPS track point"""
    timestamp: datetime
    latitude: float
    longitude: float
    elevation: Optional[float]  # meters
    distance: float = 0.0  # cumulative distance in meters
    heart_rate: Optional[int] = None
    cadence: Optional[int] = None
    speed: Optional[float] = None  # m/s


@dataclass
class Lap:
    """Represents a lap or mile split"""
    number: int
    distance: float  # meters
    duration: timedelta
    pace: float  # min/mile
    gap: float  # grade adjusted pace min/mile
    elevation_gain: float  # meters
    elevation_loss: float  # meters
    avg_heart_rate: Optional[int] = None
    max_heart_rate: Optional[int] = None


@dataclass
class ActivityStats:
    """Complete activity statistics"""
    name: str
    activity_type: str
    start_time: datetime

    # Basic metrics
    total_distance: float  # meters
    total_time: timedelta
    moving_time: timedelta
    elapsed_time: timedelta

    # Pace
    avg_pace: float  # min/mile
    avg_gap: float  # grade adjusted pace min/mile

    # Elevation
    elevation_gain: float  # meters
    elevation_loss: float  # meters
    min_elevation: float
    max_elevation: float

    # Heart rate
    avg_heart_rate: Optional[int] = None
    max_heart_rate: Optional[int] = None

    # Calories
    calories: Optional[int] = None

    # Track data
    track_points: List[TrackPoint] = None
    laps: List[Lap] = None
    mile_splits: List[Lap] = None


class ActivityAnalyzer:
    """Analyzes Strava activity files"""

    METERS_TO_MILES = 0.000621371
    METERS_TO_FEET = 3.28084

    def __init__(self):
        self.track_points: List[TrackPoint] = []

    def analyze_file(self, file_path: Path) -> Optional[ActivityStats]:
        """Analyze a .fit or .gpx file"""
        if file_path.suffix == '.gz':
            # Handle gzipped files
            actual_path = file_path.stem
            if actual_path.endswith('.fit'):
                return self._analyze_fit(file_path, is_gzipped=True)
            elif actual_path.endswith('.gpx'):
                return self._analyze_gpx(file_path, is_gzipped=True)
        elif file_path.suffix == '.fit':
            return self._analyze_fit(file_path, is_gzipped=False)
        elif file_path.suffix == '.gpx':
            return self._analyze_gpx(file_path, is_gzipped=False)
        else:
            print(f"Unsupported file type: {file_path}")
            return None

    def _analyze_fit(self, file_path: Path, is_gzipped: bool = False) -> Optional[ActivityStats]:
        """Analyze a FIT file"""
        try:
            # Try with lenient parsing first (skip CRC checks for corrupted files)
            if is_gzipped:
                with gzip.open(file_path, 'rb') as f:
                    fitfile = FitFile(f, check_crc=False)
                    return self._parse_fit_file(fitfile)
            else:
                fitfile = FitFile(str(file_path), check_crc=False)
                return self._parse_fit_file(fitfile)
        except Exception as e:
            # If lenient parsing fails, file is likely too corrupted
            # Silently skip to avoid verbose output during batch processing
            return None

    def _parse_fit_file(self, fitfile) -> Optional[ActivityStats]:
        """Parse FIT file data"""
        self.track_points = []

        # Get session data (summary)
        session_data = {}
        for record in fitfile.get_messages('session'):
            for field in record:
                session_data[field.name] = field.value

        # Get lap data
        laps_data = []
        for record in fitfile.get_messages('lap'):
            lap_info = {}
            for field in record:
                lap_info[field.name] = field.value
            laps_data.append(lap_info)

        # Get record data (track points)
        cumulative_distance = 0.0
        for record in fitfile.get_messages('record'):
            point_data = {}
            for field in record:
                point_data[field.name] = field.value

            if 'position_lat' in point_data and 'position_long' in point_data:
                # Convert semicircles to degrees
                lat = point_data['position_lat'] * (180 / 2**31) if point_data['position_lat'] else None
                lon = point_data['position_long'] * (180 / 2**31) if point_data['position_long'] else None

                if lat and lon:
                    point = TrackPoint(
                        timestamp=point_data.get('timestamp'),
                        latitude=lat,
                        longitude=lon,
                        elevation=point_data.get('altitude') or 0,
                        distance=point_data.get('distance', cumulative_distance),
                        heart_rate=point_data.get('heart_rate'),
                        cadence=point_data.get('cadence'),
                        speed=point_data.get('speed')
                    )
                    self.track_points.append(point)
                    if 'distance' in point_data:
                        cumulative_distance = point_data['distance']

        if not self.track_points:
            return None

        # Calculate statistics
        return self._calculate_stats(session_data, laps_data)

    def _analyze_gpx(self, file_path: Path, is_gzipped: bool = False) -> Optional[ActivityStats]:
        """Analyze a GPX file"""
        try:
            if is_gzipped:
                with gzip.open(file_path, 'rt') as f:
                    gpx = gpxpy.parse(f)
            else:
                with open(file_path, 'r') as f:
                    gpx = gpxpy.parse(f)

            self.track_points = []
            cumulative_distance = 0.0

            for track in gpx.tracks:
                for segment in track.segments:
                    prev_point = None
                    for point in segment.points:
                        if prev_point:
                            distance = self._haversine_distance(
                                prev_point.latitude, prev_point.longitude,
                                point.latitude, point.longitude
                            )
                            cumulative_distance += distance

                        track_point = TrackPoint(
                            timestamp=point.time,
                            latitude=point.latitude,
                            longitude=point.longitude,
                            elevation=point.elevation if point.elevation else 0,
                            distance=cumulative_distance
                        )

                        # Extract heart rate from extensions if available
                        if hasattr(point, 'extensions') and point.extensions:
                            for ext in point.extensions:
                                if 'hr' in ext.tag.lower():
                                    try:
                                        track_point.heart_rate = int(ext.text)
                                    except:
                                        pass

                        self.track_points.append(track_point)
                        prev_point = point

            if not self.track_points:
                return None

            return self._calculate_stats({}, [])

        except Exception as e:
            print(f"Error parsing GPX file {file_path}: {e}")
            return None

    def _calculate_stats(self, session_data: dict, laps_data: list) -> ActivityStats:
        """Calculate comprehensive activity statistics"""

        if not self.track_points:
            return None

        # Basic info
        start_time = self.track_points[0].timestamp
        end_time = self.track_points[-1].timestamp
        total_time = end_time - start_time

        # Distance - calculate if not available
        if self.track_points[-1].distance is not None and self.track_points[-1].distance > 0:
            total_distance = self.track_points[-1].distance
        else:
            # Calculate total distance from GPS coordinates
            total_distance = 0.0
            for i in range(1, len(self.track_points)):
                total_distance += self._haversine_distance(
                    self.track_points[i-1].latitude, self.track_points[i-1].longitude,
                    self.track_points[i].latitude, self.track_points[i].longitude
                )

        # Moving time (exclude stopped periods)
        moving_time = self._calculate_moving_time()

        # Elevation
        elevation_gain, elevation_loss = self._calculate_elevation_change()
        elevations = [p.elevation for p in self.track_points if p.elevation is not None]
        min_elevation = min(elevations) if elevations else 0
        max_elevation = max(elevations) if elevations else 0

        # Pace (minutes per mile)
        avg_pace = self._calculate_pace(total_distance, moving_time.total_seconds())

        # Grade Adjusted Pace
        avg_gap = self._calculate_gap()

        # Heart rate
        hr_values = [p.heart_rate for p in self.track_points if p.heart_rate]
        avg_hr = int(sum(hr_values) / len(hr_values)) if hr_values else None
        max_hr = max(hr_values) if hr_values else None

        # Calories (rough estimate based on distance and weight)
        # ~100 calories per mile for running
        calories = int(total_distance * self.METERS_TO_MILES * 100) if total_distance else None

        # Generate mile splits
        mile_splits = self._generate_mile_splits()

        # Parse laps if available
        laps = self._parse_laps(laps_data)

        return ActivityStats(
            name=session_data.get('sport', 'Unknown Activity'),
            activity_type=session_data.get('sport', 'Run'),
            start_time=start_time,
            total_distance=total_distance,
            total_time=total_time,
            moving_time=moving_time,
            elapsed_time=total_time,
            avg_pace=avg_pace,
            avg_gap=avg_gap,
            elevation_gain=elevation_gain,
            elevation_loss=elevation_loss,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            avg_heart_rate=avg_hr,
            max_heart_rate=max_hr,
            calories=calories,
            track_points=self.track_points,
            laps=laps,
            mile_splits=mile_splits
        )

    def _calculate_moving_time(self) -> timedelta:
        """Calculate moving time excluding stops"""
        moving_seconds = 0.0
        speed_threshold = 0.5  # m/s, below this is considered stopped

        for i in range(1, len(self.track_points)):
            p1 = self.track_points[i-1]
            p2 = self.track_points[i]

            dt = (p2.timestamp - p1.timestamp).total_seconds()

            # Calculate distance between points
            if p1.distance is not None and p2.distance is not None:
                distance = p2.distance - p1.distance
            else:
                # Fall back to GPS distance calculation
                distance = self._haversine_distance(
                    p1.latitude, p1.longitude,
                    p2.latitude, p2.longitude
                )

            if dt > 0 and distance >= 0:
                speed = distance / dt
                if speed > speed_threshold:
                    moving_seconds += dt

        return timedelta(seconds=moving_seconds)

    def _calculate_elevation_change(self) -> Tuple[float, float]:
        """Calculate elevation gain and loss in meters"""
        gain = 0.0
        loss = 0.0

        for i in range(1, len(self.track_points)):
            elev1 = self.track_points[i-1].elevation
            elev2 = self.track_points[i].elevation

            # Skip if either elevation is None
            if elev1 is None or elev2 is None:
                continue

            diff = elev2 - elev1
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)

        return gain, loss

    def _calculate_pace(self, distance: float, time_seconds: float) -> float:
        """Calculate pace in minutes per mile"""
        if distance == 0:
            return 0.0

        miles = distance * self.METERS_TO_MILES
        minutes = time_seconds / 60
        return minutes / miles if miles > 0 else 0.0

    def _calculate_gap(self) -> float:
        """Calculate Grade Adjusted Pace (GAP)"""
        # GAP adjusts pace based on elevation grade
        # For each segment, adjust time based on grade

        total_adjusted_time = 0.0
        total_distance = 0.0

        for i in range(1, len(self.track_points)):
            p1 = self.track_points[i-1]
            p2 = self.track_points[i]

            # Calculate distance between points
            if p1.distance is not None and p2.distance is not None:
                distance = p2.distance - p1.distance
            else:
                distance = self._haversine_distance(
                    p1.latitude, p1.longitude,
                    p2.latitude, p2.longitude
                )

            time_seconds = (p2.timestamp - p1.timestamp).total_seconds()

            if distance > 0 and time_seconds > 0:
                total_distance += distance

                # Calculate elevation change if available
                if p1.elevation is not None and p2.elevation is not None:
                    elevation_change = p2.elevation - p1.elevation
                    grade = elevation_change / distance
                    # Adjustment factor based on grade (simplified model)
                    # Uphill: add time, Downhill: subtract time
                    adjustment = 1.0 + (grade * 10)  # Rough approximation
                    adjusted_time = time_seconds * adjustment
                else:
                    adjusted_time = time_seconds

                total_adjusted_time += adjusted_time

        return self._calculate_pace(total_distance, total_adjusted_time)

    def _generate_mile_splits(self) -> List[Lap]:
        """Generate splits for each mile"""
        splits = []
        mile_meters = 1609.34  # meters in a mile

        current_mile = 1
        mile_start_idx = 0
        last_mile_end_distance = 0.0

        # Ensure we have distance data, calculate if needed
        if self.track_points[0].distance is None:
            cumulative_dist = 0.0
            for i, point in enumerate(self.track_points):
                if i > 0:
                    cumulative_dist += self._haversine_distance(
                        self.track_points[i-1].latitude, self.track_points[i-1].longitude,
                        point.latitude, point.longitude
                    )
                point.distance = cumulative_dist

        for i, point in enumerate(self.track_points):
            if point.distance is not None and point.distance >= current_mile * mile_meters:
                # Found end of current mile
                lap = self._calculate_lap_stats(
                    current_mile,
                    mile_start_idx,
                    i,
                    last_mile_end_distance,
                    current_mile * mile_meters
                )
                splits.append(lap)

                last_mile_end_distance = current_mile * mile_meters
                current_mile += 1
                mile_start_idx = i

        # Add final partial mile if exists
        if mile_start_idx < len(self.track_points) - 1:
            final_distance = self.track_points[-1].distance if self.track_points[-1].distance else 0
            # Only add if there's meaningful distance left
            if final_distance > last_mile_end_distance + 10:  # At least 10 meters
                lap = self._calculate_lap_stats(
                    current_mile,
                    mile_start_idx,
                    len(self.track_points) - 1,
                    last_mile_end_distance,
                    final_distance
                )
                splits.append(lap)

        return splits

    def _calculate_lap_stats(self, lap_number: int, start_idx: int, end_idx: int,
                            start_distance: float, end_distance: float) -> Lap:
        """Calculate statistics for a lap/split"""

        points = self.track_points[start_idx:end_idx+1]

        duration = points[-1].timestamp - points[0].timestamp
        distance = end_distance - start_distance

        # Pace
        pace = self._calculate_pace(distance, duration.total_seconds())

        # Elevation
        elev_gain = 0.0
        elev_loss = 0.0
        for i in range(1, len(points)):
            elev1 = points[i-1].elevation
            elev2 = points[i].elevation

            # Skip if either elevation is None
            if elev1 is None or elev2 is None:
                continue

            diff = elev2 - elev1
            if diff > 0:
                elev_gain += diff
            else:
                elev_loss += abs(diff)

        # GAP for this lap
        gap = self._calculate_lap_gap(points, distance, duration.total_seconds())

        # Heart rate
        hr_values = [p.heart_rate for p in points if p.heart_rate]
        avg_hr = int(sum(hr_values) / len(hr_values)) if hr_values else None
        max_hr = max(hr_values) if hr_values else None

        return Lap(
            number=lap_number,
            distance=distance,
            duration=duration,
            pace=pace,
            gap=gap,
            elevation_gain=elev_gain,
            elevation_loss=elev_loss,
            avg_heart_rate=avg_hr,
            max_heart_rate=max_hr
        )

    def _calculate_lap_gap(self, points: List[TrackPoint], distance: float, time_seconds: float) -> float:
        """Calculate GAP for a specific lap"""
        if len(points) < 2:
            return self._calculate_pace(distance, time_seconds)

        # Check if both elevation values exist
        if points[0].elevation is not None and points[-1].elevation is not None:
            total_elevation = points[-1].elevation - points[0].elevation
            if distance > 0:
                grade = total_elevation / distance
                adjustment = 1.0 + (grade * 10)
                adjusted_time = time_seconds * adjustment
                return self._calculate_pace(distance, adjusted_time)

        return self._calculate_pace(distance, time_seconds)

    def _parse_laps(self, laps_data: list) -> List[Lap]:
        """Parse lap data from FIT file"""
        if not laps_data:
            return []

        laps = []
        for i, lap_info in enumerate(laps_data, 1):
            # Extract lap data
            distance = lap_info.get('total_distance', 0)
            if distance == 0:
                continue

            # Duration - use timer time (moving time, excludes pauses) like Strava
            total_time = lap_info.get('total_timer_time', 0)
            if total_time == 0:
                total_time = lap_info.get('total_elapsed_time', 0)

            duration = timedelta(seconds=total_time)

            # Pace
            pace = self._calculate_pace(distance, total_time)

            # Elevation
            elev_gain = lap_info.get('total_ascent', 0)
            elev_loss = lap_info.get('total_descent', 0)

            # GAP - for laps, we can use the lap's avg grade if available
            if 'avg_grade' in lap_info and lap_info['avg_grade']:
                grade = lap_info['avg_grade'] / 100  # Convert percentage to decimal
                adjustment = 1.0 + (grade * 10)
                adjusted_time = total_time * adjustment
                gap = self._calculate_pace(distance, adjusted_time)
            else:
                # Fall back to elevation-based calculation
                if elev_gain and elev_gain > 0 and distance > 0:
                    grade = (elev_gain - elev_loss) / distance
                    adjustment = 1.0 + (grade * 10)
                    adjusted_time = total_time * adjustment
                    gap = self._calculate_pace(distance, adjusted_time)
                else:
                    gap = pace

            # Heart rate
            avg_hr = lap_info.get('avg_heart_rate')
            max_hr = lap_info.get('max_heart_rate')

            lap = Lap(
                number=i,
                distance=distance,
                duration=duration,
                pace=pace,
                gap=gap,
                elevation_gain=elev_gain if elev_gain else 0,
                elevation_loss=elev_loss if elev_loss else 0,
                avg_heart_rate=int(avg_hr) if avg_hr else None,
                max_heart_rate=int(max_hr) if max_hr else None
            )
            laps.append(lap)

        return laps

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c


def format_pace(pace: float) -> str:
    """Format pace as MM:SS/mi"""
    if pace == 0 or math.isnan(pace) or math.isinf(pace):
        return "--:--"

    minutes = int(pace)
    seconds = int((pace - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def format_time(td: timedelta) -> str:
    """Format timedelta as H:MM:SS"""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


def print_activity_summary(stats: ActivityStats):
    """Print a formatted summary of activity statistics"""
    print("\n" + "="*80)
    print(f"ACTIVITY SUMMARY")
    print("="*80)
    print(f"Activity Type: {stats.activity_type}")
    print(f"Date: {stats.start_time.strftime('%A, %B %d, %Y at %I:%M %p')}")
    print()

    print(f"Distance:       {stats.total_distance * ActivityAnalyzer.METERS_TO_MILES:.2f} mi")
    print(f"Moving Time:    {format_time(stats.moving_time)}")
    print(f"Elapsed Time:   {format_time(stats.elapsed_time)}")
    print(f"Pace:           {format_pace(stats.avg_pace)} /mi")
    print(f"GAP:            {format_pace(stats.avg_gap)} /mi")
    print()

    print(f"Elevation Gain: {stats.elevation_gain * ActivityAnalyzer.METERS_TO_FEET:.0f} ft")
    print(f"Elevation Loss: {stats.elevation_loss * ActivityAnalyzer.METERS_TO_FEET:.0f} ft")
    print(f"Elev Range:     {stats.min_elevation * ActivityAnalyzer.METERS_TO_FEET:.0f} - "
          f"{stats.max_elevation * ActivityAnalyzer.METERS_TO_FEET:.0f} ft")
    print()

    if stats.avg_heart_rate:
        print(f"Avg Heart Rate: {stats.avg_heart_rate} bpm")
        print(f"Max Heart Rate: {stats.max_heart_rate} bpm")
        print()

    if stats.calories:
        print(f"Calories:       {stats.calories}")
        print()

    # Laps (prefer laps over mile splits)
    if stats.laps:
        print("="*80)
        print("LAPS")
        print("="*80)
        print(f"{'Lap':<6} {'Distance':<12} {'Time':<10} {'Pace':<10} {'GAP':<10} {'Elev ↑':<10} {'HR':<8}")
        print("-"*80)

        for lap in stats.laps:
            dist_mi = lap.distance * ActivityAnalyzer.METERS_TO_MILES
            elev_ft = lap.elevation_gain * ActivityAnalyzer.METERS_TO_FEET
            hr_str = f"{lap.avg_heart_rate}" if lap.avg_heart_rate else "--"

            print(f"{lap.number:<6} {dist_mi:.2f} mi     {format_time(lap.duration):<10} "
                  f"{format_pace(lap.pace):<10} {format_pace(lap.gap):<10} {elev_ft:>4.0f} ft    {hr_str:<8}")

    # Mile splits (show if no laps)
    elif stats.mile_splits:
        print("="*80)
        print("MILE SPLITS")
        print("="*80)
        print(f"{'Mile':<6} {'Distance':<12} {'Time':<10} {'Pace':<10} {'GAP':<10} {'Elev ↑':<10} {'HR':<8}")
        print("-"*80)

        for lap in stats.mile_splits:
            dist_mi = lap.distance * ActivityAnalyzer.METERS_TO_MILES
            elev_ft = lap.elevation_gain * ActivityAnalyzer.METERS_TO_FEET
            hr_str = f"{lap.avg_heart_rate}" if lap.avg_heart_rate else "--"

            print(f"{lap.number:<6} {dist_mi:.2f} mi     {format_time(lap.duration):<10} "
                  f"{format_pace(lap.pace):<10} {format_pace(lap.gap):<10} {elev_ft:>4.0f} ft    {hr_str:<8}")

    print("="*80)


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python activity_analyzer.py <activity_file.fit|.gpx>")
        print("\nExample:")
        print("  python activity_analyzer.py ~/Downloads/export_40402578/activities/12345.fit.gz")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Analyzing {file_path.name}...")

    analyzer = ActivityAnalyzer()
    stats = analyzer.analyze_file(file_path)

    if stats:
        print_activity_summary(stats)

        # Optionally save to JSON
        if len(sys.argv) > 2 and sys.argv[2] == '--json':
            output_file = file_path.stem + '_stats.json'
            # Convert to dict for JSON serialization
            # (would need custom serializer for datetime/timedelta)
            print(f"\nStats object available for JSON export")
    else:
        print("Failed to analyze activity file")
        sys.exit(1)


if __name__ == '__main__':
    main()
