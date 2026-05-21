"""
Training Intensity Feature Extractor
Computes zone distribution and workout classification
"""

from typing import Dict, List, Optional
import json
import numpy as np

from feature_engineering.utils import meters_to_miles


class TrainingIntensityExtractor:
    """Extract training intensity features"""

    def __init__(self, activity_cache):
        """Initialize with activity cache for track point queries"""
        self.cache = activity_cache

    def extract(
        self,
        activities: List[Dict],
        lt_hr: Optional[float] = None,
        aet_hr: Optional[float] = None,
        max_hr: Optional[float] = None
    ) -> Dict[str, float]:
        """Extract training intensity features

        Args:
            activities: List of activity dictionaries
            lt_hr: Lactate threshold heart rate (bpm)
            aet_hr: Aerobic threshold heart rate (bpm)
            max_hr: Maximum heart rate (bpm)

        Returns:
            Dictionary with intensity features
        """
        if not activities:
            return self._default_features()

        # Define HR zones
        zones = self._define_zones(lt_hr, aet_hr, max_hr)

        # Calculate zone distribution
        zone_distribution = self._calculate_zone_distribution(activities, zones)

        # Classify workouts
        tempo_count, interval_count = self._classify_workouts(activities)

        # Calculate features
        features = {}

        # Zone percentages
        total_points = sum(zone_distribution.values())
        if total_points > 0:
            features['zone1_percent'] = (zone_distribution.get(1, 0) / total_points) * 100
            features['zone2_percent'] = (zone_distribution.get(2, 0) / total_points) * 100
            features['zone3_percent'] = (zone_distribution.get(3, 0) / total_points) * 100
            features['zone4_percent'] = (zone_distribution.get(4, 0) / total_points) * 100
            features['zone5_percent'] = (zone_distribution.get(5, 0) / total_points) * 100
        else:
            features['zone1_percent'] = 0.0
            features['zone2_percent'] = 0.0
            features['zone3_percent'] = 0.0
            features['zone4_percent'] = 0.0
            features['zone5_percent'] = 0.0

        # Workout counts
        features['tempo_workout_count'] = tempo_count
        features['interval_workout_count'] = interval_count

        # Quality workout percentage
        total_workouts = len(activities)
        quality_workouts = tempo_count + interval_count
        features['quality_workout_percent'] = (quality_workouts / total_workouts * 100) if total_workouts > 0 else 0.0

        return features

    def _define_zones(
        self,
        lt_hr: Optional[float],
        aet_hr: Optional[float],
        max_hr: Optional[float]
    ) -> Dict[int, tuple]:
        """Define 5-zone heart rate training zones

        Zone 1: Recovery (< AET)
        Zone 2: Aerobic (AET to ~92% LT)
        Zone 3: Tempo (~92-100% LT)
        Zone 4: Threshold (LT to ~95% max)
        Zone 5: VO2max+ (> 95% max)
        """
        zones = {}

        if aet_hr and lt_hr and max_hr:
            zones[1] = (0, aet_hr)
            zones[2] = (aet_hr, lt_hr * 0.92)
            zones[3] = (lt_hr * 0.92, lt_hr)
            zones[4] = (lt_hr, max_hr * 0.95)
            zones[5] = (max_hr * 0.95, 250)
        elif lt_hr and max_hr:
            # Use LT and max HR only
            zones[1] = (0, lt_hr * 0.75)
            zones[2] = (lt_hr * 0.75, lt_hr * 0.92)
            zones[3] = (lt_hr * 0.92, lt_hr)
            zones[4] = (lt_hr, max_hr * 0.95)
            zones[5] = (max_hr * 0.95, 250)
        elif max_hr:
            # Use max HR only (rough estimates)
            zones[1] = (0, max_hr * 0.60)
            zones[2] = (max_hr * 0.60, max_hr * 0.75)
            zones[3] = (max_hr * 0.75, max_hr * 0.85)
            zones[4] = (max_hr * 0.85, max_hr * 0.95)
            zones[5] = (max_hr * 0.95, 250)
        else:
            # Default zones (assuming max HR ~190)
            zones[1] = (0, 115)
            zones[2] = (115, 140)
            zones[3] = (140, 160)
            zones[4] = (160, 180)
            zones[5] = (180, 250)

        return zones

    def _calculate_zone_distribution(
        self,
        activities: List[Dict],
        zones: Dict[int, tuple]
    ) -> Dict[int, int]:
        """Calculate time spent in each zone using track point summaries"""
        import sqlite3
        zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        for activity in activities:
            file_name = activity['file_name']

            # Query track point summary for HR data
            cursor.execute("""
                SELECT avg_hr, point_count
                FROM track_point_summary
                WHERE file_name = ? AND avg_hr IS NOT NULL
            """, (file_name,))

            for row in cursor.fetchall():
                avg_hr = row[0]
                count = row[1]

                # Classify into zone
                for zone_num, (lower, upper) in zones.items():
                    if lower <= avg_hr < upper:
                        zone_counts[zone_num] += count
                        break

        conn.close()

        return zone_counts

    def _classify_workouts(self, activities: List[Dict]) -> tuple:
        """Classify workouts as tempo or interval

        Returns:
            (tempo_count, interval_count)
        """
        tempo_count = 0
        interval_count = 0

        for activity in activities:
            if not activity.get('laps_json'):
                continue

            laps = json.loads(activity['laps_json'])

            # Check for tempo workout
            if self._is_tempo_workout(laps, activity):
                tempo_count += 1
            # Check for interval workout
            elif self._is_interval_workout(laps, activity):
                interval_count += 1

        return tempo_count, interval_count

    def _is_tempo_workout(self, laps: List[Dict], activity: Dict) -> bool:
        """Detect tempo workout

        Criteria:
        - 1-4 mile laps
        - Consistent pace (CV < 15%)
        - Hard effort (HR >= 150 bpm)
        - Duration 6-30 minutes per lap
        - Pace 5-9 min/mile
        """
        tempo_laps = []

        for lap in laps:
            distance_mi = meters_to_miles(lap.get('distance', 0))
            duration_min = lap.get('duration_seconds', 0) / 60
            pace = lap.get('pace', 0)
            hr = lap.get('avg_hr', 0) or activity.get('avg_heart_rate', 0)

            # Check tempo criteria
            if (1.0 <= distance_mi <= 4.0 and
                6 <= duration_min <= 30 and
                5 < pace < 9 and
                hr >= 150):
                tempo_laps.append(lap)

        # Need at least 1 tempo lap with consistent pace
        if len(tempo_laps) >= 1:
            paces = [lap.get('pace', 0) for lap in tempo_laps]
            if len(paces) > 1:
                cv = np.std(paces) / np.mean(paces)
                return cv < 0.15
            return True

        return False

    def _is_interval_workout(self, laps: List[Dict], activity: Dict) -> bool:
        """Detect interval workout

        Criteria:
        - Multiple laps of similar distance (within 0.3 miles)
        - Fast pace (5-9 min/mile)
        - High HR (>= 140 bpm at activity level)
        - At least 2 reps
        """
        if not activity.get('avg_heart_rate') or activity['avg_heart_rate'] < 140:
            return False

        # Group laps by similar distance
        distance_groups = {}

        for lap in laps:
            dist_mi = meters_to_miles(lap.get('distance', 0))
            pace = lap.get('pace', 0)

            # Must be reasonable interval distance and pace
            if not (0.8 <= dist_mi <= 5.0 and 5 < pace < 9):
                continue

            # Find matching distance group
            found_group = False
            for group_dist in distance_groups:
                if abs(dist_mi - group_dist) < 0.3:
                    distance_groups[group_dist].append(lap)
                    found_group = True
                    break

            if not found_group:
                distance_groups[dist_mi] = [lap]

        # Check for groups with 2+ intervals
        for group_laps in distance_groups.values():
            if len(group_laps) >= 2:
                # Check pace consistency
                paces = [lap.get('pace', 0) for lap in group_laps]
                cv = np.std(paces) / np.mean(paces)
                if cv < 0.15:
                    return True

        return False

    def _default_features(self) -> Dict[str, float]:
        """Return default features when no activities"""
        return {
            'zone1_percent': 0.0,
            'zone2_percent': 0.0,
            'zone3_percent': 0.0,
            'zone4_percent': 0.0,
            'zone5_percent': 0.0,
            'tempo_workout_count': 0,
            'interval_workout_count': 0,
            'quality_workout_percent': 0.0,
        }
