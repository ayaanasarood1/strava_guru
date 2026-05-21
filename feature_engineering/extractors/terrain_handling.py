"""
Terrain Handling Feature Extractor
Analyzes performance on hills and elevation changes
"""

from typing import Dict, List
import numpy as np
import pickle

from feature_engineering.utils import calculate_grade, meters_to_miles


class TerrainHandlingExtractor:
    """Extract terrain handling features"""

    def __init__(self, activity_cache):
        """Initialize with activity cache"""
        self.cache = activity_cache

    def extract(self, activities: List[Dict]) -> Dict[str, float]:
        """Extract terrain handling features

        Args:
            activities: List of activity dictionaries

        Returns:
            Dictionary with terrain features
        """
        if not activities:
            return self._default_features()

        # Analyze grade-HR relationship
        uphill_hr_per_grade = self._analyze_uphill_hr(activities)
        downhill_hr_per_grade = self._analyze_downhill_hr(activities)

        # Calculate hill recovery rate
        hill_recovery = self._calculate_hill_recovery(activities)

        # Calculate elevation tolerance
        elevation_tolerance = self._calculate_elevation_tolerance(activities)

        features = {
            'hr_per_grade_uphill': uphill_hr_per_grade,
            'hr_per_grade_downhill': downhill_hr_per_grade,
            'hill_recovery_rate': hill_recovery,
            'elevation_tolerance': elevation_tolerance,
        }

        return features

    def _analyze_uphill_hr(self, activities: List[Dict]) -> float:
        """Calculate HR increase per % grade uphill

        Returns:
            HR increase per 1% grade (bpm/%)
        """
        grade_hr_pairs = []

        for activity in activities:
            # Skip if no elevation data
            if not activity.get('elevation_gain_meters'):
                continue

            # Load track points
            track_points = self._load_track_points(activity)
            if not track_points or len(track_points) < 50:
                continue

            # Analyze uphill sections
            for i in range(1, len(track_points)):
                curr = track_points[i]
                prev = track_points[i - 1]

                # Need HR and elevation data
                if not (curr.heart_rate and curr.elevation is not None and
                        prev.elevation is not None):
                    continue

                # Calculate grade
                if curr.distance and prev.distance:
                    distance_m = curr.distance - prev.distance
                    if distance_m > 0:
                        elev_change = curr.elevation - prev.elevation

                        # Only uphill segments
                        if elev_change > 0:
                            grade = calculate_grade(elev_change, distance_m)

                            # Reasonable grade range (1-20%)
                            if 1 <= grade <= 20:
                                grade_hr_pairs.append((grade, curr.heart_rate))

        if len(grade_hr_pairs) < 20:
            return 0.0

        # Calculate HR change per grade using linear regression
        grades = np.array([g for g, _ in grade_hr_pairs])
        hrs = np.array([hr for _, hr in grade_hr_pairs])

        # Group by grade bins and calculate median HR
        grade_bins = np.arange(1, 15, 1)  # 1-14% in 1% bins
        binned_hr_changes = []

        for i in range(len(grade_bins) - 1):
            mask = (grades >= grade_bins[i]) & (grades < grade_bins[i + 1])
            if np.sum(mask) > 5:
                median_hr = np.median(hrs[mask])
                binned_hr_changes.append(median_hr)

        if len(binned_hr_changes) >= 3:
            # Calculate slope (HR per grade)
            x = np.arange(len(binned_hr_changes))
            slope = np.polyfit(x, binned_hr_changes, 1)[0]
            return slope

        return 0.0

    def _analyze_downhill_hr(self, activities: List[Dict]) -> float:
        """Calculate HR change per % grade downhill

        Returns:
            HR change per 1% grade (bpm/%)
        """
        grade_hr_pairs = []

        for activity in activities:
            if not activity.get('elevation_loss_meters'):
                continue

            track_points = self._load_track_points(activity)
            if not track_points or len(track_points) < 50:
                continue

            # Analyze downhill sections
            for i in range(1, len(track_points)):
                curr = track_points[i]
                prev = track_points[i - 1]

                if not (curr.heart_rate and curr.elevation is not None and
                        prev.elevation is not None):
                    continue

                if curr.distance and prev.distance:
                    distance_m = curr.distance - prev.distance
                    if distance_m > 0:
                        elev_change = curr.elevation - prev.elevation

                        # Only downhill segments
                        if elev_change < 0:
                            grade = abs(calculate_grade(elev_change, distance_m))

                            if 1 <= grade <= 20:
                                grade_hr_pairs.append((grade, curr.heart_rate))

        if len(grade_hr_pairs) < 20:
            return 0.0

        # Calculate HR change for downhills
        grades = np.array([g for g, _ in grade_hr_pairs])
        hrs = np.array([hr for _, hr in grade_hr_pairs])

        grade_bins = np.arange(1, 15, 1)
        binned_hr_changes = []

        for i in range(len(grade_bins) - 1):
            mask = (grades >= grade_bins[i]) & (grades < grade_bins[i + 1])
            if np.sum(mask) > 5:
                median_hr = np.median(hrs[mask])
                binned_hr_changes.append(median_hr)

        if len(binned_hr_changes) >= 3:
            x = np.arange(len(binned_hr_changes))
            slope = np.polyfit(x, binned_hr_changes, 1)[0]
            return slope

        return 0.0

    def _calculate_hill_recovery(self, activities: List[Dict]) -> float:
        """Calculate HR recovery rate after hills (bpm/min)

        Measures how quickly HR drops after uphill efforts
        """
        recovery_rates = []

        for activity in activities:
            if not activity.get('elevation_gain_meters'):
                continue

            track_points = self._load_track_points(activity)
            if not track_points or len(track_points) < 100:
                continue

            # Find hill climbs followed by flat/downhill
            i = 0
            while i < len(track_points) - 20:
                # Look for uphill section (at least 10 points climbing)
                climb_start = i
                climb_hr_peak = 0

                # Scan for climb
                climbing = True
                climb_length = 0
                while climbing and i < len(track_points) - 10:
                    curr = track_points[i]
                    if i > 0:
                        prev = track_points[i - 1]
                        if (curr.elevation is not None and prev.elevation is not None and
                            curr.elevation > prev.elevation):
                            climb_length += 1
                            if curr.heart_rate:
                                climb_hr_peak = max(climb_hr_peak, curr.heart_rate)
                        else:
                            climbing = False
                    i += 1

                # Need significant climb
                if climb_length < 10 or climb_hr_peak == 0:
                    continue

                # Now measure recovery (next 10-20 points)
                recovery_start = i
                recovery_points = []
                for j in range(recovery_start, min(recovery_start + 20, len(track_points))):
                    if track_points[j].heart_rate:
                        recovery_points.append(track_points[j].heart_rate)

                if len(recovery_points) >= 5:
                    # Calculate recovery rate (HR drop per minute)
                    hr_drop = climb_hr_peak - recovery_points[-1]
                    time_min = len(recovery_points) * 10 / 60  # 10-sec buckets
                    if time_min > 0:
                        recovery_rate = hr_drop / time_min
                        if recovery_rate > 0:
                            recovery_rates.append(recovery_rate)

        if recovery_rates:
            return np.mean(recovery_rates)
        return 0.0

    def _calculate_elevation_tolerance(self, activities: List[Dict]) -> float:
        """Calculate performance on hilly vs flat runs

        Returns ratio of performance (higher = better on hills)
        """
        hilly_paces = []
        flat_paces = []

        for activity in activities:
            distance_mi = meters_to_miles(activity.get('distance_meters', 0))
            if distance_mi < 3:  # Skip short runs
                continue

            # Calculate elevation per mile
            elev_gain = activity.get('elevation_gain_meters', 0)
            elev_per_mile = (elev_gain * 3.28084) / distance_mi  # ft/mile

            pace = activity.get('avg_pace')
            if not pace or not (5 < pace < 12):  # Reasonable pace
                continue

            # Classify as hilly or flat
            if elev_per_mile > 100:  # > 100 ft/mile = hilly
                hilly_paces.append(pace)
            elif elev_per_mile < 30:  # < 30 ft/mile = flat
                flat_paces.append(pace)

        # Calculate tolerance (ratio of flat pace to hilly pace)
        if hilly_paces and flat_paces:
            avg_hilly = np.mean(hilly_paces)
            avg_flat = np.mean(flat_paces)
            if avg_hilly > 0:
                # Higher ratio = better hill tolerance
                # (closer to 1.0 = maintains pace on hills)
                return avg_flat / avg_hilly

        return 0.0

    def _load_track_points(self, activity: Dict) -> list:
        """Load track points for an activity"""
        track_points_file = activity.get('track_points_file')
        if not track_points_file:
            return []

        track_points_path = self.cache.cache_dir / track_points_file
        if not track_points_path.exists():
            return []

        try:
            with open(track_points_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return []

    def _default_features(self) -> Dict[str, float]:
        """Return default features when no activities"""
        return {
            'hr_per_grade_uphill': 0.0,
            'hr_per_grade_downhill': 0.0,
            'hill_recovery_rate': 0.0,
            'elevation_tolerance': 0.0,
        }
