"""
Training Volume Feature Extractor
Computes weekly mileage, long runs, and consistency metrics
"""

from typing import Dict, List
import numpy as np

from feature_engineering.utils import (
    group_by_week,
    meters_to_miles,
    calculate_coefficient_of_variation
)


class TrainingVolumeExtractor:
    """Extract training volume features"""

    def extract(self, activities: List[Dict]) -> Dict[str, float]:
        """Extract training volume features

        Args:
            activities: List of activity dictionaries from cache

        Returns:
            Dictionary with volume features
        """
        if not activities:
            return self._default_features()

        # Group activities by week
        weeks = group_by_week(activities)

        # Calculate weekly mileages
        weekly_mileages = []
        for week_activities in weeks.values():
            week_distance = sum(a.get('distance_meters', 0) for a in week_activities)
            weekly_mileages.append(meters_to_miles(week_distance))

        # Find long runs
        long_runs = self._find_long_runs(activities, weeks)

        # Calculate features
        features = {}

        # Total weekly mileage (average)
        features['total_weekly_mileage'] = np.mean(weekly_mileages) if weekly_mileages else 0.0

        # Peak weekly mileage
        features['peak_weekly_mileage'] = max(weekly_mileages) if weekly_mileages else 0.0

        # Long run distance (average)
        if long_runs:
            features['long_run_distance'] = np.mean(long_runs)
        else:
            features['long_run_distance'] = 0.0

        # Long run as percent of weekly volume
        if features['total_weekly_mileage'] > 0 and features['long_run_distance'] > 0:
            features['long_run_percent_weekly'] = (
                features['long_run_distance'] / features['total_weekly_mileage']
            )
        else:
            features['long_run_percent_weekly'] = 0.0

        # Total runs
        features['total_runs'] = len(activities)

        # Runs per week
        num_weeks = len(weeks) if weeks else 1
        features['runs_per_week'] = len(activities) / num_weeks

        # Mileage consistency (coefficient of variation)
        features['mileage_consistency'] = calculate_coefficient_of_variation(weekly_mileages)

        return features

    def _find_long_runs(self, activities: List[Dict], weeks: Dict[int, List[Dict]]) -> List[float]:
        """Find long run for each week

        Long run is defined as the longest run of the week
        """
        long_runs = []

        for week_activities in weeks.values():
            if not week_activities:
                continue

            # Find longest run of the week
            longest = max(week_activities, key=lambda a: a.get('distance_meters', 0))
            long_run_miles = meters_to_miles(longest.get('distance_meters', 0))

            # Only count if it's significant (> 5 miles)
            if long_run_miles > 5.0:
                long_runs.append(long_run_miles)

        return long_runs

    def _default_features(self) -> Dict[str, float]:
        """Return default features when no activities"""
        return {
            'total_weekly_mileage': 0.0,
            'peak_weekly_mileage': 0.0,
            'long_run_distance': 0.0,
            'long_run_percent_weekly': 0.0,
            'total_runs': 0,
            'runs_per_week': 0.0,
            'mileage_consistency': 0.0,
        }
