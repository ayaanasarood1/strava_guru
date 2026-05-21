"""
Race Context Feature Extractor
Extracts race-specific contextual features
"""

from typing import Dict, List
from datetime import datetime, timedelta

from feature_engineering.utils import calculate_taper_quality


class RaceContextExtractor:
    """Extract race context features"""

    def extract(
        self,
        activities: List[Dict],
        race_date: datetime,
        race_distance_miles: float,
        pre_taper_weekly_avg: float
    ) -> Dict[str, float]:
        """Extract race context features

        Args:
            activities: List of activity dictionaries
            race_date: Date of target race
            race_distance_miles: Distance of target race
            pre_taper_weekly_avg: Average weekly mileage before taper

        Returns:
            Dictionary with race context features
        """
        features = {}

        # Race distance (input feature)
        features['race_distance_miles'] = race_distance_miles

        # Taper quality score
        taper_start = race_date - timedelta(days=14)  # 2-week taper
        features['taper_quality_score'] = calculate_taper_quality(
            activities,
            taper_start,
            race_date,
            pre_taper_weekly_avg
        )

        # Days since last hard effort
        features['days_since_last_hard_effort'] = self._days_since_hard_effort(
            activities,
            race_date
        )

        return features

    def _days_since_hard_effort(self, activities: List[Dict], race_date: datetime) -> int:
        """Calculate days since last hard effort

        Hard effort defined as:
        - HR > 160 bpm
        - Duration > 20 minutes
        - Pace < 8:30/mile
        """
        hard_efforts = []

        for activity in activities:
            # Parse activity date
            act_date = activity['activity_date']
            if isinstance(act_date, str):
                act_date = datetime.fromisoformat(act_date)

            # Must be before race
            if act_date >= race_date:
                continue

            # Check hard effort criteria
            avg_hr = activity.get('avg_heart_rate', 0) or 0
            duration_min = activity.get('moving_time_seconds', 0) / 60
            avg_pace = activity.get('avg_pace', 999) or 999

            if avg_hr > 160 and duration_min > 20 and avg_pace < 8.5:
                hard_efforts.append(act_date)

        if hard_efforts:
            # Find most recent
            most_recent = max(hard_efforts)
            days_since = (race_date - most_recent).days
            return days_since

        return 999  # No recent hard effort found
