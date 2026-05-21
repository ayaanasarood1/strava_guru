"""
Runner Profile Feature Extractor
Normalizes runner personalization features
"""

from typing import Dict, List

from feature_engineering.runner_context import RunnerContext
from feature_engineering.utils import (
    normalize_age,
    normalize_hr,
    calculate_coefficient_of_variation,
    group_by_week,
    meters_to_miles
)


class RunnerProfileExtractor:
    """Extract runner personalization features"""

    def extract(
        self,
        runner_context: RunnerContext,
        activities: List[Dict]
    ) -> Dict[str, float]:
        """Extract runner profile features

        Args:
            runner_context: Runner personal characteristics
            activities: List of activity dictionaries (for consistency)

        Returns:
            Dictionary with profile features
        """
        features = {}

        # Age normalized (peak at 35)
        features['age_normalized'] = normalize_age(runner_context.age)

        # Sex encoded (0 = F, 1 = M)
        features['sex_encoded'] = 1.0 if runner_context.sex == 'M' else 0.0

        # Max HR normalized
        features['max_hr_normalized'] = normalize_hr(
            runner_context.max_hr,
            min_hr=150,
            max_hr=220
        )

        # Experience years
        if runner_context.experience_years:
            # Normalize to 0-1 (capped at 15 years = expert)
            features['experience_years'] = min(runner_context.experience_years / 15.0, 1.0)
        else:
            # Estimate from age (assume started running at 25)
            estimated_years = max(runner_context.age - 25, 0)
            features['experience_years'] = min(estimated_years / 15.0, 1.0)

        # Recent injury flag
        features['recent_injury_flag'] = 1.0 if runner_context.recent_injury_flag else 0.0

        # Training consistency score
        features['training_consistency_score'] = self._calculate_consistency(activities)

        return features

    def _calculate_consistency(self, activities: List[Dict]) -> float:
        """Calculate training consistency score (0-1)

        Higher score = more consistent training
        Based on:
        - Week-to-week mileage consistency
        - Number of weeks with at least 3 runs
        """
        if not activities:
            return 0.0

        # Group by week
        weeks = group_by_week(activities)

        # Calculate weekly mileages
        weekly_mileages = []
        weeks_with_3plus_runs = 0

        for week_activities in weeks.values():
            week_distance = sum(a.get('distance_meters', 0) for a in week_activities)
            weekly_mileages.append(meters_to_miles(week_distance))

            if len(week_activities) >= 3:
                weeks_with_3plus_runs += 1

        if not weekly_mileages:
            return 0.0

        # Consistency component (inverse of CV)
        cv = calculate_coefficient_of_variation(weekly_mileages)
        consistency_score = 1.0 / (1.0 + cv) if cv > 0 else 1.0

        # Frequency component (weeks with 3+ runs)
        frequency_score = weeks_with_3plus_runs / len(weeks) if weeks else 0.0

        # Combined score (weighted average)
        combined_score = 0.6 * consistency_score + 0.4 * frequency_score

        return combined_score
