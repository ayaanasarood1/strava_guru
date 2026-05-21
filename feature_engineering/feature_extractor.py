"""
Training Feature Extractor
Main orchestrator for feature extraction pipeline
"""

from datetime import datetime
from typing import Optional
from pathlib import Path

from activity_cache import ActivityCache
from .runner_context import RunnerContext
from .feature_vector import TrainingFeatureVector
from .utils import get_training_window, meters_to_miles, group_by_week

from .extractors import (
    LactateThresholdExtractor,
    TrainingVolumeExtractor,
    TrainingIntensityExtractor,
    RunningEfficiencyExtractor,
    TerrainHandlingExtractor,
    RaceContextExtractor,
    RunnerProfileExtractor,
)


class TrainingFeatureExtractor:
    """Main feature extraction orchestrator"""

    def __init__(self, activity_cache: ActivityCache, cache_dir: Path = None):
        """Initialize feature extractor

        Args:
            activity_cache: ActivityCache instance for querying activities
            cache_dir: Optional cache directory for LT caching
        """
        self.cache = activity_cache

        # Initialize extractors
        self.lt_extractor = LactateThresholdExtractor(cache_dir)
        self.volume_extractor = TrainingVolumeExtractor()
        self.intensity_extractor = TrainingIntensityExtractor(activity_cache)
        self.efficiency_extractor = RunningEfficiencyExtractor(activity_cache)
        self.terrain_extractor = TerrainHandlingExtractor(activity_cache)
        self.race_context_extractor = RaceContextExtractor()
        self.profile_extractor = RunnerProfileExtractor()

    def extract_features(
        self,
        runner_id: str,
        race_date: datetime,
        lookback_weeks: int,
        race_distance_miles: float,
        runner_context: RunnerContext
    ) -> TrainingFeatureVector:
        """Extract all training features for race prediction

        Args:
            runner_id: Identifier for runner (for logging)
            race_date: Date of target race
            lookback_weeks: Number of weeks to look back for training data
            race_distance_miles: Distance of target race (miles)
            runner_context: Runner personal characteristics

        Returns:
            TrainingFeatureVector with all extracted features
        """
        print(f"Extracting features for {runner_id}...")
        print(f"  Race date: {race_date.strftime('%Y-%m-%d')}")
        print(f"  Lookback: {lookback_weeks} weeks")
        print(f"  Race distance: {race_distance_miles} miles")

        # Get training window
        start_date, end_date = get_training_window(race_date, lookback_weeks)
        print(f"  Training window: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Query activities
        activities = self.cache.get_activities_by_date_range(start_date, end_date)
        print(f"  Found {len(activities)} activities")

        if not activities:
            print("  WARNING: No activities found in training window")
            return self._default_feature_vector(race_distance_miles, runner_context)

        # Initialize feature vector
        features = TrainingFeatureVector()

        # Extract features by category
        try:
            # 1. Lactate Threshold
            print("  Extracting lactate threshold features...")
            lt_features = self.lt_extractor.extract(
                self.cache,
                activities,
                runner_context.max_hr
            )
            features.lt_heart_rate = lt_features.get('lt_heart_rate')
            features.lt_pace = lt_features.get('lt_pace')
            features.lt_percent_max_hr = lt_features.get('lt_percent_max_hr')
            features.aet_heart_rate = lt_features.get('aet_heart_rate')
            features.aet_pace = lt_features.get('aet_pace')

            # 2. Training Volume
            print("  Extracting training volume features...")
            volume_features = self.volume_extractor.extract(activities)
            features.total_weekly_mileage = volume_features['total_weekly_mileage']
            features.peak_weekly_mileage = volume_features['peak_weekly_mileage']
            features.long_run_distance = volume_features['long_run_distance']
            features.long_run_percent_weekly = volume_features['long_run_percent_weekly']
            features.total_runs = volume_features['total_runs']
            features.runs_per_week = volume_features['runs_per_week']
            features.mileage_consistency = volume_features['mileage_consistency']

            # 3. Training Intensity
            print("  Extracting training intensity features...")
            intensity_features = self.intensity_extractor.extract(
                activities,
                lt_hr=features.lt_heart_rate,
                aet_hr=features.aet_heart_rate,
                max_hr=runner_context.max_hr
            )
            features.zone1_percent = intensity_features['zone1_percent']
            features.zone2_percent = intensity_features['zone2_percent']
            features.zone3_percent = intensity_features['zone3_percent']
            features.zone4_percent = intensity_features['zone4_percent']
            features.zone5_percent = intensity_features['zone5_percent']
            features.tempo_workout_count = intensity_features['tempo_workout_count']
            features.interval_workout_count = intensity_features['interval_workout_count']
            features.quality_workout_percent = intensity_features['quality_workout_percent']

            # 4. Running Efficiency
            print("  Extracting running efficiency features...")
            efficiency_features = self.efficiency_extractor.extract(
                activities,
                lt_pace=features.lt_pace
            )
            features.hr_at_easy_pace = efficiency_features['hr_at_easy_pace']
            features.hr_at_marathon_pace = efficiency_features['hr_at_marathon_pace']
            features.cardiac_drift = efficiency_features['cardiac_drift']
            features.aerobic_decoupling = efficiency_features['aerobic_decoupling']
            features.hr_variability_coefficient = efficiency_features['hr_variability_coefficient']

            # 5. Terrain Handling
            print("  Extracting terrain handling features...")
            terrain_features = self.terrain_extractor.extract(activities)
            features.hr_per_grade_uphill = terrain_features['hr_per_grade_uphill']
            features.hr_per_grade_downhill = terrain_features['hr_per_grade_downhill']
            features.hill_recovery_rate = terrain_features['hill_recovery_rate']
            features.elevation_tolerance = terrain_features['elevation_tolerance']

            # 6. Race Context
            print("  Extracting race context features...")
            # Calculate pre-taper weekly average
            pre_taper_weekly_avg = self._calculate_pre_taper_weekly_avg(
                activities,
                race_date
            )
            context_features = self.race_context_extractor.extract(
                activities,
                race_date,
                race_distance_miles,
                pre_taper_weekly_avg
            )
            features.race_distance_miles = context_features['race_distance_miles']
            features.taper_quality_score = context_features['taper_quality_score']
            features.days_since_last_hard_effort = context_features['days_since_last_hard_effort']

            # 7. Runner Personalization
            print("  Extracting runner profile features...")
            profile_features = self.profile_extractor.extract(
                runner_context,
                activities
            )
            features.age_normalized = profile_features['age_normalized']
            features.sex_encoded = profile_features['sex_encoded']
            features.max_hr_normalized = profile_features['max_hr_normalized']
            features.experience_years = profile_features['experience_years']
            features.recent_injury_flag = profile_features['recent_injury_flag']
            features.training_consistency_score = profile_features['training_consistency_score']

            print(f"  ✓ Extracted {features.feature_count()} features")

            # Validate
            if not features.validate():
                print("  WARNING: Feature validation failed")

            return features

        except Exception as e:
            print(f"  ERROR: Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._default_feature_vector(race_distance_miles, runner_context)

    def _calculate_pre_taper_weekly_avg(
        self,
        activities: list,
        race_date: datetime
    ) -> float:
        """Calculate average weekly mileage before taper period"""
        # Use 8 weeks before taper (2 weeks before race)
        from datetime import timedelta
        taper_start = race_date - timedelta(days=14)
        pre_taper_start = taper_start - timedelta(weeks=8)

        pre_taper_activities = []
        for a in activities:
            act_date = a['activity_date']
            if isinstance(act_date, str):
                act_date = datetime.fromisoformat(act_date)
            if pre_taper_start <= act_date < taper_start:
                pre_taper_activities.append(a)

        if not pre_taper_activities:
            return 0.0

        weeks = group_by_week(pre_taper_activities)
        weekly_mileages = []

        for week_activities in weeks.values():
            week_distance = sum(a.get('distance_meters', 0) for a in week_activities)
            weekly_mileages.append(meters_to_miles(week_distance))

        if weekly_mileages:
            import numpy as np
            return np.mean(weekly_mileages)

        return 0.0

    def _default_feature_vector(
        self,
        race_distance_miles: float,
        runner_context: RunnerContext
    ) -> TrainingFeatureVector:
        """Return feature vector with defaults when extraction fails"""
        features = TrainingFeatureVector()

        # Set basic context
        features.race_distance_miles = race_distance_miles

        # Set runner profile features
        profile_features = self.profile_extractor.extract(runner_context, [])
        features.age_normalized = profile_features['age_normalized']
        features.sex_encoded = profile_features['sex_encoded']
        features.max_hr_normalized = profile_features['max_hr_normalized']
        features.experience_years = profile_features['experience_years']
        features.recent_injury_flag = profile_features['recent_injury_flag']
        features.training_consistency_score = 0.0

        return features
