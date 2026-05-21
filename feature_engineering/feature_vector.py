"""
Training Feature Vector
Dataclass containing all 41 features for race prediction
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class TrainingFeatureVector:
    """Training features extracted from activity data

    Contains 41 features across 7 categories:
    - Lactate Threshold (5)
    - Training Volume (7)
    - Training Intensity (8)
    - Running Efficiency (5)
    - Terrain Handling (4)
    - Race Context (3)
    - Runner Personalization (6)
    - Reserved for future (3)
    """

    # Category 1: Lactate Threshold (5 features)
    lt_heart_rate: Optional[float] = None  # LT heart rate (bpm)
    lt_pace: Optional[float] = None  # LT pace (min/mile)
    lt_percent_max_hr: Optional[float] = None  # LT as % of max HR
    aet_heart_rate: Optional[float] = None  # Aerobic threshold HR (bpm)
    aet_pace: Optional[float] = None  # Aerobic threshold pace (min/mile)

    # Category 2: Training Volume (7 features)
    total_weekly_mileage: float = 0.0  # Average weekly miles
    peak_weekly_mileage: float = 0.0  # Highest weekly miles
    long_run_distance: float = 0.0  # Average long run distance (miles)
    long_run_percent_weekly: float = 0.0  # Long run as % of weekly volume
    total_runs: int = 0  # Total number of runs
    runs_per_week: float = 0.0  # Average runs per week
    mileage_consistency: float = 0.0  # Coefficient of variation (lower = more consistent)

    # Category 3: Training Intensity (8 features)
    zone1_percent: float = 0.0  # % time in Zone 1 (easy)
    zone2_percent: float = 0.0  # % time in Zone 2 (aerobic)
    zone3_percent: float = 0.0  # % time in Zone 3 (tempo)
    zone4_percent: float = 0.0  # % time in Zone 4 (threshold)
    zone5_percent: float = 0.0  # % time in Zone 5 (VO2max+)
    tempo_workout_count: int = 0  # Number of tempo workouts
    interval_workout_count: int = 0  # Number of interval workouts
    quality_workout_percent: float = 0.0  # % of workouts that are quality

    # Category 4: Running Efficiency (5 features)
    hr_at_easy_pace: Optional[float] = None  # HR at 9:00/mile (bpm)
    hr_at_marathon_pace: Optional[float] = None  # HR at estimated MP (bpm)
    cardiac_drift: float = 0.0  # HR drift in long runs (bpm/hour)
    aerobic_decoupling: float = 0.0  # HR-pace decoupling (%)
    hr_variability_coefficient: float = 0.0  # Coefficient of variation of HR

    # Category 5: Terrain Handling (4 features)
    hr_per_grade_uphill: float = 0.0  # HR increase per % grade uphill (bpm/%)
    hr_per_grade_downhill: float = 0.0  # HR change per % grade downhill (bpm/%)
    hill_recovery_rate: float = 0.0  # HR recovery after hills (bpm/min)
    elevation_tolerance: float = 0.0  # Performance on hilly vs flat (ratio)

    # Category 6: Race Context (3 features)
    race_distance_miles: float = 0.0  # Target race distance
    taper_quality_score: float = 0.0  # Quality of taper (0-1)
    days_since_last_hard_effort: int = 0  # Days since last quality workout

    # Category 7: Runner Personalization (6 features)
    age_normalized: float = 0.0  # Age normalized to 0-1 (35 = peak)
    sex_encoded: float = 0.0  # 0 = F, 1 = M
    max_hr_normalized: float = 0.0  # Max HR normalized to 0-1
    experience_years: float = 0.0  # Years of running experience
    recent_injury_flag: float = 0.0  # 0 = no injury, 1 = recent injury
    training_consistency_score: float = 0.0  # Consistency score (0-1)

    # Reserved for future categories (3 features)
    # Weather, course profile, race-day conditions will go here
    reserved_1: Optional[float] = None
    reserved_2: Optional[float] = None
    reserved_3: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML pipelines"""
        return asdict(self)

    def to_dataframe(self):
        """Convert to single-row pandas DataFrame

        Requires pandas to be installed
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")
        import pandas as pd
        return pd.DataFrame([self.to_dict()])

    def feature_count(self) -> int:
        """Count non-None features"""
        d = self.to_dict()
        return sum(1 for v in d.values() if v is not None)

    def validate(self) -> bool:
        """Check if feature vector has reasonable values"""
        issues = []

        # HR features should be in reasonable range
        hr_features = [
            self.lt_heart_rate, self.aet_heart_rate,
            self.hr_at_easy_pace, self.hr_at_marathon_pace
        ]
        for hr in hr_features:
            if hr is not None and (hr < 50 or hr > 220):
                issues.append(f"Invalid HR value: {hr}")

        # Pace features should be reasonable
        pace_features = [self.lt_pace, self.aet_pace]
        for pace in pace_features:
            if pace is not None and (pace < 4 or pace > 15):
                issues.append(f"Invalid pace value: {pace}")

        # Percentages should sum to ~100
        zone_sum = (self.zone1_percent + self.zone2_percent + self.zone3_percent +
                    self.zone4_percent + self.zone5_percent)
        if zone_sum > 0 and abs(zone_sum - 100) > 5:
            issues.append(f"Zone percentages sum to {zone_sum}, expected ~100")

        # Normalized features should be 0-1
        normalized = [
            self.lt_percent_max_hr, self.age_normalized,
            self.max_hr_normalized, self.training_consistency_score,
            self.taper_quality_score
        ]
        for val in normalized:
            if val is not None and (val < 0 or val > 1):
                issues.append(f"Normalized value out of range: {val}")

        if issues:
            print(f"Validation issues: {issues}")
            return False

        return True
