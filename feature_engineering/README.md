# Feature Engineering Pipeline

A modular pipeline for extracting training features from running activities for race time prediction.

## Overview

This pipeline extracts **41 features** across 7 categories from your training data:

1. **Lactate Threshold** (5 features) - LT heart rate, pace, and aerobic threshold
2. **Training Volume** (7 features) - Weekly mileage, long runs, consistency
3. **Training Intensity** (8 features) - HR zones, tempo/interval workouts
4. **Running Efficiency** (5 features) - HR-pace relationship, cardiac drift
5. **Terrain Handling** (4 features) - Performance on hills
6. **Race Context** (3 features) - Race distance, taper quality, recency
7. **Runner Personalization** (6 features) - Age, sex, experience, consistency
8. **Reserved** (3 features) - For future weather/course features

## Quick Start

```python
from datetime import datetime
from activity_cache import ActivityCache
from feature_engineering import TrainingFeatureExtractor, RunnerContext

# Initialize
cache = ActivityCache()
extractor = TrainingFeatureExtractor(cache)

# Define runner profile
runner = RunnerContext(
    age=35,
    sex='M',
    max_hr=185,
    experience_years=8,
    resting_hr=52
)

# Extract features
features = extractor.extract_features(
    runner_id="runner_001",
    race_date=datetime(2026, 10, 15),
    lookback_weeks=12,
    race_distance_miles=26.2,
    runner_context=runner
)

# Export for ML
feature_dict = features.to_dict()  # Dictionary
# feature_df = features.to_dataframe()  # Pandas DataFrame (requires pandas)
```

## Architecture

### Core Components

- **`TrainingFeatureExtractor`**: Main orchestrator that coordinates all extractors
- **`TrainingFeatureVector`**: Dataclass containing all 41 features with type hints
- **`RunnerContext`**: Input dataclass for personal runner characteristics

### Specialized Extractors

Each feature category has a dedicated extractor in `extractors/`:

- `LactateThresholdExtractor` - Estimates LT using HR-pace deflection point method
- `TrainingVolumeExtractor` - Computes weekly mileage and long run statistics
- `TrainingIntensityExtractor` - Calculates zone distribution and workout classification
- `RunningEfficiencyExtractor` - Builds HR-pace model and measures cardiac drift
- `TerrainHandlingExtractor` - Analyzes performance on hills and elevation
- `RaceContextExtractor` - Computes taper quality and recency metrics
- `RunnerProfileExtractor` - Normalizes runner personalization features

## Feature Details

### Lactate Threshold Features
- `lt_heart_rate`: LT heart rate (bpm)
- `lt_pace`: LT pace (min/mile)
- `lt_percent_max_hr`: LT as % of max HR
- `aet_heart_rate`: Aerobic threshold HR (bpm)
- `aet_pace`: Aerobic threshold pace (min/mile)

### Training Volume Features
- `total_weekly_mileage`: Average weekly miles
- `peak_weekly_mileage`: Highest weekly miles
- `long_run_distance`: Average long run distance
- `long_run_percent_weekly`: Long run as % of weekly volume
- `total_runs`: Total number of runs
- `runs_per_week`: Average runs per week
- `mileage_consistency`: Coefficient of variation (lower = more consistent)

### Training Intensity Features
- `zone1_percent` through `zone5_percent`: % time in each HR zone
- `tempo_workout_count`: Number of tempo workouts
- `interval_workout_count`: Number of interval workouts
- `quality_workout_percent`: % of workouts that are quality

### Running Efficiency Features
- `hr_at_easy_pace`: HR at 9:00/mile (bpm)
- `hr_at_marathon_pace`: HR at estimated MP (bpm)
- `cardiac_drift`: HR drift in long runs (bpm/hour)
- `aerobic_decoupling`: HR-pace decoupling (%)
- `hr_variability_coefficient`: Coefficient of variation of HR

### Terrain Handling Features
- `hr_per_grade_uphill`: HR increase per % grade uphill (bpm/%)
- `hr_per_grade_downhill`: HR change per % grade downhill (bpm/%)
- `hill_recovery_rate`: HR recovery after hills (bpm/min)
- `elevation_tolerance`: Performance on hilly vs flat (ratio)

### Race Context Features
- `race_distance_miles`: Target race distance
- `taper_quality_score`: Quality of taper (0-1)
- `days_since_last_hard_effort`: Days since last quality workout

### Runner Personalization Features
- `age_normalized`: Age normalized to 0-1 (peak at 35)
- `sex_encoded`: 0 = F, 1 = M
- `max_hr_normalized`: Max HR normalized to 0-1
- `experience_years`: Years of running experience (normalized)
- `recent_injury_flag`: 0 = no injury, 1 = recent injury
- `training_consistency_score`: Consistency score (0-1)

## Design Decisions

### Training Window
- Default: race_date minus lookback_weeks to race_date minus 7 days
- Excludes final 7 days (taper period) to focus on training adaptations

### HR Zones
Uses 5-zone model based on lactate threshold:
- Zone 1: Recovery (< AET)
- Zone 2: Aerobic (AET to 92% LT)
- Zone 3: Tempo (92-100% LT)
- Zone 4: Threshold (LT to 95% max)
- Zone 5: VO2max+ (> 95% max)

### Workout Classification

**Tempo Workouts:**
- 1-4 mile laps
- Consistent pace (CV < 15%)
- Hard effort (HR ≥ 150 bpm)
- Duration 6-30 minutes per lap
- Pace 5-9 min/mile

**Interval Workouts:**
- Multiple laps of similar distance (within 0.3 miles)
- Fast pace (5-9 min/mile)
- High HR (≥ 140 bpm at activity level)
- At least 2 reps with consistent pace

### Caching Strategy
- **LT estimates**: Cached by activity file set (keyed by MD5 hash)
- **Track point summaries**: Pre-computed in activity cache (10-second buckets)
- **On-demand computation**: Other features computed when requested

## Performance

- Feature extraction: < 5 seconds per runner
- LT calculation (first time): ~10-30 seconds
- LT calculation (cached): < 1 second

## Validation

The `TrainingFeatureVector.validate()` method checks:
- HR features in range 50-220 bpm
- Pace features in range 4-15 min/mile
- Zone percentages sum to ~100%
- Normalized features in range 0-1

## Testing

Run the test script to verify functionality:

```bash
python test_feature_extraction.py
```

This will:
1. Load activities from cache
2. Extract features for a sample runner
3. Display all 41 features
4. Validate feature ranges
5. Test export functions

## Extending the Pipeline

### Adding New Features

1. Create a new extractor in `extractors/`:

```python
class WeatherFeatureExtractor:
    def extract(self, activities, weather_api_key):
        # Match timestamps to weather API
        return {
            'training_temp_avg': ...,
            'training_humidity_avg': ...
        }
```

2. Add fields to `TrainingFeatureVector`:

```python
# Category: Weather
training_temp_avg: Optional[float] = None
training_humidity_avg: Optional[float] = None
```

3. Wire up in `TrainingFeatureExtractor.extract_features()`:

```python
weather_features = self.weather_extractor.extract(activities, api_key)
features.training_temp_avg = weather_features['training_temp_avg']
```

## Dependencies

**Required:**
- numpy
- scipy
- scikit-learn

**Optional:**
- pandas (for `to_dataframe()` method)

## Integration with Activity Cache

The pipeline leverages the existing `ActivityCache` system:

- **Activity queries**: `get_activities_by_date_range()`
- **HR-pace data**: `track_point_summary` table for efficient zone calculations
- **Track points**: Loaded on-demand for terrain and drift analysis

## Notes

- Features are stored in raw units (normalization happens in ML pipeline)
- Missing data handled with reasonable defaults (0 for counts, None for derived metrics)
- All extractors are independent and can be tested separately
- The design supports adding weather/GPX features without major refactoring
