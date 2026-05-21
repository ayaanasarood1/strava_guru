"""
Race Data Collector
Collects training features and race results for ML training
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from activity_cache import ActivityCache
from feature_engineering import TrainingFeatureExtractor, RunnerContext


@dataclass
class RaceResult:
    """A race result with training features"""
    race_id: str
    runner_id: str
    race_date: datetime
    race_distance_miles: float
    actual_time_minutes: float  # Target variable

    # Runner context
    age: int
    sex: str
    max_hr: int
    experience_years: Optional[int] = None

    # Training window
    lookback_weeks: int = 12

    # Features (populated after extraction)
    features: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        if isinstance(d['race_date'], datetime):
            d['race_date'] = d['race_date'].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'RaceResult':
        """Create from dictionary"""
        if isinstance(d['race_date'], str):
            d['race_date'] = datetime.fromisoformat(d['race_date'])
        return cls(**d)


class RaceDataCollector:
    """Collects and manages race training data"""

    def __init__(self, cache: ActivityCache, data_dir: Path = None):
        """Initialize data collector

        Args:
            cache: ActivityCache instance
            data_dir: Directory to store collected data
        """
        self.cache = cache
        self.data_dir = data_dir or Path.home() / ".strava_guru_cache" / "race_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = TrainingFeatureExtractor(cache)
        self.race_results: List[RaceResult] = []

    def add_race_result(
        self,
        race_id: str,
        runner_id: str,
        race_date: datetime,
        race_distance_miles: float,
        actual_time_minutes: float,
        runner_context: RunnerContext,
        lookback_weeks: int = 12
    ) -> RaceResult:
        """Add a race result and extract features

        Args:
            race_id: Unique race identifier
            runner_id: Runner identifier
            race_date: Date of race
            race_distance_miles: Race distance
            actual_time_minutes: Actual finish time in minutes
            runner_context: Runner personal characteristics
            lookback_weeks: Training lookback period

        Returns:
            RaceResult with extracted features
        """
        print(f"\nAdding race result: {race_id}")
        print(f"  Runner: {runner_id}")
        print(f"  Date: {race_date.strftime('%Y-%m-%d')}")
        print(f"  Distance: {race_distance_miles} miles")
        print(f"  Time: {self._format_time(actual_time_minutes)}")

        # Create race result
        race_result = RaceResult(
            race_id=race_id,
            runner_id=runner_id,
            race_date=race_date,
            race_distance_miles=race_distance_miles,
            actual_time_minutes=actual_time_minutes,
            age=runner_context.age,
            sex=runner_context.sex,
            max_hr=runner_context.max_hr,
            experience_years=runner_context.experience_years,
            lookback_weeks=lookback_weeks
        )

        # Extract features
        try:
            features = self.extractor.extract_features(
                runner_id=runner_id,
                race_date=race_date,
                lookback_weeks=lookback_weeks,
                race_distance_miles=race_distance_miles,
                runner_context=runner_context
            )

            race_result.features = features.to_dict()
            print(f"  ✓ Extracted {features.feature_count()} features")

        except Exception as e:
            print(f"  ✗ Feature extraction failed: {e}")
            race_result.features = None

        self.race_results.append(race_result)
        return race_result

    def load_from_json(self, json_file: Path):
        """Load race results from JSON file

        Expected format:
        [
            {
                "race_id": "boston_2024_runner1",
                "runner_id": "runner1",
                "race_date": "2024-04-15",
                "race_distance_miles": 26.2,
                "actual_time_minutes": 195.5,
                "age": 35,
                "sex": "M",
                "max_hr": 185,
                "experience_years": 8
            },
            ...
        ]
        """
        print(f"\nLoading race results from {json_file}")

        with open(json_file) as f:
            data = json.load(f)

        for item in data:
            # Parse date
            race_date = datetime.fromisoformat(item['race_date'])

            # Create runner context
            runner_context = RunnerContext(
                age=item['age'],
                sex=item['sex'],
                max_hr=item['max_hr'],
                experience_years=item.get('experience_years'),
                resting_hr=item.get('resting_hr'),
                recent_injury_flag=item.get('recent_injury_flag', False)
            )

            # Add race result
            self.add_race_result(
                race_id=item['race_id'],
                runner_id=item['runner_id'],
                race_date=race_date,
                race_distance_miles=item['race_distance_miles'],
                actual_time_minutes=item['actual_time_minutes'],
                runner_context=runner_context,
                lookback_weeks=item.get('lookback_weeks', 12)
            )

        print(f"\n✓ Loaded {len(self.race_results)} race results")

    def save_dataset(self, filename: str = "race_dataset.json"):
        """Save collected dataset to JSON"""
        output_file = self.data_dir / filename

        data = [result.to_dict() for result in self.race_results]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n✓ Saved dataset to {output_file}")
        print(f"  Total samples: {len(self.race_results)}")

        return output_file

    def load_dataset(self, filename: str = "race_dataset.json") -> List[RaceResult]:
        """Load previously saved dataset"""
        input_file = self.data_dir / filename

        if not input_file.exists():
            print(f"Dataset file not found: {input_file}")
            return []

        with open(input_file) as f:
            data = json.load(f)

        self.race_results = [RaceResult.from_dict(item) for item in data]

        print(f"✓ Loaded {len(self.race_results)} race results from {input_file}")
        return self.race_results

    def get_training_data(self) -> Tuple[List[Dict], List[float]]:
        """Get features (X) and targets (y) for ML training

        Returns:
            Tuple of (feature_dicts, target_times)
        """
        X = []
        y = []

        for result in self.race_results:
            if result.features is not None:
                X.append(result.features)
                y.append(result.actual_time_minutes)

        print(f"\nTraining data: {len(X)} samples")
        if X:
            # Count non-None features
            sample = X[0]
            non_none = sum(1 for v in sample.values() if v is not None)
            print(f"  Features per sample: {non_none} / {len(sample)}")

        return X, y

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about collected dataset"""
        if not self.race_results:
            return {}

        stats = {
            'total_samples': len(self.race_results),
            'with_features': sum(1 for r in self.race_results if r.features),
            'runners': len(set(r.runner_id for r in self.race_results)),
            'race_distances': {},
            'time_range': {
                'min': min(r.actual_time_minutes for r in self.race_results),
                'max': max(r.actual_time_minutes for r in self.race_results),
                'mean': sum(r.actual_time_minutes for r in self.race_results) / len(self.race_results)
            }
        }

        # Count by distance
        for result in self.race_results:
            dist = result.race_distance_miles
            stats['race_distances'][dist] = stats['race_distances'].get(dist, 0) + 1

        return stats

    def _format_time(self, minutes: float) -> str:
        """Format time in HH:MM:SS"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        secs = int((minutes % 1) * 60)

        if hours > 0:
            return f"{hours}:{mins:02d}:{secs:02d}"
        else:
            return f"{mins}:{secs:02d}"
