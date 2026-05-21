"""
Lactate Threshold Feature Extractor
Wraps existing LT analyzer with caching
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle
import hashlib

from lactate_threshold_analyzer import LactateThresholdAnalyzer, LactateThresholdEstimate
from activity_cache import ActivityCache


class LactateThresholdExtractor:
    """Extract lactate threshold features with caching"""

    def __init__(self, cache_dir: Path = None):
        """Initialize with cache directory"""
        self.cache_dir = cache_dir or Path.home() / ".strava_guru_cache" / "lt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        activity_cache: ActivityCache,
        activities: list,
        runner_max_hr: Optional[int] = None
    ) -> Dict[str, float]:
        """Extract lactate threshold features

        Args:
            activity_cache: ActivityCache instance
            activities: List of activity dictionaries from cache
            runner_max_hr: Runner's max heart rate (optional)

        Returns:
            Dictionary with LT features
        """
        # Check cache first
        cache_key = self._get_cache_key(activities)
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        # Compute LT estimate
        lt_estimate = self._compute_lt(activity_cache, activities, runner_max_hr)

        # Extract features
        features = self._extract_features(lt_estimate, runner_max_hr)

        # Cache results
        self._save_to_cache(cache_key, features)

        return features

    def _get_cache_key(self, activities: list) -> str:
        """Generate cache key from activity set"""
        # Use sorted file names to create deterministic key
        file_names = sorted([a['file_name'] for a in activities])
        key_str = '|'.join(file_names)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Load cached LT features"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, features: Dict[str, float]):
        """Save LT features to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass

    def _compute_lt(
        self,
        activity_cache: ActivityCache,
        activities: list,
        runner_max_hr: Optional[int]
    ) -> LactateThresholdEstimate:
        """Compute LT estimate using existing analyzer

        This is a simplified version that uses track point summary data
        instead of loading full activities.
        """
        try:
            import sqlite3
            # Get HR-pace data from track point summaries
            hr_pace_data = []
            conn = sqlite3.connect(activity_cache.db_path)
            cursor = conn.cursor()

            for activity in activities:
                file_name = activity['file_name']

                # Query track point summary
                cursor.execute("""
                    SELECT avg_hr, avg_pace
                    FROM track_point_summary
                    WHERE file_name = ? AND avg_hr IS NOT NULL AND avg_pace IS NOT NULL
                """, (file_name,))

                for row in cursor.fetchall():
                    hr_pace_data.append((row[0], row[1]))

            conn.close()

            if not hr_pace_data or len(hr_pace_data) < 100:
                return self._default_estimate()

            # Create analyzer and use simplified estimation
            # Since we don't have full ActivityStats objects, we'll use a simpler method
            analyzer = LactateThresholdAnalyzer()
            analyzer.hr_pace_data = hr_pace_data

            # Use direct estimation from HR-pace data
            estimate = self._estimate_from_hr_pace_data(hr_pace_data, runner_max_hr)

            # Set max HR if provided
            if runner_max_hr and estimate.lt_heart_rate:
                estimate.max_heart_rate = runner_max_hr
                estimate.lt_percent_max = estimate.lt_heart_rate / runner_max_hr

            return estimate

        except Exception as e:
            print(f"Error computing LT: {e}")
            return self._default_estimate()

    def _estimate_from_hr_pace_data(
        self,
        hr_pace_data: list,
        runner_max_hr: Optional[int]
    ) -> LactateThresholdEstimate:
        """Estimate LT from HR-pace data using deflection point method"""
        import numpy as np

        try:
            # Convert to array
            hr_pace_array = np.array(hr_pace_data)

            # Bin data by pace
            pace_bins = np.linspace(5, 12, 30)
            binned_hr = []
            binned_pace = []

            for i in range(len(pace_bins) - 1):
                mask = (hr_pace_array[:, 1] >= pace_bins[i]) & (hr_pace_array[:, 1] < pace_bins[i+1])
                if np.sum(mask) > 10:
                    binned_hr.append(np.median(hr_pace_array[mask, 0]))
                    binned_pace.append(np.median(hr_pace_array[mask, 1]))

            if len(binned_hr) < 10:
                return self._default_estimate()

            binned_hr = np.array(binned_hr)
            binned_pace = np.array(binned_pace)

            # Sort by pace
            sort_idx = np.argsort(binned_pace)
            binned_pace = binned_pace[sort_idx]
            binned_hr = binned_hr[sort_idx]

            # Find deflection point
            hr_rate = np.diff(binned_hr) / np.diff(binned_pace)
            hr_accel = np.diff(hr_rate)
            deflection_idx = np.argmax(np.abs(hr_accel)) + 1

            lt_hr = binned_hr[deflection_idx]
            lt_pace = binned_pace[deflection_idx]

            # Estimate aerobic threshold (80% of LT HR)
            aet_hr = lt_hr * 0.82
            aet_pace = lt_pace + 1.0

            # Create estimate
            estimate = LactateThresholdEstimate(
                lt_heart_rate=lt_hr,
                lt_pace=lt_pace,
                confidence=min(len(binned_hr) / 20, 1.0) * 0.7,
                method="HR Deflection Point",
                max_heart_rate=runner_max_hr,
                aerobic_threshold_hr=aet_hr,
                aerobic_threshold_pace=aet_pace
            )

            if runner_max_hr:
                estimate.lt_percent_max = lt_hr / runner_max_hr

            return estimate

        except Exception as e:
            print(f"Error in LT estimation: {e}")
            return self._default_estimate()

    def _default_estimate(self) -> LactateThresholdEstimate:
        """Return default LT estimate when computation fails"""
        return LactateThresholdEstimate(
            lt_heart_rate=0,
            lt_pace=0,
            confidence=0,
            method="default"
        )

    def _extract_features(
        self,
        estimate: LactateThresholdEstimate,
        runner_max_hr: Optional[int]
    ) -> Dict[str, float]:
        """Extract features from LT estimate"""
        features = {}

        # LT heart rate and pace
        features['lt_heart_rate'] = estimate.lt_heart_rate if estimate.lt_heart_rate else None
        features['lt_pace'] = estimate.lt_pace if estimate.lt_pace else None

        # LT as percent of max HR
        if estimate.lt_heart_rate and runner_max_hr:
            features['lt_percent_max_hr'] = estimate.lt_heart_rate / runner_max_hr
        elif estimate.lt_percent_max:
            features['lt_percent_max_hr'] = estimate.lt_percent_max
        else:
            features['lt_percent_max_hr'] = None

        # Aerobic threshold (typically ~80-85% of LT HR)
        if estimate.aerobic_threshold_hr:
            features['aet_heart_rate'] = estimate.aerobic_threshold_hr
        elif estimate.lt_heart_rate:
            features['aet_heart_rate'] = estimate.lt_heart_rate * 0.82
        else:
            features['aet_heart_rate'] = None

        # Aerobic threshold pace
        if estimate.aerobic_threshold_pace:
            features['aet_pace'] = estimate.aerobic_threshold_pace
        elif estimate.lt_pace:
            features['aet_pace'] = estimate.lt_pace + 1.0  # ~1 min/mile slower
        else:
            features['aet_pace'] = None

        return features
