"""
Running Efficiency Feature Extractor
Analyzes HR-pace relationship and cardiac drift
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from feature_engineering.utils import estimate_marathon_pace


class RunningEfficiencyExtractor:
    """Extract running efficiency features"""

    def __init__(self, activity_cache):
        """Initialize with activity cache for track point queries"""
        self.cache = activity_cache

    def extract(
        self,
        activities: List[Dict],
        lt_pace: Optional[float] = None
    ) -> Dict[str, float]:
        """Extract running efficiency features

        Args:
            activities: List of activity dictionaries
            lt_pace: Lactate threshold pace (min/mile)

        Returns:
            Dictionary with efficiency features
        """
        if not activities:
            return self._default_features()

        # Build HR-pace model
        hr_pace_model = self._build_hr_pace_model(activities)

        # Calculate cardiac drift
        cardiac_drift = self._calculate_cardiac_drift(activities)

        # Calculate aerobic decoupling
        aerobic_decoupling = self._calculate_aerobic_decoupling(activities)

        # Calculate HR variability
        hr_variability = self._calculate_hr_variability(activities)

        # Extract features
        features = {}

        # HR at specific paces
        if hr_pace_model is not None:
            features['hr_at_easy_pace'] = self._predict_hr_at_pace(hr_pace_model, 9.0)  # 9:00/mile

            # Marathon pace (estimate from LT if available)
            if lt_pace:
                mp = estimate_marathon_pace(lt_pace)
                features['hr_at_marathon_pace'] = self._predict_hr_at_pace(hr_pace_model, mp)
            else:
                features['hr_at_marathon_pace'] = self._predict_hr_at_pace(hr_pace_model, 8.5)
        else:
            features['hr_at_easy_pace'] = None
            features['hr_at_marathon_pace'] = None

        # Cardiac drift and decoupling
        features['cardiac_drift'] = cardiac_drift
        features['aerobic_decoupling'] = aerobic_decoupling
        features['hr_variability_coefficient'] = hr_variability

        return features

    def _build_hr_pace_model(self, activities: List[Dict]) -> Optional[tuple]:
        """Build polynomial HR-pace model

        Returns:
            Tuple of (model, poly_features) or None
        """
        import sqlite3
        # Collect HR-pace data from track point summaries
        hr_data = []
        pace_data = []

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        for activity in activities:
            file_name = activity['file_name']

            cursor.execute("""
                SELECT avg_hr, avg_pace
                FROM track_point_summary
                WHERE file_name = ? AND avg_hr IS NOT NULL AND avg_pace IS NOT NULL
            """, (file_name,))

            for row in cursor.fetchall():
                hr = row[0]
                pace = row[1]

                # Filter reasonable values
                if 50 <= hr <= 220 and 4 <= pace <= 15:
                    hr_data.append(hr)
                    pace_data.append(pace)

        conn.close()

        if len(hr_data) < 50:
            return None

        # Build polynomial model (degree 2)
        try:
            X = np.array(pace_data).reshape(-1, 1)
            y = np.array(hr_data)

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, y)

            return (model, poly)

        except Exception:
            return None

    def _predict_hr_at_pace(self, hr_pace_model: tuple, pace: float) -> Optional[float]:
        """Predict HR at given pace using model"""
        try:
            model, poly = hr_pace_model
            X = np.array([[pace]])
            X_poly = poly.transform(X)
            hr = model.predict(X_poly)[0]

            # Sanity check
            if 50 <= hr <= 220:
                return hr
            return None

        except Exception:
            return None

    def _calculate_cardiac_drift(self, activities: List[Dict]) -> float:
        """Calculate cardiac drift in long runs (bpm/hour)

        Compares first half vs second half HR
        """
        import sqlite3
        drift_values = []

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        for activity in activities:
            # Only analyze long runs (> 60 minutes)
            duration_min = activity.get('moving_time_seconds', 0) / 60
            if duration_min < 60:
                continue

            file_name = activity['file_name']

            # Get track point summary
            cursor.execute("""
                SELECT time_bucket, avg_hr
                FROM track_point_summary
                WHERE file_name = ? AND avg_hr IS NOT NULL
                ORDER BY time_bucket
            """, (file_name,))

            hrs = [row[1] for row in cursor.fetchall()]

            if len(hrs) < 10:
                continue

            # Split into first and second half
            mid = len(hrs) // 2
            first_half_hr = np.mean(hrs[:mid])
            second_half_hr = np.mean(hrs[mid:])

            # Calculate drift (bpm/hour)
            drift = (second_half_hr - first_half_hr) / (duration_min / 60)
            drift_values.append(drift)

        conn.close()

        if drift_values:
            return np.mean(drift_values)
        return 0.0

    def _calculate_aerobic_decoupling(self, activities: List[Dict]) -> float:
        """Calculate aerobic decoupling (%)

        Measures HR-pace decoupling in steady efforts
        """
        import sqlite3
        decoupling_values = []

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        for activity in activities:
            # Only analyze medium-long runs (40-90 minutes)
            duration_min = activity.get('moving_time_seconds', 0) / 60
            if not (40 <= duration_min <= 90):
                continue

            file_name = activity['file_name']

            # Get HR and pace data
            cursor.execute("""
                SELECT time_bucket, avg_hr, avg_pace
                FROM track_point_summary
                WHERE file_name = ? AND avg_hr IS NOT NULL AND avg_pace IS NOT NULL
                ORDER BY time_bucket
            """, (file_name,))

            data = [(row[1], row[2]) for row in cursor.fetchall()]

            if len(data) < 10:
                continue

            # Split into first and second half
            mid = len(data) // 2
            first_half = data[:mid]
            second_half = data[mid:]

            # Calculate HR/pace ratio for each half
            first_ratio = np.mean([hr / pace for hr, pace in first_half if pace > 0])
            second_ratio = np.mean([hr / pace for hr, pace in second_half if pace > 0])

            # Decoupling percentage
            if first_ratio > 0:
                decoupling = ((second_ratio - first_ratio) / first_ratio) * 100
                decoupling_values.append(decoupling)

        conn.close()

        if decoupling_values:
            return np.mean(decoupling_values)
        return 0.0

    def _calculate_hr_variability(self, activities: List[Dict]) -> float:
        """Calculate coefficient of variation of HR across all activities"""
        import sqlite3
        all_hr = []

        conn = sqlite3.connect(self.cache.db_path)
        cursor = conn.cursor()

        for activity in activities:
            file_name = activity['file_name']

            cursor.execute("""
                SELECT avg_hr
                FROM track_point_summary
                WHERE file_name = ? AND avg_hr IS NOT NULL
            """, (file_name,))

            for row in cursor.fetchall():
                all_hr.append(row[0])

        conn.close()

        if len(all_hr) >= 10:
            mean_hr = np.mean(all_hr)
            std_hr = np.std(all_hr)
            if mean_hr > 0:
                return std_hr / mean_hr

        return 0.0

    def _default_features(self) -> Dict[str, float]:
        """Return default features when no activities"""
        return {
            'hr_at_easy_pace': None,
            'hr_at_marathon_pace': None,
            'cardiac_drift': 0.0,
            'aerobic_decoupling': 0.0,
            'hr_variability_coefficient': 0.0,
        }
