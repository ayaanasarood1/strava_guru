#!/usr/bin/env python3
"""
Lactate Threshold Analyzer
Predicts lactate threshold using HR, pace, and activity data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from activity_analyzer import ActivityAnalyzer, ActivityStats, format_pace


@dataclass
class LactateThresholdEstimate:
    """Lactate threshold estimation results"""
    lt_heart_rate: float  # Estimated LT heart rate (bpm)
    lt_pace: float  # Estimated LT pace (min/mile)
    confidence: float  # Confidence score (0-1)
    method: str  # Method used for estimation

    # Additional metrics
    max_heart_rate: Optional[float] = None
    lt_percent_max: Optional[float] = None
    aerobic_threshold_hr: Optional[float] = None
    aerobic_threshold_pace: Optional[float] = None

    # Training zones
    zone1_hr: Optional[Tuple[float, float]] = None
    zone2_hr: Optional[Tuple[float, float]] = None
    zone3_hr: Optional[Tuple[float, float]] = None
    zone4_hr: Optional[Tuple[float, float]] = None
    zone5_hr: Optional[Tuple[float, float]] = None


class LactateThresholdAnalyzer:
    """Analyzes multiple activities to estimate lactate threshold"""

    def __init__(self):
        self.activities: List[ActivityStats] = []
        self.hr_pace_data: List[Tuple[float, float]] = []  # (HR, GAP) pairs
        self.use_gap = True  # Use Grade Adjusted Pace by default

    def load_activities(self, activities_dir: Path, limit: Optional[int] = None):
        """Load multiple activities for analysis"""
        print(f"Loading activities from {activities_dir}...")

        activity_files = sorted(activities_dir.glob("*.fit.gz"))[:limit] if limit else sorted(activities_dir.glob("*.fit.gz"))

        analyzer = ActivityAnalyzer()
        loaded = 0

        for file_path in activity_files:
            try:
                stats = analyzer.analyze_file(file_path)
                if stats and stats.avg_heart_rate and stats.avg_pace > 0:
                    self.activities.append(stats)
                    loaded += 1
            except Exception as e:
                continue

        print(f"Loaded {loaded} activities with HR data")
        self._extract_hr_pace_data()

    def _extract_hr_pace_data(self):
        """Extract HR-pace pairs from all track points, using GAP for elevation adjustment"""
        for activity in self.activities:
            if not activity.track_points:
                continue

            for i, point in enumerate(activity.track_points):
                if not (point.heart_rate and point.speed and point.speed > 0):
                    continue

                # Convert speed (m/s) to pace (min/mile)
                pace = 26.8224 / point.speed  # 1609.34 / 60 = 26.8224

                # Calculate GAP (Grade Adjusted Pace) if elevation data available
                if i > 0 and point.elevation is not None and activity.track_points[i-1].elevation is not None:
                    prev_point = activity.track_points[i-1]

                    # Calculate grade
                    if point.distance and prev_point.distance:
                        distance_delta = point.distance - prev_point.distance
                        elev_delta = point.elevation - prev_point.elevation

                        if distance_delta > 0:
                            grade = elev_delta / distance_delta

                            # Adjust pace for grade (uphill = slower equivalent, downhill = faster)
                            # Rule of thumb: 10-12 seconds per mile per 1% grade
                            grade_percent = grade * 100
                            pace_adjustment = grade_percent * 0.20  # 12 sec/mile per 1% grade = 0.2 min/mile
                            gap = pace - pace_adjustment  # Subtract because slower uphill = faster flat equivalent

                            # Use GAP instead of raw pace
                            pace = gap

                if 4 < pace < 15:  # Filter reasonable paces (4-15 min/mile)
                    self.hr_pace_data.append((point.heart_rate, pace))

    def analyze_elevation_profile(self):
        """Analyze how much elevation gain is in the dataset"""
        flat_activities = []  # < 50 ft/mile
        rolling_activities = []  # 50-100 ft/mile
        hilly_activities = []  # > 100 ft/mile

        for activity in self.activities:
            distance_mi = activity.total_distance * ActivityAnalyzer.METERS_TO_MILES
            if distance_mi < 1:
                continue

            elev_gain_ft = activity.elevation_gain * ActivityAnalyzer.METERS_TO_FEET
            elev_per_mile = elev_gain_ft / distance_mi

            if elev_per_mile < 50:
                flat_activities.append(activity)
            elif elev_per_mile < 100:
                rolling_activities.append(activity)
            else:
                hilly_activities.append(activity)

        print(f"\nElevation Profile of Activities:")
        print(f"  Flat runs (< 50 ft/mi):    {len(flat_activities):4d} activities ({len(flat_activities)/len(self.activities)*100:.1f}%)")
        print(f"  Rolling (50-100 ft/mi):    {len(rolling_activities):4d} activities ({len(rolling_activities)/len(self.activities)*100:.1f}%)")
        print(f"  Hilly (> 100 ft/mi):       {len(hilly_activities):4d} activities ({len(hilly_activities)/len(self.activities)*100:.1f}%)")

        return flat_activities, rolling_activities, hilly_activities

    def estimate_lactate_threshold(self) -> LactateThresholdEstimate:
        """
        Estimate lactate threshold using multiple methods and return best estimate
        """
        if len(self.activities) < 5:
            raise ValueError("Need at least 5 activities with HR data for reliable LT estimation")

        # Analyze elevation profile
        self.analyze_elevation_profile()

        print("\n" + "="*60)
        print("Estimating Lactate Threshold (using GAP for elevation adjustment)...")
        print("="*60)

        # Method 1: HR Deflection Point (Conconi-like)
        lt1 = self._estimate_hr_deflection_point()
        print(f"Method 1 (HR Deflection): LT HR={lt1.lt_heart_rate:.0f} bpm, Pace={format_pace(lt1.lt_pace)}")

        # Method 2: Critical Pace Analysis
        lt2 = self._estimate_critical_pace()
        print(f"Method 2 (Critical Pace): LT HR={lt2.lt_heart_rate:.0f} bpm, Pace={format_pace(lt2.lt_pace)}")

        # Method 3: Tempo Run Detection
        lt3 = self._estimate_from_tempo_runs()
        print(f"Method 3 (Tempo Analysis): LT HR={lt3.lt_heart_rate:.0f} bpm, Pace={format_pace(lt3.lt_pace)}")

        # Method 4: Physiological Model (85-90% max HR)
        lt4 = self._estimate_physiological_model()
        print(f"Method 4 (Physio Model): LT HR={lt4.lt_heart_rate:.0f} bpm, Pace={format_pace(lt4.lt_pace)}")

        # Weighted average based on confidence
        estimates = [lt1, lt2, lt3, lt4]
        total_confidence = sum(e.confidence for e in estimates)

        weighted_hr = sum(e.lt_heart_rate * e.confidence for e in estimates) / total_confidence
        weighted_pace = sum(e.lt_pace * e.confidence for e in estimates) / total_confidence
        avg_confidence = total_confidence / len(estimates)

        # Get max HR
        max_hr = max(a.max_heart_rate for a in self.activities if a.max_heart_rate)

        # Calculate final estimate
        final_estimate = LactateThresholdEstimate(
            lt_heart_rate=weighted_hr,
            lt_pace=weighted_pace,
            confidence=avg_confidence,
            method="Ensemble (weighted average of 4 methods)",
            max_heart_rate=max_hr,
            lt_percent_max=(weighted_hr / max_hr * 100) if max_hr else None,
            aerobic_threshold_hr=weighted_hr * 0.85,  # AeT ~85% of LT HR
            aerobic_threshold_pace=weighted_pace * 1.15  # AeT pace ~15% slower
        )

        # Calculate training zones
        final_estimate.zone1_hr = (max_hr * 0.50, max_hr * 0.60)  # Recovery
        final_estimate.zone2_hr = (max_hr * 0.60, weighted_hr * 0.85)  # Aerobic
        final_estimate.zone3_hr = (weighted_hr * 0.85, weighted_hr * 0.95)  # Tempo
        final_estimate.zone4_hr = (weighted_hr * 0.95, weighted_hr * 1.05)  # Threshold
        final_estimate.zone5_hr = (weighted_hr * 1.05, max_hr)  # VO2max/Anaerobic

        print("="*60)
        print(f"\n🎯 FINAL ESTIMATE:")
        print(f"   Lactate Threshold HR: {final_estimate.lt_heart_rate:.0f} bpm ({final_estimate.lt_percent_max:.0f}% of max)")
        print(f"   Lactate Threshold Pace: {format_pace(final_estimate.lt_pace)} /mi")
        print(f"   Confidence: {final_estimate.confidence:.1%}")

        return final_estimate

    def _estimate_hr_deflection_point(self) -> LactateThresholdEstimate:
        """
        Find the HR deflection point where HR starts rising faster than pace increases
        This mimics the Conconi test
        """
        if not self.hr_pace_data:
            return LactateThresholdEstimate(0, 0, 0, "HR Deflection")

        # Bin data by pace
        hr_pace_array = np.array(self.hr_pace_data)
        pace_bins = np.linspace(5, 12, 30)  # 5-12 min/mile in 30 bins

        binned_hr = []
        binned_pace = []

        for i in range(len(pace_bins) - 1):
            mask = (hr_pace_array[:, 1] >= pace_bins[i]) & (hr_pace_array[:, 1] < pace_bins[i+1])
            if np.sum(mask) > 10:  # Need enough samples
                binned_hr.append(np.median(hr_pace_array[mask, 0]))
                binned_pace.append(np.median(hr_pace_array[mask, 1]))

        if len(binned_hr) < 10:
            return LactateThresholdEstimate(0, 0, 0, "HR Deflection")

        binned_hr = np.array(binned_hr)
        binned_pace = np.array(binned_pace)

        # Sort by pace (faster to slower)
        sort_idx = np.argsort(binned_pace)
        binned_pace = binned_pace[sort_idx]
        binned_hr = binned_hr[sort_idx]

        # Find deflection point: where second derivative changes most
        # Calculate rate of change of HR per pace unit
        hr_rate = np.diff(binned_hr) / np.diff(binned_pace)

        # Second derivative (acceleration)
        hr_accel = np.diff(hr_rate)

        # Find maximum acceleration point (deflection)
        deflection_idx = np.argmax(np.abs(hr_accel)) + 1

        lt_hr = binned_hr[deflection_idx]
        lt_pace = binned_pace[deflection_idx]

        # Confidence based on data quality
        confidence = min(len(binned_hr) / 20, 1.0) * 0.7

        return LactateThresholdEstimate(lt_hr, lt_pace, confidence, "HR Deflection Point")

    def _estimate_critical_pace(self) -> LactateThresholdEstimate:
        """
        Estimate LT using critical velocity/pace model
        LT is approximately at critical pace
        """
        # Use activities of different distances and find critical pace
        # CP is the pace that can be sustained for ~40-60 min

        tempo_activities = [
            a for a in self.activities
            if 20 < a.moving_time.total_seconds() / 60 < 80  # 20-80 min runs
            and a.avg_heart_rate
            and 6 < a.avg_pace < 11  # Reasonable pace
        ]

        if len(tempo_activities) < 3:
            return LactateThresholdEstimate(0, 0, 0, "Critical Pace")

        # Sort by pace
        tempo_activities.sort(key=lambda a: a.avg_pace)

        # Critical pace is around median pace for 40-60 min efforts
        paces = [a.avg_pace for a in tempo_activities]
        hrs = [a.avg_heart_rate for a in tempo_activities]

        # Use weighted median (favor longer runs)
        weights = [a.moving_time.total_seconds() for a in tempo_activities]

        lt_pace = np.average(paces, weights=weights)
        lt_hr = np.average(hrs, weights=weights)

        confidence = min(len(tempo_activities) / 10, 1.0) * 0.8

        return LactateThresholdEstimate(lt_hr, lt_pace, confidence, "Critical Pace Model")

    def _estimate_from_tempo_runs(self) -> LactateThresholdEstimate:
        """
        Identify tempo runs (sustained hard effort) and estimate LT from those
        """
        # Look for activities with:
        # 1. Consistent pace (low variability)
        # 2. Hard effort (high HR)
        # 3. Duration 20-60 minutes

        tempo_candidates = []

        for activity in self.activities:
            if not activity.laps or not activity.avg_heart_rate:
                continue

            duration_min = activity.moving_time.total_seconds() / 60
            if duration_min < 20 or duration_min > 70:
                continue

            # Check pace consistency
            if len(activity.laps) >= 3:
                lap_paces = [lap.pace for lap in activity.laps if lap.pace > 0]
                if lap_paces:
                    pace_cv = np.std(lap_paces) / np.mean(lap_paces)

                    # Low variability = steady effort
                    if pace_cv < 0.15:  # Less than 15% variation
                        tempo_candidates.append({
                            'hr': activity.avg_heart_rate,
                            'pace': activity.avg_pace,
                            'duration': duration_min,
                            'consistency': 1 - pace_cv
                        })

        if len(tempo_candidates) < 2:
            return LactateThresholdEstimate(0, 0, 0, "Tempo Analysis")

        # Weight by consistency and duration
        weights = [t['consistency'] * min(t['duration'] / 40, 1.0) for t in tempo_candidates]
        lt_hr = np.average([t['hr'] for t in tempo_candidates], weights=weights)
        lt_pace = np.average([t['pace'] for t in tempo_candidates], weights=weights)

        confidence = min(len(tempo_candidates) / 8, 1.0) * 0.85

        return LactateThresholdEstimate(lt_hr, lt_pace, confidence, "Tempo Run Analysis")

    def _estimate_physiological_model(self) -> LactateThresholdEstimate:
        """
        Use physiological guidelines: LT is typically 85-90% of max HR
        """
        max_hr = max(a.max_heart_rate for a in self.activities if a.max_heart_rate)

        # LT typically at 88% of max HR
        lt_hr = max_hr * 0.88

        # Find pace at this HR using regression
        if len(self.hr_pace_data) > 100:
            hr_pace_array = np.array(self.hr_pace_data)

            # Filter to HR around LT estimate
            mask = (hr_pace_array[:, 0] > lt_hr - 10) & (hr_pace_array[:, 0] < lt_hr + 10)
            if np.sum(mask) > 20:
                lt_pace = np.median(hr_pace_array[mask, 1])
            else:
                # Fallback: fit curve to all data
                X = hr_pace_array[:, 0].reshape(-1, 1)
                y = hr_pace_array[:, 1]

                # Polynomial regression
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)

                lt_pace = model.predict(poly.transform([[lt_hr]]))[0]
        else:
            lt_pace = 8.0  # Fallback

        confidence = 0.75  # Moderate confidence in physiological model

        return LactateThresholdEstimate(lt_hr, lt_pace, confidence, "Physiological Model (88% max HR)")

    def visualize_analysis(self, lt_estimate: LactateThresholdEstimate, output_path: Optional[Path] = None):
        """Create comprehensive visualization of LT analysis"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Lactate Threshold Analysis', fontsize=16, fontweight='bold')

        # 1. HR vs Pace Scatter with LT line
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_hr_pace_relationship(ax1, lt_estimate)

        # 2. Training Zones
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_training_zones(ax2, lt_estimate)

        # 3. Activity Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_activity_distribution(ax3, lt_estimate)

        # 4. HR Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_hr_distribution(ax4, lt_estimate)

        # 5. Summary Stats
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_summary_stats(ax5, lt_estimate)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def _plot_hr_pace_relationship(self, ax, lt_estimate):
        """Plot HR vs Pace scatter with LT threshold"""
        if not self.hr_pace_data:
            return

        hr_pace_array = np.array(self.hr_pace_data)

        # Sample if too many points
        if len(hr_pace_array) > 5000:
            idx = np.random.choice(len(hr_pace_array), 5000, replace=False)
            hr_pace_array = hr_pace_array[idx]

        # Scatter plot
        ax.scatter(hr_pace_array[:, 1], hr_pace_array[:, 0],
                  alpha=0.3, s=10, c='#4AB4DC', label='All data points')

        # LT line
        ax.axhline(y=lt_estimate.lt_heart_rate, color='red',
                  linestyle='--', linewidth=2, label=f'LT HR ({lt_estimate.lt_heart_rate:.0f} bpm)')
        ax.axvline(x=lt_estimate.lt_pace, color='orange',
                  linestyle='--', linewidth=2, label=f'LT Pace ({format_pace(lt_estimate.lt_pace)})')

        # Add aerobic threshold
        if lt_estimate.aerobic_threshold_hr:
            ax.axhline(y=lt_estimate.aerobic_threshold_hr, color='green',
                      linestyle=':', linewidth=2, alpha=0.7, label='Aerobic Threshold')

        ax.set_xlabel('GAP - Grade Adjusted Pace (min/mile)', fontweight='bold')
        ax.set_ylabel('Heart Rate (bpm)', fontweight='bold')
        ax.set_title('Heart Rate vs GAP Relationship (Elevation Adjusted)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Faster pace on right

    def _plot_training_zones(self, ax, lt_estimate):
        """Plot training zone chart"""
        ax.axis('off')

        zones = [
            ('Zone 1\nRecovery', lt_estimate.zone1_hr, '#90EE90'),
            ('Zone 2\nAerobic', lt_estimate.zone2_hr, '#87CEEB'),
            ('Zone 3\nTempo', lt_estimate.zone3_hr, '#FFD700'),
            ('Zone 4\nThreshold', lt_estimate.zone4_hr, '#FFA500'),
            ('Zone 5\nVO2max', lt_estimate.zone5_hr, '#FF6347'),
        ]

        y_pos = 0.9

        ax.text(0.5, 0.95, 'Training Zones Based on Lactate Threshold',
               ha='center', fontsize=13, fontweight='bold', transform=ax.transAxes)

        for zone_name, hr_range, color in zones:
            if hr_range:
                ax.add_patch(plt.Rectangle((0.1, y_pos - 0.08), 0.8, 0.12,
                                          facecolor=color, alpha=0.6, transform=ax.transAxes))
                ax.text(0.15, y_pos - 0.02, zone_name,
                       fontsize=10, fontweight='bold', transform=ax.transAxes, va='center')
                ax.text(0.75, y_pos - 0.02, f'{hr_range[0]:.0f} - {hr_range[1]:.0f} bpm',
                       fontsize=10, transform=ax.transAxes, va='center', ha='right')
                y_pos -= 0.16

    def _plot_activity_distribution(self, ax, lt_estimate):
        """Plot distribution of activities by HR zone"""
        zone_counts = [0, 0, 0, 0, 0]
        zone_labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']

        for activity in self.activities:
            if not activity.avg_heart_rate:
                continue

            hr = activity.avg_heart_rate
            if hr < lt_estimate.zone1_hr[1]:
                zone_counts[0] += 1
            elif hr < lt_estimate.zone2_hr[1]:
                zone_counts[1] += 1
            elif hr < lt_estimate.zone3_hr[1]:
                zone_counts[2] += 1
            elif hr < lt_estimate.zone4_hr[1]:
                zone_counts[3] += 1
            else:
                zone_counts[4] += 1

        colors = ['#90EE90', '#87CEEB', '#FFD700', '#FFA500', '#FF6347']
        ax.bar(zone_labels, zone_counts, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Training Zone', fontweight='bold')
        ax.set_ylabel('Number of Activities', fontweight='bold')
        ax.set_title('Activity Distribution by Zone', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_hr_distribution(self, ax, lt_estimate):
        """Plot heart rate distribution histogram"""
        all_hrs = [a.avg_heart_rate for a in self.activities if a.avg_heart_rate]

        ax.hist(all_hrs, bins=30, color='#4AB4DC', alpha=0.7, edgecolor='black')
        ax.axvline(x=lt_estimate.lt_heart_rate, color='red',
                  linestyle='--', linewidth=2, label='LT HR')
        ax.axvline(x=lt_estimate.aerobic_threshold_hr, color='green',
                  linestyle=':', linewidth=2, label='AeT HR')

        ax.set_xlabel('Average Heart Rate (bpm)', fontweight='bold')
        ax.set_ylabel('Number of Activities', fontweight='bold')
        ax.set_title('Heart Rate Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_summary_stats(self, ax, lt_estimate):
        """Plot summary statistics"""
        ax.axis('off')

        summary = f"""
LACTATE THRESHOLD ESTIMATE
(Using GAP - Grade Adjusted Pace)

Threshold Metrics:
  LT Heart Rate:        {lt_estimate.lt_heart_rate:.0f} bpm
  LT as % of Max HR:    {lt_estimate.lt_percent_max:.0f}%
  LT Pace (GAP):        {format_pace(lt_estimate.lt_pace)} /mi

Aerobic Threshold:
  AeT Heart Rate:       {lt_estimate.aerobic_threshold_hr:.0f} bpm
  AeT Pace (GAP):       {format_pace(lt_estimate.aerobic_threshold_pace)} /mi

Model Confidence:       {lt_estimate.confidence:.0%}
Estimation Method:      {lt_estimate.method}

Data Summary:
  Activities Analyzed:  {len(self.activities)}
  HR-Pace Data Points:  {len(self.hr_pace_data):,}
  Max HR Observed:      {lt_estimate.max_heart_rate:.0f} bpm

Training Recommendations:
  Easy runs:            < {lt_estimate.zone2_hr[1]:.0f} bpm
  Tempo runs:           {lt_estimate.zone3_hr[0]:.0f}-{lt_estimate.zone3_hr[1]:.0f} bpm
  Threshold intervals:  {lt_estimate.zone4_hr[0]:.0f}-{lt_estimate.zone4_hr[1]:.0f} bpm
  VO2max intervals:     > {lt_estimate.zone5_hr[0]:.0f} bpm
"""

        ax.text(0.1, 0.95, summary, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lactate_threshold_analyzer.py <activities_directory> [--limit N] [--output chart.png]")
        print("\nExample:")
        print("  python lactate_threshold_analyzer.py ~/Downloads/export_40402578/activities/")
        print("  python lactate_threshold_analyzer.py ~/Downloads/export_40402578/activities/ --limit 50 --output lt_analysis.png")
        sys.exit(1)

    activities_dir = Path(sys.argv[1])

    # Parse options
    limit = None
    output_path = None

    if '--limit' in sys.argv:
        limit_idx = sys.argv.index('--limit')
        if limit_idx + 1 < len(sys.argv):
            limit = int(sys.argv[limit_idx + 1])

    if '--output' in sys.argv:
        output_idx = sys.argv.index('--output')
        if output_idx + 1 < len(sys.argv):
            output_path = Path(sys.argv[output_idx + 1])

    # Analyze
    analyzer = LactateThresholdAnalyzer()
    analyzer.load_activities(activities_dir, limit=limit)

    if len(analyzer.activities) < 5:
        print(f"Error: Only found {len(analyzer.activities)} activities with HR data.")
        print("Need at least 5 activities for reliable LT estimation.")
        sys.exit(1)

    # Estimate LT
    lt_estimate = analyzer.estimate_lactate_threshold()

    # Visualize
    print("\nGenerating visualization...")
    analyzer.visualize_analysis(lt_estimate, output_path)

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
