#!/usr/bin/env python3
"""
Strava Activity Visualizer
Creates charts and visualizations from activity data
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import timedelta

from activity_analyzer import ActivityAnalyzer, ActivityStats, format_pace, format_time


class ActivityVisualizer:
    """Creates visualizations for activity data"""

    def __init__(self, stats: ActivityStats):
        self.stats = stats

    def create_full_report(self, output_path: Optional[Path] = None):
        """Create a comprehensive visualization report"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Title
        fig.suptitle(f'{self.stats.activity_type.title()} - {self.stats.start_time.strftime("%B %d, %Y")}',
                    fontsize=16, fontweight='bold')

        # 1. Pace Analysis Bar Chart (like Strava)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_pace_bars(ax1)

        # 2. Elevation Profile
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_elevation_profile(ax2)

        # 3. Heart Rate Over Distance
        ax3 = fig.add_subplot(gs[2, :])
        self._plot_heart_rate(ax3)

        # 4. Summary Stats
        ax4 = fig.add_subplot(gs[3, 0])
        self._plot_summary_stats(ax4)

        # 5. Split Table
        ax5 = fig.add_subplot(gs[3, 1])
        self._plot_splits_table(ax5)

        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()

    def _plot_pace_bars(self, ax):
        """Plot pace analysis with bars and elevation (like Strava)"""
        # Prefer laps over mile splits
        splits = self.stats.laps if self.stats.laps else self.stats.mile_splits

        if not splits:
            ax.text(0.5, 0.5, 'No split data available', ha='center', va='center')
            return

        split_numbers = [s.number for s in splits]
        paces = [s.pace for s in splits]
        elevations = [s.elevation_gain * ActivityAnalyzer.METERS_TO_FEET for s in splits]

        # Create twin axis for elevation
        ax2 = ax.twinx()

        # Plot pace bars
        bars = ax.bar(split_numbers, paces, color='#4AB4DC', alpha=0.8, edgecolor='white', linewidth=1)

        # Plot elevation area
        ax2.fill_between(split_numbers, 0, elevations, color='#E0E0E0', alpha=0.5, label='Elevation Gain')
        ax2.plot(split_numbers, elevations, color='#999999', linewidth=1)

        # Formatting
        ax.set_xlabel('Mile', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pace (min/mi)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Elevation Gain (ft)', fontsize=11, fontweight='bold')

        ax.set_title('Pace Analysis', fontsize=13, fontweight='bold', pad=10)

        # Set y-axis limits for pace (inverted because faster = lower time)
        if paces:
            valid_paces = [p for p in paces if p > 0 and p < 999]
            if valid_paces:
                min_pace = min(valid_paces)
                max_pace = max(valid_paces)
                padding = (max_pace - min_pace) * 0.2
                ax.set_ylim(max_pace + padding, max(0, min_pace - padding))

        # Format y-axis as MM:SS
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([format_pace(y) for y in y_ticks])

        # Grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Set x-axis to show all splits
        ax.set_xticks(split_numbers)

    def _plot_elevation_profile(self, ax):
        """Plot elevation profile over distance"""
        if not self.stats.track_points:
            ax.text(0.5, 0.5, 'No elevation data available', ha='center', va='center')
            return

        points = self.stats.track_points
        distances = [(p.distance * ActivityAnalyzer.METERS_TO_MILES if p.distance else 0) for p in points]
        elevations = [p.elevation * ActivityAnalyzer.METERS_TO_FEET if p.elevation else 0 for p in points]

        # Plot elevation
        ax.fill_between(distances, elevations, color='#D0D0D0', alpha=0.7)
        ax.plot(distances, elevations, color='#888888', linewidth=1.5)

        ax.set_xlabel('Distance (mi)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Elevation (ft)', fontsize=11, fontweight='bold')
        ax.set_title('Elevation Profile', fontsize=13, fontweight='bold', pad=10)

        ax.grid(True, alpha=0.3)

        # Add elevation gain/loss text
        gain_ft = self.stats.elevation_gain * ActivityAnalyzer.METERS_TO_FEET
        loss_ft = self.stats.elevation_loss * ActivityAnalyzer.METERS_TO_FEET
        ax.text(0.02, 0.98, f'↑ {gain_ft:.0f} ft  ↓ {loss_ft:.0f} ft',
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_heart_rate(self, ax):
        """Plot heart rate over distance"""
        if not self.stats.track_points or not any(p.heart_rate for p in self.stats.track_points):
            ax.text(0.5, 0.5, 'No heart rate data available', ha='center', va='center')
            ax.set_title('Heart Rate', fontsize=13, fontweight='bold', pad=10)
            return

        points = [p for p in self.stats.track_points if p.heart_rate and p.distance]
        distances = [(p.distance * ActivityAnalyzer.METERS_TO_MILES if p.distance else 0) for p in points]
        heart_rates = [p.heart_rate for p in points]

        # Plot heart rate
        ax.plot(distances, heart_rates, color='#FC4C02', linewidth=2, label='Heart Rate')
        ax.fill_between(distances, heart_rates, alpha=0.3, color='#FC4C02')

        # Add average line
        if self.stats.avg_heart_rate:
            ax.axhline(y=self.stats.avg_heart_rate, color='#666666', linestyle='--',
                      linewidth=1.5, label=f'Avg: {self.stats.avg_heart_rate} bpm')

        ax.set_xlabel('Distance (mi)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Heart Rate (bpm)', fontsize=11, fontweight='bold')
        ax.set_title('Heart Rate', fontsize=13, fontweight='bold', pad=10)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

        # Add HR zones if available
        if self.stats.avg_heart_rate:
            # Rough HR zones based on max HR
            ax.set_ylim(bottom=min(heart_rates) - 10, top=max(heart_rates) + 10)

    def _plot_summary_stats(self, ax):
        """Plot summary statistics as text"""
        ax.axis('off')

        stats_text = f"""
SUMMARY STATISTICS

Distance:           {self.stats.total_distance * ActivityAnalyzer.METERS_TO_MILES:.2f} mi

Time:
  Moving Time:      {format_time(self.stats.moving_time)}
  Elapsed Time:     {format_time(self.stats.elapsed_time)}

Pace:
  Average Pace:     {format_pace(self.stats.avg_pace)} /mi
  GAP:              {format_pace(self.stats.avg_gap)} /mi

Elevation:
  Gain:             {self.stats.elevation_gain * ActivityAnalyzer.METERS_TO_FEET:.0f} ft
  Loss:             {self.stats.elevation_loss * ActivityAnalyzer.METERS_TO_FEET:.0f} ft
  Range:            {self.stats.min_elevation * ActivityAnalyzer.METERS_TO_FEET:.0f} - {self.stats.max_elevation * ActivityAnalyzer.METERS_TO_FEET:.0f} ft
"""

        if self.stats.avg_heart_rate:
            stats_text += f"""
Heart Rate:
  Average:          {self.stats.avg_heart_rate} bpm
  Maximum:          {self.stats.max_heart_rate} bpm
"""

        if self.stats.calories:
            stats_text += f"""
Calories:           {self.stats.calories}
"""

        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def _plot_splits_table(self, ax):
        """Plot splits as a table"""
        ax.axis('off')

        # Prefer laps over mile splits
        splits = self.stats.laps if self.stats.laps else self.stats.mile_splits
        splits_label = 'Lap' if self.stats.laps else 'Mile'

        if not splits:
            ax.text(0.5, 0.5, 'No split data available', ha='center', va='center')
            return

        # Prepare table data
        headers = [splits_label, 'Distance', 'Time', 'Pace', 'GAP', 'HR']
        rows = []

        for split in splits:
            dist_mi = split.distance * ActivityAnalyzer.METERS_TO_MILES
            hr_str = str(split.avg_heart_rate) if split.avg_heart_rate else '--'

            rows.append([
                str(split.number),
                f'{dist_mi:.2f} mi',
                format_time(split.duration),
                format_pace(split.pace),
                format_pace(split.gap),
                hr_str
            ])

        # Create table
        table = ax.table(cellText=rows, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4AB4DC')
            cell.set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('white')

        # Set title based on split type
        title = 'Laps' if self.stats.laps else 'Mile Splits'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    def create_pace_chart_only(self, output_path: Optional[Path] = None):
        """Create just the pace analysis chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_pace_bars(ax)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved pace chart to: {output_path}")
        else:
            plt.show()

        plt.close()

    def create_route_map(self, output_path: Optional[Path] = None):
        """Create a simple route map from GPS coordinates"""
        if not self.stats.track_points:
            print("No GPS data available for route map")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract coordinates
        lats = [p.latitude for p in self.stats.track_points]
        lons = [p.longitude for p in self.stats.track_points]

        # Plot route
        ax.plot(lons, lats, color='#FC4C02', linewidth=3, alpha=0.8)

        # Mark start and finish
        ax.plot(lons[0], lats[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(lons[-1], lats[-1], 'ro', markersize=12, label='Finish', zorder=5)

        # Add mile markers
        if self.stats.mile_splits:
            mile_distances = [i * 1609.34 for i in range(1, len(self.stats.mile_splits) + 1)]
            for mile_num, target_dist in enumerate(mile_distances, 1):
                # Find closest point to this mile distance
                closest_point = min(self.stats.track_points,
                                  key=lambda p: abs(p.distance - target_dist) if p.distance else float('inf'))
                ax.plot(closest_point.longitude, closest_point.latitude,
                       'o', color='white', markersize=8, zorder=4,
                       markeredgecolor='#4AB4DC', markeredgewidth=2)
                ax.text(closest_point.longitude, closest_point.latitude,
                       str(mile_num), fontsize=8, ha='center', va='center',
                       fontweight='bold', zorder=5)

        ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
        ax.set_title(f'Route Map - {self.stats.total_distance * ActivityAnalyzer.METERS_TO_MILES:.2f} mi',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved route map to: {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <activity_file.fit|.gpx> [output_dir]")
        print("\nExample:")
        print("  python visualizer.py ~/Downloads/export_40402578/activities/12345.fit.gz")
        print("  python visualizer.py ~/Downloads/export_40402578/activities/12345.fit.gz ./charts/")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Parse output directory
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {file_path.name}...")

    # Analyze activity
    analyzer = ActivityAnalyzer()
    stats = analyzer.analyze_file(file_path)

    if not stats:
        print("Failed to analyze activity file")
        sys.exit(1)

    print("Creating visualizations...")

    # Create visualizer
    viz = ActivityVisualizer(stats)

    # Generate all visualizations
    if output_dir:
        base_name = file_path.stem.replace('.fit', '').replace('.gpx', '')

        # Full report
        viz.create_full_report(output_dir / f"{base_name}_full_report.png")

        # Individual charts
        viz.create_pace_chart_only(output_dir / f"{base_name}_pace.png")
        viz.create_route_map(output_dir / f"{base_name}_route.png")

        print(f"\nAll visualizations saved to: {output_dir}")
    else:
        # Show interactive plots
        print("\nGenerating full report...")
        viz.create_full_report()

        print("Generating route map...")
        viz.create_route_map()


if __name__ == '__main__':
    main()
