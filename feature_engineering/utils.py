"""
Utility functions for feature extraction
Date windowing, normalization, and helper calculations
"""

from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import numpy as np


def get_training_window(
    race_date: datetime,
    lookback_weeks: int,
    exclude_taper_days: int = 7
) -> Tuple[datetime, datetime]:
    """Calculate training window for feature extraction

    Args:
        race_date: Date of target race
        lookback_weeks: Number of weeks to look back
        exclude_taper_days: Days before race to exclude (taper period)

    Returns:
        Tuple of (start_date, end_date) for training window
    """
    end_date = race_date - timedelta(days=exclude_taper_days)
    start_date = end_date - timedelta(weeks=lookback_weeks)
    return start_date, end_date


def normalize_age(age: int, peak_age: int = 35) -> float:
    """Normalize age to 0-1 scale where peak_age = 1.0

    Uses inverse quadratic decay from peak age
    """
    if age < 18:
        # Young runners: linear scale from 0.7 to 1.0
        return 0.7 + (age - 18) / (peak_age - 18) * 0.3
    elif age <= peak_age:
        # Pre-peak: linear scale from 0.7 to 1.0
        return 0.7 + (age - 18) / (peak_age - 18) * 0.3
    else:
        # Post-peak: exponential decay
        years_past_peak = age - peak_age
        decay = np.exp(-0.02 * years_past_peak)
        return decay


def normalize_hr(hr: float, min_hr: float = 50, max_hr: float = 220) -> float:
    """Normalize heart rate to 0-1 scale"""
    return (hr - min_hr) / (max_hr - min_hr)


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """Calculate coefficient of variation (CV = std / mean)

    Lower CV = more consistent
    """
    if not values or len(values) < 2:
        return 0.0

    mean = np.mean(values)
    if mean == 0:
        return 0.0

    std = np.std(values)
    return std / mean


def group_by_week(activities: List[Dict], date_field: str = 'activity_date') -> Dict[int, List[Dict]]:
    """Group activities by week number

    Args:
        activities: List of activity dictionaries with date field
        date_field: Name of date field in activity dict

    Returns:
        Dictionary mapping week number (0-N) to list of activities
    """
    if not activities:
        return {}

    # Sort by date
    sorted_activities = sorted(activities, key=lambda x: x[date_field])

    # Find start date
    start_date = sorted_activities[0][date_field]
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)

    weeks = {}
    for activity in sorted_activities:
        act_date = activity[date_field]
        if isinstance(act_date, str):
            act_date = datetime.fromisoformat(act_date)

        # Calculate week number
        week_num = (act_date - start_date).days // 7

        if week_num not in weeks:
            weeks[week_num] = []

        weeks[week_num].append(activity)

    return weeks


def meters_to_miles(meters: float) -> float:
    """Convert meters to miles"""
    return meters * 0.000621371


def seconds_to_minutes(seconds: float) -> float:
    """Convert seconds to minutes"""
    return seconds / 60.0


def pace_to_speed_mph(pace_min_per_mile: float) -> float:
    """Convert pace (min/mile) to speed (mph)"""
    if pace_min_per_mile <= 0:
        return 0.0
    return 60.0 / pace_min_per_mile


def calculate_grade(elevation_change: float, distance: float) -> float:
    """Calculate grade as percentage

    Args:
        elevation_change: Elevation change in meters
        distance: Horizontal distance in meters

    Returns:
        Grade as percentage (e.g., 5.0 for 5% grade)
    """
    if distance <= 0:
        return 0.0
    return (elevation_change / distance) * 100


def estimate_marathon_pace(lt_pace: float) -> float:
    """Estimate marathon pace from lactate threshold pace

    Rule of thumb: Marathon pace is ~15-20 seconds slower per mile than LT pace
    """
    return lt_pace + 0.3  # Add ~18 seconds per mile


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_taper_quality(
    activities: List[Dict],
    taper_start: datetime,
    race_date: datetime,
    pre_taper_weekly_avg: float
) -> float:
    """Calculate taper quality score (0-1)

    Good taper: 50-70% reduction in volume with some intensity
    """
    taper_activities = []
    for a in activities:
        act_date = a['activity_date']
        if isinstance(act_date, str):
            act_date = datetime.fromisoformat(act_date)
        if taper_start <= act_date < race_date:
            taper_activities.append(a)

    if not taper_activities:
        return 0.5  # Neutral score if no taper data

    # Calculate taper volume
    taper_distance = sum(a.get('distance_meters', 0) for a in taper_activities)
    taper_days = (race_date - taper_start).days
    taper_weekly = (taper_distance / taper_days) * 7 if taper_days > 0 else 0

    # Calculate reduction percentage
    if pre_taper_weekly_avg > 0:
        reduction = 1 - (taper_weekly / pre_taper_weekly_avg)
    else:
        reduction = 0

    # Ideal taper: 50-70% reduction
    if 0.5 <= reduction <= 0.7:
        quality = 1.0
    elif 0.3 <= reduction < 0.5:
        quality = 0.7 + (reduction - 0.3) / 0.2 * 0.3
    elif 0.7 < reduction <= 0.85:
        quality = 0.7 + (0.85 - reduction) / 0.15 * 0.3
    else:
        quality = 0.5  # Too much or too little taper

    return quality
