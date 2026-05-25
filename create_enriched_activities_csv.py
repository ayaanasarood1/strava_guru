#!/usr/bin/env python3
"""
Create an enriched activities CSV with features from both CSV and FIT files.
Uses the 'Filename' column from activities.csv for direct FIT file matching.
Adds a 'bonked' column for races that should be excluded from training.
"""

import argparse
import pandas as pd
import numpy as np
import zipfile
import gzip
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

try:
    import fitdecode
    FIT_AVAILABLE = True
    FIT_LIBRARY = 'fitdecode'
except ImportError:
    try:
        from fitparse import FitFile
        FIT_AVAILABLE = True
        FIT_LIBRARY = 'fitparse'
    except ImportError:
        FIT_AVAILABLE = False
        FIT_LIBRARY = None
        print("Warning: neither fitdecode nor fitparse installed. FIT features will be empty.")

import xml.etree.ElementTree as ET


def parse_gpx_file(gpx_content):
    """Parse GPX file and extract detailed metrics"""
    try:
        root = ET.fromstring(gpx_content)

        # Handle GPX namespace
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1',
              'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'}

        # Find all trackpoints
        trackpoints = root.findall('.//gpx:trkpt', ns)
        if not trackpoints:
            # Try without namespace
            trackpoints = root.findall('.//{http://www.topografix.com/GPX/1/1}trkpt')
        if not trackpoints:
            trackpoints = root.findall('.//trkpt')

        if not trackpoints:
            return None

        timestamps = []
        heart_rates = []
        elevations = []
        lats = []
        lons = []

        for tp in trackpoints:
            # Get lat/lon
            lat = tp.get('lat')
            lon = tp.get('lon')
            if lat and lon:
                lats.append(float(lat))
                lons.append(float(lon))

            # Get timestamp
            time_elem = tp.find('gpx:time', ns) or tp.find('{http://www.topografix.com/GPX/1/1}time') or tp.find('time')
            if time_elem is not None and time_elem.text:
                from datetime import datetime
                try:
                    ts = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    pass

            # Get elevation
            ele_elem = tp.find('gpx:ele', ns) or tp.find('{http://www.topografix.com/GPX/1/1}ele') or tp.find('ele')
            if ele_elem is not None and ele_elem.text:
                try:
                    elevations.append(float(ele_elem.text))
                except:
                    pass

            # Get heart rate from extensions
            ext = tp.find('.//gpxtpx:hr', ns)
            if ext is None:
                ext = tp.find('.//{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr')
            if ext is not None and ext.text:
                try:
                    heart_rates.append(int(ext.text))
                except:
                    pass

        if not timestamps or len(lats) < 2:
            return None

        # Calculate distance using haversine
        from math import radians, sin, cos, sqrt, atan2

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            phi1, phi2 = radians(lat1), radians(lat2)
            dphi = radians(lat2 - lat1)
            dlambda = radians(lon2 - lon1)
            a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
            return 2*R*atan2(sqrt(a), sqrt(1-a))

        total_distance_m = sum(haversine(lats[i], lons[i], lats[i+1], lons[i+1])
                               for i in range(len(lats)-1))
        distance_miles = total_distance_m / 1609.34

        if len(timestamps) > 1:
            duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()
        else:
            duration_sec = 0

        duration_min = duration_sec / 60
        pace = duration_min / distance_miles if distance_miles > 0.5 else None

        # HR metrics
        hr_valid = [hr for hr in heart_rates if hr and hr > 0]
        avg_hr = np.mean(hr_valid) if hr_valid else None
        max_hr = max(hr_valid) if hr_valid else None
        min_hr = min(hr_valid) if hr_valid else None

        # HR zones
        if hr_valid:
            zone1 = sum(1 for hr in hr_valid if hr < 114) / len(hr_valid) * 100
            zone2 = sum(1 for hr in hr_valid if 114 <= hr < 133) / len(hr_valid) * 100
            zone3 = sum(1 for hr in hr_valid if 133 <= hr < 152) / len(hr_valid) * 100
            zone4 = sum(1 for hr in hr_valid if 152 <= hr < 171) / len(hr_valid) * 100
            zone5 = sum(1 for hr in hr_valid if hr >= 171) / len(hr_valid) * 100
        else:
            zone1 = zone2 = zone3 = zone4 = zone5 = None

        # Elevation
        if elevations and len(elevations) > 1:
            elevation_gain = sum(max(0, elevations[i+1] - elevations[i])
                                for i in range(len(elevations)-1))
            elevation_loss = sum(max(0, elevations[i] - elevations[i+1])
                                for i in range(len(elevations)-1))
        else:
            elevation_gain = elevation_loss = 0

        return {
            'fit_distance_miles': round(distance_miles, 2),
            'fit_duration_min': round(duration_min, 2),
            'fit_pace': round(pace, 2) if pace else None,
            'fit_avg_hr': round(avg_hr, 1) if avg_hr else None,
            'fit_max_hr': int(max_hr) if max_hr else None,
            'fit_min_hr': int(min_hr) if min_hr else None,
            'fit_zone1_pct': round(zone1, 1) if zone1 is not None else None,
            'fit_zone2_pct': round(zone2, 1) if zone2 is not None else None,
            'fit_zone3_pct': round(zone3, 1) if zone3 is not None else None,
            'fit_zone4_pct': round(zone4, 1) if zone4 is not None else None,
            'fit_zone5_pct': round(zone5, 1) if zone5 is not None else None,
            'fit_elevation_gain_m': round(elevation_gain, 1),
            'fit_elevation_loss_m': round(elevation_loss, 1),
            'fit_avg_cadence': None,  # GPX typically doesn't have cadence
            'fit_pace_variability': None,
        }
    except Exception as e:
        return None


def parse_fit_file(fit_content):
    """Parse FIT file and extract detailed metrics using fitdecode (more robust) or fitparse"""
    if not FIT_AVAILABLE:
        return None

    try:
        import warnings
        warnings.filterwarnings('ignore')  # Suppress fitdecode warnings about field sizes

        distances = []
        heart_rates = []
        timestamps = []
        altitudes = []
        cadences = []

        if FIT_LIBRARY == 'fitdecode':
            with fitdecode.FitReader(BytesIO(fit_content)) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'record':
                        for field in frame.fields:
                            if field.name == 'distance' and field.value is not None:
                                distances.append(field.value)
                            elif field.name == 'heart_rate' and field.value is not None:
                                heart_rates.append(field.value)
                            elif field.name == 'timestamp' and field.value is not None:
                                timestamps.append(field.value)
                            elif field.name in ('altitude', 'enhanced_altitude') and field.value is not None:
                                altitudes.append(field.value)
                            elif field.name == 'cadence' and field.value is not None:
                                cadences.append(field.value)
        else:
            # Fallback to fitparse
            from fitparse import FitFile
            fitfile = FitFile(BytesIO(fit_content))
            for record in fitfile.get_messages('record'):
                for field in record:
                    if field.name == 'distance' and field.value is not None:
                        distances.append(field.value)
                    elif field.name == 'heart_rate' and field.value is not None:
                        heart_rates.append(field.value)
                    elif field.name == 'timestamp' and field.value is not None:
                        timestamps.append(field.value)
                    elif field.name in ('altitude', 'enhanced_altitude') and field.value is not None:
                        altitudes.append(field.value)
                    elif field.name == 'cadence' and field.value is not None:
                        cadences.append(field.value)

        if not timestamps:
            return None

        total_distance_m = max(distances) if distances else 0
        distance_miles = total_distance_m / 1609.34

        if len(timestamps) > 1:
            duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()
        else:
            duration_sec = 0

        duration_min = duration_sec / 60
        pace = duration_min / distance_miles if distance_miles > 0.5 else None

        # HR metrics
        hr_valid = [hr for hr in heart_rates if hr and hr > 0]
        avg_hr = np.mean(hr_valid) if hr_valid else None
        max_hr = max(hr_valid) if hr_valid else None
        min_hr = min(hr_valid) if hr_valid else None

        # HR zones (assuming max HR ~190)
        if hr_valid:
            zone1 = sum(1 for hr in hr_valid if hr < 114) / len(hr_valid) * 100
            zone2 = sum(1 for hr in hr_valid if 114 <= hr < 133) / len(hr_valid) * 100
            zone3 = sum(1 for hr in hr_valid if 133 <= hr < 152) / len(hr_valid) * 100
            zone4 = sum(1 for hr in hr_valid if 152 <= hr < 171) / len(hr_valid) * 100
            zone5 = sum(1 for hr in hr_valid if hr >= 171) / len(hr_valid) * 100
        else:
            zone1 = zone2 = zone3 = zone4 = zone5 = None

        # Elevation
        if altitudes and len(altitudes) > 1:
            alt_valid = [a for a in altitudes if a is not None]
            if alt_valid:
                elevation_gain = sum(max(0, alt_valid[i+1] - alt_valid[i])
                                    for i in range(len(alt_valid)-1))
                elevation_loss = sum(max(0, alt_valid[i] - alt_valid[i+1])
                                    for i in range(len(alt_valid)-1))
            else:
                elevation_gain = elevation_loss = 0
        else:
            elevation_gain = elevation_loss = 0

        # Cadence
        cadence_valid = [c for c in cadences if c and c > 0]
        avg_cadence = np.mean(cadence_valid) * 2 if cadence_valid else None  # *2 for steps/min

        # Pace variability
        if len(timestamps) > 10 and len(distances) > 10:
            segment_paces = []
            min_len = min(len(timestamps), len(distances))
            for i in range(1, min_len):
                if distances[i] > distances[i-1]:
                    seg_dist = (distances[i] - distances[i-1]) / 1609.34
                    seg_time = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                    if seg_dist > 0.01:
                        segment_paces.append(seg_time / seg_dist)

            if segment_paces:
                pace_variability = np.std(segment_paces) / np.mean(segment_paces) if np.mean(segment_paces) > 0 else 0
            else:
                pace_variability = None
        else:
            pace_variability = None

        return {
            'fit_distance_miles': round(distance_miles, 2),
            'fit_duration_min': round(duration_min, 2),
            'fit_pace': round(pace, 2) if pace else None,
            'fit_avg_hr': round(avg_hr, 1) if avg_hr else None,
            'fit_max_hr': int(max_hr) if max_hr else None,
            'fit_min_hr': int(min_hr) if min_hr else None,
            'fit_zone1_pct': round(zone1, 1) if zone1 is not None else None,
            'fit_zone2_pct': round(zone2, 1) if zone2 is not None else None,
            'fit_zone3_pct': round(zone3, 1) if zone3 is not None else None,
            'fit_zone4_pct': round(zone4, 1) if zone4 is not None else None,
            'fit_zone5_pct': round(zone5, 1) if zone5 is not None else None,
            'fit_elevation_gain_m': round(elevation_gain, 1),
            'fit_elevation_loss_m': round(elevation_loss, 1),
            'fit_avg_cadence': round(avg_cadence, 1) if avg_cadence else None,
            'fit_pace_variability': round(pace_variability, 3) if pace_variability else None,
        }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description='Create enriched activities CSV')
    parser.add_argument('input_path', help='Path to Strava export folder or zip')
    parser.add_argument('--output', default='enriched_activities.csv', help='Output CSV file')
    parser.add_argument('--bonked-dates', nargs='*', default=[],
                       help='Race dates to mark as bonked (YYYY-MM-DD)')
    args = parser.parse_args()

    input_path = Path(args.input_path)

    # Determine if zip or folder
    if input_path.suffix == '.zip':
        is_zip = True
        zip_path = input_path
    elif input_path.is_dir():
        is_zip = False
        folder_path = input_path
    else:
        print(f"Error: {input_path} is not a valid zip file or folder")
        return

    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    if args.bonked_dates:
        print(f"Bonked dates: {args.bonked_dates}")
    print()

    # Parse activities.csv
    print("Reading activities.csv...")
    if is_zip:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_file = None
            for name in z.namelist():
                if name.endswith('activities.csv'):
                    csv_file = name
                    break
            if not csv_file:
                print("Error: No activities.csv found")
                return
            with z.open(csv_file) as f:
                df = pd.read_csv(f)
    else:
        csv_path = folder_path / 'activities.csv'
        if not csv_path.exists():
            print(f"Error: {csv_path} not found")
            return
        df = pd.read_csv(csv_path)

    # Filter to runs only
    df = df[df['Activity Type'] == 'Run'].copy()
    print(f"Found {len(df)} running activities")

    # Parse date column
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
        except:
            try:
                return datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
            except:
                return None

    df['parsed_date'] = df['Activity Date'].apply(parse_date)
    df = df[df['parsed_date'].notna()]

    # Calculate basic features from CSV
    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else None
        except:
            return None

    # Handle duplicate Distance columns
    if 'Distance.1' in df.columns:
        df['distance_meters'] = df['Distance.1'].apply(safe_float)
    else:
        df['distance_meters'] = df['Distance'].apply(lambda x: safe_float(x) * 1609.34 if safe_float(x) and safe_float(x) < 100 else safe_float(x))

    df['csv_distance_miles'] = df['distance_meters'].apply(lambda x: round(x / 1609.34, 2) if x else None)
    df['csv_moving_time_sec'] = df['Moving Time'].apply(safe_float)
    df['csv_elapsed_time_sec'] = df['Elapsed Time'].apply(safe_float)
    df['csv_duration_min'] = df.apply(
        lambda r: round((r['csv_moving_time_sec'] or r['csv_elapsed_time_sec'] or 0) / 60, 2), axis=1
    )
    df['csv_pace'] = df.apply(
        lambda r: round(r['csv_duration_min'] / r['csv_distance_miles'], 2)
        if r['csv_distance_miles'] and r['csv_distance_miles'] > 0.5 else None, axis=1
    )
    df['csv_avg_hr'] = df['Average Heart Rate'].apply(safe_float)
    df['csv_max_hr'] = df['Max Heart Rate'].apply(safe_float)
    df['csv_elevation_gain'] = df['Elevation Gain'].apply(safe_float)
    df['csv_calories'] = df['Calories'].apply(safe_float)

    # Parse FIT files using Filename column (direct mapping)
    fit_columns = ['fit_distance_miles', 'fit_duration_min', 'fit_pace', 'fit_avg_hr',
                   'fit_max_hr', 'fit_min_hr', 'fit_zone1_pct', 'fit_zone2_pct',
                   'fit_zone3_pct', 'fit_zone4_pct', 'fit_zone5_pct',
                   'fit_elevation_gain_m', 'fit_elevation_loss_m',
                   'fit_avg_cadence', 'fit_pace_variability']

    for col in fit_columns:
        df[col] = None

    if FIT_AVAILABLE and 'Filename' in df.columns:
        print("\nParsing FIT files using Filename column...")

        # Count how many have filenames
        has_filename = df['Filename'].notna().sum()
        print(f"  Activities with Filename: {has_filename}/{len(df)}")

        matched = 0
        parsed = 0

        for idx, row in df.iterrows():
            filename = row.get('Filename')
            if pd.isna(filename) or not filename:
                continue

            try:
                if is_zip:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        # Filename might be relative path like "activities/123.fit.gz"
                        fit_path = filename
                        if fit_path not in z.namelist():
                            # Try with folder prefix stripped
                            fit_path = filename.split('/')[-1]
                            for name in z.namelist():
                                if name.endswith(fit_path):
                                    fit_path = name
                                    break

                        with z.open(fit_path) as f:
                            content = f.read()
                            if filename.endswith('.gz'):
                                content = gzip.decompress(content)

                        # Parse based on file type
                        if filename.endswith('.fit.gz') or filename.endswith('.fit'):
                            data = parse_fit_file(content)
                        elif filename.endswith('.gpx.gz') or filename.endswith('.gpx'):
                            data = parse_gpx_file(content)
                        else:
                            continue
                else:
                    fit_path = folder_path / filename
                    if not fit_path.exists():
                        # Try just the filename in activities folder
                        fit_path = folder_path / 'activities' / filename.split('/')[-1]

                    if fit_path.exists():
                        filepath_str = str(fit_path)

                        # Parse based on file type
                        if filepath_str.endswith('.fit.gz'):
                            with gzip.open(fit_path, 'rb') as f:
                                content = f.read()
                            data = parse_fit_file(content)
                        elif filepath_str.endswith('.fit'):
                            with open(fit_path, 'rb') as f:
                                content = f.read()
                            data = parse_fit_file(content)
                        elif filepath_str.endswith('.gpx.gz'):
                            with gzip.open(fit_path, 'rb') as f:
                                content = f.read()
                            data = parse_gpx_file(content)
                        elif filepath_str.endswith('.gpx'):
                            with open(fit_path, 'rb') as f:
                                content = f.read()
                            data = parse_gpx_file(content)
                        else:
                            # Unknown format, skip
                            continue
                    else:
                        continue

                # data is already set above
                parsed += 1

                if data:
                    for col in fit_columns:
                        df.at[idx, col] = data.get(col)
                    matched += 1

            except Exception as e:
                continue

            if parsed % 200 == 0:
                print(f"  Processed {parsed}... (matched: {matched})")

        print(f"  Successfully parsed {parsed} FIT files")
        print(f"  Matched {matched} activities with FIT data")

    elif not FIT_AVAILABLE:
        print("\nSkipping FIT parsing (fitparse not available)")
    else:
        print("\nNo Filename column found - cannot match FIT files")

    # Add derived features
    print("\nAdding derived features...")

    # Use best available data (FIT preferred, CSV fallback)
    df['distance_miles'] = df['fit_distance_miles'].fillna(df['csv_distance_miles'])
    df['duration_min'] = df['fit_duration_min'].fillna(df['csv_duration_min'])
    df['pace_min_per_mile'] = df['fit_pace'].fillna(df['csv_pace'])
    df['avg_hr'] = df['fit_avg_hr'].fillna(df['csv_avg_hr'])
    df['max_hr'] = df['fit_max_hr'].fillna(df['csv_max_hr'])

    # Workout type classification
    df['is_long_run'] = df['distance_miles'].apply(lambda x: x >= 15 if x else False)
    df['is_tempo'] = df['pace_min_per_mile'].apply(lambda x: 7.0 <= x < 8.0 if x else False)
    df['is_speed_work'] = df['pace_min_per_mile'].apply(lambda x: x < 7.0 if x else False)
    df['is_easy'] = df['pace_min_per_mile'].apply(lambda x: x >= 9.0 if x else False)
    df['is_marathon'] = df['distance_miles'].apply(lambda x: 25.0 <= x <= 27.5 if x else False)

    # Bonked flag
    bonked_dates = set(args.bonked_dates)
    df['is_bonked'] = df['parsed_date'].apply(
        lambda d: d.strftime('%Y-%m-%d') in bonked_dates if d else False
    )

    # Select and order columns for output
    output_columns = [
        # Identifiers
        'Activity Date', 'Activity Name',

        # Combined best values
        'distance_miles', 'duration_min', 'pace_min_per_mile', 'avg_hr', 'max_hr',

        # CSV values
        'csv_distance_miles', 'csv_duration_min', 'csv_pace', 'csv_avg_hr',
        'csv_max_hr', 'csv_elevation_gain', 'csv_calories',

        # FIT values
        'fit_distance_miles', 'fit_duration_min', 'fit_pace', 'fit_avg_hr',
        'fit_max_hr', 'fit_min_hr', 'fit_zone1_pct', 'fit_zone2_pct',
        'fit_zone3_pct', 'fit_zone4_pct', 'fit_zone5_pct',
        'fit_elevation_gain_m', 'fit_elevation_loss_m',
        'fit_avg_cadence', 'fit_pace_variability',

        # Classifications
        'is_long_run', 'is_tempo', 'is_speed_work', 'is_easy', 'is_marathon',
        'is_bonked',
    ]

    # Only include columns that exist
    output_columns = [c for c in output_columns if c in df.columns]

    # Sort by date descending
    df = df.sort_values('parsed_date', ascending=False)

    # Save
    df[output_columns].to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(f"Total activities: {len(df)}")
    print(f"With FIT data: {df['fit_distance_miles'].notna().sum()}")
    print(f"Long runs: {df['is_long_run'].sum()}")
    print(f"Tempo runs: {df['is_tempo'].sum()}")
    print(f"Marathons: {df['is_marathon'].sum()}")
    print(f"Bonked: {df['is_bonked'].sum()}")


if __name__ == '__main__':
    main()
