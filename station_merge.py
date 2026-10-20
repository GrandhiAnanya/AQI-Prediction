from pathlib import Path
import re
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_FOLDER = Path("Datasets")
COMBINED_FILE = ROOT_FOLDER / "Combined_Stations.csv"
OUTPUT_FILE = ROOT_FOLDER / "Combined_Stations_Fixed.csv"

WEATHER_COLUMNS = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "wind_speed_10m (km/h)",
    "wind_direction_10m (°)",
    "precipitation (mm)",
    "surface_pressure (hPa)",
    "cloud_cover (%)",
    "shortwave_radiation (W/m²)",
]

# Final output order requested in Merge_fix.py.
COLUMN_ORDER = ['Day', 'Month', 'Year', 'Time', 'Day_of_Week', 'Season', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)', 'NH3 (µg/m³)', 'temperature_2m (°C)', 'relative_humidity_2m (%)', 'dew_point_2m (°C)', 'precipitation (mm)', 'surface_pressure (hPa)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)', 'shortwave_radiation (W/m²)', 'cloud_cover (%)', 'elevation', 'latitude', 'longitude', 'AQI', 'AQI_lag_1', 'AQI_lag_6', 'AQI_lag_12', 'AQI_lag_24', 'PM2.5_lag_1', 'PM2.5_lag_6', 'PM2.5_lag_12', 'PM2.5_lag_24', 'PM10_lag_1', 'PM10_lag_6', 'PM10_lag_12', 'PM10_lag_24', 'NO2_lag_1', 'NO2_lag_6', 'NO2_lag_12', 'NO2_lag_24', 'SO2_lag_1', 'SO2_lag_6', 'SO2_lag_12', 'SO2_lag_24', 'CO_lag_1', 'CO_lag_6', 'CO_lag_12', 'CO_lag_24', 'Ozone_lag_1', 'Ozone_lag_6', 'Ozone_lag_12', 'Ozone_lag_24', 'NH3_lag_1', 'NH3_lag_6', 'NH3_lag_12', 'NH3_lag_24', 'AQI_roll_mean_1', 'AQI_roll_median_1', 'AQI_roll_mean_6', 'AQI_roll_median_6', 'AQI_roll_std_6', 'AQI_roll_mean_12', 'AQI_roll_median_12', 'AQI_roll_std_12', 'AQI_roll_mean_24', 'AQI_roll_median_24', 'AQI_roll_std_24', 'PM2.5_roll_mean_1', 'PM2.5_roll_median_1', 'PM2.5_roll_mean_6', 'PM2.5_roll_median_6', 'PM2.5_roll_std_6', 'PM2.5_roll_mean_12', 'PM2.5_roll_median_12', 'PM2.5_roll_std_12', 'PM2.5_roll_mean_24', 'PM2.5_roll_median_24', 'PM2.5_roll_std_24', 'PM10_roll_mean_1', 'PM10_roll_median_1', 'PM10_roll_mean_6', 'PM10_roll_median_6', 'PM10_roll_std_6', 'PM10_roll_mean_12', 'PM10_roll_median_12', 'PM10_roll_std_12', 'PM10_roll_mean_24', 'PM10_roll_median_24', 'PM10_roll_std_24', 'NO2_roll_mean_1', 'NO2_roll_median_1', 'NO2_roll_mean_6', 'NO2_roll_median_6', 'NO2_roll_std_6', 'NO2_roll_mean_12', 'NO2_roll_median_12', 'NO2_roll_std_12', 'NO2_roll_mean_24', 'NO2_roll_median_24', 'NO2_roll_std_24', 'SO2_roll_mean_1', 'SO2_roll_median_1', 'SO2_roll_mean_6', 'SO2_roll_median_6', 'SO2_roll_std_6', 'SO2_roll_mean_12', 'SO2_roll_median_12', 'SO2_roll_std_12', 'SO2_roll_mean_24', 'SO2_roll_median_24', 'SO2_roll_std_24', 'CO_roll_mean_1', 'CO_roll_median_1', 'CO_roll_mean_6', 'CO_roll_median_6', 'CO_roll_std_6', 'CO_roll_mean_12', 'CO_roll_median_12', 'CO_roll_std_12', 'CO_roll_mean_24', 'CO_roll_median_24', 'CO_roll_std_24', 'Ozone_roll_mean_1', 'Ozone_roll_median_1', 'Ozone_roll_mean_6', 'Ozone_roll_median_6', 'Ozone_roll_std_6', 'Ozone_roll_mean_12', 'Ozone_roll_median_12', 'Ozone_roll_std_12', 'Ozone_roll_mean_24', 'Ozone_roll_median_24', 'Ozone_roll_std_24', 'NH3_roll_mean_1', 'NH3_roll_median_1', 'NH3_roll_mean_6', 'NH3_roll_median_6', 'NH3_roll_std_6', 'NH3_roll_mean_12', 'NH3_roll_median_12', 'NH3_roll_std_12', 'NH3_roll_mean_24', 'NH3_roll_median_24', 'NH3_roll_std_24', 'AQI_future_1', 'AQI_future_24']


def clean_column_name(column):
    """Clean whitespace and common encoding issues in column names."""
    column = str(column).strip().replace("\ufeff", "").replace("Â", "")
    column = column.replace("º", "°")
    return re.sub(r"\s+", " ", column)


def column_key(column):
    """Normalize a column name for matching despite punctuation/spacing."""
    return re.sub(r"[^a-z0-9]", "", clean_column_name(column).lower())



# Station filenames are normalized before matching so spaces, underscores,
# hyphens, and punctuation do not affect the unseen-station classification.
UNSEEN_STATION_KEYS = {
    "powaimumbai": "Mumbai Powai",
    "velacheryresareachennai": "Chennai Velachery Res Area",
    "bidhannagarkolkata": "Kolkata Bidhannagar",
    "alipurdelhi": "Delhi Alipur",
    "lonighaziabad": "Ghaziabad Loni",
    "kukrailpicnicspot1lucknow": "Lucknow Kukrail Picnic Spot-1",
    "pallavpuramphase2meerut": "Meerut Pallavpuram Phase 2",
    "zooparkhyderabad": "Hyderabad Zoo Park",
}


def normalized_station_key(filename):
    """Convert a station filename/stem into lowercase alphanumeric text."""
    stem = Path(str(filename)).stem
    stem = re.sub(r"_final$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def classify_station(filename):
    """Return 'Unseen' for the eight held-out stations; otherwise 'Seen'."""
    key = normalized_station_key(filename)
    for unseen_key in UNSEEN_STATION_KEYS:
        if unseen_key in key:
            return "Unseen"
    return "Seen"


def add_station_helper_columns(df):
    """
    Add deterministic IDs and seen/unseen labels using source filenames.

    IDs are assigned from sorted unique source filenames. The mapping is
    saved alongside the final dataset so the IDs can be reused consistently.
    """
    if "source_file" not in df.columns:
        raise KeyError("Expected 'source_file' before creating Station_ID.")

    station_files = sorted(df["source_file"].dropna().astype(str).unique())
    station_id_map = {
        filename: f"GEO_{index:04d}"
        for index, filename in enumerate(station_files, start=1)
    }

    df["Station_ID"] = df["source_file"].astype(str).map(station_id_map)
    df["Station_Class"] = df["source_file"].apply(classify_station)

    mapping_df = pd.DataFrame({
        "source_file": station_files,
        "Station_ID": [station_id_map[name] for name in station_files],
        "Station_Class": [classify_station(name) for name in station_files],
    })
    mapping_file = ROOT_FOLDER / "Station_ID_Mapping.csv"
    mapping_df.to_csv(mapping_file, index=False, encoding="utf-8-sig")

    print("\\nStation classification:")
    print(mapping_df["Station_Class"].value_counts().to_string())
    print(f"Station ID mapping saved to: {mapping_file}")

    # Warn if any of the eight requested unseen stations was not matched.
    matched_unseen = {
        requested_name
        for filename in station_files
        for key, requested_name in UNSEEN_STATION_KEYS.items()
        if key in normalized_station_key(filename)
    }
    unmatched = sorted(set(UNSEEN_STATION_KEYS.values()) - matched_unseen)
    if unmatched:
        print("\\nWARNING: These requested unseen stations were not matched to filenames:")
        for name in unmatched:
            print("-", name)

    return df

def merge_station_files():
    """Combine all station files ending in _Final.csv under Datasets/."""
    if not ROOT_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder '{ROOT_FOLDER}' was not found. Put this script beside the Datasets folder."
        )

    weather_lookup = {column_key(name): name for name in WEATHER_COLUMNS}
    dataframes = []
    failed_files = []

    # Do not accidentally read our own output files as station inputs.
    csv_files = sorted(
        path for path in ROOT_FOLDER.rglob("*.csv")
        if path.name.lower().endswith("_final.csv")
        and path.resolve() not in {COMBINED_FILE.resolve(), OUTPUT_FILE.resolve()}
    )

    print(f"Found {len(csv_files)} station CSV files.")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            df.columns = [clean_column_name(col) for col in df.columns]

            rename_map = {}
            for col in df.columns:
                key = column_key(col)
                if key in weather_lookup:
                    rename_map[col] = weather_lookup[key]
            df = df.rename(columns=rename_map)

            # Preserve source during merging for diagnostics; removed in final output.
            df["source_file"] = file_path.name
            dataframes.append(df)
            print(f"Loaded: {file_path.name} | Rows: {len(df)}")

        except Exception as exc:
            failed_files.append((file_path, str(exc)))
            print(f"Could not read {file_path}: {exc}")

    if not dataframes:
        raise RuntimeError(
            f"No station CSV files ending in '_Final.csv' were successfully loaded under '{ROOT_FOLDER}'."
        )

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    combined_df = add_station_helper_columns(combined_df)

    # Ensure every expected weather column exists.
    for col in WEATHER_COLUMNS:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA

    # Keep non-weather columns first, then the weather columns in the specified order.
    other_columns = [col for col in combined_df.columns if col not in WEATHER_COLUMNS]
    combined_df = combined_df[other_columns + WEATHER_COLUMNS]

    # Intermediate combined file, useful for debugging.
    # combined_df.to_csv(COMBINED_FILE, index=False, encoding="utf-8-sig")

    print("\nMerge complete.")
    print(f"Total files loaded: {len(dataframes)}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Intermediate file: {COMBINED_FILE}")

    if failed_files:
        print("\nFiles that could not be processed:")
        for path, error in failed_files:
            print(f"- {path}: {error}")

    return combined_df


def fix_and_save_dataset(df):
    """Merge duplicate target columns, remove source_file, and order columns."""
    target_pairs = [
        ("AQI_future_1", "AQI_t+1"),
        ("AQI_future_24", "AQI_t+24"),
    ]

    for target, duplicate in target_pairs:
        if target in df.columns and duplicate in df.columns:
            both_present = df[target].notna() & df[duplicate].notna()
            conflicts = both_present & (
                df[target].astype("string") != df[duplicate].astype("string")
            )

            print(f"\n{target} and {duplicate}")
            print("Rows where both have values:", int(both_present.sum()))
            print("Rows with conflicting values:", int(conflicts.sum()))

            # Prefer the primary target value; fill only its missing values from duplicate.
            df[target] = df[target].combine_first(df[duplicate])
            df.drop(columns=[duplicate], inplace=True)

        elif target not in df.columns and duplicate in df.columns:
            df.rename(columns={duplicate: target}, inplace=True)

        elif target not in df.columns:
            print(f"WARNING: Neither {target} nor {duplicate} exists.")

    if "source_file" in df.columns:
        df.drop(columns=["source_file"], inplace=True)

    missing_columns = [col for col in COLUMN_ORDER if col not in df.columns]
    helper_columns = ["Station_ID", "Station_Class"]
    extra_columns = [
        col for col in df.columns
        if col not in COLUMN_ORDER and col not in helper_columns
    ]

    if missing_columns:
        print("\nWARNING: Expected columns missing from the merged data:")
        for col in missing_columns:
            print("-", col)

    if extra_columns:
        print("\nAdditional columns retained at the end:")
        for col in extra_columns:
            print("-", col)

    final_order = (
        [col for col in helper_columns if col in df.columns]
        + [col for col in COLUMN_ORDER if col in df.columns and col not in helper_columns]
        + extra_columns
    )
    df = df[final_order]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nFinal dataset saved to:", OUTPUT_FILE)
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))
    print("\nFinal column order:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    return df


def main():
    ROOT_FOLDER.mkdir(parents=True, exist_ok=True)
    combined_df = merge_station_files()
    fix_and_save_dataset(combined_df)


if __name__ == "__main__":
    main()
