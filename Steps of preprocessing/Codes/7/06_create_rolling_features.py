
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 5 — ROLLING FEATURE CREATION
# ============================================================

print("=" * 70)
print("STEP 5 — STATION-WISE ROLLING FEATURE CREATION")
print("=" * 70)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = (
    "merged_with_weather_with_elevation_with_AQI_lags.csv"
)

OUTPUT_FILENAME = (
    "merged_with_weather_with_elevation_final.csv"
)


# ============================================================
# 2. VARIABLES
# ============================================================

ROLLING_COLUMNS = [
    "AQI",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)"
]

TIME_COLUMNS = [
    "Day",
    "Month",
    "Year",
    "Time"
]

REQUIRED_COLUMNS = ROLLING_COLUMNS + TIME_COLUMNS


# ============================================================
# 3. ROLLING WINDOWS
# ============================================================

MEAN_MEDIAN_WINDOWS = [6, 12, 24]

STD_WINDOWS = [6, 12, 24]


# ============================================================
# 4. PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    station_folder = input_file.parent

    output_file = station_folder / OUTPUT_FILENAME

    print("\n" + "=" * 70)
    print(f"Processing station: {station_folder.name}")
    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print(
        f"Loaded dataset: {df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    # --------------------------------------------------------
    # CREATE TIMESTAMP
    # --------------------------------------------------------

    df["_Timestamp"] = pd.to_datetime(
        df["Day"].astype(str).str.zfill(2) + "-"
        + df["Month"].astype(str).str.zfill(2) + "-"
        + df["Year"].astype(str) + " "
        + df["Time"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    # --------------------------------------------------------
    # CHECK INVALID TIMESTAMPS
    # --------------------------------------------------------

    invalid_timestamp_count = int(
        df["_Timestamp"].isna().sum()
    )

    if invalid_timestamp_count > 0:
        raise ValueError(
            f"Found {invalid_timestamp_count:,} invalid timestamps."
        )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    df = (
        df.sort_values("_Timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CHECK DUPLICATE TIMESTAMPS
    # --------------------------------------------------------

    duplicate_count = int(
        df["_Timestamp"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate timestamps. "
            "Each timestamp must be unique."
        )

    print(
        f"Time range: {df['_Timestamp'].min()} → "
        f"{df['_Timestamp'].max()}"
    )

    # --------------------------------------------------------
    # DETECT TIMESTAMP GAPS
    # --------------------------------------------------------

    timestamp_difference = df["_Timestamp"].diff()

    gap_mask = (
        timestamp_difference.notna()
        & (
            timestamp_difference
            != pd.Timedelta(hours=1)
        )
    )

    gap_count = int(gap_mask.sum())

    if gap_count > 0:

        print(
            f"\nWARNING: {gap_count:,} timestamp gaps "
            "larger than one hour were detected."
        )

        print(
            "Rolling features will be NaN across gaps "
            "until a complete consecutive window is available."
        )

    else:

        print("\nTimestamp frequency check: hourly")

    # --------------------------------------------------------
    # CONVERT ROLLING COLUMNS TO NUMERIC
    # --------------------------------------------------------

    for column in ROLLING_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # CREATE ROLLING FEATURES
    # --------------------------------------------------------

    print("\nCreating rolling features...")

    new_features = {}

    feature_count = 0

    for column in ROLLING_COLUMNS:

        print(f"Processing: {column}")

        # Exclude the current observation.
        past_values = df[column].shift(1)

        base_name = column.split(" (")[0]

        # ====================================================
        # MEAN AND MEDIAN: 6, 12, 24 HOURS
        # ====================================================

        for window in MEAN_MEDIAN_WINDOWS:

            # Check whether the previous window spans exactly
            # the expected number of consecutive hours.
            oldest_timestamp = (
                df["_Timestamp"].shift(window)
            )

            continuous_window = (
                (
                    df["_Timestamp"] - oldest_timestamp
                )
                == pd.Timedelta(hours=window)
            )

            # Rolling mean
            mean_name = (
                f"{base_name}_roll_mean_{window}"
            )

            rolling_mean = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window
                )
                .mean()
            )

            rolling_mean = rolling_mean.where(
                continuous_window
            )

            new_features[mean_name] = rolling_mean

            feature_count += 1

            # Rolling median
            median_name = (
                f"{base_name}_roll_median_{window}"
            )

            rolling_median = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window
                )
                .median()
            )

            rolling_median = rolling_median.where(
                continuous_window
            )

            new_features[median_name] = rolling_median

            feature_count += 1

        # ====================================================
        # STANDARD DEVIATION: 6, 12, 24 HOURS
        # ====================================================

        for window in STD_WINDOWS:

            oldest_timestamp = (
                df["_Timestamp"].shift(window)
            )

            continuous_window = (
                (
                    df["_Timestamp"] - oldest_timestamp
                )
                == pd.Timedelta(hours=window)
            )

            std_name = (
                f"{base_name}_roll_std_{window}"
            )

            rolling_std = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window
                )
                .std()
            )

            rolling_std = rolling_std.where(
                continuous_window
            )

            new_features[std_name] = rolling_std

            feature_count += 1

    # --------------------------------------------------------
    # ADD ALL FEATURES
    # --------------------------------------------------------

    rolling_features_df = pd.DataFrame(
        new_features,
        index=df.index
    )

    df = pd.concat(
        [df, rolling_features_df],
        axis=1
    )

    # --------------------------------------------------------
    # REMOVE TEMPORARY TIMESTAMP
    # --------------------------------------------------------

    df = df.drop(
        columns=["_Timestamp"]
    )

    # --------------------------------------------------------
    # SAVE FINAL DATASET
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY FEATURE NAMES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROLLING FEATURES CREATED")
    print("=" * 70)

    for column in ROLLING_COLUMNS:

        base_name = column.split(" (")[0]

        print(f"\n{base_name}:")

        for window in MEAN_MEDIAN_WINDOWS:

            print(f"  {base_name}_roll_mean_{window}")
            print(f"  {base_name}_roll_median_{window}")

        for window in STD_WINDOWS:

            print(f"  {base_name}_roll_std_{window}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATION ROLLING FEATURE CREATION COMPLETED")
    print("=" * 70)

    print(
        f"Final dataset shape: "
        f"{df.shape[0]:,} rows × {df.shape[1]:,} columns"
    )

    print(f"Rolling features: {feature_count:,}")
    print(f"Output file: {output_file}")

    print("\nRolling windows:")
    print("  6 hours  -> previous 6 consecutive hours")
    print("  12 hours -> previous 12 consecutive hours")
    print("  24 hours -> previous 24 consecutive hours")

    print("\nStatistics:")
    print("  Mean")
    print("  Median")
    print("  Standard deviation")

    print("\nCurrent observation:")
    print("  Excluded from rolling calculations")

    print("\nTimestamp gaps:")
    print("  Rolling windows are invalid across timestamp gaps")

    print("\nData leakage:")
    print("  Current-row values are not used in rolling features")


# ============================================================
# 5. MAIN — PROCESS ALL STATIONS
# ============================================================

def main():

    if not MAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{MAIN_FOLDER}"
        )

    input_files = sorted(
        MAIN_FOLDER.rglob(INPUT_FILENAME)
    )

    # Exclude files inside Raw data folders.
    input_files = [
        file for file in input_files
        if file.parent.name.lower() != "raw data"
    ]

    print(
        f"\nNumber of station files found: {len(input_files)}"
    )

    if not input_files:
        raise FileNotFoundError(
            f"No '{INPUT_FILENAME}' files found under:\n"
            f"{MAIN_FOLDER}\n"
            "Run Part 4 first."
        )

    successful = 0
    failed = 0

    for input_file in input_files:

        try:
            process_station(input_file)
            successful += 1

        except Exception as error:

            failed += 1

            print("\nERROR processing station:")
            print(input_file)
            print(f"Reason: {error}")

            # Continue processing the remaining stations.
            continue

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5 — ALL STATIONS COMPLETED")
    print("=" * 70)

    print(f"Files found: {len(input_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
