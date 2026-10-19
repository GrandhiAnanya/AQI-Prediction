
import pandas as pd
from pathlib import Path


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = "merged_with_weather_with_elevation_with_AQI.csv"

OUTPUT_FILENAME = "merged_with_weather_with_elevation_with_AQI_lags.csv"


# ============================================================
# 2. VARIABLES
# ============================================================

lag_columns = [
    "AQI",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)"
]


# ============================================================
# 3. LAG HOURS
# ============================================================

lag_hours = [
    1,
    6,
    12,
    24
]


# ============================================================
# 4. SHORT NAMES
# ============================================================

name_map = {
    "AQI": "AQI",
    "PM2.5 (µg/m³)": "PM2.5",
    "PM10 (µg/m³)": "PM10",
    "NO2 (µg/m³)": "NO2",
    "SO2 (µg/m³)": "SO2",
    "CO (mg/m³)": "CO",
    "Ozone (µg/m³)": "Ozone",
    "NH3 (µg/m³)": "NH3"
}


# ============================================================
# 5. PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    station_folder = input_file.parent
    output_file = station_folder / OUTPUT_FILENAME

    print("\n" + "=" * 70)
    print(f"Processing station: {station_folder.name}")
    print(f"Input : {input_file}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "Day",
        "Month",
        "Year",
        "Time"
    ] + lag_columns

    missing_columns = [
        column
        for column in required_columns
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

    df["_timestamp"] = pd.to_datetime(
        df["Day"].astype(str)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
        + "-"
        + df["Year"].astype(str)
        + " "
        + df["Time"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce"
    )

    if df["_timestamp"].isna().any():

        bad = int(df["_timestamp"].isna().sum())

        raise ValueError(
            f"{bad} invalid timestamps found."
        )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    df = (
        df.sort_values("_timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CHECK DUPLICATE TIMESTAMPS
    # --------------------------------------------------------

    duplicate_timestamps = int(
        df["_timestamp"].duplicated().sum()
    )

    if duplicate_timestamps:

        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps."
        )

    # --------------------------------------------------------
    # CREATE LAG FEATURES
    # --------------------------------------------------------

    for column in lag_columns:

        prefix = name_map[column]

        # Timestamp-based lookup:
        # current timestamp minus the specified lag duration.

        lookup = pd.Series(
            df[column].values,
            index=df["_timestamp"]
        )

        for lag in lag_hours:

            lag_timestamp = (
                df["_timestamp"]
                - pd.Timedelta(hours=lag)
            )

            new_column = f"{prefix}_lag_{lag}"

            df[new_column] = lag_timestamp.map(lookup)

    # --------------------------------------------------------
    # REMOVE TEMPORARY TIMESTAMP
    # --------------------------------------------------------

    df = df.drop(columns=["_timestamp"])

    # --------------------------------------------------------
    # SAVE OUTPUT IN THE SAME STATION FOLDER
    # --------------------------------------------------------

    df.to_csv(output_file, index=False)

    print("\nLag variables created:")
    print("1, 6, 12 and 24 hours")

    print("Final shape:", df.shape)
    print("Saved:", output_file)
    print("Part 4 completed successfully.")


# ============================================================
# 6. MAIN: PROCESS ALL STATIONS
# ============================================================

def main():

    print("=" * 70)
    print("STEP 4 — STATION-WISE LAG FEATURE CREATION")
    print("=" * 70)

    if not MAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{MAIN_FOLDER}"
        )

    # Find the AQI file in every station subfolder.
    files = sorted(
        MAIN_FOLDER.rglob(INPUT_FILENAME)
    )

    # Exclude files inside Raw data folders.
    files = [
        file for file in files
        if file.parent.name.lower() != "raw data"
    ]

    print("Number of station files found:", len(files))

    if not files:
        raise FileNotFoundError(
            f"No '{INPUT_FILENAME}' files found under:\n"
            f"{MAIN_FOLDER}\n"
            "Run Part 3 first."
        )

    successful = 0
    failed = 0

    for input_file in files:

        try:
            process_station(input_file)
            successful += 1

        except Exception as error:

            failed += 1

            print("\nERROR processing station:")
            print(input_file)
            print("Reason:", error)

            # Continue processing the remaining stations.
            continue

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATION-WISE LAG CREATION COMPLETED")
    print("=" * 70)

    print("Files found:", len(files))
    print("Successfully processed:", successful)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
