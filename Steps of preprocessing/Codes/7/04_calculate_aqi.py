
import pandas as pd
import numpy as np
from pathlib import Path
import warnings


# ============================================================
# PATH CONFIGURATION
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = "merged_with_weather_with_elevation_imputed.csv"
OUTPUT_FILENAME = "merged_with_weather_with_elevation_with_AQI.csv"


# ============================================================
# CPCB AQI BREAKPOINTS
# ============================================================

breakpoints = {

    "PM2.5 (µg/m³)": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400)
    ],

    "PM10 (µg/m³)": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400)
    ],

    "NO2 (µg/m³)": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400)
    ],

    "SO2 (µg/m³)": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400)
    ],

    "NH3 (µg/m³)": [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400)
    ],

    "CO (mg/m³)": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400)
    ],

    "Ozone (µg/m³)": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400)
    ]
}


# ============================================================
# FUNCTION TO CALCULATE POLLUTANT SUB-INDEX
# ============================================================

def calculate_sub_index(concentration, pollutant):

    if pd.isna(concentration):
        return np.nan

    if concentration < 0:
        return np.nan

    ranges = breakpoints[pollutant]

    # Normal CPCB breakpoint ranges
    for BPL, BPH, ILo, IHi in ranges:

        if BPL <= concentration <= BPH:

            index = (
                ((IHi - ILo) / (BPH - BPL))
                * (concentration - BPL)
                + ILo
            )

            return index

    # Handle decimal values between adjacent breakpoints
    for i in range(len(ranges) - 1):

        lower = ranges[i]
        upper = ranges[i + 1]

        lower_concentration = lower[1]
        lower_index = lower[3]

        upper_concentration = upper[0]
        upper_index = upper[2]

        if (
            lower_concentration
            < concentration
            < upper_concentration
        ):

            index = (
                (upper_index - lower_index)
                / (upper_concentration - lower_concentration)
            ) * (
                concentration - lower_concentration
            ) + lower_index

            return index

    # Original upper-breakpoint handling preserved
    if concentration > ranges[-1][1]:
        BPL,BPH,ILo,IHi=ranges[-1]
        index=(
            ((IHi-ILo)/(BPH-BPL))
            *(concentration-BPL)
            +ILo
        )

        return min(index,500)

    return np.nan


# ============================================================
# PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    print("\n==========================================")
    print("Processing station:", input_file.parent.name)
    print("Input:", input_file)
    print("==========================================")

    df = pd.read_csv(input_file)

    # Check required pollutant columns
    missing_columns = [
        pollutant
        for pollutant in breakpoints.keys()
        if pollutant not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing pollutant columns in {input_file.name}: "
            f"{missing_columns}"
        )

    # Create timestamp
    if "Timestamp" in df.columns:

        df["_timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

    else:

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

    # Sort chronologically
    df = (
        df.sort_values("_timestamp")
        .reset_index(drop=True)
    )

    # Check duplicate timestamps
    duplicate_timestamps = int(
        df["_timestamp"].duplicated().sum()
    )

    if duplicate_timestamps:

        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps "
            f"in {input_file.name}."
        )

    # CPCB running averages
    average_periods = {
        "PM2.5 (µg/m³)": ("24h", 16),
        "PM10 (µg/m³)": ("24h", 16),
        "NO2 (µg/m³)": ("24h", 16),
        "SO2 (µg/m³)": ("24h", 16),
        "NH3 (µg/m³)": ("24h", 16),
        "CO (mg/m³)": ("8h", 8),
        "Ozone (µg/m³)": ("8h", 8),
    }

    time_df = df.set_index("_timestamp")
    averaged_values = {}

    for pollutant, (period, min_periods) in average_periods.items():

        averaged_values[pollutant] = (
            time_df[pollutant]
            .rolling(
                window=period,
                min_periods=min_periods,
                closed="right"
            )
            .mean()
            .reset_index(drop=True)
        )

    # Calculate pollutant sub-indices
    sub_indices = {}

    for pollutant in breakpoints.keys():

        sub_indices[pollutant] = (
            averaged_values[pollutant]
            .apply(
                lambda x: calculate_sub_index(
                    x,
                    pollutant
                )
            )
        )

    sub_index_df = pd.DataFrame(
        sub_indices,
        index=df.index
    )

    # AQI = maximum pollutant sub-index
    valid_subindices = (
        sub_index_df.notna().sum(axis=1)
    )

    particulate_available = (
        sub_index_df["PM2.5 (µg/m³)"].notna()
        |
        sub_index_df["PM10 (µg/m³)"].notna()
    )

    df["AQI"] = sub_index_df.max(axis=1)

    df.loc[
        (valid_subindices < 3)
        | (~particulate_available),
        "AQI"
    ] = np.nan

    # Round final AQI
    df["AQI"] = (
        df["AQI"]
        .round()
        .astype("Int64")
    )

    # Remove temporary timestamp
    df = df.drop(columns=["_timestamp"])

    # Save output in the same station folder
    output_file = input_file.parent / OUTPUT_FILENAME

    df.to_csv(output_file, index=False)

    print("Saved:", output_file)
    print("AQI calculated successfully.")
    print("Rows:", len(df))


# ============================================================
# MAIN: PROCESS ALL STATIONS
# ============================================================

def main():

    if not MAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {MAIN_FOLDER}"
        )

    files = sorted(
        MAIN_FOLDER.rglob(INPUT_FILENAME)
    )

    # Avoid processing files inside Raw data folders
    files = [
        file for file in files
        if file.parent.name.lower() != "raw data"
    ]

    print("Number of station files found:", len(files))

    if not files:
        raise FileNotFoundError(
            f"No '{INPUT_FILENAME}' files found under:\n"
            f"{MAIN_FOLDER}"
        )

    successful = 0
    failed = 0

    for file in files:

        try:
            process_station(file)
            successful += 1

        except Exception as error:

            failed += 1

            print("\nERROR processing:", file)
            print("Reason:", error)

            # Continue with other stations
            continue

    print("\n==========================================")
    print("STATION-WISE AQI CALCULATION COMPLETED")
    print("==========================================")
    print("Files found:", len(files))
    print("Successfully processed:", successful)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
