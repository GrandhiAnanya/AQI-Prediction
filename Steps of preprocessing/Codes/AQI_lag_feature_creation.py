import pandas as pd
import numpy as np
import glob
import os


# ============================================================
# PATH
# ============================================================

base_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data"


# ============================================================
# FIND AQI FILES
# ============================================================

files = glob.glob(
    os.path.join(base_folder, "**", "*_with_AQI.csv"),
    recursive=True
)

print("Number of files found:", len(files))


# ============================================================
# VARIABLES FOR LAG CREATION
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

lag_hours = [1, 6, 12, 24]


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file in files:

    print("\n==========================================")
    print("Processing:", os.path.basename(file))
    print("==========================================")

    df = pd.read_csv(file)

    original_rows = len(df)

    # --------------------------------------------------------
    # Create temporary timestamp
    # --------------------------------------------------------

    df["_timestamp"] = pd.to_datetime(
        df["Day"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Year"].astype(str) + " " +
        df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    # Check invalid timestamps
    invalid_timestamps = df["_timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(
            "WARNING:",
            invalid_timestamps,
            "invalid timestamps found."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values("_timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # Handle duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = df["_timestamp"].duplicated().sum()

    print("Duplicate timestamps:", duplicate_count)

    if duplicate_count > 0:

        # Keep the first record for each timestamp
        df = df.drop_duplicates(
            subset=["_timestamp"],
            keep="first"
        ).reset_index(drop=True)

        print("Rows after removing duplicate timestamps:", len(df))

    # --------------------------------------------------------
    # Create time-based lag variables
    # --------------------------------------------------------

    for column in lag_columns:

        for lag in lag_hours:

            lag_name = column

            # Make shorter names for output columns
            if column == "AQI":
                prefix = "AQI"

            elif column == "PM2.5 (µg/m³)":
                prefix = "PM2.5"

            elif column == "PM10 (µg/m³)":
                prefix = "PM10"

            elif column == "NO2 (µg/m³)":
                prefix = "NO2"

            elif column == "SO2 (µg/m³)":
                prefix = "SO2"

            elif column == "CO (mg/m³)":
                prefix = "CO"

            elif column == "Ozone (µg/m³)":
                prefix = "Ozone"

            elif column == "NH3 (µg/m³)":
                prefix = "NH3"

            new_column = f"{prefix}_lag_{lag}"

            # ------------------------------------------------
            # Time-aware lag
            # ------------------------------------------------

            lag_timestamp = df["_timestamp"] - pd.Timedelta(
                hours=lag
            )

            lookup = pd.Series(
                df[column].values,
                index=df["_timestamp"]
            )

            df[new_column] = lag_timestamp.map(lookup)

    # --------------------------------------------------------
    # Remove temporary timestamp
    # --------------------------------------------------------

    df = df.drop(columns=["_timestamp"])

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_file = file.replace(
        "_with_AQI.csv",
        "_with_AQI_lags.csv"
    )

    df.to_csv(output_file, index=False)

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print("Original rows:", original_rows)
    print("Final rows:", len(df))

    print("\nLag variables created:")

    for column in lag_columns:

        if column == "AQI":
            prefix = "AQI"
        elif column == "PM2.5 (µg/m³)":
            prefix = "PM2.5"
        elif column == "PM10 (µg/m³)":
            prefix = "PM10"
        elif column == "NO2 (µg/m³)":
            prefix = "NO2"
        elif column == "SO2 (µg/m³)":
            prefix = "SO2"
        elif column == "CO (mg/m³)":
            prefix = "CO"
        elif column == "Ozone (µg/m³)":
            prefix = "Ozone"
        elif column == "NH3 (µg/m³)":
            prefix = "NH3"

        print(
            f"{prefix}:",
            [f"{prefix}_lag_{lag}" for lag in lag_hours]
        )

    print("\nSaved to:")
    print(output_file)


print("\n==========================================")
print("LAG FEATURE CREATION COMPLETED")
print("==========================================")