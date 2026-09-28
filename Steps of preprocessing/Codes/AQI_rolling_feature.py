import pandas as pd
import numpy as np
import glob
import os


# ============================================================
# PATH
# ============================================================

base_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data"


# ============================================================
# FIND FILES
# ============================================================

files = glob.glob(
    os.path.join(
        base_folder,
        "**",
        "*_with_AQI_lags.csv"
    ),
    recursive=True
)

print("Number of files found:", len(files))


# ============================================================
# VARIABLES
# ============================================================

rolling_columns = [
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
# ROLLING WINDOWS
# ============================================================

windows = [1, 6, 12, 24]


# ============================================================
# SHORT NAMES
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
# PROCESS EACH FILE
# ============================================================

for file in files:

    print("\n==========================================")
    print("Processing:", os.path.basename(file))
    print("==========================================")

    df = pd.read_csv(file)

    original_rows = len(df)

    # --------------------------------------------------------
    # Temporary timestamp
    # --------------------------------------------------------

    df["_timestamp"] = pd.to_datetime(
        df["Day"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Year"].astype(str) + " " +
        df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "_timestamp"
    ).reset_index(drop=True)


    # ========================================================
    # CREATE ROLLING FEATURES
    # ========================================================

    for column in rolling_columns:

        prefix = name_map[column]

        # Only previous observations
        past_values = df[column].shift(1)

        for window in windows:

            # ------------------------------------------------
            # Rolling Mean
            # ------------------------------------------------

            mean_column = (
                f"{prefix}_roll_mean_{window}"
            )

            df[mean_column] = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window
                )
                .mean()
            )


            # ------------------------------------------------
            # Rolling Median
            # ------------------------------------------------

            median_column = (
                f"{prefix}_roll_median_{window}"
            )

            df[median_column] = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window
                )
                .median()
            )


            # ------------------------------------------------
            # Rolling Standard Deviation
            # ------------------------------------------------

            std_column = (
                f"{prefix}_roll_std_{window}"
            )

            df[std_column] = (
                past_values
                .rolling(
                    window=window,
                    min_periods=window
                )
                .std()
            )


    # ========================================================
    # REMOVE TEMPORARY TIMESTAMP
    # ========================================================

    df = df.drop(
        columns=["_timestamp"]
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_file = file.replace(
        "_with_AQI_lags.csv",
        "_with_AQI_lags_rolling.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("Original rows:", original_rows)
    print("Final rows:", len(df))

    print("\nRolling features:")
    print("Mean   : 1, 6, 12, 24 hours")
    print("Median : 1, 6, 12, 24 hours")
    print("Std    : 1, 6, 12, 24 hours")

    print("\nVariables:")
    print("AQI + 7 pollutants")

    print("\nSaved to:")
    print(output_file)


print("\n==========================================")
print("ROLLING STATISTICS COMPLETED")
print("==========================================")