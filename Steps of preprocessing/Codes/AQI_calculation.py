import pandas as pd
import numpy as np
import glob
import os


# ============================================================
# CPCB AQI BREAKPOINTS
# ============================================================

breakpoints = {

    "PM2.5 (µg/m³)": [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400)
    ],

    "PM10 (µg/m³)": [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400)
    ],

    "NO2 (µg/m³)": [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 180, 100, 200),
        (180, 280, 200, 300),
        (280, 400, 300, 400)
    ],

    "SO2 (µg/m³)": [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 380, 100, 200),
        (380, 800, 200, 300),
        (800, 1600, 300, 400)
    ],

    "NH3 (µg/m³)": [
        (0, 200, 0, 50),
        (200, 400, 50, 100),
        (400, 800, 100, 200),
        (800, 1200, 200, 300),
        (1200, 1800, 300, 400)
    ],

    "CO (mg/m³)": [
        (0, 1.0, 0, 50),
        (1.0, 2.0, 50, 100),
        (2.0, 10, 100, 200),
        (10.0, 17, 200, 300),
        (17.0, 34, 300, 400)
    ],

    "Ozone (µg/m³)": [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 168, 100, 200),
        (168, 208, 200, 300),
        (208, 748, 300, 400)
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

    for BPL, BPH, ILo, IHi in breakpoints[pollutant]:

        if BPL <= concentration <= BPH:

            index = ((IHi - ILo) / (BPH - BPL)) * \
                    (concentration - BPL) + ILo

            return index

    # Above highest breakpoint
    return 500


# ============================================================
# PROCESS ALL IMPUTED FILES
# ============================================================

base_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data"

files = glob.glob(
    os.path.join(base_folder, "**", "*_imputed.csv"),
    recursive=True
)

print("Number of files found:", len(files))


for file in files:

    print("\nProcessing:", file)

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Create timestamp for time-based averaging
    # --------------------------------------------------------

    had_timestamp = "Timestamp" in df.columns

    if had_timestamp:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        df["Timestamp"] = pd.to_datetime(
            df[["Year", "Month", "Day"]].astype(str).agg("-".join, axis=1)
            + " "
            + df["Time"].astype(str)
        )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # Calculate CPCB running averages
    #
    # 24-hour average:
    # PM2.5, PM10, NO2, SO2, NH3
    #
    # 8-hour average:
    # CO, Ozone
    #
    # CPCB requires at least 16 hours of data for a sub-index.
    # --------------------------------------------------------

    average_periods = {
        "PM2.5 (µg/m³)": 24,
        "PM10 (µg/m³)": 24,
        "NO2 (µg/m³)": 24,
        "SO2 (µg/m³)": 24,
        "NH3 (µg/m³)": 24,
        "CO (mg/m³)": 8,
        "Ozone (µg/m³)": 8
    }

    averaged_values = {}

    for pollutant, hours in average_periods.items():

        if pollutant not in df.columns:
            print("WARNING: Missing pollutant column:", pollutant)
            averaged_values[pollutant] = pd.Series(
                np.nan, index=df.index
            )
            continue

        averaged_values[pollutant] = (
            df[pollutant]
            .rolling(window=hours, min_periods=hours)
            .mean()
        )

    # --------------------------------------------------------
    # Calculate all pollutant sub-indices TEMPORARILY
    # --------------------------------------------------------

    sub_indices = []

    for pollutant in breakpoints.keys():

        sub_index = averaged_values[pollutant].apply(
            lambda x: calculate_sub_index(x, pollutant)
        )

        sub_indices.append(sub_index)

    # --------------------------------------------------------
    # AQI = maximum pollutant sub-index
    # --------------------------------------------------------

    sub_index_df = pd.concat(sub_indices, axis=1)

    valid_count = sub_index_df.notna().sum(axis=1)
    particulate_available = (
        sub_index_df["PM2.5 (µg/m³)"].notna()
        | sub_index_df["PM10 (µg/m³)"].notna()
    )

    df["AQI"] = sub_index_df.max(axis=1)

    df.loc[
        (valid_count < 3) | (~particulate_available),
        "AQI"
    ] = np.nan

    df["AQI"] = df["AQI"].round().astype("Int64")

    # --------------------------------------------------------
    # Remove temporary Timestamp if it was not in the input
    # --------------------------------------------------------

    if not had_timestamp:
        df.drop(columns=["Timestamp"], inplace=True)

    # --------------------------------------------------------
    # Save in SAME folder
    # --------------------------------------------------------

    output_file = file.replace(
        "_imputed.csv",
        "_with_AQI.csv"
    )

    df.to_csv(output_file, index=False)

    print("Saved:", output_file)

    print("AQI calculated successfully.")
    print("Rows:", len(df))


print("\n======================================")
print("AQI CALCULATION COMPLETED")
print("======================================")
