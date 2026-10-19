
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = "merged_with_weather_with_elevation.csv"
OUTPUT_FILENAME = "Long_Gap_Analysis.xlsx"


# ============================================================
# POLLUTANT COLUMNS
# ============================================================


POLLUTANT_COLS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)"
]



# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLS = [
    "Day",
    "Month",
    "Year",
    "Time",
    "Day_of_Week",
    "Season"
] + POLLUTANT_COLS


# ============================================================
# BUILD TIMESTAMP
# ============================================================

def build_timestamp(df):

    timestamp = pd.to_datetime(
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Day"].astype(str) + " " +
        df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    if timestamp.isna().any():
        invalid_count = timestamp.isna().sum()

        raise ValueError(
            f"{invalid_count} rows have invalid timestamps."
        )

    return timestamp


# ============================================================
# CLASSIFY GAP
# ============================================================

def classify_gap(gap_hours):

    if gap_hours <= 3:
        return "1-3 h"

    elif gap_hours <= 72:
        return "4-72 h"

    else:
        return ">72 h"


# ============================================================
# PROPOSE ACTION
# ============================================================

def proposed_action(gap_category):

    if gap_category == "1-3 h":
        return "Time interpolation"

    elif gap_category == "4-72 h":
        return "Contextual median"

    else:
        return "Leave / Drop"


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_columns(df, file_path):

    missing_cols = [
        col for col in REQUIRED_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {file_path.name}: "
            f"{missing_cols}"
        )


# ============================================================
# IDENTIFY CONSECUTIVE MISSING-VALUE RUNS
# ============================================================

def identify_missing_runs(df):

    records = []

    for pollutant in POLLUTANT_COLS:

        missing_mask = df[pollutant].isna().to_numpy()

        if not missing_mask.any():
            continue

        # Find the beginning and end of each missing run
        starts = np.where(
            missing_mask &
            ~np.r_[False, missing_mask[:-1]]
        )[0]

        ends = np.where(
            missing_mask &
            ~np.r_[missing_mask[1:], False]
        )[0]

        for gap_id, (start_idx, end_idx) in enumerate(
            zip(starts, ends),
            start=1
        ):

            gap_start = df.loc[start_idx, "_timestamp"]
            gap_end = df.loc[end_idx, "_timestamp"]

            # Preserve the original gap-duration calculation
            gap_hours = int(
                round(
                    (gap_end - gap_start).total_seconds() / 3600
                ) + 1
            )

            gap_category = classify_gap(gap_hours)

            records.append({
                "Variable": pollutant,
                "Gap_ID": gap_id,
                "Gap_Start": gap_start,
                "Gap_End": gap_end,
                "Gap_Hours": gap_hours,
                "Gap_Category": gap_category,
                "Proposed_Action": proposed_action(gap_category),
                "Gap_Year": gap_start.year
            })

    return pd.DataFrame(records)


# ============================================================
# PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    print("\n" + "=" * 70)
    print(f"Processing station: {input_file.parent.name}")
    print(f"Input file: {input_file}")
    print("=" * 70)

    # Load station dataset
    df = pd.read_csv(input_file)

    # Validate required columns
    validate_columns(df, input_file)

    # Build timestamps
    df["_timestamp"] = build_timestamp(df)

    # Sort chronologically
    df = df.sort_values("_timestamp").reset_index(drop=True)

    # Check duplicate timestamps
    duplicate_count = df["_timestamp"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    # Check whether timestamps are hourly
    time_differences = df["_timestamp"].diff().dropna()

    non_hourly_count = (
        time_differences != pd.Timedelta(hours=1)
    ).sum()

    if non_hourly_count > 0:
        warnings.warn(
            f"{non_hourly_count} timestamp intervals are not "
            f"exactly one hour in {input_file.parent.name}. "
            "Check for missing timestamp rows or irregular intervals."
        )

    # Identify missing-value runs
    gap_df = identify_missing_runs(df)

    # Prepare output path in the same station folder
    output_file = input_file.parent / OUTPUT_FILENAME

    # ========================================================
    # EXCEL REPORT
    # ========================================================

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        # All missing-value runs
        gap_df.to_excel(
            writer,
            sheet_name="All_Gaps",
            index=False
        )

        # Separate reports by gap category
        gap_df[
            gap_df["Gap_Category"] == "1-3 h"
        ].to_excel(
            writer,
            sheet_name="1-3_Hours",
            index=False
        )

        gap_df[
            gap_df["Gap_Category"] == "4-72 h"
        ].to_excel(
            writer,
            sheet_name="4-72_Hours",
            index=False
        )

        gap_df[
            gap_df["Gap_Category"] == ">72 h"
        ].to_excel(
            writer,
            sheet_name="Over_72_Hours",
            index=False
        )

        # Summary by pollutant
        if not gap_df.empty:

            variable_summary = (
                gap_df.groupby("Variable")
                .agg(
                    Total_Gaps=("Gap_ID", "count"),
                    Total_Missing_Hours=("Gap_Hours", "sum"),
                    Average_Gap_Hours=("Gap_Hours", "mean"),
                    Maximum_Gap_Hours=("Gap_Hours", "max")
                )
                .reset_index()
            )

        else:

            variable_summary = pd.DataFrame(
                columns=[
                    "Variable",
                    "Total_Gaps",
                    "Total_Missing_Hours",
                    "Average_Gap_Hours",
                    "Maximum_Gap_Hours"
                ]
            )

        variable_summary.to_excel(
            writer,
            sheet_name="Variable_Summary",
            index=False
        )

        # Summary by gap category
        if not gap_df.empty:

            category_summary = (
                gap_df.groupby("Gap_Category")
                .agg(
                    Total_Gaps=("Gap_ID", "count"),
                    Total_Missing_Hours=("Gap_Hours", "sum")
                )
                .reset_index()
            )

        else:

            category_summary = pd.DataFrame(
                columns=[
                    "Gap_Category",
                    "Total_Gaps",
                    "Total_Missing_Hours"
                ]
            )

        category_summary.to_excel(
            writer,
            sheet_name="Category_Summary",
            index=False
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(f"Total rows: {len(df)}")
    print(f"Total missing-value runs: {len(gap_df)}")

    if not gap_df.empty:

        print("\nGap category counts:")
        print(
            gap_df["Gap_Category"]
            .value_counts()
            .reindex(
                ["1-3 h", "4-72 h", ">72 h"],
                fill_value=0
            )
        )

    print(f"\nReport saved to: {output_file}")


# ============================================================
# MAIN: PROCESS ALL STATIONS AUTOMATICALLY
# ============================================================

def main():

    if not MAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {MAIN_FOLDER}"
        )

    # Search for the input file in every subfolder
    input_files = list(
        MAIN_FOLDER.rglob(INPUT_FILENAME)
    )

    # Avoid accidentally processing files inside Raw data folders
    input_files = [
        file for file in input_files
        if file.parent.name.lower() != "raw data"
    ]

    if not input_files:
        print(
            f"No files named '{INPUT_FILENAME}' "
            f"were found under:\n{MAIN_FOLDER}"
        )
        return

    print(f"Stations/files found: {len(input_files)}")

    successful = 0
    failed = 0

    for input_file in input_files:

        try:
            process_station(input_file)
            successful += 1

        except Exception as error:

            failed += 1

            print("\nERROR processing:")
            print(input_file)
            print(f"Reason: {error}")

            # Continue processing the remaining stations
            continue

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("LONG GAP ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"Files found: {len(input_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")

    print(
        "\nEach successful station has its own "
        f"'{OUTPUT_FILENAME}' report."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
