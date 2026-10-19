
import pandas as pd
from pathlib import Path

# ============================================================
# STEP 9 — DATE TRIMMING AND NULL TARGET REMOVAL
# ============================================================

print("=" * 70)
print("STEP 9 — DATE TRIMMING AND NULL TARGET REMOVAL")
print("=" * 70)

# ============================================================
# 1. FILE PATHS
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = (
    "merged_with_weather_with_elevation_final_with_future_AQI.csv"
)

OUTPUT_FILENAME = (
    "merged_with_weather_with_elevation_final_with_future_AQI_trimmed.csv"
)

# ============================================================
# 2. TRAIN / VALIDATION / TEST PERIODS
# ============================================================

SPLIT_PERIODS = {
    "train": (
        pd.Timestamp("2020-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:00:00")
    ),
    "validation": (
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 00:00:00")
    ),
    "test": (
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 00:00:00")
    )
}

TARGET_HORIZONS = {
    "AQI_future_1": 1,
    "AQI_future_24": 24
}

# ============================================================
# 3. PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    output_file = input_file.parent / OUTPUT_FILENAME

    print("\n" + "=" * 70)
    print(f"Processing station: {input_file.parent.name}")
    print("=" * 70)

    df = pd.read_csv(input_file)
    original_rows = len(df)

    print(f"Original rows: {original_rows:,}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "Day", "Month", "Year", "Time",
        "AQI_future_1", "AQI_future_24"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Create timestamp
    # --------------------------------------------------------

    timestamp_string = (
        df["Day"].astype(str).str.zfill(2) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Year"].astype(str) + " " +
        df["Time"].astype(str)
    )

    df["_Timestamp"] = pd.to_datetime(
        timestamp_string,
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    invalid_timestamps = df["_Timestamp"].isna().sum()

    if invalid_timestamps > 0:
        raise ValueError(
            f"Found {invalid_timestamps:,} invalid timestamps."
        )

    # --------------------------------------------------------
    # Sort and check duplicates
    # --------------------------------------------------------

    df = df.sort_values(
        "_Timestamp"
    ).reset_index(drop=True)

    duplicate_count = df["_Timestamp"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate timestamps."
        )

    # --------------------------------------------------------
    # Convert targets to numeric
    # --------------------------------------------------------

    for target in TARGET_HORIZONS:
        df[target] = pd.to_numeric(
            df[target],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove January 1, 2020
    # --------------------------------------------------------

    remove_jan_1 = (
        (df["_Timestamp"] >= pd.Timestamp("2020-01-01"))
        & (df["_Timestamp"] < pd.Timestamp("2020-01-02"))
    )

    jan_rows_removed = int(remove_jan_1.sum())

    df = df.loc[~remove_jan_1].copy()

    print(
        f"Rows removed for January 1, 2020: "
        f"{jan_rows_removed:,}"
    )

    # --------------------------------------------------------
    # Keep only the defined study period: 2020–2025
    # --------------------------------------------------------

    study_period_mask = (
        (df["_Timestamp"] >= pd.Timestamp("2020-01-01"))
        & (df["_Timestamp"] < pd.Timestamp("2026-01-01"))
    )

    outside_period_removed = int((~study_period_mask).sum())

    df = df.loc[study_period_mask].copy()

    print(
        f"Rows outside 2020–2025 removed: "
        f"{outside_period_removed:,}"
    )

    # --------------------------------------------------------
    # Assign each row to its temporal split
    # --------------------------------------------------------

    current_split = pd.Series(
        index=df.index,
        dtype="object"
    )

    for split_name, (start, end) in SPLIT_PERIODS.items():

        mask = (
            (df["_Timestamp"] >= start)
            & (df["_Timestamp"] < end)
        )

        current_split.loc[mask] = split_name

    # --------------------------------------------------------
    # Remove rows whose future targets cross split boundaries
    # --------------------------------------------------------

    keep_mask = current_split.notna()

    for target_column, horizon in TARGET_HORIZONS.items():

        future_timestamp = (
            df["_Timestamp"]
            + pd.Timedelta(hours=horizon)
        )

        future_split = pd.Series(
            index=df.index,
            dtype="object"
        )

        for split_name, (start, end) in SPLIT_PERIODS.items():

            mask = (
                (future_timestamp >= start)
                & (future_timestamp < end)
            )

            future_split.loc[mask] = split_name

        same_split = (
            current_split.notna()
            & future_split.notna()
            & (current_split == future_split)
        )

        keep_mask &= same_split

    boundary_rows_removed = int((~keep_mask).sum())

    df = df.loc[keep_mask].copy()

    print(
        f"Rows removed at split boundaries: "
        f"{boundary_rows_removed:,}"
    )

    # --------------------------------------------------------
    # Remove rows missing either future target
    # --------------------------------------------------------

    before_target_drop = len(df)

    df = df.dropna(
        subset=["AQI_future_1", "AQI_future_24"]
    ).copy()

    null_target_rows_removed = before_target_drop - len(df)

    print(
        f"Rows removed due to missing future targets: "
        f"{null_target_rows_removed:,}"
    )

    # --------------------------------------------------------
    # Remove temporary timestamp and reset index
    # --------------------------------------------------------

    df.drop(
        columns=["_Timestamp"],
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Move targets to the end
    # --------------------------------------------------------

    target_columns = [
        "AQI_future_1",
        "AQI_future_24"
    ]

    other_columns = [
        col for col in df.columns
        if col not in target_columns
    ]

    df = df[other_columns + target_columns]

    # --------------------------------------------------------
    # Final target null check
    # --------------------------------------------------------

    remaining_target_nulls = int(
        df[target_columns].isna().sum().sum()
    )

    if remaining_target_nulls > 0:
        raise ValueError(
            "Future target null values remain after trimming."
        )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False
    )

    print("\nStation summary:")
    print(f"  Original rows:          {original_rows:,}")
    print(f"  Final rows:             {len(df):,}")
    print(
        f"  Total rows removed:     "
        f"{original_rows - len(df):,}"
    )
    print(
        f"  Remaining target nulls: "
        f"{remaining_target_nulls:,}"
    )
    print(f"  Saved to: {output_file}")

    return {
        "station": input_file.parent.name,
        "status": "SUCCESS",
        "original_rows": original_rows,
        "final_rows": len(df),
        "rows_removed": original_rows - len(df),
        "remaining_target_nulls": remaining_target_nulls,
        "output": str(output_file)
    }


# ============================================================
# 4. FIND ALL STATION FILES
# ============================================================

if not MAIN_FOLDER.exists():
    raise FileNotFoundError(
        f"Main folder not found: {MAIN_FOLDER}"
    )

input_files = sorted([
    path
    for path in MAIN_FOLDER.rglob(INPUT_FILENAME)
    if path.is_file()
    and path.parent.name.lower() != "raw data"
])

if not input_files:
    raise FileNotFoundError(
        f"No files named '{INPUT_FILENAME}' found "
        f"under {MAIN_FOLDER}"
    )

print(f"\nFound {len(input_files)} station files.")

# ============================================================
# 5. PROCESS ALL STATIONS
# ============================================================

results = []

for input_file in input_files:

    try:
        results.append(process_station(input_file))

    except Exception as e:

        print(
            f"\nERROR processing {input_file.parent.name}: {e}"
        )

        results.append({
            "station": input_file.parent.name,
            "status": f"FAILED: {e}",
            "original_rows": None,
            "final_rows": None,
            "rows_removed": None,
            "remaining_target_nulls": None,
            "output": None
        })

# ============================================================
# 6. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("STEP 9 — FINAL SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))

print(
    "\nSuccessful stations:",
    (summary_df["status"] == "SUCCESS").sum()
)

print(
    "Failed stations:",
    (summary_df["status"] != "SUCCESS").sum()
)

print("\nSTEP 9 COMPLETED")
print("=" * 70)
