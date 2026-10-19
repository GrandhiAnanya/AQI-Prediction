
import pandas as pd
from pathlib import Path

# ============================================================
# STEP 8 — FUTURE AQI TARGET CREATION (STATION-WISE)
# ============================================================

print("=" * 70)
print("STEP 8 — FUTURE AQI TARGET CREATION")
print("=" * 70)

# ============================================================
# 1. FILE PATHS
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = "merged_with_weather_with_elevation_final.csv"

OUTPUT_FILENAME = (
    "merged_with_weather_with_elevation_final_with_future_AQI.csv"
)

# ============================================================
# 2. PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    output_file = input_file.parent / OUTPUT_FILENAME

    print("\n" + "=" * 70)
    print(f"Processing station: {input_file.parent.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print(
        f"Loaded: {df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Day", "Month", "Year", "Time", "AQI"
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

    df["_Timestamp"] = pd.to_datetime(
        df["Day"].astype(str).str.zfill(2) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Year"].astype(str) + " " +
        df["Time"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    invalid_count = df["_Timestamp"].isna().sum()

    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count:,} invalid timestamps."
        )

    # --------------------------------------------------------
    # Check duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = df["_Timestamp"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate timestamps."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "_Timestamp"
    ).reset_index(drop=True)

    print(
        f"Time range: {df['_Timestamp'].min()} → "
        f"{df['_Timestamp'].max()}"
    )

    # --------------------------------------------------------
    # Convert AQI to numeric
    # --------------------------------------------------------

    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Create timestamp-based lookup
    # --------------------------------------------------------

    aqi_by_timestamp = pd.Series(
        df["AQI"].values,
        index=df["_Timestamp"]
    )

    # --------------------------------------------------------
    # Create future AQI targets using exact timestamps
    # --------------------------------------------------------

    df["AQI_future_1"] = (
        df["_Timestamp"] + pd.Timedelta(hours=1)
    ).map(aqi_by_timestamp)

    df["AQI_future_24"] = (
        df["_Timestamp"] + pd.Timedelta(hours=24)
    ).map(aqi_by_timestamp)

    # --------------------------------------------------------
    # Target availability
    # --------------------------------------------------------

    missing_future_1 = df["AQI_future_1"].isna().sum()
    missing_future_24 = df["AQI_future_24"].isna().sum()

    print("\nFuture target availability:")
    print(
        f"  AQI_future_1  : "
        f"{df['AQI_future_1'].notna().sum():,} valid, "
        f"{missing_future_1:,} missing"
    )
    print(
        f"  AQI_future_24 : "
        f"{df['AQI_future_24'].notna().sum():,} valid, "
        f"{missing_future_24:,} missing"
    )

    # --------------------------------------------------------
    # Check timestamp gaps
    # --------------------------------------------------------

    timestamp_difference = df["_Timestamp"].diff()

    gap_mask = (
        timestamp_difference.notna()
        & (
            timestamp_difference
            != pd.Timedelta(hours=1)
        )
    )

    gap_count = gap_mask.sum()

    if gap_count > 0:
        print(
            f"\nWARNING: {gap_count:,} non-hourly timestamp "
            f"intervals detected."
        )
        print(
            "Targets are matched by exact timestamps; "
            "missing future hours will remain NaN."
        )
    else:
        print("\nTimestamp frequency check: hourly")

    # --------------------------------------------------------
    # Remove temporary timestamp
    # --------------------------------------------------------

    df = df.drop(columns=["_Timestamp"])

    # --------------------------------------------------------
    # Move target columns to the end
    # --------------------------------------------------------

    future_columns = [
        "AQI_future_1",
        "AQI_future_24"
    ]

    other_columns = [
        col for col in df.columns
        if col not in future_columns
    ]

    df = df[other_columns + future_columns]

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(
        f"Final shape: {df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    return {
        "station": input_file.parent.name,
        "status": "SUCCESS",
        "rows": len(df),
        "missing_AQI_future_1": int(missing_future_1),
        "missing_AQI_future_24": int(missing_future_24),
        "output": str(output_file)
    }


# ============================================================
# 3. FIND ALL STATION FILES
# ============================================================

if not MAIN_FOLDER.exists():
    raise FileNotFoundError(
        f"Main folder not found: {MAIN_FOLDER}"
    )

input_files = [
    path for path in MAIN_FOLDER.rglob(INPUT_FILENAME)
    if path.is_file()
    and path.parent.name.lower() != "raw data"
    and OUTPUT_FILENAME not in path.name
]

print(f"\nFound {len(input_files)} station input files.")

# ============================================================
# 4. PROCESS ALL STATIONS
# ============================================================

results = []

for input_file in input_files:

    try:
        result = process_station(input_file)
        results.append(result)

    except Exception as e:
        print(f"\nERROR processing {input_file.parent.name}: {e}")

        results.append({
            "station": input_file.parent.name,
            "status": f"FAILED: {e}",
            "rows": None,
            "missing_AQI_future_1": None,
            "missing_AQI_future_24": None,
            "output": None
        })

# ============================================================
# 5. FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(results)

print("\n" + "=" * 70)
print("STEP 8 — FINAL SUMMARY")
print("=" * 70)

if not summary.empty:
    print(summary.to_string(index=False))

    print("\nSuccessful stations:", 
          (summary["status"] == "SUCCESS").sum())
    print("Failed stations:", 
          (summary["status"] != "SUCCESS").sum())

print("\nSTEP 8 COMPLETED")
print("=" * 70)
