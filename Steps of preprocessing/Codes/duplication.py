import pandas as pd
import glob
import os

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\AQI-Prediction"

IMPUTED_DIR = os.path.join(
    PROJECT_ROOT,
    "Imputed_Data"
)

CITIES = [
    "Bhopal",
    "Mumbai",
    "Chennai",
    "Kolkata",
    "Visakhapatnam"
]


# ============================================================
# FUNCTION TO CHECK ONE DATASET
# ============================================================

def check_dataset(file_path):

    df = pd.read_csv(file_path)

    filename = os.path.basename(file_path)

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    print(f"Total rows              : {len(df):,}")

    # --------------------------------------------------------
    # CREATE TIMESTAMP
    # --------------------------------------------------------

    required_time_columns = [
        "Year",
        "Month",
        "Day",
        "Time"
    ]

    if all(col in df.columns for col in required_time_columns):

        df["Timestamp_Check"] = pd.to_datetime(
            df["Year"].astype(str) + "-" +
            df["Month"].astype(str).str.zfill(2) + "-" +
            df["Day"].astype(str).str.zfill(2) + " " +
            df["Time"].astype(str),
            errors="coerce"
        )

    else:
        print("\nWARNING: Time columns are missing.")
        return


    # --------------------------------------------------------
    # INVALID TIMESTAMPS
    # --------------------------------------------------------

    invalid = df["Timestamp_Check"].isna().sum()

    print(f"Invalid timestamps       : {invalid:,}")


    # --------------------------------------------------------
    # EXACT DUPLICATE ROWS
    # --------------------------------------------------------

    exact_duplicates = (
        df.drop(columns=["Timestamp_Check"])
        .duplicated()
        .sum()
    )

    print(f"Exact duplicate rows     : {exact_duplicates:,}")


    # --------------------------------------------------------
    # DUPLICATE TIMESTAMPS
    # --------------------------------------------------------

    timestamp_counts = (
        df["Timestamp_Check"]
        .value_counts()
    )

    duplicate_timestamps = timestamp_counts[
        timestamp_counts > 1
    ]

    duplicate_rows = duplicate_timestamps.sum()

    unique_timestamps = (
        df["Timestamp_Check"]
        .dropna()
        .nunique()
    )

    print(f"Unique timestamps        : {unique_timestamps:,}")
    print(
        f"Duplicate timestamps     : "
        f"{len(duplicate_timestamps):,}"
    )
    print(
        f"Rows in duplicate times  : "
        f"{duplicate_rows:,}"
    )


    # --------------------------------------------------------
    # DUPLICATE TIMESTAMPS WITH DIFFERENT VALUES
    # --------------------------------------------------------

    pollutants = [
        "PM2.5 (µg/m³)",
        "PM10 (µg/m³)",
        "NO2 (µg/m³)",
        "SO2 (µg/m³)",
        "CO (mg/m³)",
        "Ozone (µg/m³)",
        "NH3 (µg/m³)"
    ]

    available_pollutants = [
        col for col in pollutants
        if col in df.columns
    ]

    different_value_count = 0

    if available_pollutants:

        for timestamp, group in df.groupby(
            "Timestamp_Check"
        ):

            if len(group) > 1:

                different = (
                    group[available_pollutants]
                    .nunique(dropna=False)
                    .gt(1)
                    .any()
                )

                if different:
                    different_value_count += 1

    print(
        f"Duplicate times with    "
        f"different values       : "
        f"{different_value_count:,}"
    )


    # --------------------------------------------------------
    # HOURLY CONTINUITY
    # --------------------------------------------------------

    unique_time_values = (
        df["Timestamp_Check"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    differences = unique_time_values.diff()

    non_hourly = differences[
        differences.notna() &
        (differences != pd.Timedelta(hours=1))
    ]

    print(
        f"Non-1-hour differences   : "
        f"{len(non_hourly):,}"
    )


    # --------------------------------------------------------
    # MISSING HOURLY TIMESTAMPS
    # --------------------------------------------------------

    if len(unique_time_values) > 0:

        expected = pd.date_range(
            start=unique_time_values.iloc[0],
            end=unique_time_values.iloc[-1],
            freq="h"
        )

        actual = set(unique_time_values)

        missing_hours = [
            timestamp
            for timestamp in expected
            if timestamp not in actual
        ]

    else:
        missing_hours = []

    print(
        f"Missing hourly times     : "
        f"{len(missing_hours):,}"
    )


    # --------------------------------------------------------
    # DATE COVERAGE
    # --------------------------------------------------------

    if len(unique_time_values) > 0:

        print(
            f"First timestamp          : "
            f"{unique_time_values.iloc[0]}"
        )

        print(
            f"Last timestamp           : "
            f"{unique_time_values.iloc[-1]}"
        )


    # --------------------------------------------------------
    # MISSING POLLUTANT VALUES
    # --------------------------------------------------------

    pollutant_missing = 0

    if available_pollutants:

        pollutant_missing = (
            df[available_pollutants]
            .isna()
            .sum()
            .sum()
        )

    print(
        f"Missing pollutant values : "
        f"{pollutant_missing:,}"
    )


    # --------------------------------------------------------
    # NEGATIVE POLLUTANT VALUES
    # --------------------------------------------------------

    negative_values = 0

    for col in available_pollutants:

        numeric_col = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        negative_values += (
            numeric_col < 0
        ).sum()

    print(
        f"Negative pollutant values: "
        f"{negative_values:,}"
    )


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if (
        len(duplicate_timestamps) == 0
        and len(non_hourly) == 0
        and len(missing_hours) == 0
        and pollutant_missing == 0
        and negative_values == 0
    ):

        print("\nSTATUS: PASS")

    else:

        print("\nSTATUS: CHECK REQUIRED")


# ============================================================
# FIND ALL IMPUTED DATASETS
# ============================================================

print("\n")
print("#" * 80)
print("CHECKING ALL IMPUTED DATASETS")
print("#" * 80)

total_files = 0

for city in CITIES:

    city_path = os.path.join(
        IMPUTED_DIR,
        city
    )

    files = glob.glob(
        os.path.join(
            city_path,
            "*_imputed.csv"
        )
    )

    print("\n" + "#" * 80)
    print(f"CITY: {city}")
    print("#" * 80)

    if not files:

        print("No imputed datasets found.")

        continue

    for file_path in sorted(files):

        total_files += 1

        check_dataset(file_path)


# ============================================================
# END
# ============================================================

print("\n" + "#" * 80)
print("CHECK COMPLETED")
print("#" * 80)

print(
    f"Total imputed datasets checked: {total_files}"
)