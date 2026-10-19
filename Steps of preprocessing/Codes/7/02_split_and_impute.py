
"""
02_split_and_impute.py

Station-wise AQI pollutant imputation.

Rules:
    1-3 hours   -> Past-only forward fill (causal)
    4-72 hours  -> Contextual median fitted on observed TRAIN data
    >72 hours   -> Leave missing; unresolved rows are dropped

Splits:
    Train      = 2020-2023
    Validation = 2024
    Test       = 2025

Each station is processed independently.
Split DataFrames remain in memory and are recombined into one CSV.
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAIN_FOLDER = Path(
    r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"
)

INPUT_FILENAME = "merged_with_weather_with_elevation.csv"
GAP_ANALYSIS_FILENAME = "Long_Gap_Analysis.xlsx"

OUTPUT_FILENAME = "merged_with_weather_with_elevation_imputed.csv"
AUDIT_FILENAME = "imputation_audit.csv"

MIN_CONTEXT_COUNT = 5

POLLUTANT_COLS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)",
]

REQUIRED_COLUMNS = [
    "Day",
    "Month",
    "Year",
    "Time",
    "Day_of_Week",
    "Season",
] + POLLUTANT_COLS


# ============================================================
# 2. HELPERS
# ============================================================

def build_timestamp(df):

    timestamp = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
        + "-"
        + df["Day"].astype(str).str.zfill(2)
        + " "
        + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )

    if timestamp.isna().any():
        bad = int(timestamp.isna().sum())
        raise ValueError(f"{bad} invalid timestamps found.")

    return timestamp


def validate_columns(df, file_path):

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {file_path.name}:\n"
            + "\n".join(f"- {col}" for col in missing)
        )


def normalize_action(value):
    return " ".join(str(value).strip().lower().split())


def check_hourly_order(df):

    diffs = df["_timestamp"].diff().dropna()

    bad = int(
        (diffs != pd.Timedelta(hours=1)).sum()
    )

    if bad:
        warnings.warn(
            f"Found {bad} non-hourly timestamp intervals. "
            "Check for missing timestamps or irregular intervals."
        )


def load_gap_analysis(gap_file):

    if not gap_file.exists():
        raise FileNotFoundError(
            f"Gap analysis workbook not found:\n{gap_file}\n\n"
            "Run Part 1 for this station first."
        )

    gaps = pd.read_excel(
        gap_file,
        sheet_name="All_Gaps",
    )

    required = [
        "Variable",
        "Gap_ID",
        "Gap_Start",
        "Gap_End",
        "Gap_Hours",
        "Gap_Category",
        "Proposed_Action",
    ]

    missing = [
        col for col in required
        if col not in gaps.columns
    ]

    if missing:
        raise ValueError(
            f"Gap analysis workbook is missing columns: {missing}"
        )

    if gaps.empty:
        return gaps

    gaps["Gap_Start"] = pd.to_datetime(
        gaps["Gap_Start"], errors="coerce"
    )

    gaps["Gap_End"] = pd.to_datetime(
        gaps["Gap_End"], errors="coerce"
    )

    if gaps[["Gap_Start", "Gap_End"]].isna().any().any():
        raise ValueError(
            "Invalid Gap_Start or Gap_End values found."
        )

    unknown_variables = sorted(
        set(gaps["Variable"].dropna()) - set(POLLUTANT_COLS)
    )

    if unknown_variables:
        raise ValueError(
            "Gap workbook pollutant names do not match the dataset: "
            f"{unknown_variables}"
        )

    return gaps


# ============================================================
# 3. MAP ROWS TO GAP CATEGORIES
# ============================================================

def identify_row_gap_map(df, gap_df):
    """
    Associate each missing pollutant value with its gap category.
    Gap identification comes from Part 1.
    """

    row_gap_maps = {
        variable: pd.Series(
            index=df.index,
            dtype=object
        )
        for variable in POLLUTANT_COLS
    }

    for _, gap in gap_df.iterrows():

        variable = gap["Variable"]

        mask = (
            (df["_timestamp"] >= gap["Gap_Start"])
            & (df["_timestamp"] <= gap["Gap_End"])
            & df[variable].isna()
        )

        info = (
            gap["Gap_Start"],
            gap["Gap_End"],
            int(gap["Gap_Hours"]),
            gap["Gap_Category"],
            gap["Proposed_Action"],
        )

        row_gap_maps[variable].loc[mask] = [info] * int(mask.sum())

    return row_gap_maps


# ============================================================
# 4. FIT CONTEXTUAL MEDIANS USING TRAIN DATA ONLY
# ============================================================

def fit_contextual_medians(train_df):
    """
    Fit median lookup tables using only observed pollutant values
    from 2020-2023.

    Hierarchy:
        Season + Hour + Day_of_Week
        -> Season + Hour
        -> Season
        -> Overall station median

    Missing pollutant values are excluded when fitting medians.
    """

    hierarchy = [
        ["Season", "_Hour", "Day_of_Week"],
        ["Season", "_Hour"],
        ["Season"],
        [],
    ]

    median_maps = {}
    count_maps = {}

    for variable in POLLUTANT_COLS:

        observed = train_df.loc[
            train_df[variable].notna()
        ].copy()

        median_maps[variable] = []
        count_maps[variable] = []

        for keys in hierarchy:

            if keys:

                grouped = (
                    observed.groupby(keys, dropna=True)[variable]
                    .agg(["median", "count"])
                )

                median_maps[variable].append(
                    grouped["median"].to_dict()
                )

                count_maps[variable].append(
                    grouped["count"].to_dict()
                )

            else:

                values = observed[variable].dropna()

                median = (
                    values.median()
                    if len(values)
                    else np.nan
                )

                count = int(values.notna().sum())

                median_maps[variable].append(
                    {(): median}
                )

                count_maps[variable].append(
                    {(): count}
                )

    return median_maps, count_maps


# ============================================================
# 5. APPLY CONTEXTUAL MEDIAN
# ============================================================

def apply_contextual_median(
    df,
    mask,
    variable,
    median_maps,
    count_maps,
):
    """
    Fill selected missing rows using training-only median maps.
    Returns filled count and counts by hierarchy level.
    """

    target = df.index[
        mask & df[variable].isna()
    ]

    if len(target) == 0:
        return 0, {}

    hierarchy = [
        ["Season", "_Hour", "Day_of_Week"],
        ["Season", "_Hour"],
        ["Season"],
        [],
    ]

    filled = 0
    level_counts = {}

    for level, keys in enumerate(hierarchy, start=1):

        remaining = target[
            df.loc[target, variable].isna()
        ]

        if len(remaining) == 0:
            break

        if keys:

            for idx in remaining:

                row = df.loc[idx]

                # Do not use a contextual group with missing keys.
                if any(pd.isna(row[key]) for key in keys):
                    continue

                key = tuple(row[k] for k in keys)

                count = count_maps[level - 1].get(key, 0)
                median = median_maps[level - 1].get(key, np.nan)

                if (
                    count >= MIN_CONTEXT_COUNT
                    and pd.notna(median)
                ):
                    df.at[idx, variable] = median
                    filled += 1

                    level_counts[level] = (
                        level_counts.get(level, 0) + 1
                    )

        else:

            count = count_maps[level - 1].get((), 0)
            median = median_maps[level - 1].get((), np.nan)

            if (
                count >= MIN_CONTEXT_COUNT
                and pd.notna(median)
            ):
                remaining = target[
                    df.loc[target, variable].isna()
                ]

                df.loc[remaining, variable] = median

                n = len(remaining)
                filled += n

                level_counts[level] = (
                    level_counts.get(level, 0) + n
                )

    return filled, level_counts


# ============================================================
# 6. IMPUTE SHORT GAPS USING PAST OBSERVATIONS ONLY
# ============================================================

def impute_short_gaps(part, variable, gap_series):
    """
    Fill only detected 1-3 hour gaps using forward fill.

    This uses the most recent observed value and does not use
    observations after the missing run.

    Filling is restricted to detected short gaps and to a maximum
    of three consecutive missing rows.
    """

    short_mask = gap_series.map(
        lambda x: (
            isinstance(x, tuple)
            and x[3] == "1-3 h"
        )
    )

    if not short_mask.any():
        return 0

    original_missing = part[variable].isna()

    # Forward fill uses past values only.
    past_values = part[variable].ffill(limit=3)

    fill_mask = (
        short_mask
        & original_missing
        & past_values.notna()
    )

    count = int(fill_mask.sum())

    if count:
        part.loc[fill_mask, variable] = past_values.loc[fill_mask]

    return count


# ============================================================
# 7. PROCESS ONE SPLIT
# ============================================================

def process_split(
    part,
    split_name,
    row_gap_maps,
    median_maps,
    count_maps,
):
    """Impute one in-memory split and return audit records."""

    audit = []

    for variable in POLLUTANT_COLS:

        gap_series = row_gap_maps[variable].loc[part.index]

        before = int(part[variable].isna().sum())

        # ----------------------------------------------------
        # 1. SHORT GAPS: 1-3 HOURS
        # ----------------------------------------------------

        filled_short = impute_short_gaps(
            part,
            variable,
            gap_series,
        )

        if filled_short:

            audit.append({
                "Split": split_name,
                "Variable": variable,
                "Method": "1-3 h -> Past-only forward fill",
                "Rows_Filled": filled_short,
                "Rows_Dropped": 0,
                "Rows_Left_Missing": 0,
            })

        # ----------------------------------------------------
        # 2. MEDIUM GAPS: 4-72 HOURS
        # ----------------------------------------------------

        medium_mask = gap_series.map(
            lambda x: (
                isinstance(x, tuple)
                and x[3] == "4-72 h"
            )
        )

        if medium_mask.any():

            filled_count, level_counts = apply_contextual_median(
                part,
                medium_mask,
                variable,
                median_maps[variable],
                count_maps[variable],
            )

            for level, count in level_counts.items():

                audit.append({
                    "Split": split_name,
                    "Variable": variable,
                    "Method": (
                        "4-72 h -> Contextual median "
                        f"(Level {level})"
                    ),
                    "Rows_Filled": count,
                    "Rows_Dropped": 0,
                    "Rows_Left_Missing": 0,
                })

        # ----------------------------------------------------
        # 3. LONG GAPS: MORE THAN 72 HOURS
        # ----------------------------------------------------

        long_mask = gap_series.map(
            lambda x: (
                isinstance(x, tuple)
                and x[3] == ">72 h"
            )
        )

        remaining_long = int(
            (long_mask & part[variable].isna()).sum()
        )

        if remaining_long:

            audit.append({
                "Split": split_name,
                "Variable": variable,
                "Method": ">72 h -> Leave missing",
                "Rows_Filled": 0,
                "Rows_Dropped": 0,
                "Rows_Left_Missing": remaining_long,
            })

        # ----------------------------------------------------
        # FINAL POLLUTANT SUMMARY FOR THIS SPLIT
        # ----------------------------------------------------

        after = int(part[variable].isna().sum())

        audit.append({
            "Split": split_name,
            "Variable": variable,
            "Method": "FINAL",
            "Rows_Filled": before - after,
            "Rows_Dropped": 0,
            "Rows_Left_Missing": after,
        })

    return part, audit


# ============================================================
# 8. PROCESS ONE STATION
# ============================================================

def process_station(input_file):

    station_folder = input_file.parent

    gap_file = station_folder / GAP_ANALYSIS_FILENAME
    output_file = station_folder / OUTPUT_FILENAME
    audit_file = station_folder / AUDIT_FILENAME

    print("\n" + "=" * 75)
    print(f"Station: {station_folder.name}")
    print(f"Input:   {input_file}")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD AND VALIDATE DATA
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print(f"Original shape: {df.shape}")

    validate_columns(df, input_file)

    df["_timestamp"] = build_timestamp(df)

    df = (
        df.sort_values("_timestamp")
        .reset_index(drop=True)
    )

    duplicate_count = int(
        df["_timestamp"].duplicated().sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps. "
            "Resolve duplicates before imputation."
        )

    check_hourly_order(df)

    df["_Hour"] = df["_timestamp"].dt.hour

    # --------------------------------------------------------
    # LOAD THIS STATION'S GAP REPORT
    # --------------------------------------------------------

    gap_df = load_gap_analysis(gap_file)

    print(f"Gap-analysis records: {len(gap_df):,}")

    # Check gap-analysis rules.
    allowed_actions = {
        "1-3 h": "time interpolation",
        "4-72 h": "contextual median",
        ">72 h": "leave / drop",
    }

    for _, row in gap_df.iterrows():

        category = row["Gap_Category"]

        expected_action = allowed_actions.get(category)

        actual_action = normalize_action(
            row["Proposed_Action"]
        )

        if expected_action is None:
            raise ValueError(
                f"Unknown gap category: {category}"
            )

        if actual_action != expected_action:
            raise ValueError(
                f"Gap analysis rule mismatch for {category}: "
                f"expected '{expected_action}', "
                f"got '{row['Proposed_Action']}'."
            )

    row_gap_maps = identify_row_gap_map(df, gap_df)

    # --------------------------------------------------------
    # SPLIT IN MEMORY
    # --------------------------------------------------------

    train = df[df["Year"].between(2020, 2023)].copy()
    validation = df[df["Year"].eq(2024)].copy()
    test = df[df["Year"].eq(2025)].copy()

    print(f"Train      (2020-2023): {len(train):,} rows")
    print(f"Validation (2024)     : {len(validation):,} rows")
    print(f"Test       (2025)     : {len(test):,} rows")

    included_rows = len(train) + len(validation) + len(test)

    excluded_year_rows = len(df) - included_rows

    if excluded_year_rows:
        warnings.warn(
            f"{excluded_year_rows:,} rows are outside 2020-2025 "
            "and will not appear in the final output."
        )

    if train.empty:
        raise ValueError(
            "No training rows from 2020-2023 were found. "
            "Cannot fit contextual medians."
        )

    # --------------------------------------------------------
    # FIT MEDIANS ON OBSERVED TRAIN VALUES ONLY
    # --------------------------------------------------------

    median_maps, count_maps = fit_contextual_medians(train)

    split_data = {
        "Train": train,
        "Validation": validation,
        "Test": test,
    }

    all_audit = []

    # --------------------------------------------------------
    # IMPUTE EACH SPLIT
    # --------------------------------------------------------

    for split_name, part in split_data.items():

        part, audit = process_split(
            part,
            split_name,
            row_gap_maps,
            median_maps,
            count_maps,
        )

        split_data[split_name] = part
        all_audit.extend(audit)

    # --------------------------------------------------------
    # RECOMBINE SPLITS INTO ONE DATASET
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            split_data["Train"],
            split_data["Validation"],
            split_data["Test"],
        ],
        axis=0,
    ).sort_values("_timestamp")

    before_drop = len(final_df)

    # Drop rows that still have missing pollutant values.
    # Weather missingness alone does not trigger row deletion.
    residual_missing = final_df[POLLUTANT_COLS].isna().any(axis=1)

    rows_dropped = int(residual_missing.sum())

    final_df = final_df.loc[~residual_missing].copy()

    all_audit.append({
        "Split": "ALL",
        "Variable": "ALL POLLUTANTS",
        "Method": "Drop rows with unresolved pollutant missingness",
        "Rows_Filled": 0,
        "Rows_Dropped": rows_dropped,
        "Rows_Left_Missing": int(
            final_df[POLLUTANT_COLS].isna().sum().sum()
        ),
    })

    # Remove temporary columns.
    final_df.drop(
        columns=["_timestamp", "_Hour"],
        errors="ignore",
        inplace=True,
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS INSIDE THE STATION FOLDER
    # --------------------------------------------------------

    final_df.to_csv(output_file, index=False)

    audit_df = pd.DataFrame(all_audit)
    audit_df.to_csv(audit_file, index=False)

    # --------------------------------------------------------
    # PRINT STATION SUMMARY
    # --------------------------------------------------------

    print("\nIMPUTATION SUMMARY")
    print("-" * 50)
    print(f"Rows before final drop : {before_drop:,}")
    print(f"Rows dropped           : {rows_dropped:,}")
    print(f"Final rows             : {len(final_df):,}")

    remaining_nans = int(
        final_df[POLLUTANT_COLS].isna().sum().sum()
    )

    print(f"Remaining pollutant NaNs: {remaining_nans:,}")
    print(f"Imputed dataset saved  : {output_file}")
    print(f"Audit report saved     : {audit_file}")


# ============================================================
# 9. MAIN: PROCESS ALL STATIONS
# ============================================================

def main():

    print("=" * 75)
    print("STATION-WISE AQI IMPUTATION")
    print("=" * 75)

    if not MAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{MAIN_FOLDER}"
        )

    input_files = sorted(
        MAIN_FOLDER.rglob(INPUT_FILENAME)
    )

    # Exclude files that may be inside Raw data folders.
    input_files = [
        file for file in input_files
        if file.parent.name.lower() != "raw data"
    ]

    if not input_files:
        print(
            f"No files named '{INPUT_FILENAME}' found under:\n"
            f"{MAIN_FOLDER}"
        )
        return

    print(f"Station files found: {len(input_files)}")

    successful = 0
    failed = 0
    failures = []

    for input_file in input_files:

        try:
            process_station(input_file)
            successful += 1

        except Exception as error:

            failed += 1
            failures.append((input_file, str(error)))

            print("\nERROR processing station:")
            print(input_file)
            print(f"Reason: {error}")

            # Continue processing other stations.
            continue

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STATION-WISE IMPUTATION COMPLETED")
    print("=" * 75)

    print(f"Files found           : {len(input_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed                : {failed}")

    if failures:
        print("\nStations requiring attention:")

        for file_path, reason in failures:
            print(f"\n- {file_path}")
            print(f"  Reason: {reason}")


if __name__ == "__main__":
    main()
