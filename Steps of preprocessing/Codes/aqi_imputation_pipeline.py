"""
AQI station-wise imputation pipeline
------------------------------------

Cities:
    Bhopal, Mumbai, Chennai, Kolkata, Visakhapatnam

Split (kept ONLY in memory):
    Train      = 2020-2023
    Validation = 2024
    Test       = 2025

Imputation:
    1-3 hours      -> time interpolation
    4-72 hours     -> contextual median
    >72 hours      -> use Long_Gap_Analysis_5_Cities.xlsx:
                         Proposed_Action = Contextual median -> impute
                         Proposed_Action = Leave / Drop     -> keep missing

    After all imputation, any row still missing one or more pollutant
    values is dropped. Weather missingness does not trigger row deletion.

Contextual-median hierarchy:
    Station + Season + Hour + Day-of-Week
        -> Station + Season + Hour
        -> Station + Season
        -> Station median

Important leakage rule:
    Contextual medians are FIT ONLY on observed TRAIN (2020-2023)
    values and then applied to Train/Validation/Test.
    The three split DataFrames are never written as separate files.

Output:
    One final imputed CSV per station, plus one overall audit summary.
"""

from pathlib import Path
import os
import re
import warnings
import pandas as pd
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\prave\Desktop\college projects\AQI-Prediction")
DATASETS_DIR = Path(r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets")
LONG_GAP_FILE = Path(r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets\Long_Gap_Analysis_5_Cities.xlsx")
STATION_METADATA_FILE = Path(r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets\stations meta data.txt")

# If your workbook has "(1)" because of downloading/uploading,
# the filename normalizer below handles it automatically.
TARGET_CITIES = [
    "Bhopal",
    "Mumbai",
    "Chennai",
    "Kolkata",
    "Visakhapatnam"
]

# Minimum number of observed TRAIN values required before a
# contextual median is considered reliable enough.
MIN_CONTEXT_COUNT = 5

# Output folders. These are NOT train/validation/test files.
OUTPUT_DIR = PROJECT_ROOT / "Imputed_Data"
AUDIT_DIR = PROJECT_ROOT / "Imputation_Audit"

SAVE_IMPUTED_FILES = True
SAVE_AUDIT_SUMMARY = True

POLLUTANT_COLS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)"
]

REQUIRED_COLUMNS = [
    "Day", "Month", "Year", "Time",
    "Day_of_Week", "Season"
] + POLLUTANT_COLS


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def normalize_filename(name):
    """
    Makes filename matching robust to:
        file.csv
        file(1).csv
        file(2).csv
    """
    name = os.path.basename(str(name)).strip()
    name = re.sub(r"\(\d+\)(?=\.csv$)", "", name, flags=re.IGNORECASE)
    return name.lower()


def normalize_text(text):
    """Normalize text for safe comparisons."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_timestamp(df):
    """
    Construct timestamp from the existing Day/Month/Year/Time columns.
    The original columns are retained.
    """
    timestamp = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
        + "-"
        + df["Day"].astype(str).str.zfill(2)
        + " "
        + df["Time"].astype(str).str.strip(),
        errors="coerce"
    )

    if timestamp.isna().any():
        bad = int(timestamp.isna().sum())
        raise ValueError(f"{bad} invalid timestamps found.")

    return timestamp


def check_required_columns(df, file_path):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"\nMissing required columns in {file_path.name}:\n{missing}"
        )


def check_hourly_order(df, station_name):
    """
    Your datasets are hourly. This checks that timestamps are sorted
    and reports any unexpected gaps in the row timeline.
    """
    diffs = df["_timestamp"].diff().dropna()
    bad = (diffs != pd.Timedelta(hours=1)).sum()

    if bad > 0:
        warnings.warn(
            f"{station_name}: {bad} non-1-hour timestamp differences found. "
            "Gap detection assumes the dataset contains one row per hour."
        )


# ============================================================
# 3. READ LONG-GAP ANALYSIS
# ============================================================

def load_long_gap_analysis():
    if not LONG_GAP_FILE.exists():
        raise FileNotFoundError(
            f"Long-gap workbook not found:\n{LONG_GAP_FILE}"
        )

    long_df = pd.read_excel(
        LONG_GAP_FILE,
        sheet_name="All_Long_Gaps"
    )

    required = [
        "City", "Station", "File", "Variable",
        "Gap_Start", "Gap_End", "Gap_Hours",
        "Long_Gap_Class", "Proposed_Action"
    ]

    missing = [c for c in required if c not in long_df.columns]

    if missing:
        raise ValueError(
            f"Long-gap workbook is missing columns: {missing}"
        )

    long_df["Gap_Start"] = pd.to_datetime(
        long_df["Gap_Start"], errors="coerce"
    )
    long_df["Gap_End"] = pd.to_datetime(
        long_df["Gap_End"], errors="coerce"
    )

    long_df["File_norm"] = long_df["File"].map(normalize_filename)

    long_df = long_df[
        long_df["City"].isin(TARGET_CITIES)
    ].copy()

    return long_df


def make_long_gap_lookup(long_df, file_name):
    """
    Returns:
        {(variable, gap_start, gap_end): proposed_action}
    """
    file_norm = normalize_filename(file_name)

    station_gaps = long_df[
        long_df["File_norm"] == file_norm
    ].copy()

    lookup = {}

    for _, row in station_gaps.iterrows():
        key = (
            row["Variable"],
            row["Gap_Start"],
            row["Gap_End"]
        )

        lookup[key] = {
            "class": row["Long_Gap_Class"],
            "action": row["Proposed_Action"],
            "gap_hours": row["Gap_Hours"],
            "city": row["City"],
            "station": row["Station"]
        }

    return lookup, station_gaps


# ============================================================
# 4. FIND STATION FILES
# ============================================================

def find_station_files(long_df):
    """
    The long-gap workbook contains the authoritative station-file names
    for the five cities. We use those names to locate the station CSVs
    inside the corresponding city folders.

    This avoids accidentally processing raw files or duplicate versions.
    """

    pairs = (
        long_df[["City", "Station", "File"]]
        .drop_duplicates()
        .sort_values(["City", "Station"])
    )

    found = []

    all_csvs = list(DATASETS_DIR.rglob("*.csv"))

    normalized_paths = {}
    for p in all_csvs:
        normalized_paths.setdefault(
            normalize_filename(p.name), []
        ).append(p)

    for _, row in pairs.iterrows():

        city = row["City"]
        station = row["Station"]
        expected_file = row["File"]

        # Prefer searching inside the correct city directory.
        city_dir = DATASETS_DIR / city

        candidates = []
        if city_dir.exists():
            candidates = [
                p for p in city_dir.rglob("*.csv")
                if normalize_filename(p.name)
                == normalize_filename(expected_file)
            ]

        # Fallback: search entire Datasets folder.
        if not candidates:
            candidates = normalized_paths.get(
                normalize_filename(expected_file), []
            )

        if not candidates:
            warnings.warn(
                f"Could not find CSV for:\n"
                f"  City   : {city}\n"
                f"  Station: {station}\n"
                f"  File   : {expected_file}"
            )
            continue

        if len(candidates) > 1:
            warnings.warn(
                f"Multiple matches found for {expected_file}. "
                f"Using the first match:\n{candidates}"
            )

        found.append({
            "city": city,
            "station": station,
            "expected_file": expected_file,
            "path": candidates[0]
        })

    return found


# ============================================================
# 5. IDENTIFY EVERY MISSING RUN BEFORE SPLITTING
# ============================================================

def identify_missing_runs(df):
    """
    Identify complete missing runs on the ORIGINAL station dataset.

    This is done BEFORE train/validation/test splitting so that a
    long gap crossing a year boundary cannot accidentally look like
    a short gap after splitting.
    """

    records = []

    for variable in POLLUTANT_COLS:

        missing = df[variable].isna().to_numpy()

        if not missing.any():
            continue

        starts = np.flatnonzero(
            missing & ~np.r_[False, missing[:-1]]
        )

        ends = np.flatnonzero(
            missing & ~np.r_[missing[1:], False]
        )

        for gap_id, (start_idx, end_idx) in enumerate(
            zip(starts, ends), start=1
        ):

            gap_start = df.loc[start_idx, "_timestamp"]
            gap_end = df.loc[end_idx, "_timestamp"]

            gap_hours = int(
                round(
                    (gap_end - gap_start).total_seconds() / 3600
                )
            ) + 1

            if gap_hours <= 3:
                category = "1-3 h"
            elif gap_hours <= 72:
                category = "4-72 h"
            else:
                category = ">72 h"

            records.append({
                "Variable": variable,
                "Gap_ID": gap_id,
                "Gap_Start": gap_start,
                "Gap_End": gap_end,
                "Gap_Hours": gap_hours,
                "Category": category
            })

    return pd.DataFrame(records)


# ============================================================
# 6. FIT CONTEXTUAL MEDIANS FROM TRAIN ONLY
# ============================================================

def fit_contextual_medians(train_df):
    """
    Fit contextual median lookup tables using ONLY observed TRAIN values.

    Hierarchy:
        Level 1: Season + Hour + Day_of_Week
        Level 2: Season + Hour
        Level 3: Season
        Level 4: Station median

    Because each station is processed separately, 'Station' is implicit.
    """

    median_maps = {}
    count_maps = {}

    hierarchy = [
        ["Season", "_Hour", "Day_of_Week"],
        ["Season", "_Hour"],
        ["Season"],
        []
    ]

    for variable in POLLUTANT_COLS:

        observed = train_df[
            train_df[variable].notna()
        ].copy()

        median_maps[variable] = []
        count_maps[variable] = []

        for keys in hierarchy:

            if keys:

                grouped = (
                    observed
                    .groupby(keys)[variable]
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

                median = values.median() if len(values) else np.nan

                median_maps[variable].append({
                    (): median
                })

                count_maps[variable].append({
                    (): len(values)
                })

    return median_maps, count_maps


def apply_contextual_median(
    df,
    mask,
    variable,
    median_maps,
    count_maps,
    min_count=5
):
    """
    Fill missing rows using the four-level hierarchy.

    Returns:
        filled_count
        level_counts
    """

    missing_index = df.index[
        mask & df[variable].isna()
    ]

    if len(missing_index) == 0:
        return 0, {}

    hierarchy = [
        ["Season", "_Hour", "Day_of_Week"],
        ["Season", "_Hour"],
        ["Season"],
        []
    ]

    filled = 0
    level_counts = {}

    remaining = missing_index

    for level, keys in enumerate(hierarchy, start=1):

        remaining = remaining[
            df.loc[remaining, variable].isna()
        ]

        if len(remaining) == 0:
            break

        if keys:

            for idx in remaining:

                row = df.loc[idx]

                key = tuple(
                    row[k] for k in keys
                )

                count = count_maps[level - 1].get(
                    key, 0
                )

                median = median_maps[level - 1].get(
                    key, np.nan
                )

                if count >= min_count and pd.notna(median):

                    df.at[idx, variable] = median

                    filled += 1
                    level_counts[level] = (
                        level_counts.get(level, 0) + 1
                    )

        else:

            count = count_maps[level - 1].get(
                (), 0
            )

            median = median_maps[level - 1].get(
                (), np.nan
            )

            if count >= min_count and pd.notna(median):

                df.loc[remaining, variable] = median

                n = len(remaining)

                filled += n
                level_counts[level] = n

    return filled, level_counts


# ============================================================
# 7. BUILD ROW -> GAP INFORMATION
# ============================================================

def build_row_gap_map(df, gap_info):
    """
    For every pollutant and every row, store the complete missing-gap
    information identified BEFORE splitting.

    This prevents a >24h gap from being misclassified as a short gap
    merely because it crosses 2023/2024 or 2024/2025.
    """

    row_gap_maps = {
        variable: pd.Series(
            index=df.index,
            dtype=object
        )
        for variable in POLLUTANT_COLS
    }

    for _, gap in gap_info.iterrows():

        mask = (
            (df["_timestamp"] >= gap["Gap_Start"])
            & (df["_timestamp"] <= gap["Gap_End"])
        )

        info = (
            gap["Gap_Start"],
            gap["Gap_End"],
            gap["Gap_Hours"],
            gap["Category"]
        )

        row_gap_maps[
            gap["Variable"]
        ].loc[mask] = [info] * int(mask.sum())

    return row_gap_maps


# ============================================================
# 8. IMPUTE ONE STATION
# ============================================================

def impute_station(
    df,
    city,
    station,
    file_name,
    long_gap_lookup
):
    """
    Main station-wise imputation function.
    """

    df = df.copy()

    check_required_columns(df, Path(file_name))

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["_timestamp"] = build_timestamp(df)

    df = (
        df.sort_values("_timestamp")
        .reset_index(drop=True)
    )

    check_hourly_order(df, station)

    df["_Hour"] = df["_timestamp"].dt.hour

    # --------------------------------------------------------
    # Identify complete missing runs BEFORE splitting
    # --------------------------------------------------------

    gap_info = identify_missing_runs(df)

    row_gap_maps = build_row_gap_map(
        df,
        gap_info
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST
    # Kept only in memory.
    # --------------------------------------------------------

    train_mask = df["Year"].between(2020, 2023)
    validation_mask = df["Year"].eq(2024)
    test_mask = df["Year"].eq(2025)

    train = df.loc[train_mask].copy()
    validation = df.loc[validation_mask].copy()
    test = df.loc[test_mask].copy()

    # --------------------------------------------------------
    # FIT contextual medians ONLY from observed TRAIN data
    # --------------------------------------------------------

    median_maps, count_maps = fit_contextual_medians(
        train
    )

    audit = []

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    split_data = {
        "Train": train,
        "Validation": validation,
        "Test": test
    }

    for split_name, part in split_data.items():

        for variable in POLLUTANT_COLS:

            gap_series = row_gap_maps[
                variable
            ].loc[part.index]

            # ----------------------------------------------
            # 1. SHORT GAPS: 1-3 hours
            # ----------------------------------------------

            short_mask = gap_series.map(
                lambda x:
                    isinstance(x, tuple)
                    and x[3] == "1-3 h"
            )

            before = int(
                part[variable].isna().sum()
            )

            if short_mask.any():

                # Interpolation is performed only inside this
                # Train/Validation/Test split.
                interpolated = (
                    part[variable]
                    .interpolate(
                        method="linear",
                        limit=3,
                        limit_area="inside"
                    )
                )

                fill_mask = (
                    short_mask
                    & part[variable].isna()
                    & interpolated.notna()
                )

                part.loc[
                    fill_mask,
                    variable
                ] = interpolated.loc[fill_mask]

                n = int(fill_mask.sum())

                if n:
                    audit.append({
                        "City": city,
                        "Station": station,
                        "File": file_name,
                        "Split": split_name,
                        "Variable": variable,
                        "Method": "Interpolation",
                        "Rows_Filled": n
                    })

            # ----------------------------------------------
            # 2. 4-72 HOURS: CONTEXTUAL MEDIAN
            # ----------------------------------------------

            medium_mask = gap_series.map(
                lambda x:
                    isinstance(x, tuple)
                    and x[3] == "4-72 h"
            )

            if medium_mask.any():

                n, levels = apply_contextual_median(
                    part,
                    medium_mask,
                    variable,
                    median_maps[variable],
                    count_maps[variable],
                    MIN_CONTEXT_COUNT
                )

                if n:
                    for level, count in levels.items():
                        audit.append({
                            "City": city,
                            "Station": station,
                            "File": file_name,
                            "Split": split_name,
                            "Variable": variable,
                            "Method":
                                f"Contextual median - Level {level}",
                            "Rows_Filled": count
                        })

            # ----------------------------------------------
            # 3. >72 HOURS: USE LONG-GAP ANALYSIS FILE
            # ----------------------------------------------

            long_mask = gap_series.map(
                lambda x:
                    isinstance(x, tuple)
                    and x[3] == ">72 h"
            )

            if long_mask.any():

                fill_long_mask = pd.Series(
                    False,
                    index=part.index
                )

                leave_long_mask = pd.Series(
                    False,
                    index=part.index
                )

                unclassified_mask = pd.Series(
                    False,
                    index=part.index
                )

                for idx in part.index[long_mask]:

                    info = gap_series.loc[idx]

                    gap_start = info[0]
                    gap_end = info[1]

                    key = (
                        variable,
                        gap_start,
                        gap_end
                    )

                    decision = long_gap_lookup.get(key)

                    if decision is None:

                        # Do NOT guess.
                        # If the long-gap analysis file does not
                        # classify the gap, leave it missing.
                        unclassified_mask.loc[idx] = True

                    elif normalize_text(decision["action"]) == "contextual median":

                        fill_long_mask.loc[idx] = True

                    else:

                        # Very long / Extreme / unreliable
                        # -> Leave / Drop
                        leave_long_mask.loc[idx] = True

                # Moderate & usable -> contextual median
                if fill_long_mask.any():

                    n, levels = apply_contextual_median(
                        part,
                        fill_long_mask,
                        variable,
                        median_maps[variable],
                        count_maps[variable],
                        MIN_CONTEXT_COUNT
                    )

                    if n:
                        for level, count in levels.items():
                            audit.append({
                                "City": city,
                                "Station": station,
                                "File": file_name,
                                "Split": split_name,
                                "Variable": variable,
                                "Method":
                                    f"Long gap -> Contextual median "
                                    f"(Level {level})",
                                "Rows_Filled": count
                            })

                # Leave/drop
                leave_count = int(
                    (
                        leave_long_mask
                        & part[variable].isna()
                    ).sum()
                )

                if leave_count:
                    audit.append({
                        "City": city,
                        "Station": station,
                        "File": file_name,
                        "Split": split_name,
                        "Variable": variable,
                        "Method": "Long gap -> Leave/Drop",
                        "Rows_Filled": 0,
                        "Rows_Left_Missing": leave_count
                    })

                # Unclassified
                unclassified_count = int(
                    (
                        unclassified_mask
                        & part[variable].isna()
                    ).sum()
                )

                if unclassified_count:
                    audit.append({
                        "City": city,
                        "Station": station,
                        "File": file_name,
                        "Split": split_name,
                        "Variable": variable,
                        "Method":
                            "Long gap -> Unclassified; left missing",
                        "Rows_Filled": 0,
                        "Rows_Left_Missing":
                            unclassified_count
                    })

            after = int(
                part[variable].isna().sum()
            )

            # Overall variable summary
            audit.append({
                "City": city,
                "Station": station,
                "File": file_name,
                "Split": split_name,
                "Variable": variable,
                "Method": "FINAL",
                "Rows_Filled":
                    before - after,
                "Rows_Left_Missing": after
            })

        split_data[split_name] = part

    # --------------------------------------------------------
    # Put the three splits back into ONE final station dataset
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We are NOT creating Train.csv / Validation.csv / Test.csv.
    #
    # We recombine the already-imputed rows so that the final
    # station dataset remains one chronological dataset for later
    # ML processing.
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            split_data["Train"],
            split_data["Validation"],
            split_data["Test"]
        ],
        axis=0
    ).sort_index()

    # Restore original row order and remove temporary columns
    final_df = (
        final_df
        .sort_index()
        .drop(columns=["_timestamp", "_Hour"], errors="ignore")
    )

    # --------------------------------------------------------
    # DROP ROWS THAT STILL HAVE MISSING POLLUTANT VALUES
    # --------------------------------------------------------
    # These are the values that were deliberately marked
    # "Leave / Drop" in the long-gap analysis, or values for
    # which a contextual median could not be obtained.
    # Weather columns are NOT used for this drop decision.
    # --------------------------------------------------------

    rows_before_drop = len(final_df)

    remaining_missing_mask = final_df[POLLUTANT_COLS].isna().any(axis=1)
    rows_dropped = int(remaining_missing_mask.sum())

    if rows_dropped:
        final_df = final_df.loc[~remaining_missing_mask].copy()

    # Add station-level row-drop audit
    audit.append({
        "City": city,
        "Station": station,
        "File": file_name,
        "Split": "ALL",
        "Variable": "ALL POLLUTANTS",
        "Method": "Final residual missing -> Drop row",
        "Rows_Filled": 0,
        "Rows_Dropped": rows_dropped,
        "Rows_Left_Missing": int(
            final_df[POLLUTANT_COLS].isna().sum().sum()
        )
    })

    print(
        f"Rows before final drop: {rows_before_drop:,}"
    )
    print(
        f"Rows dropped due to remaining pollutant NaNs: {rows_dropped:,}"
    )
    print(
        f"Final shape: {final_df.shape}"
    )

    audit_df = pd.DataFrame(audit)

    return final_df, audit_df


# ============================================================
# 9. PROCESS ALL STATIONS
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AQI STATION-WISE IMPUTATION")
    print("=" * 70)

    print("\nTarget cities:")
    for city in TARGET_CITIES:
        print("  -", city)

    # --------------------------------------------------------
    # Load long-gap analysis
    # --------------------------------------------------------

    long_df = load_long_gap_analysis()

    print(
        f"\nLong-gap records loaded: {len(long_df):,}"
    )

    # --------------------------------------------------------
    # Find station CSV files
    # --------------------------------------------------------

    station_files = find_station_files(
        long_df
    )

    print(
        f"Station files found: {len(station_files)}"
    )

    if not station_files:
        raise RuntimeError(
            "No station CSV files were found."
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    if SAVE_IMPUTED_FILES:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    if SAVE_AUDIT_SUMMARY:
        AUDIT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    all_audits = []

    # --------------------------------------------------------
    # Process each station independently
    # --------------------------------------------------------

    for item in station_files:

        city = item["city"]
        station = item["station"]
        csv_path = item["path"]

        print("\n" + "-" * 70)
        print(f"City    : {city}")
        print(f"Station : {station}")
        print(f"File    : {csv_path.name}")

        df = pd.read_csv(csv_path)

        print(
            f"Original shape: {df.shape}"
        )

        # Long-gap decisions for this exact station file
        long_lookup, station_long_gaps = (
            make_long_gap_lookup(
                long_df,
                csv_path.name
            )
        )

        print(
            f"Long-gap records for station: "
            f"{len(station_long_gaps)}"
        )

        # ----------------------------------------------------
        # Impute station
        # ----------------------------------------------------

        imputed_df, audit_df = impute_station(
            df=df,
            city=city,
            station=station,
            file_name=csv_path.name,
            long_gap_lookup=long_lookup
        )

        # ----------------------------------------------------
        # Report missing values
        # ----------------------------------------------------

        before_missing = int(
            df[POLLUTANT_COLS]
            .isna()
            .sum()
            .sum()
        )

        after_missing = int(
            imputed_df[POLLUTANT_COLS]
            .isna()
            .sum()
            .sum()
        )

        print(
            f"Missing pollutant values BEFORE: "
            f"{before_missing:,}"
        )

        print(
            f"Missing pollutant values AFTER : "
            f"{after_missing:,}"
        )

        # ----------------------------------------------------
        # Save ONE final imputed station file
        # ----------------------------------------------------

        if SAVE_IMPUTED_FILES:

            city_output_dir = (
                OUTPUT_DIR / city
            )

            city_output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_name = (
                csv_path.stem
                + "_imputed.csv"
            )

            output_path = (
                city_output_dir
                / output_name
            )

            imputed_df.to_csv(
                output_path,
                index=False
            )

            print(
                f"Saved: {output_path}"
            )

        # ----------------------------------------------------
        # Collect audit
        # ----------------------------------------------------

        if not audit_df.empty:
            all_audits.append(audit_df)

    # --------------------------------------------------------
    # Overall audit summary
    # --------------------------------------------------------

    if all_audits and SAVE_AUDIT_SUMMARY:

        audit_all = pd.concat(
            all_audits,
            ignore_index=True
        )

        audit_path = (
            AUDIT_DIR
            / "overall_imputation_audit.csv"
        )

        audit_all.to_csv(
            audit_path,
            index=False
        )

        print(
            "\nOverall audit saved to:"
        )
        print(audit_path)

    print("\n" + "=" * 70)
    print("IMPUTATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
