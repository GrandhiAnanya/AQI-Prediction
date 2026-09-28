import pandas as pd
import glob
import os

# ============================================================
# 1. CITIES TO ANALYZE
# ============================================================

TARGET_CITIES = [
    "Bhopal",
    "Mumbai",
    "Chennai",
    "Kolkata",
    "Visakhapatnam"
]

BASE_FOLDER = "Datasets"


# ============================================================
# 2. FIND FINAL MERGED FILES
# ============================================================

all_files = glob.glob(
    os.path.join(
        BASE_FOLDER,
        "**",
        "*_with_elevation.csv"
    ),
    recursive=True
)

# Keep only files belonging to the five selected cities
files = []

for file in all_files:

    relative_path = os.path.relpath(file, BASE_FOLDER)
    city = relative_path.split(os.sep)[0]

    if city in TARGET_CITIES:
        files.append(file)


print("=" * 70)
print("GAP ANALYSIS")
print("=" * 70)

print("\nCities being analyzed:")
for city in TARGET_CITIES:
    print(" -", city)

print(f"\nTotal station files found: {len(files)}")


# ============================================================
# 3. COLUMNS NOT USED FOR GAP ANALYSIS
# ============================================================

ignore_columns = [
    "Day",
    "Month",
    "Year",
    "Time",
    "Day_of_Week",
    "Season",
    "latitude",
    "longitude",
    "elevation"
]


# ============================================================
# 4. STORE RESULTS
# ============================================================

summary_results = []
gap_results = []


# ============================================================
# 5. PROCESS EACH STATION
# ============================================================

for file in files:

    print("\n" + "-" * 70)
    print("Processing:")
    print(file)

    try:
        df = pd.read_csv(file)

    except Exception as e:
        print("ERROR:", e)
        continue


    # --------------------------------------------------------
    # Identify city and station
    # --------------------------------------------------------

    relative_path = os.path.relpath(file, BASE_FOLDER)

    path_parts = relative_path.split(os.sep)

    city = path_parts[0]

    station = os.path.basename(
        os.path.dirname(file)
    )

    filename = os.path.basename(file)

    print("City    :", city)
    print("Station :", station)
    print("Rows    :", len(df))


    # ========================================================
    # 6. CREATE TEMPORARY DATETIME
    # ========================================================

    datetime_string = (
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
        + "-"
        + df["Day"].astype(str).str.zfill(2)
        + " "
        + df["Time"].astype(str)
    )

    df["datetime_temp"] = pd.to_datetime(
        datetime_string,
        errors="coerce"
    )

    invalid_dates = df["datetime_temp"].isna().sum()

    if invalid_dates > 0:
        print(
            f"WARNING: {invalid_dates} invalid datetime values."
        )

    df = df.sort_values(
        "datetime_temp"
    ).reset_index(drop=True)


    # ========================================================
    # 7. VARIABLES TO ANALYZE
    # ========================================================

    analysis_columns = [
        col
        for col in df.columns
        if col not in ignore_columns
        and col != "datetime_temp"
    ]


    # ========================================================
    # 8. ANALYZE EACH VARIABLE
    # ========================================================

    for column in analysis_columns:

        missing_mask = df[column].isna()

        total_rows = len(df)

        missing_count = missing_mask.sum()

        missing_percentage = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0
        )


        # ----------------------------------------------------
        # No missing values
        # ----------------------------------------------------

        if missing_count == 0:

            summary_results.append({

                "City": city,
                "Station": station,
                "File": filename,

                "Variable": column,

                "Total_Observations": total_rows,

                "Missing_Observations": 0,

                "Missing_Percentage": 0,

                "Total_Gaps": 0,

                "Gaps_1_to_3h": 0,
                "Gaps_4_to_72h": 0,
                "Gaps_Above_72h": 0,

                "Longest_Gap_Hours": 0
            })

            continue


        # ====================================================
        # 9. FIND CONTINUOUS MISSING GROUPS
        # ====================================================

        groups = (
            missing_mask
            .ne(missing_mask.shift())
            .cumsum()
        )

        missing_groups = groups[missing_mask]

        variable_gaps = []


        # ====================================================
        # 10. ANALYZE EACH GAP
        # ====================================================

        for group_id, group_data in df[
            missing_mask
        ].groupby(missing_groups):

            start_index = group_data.index.min()

            end_index = group_data.index.max()


            start_time = df.loc[
                start_index,
                "datetime_temp"
            ]

            end_time = df.loc[
                end_index,
                "datetime_temp"
            ]


            # Number of missing records
            gap_records = len(group_data)


            # Calculate gap duration
            if (
                pd.notna(start_time)
                and pd.notna(end_time)
            ):

                gap_hours = (
                    (
                        end_time - start_time
                    ).total_seconds()
                    / 3600
                ) + 1

            else:

                gap_hours = gap_records


            # ------------------------------------------------
            # Determine season
            # ------------------------------------------------

            if pd.notna(start_time):

                month = start_time.month

                if month in [3, 4, 5]:
                    season = "Summer"

                elif month in [6, 7, 8, 9]:
                    season = "Monsoon"

                elif month in [10, 11]:
                    season = "Autumn"

                elif month in [12, 1, 2]:
                    season = "Winter"

                else:
                    season = "Unknown"

            else:

                season = "Unknown"


            # ------------------------------------------------
            # Gap category
            # ------------------------------------------------

            if gap_hours <= 3:

                gap_category = "Short (1-3 hours) - Interpolation"

            elif gap_hours <= 72:

                gap_category = "Moderate (4-72 hours) - Contextual Median"

            else:

                gap_category = "Long (>72 hours) - Leave/Drop"


            # ------------------------------------------------
            # Day of week
            # ------------------------------------------------

            if pd.notna(start_time):

                day_of_week = start_time.day_name()

            else:

                day_of_week = "Unknown"


            # =================================================
            # STORE GAP DETAILS
            # =================================================

            gap_results.append({

                "City": city,
                "Station": station,
                "File": filename,

                "Variable": column,

                "Gap_Start": start_time,
                "Gap_End": end_time,

                "Gap_Hours": gap_hours,

                "Missing_Observations": gap_records,

                "Season": season,

                "Day_of_Week": day_of_week,

                "Gap_Category": gap_category
            })


            variable_gaps.append(gap_hours)


        # ====================================================
        # 11. GAP SUMMARY
        # ====================================================

        total_gaps = len(variable_gaps)

        gaps_1_to_3 = sum(
            1 <= x <= 3
            for x in variable_gaps
        )

        gaps_4_to_72 = sum(
            4 <= x <= 72
            for x in variable_gaps
        )

        gaps_above_72 = sum(
            x > 72
            for x in variable_gaps
        )

        longest_gap = (
            max(variable_gaps)
            if variable_gaps
            else 0
        )


        summary_results.append({

            "City": city,
            "Station": station,
            "File": filename,

            "Variable": column,

            "Total_Observations": total_rows,

            "Missing_Observations": missing_count,

            "Missing_Percentage": round(
                missing_percentage,
                2
            ),

            "Total_Gaps": total_gaps,

            "Gaps_1_to_3h": gaps_1_to_3,

            "Gaps_4_to_72h": gaps_4_to_72,
            "Gaps_Above_72h": gaps_above_72,

            "Longest_Gap_Hours": round(
                longest_gap,
                2
            )
        })


# ============================================================
# 12. CREATE DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)

gap_df = pd.DataFrame(
    gap_results
)


# ============================================================
# 13. SAVE OUTPUT
# ============================================================

summary_output = os.path.join(
    BASE_FOLDER,
    "gap_analysis_summary_5_cities.csv"
)

details_output = os.path.join(
    BASE_FOLDER,
    "gap_analysis_details_5_cities.csv"
)


summary_df.to_csv(
    summary_output,
    index=False
)

gap_df.to_csv(
    details_output,
    index=False
)


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("GAP ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nCities analyzed:",
    ", ".join(TARGET_CITIES)
)

print(
    "\nStation files analyzed:",
    len(files)
)

print(
    "\nVariable summaries:",
    len(summary_df)
)

print(
    "\nIndividual gaps identified:",
    len(gap_df)
)

print(
    "\nSummary saved at:"
)

print(summary_output)

print(
    "\nDetailed gaps saved at:"
)

print(details_output)

print("\nDone!")