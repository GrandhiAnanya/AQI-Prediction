import pandas as pd
import glob
import os

# ============================================================
# SELECT CITIES TO ANALYZE
# ============================================================

selected_cities = [
    "Bhopal",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Visakhapatnam"
]

main_folder = "Datasets"


# ============================================================
# EXPECTED COLUMNS
# ============================================================

expected_columns = [
    "Day",
    "Month",
    "Year",
    "Time",
    "Day_of_Week",
    "Season",

    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)",

    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "precipitation (mm)",
    "surface_pressure (hPa)",
    "wind_speed_10m (km/h)",
    "wind_direction_10m (°)",
    "shortwave_radiation (W/m²)",
    "cloud_cover (%)",

    "latitude",
    "longitude"
]


# ============================================================
# POLLUTANT COLUMNS
# ============================================================

pollutant_columns = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)"
]


# ============================================================
# WEATHER COLUMNS
# ============================================================

weather_numeric_columns = [
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "precipitation (mm)",
    "surface_pressure (hPa)",
    "wind_speed_10m (km/h)",
    "wind_direction_10m (°)",
    "shortwave_radiation (W/m²)",
    "cloud_cover (%)"
]


# ============================================================
# FUNCTION: CONSECUTIVE MISSING VALUES
# ============================================================

def longest_missing_gap(series):

    missing = series.isna()

    groups = (
        missing
        .ne(missing.shift())
        .cumsum()
    )

    gap_lengths = (
        missing
        .groupby(groups)
        .sum()
    )

    if len(gap_lengths) == 0:
        return 0

    return int(gap_lengths.max())


# ============================================================
# PROCESS EACH CITY
# ============================================================

for city in selected_cities:

    print("\n")
    print("=" * 80)
    print("CITY:", city)
    print("=" * 80)

    cpcb_folder = os.path.join(
        main_folder,
        city,
        "CPCB data"
    )

    if not os.path.exists(cpcb_folder):

        print("CPCB folder not found!")
        continue


    # ========================================================
    # FIND FINAL MERGED FILES
    # ========================================================

    files = glob.glob(
        os.path.join(
            cpcb_folder,
            "**",
            "*_weather_merged.csv"
        ),
        recursive=True
    )

    print("Merged station files found:", len(files))


    # ========================================================
    # ANALYZE EACH STATION
    # ========================================================

    for file in files:

        filename = os.path.basename(file)

        print("\n")
        print("-" * 80)
        print("STATION:", filename)
        print("-" * 80)


        # ====================================================
        # 1. READ DATA
        # ====================================================

        df = pd.read_csv(file)


        # ====================================================
        # 2. BASIC INFORMATION
        # ====================================================

        print("\n--- BASIC INFORMATION ---")

        print("Rows    :", df.shape[0])
        print("Columns :", df.shape[1])


        # ====================================================
        # 3. CHECK COLUMNS
        # ====================================================

        print("\n--- COLUMN CHECK ---")

        missing_columns = [
            col
            for col in expected_columns
            if col not in df.columns
        ]

        extra_columns = [
            col
            for col in df.columns
            if col not in expected_columns
        ]

        if missing_columns:

            print("Missing expected columns:")
            for col in missing_columns:
                print("  ", col)

        else:
            print("All expected columns are present.")

        if extra_columns:

            print("\nExtra columns:")
            for col in extra_columns:
                print("  ", col)


        # ====================================================
        # 4. MISSING VALUE ANALYSIS
        # ====================================================

        print("\n--- MISSING VALUE ANALYSIS ---")

        missing_count = df.isna().sum()

        missing_percent = (
            df.isna().mean() * 100
        )

        missing_table = pd.DataFrame({
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_percent.round(2)
        })

        missing_table = missing_table[
            missing_table["Missing_Count"] > 0
        ]

        if len(missing_table) == 0:

            print("No missing values!")

        else:

            print(
                missing_table
                .sort_values(
                    "Missing_Percentage",
                    ascending=False
                )
                .to_string()
            )


        # ====================================================
        # 5. CONSECUTIVE MISSING GAPS
        # ====================================================

        print("\n--- LONGEST CONSECUTIVE MISSING GAP ---")

        for col in expected_columns:

            if col not in df.columns:
                continue

            gap = longest_missing_gap(
                df[col]
            )

            if gap > 0:

                print(
                    f"{col}: {gap} consecutive rows"
                )


        # ====================================================
        # 6. DUPLICATE RECORDS
        # ====================================================

        print("\n--- DUPLICATE CHECK ---")

        duplicates = df.duplicated().sum()

        print(
            "Duplicate complete rows:",
            duplicates
        )


        # ====================================================
        # 7. DATE RANGE
        # ====================================================

        print("\n--- DATE COVERAGE ---")

        if all(
            col in df.columns
            for col in ["Day", "Month", "Year"]
        ):

            dates = pd.to_datetime(
                {
                    "year": df["Year"],
                    "month": df["Month"],
                    "day": df["Day"]
                },
                errors="coerce"
            )

            print(
                "Start date:",
                dates.min()
            )

            print(
                "End date:",
                dates.max()
            )


        # ====================================================
        # 8. TIME COVERAGE
        # ====================================================

        print("\n--- TIME INFORMATION ---")

        if "Time" in df.columns:

            print(
                "Unique time values:",
                df["Time"].nunique()
            )

            print(
                "First few times:",
                df["Time"].dropna().unique()[:10]
            )


        # ====================================================
        # 9. CHECK FOR DUPLICATE DATE + TIME
        # ====================================================

        print("\n--- DATE + TIME DUPLICATE CHECK ---")

        if all(
            col in df.columns
            for col in [
                "Day",
                "Month",
                "Year",
                "Time"
            ]
        ):

            date_time_duplicates = df.duplicated(
                subset=[
                    "Day",
                    "Month",
                    "Year",
                    "Time"
                ]
            ).sum()

            print(
                "Duplicate date/time records:",
                date_time_duplicates
            )


        # ====================================================
        # 10. NEGATIVE VALUE CHECK
        # ====================================================

        print("\n--- NEGATIVE VALUE CHECK ---")

        for col in (
            pollutant_columns
            + weather_numeric_columns
        ):

            if col not in df.columns:
                continue

            negative_count = (
                df[col] < 0
            ).sum()

            if negative_count > 0:

                print(
                    f"{col}: {negative_count} negative values"
                )


        # ====================================================
        # 11. BASIC STATISTICS
        # ====================================================

        print("\n--- BASIC STATISTICS ---")

        numeric_columns = (
            pollutant_columns
            + weather_numeric_columns
        )

        available_numeric = [
            col
            for col in numeric_columns
            if col in df.columns
        ]

        statistics = df[
            available_numeric
        ].describe().T

        print(
            statistics[
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "50%",
                    "max"
                ]
            ].round(2).to_string()
        )


        # ====================================================
        # 12. SEASON DISTRIBUTION
        # ====================================================

        print("\n--- SEASON DISTRIBUTION ---")

        if "Season" in df.columns:

            print(
                df["Season"]
                .value_counts(dropna=False)
                .to_string()
            )


        # ====================================================
        # 13. DAY OF WEEK DISTRIBUTION
        # ====================================================

        print("\n--- DAY OF WEEK DISTRIBUTION ---")

        if "Day_of_Week" in df.columns:

            print(
                df["Day_of_Week"]
                .value_counts(dropna=False)
                .to_string()
            )


        # ====================================================
        # 14. COORDINATE CHECK
        # ====================================================

        print("\n--- COORDINATE CHECK ---")

        if "latitude" in df.columns:

            print(
                "Latitude:",
                df["latitude"].dropna().unique()
            )

        if "longitude" in df.columns:

            print(
                "Longitude:",
                df["longitude"].dropna().unique()
            )


        print("\nAnalysis completed for:", filename)


print("\n")
print("=" * 80)
print("ALL SELECTED STATIONS ANALYZED")
print("=" * 80)