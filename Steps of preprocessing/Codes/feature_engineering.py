import pandas as pd
import glob
import os

# ============================================================
# SELECT CITIES TO PROCESS
# ============================================================

selected_cities = [
    "Bhopal",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Visakhapatnam"
]

# Main Datasets folder
main_folder = "Datasets"


# ============================================================
# FUNCTION TO DETERMINE SEASON
# ============================================================

def get_season(month):

    if month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    elif month in [10, 11]:
        return "Autumn"

    else:
        return "Winter"


# ============================================================
# FUNCTION TO PARSE DIFFERENT TIMESTAMP FORMATS
# ============================================================

def parse_timestamp(value):

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    # Try YYYY-MM-DD HH:MM:SS
    result = pd.to_datetime(
        value,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    if pd.notna(result):
        return result

    # Try YYYY-MM-DD HH:MM
    result = pd.to_datetime(
        value,
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )

    if pd.notna(result):
        return result

    # Try DD-MM-YYYY HH:MM:SS
    result = pd.to_datetime(
        value,
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    if pd.notna(result):
        return result

    # Try DD-MM-YYYY HH:MM
    result = pd.to_datetime(
        value,
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    if pd.notna(result):
        return result

    # Try YYYY-MM-DD (date only)
    result = pd.to_datetime(
        value,
        format="%Y-%m-%d",
        errors="coerce"
    )

    if pd.notna(result):
        return result

    # Try DD-MM-YYYY (date only)
    result = pd.to_datetime(
        value,
        format="%d-%m-%Y",
        errors="coerce"
    )

    return result


# ============================================================
# PROCESS SELECTED CITIES
# ============================================================

for city in selected_cities:

    cpcb_folder = os.path.join(
        main_folder,
        city,
        "CPCB data"
    )

    if not os.path.exists(cpcb_folder):
        print(f"\nCPCB folder not found for: {city}")
        continue

    # Find pollutant datasets
    files = glob.glob(
        os.path.join(
            cpcb_folder,
            "**",
            "*_pollutants.csv"
        ),
        recursive=True
    )

    print(f"\n{'='*60}")
    print(f"Processing city: {city}")
    print(f"Number of files found: {len(files)}")
    print(f"{'='*60}")


    # ========================================================
    # PROCESS EACH FILE
    # ========================================================

    for file in files:

        filename = os.path.basename(file)

        # Don't process an already processed file
        if "_time_features" in filename:
            continue

        print("\nProcessing:", filename)

        # ----------------------------------------------------
        # 1. READ DATASET
        # ----------------------------------------------------

        df = pd.read_csv(file)

        # ----------------------------------------------------
        # 2. PARSE TIMESTAMP
        # ----------------------------------------------------

        print("Parsing timestamps...")

        df["Timestamp_parsed"] = df["Timestamp"].apply(
            parse_timestamp
        )

        # ----------------------------------------------------
        # 3. CHECK INVALID TIMESTAMPS
        # ----------------------------------------------------

        invalid = df["Timestamp_parsed"].isna().sum()

        if invalid > 0:

            print(
                "WARNING:",
                invalid,
                "invalid timestamps found"
            )

            # Show examples
            print("Examples of invalid timestamps:")

            print(
                df.loc[
                    df["Timestamp_parsed"].isna(),
                    "Timestamp"
                ].head(10).tolist()
            )

        # ----------------------------------------------------
        # 4. EXTRACT DAY
        # ----------------------------------------------------

        df["Day"] = df["Timestamp_parsed"].dt.day

        # ----------------------------------------------------
        # 5. EXTRACT MONTH
        # ----------------------------------------------------

        df["Month"] = df["Timestamp_parsed"].dt.month

        # ----------------------------------------------------
        # 6. EXTRACT YEAR
        # ----------------------------------------------------

        df["Year"] = df["Timestamp_parsed"].dt.year

        # ----------------------------------------------------
        # 7. EXTRACT TIME
        # ----------------------------------------------------

        df["Time"] = df["Timestamp_parsed"].dt.strftime(
            "%H:%M:%S"
        )

        # ----------------------------------------------------
        # 8. EXTRACT DAY OF WEEK
        # ----------------------------------------------------

        df["Day_of_Week"] = df["Timestamp_parsed"].dt.day_name()

        # ----------------------------------------------------
        # 9. CREATE SEASON
        # ----------------------------------------------------

        df["Season"] = df["Month"].apply(
            lambda x: get_season(x)
            if pd.notna(x)
            else None
        )

        # ----------------------------------------------------
        # 10. DROP TIMESTAMP COLUMNS
        # ----------------------------------------------------

        df.drop(
            columns=[
                "Timestamp",
                "Timestamp_parsed"
            ],
            inplace=True
        )

        # ----------------------------------------------------
        # 11. ARRANGE COLUMNS
        # ----------------------------------------------------

        time_columns = [
            "Day",
            "Month",
            "Year",
            "Time",
            "Day_of_Week",
            "Season"
        ]

        other_columns = [
            col for col in df.columns
            if col not in time_columns
        ]

        df = df[
            time_columns + other_columns
        ]

        # ----------------------------------------------------
        # 12. SAVE NEW DATASET
        # ----------------------------------------------------

        output_file = os.path.join(
            os.path.dirname(file),
            os.path.splitext(filename)[0]
            + "_time_features.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print("Saved:", output_file)


print("\n\nAll selected cities processed successfully!")