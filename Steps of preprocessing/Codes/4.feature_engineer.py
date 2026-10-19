
import pandas as pd
import glob
import os

# ============================================================
# MAIN DATASETS FOLDER
# ============================================================

main_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"

# Find all station-level input files
input_files = glob.glob(
    os.path.join(
        main_folder,
        "**",
        "merged_selected_columns.csv"
    ),
    recursive=True
)

# ============================================================
# SEASON FUNCTION
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
# PROCESS EACH STATION
# ============================================================

if not input_files:
    print("No merged_selected_columns.csv files found.")

for input_file in input_files:

    station_folder = os.path.dirname(input_file)
    station_name = os.path.basename(station_folder)

    output_file = os.path.join(
        station_folder,
        "merged_feature_engineered.csv"
    )

    print("\n" + "=" * 60)
    print("Processing station:", station_name)
    print("=" * 60)

    try:
        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        df = pd.read_csv(input_file)

        print("Original columns:")
        print(df.columns.tolist())

        # ----------------------------------------------------
        # PARSE TIMESTAMP
        # ----------------------------------------------------

        df["Timestamp_parsed"] = pd.to_datetime(
            df["Timestamp"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

        # ----------------------------------------------------
        # CHECK INVALID TIMESTAMPS
        # ----------------------------------------------------

        invalid_timestamps = df["Timestamp_parsed"].isna().sum()

        if invalid_timestamps > 0:
            print(
                f"\nERROR: {invalid_timestamps} invalid timestamp(s) found."
            )

            print(
                df.loc[
                    df["Timestamp_parsed"].isna(),
                    "Timestamp"
                ].head(10)
            )

            print("Skipping this station.")
            continue

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        df["Day"] = df["Timestamp_parsed"].dt.day
        df["Month"] = df["Timestamp_parsed"].dt.month
        df["Year"] = df["Timestamp_parsed"].dt.year
        df["Time"] = df["Timestamp_parsed"].dt.strftime("%H:%M:%S")
        df["Day_of_Week"] = df["Timestamp_parsed"].dt.day_name()

        df["Season"] = df["Month"].apply(get_season)

        # ----------------------------------------------------
        # REMOVE TEMPORARY COLUMN
        # ----------------------------------------------------

        df.drop(columns=["Timestamp_parsed"], inplace=True)

        # ----------------------------------------------------
        # REORDER COLUMNS
        # ----------------------------------------------------

        final_columns = [
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
            "latitude",
            "longitude",
            "Timestamp"
        ]

        missing_columns = [
            col for col in final_columns
            if col not in df.columns
        ]

        if missing_columns:
            print("Missing required columns:", missing_columns)
            print("Skipping this station.")
            continue

        df = df[final_columns]

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        df.to_csv(output_file, index=False)

        print("\nFeature engineering completed successfully!")
        print("\nFinal columns:")
        for i, column in enumerate(df.columns, start=1):
            print(f"{i}. {column}")

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nOutput file:")
        print(output_file)

    except Exception as e:
        print("Error processing:", station_name)
        print("Reason:", e)

print("\nAll station folders processed.")
