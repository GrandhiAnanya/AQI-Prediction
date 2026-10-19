
import pandas as pd
import glob
import os
import csv

# ============================================================
# MAIN DATASETS FOLDER
# ============================================================

main_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"

# Find station-level merged.csv files
merged_files = glob.glob(
    os.path.join(main_folder, "**", "merged.csv"),
    recursive=True
)

# Exclude files inside Raw data folders
merged_files = [
    f for f in merged_files
    if os.path.basename(os.path.dirname(f)).lower() != "raw data"
]

# ============================================================
# PROCESS EACH STATION
# ============================================================

for input_file in merged_files:

    station_folder = os.path.dirname(input_file)
    station_name = os.path.basename(station_folder)

    # Locate the corresponding weather data folder
    city_folder = os.path.dirname(
        os.path.dirname(station_folder)
    )

    weather_folder = os.path.join(
        city_folder,
        "weather data"
    )

    weather_file = os.path.join(
        weather_folder,
        station_name + ".csv"
    )

    print("\n" + "=" * 60)
    print("Processing:", station_name)
    print("=" * 60)

    if not os.path.exists(weather_file):
        print("Weather file not found:", weather_file)
        continue

    try:
        # ----------------------------------------------------
        # 1. Read latitude and longitude from weather metadata
        # ----------------------------------------------------

        with open(weather_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)

            metadata_header = next(reader)
            metadata_values = next(reader)

            latitude = float(metadata_values[0])
            longitude = float(metadata_values[1])

        print("Latitude :", latitude)
        print("Longitude:", longitude)

        # ----------------------------------------------------
        # 2. Read merged pollutant data
        # ----------------------------------------------------

        df = pd.read_csv(input_file, encoding="utf-8-sig")

        # ----------------------------------------------------
        # 3. Append latitude and longitude
        # ----------------------------------------------------

        df["latitude"] = latitude
        df["longitude"] = longitude

        # ----------------------------------------------------
        # 4. Save inside the station folder
        # ----------------------------------------------------

        output_file = os.path.join(
            station_folder,
            "merged_modified.csv"
        )

        df.to_csv(output_file, index=False)

        print("Saved:", output_file)
        print("Total rows:", len(df))

    except Exception as e:
        print("Error processing:", station_name)
        print("Reason:", e)

print("\nAll station folders processed.")
