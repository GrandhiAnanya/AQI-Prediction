
import pandas as pd
import glob
import os
import csv

# ============================================================
# MAIN DATASETS FOLDER
# ============================================================

main_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"

# Find all station-level merged_with_weather.csv files
input_files = glob.glob(
    os.path.join(
        main_folder,
        "**",
        "merged_with_weather.csv"
    ),
    recursive=True
)

# ============================================================
# PROCESS EACH STATION
# ============================================================

if not input_files:
    print("No merged_with_weather.csv files found.")

for input_file in input_files:

    station_folder = os.path.dirname(input_file)
    station_name = os.path.basename(station_folder)

    # Identify city folder
    city_folder = os.path.dirname(
        os.path.dirname(station_folder)
    )

    weather_folder = os.path.join(
        city_folder,
        "weather data"
    )

    # Weather filename matches the station folder name
    weather_file = os.path.join(
        weather_folder,
        station_name + ".csv"
    )

    output_file = os.path.join(
        station_folder,
        "merged_with_weather_with_elevation.csv"
    )

    print("\n" + "=" * 70)
    print("Processing station:", station_name)
    print("=" * 70)

    if not os.path.exists(weather_file):
        print("Weather file not found:", weather_file)
        print("Skipping this station.")
        continue

    try:
        # ----------------------------------------------------
        # 1. READ ELEVATION FROM WEATHER METADATA
        # ----------------------------------------------------

        with open(
            weather_file,
            "r",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.reader(f)

            metadata_header = next(reader)
            metadata_values = next(reader)

        # Find the elevation field by its metadata header
        normalized_headers = [
            header.strip().lower()
            for header in metadata_header
        ]

        if "elevation" not in normalized_headers:
            raise ValueError(
                "Elevation field not found in weather metadata. "
                f"Available fields: {metadata_header}"
            )

        elevation_index = normalized_headers.index("elevation")
        elevation = float(metadata_values[elevation_index])

        print("Elevation:", elevation, "m")

        # ----------------------------------------------------
        # 2. READ MAIN CSV
        # ----------------------------------------------------

        df = pd.read_csv(input_file)

        # ----------------------------------------------------
        # 3. INSERT ELEVATION BEFORE LATITUDE
        # ----------------------------------------------------

        if "latitude" not in df.columns:
            raise ValueError(
                "Latitude column not found in the main CSV."
            )

        # Avoid creating a duplicate elevation column
        if "elevation" in df.columns:
            df.drop(columns=["elevation"], inplace=True)

        latitude_index = df.columns.get_loc("latitude")

        df.insert(
            latitude_index,
            "elevation",
            elevation
        )

        # ----------------------------------------------------
        # 4. SAVE
        # ----------------------------------------------------

        df.to_csv(output_file, index=False)

        print("Elevation added successfully.")
        print("Saved as:", output_file)
        print("Shape:", df.shape)

    except Exception as e:
        print("Error processing:", station_name)
        print("Reason:", e)

print("\nAll station folders processed.")
