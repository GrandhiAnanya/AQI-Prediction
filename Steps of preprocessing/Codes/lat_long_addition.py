import pandas as pd
import glob
import os

# ============================================================
# SELECT STATES / CITY FOLDERS TO PROCESS
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
# PROCESS SELECTED FOLDERS
# ============================================================

for state in selected_cities:

    weather_folder = os.path.join(
        main_folder,
        state,
        "weather data"
    )

    # Check whether folder exists
    if not os.path.exists(weather_folder):
        print(f"\nWeather folder not found for: {state}")
        continue

    # Find only CSV files inside weather data
    files = glob.glob(
        os.path.join(weather_folder, "*.csv")
    )

    print(f"\n{'='*50}")
    print(f"Processing: {state}")
    print(f"Number of files found: {len(files)}")
    print(f"{'='*50}")

    # ========================================================
    # PROCESS EACH WEATHER FILE
    # ========================================================

    for file in files:

        filename = os.path.basename(file)

        # Don't process an already updated file
        if "_updated" in filename:
            continue

        print("\nProcessing:", filename)

        # ----------------------------------------------------
        # 1. Read latitude and longitude
        # ----------------------------------------------------

        info = pd.read_csv(file, nrows=1)

        latitude = info["latitude"].iloc[0]
        longitude = info["longitude"].iloc[0]

        print("Latitude :", latitude)
        print("Longitude:", longitude)

        # ----------------------------------------------------
        # 2. Read actual weather data
        # ----------------------------------------------------

        df = pd.read_csv(file, skiprows=2)

        # ----------------------------------------------------
        # 3. Add latitude and longitude
        # ----------------------------------------------------

        df["latitude"] = latitude
        df["longitude"] = longitude

        # ----------------------------------------------------
        # 4. Check cloud_cover column
        # ----------------------------------------------------

        if "cloud_cover (%)" not in df.columns:

            print("WARNING: cloud_cover (%) not found")
            print("Columns available:")
            print(df.columns.tolist())

            continue

        # ----------------------------------------------------
        # 5. Move coordinates after cloud_cover (%)
        # ----------------------------------------------------

        cols = list(df.columns)

        cols.remove("latitude")
        cols.remove("longitude")

        cloud_index = cols.index("cloud_cover (%)")

        cols.insert(cloud_index + 1, "latitude")
        cols.insert(cloud_index + 2, "longitude")

        df = df[cols]

        # ----------------------------------------------------
        # 6. Save updated file
        # ----------------------------------------------------

        output_file = os.path.join(
            weather_folder,
            os.path.splitext(filename)[0] + "_updated.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print("Saved:", output_file)


print("\n\nAll selected states processed successfully!")