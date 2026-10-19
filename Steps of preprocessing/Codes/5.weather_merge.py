
import pandas as pd
import glob
import os

# ============================================================
# MAIN DATASETS FOLDER
# ============================================================

main_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"

# Find all station-level feature-engineered files
main_files = glob.glob(
    os.path.join(
        main_folder,
        "**",
        "merged_feature_engineered.csv"
    ),
    recursive=True
)

# ============================================================
# PROCESS EACH STATION
# ============================================================

if not main_files:
    print("No merged_feature_engineered.csv files found.")

for main_file in main_files:

    station_folder = os.path.dirname(main_file)
    station_name = os.path.basename(station_folder)

    # Identify the city folder
    # Example:
    # Datasets/Patna/CPCB data/Station name
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

    output_file = os.path.join(
        station_folder,
        "merged_with_weather.csv"
    )

    print("\n" + "=" * 70)
    print("Processing station:", station_name)
    print("=" * 70)

    if not os.path.exists(weather_file):
        print("Weather file not found:", weather_file)
        print("Skipping this station.")
        continue

    try:
        # ====================================================
        # READ FILES
        # ====================================================

        main_df = pd.read_csv(main_file)
        weather_df = pd.read_csv(weather_file,skiprows=2)

        # ====================================================
        # PARSE TIMESTAMPS
        # ====================================================

        main_df["Timestamp"] = pd.to_datetime(
            main_df["Timestamp"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

        weather_df["time"] = pd.to_datetime(
            weather_df["time"],
            format="%Y-%m-%dT%H:%M",
            errors="coerce"
        )

        # ====================================================
        # CHECK INVALID TIMESTAMPS
        # ====================================================

        invalid_main = main_df["Timestamp"].isna().sum()
        invalid_weather = weather_df["time"].isna().sum()

        if invalid_main > 0:
            raise ValueError(
                f"Main CSV contains {invalid_main} invalid timestamps."
            )

        if invalid_weather > 0:
            raise ValueError(
                f"Weather CSV contains {invalid_weather} invalid timestamps."
            )

        # ====================================================
        # RENAME WEATHER TIMESTAMP
        # ====================================================

        weather_df.rename(
            columns={"time": "Timestamp"},
            inplace=True
        )

        # ====================================================
        # WEATHER COLUMNS TO ADD
        # ====================================================

        weather_columns = [
            "Timestamp",
            "temperature_2m (°C)",
            "relative_humidity_2m (%)",
            "dew_point_2m (°C)",
            "precipitation (mm)",
            "surface_pressure (hPa)",
            "wind_speed_10m (km/h)",
            "wind_direction_10m (°)",
            "shortwave_radiation (W/m²)",
            "cloud_cover (%)",
            
        ]

        missing_weather_columns = [
            col for col in weather_columns
            if col not in weather_df.columns
        ]

        if missing_weather_columns:
            raise ValueError(
                f"Weather columns missing: {missing_weather_columns}"
            )

        weather_df = weather_df[weather_columns]

        # ====================================================
        # CHECK DUPLICATE WEATHER TIMESTAMPS
        # ====================================================

        duplicate_weather_timestamps = (
            weather_df["Timestamp"].duplicated().sum()
        )

        if duplicate_weather_timestamps > 0:
            raise ValueError(
                f"Weather CSV contains "
                f"{duplicate_weather_timestamps} duplicate timestamps."
            )

        # ====================================================
        # JOIN BASED ON TIMESTAMP
        # ====================================================

        merged_df = pd.merge(
            main_df,
            weather_df,
            on="Timestamp",
            how="inner",
            suffixes=("_pollutant", "_weather")
        )

        # ====================================================
        # CHECK JOIN RESULTS
        # ====================================================

        print("Main CSV rows   :", len(main_df))
        print("Weather CSV rows:", len(weather_df))
        print("Joined rows     :", len(merged_df))

        if len(merged_df) < len(main_df):
            print(
                "WARNING:",
                len(main_df) - len(merged_df),
                "main rows had no matching weather timestamp."
            )
        else:
            print("All main timestamps matched weather data.")

        # ====================================================
        # DROP TIMESTAMP (ORIGINAL LOGIC)
        # ====================================================

        merged_df.drop(
            columns=["Timestamp"],
            inplace=True
        )

        # ====================================================
        # FINAL COLUMN ORDER
        # ====================================================

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

        missing_final_columns = [
            col for col in final_columns
            if col not in merged_df.columns
        ]

        if missing_final_columns:
            raise ValueError(
                f"Final columns missing: {missing_final_columns}"
            )

        merged_df = merged_df[final_columns]

        # ====================================================
        # SAVE FINAL CSV
        # ====================================================

        merged_df.to_csv(
            output_file,
            index=False
        )

        print("\nSuccessfully created:")
        print(output_file)

        print("\nFinal shape:")
        print(merged_df.shape)

        print("\nFinal columns:")
        for i, column in enumerate(merged_df.columns, start=1):
            print(f"{i}. {column}")

        print("\nFirst 5 rows:")
        print(merged_df.head())

    except Exception as e:
        print("Error processing:", station_name)
        print("Reason:", e)

print("\nAll station folders processed.")
