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

main_folder = "Datasets"


# ============================================================
# CPCB COLUMNS
# ============================================================

cpcb_columns = [
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
    "NH3 (µg/m³)"
]


# ============================================================
# WEATHER COLUMNS
# ============================================================

weather_columns = [
    "time",
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
# PROCESS EACH CITY
# ============================================================

for city in selected_cities:

    print("\n" + "=" * 70)
    print("PROCESSING CITY:", city)
    print("=" * 70)

    cpcb_folder = os.path.join(
        main_folder,
        city,
        "CPCB data"
    )

    weather_folder = os.path.join(
        main_folder,
        city,
        "weather data"
    )

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    if not os.path.exists(cpcb_folder):
        print("CPCB folder not found!")
        continue

    if not os.path.exists(weather_folder):
        print("Weather folder not found!")
        continue


    # ========================================================
    # FIND CPCB TIME-FEATURE FILES
    # ========================================================

    cpcb_files = glob.glob(
        os.path.join(
            cpcb_folder,
            "**",
            "*_time_features.csv"
        ),
        recursive=True
    )


    # ========================================================
    # FIND UPDATED WEATHER FILES
    # ========================================================

    weather_files = glob.glob(
        os.path.join(
            weather_folder,
            "*_updated.csv"
        )
    )

    print("\nCPCB files found:", len(cpcb_files))
    print("Weather files found:", len(weather_files))


    # ========================================================
    # PROCESS EACH CPCB STATION
    # ========================================================

    for cpcb_file in cpcb_files:

        cpcb_filename = os.path.basename(cpcb_file)

        print("\n" + "-" * 70)
        print("CPCB file:", cpcb_filename)


        # ====================================================
        # READ CPCB
        # ====================================================

        cpcb = pd.read_csv(cpcb_file)


        # ====================================================
        # CHECK CPCB COLUMNS
        # ====================================================

        missing_cpcb = [
            col for col in cpcb_columns
            if col not in cpcb.columns
        ]

        if missing_cpcb:

            print("Missing CPCB columns:")
            print(missing_cpcb)

            continue


        # ====================================================
        # CREATE TEMPORARY DATE KEY
        # ====================================================

        cpcb["merge_date"] = pd.to_datetime(
            {
                "year": cpcb["Year"],
                "month": cpcb["Month"],
                "day": cpcb["Day"]
            },
            errors="coerce"
        )

        # Temporary time key
        cpcb["merge_time"] = pd.to_datetime(
            cpcb["Time"].astype(str),
            format="mixed",
            errors="coerce"
        ).dt.strftime("%H:%M:%S")


        # ====================================================
        # CHECK CPCB DATE/TIME
        # ====================================================

        invalid_cpcb = (
            cpcb["merge_date"].isna()
            | cpcb["merge_time"].isna()
        ).sum()

        if invalid_cpcb > 0:

            print(
                "WARNING:",
                invalid_cpcb,
                "CPCB rows have invalid date/time"
            )

            continue


        # ====================================================
        # FIND STATION'S WEATHER FILE
        # ====================================================

        station_folder = os.path.basename(
            os.path.dirname(cpcb_file)
        )

        print("CPCB station:", station_folder)


        # ----------------------------------------------------
        # Normalize names for matching
        # ----------------------------------------------------

        def normalize_name(name):

            name = name.lower()

            for char in [
                " ",
                "_",
                "-",
                ",",
                ".",
                "(",
                ")"
            ]:
                name = name.replace(char, "")

            for word in [
                "cpcb",
                "cppcb",
                "tnpcb",
                "mpcb",
                "mppcb",
                "wbpcb",
                "appcb",
                "updated"
            ]:
                name = name.replace(word, "")

            return name


        station_key = normalize_name(
            station_folder
        )


        # ----------------------------------------------------
        # Find matching weather file
        # ----------------------------------------------------

        matching_weather = []

        for weather_file in weather_files:

            weather_filename = os.path.splitext(
                os.path.basename(weather_file)
            )[0]

            weather_key = normalize_name(
                weather_filename
            )

            if (
                station_key in weather_key
                or weather_key in station_key
            ):
                matching_weather.append(
                    weather_file
                )


        # ====================================================
        # CHECK WEATHER MATCH
        # ====================================================

        if len(matching_weather) == 0:

            print(
                "WARNING: No matching weather file found."
            )

            continue


        if len(matching_weather) > 1:

            print(
                "WARNING: Multiple weather files found:"
            )

            for f in matching_weather:
                print(
                    "   ",
                    os.path.basename(f)
                )

            print("Skipping this station.")

            continue


        weather_file = matching_weather[0]

        print(
            "Weather file:",
            os.path.basename(weather_file)
        )


        # ====================================================
        # READ WEATHER DATA
        # ====================================================

        weather = pd.read_csv(
            weather_file
        )


        # ====================================================
        # CHECK WEATHER COLUMNS
        # ====================================================

        missing_weather = [
            col for col in weather_columns
            if col not in weather.columns
        ]

        if missing_weather:

            print("Missing weather columns:")
            print(missing_weather)

            continue


        # ====================================================
        # CREATE TEMPORARY WEATHER DATE/TIME KEYS
        # ====================================================

        weather_datetime = pd.to_datetime(
            weather["time"],
            format="mixed",
            errors="coerce"
        )


        weather["merge_date"] = (
            weather_datetime.dt.normalize()
        )


        weather["merge_time"] = (
            weather_datetime.dt.strftime(
                "%H:%M:%S"
            )
        )


        # ====================================================
        # CHECK WEATHER TIMESTAMPS
        # ====================================================

        invalid_weather = (
            weather["merge_date"].isna()
            | weather["merge_time"].isna()
        ).sum()

        if invalid_weather > 0:

            print(
                "WARNING:",
                invalid_weather,
                "invalid weather date/time values"
            )

            continue


        # ====================================================
        # KEEP ONLY REQUIRED COLUMNS
        # ====================================================

        cpcb_merge = cpcb[
            cpcb_columns
            + ["merge_date", "merge_time"]
        ]

        weather_merge = weather[
            weather_columns
            + ["merge_date", "merge_time"]
        ]


        # ====================================================
        # MERGE USING DATE + TIME
        # ====================================================

        merged = pd.merge(
            cpcb_merge,
            weather_merge,
            on=[
                "merge_date",
                "merge_time"
            ],
            how="inner"
        )


        # ====================================================
        # DISPLAY MERGE INFORMATION
        # ====================================================

        print("\nCPCB records   :", len(cpcb))
        print("Weather records:", len(weather))
        print("Matched records:", len(merged))


        # ====================================================
        # REMOVE TEMPORARY MERGE COLUMNS
        # ====================================================

        merged.drop(
            columns=[
                "merge_date",
                "merge_time",
                "time"
            ],
            inplace=True
        )


        # ====================================================
        # ARRANGE FINAL COLUMNS
        # ====================================================

        final_columns = [

            # CPCB time features
            "Day",
            "Month",
            "Year",
            "Time",
            "Day_of_Week",
            "Season",

            # CPCB pollutants
            "PM2.5 (µg/m³)",
            "PM10 (µg/m³)",
            "NO2 (µg/m³)",
            "SO2 (µg/m³)",
            "CO (mg/m³)",
            "Ozone (µg/m³)",
            "NH3 (µg/m³)",

            # Weather
            "temperature_2m (°C)",
            "relative_humidity_2m (%)",
            "dew_point_2m (°C)",
            "precipitation (mm)",
            "surface_pressure (hPa)",
            "wind_speed_10m (km/h)",
            "wind_direction_10m (°)",
            "shortwave_radiation (W/m²)",
            "cloud_cover (%)",

            # Coordinates
            "latitude",
            "longitude"
        ]


        merged = merged[
            final_columns
        ]


        # ====================================================
        # SAVE
        # ====================================================

        output_file = os.path.join(
            os.path.dirname(cpcb_file),
            os.path.splitext(cpcb_filename)[0]
            + "_weather_merged.csv"
        )


        merged.to_csv(
            output_file,
            index=False
        )


        print(
            "Saved:",
            output_file
        )


print("\n\nAll selected cities processed successfully!")