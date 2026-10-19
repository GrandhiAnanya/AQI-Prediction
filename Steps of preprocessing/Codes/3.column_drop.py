
import pandas as pd
import glob
import os

# ============================================================
# MAIN DATASETS FOLDER
# ============================================================

main_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"

# Find all station-level merged_modified.csv files
input_files = glob.glob(
    os.path.join(main_folder, "**", "merged_modified.csv"),
    recursive=True
)

# ============================================================
# COLUMNS TO RETAIN
# ============================================================

columns_to_keep = [
    "Timestamp",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "NH3 (µg/m³)",
    "latitude",
    "longitude"
]

# ============================================================
# PROCESS EACH STATION
# ============================================================

if not input_files:
    print("No merged_modified.csv files found. Check the folder path.")

for input_file in input_files:

    station_folder = os.path.dirname(input_file)
    station_name = os.path.basename(station_folder)

    output_file = os.path.join(
        station_folder,
        "merged_selected_columns.csv"
    )

    print("\n" + "=" * 60)
    print("Processing station:", station_name)
    print("=" * 60)

    try:
        # Read CSV
        df = pd.read_csv(input_file)

        # Check which requested columns are missing
        missing_columns = [
            col for col in columns_to_keep
            if col not in df.columns
        ]

        if missing_columns:
            print("ERROR: The following columns were not found:")
            for col in missing_columns:
                print(" -", col)

            print("Skipping this station.")
            continue

        # Keep only required columns in the specified order
        df = df[columns_to_keep]

        # Save inside the specific station folder
        df.to_csv(output_file, index=False)

        print("Successfully created:", output_file)

        print("\nColumns retained:")
        for col in df.columns:
            print(" -", col)

        print("\nShape:", df.shape)

    except Exception as e:
        print("Error processing:", station_name)
        print("Reason:", e)

print("\nAll station folders processed.")
