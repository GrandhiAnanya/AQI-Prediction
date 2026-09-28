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

    # Find all merged CPCB CSV files inside city folders
    files = glob.glob(
        os.path.join(cpcb_folder, "**", "*merged.csv"),
        recursive=True
    )

    print(f"\n{'='*60}")
    print(f"Processing city: {city}")
    print(f"Number of merged files found: {len(files)}")
    print(f"{'='*60}")

    # ========================================================
    # PROCESS EACH MERGED DATASET
    # ========================================================

    for file in files:

        filename = os.path.basename(file)

        print("\nProcessing:", filename)

        # Read dataset
        df = pd.read_csv(file)

        # ====================================================
        # COLUMNS TO KEEP
        # ====================================================

        columns_to_keep = [
            "Timestamp",
            "PM2.5 (µg/m³)",
            "PM10 (µg/m³)",
            "NO2 (µg/m³)",
            "SO2 (µg/m³)",
            "CO (mg/m³)",
            "Ozone (µg/m³)",
            "NH3 (µg/m³)"
        ]

        # ====================================================
        # CHECK FOR MISSING COLUMNS
        # ====================================================

        missing_columns = [
            col for col in columns_to_keep
            if col not in df.columns
        ]

        if missing_columns:

            print("WARNING: These columns were not found:")
            print(missing_columns)

            print("\nAvailable columns:")
            print(df.columns.tolist())

            continue

        # ====================================================
        # KEEP ONLY REQUIRED POLLUTANTS
        # ====================================================

        df = df[columns_to_keep]

        # ====================================================
        # SAVE CLEANED DATASET
        # ====================================================

        output_file = os.path.join(
            os.path.dirname(file),
            os.path.splitext(filename)[0] + "_pollutants.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print("Saved:", output_file)
        print("New shape:", df.shape)

print("\n\nAll selected cities processed successfully!")