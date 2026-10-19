
import pandas as pd
import glob
import os
import re

# ============================================================
# BASE DATASET FOLDER
# ============================================================

base_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Datasets"

# Find all station folders containing a "Raw data" directory
raw_folders = glob.glob(
    os.path.join(base_folder, "**", "Raw data"),
    recursive=True
)

if not raw_folders:
    print("No 'Raw data' folders found. Check the base folder path.")

# ============================================================
# PROCESS EACH STATION SEPARATELY
# ============================================================

for folder in raw_folders:

    # Get every original CSV file in this station's Raw data folder
    files = glob.glob(os.path.join(folder, "*.csv"))

    # Exclude previously generated merged.csv to avoid re-merging it
    files = [
        file for file in files
        if os.path.basename(file).lower() != "merged.csv"
    ]

    if not files:
        print(f"\nNo CSV files found in: {folder}")
        continue

    # Sort by year in filename
    files.sort(
        key=lambda x: int(
            re.search(r'(\d{4})', os.path.basename(x)).group(1)
        ) if re.search(r'(\d{4})', os.path.basename(x)) else 0
    )

    print("\n" + "=" * 70)
    print(f"STATION FOLDER: {folder}")
    print("=" * 70)

    print("Files will be merged in this order:")
    for file in files:
        print(os.path.basename(file))

    # Merge 
    merged_df = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    # Save merged.csv in the specific station folder
    station_folder = os.path.dirname(folder)

    output_file = os.path.join(station_folder, "merged.csv")
    merged_df.to_csv(output_file, index=False)

    print(f"\nMerged file saved as: {output_file}")
    print(f"Total rows: {len(merged_df)}")

print("\nAll station folders processed.")
