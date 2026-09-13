import pandas as pd
import glob
import os

folder = "Datasets/Visakhapatnam/CPCB data/GVM Corporation, Visakhapatnam - APPCB"

# Find all CSV files in the station folder
files = glob.glob(os.path.join(folder, "*.csv"))

print("Number of files:", len(files))

data = []

for file in files:
    print("Reading:", os.path.basename(file))
    
    df = pd.read_csv(file)
    data.append(df)

# Combine all years
station_data = pd.concat(data, ignore_index=True)

# Save merged file
output_file = os.path.join(
    folder,
    "GVM_Corporation_Visakhapatnam_merged.csv"
)

station_data.to_csv(output_file, index=False)

print("\nMerged successfully!")
print("Shape:", station_data.shape)
print("Saved to:", output_file)