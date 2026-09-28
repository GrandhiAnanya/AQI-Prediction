import pandas as pd
import os

# ============================================================
# 1. MAIN CSV FILE
# ============================================================

folder = "Datasets/Visakhapatnam/CPCB data/GVM Corporation, Visakhapatnam - APPCB"

input_file = os.path.join(
    folder,
    "GVM_Corporation_Visakhapatnam_merged_pollutants_time_features_weather_merged.csv"
)

output_file = os.path.join(
    folder,
    "GVM_Corporation_Visakhapatnam_merged_pollutants_time_features_weather_merged_with_elevation.csv"
)

# Hardcode station elevation here (in meters)
ELEVATION = 78


# ============================================================
# 2. READ CSV
# ============================================================

df = pd.read_csv(input_file)


# ============================================================
# 3. ADD ELEVATION COLUMN
# ============================================================

# Insert elevation immediately before latitude
latitude_index = df.columns.get_loc("latitude")

df.insert(
    latitude_index,
    "elevation",
    ELEVATION
)


# ============================================================
# 4. SAVE IN SAME CPCB FOLDER
# ============================================================

df.to_csv(output_file, index=False)

print(f"Elevation ({ELEVATION} m) added successfully.")
print(f"Saved as: {output_file}")