import pandas as pd

# Load the dataset
file_path = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data\Chennai\Perungudi_Chennai_merged_pollutants_time_features_weather_merged_with_elevation_with_AQI.csv"

df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Check exact duplicate rows
# --------------------------------------------------

exact_duplicates = df.duplicated().sum()

print("Exact duplicate rows:", exact_duplicates)


# --------------------------------------------------
# 2. Check duplicates based on date/time
# --------------------------------------------------

# If your dataset has separate Day, Month, Year and Time columns
df['datetime'] = pd.to_datetime(
    df['Year'].astype(str) + '-' +
    df['Month'].astype(str) + '-' +
    df['Day'].astype(str) + ' ' +
    df['Time'].astype(str),
    errors='coerce'
)

print("Invalid datetime values:", df['datetime'].isna().sum())

# Find duplicate timestamps
duplicate_datetime = df[df.duplicated(
    subset=['datetime'],
    keep=False
)].sort_values('datetime')

print("Rows having duplicate timestamps:", len(duplicate_datetime))

print("\nDuplicate timestamps:")
print(duplicate_datetime[['datetime']].value_counts().head(20))


# --------------------------------------------------
# 3. Remove duplicate timestamps
# --------------------------------------------------

# Keep the first occurrence of each timestamp
df = df.drop_duplicates(
    subset=['datetime'],
    keep='first'
).copy()


# --------------------------------------------------
# 4. Remove helper column
# --------------------------------------------------

df.drop(columns=['datetime'], inplace=True)


# --------------------------------------------------
# 5. Reset index
# --------------------------------------------------

df.reset_index(drop=True, inplace=True)


# --------------------------------------------------
# 6. Check result
# --------------------------------------------------

print("\nAfter removing duplicates:")
print("New shape:", df.shape)

print("Remaining exact duplicates:", df.duplicated().sum())


# --------------------------------------------------
# 7. Save cleaned file
# --------------------------------------------------

output_file = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data\Chennai\Perungudi_Chennai_merged_pollutants_time_features_weather_merged_with_elevation_with_AQI.csv"

df.to_csv(output_file, index=False)

print("\nSaved as:", output_file)