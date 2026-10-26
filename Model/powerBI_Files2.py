from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION — keep aligned with powerForest.py
# ============================================================

DATASET_FILE = Path("../Datasets/Final_Dataset.csv")
OUTPUT_DIR = Path("../Outputs/PowerBI/Environmental_Analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_COLUMNS = [
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

WIND_COLUMN = "wind_speed_10m (km/h)"

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_season(month):
    """
    Match the season definitions used in the project's
    feature engineering.
    """
    if month in [3, 4, 5]:
        return "Summer"
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    if month in [10, 11]:
        return "Autumn"
    return "Winter"


def build_timestamp(data):
    """
    Build a timestamp from existing Timestamp or Year, Month,
    Day and Time columns.
    """
    if "Timestamp" in data.columns:
        return pd.to_datetime(data["Timestamp"], errors="coerce")

    required_date_cols = ["Year", "Month", "Day", "Time"]
    missing_date_cols = [
        col for col in required_date_cols
        if col not in data.columns
    ]

    if missing_date_cols:
        raise KeyError(
            "Could not build Timestamp. Missing columns: "
            + ", ".join(missing_date_cols)
        )

    date_text = (
        data["Year"].astype("Int64").astype(str)
        + "-"
        + data["Month"].astype("Int64").astype(str)
        + "-"
        + data["Day"].astype("Int64").astype(str)
        + " "
        + data["Time"].astype(str).str.strip()
    )

    return pd.to_datetime(
        date_text,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

print(f"Loading dataset: {DATASET_FILE.resolve()}")

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_FILE.resolve()}\n"
        "Update DATASET_FILE if your folder layout differs."
    )

df = pd.read_csv(DATASET_FILE, low_memory=False)

if "AQI" not in df.columns:
    raise KeyError(
        "Required column 'AQI' was not found in Final_Dataset.csv."
    )

# Build and validate timestamps.
df["Timestamp"] = build_timestamp(df)

df = df.dropna(subset=["Timestamp"]).copy()

# Ensure AQI is numeric and retain valid observations.
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
df = df.dropna(subset=["AQI"]).copy()

# Standardize station identifiers.
if "Station_ID" not in df.columns:
    df["Station_ID"] = "Unknown"

df["Station_ID"] = (
    df["Station_ID"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

# Identify station class/split where available.
if "Station_Class" in df.columns:
    df["Station_Class"] = (
        df["Station_Class"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
elif "Station_Split" in df.columns:
    df["Station_Class"] = (
        df["Station_Split"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
else:
    df["Station_Class"] = "Unknown"

# Validate weather columns. Missing columns are created as NaN
# so the script can still produce the other analyses.
for col in WEATHER_COLUMNS:
    if col not in df.columns:
        print(
            f"WARNING: Weather column missing; "
            f"creating blank column: {col}"
        )
        df[col] = np.nan

    df[col] = pd.to_numeric(df[col], errors="coerce")

# Derive calendar features from the validated timestamp.
df["Date"] = df["Timestamp"].dt.normalize()
df["Year"] = df["Timestamp"].dt.year
df["Month_Number"] = df["Timestamp"].dt.month
df["Month_Name"] = df["Timestamp"].dt.strftime("%b")
df["Season"] = df["Month_Number"].map(get_season)
df["Day_of_Week"] = df["Timestamp"].dt.day_name()
df["Weekday_Order"] = df["Timestamp"].dt.dayofweek
df["Hour"] = df["Timestamp"].dt.hour

# Monday-Friday are weekdays; Saturday-Sunday are weekends.
df["Day_Type"] = np.where(
    df["Weekday_Order"] < 5,
    "Weekday",
    "Weekend",
)

station_columns = ["Station_ID", "Station_Class"]

print(f"Valid AQI observations loaded: {len(df):,}")
print(f"Stations represented: {df['Station_ID'].nunique():,}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")


# ============================================================
# 2. SEASONAL AQI TRENDS
#    One row per station, year and season.
# ============================================================

seasonal = (
    df.groupby(
        station_columns + ["Year", "Season"],
        dropna=False,
        as_index=False,
    )
    .agg(
        Mean_AQI=("AQI", "mean"),
        Median_AQI=("AQI", "median"),
        Min_AQI=("AQI", "min"),
        Max_AQI=("AQI", "max"),
        Observation_Count=("AQI", "count"),
    )
)

season_order = {
    "Winter": 1,
    "Summer": 2,
    "Monsoon": 3,
    "Autumn": 4,
}

seasonal["Season_Order"] = seasonal["Season"].map(season_order)

seasonal = seasonal.sort_values(
    ["Year", "Season_Order", "Station_ID"]
)

seasonal.to_csv(
    OUTPUT_DIR / "Seasonal_AQI_Trends.csv",
    index=False,
)


# ============================================================
# 3. DAY-OF-WEEK AQI TRENDS
#    One row per station and weekday, pooled across years.
# ============================================================

weekday = (
    df.groupby(
        station_columns + ["Day_of_Week", "Weekday_Order"],
        dropna=False,
        as_index=False,
    )
    .agg(
        Mean_AQI=("AQI", "mean"),
        Median_AQI=("AQI", "median"),
        Min_AQI=("AQI", "min"),
        Max_AQI=("AQI", "max"),
        Observation_Count=("AQI", "count"),
        Mean_Wind_Speed=(
            WIND_COLUMN,
            "mean",
        ),
        Mean_Precipitation=(
            "precipitation (mm)",
            "mean",
        ),
    )
    .sort_values(["Station_ID", "Weekday_Order"])
)

weekday.to_csv(
    OUTPUT_DIR / "Weekday_AQI_Trends.csv",
    index=False,
)


# ============================================================
# 4. WEEKDAY VS WEEKEND AQI
#    One row per station and day type, pooled across years.
# ============================================================

weekday_weekend = (
    df.groupby(
        station_columns + ["Day_Type"],
        dropna=False,
        as_index=False,
    )
    .agg(
        Mean_AQI=("AQI", "mean"),
        Median_AQI=("AQI", "median"),
        Min_AQI=("AQI", "min"),
        Max_AQI=("AQI", "max"),
        Observation_Count=("AQI", "count"),
        Mean_Wind_Speed=(
            WIND_COLUMN,
            "mean",
        ),
        Mean_Precipitation=(
            "precipitation (mm)",
            "mean",
        ),
    )
)

weekday_weekend.to_csv(
    OUTPUT_DIR / "Weekday_Weekend_AQI.csv",
    index=False,
)


# ============================================================
# 5. WEATHER/AQI OBSERVATIONS FOR SCATTER PLOTS
#    One row per valid AQI observation.
# ============================================================

weather_output_cols = [
    "Timestamp",
    "Date",
    "Year",
    "Month_Number",
    "Month_Name",
    "Season",
    "Day_of_Week",
    "Weekday_Order",
    "Day_Type",
    "Hour",
    "Station_ID",
    "Station_Class",
    "AQI",
] + WEATHER_COLUMNS

weather_observations = df[weather_output_cols].copy()

weather_observations.to_csv(
    OUTPUT_DIR / "AQI_Weather_Observations.csv",
    index=False,
)


# ============================================================
# 6. DAILY WIND/AQI COMPARISON
#    Aggregate hourly observations to station-day first.
# ============================================================

daily_agg = {
    "Mean_Daily_AQI": ("AQI", "mean"),
    "Median_Daily_AQI": ("AQI", "median"),
    "Max_Hourly_AQI": ("AQI", "max"),
    "Hourly_Observation_Count": ("AQI", "count"),
    "Mean_Wind_Speed_kmh": (WIND_COLUMN, "mean"),
    "Max_Wind_Speed_kmh": (WIND_COLUMN, "max"),
    "Total_Precipitation_mm": (
        "precipitation (mm)",
        "sum",
    ),
    "Mean_Temperature_C": (
        "temperature_2m (°C)",
        "mean",
    ),
    "Mean_Relative_Humidity_pct": (
        "relative_humidity_2m (%)",
        "mean",
    ),
}

daily_wind = (
    df.groupby(
        station_columns
        + ["Date", "Year", "Month_Number", "Season"],
        dropna=False,
        as_index=False,
    )
    .agg(**daily_agg)
)

# Assign relative wind categories using only valid daily wind speeds.
# These categories are dataset-relative thirds, not fixed
# meteorological thresholds.
valid_wind = daily_wind["Mean_Wind_Speed_kmh"].notna()

daily_wind["Wind_Category"] = "Unknown"

wind_rank = daily_wind.loc[
    valid_wind,
    "Mean_Wind_Speed_kmh",
].rank(
    method="average",
    pct=True,
)

daily_wind.loc[valid_wind, "Wind_Category"] = np.select(
    [
        wind_rank <= (1 / 3),
        wind_rank <= (2 / 3),
    ],
    [
        "Calmer third",
        "Middle third",
    ],
    default="Windier third",
)

daily_wind.to_csv(
    OUTPUT_DIR / "Daily_Wind_AQI.csv",
    index=False,
)


# ============================================================
# 7. SUMMARY
# ============================================================

print("\nEnvironmental-analysis CSVs created:")

output_files = [
    "Seasonal_AQI_Trends.csv",
    "Weekday_AQI_Trends.csv",
    "Weekday_Weekend_AQI.csv",
    "AQI_Weather_Observations.csv",
    "Daily_Wind_AQI.csv",
]

for filename in output_files:
    path = OUTPUT_DIR / filename

    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f" - {filename}: {size_mb:.2f} MB")

print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")

print("\nSuggested Power BI visuals:")
print(
    " - Seasonal_AQI_Trends.csv: "
    "seasonal AQI by year and station."
)
print(
    " - Weekday_AQI_Trends.csv: "
    "Mean_AQI by Day_of_Week, ordered Monday-Sunday."
)
print(
    " - Weekday_Weekend_AQI.csv: "
    "compare Mean_AQI for weekdays and weekends."
)
print(
    " - AQI_Weather_Observations.csv: "
    "scatter plots of AQI against wind, humidity, temperature, etc."
)
print(
    " - Daily_Wind_AQI.csv: "
    "compare Mean_Daily_AQI across Wind_Category."
)

print(
    "\nNOTE: Wind categories are relative to this dataset's "
    "daily station-level wind-speed distribution."
)
