from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path("../Datasets/Final_Dataset.csv")

MODEL_DIRS = {
    "XGBoost": Path("../Outputs/XGBoost"),
    "Random Forest": Path("../Outputs/Random_Forest"),
}

OUTPUT_DIR = Path("../Outputs/PowerBI")

TARGETS = {
    "AQI_future_1": 1,
    "AQI_future_24": 24,
}

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD ORIGINAL DATASET AND BUILD WEATHER LOOKUP
# ============================================================

print("Loading original dataset...")

df = pd.read_csv(DATASET_FILE, low_memory=False)

date_text = (
    df["Year"].astype("Int64").astype(str) + "-"
    + df["Month"].astype("Int64").astype(str) + "-"
    + df["Day"].astype("Int64").astype(str) + " "
    + df["Time"].astype(str).str.strip()
)

df["Timestamp"] = pd.to_datetime(
    date_text,
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)

df["Station_ID"] = df["Station_ID"].astype(str).str.strip()
df["Station_Split"] = (
    df["Station_Class"].astype(str).str.strip().str.lower()
)

df = df.dropna(subset=["Timestamp"]).copy()

context_columns = [
    "Timestamp",
    "Station_ID",
    "latitude",
    "longitude",
    "elevation",
    *WEATHER_COLUMNS,
]

missing = [c for c in context_columns if c not in df.columns]
if missing:
    raise KeyError(f"Missing dataset columns: {missing}")

weather_lookup = df[context_columns].copy()

# Do not silently duplicate prediction rows when joining weather.
duplicate_keys = weather_lookup.duplicated(
    ["Timestamp", "Station_ID"], keep=False
)

if duplicate_keys.any():
    raise ValueError(
        f"{duplicate_keys.sum():,} duplicate weather lookup rows found "
        "for Timestamp + Station_ID. Resolve these before continuing."
    )

stations = (
    df[
        [
            "Station_ID",
            "Station_Split",
            "latitude",
            "longitude",
            "elevation",
        ]
    ]
    .drop_duplicates(subset=["Station_ID"])
    .sort_values("Station_ID")
    .reset_index(drop=True)
)

stations.to_csv(OUTPUT_DIR / "Stations.csv", index=False)


# ============================================================
# 2. LOAD AND STANDARDIZE BOTH MODEL FAMILIES
# ============================================================

def load_predictions(algorithm, source_dir, target):
    """Read one prediction file and standardize its column names."""

    path = source_dir / f"predictions_{target}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    pred = pd.read_csv(path, low_memory=False)

    required = [
        "_Timestamp",
        "Station_ID",
        "Station_Split",
        target,
        "Model",
        "Prediction",
        "Residual",
    ]

    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    pred["Timestamp"] = pd.to_datetime(
        pred["_Timestamp"], errors="coerce"
    )
    pred["Station_ID"] = pred["Station_ID"].astype(str).str.strip()
    pred["Station_Split"] = (
        pred["Station_Split"].astype(str).str.strip().str.lower()
    )
    pred["Model"] = pred["Model"].astype(str).str.strip()

    pred = pred.rename(
        columns={
            target: "Actual_AQI",
            "Prediction": "Predicted_AQI",
            "Residual": "Residual_AQI",
        }
    )

    pred["Algorithm"] = algorithm
    pred["Target"] = target
    pred["Horizon_Hours"] = TARGETS[target]

    pred = pred.dropna(
        subset=[
            "Timestamp",
            "Station_ID",
            "Actual_AQI",
            "Predicted_AQI",
        ]
    ).copy()

    # The residual convention is actual minus prediction.
    # Recalculate derived fields consistently.
    pred["Residual_AQI"] = (
        pred["Actual_AQI"] - pred["Predicted_AQI"]
    )
    pred["Absolute_Error"] = pred["Residual_AQI"].abs()
    pred["Squared_Error"] = pred["Residual_AQI"] ** 2

    pred["Forecast_Target_Timestamp"] = (
        pred["Timestamp"]
        + pd.to_timedelta(pred["Horizon_Hours"], unit="h")
    )

    return pred


all_predictions = []

for algorithm, source_dir in MODEL_DIRS.items():
    print(f"\nReading {algorithm} predictions...")

    for target in TARGETS:
        pred = load_predictions(algorithm, source_dir, target)
        all_predictions.append(pred)

predictions = pd.concat(all_predictions, ignore_index=True)

# Calendar fields for Power BI slicers and charts.
predictions["Date"] = predictions["Timestamp"].dt.normalize()
predictions["Year"] = predictions["Timestamp"].dt.year
predictions["Month_Number"] = predictions["Timestamp"].dt.month
predictions["Month_Name"] = predictions["Timestamp"].dt.strftime("%b")
predictions["Day"] = predictions["Timestamp"].dt.day
predictions["Hour"] = predictions["Timestamp"].dt.hour
predictions["Day_of_Week"] = predictions["Timestamp"].dt.day_name()

predictions["Test_Group"] = predictions["Station_Split"].map(
    {
        "seen": "Seen Stations",
        "unseen": "Unseen Stations",
    }
).fillna(predictions["Station_Split"].str.title())

# Add weather and station context without changing prediction counts.
predictions = predictions.merge(
    weather_lookup,
    on=["Timestamp", "Station_ID"],
    how="left",
    validate="many_to_one",
    suffixes=("", "_Source"),
)

# Keep coordinates from the prediction output when present;
# otherwise fill them from the original dataset.
for col in ["latitude", "longitude", "elevation"]:
    source_col = f"{col}_Source"

    if source_col in predictions.columns:
        if col in predictions.columns:
            predictions[col] = predictions[col].fillna(
                predictions[source_col]
            )
            predictions.drop(columns=source_col, inplace=True)
        else:
            predictions.rename(columns={source_col: col}, inplace=True)

prediction_columns = [
    "Algorithm",
    "Model",
    "Target",
    "Horizon_Hours",
    "Timestamp",
    "Forecast_Target_Timestamp",
    "Date",
    "Year",
    "Month_Number",
    "Month_Name",
    "Day",
    "Hour",
    "Day_of_Week",
    "Station_ID",
    "Station_Split",
    "Test_Group",
    "latitude",
    "longitude",
    "elevation",
    "Training_Period",
    "Actual_AQI",
    "Predicted_AQI",
    "Residual_AQI",
    "Absolute_Error",
    "Squared_Error",
    *WEATHER_COLUMNS,
]

prediction_columns = [
    c for c in prediction_columns if c in predictions.columns
]

predictions = predictions[prediction_columns].copy()

predictions.to_csv(
    OUTPUT_DIR / "Forecast_Predictions.csv",
    index=False,
)

print(f"\nForecast rows exported: {len(predictions):,}")

# ============================================================
# 2B. COMMON OBSERVATIONS FOR FAIR ALGORITHM COMPARISON
# ============================================================

comparison_keys = [
    "Timestamp",
    "Station_ID",
    "Target",
    "Horizon_Hours",
    "Model",
]

algorithm_counts = (
    predictions.groupby(comparison_keys)["Algorithm"]
    .nunique()
    .rename("Algorithm_Count")
    .reset_index()
)

common_keys = algorithm_counts.loc[
    algorithm_counts["Algorithm_Count"] == 2,
    comparison_keys,
]

common_predictions = predictions.merge(
    common_keys,
    on=comparison_keys,
    how="inner",
    validate="many_to_one",
)

common_predictions.to_csv(
    OUTPUT_DIR / "Common_Observation_Comparison.csv",
    index=False,
)

print(
    "Common-observation rows:",
    f"{len(common_predictions):,}"
)

print(
    "Common observations per algorithm/model/horizon:",
    common_predictions.groupby(
        ["Algorithm", "Model", "Horizon_Hours"]
    ).size().to_dict(),
)


# ============================================================
# 3. COMBINE MODEL METRICS
# ============================================================

metric_frames = []

for algorithm, source_dir in MODEL_DIRS.items():
    path = source_dir / "metrics.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    metrics = pd.read_csv(path)
    metrics["Algorithm"] = algorithm
    metrics["Horizon_Hours"] = metrics["Target"].map(TARGETS)

    metric_frames.append(metrics)

all_metrics = pd.concat(metric_frames, ignore_index=True)

# Put identifiers first; retain all metric columns from both files.
id_columns = [
    "Algorithm",
    "Target",
    "Horizon_Hours",
    "Model",
    "Evaluation",
    "Training_Period",
]

all_metrics = all_metrics[
    [c for c in id_columns if c in all_metrics.columns]
    + [
        c for c in all_metrics.columns
        if c not in id_columns
    ]
]

all_metrics.to_csv(
    OUTPUT_DIR / "Model_Metrics.csv",
    index=False,
)

print(f"Metric rows exported: {len(all_metrics):,}")


# ============================================================
# 4. STATION-LEVEL PERFORMANCE
# ============================================================

station_performance = (
    predictions.groupby(
        [
            "Algorithm",
            "Model",
            "Target",
            "Horizon_Hours",
            "Station_ID",
            "Station_Split",
            "Test_Group",
        ],
        as_index=False,
    )
    .agg(
        MAE=("Absolute_Error", "mean"),
        Mean_Squared_Error=("Squared_Error", "mean"),
        Mean_Residual=("Residual_AQI", "mean"),
        Forecast_Count=("Actual_AQI", "size"),
        Mean_Actual_AQI=("Actual_AQI", "mean"),
        Mean_Predicted_AQI=("Predicted_AQI", "mean"),
    )
)

station_performance["RMSE"] = np.sqrt(
    station_performance["Mean_Squared_Error"]
)

station_performance.drop(
    columns=["Mean_Squared_Error"], inplace=True
)

station_performance.to_csv(
    OUTPUT_DIR / "Station_Performance.csv",
    index=False,
)

print(f"Station performance rows: {len(station_performance):,}")


# ============================================================
# 5. MODEL COMPARISON SUMMARY
# ============================================================

summary = (
    predictions.groupby(
        [
            "Algorithm",
            "Model",
            "Target",
            "Horizon_Hours",
            "Test_Group",
        ],
        as_index=False,
    )
    .agg(
        MAE=("Absolute_Error", "mean"),
        Mean_Squared_Error=("Squared_Error", "mean"),
        Mean_Residual=("Residual_AQI", "mean"),
        Forecast_Count=("Actual_AQI", "size"),
        Mean_Actual_AQI=("Actual_AQI", "mean"),
        Mean_Predicted_AQI=("Predicted_AQI", "mean"),
    )
)

summary["RMSE"] = np.sqrt(summary["Mean_Squared_Error"])
summary.drop(columns=["Mean_Squared_Error"], inplace=True)

summary.to_csv(
    OUTPUT_DIR / "Model_Comparison_Summary.csv",
    index=False,
)

print(f"Summary rows: {len(summary):,}")


# ============================================================
# 6. DATA QUALITY AND SAMPLE-COUNT CHECKS
# ============================================================

print("\nRows by algorithm, horizon, model, and station group:")
print(
    predictions.groupby(
        ["Algorithm", "Horizon_Hours", "Model", "Test_Group"]
    ).size().to_string()
)

print("\nMissing weather values in prediction rows:")
print(
    predictions[WEATHER_COLUMNS]
    .isna()
    .sum()
    .sort_values(ascending=False)
    .to_string()
)

print("\nExported Power BI files:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"{path.name}: {path.stat().st_size / (1024**2):.2f} MB")

print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")
