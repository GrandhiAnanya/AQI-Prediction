"""
Standalone GPU Random Forest hyperparameter tuning for the AQI project.

Run from the AQI-Prediction project root:
    conda activate aqi-rapids
    python tune_aqi_gpu_random_forest.py

Outputs:
    temporal_spatial_results_RF_GPU/rf_tuning_results.csv
    temporal_spatial_results_RF_GPU/best_hyperparameters.json
    temporal_spatial_results_RF_GPU/best_hyperparameters.csv

The script tunes separately for:
    AQI_future_1  / Model A (without weather features)
    AQI_future_1  / Model B (with weather features)
    AQI_future_24 / Model A (without weather features)
    AQI_future_24 / Model B (with weather features)

It uses 2020-2023 seen-station data for fitting and 2024 seen-station
data for validation. It does not fit or evaluate on 2025 test data.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
import cupy as cp
from cuml.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

INPUT_CSV = Path("../Datasets/Final_Dataset.csv")
OUTPUT_DIR = Path("../Outputs/Random_Forest")
RANDOM_SEED = 42

TARGETS = ["AQI_future_1", "AQI_future_24"]

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

EXCLUDE = {
    "Day", "Month", "Year", "Time", "Day_of_Week", "Season",
    "Station_ID", "Station_Class", "Station_Split",
    "_Timestamp", "_year", "_target_time", *TARGETS,
}

# Since your full GPU run completed, default to all eligible training rows.
# Change to 600_000 if you want a quicker first pass or hit GPU memory errors.
MAX_TUNING_TRAIN_ROWS = None

# Validation is sampled to make repeated scoring faster. Set to None to use
# every eligible 2024 validation row.
MAX_TUNING_VAL_ROWS = None

# Twelve candidates per target/model combination = up to 48 GPU fits.
# This is a focused search, not an exhaustive Cartesian grid.
PARAM_CANDIDATES = [
    # Baseline close to the configuration that already worked for you.
    {"n_estimators": 200, "max_depth": 16, "max_features": 0.7,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 100, "max_depth": 12, "max_features": 0.5,
     "min_samples_leaf": 1, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 12, "max_features": 0.7,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 300, "max_depth": 16, "max_features": 0.7,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 20, "max_features": 0.7,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 16, "max_features": 0.5,
     "min_samples_leaf": 1, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 16, "max_features": 1.0,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 16, "max_features": 0.7,
     "min_samples_leaf": 4, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 20, "max_features": 0.5,
     "min_samples_leaf": 4, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 300, "max_depth": 20, "max_features": 0.7,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 0.8},

    {"n_estimators": 200, "max_depth": 16, "max_features": 0.7,
     "min_samples_leaf": 2, "n_bins": 128, "max_samples": 0.8},

    {"n_estimators": 300, "max_depth": 16, "max_features": 0.5,
     "min_samples_leaf": 2, "n_bins": 64, "max_samples": 1.0},
]

# cuML-specific/common parameters held constant during tuning.
BASE_RF_PARAMS = {
    "min_samples_split": 2,
    "bootstrap": True,
    "n_streams": 4,
    "random_state": RANDOM_SEED,
    "output_type": "cupy",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def free_gpu_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def clean_features(frame, columns):
    """Convert model features to float32 after removing invalid values."""
    X = frame.loc[:, columns]
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().to_numpy().any():
        raise ValueError(
            "Missing/non-finite features remain after model-specific filtering."
        )
    return X.astype(np.float32)


def candidate_key(target, model_name):
    return f"{target}__Model_{model_name}"


def save_progress(rows, best_params_by_key):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "rf_tuning_results.csv", index=False
    )

    # JSON is the canonical file the final training script should load.
    with open(
        OUTPUT_DIR / "best_hyperparameters.json", "w", encoding="utf-8"
    ) as f:
        json.dump(best_params_by_key, f, indent=2)

    best_rows = []
    for key, item in best_params_by_key.items():
        best_rows.append({
            "Key": key,
            "Target": item["target"],
            "Model": item["model"],
            "Best_Validation_MAE": item["validation_mae"],
            "Best_Validation_RMSE": item["validation_rmse"],
            "Best_Validation_R2": item["validation_r2"],
            "Best_Params": json.dumps(item["params"]),
        })

    pd.DataFrame(best_rows).to_csv(
        OUTPUT_DIR / "best_hyperparameters.csv", index=False
    )


# ---------------------------------------------------------------------
# LOAD AND PREPARE DATA (same feature/split logic as the final script)
# ---------------------------------------------------------------------

print("CuPy:", cp.__version__)
print("CUDA GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("Loading dataset...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_CSV.resolve()}\n"
        "Run this script from the AQI-Prediction project root, or update INPUT_CSV."
    )

df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

required = {
    "Year", "Month", "Day", "Time", "Day_of_Week", "Season",
    "Station_ID", "Station_Class", "latitude", "longitude", "elevation",
    *TARGETS, *WEATHER_COLUMNS,
}
missing = sorted(required - set(df.columns))
if missing:
    raise KeyError(f"Missing required columns: {missing}")

date_text = (
    df["Year"].astype("Int64").astype(str) + "-"
    + df["Month"].astype("Int64").astype(str) + "-"
    + df["Day"].astype("Int64").astype(str) + " "
    + df["Time"].astype(str).str.strip()
)

df["_Timestamp"] = pd.to_datetime(
    date_text, format="%Y-%m-%d %H:%M:%S", errors="coerce"
)
df = df.dropna(subset=["_Timestamp"]).copy()
df["_year"] = df["_Timestamp"].dt.year
df["Hour"] = df["_Timestamp"].dt.hour

for col in ["latitude", "longitude", "elevation", *TARGETS, *WEATHER_COLUMNS]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = pd.get_dummies(df, columns=["Day_of_Week", "Season"], dtype=np.uint8)
df["Station_ID"] = df["Station_ID"].astype(str).str.strip()
df["Station_Class"] = df["Station_Class"].astype(str).str.strip().str.lower()

valid_classes = {"seen", "unseen"}
observed = set(df["Station_Class"].dropna().unique())
if not observed.issubset(valid_classes):
    raise ValueError(
        f"Station_Class must contain only seen/unseen; found {observed}"
    )

df = df[
    df["Station_ID"].ne("") & df["Station_Class"].isin(valid_classes)
].copy()

class_counts = df.groupby("Station_ID")["Station_Class"].nunique()
if (class_counts > 1).any():
    raise ValueError(
        "Some Station_ID values have inconsistent Station_Class values: "
        f"{class_counts[class_counts > 1].index.tolist()[:10]}"
    )

seen_ids = set(df.loc[df["Station_Class"] == "seen", "Station_ID"])
unseen_ids = set(df.loc[df["Station_Class"] == "unseen", "Station_ID"])
if not seen_ids or not unseen_ids:
    raise ValueError("Both seen and unseen stations must be present.")

# Same feature-selection rule as the final training script.
features = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in EXCLUDE
    and not col.startswith("AQI_future_")
    and not col.lower().startswith(("target", "label", "future_"))
]

model_features = {
    "A": [col for col in features if col not in WEATHER_COLUMNS],
    "B": features,
}

print(f"Seen stations: {len(seen_ids)} | Unseen stations: {len(unseen_ids)}")
print(f"Feature counts: Model A={len(model_features['A'])}, "
      f"Model B={len(model_features['B'])}")
print(f"Training sample cap: {MAX_TUNING_TRAIN_ROWS}")
print(f"Validation sample cap: {MAX_TUNING_VAL_ROWS}")
print(f"Candidates per target/model: {len(PARAM_CANDIDATES)}")
print("This may take several hours because each combination is tuned separately.")

all_results = []
best_params_by_key = {}


# ---------------------------------------------------------------------
# TUNE FOUR TARGET / FEATURE-SET COMBINATIONS
# ---------------------------------------------------------------------

for target in TARGETS:
    horizon = 1 if target == "AQI_future_1" else 24
    train_cutoff = pd.Timestamp("2024-01-01") - pd.Timedelta(hours=horizon)
    val_cutoff = pd.Timestamp("2025-01-01") - pd.Timedelta(hours=horizon)

    for model_name, cols in model_features.items():
        print("\n" + "=" * 78)
        print(f"TUNING | Target={target} | Model={model_name}")
        print("=" * 78)

        # Keep only metadata, features, and this target for this run.
        needed = list(dict.fromkeys([
            "_Timestamp", "_year", "Station_ID", "Station_Class",
            *cols, target,
        ]))
        model_data = df.loc[:, needed].replace(
            [np.inf, -np.inf], np.nan
        )
        model_data = model_data.dropna(subset=cols + [target])

        train_mask = (
            model_data["_year"].between(2020, 2023)
            & (model_data["_Timestamp"] < train_cutoff)
            & model_data["Station_ID"].isin(seen_ids)
        )
        val_mask = (
            (model_data["_year"] == 2024)
            & (model_data["_Timestamp"] < val_cutoff)
            & model_data["Station_ID"].isin(seen_ids)
        )

        train_df = model_data.loc[train_mask]
        val_df = model_data.loc[val_mask]

        print(f"Eligible train rows: {len(train_df):,}")
        print(f"Eligible validation rows: {len(val_df):,}")

        if train_df.empty or val_df.empty:
            raise ValueError(f"Empty train/validation split for {target}, Model {model_name}")

        if MAX_TUNING_TRAIN_ROWS is not None and len(train_df) > MAX_TUNING_TRAIN_ROWS:
            train_df = train_df.sample(
                n=MAX_TUNING_TRAIN_ROWS,
                random_state=RANDOM_SEED,
            )

        if MAX_TUNING_VAL_ROWS is not None and len(val_df) > MAX_TUNING_VAL_ROWS:
            val_df = val_df.sample(
                n=MAX_TUNING_VAL_ROWS,
                random_state=RANDOM_SEED,
            )

        print(f"Rows used for tuning: train={len(train_df):,}, "
              f"validation={len(val_df):,}")

        X_train_cpu = clean_features(train_df, cols)
        y_train_np = train_df[target].to_numpy(dtype=np.float32, copy=True)
        X_val_cpu = clean_features(val_df, cols)
        y_val_np = val_df[target].to_numpy(dtype=np.float32, copy=True)

        # Keep the fixed tuning data on GPU across trials.
        X_train_gpu = cp.asarray(
            X_train_cpu.to_numpy(dtype=np.float32, copy=True)
        )
        y_train_gpu = cp.asarray(y_train_np)
        X_val_gpu = cp.asarray(
            X_val_cpu.to_numpy(dtype=np.float32, copy=True)
        )

        best_mae = float("inf")
        best_item = None

        try:
            for trial, candidate in enumerate(PARAM_CANDIDATES, start=1):
                # Keep the Random Forest seed identical across candidates so
                # validation differences primarily reflect hyperparameters.
                # BASE_RF_PARAMS already sets random_state=RANDOM_SEED; explicitly
                # enforce it here in case a candidate ever includes that key.
                params = {**BASE_RF_PARAMS, **candidate}
                params["random_state"] = RANDOM_SEED

                print(f"\nTrial {trial}/{len(PARAM_CANDIDATES)}")
                print("Candidate hyperparameters:")
                print(json.dumps(candidate, indent=2))
                print(f"Fixed Random Forest random_state: {params['random_state']}")

                model = None
                pred_gpu = None
                pred_np = None
                start = time.perf_counter()

                try:
                    model = RandomForestRegressor(**params)
                    model.fit(X_train_gpu, y_train_gpu)
                    cp.cuda.Stream.null.synchronize()
                    fit_seconds = time.perf_counter() - start

                    pred_gpu = model.predict(X_val_gpu)
                    pred_np = cp.asnumpy(
                        cp.asarray(pred_gpu)
                    ).reshape(-1)

                    mae = float(mean_absolute_error(y_val_np, pred_np))
                    rmse = float(np.sqrt(mean_squared_error(y_val_np, pred_np)))
                    r2 = float(r2_score(y_val_np, pred_np))

                    result = {
                        "Target": target,
                        "Model": model_name,
                        "Trial": trial,
                        **candidate,
                        "Validation_MAE": mae,
                        "Validation_RMSE": rmse,
                        "Validation_R2": r2,
                        "Fit_Seconds": fit_seconds,
                        "Train_Rows": len(train_df),
                        "Validation_Rows": len(val_df),
                    }
                    all_results.append(result)

                    print(
                        f"MAE={mae:.4f} | RMSE={rmse:.4f} | "
                        f"R2={r2:.4f} | fit={fit_seconds / 60:.1f} min"
                    )

                    if mae < best_mae:
                        best_mae = mae
                        # Store all required cuML parameters for final fitting.
                        best_item = {
                            "target": target,
                            "model": model_name,
                            "validation_mae": mae,
                            "validation_rmse": rmse,
                            "validation_r2": r2,
                            "params": params.copy(),
                        }
                        best_params_by_key[candidate_key(target, model_name)] = best_item

                finally:
                    if model is not None:
                        del model
                    if pred_gpu is not None:
                        del pred_gpu
                    if pred_np is not None:
                        del pred_np
                    free_gpu_memory()

                # Persist after every successful trial, so completed work is kept.
                save_progress(all_results, best_params_by_key)

        finally:
            del X_train_gpu, y_train_gpu, X_val_gpu
            del X_train_cpu, X_val_cpu, y_train_np, y_val_np
            del train_df, val_df, model_data
            free_gpu_memory()

        print(f"\nBest for {target} / Model {model_name}:")
        print(json.dumps(best_item, indent=2))

        del train_mask, val_mask
        gc.collect()

print("\nTUNING COMPLETE")
print("Results:", (OUTPUT_DIR / "rf_tuning_results.csv").resolve())
print("Best parameters:", (OUTPUT_DIR / "best_hyperparameters.json").resolve())
print("Summary:", (OUTPUT_DIR / "best_hyperparameters.csv").resolve())
print("\nBest configurations:")
for key, item in best_params_by_key.items():
    print(
        f"{key}: MAE={item['validation_mae']:.4f}, "
        f"RMSE={item['validation_rmse']:.4f}, "
        f"R2={item['validation_r2']:.4f}"
    )
