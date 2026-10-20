
from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import cupy as cp
from cuml.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

INPUT_CSV = Path("../Datasets/Final_Dataset.csv")
OUTPUT_DIR = Path("../Outputs/Random_Forest")
MODEL_DIR = OUTPUT_DIR / "trained_models"
TARGETS = ["AQI_future_1", "AQI_future_24"]
RANDOM_SEED = 42

WEATHER_COLUMNS = [
    "temperature_2m (°C)", "relative_humidity_2m (%)",
    "dew_point_2m (°C)", "precipitation (mm)",
    "surface_pressure (hPa)", "wind_speed_10m (km/h)",
    "wind_direction_10m (°)", "shortwave_radiation (W/m²)",
    "cloud_cover (%)",
]

# Load the best parameters selected by the completed tuning run.
BEST_PARAMS_FILE = OUTPUT_DIR / "best_hyperparameters.json"

if not BEST_PARAMS_FILE.exists():
    raise FileNotFoundError(
        f"Best-parameters file not found: {BEST_PARAMS_FILE.resolve()}\\n"
        "Run tune_aqi_gpu_random_forest.py first."
    )

with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
    BEST_PARAMS_MAP = json.load(f)

EXPECTED_PARAM_KEYS = {
    f"{target}__Model_{model_name}"
    for target in TARGETS
    for model_name in ("A", "B")
}
missing_param_keys = EXPECTED_PARAM_KEYS - set(BEST_PARAMS_MAP)
if missing_param_keys:
    raise KeyError(
        "The tuning JSON is missing configurations for: "
        + ", ".join(sorted(missing_param_keys))
        + ". Ensure all four target/model combinations completed successfully."
    )

# Keep final fitting on the same fixed random seed used during tuning.
for _key, _item in BEST_PARAMS_MAP.items():
    _item["params"]["random_state"] = RANDOM_SEED

print("Loaded tuned Random Forest configurations:")
for _key in sorted(EXPECTED_PARAM_KEYS):
    _item = BEST_PARAMS_MAP[_key]
    print(
        f"  {_key}: validation MAE={_item['validation_mae']:.4f}; "
        f"params={json.dumps(_item['params'], sort_keys=True)}"
    )

# None uses every eligible training row. If GPU memory is insufficient, try 250_000.
MAX_TRAIN_ROWS = None
PREDICT_BATCH_SIZE = 50_000

EXCLUDE = {
    "Day", "Month", "Year", "Time", "Day_of_Week", "Season",
    "Station_ID", "Station_Class", "Station_Split",
    "_Timestamp", "_year", "_target_time", *TARGETS,
}


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "N": len(y_true),
    }


def clean_features(frame, columns):
    X = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Missing features remain after model-specific dropna.")
    return X.astype(np.float32)


def select_training_rows(frame, seed):
    if MAX_TRAIN_ROWS is None or len(frame) <= MAX_TRAIN_ROWS:
        return frame
    sampled = frame.sample(n=MAX_TRAIN_ROWS, random_state=seed)
    print(f"Training subsample: {len(sampled):,} / {len(frame):,} rows")
    return sampled


def predict_gpu_batched(model, X_cpu, batch_size=PREDICT_BATCH_SIZE):
    predictions = []
    for start in range(0, len(X_cpu), batch_size):
        stop = min(start + batch_size, len(X_cpu))
        batch_np = X_cpu.iloc[start:stop].to_numpy(dtype=np.float32, copy=True)
        batch_gpu = cp.asarray(batch_np)
        pred_gpu = model.predict(batch_gpu)
        predictions.append(cp.asnumpy(cp.asarray(pred_gpu)).reshape(-1))
        del batch_np, batch_gpu, pred_gpu
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def free_gpu_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def make_model(target, model_name):
    """Create the cuML Random Forest using this target/model's tuned parameters."""
    key = f"{target}__Model_{model_name}"
    params = BEST_PARAMS_MAP[key]["params"].copy()
    params["random_state"] = RANDOM_SEED
    return RandomForestRegressor(**params)


print("CuPy:", cp.__version__)
print("CUDA GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"CSV not found: {INPUT_CSV.resolve()}. Set INPUT_CSV to your dataset path."
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
    raise ValueError(f"Station_Class must contain only seen/unseen; found {observed}")

df = df[df["Station_ID"].ne("") & df["Station_Class"].isin(valid_classes)].copy()
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

df["Station_Split"] = df["Station_Class"]
df[[
    "Station_ID", "Station_Class", "Station_Split",
    "latitude", "longitude", "elevation"
]].drop_duplicates("Station_ID").to_csv(OUTPUT_DIR / "station_split.csv", index=False)

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
if not model_features["A"] or not model_features["B"]:
    raise ValueError("A model feature set is empty.")

with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(model_features, f, indent=2)

with open(OUTPUT_DIR / "applied_best_hyperparameters.json", "w", encoding="utf-8") as f:
    json.dump(BEST_PARAMS_MAP, f, indent=2)

print(f"Seen stations: {len(seen_ids)} | Unseen stations: {len(unseen_ids)}")
print(f"Features: A={len(model_features['A'])}, B={len(model_features['B'])}")

for target in TARGETS:
    out = OUTPUT_DIR / f"predictions_{target}.csv"
    if out.exists():
        out.unlink()

metrics = []

for target in TARGETS:
    horizon = 1 if target == "AQI_future_1" else 24
    data = df.replace([np.inf, -np.inf], np.nan).copy()
    data["_target_time"] = data["_Timestamp"] + pd.to_timedelta(horizon, unit="h")

    for model_name, cols in model_features.items():
        print(f"\n{'=' * 70}\nTarget={target} | Model={model_name}")
        # Drop only missing values in this model's inputs and current target.
        model_data = data.dropna(subset=cols + [target]).copy()
        print(f"Rows after model-specific filtering: {len(model_data):,}")

        train = model_data[
            model_data["_year"].between(2020, 2023)
            & (model_data["_target_time"] < "2024-01-01")
            & model_data["Station_ID"].isin(seen_ids)
        ]
        validation = model_data[
            (model_data["_year"] == 2024)
            & (model_data["_target_time"] < "2025-01-01")
            & model_data["Station_ID"].isin(seen_ids)
        ]
        final_train = model_data[
            model_data["_year"].between(2020, 2024)
            & (model_data["_target_time"] < "2025-01-01")
            & model_data["Station_ID"].isin(seen_ids)
        ]
        test_seen = model_data[
            (model_data["_year"] == 2025)
            & (model_data["_target_time"] < "2026-01-01")
            & model_data["Station_ID"].isin(seen_ids)
        ]
        test_unseen = model_data[
            (model_data["_year"] == 2025)
            & (model_data["_target_time"] < "2026-01-01")
            & model_data["Station_ID"].isin(unseen_ids)
        ]

        splits = {
            "train": train, "validation": validation,
            "final_train": final_train, "test_seen": test_seen,
            "test_unseen": test_unseen,
        }
        print("Split sizes:", {k: len(v) for k, v in splits.items()})
        if any(v.empty for v in splits.values()):
            raise ValueError(f"Empty split for {target}, Model {model_name}")

        # Phase 1: fit 2020-2023 and validate on 2024.
        train_fit = select_training_rows(
            train, RANDOM_SEED + horizon + (0 if model_name == "A" else 100)
        )
        X_train_cpu = clean_features(train_fit, cols)
        X_val_cpu = clean_features(validation, cols)
        y_train_np = train_fit[target].to_numpy(dtype=np.float32, copy=True)

        model = make_model(target, model_name)
        start = time.perf_counter()
        X_train_gpu = cp.asarray(X_train_cpu.to_numpy(dtype=np.float32, copy=True))
        y_train_gpu = cp.asarray(y_train_np)
        model.fit(X_train_gpu, y_train_gpu)
        cp.cuda.Stream.null.synchronize()
        fit_minutes = (time.perf_counter() - start) / 60
        print(f"Validation-fit time: {fit_minutes:.2f} min")

        val_pred = predict_gpu_batched(model, X_val_cpu)
        result = {
            "Target": target, "Model": model_name,
            "Evaluation": "validation_seen_2024",
            "Training_Period": "2020-2023",
            "Fit_Time_Minutes": fit_minutes,
            **evaluate(validation[target].to_numpy(), val_pred),
        }
        metrics.append(result)
        print(
            f"Validation MAE={result['MAE']:.4f} | "
            f"RMSE={result['RMSE']:.4f} | R2={result['R2']:.4f}"
        )

        del model, X_train_gpu, y_train_gpu, X_train_cpu, X_val_cpu
        del y_train_np, val_pred, train_fit
        free_gpu_memory()

        # Phase 2: final fit 2020-2024.
        final_train_fit = select_training_rows(
            final_train,
            RANDOM_SEED + horizon + (1000 if model_name == "A" else 1100),
        )
        X_final_cpu = clean_features(final_train_fit, cols)
        y_final_np = final_train_fit[target].to_numpy(dtype=np.float32, copy=True)

        final_model = make_model(target, model_name)
        start = time.perf_counter()
        X_final_gpu = cp.asarray(X_final_cpu.to_numpy(dtype=np.float32, copy=True))
        y_final_gpu = cp.asarray(y_final_np)
        final_model.fit(X_final_gpu, y_final_gpu)
        cp.cuda.Stream.null.synchronize()
        final_fit_minutes = (time.perf_counter() - start) / 60
        print(f"Final-fit time: {final_fit_minutes:.2f} min")

        model_path = MODEL_DIR / f"cuRF_{model_name}_{target}.joblib"
        joblib.dump(final_model, model_path)
        print(f"Saved: {model_path}")

        del X_final_gpu, y_final_gpu, X_final_cpu, y_final_np, final_train_fit
        free_gpu_memory()

        # Phase 3: evaluate all eligible 2025 rows.
        for split_name, test_df in [
            ("seen_stations_2025", test_seen),
            ("unseen_stations_2025", test_unseen),
        ]:
            X_test_cpu = clean_features(test_df, cols)
            pred = predict_gpu_batched(final_model, X_test_cpu)
            result = {
                "Target": target, "Model": model_name,
                "Evaluation": split_name, "Training_Period": "2020-2024",
                "Fit_Time_Minutes": final_fit_minutes,
                **evaluate(test_df[target].to_numpy(), pred),
            }
            metrics.append(result)
            print(
                f"{split_name} | MAE={result['MAE']:.4f} | "
                f"RMSE={result['RMSE']:.4f} | R2={result['R2']:.4f}"
            )

            output = test_df[[
                "_Timestamp", "Station_ID", "Station_Split",
                "latitude", "longitude", "elevation", target,
            ]].copy()
            output["Model"] = model_name
            output["Training_Period"] = "2020-2024"
            output["Prediction"] = pred
            output["Residual"] = output[target].to_numpy() - pred
            output.to_csv(
                OUTPUT_DIR / f"predictions_{target}.csv",
                mode="a",
                header=not (OUTPUT_DIR / f"predictions_{target}.csv").exists(),
                index=False,
            )
            del X_test_cpu, pred, output

        del final_model, train, validation, final_train, test_seen, test_unseen, model_data
        free_gpu_memory()

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
print("\nFinal metrics:")
print(metrics_df.to_string(index=False))
print(f"\nGPU Random Forest outputs: {OUTPUT_DIR.resolve()}")
