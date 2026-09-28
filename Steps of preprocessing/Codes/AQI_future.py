import pandas as pd
import glob
import os

# ============================================================
# PATH
# ============================================================

base_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data"


# ============================================================
# FIND FILES
# ============================================================

files = glob.glob(
    os.path.join(base_folder, "**", "*_with_AQI_lags_rolling.csv"),
    recursive=True
)

print("Number of files found:", len(files))


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file in files:

    print("\n==========================================")
    print("Processing:", os.path.basename(file))
    print("==========================================")

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Create timestamp
    # --------------------------------------------------------

    df["_timestamp"] = pd.to_datetime(
        df["Day"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Year"].astype(str) + " " +
        df["Time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    # Sort chronologically
    df = df.sort_values("_timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # Create lookup for AQI
    # --------------------------------------------------------

    lookup = pd.Series(
        df["AQI"].values,
        index=df["_timestamp"]
    )

    # --------------------------------------------------------
    # Create future AQI timestamps
    # --------------------------------------------------------

    future_1_timestamp = (
        df["_timestamp"] + pd.Timedelta(hours=1)
    )

    future_24_timestamp = (
        df["_timestamp"] + pd.Timedelta(hours=24)
    )

    # --------------------------------------------------------
    # Create future AQI features
    # --------------------------------------------------------

    df["AQI_future_1"] = future_1_timestamp.map(lookup)

    df["AQI_future_24"] = future_24_timestamp.map(lookup)

    # --------------------------------------------------------
    # Remove temporary timestamp
    # --------------------------------------------------------

    df = df.drop(columns=["_timestamp"])

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = file.replace(
        "_with_AQI_lags_rolling.csv",
        "_with_future_AQI.csv"
    )

    df.to_csv(output_file, index=False)

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("AQI_future_1 NaN:", df["AQI_future_1"].isna().sum())
    print("AQI_future_24 NaN:", df["AQI_future_24"].isna().sum())

    print("Saved to:")
    print(output_file)


print("\n==========================================")
print("FUTURE AQI CREATION COMPLETED")
print("==========================================")