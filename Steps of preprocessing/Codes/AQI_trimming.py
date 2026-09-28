import pandas as pd
import glob
import os

# ============================================================
# PATH
# ============================================================

base_folder = r"C:\Users\prave\Desktop\college projects\AQI-Prediction\Imputed_Data"


# ============================================================
# FIND ALL CSV FILES
# ============================================================

files = glob.glob(
    os.path.join(base_folder, "**", "*_with_future_AQI.csv"),
    recursive=True
)

print("Number of files found:", len(files))


# ============================================================
# PROCESS EACH CITY FILE
# ============================================================

for file in files:

    print("\n==========================================")
    print("Processing:", os.path.basename(file))
    print("==========================================")

    df = pd.read_csv(file)

    original_rows = len(df)

    # --------------------------------------------------------
    # Create timestamp
    # --------------------------------------------------------

    if "Timestamp" in df.columns:

        df["_temp_timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

    else:

        df["_temp_timestamp"] = pd.to_datetime(
            df["Day"].astype(str) + "-" +
            df["Month"].astype(str) + "-" +
            df["Year"].astype(str) + " " +
            df["Time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )


    # --------------------------------------------------------
    # Remove first 24 hours of 2020
    # --------------------------------------------------------

    remove_2020 = (
        (df["_temp_timestamp"] >= "2020-01-01 00:00:00") &
        (df["_temp_timestamp"] <  "2020-01-02 00:00:00")
    )


    # --------------------------------------------------------
    # Remove last 24 hours of 2024
    # --------------------------------------------------------

    remove_2024 = (
        (df["_temp_timestamp"] >= "2024-12-31 00:00:00") &
        (df["_temp_timestamp"] <  "2025-01-01 00:00:00")
    )


    # --------------------------------------------------------
    # Remove last 24 hours of 2025
    # --------------------------------------------------------

    remove_2025 = (
        (df["_temp_timestamp"] >= "2025-12-31 00:00:00") &
        (df["_temp_timestamp"] <  "2026-01-01 00:00:00")
    )


    # --------------------------------------------------------
    # Combine all removal conditions
    # --------------------------------------------------------

    rows_to_remove = remove_2020 | remove_2024 | remove_2025

    removed_count = rows_to_remove.sum()

    # Keep everything else
    df = df.loc[~rows_to_remove].copy()


    # --------------------------------------------------------
    # Remove temporary timestamp
    # --------------------------------------------------------

    df.drop(columns=["_temp_timestamp"], inplace=True)


    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df.reset_index(drop=True, inplace=True)


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = file.replace(
            "_with_future_AQI.csv",
            "_with_future_AQI_trimmed.csv"
        )
    
    df.to_csv(
            output_file,
            index=False
        )
    


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("Original rows :", original_rows)
    print("Rows removed  :", removed_count)
    print("Final rows    :", len(df))

    print("\nRemoved:")
    print("2020-01-01 →", remove_2020.sum(), "rows")
    print("2024-12-31 →", remove_2024.sum(), "rows")
    print("2025-12-31 →", remove_2025.sum(), "rows")


print("\n==========================================")
print("DATE TRIMMING COMPLETED")
print("==========================================")