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
    os.path.join(base_folder, "**", "*_with_future_AQI_trimmed.csv"),
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

    # Find all roll_std_1 columns
    columns_to_drop = [
        col for col in df.columns
        if col.endswith("_roll_std_1")
    ]

    if columns_to_drop:

        print("Columns being removed:")
        for col in columns_to_drop:
            print(" -", col)

        # Drop the columns
        df = df.drop(columns=columns_to_drop)

        # Save back to the same file
        output_file = file.replace(
                    "_with_future_AQI_trimmed.csv",
                    "_with_future_AQI_trimmed_2.csv"
                )
            
        df.to_csv(
                    output_file,
                    index=False
                )
            

        print("Removed:", len(columns_to_drop), "columns")
        print("New shape:", df.shape)

    else:
        print("No _roll_std_1 columns found.")


print("\n==========================================")
print("COMPLETED")
print("==========================================")