import pandas as pd

# ============================================================
# FILE PATH
# ============================================================

details_file = "Datasets/gap_analysis_details_5_cities.csv"

# Read detailed gap analysis
details = pd.read_csv(details_file)

print("Original shape:", details.shape)


# ============================================================
# CONVERT DATES
# ============================================================

details["Gap_Start"] = pd.to_datetime(
    details["Gap_Start"],
    errors="coerce"
)

details["Gap_End"] = pd.to_datetime(
    details["Gap_End"],
    errors="coerce"
)


# ============================================================
# SELECT ONLY LONG GAPS (>24 HOURS)
# ============================================================

long_gaps = details[
    details["Gap_Hours"] >= 4
].copy()


# ============================================================
# CLASSIFY LONG GAPS
# ============================================================

def classify_gap(hours):

    if 4 <= hours <= 72:
        return "Moderate (4-72 hours)"

    elif hours > 72:
        return "Long (>72 hours)"


long_gaps["Long_Gap_Class"] = (
    long_gaps["Gap_Hours"]
    .apply(classify_gap)
)


# ============================================================
# PROPOSED IMPUTATION ACTION
# ============================================================

long_gaps["Proposed_Action"] = long_gaps[
    "Long_Gap_Class"
].map({
    "Moderate (4-72 hours)": "Contextual median",
    "Long (>72 hours)": "Leave / Drop"
})


# ============================================================
# EXTRACT YEAR
# ============================================================

long_gaps["Gap_Year"] = (
    long_gaps["Gap_Start"].dt.year
)


# ============================================================
# CREATE SEPARATE SHEETS
# ============================================================

all_long_gaps = long_gaps.copy()

moderate_4_to_72 = long_gaps[
    long_gaps["Long_Gap_Class"] == "Moderate (4-72 hours)"
].copy()

long_above_72 = long_gaps[
    long_gaps["Long_Gap_Class"] == "Long (>72 hours)"
].copy()


# ============================================================
# STATION + VARIABLE SUMMARY
# ============================================================

station_variable_summary = (
    long_gaps
    .groupby(
        [
            "City",
            "Station",
            "Variable",
            "Long_Gap_Class"
        ]
    )
    .agg(
        Number_of_Gaps=("Gap_Hours", "count"),
        Total_Missing_Hours=("Gap_Hours", "sum"),
        Maximum_Gap_Hours=("Gap_Hours", "max"),
        Average_Gap_Hours=("Gap_Hours", "mean")
    )
    .reset_index()
)

station_variable_summary["Average_Gap_Hours"] = (
    station_variable_summary["Average_Gap_Hours"]
    .round(2)
)


# ============================================================
# CITY SUMMARY
# ============================================================

city_summary = (
    long_gaps
    .groupby(
        [
            "City",
            "Long_Gap_Class"
        ]
    )
    .agg(
        Number_of_Gaps=("Gap_Hours", "count"),
        Total_Missing_Hours=("Gap_Hours", "sum"),
        Maximum_Gap_Hours=("Gap_Hours", "max")
    )
    .reset_index()
)


# ============================================================
# TOP 30 LONGEST GAPS
# ============================================================

top_30_longest = (
    long_gaps
    .sort_values(
        "Gap_Hours",
        ascending=False
    )
    .head(30)
    .copy()
)


# ============================================================
# OUTPUT EXCEL FILE
# ============================================================

output_file = (
    "Datasets/Long_Gap_Analysis_5_Cities.xlsx"
)


with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    all_long_gaps.to_excel(
        writer,
        sheet_name="All_Long_Gaps",
        index=False
    )
    moderate_4_to_72.to_excel(
        writer,
        sheet_name="Moderate_4_to_72h",
        index=False
    )

    long_above_72.to_excel(
        writer,
        sheet_name="Long_Above_72h",
        index=False
    )
    station_variable_summary.to_excel(
        writer,
        sheet_name="Station_Variable_Summary",
        index=False
    )

    city_summary.to_excel(
        writer,
        sheet_name="City_Summary",
        index=False
    )

    top_30_longest.to_excel(
        writer,
        sheet_name="Top_30_Longest",
        index=False
    )


print("\n==========================================")
print("LONG GAP ANALYSIS COMPLETED")
print("==========================================")

print("Total gaps >=4 hours:", len(all_long_gaps))

print("\nGap classification:")
print(
    all_long_gaps["Long_Gap_Class"]
    .value_counts()
)

print("\nExcel file saved at:")
print(output_file)