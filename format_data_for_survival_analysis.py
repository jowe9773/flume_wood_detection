import geopandas as gpd
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
RFSD = "2_0"

INPUT_SHP = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/QGIS/first5_{RFSD}_noshort.shp"
OUTPUT_CSV = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/data_for_survival_analysis/first5_{RFSD}_survival_dataset.csv"

ID_COL = "track_id"
X_THRESHOLD = 9000  # censoring boundary

# OPTIONAL: define deposition rule
# If you already have a column, set EVENT_COL = "deposited"
EVENT_COL = None

# -----------------------------
# LOAD DATA
# -----------------------------
gdf = gpd.read_file(INPUT_SHP)

# Extract x coordinate
gdf["x"] = gdf.geometry.x

gdf = gdf.sort_values([ID_COL, "x"])

# -----------------------------
# BUILD SURVIVAL DATASET
# -----------------------------
rows = []

for wood_id, group in gdf.groupby(ID_COL):

    group = group.sort_values("x")

    x_last = group["x"].iloc[-1]
    class_name = group["class_name"].iloc[-1]

    # -------------------------
    # EVENT DEFINITION
    # -------------------------
    if EVENT_COL and EVENT_COL in group.columns:
        event = bool(group[EVENT_COL].max())

    else:
        # Example assumption:
        # event happens if trajectory actually passes a "deposition condition"
        # YOU MUST CUSTOMIZE THIS if you have a better rule
        event = x_last < X_THRESHOLD

    # -------------------------
    # CENSORING RULE
    # -------------------------
    # If it reaches or exceeds experiment end → censored
    if x_last >= X_THRESHOLD:
        event = False

    # -------------------------
    # TIME VARIABLE
    # -------------------------
    time = min(x_last, X_THRESHOLD)

    rows.append({
        "track_id": wood_id,
        "max_x": time,
        "deposited": event,
        "class_name": class_name
    })

df = pd.DataFrame(rows)

df["deposited"] = df["deposited"].astype(bool)
df["max_x"] = df["max_x"].astype(float)

df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print(df)