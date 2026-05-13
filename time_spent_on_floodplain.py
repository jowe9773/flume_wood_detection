import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
import itertools

def get_ratios(shapefile):
    gdf = gpd.read_file(shapefile)
    gdf["location"] = gdf["location"].str.lower().str.strip()

    total = gdf.groupby("track_id").size()
    flood = gdf[gdf["location"] == "floodplain"].groupby("track_id").size()

    df = pd.DataFrame({
        "total": total,
        "flood": flood
    }).fillna(0)

    return df["flood"] / df["total"]


# -----------------------------
# COMPUTE RATIOS FOR EACH CASE
# -----------------------------
ratios1 = get_ratios("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/QGIS/first5_0_25_noshort.shp")
ratios2 = get_ratios("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/QGIS/first5_0_5_noshort.shp")
ratios3 = get_ratios("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/QGIS/first5_1_0_noshort.shp")

# global test
stat, p = kruskal(ratios1, ratios2, ratios3)
print("Kruskal-Wallis p-value:", p)

# -----------------------------
# POST-HOC PAIRWISE TESTS
# -----------------------------
pairs = [("0.25", ratios1), ("0.5", ratios2), ("1.0", ratios3)]

for (name1, r1), (name2, r2) in itertools.combinations(pairs, 2):
    stat, p = mannwhitneyu(r1, r2, alternative="two-sided")
    print(f"{name1} vs {name2}: p = {p}")

# -----------------------------
# PLOT: THREE BOXPLOTS
# -----------------------------
plt.figure()

plt.boxplot([ratios1, ratios2, ratios3])

plt.xticks([1, 2, 3], ["Case 1", "Case 2", "Case 3"])
plt.ylabel("Floodplain Ratio")
plt.title("Floodplain Occupancy Comparison")

plt.show()


