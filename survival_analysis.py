import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

# -----------------------------
# DEFINE YOUR DATASETS
# -----------------------------
datasets = {
    "0_25": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/data_for_survival_analysis/first5_0_25_survival_dataset.csv",
    "0_5": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/data_for_survival_analysis/first5_0_5_survival_dataset.csv",
    "1_0": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/data_for_survival_analysis/first5_1_0_survival_dataset.csv",
}

# -----------------------------
# SETUP FIGURE
# -----------------------------
plt.figure(figsize=(8, 5))

all_dfs = []

# -----------------------------
# LOOP THROUGH DATASETS
# -----------------------------
for label, path in datasets.items():
    df = pd.read_csv(path)

    df["max_x"] = df["max_x"].astype(float)
    df["deposited"] = df["deposited"].astype(bool)
    df["group"] = label

    all_dfs.append(df)

    y = Surv.from_arrays(
        event=df["deposited"].values,
        time=df["max_x"].values
    )

    # KM with confidence intervals
    time, survival_prob, conf_int = kaplan_meier_estimator(
        y["event"],
        y["time"],
        conf_type="log-log"   # robust CI
    )

    # Plot main curve
    plt.step(time, survival_prob, where="post", label=label)

    # Plot confidence interval shading
    plt.fill_between(
        time,
        conf_int[0],
        conf_int[1],
        step="post",
        alpha=0.2
    )

# -----------------------------
# LOG-RANK TEST (GLOBAL)
# -----------------------------
df_all = pd.concat(all_dfs)

y_all = Surv.from_arrays(
    event=df_all["deposited"],
    time=df_all["max_x"]
)

result = compare_survival(y_all, df_all["group"])
stat, p_value = compare_survival(y_all, df_all["group"])

# -----------------------------
# ANNOTATE P-VALUE
# -----------------------------
plt.text(
    0.05, 0.05,
    f"log-rank p = {p_value:.3e}",
    transform=plt.gca().transAxes
)

# -----------------------------
# FORMATTING
# -----------------------------
plt.xlabel("Travel distance (mm)")
plt.ylabel("Survival probability")
plt.title("Kaplan–Meier Survival Curves with Confidence Intervals")
plt.grid(True)
plt.legend()

plt.show()