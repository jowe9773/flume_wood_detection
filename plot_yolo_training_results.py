import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
csv_path = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/yolo26n/YOLOv26n tuning results short.csv"  # <-- CHANGE THIS

df = pd.read_csv(csv_path)

# -----------------------------
# SELECT METRICS
# -----------------------------
metrics = [
    ("mean_MAP50", "std_MAP50", "mAP@50"),
    ("mean_MAP50-95", "std_MAP50-95", "mAP@50-95"),
    ("mean_precision", "std_precision", "Precision"),
    ("mean_recall", "std_recall", "Recall"),
]

runs = df["train_id"].values

# -----------------------------
# PLOTTING
# -----------------------------
plt.rcParams.update({"font.size": 11})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

x = np.arange(len(runs))

for ax, (mean_col, std_col, title) in zip(axes, metrics):
    
    means = df[mean_col].values
    stds = df[std_col].values
    
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4
    )
    
    # Highlight best run
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    # Labels
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    
    # Clean styling
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.35)

# -----------------------------
# SAVE
# -----------------------------
plt.savefig("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/yolo26n/result_images/yolo_tuning_results_short.png", dpi=300, bbox_inches="tight")
plt.show()