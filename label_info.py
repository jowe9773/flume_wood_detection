import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
LABELS_DIR = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_data/400_merged_vert/labels"   
SAVE_DIR = "C:/Users/josie/OneDrive - UCB-O365/Conferences/2026 Hydrosciences Symposium"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 14
})

# -----------------------------
# LOAD LABEL DATA (recursive)
# -----------------------------
classes = []
x_centers = []
y_centers = []
widths = []
heights = []

label_files = glob.glob(os.path.join(LABELS_DIR, "**", "*.txt"), recursive=True)

for file in label_files:
    if os.path.getsize(file) == 0:
        continue

    try:
        data = np.loadtxt(file).reshape(-1, 5)
    except:
        continue

    classes.extend(data[:, 0])
    x_centers.extend(data[:, 1])
    y_centers.extend(data[:, 2])
    widths.extend(data[:, 3])
    heights.extend(data[:, 4])

classes = np.array(classes)
x_centers = np.array(x_centers)
y_centers = np.array(y_centers)
widths = np.array(widths)
heights = np.array(heights)
areas = widths * heights

print(f"Loaded {len(classes)} labels from {len(label_files)} files")
print(f"Total number of bounding boxes: {len(classes)}")

# -----------------------------
# Helper: auto axis limits
# -----------------------------
def auto_limits(data, pad=0.05):
    dmin, dmax = np.min(data), np.max(data)
    span = dmax - dmin
    return dmin - pad * span, dmax + pad * span

# -----------------------------
# Helper: save figure
# -----------------------------
def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{name}.svg", format="svg")
    plt.savefig(f"{SAVE_DIR}/{name}.png", dpi=300)
    plt.close()

# -----------------------------
# 1. CLASS HISTOGRAM
# -----------------------------
plt.figure()
unique, counts = np.unique(classes, return_counts=True)
plt.bar(unique, counts)

plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution")

save_fig("class_distribution")

# -----------------------------
# 2. CENTER LOCATIONS
# -----------------------------
plt.figure()
plt.scatter(x_centers, y_centers, s=5, alpha=0.5)

plt.xlabel("x_center")
plt.ylabel("y_center")
plt.title("Bounding Box Centers")

plt.xlim(*auto_limits(x_centers))
plt.ylim(*auto_limits(y_centers))

save_fig("bbox_centers")

# -----------------------------
# 3. WIDTH vs HEIGHT
# -----------------------------
plt.figure()
plt.scatter(widths, heights, s=5, alpha=0.5)

plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Bounding Box Width vs Height")

plt.xlim(*auto_limits(widths))
plt.ylim(*auto_limits(heights))

save_fig("bbox_wh")

# -----------------------------
# 4. AREA HISTOGRAM
# -----------------------------
plt.figure()
plt.hist(areas, bins=50)

plt.xlabel("Area")
plt.ylabel("Frequency")
plt.title("Bounding Box Area Distribution")

save_fig("bbox_area")

print("Saved SVG + PNG plots to:", SAVE_DIR)