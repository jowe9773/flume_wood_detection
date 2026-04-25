

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/bot-SORT tuning results.csv")

# Filter rows with missing data
df = df.dropna(subset=['raw_precision', 'post_precision', 'raw_recall', 'post_recall'])

segments = df['segment']
x = np.arange(len(segments))
width = 0.35

# Create figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(14,6))

# --- Precision subplot ---
axs[0].bar(x - width/2, df['raw_precision'], width, label='Raw Precision', color='skyblue')
axs[0].bar(x + width/2, df['post_precision'], width, label='Post Precision', color='dodgerblue')
axs[0].set_xticks(x)
axs[0].set_xticklabels(segments, rotation=45, ha='right')
axs[0].set_ylabel('Precision')
axs[0].set_title('Precision: Raw vs Post-correction')
axs[0].set_ylim(0, 1)
axs[0].set_yticks(np.arange(0, 1.01, 0.2))
axs[0].legend()
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# --- Recall subplot ---
axs[1].bar(x - width/2, df['raw_recall'], width, label='Raw Recall', color='lightgreen')
axs[1].bar(x + width/2, df['post_recall'], width, label='Post Recall', color='green')
axs[1].set_xticks(x)
axs[1].set_xticklabels(segments, rotation=45, ha='right')
axs[1].set_ylabel('Recall')
axs[1].set_title('Recall: Raw vs Post-correction')
axs[1].set_ylim(0, 1)
axs[1].set_yticks(np.arange(0, 1.01, 0.2))
axs[1].legend()
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()