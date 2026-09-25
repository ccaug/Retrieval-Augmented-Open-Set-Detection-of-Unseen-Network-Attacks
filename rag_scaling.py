#!/usr/bin/env python3
"""
Generate rag_scaling.png — RAG corpus scaling figure.

Plots known accuracy, unknown recall, and FRR versus retrieval corpus size,
using the data reported in the "RAG CORPUS SCALING" summary.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------------------
# Data (from the corpus-scaling summary)
# ---------------------------------------------------------------------------
corpus_size = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])

known_accuracy = np.array([
    0.882030, 0.926703, 0.925394, 0.919069, 0.914577,
    0.905945, 0.883376, 0.853612, 0.769928,
])

unknown_recall = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
])

frr = np.array([
    0.117970, 0.073297, 0.074606, 0.080931, 0.085423,
    0.094055, 0.116624, 0.146388, 0.230072,
])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax1 = plt.subplots(figsize=(6.5, 4.2))

# --- Left axis: accuracy / recall -----------------------------------------
color_known = "#1f77b4"      # blue
color_unknown = "#2ca02c"    # green

ln1 = ax1.plot(
    corpus_size, known_accuracy,
    marker="o", markersize=5, linewidth=1.8,
    color=color_known, label="Known accuracy",
)
ln2 = ax1.plot(
    corpus_size, unknown_recall,
    marker="s", markersize=5, linewidth=1.8, linestyle="--",
    color=color_unknown, label="Unknown recall",
)

ax1.set_xlabel("Retrieval corpus size")
ax1.set_ylabel("Accuracy / Recall")
ax1.set_ylim(0.0, 1.05)
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_xticks(corpus_size)
ax1.grid(True, alpha=0.3, linestyle=":")

# Annotate the peak of known accuracy
peak_idx = int(np.argmax(known_accuracy))
ax1.annotate(
    f"peak {known_accuracy[peak_idx]*100:.2f}%",
    xy=(corpus_size[peak_idx], known_accuracy[peak_idx]),
    xytext=(corpus_size[peak_idx] + 300, known_accuracy[peak_idx] + 0.045),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    fontsize=8.5, color="gray",
)

# --- Right axis: FRR ------------------------------------------------------
ax2 = ax1.twinx()
color_frr = "#d62728"        # red

ln3 = ax2.plot(
    corpus_size, frr,
    marker="^", markersize=5, linewidth=1.8, linestyle="-.",
    color=color_frr, label="FRR",
)

ax2.set_ylabel("False-rejection rate (FRR)")
ax2.set_ylim(0.0, 0.30)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

# --- Combined legend ------------------------------------------------------
lines = ln1 + ln2 + ln3
labels = [l.get_label() for l in lines]
ax1.legend(
    lines, labels,
    loc="center right",
    framealpha=0.95,
    edgecolor="gray",
)

ax1.set_title("RAG corpus scaling: known accuracy, unknown recall, FRR")

fig.tight_layout()
fig.savefig("rag_scaling.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved rag_scaling.png")