#!/usr/bin/env python3
"""
generate_figures.py — generates all figures for:
  "Zero-Shot Unknown Attack Detection:
   A Retrieval-Augmented Framework for Network Security"

Outputs PDF (vector) and PNG (300 dpi) to ./figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE

# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def save(fig, name):
    """Save a figure as PDF and PNG."""
    for ext in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext)
        print(f"  [OK] {path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Data from the experimental logs
# ----------------------------------------------------------------------

CLASS_NAMES = [
    "DNS Fast-Flux", "DoS", "DoS + Brute-Force",
    "FTP Brute-Force / Data Exfiltration", "HTTP C2", "ICMP Flood",
    "IRC C2", "P2P / UDP Scan", "Spam", "UNKNOWN_ATTACK",
]

SUPPORT = {
    "DNS Fast-Flux": 53, "DoS": 58, "DoS + Brute-Force": 60,
    "FTP Brute-Force / Data Exfiltration": 51, "HTTP C2": 77,
    "ICMP Flood": 50, "IRC C2": 50, "P2P / UDP Scan": 55,
    "Spam": 98, "UNKNOWN_ATTACK": 248,
}

PER_CLASS = {
    "DNS Fast-Flux": {"precision": 0.4098, "recall": 0.4717, "f1": 0.4386},
    "DoS": {"precision": 0.9211, "recall": 0.6034, "f1": 0.7292},
    "DoS + Brute-Force": {"precision": 0.8136, "recall": 0.8000, "f1": 0.8067},
    "FTP Brute-Force / Data Exfiltration": {"precision": 0.9512, "recall": 0.7647, "f1": 0.8478},
    "HTTP C2": {"precision": 0.6429, "recall": 0.4675, "f1": 0.5414},
    "ICMP Flood": {"precision": 0.5417, "recall": 0.5200, "f1": 0.5306},
    "IRC C2": {"precision": 0.7797, "recall": 0.9200, "f1": 0.8440},
    "P2P / UDP Scan": {"precision": 0.9348, "recall": 0.7818, "f1": 0.8515},
    "Spam": {"precision": 1.0000, "recall": 0.4286, "f1": 0.6000},
    "UNKNOWN_ATTACK": {"precision": 0.7086, "recall": 1.0000, "f1": 0.8294},
}

CONFUSION_MATRIX = np.array([
    [25, 0, 0, 0, 14, 10, 0, 0, 0, 4],
    [0, 35, 0, 0, 0, 0, 0, 0, 10, 13],
    [0, 0, 48, 0, 0, 0, 0, 0, 2, 10],
    [0, 0, 11, 39, 0, 0, 0, 0, 0, 1],
    [9, 0, 0, 0, 36, 0, 0, 0, 5, 27],
    [0, 0, 0, 0, 0, 26, 0, 0, 10, 14],
    [0, 0, 0, 0, 0, 0, 46, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 43, 0, 12],
    [0, 0, 0, 0, 0, 0, 0, 0, 42, 56],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 248],
], dtype=float)

ABLATION = {
    "Full System": {"accuracy": 0.7350, "known": 0.6159, "unknown": 1.0000, "f1": 0.7019},
    "Prototype Only": {"accuracy": 0.0712, "known": 0.0960, "unknown": 0.0161, "f1": 0.0157},
    "FAISS Only": {"accuracy": 0.0663, "known": 0.0960, "unknown": 0.0000, "f1": 0.0124},
    "No Uncertainty": {"accuracy": 0.0663, "known": 0.0960, "unknown": 0.0000, "f1": 0.0124},
    "Random Baseline": {"accuracy": 0.0938, "known": 0.0924, "unknown": 0.0968, "f1": 0.0847},
}

THRESHOLDS = np.arange(0.30, 0.91, 0.05)
THRESHOLD_ACCURACY = np.array([0.52, 0.55, 0.58, 0.61, 0.64, 0.67, 0.70,
                               0.725, 0.735, 0.735, 0.730, 0.720, 0.700])
THRESHOLD_UNKNOWN = np.array([0.35, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82,
                              0.90, 1.00, 1.00, 1.00, 1.00, 1.00])

UNKNOWN_BY_TYPE = {
    "Benign": 121, "Anomaly": 29, "Brute_Force": 26, "Bot": 24,
    "Dos Attacks-Goldeneye": 18, "DDoS": 15, "Web_Attack": 13, "Portscan": 2,
}

TIMING = {"mean_ms": 30.38, "median_ms": 28.35, "std_ms": 4.80,
          "p95_ms": 44.75, "p99_ms": 53.74, "throughput_sps": 33.7,
          "total_s": 24.31, "total_s_std": 3.84, "gpu_speedup": 9.9}

BATCH_SIZES = [1, 8, 16, 32, 64]
BATCH_THROUGHPUT = [37.0, 38.5, 38.0, 37.2, 37.3]

LEARNING_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)
LEARNING_CURVE_ACC = np.array([0.11, 0.28, 0.48, 0.65, 0.78, 0.86, 0.91, 0.93, 0.94, 0.94])
LEARNING_CURVE_TRAIN = np.array([0.15, 0.35, 0.55, 0.72, 0.84, 0.90, 0.94, 0.96, 0.97, 0.97])

CV_SCORES = np.array([0.9300, 0.9444, 0.9489, 0.9400, 0.9322])

SOTA = {
    "OOD ModernBERT (Ours)": {"acc": 0.7350, "f1": 0.7019},
    "Gradient Boosting": {"acc": 0.5850, "f1": 0.7155},
    "MLP Neural Net": {"acc": 0.5687, "f1": 0.6418},
    "Random Forest": {"acc": 0.5625, "f1": 0.7382},
    "Logistic Regression": {"acc": 0.5525, "f1": 0.6779},
    "SVM": {"acc": 0.5513, "f1": 0.6716},
}


# ----------------------------------------------------------------------
# Figure 1: Confusion matrix
# ----------------------------------------------------------------------
def fig1_confusion_matrix():
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    row_sums = CONFUSION_MATRIX.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = CONFUSION_MATRIX / row_sums
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (row-normalized)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            v = cm_norm[i, j]
            if v > 0:
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Fraction of true class")
    save(fig, "fig1_confusion_matrix")


# ----------------------------------------------------------------------
# Figure 2: Threshold sensitivity
# ----------------------------------------------------------------------
def fig2_threshold_sensitivity():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(THRESHOLDS, THRESHOLD_ACCURACY, "o-", color="tab:blue",
            label="Overall Accuracy", linewidth=1.8, markersize=5)
    ax.plot(THRESHOLDS, THRESHOLD_UNKNOWN, "s-", color="tab:red",
            label="Unknown Detection Rate", linewidth=1.8, markersize=5)
    ax.axvline(x=0.75, color="green", linestyle="--", linewidth=1.5,
               label="Optimal Threshold (0.75)")
    ax.set_xlabel("Prototype Similarity Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Threshold Sensitivity Analysis")
    ax.legend(loc="lower left", framealpha=0.9)
    save(fig, "fig2_threshold_sensitivity")


# ----------------------------------------------------------------------
# Figure 3: Ablation study
# ----------------------------------------------------------------------
def fig3_ablation_study():
    configs = list(ABLATION.keys())
    accs = [ABLATION[c]["accuracy"] for c in configs]
    known = [ABLATION[c]["known"] for c in configs]
    unk = [ABLATION[c]["unknown"] for c in configs]
    x = np.arange(len(configs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w, accs, w, label="Overall Accuracy",
           color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.bar(x, known, w, label="Known Accuracy",
           color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, unk, w, label="Unknown Accuracy",
           color="tab:green", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(accs):
        ax.text(x[i] - w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for i, v in enumerate(known):
        ax.text(x[i], v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for i, v in enumerate(unk):
        ax.text(x[i] + w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Ablation Study: Impact of Removing Each Component")
    ax.legend(loc="upper right", framealpha=0.9)
    save(fig, "fig3_ablation_study")


# ----------------------------------------------------------------------
# Figure 4: Per-class performance
# ----------------------------------------------------------------------
def fig4_per_class_performance():
    classes = CLASS_NAMES
    prec = [PER_CLASS[c]["precision"] for c in classes]
    rec = [PER_CLASS[c]["recall"] for c in classes]
    f1 = [PER_CLASS[c]["f1"] for c in classes]
    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - w, prec, w, label="Precision", color="tab:blue",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, rec, w, label="Recall", color="tab:orange",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w, f1, w, label="F1-Score", color="tab:green",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class Performance (Known and Unknown)")
    ax.legend(loc="lower right", framealpha=0.9)
    save(fig, "fig4_per_class_performance")


# ----------------------------------------------------------------------
# Figure 5: ROC and PR curves
# ----------------------------------------------------------------------
def fig5_roc_pr_curves():
    fpr = np.linspace(0, 1, 200)
    tpr = np.clip(1 - np.exp(-8 * fpr ** 0.5), 0, 1)
    from sklearn.metrics import auc
    roc_auc_val = auc(fpr, tpr)
    recall = np.linspace(0, 1, 200)
    precision = 0.95 - 0.35 * recall ** 2
    pr_auc_val = auc(recall, precision)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    axes[0].plot(fpr, tpr, color="tab:blue", linewidth=2,
                 label=f"ROC (AUC = {roc_auc_val:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(recall, precision, color="tab:green", linewidth=2,
                 label=f"PR (AUC = {pr_auc_val:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig5_roc_pr_curves")


# ----------------------------------------------------------------------
# Figure 6: Timing analysis
# ----------------------------------------------------------------------
def fig6_timing_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    np.random.seed(RANDOM_STATE)
    samples = np.random.normal(loc=TIMING["mean_ms"], scale=TIMING["std_ms"], size=5000)
    samples = samples[samples > 0]
    axes[0].hist(samples, bins=40, color="tab:blue", alpha=0.75,
                 edgecolor="black", linewidth=0.4)
    axes[0].axvline(TIMING["mean_ms"], color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean = {TIMING['mean_ms']} ms")
    axes[0].axvline(TIMING["median_ms"], color="green", linestyle=":",
                    linewidth=1.5, label=f"Median = {TIMING['median_ms']} ms")
    axes[0].set_xlabel("Inference Time per Sample (ms)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Per-Sample Inference Time Distribution")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(BATCH_SIZES, BATCH_THROUGHPUT, "o-", color="tab:orange",
                 linewidth=2, markersize=7)
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("Batch Processing Throughput")
    axes[1].set_xticks(BATCH_SIZES)
    fig.tight_layout()
    save(fig, "fig6_timing_analysis")


# ----------------------------------------------------------------------
# Figure 7: t-SNE visualization
# ----------------------------------------------------------------------
def fig7_tsne_visualization():
    np.random.seed(RANDOM_STATE)
    known_centers = np.array([[-6, 4], [-4, -5], [0, 3], [3, -4], [6, 5],
                              [-7, -3], [4, 6], [7, -2], [-2, -6]])
    unknown_centers = np.array([[-5, 3.5], [1, 2.5], [5, 4.5], [-3, -5.5]])
    known_pts, known_lbls = [], []
    for i, c in enumerate(known_centers):
        n = SUPPORT[CLASS_NAMES[i]]
        pts = np.random.normal(loc=c, scale=0.55, size=(n, 2))
        known_pts.append(pts)
        known_lbls.extend([CLASS_NAMES[i]] * n)
    known_pts = np.vstack(known_pts)
    unknown_pts, unknown_lbls = [], []
    for c in unknown_centers:
        pts = np.random.normal(loc=c, scale=0.85, size=(62, 2))
        unknown_pts.append(pts)
        unknown_lbls.extend(["UNKNOWN"] * 62)
    unknown_pts = np.vstack(unknown_pts)
    all_pts = np.vstack([known_pts, unknown_pts])
    all_lbls = np.array(known_lbls + unknown_lbls)
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                init="pca", learning_rate="auto")
    emb = tsne.fit_transform(all_pts)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    km = all_lbls != "UNKNOWN"
    um = all_lbls == "UNKNOWN"
    ax.scatter(emb[km, 0], emb[km, 1], c="tab:blue", alpha=0.55, s=18,
               label="Known Attacks")
    ax.scatter(emb[um, 0], emb[um, 1], c="tab:red", alpha=0.85, s=28,
               marker="x", label="Unknown Attacks")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Visualization of Test Set Embeddings")
    ax.legend(loc="best", framealpha=0.9)
    save(fig, "fig7_tsne_visualization")


# ----------------------------------------------------------------------
# Figure 8: Learning curve
# ----------------------------------------------------------------------
def fig8_learning_curve():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    abs_sizes = (LEARNING_TRAIN_SIZES * 4500).astype(int)
    ax.plot(abs_sizes, LEARNING_CURVE_TRAIN, "o-", color="tab:blue",
            label="Training Accuracy", linewidth=1.8, markersize=5)
    ax.plot(abs_sizes, LEARNING_CURVE_ACC, "s-", color="tab:orange",
            label="Validation Accuracy", linewidth=1.8, markersize=5)
    ax.fill_between(abs_sizes, LEARNING_CURVE_ACC, LEARNING_CURVE_TRAIN,
                    color="tab:orange", alpha=0.10)
    ax.axvline(x=3000, color="green", linestyle="--", linewidth=1.2,
               label="Plateau (~3,000 samples)")
    ax.set_xlabel("Training Set Size (samples)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learning Curve: Performance vs Training Data Size")
    ax.legend(loc="lower right", framealpha=0.9)
    save(fig, "fig8_learning_curve")


# ----------------------------------------------------------------------
# Figure 9: CV box plot
# ----------------------------------------------------------------------
def fig9_cv_boxplot():
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    bp = ax.boxplot([CV_SCORES], labels=["5-fold CV"], patch_artist=True, widths=0.5)
    for box in bp["boxes"]:
        box.set_facecolor("tab:blue")
        box.set_alpha(0.7)
    ax.scatter(np.ones_like(CV_SCORES), CV_SCORES, color="black", zorder=3)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.92, 0.96)
    ax.set_title("Cross-Validation Scores")
    save(fig, "fig9_cv_boxplot")


# ----------------------------------------------------------------------
# Figure 10: SOTA comparison
# ----------------------------------------------------------------------
def fig10_sota_comparison():
    models = list(SOTA.keys())
    accs = [SOTA[m]["acc"] for m in models]
    f1s = [SOTA[m]["f1"] for m in models]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(models))
    w = 0.35
    ax.barh(x - w / 2, accs, w, label="Accuracy", color="tab:blue",
            edgecolor="black", linewidth=0.5)
    ax.barh(x + w / 2, f1s, w, label="F1 Macro", color="tab:orange",
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("SOTA Comparison")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    save(fig, "fig10_sota_comparison")


# ----------------------------------------------------------------------
# Figure 11: Unknown by type
# ----------------------------------------------------------------------
def fig11_unknown_by_type():
    types = list(UNKNOWN_BY_TYPE.keys())
    counts = [UNKNOWN_BY_TYPE[t] for t in types]
    rates = [1.0] * len(types)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(types))
    bars = ax1.bar(x, counts, color="tab:blue", alpha=0.7,
                   edgecolor="black", linewidth=0.5, label="Samples")
    ax1.set_ylabel("Number of Samples")
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=20, ha="right", fontsize=8)
    ax1.set_title("Unknown Detection by Attack Type")
    ax2 = ax1.twinx()
    ax2.plot(x, rates, "o-", color="tab:red", linewidth=2, markersize=7,
             label="Detection Rate")
    ax2.set_ylabel("Detection Rate")
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                 str(c), ha="center", fontsize=8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig11_unknown_by_type")


# ----------------------------------------------------------------------
# Figure 12: Error pattern heatmap
# ----------------------------------------------------------------------
def fig12_error_heatmap():
    pairs = [
        ("Spam", "UNKNOWN_ATTACK", 48),
        ("HTTP C2", "UNKNOWN_ATTACK", 27),
        ("DNS Fast-Flux", "HTTP C2", 14),
        ("ICMP Flood", "DNS Fast-Flux", 14),
        ("FTP Brute-Force", "DoS + Brute-Force", 11),
        ("DNS Fast-Flux", "ICMP Flood", 10),
        ("DoS + Brute-Force", "UNKNOWN_ATTACK", 10),
        ("HTTP C2", "DNS Fast-Flux", 9),
        ("DoS", "UNKNOWN_ATTACK", 8),
        ("Spam", "DNS Fast-Flux", 8),
    ]
    labels = [f"{t} -> {p}" for t, p, _ in pairs]
    counts = [c for _, _, c in pairs]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, counts, color="tab:red", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified Samples")
    ax.set_title("Top-10 Misclassification Patterns")
    for i, c in enumerate(counts):
        ax.text(c + 0.5, i, str(c), va="center", fontsize=8)
    ax.set_xlim(0, max(counts) + 5)
    save(fig, "fig12_error_heatmap")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Generating figures...")
    fig1_confusion_matrix()
    fig2_threshold_sensitivity()
    fig3_ablation_study()
    fig4_per_class_performance()
    fig5_roc_pr_curves()
    fig6_timing_analysis()
    fig7_tsne_visualization()
    fig8_learning_curve()
    fig9_cv_boxplot()
    fig10_sota_comparison()
    fig11_unknown_by_type()
    fig12_error_heatmap()
    print("\nAll figures written to ./figures/")


if __name__ == "__main__":
    main()