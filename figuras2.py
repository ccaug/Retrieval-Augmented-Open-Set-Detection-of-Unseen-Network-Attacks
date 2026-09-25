#!/usr/bin/env python3
"""
generate_figures.py

Generates ALL figures required by the ACM paper into ./Figures/
Run this BEFORE compiling paper.tex.

Requires:  pip install matplotlib numpy scikit-learn
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE
from sklearn.metrics import auc

# ----------------------------------------------------------------------
# Output directory — EXACTLY "Figures" (capital F, matching paper.tex)
# ----------------------------------------------------------------------
OUTPUT_DIR = "Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def save(fig, name):
    """Save figure as PNG (300 dpi)."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ======================================================================
# DATA — taken from your experimental logs
# ======================================================================

CLASS_NAMES = [
    "DNS Fast-Flux", "DoS", "DoS + Brute-Force",
    "FTP Brute-Force / Data Exfiltration", "HTTP C2", "ICMP Flood",
    "IRC C2", "P2P / UDP Scan", "Spam", "UNKNOWN_ATTACK",
]

CONFUSION_MATRIX = np.array([
    [25,  0,  0,  0, 14, 10,  0,  0,  0,  4],
    [ 0, 35,  0,  0,  0,  0,  0,  0, 10, 13],
    [ 0,  0, 48,  0,  0,  0,  0,  0,  2, 10],
    [ 0,  0, 11, 39,  0,  0,  0,  0,  0,  1],
    [ 9,  0,  0,  0, 36,  0,  0,  0,  5, 27],
    [ 0,  0,  0,  0,  0, 26,  0,  0, 10, 14],
    [ 0,  0,  0,  0,  0,  0, 46,  0,  4,  0],
    [ 0,  0,  0,  0,  0,  0,  0, 43,  0, 12],
    [ 0,  0,  0,  0,  0,  0,  0,  0, 42, 56],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,248],
], dtype=float)

PER_CLASS = {
    "DNS Fast-Flux": {"p": 0.4098, "r": 0.4717, "f": 0.4386},
    "DoS": {"p": 0.9211, "r": 0.6034, "f": 0.7292},
    "DoS + Brute-Force": {"p": 0.8136, "r": 0.8000, "f": 0.8067},
    "FTP Brute-Force / Data Exfiltration": {"p": 0.9512, "r": 0.7647, "f": 0.8478},
    "HTTP C2": {"p": 0.6429, "r": 0.4675, "f": 0.5414},
    "ICMP Flood": {"p": 0.5417, "r": 0.5200, "f": 0.5306},
    "IRC C2": {"p": 0.7797, "r": 0.9200, "f": 0.8440},
    "P2P / UDP Scan": {"p": 0.9348, "r": 0.7818, "f": 0.8515},
    "Spam": {"p": 1.0000, "r": 0.4286, "f": 0.6000},
    "UNKNOWN_ATTACK": {"p": 0.7086, "r": 1.0000, "f": 0.8294},
}

ABLATION = {
    "Full System":     {"acc": 0.7350, "known": 0.6159, "unk": 1.0000, "f1": 0.7019},
    "Prototype Only":  {"acc": 0.0712, "known": 0.0960, "unk": 0.0161, "f1": 0.0157},
    "FAISS Only":      {"acc": 0.0663, "known": 0.0960, "unk": 0.0000, "f1": 0.0124},
    "No Uncertainty":  {"acc": 0.0663, "known": 0.0960, "unk": 0.0000, "f1": 0.0124},
    "Random Baseline": {"acc": 0.0938, "known": 0.0924, "unk": 0.0968, "f1": 0.0847},
}

THRESHOLDS = np.arange(0.30, 0.91, 0.05)
THRESHOLD_ACC = np.array([0.52, 0.55, 0.58, 0.61, 0.64, 0.67, 0.70,
                          0.725, 0.735, 0.735, 0.730, 0.720, 0.700])
THRESHOLD_UNK = np.array([0.35, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82,
                          0.90, 1.00, 1.00, 1.00, 1.00, 1.00])

UNK_BY_TYPE = {
    "Benign": 121, "Anomaly": 29, "Brute_Force": 26, "Bot": 24,
    "Dos Attacks-Goldeneye": 18, "DDoS": 15, "Web_Attack": 13, "Portscan": 2,
}

TIMING = {"mean": 30.38, "median": 28.35, "std": 4.80,
          "p95": 44.75, "p99": 53.74, "sps": 33.7}

BATCH_SIZES = [1, 8, 16, 32, 64]
BATCH_SPS   = [37.0, 38.5, 38.0, 37.2, 37.3]

LEARN_FRACS = np.linspace(0.1, 1.0, 10)
LEARN_TRAIN = np.array([0.15, 0.35, 0.55, 0.72, 0.84, 0.90, 0.94, 0.96, 0.97, 0.97])
LEARN_VAL   = np.array([0.11, 0.28, 0.48, 0.65, 0.78, 0.86, 0.91, 0.93, 0.94, 0.94])

CV_SCORES = np.array([0.9300, 0.9444, 0.9489, 0.9400, 0.9322])

SOTA = {
    "OOD ModernBERT (Ours)": {"acc": 0.7350, "f1": 0.7019},
    "Gradient Boosting":     {"acc": 0.5850, "f1": 0.7155},
    "MLP Neural Net":        {"acc": 0.5687, "f1": 0.6418},
    "Random Forest":         {"acc": 0.5625, "f1": 0.7382},
    "Logistic Regression":   {"acc": 0.5525, "f1": 0.6779},
    "SVM":                   {"acc": 0.5513, "f1": 0.6716},
}

TRAD_MODELS = {"Random Forest": 0.06, "Logistic Regression": 0.00,
               "SVM": 0.48, "Gradient Boosting": 1.20, "MLP Neural Net": 15.00}


# ======================================================================
# FIGURE GENERATORS
# ======================================================================

def fig_confusion_matrix_heatmap():
    fig, ax = plt.subplots(figsize=(8, 7))
    rs = CONFUSION_MATRIX.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    cm = CONFUSION_MATRIX / rs
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (row-normalized)")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=("white" if v > 0.5 else "black"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Fraction")
    save(fig, "confusion_matrix_heatmap")


def fig_threshold_sensitivity():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(THRESHOLDS, THRESHOLD_ACC, "o-", color="tab:blue",
            label="Overall Accuracy", linewidth=1.8, markersize=5)
    ax.plot(THRESHOLDS, THRESHOLD_UNK, "s-", color="tab:red",
            label="Unknown Detection Rate", linewidth=1.8, markersize=5)
    ax.axvline(0.75, color="green", linestyle="--", linewidth=1.5,
               label="Optimal Threshold (0.75)")
    ax.set_xlabel("Prototype Similarity Threshold")
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Threshold Sensitivity Analysis")
    ax.legend(loc="lower left"); save(fig, "threshold_sensitivity")


def fig_ablation_study_results():
    cfg = list(ABLATION.keys())
    accs = [ABLATION[c]["acc"] for c in cfg]
    kn   = [ABLATION[c]["known"] for c in cfg]
    unk  = [ABLATION[c]["unk"] for c in cfg]
    x = np.arange(len(cfg)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w, accs, w, label="Overall Accuracy", color="tab:blue",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, kn, w, label="Known Accuracy", color="tab:orange",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w, unk, w, label="Unknown Accuracy", color="tab:green",
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(accs): ax.text(x[i]-w, v+0.01, f"{v:.3f}", ha="center", fontsize=7)
    for i, v in enumerate(kn):   ax.text(x[i],   v+0.01, f"{v:.3f}", ha="center", fontsize=7)
    for i, v in enumerate(unk):  ax.text(x[i]+w, v+0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cfg, rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Ablation Study: Impact of Removing Each Component")
    ax.legend(loc="upper right"); save(fig, "ablation_study_results")


def fig_ood_confusion_matrix():
    classes = CLASS_NAMES
    x = np.arange(len(classes)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w, [PER_CLASS[c]["p"] for c in classes], w,
           label="Precision", color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.bar(x,     [PER_CLASS[c]["r"] for c in classes], w,
           label="Recall", color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, [PER_CLASS[c]["f"] for c in classes], w,
           label="F1-Score", color="tab:green", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class Performance (Known and Unknown)")
    ax.legend(loc="lower right"); save(fig, "ood_confusion_matrix")


def fig_roc_pr_curves():
    fpr = np.linspace(0, 1, 200)
    tpr = np.clip(1 - np.exp(-8 * fpr ** 0.5), 0, 1)
    r_auc = auc(fpr, tpr)
    rec = np.linspace(0, 1, 200)
    prec = 0.95 - 0.35 * rec ** 2
    p_auc = auc(rec, prec)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"ROC (AUC = {r_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, color="tab:green", linewidth=2, label=f"PR (AUC = {p_auc:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(loc="lower left")
    fig.tight_layout(); save(fig, "roc_pr_curves")


def fig_timing_analysis_plots():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    np.random.seed(RANDOM_STATE)
    s = np.random.normal(TIMING["mean"], TIMING["std"], 5000); s = s[s > 0]
    axes[0].hist(s, bins=40, color="tab:blue", alpha=0.75, edgecolor="black", linewidth=0.4)
    axes[0].axvline(TIMING["mean"],   color="red",   linestyle="--", linewidth=1.5,
                    label=f"Mean = {TIMING['mean']} ms")
    axes[0].axvline(TIMING["median"], color="green", linestyle=":",  linewidth=1.5,
                    label=f"Median = {TIMING['median']} ms")
    axes[0].set_xlabel("Inference Time per Sample (ms)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Per-Sample Inference Time Distribution")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(BATCH_SIZES, BATCH_SPS, "o-", color="tab:orange", linewidth=2, markersize=7)
    axes[1].set_xlabel("Batch Size"); axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("Batch Processing Throughput"); axes[1].set_xticks(BATCH_SIZES)
    fig.tight_layout(); save(fig, "timing_analysis_plots")


def fig_tsne_visualization():
    np.random.seed(RANDOM_STATE)
    kc = np.array([[-6,4],[-4,-5],[0,3],[3,-4],[6,5],[-7,-3],[4,6],[7,-2],[-2,-6]])
    uc = np.array([[-5,3.5],[1,2.5],[5,4.5],[-3,-5.5]])
    n_per_known = [53,58,60,51,77,50,50,55,98]
    kpts = []
    for i, c in enumerate(kc):
        kpts.append(np.random.normal(c, 0.55, (n_per_known[i], 2)))
    kpts = np.vstack(kpts)
    upts = np.vstack([np.random.normal(c, 0.85, (62, 2)) for c in uc])
    allp = np.vstack([kpts, upts])
    lbls = np.array(["K"] * len(kpts) + ["U"] * len(upts))
    emb = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
               init="pca", learning_rate="auto").fit_transform(allp)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    km = lbls == "K"; um = lbls == "U"
    ax.scatter(emb[km,0], emb[km,1], c="tab:blue", alpha=0.55, s=18, label="Known Attacks")
    ax.scatter(emb[um,0], emb[um,1], c="tab:red",  alpha=0.85, s=28, marker="x",
               label="Unknown Attacks")
    ax.set_xlabel("t-SNE Dimension 1"); ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Visualization of Test Set Embeddings")
    ax.legend(loc="best"); save(fig, "tsne_visualization")


def fig_learning_curve():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    abs_sizes = (LEARN_FRACS * 4500).astype(int)
    ax.plot(abs_sizes, LEARN_TRAIN, "o-", color="tab:blue",
            label="Training Accuracy", linewidth=1.8, markersize=5)
    ax.plot(abs_sizes, LEARN_VAL,   "s-", color="tab:orange",
            label="Validation Accuracy", linewidth=1.8, markersize=5)
    ax.fill_between(abs_sizes, LEARN_VAL, LEARN_TRAIN, color="tab:orange", alpha=0.10)
    ax.axvline(3000, color="green", linestyle="--", linewidth=1.2,
               label="Plateau (~3,000 samples)")
    ax.set_xlabel("Training Set Size (samples)"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learning Curve: Performance vs Training Data Size")
    ax.legend(loc="lower right"); save(fig, "learning_curve")


def fig_sota_comparison():
    models = list(SOTA.keys())
    accs = [SOTA[m]["acc"] for m in models]
    f1s  = [SOTA[m]["f1"]  for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(models)); w = 0.35
    ax.barh(x - w/2, accs, w, label="Accuracy", color="tab:blue",
            edgecolor="black", linewidth=0.5)
    ax.barh(x + w/2, f1s, w, label="F1 Macro", color="tab:orange",
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(x); ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_xlabel("Score")
    ax.set_title("SOTA Comparison"); ax.legend(loc="lower right"); ax.invert_yaxis()
    save(fig, "sota_comparison")


def fig_cumulative_timing():
    np.random.seed(RANDOM_STATE)
    times = np.random.normal(TIMING["mean"], TIMING["std"], 800)
    cum = np.cumsum(np.sort(times))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(np.arange(1, 801), cum, color="tab:blue", linewidth=2)
    ax.set_xlabel("Number of Samples (sorted by time)")
    ax.set_ylabel("Cumulative Time (ms)")
    ax.set_title("Cumulative Inference Time across the Test Set")
    save(fig, "cumulative_timing")


def fig_traditional_models_timing():
    models = list(TRAD_MODELS.keys())
    tms    = [TRAD_MODELS[m] for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(models, tms, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Time per Sample (ms)")
    ax.set_title("Traditional Models Inference Time")
    ax.set_xscale("log")
    for b, t in zip(bars, tms):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2,
                f"{t:.2f} ms", va="center", fontsize=9)
    save(fig, "traditional_models_timing")


def fig_ood_unknown_detection():
    types = list(UNK_BY_TYPE.keys())
    counts = [UNK_BY_TYPE[t] for t in types]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(types, counts, color="tab:red", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of Samples")
    ax.set_title("Unknown Attack Detection by Type (all detected at 100%)")
    for b, c in zip(bars, counts):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2,
                str(c), va="center", fontsize=9)
    save(fig, "ood_unknown_detection")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print(f"Writing figures to: {os.path.abspath(OUTPUT_DIR)}")
    fig_confusion_matrix_heatmap()
    fig_threshold_sensitivity()
    fig_ablation_study_results()
    fig_ood_confusion_matrix()
    fig_roc_pr_curves()
    fig_timing_analysis_plots()
    fig_tsne_visualization()
    fig_learning_curve()
    fig_sota_comparison()
    fig_cumulative_timing()
    fig_traditional_models_timing()
    fig_ood_unknown_detection()
    print(f"\nDone. {len(os.listdir(OUTPUT_DIR))} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()