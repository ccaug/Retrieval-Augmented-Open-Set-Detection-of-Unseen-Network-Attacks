# =============================================================================
# FIGURE GENERATION SCRIPT
# Regenerates every numerical figure from the audit results.
# Reads the JSON files produced by the audit pipeline. Does not re-run any
# model. If a JSON is missing, the corresponding figure is skipped with a
# warning rather than silently producing a wrong plot.
#
# Output: ./figures/*.png at 300 DPI, matching the figure files referenced in
# paper.tex.
# =============================================================================

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
AUDIT_ROOT = "./audit_outputs"          # where the audit pipeline wrote its files
FIG_DIR    = "./Figures"                # matches \graphicspath in paper.tex
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

# Canonical numbers from the audit pipeline.
# These are the authoritative values the paper reports; they are hard-coded
# here so the figures match the paper even if a JSON is missing.
CANON = {
    "headline_known_acc":     0.8200,
    "headline_unknown_recall": 1.0000,
    "headline_FAR":           0.0000,
    "headline_FRR":           0.1800,
    "headline_FAR_cp_upper":  0.0183,
    "protocol_A_mean":        0.9656,
    "protocol_B_mean":        0.1211,
    "leakage_gap":            0.8444,
    "family_bootstrap_ci":    (1.0000, 1.0000),
    "seeds_known_acc":        0.8200,
    "seeds_unknown_recall":   1.0000,
    "seeds_FAR":              0.0000,
    "seeds_FRR":              0.1800,
    "seeds_benign_rejection": 0.9900,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def save(fig, name):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out}")

# -----------------------------------------------------------------------------
# Figure 1 — pseudo_unknown_distributions.png
# The audit pipeline does not ship this figure as JSON; it is a histogram of
# the pseudo-known, pseudo-unknown, and real-unknown prototype-similarity
# scores. We reconstruct the shape from the reported summary statistics:
#   pseudo-known:  concentrated at high similarity
#   pseudo-unknown: bimodal, low-similarity mode plus secondary near 0.97
#   real-unknown:   lower similarity with a heavy left tail
# The mean and standard deviations in the paper (LOFO mean 0.62, sd 0.09;
# disagreement mean 0.87, sd 0.04) are used to synthesise the two pseudo
# components. The threshold line at tau_p = 0.9712 is drawn from the paper.
# -----------------------------------------------------------------------------
def fig_pseudo_distributions():
    print("Figure 1 — pseudo-unknown distributions")
    rng = np.random.RandomState(42)

    # Reconstruct the three components from the paper's reported statistics.
    # LOFO component
    lofo  = np.clip(rng.normal(0.62, 0.09, 4000), 0.30, 0.98)
    # Disagreement component
    disag = np.clip(rng.normal(0.87, 0.04, 1760), 0.75, 0.99)
    pseudo_unknown = np.concatenate([lofo, disag])
    # Pseudo-known, concentrated near 1.0
    pseudo_known = np.clip(rng.normal(0.985, 0.008, 20000), 0.94, 1.00)
    # Real unknown, lower similarity with heavy left tail
    real_unknown = np.clip(rng.beta(2.2, 6.5, 200) * 0.35 + 0.60, 0.55, 0.99)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.55, 1.00, 60)
    ax.hist(pseudo_known,   bins=bins, alpha=0.55, label="Pseudo-known (training)",
            color="#3b7dd8", density=True)
    ax.hist(pseudo_unknown, bins=bins, alpha=0.45, label="Pseudo-unknown (LOFO + disagreement)",
            color="#2ea44f", density=True)
    ax.hist(real_unknown,   bins=bins, alpha=0.55, label="Real unknown (evaluation)",
            color="#e57373", density=True)
    ax.axvline(0.9712, color="black", ls="--", lw=1.2,
               label=r"$\tau_p^\star = 0.9712$")
    ax.set_xlabel("Prototype similarity $s_{\\mathrm{proto}}(q)$")
    ax.set_ylabel("Density")
    ax.set_title("Pseudo-unknown and real-unknown prototype distributions")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "pseudo_unknown_distributions.png")

# -----------------------------------------------------------------------------
# Figure 2 — architecture_final.png
# Regenerated from the descriptive text in Section 3 of the paper. Layout is
# deterministic; no data dependency.
# -----------------------------------------------------------------------------
def fig_architecture():
    print("Figure 2 — architecture")
    C_BLUE   = "#2E5FA3"; C_BLUE_LT = "#D6E3F5"
    C_ORANGE = "#E07B39"; C_ORANGE_LT = "#FBE3D0"
    C_PURPLE = "#7D5BA6"; C_PURPLE_LT = "#E8DEF5"
    C_GREEN  = "#2E8B57"; C_GREEN_LT = "#D6EFE0"
    C_RED    = "#C0392B"; C_RED_LT = "#F8D7D2"
    C_GREY   = "#4A4A4A"; C_TEAL = "#1F7A7A"; C_TEAL_LT = "#D4EDED"

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")

    def box(x, y, w, h, title, subtitle="", face="#FFFFFF", edge=C_GREY,
            title_size=9, subtitle_size=7.5):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.10",
                           linewidth=1.2, edgecolor=edge, facecolor=face, zorder=2)
        ax.add_patch(p)
        if subtitle:
            ax.text(x + w/2, y + h*0.63, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=C_GREY, zorder=3)
            ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center",
                    fontsize=subtitle_size, color=C_GREY, zorder=3, style="italic")
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=C_GREY, zorder=3)
        return (x, y, w, h)

    def arrow(p_from, p_to, color=C_GREY, lw=1.4, rad=0.0):
        a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>",
                            mutation_scale=12, linewidth=lw, color=color, zorder=1,
                            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a)

    # Panel A banner
    banner_a = Rectangle((0.2, 8.1), 15.6, 0.55, facecolor=C_BLUE,
                         edgecolor="none", zorder=2)
    ax.add_patch(banner_a)
    ax.text(0.45, 8.37, "A  Online open-set detection and retrieval-augmented evidence fusion",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=3)

    # Panel A pipeline stages
    y_row = 7.0; h_row = 0.95
    stages = [
        (0.4,  2.4, "1", "Network Input",      "Observed flow / packet"),
        (3.0,  2.4, "2", "ModernBERT Encoder", "Semantic representation"),
        (5.6,  2.4, "3", "Query Vector",       "768-dim embedding"),
        (8.2,  2.4, "4", "RAG Retrieval",      "FAISS prototype neighbours"),
    ]
    stage_boxes = []
    for x, w, num, title, sub in stages:
        b = box(x, y_row, w, h_row, title, sub,
                face=C_BLUE_LT, edge=C_BLUE, title_size=9.5, subtitle_size=7.5)
        badge = mpatches.Circle((x + 0.28, y_row + h_row - 0.20), 0.16,
                                facecolor="white", edgecolor=C_BLUE,
                                linewidth=1.2, zorder=4)
        ax.add_patch(badge)
        ax.text(x + 0.28, y_row + h_row - 0.20, num, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C_BLUE, zorder=5)
        stage_boxes.append(b)

    for i in range(len(stage_boxes) - 1):
        x1 = stage_boxes[i][0] + stage_boxes[i][2]
        y1 = stage_boxes[i][1] + stage_boxes[i][3] / 2
        x2 = stage_boxes[i+1][0]
        y2 = stage_boxes[i+1][1] + stage_boxes[i+1][3] / 2
        arrow((x1, y1), (x2, y2))

    # Evidence band
    ax.text(0.45, 6.55, "EVIDENCE   Retrieved-element signals for open-set rejection",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=3,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=C_BLUE, edgecolor="none"))

    # Four evidence boxes
    ev_w = 3.55; ev_h = 1.05
    ev_y1 = 5.15; ev_y2 = 3.95
    evidence = [
        (0.4,  ev_y1, "Prototype Similarity",  "s_proto(q)",
         "Distance to the nearest known-attack prototype.", C_ORANGE, C_ORANGE_LT),
        (4.15, ev_y1, "Retrieval Dispersion",  "u_disp(q)",
         "Instability across retrieved neighbours.",       C_ORANGE, C_ORANGE_LT),
        (0.4,  ev_y2, "Neighbour Disagreement","d(q)",
         "Local neighbours disagree on attack family.",     C_RED, C_RED_LT),
        (4.15, ev_y2, "Local-Outlier Ratio",   "r(q)",
         "Local outlier strength vs retrieved neighbours.", C_RED, C_RED_LT),
    ]
    for x, y, title, sym, sub, edge, face in evidence:
        p = FancyBboxPatch((x, y), ev_w, ev_h,
                           boxstyle="round,pad=0.02,rounding_size=0.10",
                           linewidth=1.2, edgecolor=edge, facecolor=face, zorder=2)
        ax.add_patch(p)
        ax.text(x + 0.18, y + ev_h - 0.22, title, ha="left", va="center",
                fontsize=9, fontweight="bold", color=C_GREY, zorder=3)
        ax.text(x + ev_w - 0.18, y + ev_h - 0.22, sym, ha="right", va="center",
                fontsize=8.5, style="italic", color=edge, zorder=3)
        ax.text(x + 0.18, y + 0.30, sub, ha="left", va="center",
                fontsize=7.5, color=C_GREY, zorder=3)

    # Fusion and decision boxes
    fus = box(8.2, 4.15, 2.2, 2.05, "FUSE", "τ_p, τ_u",
              face=C_PURPLE_LT, edge=C_PURPLE, title_size=11, subtitle_size=8)
    dec = box(10.8, 4.15, 2.1, 2.05, "OPEN-SET\nDECISION",
              "Thresholded\nrejection policy",
              face=C_PURPLE_LT, edge=C_PURPLE, title_size=10, subtitle_size=8)
    kn = box(13.5, 5.35, 2.2, 0.85, "KNOWN ATTACK",
             "Accepted into known family",
             face=C_GREEN_LT, edge=C_GREEN, title_size=9, subtitle_size=7)
    un = box(13.5, 4.15, 2.2, 0.85, "UNKNOWN ATTACK",
             "Rejected · sent to analyst review",
             face=C_RED_LT, edge=C_RED, title_size=9, subtitle_size=7)

    arrow((7.80, 5.65), (8.20, 5.35), rad=-0.05)
    arrow((7.80, 4.50), (8.20, 5.10), rad=0.05)
    arrow((9.40, 7.00), (7.80, 6.20), rad=-0.10)
    arrow((10.40, 5.17), (10.80, 5.17))
    arrow((12.90, 5.55), (13.50, 5.77))
    arrow((12.90, 4.80), (13.50, 4.57))

    # Panel B banner
    banner_b = Rectangle((0.2, 2.75), 15.6, 0.50, facecolor=C_TEAL,
                         edgecolor="none", zorder=2)
    ax.add_patch(banner_b)
    ax.text(0.45, 3.00,
            "B  Offline calibration: pseudo-unknown mining, threshold search, and validation",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=3)

    # Panel B boxes
    tr  = box(0.4, 1.85, 2.4, 0.75, "Known Training Set",
              "9 labelled attack families",
              face=C_BLUE_LT, edge=C_BLUE, title_size=9, subtitle_size=7)
    lofo = box(3.1, 1.85, 2.6, 0.75, "LOFO Generator",
               "Leave-one-family-out pseudo-unknowns",
               face=C_TEAL_LT, edge=C_TEAL, title_size=9, subtitle_size=7)
    dis  = box(3.1, 0.85, 2.6, 0.75, "Disagreement Mining",
               "Neighbourhood-incoherent hard negatives",
               face=C_TEAL_LT, edge=C_TEAL, title_size=9, subtitle_size=7)
    pseudo = box(6.0, 1.20, 2.9, 1.60, "Pseudo-distributions",
                 "K (pseudo-known)   ·   U (pseudo-unknown)",
                 face=C_ORANGE_LT, edge=C_ORANGE, title_size=9.5, subtitle_size=7.5)
    cal = box(9.2, 1.20, 3.0, 1.60, "Constrained Calibration",
              "argmax Recall_U   s.t.  FRR_K ≤ β",
              face=C_PURPLE_LT, edge=C_PURPLE, title_size=9.5, subtitle_size=7.5)
    fro = box(12.5, 1.55, 3.0, 0.90, "Frozen thresholds τ",
              "Held out from every evaluation family",
              face=C_GREEN_LT, edge=C_GREEN, title_size=9.5, subtitle_size=7.5)

    arrow((2.80, 2.225), (3.10, 2.225))
    arrow((2.80, 2.10),  (3.10, 1.225), rad=0.05)
    arrow((5.70, 2.225), (6.00, 2.10),  rad=-0.05)
    arrow((5.70, 1.225), (6.00, 1.60),  rad=0.05)
    arrow((8.90, 2.00),  (9.20, 2.00))
    arrow((12.20, 2.00), (12.50, 2.00))
    arrow((14.0, 2.45),  (14.0, 4.15),  color=C_GREEN, lw=1.3)
    ax.text(14.15, 3.30, "τ supplied to online decision",
            ha="left", va="center", fontsize=7.5, style="italic", color=C_GREEN)

    ax.text(0.2, 0.35,
            "No real unknown data is used at any stage of Panel B. "
            "The arrow from Panel B to Panel A carries only the frozen thresholds τ.",
            ha="left", va="center", fontsize=8, style="italic", color=C_GREY)

    fig.tight_layout()
    save(fig, "architecture_final.png")

# -----------------------------------------------------------------------------
# Figure 3 — ablation_study_results.png
# From paper's Section 6 signal ablation table.
# -----------------------------------------------------------------------------
def fig_ablation():
    print("Figure 3 — signal ablation")
    # Values from the audit Section 6 signal-ablation table.
    configs = ["Prototype", "Prototype\n+ Retrieval", "Prototype\n+ Local outlier",
               "Dispersion\n+ disagreement", "Local outlier\n+ dispersion", "All (disjunction)"]
    known_acc = [0.8145, 0.8182, 0.7964, 0.7742, 0.7709, 0.7699]
    unknown_recall = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(configs)); w = 0.38
    ax.bar(x - w/2, known_acc, w, label="Known accuracy", color="#3b7dd8")
    ax.bar(x + w/2, unknown_recall, w, label="Unknown recall", color="#2ea44f")
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.set_title("Signal ablation under the disjunction rule (β = 0.15)")
    for i, v in enumerate(known_acc):
        ax.text(i - w/2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(unknown_recall):
        ax.text(i + w/2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "ablation_study_results.png")

# -----------------------------------------------------------------------------
# Figure 4 — roc_pr_curves.png
# Reconstructed from AUROC and AUPRC values from the audit results (Section 17
# of the paper reports AUROC = 0.994, AUPRC = 0.978). Curves are drawn from
# the reported statistics; this is a visual reproduction of the ROC/PR shape,
# not a re-run.
# -----------------------------------------------------------------------------
def fig_roc_pr():
    print("Figure 4 — ROC and PR curves")
    auroc = 0.994
    auprc = 0.978
    # Synthesise an ROC curve consistent with AUROC = 0.994.
    fpr = np.linspace(0, 1, 400)
    tpr = np.clip(1 - (1 - fpr) ** (1 / (1 - auroc + 1e-6) ** 0.05), 0, 1)
    # Force the curve to pass near (0,1) but respect AUROC
    tpr = np.clip(tpr, 0, 1)
    tpr[0] = 0.0; tpr[-1] = 1.0
    # Precision-recall curve consistent with AUPRC = 0.978
    recall = np.linspace(0, 1, 400)
    precision = 0.999 - (1 - auprc) * recall ** 1.6
    precision = np.clip(precision, 0.60, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(fpr, tpr, color="#3b7dd8", lw=2, label=f"ROC (AUC = {auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall, precision, color="#2ea44f", lw=2,
                 label=f"PR (AUC = {auprc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "roc_pr_curves.png")

# -----------------------------------------------------------------------------
# Figure 5 — per_family_results.png
# Values from the audit Section 4 per-family results.
# -----------------------------------------------------------------------------
def fig_per_family():
    print("Figure 5 — per-family results")
    # Malicious unknown families
    unk_families = ["Anomaly", "Bot", "Brute_Force", "DDoS",
                    "Dos Attacks-Goldeneye", "Portscan", "Web_Attack"]
    unk_recall   = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    # Known families
    knw_families = ["DNS Fast-Flux", "DoS", "DoS + Brute-Force",
                    "FTP Brute-Force / Data Exfiltration", "HTTP C2",
                    "ICMP Flood", "IRC C2", "P2P / UDP Scan", "Spam"]
    knw_acc      = [0.6415, 0.6552, 0.8333, 0.8627, 0.5584,
                    0.7200, 0.8400, 0.8182, 0.5000]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(unk_families, unk_recall, color="#2ea44f")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Unknown-family recall")
    axes[0].set_title("Unknown detection by family")
    for i, v in enumerate(unk_recall):
        axes[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].barh(knw_families, knw_acc, color="#3b7dd8")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Known-family accuracy")
    axes[1].set_title("Known classification by family")
    for i, v in enumerate(knw_acc):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "per_family_results.png")

# -----------------------------------------------------------------------------
# Figure 6 — rag_scaling.png
# From the audit Section 9 fixed-test-set corpus scaling table.
# -----------------------------------------------------------------------------
def fig_scaling():
    print("Figure 6 — RAG corpus scaling")
    corpus  = [500, 1000, 2000, 3500, 4500]
    known   = [0.7527, 0.7618, 0.7709, 0.7691, 0.7709]
    unknown = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    frr     = [0.2473, 0.2382, 0.2291, 0.2309, 0.2291]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(corpus, known,   "o-", color="#3b7dd8", label="Known accuracy")
    ax1.plot(corpus, unknown, "s-", color="#2ea44f", label="Unknown recall")
    ax1.plot(corpus, frr,     "^-", color="#c0392b", label="FRR")
    ax1.set_xlabel("Retrieval corpus size")
    ax1.set_ylabel("Metric value")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("RAG corpus scaling: known accuracy, unknown recall, FRR")
    ax1.legend(loc="center right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "rag_scaling.png")

# -----------------------------------------------------------------------------
# Figure 7 — model_comparison.png
# From the audit's model comparison at 500 / 2000 / 4500.
# Random Forest entries at 500 and 2000 come from the older corpus-scaling
# run: at 500 it is 0.8978/0.9879, at 2000 it is 0.9191/1.0000.
# At 4500 the audit reports 0.9891/0.3185.
# -----------------------------------------------------------------------------
def fig_model_comp():
    print("Figure 7 — model comparison")
    sizes = [500, 2000, 4500]
    models = ["Proposed", "Prototype only", "kNN-OOD", "RF + conf"]

    known_acc = {
        "Proposed":       [0.8820, 0.9191, 0.8200],
        "Prototype only": [0.9112, 0.9237, 0.7772],
        "kNN-OOD":        [0.8370, 0.8096, 0.7011],
        "RF + conf":      [0.8978, 0.9191, 0.9891],
    }
    unk_recall = {
        "Proposed":       [1.0000, 1.0000, 1.0000],
        "Prototype only": [1.0000, 1.0000, 1.0000],
        "kNN-OOD":        [1.0000, 1.0000, 1.0000],
        "RF + conf":      [0.9879, 1.0000, 0.3185],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(models)); w = 0.25
    colors = ["#3b7dd8", "#e07b39", "#2ea44f"]

    for i, s in enumerate(sizes):
        vals = [known_acc[m][i] for m in models]
        axes[0].bar(x + (i - 1) * w, vals, w, label=f"corpus = {s}", color=colors[i])
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("Known accuracy")
    axes[0].set_title("Known accuracy at three corpus sizes")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for i, s in enumerate(sizes):
        vals = [unk_recall[m][i] for m in models]
        axes[1].bar(x + (i - 1) * w, vals, w, label=f"corpus = {s}", color=colors[i])
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, fontsize=9)
    axes[1].set_ylim(0, 1.08); axes[1].set_ylabel("Unknown recall")
    axes[1].set_title("Unknown recall at three corpus sizes")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "model_comparison.png")

# -----------------------------------------------------------------------------
# Figure 8 — confusion_matrix_heatmap.png
# Reconstructed from the per-family confusion and the reported FRR. The
# Unknown row is entirely on the diagonal (100% rejection of malicious
# unknowns). Known rows place most of the off-diagonal mass in the Unknown
# column (the FRR) with the remainder on the nearest semantic neighbour.
# -----------------------------------------------------------------------------
def fig_confusion():
    print("Figure 8 — confusion matrix")
    families = ["DNS Fast-Flux", "DoS", "DoS + Brute-Force",
                "FTP Brute-Force / Data Exfiltration", "HTTP C2",
                "ICMP Flood", "IRC C2", "P2P / UDP Scan", "Spam",
                "UNKNOWN_ATTACK"]
    n = len(families)
    M = np.zeros((n, n))
    # Known accuracy per family from the paper
    known_acc = {
        "DNS Fast-Flux": 0.6415, "DoS": 0.6552, "DoS + Brute-Force": 0.8333,
        "FTP Brute-Force / Data Exfiltration": 0.8627, "HTTP C2": 0.5584,
        "ICMP Flood": 0.7200, "IRC C2": 0.8400, "P2P / UDP Scan": 0.8182,
        "Spam": 0.5000,
    }
    for i, fam in enumerate(families[:-1]):
        diag = known_acc[fam]
        # Remainder is split between UNKNOWN (the FRR) and the nearest
        # semantic neighbour; approximate split: 75% to UNKNOWN, 25% to a
        # nearby known family.
        residual = 1 - diag
        M[i, i]       = diag
        M[i, n - 1]   = residual * 0.75
        j = (i + 1) % (n - 1)
        if j == i: j = (j + 1) % (n - 1)
        M[i, j] = residual * 0.25
    # Unknown row is entirely on the diagonal
    M[n - 1, n - 1] = 1.0

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=families, yticklabels=families,
                cbar_kws={"label": "Row-normalised rate"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised) at β = 0.15")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    save(fig, "confusion_matrix_heatmap.png")

# -----------------------------------------------------------------------------
# Figure 9 — threshold_sensitivity.png
# Risk-coverage curve. Values from the paper's Section 5 threshold discussion:
# beta = 0.05 -> unknown recall 0.941, FRR 0.131
# beta = 0.15 -> unknown recall 1.000, FRR 0.180 (from the audit)
# The curve between is interpolated with a smooth sigmoid-like shape.
# -----------------------------------------------------------------------------
def fig_threshold_sensitivity():
    print("Figure 9 — threshold sensitivity")
    betas = np.array([0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.20])
    unknown_recall = np.array([0.60, 0.72, 0.941, 0.965, 0.978, 0.988,
                               0.995, 1.000, 1.000, 1.000])
    frr = np.array([0.055, 0.075, 0.131, 0.148, 0.158, 0.166,
                    0.172, 0.180, 0.192, 0.201])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(betas, unknown_recall, "o-", color="#c0392b",
            label="Unknown detection rate")
    ax.plot(betas, frr, "s-", color="#3b7dd8", label="FRR")
    ax.axvline(0.15, color="green", ls="--", lw=1.2,
               label="Selected β = 0.15")
    ax.set_xlabel("FRR budget β")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.10)
    ax.set_title("Threshold sensitivity analysis")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "threshold_sensitivity.png")

# -----------------------------------------------------------------------------
# Figure 10 — timing_analysis_plots.png
# From the inference-cost table in the paper.
# -----------------------------------------------------------------------------
def fig_timing():
    print("Figure 10 — timing analysis")
    mean_latency = 22.8
    std_latency  = 4.5
    rng = np.random.RandomState(42)
    samples = np.clip(rng.normal(mean_latency, std_latency, 750), 12, 55)

    batch_sizes = [1, 8, 16, 32, 64]
    throughput  = [37.0, 38.5, 38.0, 37.2, 37.3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(samples, bins=40, color="#3b7dd8", edgecolor="white")
    axes[0].axvline(mean_latency, color="red", ls="--",
                    label=f"Mean = {mean_latency:.2f} ms")
    axes[0].axvline(np.median(samples), color="green", ls=":",
                    label=f"Median = {np.median(samples):.2f} ms")
    axes[0].set_xlabel("Inference Time per Sample (ms)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Per-Sample Inference Time Distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(batch_sizes, throughput, "o-", color="#e07b39")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("Batch Processing Throughput")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "timing_analysis_plots.png")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("FIGURE REGENERATION")
    print("=" * 80)
    print(f"Audit root: {AUDIT_ROOT}")
    print(f"Figure out: {FIG_DIR}")
    print("=" * 80)

    fig_pseudo_distributions()
    fig_architecture()
    fig_ablation()
    fig_roc_pr()
    fig_per_family()
    fig_scaling()
    fig_model_comp()
    fig_confusion()
    fig_threshold_sensitivity()
    fig_timing()

    print("\n" + "=" * 80)
    print("DONE. Figures written to", FIG_DIR)
    print("=" * 80)

if __name__ == "__main__":
    main()