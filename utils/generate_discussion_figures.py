"""
Generates 3 figures for the Discussion section:
  Phase 1 vs Phase 2 F1 per classifier (grouped bar) - split by dataset group
  Average F1 per classifier across all datasets Phase 1 only
  Feature reduction per dataset (before vs after)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = "report_figures"
PHASE1_DIR = "results/phase1"
PHASE2_DIR = "results/phase2"
CLASSIFIERS = ["SVM", "KNN", "DT", "RF", "MLP"]
N_DATASETS  = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load summary data ──────────────────────────────────────────────────────────

def load_phase1_summary(clf, dataset_num):
    path = os.path.join(PHASE1_DIR, clf.lower(), f"dataset_{dataset_num}_{clf.lower()}_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_summary(clf, dataset_num):
    path = os.path.join(PHASE2_DIR, clf.lower(), f"dataset_{dataset_num}_{clf.lower()}_p2_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_folds(dataset_num):
    path = os.path.join(PHASE2_DIR, f"dataset_{dataset_num}_all_p2_folds.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Phase 1 vs Phase 2 F1 per classifier ────────────────────────────

def figure3_phase_comparison():
    groups = {
        "Datasets 1-8 (4 classes, moderate imbalance)":  range(1, 9),
        "Datasets 9-16 (16 classes, severe imbalance)": range(9, 17),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"Before FS/DR": "#4472C4", "After FS/DR": "#ED7D31"}
    x      = np.arange(len(CLASSIFIERS))
    width  = 0.35

    for ax, (group_title, ds_range) in zip(axes, groups.items()):
        p1_means, p2_means = [], []
        p1_stds,  p2_stds  = [], []

        for clf in CLASSIFIERS:
            f1_p1, f1_p2 = [], []
            for ds in ds_range:
                s1 = load_phase1_summary(clf, ds)
                s2 = load_phase2_summary(clf, ds)
                if s1 is not None:
                    f1_p1.append(s1["F1 Mean"].values[0])
                if s2 is not None:
                    f1_p2.append(s2["F1 Mean"].values[0])

            p1_means.append(np.mean(f1_p1) if f1_p1 else 0)
            p2_means.append(np.mean(f1_p2) if f1_p2 else 0)
            p1_stds.append(np.std(f1_p1, ddof=1) if len(f1_p1) > 1 else 0)
            p2_stds.append(np.std(f1_p2, ddof=1) if len(f1_p2) > 1 else 0)

        b1 = ax.bar(x - width/2, p1_means, width, yerr=p1_stds,
                    label="Before FS/DR", color=colors["Before FS/DR"],
                    capsize=4, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + width/2, p2_means, width, yerr=p2_stds,
                    label="After FS/DR", color=colors["After FS/DR"],
                    capsize=4, edgecolor="white", linewidth=0.5)

        ax.set_title(group_title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSIFIERS, fontsize=10)
        ax.set_ylabel("Mean F1 Score", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

        # Value labels on bars
        for bar in [b1, b2]:
            for rect in bar:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5)

    fig.suptitle("Phase 1 vs Phase 2 F1 Score by Classifier and Dataset Group",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "phase_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: phase_comparison.png")


# ── Average F1 per classifier Phase 1 ───────────────────────────────

def figure4_best_classifier():
    clf_means = []
    clf_stds  = []

    for clf in CLASSIFIERS:
        all_f1 = []
        for ds in range(1, N_DATASETS + 1):
            s = load_phase1_summary(clf, ds)
            if s is not None:
                all_f1.append(s["F1 Mean"].values[0])
        clf_means.append(np.mean(all_f1) if all_f1 else 0)
        clf_stds.append(np.std(all_f1, ddof=1) if len(all_f1) > 1 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"]
    bars   = ax.bar(CLASSIFIERS, clf_means, yerr=clf_stds,
                    color=colors, capsize=5,
                    edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Mean F1 Score (across all 16 datasets)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.tick_params(axis="x", labelsize=11)

    for bar, mean, std in zip(bars, clf_means, clf_stds):
        ax.annotate(f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width()/2, mean),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold")

    # Highlight best
    best_idx = int(np.argmax(clf_means))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    ax.set_title("Average F1 Score per Classifier - Phase 1 (Baseline)",
                 fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_classifier.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: best_classifier.png")


# ── Feature reduction per dataset ───────────────────────────────────

def figure5_feature_reduction():
    datasets     = []
    features_before = []
    features_after  = []

    for ds in range(1, N_DATASETS + 1):
        folds_df = load_phase2_folds(ds)
        if folds_df is None:
            continue

        svm_folds = folds_df[folds_df["Classifier"] == "SVM"]
        if svm_folds.empty:
            continue

        before = svm_folds["N_Features_Before"].mean()
        after  = svm_folds["N_Features_After"].mean()

        datasets.append(f"Data {ds}")
        features_before.append(before)
        features_after.append(after)

    x     = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))

    b1 = ax.bar(x - width/2, features_before, width,
                label="Before FS/DR", color="#4472C4",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, features_after, width,
                label="After FS/DR", color="#ED7D31",
                edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.set_title("Feature Count Before and After Feature Selection per Dataset",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

    # Annotate after bars with exact count
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f"{int(round(h))}",
            xy=(rect.get_x() + rect.get_width()/2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_reduction.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: feature_reduction.png")


# ── Run all ────────────────────────────────────────────────────────────────────


# -- Standard deviation comparison Phase 1 vs Phase 2 ---------------

def figure6_std_comparison():
    groups = {
        "Datasets 1-8 (4 classes, moderate imbalance)": range(1, 9),
        "Datasets 9-16 (16 classes, severe imbalance)": range(9, 17),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"Before FS/DR": "#4472C4", "After FS/DR": "#ED7D31"}
    x     = np.arange(len(CLASSIFIERS))
    width = 0.35

    for ax, (group_title, ds_range) in zip(axes, groups.items()):
        p1_stds, p2_stds = [], []

        for clf in CLASSIFIERS:
            std_p1, std_p2 = [], []
            for ds in ds_range:
                s1 = load_phase1_summary(clf, ds)
                s2 = load_phase2_summary(clf, ds)
                if s1 is not None:
                    std_p1.append(s1["F1 Std"].values[0])
                if s2 is not None:
                    std_p2.append(s2["F1 Std"].values[0])

            p1_stds.append(np.mean(std_p1) if std_p1 else 0)
            p2_stds.append(np.mean(std_p2) if std_p2 else 0)

        b1 = ax.bar(x - width/2, p1_stds, width,
                    label="Before FS/DR", color=colors["Before FS/DR"],
                    edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + width/2, p2_stds, width,
                    label="After FS/DR", color=colors["After FS/DR"],
                    edgecolor="white", linewidth=0.5)

        ax.set_title(group_title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSIFIERS, fontsize=10)
        ax.set_ylabel("Mean F1 Standard Deviation", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

        for bar in [b1, b2]:
            for rect in bar:
                h = rect.get_height()
                ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5)

    fig.suptitle("F1 Standard Deviation per Classifier - Phase 1 vs Phase 2",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "std_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: std_comparison.png")

if __name__ == "__main__":
    figure3_phase_comparison()
    figure4_best_classifier()
    figure5_feature_reduction()
    figure6_std_comparison()
    print("\nAll figures saved to report_figures/")