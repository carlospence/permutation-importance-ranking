import os
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================================
# Configuration
# ============================================================================

PHASE1_SUMMARY_DIR = "results/phase1"
PHASE2_SUMMARY_DIR = "results/phase2"

PHASE1_TEST_RESULTS = "results/final_test/phase1/all_phase1_final_test_results.csv"
PHASE2_TEST_RESULTS = "results/final_test/phase2/all_phase2_final_test_results.csv"

OUTPUT_DIR = "report_figures"
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test_results")
CONFUSION_DIR = os.path.join(TEST_OUTPUT_DIR, "confusion_matrices")

CLASSIFIERS_SUMMARY = ["SVM", "KNN", "DT", "RF", "MLP"]
CLASSIFIERS_TEST = ["SVM", "kNN", "DT", "RF", "MLP"]
GROUP_ORDER = ["Datasets 1-8", "Datasets 9-16"]
N_DATASETS = 16

# Bigger figures without losing clarity:
# - increase FIG_SCALE for larger dimensions
# - keep SAVE_DPI at 300 or higher for print quality
# - SAVE_PDF=True also exports vector PDF versions where supported
FIG_SCALE = 1.35
SAVE_DPI = 300
SAVE_PDF = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)


# ============================================================================
# Utility helpers
# ============================================================================

def scaled_size(width: float, height: float) -> Tuple[float, float]:
    return width * FIG_SCALE, height * FIG_SCALE


def save_figure(fig: plt.Figure, png_path: str) -> None:
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    if SAVE_PDF:
        pdf_path = os.path.splitext(png_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def dataset_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Could not extract dataset number from: {value}")
    return int(match.group(1))


def normalize_test_classifier(value: str) -> str:
    c = str(value).strip().upper()
    if c in {"KNN", "K-NN", "K_NN", "K NEAREST"}:
        return "kNN"
    mapping = {
        "SVM": "SVM",
        "DT": "DT",
        "RF": "RF",
        "MLP": "MLP",
    }
    return mapping.get(c, c.title())


# ============================================================================
# Phase summary loaders
# ============================================================================

def load_phase1_summary(clf: str, dataset_num: int) -> Optional[pd.DataFrame]:
    path = os.path.join(
        PHASE1_SUMMARY_DIR,
        clf.lower(),
        f"dataset_{dataset_num}_{clf.lower()}_summary.csv",
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_summary(clf: str, dataset_num: int) -> Optional[pd.DataFrame]:
    path = os.path.join(
        PHASE2_SUMMARY_DIR,
        clf.lower(),
        f"dataset_{dataset_num}_{clf.lower()}_p2_summary.csv",
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_folds(dataset_num: int) -> Optional[pd.DataFrame]:
    path = os.path.join(PHASE2_SUMMARY_DIR, f"dataset_{dataset_num}_all_p2_folds.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ============================================================================
# Final test loaders
# ============================================================================

def load_test_results(path: str, phase_label: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"Skipping missing results file: {path}")
        return None

    df = pd.read_csv(path)
    required = {"Dataset", "Classifier", "Test_Accuracy", "Test_F1_Macro"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["Dataset_Num"] = df["Dataset"].apply(dataset_number)
    df["Dataset_Group"] = df["Dataset_Num"].apply(
        lambda n: "Datasets 1-8" if 1 <= n <= 8 else "Datasets 9-16"
    )
    df["Phase_Label"] = phase_label
    df["Classifier_Norm"] = df["Classifier"].apply(normalize_test_classifier)
    return df


# ============================================================================
# Figure 1: Phase 1 vs Phase 2 summary F1 by classifier and dataset group
# ============================================================================

def figure_phase_comparison() -> None:
    groups = {
        "Datasets 1-8 (4 classes, moderate imbalance)": range(1, 9),
        "Datasets 9-16 (16 classes, severe imbalance)": range(9, 17),
    }

    fig, axes = plt.subplots(1, 2, figsize=scaled_size(16, 6))

    colors = {"Before FS/DR": "#4472C4", "After FS/DR": "#ED7D31"}
    x = np.arange(len(CLASSIFIERS_SUMMARY))
    width = 0.35

    for ax, (group_title, ds_range) in zip(axes, groups.items()):
        p1_means, p2_means = [], []
        p1_stds, p2_stds = [], []

        for clf in CLASSIFIERS_SUMMARY:
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

        bars1 = ax.bar(
            x - width / 2,
            p1_means,
            width,
            yerr=p1_stds,
            label="Before FS/DR",
            color=colors["Before FS/DR"],
            capsize=4,
            edgecolor="white",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            p2_means,
            width,
            yerr=p2_stds,
            label="After FS/DR",
            color=colors["After FS/DR"],
            capsize=4,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title(group_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSIFIERS_SUMMARY, fontsize=11)
        ax.set_ylabel("Mean F1 Score", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

        for bar_group in (bars1, bars2):
            for rect in bar_group:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    fig.suptitle(
        "Phase 1 vs Phase 2 F1 Score by Classifier and Dataset Group",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "phase_comparison.png"))


# ============================================================================
# Figure 2: Average Phase 1 F1 per classifier
# ============================================================================

def figure_best_classifier() -> None:
    clf_means = []
    clf_stds = []

    for clf in CLASSIFIERS_SUMMARY:
        all_f1 = []
        for ds in range(1, N_DATASETS + 1):
            s = load_phase1_summary(clf, ds)
            if s is not None:
                all_f1.append(s["F1 Mean"].values[0])
        clf_means.append(np.mean(all_f1) if all_f1 else 0)
        clf_stds.append(np.std(all_f1, ddof=1) if len(all_f1) > 1 else 0)

    fig, ax = plt.subplots(figsize=scaled_size(8, 5))
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"]

    bars = ax.bar(
        CLASSIFIERS_SUMMARY,
        clf_means,
        yerr=clf_stds,
        color=colors,
        capsize=5,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Mean F1 Score (across all 16 datasets)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.tick_params(axis="x", labelsize=11)

    for bar, mean in zip(bars, clf_means):
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    best_idx = int(np.argmax(clf_means))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    ax.set_title(
        "Average F1 Score per Classifier - Phase 1 (Baseline)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "best_classifier.png"))


# ============================================================================
# Figure 3: Feature count reduction per dataset
# ============================================================================

def figure_feature_reduction() -> None:
    datasets = []
    features_before = []
    features_after = []

    for ds in range(1, N_DATASETS + 1):
        folds_df = load_phase2_folds(ds)
        if folds_df is None:
            continue

        svm_folds = folds_df[folds_df["Classifier"] == "SVM"]
        if svm_folds.empty:
            continue

        before = svm_folds["N_Features_Before"].mean()
        after = svm_folds["N_Features_After"].mean()

        datasets.append(f"Data {ds}")
        features_before.append(before)
        features_after.append(after)

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=scaled_size(16, 6))

    bars1 = ax.bar(
        x - width / 2,
        features_before,
        width,
        label="Before FS/DR",
        color="#4472C4",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        features_after,
        width,
        label="After FS/DR",
        color="#ED7D31",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of Features", fontsize=11)
    ax.set_title(
        "Feature Count Before and After Feature Selection per Dataset",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

    for rect in bars2:
        height = rect.get_height()
        ax.annotate(
            f"{int(round(height))}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "feature_reduction.png"))


# ============================================================================
# Figure 4: F1 standard deviation comparison
# ============================================================================

def figure_std_comparison() -> None:
    groups = {
        "Datasets 1-8 (4 classes, moderate imbalance)": range(1, 9),
        "Datasets 9-16 (16 classes, severe imbalance)": range(9, 17),
    }

    fig, axes = plt.subplots(1, 2, figsize=scaled_size(16, 6))
    colors = {"Before FS/DR": "#4472C4", "After FS/DR": "#ED7D31"}
    x = np.arange(len(CLASSIFIERS_SUMMARY))
    width = 0.35

    for ax, (group_title, ds_range) in zip(axes, groups.items()):
        p1_stds, p2_stds = [], []

        for clf in CLASSIFIERS_SUMMARY:
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

        bars1 = ax.bar(
            x - width / 2,
            p1_stds,
            width,
            label="Before FS/DR",
            color=colors["Before FS/DR"],
            edgecolor="white",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            p2_stds,
            width,
            label="After FS/DR",
            color=colors["After FS/DR"],
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title(group_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSIFIERS_SUMMARY, fontsize=11)
        ax.set_ylabel("Mean F1 Standard Deviation", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

        for bar_group in (bars1, bars2):
            for rect in bar_group:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    fig.suptitle(
        "F1 Standard Deviation per Classifier - Phase 1 vs Phase 2",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "std_comparison.png"))


# ============================================================================
# Figure 5: Combined average test macro-F1 per classifier
# ============================================================================

def figure_combined_average_test_macro_f1(all_df: pd.DataFrame) -> None:
    summary = (
        all_df.groupby(["Phase_Label", "Classifier_Norm"], as_index=False)["Test_F1_Macro"]
        .mean()
        .rename(columns={"Test_F1_Macro": "Avg_Macro_F1"})
    )

    phase_order = list(summary["Phase_Label"].drop_duplicates())
    pivot = summary.pivot(index="Classifier_Norm", columns="Phase_Label", values="Avg_Macro_F1")
    pivot = pivot.reindex(CLASSIFIERS_TEST)

    fig, ax = plt.subplots(figsize=scaled_size(10, 5))
    x = list(range(len(CLASSIFIERS_TEST)))
    width = 0.35

    if len(phase_order) == 1:
        phase = phase_order[0]
        values = pivot[phase].fillna(0).tolist()
        bars = ax.bar(x, values, width=0.5, label=phase)
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        left_phase, right_phase = phase_order[0], phase_order[1]
        left_vals = pivot[left_phase].fillna(0).tolist()
        right_vals = pivot[right_phase].fillna(0).tolist()

        bars1 = ax.bar([i - width / 2 for i in x], left_vals, width=width, label=left_phase)
        bars2 = ax.bar([i + width / 2 for i in x], right_vals, width=width, label=right_phase)

        for bars in (bars1, bars2):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title("Average Test Macro-F1 per Classifier (Phase 1 vs Phase 2)")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Average Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSIFIERS_TEST)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, os.path.join(TEST_OUTPUT_DIR, "combined_average_test_macro_f1.png"))


# ============================================================================
# Figure 6: Test macro-F1 by classifier and dataset group
# ============================================================================

def figure_combined_group_macro_f1(all_df: pd.DataFrame) -> None:
    summary = (
        all_df.groupby(["Phase_Label", "Classifier_Norm", "Dataset_Group"], as_index=False)["Test_F1_Macro"]
        .mean()
        .rename(columns={"Test_F1_Macro": "Avg_Macro_F1"})
    )

    phases = list(summary["Phase_Label"].drop_duplicates())
    fig, axes = plt.subplots(1, len(phases), figsize=scaled_size(6 * len(phases), 5), squeeze=False)

    for idx, phase in enumerate(phases):
        phase_df = summary[summary["Phase_Label"] == phase]
        pivot = phase_df.pivot(index="Classifier_Norm", columns="Dataset_Group", values="Avg_Macro_F1")
        pivot = pivot.reindex(CLASSIFIERS_TEST)

        ax = axes[0][idx]
        x = list(range(len(CLASSIFIERS_TEST)))
        width = 0.35

        vals_1 = pivot[GROUP_ORDER[0]].fillna(0).tolist() if GROUP_ORDER[0] in pivot.columns else [0] * len(CLASSIFIERS_TEST)
        vals_2 = pivot[GROUP_ORDER[1]].fillna(0).tolist() if GROUP_ORDER[1] in pivot.columns else [0] * len(CLASSIFIERS_TEST)

        bars1 = ax.bar([i - width / 2 for i in x], vals_1, width=width, label=GROUP_ORDER[0])
        bars2 = ax.bar([i + width / 2 for i in x], vals_2, width=width, label=GROUP_ORDER[1])

        for bars in (bars1, bars2):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_title(phase)
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Average Macro-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSIFIERS_TEST)
        ax.set_ylim(0, 1.05)
        ax.legend()

    fig.suptitle("Test Macro-F1 by Classifier and Dataset Group")
    fig.tight_layout()
    save_figure(fig, os.path.join(TEST_OUTPUT_DIR, "combined_test_macro_f1_by_dataset_group.png"))


# ============================================================================
# Figure 7: Per-dataset test macro-F1 line chart
# ============================================================================

def figure_line_chart(p1_df: pd.DataFrame, p2_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=scaled_size(18, 6), sharey=True)

    colors = {
        "SVM": "#E74C3C",
        "kNN": "#3498DB",
        "DT": "#2ECC71",
        "RF": "#F39C12",
        "MLP": "#9B59B6",
    }

    datasets = list(range(1, N_DATASETS + 1))

    for ax, (df, title) in zip(
        axes,
        [
            (p1_df, "Phase 1 - Test Macro-F1 per Dataset"),
            (p2_df, "Phase 2 - Test Macro-F1 per Dataset"),
        ],
    ):
        for clf in CLASSIFIERS_TEST:
            clf_df = df[df["Classifier_Norm"] == clf]
            f1_vals = []
            for ds in datasets:
                row = clf_df[clf_df["Dataset_Num"] == ds]
                f1_vals.append(float(row["Test_F1_Macro"].values[0]) if not row.empty else np.nan)

            ax.plot(
                datasets,
                f1_vals,
                marker="o",
                markersize=4,
                label=clf,
                color=colors[clf],
                linewidth=1.8,
            )

        ax.axvline(x=8.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(4.5, 0.02, "Datasets 1-8", ha="center", fontsize=9, color="gray")
        ax.text(12.5, 0.02, "Datasets 9-16", ha="center", fontsize=9, color="gray")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Test Macro-F1", fontsize=11)
        ax.set_xticks(datasets)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

    fig.suptitle("Per-Dataset Test Macro-F1 by Classifier", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "figure_line_chart.png"))


# ============================================================================
# Figure 8: Heatmaps for Phase 1 and Phase 2 test macro-F1
# ============================================================================

def figure_heatmaps(p1_df: pd.DataFrame, p2_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=scaled_size(16, 8))
    ds_labels = [f"Data {i}" for i in range(1, N_DATASETS + 1)]

    for ax, (df, title) in zip(
        axes,
        [
            (p1_df, "Phase 1 - Test Macro-F1"),
            (p2_df, "Phase 2 - Test Macro-F1"),
        ],
    ):
        matrix = np.full((N_DATASETS, len(CLASSIFIERS_TEST)), np.nan)

        for j, clf in enumerate(CLASSIFIERS_TEST):
            clf_df = df[df["Classifier_Norm"] == clf]
            for ds in range(1, N_DATASETS + 1):
                row = clf_df[clf_df["Dataset_Num"] == ds]
                if not row.empty:
                    matrix[ds - 1, j] = float(row["Test_F1_Macro"].values[0])

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CLASSIFIERS_TEST)))
        ax.set_xticklabels(CLASSIFIERS_TEST, fontsize=10)
        ax.set_yticks(range(N_DATASETS))
        ax.set_yticklabels(ds_labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        for i in range(N_DATASETS):
            for j in range(len(CLASSIFIERS_TEST)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.3 or val > 0.75 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold",
                    )

        ax.axhline(y=7.5, color="white", linewidth=2.5)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Macro-F1")

    fig.suptitle("Test Macro-F1 Heatmap - Phase 1 vs Phase 2", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "figure_heatmaps.png"))


# ============================================================================
# Figure 9+: Confusion matrices
# ============================================================================

def plot_confusion_matrix_from_csv(csv_path: str, title: str, output_filename: str) -> None:
    if not os.path.exists(csv_path):
        print(f"Skipping missing confusion matrix file: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=scaled_size(7, 6))

    im = ax.imshow(df.values, cmap="Blues", interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=45)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(i) for i in df.index])

    max_value = df.values.max() if df.values.size else 0
    threshold = max_value / 2 if max_value > 0 else 0

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iat[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    save_figure(fig, os.path.join(CONFUSION_DIR, output_filename))


def build_confusion_targets() -> List[Tuple[str, str, str]]:
    return [
        (
            "results/final_test/phase1/dataset_1_rf_confusion_matrix.csv",
            "Confusion Matrix - Random Forest - Dataset 1 (Phase 1)",
            "phase1_rf_dataset1_confusion_matrix.png",
        ),
        (
            "results/final_test/phase1/dataset_15_rf_confusion_matrix.csv",
            "Confusion Matrix - Random Forest - Dataset 15 (Phase 1)",
            "phase1_rf_dataset15_confusion_matrix.png",
        ),
        (
            "results/final_test/phase2/dataset_1_rf_confusion_matrix.csv",
            "Confusion Matrix - Random Forest - Dataset 1 (Phase 2)",
            "phase2_rf_dataset1_confusion_matrix.png",
        ),
        (
            "results/final_test/phase2/dataset_15_rf_confusion_matrix.csv",
            "Confusion Matrix - Random Forest - Dataset 15 (Phase 2)",
            "phase2_rf_dataset15_confusion_matrix.png",
        ),
    ]


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Generating summary-based figures...")
    figure_phase_comparison()
    figure_best_classifier()
    figure_feature_reduction()
    figure_std_comparison()

    print("Generating test-result figures...")
    phase1_df = load_test_results(PHASE1_TEST_RESULTS, "Phase 1")
    phase2_df = load_test_results(PHASE2_TEST_RESULTS, "Phase 2")

    available_frames = [df for df in (phase1_df, phase2_df) if df is not None]
    if not available_frames:
        raise FileNotFoundError("No Phase 1 or Phase 2 test result files were found.")

    all_df = pd.concat(available_frames, ignore_index=True)

    figure_combined_average_test_macro_f1(all_df)
    figure_combined_group_macro_f1(all_df)

    if phase1_df is not None and phase2_df is not None:
        figure_line_chart(phase1_df, phase2_df)
        figure_heatmaps(phase1_df, phase2_df)
    else:
        print("Skipping line chart and heatmaps because both phase files are required.")

    print("Generating confusion matrices...")
    for csv_path, title, output_filename in build_confusion_targets():
        plot_confusion_matrix_from_csv(csv_path, title, output_filename)

    print("\nDone.")
    print(f"Main figures: {OUTPUT_DIR}")
    print(f"Test figures: {TEST_OUTPUT_DIR}")
    print(f"Confusion matrices: {CONFUSION_DIR}")


if __name__ == "__main__":
    main()
