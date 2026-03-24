"""
Generates 3 additional discussion figures from test result CSVs:
  figure_line_chart.png  — per-dataset test F1 line chart Phase 1 vs Phase 2
  figure_heatmaps.png    — side by side heatmap Phase 1 and Phase 2
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR  = "report_figures"
P1_CSV      = "results/final_test/phase1/all_phase1_final_test_results.csv"
P2_CSV      = "results/final_test/phase2/all_phase2_final_test_results.csv"
N_DATASETS  = 16
CLASSIFIERS = ["SVM", "kNN", "DT", "RF", "MLP"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load & normalise ───────────────────────────────────────────────────────────

def load_and_normalise(path):
    df = pd.read_csv(path)

    # Extract dataset number from "dataset_1" etc.
    df["Dataset_Num"] = df["Dataset"].str.extract(r"(\d+)").astype(int)

    # Normalise classifier names to match CLASSIFIERS list
    def norm_clf(c):
        c = str(c).strip().upper()
        if c in ["KNN", "K-NN", "K_NN", "K NEAREST"]:
            return "kNN"
        mapping = {"SVM": "SVM", "DT": "DT", "RF": "RF", "MLP": "MLP"}
        return mapping.get(c, c.title())

    df["Classifier_Norm"] = df["Classifier"].apply(norm_clf)
    return df


# ── Figure: Line chart ─────────────────────────────────────────────────────────

def figure_line_chart(p1_df, p2_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    colors = {
        "SVM": "#E74C3C",
        "kNN": "#3498DB",
        "DT":  "#2ECC71",
        "RF":  "#F39C12",
        "MLP": "#9B59B6",
    }

    datasets = list(range(1, N_DATASETS + 1))

    for ax, (df, title) in zip(axes, [
        (p1_df, "Phase 1 - Test Macro-F1 per Dataset"),
        (p2_df, "Phase 2 - Test Macro-F1 per Dataset"),
    ]):
        for clf in CLASSIFIERS:
            clf_df  = df[df["Classifier_Norm"] == clf]
            f1_vals = []
            for ds in datasets:
                row = clf_df[clf_df["Dataset_Num"] == ds]
                f1_vals.append(float(row["Test_F1_Macro"].values[0]) if not row.empty else np.nan)

            ax.plot(datasets, f1_vals, marker="o", markersize=4,
                    label=clf, color=colors[clf], linewidth=1.8)

        # Divider between groups
        ax.axvline(x=8.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(4.5,  0.02, "Datasets 1-8",  ha="center", fontsize=9, color="gray")
        ax.text(12.5, 0.02, "Datasets 9-16", ha="center", fontsize=9, color="gray")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Dataset", fontsize=10)
        ax.set_ylabel("Test Macro-F1", fontsize=10)
        ax.set_xticks(datasets)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

    fig.suptitle("Per-Dataset Test Macro-F1 by Classifier",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure_line_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure: Heatmaps ──────────────────────────────────────────────────────────

def figure_heatmaps(p1_df, p2_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ds_labels = [f"Data {i}" for i in range(1, N_DATASETS + 1)]

    for ax, (df, title) in zip(axes, [
        (p1_df, "Phase 1 - Test Macro-F1"),
        (p2_df, "Phase 2 - Test Macro-F1"),
    ]):
        # Build matrix: rows = datasets, cols = classifiers
        matrix = np.full((N_DATASETS, len(CLASSIFIERS)), np.nan)

        for i, clf in enumerate(CLASSIFIERS):
            clf_df = df[df["Classifier_Norm"] == clf]
            for ds in range(1, N_DATASETS + 1):
                row = clf_df[clf_df["Dataset_Num"] == ds]
                if not row.empty:
                    matrix[ds - 1, i] = float(row["Test_F1_Macro"].values[0])

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(CLASSIFIERS)))
        ax.set_xticklabels(CLASSIFIERS, fontsize=10)
        ax.set_yticks(range(N_DATASETS))
        ax.set_yticklabels(ds_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        # Annotate each cell
        for i in range(N_DATASETS):
            for j in range(len(CLASSIFIERS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.3 or val > 0.75 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, color=text_color, fontweight="bold")

        # White divider between dataset groups
        ax.axhline(y=7.5, color="white", linewidth=2.5)

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Macro-F1")

    fig.suptitle("Test Macro-F1 Heatmap - Phase 1 vs Phase 2",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "figure_heatmaps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    for path in [P1_CSV, P2_CSV]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

    p1_df = load_and_normalise(P1_CSV)
    p2_df = load_and_normalise(P2_CSV)

    print(f"Phase 1: {len(p1_df)} rows | Classifiers: {p1_df['Classifier_Norm'].unique()}")
    print(f"Phase 2: {len(p2_df)} rows | Classifiers: {p2_df['Classifier_Norm'].unique()}")

    figure_line_chart(p1_df, p2_df)
    figure_heatmaps(p1_df, p2_df)

    print("\nAll figures saved to report_figures/")


if __name__ == "__main__":
    main()