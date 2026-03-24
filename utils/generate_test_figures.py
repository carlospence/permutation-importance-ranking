import os
import re
from typing import Optional, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


PHASE1_RESULTS = "results/final_test/phase1/all_phase1_final_test_results.csv"
PHASE2_RESULTS = "results/final_test/phase2/all_phase2_final_test_results.csv"

OUTPUT_DIR = "report_figures/test_results"
CONFUSION_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)

CLASSIFIER_ORDER = ["SVM", "kNN", "DT", "RF", "MLP"]
GROUP_ORDER = ["Datasets 1-8", "Datasets 9-16"]


def dataset_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Could not extract dataset number from: {value}")
    return int(match.group(1))


def load_results(path: str, phase_label: str) -> Optional[pd.DataFrame]:
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
    return df


def plot_combined_average_macro_f1(all_df: pd.DataFrame) -> None:
    summary = (
        all_df.groupby(["Phase_Label", "Classifier"], as_index=False)["Test_F1_Macro"]
        .mean()
        .rename(columns={"Test_F1_Macro": "Avg_Macro_F1"})
    )

    phase_order = list(summary["Phase_Label"].drop_duplicates())

    pivot = summary.pivot(
        index="Classifier",
        columns="Phase_Label",
        values="Avg_Macro_F1",
    ).reindex(CLASSIFIER_ORDER)

    plt.figure(figsize=(10, 5))
    x = list(range(len(CLASSIFIER_ORDER)))
    width = 0.35

    if len(phase_order) == 1:
        phase = phase_order[0]
        values = pivot[phase].fillna(0).tolist()
        bars = plt.bar(x, values, width=0.5, label=phase)
        for bar in bars:
            value = bar.get_height()
            plt.text(
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

        bars1 = plt.bar([i - width / 2 for i in x], left_vals, width=width, label=left_phase)
        bars2 = plt.bar([i + width / 2 for i in x], right_vals, width=width, label=right_phase)

        for bars in (bars1, bars2):
            for bar in bars:
                value = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.title("Average Test Macro-F1 per Classifier (Phase 1 vs Phase 2)")
    plt.xlabel("Classifier")
    plt.ylabel("Average Macro-F1")
    plt.xticks(x, CLASSIFIER_ORDER)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "combined_average_test_macro_f1.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_combined_group_macro_f1(all_df: pd.DataFrame) -> None:
    summary = (
        all_df.groupby(["Phase_Label", "Classifier", "Dataset_Group"], as_index=False)["Test_F1_Macro"]
        .mean()
        .rename(columns={"Test_F1_Macro": "Avg_Macro_F1"})
    )

    phases = list(summary["Phase_Label"].drop_duplicates())
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 5), squeeze=False)

    for idx, phase in enumerate(phases):
        phase_df = summary[summary["Phase_Label"] == phase]
        pivot = phase_df.pivot(
            index="Classifier",
            columns="Dataset_Group",
            values="Avg_Macro_F1",
        ).reindex(CLASSIFIER_ORDER)

        ax = axes[0][idx]
        x = list(range(len(CLASSIFIER_ORDER)))
        width = 0.35

        vals_1 = pivot[GROUP_ORDER[0]].fillna(0).tolist()
        vals_2 = pivot[GROUP_ORDER[1]].fillna(0).tolist()

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
        ax.set_xticklabels(CLASSIFIER_ORDER)
        ax.set_ylim(0, 1.05)
        ax.legend()

    fig.suptitle("Test Macro-F1 by Classifier and Dataset Group")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "combined_test_macro_f1_by_dataset_group.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion_matrix_from_csv(
    csv_path: str,
    title: str,
    output_filename: str,
) -> None:
    if not os.path.exists(csv_path):
        print(f"Skipping missing confusion matrix file: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0)

    plt.figure(figsize=(7, 6))
    plt.imshow(df.values, cmap="Blues", interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(len(df.columns)), [str(c) for c in df.columns], rotation=45)
    plt.yticks(range(len(df.index)), [str(i) for i in df.index])

    max_value = df.values.max() if df.values.size else 0
    threshold = max_value / 2 if max_value > 0 else 0

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iat[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(CONFUSION_DIR, output_filename), dpi=300, bbox_inches="tight")
    plt.close()


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


def main() -> None:
    phase1_df = load_results(PHASE1_RESULTS, "Phase 1")
    phase2_df = load_results(PHASE2_RESULTS, "Phase 2")

    available_frames = [df for df in [phase1_df, phase2_df] if df is not None]
    if not available_frames:
        raise FileNotFoundError("No Phase 1 or Phase 2 result files were found.")

    all_df = pd.concat(available_frames, ignore_index=True)

    plot_combined_average_macro_f1(all_df)
    plot_combined_group_macro_f1(all_df)

    for csv_path, title, output_filename in build_confusion_targets():
        plot_confusion_matrix_from_csv(csv_path, title, output_filename)

    print(f"Saved figures to: {OUTPUT_DIR}")
    print(f"Saved confusion matrices to: {CONFUSION_DIR}")


if __name__ == "__main__":
    main()