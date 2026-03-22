"""
Generates two results tables for the report:
  - Table A: Accuracy mean +/- std per dataset per phase per classifier
  - Table B: F1 mean +/- std per dataset per phase per classifier

Reads from individual summary CSVs in results/phase1 and results/phase2.
Outputs to report_figures/results_table_accuracy.csv and results_table_f1.csv
"""

import os
import pandas as pd

PHASE1_DIR = "results/phase1"
PHASE2_DIR = "results/phase2"
OUTPUT_DIR = "report_figures"
N_DATASETS = 16

CLASSIFIERS = ["svm", "knn", "dt", "rf", "mlp"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_phase1_summary(clf, dataset_num):
    path = os.path.join(PHASE1_DIR, clf, f"dataset_{dataset_num}_{clf}_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_phase2_summary(clf, dataset_num):
    path = os.path.join(PHASE2_DIR, clf, f"dataset_{dataset_num}_{clf}_p2_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def build_table(metric):
    """
    metric: 'Accuracy' or 'F1'
    Returns a DataFrame with columns: Dataset, Phase, SVM, kNN, DT, RF, MLP
    """
    rows = []

    mean_col = f"{metric} Mean"
    std_col  = f"{metric} Std"

    for ds in range(1, N_DATASETS + 1):
        for phase, loader in [("Before FS/DR", load_phase1_summary),
                               ("After FS/DR",  load_phase2_summary)]:
            row = {"Dataset": f"Data {ds}", "Phase": phase}

            for clf in CLASSIFIERS:
                df = loader(clf, ds)
                if df is None or df.empty:
                    row[clf.upper()] = "N/A"
                    continue

                mean = df[mean_col].values[0]
                std  = df[std_col].values[0]
                row[clf.upper()] = f"{mean:.4f} +/- {std:.4f}"

            rows.append(row)

    return pd.DataFrame(rows, columns=["Dataset", "Phase", "SVM", "KNN", "DT", "RF", "MLP"])


def main():
    for metric, label in [("Accuracy", "accuracy"), ("F1", "f1")]:
        table    = build_table(metric)
        out_path = os.path.join(OUTPUT_DIR, f"results_table_{label}.csv")
        table.to_csv(out_path, index=False)

        print(f"\n{metric} Results Table")
        print("=" * 100)
        print(table.to_string(index=False))
        print("=" * 100)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()