"""
Generates hyperparameter summary tables for Phase 1 and Phase 2.
For each classifier, finds the most frequently selected hyperparameter
combination across all folds and datasets.
"""

import os
import pandas as pd
from collections import Counter


PHASE1_DIR = "results/phase1"
PHASE2_DIR = "results/phase2"

CLASSIFIERS = {
    "SVM":  {"key": "svm", "params": ["Best_Kernel", "Best_C", "Best_Gamma"]},
    "kNN":  {"key": "knn", "params": ["Best_K", "Best_Metric"]},
    "DT":   {"key": "dt",  "params": ["Best_Max_Depth", "Best_Min_Split", "Best_Criterion"]},
    "RF":   {"key": "rf",  "params": ["Best_N_Estimators", "Best_Max_Depth", "Best_Min_Samples_Split"]},
    "MLP":  {"key": "mlp", "params": ["Best_Hidden_Layers", "Best_Learning_Rate", "Best_Alpha"]},
}


def get_most_common(series):
    """Return most frequent non-null value in a series."""
    counts = Counter(series.dropna().astype(str).tolist())
    if not counts:
        return "N/A"
    return counts.most_common(1)[0][0]


def load_all_folds_phase1(clf_key):
    """Load and combine all fold CSVs for a classifier in Phase 1."""
    dfs = []
    for ds in range(1, 17):
        path = os.path.join(PHASE1_DIR, clf_key, f"dataset_{ds}_{clf_key}_folds.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_all_folds_phase2(clf_key):
    """Load and combine all fold CSVs for a classifier in Phase 2."""
    dfs = []
    for ds in range(1, 17):
        path = os.path.join(PHASE2_DIR, f"dataset_{ds}_all_p2_folds.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            clf_df = df[df["Classifier"] == clf_key.upper()]
            if not clf_df.empty:
                dfs.append(clf_df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def format_params(row, clf_name):
    """Format hyperparameters into a readable string."""
    if clf_name == "SVM":
        kernel = row.get("Best_Kernel", "N/A")
        C      = row.get("Best_C", "N/A")
        gamma  = row.get("Best_Gamma", "N/A")
        if kernel == "linear":
            return f"kernel=linear, C={C}"
        return f"kernel=rbf, C={C}, gamma={gamma}"

    elif clf_name == "kNN":
        return f"k={row.get('Best_K', 'N/A')}, metric={row.get('Best_Metric', 'N/A')}"

    elif clf_name == "DT":
        depth    = row.get("Best_Max_Depth", "N/A")
        split    = row.get("Best_Min_Split", "N/A")
        criterion= row.get("Best_Criterion", "N/A")
        return f"max_depth={depth}, min_samples_split={split}, criterion={criterion}"

    elif clf_name == "RF":
        n_est  = row.get("Best_N_Estimators", "N/A")
        depth  = row.get("Best_Max_Depth", "N/A")
        split  = row.get("Best_Min_Samples_Split", "N/A")
        return f"n_estimators={n_est}, max_depth={depth}, min_samples_split={split}"

    elif clf_name == "MLP":
        layers = row.get("Best_Hidden_Layers", "N/A")
        lr     = row.get("Best_Learning_Rate", "N/A")
        alpha  = row.get("Best_Alpha", "N/A")
        return f"hidden_layers={layers}, lr={lr}, alpha={alpha}"

    return "N/A"


def get_best_params(df, clf_name, param_cols):
    """Get most common value per param column, return formatted string."""
    if df.empty:
        return "No data"
    row = {col: get_most_common(df[col]) for col in param_cols if col in df.columns}
    return format_params(row, clf_name)


def main():
    rows = []

    for clf_name, cfg in CLASSIFIERS.items():
        clf_key   = cfg["key"]
        param_cols = cfg["params"]

        # Phase 1
        df1     = load_all_folds_phase1(clf_key)
        params1 = get_best_params(df1, clf_name, param_cols)

        # Phase 2
        df2     = load_all_folds_phase2(clf_key)
        params2 = get_best_params(df2, clf_name, param_cols)

        rows.append({
            "Classifier": clf_name,
            "Phase":      "Before FS/DR",
            "Best Hyperparameters": params1,
        })
        rows.append({
            "Classifier": clf_name,
            "Phase":      "After FS/DR",
            "Best Hyperparameters": params2,
        })

    result = pd.DataFrame(rows)

    print("\nHyperparameter Summary Table")
    print("=" * 80)
    print(result.to_string(index=False))
    print("=" * 80)

    result.to_csv("report_figures/hyperparameter_table.csv", index=False)
    print("\nSaved: report_figures/hyperparameter_table.csv")


if __name__ == "__main__":
    os.makedirs("report_figures", exist_ok=True)
    main()