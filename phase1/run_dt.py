import os
import time
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR     = "data"
RESULTS_DIR  = "results/phase1/dt"
TARGET_COL   = "Label"
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Pipeline & grid ────────────────────────────────────────────────────────────

def get_dt_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("dt",      DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])

def get_dt_param_grid():
    """
    Tuning max_depth and min_samples_split as specified in the instructions.
    None for max_depth = fully grown tree (no depth limit).
    Higher min_samples_split = more regularisation, avoids overfitting on
    imbalanced datasets like 9-16.
    """
    return {
        "dt__max_depth":         [3, 5, 10, None],
        "dt__min_samples_split": [2, 5, 10],
        "dt__criterion":         ["gini", "entropy"],
    }


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_dataset(csv_path: str, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]

    n_classes = y.nunique()

    print(f"\n{'='*60}")
    print(f"Dataset  : {dataset_name}")
    print(f"Shape    : {X.shape}  |  Classes: {n_classes}")
    print(f"{'='*60}")

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = ShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)

    fold_rows     = []
    dataset_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.time()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid = GridSearchCV(
            estimator  = get_dt_pipeline(),
            param_grid = get_dt_param_grid(),
            cv         = inner_cv,
            scoring    = "f1_macro",
            n_jobs     = -1,
            refit      = True,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        best_params    = grid.best_params_
        y_pred         = best_estimator.predict(X_test)

        accuracy  = round(accuracy_score(y_test, y_pred), 4)
        f1_macro  = round(f1_score(y_test, y_pred, average="macro"), 4)
        fold_time = round(time.time() - fold_start, 2)

        clean_params = {k.replace("dt__", ""): v for k, v in best_params.items()}

        fold_rows.append({
            "Dataset":           dataset_name,
            "Fold":              fold,
            "Accuracy":          accuracy,
            "F1":                f1_macro,
            "Best_Max_Depth":    clean_params.get("max_depth"),
            "Best_Min_Split":    clean_params.get("min_samples_split"),
            "Best_Criterion":    clean_params.get("criterion"),
            "Parameters":        str(clean_params),
            "Fold_Time_Seconds": fold_time,
        })

        print(
            f"  Fold {fold:>2}/10 | "
            f"Acc={accuracy:.4f} | F1={f1_macro:.4f} | "
            f"Params={clean_params} | Time={fold_time:.1f}s"
        )

    folds_df   = pd.DataFrame(fold_rows)
    total_time = round(time.time() - dataset_start, 2)
    acc_mean   = folds_df["Accuracy"].mean()
    acc_std    = folds_df["Accuracy"].std(ddof=1)
    f1_mean    = folds_df["F1"].mean()
    f1_std     = folds_df["F1"].std(ddof=1)

    print(
        f"\n  Summary → "
        f"Acc: {acc_mean:.4f} ± {acc_std:.4f} | "
        f"F1: {f1_mean:.4f} ± {f1_std:.4f} | "
        f"Total: {total_time:.1f}s"
    )

    summary_df = pd.DataFrame([{
        "Dataset":             dataset_name,
        "Accuracy Mean":       round(acc_mean, 4),
        "Accuracy Std":        round(acc_std, 4),
        "F1 Mean":             round(f1_mean, 4),
        "F1 Std":              round(f1_std, 4),
        "Accuracy Mean ± Std": f"{acc_mean:.4f} ± {acc_std:.4f}",
        "F1 Mean ± Std":       f"{f1_mean:.4f} ± {f1_std:.4f}",
        "Total_Time_Seconds":  total_time,
    }])

    return folds_df, summary_df


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_results(folds_df: pd.DataFrame, summary_df: pd.DataFrame, dataset_name: str) -> None:
    folds_path   = os.path.join(RESULTS_DIR, f"{dataset_name}_dt_folds.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{dataset_name}_dt_summary.csv")
    folds_df.to_csv(folds_path,    index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {folds_path}")
    print(f"  Saved: {summary_path}")


def combine_all_summaries(frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    out_path = os.path.join(RESULTS_DIR, "all_datasets_dt_summary.csv")
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"\nCombined summary → {out_path}")


def get_dataset_folders(data_dir: str) -> list[str]:
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    def sort_key(v):
        return (0, int(v)) if v.isdigit() else (1, v.lower())
    return sorted(folders, key=sort_key)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    folders = get_dataset_folders(DATA_DIR)
    if not folders:
        raise ValueError(f"No dataset folders found in: {DATA_DIR}")

    all_summaries = []

    for folder in folders:
        csv_path = os.path.join(DATA_DIR, folder, "train.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {folder}: train.csv not found")
            continue

        dataset_name = f"dataset_{folder}"
        try:
            folds_df, summary_df = evaluate_dataset(csv_path, dataset_name)
            save_results(folds_df, summary_df, dataset_name)
            all_summaries.append(summary_df)
        except Exception as exc:
            print(f"Failed on {dataset_name}: {exc}")
            raise

    combine_all_summaries(all_summaries)
    print("\nDone.")


if __name__ == "__main__":
    main()