"""
Phase 2 - Decision Tree with Permutation Importance Feature Selection
=====================================================================
Feature selection is computed ONCE per fold using Random Forest permutation
importance, then Decision Tree is tuned on reduced features.

Process per fold:
  1. Split train/test via outer StratifiedKFold
  2. Fit RF + compute permutation importance (once)
  3. Keep features with mean importance above mean importance score
  4. Tune DT on reduced features, evaluate on reduced test fold
  5. Record accuracy, F1, selected features, parameters
"""

import os
import time
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "data"
RESULTS_DIR = "results/phase2/dt"
TARGET_COL = "Label"
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    X_sc = scaler.fit_transform(X_imp)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )
    rf.fit(X_sc, y_train)

    result = permutation_importance(
        rf, X_sc, y_train,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=1,
    )

    importances = result.importances_mean
    mean_thresh = importances.mean()
    selected_idx = np.where(importances > mean_thresh)[0]

    if len(selected_idx) == 0:
        selected_idx = np.argsort(importances)[
            ::-1][:max(1, len(importances) // 2)]

    return list(X_train.columns[selected_idx])


# ══════════════════════════════════════════════════════════════════════════════
#  DECISION TREE CLASSIFIER DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _base():
    return [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ]


def get_dt_pipeline():
    return Pipeline(_base() + [("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))])


def get_dt_param_grid():
    return {
        "dt__max_depth": [3, 5, 10, None],
        "dt__min_samples_split": [2, 5, 10],
        "dt__criterion": ["gini", "entropy"],
    }


def run_grid(pipeline, param_grid, X_train, y_train, inner_cv):
    grid = GridSearchCV(
        estimator=pipeline, param_grid=param_grid,
        cv=inner_cv, scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ══════════════════════════════════════════════════════════════════════════════
#  CORE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(csv_path, dataset_name):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    n_classes = y.nunique()

    print(f"\n{'='*60}")
    print(f"Dataset  : {dataset_name}")
    print(f"Shape    : {X.shape}  |  Classes: {n_classes}")
    print(f"{'='*60}")

    outer_cv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = ShuffleSplit(n_splits=1, test_size=0.10,
                            random_state=RANDOM_STATE)

    fold_rows = []
    dataset_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.time()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Feature selection — once per fold
        fs_start = time.time()
        selected_features = select_features(X_train, y_train)
        n_selected = len(selected_features)
        print(
            f"\n  Fold {fold:>2}/10 | Features: {X.shape[1]} → {n_selected} | FS: {time.time()-fs_start:.1f}s")

        X_tr = X_train[selected_features]
        X_te = X_test[selected_features]
        feat_str = ", ".join(selected_features)

        def base_row():
            return {
                "Dataset": dataset_name, "Fold": fold,
                "N_Features_Before": X.shape[1], "N_Features_After": n_selected,
                "Selected_Features": feat_str,
            }

        # ── DT ─────────────────────────────────────────────────────────────────
        t = time.time()
        est, p = run_grid(get_dt_pipeline(),
                          get_dt_param_grid(), X_tr, y_train, inner_cv)
        y_pred = est.predict(X_te)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1 = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        cp = {k.replace("dt__", ""): v for k, v in p.items()}
        fold_rows.append({**base_row(), "Accuracy": acc, "F1": f1,
                          "Best_Max_Depth": cp.get("max_depth"),
                          "Best_Min_Split": cp.get("min_samples_split"),
                          "Best_Criterion": cp.get("criterion"),
                          "Parameters": str(cp), "Fold_Time_Seconds": round(time.time()-t, 2)})
        print(
            f"    DT   | Acc={acc:.4f} F1={f1:.4f} | {round(time.time()-t, 1)}s")
        print(f"  Fold {fold} total: {round(time.time()-fold_start, 1)}s")

    # Build DataFrames and summaries
    folds_df = pd.DataFrame(fold_rows)
    acc_mean = folds_df["Accuracy"].mean()
    acc_std = folds_df["Accuracy"].std(ddof=1)
    f1_mean = folds_df["F1"].mean()
    f1_std = folds_df["F1"].std(ddof=1)
    avg_features = folds_df["N_Features_After"].mean()
    total_time = round(time.time() - dataset_start, 2)

    summary_df = pd.DataFrame([{
        "Dataset":             dataset_name,
        "Classifier":          "DT",
        "Accuracy Mean":       round(acc_mean, 4),
        "Accuracy Std":        round(acc_std, 4),
        "F1 Mean":             round(f1_mean, 4),
        "F1 Std":              round(f1_std, 4),
        "Accuracy Mean ± Std": f"{acc_mean:.4f} ± {acc_std:.4f}",
        "F1 Mean ± Std":       f"{f1_mean:.4f} ± {f1_std:.4f}",
        "Avg_Features_Before": X.shape[1],
        "Avg_Features_After":  round(avg_features, 1),
        "Total_Time_Seconds":  total_time,
    }])

    print(f"\n  [DT] Acc: {acc_mean:.4f} ± {acc_std:.4f} | "
          f"F1: {f1_mean:.4f} ± {f1_std:.4f} | "
          f"Avg features: {avg_features:.1f}/{X.shape[1]}")

    return folds_df, summary_df


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_results(folds_df, summary_df, dataset_name):
    folds_path = os.path.join(RESULTS_DIR, f"{dataset_name}_dt_p2_folds.csv")
    summary_path = os.path.join(
        RESULTS_DIR, f"{dataset_name}_dt_p2_summary.csv")
    folds_df.to_csv(folds_path,    index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {folds_path}")
    print(f"  Saved: {summary_path}")


def combine_all_summaries(frames):
    if not frames:
        return
    out_path = os.path.join(RESULTS_DIR, "all_datasets_dt_p2_summary.csv")
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"\nCombined summary → {out_path}")


def get_dataset_folders(data_dir):
    folders = [f for f in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    def sort_key(v):
        return (0, int(v)) if v.isdigit() else (1, v.lower())
    return sorted(folders, key=sort_key)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

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
