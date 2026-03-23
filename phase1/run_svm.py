import os
import time
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR      = "data"
RESULTS_DIR   = "results/phase1/svm"
TARGET_COL    = "Label"
RANDOM_STATE  = 42

# Thresholds for switching solver
RBF_CLASS_LIMIT  = 10      # skip RBF if more than 10 classes
SGD_CLASS_LIMIT  = 10      # use SGD linear SVM if more than 10 classes

# Hyperparameter search space
C_VALUES     = [0.1, 1, 10]
GAMMA_VALUES = ["scale", "auto"]

# Approximate training fold size (90% of dataset)
# Used to convert C → SGD alpha: alpha = 1 / (C * n_samples)
TRAIN_FOLD_SIZE = 0.9

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Pipelines ──────────────────────────────────────────────────────────────────

def make_linear_pipeline(C: float) -> Pipeline:
    """Standard LinearSVC — used for datasets 1-8 (4 classes, low dimensional)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("svm",     LinearSVC(C=C, max_iter=2000, dual=False,
                              tol=1e-3, random_state=RANDOM_STATE)),
    ])

def make_sgd_pipeline(C: float, n_train: int) -> Pipeline:
    """
    SGDClassifier with hinge loss = linear SVM.
    Used for datasets 9-16 (16 classes, 265 features, severe imbalance).
    Mathematically equivalent to LinearSVC but uses stochastic gradient
    descent which handles high-dimensional imbalanced multiclass problems
    orders of magnitude faster than the liblinear solver.
    alpha = 1 / (C * n_samples) is the exact SGD equivalent of SVM's C.
    class_weight='balanced' corrects for severe class imbalance.
    """
    alpha = 1.0 / (C * n_train)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("svm",     SGDClassifier(
            loss         = "hinge",
            alpha        = alpha,
            max_iter     = 1000,
            tol          = 1e-3,
            random_state = RANDOM_STATE,
            class_weight = "balanced",
            n_jobs       = 1,
        )),
    ])

def make_rbf_pipeline(C: float, gamma) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("svm",     SVC(C=C, kernel="rbf", gamma=gamma,
                        random_state=RANDOM_STATE, cache_size=1000)),
    ])


# ── Manual inner tuning ────────────────────────────────────────────────────────

def tune_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_classes: int,
) -> tuple[Pipeline, dict]:
    """
    Iterates over hyperparameter configs using a single 90/10 inner split
    as specified in the instructions. Picks best by macro F1, then refits
    on the full training fold before returning.
    """
    inner_split = ShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)
    sub_train_idx, val_idx = next(inner_split.split(X_train, y_train))

    X_sub = X_train.iloc[sub_train_idx]
    y_sub = y_train.iloc[sub_train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    n_sub = len(X_sub)

    best_score    = -np.inf
    best_params   = {}
    best_pipeline = None

    use_sgd = n_classes > SGD_CLASS_LIMIT

    if use_sgd:
        print(f"    [!] Using SGD linear SVM (hinge loss) — "
              f"{n_classes} classes, solver would hang with LibLinear")

    # ── Linear / SGD candidates ────────────────────────────────────────────────
    for C in C_VALUES:
        pipeline = make_sgd_pipeline(C, n_sub) if use_sgd else make_linear_pipeline(C)
        pipeline.fit(X_sub, y_sub)
        score = f1_score(y_val, pipeline.predict(X_val),
                         average="macro", zero_division=0)

        if score > best_score:
            best_score    = score
            best_params   = {"kernel": "linear", "C": C, "gamma": "N/A"}
            best_pipeline = pipeline

    # ── RBF candidates ─────────────────────────────────────────────────────────
    if n_classes <= RBF_CLASS_LIMIT:
        for C in C_VALUES:
            for gamma in GAMMA_VALUES:
                pipeline = make_rbf_pipeline(C, gamma)
                pipeline.fit(X_sub, y_sub)
                score = f1_score(y_val, pipeline.predict(X_val),
                                 average="macro", zero_division=0)

                if score > best_score:
                    best_score    = score
                    best_params   = {"kernel": "rbf", "C": C, "gamma": gamma}
                    best_pipeline = pipeline
    else:
        print(f"    [!] RBF skipped — {n_classes} classes = "
              f"{n_classes*(n_classes-1)//2} binary SVMs per fit")

    # Refit best config on full training fold
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, best_params


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

    outer_cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    dataset_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.time()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_pipeline, best_params = tune_svm(X_train, y_train, n_classes)

        y_pred    = best_pipeline.predict(X_test)
        accuracy  = round(accuracy_score(y_test, y_pred), 4)
        f1_macro  = round(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0), 4)
        fold_time = round(time.time() - fold_start, 2)

        fold_rows.append({
            "Dataset":           dataset_name,
            "Fold":              fold,
            "Accuracy":          accuracy,
            "F1":                f1_macro,
            "Best_Kernel":       best_params.get("kernel"),
            "Best_C":            best_params.get("C"),
            "Best_Gamma":        best_params.get("gamma"),
            "Parameters":        str(best_params),
            "Fold_Time_Seconds": fold_time,
        })

        print(
            f"  Fold {fold:>2}/10 | "
            f"Acc={accuracy:.4f} | F1={f1_macro:.4f} | "
            f"Params={best_params} | Time={fold_time:.1f}s"
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
    folds_path   = os.path.join(RESULTS_DIR, f"{dataset_name}_svm_folds.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{dataset_name}_svm_summary.csv")
    folds_df.to_csv(folds_path,    index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {folds_path}")
    print(f"  Saved: {summary_path}")


def combine_all_summaries(frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    out_path = os.path.join(RESULTS_DIR, "all_datasets_svm_summary.csv")
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