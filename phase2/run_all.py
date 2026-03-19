"""
Phase 2 - All Classifiers with Permutation Importance Feature Selection
=======================================================================
Feature selection is computed ONCE per fold using Random Forest permutation
importance, then shared across all 5 classifiers — avoids recomputing the
expensive permutation importance step 5 times per fold.

Process per fold:
  1. Split train/test via outer StratifiedKFold
  2. Fit RF + compute permutation importance (once)
  3. Keep features with mean importance above mean importance score
  4. For each classifier: tune on reduced features, evaluate on reduced test fold
  5. Record accuracy, F1, selected features, parameters per classifier
"""

import os
import time
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR        = "data"
RESULTS_DIR     = "results/phase2"
TARGET_COL      = "Label"
RANDOM_STATE    = 42
RBF_CLASS_LIMIT = 10
SGD_CLASS_LIMIT = 10
C_VALUES        = [0.1, 1, 10]
GAMMA_VALUES    = ["scale", "auto"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    imputer = SimpleImputer(strategy="mean")
    scaler  = StandardScaler()
    X_imp   = imputer.fit_transform(X_train)
    X_sc    = scaler.fit_transform(X_imp)

    rf = RandomForestClassifier(
        n_estimators = 100,
        random_state = RANDOM_STATE,
        n_jobs       = 1,
        class_weight = "balanced",
    )
    rf.fit(X_sc, y_train)

    result = permutation_importance(
        rf, X_sc, y_train,
        n_repeats    = 10,
        random_state = RANDOM_STATE,
        scoring      = "f1_macro",
        n_jobs       = 1,
    )

    importances  = result.importances_mean
    mean_thresh  = importances.mean()
    selected_idx = np.where(importances > mean_thresh)[0]

    if len(selected_idx) == 0:
        selected_idx = np.argsort(importances)[::-1][:max(1, len(importances) // 2)]

    return list(X_train.columns[selected_idx])


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFIER DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _base():
    return [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ]

# ── SVM ────────────────────────────────────────────────────────────────────────

def make_linear_pipeline(C):
    return Pipeline(_base() + [
        ("svm", LinearSVC(C=C, max_iter=2000, dual=False, tol=1e-3, random_state=RANDOM_STATE))
    ])

def make_sgd_pipeline(C, n_train):
    return Pipeline(_base() + [
        ("svm", SGDClassifier(
            loss="hinge", alpha=1.0/(C*n_train), max_iter=1000, tol=1e-3,
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1,
        ))
    ])

def make_rbf_pipeline(C, gamma):
    return Pipeline(_base() + [
        ("svm", SVC(C=C, kernel="rbf", gamma=gamma, random_state=RANDOM_STATE, cache_size=1000))
    ])

def tune_svm(X_train, y_train, n_classes):
    inner = ShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)
    sub_idx, val_idx = next(inner.split(X_train, y_train))
    X_sub, y_sub = X_train.iloc[sub_idx], y_train.iloc[sub_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    best_score, best_params, best_pipeline = -np.inf, {}, None
    use_sgd = n_classes > SGD_CLASS_LIMIT

    for C in C_VALUES:
        pipe  = make_sgd_pipeline(C, len(X_sub)) if use_sgd else make_linear_pipeline(C)
        pipe.fit(X_sub, y_sub)
        score = f1_score(y_val, pipe.predict(X_val), average="macro", zero_division=0)
        if score > best_score:
            best_score, best_params, best_pipeline = score, {"kernel": "linear", "C": C, "gamma": "N/A"}, pipe

    if n_classes <= RBF_CLASS_LIMIT:
        for C in C_VALUES:
            for gamma in GAMMA_VALUES:
                pipe = make_rbf_pipeline(C, gamma)
                pipe.fit(X_sub, y_sub)
                score = f1_score(y_val, pipe.predict(X_val), average="macro", zero_division=0)
                if score > best_score:
                    best_score, best_params, best_pipeline = score, {"kernel": "rbf", "C": C, "gamma": gamma}, pipe

    best_pipeline.fit(X_train, y_train)
    return best_pipeline, best_params

# ── kNN, DT, RF, MLP ──────────────────────────────────────────────────────────

def get_knn_pipeline():
    return Pipeline(_base() + [("knn", KNeighborsClassifier())])

def get_knn_param_grid():
    return {"knn__n_neighbors": [3,5,7,9,11], "knn__metric": ["euclidean","manhattan"]}

def get_dt_pipeline():
    return Pipeline(_base() + [("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))])

def get_dt_param_grid():
    return {
        "dt__max_depth": [3,5,10,None],
        "dt__min_samples_split": [2,5,10],
        "dt__criterion": ["gini","entropy"],
    }

def get_rf_pipeline():
    return Pipeline(_base() + [("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])

def get_rf_param_grid():
    return {
        "rf__n_estimators": [100,200],
        "rf__max_depth": [5,10,None],
        "rf__min_samples_split": [2,5],
    }

def get_mlp_pipeline():
    return Pipeline(_base() + [
        ("mlp", MLPClassifier(max_iter=500, random_state=RANDOM_STATE,
                              early_stopping=True, n_iter_no_change=15))
    ])

def get_mlp_param_grid():
    return {
        "mlp__hidden_layer_sizes": [(64,),(128,),(64,64)],
        "mlp__learning_rate_init": [0.001,0.01],
        "mlp__alpha": [0.0001,0.001],
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
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    n_classes = y.nunique()

    print(f"\n{'='*60}")
    print(f"Dataset  : {dataset_name}")
    print(f"Shape    : {X.shape}  |  Classes: {n_classes}")
    print(f"{'='*60}")

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = ShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)

    # One list per classifier
    rows = {clf: [] for clf in ["svm","knn","dt","rf","mlp"]}
    dataset_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.time()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Feature selection — once per fold
        fs_start          = time.time()
        selected_features = select_features(X_train, y_train)
        n_selected        = len(selected_features)
        print(f"\n  Fold {fold:>2}/10 | Features: {X.shape[1]} → {n_selected} | FS: {time.time()-fs_start:.1f}s")

        X_tr = X_train[selected_features]
        X_te = X_test[selected_features]
        feat_str = ", ".join(selected_features)

        def base_row():
            return {
                "Dataset": dataset_name, "Fold": fold,
                "N_Features_Before": X.shape[1], "N_Features_After": n_selected,
                "Selected_Features": feat_str,
            }

        # ── SVM ────────────────────────────────────────────────────────────────
        t = time.time()
        pipe, params = tune_svm(X_tr, y_train, n_classes)
        y_pred = pipe.predict(X_te)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        rows["svm"].append({**base_row(), "Accuracy": acc, "F1": f1,
            "Best_Kernel": params.get("kernel"), "Best_C": params.get("C"),
            "Best_Gamma": params.get("gamma"), "Parameters": str(params),
            "Fold_Time_Seconds": round(time.time()-t, 2)})
        print(f"    SVM  | Acc={acc:.4f} F1={f1:.4f} | {round(time.time()-t,1)}s")

        # ── kNN ────────────────────────────────────────────────────────────────
        t = time.time()
        est, p = run_grid(get_knn_pipeline(), get_knn_param_grid(), X_tr, y_train, inner_cv)
        y_pred = est.predict(X_te)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        cp  = {k.replace("knn__",""): v for k,v in p.items()}
        rows["knn"].append({**base_row(), "Accuracy": acc, "F1": f1,
            "Best_K": cp.get("n_neighbors"), "Best_Metric": cp.get("metric"),
            "Parameters": str(cp), "Fold_Time_Seconds": round(time.time()-t, 2)})
        print(f"    kNN  | Acc={acc:.4f} F1={f1:.4f} | {round(time.time()-t,1)}s")

        # ── DT ─────────────────────────────────────────────────────────────────
        t = time.time()
        est, p = run_grid(get_dt_pipeline(), get_dt_param_grid(), X_tr, y_train, inner_cv)
        y_pred = est.predict(X_te)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        cp  = {k.replace("dt__",""): v for k,v in p.items()}
        rows["dt"].append({**base_row(), "Accuracy": acc, "F1": f1,
            "Best_Max_Depth": cp.get("max_depth"),
            "Best_Min_Split": cp.get("min_samples_split"),
            "Best_Criterion": cp.get("criterion"),
            "Parameters": str(cp), "Fold_Time_Seconds": round(time.time()-t, 2)})
        print(f"    DT   | Acc={acc:.4f} F1={f1:.4f} | {round(time.time()-t,1)}s")

        # ── RF ─────────────────────────────────────────────────────────────────
        t = time.time()
        est, p = run_grid(get_rf_pipeline(), get_rf_param_grid(), X_tr, y_train, inner_cv)
        y_pred = est.predict(X_te)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        cp  = {k.replace("rf__",""): v for k,v in p.items()}
        rows["rf"].append({**base_row(), "Accuracy": acc, "F1": f1,
            "Best_N_Estimators": cp.get("n_estimators"),
            "Best_Max_Depth": cp.get("max_depth"),
            "Best_Min_Split": cp.get("min_samples_split"),
            "Parameters": str(cp), "Fold_Time_Seconds": round(time.time()-t, 2)})
        print(f"    RF   | Acc={acc:.4f} F1={f1:.4f} | {round(time.time()-t,1)}s")

        # ── MLP ────────────────────────────────────────────────────────────────
        t = time.time()
        est, p = run_grid(get_mlp_pipeline(), get_mlp_param_grid(), X_tr, y_train, inner_cv)
        y_pred = est.predict(X_te)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1  = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        cp  = {k.replace("mlp__",""): v for k,v in p.items()}
        rows["mlp"].append({**base_row(), "Accuracy": acc, "F1": f1,
            "Best_Hidden_Layers": str(cp.get("hidden_layer_sizes")),
            "Best_Learning_Rate": cp.get("learning_rate_init"),
            "Best_Alpha": cp.get("alpha"),
            "Parameters": str(cp), "Fold_Time_Seconds": round(time.time()-t, 2)})
        print(f"    MLP  | Acc={acc:.4f} F1={f1:.4f} | {round(time.time()-t,1)}s")
        print(f"  Fold {fold} total: {round(time.time()-fold_start, 1)}s")

    # Build DataFrames and summaries
    all_folds, all_summaries = [], []

    for clf in ["svm","knn","dt","rf","mlp"]:
        folds_df     = pd.DataFrame(rows[clf])
        acc_mean     = folds_df["Accuracy"].mean()
        acc_std      = folds_df["Accuracy"].std(ddof=1)
        f1_mean      = folds_df["F1"].mean()
        f1_std       = folds_df["F1"].std(ddof=1)
        avg_features = folds_df["N_Features_After"].mean()
        total_time   = round(time.time() - dataset_start, 2)

        folds_df["Classifier"] = clf.upper()
        all_folds.append(folds_df)

        all_summaries.append(pd.DataFrame([{
            "Dataset":             dataset_name,
            "Classifier":          clf.upper(),
            "Accuracy Mean":       round(acc_mean, 4),
            "Accuracy Std":        round(acc_std, 4),
            "F1 Mean":             round(f1_mean, 4),
            "F1 Std":              round(f1_std, 4),
            "Accuracy Mean ± Std": f"{acc_mean:.4f} ± {acc_std:.4f}",
            "F1 Mean ± Std":       f"{f1_mean:.4f} ± {f1_std:.4f}",
            "Avg_Features_Before": X.shape[1],
            "Avg_Features_After":  round(avg_features, 1),
            "Total_Time_Seconds":  total_time,
        }]))

        print(f"\n  [{clf.upper()}] Acc: {acc_mean:.4f} ± {acc_std:.4f} | "
              f"F1: {f1_mean:.4f} ± {f1_std:.4f} | "
              f"Avg features: {avg_features:.1f}/{X.shape[1]}")

    return pd.concat(all_folds, ignore_index=True), pd.concat(all_summaries, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_results(folds_df, summary_df, dataset_name):
    # Save combined
    folds_path   = os.path.join(RESULTS_DIR, f"{dataset_name}_all_p2_folds.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{dataset_name}_all_p2_summary.csv")
    folds_df.to_csv(folds_path,    index=False)
    summary_df.to_csv(summary_path, index=False)

    # Save per classifier
    for clf in ["svm","knn","dt","rf","mlp"]:
        clf_dir = os.path.join(RESULTS_DIR, clf)
        os.makedirs(clf_dir, exist_ok=True)

        clf_folds   = folds_df[folds_df["Classifier"] == clf.upper()].copy()
        clf_summary = summary_df[summary_df["Classifier"] == clf.upper()].copy()

        clf_folds.to_csv(os.path.join(clf_dir, f"{dataset_name}_{clf}_p2_folds.csv"), index=False)
        clf_summary.to_csv(os.path.join(clf_dir, f"{dataset_name}_{clf}_p2_summary.csv"), index=False)

    print(f"  Saved: {folds_path}")
    print(f"  Saved: {summary_path}")


def combine_all_summaries(frames):
    if not frames:
        return
    out_path = os.path.join(RESULTS_DIR, "all_datasets_all_clfs_p2_summary.csv")
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"\nCombined summary → {out_path}")


def get_dataset_folders(data_dir):
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
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