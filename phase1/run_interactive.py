import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "data"
RESULTS_DIR = "results/interactive/phase1"
TARGET_COL = "Label"
RANDOM_STATE = 42

# SVM-specific thresholds
RBF_CLASS_LIMIT = 10
SGD_CLASS_LIMIT = 10
C_VALUES = [0.1, 1, 10]
GAMMA_VALUES = ["scale", "auto"]
TRAIN_FOLD_SIZE = 0.9

CLASSIFIERS = ["dt", "knn", "mlp", "rf", "svm"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINES & PARAM GRIDS
# ════════════════════════════════════════════════════════════════════════════════

def get_dt_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("dt",      DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


def get_dt_param_grid() -> dict:
    return {
        "dt__max_depth":         [3, 5, 10, None],
        "dt__min_samples_split": [2, 5, 10],
        "dt__criterion":         ["gini", "entropy"],
    }


def get_knn_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("knn",     KNeighborsClassifier()),
    ])


def get_knn_param_grid() -> dict:
    return {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__metric":      ["euclidean", "manhattan"],
    }


def get_mlp_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("mlp",     MLPClassifier(
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=15,
        )),
    ])


def get_mlp_param_grid() -> dict:
    return {
        "mlp__hidden_layer_sizes": [(64,), (128,), (64, 64)],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__alpha":              [0.0001, 0.001],
    }


def get_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def get_rf_param_grid() -> dict:
    return {
        "rf__n_estimators":      [100, 200],
        "rf__max_depth":         [5, 10, None],
        "rf__min_samples_split": [2, 5],
    }


def make_linear_svm_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("svm",     LinearSVC(C=C, max_iter=2000, dual=False,
                              tol=1e-3, random_state=RANDOM_STATE)),
    ])


def make_sgd_svm_pipeline(C: float, n_train: int) -> Pipeline:
    alpha = 1.0 / (C * n_train)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("svm",     SGDClassifier(
            loss="hinge",
            alpha=alpha,
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        )),
    ])


def make_rbf_svm_pipeline(C: float, gamma) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("svm",     SVC(C=C, kernel="rbf", gamma=gamma,
                        random_state=RANDOM_STATE, cache_size=1000)),
    ])


# ════════════════════════════════════════════════════════════════════════════════
# CLASSIFIER-SPECIFIC TUNING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def tune_gridcv_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    param_grid: dict,
) -> Tuple[Pipeline, dict]:
    """Standard GridSearchCV tuning for DT, KNN, MLP, RF."""
    inner_cv = ShuffleSplit(n_splits=1, test_size=0.10,
                            random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_params = {k.split("__")[-1]: v for k, v in grid.best_params_.items()}
    return grid.best_estimator_, best_params


def tune_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_classes: int,
) -> Tuple[Pipeline, dict]:
    """Manual SVM tuning with linear/RBF selection based on class count."""
    inner_split = ShuffleSplit(
        n_splits=1, test_size=0.10, random_state=RANDOM_STATE)
    sub_train_idx, val_idx = next(inner_split.split(X_train, y_train))

    X_sub = X_train.iloc[sub_train_idx]
    y_sub = y_train.iloc[sub_train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    n_sub = len(X_sub)

    best_score = -np.inf
    best_params = {}
    best_pipeline = None

    use_sgd = n_classes > SGD_CLASS_LIMIT

    if use_sgd:
        print(f"       [SVM] Using SGD linear SVM — {n_classes} classes")

    # Linear / SGD candidates
    for C in C_VALUES:
        pipeline = make_sgd_svm_pipeline(
            C, n_sub) if use_sgd else make_linear_svm_pipeline(C)
        pipeline.fit(X_sub, y_sub)
        score = f1_score(y_val, pipeline.predict(X_val),
                         average="macro", zero_division=0)

        if score > best_score:
            best_score = score
            best_params = {"kernel": "linear", "C": C, "gamma": "N/A"}
            best_pipeline = pipeline

    # RBF candidates
    if n_classes <= RBF_CLASS_LIMIT:
        for C in C_VALUES:
            for gamma in GAMMA_VALUES:
                pipeline = make_rbf_svm_pipeline(C, gamma)
                pipeline.fit(X_sub, y_sub)
                score = f1_score(y_val, pipeline.predict(X_val),
                                 average="macro", zero_division=0)

                if score > best_score:
                    best_score = score
                    best_params = {"kernel": "rbf", "C": C, "gamma": gamma}
                    best_pipeline = pipeline
    else:
        print(f"       [SVM] RBF skipped — {n_classes} classes too many")

    best_pipeline.fit(X_train, y_train)
    return best_pipeline, best_params


# ════════════════════════════════════════════════════════════════════════════════
# FORMAT RESULTS HELPER
# ════════════════════════════════════════════════════════════════════════════════

def format_classifier_results(classifier: str, best_params: dict, accuracy: float, f1_macro: float) -> dict:
    """Format results row based on classifier type."""
    base_row = {
        "Classifier":    classifier.upper(),
        "Accuracy":      accuracy,
        "F1":            f1_macro,
        "Parameters":    str(best_params),
    }

    if classifier == "dt":
        base_row.update({
            "Best_Max_Depth":     best_params.get("max_depth"),
            "Best_Min_Split":     best_params.get("min_samples_split"),
            "Best_Criterion":     best_params.get("criterion"),
        })
    elif classifier == "knn":
        base_row.update({
            "Best_K":      best_params.get("n_neighbors"),
            "Best_Metric": best_params.get("metric"),
        })
    elif classifier == "mlp":
        base_row.update({
            "Best_Hidden_Layers": str(best_params.get("hidden_layer_sizes")),
            "Best_Learning_Rate": best_params.get("learning_rate_init"),
            "Best_Alpha":         best_params.get("alpha"),
        })
    elif classifier == "rf":
        base_row.update({
            "Best_N_Estimators":      best_params.get("n_estimators"),
            "Best_Max_Depth":         best_params.get("max_depth"),
            "Best_Min_Samples_Split": best_params.get("min_samples_split"),
        })
    elif classifier == "svm":
        base_row.update({
            "Best_Kernel": best_params.get("kernel"),
            "Best_C":      best_params.get("C"),
            "Best_Gamma":  best_params.get("gamma"),
        })

    return base_row


# ════════════════════════════════════════════════════════════════════════════════
# CORE EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_classifiers(csv_path: str, dataset_name: str, classifiers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selected classifiers on a single dataset."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    n_classes = y.nunique()

    print(f"\n{'='*80}")
    print(f"Dataset  : {dataset_name}")
    print(f"Shape    : {X.shape}  |  Classes: {n_classes}")
    print(f"Classifiers: {', '.join(c.upper() for c in classifiers)}")
    print(f"{'='*80}")

    outer_cv = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    fold_rows_per_clf = {clf: [] for clf in classifiers}
    dataset_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.time()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\n  Fold {fold:>2}/10:")

        # Evaluate each selected classifier
        for clf in classifiers:
            try:
                if clf == "dt":
                    best_estimator, best_params = tune_gridcv_classifier(
                        X_train, y_train, get_dt_pipeline(), get_dt_param_grid()
                    )
                elif clf == "knn":
                    best_estimator, best_params = tune_gridcv_classifier(
                        X_train, y_train, get_knn_pipeline(), get_knn_param_grid()
                    )
                elif clf == "mlp":
                    best_estimator, best_params = tune_gridcv_classifier(
                        X_train, y_train, get_mlp_pipeline(), get_mlp_param_grid()
                    )
                elif clf == "rf":
                    best_estimator, best_params = tune_gridcv_classifier(
                        X_train, y_train, get_rf_pipeline(), get_rf_param_grid()
                    )
                elif clf == "svm":
                    best_estimator, best_params = tune_svm(
                        X_train, y_train, n_classes)

                y_pred = best_estimator.predict(X_test)
                accuracy = round(accuracy_score(y_test, y_pred), 4)
                f1_macro = round(f1_score(y_test, y_pred, average="macro",
                                          zero_division=0), 4)

                result = format_classifier_results(
                    clf, best_params, accuracy, f1_macro)
                result["Dataset"] = dataset_name
                result["Fold"] = fold
                result["Fold_Time_Seconds"] = round(
                    time.time() - fold_start, 2)

                fold_rows_per_clf[clf].append(result)

                print(
                    f"    {clf.upper():3s} | Acc={accuracy:.4f} | F1={f1_macro:.4f} | "
                    f"Params={best_params}"
                )

            except Exception as exc:
                print(f"    {clf.upper():3s} | ERROR: {exc}")
                raise

        fold_time = round(time.time() - fold_start, 2)

    # Combine all classifier results
    all_folds = []
    for clf in classifiers:
        if fold_rows_per_clf[clf]:
            all_folds.extend(fold_rows_per_clf[clf])

    folds_df = pd.DataFrame(all_folds)

    # Summary statistics per classifier
    summary_rows = []
    for clf in classifiers:
        clf_data = folds_df[folds_df["Classifier"] == clf.upper()]
        if not clf_data.empty:
            acc_mean = clf_data["Accuracy"].mean()
            acc_std = clf_data["Accuracy"].std(ddof=1)
            f1_mean = clf_data["F1"].mean()
            f1_std = clf_data["F1"].std(ddof=1)

            summary_rows.append({
                "Dataset":             dataset_name,
                "Classifier":          clf.upper(),
                "Accuracy Mean":       round(acc_mean, 4),
                "Accuracy Std":        round(acc_std, 4),
                "F1 Mean":             round(f1_mean, 4),
                "F1 Std":              round(f1_std, 4),
                "Accuracy Mean ± Std": f"{acc_mean:.4f} ± {acc_std:.4f}",
                "F1 Mean ± Std":       f"{f1_mean:.4f} ± {f1_std:.4f}",
                "Total_Time_Seconds":  round(time.time() - dataset_start, 2),
            })

            print(
                f"\n  {clf.upper()} Summary → "
                f"Acc: {acc_mean:.4f} ± {acc_std:.4f} | "
                f"F1: {f1_mean:.4f} ± {f1_std:.4f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    total_time = round(time.time() - dataset_start, 2)
    print(f"\n  Dataset Total Time: {total_time:.1f}s")

    return folds_df, summary_df


# ════════════════════════════════════════════════════════════════════════════════
# I/O & UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def save_results(folds_df: pd.DataFrame, summary_df: pd.DataFrame, dataset_name: str, classifiers: List[str]) -> None:
    """Save per-fold and summary results."""
    clf_results_dir = os.path.join(RESULTS_DIR, "training_results")
    os.makedirs(clf_results_dir, exist_ok=True)

    clf_suffix = "_".join(classifiers)
    folds_path = os.path.join(
        clf_results_dir, f"{dataset_name}_{clf_suffix}_folds.csv")
    summary_path = os.path.join(
        clf_results_dir, f"{dataset_name}_{clf_suffix}_summary.csv")

    folds_df.to_csv(folds_path,    index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"  Saved: {folds_path}")
    print(f"  Saved: {summary_path}")


def get_dataset_folders(data_dir: str) -> List[str]:
    """Get sorted list of dataset folders."""
    folders = [f for f in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    def sort_key(v):
        return (0, int(v)) if v.isdigit() else (1, v.lower())
    return sorted(folders, key=sort_key)


# ════════════════════════════════════════════════════════════════════════════════
# INTERACTIVE INPUT
# ════════════════════════════════════════════════════════════════════════════════

def display_menu() -> None:
    """Display main menu."""
    print("\n" + "="*80)
    print("Interactive Classifier Evaluation")
    print("="*80)
    print("\nUsage:")
    print("  - Enter dataset numbers as: 1 or 1,5,10 or 1-5 (range)")
    print("  - Enter classifier names as: dt or dt,svm or knn,rf")
    print("  - Available classifiers: dt, knn, mlp, rf, svm")
    print("="*80)


def parse_datasets(user_input: str, available_folders: List[str]) -> List[str]:
    """Parse user input for dataset selection."""
    user_input = user_input.strip()
    selected = []

    try:
        # Handle range input (1-5)
        if "-" in user_input:
            parts = user_input.split("-")
            if len(parts) == 2:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                selected = [str(i) for i in range(start, end + 1)]
            else:
                raise ValueError("Invalid range format")
        else:
            # Handle comma-separated input
            tokens = [t.strip() for t in user_input.split(",")]
            for token in tokens:
                if token.isdigit():
                    selected.append(token)
                else:
                    raise ValueError(f"Invalid dataset number: {token}")

        # Validate against available folders
        valid_selected = [s for s in selected if s in available_folders]

        if not valid_selected:
            print(
                f"⚠️  No valid datasets found. Available: {', '.join(available_folders)}")
            return []

        if len(valid_selected) < len(selected):
            print(
                f"⚠️  Some datasets not found. Using: {', '.join(valid_selected)}")

        return valid_selected

    except Exception as e:
        print(f"❌ Error parsing datasets: {e}")
        return []


def parse_classifiers(user_input: str) -> List[str]:
    """Parse user input for classifier selection."""
    user_input = user_input.strip().lower()

    try:
        selected = [c.strip() for c in user_input.split(",")]

        valid_selected = [c for c in selected if c in CLASSIFIERS]

        if not valid_selected:
            print(
                f"❌ No valid classifiers selected. Available: {', '.join(CLASSIFIERS)}")
            return []

        if len(valid_selected) < len(selected):
            invalid = [c for c in selected if c not in CLASSIFIERS]
            print(f"⚠️  Invalid classifiers ignored: {', '.join(invalid)}")

        return valid_selected

    except Exception as e:
        print(f"❌ Error parsing classifiers: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ════════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        sys.exit(1)

    available_folders = get_dataset_folders(DATA_DIR)
    if not available_folders:
        print(f"❌ No dataset folders found in: {DATA_DIR}")
        sys.exit(1)

    display_menu()

    while True:
        print("\n" + "-"*80)
        print("Enter '0' to exit, or proceed with selection:")
        print("-"*80)

        # Dataset selection
        print(f"\n📁 Available datasets: {', '.join(available_folders)}")
        dataset_input = input(
            "\n📌 Enter dataset number(s) [e.g., 1 or 1,5,10 or 1-5]: ").strip()

        if dataset_input == "0":
            print("\n👋 Exiting...\n")
            break

        selected_datasets = parse_datasets(dataset_input, available_folders)
        if not selected_datasets:
            print("❌ No valid datasets selected. Please try again.")
            continue

        # Classifier selection
        print(f"\n🤖 Available classifiers: {', '.join(CLASSIFIERS)}")
        classifier_input = input(
            "\n📌 Enter classifier(s) [e.g., dt or dt,svm or knn,rf]: ").strip()

        if classifier_input == "0":
            print("\n👋 Exiting...\n")
            break

        selected_classifiers = parse_classifiers(classifier_input)
        if not selected_classifiers:
            print("❌ No valid classifiers selected. Please try again.")
            continue

        # Confirmation
        print(f"\n✓ Datasets: {', '.join(selected_datasets)}")
        print(
            f"✓ Classifiers: {', '.join(c.upper() for c in selected_classifiers)}")
        confirm = input("\nProceed with evaluation? (y/n): ").strip().lower()

        if confirm != "y":
            print("⏭️  Skipped.")
            continue

        # Run evaluation
        print("\n🚀 Starting evaluation...\n")
        eval_start = time.time()
        all_summaries = []

        for folder in selected_datasets:
            csv_path = os.path.join(DATA_DIR, folder, "train.csv")
            if not os.path.exists(csv_path):
                print(f"⚠️  Skipping dataset_{folder}: train.csv not found")
                continue

            dataset_name = f"dataset_{folder}"
            try:
                folds_df, summary_df = evaluate_classifiers(
                    csv_path, dataset_name, selected_classifiers
                )
                save_results(folds_df, summary_df,
                             dataset_name, selected_classifiers)
                all_summaries.append(summary_df)
            except Exception as exc:
                print(f"\n❌ Failed on {dataset_name}: {exc}")
                continue

        # Final summary
        if all_summaries:
            combined = pd.concat(all_summaries, ignore_index=True)
            clf_suffix = "_".join(selected_classifiers)
            combined_path = os.path.join(
                RESULTS_DIR, "interactive",
                f"combined_results_{clf_suffix}.csv"
            )
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            combined.to_csv(combined_path, index=False)
            print(f"\n✓ Combined results → {combined_path}")

        total_eval_time = round(time.time() - eval_start, 2)
        print(f"\n✅ Evaluation completed in {total_eval_time:.1f}s\n")

        # Continue loop
        again = input("Evaluate another selection? (y/n): ").strip().lower()
        if again != "y":
            print("\n👋 Exiting...\n")
            break


if __name__ == "__main__":
    main()
