import os
import re
import ast
import json
import time
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


DATA_DIR = "data"
HYPERPARAMETER_TABLE = "report_figures/hyperparameter_table.csv"
RESULTS_DIR = os.path.join("results", "final_test", "phase2")
MODELS_DIR = "models/phase2"
TARGET_COL = "Label"
RANDOM_STATE = 42
SGD_CLASS_LIMIT = 10

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def normalize_classifier_name(value: str) -> str:
    value = value.strip().lower()
    mapping = {
        "svm": "SVM",
        "knn": "kNN",
        "k-nearest neighbors": "kNN",
        "k-nearest neighbours": "kNN",
        "dt": "DT",
        "decision tree": "DT",
        "rf": "RF",
        "random forest": "RF",
        "mlp": "MLP",
        "multilayer perceptron": "MLP",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported classifier: {value}")
    return mapping[value]


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"N/A", "None"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("(") and value.endswith(")"):
        return ast.literal_eval(value)
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return value


def parse_hyperparameter_string(param_text: str, classifier: str) -> Dict[str, Any]:
    pairs = re.findall(
        r"([A-Za-z_]+)\s*=\s*(\([^)]*\)|\[[^\]]*\]|[^,]+)", param_text)
    params = {key.strip(): parse_scalar(raw_value.strip())
              for key, raw_value in pairs}

    if classifier == "kNN" and "k" in params:
        params["n_neighbors"] = int(params.pop("k"))
    if classifier == "MLP" and "hidden_layers" in params:
        params["hidden_layer_sizes"] = params.pop("hidden_layers")
    if classifier == "MLP" and "lr" in params:
        params["learning_rate_init"] = params.pop("lr")

    return params


def load_phase2_best_hyperparameters(path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(path)
    required_columns = {"Classifier", "Phase", "Best Hyperparameters"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in hyperparameter table: {sorted(missing)}")

    phase2_df = df[df["Phase"].astype(
        str).str.strip().eq("After FS/DR")].copy()
    if phase2_df.empty:
        raise ValueError("No 'After FS/DR' rows found in hyperparameter table")

    best_params: Dict[str, Dict[str, Any]] = {}
    for _, row in phase2_df.iterrows():
        classifier = normalize_classifier_name(str(row["Classifier"]))
        best_params[classifier] = parse_hyperparameter_string(
            str(row["Best Hyperparameters"]),
            classifier,
        )
    return best_params


# Phase 2 feature selection copied from run_all.py logic.
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
        rf,
        X_sc,
        y_train,
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
            ::-1][: max(1, len(importances) // 2)]

    return list(X_train.columns[selected_idx])


def make_svm_pipeline(params: Dict[str, Any], n_classes: int, n_train: int) -> Pipeline:
    kernel = str(params.get("kernel", "linear")).lower()
    c_value = float(params.get("C", 1.0))

    if kernel == "linear":
        if n_classes > SGD_CLASS_LIMIT:
            alpha = 1.0 / (c_value * n_train)
            estimator = SGDClassifier(
                loss="hinge",
                alpha=alpha,
                max_iter=1000,
                tol=1e-3,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=1,
            )
        else:
            estimator = LinearSVC(
                C=c_value,
                max_iter=2000,
                dual=False,
                tol=1e-3,
                random_state=RANDOM_STATE,
            )
    elif kernel == "rbf":
        estimator = SVC(
            C=c_value,
            kernel="rbf",
            gamma=params.get("gamma", "scale"),
            random_state=RANDOM_STATE,
            cache_size=1000,
        )
    else:
        raise ValueError(
            f"Unsupported SVM kernel for final test pipeline: {kernel}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svm", estimator),
    ])


def make_knn_pipeline(params: Dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            metric=str(params["metric"]),
        )),
    ])


def make_dt_pipeline(params: Dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("dt", DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            criterion=str(params.get("criterion", "gini")),
            random_state=RANDOM_STATE,
        )),
    ])


def make_rf_pipeline(params: Dict[str, Any]) -> Pipeline:
    rf_kwargs = {
        "n_estimators": int(params.get("n_estimators", 100)),
        "max_depth": params.get("max_depth"),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params.get("min_samples_split") is not None:
        rf_kwargs["min_samples_split"] = int(params["min_samples_split"])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(**rf_kwargs)),
    ])


def make_mlp_pipeline(params: Dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64,)),
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            alpha=float(params.get("alpha", 0.0001)),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )),
    ])


def build_pipeline(classifier: str, params: Dict[str, Any], n_classes: int, n_train: int) -> Pipeline:
    if classifier == "SVM":
        return make_svm_pipeline(params, n_classes, n_train)
    if classifier == "kNN":
        return make_knn_pipeline(params)
    if classifier == "DT":
        return make_dt_pipeline(params)
    if classifier == "RF":
        return make_rf_pipeline(params)
    if classifier == "MLP":
        return make_mlp_pipeline(params)
    raise ValueError(f"Unsupported classifier: {classifier}")


def get_dataset_folders(data_dir: str) -> List[str]:
    folders = [f for f in os.listdir(
        data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    def sort_key(value: str) -> Tuple[int, Any]:
        return (0, int(value)) if value.isdigit() else (1, value.lower())

    return sorted(folders, key=sort_key)


def save_model(pipeline: Pipeline, dataset_name: str, classifier_name: str) -> None:
    """Save trained pipeline to disk."""
    model_path = os.path.join(
        MODELS_DIR, f"{dataset_name}_{classifier_name.lower()}.pkl")
    joblib_dump(pipeline, model_path)
    print(f"  Model saved: {model_path}")


def evaluate_single_dataset(classifier: str, params: Dict[str, Any], folder: str) -> Dict[str, Any]:
    train_path = os.path.join(DATA_DIR, folder, "train.csv")
    test_path = os.path.join(DATA_DIR, folder, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing train.csv for dataset folder: {folder}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Missing test.csv for dataset folder: {folder}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_full = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test_full = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    dataset_name = f"dataset_{folder}"
    n_classes = y_train.nunique()

    start = time.time()

    selected_features = select_features(X_train_full, y_train)
    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    pipeline = build_pipeline(
        classifier, params, n_classes=n_classes, n_train=len(X_train))
    pipeline.fit(X_train, y_train)

    # SAVE MODEL
    save_model(pipeline, dataset_name, classifier)

    y_pred = pipeline.predict(X_test)
    elapsed = round(time.time() - start, 2)

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1_macro = round(
        f1_score(y_test, y_pred, average="macro", zero_division=0), 4)

    labels = sorted(pd.unique(pd.concat([y_train, y_test], ignore_index=True)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    cm_path = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_{classifier.lower()}_confusion_matrix.csv",
    )
    cm_df.to_csv(cm_path)

    features_path = os.path.join(
        RESULTS_DIR,
        f"{dataset_name}_{classifier.lower()}_selected_features.json",
    )
    with open(features_path, "w", encoding="utf-8") as handle:
        json.dump(selected_features, handle, indent=2)

    return {
        "Dataset": dataset_name,
        "Phase": "After FS/DR",
        "Classifier": classifier,
        "Test_Accuracy": accuracy,
        "Test_F1_Macro": f1_macro,
        "Hyperparameters": json.dumps(params, sort_keys=True),
        "Train_Rows": len(X_train_full),
        "Test_Rows": len(X_test_full),
        "Num_Features_Before": X_train_full.shape[1],
        "Num_Features_After": len(selected_features),
        "Selected_Features_File": features_path,
        "Confusion_Matrix_File": cm_path,
        "Num_Classes": int(n_classes),
        "Time_Seconds": elapsed,
    }


def run_classifier(classifier: str, params: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    folders = get_dataset_folders(DATA_DIR)
    if not folders:
        raise ValueError(f"No dataset folders found in: {DATA_DIR}")

    print(f"\n{'=' * 80}")
    print(f"Classifier: {classifier}")
    print(f"Phase     : After FS/DR (WITH MODEL SAVING)")
    print(f"Params    : {params}")
    print(f"{'=' * 80}")

    for folder in folders:
        dataset_name = f"dataset_{folder}"
        try:
            result = evaluate_single_dataset(classifier, params, folder)
            rows.append(result)
            print(
                f"{dataset_name:<12} | "
                f"Feat={result['Num_Features_After']}/{result['Num_Features_Before']} | "
                f"Acc={result['Test_Accuracy']:.4f} | "
                f"F1={result['Test_F1_Macro']:.4f} | "
                f"Time={result['Time_Seconds']:.2f}s"
            )
        except Exception as exc:
            print(f"Failed on {dataset_name}: {exc}")
            raise

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(
        RESULTS_DIR, f"all_phase2_{classifier.lower()}_final_test_results.csv")
    result_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Phase 2 test-set evaluation using chosen best hyperparameters and permutation-importance feature selection, WITH MODEL SAVING."
    )
    parser.add_argument(
        "--classifier",
        choices=["svm", "knn", "dt", "rf", "mlp", "all"],
        default="all",
        help="Classifier to run. Default: all",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"Root directory containing dataset folders. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--hyperparams",
        default=HYPERPARAMETER_TABLE,
        help=f"CSV file containing best hyperparameters. Default: {HYPERPARAMETER_TABLE}",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIR
    args = parse_args()
    DATA_DIR = args.data_dir

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    if not os.path.exists(args.hyperparams):
        raise FileNotFoundError(
            f"Hyperparameter table not found: {args.hyperparams}")

    phase2_params = load_phase2_best_hyperparameters(args.hyperparams)

    if args.classifier == "all":
        classifiers = ["SVM", "kNN", "DT", "RF", "MLP"]
    else:
        classifiers = [normalize_classifier_name(args.classifier)]

    all_frames: List[pd.DataFrame] = []
    for classifier in classifiers:
        if classifier not in phase2_params:
            raise ValueError(
                f"No Phase 2 hyperparameters found for classifier: {classifier}")
        all_frames.append(run_classifier(
            classifier, phase2_params[classifier]))

    combined_df = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(
        RESULTS_DIR, "all_phase2_final_test_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined results saved: {combined_path}")
    print(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
