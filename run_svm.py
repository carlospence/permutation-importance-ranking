import os
import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


DATA_DIR = "data"
RESULTS_DIR = "results/svm"
TARGET_COL = "Label"
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def build_svm_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ])


def get_param_grid():
    return {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale"]
    }


def evaluate_dataset(csv_path: str, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    print(f"\nStarting {dataset_name}")
    print(f"Shape: {X.shape}, classes: {y.value_counts().to_dict()}")

    dataset_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        fold_start = time.time()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = KFold(n_splits=9, shuffle=True, random_state=RANDOM_STATE)

        grid = GridSearchCV(
            estimator=build_svm_pipeline(),
            param_grid=get_param_grid(),
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        best_params = grid.best_params_

        fold_elapsed = time.time() - fold_start

        fold_rows.append({
            "Dataset": dataset_name,
            "Fold": fold,
            "Accuracy": accuracy,
            "F1": f1_macro,
            "Parameters": str(best_params),
            "Best_C": best_params.get("svm__C"),
            "Best_Kernel": best_params.get("svm__kernel"),
            "Best_Gamma": best_params.get("svm__gamma"),
            "Fold_Time_Seconds": round(fold_elapsed, 2)
        })

        print(
            f"[{dataset_name}] Fold {fold}/10 | "
            f"Accuracy={accuracy:.4f} | F1={f1_macro:.4f} | "
            f"Params={best_params} | Time={fold_elapsed:.1f}s"
        )

    folds_df = pd.DataFrame(fold_rows)

    total_elapsed = time.time() - dataset_start

    summary_df = pd.DataFrame([{
        "Dataset": dataset_name,
        "Accuracy Mean": folds_df["Accuracy"].mean(),
        "Accuracy Std": folds_df["Accuracy"].std(ddof=1),
        "F1 Mean": folds_df["F1"].mean(),
        "F1 Std": folds_df["F1"].std(ddof=1),
        "Accuracy Mean ± Std": f"{folds_df['Accuracy'].mean():.4f} ± {folds_df['Accuracy'].std(ddof=1):.4f}",
        "F1 Mean ± Std": f"{folds_df['F1'].mean():.4f} ± {folds_df['F1'].std(ddof=1):.4f}",
        "Total_Time_Seconds": round(total_elapsed, 2)
    }])

    return folds_df, summary_df


def save_results(folds_df: pd.DataFrame, summary_df: pd.DataFrame, dataset_name: str) -> None:
    folds_path = os.path.join(RESULTS_DIR, f"{dataset_name}_svm_folds.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{dataset_name}_svm_summary.csv")

    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {folds_path}")
    print(f"Saved: {summary_path}")


def combine_all_summaries(summary_frames: list[pd.DataFrame]) -> None:
    if not summary_frames:
        return

    all_summary_df = pd.concat(summary_frames, ignore_index=True)
    all_summary_path = os.path.join(RESULTS_DIR, "all_datasets_svm_summary.csv")
    all_summary_df.to_csv(all_summary_path, index=False)
    print(f"\nSaved combined summary: {all_summary_path}")


def get_dataset_folders(data_dir: str) -> list[str]:
    folders = [
        folder for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
    ]

    def sort_key(value: str):
        return (0, int(value)) if value.isdigit() else (1, value.lower())

    return sorted(folders, key=sort_key)


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    dataset_folders = get_dataset_folders(DATA_DIR)

    if not dataset_folders:
        raise ValueError(f"No dataset folders found in: {DATA_DIR}")

    all_summary_frames = []

    for folder in dataset_folders:
        csv_path = os.path.join(DATA_DIR, folder, "train.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping {folder}: train.csv not found")
            continue

        dataset_name = f"dataset_{folder}"

        try:
            folds_df, summary_df = evaluate_dataset(csv_path, dataset_name)
            save_results(folds_df, summary_df, dataset_name)
            all_summary_frames.append(summary_df)
        except Exception as exc:
            print(f"Failed on {dataset_name}: {exc}")

    combine_all_summaries(all_summary_frames)
    print("\nDone.")


if __name__ == "__main__":
    main()
