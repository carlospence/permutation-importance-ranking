#!/usr/bin/env python3
"""
Interactive Model Tester & Model Loader
=================================================
An interactive model tester and model loader utility.
Load saved trained models and test them on either existing dataset test files or custom data.

Usage:
    python utils/combined_interactive_model_tester.py

Features:
    - Choose Phase (1 or 2)
    - View available models
    - Load any saved model
    - Test on existing test data or custom CSV file
    - Display predictions and metrics
    - Option to test multiple models in one session
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
from joblib import load as joblib_load
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from pathlib import Path


# ============================================================================
# MODEL LOADER FUNCTIONS
# ============================================================================

# Constants
MODELS_BASE_DIR = "models"
VALID_PHASES = {"phase1", "phase2"}
VALID_CLASSIFIERS = {"svm", "knn", "dt", "rf", "mlp"}
DATA_DIR = "data"
RESULTS_DIR = os.path.join("results", "predictions", "interactive_tests")


def load_model(phase: str, dataset_name: str, classifier: str) -> Pipeline:
    """
    Load a saved trained model from disk.

    Parameters
    ----------
    phase : str
        Either "phase1" or "phase2"
    dataset_name : str
        Dataset identifier, e.g., "dataset_1", "dataset_5", "dataset_16"
    classifier : str
        Classifier name: "svm", "knn", "dt", "rf", or "mlp"

    Returns
    -------
    Pipeline
        The trained scikit-learn Pipeline object

    Raises
    ------
    ValueError
        If phase, dataset_name, or classifier are invalid
    FileNotFoundError
        If the model file does not exist
    """
    # Validate inputs
    if phase.lower() not in VALID_PHASES:
        raise ValueError(f"phase must be one of {VALID_PHASES}, got '{phase}'")

    if classifier.lower() not in VALID_CLASSIFIERS:
        raise ValueError(
            f"classifier must be one of {VALID_CLASSIFIERS}, got '{classifier}'")

    if not isinstance(dataset_name, str) or not dataset_name.startswith("dataset_"):
        raise ValueError(
            f"dataset_name should be 'dataset_X', got '{dataset_name}'")

    # Construct path
    phase_lower = phase.lower()
    classifier_lower = classifier.lower()
    model_path = os.path.join(
        MODELS_BASE_DIR,
        phase_lower,
        f"{dataset_name}_{classifier_lower}.pkl"
    )

    # Check existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Make sure to run the corresponding training script first:\n"
            f"  python phase1/run_phase1_final_test_with_save.py  (for phase1)\n"
            f"  python phase2/run_phase2_final_test_with_save.py  (for phase2)"
        )

    # Load and return
    model = joblib_load(model_path)
    print(f"✓ Loaded model: {model_path}")
    return model


def predict_on_data(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data using a loaded model.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn Pipeline object
    X : pd.DataFrame
        Input features (no label column)

    Returns
    -------
    np.ndarray
        Predicted labels
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

    if X.empty:
        raise ValueError("X cannot be empty")

    # Attempt to align features with model's expected features
    try:
        # Try to get feature names from the pipeline
        try:
            expected_features = model.get_feature_names_out()
        except Exception:
            # If the final estimator doesn't support get_feature_names_out,
            # try to get it from the preprocessing steps
            expected_features = model[:-1].get_feature_names_out()

        if list(X.columns) != list(expected_features):
            print(
                f"⚠ Realigning {X.shape[1]} features to match model's {len(expected_features)} expected features")
            X = X[expected_features]
            print(f"✓ Features aligned successfully")
    except Exception as e:
        print(f"⚠ Could not align features: {e}")

    predictions = model.predict(X)
    print(f"✓ Generated {len(predictions)} predictions")
    return predictions


def predict_proba_on_data(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Get probability predictions on new data using a loaded model.
    Only works if the model supports predict_proba (SVM with probability=True, etc.).

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn Pipeline object
    X : pd.DataFrame
        Input features (no label column)

    Returns
    -------
    np.ndarray
        Probability predictions (n_samples, n_classes)

    Raises
    ------
    AttributeError
        If the model does not support predict_proba
    """
    if not hasattr(model, 'predict_proba'):
        estimator_name = model.named_steps.get('svm') or model.named_steps.get('knn') \
            or model.named_steps.get('dt') or model.named_steps.get('rf') \
            or model.named_steps.get('mlp')
        raise AttributeError(
            f"Model does not support predict_proba. "
            f"Estimator: {type(estimator_name).__name__}"
        )

    # Attempt to align features with model's expected features
    try:
        # Try to get feature names from the pipeline
        try:
            expected_features = model.get_feature_names_out()
        except Exception:
            # If the final estimator doesn't support get_feature_names_out,
            # try to get it from the preprocessing steps
            expected_features = model[:-1].get_feature_names_out()

        if list(X.columns) != list(expected_features):
            print(
                f"⚠ Realigning {X.shape[1]} features to match model's {len(expected_features)} expected features")
            X = X[expected_features]
            print(f"✓ Features aligned successfully")
    except Exception as e:
        print(f"⚠ Could not align features: {e}")

    probabilities = model.predict_proba(X)
    print(
        f"✓ Generated probability predictions for {len(probabilities)} samples")
    return probabilities


def evaluate_on_data(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a model on labeled data.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn Pipeline object
    X : pd.DataFrame
        Input features (no label column)
    y : pd.Series
        Target labels

    Returns
    -------
    dict
        Dictionary with keys: {accuracy, f1_macro, n_samples, n_classes}
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            f"y must be a pandas Series or numpy array, got {type(y)}")

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length, got {len(X)} and {len(y)}")

    # Attempt to align features with model's expected features
    try:
        try:
            expected_features = model.get_feature_names_out()
        except Exception:
            # If the final estimator doesn't support get_feature_names_out,
            # try to get it from the preprocessing steps
            expected_features = model[:-1].get_feature_names_out()

        if list(X.columns) != list(expected_features):
            X = X[expected_features]
    except Exception as e:
        pass  # Silently continue if feature alignment fails

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1_macro = f1_score(y, predictions, average="macro", zero_division=0)
    n_classes = len(np.unique(y))
    cm = confusion_matrix(y, predictions)

    results = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "n_samples": len(X),
        "n_classes": n_classes,
        "confusion_matrix": cm,
        "predictions": predictions,
        "true_labels": y,
    }

    print(f"✓ Evaluation Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print(
        f"  Samples: {results['n_samples']}, Classes: {results['n_classes']}")

    return results


def list_available_models(phase: str) -> Dict[str, list]:
    """
    List all available saved models for a given phase.

    Parameters
    ----------
    phase : str
        Either "phase1" or "phase2"

    Returns
    -------
    dict
        Dictionary mapping dataset names to list of classifiers
    """
    if phase.lower() not in VALID_PHASES:
        raise ValueError(f"phase must be one of {VALID_PHASES}, got '{phase}'")

    phase_dir = os.path.join(MODELS_BASE_DIR, phase.lower())

    if not os.path.exists(phase_dir):
        return {}

    available = {}
    for pkl_file in os.listdir(phase_dir):
        if pkl_file.endswith(".pkl"):
            # Extract dataset_name and classifier from filename
            name_parts = pkl_file.replace(".pkl", "").rsplit("_", 1)
            if len(name_parts) == 2:
                dataset_name, classifier = name_parts
                if dataset_name not in available:
                    available[dataset_name] = []
                available[dataset_name].append(classifier)

    # Sort classifiers in each list
    for dataset_name in available:
        available[dataset_name] = sorted(available[dataset_name])

    return available


def load_test_data_from_csv(csv_path: str, label_col: str = "Label") -> tuple:
    """
    Load test data from a CSV file, separating features and labels.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    label_col : str
        Name of the label column

    Returns
    -------
    tuple
        (X, y) where X is features DataFrame and y is labels Series
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in CSV. Available: {df.columns.tolist()}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    print(
        f"✓ Loaded data from {csv_path}: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def get_model_feature_info(model: Pipeline) -> Dict[str, Any]:
    """
    Extract and display model's expected features for debugging.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn Pipeline object

    Returns
    -------
    dict
        Information about model's expected features
    """
    info = {}
    try:
        # Try to get feature names from the pipeline
        expected_features = model.get_feature_names_out()
        info["feature_names"] = list(expected_features)
        info["n_features"] = len(expected_features)
    except Exception as e:
        # If the final estimator doesn't support get_feature_names_out,
        # try to get it from the preprocessing steps
        try:
            expected_features = model[:-1].get_feature_names_out()
            info["feature_names"] = list(expected_features)
            info["n_features"] = len(expected_features)
        except Exception as e2:
            info["feature_names"] = None
            info["n_features"] = None
            info["error"] = f"Could not get feature names: {str(e2)}"
    return info


def display_model_feature_info(model: Pipeline, X_test: pd.DataFrame):
    """
    Display model's expected features vs actual test data features for debugging.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn Pipeline object
    X_test : pd.DataFrame
        Test data features
    """
    print("\n" + "─" * 70)
    print("  Model Feature Information (Debug)")
    print("─" * 70)

    info = get_model_feature_info(model)

    if info.get("feature_names"):
        print(f"\n✓ Model expects {info['n_features']} features:")
        for i, fname in enumerate(info["feature_names"], 1):
            print(f"    {i:2d}. {fname}")
    else:
        print(
            f"⚠ Could not retrieve model's expected features: {info.get('error')}")

    print(f"\n✓ Test data has {X_test.shape[1]} features:")

    """Print Test data features with indices for debugging. Commented out to reduce clutter, but can be uncommented if needed."""
    # for i, fname in enumerate(X_test.columns, 1):
    #     print(f"    {i:2d}. {fname}")

    if info.get("feature_names"):
        if set(info["feature_names"]) == set(X_test.columns):
            if list(info["feature_names"]) == list(X_test.columns):
                print("\n✓ Features match and are in same order")
            else:
                print(
                    "\n⚠ Features match but are in DIFFERENT order - will be reordered automatically")
        else:
            missing_in_test = set(info["feature_names"]) - set(X_test.columns)
            missing_in_model = set(X_test.columns) - set(info["feature_names"])
            if missing_in_test:
                print(f"\n❌ Missing in test data: {missing_in_test}")
            if missing_in_model:
                print(f"\n❌ Extra in test data: {missing_in_model}")


# ============================================================================
# INTERACTIVE MODEL TESTER FUNCTIONS
# ============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")


def format_confusion_matrix_ascii_box(cm: np.ndarray, n_classes: int) -> str:
    """
    Format confusion matrix as ASCII box (Format 2).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    n_classes : int
        Number of classes

    Returns
    -------
    str
        Formatted confusion matrix string
    """
    lines = []

    # Header
    header = "Predicted→ | "
    for i in range(n_classes):
        header += f"{i:4d} "
    lines.append("┌" + "─" * (len(header) + 8) + "┐")
    lines.append("│ " + header + "│")
    lines.append("├" + "─" * (len(header) + 8) + "┤")

    # Data rows
    for i in range(n_classes):
        row = f"│ Actual {i}  | "
        for j in range(n_classes):
            row += f"{cm[i, j]:4d} "
        row += "│"
        lines.append(row)

    lines.append("└" + "─" * (len(header) + 8) + "┘")

    return "\n".join(lines)


def format_confusion_matrix_with_percentages(cm: np.ndarray, n_classes: int) -> str:
    """
    Format confusion matrix with percentages (Format 4).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    n_classes : int
        Number of classes

    Returns
    -------
    str
        Formatted confusion matrix string with percentages
    """
    lines = []

    # Header
    header = "         Predicted→ | "
    for i in range(n_classes):
        header += f"{i:8d} "
    header += "| Total"
    lines.append(header)
    lines.append("Actual↓  | " + "─" * (len(header) - 10))

    # Data rows
    for i in range(n_classes):
        row = f"   {i}     | "
        row_sum = cm[i].sum()
        for j in range(n_classes):
            count = cm[i, j]
            if row_sum > 0:
                pct = (count / row_sum) * 100
                row += f"{count:3d}({pct:3.0f}%) "
            else:
                row += f"{count:3d}(  0%) "
        row += f"| {row_sum:5d}"
        lines.append(row)

    # Bottom separator and column totals
    lines.append("         | " + "─" * (8 * n_classes + 8))
    bottom = "Total    | "
    grand_total = cm.sum()
    for j in range(n_classes):
        col_sum = cm[:, j].sum()
        bottom += f"{col_sum:8d} "
    bottom += f"| {grand_total:5d}"
    lines.append(bottom)

    return "\n".join(lines)


def get_phase() -> str:
    """
    Interactive prompt to select a phase.

    Returns
    -------
    str
        Either "phase1" or "phase2"
    """
    print_section("Step 1: Select Phase")
    print("Available phases:")
    print("  1) Phase 1 (Before FS/DR)")
    print("  2) Phase 2 (After FS/DR)")

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            return "phase1"
        elif choice == "2":
            return "phase2"
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def display_available_models(phase: str):
    """Display all available models for a phase."""
    print_section("Available Models")

    try:
        available = list_available_models(phase)
        if not available:
            print(f"❌ No models found for {phase}.")
            print(
                f"   Run 'python phase1/run_phase1_final_test_with_save.py' to generate models.")
            return None

        print(f"Found {len(available)} datasets with saved models:\n")
        for dataset_name in sorted(available.keys()):
            classifiers = available[dataset_name]
            print(f"  {dataset_name}: {', '.join(classifiers)}")

        return available
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return None


def get_dataset_choice(available: dict) -> Optional[str]:
    """
    Interactive prompt to select a dataset.

    Parameters
    ----------
    available : dict
        Available models dictionary from list_available_models()

    Returns
    -------
    str or None
        Dataset name like "dataset_1", or None if invalid
    """
    print_section("Step 2: Select Dataset")

    datasets = sorted(available.keys())
    print("Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i:2}) {dataset}")

    while True:
        try:
            choice = input("\nEnter dataset number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                dataset = datasets[idx]
                print(f"✓ Selected: {dataset}")
                return dataset
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(datasets)}.")
        except ValueError:
            print("❌ Please enter a valid number.")


def get_classifier_choice(available: dict, dataset: str) -> Optional[str]:
    """
    Interactive prompt to select a classifier.

    Parameters
    ----------
    available : dict
        Available models dictionary
    dataset : str
        Selected dataset

    Returns
    -------
    str or None
        Classifier name, or None if invalid
    """
    print_section("Step 3: Select Classifier")

    classifiers = available[dataset]
    print(f"Available classifiers for {dataset}:")
    for i, clf in enumerate(classifiers, 1):
        clf_full = {
            "dt": "Decision Tree",
            "knn": "k-Nearest Neighbors",
            "mlp": "Neural Network (MLP)",
            "rf": "Random Forest",
            "svm": "Support Vector Machine",
        }
        print(f"  {i}) {clf.upper():5} - {clf_full.get(clf, clf)}")

    while True:
        try:
            choice = input("\nEnter classifier number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(classifiers):
                clf = classifiers[idx]
                print(f"✓ Selected: {clf.upper()}")
                return clf
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(classifiers)}.")
        except ValueError:
            print("❌ Please enter a valid number.")


def get_test_data_source(dataset: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    """
    Interactive prompt to choose between existing test data or custom data.

    Parameters
    ----------
    dataset : str
        Selected dataset (e.g., "dataset_5")

    Returns
    -------
    tuple
        (X_test, y_test, source_name) where y_test can be None
    """

    print_section("Step 4: Choose Test Data Source")
    print("Options:")
    print("  1) Use existing dataset test file (data/X/test.csv)")
    print("  2) Use custom CSV file")

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "1":
            return load_existing_test_data(dataset)
        elif choice == "2":
            return load_custom_test_data()
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def load_existing_test_data(dataset: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    """Load test data from existing dataset folder."""
    print_section("Loading Existing Test Data")

    # Extract dataset number from "dataset_X"
    try:
        dataset_num = dataset.split("_")[1]
        test_path = os.path.join(DATA_DIR, dataset_num, "test.csv")

        if not os.path.exists(test_path):
            print(f"❌ Test file not found: {test_path}")
            return None, None, ""

        X_test, y_test = load_test_data_from_csv(test_path)
        print(
            f"✓ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test, f"Existing dataset ({test_path})"

    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None, ""


def load_custom_test_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str]:
    """Load test data from custom CSV file."""
    print_section("Load Custom Data")

    csv_path = input(
        "Enter path to CSV file (or press Enter to skip): ").strip()

    if not csv_path:
        print("⊘ Skipped")
        return None, None, ""

    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None, None, ""

    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")

        # Check if there's a Label column
        if "Label" in df.columns:
            X_test, y_test = load_test_data_from_csv(csv_path)
            print("✓ Label column found - can provide evaluation metrics")
            return X_test, y_test, csv_path
        else:
            X_test = df
            print("⊘ No 'Label' column - will provide predictions only (no metrics)")
            return X_test, None, csv_path

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None, None, ""


def test_model(phase: str, dataset: str, classifier: str,
               X_test: pd.DataFrame, y_test: Optional[pd.Series],
               data_source: str):
    """
    Load model and run predictions/evaluation.

    Parameters
    ----------
    phase : str
        Phase ("phase1" or "phase2")
    dataset : str
        Dataset name
    classifier : str
        Classifier name
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series or None
        Test labels (optional)
    data_source : str
        Description of data source
    """

    print_section(f"Testing {classifier.upper()} on {dataset}")

    try:
        # Load model
        print("Loading model...")
        model = load_model(phase, dataset, classifier)

        # Display model's feature information for debugging
        display_model_feature_info(model, X_test)

        # Make predictions
        print("\nMaking predictions...")
        predictions = predict_on_data(model, X_test)

        # Display predictions summary
        print_section("Prediction Summary")
        unique_predictions = np.unique(predictions)
        print(f"Predicted classes: {sorted(unique_predictions)}")
        print(f"Predictions distribution:")
        for cls in sorted(unique_predictions):
            count = (predictions == cls).sum()
            pct = 100.0 * count / len(predictions)
            print(f"  Class {cls}: {count:4d} ({pct:5.1f}%)")

        # Try to get probabilities if available
        try:
            print("\nComputing prediction probabilities...")
            probas = predict_proba_on_data(model, X_test)
            print(f"✓ Probability estimates available (shape: {probas.shape})")
        except AttributeError:
            print("⊘ Model does not support probability estimates")

        # Evaluate if labels available
        if y_test is not None:
            print_section("Evaluation Metrics")
            results = evaluate_on_data(model, X_test, y_test)
            print("\nMetrics Summary:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  F1 (macro): {results['f1_macro']:.4f}")
            print(f"  Samples:   {results['n_samples']}")
            print(f"  Classes:   {results['n_classes']}")

            # Display confusion matrix in Format 2 (ASCII Box)
            print("\n" + "─" * 70)
            print("  Confusion Matrix: Simple ASCII Box")
            print("─" * 70)
            cm_ascii = format_confusion_matrix_ascii_box(
                results['confusion_matrix'], results['n_classes'])
            print(cm_ascii)

            # Display confusion matrix in Format 4 (With Percentages)
            print("\n" + "─" * 70)
            print("  Confusion Matrix: With Percentages (Row-wise)")
            print("─" * 70)
            cm_pct = format_confusion_matrix_with_percentages(
                results['confusion_matrix'], results['n_classes'])
            print(cm_pct)
        else:
            print_section("Note")
            print("No labels available - classification metrics cannot be computed.")
            print("Predictions have been generated successfully.")

        # Save results option
        save_results(predictions, y_test, phase, dataset, classifier)

    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()


def save_results(predictions: np.ndarray, y_test: Optional[pd.Series],
                 phase: str, dataset: str, classifier: str):
    """Ask if user wants to save predictions to file."""
    print_section("Save Results")

    response = input("Save predictions to CSV? (y/n): ").strip().lower()
    if response != "y":
        return

    try:
        output_filename = f"predictions_{phase}_{dataset}_{classifier}.csv"
        formatted_date = datetime.now().strftime("%b-%d")
        output_path = os.path.join(
            RESULTS_DIR, formatted_date, output_filename)
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame({"Prediction": predictions})

        if y_test is not None:
            results_df["Actual"] = y_test.values
            results_df["Correct"] = predictions == y_test.values

        results_df.to_csv(file_path, index=False)
        print(f"✓ Predictions saved to: {output_filename}")

    except Exception as e:
        print(f"❌ Error saving predictions: {e}")


def main():
    """Main interactive loop."""
    clear_screen()
    print_header("Interactive Model Tester")
    print("Test saved trained models on existing or custom data\n")

    while True:
        try:
            # Step 1: Choose phase
            phase = get_phase()

            # Step 2: Display available models and select dataset
            available = display_available_models(phase)
            if not available:
                again = input("\nTry again? (y/n): ").strip().lower()
                if again != "y":
                    break
                clear_screen()
                print_header("Interactive Model Tester")
                continue

            dataset = get_dataset_choice(available)
            if not dataset:
                continue

            # Step 3: Select classifier
            classifier = get_classifier_choice(available, dataset)
            if not classifier:
                continue

            # Step 4: Choose test data source
            X_test, y_test, data_source = get_test_data_source(dataset)
            if X_test is None:
                print("❌ No valid test data loaded.")
                again = input("Try again? (y/n): ").strip().lower()
                if again == "y":
                    clear_screen()
                    print_header("Interactive Model Tester")
                continue

            # Step 5: Run test
            test_model(phase, dataset, classifier, X_test, y_test, data_source)

            # Ask to test another model
            print_section("Continue?")
            again = input("Test another model? (y/n): ").strip().lower()
            if again != "y":
                break

            clear_screen()
            print_header("nteractive Model Tester")

        except KeyboardInterrupt:
            print("\n\n⊘ Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            again = input("Try again? (y/n): ").strip().lower()
            if again != "y":
                break
            clear_screen()
            print_header("Interactive Model Tester")

    print_section("Thank You")
    print("Goodbye!")


if __name__ == "__main__":
    main()
