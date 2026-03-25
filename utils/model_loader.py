"""
Model Loader Utility Module
============================
Provides functions to load and use saved trained models from Phase 1 and Phase 2.

Usage Example:
    from utils.model_loader import load_model, predict_on_data

    # Load a saved model
    model = load_model("phase1", "dataset_5", "svm")

    # Use on test data
    X_test = pd.read_csv("test_data.csv").drop("Label", axis=1)
    predictions = predict_on_data(model, X_test)

    # Evaluate if labels are available
    y_test = pd.read_csv("test_data.csv")["Label"]
    results = evaluate_on_data(model, X_test, y_test)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Constants
MODELS_BASE_DIR = "models"
VALID_PHASES = {"phase1", "phase2"}
VALID_CLASSIFIERS = {"svm", "knn", "dt", "rf", "mlp"}


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

    Examples
    --------
    >>> model = load_model("phase1", "dataset_1", "svm")
    >>> predictions = model.predict(X_test)
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

    Examples
    --------
    >>> model = load_model("phase1", "dataset_1", "svm")
    >>> X_test = pd.read_csv("test.csv").drop("Label", axis=1)
    >>> predictions = predict_on_data(model, X_test)
    >>> print(predictions)
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

    if X.empty:
        raise ValueError("X cannot be empty")

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

    Examples
    --------
    >>> model = load_model("phase1", "dataset_1", "knn")
    >>> X_test = pd.read_csv("test.csv").drop("Label", axis=1)
    >>> probabilities = predict_proba_on_data(model, X_test)
    """
    if not hasattr(model, 'predict_proba'):
        estimator_name = model.named_steps.get('svm') or model.named_steps.get('knn') \
            or model.named_steps.get('dt') or model.named_steps.get('rf') \
            or model.named_steps.get('mlp')
        raise AttributeError(
            f"Model does not support predict_proba. "
            f"Estimator: {type(estimator_name).__name__}"
        )

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

    Examples
    --------
    >>> model = load_model("phase1", "dataset_1", "svm")
    >>> X_test = pd.read_csv("test.csv").drop("Label", axis=1)
    >>> y_test = pd.read_csv("test.csv")["Label"]
    >>> results = evaluate_on_data(model, X_test, y_test)
    >>> print(f"Accuracy: {results['accuracy']:.4f}, F1: {results['f1_macro']:.4f}")
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            f"y must be a pandas Series or numpy array, got {type(y)}")

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length, got {len(X)} and {len(y)}")

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1_macro = f1_score(y, predictions, average="macro", zero_division=0)
    n_classes = len(np.unique(y))

    results = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "n_samples": len(X),
        "n_classes": n_classes,
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

    Examples
    --------
    >>> models = list_available_models("phase1")
    >>> print(models)
    {'dataset_1': ['dt', 'knn', 'mlp', 'rf', 'svm'],
     'dataset_2': ['dt', 'knn', 'mlp', 'rf', 'svm'],
     ...}
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

    Examples
    --------
    >>> X_test, y_test = load_test_data_from_csv("data/1/test.csv")
    >>> model = load_model("phase1", "dataset_1", "svm")
    >>> results = evaluate_on_data(model, X_test, y_test)
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


if __name__ == "__main__":
    # Example usage
    print("Model Loader Utility Module")
    print("=" * 60)
    print("\nExample: Load model and evaluate on test data")
    print("-" * 60)
    print(">>> from utils.model_loader import load_model, evaluate_on_data, load_test_data_from_csv")
    print(">>> model = load_model('phase1', 'dataset_1', 'svm')")
    print(">>> X_test, y_test = load_test_data_from_csv('data/1/test.csv')")
    print(">>> results = evaluate_on_data(model, X_test, y_test)")
    print()
    print("Available functions:")
    print("  - load_model(phase, dataset_name, classifier)")
    print("  - predict_on_data(model, X)")
    print("  - predict_proba_on_data(model, X)")
    print("  - evaluate_on_data(model, X, y)")
    print("  - list_available_models(phase)")
    print("  - load_test_data_from_csv(csv_path, label_col)")
