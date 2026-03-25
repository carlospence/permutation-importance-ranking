#!/usr/bin/env python3
"""
Interactive Model Tester Utility
==================================
A user-friendly interactive script to load saved trained models and test them on
either existing dataset test files or custom data.

Usage:
    python utils/interactive_model_tester.py

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
from typing import Optional, Tuple

import pandas as pd
import numpy as np

# Import from model_loader utility
try:
    from utils.model_loader import (
        load_model,
        predict_on_data,
        predict_proba_on_data,
        evaluate_on_data,
        list_available_models,
        load_test_data_from_csv,
    )
except ImportError as e:
    print(f"Error: Could not import model_loader. Make sure you're running from project root.")
    print(f"Details: {e}")
    sys.exit(1)


# Constants
DATA_DIR = "data"
CLASSIFIERS = ["svm", "knn", "dt", "rf", "mlp"]


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
        results_df = pd.DataFrame({"Prediction": predictions})

        if y_test is not None:
            results_df["Actual"] = y_test.values
            results_df["Correct"] = predictions == y_test.values

        results_df.to_csv(output_filename, index=False)
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
            print_header("Interactive Model Tester")

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
