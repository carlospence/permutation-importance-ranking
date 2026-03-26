# Permutation Importance Ranking

This project compares five classifiers on 16 tabular classification datasets and studies the effect of feature selection using permutation importance.

Phase 1 evaluates models on the original feature space. Phase 2 applies permutation-importance-based feature selection before training and compares the same classifiers again.

## Classifiers

- SVM
- kNN
- Decision Tree
- Random Forest
- MLP

## Project Goals

- Benchmark all five classifiers across 16 datasets.
- Measure accuracy and macro F1 with cross-validation.
- Compare performance before and after feature selection / dimensionality reduction.
- Save final test-set results, trained models, tables, and report figures.

## Repository Structure

```text
data/                  16 datasets, each with train.csv and test.csv
phase1/                Baseline training and evaluation scripts
phase2/                Feature-selection-based training and evaluation scripts
models/                Saved trained models for phase 1 and phase 2
results/               Cross-validation outputs, final test results, predictions
report_figures/        Generated plots and report-ready CSV tables
utils/                 Reporting, workbook-filling, and interactive model utilities
sample.ipynb           Notebook for experimentation
requirements.txt       Python dependencies
CS6735_PROJECT_COMPLETE.xlsx  Project workbook
```

## Data Format

Each dataset lives under `data/<dataset_id>/` and contains:

- `train.csv`
- `test.csv`

## Code Overview

**Our Code** trains and evaluates **all 5 classifiers simultaneously** (DT, kNN, MLP, RF, SVM) on all datasets using **nested cross-validation with hyperparameter tuning**. It unifies the individual classifier training into one consolidated script. The Phase 2 uses **Permutation Importance Feature Selection**. This is the key difference from Phase 1: features are selected once per fold, then shared across all classifiers to avoid expensive recomputation.

We have different folder for Phase 1 and Phase 2 codes with similar code structure.

## Methodology

### Phase 1

- Uses the original feature set.
- Runs 10-fold stratified cross-validation on each training set.
- Tunes model hyperparameters with a small inner validation split / grid search.
- Writes per-fold and summary metrics to `results/phase1/`.

Main script:

```bash
python phase1/run_all_classifiers.py
```

### Phase 2

- Computes permutation importance with a Random Forest once per fold.
- Keeps features whose importance is above the mean importance threshold.
- Retrains and evaluates all five classifiers using the reduced feature set.
- Writes outputs to `results/phase2/`.

Main script:

```bash
python phase2/run_all.py
```

**Save Results:**

- Per-fold: `dataset_X_all_folds.csv` (all classifiers, all folds)
- Per-dataset summary: `dataset_X_all_summary.csv` (summary stats per classifier)
- Combined: `all_datasets_all_classifiers_summary.csv` (all datasets + all classifiers)

## Output Files

```
results/combined/phase{1or2}/all/
├── dataset_1_all_folds.csv              # Per-fold results
├── dataset_1_all_summary.csv            # Dataset-1 summary
├── dataset_2_all_folds.csv
├── ...
└── all_datasets_all_classifiers_summary.csv  # Master summary
```

**Run_interactive.py** — an interactive script with full menu-driven evaluation.

## Features:

**Interactive Selection**

- **Datasets**: Enter single (e.g., `5`), multiple (e.g., `1,5,10`), or range (e.g., `1-5`)
- **Classifiers**: Enter single (e.g., `dt`), multiple (e.g., `dt,svm`), or all (e.g., `dt,knn,mlp,rf,svm`)
- Input validation with helpful error messages and available options displayed

**Multi-Selection Support**

- Datasets: `1` | `1,5,10` | `1-5`
- Classifiers: `dt` | `knn,rf` | `dt,svm,mlp`

**Usage:**

```bash
python phase1/run_interactive.py
```

Then follow the on-screen prompts. Example session:

```
📁 Available datasets: 1, 2, 3, ..., 16
📌 Enter dataset number(s): 1,5,10

🤖 Available classifiers: dt, knn, mlp, rf, svm
📌 Enter classifier(s): dt,svm

✓ Datasets: 1, 5, 10
✓ Classifiers: DT, SVM
Proceed with evaluation? (y/n): y
```

## Final Test Evaluation

After selecting hyperparameters from cross-validation, the project can run final train/test evaluation and save trained models.

## Command-Line Arguments

```bash
--classifier {svm|knn|dt|rf|mlp|all}  # Which classifier(s) to test (default: all)
--data-dir <path>                     # Dataset directory (default: data/)
--hyperparams <path>                  # Hyperparameter table CSV (default: report_figures/hyperparameter_table.csv)
```

Phase 1:

```bash
# Test all classifiers
python phase1/run_phase1_final_test_with_save.py --classifier all
```

```bash
# Test only SVM
python phase1/run_phase1_final_test_with_save.py --classifier svm
```

Phase 2:

```bash
# Test all classifiers
python phase2/run_phase2_final_test_with_save.py --classifier all
```

```bash
# Test only KNN
python phase2/run_phase2_final_test_with_save.py --classifier knn
```

These scripts produce:

- combined final test CSVs in `results/final_test/phase1/` and `results/final_test/phase2/`
- confusion matrices as CSV files
- saved models in `models/phase1/` and `models/phase2/`
- selected-feature JSON files for phase 2

**run_phase{1or2}_final_test_with_save.py** and **run_phase{1or2}_final_test.py** are **final test/evaluation script** for Phase 1 and 2. It loads the best hyperparameters discovered during Phase 1 or 2 training and evaluates each classifier on the actual test sets (separate from the training data used for hyperparameter tuning). The version **"With Save"** saves the model generated so that it can easily be loaded for predictions without retraining while the version without ***"With Save"*** does not save the model.

## Interactive Model Loader and Testing Utility

This is an **interactive model loader & testing utility** for a machine learning project. It allows you to load pre-trained models and test them on either existing datasets or custom data.

**Main Workflow (`main()`):**

1. Select phase (1 or 2)
2. Display available trained models
3. Choose a dataset and classifier
4. Load test data (existing or custom)
5. Load the model and generate predictions
6. Display results (predictions, probabilities, evaluation metrics, confusion matrix)
7. Optionally save results to CSV
8. Loop to test another model or exit

### Key Features:

- Feature alignment checks (warns if features don't match)
- Multiple confusion matrix formats
- Probability estimates when available
- Results saved with timestamps organized by date
- Full error handling with helpful error messages

The file can be executed in several ways:

## **1. From project root (recommended)**

```powershell
python utils/interactive_model_loader_tester.py
```

## **2. From within the utils directory**

```powershell
cd utils
python interactive_model_loader_tester.py
```

## **3. Using Python module execution**

```powershell
python -m utils.interactive_model_loader_tester
```

## **4. On Unix/Linux with shebang**

```bash
./utils/interactive_model_loader_tester.py
```

(Requires execute permissions: `chmod +x utils/interactive_model_loader_tester.py`)

---

## **What Happens When You Run It:**

The file has a `if __name__ == "__main__":` block that calls the `main()` function, which launches an **interactive menu** where you:

1. Select Phase (1 or 2)
2. View available trained models
3. Choose a dataset and classifier
4. Select test data (existing or custom)
5. Run predictions and view evaluation metrics
6. Optionally save results to CSV
7. Test another model or exit

**Requirements before running:**

- Pre-trained models must exist in phase1 and phase2 directories
- Test data available in `data/<N>/test.csv` format or a custom CSV file
