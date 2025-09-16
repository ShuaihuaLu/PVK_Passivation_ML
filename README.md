# PVK_Passivation_ML
Activate Learning Application to PVK Passivation Molecule Design

# NGBoost Regression for Property Prediction (Section B)

A complete pipeline to train an NGBoost regressor for workfunction and formation energy prediction with Bayesian hyperparameter optimization, evaluation, visualization, SHAP-based feature importance, and saving model outputs.

This repository contains:

- `NGBoost_Regression.py` — main script:
  - Loads training and prediction CSVs
  - Scales features with a named `StandardScaler`
  - Tunes NGBoost hyperparameters using `bayes_opt`
  - Trains final `NGBRegressor` on the training set
  - Evaluates on a validation split and reports R², MAE, MSE and 95% coverage of predictive intervals
  - Produces predictive distributions (mean, 95% CI, uncertainty) for new samples
  - Saves predictions to CSV and produces publication-style plots (performance, histogram, SHAP)
  - Saves trained model and scaler with `joblib`
- Example outputs (created by the script):
  - `material_predictions.csv`
  - `performance_plot.png`
  - `prediction_distribution.png`
  - `feature_importance.png`
  - `trained_ngb_model.pkl` and `feature_scaler.pkl`
  - `output.log`

---

## Input data format
### Training CSV
Expect columns in this order:
- `id, thickness, comp_A_pct, comp_B_pct, bandgap`
First column: identifier (string or numeric)
Columns 2..N+1: numeric features
Last column: numeric target (e.g., bandgap in eV)
### Prediction CSV
Same features but without the target:
- `material_id, feature_1, feature_2, ..., feature_N`

# Perovskite Bayesian Optimization (Section C)

This repository provides a Python script (`perovskite_optimizer.py`) that wraps a Bayesian Optimization loop for perovskite experiments using **nextorch**. The script implements **Scenario A**: you have a small labeled initial dataset which is used to initialize a surrogate model; at each iteration the model prints its Top-K recommended candidates, and the loop simulates performing experiments by reading a sequence of actual measured samples from the Excel file and updating the surrogate.

## What this script does

- Auto-detects columns from an Excel file (Sheet name configurable).
- Uses the first `n_initial_samples` labeled rows to initialize the surrogate (nextorch.Experiment).
- At every iteration:
  - Requests Top-K candidate recommendations from the model (`generate_next_point(n_candidates=top_k)`).
  - Prints the Top-K candidates and (if available) acquisition values.
  - Takes the next `n_samples_per_iteration` rows from the dataset as "actual experiments" (simulating measurement), and updates the surrogate with these measured Y values.
- After all iterations, prints final Top-K recommendations and saves a compact progress figure (`{Y_name}_optimization_progress.pdf` and `.png`).

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib
- nextorch (the script calls `nextorch.bo.Experiment`, `generate_next_point`, and `run_trial`; API differences between nextorch versions may require small adjustments)

Install common dependencies:

```bash
pip install numpy pandas matplotlib openpyxl
