# PVK_Passivation_ML
Activate Learning Application to PVK Passivation Molecule Design

# Molecular Generator (Section A)

This repository contains a script to generate substituted benzene derivatives using RDKit. The pipeline performs two stages:

1. **Functionalization** — attach functional groups to benzene ring carbons (supported groups: `CN`, `CF3`, `CH3`, `OCH3`, `COOH`, `OH`).
2. **Halogenation** — substitute selected ring hydrogens with halogens (`F`, `Cl`, `Br`).

The script produces SDF files for each derivative and a grid image showing up to the first 16 generated derivatives for each base molecule.

## Files

- `benzene_derivatives.py` — main generation script (RDKit-based).
- Output directories created at runtime under the configured `output_dir` (default `output/`):
  - `output/sdf/` — SDF files for generated derivatives
  - `output/images/` — PNG grid images per base molecule

## Requirements

- Python 3.7+ (recommended)
- [RDKit](https://www.rdkit.org/) — install via conda:
  ```bash
  conda create -n rdkit-env -c conda-forge rdkit python=3.8
  conda activate rdkit-env
## Parameters to change

You can customize the generation by editing the `process_base_smiles(...)` call at the bottom of the script, or by importing and calling `process_base_smiles()` from another Python script.  
Below are the key parameters you can change, their types, purpose, and examples.

### Parameters

- **`base_smiles_dict`** — `dict[str, str]`  
  A dictionary mapping a short name (string) to a SMILES string of the base molecule.  
  Example:
  ```py
  base_smiles_dict = {
      "PEA": "C1=CC=C(C=C1)CCN",
      "PBA": "C1=CC=C(C=C1)CCCCN"
  }
  ## Parameters

### `max_substituents` — int (≥ 0)
Maximum number of functional groups to add to the benzene ring during the functionalization stage.

- `0` → functionalization is skipped and the original molecule is carried forward
- `1` → add up to one substituent (typical small combinatorial load)

**Examples:**
```python
max_substituents = 0   # skip functionalization
max_substituents = 1   # add up to one functional group per product
max_substituents = 2   # allow up to two substituents (combinatorics increases)

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
- `material_id, feature_1, feature_2, ..., feature_N, Target_property`
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
