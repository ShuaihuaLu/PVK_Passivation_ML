# PVK_Passivation_ML

Predictive ML datasets and examples for molecular passivation candidates (perovskite passivation project).

## Overview

This repository collects tabular molecular datasets used for machine-learning prediction of properties relevant to passivation of perovskite (PVK) surfaces. The Data/ folder contains CSVs for training/testing and for model predictions for two endpoints:

- Formation energy (FormationEnergy)
- Work function (WorkFunction)

Each row corresponds to one molecule (SMILES) and a set of computed molecular descriptors and/or targets/predictions.

## Repository structure

- Data/
  - `Train&test_FormationEnergy.csv` — training + test data for formation energy (includes descriptors and target values).
  - `Prediction_FormationEnergy.csv` — model prediction table(s) for formation energy (descriptor columns plus predictions).
  - `Train&test_WorkFunction.csv` — training + test data for work function.
  - `Prediction_WorkFunction.csv` — model prediction table(s) for work function.
- (Other code / notebooks can be added here: `notebooks/`, `src/`, etc.)

> Note: CSVs in `Data/` are the authoritative dataset files. Column lists are long; key columns that appear in the files include:
> `ID`, `file_name`, `SMILES`, `Molar_volume`, `HOMO`, `LUMO`, `HOMO_LUMO_Gap`, `Dipole_Moment`, `MaxAbsEStateIndex`, `MaxEStateIndex`, `MinAbsEStateIndex`, `MinEStateIndex`, `qed`, `SPS`, `MolWt`, `HeavyAtomMolWt`, `ExactMolWt`, `NumValenceElectrons`, ... (plus target/prediction columns in the train/prediction files).

## Quickstart — inspect the data

Install recommended packages (example):

```bash
python -m venv .venv
source .venv/bin/activate     # or .\.venv\Scripts\activate on Windows
pip install pandas numpy scikit-learn matplotlib seaborn
# Optional for advanced models / feature generation:
pip install xgboost lightgbm rdkit-pypi
```

Load and preview a CSV with pandas:

```python
import pandas as pd

df = pd.read_csv("Data/Train&test_FormationEnergy.csv")
print(df.shape)
print(df.columns.tolist()[:30])  # preview first columns
print(df.head())
```

## Example: simple modeling flow (sketch)

1. Load `Train&test_*.csv`.
2. Select numeric features (drop IDs, SMILES, file_name, any target/prediction columns).
3. Train a regressor (RandomForest / XGBoost / LightGBM).
4. Evaluate on test set using MAE / RMSE / R².
5. Save predictions to CSV for later comparison with `Prediction_*.csv`.

Minimal example (sketch):

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Data/Train&test_FormationEnergy.csv")
target_name = "FormationEnergy"  # adapt if column has a different name
# simple feature selection: numeric columns only (drop target)
X = df.select_dtypes("number").drop(columns=[target_name, "ID"], errors="ignore")
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
```

Adjust `target_name` and column selection according to the actual CSV header.

## Recommended workflow / tips

- Validate SMILES strings and generate additional descriptors (RDKit) if needed.
- Normalize/standardize features when using linear models.
- Use cross-validation with scaffold or stratified splits (if targets display heterogeneity).
- Track experiments with MLflow, Weights & Biases, or simple CSV logging.
- Keep random seeds and package versions to ensure reproducibility.

## Evaluation metrics

For regression endpoints such as formation energy and work function, consider:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Report performance on hold-out test set(s) and, if available, on an external validation set.

## Reproducibility

- Record package versions (pip freeze > requirements.txt).
- Fix random seeds in model training and splitting (e.g., `random_state=42`).
- Save model artifacts and the exact CSV used for training and evaluation.

## Adding notebooks / scripts

Suggested additions to the repo:
- `requirements.txt`
- `notebooks/EDA.ipynb` — exploratory data analysis
- `notebooks/Modeling_FormationEnergy.ipynb`
- `notebooks/Modeling_WorkFunction.ipynb`
- `src/train.py`, `src/predict.py` — CLI scripts for training and inference

I can create these files (requirements, example notebook, or script) on request.

## Contributing

- Open an issue to propose new features or report data issues.
- Provide clear descriptions and minimal reproducible examples for code changes.
- Add unit tests where appropriate.

## Contact & citation

If you use this dataset or models from this repository in publications, please reference the repository (add the repository citation entry once you create a citable DOI or paper).

## License

No LICENSE file is included in the repository by default. Add a license (MIT, Apache-2.0, etc.) to clarify reuse terms.

## Acknowledgements

This repo was prepared to hold molecular descriptor datasets and examples for ML-driven passivation screening. Thank you for contributing — open an issue if you'd like me to extend this README or generate example notebooks and scripts.