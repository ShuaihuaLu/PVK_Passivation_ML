# PVK_Passivation_ML
Activate Learning Application to PVK Passivation Molecule Design
# Perovskite Bayesian Optimization (Scenario A)

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
