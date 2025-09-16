#!/usr/bin/env python3
"""
perovskite_optimizer.py

Perovskite Bayesian Optimization wrapper (Scenario A).

- Auto detects columns in an Excel file:
    Column 1: ID
    Column 2: Iteration number
    Columns 3..(-3): Input variables (X)
    Column -2: Target property (Y)
    Column -1: Target property std (Y_std)

- Requires an initial labeled dataset (n_initial_samples) to initialize the surrogate.
- At each iteration the model prints its Top-K recommended candidates (by
  calling nextorch's generate_next_point with n_candidates=top_k).
- The code then uses the next n_samples_per_iteration samples from the
  input dataset (simulating "do experiment and return measured Y") to update the surrogate.

Note: nextorch API (Experiment.input_data, generate_next_point, run_trial, etc.)
may vary between versions. If you hit API errors, please refer to your nextorch
version docs and adapt the calls where indicated.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# nextorch imports - ensure nextorch is installed in your Python environment
from nextorch import bo, doe, io, utils

# reproducible seed
GLOBAL_RANDOM_SEED = 25
np.random.seed(GLOBAL_RANDOM_SEED)


class PerovskiteOptimizer:
    """
    Bayesian optimization wrapper for perovskite experiments.

    Key features:
    - auto-detects column names from an Excel file
    - computes input bounds from observed data
    - initializes a nextorch Experiment with initial labeled data
    - at each iteration, prints Top-K model recommendations (generate_next_point)
    - updates the surrogate using a stored sequence of "actual" experiments
    """

    def __init__(self,
                 data_file: str = "data.xlsx",
                 random_seed: int = GLOBAL_RANDOM_SEED,
                 n_initial_samples: int = 5,
                 n_samples_per_iteration: int = 1,
                 acq_func_name: str = "qEI",
                 top_k: int = 5,
                 sheet_name: str = "Sheet1"):
        self.data_file = data_file
        self.random_seed = random_seed
        self.n_initial_samples = n_initial_samples
        self.n_samples_per_iteration = n_samples_per_iteration
        self.acq_func_name = acq_func_name
        self.top_k = top_k
        self.sheet_name = sheet_name

        np.random.seed(self.random_seed)

        # auto-detected names
        self.id_col = ""
        self.iter_col = ""
        self.X_names: List[str] = []
        self.Y_name = ""
        self.Y_std_name = ""

        # data containers
        self.df: Optional[pd.DataFrame] = None
        self.X_ranges: List[List[float]] = []
        self.Y_plot_range: List[float] = []
        self.Exp = None

        # tracking
        self.mean_responses: List[float] = []
        self.std_responses: List[float] = []
        self.Y_init_values: Optional[np.ndarray] = None

    def auto_detect_columns(self) -> None:
        """Auto-detect columns by position and perform basic checks."""
        if self.df is None:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

        cols = list(self.df.columns)
        n = len(cols)
        if n < 5:
            raise ValueError(f"Expected at least 5 columns (ID, Iter, X..., Y, Y_std); found {n}: {cols}")

        self.id_col = cols[0]
        self.iter_col = cols[1]
        self.X_names = cols[2:-2]
        self.Y_name = cols[-2]
        self.Y_std_name = cols[-1]

        # ensure X_names is a list
        if isinstance(self.X_names, str):
            self.X_names = [self.X_names]

        logger.info("Auto-detected columns:")
        logger.info(f"  ID column: {self.id_col}")
        logger.info(f"  Iteration column: {self.iter_col}")
        logger.info(f"  Input X columns ({len(self.X_names)}): {self.X_names}")
        logger.info(f"  Target Y: {self.Y_name}")
        logger.info(f"  Target Y std: {self.Y_std_name}")

    def load_data(self) -> pd.DataFrame:
        """Load Excel file and coerce numeric columns where reasonable."""
        abs_path = os.path.abspath(os.path.join(os.getcwd(), self.data_file))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Data file not found: {abs_path}")

        try:
            self.df = pd.read_excel(abs_path, sheet_name=self.sheet_name)
        except Exception as exc:
            logger.error(f"Failed to read Excel file: {exc}")
            raise

        logger.info(f"Loaded data from: {abs_path} shape={self.df.shape}")
        self.auto_detect_columns()

        # coerce numeric types for X, Y, Y_std, and iteration
        cols_to_cast = list(self.X_names) + [self.Y_name, self.Y_std_name, self.iter_col]
        for c in cols_to_cast:
            if c not in self.df.columns:
                logger.warning(f"Expected column '{c}' not found in data.")
                continue
            if not pd.api.types.is_numeric_dtype(self.df[c]):
                coerced = pd.to_numeric(self.df[c], errors="coerce")
                n_nan = coerced.isna().sum()
                if n_nan > 0:
                    logger.warning(f"Column '{c}' coerced to numeric; {n_nan} NaN(s) introduced.")
                self.df[c] = coerced

        return self.df

    def compute_bounds(self) -> List[List[float]]:
        """Compute parameter bounds from the dataset and determine Y plot range."""
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_data() first.")

        self.X_ranges = []
        logger.info("Computing parameter bounds:")
        for name in self.X_names:
            valid = self.df[name].dropna()
            if len(valid) == 0:
                logger.warning(f"No valid samples for {name}. Using fallback [0,1].")
                self.X_ranges.append([0.0, 1.0])
                continue
            lo = float(valid.min())
            hi = float(valid.max())
            if lo == hi:
                hi = lo + 1e-6
            self.X_ranges.append([lo, hi])
            logger.info(f"  {name}: [{lo:.6g}, {hi:.6g}]")

        # Y plot range: prefer using Y +/- Y_std if available
        y = self.df[self.Y_name] if self.Y_name in self.df.columns else pd.Series(dtype=float)
        ystd = self.df[self.Y_std_name] if self.Y_std_name in self.df.columns else pd.Series(dtype=float)

        if y.dropna().empty:
            self.Y_plot_range = [0.0, 1.0]
        else:
            if not ystd.dropna().empty:
                common = self.df[self.Y_name].notna() & self.df[self.Y_std_name].notna()
                if common.any():
                    y_vals = self.df.loc[common, self.Y_name]
                    ystd_vals = self.df.loc[common, self.Y_std_name]
                    y_min = (y_vals - ystd_vals).min()
                    y_max = (y_vals + ystd_vals).max()
                    self.Y_plot_range = [float(y_min), float(y_max)]
                else:
                    self.Y_plot_range = [float(y.min()), float(y.max())]
            else:
                y_min = float(y.min())
                y_max = float(y.max())
                pad = 0.05 * (abs(y_max - y_min) if (y_max - y_min) != 0 else max(1.0, abs(y_max)))
                self.Y_plot_range = [y_min - pad, y_max + pad]

        logger.info(f"Y plot range: {self.Y_plot_range}")
        return self.X_ranges

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training arrays:
        - sort by iteration number
        - take first n_initial_samples as the initial labeled set
        - return remaining arrays for iterative updating
        """
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_data() first.")

        required_cols = self.X_names + [self.Y_name, self.iter_col]
        clean = self.df.dropna(subset=required_cols).copy()
        logger.info(f"Data preparation: original_rows={len(self.df)} clean_rows={len(clean)}")
        if len(clean) == 0:
            raise ValueError("No complete rows after dropping missing X/Y/iteration.")

        X_all = clean[self.X_names].values
        Y_all = clean[self.Y_name].values.reshape(-1, 1)
        Y_std_all = clean[self.Y_std_name].values if self.Y_std_name in clean.columns else np.full(len(Y_all), np.nan)
        iter_all = clean[self.iter_col].values

        # sort by iteration
        idx = np.argsort(iter_all)
        X_all = X_all[idx]
        Y_all = Y_all[idx]
        Y_std_all = Y_std_all[idx]
        iter_all = iter_all[idx]

        available = len(X_all)
        if self.n_initial_samples > available:
            logger.warning(f"Requested initial samples {self.n_initial_samples} > available {available}. Reducing initial to {available}.")
            self.n_initial_samples = available

        X_init = X_all[:self.n_initial_samples]
        Y_init = Y_all[:self.n_initial_samples]
        Y_std_init = Y_std_all[:self.n_initial_samples]
        self.Y_init_values = Y_init.flatten() if len(Y_init) > 0 else None

        X_remaining = X_all[self.n_initial_samples:]
        Y_remaining = Y_all[self.n_initial_samples:]
        Y_std_remaining = Y_std_all[self.n_initial_samples:]

        logger.info(f"Initial training samples: {len(X_init)} Remaining samples for iteration: {len(X_remaining)} Samples per iteration: {self.n_samples_per_iteration}")

        if self.Y_init_values is not None and len(self.Y_init_values) > 0:
            logger.info(f"Initial Y mean: {np.mean(self.Y_init_values):.6g} std: {np.std(self.Y_init_values):.6g}")

        return X_init, Y_init, Y_std_init, len(X_remaining), X_remaining, Y_remaining, Y_std_remaining

    def initialize_experiment(self, X_init_real: np.ndarray, Y_init_real: np.ndarray):
        """Initialize nextorch Experiment and supply initial labeled data."""
        self.Exp = bo.Experiment("auto_detected_BO")
        try:
            # nextorch signatures vary by version: adjust Y_names type if needed
            self.Exp.input_data(
                X_init_real,
                Y_init_real,
                X_names=self.X_names,
                Y_names=[self.Y_name] if isinstance(self.Y_name, str) else self.Y_name,
                X_ranges=self.X_ranges,
                unit_flag=False
            )
            # we assume maximizing the target (change if you need minimization)
            self.Exp.set_optim_specs(maximize=True)
        except Exception as exc:
            logger.error(f"Failed to initialize nextorch Experiment: {exc}")
            raise

        logger.info(f"Initialized nextorch Experiment with {len(X_init_real)} initial samples targeting maximize {self.Y_name}.")

    def run_bo_iterations(self, X_remaining: np.ndarray, Y_remaining: np.ndarray, Y_std_remaining: np.ndarray) -> List[np.ndarray]:
        """
        Run iterative BO using the stored sequence of actual experiments.

        Workflow per iteration:
          1) Ask model for top_k recommendations via generate_next_point(n_candidates=top_k)
          2) Print the top_k candidates (predicted)
          3) Take next n_samples_per_iteration points from the provided real sequence,
             update model with their measured Y values via Exp.run_trial(...)
        """
        if self.Y_init_values is not None:
            init_mean = float(np.mean(self.Y_init_values))
            init_std = float(np.std(self.Y_init_values))
        else:
            init_mean = 0.0
            init_std = 0.0

        self.mean_responses = [init_mean]
        self.std_responses = [init_std]

        future_recommendations: List[np.ndarray] = []
        idx = 0
        it = 1

        logger.info("=" * 70)
        logger.info(f"RUNNING BO ITERATIONS (printing model Top-{self.top_k} each iteration)")
        logger.info("=" * 70)

        while idx < len(X_remaining):
            logger.info(f"\n--- Iteration {it} ---")

            # Get top_k recommendations from the surrogate / acquisition
            try:
                X_top_unit, X_top_real_pred, acq_vals = self.Exp.generate_next_point(
                    n_candidates=self.top_k,
                    acq_func_name=self.acq_func_name
                )

                # X_top_real_pred shape: (top_k, n_features)
                if X_top_real_pred is not None:
                    top_df = pd.DataFrame(X_top_real_pred, columns=self.X_names)
                    logger.info(f"Model Top-{self.top_k} recommendations (predicted):")
                    logger.info("\n" + top_df.round(6).to_string(index=False))
                    if acq_vals is not None:
                        try:
                            av = np.array(acq_vals).flatten()
                            logger.info(f"Acquisition values (approx): {av.round(6).tolist()}")
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning(f"generate_next_point failed at iteration {it}: {exc}")
                X_top_real_pred = None

            # Simulate "perform experiments" by taking next chunk from X_remaining
            samples_this_iter = min(self.n_samples_per_iteration, len(X_remaining) - idx)
            X_new_real = X_remaining[idx:idx + samples_this_iter]
            Y_new_real = Y_remaining[idx:idx + samples_this_iter]
            Y_std_new_real = Y_std_remaining[idx:idx + samples_this_iter]

            logger.info("Actual experiments performed (from dataset):")
            logger.info(pd.DataFrame(X_new_real, columns=self.X_names).round(6).to_string(index=False))
            logger.info("Actual results:")
            for s_i, (yval, ystd) in enumerate(zip(Y_new_real.flatten(), Y_std_new_real)):
                logger.info(f"  Experiment {s_i+1}: {self.Y_name} = {yval:.6g} ± {ystd if not np.isnan(ystd) else 'nan'}")
            logger.info(f"Mean this iteration: {Y_new_real.mean():.6g} ± {np.nanmean(Y_std_new_real):.6g}")

            # Unit-scale the X_new_real before calling run_trial
            try:
                X_new_unit = utils.unitscale_X(X_new_real, self.X_ranges)
            except Exception:
                # fallback min-max scaling
                X_new_unit = np.empty_like(X_new_real, dtype=float)
                for j, (lo, hi) in enumerate(self.X_ranges):
                    X_new_unit[:, j] = (X_new_real[:, j] - lo) / (hi - lo) if hi != lo else 0.5

            # Update model with actual observed result(s)
            try:
                self.Exp.run_trial(X_new_unit, X_new_real, Y_new_real)
            except Exception as exc:
                logger.warning(f"Exp.run_trial failed at iteration {it}: {exc}")

            # track statistics
            self.mean_responses.append(float(np.mean(Y_new_real)))
            mean_std = float(np.nanmean(Y_std_new_real)) if not np.all(np.isnan(Y_std_new_real)) else float(np.std(Y_new_real.flatten()))
            self.std_responses.append(mean_std)

            idx += samples_this_iter
            it += 1

        # After finishing, request final Top-K recommendations
        logger.info("\n--- Final Top-K recommendations for future experiments ---")
        try:
            X_future_unit, X_future_real_pred, _ = self.Exp.generate_next_point(
                n_candidates=self.top_k,
                acq_func_name=self.acq_func_name
            )
            if X_future_real_pred is not None:
                fut_df = pd.DataFrame(X_future_real_pred, columns=self.X_names)
                logger.info(f"Final Top-{self.top_k} recommendations:")
                logger.info("\n" + fut_df.round(6).to_string(index=False))
                future_recommendations.append(X_future_real_pred)
        except Exception as exc:
            logger.warning(f"Final generate_next_point failed: {exc}")

        return future_recommendations

    def plot_results(self, save_fig: bool = True, show_fig: bool = False, highlight_best: bool = False):
        """
        Plot optimization progress (mean ± std) in a compact, publication-friendly style.
        Saves PDF and PNG by default.
        """
        original_rc = plt.rcParams.copy()
        try:
            plt.rcParams.update({
                "font.family": "Arial",
                "font.size": 8,
                "axes.linewidth": 0.5,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "legend.frameon": False,
                "figure.dpi": 600
            })

            iters = np.arange(len(self.mean_responses))
            fig, ax = plt.subplots(figsize=(2.5, 3.5))

            main_color = "#2166AC"
            err_color = "#4D4D4D"

            ax.errorbar(iters, self.mean_responses, yerr=self.std_responses,
                        marker="o", linestyle="-", linewidth=1, markersize=3,
                        color=main_color, capsize=2, capthick=0.5,
                        ecolor=err_color, elinewidth=0.5, alpha=0.9)

            if highlight_best and len(self.mean_responses) > 0:
                best_idx = int(np.argmax(self.mean_responses))
                ax.scatter(best_idx, self.mean_responses[best_idx],
                           color="#D73027", s=25, marker="o", zorder=5,
                           edgecolor="white", linewidth=0.5)

            ax.set_xlabel("Experiment Iteration", fontsize=8)
            ylabel = f"{self.Y_name} (%)" if self.Y_name else "Target (%)"
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7, direction="out")

            n = len(iters)
            if n <= 10:
                ax.set_xticks(iters)
            else:
                ax.set_xticks(np.arange(0, n, max(1, n // 5)))

            if self.Y_plot_range and len(self.Y_plot_range) == 2:
                ylo, yhi = self.Y_plot_range
                pad = 0.05 * (yhi - ylo) if (yhi > ylo) else 0.05 * max(abs(yhi), 1.0)
                ax.set_ylim(ylo - pad, yhi + pad)
            else:
                if len(self.mean_responses) > 0:
                    arr = np.array(self.mean_responses)
                    ax.set_ylim(arr.min() - 0.1 * abs(arr.min()), arr.max() + 0.1 * abs(arr.max()))

            plt.tight_layout()

            if save_fig:
                base = f"{self.Y_name}_optimization_progress" if self.Y_name else "optimization_progress"
                pdfname = f"{base}.pdf"
                pngname = f"{base}.png"
                fig.savefig(pdfname, dpi=300, bbox_inches="tight", facecolor="white", format="pdf")
                fig.savefig(pngname, dpi=600, bbox_inches="tight", facecolor="white", format="png")
                logger.info(f"Saved figures: {pdfname}, {pngname}")

            if show_fig:
                plt.show()
            else:
                plt.close(fig)
        finally:
            plt.rcParams.update(original_rc)

    def print_summary(self):
        """Print a compact summary of the optimization results."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Data file: {self.data_file}")
        print(f"Target: {self.Y_name}")
        print(f"Input variables: {len(self.X_names)} ({', '.join(self.X_names)})")
        print(f"Initial training samples: {self.n_initial_samples}")
        print(f"Samples per iteration: {self.n_samples_per_iteration}")
        print(f"Top-k per iteration: {self.top_k}")
        print(f"Iterations performed (excluding initial): {max(0, len(self.mean_responses)-1)}")

        if len(self.mean_responses) > 0:
            init_val = self.mean_responses[0]
            init_std = self.std_responses[0] if len(self.std_responses) > 0 else 0.0
            best_val = max(self.mean_responses)
            best_idx = int(np.argmax(self.mean_responses))
            best_std = self.std_responses[best_idx] if best_idx < len(self.std_responses) else 0.0

            print(f"Initial {self.Y_name} (mean ± std): {init_val:.6g} ± {init_std:.6g}")
            print(f"Best {self.Y_name} achieved: {best_val:.6g} ± {best_std:.6g}")
            print(f"Improvement: {best_val - init_val:.6g}")

            if init_std > 0 and best_std > 0:
                combined = np.sqrt(init_std**2 + best_std**2)
                significance = abs(best_val - init_val) / combined
                print(f"Improvement significance ratio: {significance:.3f}")
                if significance > 2:
                    print("  -> Statistically significant improvement (>2σ)")
                elif significance > 1:
                    print("  -> Moderate improvement (1-2σ)")
                else:
                    print("  -> Within noise level (<1σ)")

        print(f"Optimization direction: maximize {self.Y_name}")


def main(data_file: str = "data.xlsx",
         n_initial_samples: int = 5,
         n_samples_per_iteration: int = 1,
         acq_func_name: str = "qEI",
         top_k: int = 5,
         sheet_name: str = "Sheet1"):
    """
    Example main that runs the optimizer end-to-end using an Excel file.
    """
    optimizer = PerovskiteOptimizer(
        data_file=data_file,
        random_seed=GLOBAL_RANDOM_SEED,
        n_initial_samples=n_initial_samples,
        n_samples_per_iteration=n_samples_per_iteration,
        acq_func_name=acq_func_name,
        top_k=top_k,
        sheet_name=sheet_name
    )

    optimizer.load_data()
    optimizer.compute_bounds()

    X_init, Y_init, Y_std_init, n_remaining, X_remaining, Y_remaining, Y_std_remaining = optimizer.prepare_training_data()
    optimizer.initialize_experiment(X_init, Y_init)

    future_recs = optimizer.run_bo_iterations(X_remaining, Y_remaining, Y_std_remaining)

    optimizer.plot_results(save_fig=True, show_fig=False)
    optimizer.print_summary()

    if future_recs and len(future_recs) > 0:
        print(f"\nNEXT RECOMMENDED EXPERIMENTS (Top-{optimizer.top_k}):")
        print("-" * 50)
        for i, rec in enumerate(future_recs[0]):
            print(f"Experiment {i+1}:")
            for pname, pval in zip(optimizer.X_names, rec):
                print(f"  {pname}: {pval:.6g}")
            print()

    return optimizer


if __name__ == "__main__":
    # Example usage: adjust file name and parameters as needed
    DATA_FILE = "data_1.xlsx"
    INITIAL_SAMPLES = 5
    SAMPLES_PER_ITER = 1
    ACQ_FUNCTION = "qEI"
    TOP_K = 5
    SHEET = "Sheet1"

    logger.info(f"Starting BO. Data: {DATA_FILE}, initial_samples={INITIAL_SAMPLES}, samples_per_iter={SAMPLES_PER_ITER}, acq={ACQ_FUNCTION}, top_k={TOP_K}")
    main(data_file=DATA_FILE,
         n_initial_samples=INITIAL_SAMPLES,
         n_samples_per_iteration=SAMPLES_PER_ITER,
         acq_func_name=ACQ_FUNCTION,
         top_k=TOP_K,
         sheet_name=SHEET)
