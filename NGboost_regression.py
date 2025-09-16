#!/usr/bin/env python3
"""
bandgap_model_pipeline.py

Regression pipeline for predicting bandgap (or similar target) using NGBoost with
Bayesian hyperparameter optimization, plus visualization and SHAP analysis.

Features:
- Load training and prediction CSVs
- Preprocessing with a named StandardScaler (preserves feature names)
- Hyperparameter tuning via BayesianOptimization
- Train final NGBRegressor with best hyperparameters
- Evaluate on validation set and report R^2, MAE, MSE and 95% coverage of predictive intervals
- Save predictions (mean, 95% CI, uncertainty) to CSV
- Create publication-style performance plot, histogram and SHAP summary plot
- Save trained model and scaler using joblib
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ngboost import NGBRegressor
from ngboost.distns import Normal
from bayes_opt import BayesianOptimization
import shap
from sklearn.tree import DecisionTreeRegressor
import joblib
import logging
import warnings

warnings.filterwarnings("ignore")

# ======================================================================
# User-editable configuration block
# ======================================================================

# File path and names
FILE_SETTINGS = {
    "filepath": "/ahome/shlu01/PASS/ML/ML/dataset/",  # data folder path
    "train_dataset": "Train_Bandgap.csv",            # training CSV filename
    "pred_dataset": "Prediction.csv",                 # prediction CSV filename (features only)
    "output_log": "output.log",                       # log file
    "model_save": "trained_ngb_model.pkl",            # trained model save path
    "scaler_save": "feature_scaler.pkl"               # scaler save path
}

# Visualization configuration
VISUAL_SETTINGS = {
    # Performance scatter plot settings
    "performance_plot": {
        "output_path": "performance_plot.png",
        "colors": {
            "val_edge": "#2a75bc",
            "val_face": "#89cff0",
            "train_edge": "#e1822b",
            "train_face": "#f8d49a",
            "diagonal": "#404040"
        },
        "markers": {
            "val": "o",
            "train": "s"
        },
        "figsize": (6.5, 5)
    },

    # Histogram settings (Nature single-column style)
    "histogram": {
        "output_path": "prediction_distribution.png",
        "bins": 15,
        "color": "#1f77b4",
        "edgecolor": "white",
        "alpha": 0.8,
        "figsize": (3.5, 3)
    },

    # SHAP summary settings
    "shap_plot": {
        "output_path": "feature_importance.png",
        "plot_type": "bar",
        "max_display": 20,
        "figsize": (10, 6)
    }
}

# Model training and optimization configuration
MODEL_SETTINGS = {
    "test_size": 0.2,            # validation fraction
    "random_state": 42,          # RNG seed
    "bayesian_opt": {
        "init_points": 10,       # BO initial random evaluations
        "n_iter": 200,           # BO iterations (increase/decrease for speed)
        "param_ranges": {        # hyperparameter bounds for BO
            "n_estimators": (50, 800),
            "learning_rate": (0.001, 0.2),
            "max_depth": (3, 15)
        }
    },
    "tree_params": {             # base decision tree params used inside NGBoost
        "min_samples_leaf": 5,
        "splitter": "best"
    }
}

# ======================================================================
# Core code - do not modify unless you know what you're changing
# ======================================================================

class NamedStandardScaler(StandardScaler):
    """
    Custom StandardScaler that preserves feature names when transforming/inversing.
    If a DataFrame is passed to fit_transform, it stores column names.
    """
    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        return super().fit_transform(X)

    def inverse_transform(self, X):
        X_inv = super().inverse_transform(X)
        if hasattr(self, "feature_names"):
            return pd.DataFrame(X_inv, columns=self.feature_names)
        return X_inv


def plot_performance(y_train_true, y_train_pred, y_val_true, y_val_pred, r2_value, settings):
    """Create a scatter performance plot for training and validation predictions."""
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2
    })

    fig, ax = plt.subplots(figsize=settings["figsize"], dpi=600)

    # Training points
    ax.scatter(
        y_train_true, y_train_pred,
        alpha=0.8,
        edgecolors=settings["colors"]["train_edge"],
        facecolors=settings["colors"]["train_face"],
        linewidth=0.8,
        marker=settings["markers"]["train"],
        s=60,
        label="Training Set"
    )

    # Validation points
    ax.scatter(
        y_val_true, y_val_pred,
        alpha=0.8,
        edgecolors=settings["colors"]["val_edge"],
        facecolors=settings["colors"]["val_face"],
        linewidth=0.8,
        marker=settings["markers"]["val"],
        s=60,
        label="Validation Set"
    )

    # Diagonal reference line y = x
    min_val = min(np.min(y_train_true), np.min(y_val_true))
    max_val = max(np.max(y_train_true), np.max(y_val_true))
    ax.plot([min_val, max_val], [min_val, max_val],
            color=settings["colors"]["diagonal"],
            linestyle="--",
            lw=1.5,
            dashes=(5, 3),
            alpha=0.8,
            label=f"R² = {r2_value:.3f}")

    ax.set_xlabel("True Values", fontsize=16, labelpad=8)
    ax.set_ylabel("Predictions", fontsize=16, labelpad=8)
    ax.set_title(f"Validation Performance (R² = {r2_value:.3f})", fontsize=14, pad=14)

    ax.tick_params(axis="both", labelsize=14)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
    ax.grid(alpha=0.4, linestyle=":", linewidth=0.8)
    ax.legend(frameon=False, fontsize=14, handletextpad=0.4, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout(pad=1.8)
    plt.savefig(settings["output_path"], bbox_inches="tight", transparent=True)
    plt.close()


def plot_histogram(predictions, settings):
    """Create a publication-friendly histogram of predictions."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, ax = plt.subplots(figsize=settings["figsize"])

    sns.histplot(predictions,
                 bins=settings["bins"],
                 kde=False,
                 color=settings["color"],
                 edgecolor=settings["edgecolor"],
                 alpha=settings["alpha"],
                 ax=ax)

    mean_val = np.mean(predictions)
    std_val = np.std(predictions)
    text_str = f"Mean = {mean_val:.2f}\nSD = {std_val:.2f}"
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

    ax.set_xlabel("Predicted Bandgap (eV)", labelpad=2)
    ax.set_ylabel("Count", labelpad=2)
    plt.tight_layout(pad=0.5)
    plt.savefig(settings["output_path"], bbox_inches="tight")
    plt.close()


def main():
    # Initialize logging to file and console
    logging.basicConfig(filename=FILE_SETTINGS["output_log"],
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    try:
        # Load CSVs
        train_path = os.path.join(FILE_SETTINGS["filepath"], FILE_SETTINGS["train_dataset"])
        pred_path = os.path.join(FILE_SETTINGS["filepath"], FILE_SETTINGS["pred_dataset"])
        train_df = pd.read_csv(train_path)
        pred_df = pd.read_csv(pred_path)
        logging.info("Data loaded successfully.")

        # Expect: first column = material id, features in middle, last column = target
        material_ids_train = train_df.iloc[:, 0]
        X_train = train_df.iloc[:, 1:-1]
        y_train = train_df.iloc[:, -1]
        material_ids_pred = pred_df.iloc[:, 0]
        X_pred = pred_df.iloc[:, 1:]

        # Handle missing values with median imputation
        X_train = X_train.fillna(X_train.median())
        y_train = y_train.fillna(y_train.median())
        X_pred = X_pred.fillna(X_pred.median())

        # Standardize features and preserve names
        scaler = NamedStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        # Train/validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train,
            test_size=MODEL_SETTINGS["test_size"],
            random_state=MODEL_SETTINGS["random_state"]
        )

        # Define objective for Bayesian Optimization (maximize negative logscore -> minimize logscore)
        def ngboost_hyperopt(n_estimators, learning_rate, max_depth):
            # n_estimators and max_depth will be floats from BO; cast to int where necessary
            ngb = NGBRegressor(
                Dist=Normal,
                Base=DecisionTreeRegressor(
                    max_depth=int(max_depth),
                    min_samples_leaf=MODEL_SETTINGS["tree_params"]["min_samples_leaf"],
                    splitter=MODEL_SETTINGS["tree_params"]["splitter"],
                    random_state=MODEL_SETTINGS["random_state"]
                ),
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                natural_gradient=True,
                verbose=False,
                random_state=MODEL_SETTINGS["random_state"]
            )
            # NGBoost fit with early-stopping style validation (X_val, Y_val). If this version of ngboost
            # doesn't accept X_val/Y_val, adapt accordingly.
            ngb.fit(X_train_split, y_train_split, X_val=X_val_split, Y_val=y_val_split)
            # Check evals_result for validation logscore; return a metric to maximize
            try:
                # we want to maximize negative validation logscore -> effectively minimize logscore
                val_logscore = ngb.evals_result["val"]["LOGSCORE"][-1]
                return -val_logscore
            except Exception:
                # If evals_result isn't available, return a large negative placeholder
                return -1e3

        # Bayesian optimization of hyperparameters
        optimizer = BayesianOptimization(
            f=ngboost_hyperopt,
            pbounds=MODEL_SETTINGS["bayesian_opt"]["param_ranges"],
            random_state=MODEL_SETTINGS["random_state"]
        )
        logging.info("Starting Bayesian optimization of NGBoost hyperparameters...")
        optimizer.maximize(
            init_points=MODEL_SETTINGS["bayesian_opt"]["init_points"],
            n_iter=MODEL_SETTINGS["bayesian_opt"]["n_iter"]
        )

        # Extract best parameters from optimizer
        best_params = {
            "n_estimators": int(optimizer.max["params"]["n_estimators"]),
            "learning_rate": float(optimizer.max["params"]["learning_rate"]),
            "max_depth": int(optimizer.max["params"]["max_depth"])
        }
        logging.info(f"Optimal hyperparameters found by BO: {best_params}")

        # Train final NGBoost model on full training data
        final_ngb = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(
                max_depth=best_params["max_depth"],
                min_samples_leaf=MODEL_SETTINGS["tree_params"]["min_samples_leaf"],
                splitter=MODEL_SETTINGS["tree_params"]["splitter"],
                random_state=MODEL_SETTINGS["random_state"]
            ),
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            natural_gradient=True,
            verbose=0,
            random_state=MODEL_SETTINGS["random_state"]
        )
        logging.info("Training final NGBoost model on full training data...")
        final_ngb.fit(X_train_scaled, y_train)

        # Evaluate model: use previously reserved split for validation metrics
        y_train_pred = final_ngb.predict(X_train_split)
        y_val_pred = final_ngb.predict(X_val_split)

        # Predictive distribution on validation set for interval coverage
        y_val_dist = final_ngb.pred_dist(X_val_split)
        try:
            lower_val, upper_val = y_val_dist.interval(0.95)
        except Exception:
            # If pred_dist returns params instead, attempt to compute intervals using mean ± 1.96*std
            try:
                loc = y_val_dist.params["loc"]
                scale = y_val_dist.params["scale"]
                lower_val = loc - 1.96 * scale
                upper_val = loc + 1.96 * scale
            except Exception:
                lower_val = np.full_like(y_val_pred, np.nan)
                upper_val = np.full_like(y_val_pred, np.nan)

        metrics = {
            "R2": r2_score(y_val_split, y_val_pred),
            "MAE": mean_absolute_error(y_val_split, y_val_pred),
            "MSE": mean_squared_error(y_val_split, y_val_pred),
            "Coverage_95%": float(np.mean((y_val_split >= lower_val) & (y_val_split <= upper_val)))
        }
        logging.info(f"Validation metrics: {metrics}")

        # Predict on new data: return predictive distribution and central estimate
        pred_dist = final_ngb.pred_dist(X_pred_scaled)
        try:
            pred_loc = pred_dist.params["loc"]
            pred_scale = pred_dist.params["scale"]
            pred_lower, pred_upper = pred_dist.interval(0.95)
        except Exception:
            # fallback to mean and ±1.96*scale if available
            try:
                pred_loc = pred_dist.loc
                pred_scale = pred_dist.scale
                pred_lower = pred_loc - 1.96 * pred_scale
                pred_upper = pred_loc + 1.96 * pred_scale
            except Exception:
                pred_loc = np.full(X_pred_scaled.shape[0], np.nan)
                pred_scale = np.full(X_pred_scaled.shape[0], np.nan)
                pred_lower = np.full(X_pred_scaled.shape[0], np.nan)
                pred_upper = np.full(X_pred_scaled.shape[0], np.nan)

        results_df = pd.DataFrame({
            "material_id": material_ids_pred,
            "predicted_bandgap": pred_loc,
            "lower_95": pred_lower,
            "upper_95": pred_upper,
            "uncertainty": pred_scale
        })
        results_csv = "material_predictions.csv"
        results_df.to_csv(results_csv, index=False)
        logging.info(f"Prediction results saved to: {results_csv}")

        # Visualization: performance plot and histogram
        plot_performance(y_train_split, y_train_pred,
                         y_val_split, y_val_pred,
                         metrics["R2"],
                         VISUAL_SETTINGS["performance_plot"])

        plot_histogram(results_df["predicted_bandgap"], VISUAL_SETTINGS["histogram"])

        # SHAP analysis (TreeExplainer typically works with tree-based models)
        logging.info("Computing SHAP values (may take some time)...")
        try:
            # Use the final_ngb.model if tree ensemble is wrapped differently depending on version
            explainer = shap.TreeExplainer(final_ngb, model_output=1)
            shap_values = explainer.shap_values(X_pred_scaled)
            plt.figure(figsize=VISUAL_SETTINGS["shap_plot"]["figsize"])
            shap.summary_plot(shap_values, X_pred_scaled,
                              feature_names=getattr(scaler, "feature_names", None),
                              plot_type=VISUAL_SETTINGS["shap_plot"]["plot_type"],
                              max_display=VISUAL_SETTINGS["shap_plot"]["max_display"])
            plt.title("Feature Importance (SHAP Values)", fontsize=14)
            plt.tight_layout()
            plt.savefig(VISUAL_SETTINGS["shap_plot"]["output_path"])
            plt.close()
            logging.info(f"SHAP summary plot saved to: {VISUAL_SETTINGS['shap_plot']['output_path']}")
        except Exception as exc:
            logging.warning(f"SHAP analysis failed or is incompatible with this model: {exc}")

        # Save final model and scaler
        joblib.dump(final_ngb, FILE_SETTINGS["model_save"])
        joblib.dump(scaler, FILE_SETTINGS["scaler_save"])
        logging.info(f"Saved trained model to: {FILE_SETTINGS['model_save']}")
        logging.info(f"Saved feature scaler to: {FILE_SETTINGS['scaler_save']}")
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
