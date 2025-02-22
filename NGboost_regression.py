import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ngboost import NGBRegressor
from ngboost.distns import Normal
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import shap
from sklearn.tree import DecisionTreeRegressor
import joblib
import logging
from scipy.stats import norm

import warnings

warnings.filterwarnings("ignore")

# Custom scaler preserving feature names
class NamedStandardScaler(StandardScaler):
    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        return super().fit_transform(X)
    
    def inverse_transform(self, X):
        X_trans = super().inverse_transform(X)
        if hasattr(self, 'feature_names'):
            return pd.DataFrame(X_trans, columns=self.feature_names)
        return X_trans

def main():
    # Configure logging
    logging.basicConfig(filename='material_property_prediction.log', 
                       level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load training data
    try:
        train_df = pd.read_csv("Labeled_Data_E_f.csv")
        logging.info("Training data loaded successfully")
    except FileNotFoundError:
        logging.error("Training data file not found")
        raise

    # Process training data
    material_ids_train = train_df.iloc[:, 0]  # First column: material IDs
    X_train = train_df.iloc[:, 1:-1]  # Features columns
    y_train = train_df.iloc[:, -1]   # Last column: target property

    # Handle missing values
    if X_train.isnull().sum().any():
        logging.warning("Missing values found in training data, applying median imputation")
        X_train.fillna(X_train.median(), inplace=True)
        y_train.fillna(y_train.median(), inplace=True)

    # Load prediction data
    try:
        pred_df = pd.read_csv("pred_data.csv")
        logging.info("Prediction data loaded successfully")
    except FileNotFoundError:
        logging.error("Prediction data file not found")
        raise

    # Process prediction data
    material_ids_pred = pred_df.iloc[:, 0]  # First column: material IDs
    X_pred = pred_df.iloc[:, 1:]  # Feature columns

    # Handle missing values in prediction data
    if X_pred.isnull().sum().any():
        logging.warning("Missing values found in prediction data, applying median imputation")
        X_pred.fillna(X_pred.median(), inplace=True)

    # Initialize and fit scaler
    scaler = NamedStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Train-validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Bayesian optimization objective function
    def ngboost_hyperopt(n_estimators, learning_rate, max_depth):
        ngb = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(
                max_depth=int(max_depth),
                min_samples_leaf=5,
                splitter="best",
                random_state=42
            ),
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            natural_gradient=True,
            verbose=True,
            random_state=42
        )
        ngb.fit(X_train_split, y_train_split,
                X_val=X_val_split, Y_val=y_val_split, 
                #early_stopping_rounds=20
                )
        # Retrieve validation scores; use negative loss for maximization
        if 'val' in ngb.evals_result:
            #print(ngb.evals_result['val']['LOGSCORE'][-1])
            val_loss = ngb.evals_result['val']['LOGSCORE'][-1]
            return -val_loss  # Negative because we want to minimize the loss
        else:
            # Fallback: Calculate validation score manually if needed
            y_pred = ngb.predict(X_val)

        return -mean_squared_error(y_val, y_pred)  # Use validation score for optimization

    # Run Bayesian optimization
    optimizer = BayesianOptimization(
        f=ngboost_hyperopt,
        pbounds={
            'n_estimators': (50, 800),
            'learning_rate': (0.001, 0.2),
            'max_depth': (3, 10)
        },
        random_state=42
    )
    optimizer.maximize(init_points=10, n_iter=50)
    best_params = optimizer.max['params']
    best_params.update({
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth'])
    })
    logging.info(f"Best parameters obtained: {best_params}")

    # Train final model
    final_ngb = NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(
            max_depth=best_params['max_depth'],
            min_samples_leaf=5
        ),
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        natural_gradient=True,
        verbose=2
    )
    final_ngb.fit(X_train_scaled, y_train)
    
    y_train_pred = final_ngb.predict(X_train_scaled)
    # Model evaluation
    y_pred = final_ngb.predict(X_val_split)
    y_dist = final_ngb.pred_dist(X_val_split)
    lower, upper = y_dist.interval(0.95)
    
    metrics = {
        'R²': r2_score(y_val_split, y_pred),
        'MAE': mean_absolute_error(y_val_split, y_pred),
        'MSE': mean_squared_error(y_val_split, y_pred),
        'Coverage_95%': np.mean((y_val_split >= lower) & (y_val_split <= upper))
    }
    
    logging.info("Validation metrics:")
    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")
        print(f"{k}: {v:.4f}")
    
    def calculate_uncertainty(pred_dist):
        """Calculate uncertainty metrics from predicted distribution"""
        if isinstance(pred_dist, np.ndarray):  # 处理数组格式的输出
            return pred_dist[:, 1]**2  # 假设第二列为scale参数
        elif hasattr(pred_dist, 'params'):  # 处理分布对象
            return pred_dist.params['scale']**2

        raise ValueError("Unsupported distribution type")

    def select_samples(model, X_unlabeled, n_samples=20):
        """Select samples with highest prediction uncertainty"""
        # Get predicted probability distribution
        pred_dist = model.pred_dist(X_unlabeled)
    
        # Calculate uncertainties
        uncertainties = calculate_uncertainty(pred_dist)
    
        # Select indices with highest uncertainty
        sorted_indices = np.argsort(uncertainties)[::-1]

        point_pred = pred_dist.params['loc']
        
        lower_pred, upper_pred = pred_dist.interval(0.95)

        # Create prediction output dataframe
        results_df = pd.DataFrame({
            'material_id': material_ids_pred,
            'predicted_strength': point_pred,
            'lower_95': lower_pred,
            'upper_95': upper_pred,
            'uncertainties': uncertainties
        })

        # Save predictions
        results_df.to_csv("material_strength_predictions.csv", index=False)
        logging.info("Predictions saved to material_strength_predictions.csv")

        return sorted_indices[:n_samples]
    
    n_annotate = 20
    selected_idx = select_samples(final_ngb, X_pred_scaled, n_annotate)

    # Display results
    print(f"\nSelected samples for annotation (indices {selected_idx}):")
    print("Corresponding features:")
    print(material_ids_pred[selected_idx])
    

    # Visualization 1: Validation performance
    def plot_science_style_performance(y_train_true, y_train_pred, y_val_true, y_val_pred, r2_score,
                                      output_path='science_performance_plot.png',
                                      colors={'val_edge': '#2a75bc', 'val_face': '#89cff0',
                                              'train_edge': '#e1822b', 'train_face': '#f8d49a',
                                              'diagonal': '#404040'},
                                      markers={'val': 'o', 'train': 's'},
                                      figsize=(6.5, 5)):

        """
        Generate Science-styled performance plot with training/validation data

        Parameters:
        y_train_true (array): True values for training set
        y_train_pred (array): Predicted values for training set
        y_val_true (array): True values for validation set
        y_val_pred (array): Predicted values for validation set
        r2_score (float): R-squared value for validation set
        output_path (str): Output file path (default: 'science_performance_plot.png')
        colors (dict): Custom color scheme dictionary
        markers (dict): Marker styles dictionary
        figsize (tuple): Figure dimensions in inches (default: (6.5,5))
        """
        # Set science style parameters
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.major.width'] = 1.2
        plt.rcParams['ytick.major.width'] = 1.2

        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        # Plot training set
        ax.scatter(y_train_true, y_train_pred,
                  alpha=0.8,
                  edgecolors=colors['train_edge'],
                  facecolors=colors['train_face'],
                  linewidth=0.8,
                  marker=markers['train'],
                  s=60,
                  label='Training Set')

        # Plot validation set
        ax.scatter(y_val_true, y_val_pred,
                   alpha=0.8,
                   edgecolors=colors['val_edge'],
                   facecolors=colors['val_face'],
                   linewidth=0.8,
                   marker=markers['val'],
                   s=60,
                   label='Validation Set')

        # Plot ideal diagonal line
        min_val = min(y_train_true.min(), y_val_true.min())
        max_val = max(y_train_true.max(), y_val_true.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color=colors['diagonal'],
                linestyle='--',
                lw=1.5,
                dashes=(5, 3),
                alpha=0.8,
                label=(f'R² = {r2_score:.3f}'))
    
        # Axis labels and title
        ax.set_xlabel('True Values', fontsize=11, labelpad=8)
        ax.set_ylabel('Predictions', fontsize=11, labelpad=8)
        ax.set_title(f'Validation Performance (R² = {r2_score:.3f})', fontsize=12, pad=14)

        # Axis adjustments
        ax.tick_params(axis='both', which='major', labelsize=10)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(True)

        # Grid and legend
        ax.grid(alpha=0.4, linestyle=':', linewidth=0.8)
        ax.legend(frameon=False,
                 fontsize=10,
                 handletextpad=0.4,
                 loc='upper left',
                 bbox_to_anchor=(0.02, 0.98))

        plt.tight_layout(pad=1.8)
        plt.savefig(output_path, bbox_inches='tight', transparent=True)
        plt.close()

    plot_science_style_performance(
    y_train_true=y_train,
    y_train_pred=y_train_pred,
    y_val_true=y_val_split,
    y_val_pred=y_pred,
    r2_score=metrics["R²"],
    output_path='science_plot.pdf',  # Save as PDF for publication
    figsize=(6.5, 5)
    )

    # Visualization 2: SHAP feature importance
    shap.initjs()
    explainer = shap.TreeExplainer(final_ngb, 
                               model_output=1
                              )
    shap_values = explainer.shap_values(X_pred_scaled)
    # X_train_split
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_pred_scaled, 
                     feature_names=scaler.feature_names, 
                     plot_type="bar", max_display=20)
    plt.title("Feature Importance (SHAP Values)", fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Save model and scaler
    joblib.dump(final_ngb, 'trained_ngb_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    logging.info("Model and scaler saved successfully")


if __name__ == "__main__":
    main()
