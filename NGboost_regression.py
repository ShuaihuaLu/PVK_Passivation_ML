import pandas as pd
import numpy as np
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
# 自定义配置区 (用户可修改以下参数)
# ======================================================================

# 文件路径配置
FILE_SETTINGS = {
    'filepath': '/ahome/shlu01/PASS/ML/ML/dataset/',  # 数据文件路径
    'train_dataset': "Train_Bandgap.csv",        # 训练数据文件名
    'pred_dataset': 'Prediction.csv',                 # 预测数据文件名
    'output_log': 'output.log',                       # 日志文件名
    'model_save': 'trained_ngb_model.pkl',            # 模型保存路径
    'scaler_save': 'feature_scaler.pkl'               # 标准化器保存路径
}

# 可视化参数配置
VISUAL_SETTINGS = {
    # 性能图参数
    'performance_plot': {
        'output_path': 'performance_plot.png',
        'colors': {
            'val_edge': '#2a75bc',
            'val_face': '#89cff0',
            'train_edge': '#e1822b',
            'train_face': '#f8d49a',
            'diagonal': '#404040'
        },
        'markers': {
            'val': 'o',
            'train': 's'
        },
        'figsize': (6.5, 5)
    },
    
    # 直方图参数
    'histogram': {
        'output_path': 'prediction_distribution.png',
        'bins': 15,
        'color': '#1f77b4',
        'edgecolor': 'white',
        #'linewidth': 0.5,
        'alpha': 0.8,
        #'kde_linewidth': 1.2,
        'figsize': (3.5, 3)  # Nature期刊单栏宽度(8.5cm)
    },
    
    # SHAP参数
    'shap_plot': {
        'output_path': 'feature_importance.png',
        'plot_type': "bar",
        'max_display': 20,
        'figsize': (10, 6)
    }
}

# 模型训练配置
MODEL_SETTINGS = {
    'test_size': 0.2,            # 验证集比例
    'random_state': 42,          # 随机种子
    'bayesian_opt': {
        'init_points': 10,       # 贝叶斯优化初始点
        'n_iter': 200,           # 贝叶斯优化迭代次数
        'param_ranges': {        # 超参数搜索范围
            'n_estimators': (50, 800),
            'learning_rate': (0.001, 0.2),
            'max_depth': (3, 15)
        }
    },
    'tree_params': {             # 决策树参数
        'min_samples_leaf': 5,
        'splitter': 'best'
    }
}

# ======================================================================
# 核心代码区 (以下代码不建议修改)
# ======================================================================

class NamedStandardScaler(StandardScaler):
    """自定义标准化器保留特征名称"""
    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        return super().fit_transform(X)

    def inverse_transform(self, X):
        X_trans = super().inverse_transform(X)
        if hasattr(self, 'feature_names'):
            return pd.DataFrame(X_trans, columns=self.feature_names)
        return X_trans

def plot_performance(y_train_true, y_train_pred, y_val_true, y_val_pred, r2_score, settings):
    """生成性能对比图"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2
    })
    
    fig, ax = plt.subplots(figsize=settings['figsize'], dpi=600)
    
    # 绘制训练集
    ax.scatter(y_train_true, y_train_pred,
               alpha=0.8,
               edgecolors=settings['colors']['train_edge'],
               facecolors=settings['colors']['train_face'],
               linewidth=0.8,
               marker=settings['markers']['train'],
               s=60,
               label='Training Set')
    
    # 绘制验证集
    ax.scatter(y_val_true, y_val_pred,
               alpha=0.8,
               edgecolors=settings['colors']['val_edge'],
               facecolors=settings['colors']['val_face'],
               linewidth=0.8,
               marker=settings['markers']['val'],
               s=60,
               label='Validation Set')
    
    # 绘制参考线
    min_val = min(y_train_true.min(), y_val_true.min())
    max_val = max(y_train_true.max(), y_val_true.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color=settings['colors']['diagonal'],
            linestyle='--',
            lw=1.5,
            dashes=(5, 3),
            alpha=0.8,
            label=f'R² = {r2_score:.3f}')
    
    # 设置坐标轴
    ax.set_xlabel('True Values', fontsize=16, labelpad=8)
    ax.set_ylabel('Predictions', fontsize=16, labelpad=8)
    ax.set_title(f'Validation Performance (R² = {r2_score:.3f})', fontsize=14, pad=14)
    
    # 格式调整
    ax.tick_params(axis='both', labelsize=14)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
    ax.grid(alpha=0.4, linestyle=':', linewidth=0.8)
    ax.legend(frameon=False, fontsize=14, handletextpad=0.4, 
             loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    plt.tight_layout(pad=1.8)
    plt.savefig(settings['output_path'], bbox_inches='tight', transparent=True)
    plt.close()

def plot_histogram(predictions, settings):
    """生成Nature风格直方图"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    fig, ax = plt.subplots(figsize=settings['figsize'])
    
    # 绘制直方图
    sns.histplot(predictions, 
                 bins=settings['bins'],
                 kde=False,
                 color=settings['color'],
                 edgecolor=settings['edgecolor'],
                 #linewidth=settings['linewidth'],
                 alpha=settings['alpha'],
                 ax=ax,
                 #kde_kws={'linewidth': settings['kde_linewidth']}
                 )
    
    # 添加统计信息
    mean_val = np.mean(predictions)
    std_val = np.std(predictions)
    text_str = f'Mean = {mean_val:.2f}\nSD = {std_val:.2f}'
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes,
            ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_xlabel('Predicted Bandgap (eV)', labelpad=2)
    ax.set_ylabel('Count', labelpad=2)
    plt.tight_layout(pad=0.5)
    plt.savefig(settings['output_path'], bbox_inches='tight')
    plt.close()

def main():
    # 初始化日志
    logging.basicConfig(filename=FILE_SETTINGS['output_log'],
                       level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 加载数据
        train_df = pd.read_csv(FILE_SETTINGS['filepath'] + FILE_SETTINGS['train_dataset'])
        pred_df = pd.read_csv(FILE_SETTINGS['filepath'] + FILE_SETTINGS['pred_dataset'])
        logging.info("Data Load Successfully !")
        
        # 数据处理
        material_ids_train = train_df.iloc[:, 0]
        X_train = train_df.iloc[:, 1:-1]
        y_train = train_df.iloc[:, -1]
        material_ids_pred = pred_df.iloc[:, 0]
        X_pred = pred_df.iloc[:, 1:]
        
        # 处理缺失值
        X_train.fillna(X_train.median(), inplace=True)
        y_train.fillna(y_train.median(), inplace=True)
        X_pred.fillna(X_pred.median(), inplace=True)
        
        # 数据标准化
        scaler = NamedStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)
        
        # 划分训练验证集
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, 
            test_size=MODEL_SETTINGS['test_size'],
            random_state=MODEL_SETTINGS['random_state']
        )
        
        # 贝叶斯优化
        def ngboost_hyperopt(n_estimators, learning_rate, max_depth):
            ngb = NGBRegressor(
                Dist=Normal,
                Base=DecisionTreeRegressor(
                    max_depth=int(max_depth),
                    min_samples_leaf=MODEL_SETTINGS['tree_params']['min_samples_leaf'],
                    splitter=MODEL_SETTINGS['tree_params']['splitter'],
                    random_state=MODEL_SETTINGS['random_state']
                ),
                n_estimators=int(n_estimators),
                learning_rate=learning_rate,
                natural_gradient=True,
                verbose=False,
                random_state=MODEL_SETTINGS['random_state']
            )
            ngb.fit(X_train_split, y_train_split, X_val=X_val_split, Y_val=y_val_split)
            return -ngb.evals_result['val']['LOGSCORE'][-1] if 'val' in ngb.evals_result else -1000

        optimizer = BayesianOptimization(
            f=ngboost_hyperopt,
            pbounds=MODEL_SETTINGS['bayesian_opt']['param_ranges'],
            random_state=MODEL_SETTINGS['random_state']
        )
        optimizer.maximize(
            init_points=MODEL_SETTINGS['bayesian_opt']['init_points'],
            n_iter=MODEL_SETTINGS['bayesian_opt']['n_iter']
        )
        
        # 获取最优参数
        best_params = {
            'n_estimators': int(optimizer.max['params']['n_estimators']),
            'learning_rate': optimizer.max['params']['learning_rate'],
            'max_depth': int(optimizer.max['params']['max_depth'])
        }
        logging.info(f"Optimal Parameters: {best_params}")
        
        # 训练最终模型
        final_ngb = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(
                max_depth=best_params['max_depth'],
                min_samples_leaf=MODEL_SETTINGS['tree_params']['min_samples_leaf'],
                splitter=MODEL_SETTINGS['tree_params']['splitter']
            ),
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            natural_gradient=True,
            verbose=0
        )
        final_ngb.fit(X_train_scaled, y_train)
        
        # 模型评估
        y_train_pred = final_ngb.predict(X_train_split)
        y_val_pred = final_ngb.predict(X_val_split)
        y_dist = final_ngb.pred_dist(X_val_split)
        lower, upper = y_dist.interval(0.95)
        
        metrics = {
            'R²': r2_score(y_val_split, y_val_pred),
            'MAE': mean_absolute_error(y_val_split, y_val_pred),
            'MSE': mean_squared_error(y_val_split, y_val_pred),
            'Coverage_95%': np.mean((y_val_split >= lower) & (y_val_split <= upper))
        }
        logging.info(f"Validation Indicators: {metrics}")
        
        # 生成预测
        pred_dist = final_ngb.pred_dist(X_pred_scaled)
        results_df = pd.DataFrame({
            'material_id': material_ids_pred,
            'predicted_formation_energy': pred_dist.params['loc'],
            'lower_95': pred_dist.interval(0.95)[0],
            'upper_95': pred_dist.interval(0.95)[1],
            'uncertainty': pred_dist.params['scale']
        })
        results_df.to_csv("material_predictions.csv", index=False)
        
        # 可视化
        plot_performance(y_train_split, y_train_pred, 
                        y_val_split, y_val_pred,
                        metrics['R²'],
                        VISUAL_SETTINGS['performance_plot'])
        
        plot_histogram(results_df['predicted_bandgap'],
                      VISUAL_SETTINGS['histogram'])
        
        # SHAP分析
        explainer = shap.TreeExplainer(final_ngb, model_output=1)
        shap_values = explainer.shap_values(X_pred_scaled)
        plt.figure(figsize=VISUAL_SETTINGS['shap_plot']['figsize'])
        shap.summary_plot(shap_values, X_pred_scaled,
                         feature_names=scaler.feature_names,
                         plot_type=VISUAL_SETTINGS['shap_plot']['plot_type'],
                         max_display=VISUAL_SETTINGS['shap_plot']['max_display'])
        plt.title("Feature Importance (SHAP Values)", fontsize=14)
        plt.tight_layout()
        plt.savefig(VISUAL_SETTINGS['shap_plot']['output_path'])
        plt.close()
        
        # 保存模型
        joblib.dump(final_ngb, FILE_SETTINGS['model_save'])
        joblib.dump(scaler, FILE_SETTINGS['scaler_save'])
        logging.info("Model Save Successfully !")

    except Exception as e:
        logging.error(f"Running Failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
