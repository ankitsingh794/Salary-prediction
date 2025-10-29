"""
Visualization module for model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelVisualizer:
    """Class to visualize model results"""
    
    def __init__(self):
        pass
    
    def plot_model_comparison(self, results, metric='val_r2', save_path=None):
        """
        Plot comparison of models based on a specific metric
        """
        models = []
        values = []
        
        for model_name, metrics in results.items():
            models.append(model_name)
            values.append(metrics[metric])
        
        # Sort by value
        sorted_indices = np.argsort(values)[::-1]
        models = [models[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, values, color=colors)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            if 'r2' in metric:
                label = f'{value:.4f}'
            elif 'mape' in metric:
                label = f'{value:.2f}%'
            else:
                label = f'${value:,.0f}'
            ax.text(value, bar.get_y() + bar.get_height()/2, label,
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        metric_titles = {
            'val_r2': 'R² Score (Validation)',
            'val_rmse': 'RMSE (Validation)',
            'val_mae': 'MAE (Validation)',
            'val_mape': 'MAPE % (Validation)'
        }
        
        ax.set_xlabel(metric_titles.get(metric, metric), fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, model, X_test, y_test, model_name, save_path=None):
        """
        Plot predicted vs actual values
        """
        y_pred = model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        
        ax.set_xlabel('Actual Salary ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Predicted vs Actual Salaries\nR² = {r2:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, model, X_test, y_test, model_name, save_path=None):
        """
        Plot residuals distribution
        """
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
        axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals ($)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name}: Residual Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance, model_name, save_path=None):
        """
        Plot feature importance
        """
        if feature_importance is None:
            print(f"No feature importance available for {model_name}")
            return
        
        features = [f[0] for f in feature_importance]
        importances = [f[1] for f in feature_importance]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.plasma(np.linspace(0, 1, len(features)))
        bars = ax.barh(features, importances, color=colors)
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            ax.text(importance, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, model, X_train, y_train, X_val, y_val, model_name, save_path=None):
        """
        Plot learning curves (for models that support partial_fit or have training history)
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, 'o-', color='b', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='b')
        
        ax.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='g')
        
        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_data_distribution(self, df, save_path=None):
        """
        Plot salary distribution and other data statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Salary distribution
        axes[0, 0].hist(df['salary'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Salary ($)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Salary Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Salary by job title
        if 'job_title' in df.columns:
            job_salary = df.groupby('job_title')['salary'].mean().sort_values(ascending=True)
            axes[0, 1].barh(job_salary.index, job_salary.values, color='coral')
            axes[0, 1].set_xlabel('Average Salary ($)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Job Title', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Average Salary by Job Title', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Experience vs Salary
        if 'experience_years' in df.columns:
            axes[1, 0].scatter(df['experience_years'], df['salary'], alpha=0.5, s=30, color='green')
            axes[1, 0].set_xlabel('Experience (Years)', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Experience vs Salary', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Education vs Salary
        if 'education' in df.columns:
            edu_salary = df.groupby('education')['salary'].mean().sort_values(ascending=True)
            axes[1, 1].barh(edu_salary.index, edu_salary.values, color='purple', alpha=0.7)
            axes[1, 1].set_xlabel('Average Salary ($)', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Education Level', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Average Salary by Education', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
