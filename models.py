"""
Machine Learning Models Module
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    BaggingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import config


class SalaryPredictionModels:
    """Class to manage multiple machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def build_models(self):
        """
        Build various ensemble models
        """
        print("\n" + "="*50)
        print("BUILDING MODELS")
        print("="*50)
        
        # 1. Random Forest
        self.models['Random Forest'] = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)
        print("✓ Random Forest model created")
        
        # 2. Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingRegressor(**config.GRADIENT_BOOSTING_PARAMS)
        print("✓ Gradient Boosting model created")
        
        # 3. XGBoost
        self.models['XGBoost'] = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        print("✓ XGBoost model created")
        
        # 4. LightGBM
        self.models['LightGBM'] = lgb.LGBMRegressor(**config.LIGHTGBM_PARAMS)
        print("✓ LightGBM model created")
        
        # 5. AdaBoost
        self.models['AdaBoost'] = AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE
        )
        print("✓ AdaBoost model created")
        
        # 6. Bagging Regressor
        self.models['Bagging'] = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=10),
            n_estimators=100,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        print("✓ Bagging Regressor model created")
        
        return self.models
    
    def build_ensemble_models(self):
        """
        Build advanced ensemble models (Voting and Stacking)
        """
        print("\n" + "="*50)
        print("BUILDING ENSEMBLE MODELS")
        print("="*50)
        
        # Base estimators for ensemble
        rf_base = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        gb_base = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=config.RANDOM_STATE
        )
        xgb_base = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        
        # Voting Regressor (Simple averaging)
        self.models['Voting Ensemble'] = VotingRegressor(
            estimators=[
                ('rf', rf_base),
                ('gb', gb_base),
                ('xgb', xgb_base)
            ],
            n_jobs=-1
        )
        print("✓ Voting Ensemble model created")
        
        # Stacking Regressor (Meta-learning)
        self.models['Stacking Ensemble'] = StackingRegressor(
            estimators=[
                ('rf', rf_base),
                ('gb', gb_base),
                ('xgb', xgb_base),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=config.RANDOM_STATE, verbose=-1))
            ],
            final_estimator=Ridge(),
            n_jobs=-1
        )
        print("✓ Stacking Ensemble model created")
        
        return self.models
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val):
        """
        Train a single model and evaluate on validation set
        """
        print(f"\nTraining {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Calculate MAPE
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
        
        results = {
            'model_name': model_name,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mape': train_mape,
            'val_mape': val_mape
        }
        
        self.results[model_name] = results
        
        print(f"  Training RMSE: ${train_rmse:,.2f}")
        print(f"  Validation RMSE: ${val_rmse:,.2f}")
        print(f"  Validation R²: {val_r2:.4f}")
        print(f"  Validation MAPE: {val_mape:.2f}%")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train all models
        """
        print("\n" + "="*50)
        print("TRAINING ALL MODELS")
        print("="*50)
        
        for model_name, model in self.models.items():
            self.train_model(model_name, model, X_train, y_train, X_val, y_val)
        
        return self.results
    
    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluate all trained models on test set
        """
        print("\n" + "="*50)
        print("TEST SET EVALUATION")
        print("="*50)
        
        test_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            test_results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return test_results
    
    def get_feature_importance(self, model_name, feature_names, top_n=10):
        """
        Get feature importance for tree-based models
        """
        model = self.models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not found")
            return None
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            return feature_importance[:top_n]
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def save_models(self):
        """
        Save all trained models
        """
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        
        for model_name, model in self.models.items():
            filename = f"{model_name.replace(' ', '_').lower()}.joblib"
            filepath = f"{config.MODELS_DIR}/{filename}"
            joblib.dump(model, filepath)
            print(f"✓ {model_name} saved to {filepath}")
    
    def load_model(self, model_name):
        """
        Load a saved model
        """
        filename = f"{model_name.replace(' ', '_').lower()}.joblib"
        filepath = f"{config.MODELS_DIR}/{filename}"
        model = joblib.load(filepath)
        self.models[model_name] = model
        return model
    
    def compare_models(self):
        """
        Compare all models and return summary
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        
        if not self.results:
            print("No results available. Train models first.")
            return None
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Val RMSE': f"${metrics['val_rmse']:,.2f}",
                'Val MAE': f"${metrics['val_mae']:,.2f}",
                'Val R²': f"{metrics['val_r2']:.4f}",
                'Val MAPE': f"{metrics['val_mape']:.2f}%"
            })
        
        # Sort by R² score
        comparison_data.sort(key=lambda x: float(x['Val R²']), reverse=True)
        
        # Print comparison table
        print(f"\n{'Model':<25} {'Val RMSE':<15} {'Val MAE':<15} {'Val R²':<12} {'Val MAPE':<12}")
        print("-" * 85)
        for item in comparison_data:
            print(f"{item['Model']:<25} {item['Val RMSE']:<15} {item['Val MAE']:<15} {item['Val R²']:<12} {item['Val MAPE']:<12}")
        
        # Find best model
        best_model = comparison_data[0]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        
        return comparison_data
