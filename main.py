"""
Main script to run the salary prediction pipeline
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from data_loader import DataLoader
from models import SalaryPredictionModels
from visualization import ModelVisualizer
import config


def main():
    """
    Main function to run the complete pipeline
    """
    print("="*70)
    print(" " * 15 + "EMPLOYEE SALARY PREDICTION")
    print(" " * 10 + "Using Ensemble Machine Learning Models")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and Preprocessing Data")
    print("-" * 70)
    
    data_loader = DataLoader()
    
    # Load or create sample data
    df = data_loader.load_data(create_sample=True)
    
    # Display data information
    data_loader.get_feature_info(df)
    
    # Preprocess data
    df_processed = data_loader.preprocess_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(df_processed)
    
    # Step 2: Build and train models
    print("\n[STEP 2] Building and Training Models")
    print("-" * 70)
    
    model_manager = SalaryPredictionModels()
    
    # Build base models
    model_manager.build_models()
    
    # Build ensemble models
    model_manager.build_ensemble_models()
    
    # Train all models
    results = model_manager.train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare models
    model_manager.compare_models()
    
    # Step 3: Evaluate on test set
    print("\n[STEP 3] Final Evaluation on Test Set")
    print("-" * 70)
    
    test_results = model_manager.evaluate_on_test(X_test, y_test)
    
    # Find best model
    best_model_name = max(test_results, key=lambda x: test_results[x]['r2'])
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test R²: {test_results[best_model_name]['r2']:.4f}")
    print(f"   Test RMSE: ${test_results[best_model_name]['rmse']:,.2f}")
    print(f"   Test MAE: ${test_results[best_model_name]['mae']:,.2f}")
    print(f"   Test MAPE: {test_results[best_model_name]['mape']:.2f}%")
    
    # Step 4: Feature importance
    print("\n[STEP 4] Feature Importance Analysis")
    print("-" * 70)
    
    feature_names = X_train.columns.tolist()
    
    # Get feature importance for best model
    importance = model_manager.get_feature_importance(best_model_name, feature_names, top_n=10)
    if importance:
        print(f"\nTop 10 Important Features ({best_model_name}):")
        for i, (feature, imp) in enumerate(importance, 1):
            print(f"  {i}. {feature}: {imp:.4f}")
    
    # Step 5: Save models
    print("\n[STEP 5] Saving Models")
    print("-" * 70)
    
    model_manager.save_models()
    
    # Step 6: Visualizations
    print("\n[STEP 6] Generating Visualizations")
    print("-" * 70)
    
    visualizer = ModelVisualizer()
    
    # Data distribution
    print("\nGenerating data distribution plots...")
    visualizer.plot_data_distribution(df, 
        save_path=f"{config.RESULTS_DIR}/data_distribution.png")
    
    # Model comparison
    print("\nGenerating model comparison plot...")
    visualizer.plot_model_comparison(results, metric='val_r2',
        save_path=f"{config.RESULTS_DIR}/model_comparison.png")
    
    # Predictions vs Actual for best model
    print(f"\nGenerating predictions plot for {best_model_name}...")
    best_model = model_manager.models[best_model_name]
    visualizer.plot_predictions_vs_actual(best_model, X_test, y_test, best_model_name,
        save_path=f"{config.RESULTS_DIR}/predictions_vs_actual.png")
    
    # Residuals for best model
    print(f"\nGenerating residuals plot for {best_model_name}...")
    visualizer.plot_residuals(best_model, X_test, y_test, best_model_name,
        save_path=f"{config.RESULTS_DIR}/residuals.png")
    
    # Feature importance for best model
    if importance:
        print(f"\nGenerating feature importance plot for {best_model_name}...")
        visualizer.plot_feature_importance(importance, best_model_name,
            save_path=f"{config.RESULTS_DIR}/feature_importance.png")
    
    # Step 7: Save results to CSV
    print("\n[STEP 7] Saving Results")
    print("-" * 70)
    
    # Validation results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{config.RESULTS_DIR}/validation_results.csv")
    print(f"✓ Validation results saved to {config.RESULTS_DIR}/validation_results.csv")
    
    # Test results
    test_results_df = pd.DataFrame(test_results).T
    test_results_df.to_csv(f"{config.RESULTS_DIR}/test_results.csv")
    print(f"✓ Test results saved to {config.RESULTS_DIR}/test_results.csv")
    
    # Step 8: Make sample predictions
    print("\n[STEP 8] Sample Predictions")
    print("-" * 70)
    
    # Make predictions on first 5 test samples
    n_samples = 5
    X_sample = X_test.iloc[:n_samples]
    y_sample = y_test.iloc[:n_samples]
    
    predictions = best_model.predict(X_sample)
    
    print(f"\nSample predictions using {best_model_name}:")
    print(f"{'Actual':<15} {'Predicted':<15} {'Difference':<15} {'Error %':<15}")
    print("-" * 60)
    for actual, pred in zip(y_sample, predictions):
        diff = pred - actual
        error_pct = (diff / actual) * 100
        print(f"${actual:,.2f}      ${pred:,.2f}      ${diff:,.2f}      {error_pct:+.2f}%")
    
    print("\n" + "="*70)
    print(" " * 20 + "PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Summary:")
    print(f"   • Total Models Trained: {len(model_manager.models)}")
    print(f"   • Best Model: {best_model_name}")
    print(f"   • Best Test R²: {test_results[best_model_name]['r2']:.4f}")
    print(f"   • Models saved in: {config.MODELS_DIR}")
    print(f"   • Results saved in: {config.RESULTS_DIR}")
    
    return model_manager, test_results, best_model_name


if __name__ == "__main__":
    model_manager, test_results, best_model_name = main()
