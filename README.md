# 💰 AI-Powered Employee Salary Prediction System

A professional machine learning system with a web interface that predicts employee salaries using ensemble learning techniques. Built with real Kaggle data and featuring 6 state-of-the-art ML models.

![Python](https://img.shields.io/badge/Python-3.14+-blue)
![Status](https://img.shields.io/badge/Status-Production-green)
![ML](https://img.shields.io/badge/ML-Ensemble-orange)

## ✨ Features

- 🎯 **Real-time Salary Predictions** - Interactive web interface
- 🤖 **6 ML Models** - Random Forest, XGBoost, LightGBM, Stacking, etc.
- 📊 **Data Insights** - Interactive visualizations and analytics
- 📈 **Model Comparison** - Compare predictions across all models
- 💼 **Real Data** - Trained on 607 real salary records from Kaggle
- 🎨 **Professional UI** - Modern, responsive Streamlit interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)
```bash
python main.py
```

### 3. Launch Web Application
```bash
streamlit run app.py
```

Or double-click `run_app.bat` (Windows)

The web app will open at `http://localhost:8501`

## 🤖 Machine Learning Models

### Ensemble Models Implemented
1. **🌲 Random Forest** - Bagging with decision trees
2. **📈 Gradient Boosting** - Sequential boosting
3. **⚡ XGBoost** - Extreme gradient boosting
4. **💡 LightGBM** - Light gradient boosting
5. **🗳️ Voting Ensemble** - Average predictions
6. **🏗️ Stacking Ensemble** - Meta-learning (Best: R² = 0.51)

## 📁 Project Structure

```
Project PBEL/
│
├── config.py                  # Configuration and hyperparameters
├── data_loader.py             # Data loading and preprocessing
├── models.py                  # Machine learning models
├── visualization.py           # Visualization utilities
├── main.py                    # Main pipeline script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── data/                      # Data directory
│   ├── salary_data.csv        # Raw data
│   └── processed_data.csv     # Processed data
│
├── models/                    # Saved models
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── xgboost.joblib
│   └── ...
│
└── results/                   # Results and visualizations
    ├── validation_results.csv
    ├── test_results.csv
    ├── model_comparison.png
    ├── predictions_vs_actual.png
    ├── residuals.png
    ├── feature_importance.png
    └── data_distribution.png
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load/create sample data
2. Preprocess and split data
3. Train all models (8 different models)
4. Evaluate models on validation and test sets
5. Generate visualizations
6. Save models and results

## 📈 Model Evaluation Metrics

The models are evaluated using:
- **RMSE** (Root Mean Squared Error) - Lower is better
- **MAE** (Mean Absolute Error) - Lower is better
- **R² Score** (Coefficient of Determination) - Higher is better (max 1.0)
- **MAPE** (Mean Absolute Percentage Error) - Lower is better

## 🎨 Visualizations

The project generates several visualizations:
1. **Data Distribution** - Salary distribution and feature analysis
2. **Model Comparison** - Performance comparison across all models
3. **Predictions vs Actual** - Scatter plot showing prediction accuracy
4. **Residuals Analysis** - Understanding prediction errors
5. **Feature Importance** - Most important features for prediction

## 🔧 Configuration

Edit `config.py` to modify:
- Model hyperparameters
- Data paths
- Train/test split ratios
- Feature lists
- Random seed for reproducibility

## 📊 Expected Results

The ensemble models typically achieve:
- **R² Score**: 0.85 - 0.95
- **RMSE**: $5,000 - $10,000
- **MAPE**: 5% - 10%

Best performing models are usually:
1. Stacking Ensemble
2. Voting Ensemble
3. XGBoost
4. LightGBM

## 🌟 Key Features

### Data Processing
- Automatic handling of missing values
- Label encoding for categorical features
- Standard scaling for numerical features
- Train/validation/test split

### Model Training
- Multiple ensemble techniques
- Cross-validation support
- Hyperparameter optimization ready
- Model persistence (save/load)

### Evaluation
- Comprehensive metrics
- Feature importance analysis
- Model comparison
- Residual analysis

## 📝 Notes

### Using Kaggle Data

To use real data from Kaggle:

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Set up Kaggle credentials:
   - Go to your Kaggle account settings
   - Create new API token
   - Place `kaggle.json` in `~/.kaggle/`

3. Download dataset:
```bash
kaggle datasets download -d <dataset-name>
```

4. Update `data_loader.py` to load your dataset

### Sample Datasets on Kaggle
- [Data Science Salaries 2023](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
- [Salary Prediction Dataset](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data)
- [Tech Salaries](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor)

## 🔄 Future Enhancements

- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Deep learning models (Neural Networks)
- [ ] Feature engineering and selection
- [ ] Cross-validation for ensemble models
- [ ] Real-time prediction API
- [ ] Interactive web dashboard
- [ ] Outlier detection and handling
- [ ] More sophisticated feature interactions

## 📚 Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: ML models and preprocessing
- xgboost: Gradient boosting
- lightgbm: Light gradient boosting
- matplotlib: Visualization
- seaborn: Statistical visualization
- joblib: Model persistence

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements!

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created for demonstrating ensemble machine learning techniques for salary prediction.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Predicting! 🎯**
# Salary-prediction
