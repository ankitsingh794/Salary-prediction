"""
Configuration file for the salary prediction project
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data file paths
RAW_DATA_PATH = os.path.join(DATA_DIR, 'salary_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering
CATEGORICAL_FEATURES = ['job_title', 'education', 'location', 'department', 'company_size']
NUMERICAL_FEATURES = ['experience_years', 'age', 'hours_per_week']

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.8,
    'random_state': RANDOM_STATE
}

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

LIGHTGBM_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}
