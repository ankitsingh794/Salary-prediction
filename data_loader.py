"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config


class DataLoader:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_sample_data(self, n_samples=5000):
        """
        Create sample salary data for demonstration
        In practice, you would load data from Kaggle using the Kaggle API
        """
        np.random.seed(config.RANDOM_STATE)
        
        # Job titles
        job_titles = [
            'Data Scientist', 'Software Engineer', 'Product Manager', 
            'Business Analyst', 'Data Analyst', 'Senior Developer',
            'ML Engineer', 'DevOps Engineer', 'QA Engineer', 'Tech Lead'
        ]
        
        # Education levels
        education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        
        # Locations
        locations = [
            'New York', 'San Francisco', 'Seattle', 'Austin', 'Boston',
            'Chicago', 'Los Angeles', 'Denver', 'Atlanta', 'Remote'
        ]
        
        # Departments
        departments = ['Engineering', 'Product', 'Data', 'Operations', 'Sales']
        
        # Company sizes
        company_sizes = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
        
        data = {
            'job_title': np.random.choice(job_titles, n_samples),
            'experience_years': np.random.randint(0, 25, n_samples),
            'education': np.random.choice(education_levels, n_samples, p=[0.1, 0.4, 0.35, 0.15]),
            'location': np.random.choice(locations, n_samples),
            'department': np.random.choice(departments, n_samples),
            'company_size': np.random.choice(company_sizes, n_samples),
            'age': np.random.randint(22, 65, n_samples),
            'hours_per_week': np.random.randint(35, 60, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create salary based on features (with some noise)
        base_salary = 50000
        
        # Job title impact
        job_title_salary = {
            'Data Scientist': 40000, 'Software Engineer': 35000, 'Product Manager': 45000,
            'Business Analyst': 25000, 'Data Analyst': 20000, 'Senior Developer': 50000,
            'ML Engineer': 45000, 'DevOps Engineer': 35000, 'QA Engineer': 25000, 'Tech Lead': 55000
        }
        df['salary_component_job'] = df['job_title'].map(job_title_salary)
        
        # Experience impact
        df['salary_component_exp'] = df['experience_years'] * 3000
        
        # Education impact
        education_salary = {'High School': 0, 'Bachelor': 15000, 'Master': 25000, 'PhD': 35000}
        df['salary_component_edu'] = df['education'].map(education_salary)
        
        # Location impact
        location_salary = {
            'New York': 20000, 'San Francisco': 25000, 'Seattle': 18000, 'Austin': 15000,
            'Boston': 18000, 'Chicago': 12000, 'Los Angeles': 17000, 'Denver': 13000,
            'Atlanta': 10000, 'Remote': 8000
        }
        df['salary_component_loc'] = df['location'].map(location_salary)
        
        # Company size impact
        company_size_salary = {'Startup': 5000, 'Small': 8000, 'Medium': 12000, 'Large': 18000, 'Enterprise': 25000}
        df['salary_component_size'] = df['company_size'].map(company_size_salary)
        
        # Calculate final salary with noise
        df['salary'] = (
            base_salary +
            df['salary_component_job'] +
            df['salary_component_exp'] +
            df['salary_component_edu'] +
            df['salary_component_loc'] +
            df['salary_component_size'] +
            np.random.normal(0, 10000, n_samples)  # Add noise
        )
        
        # Remove helper columns
        df = df.drop(columns=[col for col in df.columns if col.startswith('salary_component_')])
        
        # Ensure positive salaries
        df['salary'] = df['salary'].clip(lower=30000)
        
        return df
    
    def load_data(self, file_path=None, create_sample=True):
        """
        Load data from file or create sample data
        """
        # Check if ds_salaries.csv exists in current directory
        ds_salaries_path = 'ds_salaries.csv'
        import os
        if os.path.exists(ds_salaries_path) and file_path is None:
            print(f"Found existing data file: {ds_salaries_path}")
            print("Loading data...")
            df = pd.read_csv(ds_salaries_path)
            
            # Drop the first unnamed column if it exists
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            # Use salary_in_usd and drop the local currency salary column
            if 'salary_in_usd' in df.columns:
                # Drop the local currency salary column first
                if 'salary' in df.columns:
                    df = df.drop(columns=['salary', 'salary_currency'])
                # Rename salary_in_usd to salary
                df = df.rename(columns={'salary_in_usd': 'salary'})
            
            # Convert experience level to years (rough estimate)
            if 'experience_level' in df.columns:
                exp_map = {'EN': 2, 'MI': 5, 'SE': 10, 'EX': 15}
                df['experience_years'] = df['experience_level'].map(exp_map).fillna(5).astype(int)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'company_location': 'location',
                'company_size': 'company_size'
            })
            
            # Map company size to more readable format
            if 'company_size' in df.columns:
                size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
                df['company_size'] = df['company_size'].map(size_map).fillna(df['company_size'])
            
            # Add missing columns with defaults
            if 'education' not in df.columns:
                # Infer education from experience level
                edu_map = {2: 'Bachelor', 5: 'Bachelor', 10: 'Master', 15: 'PhD'}
                df['education'] = df['experience_years'].map(edu_map).fillna('Bachelor')
            
            if 'department' not in df.columns:
                df['department'] = 'Data'  # Default department
            
            if 'age' not in df.columns:
                # Estimate age based on experience: 22 (graduation) + experience years
                df['age'] = 22 + df['experience_years'] + np.random.randint(-2, 3, len(df))
            
            if 'hours_per_week' not in df.columns:
                # Map employment type to hours
                hours_map = {'FT': 40, 'PT': 20, 'CT': 35, 'FL': 30}
                if 'employment_type' in df.columns:
                    df['hours_per_week'] = df['employment_type'].map(hours_map).fillna(40).astype(int)
                else:
                    df['hours_per_week'] = 40
            
            print(f"Loaded {len(df)} records from existing file")
            return df
        elif create_sample or file_path is None:
            print("Creating sample data...")
            df = self.create_sample_data()
            df.to_csv(config.RAW_DATA_PATH, index=False)
            print(f"Sample data saved to {config.RAW_DATA_PATH}")
        else:
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path)
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data: handle missing values, encode categoricals, scale numericals
        """
        print("\nPreprocessing data...")
        
        # Create a copy for processing
        df_processed = df.copy()
        
        # Keep only required columns (drop extras from Kaggle dataset)
        required_cols = config.CATEGORICAL_FEATURES + config.NUMERICAL_FEATURES + ['salary']
        cols_to_keep = [col for col in required_cols if col in df_processed.columns]
        cols_to_drop = [col for col in df_processed.columns if col not in cols_to_keep]
        
        if cols_to_drop:
            print(f"Dropping unnecessary columns: {cols_to_drop}")
            df_processed = df_processed.drop(columns=cols_to_drop)
        
        # Check for missing values
        print(f"Missing values:\n{df_processed.isnull().sum()}")
        
        # Fill missing values if any
        for col in config.CATEGORICAL_FEATURES:
            if col in df_processed.columns:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        for col in config.NUMERICAL_FEATURES:
            if col in df_processed.columns:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Encode categorical features
        for col in config.CATEGORICAL_FEATURES:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        return df_processed
    
    def split_data(self, df):
        """
        Split data into train, validation, and test sets
        """
        # Separate features and target
        X = df.drop('salary', axis=1)
        y = df['salary']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE
        )
        
        # Scale numerical features
        numerical_cols = [col for col in config.NUMERICAL_FEATURES if col in X_train.columns]
        
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = self.scaler.transform(X_val[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print(f"\nData split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_info(self, df):
        """
        Get information about features
        """
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nBasic statistics:\n{df.describe()}")
        print(f"\nSalary statistics:")
        if 'salary' in df.columns:
            salary_mean = float(df['salary'].mean())
            salary_median = float(df['salary'].median())
            salary_std = float(df['salary'].std())
            salary_min = float(df['salary'].min())
            salary_max = float(df['salary'].max())
            print(f"  Mean: ${salary_mean:,.2f}")
            print(f"  Median: ${salary_median:,.2f}")
            print(f"  Std: ${salary_std:,.2f}")
            print(f"  Min: ${salary_min:,.2f}")
            print(f"  Max: ${salary_max:,.2f}")
