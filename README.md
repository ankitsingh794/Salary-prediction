# 💼 Salary Prediction System

An AI-powered machine learning application that predicts employee salaries using ensemble techniques. Built with Python, scikit-learn, XGBoost, LightGBM, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red)

## 🎯 Overview

Machine learning system that predicts salaries based on 8 features using real Kaggle data (607 records). Includes 8 trained ensemble models and an interactive web interface.

**Best Model**: Stacking Ensemble (R² = 0.5154, RMSE = $43,096)

## ✨ Features

- 🤖 **8 ML Models**: Random Forest, Gradient Boosting, XGBoost, LightGBM, AdaBoost, Bagging, Voting, Stacking
- 🎨 **Interactive UI**: Real-time predictions with Streamlit
- 📊 **Visualizations**: Performance metrics and data insights with Plotly
- 💾 **Pre-trained Models**: Ready-to-use saved models
- 📈 **Comprehensive Metrics**: R², RMSE, MAE, MAPE

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ankitsingh794/Salary-prediction.git
cd Salary-prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

Open browser at `http://localhost:8501`

### Windows Quick Launch
```bash
run_app.bat
```

## 💻 Usage

### Web Interface
1. Navigate to **"Predict Salary"** page
2. Enter employee details:
   - Job title, experience, education, location, company size, department, age, hours/week
3. Click **"PREDICT SALARY"**
4. View predictions from multiple models with salary breakdown

### Python API
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/stacking_ensemble.joblib')

# Prepare data (correct feature order required)
data = pd.DataFrame({
    'job_title': [25], 'location': [45], 'company_size': [1],
    'experience_years': [5], 'education': [1], 'department': [0],
    'age': [30], 'hours_per_week': [40]
})

# Predict
salary = model.predict(data)[0]
print(f"Predicted: ${salary:,.0f}")
```

## 📊 Model Performance

| Model | R² Score | RMSE ($) | MAE ($) |
|-------|----------|----------|---------|
| **Stacking Ensemble** 🥇 | 0.5154 | 43,096 | 29,126 |
| Bagging Regressor | 0.4768 | 44,780 | 30,386 |
| Random Forest | 0.4736 | 44,916 | 31,364 |
| LightGBM | 0.4204 | 47,130 | 31,910 |
| XGBoost | 0.2902 | 52,159 | 38,081 |
| Gradient Boosting | 0.2609 | 53,221 | 37,958 |

## 📁 Project Structure

```
├── app.py                  # Streamlit web app
├── main.py                 # Model training pipeline
├── models.py               # ML implementations
├── data_loader.py          # Data preprocessing
├── visualization.py        # Plotting functions
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── models/                # Trained models (8 .joblib files)
├── results/               # Metrics and visualizations
└── ds_salaries.csv       # Dataset (607 records)
```

## 🛠️ Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **Web**: Streamlit, Plotly
- **Data**: pandas, numpy
- **Viz**: matplotlib, seaborn

## 🔧 Retrain Models

```bash
python main.py
```

This will train all models, evaluate performance, and save to `models/` directory.

## 🚀 Deployment

**Streamlit Cloud** (Recommended):
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

## 📈 Dataset

- **Source**: Real Kaggle salary data
- **Records**: 607 employees
- **Features**: Job title, experience, education, location, company size, department, age, hours/week
- **Target**: Annual salary (INR)
- **Range**: ₹2,37,297 - ₹49,80,00,000
- **Mean**: ₹93,20,734

## 👨‍💻 Author

**Ankit Singh**
- GitHub: [@ankitsingh794](https://github.com/ankitsingh794)

## 📝 License

Open-source, available for educational purposes.

---

⭐ **Star this repo if you find it helpful!**
