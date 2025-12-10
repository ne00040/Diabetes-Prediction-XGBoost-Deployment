# Diabetes-Prediction-XGBoost-Deployment

## Project Overview
This project predicts whether an individual is diagnosed with diabetes based on demographic, lifestyle, and clinical factors.  
It introduces **XGBoost**, compares its performance with Logistic Regression, and deploys the best model as a **Flask RESTful API** with a simple HTML/CSS interface.


## How to Run the Project
1. Ensure the following files are in the same folder:
   - `Ekwenibe_Project3.ipynb` – Jupyter Notebook for data preprocessing, model training, and evaluation.
   - `app.py` – Flask application for deployment.
   - `best_model_xgb.pkl` **or** `best_model_logreg.pkl` – Saved best model from the Option B training run
   - `scaler.pkl` – Saved MinMaxScaler.
   - `diabetes_dataset.csv` – Dataset used for training.
   - `feature_columns.pkl` – Saved list of **post–one-hot** feature column names (used to reindex inputs in Flask).

2. Run the Jupyter Notebook file: `Ekwenibe_Project3.ipynb`

3. Install dependency: `pip install flask joblib pandas xgboost scikit-learn`
   
4. Run the Flask app: `python app.py`
   
5. Open your browser and go to: `http://127.0.0.1:5000`


## Tested with real dataset values

### Values that will predict diabetes
age = 68
gender = Male
physical_activity_minutes_per_week = 10
bmi = 34.5
waist_to_hip_ratio = 0.98
systolic_bp = 155
glucose_fasting = 160
glucose_postprandial = 240
ldl_cholesterol = 170
family_history_diabetes = 1

### Values that will not predict diabetes
age = 24
gender = Female
physical_activity_minutes_per_week = 180
bmi = 21.5
waist_to_hip_ratio = 0.78
systolic_bp = 112
glucose_fasting = 85
glucose_postprandial = 115
ldl_cholesterol = 95
family_history_diabetes = 0


> **Note:** The app dynamically reflects the trained feature set by reading `feature_columns.pkl`. Numeric inputs use dataset‑derived bounds (min/max), and categorical inputs use values found in `diabetes_dataset.csv`. The app performs one‑hot (drop_first=True), reindexes to `feature_columns.pkl`, scales with `scaler.pkl`, and predicts with the saved best model.
