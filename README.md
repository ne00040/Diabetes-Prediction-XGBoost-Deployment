# Diabetes-Prediction-XGBoost-Deployment

## 📌 Project Overview
This project predicts whether an individual is diagnosed with diabetes based on demographic, lifestyle, and clinical factors.  
It introduces **XGBoost**, compares its performance with Logistic Regression, and deploys the best model as a **Flask RESTful API** with a simple HTML/CSS interface.


## ✅ How to Run the Project
1. Ensure the following files are in the same folder:
   - `Ekwenibe_Project3.ipynb` – Jupyter Notebook for data preprocessing, model training, and evaluation.
   - `app.py` – Flask application for deployment.
   - `best_diabetes_model.pkl` – Saved XGBoost model.
   - `scaler.pkl` – Saved MinMaxScaler.
   - `diabetes_dataset.csv` – Dataset used for training.

2. Run the Jupyter Notebook file: `Ekwenibe_Project3.ipynb`
   
3. Run the Flask app: `python app.py`
   
4. Open your browser and go to: `http://127.0.0.1:5000`
