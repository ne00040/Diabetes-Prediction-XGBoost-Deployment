#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# 0) Config — final features set
# -----------------------------
FINAL_NUMERIC_FEATURES = [
    "age",
    "bmi",
    "family_history_diabetes",  
    "glucose_fasting",
    "ldl_cholesterol",
    "physical_activity_minutes_per_week",
    "systolic_bp",
    "waist_to_hip_ratio",
]
FINAL_CATEGORICAL_FEATURES = [
    "gender"  
]

# -----------------------------
# 1) Load dataset for min/max
# -----------------------------
DATASET_PATH = "diabetes_dataset.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset '{DATASET_PATH}' not found. Place it next to app_optionB.py."
    )

df = pd.read_csv(DATASET_PATH)

feature_limits = {}
for col in FINAL_NUMERIC_FEATURES:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' missing from dataset.")
    series = pd.to_numeric(df[col], errors="coerce")
    feature_limits[col] = {
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }

gender_values = sorted(df["gender"].dropna().unique().tolist())

# -----------------------------
# 2) Load files
# -----------------------------
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
SCALER_PATH           = "scaler.pkl"
MODEL_XGB_PATH        = "best_model_xgb.pkl"
MODEL_LR_PATH         = "best_model_logreg.pkl"

if not os.path.exists(FEATURE_COLUMNS_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "Missing files. Ensure 'feature_columns.pkl' and 'scaler.pkl' exist."
    )

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
scaler          = joblib.load(SCALER_PATH)

model = None
model_name = None
if os.path.exists(MODEL_XGB_PATH):
    model = joblib.load(MODEL_XGB_PATH)
    model_name = "XGBoost"
elif os.path.exists(MODEL_LR_PATH):
    model = joblib.load(MODEL_LR_PATH)
    model_name = "Logistic Regression"
else:
    raise FileNotFoundError(
        "No best model found. Expected 'best_model_xgb.pkl' or 'best_model_logreg.pkl'."
    )

# -----------------------------
# 3) Helper — build input row
# -----------------------------
def clamp(value, lo, hi):
    """Clamp to [lo, hi] and return clamped value plus a flag if clamped."""
    clamped = min(max(value, lo), hi)
    return clamped, (clamped != value)

def build_input_dataframe(form):
    """
    Build a single-row raw DataFrame from POSTed form values.
    - Validates numeric ranges using dataset min/max.
    - Categorical 'gender' comes from allowed values found in dataset.
    Returns: df_raw (1xN DataFrame), warnings (list of strings)
    """
    warnings = []

    row = {}
    for col in FINAL_NUMERIC_FEATURES:
        try:
            val = float(form[col])
        except Exception:
            val = feature_limits[col]["median"]
            warnings.append(f"'{col}' not a valid number. Using dataset median {val}.")

        lo = feature_limits[col]["min"]
        hi = feature_limits[col]["max"]
        val_clamped, was_clamped = clamp(val, lo, hi)
        if was_clamped:
            warnings.append(f"'{col}' clamped to dataset bounds [{lo}, {hi}] (input={val}).")
        row[col] = val_clamped

    # Categorical: gender
    gender = form.get("gender", None)
    if (gender is None) or (gender not in gender_values):
        most_freq = df["gender"].value_counts().idx1 if hasattr(df["gender"].value_counts(), "idx1") else df["gender"].value_counts().idxmax()
        warnings.append(f"'gender' invalid/missing. Using most frequent '{most_freq}'.")
        gender = most_freq

    row["gender"] = gender

    df_raw = pd.DataFrame([row])

    return df_raw, warnings

# -----------------------------
# 4) HTML template
# -----------------------------
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Diabetes Prediction</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#f7f7fb; color:#222; margin:0; }
    .container { max-width: 960px; margin: 40px auto; background:#fff; border-radius:12px; box-shadow:0 6px 20px rgba(0,0,0,.08); padding: 24px 28px; }
    h1 { margin:0 0 8px; }
    .subtitle { color:#555; margin-bottom:18px; }
    form { display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .group { display:flex; flex-direction:column; }
    label { font-size: 13px; color:#333; margin-bottom:6px; }
    input[type="number"], select { padding:10px; border:1px solid #ddd; border-radius:8px; background:#fafafa; }
    .btnrow { grid-column: 1 / -1; display:flex; gap:12px; margin-top:10px; }
    button { padding:10px 16px; border:none; border-radius:8px; cursor:pointer; background:#2f6fed; color:#fff; font-weight:600; }
    .secondary { background:#e9eefc; color:#2f6fed; }
    .result { margin-top: 18px; padding:12px 14px; border-radius:8px; background:#f0fdf4; color:#03543f; border:1px solid #d1fae5; }
    .error  { background:#fef2f2; color:#9b1c1c; border:1px solid #fee2e2; }
    .note { color:#666; font-size: 12px; margin-top: 4px; }
    .warnings { margin-top:12px; background:#fff7ed; color:#7c2d12; border:1px solid #fed7aa; padding:10px 12px; border-radius:8px; }
    .small { font-size: 12px; color:#666; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Diabetes Prediction</h1>

    <form method="post">
      <!-- Demographic & Lifestyle -->
      <div class="group">
        <label for="age">Age</label>
        <input type="number" step="1" name="age" id="age" required min="{{ limits.age.min }}" max="{{ limits.age.max }}">
        <span class="note small">Bounds: {{ limits.age.min }}–{{ limits.age.max }}</span>
      </div>

      <div class="group">
        <label for="gender">Gender</label>
        <select name="gender" id="gender" required>
          {% for g in gender_values %}
            <option value="{{ g }}">{{ g }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="group">
        <label for="physical_activity_minutes_per_week">Physical activity (min/week)</label>
        <input type="number" step="1" name="physical_activity_minutes_per_week" id="physical_activity_minutes_per_week" required
               min="{{ limits.physical_activity_minutes_per_week.min }}" max="{{ limits.physical_activity_minutes_per_week.max }}">
        <span class="note small">Bounds: {{ limits.physical_activity_minutes_per_week.min }}–{{ limits.physical_activity_minutes_per_week.max }}</span>
      </div>

      <!-- Clinical -->
      <div class="group">
        <label for="bmi">BMI</label>
        <input type="number" step="0.1" name="bmi" id="bmi" required min="{{ limits.bmi.min }}" max="{{ limits.bmi.max }}">
        <span class="note small">Bounds: {{ limits.bmi.min }}–{{ limits.bmi.max }}</span>
      </div>

      <div class="group">
        <label for="waist_to_hip_ratio">Waist-to-hip ratio</label>
        <input type="number" step="0.01" name="waist_to_hip_ratio" id="waist_to_hip_ratio" required
               min="{{ limits.waist_to_hip_ratio.min }}" max="{{ limits.waist_to_hip_ratio.max }}">
        <span class="note small">Bounds: {{ limits.waist_to_hip_ratio.min }}–{{ limits.waist_to_hip_ratio.max }}</span>
      </div>

      <div class="group">
        <label for="systolic_bp">Systolic BP</label>
        <input type="number" step="1" name="systolic_bp" id="systolic_bp" required min="{{ limits.systolic_bp.min }}" max="{{ limits.systolic_bp.max }}">
        <span class="note small">Bounds: {{ limits.systolic_bp.min }}–{{ limits.systolic_bp.max }}</span>
      </div>

      <div class="group">
        <label for="glucose_fasting">Glucose (fasting)</label>
        <input type="number" step="1" name="glucose_fasting" id="glucose_fasting" required
               min="{{ limits.glucose_fasting.min }}" max="{{ limits.glucose_fasting.max }}">
        <span class="note small">Bounds: {{ limits.glucose_fasting.min }}–{{ limits.glucose_fasting.max }}</span>
      </div>

      <div class="group">
        <label for="ldl_cholesterol">LDL Cholesterol</label>
        <input type="number" step="1" name="ldl_cholesterol" id="ldl_cholesterol" required
               min="{{ limits.ldl_cholesterol.min }}" max="{{ limits.ldl_cholesterol.max }}">
        <span class="note small">Bounds: {{ limits.ldl_cholesterol.min }}–{{ limits.ldl_cholesterol.max }}</span>
      </div>

      <div class="group">
        <label for="family_history_diabetes">Family history of diabetes (0 = No, 1 = Yes)</label>
        <input type="number" step="1" name="family_history_diabetes" id="family_history_diabetes" required
               min="{{ limits.family_history_diabetes.min }}" max="{{ limits.family_history_diabetes.max }}">
        <span class="note small">Bounds: {{ limits.family_history_diabetes.min }}–{{ limits.family_history_diabetes.max }}</span>
      </div>

      <div class="btnrow">
        <button type="submit">Predict</button>
        <button type="reset" class="secondary">Reset</button>
      </div>
    </form>

    {% if warnings and warnings|length > 0 %}
      <div class="warnings"><strong>Note:</strong>
        <ul>
          {% for w in warnings %}<li>{{ w }}</li>{% endfor %}
        </ul>
      </div>
    {% endif %}

    {% if error %}
      <div class="result error">Error: {{ error }}</div>
    {% endif %}

    {% if prediction is not none %}
      <div class="result">
        <div><strong>Model:</strong> {{ model_name }}</div>
        <div><strong>Prediction:</strong> {{ 'Diabetes' if prediction == 1 else 'No Diabetes' }}</div>
        {% if prob is not none %}
          <div><strong>Probability (diabetes):</strong> {{ '{:.3f}'.format(prob) }}</div>
        {% endif %}
      </div>
    {% endif %}

    <div class="small" style="margin-top:16px;">
      This app uses dataset-driven bounds and the same preprocessing as training:
      one-hot ('gender', drop_first=True) → reindex to saved feature columns → scale → predict.
    </div>
  </div>
</body>
</html>
"""

# -----------------------------
# 5) Flask app
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    prob = None
    warnings = []

    if request.method == "POST":
        try:
            df_raw, warnings = build_input_dataframe(request.form)

            df_encoded = pd.get_dummies(df_raw, columns=FINAL_CATEGORICAL_FEATURES, drop_first=True)

            X_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

            X_scaled = scaler.transform(X_aligned)

            pred = int(model.predict(X_scaled)[0])
            prediction = pred

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_scaled)[0, 1])

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        limits=feature_limits,
        gender_values=gender_values,
        prediction=prediction,
        prob=prob,
        warnings=warnings,
        error=error,
        model_name=model_name,
    )

if __name__ == "__main__":
    app.run(debug=True)