
#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# 0) Paths & required deliverables
# ----------------------------
DATASET_PATH = "diabetes_dataset.csv"          
FEATURE_COLUMNS_PATH = "feature_columns.pkl"     
SCALER_PATH = "scaler.pkl"                      
MODEL_XGB_PATH = "best_model_xgb.pkl"           
MODEL_LR_PATH  = "best_model_logreg.pkl"         

# ----------------------------
# 1) Load dataset
# ----------------------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset '{DATASET_PATH}' not found. Place it next to app.py."
    )
df = pd.read_csv(DATASET_PATH) 
if not os.path.exists(FEATURE_COLUMNS_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "Missing files. Ensure 'feature_columns.pkl' and 'scaler.pkl' exist."
    )

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)  
scaler = joblib.load(SCALER_PATH)

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

# ---------------------------------------------------------
# 2) Infer raw features from post–one-hot feature_columns
# ---------------------------------------------------------
dataset_cols = df.columns.tolist()

raw_numeric_features = []      
raw_categorical_features = {}  

for col in feature_columns:
    if col in dataset_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col not in raw_numeric_features:
                raw_numeric_features.append(col)
        else:
            if col not in raw_categorical_features:
                raw_categorical_features[col] = sorted(df[col].dropna().unique().tolist())
    else:
        if '_' in col:
            base, level = col.split('_', 1)
            if base in dataset_cols and not pd.api.types.is_numeric_dtype(df[base]):
                vals = sorted(df[base].dropna().unique().tolist())
                raw_categorical_features[base] = vals

feature_limits = {}
for col in raw_numeric_features:
    series = pd.to_numeric(df[col], errors="coerce")
    feature_limits[col] = {
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }

# ----------------------------
# 3) Helpers
# ----------------------------
def clamp(value, lo, hi):
    """Clamp to [lo, hi] and return (clamped_value, was_clamped)."""
    clamped = min(max(value, lo), hi)
    return clamped, (clamped != value)

def encode_and_align(df_raw_row, categorical_bases):
    """
    One-hot encode categorical bases with drop_first=True, then reindex to feature_columns.
    Returns X_aligned (1xN DataFrame).
    """
    if categorical_bases:
        df_encoded = pd.get_dummies(df_raw_row, columns=categorical_bases, drop_first=True)
    else:
        df_encoded = df_raw_row.copy()
    X_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return X_aligned

# ----------------------------
# 4) HTML 
# ----------------------------
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
    .error { background:#fef2f2; color:#9b1c1c; border:1px solid #fee2e2; }
    .note { color:#666; font-size: 12px; margin-top: 4px; }
    .warnings { margin-top:12px; background:#fff7ed; color:#7c2d12; border:1px solid #fed7aa; padding:10px 12px; border-radius:8px; }
    .small { font-size: 12px; color:#666; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Diabetes Prediction — {{ model_name }}</h1>
    <form method="post">

      {% for col in numeric_features %}
      <div class="group">
        <label for="{{ col }}">{{ col }}</label>
        <input type="number" step="any" name="{{ col }}" id="{{ col }}" required
               min="{{ limits[col].min }}" max="{{ limits[col].max }}">
        <span class="note small">Bounds: {{ limits[col].min }}–{{ limits[col].max }}</span>
      </div>
      {% endfor %}

      {% for col, values in categorical_features.items() %}
      <div class="group">
        <label for="{{ col }}">{{ col }}</label>
        <select name="{{ col }}" id="{{ col }}" required>
          {% for v in values %}
          <option value="{{ v }}">{{ v }}</option>
          {% endfor %}
        </select>
      </div>
      {% endfor %}

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
        <div><strong>Prediction:</strong> {{ 'Diabetes' if prediction == 1 else 'No Diabetes' }}</div>
        {% if prob is not none %}
        <div><strong>Probability (diabetes):</strong> {{ '{:.3f}'.format(prob) }}</div>
        {% endif %}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

# ----------------------------
# 5) Flask app
# ----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    prob = None
    warnings = []

    if request.method == "POST":
        try:
            row = {}
            for col in raw_numeric_features:
                try:
                    val = float(request.form[col])
                except Exception:
                    val = feature_limits[col]["median"]
                    warnings.append(f"'{col}' not a valid number. Using dataset median {val}.")
                lo = feature_limits[col]["min"]
                hi = feature_limits[col]["max"]
                val_clamped, was_clamped = clamp(val, lo, hi)
                if was_clamped:
                    warnings.append(f"'{col}' clamped to dataset bounds [{lo}, {hi}] (input={val}).")
                row[col] = val_clamped

            for col, values in raw_categorical_features.items():
                val = request.form.get(col, None)
                if (val is None) or (val not in values):
                    most_freq = df[col].value_counts().idxmax()
                    warnings.append(f"'{col}' invalid/missing. Using most frequent '{most_freq}'.")
                    val = most_freq
                row[col] = val

            df_raw = pd.DataFrame([row])

            categorical_bases = list(raw_categorical_features.keys())
            X_aligned = encode_and_align(df_raw, categorical_bases)

            X_scaled = scaler.transform(X_aligned.values)
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_scaled)[0, 1])
                pred = int(prob >= 0.5)  
            else:
                pred = int(model.predict(X_scaled)[0])
            prediction = pred

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        model_name=model_name,
        limits=feature_limits,
        numeric_features=raw_numeric_features,
        categorical_features=raw_categorical_features,
        prediction=prediction,
        prob=prob,
        warnings=warnings,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)