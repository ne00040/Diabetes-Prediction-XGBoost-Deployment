from flask import Flask, request, render_template_string
import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('best_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# HTML template with CSS styling
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0; }
        .container { width: 50%; margin: auto; background: #fff; padding: 20px; margin-top: 50px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        input, select { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; }
        button { background-color: #28a745; color: white; padding: 10px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
        button:hover { background-color: #218838; }
        .result { text-align: center; font-size: 18px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Diabetes Prediction</h2>
        <form method="POST">
            <input type="number" name="age" placeholder="Age" required>
            <select name="gender">
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>
            <select name="ethnicity">
                <option value="White">White</option>
                <option value="Black">Black</option>
                <option value="Asian">Asian</option>
                <option value="Hispanic">Hispanic</option>
                <option value="Other">Other</option>
            </select>
            <input type="number" name="bmi" placeholder="BMI" required>
            <input type="number" name="glucose_fasting" placeholder="Glucose Fasting" required>
            <input type="number" name="cholesterol_total" placeholder="Total Cholesterol" required>
            <button type="submit">Predict</button>
        </form>
        {% if prediction is not none %}
            <div class="result">Prediction: <strong>{{ prediction }}</strong></div>
        {% endif %}
    </div>
</body>
</html>
"""

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Collect input values
        age = float(request.form['age'])
        gender = request.form['gender']
        ethnicity = request.form['ethnicity']
        bmi = float(request.form['bmi'])
        glucose_fasting = float(request.form['glucose_fasting'])
        cholesterol_total = float(request.form['cholesterol_total'])

        # Create dataframe for input
        input_df = pd.DataFrame([[age, gender, ethnicity, bmi, glucose_fasting, cholesterol_total]],
                                columns=['age','gender','ethnicity','bmi','glucose_fasting','cholesterol_total'])

        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_df, columns=['gender','ethnicity'], drop_first=True)

        # Align with training columns
        all_features = scaler.feature_names_in_
        input_encoded = input_encoded.reindex(columns=all_features, fill_value=0)

        # Scale numeric features
        input_scaled = scaler.transform(input_encoded)

        # Predict
        pred = model.predict(input_scaled)[0]
        prediction = 'Diabetes' if pred == 1 else 'No Diabetes'

    return render_template_string(html_template, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)