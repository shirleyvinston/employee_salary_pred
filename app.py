"""
app.py
------
Flask web application for Employee Salary Category Prediction.

Loads the trained Random Forest model and label encoder, serves an
HTML form for users to enter employee details, and returns a
predicted salary category (High / Low).

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "trained_model.joblib")
ENCODER_PATH = os.path.join("model", "label_encoder.joblib")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURE_COLUMNS = ["age", "education_level", "experience", "income", "loan_amount"]

EDUCATION_MAP = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 2,
    "PhD": 3,
}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", education_options=list(EDUCATION_MAP.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        education_label = request.form["education_level"]
        experience = float(request.form["experience"])
        income = float(request.form["income"])
        loan_amount = float(request.form["loan_amount"])

        education_level = EDUCATION_MAP.get(education_label, 1)

        features = pd.DataFrame(
            [[age, education_level, experience, income, loan_amount]],
            columns=FEATURE_COLUMNS,
        )

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        predicted_label = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities)) * 100

        return render_template(
            "index.html",
            education_options=list(EDUCATION_MAP.keys()),
            prediction=predicted_label,
            confidence=round(confidence, 2),
            form_data=request.form,
        )

    except Exception as e:
        return render_template(
            "index.html",
            education_options=list(EDUCATION_MAP.keys()),
            error=str(e),
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint, e.g. for integration testing or a future frontend."""
    try:
        data = request.get_json()
        age = float(data["age"])
        education_level = int(data["education_level"])  # 0-3
        experience = float(data["experience"])
        income = float(data["income"])
        loan_amount = float(data["loan_amount"])

        features = pd.DataFrame(
            [[age, education_level, experience, income, loan_amount]],
            columns=FEATURE_COLUMNS,
        )
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            "salary_category": predicted_label,
            "confidence": round(float(np.max(probabilities)) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
