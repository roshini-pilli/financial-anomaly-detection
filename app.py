
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

saved = joblib.load("models/fraud_model.pkl")
model = saved["model"]
threshold = saved["threshold"]
features = saved["features"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object"}), 400

    missing = [f for f in features if f not in data]

    if missing:
        return jsonify({
            "error": "Missing transaction features",
            "missing_features": missing
        }), 400

    try:
        row = pd.DataFrame(
            [[float(data[f]) for f in features]],
            columns=features
        )
    except (ValueError, TypeError):
        return jsonify({
            "error": "All transaction features must be numeric"
        }), 400

    if row.isna().any().any() or not row.map(
        lambda value: value == value and abs(value) != float("inf")
    ).all().all():
        return jsonify({
            "error": "Features must contain finite numbers"
        }), 400

    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= threshold)

    return jsonify({
        "prediction": prediction,
        "result": "Fraudulent" if prediction else "Legitimate",
        "fraud_probability": round(probability, 6),
        "threshold": threshold
    })

if __name__ == "__main__":
    app.run(debug=False)
