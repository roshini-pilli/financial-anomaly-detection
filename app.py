
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import math
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

saved = joblib.load(MODEL_PATH)

model = saved["model"]
threshold = float(saved["threshold"])
features = saved["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object"
        }), 400

    missing = [feature for feature in features if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing transaction features",
            "missing_features": missing
        }), 400

    try:
        values = [float(data[feature]) for feature in features]
    except (ValueError, TypeError, OverflowError):
        return jsonify({
            "error": "All transaction features must be numeric"
        }), 400

    if not all(math.isfinite(value) for value in values):
        return jsonify({
            "error": "Features must contain finite numbers"
        }), 400

    row = pd.DataFrame([values], columns=features)

    try:
        probability = float(model.predict_proba(row)[0, 1])
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Unable to predict this transaction"
        }), 500

    prediction = int(probability >= threshold)

    return jsonify({
        "prediction": prediction,
        "result": "Fraudulent" if prediction else "Legitimate",
        "fraud_probability": probability,
        "fraud_percentage": probability * 100,
        "threshold": threshold,
        "threshold_percentage": threshold * 100
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(features)
    })


if __name__ == "__main__":
    app.run(debug=False)
