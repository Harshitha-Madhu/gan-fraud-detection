import os
import numpy as np
import joblib

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

# Current file directory
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Template folder path
TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "..",
    "templates"
)

# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "fraud_classifier.pkl"
)

# Create Flask app
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR
)

# Load model
model = None

try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Fraud classifier loaded successfully"
    )

except Exception as e:

    print(
        "Model loading error:",
        e
    )


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if model is None:

            return jsonify({
                "error":
                "Model not loaded"
            }), 500

        data = request.json

        features = data.get(
            "features",
            []
        )

        # validate input count
        if len(features) != 30:

            return jsonify({
                "error":
                "Please enter exactly 30 values"
            }), 400

        transaction = np.array(
            features
        ).reshape(1, -1)

        score = float(
            model.predict_proba(
                transaction
            )[0][1]
        )

        prediction = (
            "Fraud"
            if score > 0.5
            else "Normal"
        )

        result = {

            "prediction":
            prediction,

            "fraud_probability":
            round(
                score,
                4
            )
        }

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({
            "error":
            str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        debug=True
    )