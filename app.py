import os
import pickle
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request
from skimage.feature import hog

app = Flask(__name__)

_model = None
_model_lock = threading.Lock()


def extract_hog_features(img_bgr):
    img = cv2.resize(img_bgr, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )


def get_model():
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.pkl")

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Pre-trained model not found at {model_path}. Please ensure model.pkl is in the deployment."
            )

        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        return _model


@app.get("/")
def index():
    return jsonify(
        {
            "message": "Bone fracture API is running",
            "usage": "POST /predict with form-data key 'image'",
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field: image"}), 400

    uploaded = request.files["image"]
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        model = get_model()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    features = extract_hog_features(img).reshape(1, -1)
    prediction = int(model.predict(features)[0])

    label = "fractured" if prediction == 0 else "not fractured"
    return jsonify({"prediction": label})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
