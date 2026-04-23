import os
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request
from skimage.feature import hog
from sklearn.svm import SVC

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


def load_data(folder):
    data = []
    labels = []

    for label, category in enumerate(["fractured", "not fractured"]):
        path = os.path.join(folder, category)
        if not os.path.exists(path):
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            data.append(extract_hog_features(img))
            labels.append(label)

    return np.array(data), np.array(labels)


def get_model():
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_dir = os.path.join(base_dir, "train")

        x_train, y_train = load_data(train_dir)
        if len(x_train) == 0:
            raise RuntimeError("No training data found in train/fractured and train/not fractured")

        model = SVC(kernel="linear")
        model.fit(x_train, y_train)
        _model = model
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
