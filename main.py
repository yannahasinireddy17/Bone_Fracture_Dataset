import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ---------------- LOAD DATA ----------------
def load_data(folder):
    data = []
    labels = []

    for label, category in enumerate(['fractured', 'not fractured']):
        path = os.path.join(folder, category)

        if not os.path.exists(path):
            print(f"Folder not found: {path}")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping invalid image: {img_path}")
                continue

            img = cv2.resize(img, (128, 128))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            features = hog(gray,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           visualize=False)

            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels)


# ---------------- TRAIN MODEL ----------------
print("Loading data...")

X_train, y_train = load_data("train")
X_test, y_test = load_data("test")

print("Training SVM model...")

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

print("Testing SVM model...")
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_acc)


# ---------------- RANDOM FOREST ----------------
print("\nTraining Random Forest model...")

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)


# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image_path):
    print("\nReading image from:", image_path)

    # FIX: normalize path (handles \ issues)
    image_path = os.path.normpath(image_path)

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Invalid image! Check:")
        print("- File path is correct")
        print("- Image exists")
        print("- Format is .jpg or .png")
        return

    print("✅ Image loaded successfully")

    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(gray,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   visualize=False)

    features = features.reshape(1, -1)

    prediction = svm_model.predict(features)

    print("Raw Prediction:", prediction)

    if prediction[0] == 0:
        print("🦴 Prediction: Fractured")
    else:
        print("🦴 Prediction: Not Fractured")


# ---------------- USER INPUT ----------------
print("\n--- Image Prediction ---")

image_path = input("Enter image path (or just filename if in same folder): ")

predict_image(image_path)