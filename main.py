import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "dataset"
SIZE = (256, 256)

def split_image(img):
    h, w, _ = img.shape
    mid = w // 2
    rgb = img[:, :mid]
    thermal = img[:, mid:]
    thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    return rgb, thermal

def preprocess(img):
    return cv2.resize(img, SIZE)

def get_features(rgb, thermal):
    r, g, b = cv2.split(rgb)

    features = [
        np.mean(r),
        np.mean(g),
        np.mean(b),
        np.mean(thermal),
        np.max(thermal),
        np.std(thermal)
    ]
    return features

def load_data():
    X = []
    y = []

    for label, folder in enumerate(["normal", "ulcer"]):
        path = os.path.join(DATASET_PATH, folder)

        if not os.path.exists(path):
            continue

        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess(img)
            rgb, thermal = split_image(img)

            features = get_features(rgb, thermal)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def predict(model, path):
    img = cv2.imread(path)
    img = preprocess(img)

    rgb, thermal = split_image(img)
    features = get_features(rgb, thermal)

    pred = model.predict([features])[0]

    return "NORMAL" if pred == 0 else "RISK"

if __name__ == "__main__":
    X, y = load_data()

    if len(X) == 0:
        print("No images found in dataset")
        exit()

    model = train_model(X, y)

    test_image = "dataset/ulcer/img2.jpg"
    result = predict(model, test_image)

    print("Prediction:", result)
