"""Predict the dataset class for one tomato leaf image."""
from pathlib import Path
import argparse
import joblib

from features import extract_features


def main():
    parser = argparse.ArgumentParser(description="Predict tomato leaf image class")
    parser.add_argument("--image", required=True, help="Path to a tomato leaf image")
    parser.add_argument("--model", default="models/tomato_leaf_knn.joblib")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first using src/train.py."
        )

    saved = joblib.load(model_path)
    features = extract_features(str(image_path)).reshape(1, -1)
    prediction = saved["model"].predict(features)[0]

    print(f"Predicted class: {prediction}")
    if hasattr(saved["model"], "predict_proba"):
        probabilities = saved["model"].predict_proba(features)[0]
        print("Class scores:")
        for label, score in zip(saved["model"].classes_, probabilities):
            print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
