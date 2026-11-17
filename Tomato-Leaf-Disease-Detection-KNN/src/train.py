"""Train and evaluate a KNN model from class-labelled image folders."""
from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features, list_images


def load_dataset(data_dir: Path):
    features, labels = [], []
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for class_dir in class_dirs:
        image_paths = list(list_images(class_dir))
        for image_path in image_paths:
            try:
                features.append(extract_features(str(image_path)))
                labels.append(class_dir.name)
            except ValueError as exc:
                print(f"Skipping: {exc}")

    if not features:
        raise RuntimeError(
            f"No readable images found in {data_dir}. "
            "Add class subfolders containing image files."
        )

    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        raise RuntimeError("At least two classes with images are required.")
    if counts.min() < 2:
        raise RuntimeError(
            "Each class needs at least two images for a stratified train/test split."
        )

    return pd.DataFrame(features), pd.Series(labels, name="label")


def main():
    parser = argparse.ArgumentParser(description="Train tomato leaf KNN classifier")
    parser.add_argument("--data_dir", default="data/dataset", help="Root folder with one subfolder per class")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbours")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction held out for testing")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")
    if not 0 < args.test_size < 1:
        parser.error("--test_size must be between 0 and 1")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    k = min(args.k, len(X_train))
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Images loaded: {len(y)}")
    print(f"Classes: {sorted(y.unique())}")
    print(f"Training images: {len(y_train)} | Test images: {len(y_test)}")
    print(f"K value: {k}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "classes": sorted(y.unique())},
        "models/tomato_leaf_knn.joblib",
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, xticks_rotation=45, cmap="Blues"
    )
    display.figure_.tight_layout()
    display.figure_.savefig("results/confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    }).to_csv("results/test_predictions.csv", index=False)

    print("\nSaved model to models/tomato_leaf_knn.joblib")
    print("Saved evaluation files in results/")


if __name__ == "__main__":
    main()
