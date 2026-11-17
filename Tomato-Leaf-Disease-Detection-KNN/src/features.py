"""Image feature extraction for tomato leaf KNN classification."""
from pathlib import Path
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_features(image_path: str, image_size=(128, 128)) -> np.ndarray:
    """Extract a compact feature vector from one image.

    Features: normalized HSV histograms, grayscale mean/std, and
    Laplacian variance. These are starter features and can be replaced
    with the exact feature extraction used in the original project.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, image_size)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist_features = []
    # Separate histograms for hue, saturation and value.
    for channel, bins, limits in (
        (0, 16, [0, 180]),
        (1, 16, [0, 256]),
        (2, 16, [0, 256]),
    ):
        hist = cv2.calcHist([hsv], [channel], None, [bins], limits)
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist.tolist())

    intensity_mean = float(np.mean(gray)) / 255.0
    intensity_std = float(np.std(gray)) / 255.0
    texture_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Scale the variance to a less extreme numeric range.
    texture_variance = texture_variance / (texture_variance + 1000.0)

    return np.asarray(
        hist_features + [intensity_mean, intensity_std, texture_variance],
        dtype=np.float32,
    )


def list_images(class_dir: Path):
    """Yield supported image paths from a class directory."""
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
