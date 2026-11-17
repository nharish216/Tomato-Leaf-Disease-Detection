# Tomato Leaf Disease Detection Using K-Nearest Neighbours (KNN)

A Python machine-learning project for classifying tomato leaf images as healthy or diseased, and distinguishing disease classes represented in the training dataset.

The project description in my resume states that it uses image-processing techniques to extract visual information such as **color, texture, and leaf patterns**, then applies the **K-Nearest Neighbours (KNN)** algorithm for classification.

## Technology stack

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Image processing and machine learning

## Workflow

1. Collect tomato leaf images and arrange them into class-labelled folders.
2. Load and preprocess the images using OpenCV.
3. Extract image features:
   - HSV color histogram
   - Grayscale intensity statistics
   - Laplacian variance as a simple texture/detail measure
4. Split the data into training and test sets.
5. Standardize feature values and train a KNN classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Predict the class of a new tomato leaf image.

> **Important:** This repository contains a reproducible starter implementation based on the project summary in my resume. The resume does not specify the original dataset name, exact disease categories, K value, measured accuracy, or final experimental results. Those details are intentionally not invented. Run the project with your actual dataset and update this README with your real results.

## Repository structure

```text
Tomato-Leaf-Disease-Detection-KNN/
├── data/
│   └── dataset/
│       └── ADD_CLASS_FOLDERS_HERE/
├── docs/
│   └── PROJECT_DETAILS.md
├── models/
├── notebooks/
│   └── project_workflow.ipynb
├── results/
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset preparation

Place your own dataset in `data/dataset/`, with one subfolder per class. The folder name becomes the class label.

Example structure (illustrative only; use the classes available in your dataset):

```text
data/dataset/
├── Healthy/
│   ├── leaf_001.jpg
│   └── leaf_002.jpg
├── Disease_Class_1/
│   ├── leaf_001.jpg
│   └── leaf_002.jpg
└── Disease_Class_2/
    ├── leaf_001.jpg
    └── leaf_002.jpg
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

Do not upload a dataset unless you have permission to redistribute it. Check the dataset's license and cite its source.

## Setup

Python 3.9 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate

From the repository root:

```bash
python src/train.py --data_dir data/dataset --k 5
```

The script prints evaluation metrics and saves the trained pipeline and class names under `models/`. It also saves a confusion-matrix image under `results/`.

## Predict one image

After training:

```bash
python src/predict.py --image path/to/tomato_leaf.jpg
```

The predicted class is based on the labels present in the dataset used to train the model.

## Jupyter Notebook

Open `notebooks/project_workflow.ipynb` in Jupyter Notebook or VS Code to review the workflow and run the training script from a notebook.

## Limitations

- KNN predictions depend on the quality, balance, and variety of the labelled images.
- The simple hand-crafted features in this starter implementation may not capture every symptom or visual variation.
- Lighting, camera quality, background, leaf age, and image angle can affect predictions.
- This is an academic machine-learning project, not a substitute for professional agricultural diagnosis.

## Project information

This repository is based on the project summary in my resume: tomato leaf disease detection using KNN, image processing, feature extraction, preprocessing, model training, and evaluation. Replace the starter dataset and results with the exact materials and measured results from your original project before making claims about performance.
