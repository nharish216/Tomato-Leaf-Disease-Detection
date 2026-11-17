# Project Details

## Objective

Develop a machine-learning system that analyzes tomato leaf images and classifies them into the categories represented in a labelled dataset. The intended use is to support early identification of visible leaf disease symptoms.

## Method described in the resume

The resume describes the following work:

- Image processing of tomato leaf images.
- Extraction of visual features such as color, texture, and leaf patterns.
- Use of the K-Nearest Neighbours (KNN) algorithm to classify healthy and diseased leaves.
- Data preprocessing, feature extraction, model training, and evaluation.

## Starter implementation in this repository

To make the repository runnable, this starter implementation uses:

- HSV color histograms to represent color distribution.
- Grayscale mean and standard deviation as basic intensity descriptors.
- Laplacian variance as a simple image-detail/texture descriptor.
- `StandardScaler` followed by `KNeighborsClassifier` in a Scikit-learn pipeline.
- A stratified train/test split and classification metrics.

These are documented implementation choices for this GitHub starter project. They should not be represented as the exact feature extraction settings from the original academic implementation unless they match the original code.

## Information to add from the original project

- Dataset name, source, and license.
- Exact disease categories.
- Image count per category.
- Exact preprocessing and feature extraction method.
- K value and distance metric used.
- Measured evaluation results.
- Screenshots or sample predictions from the actual project.
