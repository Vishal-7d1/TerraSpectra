# 05. Machine Learning

## Purpose
This folder manages the development, training, validation, hyperparameter tuning, and comprehensive evaluation of machine learning models for disease classification.

## Contents
- `train.py`: Trains and benchmarks supervised machine learning models (e.g., Random Forest, Support Vector Machines, Gradient Boosting) using stratified cross-validation.
- `evaluate.py`: Evaluates model performance using classification metrics (Accuracy, Precision, Recall, Weighted F1-Score, Cohen's Kappa), confusion matrices, and ROC-AUC curves.
- `models/`: Directory designated for storing trained, serialized model weights and preprocessor pipelines (e.g., `.joblib`, `.pkl`).
- `outputs/`: Directory for storing model evaluation figures, confusion matrix heatmaps, ROC curves, and performance metric tables.

