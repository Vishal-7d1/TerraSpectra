"""
Inference module for predicting crop disease status and severity from spectral signatures.
"""

from typing import Union, Dict, Any
import numpy as np
import joblib


def load_model(model_path: str):
    """Loads a pre-trained pipeline or model from disk."""
    return joblib.load(model_path)


def predict_disease_stage(model: Any, spectral_vector: np.ndarray) -> Dict[str, Any]:
    """
    Predicts disease category and class probabilities for a given spectral sample.

    Parameters:
        model: Trained scikit-learn model or pipeline.
        spectral_vector (np.ndarray): Shape (1, n_features) or (n_features,).

    Returns:
        Dict[str, Any]: Prediction label and probability distribution.
    """
    if spectral_vector.ndim == 1:
        spectral_vector = spectral_vector.reshape(1, -1)

    predicted_class = model.predict(spectral_vector)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(spectral_vector)[0]
        classes = model.classes_
        probabilities = {str(c): float(p) for c, p in zip(classes, probs)}

    return {
        "prediction": predicted_class,
        "probabilities": probabilities
    }

