"""
Training & Evaluation Pipeline for TerraSpectra.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits spectral features and target labels into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_classifier(pipeline: Any, X_train: np.ndarray, y_train: np.ndarray, save_path: str = None) -> Any:
    """
    Fits the classification pipeline and optionally saves it to disk.
    """
    print("[INFO] Training classification model...")
    pipeline.fit(X_train, y_train)

    if save_path:
        joblib.dump(pipeline, save_path)
        print(f"[INFO] Model successfully saved to: {save_path}")

    return pipeline


def evaluate_classifier(pipeline: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates model performance and returns metrics.
    """
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    report = classification_report(y_test, preds, output_dict=True)
    matrix = confusion_matrix(y_test, preds)

    print(f"[RESULTS] Test Accuracy: {acc * 100:.2f}%")
    print(f"[RESULTS] Weighted F1-Score: {f1:.4f}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": matrix
    }

