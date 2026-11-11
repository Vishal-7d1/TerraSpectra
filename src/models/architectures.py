"""
Model Architectures for Hyperspectral Classification & Disease Forecasting.
Includes classical ML pipelines (RF, SVM, XGBoost) and deep 1D-CNN spectral models.
"""

from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def build_ml_pipeline(
    model_name: str = "rf",
    use_pca: bool = True,
    n_components: int = 15,
    random_state: int = 42
) -> Pipeline:
    """
    Builds a scikit-learn machine learning pipeline for spectral classification.

    Parameters:
        model_name (str): 'rf' (Random Forest), 'svm' (SVM), or 'xgb' (Gradient Boosting).
        use_pca (bool): Whether to perform dimensionality reduction on contiguous spectral bands.
        n_components (int): Number of PCA components if use_pca is True.
        random_state (int): Random seed for reproducibility.

    Returns:
        Pipeline: Scikit-learn Pipeline instance.
    """
    steps = [("scaler", StandardScaler())]

    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))

    if model_name.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_state, n_jobs=-1)
    elif model_name.lower() == "svm":
        clf = SVC(kernel="rbf", C=10.0, probability=True, random_state=random_state)
    elif model_name.lower() in ("xgb", "gb"):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    steps.append(("classifier", clf))
    return Pipeline(steps)

