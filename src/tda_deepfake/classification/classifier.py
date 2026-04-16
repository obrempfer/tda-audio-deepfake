"""Classifier wrapper for TDA deepfake detection.

Wraps scikit-learn SVM and logistic regression with a consistent interface
for training, cross-validation, and inference.
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Literal, Optional, Union

import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

from ..config import ClassifierConfig


class Classifier:
    """Sklearn-based binary classifier for real/fake speech.

    Wraps an SVM or logistic regression inside a StandardScaler pipeline.
    Intentionally simple to isolate the topological feature contribution.

    Args:
        model: Classifier type ('svm' or 'logistic').
        svm_kernel: SVM kernel (only used when model='svm').
        svm_c: SVM regularization strength (only used when model='svm').
        random_state: Random seed.
    """

    def __init__(
        self,
        model: Literal["svm", "logistic"] = ClassifierConfig.MODEL,
        svm_kernel: str = ClassifierConfig.SVM_KERNEL,
        svm_c: float = ClassifierConfig.SVM_C,
        random_state: int = ClassifierConfig.RANDOM_STATE,
    ) -> None:
        self.model_type = model
        self.random_state = random_state

        if model == "svm":
            clf = SVC(
                kernel=svm_kernel,
                C=svm_c,
                probability=True,
                random_state=random_state,
            )
        elif model == "logistic":
            clf = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {model!r}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "Classifier":
        """Fit the classifier on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary label array (0 = real, 1 = fake).

        Returns:
            self
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict labels for samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels array of shape (n_samples,).
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples, 2). Column 1 = P(fake).
        """
        return self.pipeline.predict_proba(X)

    def cross_validate(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        n_folds: int = ClassifierConfig.CV_FOLDS,
    ) -> dict:
        """Run stratified k-fold cross-validation and return metrics.

        Args:
            X: Feature matrix.
            y: Labels.
            n_folds: Number of CV folds.

        Returns:
            Dict with keys 'accuracy', 'auc' (mean ± std across folds).
        """
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        acc_scores = []
        auc_scores = []
        eer_scores = []

        for train_idx, test_idx in cv.split(X, y):
            pipeline = clone(self.pipeline)
            pipeline.fit(X[train_idx], y[train_idx])
            y_pred = pipeline.predict(X[test_idx])
            y_proba = pipeline.predict_proba(X[test_idx])[:, 1]

            acc_scores.append(accuracy_score(y[test_idx], y_pred))
            auc_scores.append(roc_auc_score(y[test_idx], y_proba))
            eer_scores.append(_compute_eer(y[test_idx], y_proba))

        acc = np.array(acc_scores, dtype=np.float64)
        auc = np.array(auc_scores, dtype=np.float64)
        eer = np.array(eer_scores, dtype=np.float64)
        return {
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc)),
            "auc_mean": float(np.mean(auc)),
            "auc_std": float(np.std(auc)),
            "eer_mean": float(np.mean(eer)),
            "eer_std": float(np.std(eer)),
        }

    def evaluate(self, X: npt.NDArray, y: npt.NDArray) -> dict:
        """Evaluate on held-out test data and return a metrics dict.

        Args:
            X: Test feature matrix.
            y: True test labels.

        Returns:
            Dict with 'report' (sklearn classification report string) and 'auc'.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        return {
            "report": classification_report(y, y_pred, target_names=["real", "fake"]),
            "auc": roc_auc_score(y, y_proba),
            "eer": _compute_eer(y, y_proba),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the fitted pipeline to disk.

        Args:
            path: File path for the saved model (e.g. 'model.pkl').
        """
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Classifier":
        """Load a previously saved pipeline from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            Classifier instance with the loaded pipeline.
        """
        obj = cls.__new__(cls)
        obj.pipeline = joblib.load(path)
        return obj


def _compute_eer(y_true: npt.NDArray, y_score: npt.NDArray) -> float:
    """Compute equal error rate from binary labels and positive-class scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)
