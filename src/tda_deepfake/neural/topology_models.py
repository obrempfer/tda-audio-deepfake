"""Small models for topology-only experiments."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureBlock:
    """One named contiguous feature block inside a concatenated vector."""

    name: str
    start: int
    end: int

    @property
    def dim(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class StageSpec:
    """One training stage for robust-core-first learning."""

    name: str
    active_blocks: tuple[str, ...]
    max_epochs: int
    learning_rate: float


class FeatureLayout:
    """Named slice metadata for concatenated topology feature blocks."""

    def __init__(self, blocks: list[FeatureBlock]) -> None:
        if not blocks:
            raise ValueError("FeatureLayout requires at least one block")
        self.blocks = blocks
        self._index = {block.name: block for block in blocks}

    @property
    def feature_dim(self) -> int:
        return self.blocks[-1].end

    @property
    def block_names(self) -> list[str]:
        return [block.name for block in self.blocks]

    def get_block(self, name: str) -> FeatureBlock:
        if name not in self._index:
            raise KeyError(f"Unknown feature block: {name}")
        return self._index[name]

    def mask_for_blocks(self, active_blocks: tuple[str, ...] | list[str]) -> npt.NDArray[np.float64]:
        mask = np.zeros(self.feature_dim, dtype=np.float64)
        for name in active_blocks:
            block = self.get_block(name)
            mask[block.start:block.end] = 1.0
        return mask

    def mask_without_blocks(self, removed_blocks: tuple[str, ...] | list[str]) -> npt.NDArray[np.float64]:
        mask = np.ones(self.feature_dim, dtype=np.float64)
        for name in removed_blocks:
            block = self.get_block(name)
            mask[block.start:block.end] = 0.0
        return mask

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_dim": self.feature_dim,
            "blocks": [asdict(block) for block in self.blocks],
        }


def stack_feature_blocks(
    block_matrices: dict[str, npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], FeatureLayout]:
    """Concatenate named block matrices column-wise and return the slice layout."""
    if not block_matrices:
        raise ValueError("Need at least one block matrix")

    names = list(block_matrices)
    first = np.asarray(block_matrices[names[0]], dtype=np.float64)
    if first.ndim != 2:
        raise ValueError(f"Expected 2-D matrix for block {names[0]!r}, got {first.shape}")

    matrices = []
    blocks: list[FeatureBlock] = []
    start = 0
    n_samples = first.shape[0]

    for name in names:
        matrix = np.asarray(block_matrices[name], dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2-D matrix for block {name!r}, got {matrix.shape}")
        if matrix.shape[0] != n_samples:
            raise ValueError(
                f"Block {name!r} has {matrix.shape[0]} samples, expected {n_samples}"
            )
        end = start + matrix.shape[1]
        matrices.append(matrix)
        blocks.append(FeatureBlock(name=name, start=start, end=end))
        start = end

    return np.concatenate(matrices, axis=1), FeatureLayout(blocks)


def binary_metrics(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Compute the main binary metrics for this project."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "eer": _compute_eer(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_eval": int(len(y_true)),
        "n_bonafide": int(np.sum(y_true == 0)),
        "n_spoof": int(np.sum(y_true == 1)),
    }


def evaluate_block_ablations(
    model: "MaskedTopologyModel",
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    layout: FeatureLayout,
) -> dict[str, dict[str, object]]:
    """Evaluate the trained model after zeroing one explicit block at a time."""
    base = model.evaluate(X, y)
    out: dict[str, dict[str, object]] = {}
    for block_name in layout.block_names:
        mask = layout.mask_without_blocks([block_name])
        metrics = model.evaluate(X, y, feature_mask=mask)
        deltas = {
            "auc": float(metrics["auc"] - base["auc"]),
            "eer": float(metrics["eer"] - base["eer"]),
            "accuracy": float(metrics["accuracy"] - base["accuracy"]),
        }
        out[block_name] = {
            "removed_blocks": [block_name],
            "metrics": metrics,
            "deltas": deltas,
        }
    return out


class MaskedTopologyModel:
    """Common prediction/evaluation interface for masked-topology models."""

    scaler: StandardScaler | None

    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
        feature_mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def evaluate(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
        feature_mask: npt.NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        y_score = self.predict_proba(X, feature_mask=feature_mask)
        return binary_metrics(y, y_score)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "MaskedTopologyModel":
        return joblib.load(path)


class TopologyLinearBaseline(MaskedTopologyModel):
    """Logistic-regression anchor on explicit topology blocks."""

    def __init__(
        self,
        c: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
    ) -> "TopologyLinearBaseline":
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        self.model = LogisticRegression(
            C=self.c,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, np.asarray(y, dtype=int))
        return self

    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
        feature_mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        if self.scaler is None or self.model is None:
            raise ValueError("Model has not been fit")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float64))
        if feature_mask is not None:
            X_scaled = X_scaled * np.asarray(feature_mask, dtype=np.float64)
        return self.model.predict_proba(X_scaled)[:, 1]


class TopologyMLP(MaskedTopologyModel):
    """Small MLP with optional staged robust-core-first training."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        alpha: float = 1e-4,
        feature_dropout: float = 0.10,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.feature_dropout = feature_dropout
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.model: MLPClassifier | None = None
        self.training_history_: list[dict[str, object]] = []
        self.best_validation_metrics_: dict[str, float] | None = None
        self.fitted_stages_: list[dict[str, object]] = []

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.int_],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.int_],
        layout: FeatureLayout,
        stages: list[StageSpec],
        monitor: str = "eer",
        patience: int = 8,
    ) -> "TopologyMLP":
        if monitor not in {"auc", "eer", "accuracy"}:
            raise ValueError(f"Unsupported monitor: {monitor}")
        if not stages:
            raise ValueError("Need at least one training stage")

        X_train = np.asarray(X_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=int)
        y_val = np.asarray(y_val, dtype=int)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        rng = np.random.default_rng(self.random_state)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=min(self.batch_size, len(X_train_scaled)),
            max_iter=1,
            shuffle=True,
            random_state=self.random_state,
        )

        initialized = False
        classes = np.array([0, 1], dtype=int)
        self.training_history_ = []
        self.fitted_stages_ = []
        best_overall_metrics: dict[str, float] | None = None

        for stage_idx, stage in enumerate(stages):
            stage_mask = layout.mask_for_blocks(stage.active_blocks)
            self.model.learning_rate_init = stage.learning_rate

            best_stage_model = copy.deepcopy(self.model) if initialized else None
            best_stage_metrics = (
                self._evaluate_scaled(X_val_scaled, y_val, feature_mask=stage_mask)
                if initialized
                else None
            )
            stale_epochs = 0

            for epoch in range(1, stage.max_epochs + 1):
                X_epoch = _apply_feature_dropout(
                    X_train_scaled,
                    active_mask=stage_mask,
                    dropout=self.feature_dropout,
                    rng=rng,
                )
                if not initialized:
                    self.model.partial_fit(X_epoch, y_train, classes=classes)
                    initialized = True
                else:
                    self.model.partial_fit(X_epoch, y_train)

                val_metrics = self._evaluate_scaled(X_val_scaled, y_val, feature_mask=stage_mask)
                history_entry = {
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "epoch": epoch,
                    "active_blocks": list(stage.active_blocks),
                    "learning_rate": stage.learning_rate,
                    "monitor": monitor,
                    "val_auc": val_metrics["auc"],
                    "val_eer": val_metrics["eer"],
                    "val_accuracy": val_metrics["accuracy"],
                    "loss": float(getattr(self.model, "loss_", np.nan)),
                }
                self.training_history_.append(history_entry)

                if best_stage_metrics is None or _is_improvement(
                    val_metrics[monitor],
                    best_stage_metrics[monitor],
                    monitor=monitor,
                ):
                    best_stage_model = copy.deepcopy(self.model)
                    best_stage_metrics = val_metrics
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if stale_epochs >= patience:
                    break

            if best_stage_model is not None:
                self.model = best_stage_model
            if best_stage_metrics is None:
                best_stage_metrics = self._evaluate_scaled(X_val_scaled, y_val, feature_mask=stage_mask)

            self.fitted_stages_.append(
                {
                    "name": stage.name,
                    "active_blocks": list(stage.active_blocks),
                    "max_epochs": stage.max_epochs,
                    "learning_rate": stage.learning_rate,
                    "best_val_auc": best_stage_metrics["auc"],
                    "best_val_eer": best_stage_metrics["eer"],
                    "best_val_accuracy": best_stage_metrics["accuracy"],
                }
            )
            best_overall_metrics = best_stage_metrics

        self.best_validation_metrics_ = best_overall_metrics
        return self

    def predict_proba(
        self,
        X: npt.NDArray[np.float64],
        feature_mask: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        if self.scaler is None or self.model is None:
            raise ValueError("Model has not been fit")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float64))
        if feature_mask is not None:
            X_scaled = X_scaled * np.asarray(feature_mask, dtype=np.float64)
        return self.model.predict_proba(X_scaled)[:, 1]

    def _evaluate_scaled(
        self,
        X_scaled: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
        feature_mask: npt.NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        if self.model is None:
            raise ValueError("Model has not been fit")
        X_eval = np.asarray(X_scaled, dtype=np.float64)
        if feature_mask is not None:
            X_eval = X_eval * np.asarray(feature_mask, dtype=np.float64)
        y_score = self.model.predict_proba(X_eval)[:, 1]
        return binary_metrics(y, y_score)


def _apply_feature_dropout(
    X: npt.NDArray[np.float64],
    active_mask: npt.NDArray[np.float64],
    dropout: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Apply input-feature dropout only on currently active blocks."""
    masked = np.asarray(X, dtype=np.float64) * np.asarray(active_mask, dtype=np.float64)
    if dropout <= 0.0:
        return masked
    if dropout >= 1.0:
        raise ValueError("feature_dropout must be < 1.0")

    keep_prob = 1.0 - dropout
    keep = rng.binomial(1, keep_prob, size=masked.shape).astype(np.float64)
    keep *= np.asarray(active_mask, dtype=np.float64)
    return masked * keep / keep_prob


def _compute_eer(y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64]) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _is_improvement(current: float, best: float, *, monitor: str) -> bool:
    if monitor == "eer":
        return current < best
    return current > best
