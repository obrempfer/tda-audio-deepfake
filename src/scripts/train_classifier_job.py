"""Train and evaluate one cached classifier job from prebuilt X/y matrices."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def _compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _positive_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    raise AttributeError("Model exposes neither predict_proba nor decision_function")


def _make_estimator(
    classifier: str,
    *,
    c_value: float,
    gamma: str | float,
    probability: bool,
    cache_size: float,
    max_iter: int,
    random_state: int,
) -> object:
    if classifier == "sklearn_svc_rbf":
        return SVC(
            kernel="rbf",
            C=c_value,
            gamma=gamma,
            probability=probability,
            cache_size=cache_size,
            random_state=random_state,
        )
    if classifier == "sklearn_logistic":
        return LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )
    if classifier == "sklearn_linear_svm":
        return LinearSVC(
            C=c_value,
            random_state=random_state,
            max_iter=max_iter,
            dual="auto",
        )
    if classifier == "thundersvm_svc_rbf":
        from thundersvm import SVC as ThunderSVC  # type: ignore

        return ThunderSVC(
            kernel="rbf",
            C=c_value,
            gamma=gamma,
            probability=probability,
        )
    raise ValueError(f"Unknown classifier {classifier!r}")


def run_job(job: dict[str, object]) -> dict[str, object]:
    feature_dir = Path(str(job["feature_dir"]))
    result_dir = Path(str(job["result_dir"]))
    result_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(feature_dir / "X_train.npy", allow_pickle=False)
    y_train = np.load(feature_dir / "y_train.npy", allow_pickle=False)
    X_eval = np.load(feature_dir / "X_eval.npy", allow_pickle=False)
    y_eval = np.load(feature_dir / "y_eval.npy", allow_pickle=False)

    params = dict(job.get("params", {}))
    classifier = str(job["classifier"])

    estimator = _make_estimator(
        classifier,
        c_value=float(params.get("C", 1.0)),
        gamma=params.get("gamma", "scale"),
        probability=bool(params.get("probability", False)),
        cache_size=float(params.get("cache_size", 8000)),
        max_iter=int(params.get("max_iter", 1000)),
        random_state=int(params.get("random_state", 42)),
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_eval)
    y_score = _positive_scores(pipeline, X_eval)
    eval_time = time.perf_counter() - t1

    metrics = {
        "run_id": job["run_id"],
        "feature_dir": str(feature_dir),
        "classifier": classifier,
        "params": params,
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_eval)),
        "n_features": int(X_train.shape[1]),
        "fit_time_seconds": float(fit_time),
        "eval_time_seconds": float(eval_time),
        "total_time_seconds": float(fit_time + eval_time),
        "auc": float(roc_auc_score(y_eval, y_score)),
        "eer": float(_compute_eer(y_eval, y_score)),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "n_support_vectors": None,
    }

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "n_support_"):
        metrics["n_support_vectors"] = int(np.sum(clf.n_support_))

    report = classification_report(y_eval, y_pred, target_names=["real", "fake"])

    np.save(result_dir / "scores.npy", y_score)
    np.save(result_dir / "eval_scores.npy", y_score)
    np.save(result_dir / "eval_labels.npy", y_eval)
    joblib.dump(pipeline, result_dir / "model.joblib")
    joblib.dump(pipeline, result_dir / "model.pkl")
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (result_dir / "report.txt").write_text(report, encoding="utf-8")
    (result_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (result_dir / "eval_report.txt").write_text(report, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    metrics = run_job(job)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
