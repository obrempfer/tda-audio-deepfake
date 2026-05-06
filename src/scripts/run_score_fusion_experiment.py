"""Run score-level late fusion across paired result directories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def _compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _wait_for_scores(result_dir: Path, timeout_seconds: int) -> None:
    waited = 0
    while not (result_dir / "eval_scores.npy").exists():
        time.sleep(10)
        waited += 10
        if timeout_seconds > 0 and waited >= timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {result_dir / 'eval_scores.npy'}")


def _load_scores(result_dir: Path, timeout_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    _wait_for_scores(result_dir, timeout_seconds)
    scores = np.load(result_dir / "eval_scores.npy", allow_pickle=False)
    labels = np.load(result_dir / "eval_labels.npy", allow_pickle=False)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.0).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "eer": float(_compute_eer(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _weighted_average(a: np.ndarray, b: np.ndarray, weight_a: float) -> np.ndarray:
    return weight_a * a + (1.0 - weight_a) * b


def _fit_meta(cal_a: np.ndarray, cal_b: np.ndarray, cal_y: np.ndarray) -> LogisticRegression:
    X = np.column_stack([cal_a, cal_b])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, cal_y)
    return clf


def _parse_target(value: str) -> tuple[str, Path, Path]:
    name, cubical_dir, morse_dir = value.split("::", 2)
    return name, Path(cubical_dir), Path(morse_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=str, required=True,
                        help="name::cubical_dir::morse_dir")
    parser.add_argument("--target", action="append", required=True,
                        help="name::cubical_dir::morse_dir")
    parser.add_argument("--weight-a", type=float, nargs="*", default=[0.5, 0.7, 0.3])
    parser.add_argument("--timeout-seconds", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cal_name, cal_c_dir, cal_m_dir = _parse_target(args.calibration)
    cal_c_scores, cal_y = _load_scores(cal_c_dir, args.timeout_seconds)
    cal_m_scores, cal_y_2 = _load_scores(cal_m_dir, args.timeout_seconds)
    if not np.array_equal(cal_y, cal_y_2):
        raise ValueError(f"Calibration labels do not match for {cal_name}")

    meta = _fit_meta(cal_c_scores, cal_m_scores, cal_y)

    summary: dict[str, object] = {
        "calibration": {
            "name": cal_name,
            "cubical_dir": str(cal_c_dir),
            "morse_dir": str(cal_m_dir),
        },
        "targets": {},
    }

    for raw_target in args.target:
        name, cubical_dir, morse_dir = _parse_target(raw_target)
        cubical_scores, y_true = _load_scores(cubical_dir, args.timeout_seconds)
        morse_scores, y_true_2 = _load_scores(morse_dir, args.timeout_seconds)
        if not np.array_equal(y_true, y_true_2):
            raise ValueError(f"Target labels do not match for {name}")

        target_payload: dict[str, object] = {
            "cubical_dir": str(cubical_dir),
            "morse_dir": str(morse_dir),
            "cubical": _metrics(y_true, cubical_scores),
            "morse": _metrics(y_true, morse_scores),
            "simple_average": _metrics(y_true, 0.5 * (cubical_scores + morse_scores)),
            "weighted_average": {},
        }

        for weight_a in args.weight_a:
            fused = _weighted_average(cubical_scores, morse_scores, weight_a)
            target_payload["weighted_average"][str(weight_a)] = _metrics(y_true, fused)

        meta_scores = meta.decision_function(np.column_stack([cubical_scores, morse_scores]))
        target_payload["logistic_meta"] = _metrics(y_true, np.asarray(meta_scores, dtype=np.float64))
        summary["targets"][name] = target_payload

    (args.out_dir / "fusion_results.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
