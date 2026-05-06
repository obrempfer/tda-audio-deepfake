import json
from pathlib import Path

import numpy as np

from scripts.train_classifier_job import run_job
from scripts.train_queue_worker import main as queue_main


def _write_bundle(feature_dir: Path) -> None:
    rng = np.random.default_rng(0)
    X_train = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.5, size=(8, 6)),
            rng.normal(loc=1.0, scale=0.5, size=(8, 6)),
        ]
    )
    y_train = np.array([0] * 8 + [1] * 8)
    X_eval = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.5, size=(4, 6)),
            rng.normal(loc=1.0, scale=0.5, size=(4, 6)),
        ]
    )
    y_eval = np.array([0] * 4 + [1] * 4)

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "X_train.npy", X_train)
    np.save(feature_dir / "y_train.npy", y_train)
    np.save(feature_dir / "X_eval.npy", X_eval)
    np.save(feature_dir / "y_eval.npy", y_eval)


def test_run_job_writes_normal_outputs(tmp_path: Path) -> None:
    feature_dir = tmp_path / "bundle"
    result_dir = tmp_path / "result"
    _write_bundle(feature_dir)

    metrics = run_job(
        {
            "run_id": "demo",
            "feature_dir": str(feature_dir),
            "result_dir": str(result_dir),
            "classifier": "sklearn_svc_rbf",
            "params": {
                "C": 1.0,
                "gamma": "scale",
                "probability": False,
                "cache_size": 8000,
                "random_state": 42,
            },
        }
    )

    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0
    assert (result_dir / "model.pkl").exists()
    assert (result_dir / "eval_results.json").exists()
    assert (result_dir / "eval_report.txt").exists()


def test_queue_worker_claims_and_completes_job(tmp_path: Path, monkeypatch) -> None:
    feature_dir = tmp_path / "bundle"
    result_dir = tmp_path / "result"
    queue_root = tmp_path / "queue"
    _write_bundle(feature_dir)
    ready_dir = queue_root / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    job_path = ready_dir / "demo.ready.json"
    job_path.write_text(
        json.dumps(
            {
                "run_id": "demo",
                "feature_dir": str(feature_dir),
                "result_dir": str(result_dir),
                "classifier": "sklearn_svc_rbf",
                "params": {
                    "C": 1.0,
                    "gamma": "scale",
                    "probability": False,
                    "cache_size": 8000,
                    "random_state": 42,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_queue_worker.py",
            "--queue-root",
            str(queue_root),
            "--once",
        ],
    )
    queue_main()

    assert not job_path.exists()
    done_dir = queue_root / "done"
    done_files = list(done_dir.glob("*.done.json"))
    assert len(done_files) == 1
    payload = json.loads(done_files[0].read_text(encoding="utf-8"))
    assert payload["run_id"] == "demo"
    assert (result_dir / "model.pkl").exists()
