"""Minimal background trainer for cached classifier smoke jobs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.train_classifier_job import run_job


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    ready_dir = args.queue_root / "ready"
    claimed_dir = args.queue_root / "claimed"
    done_dir = args.queue_root / "done"
    failed_dir = args.queue_root / "failed"
    for d in (ready_dir, claimed_dir, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    while True:
        ready_jobs = sorted(ready_dir.glob("*.ready.json"))
        if not ready_jobs:
            if args.once:
                return
            time.sleep(args.poll_seconds)
            continue

        for ready_path in ready_jobs:
            claimed_path = claimed_dir / ready_path.name.replace(".ready.json", ".claimed.json")
            try:
                ready_path.rename(claimed_path)
            except FileNotFoundError:
                continue

            try:
                job = json.loads(claimed_path.read_text(encoding="utf-8"))
                metrics = run_job(job)
                done_payload = {
                    "run_id": job["run_id"],
                    "claimed_job": str(claimed_path),
                    "metrics_path": str(Path(str(job["result_dir"])) / "metrics.json"),
                    "auc": metrics["auc"],
                    "eer": metrics["eer"],
                    "accuracy": metrics["accuracy"],
                }
                _write_json(done_dir / ready_path.name.replace(".ready.json", ".done.json"), done_payload)
            except Exception as exc:  # noqa: BLE001
                failed_payload = {
                    "run_id": job.get("run_id", ready_path.stem),
                    "claimed_job": str(claimed_path),
                    "error": repr(exc),
                }
                _write_json(failed_dir / ready_path.name.replace(".ready.json", ".failed.json"), failed_payload)
            finally:
                claimed_path.unlink(missing_ok=True)

        if args.once:
            return


if __name__ == "__main__":
    main()
