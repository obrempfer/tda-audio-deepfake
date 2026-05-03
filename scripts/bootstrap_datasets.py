#!/usr/bin/env python3
"""Bootstrap dataset storage for tda-audio-deepfake.

By default, datasets live directly inside the repo under ``data/``.
If ``--storage-root`` points elsewhere, this script creates a mirrored
``data/`` layout there and rewires the repo paths as symlinks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
MANAGED_DATA_DIRS = ("raw", "results", "protocols")
ASVSPOOF2019_LA_URL = "https://zenodo.org/records/6906306/files/LA.zip?download=1"
ASVSPOOF2021_LA_URL = "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
ASVSPOOF2021_LA_KEYS_URL = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
ASVSPOOF2021_DF_KEYS_URL = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
ASVSPOOF2021_DF_EVAL_PART00_URL = "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1"


def _try_import_hf() -> object | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    return snapshot_download


def _ensure_hf_runtime() -> object:
    snapshot_download = _try_import_hf()
    if snapshot_download is not None:
        return snapshot_download

    fallback = Path.home() / "Git" / "obrempfer" / "tda-audio-deepfake" / ".venv_lab" / "bin" / "python"
    if not fallback.exists():
        raise SystemExit(
            "huggingface_hub is unavailable and fallback runtime is missing: "
            f"{fallback}"
        )

    os.execv(str(fallback), [str(fallback), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise AssertionError("unreachable")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _download(url: str, dest: Path) -> None:
    _ensure_parent(dest)
    _run(["wget", "-c", "-O", str(dest), url])


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link(path: Path, target: Path) -> None:
    _ensure_parent(path)
    if path.exists() or path.is_symlink():
        current = path.resolve() if path.is_symlink() else None
        if path.is_symlink() and current == target.resolve():
            return
        if not path.is_symlink():
            raise RuntimeError(
                f"Refusing to replace existing real path with symlink: {path}. "
                "Complete the initial migration first."
            )
        _safe_unlink(path)
    path.symlink_to(target)


def _count_audio_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"})


def _data_root(project_root: Path) -> Path:
    return project_root / "data"


def _legacy_user_tmp_root() -> Path:
    return Path("/tmp") / os.environ.get("USER", "obrempfer")


def _dir_has_only_placeholders(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        return False
    allowed = {".gitkeep", ".gitignore"}
    for item in path.iterdir():
        if item.name not in allowed:
            return False
    return True


def _detect_asvspoof_source(repo_root: Path) -> Path | None:
    data_dir = repo_root / "data"
    direct_raw = data_dir / "raw"
    if direct_raw.exists() and not direct_raw.is_symlink():
        return direct_raw

    backups = sorted(data_dir.glob("raw_home_backup_*"))
    if backups:
        return backups[-1]
    return None


def _sync_dir(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        _run(["rsync", "-a", f"{source}/", f"{target}/"])
        return
    _copy_tree_contents(source, target, force=False)


def _ensure_repo_local_dir(repo_path: Path, migrate_existing: bool) -> None:
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if repo_path.is_symlink():
        source = repo_path.resolve()
        tmp = repo_path.parent / f".{repo_path.name}.bootstrap_tmp"
        if tmp.exists() or tmp.is_symlink():
            _safe_unlink(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        if migrate_existing and source.exists():
            _sync_dir(source, tmp)
        repo_path.unlink()
        tmp.replace(repo_path)
        return
    repo_path.mkdir(parents=True, exist_ok=True)


def _ensure_repo_symlink(repo_path: Path, target: Path, migrate_existing: bool) -> None:
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)

    if repo_path.is_symlink():
        if repo_path.resolve() == target.resolve():
            return
        if migrate_existing and repo_path.resolve().exists():
            _sync_dir(repo_path.resolve(), target)
        repo_path.unlink()
        repo_path.symlink_to(target)
        return

    if repo_path.exists():
        if migrate_existing:
            _sync_dir(repo_path, target)
        if _dir_has_only_placeholders(repo_path) or migrate_existing:
            shutil.rmtree(repo_path)
        else:
            raise RuntimeError(
                f"Refusing to replace populated repo directory without migration: {repo_path}. "
                "Re-run with --migrate-existing or choose the in-repo default storage."
            )

    repo_path.symlink_to(target)


def ensure_repo_storage_layout(repo_root: Path, storage_root: Path, migrate_existing: bool) -> Path:
    repo_root = repo_root.resolve()
    storage_root = storage_root.resolve()
    repo_data = _data_root(repo_root)
    storage_data = _data_root(storage_root)

    if storage_root == repo_root:
        for name in MANAGED_DATA_DIRS:
            _ensure_repo_local_dir(repo_data / name, migrate_existing=migrate_existing)
        return repo_root

    for name in MANAGED_DATA_DIRS:
        _ensure_repo_symlink(
            repo_data / name,
            storage_data / name,
            migrate_existing=migrate_existing,
        )
    return storage_root


def _copy_tree_contents(source: Path, target: Path, force: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        dest = target / item.name
        if dest.exists() or dest.is_symlink():
            if not force:
                continue
            _safe_unlink(dest)
        if item.is_dir():
            shutil.copytree(item, dest, symlinks=True)
        else:
            shutil.copy2(item, dest)


def restore_asvspoof(storage_root: Path, source_root: Path | None, repo_root: Path, force: bool) -> None:
    target_root = _data_root(storage_root) / "raw"
    target_root.mkdir(parents=True, exist_ok=True)

    if source_root is None or not source_root.exists():
        if source_root is not None and not source_root.exists():
            print(f"ASVspoof source root missing, falling back to public download: {source_root}")
        restore_asvspoof_from_public_sources(storage_root, repo_root=repo_root, force=force)
        return

    candidates = [
        "ASVspoof2019_LA",
        "ASVspoof2021_LA",
        "ASVspoof2021_DF_keys",
        "ASVspoof2021_DF_eval",
        "ASVspoof2021_DF_eval_part00.tar.gz",
        "DF-keys-full.tar.gz",
        "LA-keys-full.tar.gz",
    ]

    restored = 0
    for name in candidates:
        source = source_root / name
        if not source.exists() and name == "LA-keys-full.tar.gz":
            source = source_root / "ASVspoof2021_LA" / "LA-keys-full.tar.gz"
        if not source.exists():
            continue

        target = target_root / name
        if target.exists() or target.is_symlink():
            if not force:
                print(f"ASVspoof target already present, skipping: {target}")
                restored += 1
                continue
            _safe_unlink(target)

        print(f"Restoring ASVspoof asset: {source} -> {target}")
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            _ensure_parent(target)
            shutil.copy2(source, target)
        restored += 1

    if restored == 0:
        print(f"ASVspoof restore found no known assets under {source_root}")

    build_asvspoof2021_df_available_protocol(target_root)


def restore_asvspoof_from_public_sources(storage_root: Path, repo_root: Path, force: bool) -> None:
    target_root = _data_root(storage_root) / "raw"
    download_root = storage_root / "downloads" / "asvspoof"
    target_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    restore_asvspoof2019_la(target_root, download_root, force=force)
    restore_asvspoof2021_la(target_root, download_root, repo_root=repo_root, force=force)
    restore_asvspoof2021_df_keys(target_root, download_root, force=force)
    restore_asvspoof2021_df_eval_part00(target_root, download_root, force=force)


def restore_asvspoof2019_la(target_root: Path, download_root: Path, force: bool) -> None:
    target = target_root / "ASVspoof2019_LA"
    archive = download_root / "ASVspoof2019_LA.zip"
    train_audio = target / "ASVspoof2019_LA_train" / "flac"
    dev_audio = target / "ASVspoof2019_LA_dev" / "flac"
    eval_audio = target / "ASVspoof2019_LA_eval" / "flac"

    if force and target.exists():
        shutil.rmtree(target)

    if not (train_audio.exists() and dev_audio.exists() and eval_audio.exists()):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        _download(ASVSPOOF2019_LA_URL, archive)
        _run(["unzip", "-q", "-o", str(archive), "-d", str(target)])
        archive.unlink(missing_ok=True)
    else:
        print(f"ASVspoof2019_LA already present at {target}; skipping download.")

    ensure_asvspoof2019_protocol_links(target)
    build_asvspoof2019_derived(target, force=force)


def ensure_asvspoof2019_protocol_links(target: Path) -> None:
    cm_protocols = target / "ASVspoof2019_LA_cm_protocols"
    for name in (
        "ASVspoof2019.LA.cm.train.trn.txt",
        "ASVspoof2019.LA.cm.dev.trl.txt",
        "ASVspoof2019.LA.cm.eval.trl.txt",
    ):
        source = cm_protocols / name
        link = target / name
        if not source.exists():
            continue
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            _safe_unlink(link)
        link.symlink_to(source.relative_to(target))


def _read_protocol_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _label_from_protocol_line(line: str) -> str:
    return line.split()[-1].lower()


def _balanced_sample(lines: list[str], label: str, count: int, seed: int) -> list[str]:
    items = [line for line in lines if _label_from_protocol_line(line) == label]
    if count >= len(items):
        return sorted(items)
    rng = random.Random(f"{seed}:{label}:{len(items)}")
    picked = list(items)
    rng.shuffle(picked)
    return sorted(picked[:count])


def build_asvspoof2019_derived(target: Path, force: bool) -> None:
    derived_dir = target / "derived"
    train_out = derived_dir / "ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt"
    dev_out = derived_dir / "ASVspoof2019.LA.cm.dev.balanced_5000.seed42.txt"

    if not force and train_out.exists() and dev_out.exists():
        print(f"ASVspoof2019 derived files already present at {derived_dir}; skipping rebuild.")
        return

    derived_dir.mkdir(parents=True, exist_ok=True)

    train_source = target / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_source = target / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"

    train_lines = _read_protocol_lines(train_source)
    train_bonafide = sorted(line for line in train_lines if _label_from_protocol_line(line) == "bonafide")
    train_spoof = _balanced_sample(train_lines, "spoof", len(train_bonafide), seed=42)
    train_rows = train_bonafide + train_spoof
    random.Random("42:train_mix").shuffle(train_rows)
    train_out.write_text("\n".join(train_rows) + "\n", encoding="utf-8")

    dev_lines = _read_protocol_lines(dev_source)
    dev_bonafide = _balanced_sample(dev_lines, "bonafide", 2500, seed=42)
    dev_spoof = _balanced_sample(dev_lines, "spoof", 2500, seed=42)
    dev_rows = dev_bonafide + dev_spoof
    random.Random("42:dev_mix").shuffle(dev_rows)
    dev_out.write_text("\n".join(dev_rows) + "\n", encoding="utf-8")
    print(f"Built ASVspoof2019 derived files in {derived_dir}")


def restore_asvspoof2021_la(target_root: Path, download_root: Path, repo_root: Path, force: bool) -> None:
    target = target_root / "ASVspoof2021_LA"
    archive = download_root / "ASVspoof2021_LA_eval.tar.gz"
    keys_archive = target / "LA-keys-full.tar.gz"
    eval_audio = target / "ASVspoof2021_LA_eval" / "flac"
    trial_metadata = target / "keys" / "LA" / "CM" / "trial_metadata.txt"

    if force and target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)

    if not eval_audio.exists():
        _download(ASVSPOOF2021_LA_URL, archive)
        _run(["tar", "-xzf", str(archive), "-C", str(target)])
        archive.unlink(missing_ok=True)
    else:
        print(f"ASVspoof2021_LA eval audio already present at {target}; skipping eval download.")

    if force and keys_archive.exists():
        keys_archive.unlink()
    if not trial_metadata.exists() or not keys_archive.exists():
        _download(ASVSPOOF2021_LA_KEYS_URL, keys_archive)
        _run(["tar", "-xzf", str(keys_archive), "-C", str(target)])
    else:
        print(f"ASVspoof2021_LA keys already present at {target}; skipping keys download.")

    build_asvspoof2021_la_internal_split(target, repo_root=repo_root, force=force)


def build_asvspoof2021_la_internal_split(target: Path, repo_root: Path, force: bool) -> None:
    derived_dir = target / "derived" / "internal_seed42"
    prefix = "asvspoof2021_la_internal_seed42"
    needed = [
        derived_dir / f"{prefix}_train.txt",
        derived_dir / f"{prefix}_dev.txt",
        derived_dir / f"{prefix}_test.txt",
        derived_dir / f"{prefix}_summary.json",
    ]
    if not force and all(path.exists() for path in needed):
        print(f"ASVspoof2021_LA internal split already present at {derived_dir}; skipping rebuild.")
        return

    if not repo_root.exists():
        print(f"Skipping ASVspoof2021_LA internal split build; repo root missing: {repo_root}")
        return

    python_bin = repo_root / ".venv_lab" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    protocol = target / "keys" / "LA" / "CM" / "trial_metadata.txt"
    derived_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    repo_src = str(repo_root / "src")
    env["PYTHONPATH"] = repo_src if not existing_pythonpath else f"{repo_src}:{existing_pythonpath}"
    cmd = [
        str(python_bin),
        str(repo_root / "src" / "scripts" / "build_internal_protocol_split.py"),
        "--protocol",
        str(protocol),
        "--out-dir",
        str(derived_dir),
        "--prefix",
        prefix,
        "--seed",
        "42",
        "--train-ratio",
        "0.6",
        "--dev-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--allowed-partitions",
        "progress,eval",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def restore_asvspoof2021_df_keys(target_root: Path, download_root: Path, force: bool) -> None:
    target = target_root / "ASVspoof2021_DF_keys"
    archive = download_root / "DF-keys-full.tar.gz"
    trial_metadata = target / "keys" / "DF" / "CM" / "trial_metadata.txt"

    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if not trial_metadata.exists():
        _download(ASVSPOOF2021_DF_KEYS_URL, archive)
        _run(["tar", "-xzf", str(archive), "-C", str(target)])
        archive.unlink(missing_ok=True)
    else:
        print(f"ASVspoof2021_DF_keys already present at {target}; skipping download.")


def restore_asvspoof2021_df_eval_part00(target_root: Path, download_root: Path, force: bool) -> None:
    target = target_root / "ASVspoof2021_DF_eval_part00.tar.gz"
    extract_dir = target_root / "ASVspoof2021_DF_eval"
    archive = download_root / target.name

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    if target.exists() and not force:
        print(f"ASVspoof2021_DF_eval_part00 archive already present at {target}; skipping download.")
    else:
        _download(ASVSPOOF2021_DF_EVAL_PART00_URL, archive)
        _ensure_parent(target)
        shutil.copy2(archive, target)

    expected_protocol = extract_dir / "ASVspoof2021.DF.cm.eval.trl.txt"
    expected_audio = extract_dir / "flac"
    if not expected_protocol.exists() or not expected_audio.exists():
        _run(["tar", "-xzf", str(target), "-C", str(target_root)])
    else:
        print(f"ASVspoof2021_DF_eval already extracted at {extract_dir}; skipping extraction.")

    build_asvspoof2021_df_available_protocol(target_root)


def build_asvspoof2021_df_available_protocol(target_root: Path) -> None:
    eval_root = target_root / "ASVspoof2021_DF_eval"
    keys_protocol = target_root / "ASVspoof2021_DF_keys" / "keys" / "DF" / "CM" / "trial_metadata.txt"
    audio_dir = eval_root / "flac"
    if not keys_protocol.exists() or not audio_dir.exists():
        return

    derived_dir = eval_root / "derived"
    protocol_out = derived_dir / "available_trial_metadata.txt"
    summary_out = derived_dir / "available_trial_metadata_summary.json"
    derived_dir.mkdir(parents=True, exist_ok=True)

    available = {path.stem for path in audio_dir.glob("*.flac")}
    rows: list[str] = []
    labels = Counter()
    missing = 0

    with keys_protocol.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            utterance = parts[1] if len(parts) > 1 else parts[0]
            if utterance not in available:
                missing += 1
                continue
            rows.append(line)
            label = _label_from_protocol_line(line)
            if label:
                labels[label] += 1

    protocol_out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_out.write_text(
        json.dumps(
            {
                "count": len(rows),
                "labels": dict(labels),
                "available_audio_files": len(available),
                "missing_protocol_rows": missing,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Built ASVspoof2021 DF available protocol at {protocol_out}")


def restore_mlaad_tiny(snapshot_download: object, storage_root: Path, force: bool) -> None:
    target = _data_root(storage_root) / "raw" / "MLAAD-tiny"
    expected_audio = 15290
    if force and target.exists():
        shutil.rmtree(target)

    existing_audio = _count_audio_files(target) if target.exists() else 0
    if existing_audio == expected_audio and not force:
        print(f"MLAAD-tiny already present at {target} ({existing_audio} audio files); skipping download.")
    else:
        legacy_target = _legacy_user_tmp_root() / "mlaad" / "MLAAD-tiny"
        if legacy_target.exists():
            print(f"Copying legacy MLAAD-tiny into {target} from {legacy_target} ...")
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(legacy_target, target, symlinks=True)
            existing_audio = _count_audio_files(target)

        if existing_audio == expected_audio and not force:
            print(f"MLAAD-tiny restored from legacy location into {target}.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MLAAD-tiny into {target} ...")
            snapshot_download(
                repo_id="mueller91/MLAAD-tiny",
                repo_type="dataset",
                local_dir=str(target),
                max_workers=4,
            )
            print(f"MLAAD-tiny download complete: {target}")

    protocol_root = _data_root(storage_root) / "protocols" / "mlaad_tiny"
    materialize_mlaad_subset(
        src_root=target,
        out_dir=protocol_root / "mlaad_tiny_en_probe_500_restore_materialized",
        languages=["en"],
        max_per_class=500,
        seed=42,
        force=force,
    )
    materialize_mlaad_subset(
        src_root=target,
        out_dir=protocol_root / "mlaad_tiny_de_probe_1000_restore_materialized",
        languages=["de"],
        max_per_class=1000,
        seed=42,
        force=force,
    )


def _sanitize_stem(text: str) -> str:
    cleaned = []
    prev_us = False
    for char in text:
        keep = char.isalnum() or char in "._-"
        if keep:
            cleaned.append(char)
            prev_us = False
        elif not prev_us:
            cleaned.append("_")
            prev_us = True
    return "".join(cleaned).strip("_.") or "sample"


def _select_balanced(paths: list[Path], label: str, target: int, seed: int) -> list[Path]:
    # Deterministic without numpy: sort, then stride through a seeded shuffle from random.Random.
    import random

    ordered = sorted(paths)
    if target >= len(ordered):
        return ordered

    rng = random.Random(f"{seed}:{label}")
    picked = ordered[:]
    rng.shuffle(picked)
    picked = picked[:target]
    return sorted(picked)


def _collect_mlaad_audio(src_root: Path, languages: Iterable[str]) -> dict[str, list[Path]]:
    buckets = {"bonafide": [], "spoof": []}
    for language in languages:
        original_dir = src_root / "original" / language
        fake_dir = src_root / "fake" / language
        if original_dir.exists():
            buckets["bonafide"].extend(
                path for path in original_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"}
            )
        if fake_dir.exists():
            buckets["spoof"].extend(
                path for path in fake_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"}
            )
    return buckets


def materialize_mlaad_subset(
    src_root: Path,
    out_dir: Path,
    languages: list[str],
    max_per_class: int,
    seed: int,
    force: bool,
) -> None:
    if force and out_dir.exists():
        shutil.rmtree(out_dir)

    protocol_path = out_dir / "protocol.txt"
    summary_path = out_dir / "summary.json"
    if protocol_path.exists() and summary_path.exists() and not force:
        print(f"MLAAD materialized subset already present at {out_dir}; skipping rebuild.")
        return

    buckets = _collect_mlaad_audio(src_root, languages)
    target = min(len(buckets["bonafide"]), len(buckets["spoof"]), max_per_class)
    selected = {
        "bonafide": _select_balanced(buckets["bonafide"], "bonafide", target, seed),
        "spoof": _select_balanced(buckets["spoof"], "spoof", target, seed),
    }

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    summary_rows: list[dict[str, str]] = []
    counters = Counter()

    for label in ("bonafide", "spoof"):
        short = "b" if label == "bonafide" else "s"
        for path in selected[label]:
            counters[label] += 1
            rel = path.relative_to(src_root)
            link_name = f"{short}_{counters[label]:05d}_{_sanitize_stem('__'.join(rel.parts))}{path.suffix.lower()}"
            link_path = audio_dir / link_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(path)
            rows.append(f"mlaad {link_name} {label}")
            summary_rows.append(
                {
                    "linked_name": link_name,
                    "label": label,
                    "source_path": str(path),
                    "language": rel.parts[1] if len(rel.parts) > 1 else "unknown",
                }
            )

    protocol_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "count": len(rows),
                "labels": {key: len(value) for key, value in selected.items()},
                "languages": languages,
                "rows": summary_rows,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Materialized MLAAD subset at {out_dir}")


def restore_in_the_wild(storage_root: Path, force: bool) -> None:
    dataset_root = _data_root(storage_root) / "raw" / "In-The-Wild"
    zip_path = dataset_root / "release_in_the_wild.zip"
    extract_root = dataset_root / "release_in_the_wild"
    expected_count = 31779

    dataset_root.mkdir(parents=True, exist_ok=True)

    if force and extract_root.exists():
        shutil.rmtree(extract_root)

    if force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        legacy_root = _legacy_user_tmp_root() / "in_the_wild"
        legacy_zip = legacy_root / "release_in_the_wild.zip"
        if legacy_zip.exists():
            _ensure_parent(zip_path)
            shutil.copy2(legacy_zip, zip_path)
            print(f"Copied legacy In-The-Wild archive into {zip_path}.")
        else:
            _run(
                [
                    "wget",
                    "-c",
                    "-O",
                    str(zip_path),
                    "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip",
                ]
            )
    else:
        print(f"In-The-Wild archive already present at {zip_path}; skipping download.")

    current_count = _count_audio_files(extract_root) if extract_root.exists() else 0
    if current_count != expected_count:
        legacy_extract = _legacy_user_tmp_root() / "in_the_wild" / "release_in_the_wild"
        if legacy_extract.exists():
            print(f"Copying legacy In-The-Wild audio into {extract_root} ...")
            if extract_root.exists():
                shutil.rmtree(extract_root)
            shutil.copytree(legacy_extract, extract_root, symlinks=True)
        else:
            print(f"Extracting In-The-Wild into {dataset_root} ...")
            with ZipFile(zip_path) as archive:
                archive.extractall(dataset_root)
    else:
        print(f"In-The-Wild audio already extracted at {extract_root}; skipping extraction.")

    build_in_the_wild_protocol(extract_root, _data_root(storage_root) / "protocols" / "in_the_wild")


def build_in_the_wild_protocol(extract_root: Path, protocol_root: Path) -> None:
    meta_path = extract_root / "meta.csv"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata file: {meta_path}")

    protocol_root.mkdir(parents=True, exist_ok=True)
    protocol_path = protocol_root / "protocol.txt"
    summary_path = protocol_root / "summary.json"

    rows: list[str] = []
    labels = Counter()
    missing: list[str] = []

    with meta_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row["file"].strip()
            label = row["label"].strip()
            audio_path = extract_root / file_name
            if not audio_path.exists():
                missing.append(file_name)
                continue
            rows.append(f"itw {file_name} {label}")
            labels[label] += 1

    protocol_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "count": len(rows),
                "labels": dict(labels),
                "missing_count": len(missing),
                "missing": missing[:20],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Built In-The-Wild protocol at {protocol_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", "layout", "asvspoof", "mlaad_tiny", "in_the_wild"],
        default="all",
        help="Which dataset assets to set up.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=(
            "Root that should physically hold the managed data layout. "
            "Default keeps datasets inside the repo. Point this somewhere else "
            "to store data externally and replace repo data paths with symlinks."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repo root whose data paths should be configured.",
    )
    parser.add_argument(
        "--asvspoof-source-root",
        type=Path,
        default=None,
        help="Optional durable source directory containing ASVspoof assets to copy into storage.",
    )
    parser.add_argument(
        "--migrate-existing",
        action="store_true",
        help=(
            "Copy existing repo-managed raw/results/protocols content into the new "
            "storage root before switching layouts. Useful when moving data later."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload/reextract even if target files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_root = ensure_repo_storage_layout(
        args.repo_root,
        args.storage_root,
        migrate_existing=args.migrate_existing,
    )
    asvspoof_source = args.asvspoof_source_root or _detect_asvspoof_source(args.repo_root)

    if args.dataset in {"all", "asvspoof"}:
        restore_asvspoof(storage_root, asvspoof_source, args.repo_root, args.force)
    if args.dataset in {"all", "layout"}:
        if storage_root.resolve() == args.repo_root.resolve():
            print(f"Ensured in-repo data layout for {args.repo_root}")
        else:
            print(f"Ensured external storage layout at {storage_root} for {args.repo_root}")

    if args.dataset in {"all", "mlaad_tiny"}:
        snapshot_download = _ensure_hf_runtime()
        restore_mlaad_tiny(snapshot_download, storage_root, args.force)
    if args.dataset in {"all", "in_the_wild"}:
        restore_in_the_wild(storage_root, args.force)

    print("Restore complete.")


if __name__ == "__main__":
    main()
