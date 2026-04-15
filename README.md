# TDA for Audio Deepfake Detection

Explainable audio deepfake detection using persistent homology on physically motivated speech embeddings, graph complexes, and spectrogram grids.

## Overview

This project applies Topological Data Analysis (TDA) — specifically Vietoris-Rips persistent homology — to detect synthetic speech. Rather than using TDA as a black-box feature extractor, the approach constructs point clouds in a feature space where each dimension corresponds to a physically grounded property of human speech production (spectral envelope, continuity dynamics, pitch, voice quality).

See [`docs/`](docs/) for the full technical proposal.

## Current Status

This repository is an implementation-in-progress research project.

Working today:
- Package layout with installable `src/` project metadata.
- ASVspoof 2019 LA manifest parsing and audio loading.
- MFCC, delta MFCC, and optional voice-quality feature extraction.
- Point-cloud construction with uniform subsampling for tractable PH.
- Mel-spectrogram grid construction for cubical persistent homology.
- Vietoris-Rips persistent homology via Ripser, weighted kNN flag/clique persistence via Gudhi, and cubical persistent homology via Gudhi.
- Fixed-length vectorization via summary statistics, persistence images, or persistence landscapes.
- SVM / logistic regression training, cross-validation, train/eval mode, model save/load, and feature caching.
- Dataset-gated tests for ASVspoof plus synthetic end-to-end smoke tests.

Not complete yet:
- Benchmark results are pending a full ASVspoof 2019 LA run; the dataset is not included in this repository.
- The ablation analyzer exists as a module, but the CLI `--ablation` flag is not wired into the train/eval pipeline yet.
- The current README reports implementation status, not a validated detection result.

## Project Structure

```
tda-audio-deepfake/
├── data/
│   ├── raw/            # ASVspoof 2019 LA dataset (download separately)
│   ├── features/       # Extracted feature matrices (.npy)
│   └── results/        # Classification outputs, ablation reports
├── docs/               # Proposal and reference documents
├── notebooks/          # Exploratory analysis and visualization
├── scripts/            # Setup verification
├── src/
│   ├── tda_deepfake/   # Main package
│   │   ├── features/       # Audio feature extraction (librosa, parselmouth)
│   │   ├── topology/       # PH computation and vectorization (Ripser, giotto-tda)
│   │   ├── classification/ # SVM / logistic regression classifier
│   │   ├── ablation/       # Dimensional ablation for explainability
│   │   └── utils/          # Audio I/O and shared utilities
│   └── scripts/        # CLI entry points
└── tests/              # Unit tests
```

## Setup

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate tda-audio-deepfake
```

**Pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Verify setup:**
```bash
python scripts/verify_setup.py
```

## Dataset

This project uses the [ASVspoof 2019 Logical Access (LA) partition](https://www.asvspoof.org/index2019.html). Download and extract it to `data/raw/ASVspoof2019_LA/`.

The expected layout is:

```text
data/raw/ASVspoof2019_LA/
├── ASVspoof2019.LA.cm.train.trn.txt
├── ASVspoof2019.LA.cm.dev.trl.txt
├── ASVspoof2019_LA_train/flac/
└── ASVspoof2019_LA_dev/flac/
```

## Pipeline

```
VR branch:
audio file → feature extraction → point cloud → Vietoris-Rips PH → vectorization → SVM/logistic regression → label

Graph branch:
audio file → feature extraction → point cloud → kNN graph → flag/clique PH → vectorization → SVM/logistic regression → label

Cubical branch:
audio file → mel spectrogram grid → cubical PH → vectorization → SVM/logistic regression → label
```

1. **Feature extraction** (`tda_deepfake.features`): Compute 39-dim MFCC embeddings (static + Δ + Δ²) per sliding window using librosa. Optional: F0, jitter/shimmer, formants via parselmouth.
2. **Point cloud construction** (`tda_deepfake.features`): Assemble per-window feature vectors into a trajectory point cloud.
3. **Mel spectrogram construction** (`tda_deepfake.features`): Build a normalized mel-spectrogram grid for cubical persistent homology.
4. **Persistent homology** (`tda_deepfake.topology`): Compute H₀ and H₁ via Vietoris-Rips filtration on point clouds, weighted kNN flag/clique complexes on point clouds, or cubical filtration on spectrogram grids.
5. **Vectorization** (`tda_deepfake.topology`): Convert persistence diagrams to fixed-length feature vectors via summary statistics, persistence images, or persistence landscapes.
6. **Classification** (`tda_deepfake.classification`): SVM or logistic regression on topological feature vectors.
7. **Ablation** (`tda_deepfake.ablation`): Experimental module for removing feature groups and recomputing PH to isolate which physical property drives an anomaly. This is not yet wired into the CLI.

## Running

Quick cross-validation run on one split:

```bash
python -m scripts.run_pipeline \
  --protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
  --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
  --out-dir data/results/run_01 \
  --method statistics \
  --model svm \
  --max-samples 500
```

Train/eval run:

```bash
python -m scripts.run_pipeline \
  --train-protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
  --train-audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
  --eval-protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt \
  --eval-audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac \
  --out-dir data/results/baseline_dev \
  --method persistence_image \
  --n-bins 20 \
  --model svm
```

Cubical PH experiment via config:

```bash
python -m scripts.run_pipeline \
  --protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
  --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
  --out-dir data/results/cubical_mel_landscape \
  --config configs/experiments/cubical_mel_landscape_svm.yaml \
  --max-samples 500
```

kNN flag/clique PH experiment via config:

```bash
python -m scripts.run_pipeline \
  --protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
  --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
  --out-dir data/results/knn_flag_mfcc_norm_landscape \
  --config configs/experiments/knn_flag_mfcc_norm_landscape_svm.yaml \
  --max-samples 500
```

Run tests:

```bash
pytest -q
```

## References

See the technical proposal in `docs/` for full references. Key dependencies:
- [Ripser](https://github.com/scikit-tda/ripser.py) — efficient Vietoris-Rips PH
- [giotto-tda](https://giotto-ai.github.io/gtda-docs/) — sklearn-compatible TDA pipeline
- [ASVspoof 2019](https://www.asvspoof.org/index2019.html) — benchmark dataset
