# TDA for Audio Deepfake Detection

Explainable audio deepfake detection using persistent homology on physically motivated speech embeddings.

## Overview

This project applies Topological Data Analysis (TDA) — specifically Vietoris-Rips persistent homology — to detect synthetic speech. Rather than using TDA as a black-box feature extractor, the approach constructs point clouds in a feature space where each dimension corresponds to a physically grounded property of human speech production (spectral envelope, continuity dynamics, pitch, voice quality). Detected anomalies can be traced back to specific violated physical properties via dimensional ablation.

See [`docs/`](docs/) for the full technical proposal.

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
```

**Verify setup:**
```bash
python scripts/verify_setup.py
```

## Dataset

This project uses the [ASVspoof 2019 Logical Access (LA) partition](https://www.asvspoof.org/index2019.html). Download and extract it to `data/raw/ASVspoof2019_LA/`.

## Pipeline

```
audio file → feature extraction → point cloud → persistent homology → persistence images → SVM → label
                                                                                              ↓ (if flagged)
                                                                                       dimensional ablation → explanation
```

1. **Feature extraction** (`tda_deepfake.features`): Compute 39-dim MFCC embeddings (static + Δ + Δ²) per sliding window using librosa. Optional: F0, jitter/shimmer, formants via parselmouth.
2. **Point cloud construction** (`tda_deepfake.topology`): Assemble per-window feature vectors into a trajectory point cloud.
3. **Persistent homology** (`tda_deepfake.topology`): Compute H₀ and H₁ via Vietoris-Rips filtration using Ripser.
4. **Vectorization** (`tda_deepfake.topology`): Convert persistence diagrams to persistence images via giotto-tda.
5. **Classification** (`tda_deepfake.classification`): SVM or logistic regression on persistence image features.
6. **Ablation** (`tda_deepfake.ablation`): On flagged samples, systematically remove feature groups and recompute PH to isolate which physical property drives the anomaly.

## References

See the technical proposal in `docs/` for full references. Key dependencies:
- [Ripser](https://github.com/scikit-tda/ripser.py) — efficient Vietoris-Rips PH
- [giotto-tda](https://giotto-ai.github.io/gtda-docs/) — sklearn-compatible TDA pipeline
- [ASVspoof 2019](https://www.asvspoof.org/index2019.html) — benchmark dataset
