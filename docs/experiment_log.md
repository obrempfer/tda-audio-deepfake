# Experiment Log

This file is the running record for benchmark setup, implementation changes that affect results, and the best-known outcomes so far.

## Environment and Dataset

- Dataset: ASVspoof 2019 Logical Access (LA)
- Canonical local dataset root: `data/raw/ASVspoof2019_LA/`
- Derived balanced-train protocol:
  `data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt`
- Balanced protocol construction:
  all `2580` bonafide train utterances + `2580` spoof train utterances sampled with seed `42`

## Implementation Notes That Affect Results

- `2026-04-08`: fixed `--max-samples` subsampling so smoke runs are stratified instead of taking the first protocol rows, which could collapse to a single class.
- `2026-04-08`: feature-cache keys now include audio, feature, topology, vectorization, and `max_points` settings so different feature sweeps do not silently reuse stale vectors.

## Results

| Date | Protocol | Point-cloud features | Max points | Vectorization | Classifier | Accuracy | AUC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-08 | train CV, stratified subset (`n=500`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.898 ± 0.004 | 0.704 ± 0.054 | Accuracy inflated by class imbalance; useful mainly as a smoke benchmark |
| 2026-04-08 | train CV, stratified subset (`n=500`) | MFCC + delta + delta-delta | 300 | statistics | Logistic | 0.894 ± 0.021 | 0.815 ± 0.066 | Best AUC among the initial toy runs |
| 2026-04-08 | balanced train CV (`n=5160`) | MFCC + delta + delta-delta | 300 | statistics | Logistic | 0.673 ± 0.016 | 0.727 ± 0.014 | Balanced protocol removes majority-class accuracy inflation |
| 2026-04-08 | balanced train CV (`n=5160`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.734 ± 0.020 | 0.809 ± 0.018 | Current best full balanced-CV baseline |
| 2026-04-08 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta + F0 + F0 slope + spectral flux | 300 | statistics | SVM | 0.705 ± 0.020 | 0.778 ± 0.028 | Richer frame-level features are viable but substantially slower because `pyin` dominates extraction time |

## Current Read

- TDA-derived features contain real signal for this task.
- On the current benchmark, TDA looks more like a moderately informative detector than a standalone production-ready solution.
- Balanced training is necessary for meaningful interpretation; raw accuracy on the original imbalanced split is misleading.
- Added pitch/flux features did not obviously beat the strongest MFCC-only balanced baseline, though the current comparison is not apples-to-apples because the richer run was bounded to `n=1000`.

## Next Runs

1. Balanced train CV with `F0 + F0 slope + spectral flux`
2. Balanced train CV with `formants`
3. Balanced train CV with higher `max_points` (for example `500`)
4. Train/dev evaluation using the strongest balanced-train configuration
