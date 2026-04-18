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
- `2026-04-15`: added a parallel cubical-PH branch: audio can now be represented as a normalized mel spectrogram grid, passed through cubical persistence, then vectorized with the same downstream statistics / persistence-image / landscape options used by the Vietoris-Rips branch.
- `2026-04-15`: added a weighted kNN flag/clique branch on MFCC point clouds, using Gudhi simplex-tree expansion from a kNN graph so the same point cloud can be compared under a different complex family.
- `2026-04-15`: added a Morse-Smale branch on mel spectrogram grids. The branch now supports the exact `topopy.MorseSmaleComplex` path plus a local discrete fallback for environments without `topopy`.
- `2026-04-17`: expanded cubical field controls for spectrogram construction:
  dynamic-range compression mode (`db` / `log1p` / `root` / `none` / `auto`), smoothing axis (`both` / `time` / `frequency`), and frame-energy gating (`energy_gate_percentile`, `energy_gate_fill`).
- `2026-04-17`: added `configs/experiments/cubical_mel_best_field_svm.yaml` to capture the current best cubical setup in one reproducible config.
- `2026-04-17`: added optional homology-block weighting (`vectorization.homology_weights`) and a classifier scaler toggle (`classifier.scale_features`) to support H0/H1 ablation experiments.

## Results

| Date | Protocol | Point-cloud features | Max points | Vectorization | Classifier | Accuracy | AUC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-08 | train CV, stratified subset (`n=500`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.898 ± 0.004 | 0.704 ± 0.054 | Accuracy inflated by class imbalance; useful mainly as a smoke benchmark |
| 2026-04-08 | train CV, stratified subset (`n=500`) | MFCC + delta + delta-delta | 300 | statistics | Logistic | 0.894 ± 0.021 | 0.815 ± 0.066 | Best AUC among the initial toy runs |
| 2026-04-08 | balanced train CV (`n=5160`) | MFCC + delta + delta-delta | 300 | statistics | Logistic | 0.673 ± 0.016 | 0.727 ± 0.014 | Balanced protocol removes majority-class accuracy inflation |
| 2026-04-08 | balanced train CV (`n=5160`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.734 ± 0.020 | 0.809 ± 0.018 | Current best full balanced-CV baseline |
| 2026-04-09 | balanced train CV (`n=5160`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.740 ± 0.012 | 0.825 ± 0.014 | Full balanced run with per-utterance z-score normalization before PH |
| 2026-04-08 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.719 ± 0.026 | 0.800 ± 0.024 | Apples-to-apples comparison point for richer feature tests |
| 2026-04-08 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta + F0 + F0 slope + spectral flux | 300 | statistics | SVM | 0.705 ± 0.020 | 0.778 ± 0.028 | Richer frame-level features are viable but substantially slower because `pyin` dominates extraction time |
| 2026-04-08 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.737 ± 0.028 | 0.824 ± 0.035 | Per-utterance z-score normalization before PH |
| 2026-04-08 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.731 ± 0.024 | 0.807 ± 0.037 | Per-utterance z-score normalization + PCA to 16 dims before PH |
| 2026-04-08 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta | 300 | statistics | SVM | 0.732 ± 0.035 | 0.803 ± 0.028 | Per-utterance z-score normalization + JL random projection to 16 dims before PH |
| 2026-04-09 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta | 300 | persistence_image | SVM | 0.728 ± 0.027 | 0.796 ± 0.027 | Worse than normalized statistics on the matched benchmark |
| 2026-04-09 | balanced train CV, bounded subset (`n=1000`) | MFCC + delta + delta-delta | 300 | landscape | SVM | 0.746 ± 0.012 | 0.838 ± 0.025 | Best matched `n=1000` result so far |
| 2026-04-15 | train CV, stratified subset (`n=100`) | mel spectrogram (`64 x <=256`) | n/a | landscape | SVM | 0.900 ± 0.000 | 0.611 ± 0.205 | Cubical-PH smoke benchmark on the original imbalanced train split; useful only to validate the new branch end to end |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`) | n/a | landscape | SVM | 0.797 ± 0.030 | 0.877 ± 0.026 | Matched cubical baseline for tuning comparisons |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`) | n/a | statistics | SVM | 0.796 ± 0.016 | 0.859 ± 0.024 | Cubical vectorization ablation; worse than landscape |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`, Gaussian smoothing `sigma=1.0`) | n/a | landscape | SVM | 0.766 ± 0.025 | 0.831 ± 0.027 | Light smoothing hurt performance on the matched benchmark |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`96 x <=384`) | n/a | landscape | SVM | 0.783 ± 0.030 | 0.866 ± 0.033 | Denser grid did not beat the `64 x 256` cubical baseline |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`, sublevel filtration) | n/a | landscape | SVM | 0.832 ± 0.012 | 0.907 ± 0.015 | Strongest bounded cubical tweak so far; also reduced EER to `0.169 ± 0.019` |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`80 x <=256`) | n/a | landscape | SVM | 0.804 ± 0.027 | 0.872 ± 0.027 | Nearby mel-band sweep did not beat the baseline |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=320`) | n/a | landscape | SVM | 0.798 ± 0.021 | 0.873 ± 0.028 | Nearby frame-cap sweep did not beat the baseline |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`) | n/a | Morse-Smale-inspired signature (fallback approximation) | SVM | 0.789 ± 0.028 | 0.855 ± 0.033 | Local gradient-flow basin fallback; EER `0.225 ± 0.030` |
| 2026-04-15 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`) | n/a | Morse-SmaleComplex (`topopy`) | SVM | 0.834 ± 0.018 | 0.898 ± 0.016 | Exact topopy Morse-Smale path; EER `0.182 ± 0.025`, close to cubical sublevel |
| 2026-04-15 | balanced train CV (`n=5160`) | mel spectrogram (`64 x <=256`) | n/a | landscape | SVM | 0.839 ± 0.012 | 0.910 ± 0.011 | First full apples-to-apples cubical benchmark; materially stronger than the full normalized VR statistics baseline |
| 2026-04-15 | balanced train CV, bounded subset (`n=500`) | MFCC + delta + delta-delta | 300 | landscape | SVM | 0.714 ± 0.061 | 0.793 ± 0.060 | Weighted kNN flag/clique complex on the normalized MFCC point cloud (`k=15`, union graph) |
| 2026-04-15 | balanced train CV (`n=5160`) | MFCC + delta + delta-delta | 300 | landscape | SVM | 0.762 ± 0.019 | 0.845 ± 0.018 | Full weighted kNN flag/clique run (`k=15`, union graph); better than bounded smoke, still weaker than cubical |
| 2026-04-16 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`, sublevel, Gaussian `sigma=0.5`) | n/a | landscape | SVM (`C=4`) | 0.857 ± 0.007 | 0.933 ± 0.009 | Targeted cubical tune improved over earlier sublevel baseline; EER `0.134 ± 0.010` |
| 2026-04-16 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`) | n/a | Morse-Smale (`topopy`) | SVM (`C=2`, `k=12`) | 0.819 ± 0.021 | 0.896 ± 0.019 | Best refined Morse run in this sweep; EER `0.174 ± 0.024` |
| 2026-04-16 | balanced train CV, bounded subset (`n=1000`) | cubical (best at the time) + Morse (best at the time), score fusion | n/a | fused decision scores | SVM heads + linear alpha fuse | 0.867 | 0.945 | Best fusion at `alpha_cubical=0.50`; EER `0.125` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram field sweep from best cubical baseline | n/a | landscape | SVM | 0.881 ± 0.020 | 0.956 ± 0.008 | Field engineering (especially energy gate) materially improved cubical-only; best in this sweep EER `0.119 ± 0.019` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | mel spectrogram (`64 x <=256`, sublevel, `db`, gate 10%, no grid normalization) | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | 0.894 ± 0.010 | 0.963 ± 0.008 | Current best cubical-only CV result; EER `0.095 ± 0.004` |
| 2026-04-17 | train→dev held-out eval (`train n=1000`, `dev n=5000 balanced subset`) | best cubical field config | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | 0.900 | 0.959 | Held-out check preserves sub-10 behavior; EER `0.099` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | best cubical field config, homology ablation | n/a | landscape | SVM | varies | varies | H0 only: `AUC 0.732, EER 0.318`; H1 only: `AUC 0.942, EER 0.137`; H0+H1: `AUC 0.963, EER 0.095` |

## Current Read

- TDA-derived features contain strong signal for this task, and the cubical branch now performs at a competitive level on the current benchmark setup.
- Raw accuracy on imbalanced train splits remains misleading; balanced protocols and held-out evaluation are required for credible comparison.
- Field construction quality is now the dominant lever for cubical PH performance. In particular:
  - `db` compression outperformed `log1p`, `root`, and `none`.
  - Frame-energy gating (`~10-20%`) was the largest single gain.
  - Disabling grid normalization helped once gating/compression were tuned.
  - Slightly denser landscape vectorization (`layers=7`, `bins=120`) pushed CV below 10% EER.
- Best cubical-only bounded CV result so far: `AUC 0.963`, `EER 0.095` (`n=1000`, balanced train CV).
- Held-out train→dev check (`train n=1000`, `dev n=5000 balanced subset`) remained strong: `AUC 0.959`, `EER 0.099`.
- Homology ablation indicates H1 carries most of the discriminative power, but H0+H1 together remain better than H1-only on the current best config.
- Nonzero H0/H1 reweighting has little effect when `StandardScaler` is enabled (expected, because block scaling is normalized away). Disabling scaling made weighting active but degraded performance in this pipeline.

## Next Runs

1. Run full held-out dev evaluation (all dev items, not the balanced subset) with `cubical_mel_best_field_svm.yaml`.
2. Run held-out eval (`ASVspoof2019.LA.cm.eval.trl.txt`) with the same frozen best config.
3. Repeat held-out train→dev with 2-3 alternative train seeds to quantify variance for the `n=1000` training cap.
4. If more performance is needed, tune calibration/thresholding on held-out dev scores rather than broad upstream hyperparameter sweeps.
5. Keep Morse/VR/kNN as ablation and interpretability branches, but treat cubical best-field as the primary detector branch.
