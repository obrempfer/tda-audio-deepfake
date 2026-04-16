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

## Current Read

- TDA-derived features contain real signal for this task.
- On the current benchmark, TDA looks more like a moderately informative detector than a standalone production-ready solution.
- Balanced training is necessary for meaningful interpretation; raw accuracy on the original imbalanced split is misleading.
- On a matched `n=1000` balanced benchmark, adding `F0 + F0 slope + spectral flux` underperformed MFCC-only (`AUC 0.778` vs `0.800`) while taking substantially longer.
- Per-utterance feature normalization before PH improved the matched `n=1000` MFCC-only benchmark (`AUC 0.824` vs `0.800`), which is the strongest small-benchmark result so far.
- Adding PCA or JL projection after normalization did not improve on normalization alone at `16` dimensions.
- On the matched normalized benchmark, `persistence_image` underperformed `statistics`, while `landscape` improved to `AUC 0.838`, making it the strongest small-benchmark vectorization so far.
- The cubical branch is implemented and runnable, but its first imbalanced smoke benchmark is not informative enough to compare against the VR branch. The next cubical run should use the same balanced protocol used for the VR experiments.
- On the matched full balanced protocol, cubical PH on normalized mel spectrograms reached `AUC 0.910`, which is the strongest result in the repo so far and is a meaningful step above the best full VR benchmark (`AUC 0.825` with normalized MFCC statistics).
- On matched `n=1000` cubical tuning runs, the original `64 x 256` unsmoothed landscape setup remained best (`AUC 0.877`); switching to statistics, adding Gaussian smoothing, or using a denser `96 x 384` grid all underperformed it.
- The first cubical tweak that clearly helped was switching from `superlevel` to `sublevel` filtration on the same `64 x 256` grid; on the matched `n=1000` benchmark it improved from `AUC 0.877` to `0.907` and lowered EER to `0.169`.
- The exact `topopy` Morse-Smale branch materially improved on the local fallback approximation (`AUC 0.898` vs `0.855` on `n=1000`) and is now genuinely close to the strongest bounded cubical result (`AUC 0.907`).
- The first bounded kNN flag/clique benchmark is viable but not promising yet (`AUC 0.793` on `n=500`), so it currently looks weaker than both the best VR and cubical branches.

## Next Runs

1. Finish the detached train/dev evaluation for the strongest cubical setup (`sublevel`)
2. Re-run full balanced cubical CV with `sublevel` to compare directly against the current `0.910` superlevel full run
3. If the Morse-Smale branch remains interesting, sweep `graph_max_neighbors`, `normalization`, and `simplification` before considering a larger run
4. If desired, scale the kNN flag branch further only after a `k` / union-vs-mutual sweep shows a reason to
5. Keep the strongest VR setup as the interpretable comparison branch and document the tradeoff
