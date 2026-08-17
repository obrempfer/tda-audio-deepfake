# Experiment Log

This file is the running record for benchmark setup, implementation changes that affect results, and the best-known outcomes so far.

## Environment and Dataset

- Canonical project paths are `data/raw`, `data/protocols`, and
  `data/results`. On the lab machines these are symlinks into durable
  home-backed storage under `~/tda-audio-deepfake-storage/data`.
- Disposable feature, topology-neural, plotting, and package caches live under
  `/scratch/$USER/tda_deepfake_runtime` (or the compatible
  `/tmp/$USER/tda_deepfake_runtime` path). Scratch is never treated as the
  canonical result store.
- ASVspoof 2019 LA balanced train:
  `2580` bona fide + `2580` spoof, sampled with seed `42`
  (`n=5160`). The canonical derived protocol is
  `data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt`.
- ASVspoof 2019 LA dev contains `24844` files. ASVspoof 2021 LA full eval
  contains `181566` files.
- ASVspoof 2021 DF smoke/follow-up runs use the official keys and a balanced
  `part00` materialization capped at `n=5000`.
- MLAAD-tiny runs use fixed English, German, and English+German derived
  protocols.
- Full MLAAD binary runs pair MLAAD synthetic speech with M-AILABS bona fide
  speech and use fixed, label/language-balanced 70/15/15 splits:
  - English: train `84000`, dev `18000`, test `18000`
  - German: train `30800`, dev `6600`, test `6600`
  - English+German: train `114800`, dev `24600`, test `24600`
- In-the-Wild is used as a third-family held-out target with `31779` files
  (`19963` bona fide, `11816` spoof).

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
- `2026-04-18`: added frequency-band masking configs around the best cubical field setup, including low/mid/high keep/drop variants and finer low-band splits. These runs test whether the detector is using broad spectrogram structure or a localized frequency region.
- `2026-04-24`: parallelized per-utterance feature extraction with explicit `train/eval` worker controls and BLAS/OpenMP thread caps, so large eval and transfer runs can use the lab CPU nodes efficiently.
- `2026-04-24`: refactored cache reuse so related cubical configs can share intermediate mel-grid / topology / vectorization stages instead of recomputing the full pipeline for every nearby variant.
- `2026-04-25`: added reproducible experiment tooling for ASVspoof 2021 DF transfer smoke tests and internal ASVspoof 2021 LA train/dev/test splits.
- `2026-04-25`: added a topology-only neural experiment stack with explicit feature blocks, a linear baseline, a flat MLP, a staged robust-core-first MLP, and per-block ablation support. The default block layout is low-band `H1` core, low-band `H0` auxiliary, and full-field `H0+H1` auxiliary.
- `2026-04-27`: added matched topopy Morse-Smale configs and lightweight sweep runners for in-domain holdout checks, bounded parameter sweeps, and cross-dataset transfer probes.
- `2026-04-28`: completed a bounded Morse-Smale keep-low sweep over graph neighborhood size and topopy normalization; `graph_max_neighbors=4`, `normalization=None` was the clear holdout winner.
- `2026-05-01`: added a Takens / time-delay embedding branch over scalar audio signals, with configurable signal construction (`low_wave`, `low_env`, etc.), delay embedding, PH on the induced point cloud, and the same downstream vectorization/classifier path used elsewhere.
- `2026-05-01`: added MLAAD subset materialization and internal-diagnostic tooling, including Morse-Smale feature-subset masking (`counts_entropy`, `basin_fractions`, `merge_sequence`, `extrema_values`) so cubical and Morse can be compared under matched MLAAD ablations.
- `2026-05-02`: added balanced mixed-source protocol generation for ASVspoof 2019 LA + MLAAD English, with equal source contribution and per-source class balance, so source-mixing effects can be tested without quietly changing total train size.
- `2026-05-03`: added managed dataset bootstrap/storage support. Durable
  raw data, protocols, and results can live outside the clone while repo-local
  paths remain stable through symlinks.
- `2026-05-04`: added full MLAAD + M-AILABS binary protocol builders and
  fixed English, German, and English+German 70/15/15 splits.
- `2026-05-05`: changed RBF SVC defaults to `probability=False`,
  `gamma="scale"`, and `cache_size=8000`; ranking metrics now use
  `decision_function` before falling back to probabilities or labels.
- `2026-05-05`: added a non-blocking classifier queue. Pipeline runs can
  persist feature bundles and enqueue atomic `ready -> claimed -> done/failed`
  jobs while separate workers train classifiers and write the normal model and
  evaluation artifacts.
- `2026-05-06`: added full-scale mixed-source, weighted-mixture,
  compact-Morse diagnostic, and score-level fusion launchers for distributed
  multi-machine execution with machine-local caches.

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
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | local rerun reference (`db`, gate 10%, no grid normalization) | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | 0.890 ± 0.013 | 0.946 ± 0.013 | Local-machine rerun anchor for subsequent ablations; EER `0.124 ± 0.020` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | local rerun, homology ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | H0 only: `AUC 0.739, EER 0.318`; H1 only: `AUC 0.924, EER 0.141`; H0+H1: `AUC 0.946, EER 0.124` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | local rerun, energy-gate ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | gate off: `AUC 0.927, EER 0.163`; gate 8: `AUC 0.945, EER 0.126`; gate 12: `AUC 0.947, EER 0.115`; gate 16: `AUC 0.947, EER 0.120` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | local rerun, compression ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | `db` best by large margin in this sweep; `log1p`: `AUC 0.716, EER 0.351`; `root`: `AUC 0.767, EER 0.307`; `none`: `AUC 0.695, EER 0.351` |
| 2026-04-17 | balanced train CV, bounded subset (`n=1000`) | local rerun, grid normalization ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | no normalization: `AUC 0.946, EER 0.124`; minmax: `AUC 0.943, EER 0.118`; zscore: `AUC 0.946, EER 0.108` |
| 2026-04-18 | balanced train CV, bounded subset (`n=1000`) | best cubical field config, frequency-band ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | Low-band structure dominates: keep low `Acc 0.927, AUC 0.974, EER 0.075`; drop low `AUC 0.820, EER 0.265`; keep mid `AUC 0.793, EER 0.287`; keep high `AUC 0.820, EER 0.247` |
| 2026-04-18 | balanced train CV, bounded subset (`n=1000`) | best cubical field config, energy/temporal ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | Gated `db` reference stayed strong (`AUC 0.963, EER 0.095`); raw `db` without gate fell to `AUC 0.941, EER 0.127`; energy weighting and temporal transforms hurt (`AUC 0.750-0.874`) |
| 2026-04-18 | balanced train CV, bounded subset (`n=1000`) | low-band cubical field, homology ablation | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | Low-band H1 carries most of the signal: H0 only `AUC 0.877, EER 0.193`; H1 only `AUC 0.956, EER 0.106`; H0+H1 `AUC 0.974, EER 0.075` |
| 2026-04-18 | balanced train CV, bounded subset (`n=1000`) | low-band cubical field, `C` and gate robustness | n/a | landscape (`layers=7`, `bins=120`) | SVM | varies | varies | Low-band result is stable across nearby classifier settings: `C=2` `AUC 0.972, EER 0.078`; `C=4` `AUC 0.974, EER 0.075`; `C=8` `AUC 0.975, EER 0.078`; gate off remained close at `AUC 0.974, EER 0.077` |
| 2026-04-18/19 | train→dev held-out eval (`train n=1000`, full dev `n=24844`) | best cubical field vs low-band variants | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | n/a | varies | Full-dev check confirms the low-band gain: reference `AUC 0.958, EER 0.103`; keep low `AUC 0.966, EER 0.090`; low + lower mid `AUC 0.965, EER 0.093`; keep-low H1 only `AUC 0.965, EER 0.095` |
| 2026-04-25 | train→dev held-out eval (`train n=1000`, full dev `n=24844`) | low-band cubical field, H2 curiosity pass | n/a | landscape (`layers=7`, `bins=120`) | SVM (`C=4`) | varies | varies | H2 adds nothing on the 2-D cubical setup: H2 only `AUC 0.500, EER 0.500`; H1+H2 exactly matched H1 only `AUC 0.9649, EER 0.0949`; H0+H1+H2 exactly matched H0+H1 `AUC 0.9663, EER 0.0896` |

### Cross-Dataset 2019 LA → 2021 LA Full Eval

These runs use the 2019 balanced-train cubical family and evaluate on the full 2021 LA eval set (`n=181566`).

| Date | Train / Eval | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-23/24 | 2019 LA train (`n=1000`) → 2021 LA full eval | full reference | 0.8268 | 0.2277 | Full-field transfer anchor |
| 2026-04-24 | 2019 LA train (`n=1000`) → 2021 LA full eval | best field | 0.8135 | 0.2417 | Stronger in-domain than transfer; weakest of the final transfer block |
| 2026-04-24 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low | 0.8298 | 0.2203 | Best full-transfer AUC among the main three-way low-band family |
| 2026-04-24 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low H1 | 0.8293 | 0.2116 | Best full-transfer EER among the main three-way low-band family |
| 2026-04-24 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low H0 | 0.7409 | 0.3122 | Large collapse relative to H1 / H0+H1 |

### Cross-Dataset 2019 LA → 2021 LA Follow-Ups

These follow-up runs reuse the same full 2021 LA eval set (`n=181566`) and probe small gate / `C` changes around the low-band winners.

| Date | Train / Eval | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-25 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low + gate off | 0.8219 | 0.2307 | Worse than the default low-band gate |
| 2026-04-25 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low + gate10 | 0.8298 | 0.2203 | Same as the main keep-low transfer reference |
| 2026-04-25 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low + gate12 | 0.8329 | 0.2164 | Best follow-up AUC on 2021 LA transfer |
| 2026-04-25 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low H1 + `C=2` | 0.8316 | 0.2100 | Best follow-up EER on 2021 LA transfer |
| 2026-04-25 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low H1 + `C=4` | 0.8293 | 0.2116 | Same as the main H1 transfer reference |
| 2026-04-25 | 2019 LA train (`n=1000`) → 2021 LA full eval | keep low H1 + `C=8` | 0.8262 | 0.2156 | Mildly worse than `C=2/4` |

### ASVspoof 2021 DF Part 1 Transfer Smoke Tests

These are not official benchmark numbers. They use the 2019-trained saved models unchanged and evaluate on balanced subsets carved from the DF `part00` archive plus the official DF keys.

| Date | Eval subset | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-25 | DF part 1 balanced smoke subset (`n=1000`) | full reference | 0.7647 | 0.2960 | Pipeline ran cleanly on DF data |
| 2026-04-25 | DF part 1 balanced smoke subset (`n=1000`) | keep low | 0.7745 | 0.3060 | Best smoke AUC, but noisy at this scale |
| 2026-04-25 | DF part 1 balanced smoke subset (`n=1000`) | keep low H1 | 0.7657 | 0.2990 | Similar to the full-field anchor |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | full reference | 0.7777 | 0.2844 | Larger exact-transfer comparison subset |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low | 0.7907 | 0.2748 | Strongest DF transfer branch in this first pass |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low H1 | 0.7824 | 0.2798 | Retains signal, but does not beat keep-low here |

### ASVspoof 2021 DF Part 1 Follow-Ups

These follow-ups reuse the balanced DF `part00` comparison subset (`n=5000`) and probe the same gate / `C` neighborhood that mattered for the 2021 LA transfer block.

| Date | Eval subset | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low + gate off | 0.7984 | 0.2658 | Clear DF winner among the gate variants |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low + gate10 | 0.7907 | 0.2748 | Same as the original keep-low DF comparison run |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low + gate12 | 0.7920 | 0.2764 | Slightly better AUC than gate10, but still behind gate-off |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low + gate off + `C=2` | 0.7977 | 0.2664 | Nearly identical to `C=4` |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low + gate off + `C=4` | 0.7984 | 0.2658 | Best DF follow-up result |
| 2026-04-25 | DF part 1 balanced comparison subset (`n=5000`) | keep low + gate off + `C=8` | 0.7973 | 0.2670 | Essentially flat relative to `C=2/4` |

### ASVspoof 2021 LA Internal Split Topology Sweep

This is a research-only internal split, not an official challenge protocol. The split was built from the 2021 LA `trial_metadata.txt` rows with stratification over label + attack family and yielded `train=98783`, `dev=32931`, `test=32926` rows before bounded subsampling. The sweep itself used `max_train_samples=20000` and `max_eval_samples=10000` on the train/dev split.

| Date | Internal 2021 LA run | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | full reference | 0.9440 | 0.1222 | Strong internal 2021 LA baseline |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | keep low | 0.9536 | 0.1096 | Best internal-split EER (tied with gate10) |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | keep low H1 | 0.9442 | 0.1229 | H1-only is not the winner in-domain |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | keep low H0 | 0.8802 | 0.1927 | Large collapse relative to keep-low / H1 |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | drop low | 0.9079 | 0.1744 | Strong evidence that the low band still matters in 2021 LA |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | drop mid | 0.9280 | 0.1436 | Mid band removal hurts, but less than dropping low |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | drop high | 0.9469 | 0.1257 | High band appears least important among the three broad regions |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | gate off | 0.9591 | 0.1117 | Best internal-split AUC |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | gate10 | 0.9536 | 0.1096 | Same as keep-low; best EER |
| 2026-04-25 | internal train/dev split (`train max=20000`, `dev max=10000`) | gate12 | 0.9525 | 0.1108 | Very close to gate10 |

### Multi-Dataset Comparison

This table lines up the strongest directly comparable cubical-family anchors across the four main study settings.

| Setting | Train / Eval | Full-field reference | Best low-band / gate variant | Best H1-only variant | Read |
| --- | --- | --- | --- | --- | --- |
| 2019 LA in-domain held-out | 2019 LA train (`n=1000`) → full dev (`n=24844`) | `AUC 0.9576`, `EER 0.1025` | keep low: `AUC 0.9663`, `EER 0.0896` | keep low H1: `AUC 0.9649`, `EER 0.0949` | Low band dominates in-domain; H1 is strong, but H0+H1 remains best |
| 2021 LA transfer | 2019 LA train (`n=1000`) → 2021 LA full eval (`n=181566`) | `AUC 0.8268`, `EER 0.2277` | keep low + gate12: `AUC 0.8329`, `EER 0.2164` | keep low H1 + `C=2`: `AUC 0.8316`, `EER 0.2100` | Transfer is weaker, but the low-band family still survives the domain shift |
| 2021 LA internal in-domain | internal 2021 LA train/dev (`train max=20000`, `dev max=10000`) | `AUC 0.9440`, `EER 0.1222` | gate off: `AUC 0.9591`, `EER 0.1117`; keep low / gate10 best `EER 0.1096` | keep low H1: `AUC 0.9442`, `EER 0.1229` | Low band still matters, but H1-only is not the winner in-domain |
| 2021 DF bounded transfer | 2019 LA train (`n=1000`) → DF part 1 balanced subset (`n=5000`) | `AUC 0.7777`, `EER 0.2844` | keep low + gate off + `C=4`: `AUC 0.7984`, `EER 0.2658` | keep low H1: `AUC 0.7824`, `EER 0.2798` | DF retains nontrivial signal, but it is clearly weaker than either LA setting |

### Topology-Only Neural Models

These runs keep the input representation purely topological and compare three heads over the same explicit block layout:

- core block: low-band `H1`
- auxiliary block A: low-band `H0`
- auxiliary block B: full-field `H0+H1`

Training used the 2019 balanced-train subset (`n=1000`) with a balanced 2019 dev validation subset (`n=5000`). Evaluation then covered full 2019 dev, full 2021 LA transfer, and the bounded DF transfer subset (`n=5000`).

| Date | Model | 2019 dev eval | 2021 LA transfer | 2021 DF bounded transfer | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-25 | linear topology baseline | `AUC 0.9394`, `EER 0.1289` | `AUC 0.7923`, `EER 0.2573` | `AUC 0.7454`, `EER 0.3024` | Reference non-neural head on the same block layout |
| 2026-04-25 | flat MLP | `AUC 0.9716`, `EER 0.0844` | `AUC 0.8165`, `EER 0.2299` | `AUC 0.7825`, `EER 0.2746` | All blocks visible from the start; better than the linear baseline everywhere |
| 2026-04-25 | staged MLP | `AUC 0.9751`, `EER 0.0784` | `AUC 0.8275`, `EER 0.2240` | `AUC 0.7897`, `EER 0.2730` | Best topology-only neural result so far; trains `H1` core first, then adds `H0`, then full-field auxiliary blocks |

Block-ablation read from the same neural run:

- linear baseline: transfer performance still leans heavily on the full-field auxiliary block. On 2021 LA transfer, removing the full-field block dropped AUC by `0.1715`, while removing the low-band `H1` core changed AUC by only `0.0095`.
- flat MLP: improves all metrics over the linear baseline, but still leans on broader auxiliary structure in transfer. On 2021 LA transfer, removing the full-field auxiliary block dropped AUC by `0.0336`, versus `0.0059` for removing the `H1` core. On DF transfer, the same pattern held (`0.0318` vs `0.0033`).
- staged MLP: shifts the dependency back toward the low-band `H1` core. On 2021 LA transfer, removing the core dropped AUC by `0.0430`, while removing either auxiliary block changed AUC by `0.0087` or less. On DF transfer, removing the core dropped AUC by `0.0272`, again larger than either auxiliary block.

### Morse-Smale Controls and Bounded Tuning

These runs use the exact `topopy` Morse-Smale branch with statistics-vector signatures. The recent control block was designed to answer whether the apparent discrepancy between an older local CV artifact and the new transfer-oriented Morse runs came from the Morse branch itself or from protocol/config drift.

| Date | Setup | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-16 | full balanced-train CV (`n=5160`) | legacy `morse_smale_mel_svm` | 0.9101 | 0.1651 | Prior local artifact; older ungated min-max-normalized full-field Morse setup |
| 2026-04-27 | `2019 train n=1000 -> balanced 2019 dev n=5000` | legacy `morse_smale_mel_svm` | 0.8502 | 0.2350 | Protocol-matched control for the older Morse config |
| 2026-04-27 | `2019 train n=1000 -> balanced 2019 dev n=5000` | matched `keep_low` (`k=8`, `normalization=None`) | 0.8645 | 0.2212 | Better than the legacy config on the same holdout protocol |
| 2026-04-28 | full balanced-train CV (`n=5160`) | matched `keep_low`, `k=4`, `normalization=None` | 0.8695 | 0.2190 | Full-CV check on the bounded-sweep winner |

Bounded keep-low sweep on the stricter holdout protocol (`2019 train n=1000 -> balanced 2019 dev n=5000`):

| Date | Morse keep-low sweep | AUC | EER | Notes |
| --- | --- | --- | --- | --- |
| 2026-04-28 | `k=4`, `normalization=None` | 0.8723 | 0.2094 | Best bounded Morse-Smale holdout result so far |
| 2026-04-28 | `k=4`, `normalization=feature` | 0.7980 | 0.2804 | Strong collapse; feature normalization is harmful at low `k` |
| 2026-04-28 | `k=8`, `normalization=None` | 0.8645 | 0.2212 | Pre-sweep matched keep-low reference |
| 2026-04-28 | `k=8`, `normalization=feature` | 0.8523 | 0.2256 | Mild degradation relative to `None` |
| 2026-04-28 | `k=12`, `normalization=None` | 0.8645 | 0.2212 | Essentially identical to `k=8` |
| 2026-04-28 | `k=12`, `normalization=feature` | 0.8636 | 0.2206 | Nearly unchanged relative to `None` |
| 2026-04-28 | `k=16`, `normalization=None` | 0.8645 | 0.2212 | Essentially identical to `k=8/12` |
| 2026-04-28 | `k=16`, `normalization=feature` | 0.8636 | 0.2206 | Nearly unchanged relative to `None` |

### Morse-Smale Cross-Dataset Transfer

These runs use the matched topopy Morse-Smale family trained on the balanced 2019 LA train subset (`n=1000`).

| Date | Train / Eval | Config | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-27 | 2019 LA train (`n=1000`) → 2021 LA full eval (`n=181566`) | Morse full field | 0.8189 | 0.2216 | Matched full-field Morse transfer anchor |
| 2026-04-27 | 2019 LA train (`n=1000`) → 2021 LA full eval (`n=181566`) | Morse keep low | 0.8381 | 0.2080 | Best Morse transfer result so far; slightly ahead of the best cubical branch on this target |
| 2026-04-27 | 2019 LA train (`n=1000`) → 2021 LA full eval (`n=181566`) | Morse keep low + gate off | 0.8222 | 0.2498 | Gate-off hurts Morse on 2021 LA |
| 2026-04-27 | 2019 LA train (`n=1000`) → bounded 2021 DF (`n=5000`) | Morse full field | 0.7601 | 0.2852 | Best Morse EER on the bounded DF block |
| 2026-04-27 | 2019 LA train (`n=1000`) → bounded 2021 DF (`n=5000`) | Morse keep low | 0.7715 | 0.2884 | Best Morse DF AUC, but not best EER |
| 2026-04-27 | 2019 LA train (`n=1000`) → bounded 2021 DF (`n=5000`) | Morse keep low + gate off | 0.7670 | 0.3050 | Gate-off also hurts Morse on DF |

### Takens Exploratory Branch

This bounded sweep tests whether a time-delay embedding view of the signal recovers useful topology at all. The first pass used the 2019 LA bounded holdout protocol and compared low-band waveform embeddings against low-band energy-envelope embeddings.

| Date | Setup | Signal / embedding | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-05-01 | `2019 train n=1000 -> balanced 2019 dev n=5000` | low wave, `m=5`, `delay=8` | 0.687 | 0.364 | Best Takens result in the first bounded sweep |
| 2026-05-01 | `2019 train n=1000 -> balanced 2019 dev n=5000` | low wave, `m=5`, `delay=4` | 0.688 | 0.367 | Nearly tied with the best EER result |
| 2026-05-01 | `2019 train n=1000 -> balanced 2019 dev n=5000` | low env, best result (`m=3`, `delay=1`) | 0.659 | 0.385 | Envelopes carry signal, but trail waveform embeddings |

Read from the bounded sweep:

- Takens is clearly above weak/noisy behavior, so the branch is real.
- Low-band waveform embeddings are consistently better than low-band energy envelopes.
- Larger embedding dimension (`m=5`) helped, and longer delays (`4-8`) were strongest on the waveform branch.
- The branch is still far weaker than the current cubical and Morse-Smale pipelines, so it remains exploratory rather than a mainline replacement.

### MLAAD-Tiny Transfer and In-Domain Results

These runs use the current best cubical and Morse-Smale anchors:

- cubical: `keep_low_gate12`
- Morse-Smale: `keep_low_k4_norm_none`

Transfer probes are trained on `ASVspoof 2019 LA train n=1000` unless otherwise stated.

| Date | Train / Eval | Branch | AUC | EER | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-05-01 | `2019 LA train n=1000 -> MLAAD-tiny en` full balanced (`n=12140`) | cubical | 0.5467 | 0.4657 | Full English tiny transfer probe; small earlier `500/500` probe was optimistic |
| 2026-05-01 | `2019 LA train n=1000 -> MLAAD-tiny en` full balanced (`n=12140`) | Morse-Smale | 0.6265 | 0.4158 | MS clearly outperforms cubical on English MLAAD transfer |
| 2026-05-01 | `2019 LA train n=1000 -> MLAAD-tiny de` balanced (`n=2000`) | cubical | 0.4240 | 0.5570 | German transfer is much harder for both branches |
| 2026-05-01 | `2019 LA train n=1000 -> MLAAD-tiny de` balanced (`n=2000`) | Morse-Smale | 0.5550 | 0.4660 | MS remains less bad than cubical under the stronger language shift |
| 2026-05-01 | `2019 LA train n=1000 -> MLAAD-tiny en,de` full balanced (`n=14780`) | cubical | 0.5285 | 0.4798 | Mixed-language tiny transfer sits between English-only and German-only |
| 2026-05-01 | `2019 LA train n=1000 -> MLAAD-tiny en,de` full balanced (`n=14780`) | Morse-Smale | 0.6140 | 0.4234 | Same qualitative ordering as English-only |
| 2026-05-01 | `MLAAD-tiny en train+dev = 10320 -> MLAAD-tiny en test = 1820` | cubical | 0.9281 | 0.1566 | Strong in-domain English result |
| 2026-05-01 | `MLAAD-tiny en train+dev = 10320 -> MLAAD-tiny en test = 1820` | Morse-Smale | 0.9864 | 0.0357 | MS is dramatically better than cubical in-domain on English MLAAD |
| 2026-05-01 | `MLAAD-tiny de train+dev = 2244 -> MLAAD-tiny de test = 396` | cubical | 0.7486 | 0.3157 | German in-domain is materially harder for cubical |
| 2026-05-01 | `MLAAD-tiny de train+dev = 2244 -> MLAAD-tiny de test = 396` | Morse-Smale | 0.9154 | 0.1237 | MS retains a large advantage on German as well |
| 2026-05-01 | `MLAAD-tiny en,de train+dev = 12564 -> MLAAD-tiny en,de test = 2216` | cubical | 0.8885 | 0.1927 | Mixed-language MLAAD still hurts cubical relative to English-only |
| 2026-05-01 | `MLAAD-tiny en,de train+dev = 12564 -> MLAAD-tiny en,de test = 2216` | Morse-Smale | 0.9717 | 0.0523 | MS remains decisively stronger on the combined set |
| 2026-05-01 | `MLAAD-tiny en train+dev = 10320 -> ASVspoof2019 LA dev` | cubical | 0.7915 | 0.2647 | Reverse transfer is nontrivial for cubical |
| 2026-05-01 | `MLAAD-tiny en train+dev = 10320 -> ASVspoof2019 LA dev` | Morse-Smale | 0.3725 | 0.5970 | Reverse transfer collapses for MS; asymmetry is directional, not symmetric |

### MLAAD-Tiny Diagnostic Ablations

These runs freeze the current MLAAD anchors and ask what structure each branch is using on MLAAD itself.

- cubical anchor: `keep_low_gate12`
- Morse anchor: `keep_low_k4_norm_none`

#### English (`MLAAD-tiny en`)

| Date | MLAAD English diagnostic | AUC | EER | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-01 | cubical full reference | 0.8962 | 0.1764 | Full-field cubical anchor on English MLAAD |
| 2026-05-01 | cubical keep low gate10 | 0.9277 | 0.1418 | Best cubical EER on English MLAAD, tied with drop-low |
| 2026-05-01 | cubical keep low gate12 | 0.9281 | 0.1566 | Original cubical MLAAD English anchor |
| 2026-05-01 | cubical drop low | 0.9292 | 0.1418 | Dropping the low band does not hurt MLAAD the way it hurts ASVspoof |
| 2026-05-01 | cubical drop mid | 0.9169 | 0.1577 | Mid-band removal is somewhat harmful |
| 2026-05-01 | cubical drop high | 0.9169 | 0.1571 | High-band removal is also mildly harmful |
| 2026-05-01 | cubical gate off | 0.8829 | 0.2027 | Gate-off hurts cubical strongly on MLAAD |
| 2026-05-01 | cubical H0 only | 0.8578 | 0.2225 | H0-only is much weaker |
| 2026-05-01 | cubical H1 only | 0.8500 | 0.2242 | H1-only is also much weaker; the old ASVspoof H1 story does not carry over cleanly |
| 2026-05-01 | Morse full reference | 0.9875 | 0.0341 | Best broad Morse field on English MLAAD |
| 2026-05-01 | Morse keep low gate10 | 0.9864 | 0.0357 | Original Morse MLAAD English anchor |
| 2026-05-01 | Morse keep low gate12 | 0.9833 | 0.0390 | Slightly worse than gate10 |
| 2026-05-01 | Morse drop low | 0.9853 | 0.0412 | Low-band restriction is not essential for MLAAD Morse success |
| 2026-05-01 | Morse drop mid | 0.9874 | 0.0313 | Best compact Morse EER among the English band ablations |
| 2026-05-01 | Morse drop high | 0.9835 | 0.0418 | Still extremely strong |
| 2026-05-01 | Morse gate off | 0.9888 | 0.0297 | Best Morse English MLAAD result in this diagnostic block |
| 2026-05-01 | Morse counts+entropy | 0.9657 | 0.0802 | Strong subset, but well below the full signature |
| 2026-05-01 | Morse basin fractions | 0.9808 | 0.0396 | Strongest compact structural subset |
| 2026-05-01 | Morse merge sequence | 0.8728 | 0.1797 | Much weaker than the full signature |
| 2026-05-01 | Morse extrema values | 0.6106 | 0.4198 | Near-useless on English MLAAD |

#### German (`MLAAD-tiny de`, compact)

| Date | MLAAD German diagnostic | AUC | EER | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-01 | cubical full reference | 0.7691 | 0.3081 | German is much harder for cubical |
| 2026-05-01 | cubical keep low gate10 | 0.7526 | 0.3030 | Similar to the full-field anchor |
| 2026-05-01 | cubical keep low gate12 | 0.7486 | 0.3157 | Slightly worse than gate10 |
| 2026-05-01 | cubical drop low | 0.7958 | 0.2525 | Again, dropping low helps rather than hurts |
| 2026-05-01 | cubical gate off | 0.7014 | 0.3384 | Gate-off is especially harmful |
| 2026-05-01 | Morse full reference | 0.9125 | 0.1263 | Strong German Morse anchor |
| 2026-05-01 | Morse keep low gate10 | 0.9154 | 0.1237 | Best German Morse EER |
| 2026-05-01 | Morse keep low gate12 | 0.9015 | 0.1490 | Worse than gate10 |
| 2026-05-01 | Morse drop low | 0.8811 | 0.1616 | Some degradation, but still much stronger than cubical |
| 2026-05-01 | Morse gate off | 0.9185 | 0.1389 | Still strong, but not clearly better than gate10 |
| 2026-05-01 | Morse counts+entropy | 0.8633 | 0.2096 | Weaker subset |
| 2026-05-01 | Morse basin fractions | 0.9044 | 0.1414 | Strong compact subset here too |

#### English + German (`MLAAD-tiny en,de`, compact)

| Date | MLAAD English+German diagnostic | AUC | EER | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-01 | cubical full reference | 0.8680 | 0.2112 | Mixed-language cubical anchor |
| 2026-05-01 | cubical keep low gate10 | 0.8980 | 0.1742 | Best cubical keep-low variant on the combined set |
| 2026-05-01 | cubical keep low gate12 | 0.8885 | 0.1927 | Original combined-set cubical anchor |
| 2026-05-01 | cubical drop low | 0.9144 | 0.1606 | Best cubical combined-set result |
| 2026-05-01 | cubical gate off | 0.8503 | 0.2283 | Gate-off hurts again |
| 2026-05-01 | Morse full reference | 0.9758 | 0.0505 | Strong combined-set full signature |
| 2026-05-01 | Morse keep low gate10 | 0.9717 | 0.0523 | Original combined-set Morse anchor |
| 2026-05-01 | Morse keep low gate12 | 0.9697 | 0.0578 | Slightly worse than gate10 |
| 2026-05-01 | Morse drop low | 0.9668 | 0.0609 | Mild degradation only |
| 2026-05-01 | Morse gate off | 0.9732 | 0.0469 | Best combined-set Morse EER |
| 2026-05-01 | Morse counts+entropy | 0.9496 | 0.0943 | Useful, but clearly incomplete |
| 2026-05-01 | Morse basin fractions | 0.9647 | 0.0542 | Best compact Morse subset on the combined set |

English sample-level explanation pass:

- The strongest qualitative pattern is a bonafide English MLAAD sample that every cubical variant still scores as fake, while multiple Morse variants flip it correctly to bonafide with high confidence.
- On those same cases, `basin_fractions` and `counts_entropy` often preserve most of the Morse advantage, while `extrema_values` frequently collapses toward noise.
- The sample pass therefore agrees with the aggregate ablation tables: Morse is not winning on raw extrema values, but on broader basin / partition structure that cubical is mostly not exploiting on MLAAD.

### Mixed ASVspoof English + MLAAD English Training

These runs test whether a balanced mixed source set produces a more useful compromise representation.

Training conditions were matched in total size:

- `ASV-only`: balanced `ASVspoof2019 LA train`, `2580` bona fide + `2580` spoof
- `MLAAD-only`: balanced `MLAAD-tiny en train+dev`, `2580` bona fide + `2580` spoof
- `Mixed`: `1290` per label from ASV + `1290` per label from MLAAD, total `5160`

Eval targets were held fixed across all three training conditions:

- `ASVspoof2019 LA dev`
- `ASVspoof2021 LA`
- `MLAAD-tiny en` held-out test

#### Cubical (`keep_low_gate12`)

| Date | Train source | ASV2019 dev | ASV2021 LA | MLAAD English | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-05-02 | ASV-only | `AUC 0.9763`, `EER 0.0737` | `AUC 0.8368`, `EER 0.2111` | `AUC 0.4909`, `EER 0.5044` | Best pure ASV cubical point; collapses on MLAAD |
| 2026-05-02 | MLAAD-only | `AUC 0.7721`, `EER 0.2721` | `AUC 0.7063`, `EER 0.3452` | `AUC 0.9155`, `EER 0.1632` | Strong MLAAD fit, but weak back-transfer to ASV |
| 2026-05-02 | Mixed | `AUC 0.9611`, `EER 0.1000` | `AUC 0.8355`, `EER 0.2251` | `AUC 0.8921`, `EER 0.1802` | A compromise point: rescues MLAAD strongly while only modestly degrading ASV |

#### Morse-Smale (`keep_low_k4_norm_none`)

| Date | Train source | ASV2019 dev | ASV2021 LA | MLAAD English | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-05-02 | ASV-only | `AUC 0.8929`, `EER 0.1903` | `AUC 0.8390`, `EER 0.2041` | `AUC 0.6006`, `EER 0.4286` | Best pure ASV Morse point in this matrix |
| 2026-05-02 | MLAAD-only | `AUC 0.3734`, `EER 0.6013` | `AUC 0.3682`, `EER 0.6331` | `AUC 0.9849`, `EER 0.0429` | Strongest pure MLAAD fit, but catastrophic reverse transfer |
| 2026-05-02 | Mixed | `AUC 0.8659`, `EER 0.2171` | `AUC 0.8423`, `EER 0.2009` | `AUC 0.9757`, `EER 0.0681` | Best balanced Morse point; materially reduces the ASV / MLAAD asymmetry |

### Full MLAAD Binary Scale-Up

These fixed anchor runs test whether the MLAAD-tiny ordering survives when
MLAAD synthetic speech is paired with M-AILABS bona fide speech at full scale.
All rows report the held-out test split after training/monitoring on the fixed
70/15/15 protocol.

| Date | Language split | Branch | Train / test | AUC | EER |
| --- | --- | --- | ---: | ---: | ---: |
| 2026-05-04 | English | cubical keep-low gate12 | `84000 / 18000` | 0.9418 | 0.1268 |
| 2026-05-04 | English | Morse keep-low k4 | `84000 / 18000` | 0.9886 | 0.0303 |
| 2026-05-05 | German | cubical keep-low gate12 | `30800 / 6600` | 0.8421 | 0.2406 |
| 2026-05-05 | German | Morse keep-low k4 | `30800 / 6600` | 0.9468 | 0.1036 |
| 2026-05-05 | English+German | cubical keep-low gate12 | `114800 / 24600` | 0.9012 | 0.1759 |
| 2026-05-05 | English+German | Morse keep-low k4 | `114800 / 24600` | 0.9731 | 0.0533 |

The MLAAD-tiny conclusion survives scale-up: Morse-Smale remains substantially
stronger than cubical in English, German, and the combined language condition.
German remains the hardest slice for both branches.

### Full MLAAD English Morse Diagnostic

The compact diagnostic was rerun against the full English dev split
(`n=18000`) with the full English train split (`n=84000`).

| Date | Morse representation | AUC | EER | Accuracy |
| --- | --- | ---: | ---: | ---: |
| 2026-05-06 | full reference | 0.9959 | 0.0195 | 0.9817 |
| 2026-05-06 | gate off | 0.9902 | 0.0236 | 0.9793 |
| 2026-05-06 | keep low | 0.9888 | 0.0276 | 0.9746 |
| 2026-05-06 | basin fractions only | 0.9856 | 0.0308 | 0.9707 |
| 2026-05-06 | counts + entropy only | 0.9721 | 0.0707 | 0.9322 |

At full scale, the broad full-reference signature wins rather than gate-off.
The structural conclusion still survives: basin fractions alone retain most of
the full signature's performance, while counts and entropy are useful but
incomplete.

### Full-Scale Mixed-Source and In-the-Wild Matrix

The operational source anchors use different train sizes: ASV-only uses the
balanced ASVspoof train set (`n=5160`), MLAAD-only uses full English MLAAD
(`n=84000`), and mixed uses all balanced ASV plus an equal-size balanced
MLAAD sample (`n=10320` total). This is therefore a source-strategy
comparison, not a clean train-size ablation.

Each cell is `AUC / EER`.

| Date | Train source | Branch | ASV2019 dev | ASV2021 LA | MLAAD English test | In-the-Wild |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2026-05-05 | ASV-only | cubical | `0.9763 / 0.0737` | `0.8368 / 0.2111` | `0.4718 / 0.5115` | `0.4468 / 0.5499` |
| 2026-05-05 | ASV-only | Morse | `0.8929 / 0.1903` | `0.8390 / 0.2041` | `0.6132 / 0.4126` | `0.7417 / 0.3264` |
| 2026-05-05 | MLAAD-only | cubical | `0.7245 / 0.3038` | `0.6806 / 0.3616` | `0.9418 / 0.1268` | `0.3067 / 0.6380` |
| 2026-05-05 | MLAAD-only | Morse | `0.3992 / 0.5799` | `0.3761 / 0.6229` | `0.9886 / 0.0303` | `0.4577 / 0.5471` |
| 2026-05-05 | mixed ASV+MLAAD | cubical | `0.9673 / 0.0934` | `0.8400 / 0.2176` | `0.8946 / 0.1824` | `0.4306 / 0.5544` |
| 2026-05-05 | mixed ASV+MLAAD | Morse | `0.8833 / 0.1990` | `0.8480 / 0.2000` | `0.9760 / 0.0688` | `0.7389 / 0.3298` |

The third-family target changes the conclusion. ASV-only Morse is the best
In-the-Wild branch, narrowly ahead of mixed Morse. Both cubical branches fall
below chance ranking on that target. MLAAD-only Morse is highly specialized
and does not generalize back to ASV or onward to In-the-Wild.

### Weighted Mixed-Source Morse

The full-data weighted follow-up varies source composition while holding the
Morse `keep_low_k4_norm_none` representation fixed. Each cell is
`AUC / EER`.

| Date | ASV / MLAAD mix | ASV2021 LA | MLAAD English test | In-the-Wild |
| --- | ---: | ---: | ---: | ---: |
| 2026-05-06 | 75 / 25 | `0.8405 / 0.2036` | `0.9539 / 0.1101` | `0.7363 / 0.3281` |
| 2026-05-06 | 50 / 50 | `0.8417 / 0.2033` | `0.9710 / 0.0812` | `0.7306 / 0.3319` |
| 2026-05-06 | 25 / 75 | `0.8356 / 0.2161` | `0.9791 / 0.0636` | `0.7128 / 0.3444` |

Increasing the MLAAD share improves MLAAD as expected but degrades both
ASV2021 and In-the-Wild. The 75/25 ASV-heavy mix is the best weighted
In-the-Wild point, but it still does not beat ASV-only Morse
(`AUC 0.7417`, `EER 0.3264`).

### Classifier Throughput Control

The classifier control reused the exact same cached cubical feature matrices
(`84000 x 1680` train, `18000 x 1680` eval) and changed only SVC
calibration/cache behavior.

| Date | SVC setting | Fit time | Total fit+eval | AUC | EER |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026-05-05 | probability on, 200 MB cache | 4 h 22 m 50 s | 4 h 46 m 12 s | 0.9396 | 0.1271 |
| 2026-05-05 | probability off, 200 MB cache | 50 m 31 s | 1 h 13 m 36 s | 0.9396 | 0.1271 |
| 2026-05-05 | probability off, 4 GB cache | 39 m 7 s | 1 h 2 m 49 s | 0.9396 | 0.1271 |

Probability calibration was pure overhead for AUC/EER in this pipeline.
These controls motivated the shipped `probability=False`,
`decision_function` scoring, 8 GB cache default, and non-blocking classifier
queue.

## Current Read

- Balanced protocols and held-out evaluation are essential. Accuracy on the
  original imbalanced ASVspoof splits can obscure weak ranking behavior.
- Cubical field construction is the main ASVspoof lever: dB compression,
  light Gaussian smoothing, frame-energy gating, no grid normalization, and a
  low-band mask produced the strongest 2019 results.
- The best cubical-only bounded 2019 CV point is AUC `0.974` / EER
  `0.075`; the best bounded-train to full-dev point is AUC `0.966` / EER
  `0.090`.
- On ASVspoof, `H1` carries most of the useful low-band signal. `H2` is
  chance alone and does not change `H1` or `H0+H1` performance.
- Exact 2019-to-2021 transfer is weaker than in-domain performance, but both
  low-band cubical and Morse remain useful. Under the full-source matrix,
  ASV-only Morse slightly beats ASV-only cubical on ASV2021
  (`0.8390 / 0.2041` vs `0.8368 / 0.2111` AUC/EER).
- Morse-Smale is not the strongest ASV2019 in-domain branch. Its value is
  transfer retention and its much stronger alignment with MLAAD-style data.
- Full MLAAD confirms the tiny-set result. Morse test EER is `0.0303` on
  English, `0.1036` on German, and `0.0533` on English+German, materially
  below the matched cubical anchor in every case.
- The full English MLAAD diagnostic sharpens the morphology story:
  full-reference Morse is best on dev (`AUC 0.9959`, `EER 0.0195`), while
  basin fractions alone retain most of the signal
  (`AUC 0.9856`, `EER 0.0308`).
- Mixed-source Morse is the best balanced ASV2021/MLAAD branch in the
  full-scale matrix: ASV2021 `AUC 0.8480`, `EER 0.2000`; MLAAD English
  `AUC 0.9760`, `EER 0.0688`.
- In-the-Wild remains a hard out-of-family target. ASV-only Morse is best
  (`AUC 0.7417`, `EER 0.3264`), mixed Morse is nearly tied, and all tested
  cubical source strategies rank at or below chance.
- The weighted Morse study shows a smooth specialization tradeoff. More MLAAD
  improves MLAAD but hurts ASV2021 and In-the-Wild; no weighted mixture beats
  ASV-only Morse on In-the-Wild.
- The staged topology-only MLP demonstrates nonlinear headroom and shifts
  feature dependence toward the low-band `H1` core, but it does not beat the
  strongest tuned classical transfer branch.
- Takens low-band waveform topology is above noise but remains secondary
  (best bounded AUC `0.687`, EER `0.364`).
- The classifier queue and uncalibrated SVC default reduce wasted wall time
  without changing AUC/EER. On the full MLAAD control, total classifier time
  fell from 4 h 46 m to 1 h 3 m before the queue-level concurrency gain.
- The evidence supports dataset-dependent topology selection, not a universal
  winner: cubical is ASV-specialized, Morse is MLAAD-aligned and more robust to
  the tested third-family shift, and source mixing trades specialization for
  balance.

## Next Runs

1. Run a size-matched, multi-seed source-mixture study so source composition
   can be separated from the current `5160` / `84000` / `10320` train-size
   difference.
2. Complete score-level cubical/Morse fusion using a calibration split that is
   disjoint from every reported target, then compare it with branch selection.
3. Promote the full-MLAAD diagnostic winner and compact basin-fraction branch
   from dev to the untouched test split.
4. Add source-confound controls for the full MLAAD binary construction,
   especially recording conditions and bona fide corpus identity.
5. Evaluate mixed-source and weighted Morse models on the full MLAAD German
   and English+German test targets.
6. Repeat the topology-only neural comparison across multiple train seeds and
   a larger ASVspoof training budget.
7. Treat `H2` as closed for the current 2-D cubical representation unless a
   future multi-channel field changes the dimensionality of the complex.
