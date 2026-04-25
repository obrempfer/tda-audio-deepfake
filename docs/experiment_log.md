# Experiment Log

This file is the running record for benchmark setup, implementation changes that affect results, and the best-known outcomes so far.

## Environment and Dataset

- Dataset: ASVspoof 2019 Logical Access (LA)
- Canonical local dataset root: `data/raw/ASVspoof2019_LA/`
- Transfer/eval dataset root: `data/raw/ASVspoof2021_LA/`
- Derived balanced-train protocol:
  `data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt`
- Balanced protocol construction:
  all `2580` bonafide train utterances + `2580` spoof train utterances sampled with seed `42`
- Recent DF transfer smoke tests were run on `bg16` from the official ASVspoof 2021 DF keys plus the `ASVspoof2021_DF_eval_part00.tar.gz` audio archive, materialized into balanced `/tmp` subsets because the lab home filesystem hit quota during full extraction.

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

## Current Read

- TDA-derived features contain strong signal for this task, and the cubical branch now performs at a competitive level on the current benchmark setup.
- Raw accuracy on imbalanced train splits remains misleading; balanced protocols and held-out evaluation are required for credible comparison.
- Field construction quality is now the dominant lever for cubical PH performance. In particular:
  - `db` compression outperformed `log1p`, `root`, and `none`.
  - Frame-energy gating (`~10-20%`) was the largest single gain.
  - Disabling grid normalization helped once gating/compression were tuned.
  - Slightly denser landscape vectorization (`layers=7`, `bins=120`) pushed CV below 10% EER.
- Frequency-band ablations moved the current read from "full mel field is best" to "low mel band carries most of the useful topology." Keeping only the low band improved bounded CV to `AUC 0.974`, `EER 0.075`; dropping the low band caused a large collapse.
- Full-dev held-out evaluation (`train n=1000`, full dev `n=24844`) confirmed the low-band gain: keep-low `AUC 0.966`, `EER 0.090` vs full-field reference `AUC 0.958`, `EER 0.103`.
- Homology ablation indicates H1 carries most of the discriminative power. H0+H1 remains best in CV, but low-band H1-only is close on full dev (`AUC 0.965`, `EER 0.095`).
- Full 2019 LA → 2021 LA transfer is materially weaker than the in-domain 2019 dev checks, but the low-band story survives. On the full 2021 LA eval set, keep-low gave the best AUC among the main transfer configs (`0.8298`), while keep-low H1 gave the best EER (`0.2116`).
- The 2021 LA transfer follow-ups suggest only modest gains from small local retuning: gate12 lifted transfer AUC to `0.8329`, and keep-low H1 with `C=2` reduced transfer EER to `0.2100`.
- The first DF smoke block ran cleanly, so the pipeline now has a verified path onto DF data. On the larger balanced DF `part00` subset (`n=5000`), low-band transfer remained the strongest family, and the follow-up sweep moved the winner to gate-off (`AUC 0.7984`, `EER 0.2658`). The tiny `C=2/4/8` check was effectively flat, so further DF tuning is not a priority.
- The internal 2021 LA train/dev sweep says the low band still matters in-domain, but the exact winner shifts relative to the 2019-centered story: keep-low / gate10 gave the best EER (`0.1096`), gate-off gave the best AUC (`0.9591`), H1-only no longer wins, and H0-only remains much weaker.
- Across all four settings, the broad family conclusion still holds: low-band cubical structure is the most reliable motif, but the exact best gate / homology balance depends on domain. 2019 LA favors low-band strongly, 2021 LA transfer still benefits from low-band/H1 tweaks, 2021 LA in-domain prefers low-band with full H0+H1, and DF transfer prefers low-band with gate-off.
- Best cubical-only bounded CV result so far: low-band cubical field, `AUC 0.974`, `EER 0.075` (`n=1000`, balanced train CV).
- Best held-out train→dev result so far: low-band cubical field, `AUC 0.966`, `EER 0.090` (`train n=1000`, full dev `n=24844`).
- Nonzero H0/H1 reweighting has little effect when `StandardScaler` is enabled (expected, because block scaling is normalized away). Disabling scaling made weighting active but degraded performance in this pipeline.

## Next Runs

1. Promote the internal 2021 LA winner(s) from the current dev sweep to the held-out internal test split, keeping the "research-only internal split" disclaimer explicit.
2. Build the small sample-level explanation mini-demo: one 2019 LA fake, one 2021 LA fake, one bona fide sample, and optionally one failure case scored under the main field variants.
3. Expand DF transfer beyond `part00`, or at least to a larger multi-part balanced slice, so the DF conclusions are not bottlenecked by one archive shard.
4. Repeat the strongest transfer/internal checks across 2-3 train seeds to separate real gains from split variance.
5. Keep `configs/experiments/ablation/cubical_best_band_keep_low.yaml` as the primary transfer branch family, while tracking `gate_off` as the best current in-domain 2021 LA AUC variant and the best bounded DF variant.
