# RACE-MCBM

**Reporting-Aware Concept Extraction — Mechanistic Concept Bottleneck Model**

A concept bottleneck model for crash severity (KSI) prediction on UK STATS19 data. Learns `X → C → y` with artifact-penalized concept directions and a geometric local-density signal that replaces the black-box teacher.

## Core idea

STATS19 is police-reported administrative data. Some features carry reporting-process shortcuts (e.g., whether an officer attended the scene is predictive but not actionable). RACE-MCBM penalizes concept directions that load on these artifact features, producing cleaner and more stable concepts.

The signal for concept extraction:

```
signal = zscore(y) + beta * zscore(logit(local_density))
```

where `local_density` is the per-sample KSI rate among k nearest neighbors in PCA space. This gives a continuous, non-linear signal without training any model.

## Install

```bash
# Conda environment (recommended)
conda activate ltx  # or create your own with torch>=2.0

# Install dependencies
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Download data

```bash
# Individual year files for multi-year experiments (~1.5 GB)
bash scripts/download_data.sh 2020-2024
```

Raw CSV files are placed in `data/raw/`. Preprocess them into aligned feature matrices:

```bash
python scripts/preprocess_multi_year.py --years 2020 2021 2022 2023 2024
```

Preprocessed files are written to `data/processed/stats19_multi_year_raw.csv` (368 cols) and `stats19_multi_year_clean.csv` (293 cols, artifact columns removed).

## Run experiments

### Main experiments (RACE + baselines)

```bash
bash scripts/run_all.sh
```

### Teacher ablation

```bash
bash scripts/run_teacher_ablation.sh
```

### Neural CBM baselines

```bash
bash scripts/run_neural_baselines.sh
```

### Reproduce all

```bash
bash scripts/reproduce_all.sh
```

### Key parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--local_density_k` | 30 | k-NN neighbors for geometric signal (0 = disabled) |
| `--local_density_beta` | 2.0 | Weight of local density in signal |
| `--pca_components` | 20 | PCA dims before k-NN (0 = raw features) |
| `--penalty_mode` | soft_threshold | `soft_threshold` or `multiplicative` |
| `--self_distill_alpha` | 0.5 | Self-distillation weight |
| `--n_concepts` | 8 | Number of concepts |
| `--opt_n_concepts` | 16 | Concepts for optimized no-teacher variant |
| `--opt_eta` | 3.0 | Artifact penalty strength for optimized variant |

## Data

STATS19 multi-year data (2020-2024), 147,256 rows, 368 features.

| Year | Rows | Target rate |
|------|------|-------------|
| 2020 | 18,503 | 17.6% |
| 2021 | 51,438 | 19.4% |
| 2022 | 10,167 | 21.6% |
| 2023 | 52,155 | 20.2% |
| 2024 | 14,993 | 21.5% |
| **Total** | **147,256** | **19.8%** |

### Splits

| Split | Train | Val | Test | Description |
|-------|-------|-----|------|-------------|
| Random | 64% (stratified) | 16% | 20% | IID generalization |
| Group | 64% (group-shuffled) | 16% | 20% | OOD on road type × urban/rural × speed |
| Temporal | 2020-2022 | 2023 | 2024 | Cross-year temporal generalization |

## Method

### Concept extractor

`RaceConceptExtractor` in `src/race_mcbm/models.py` performs iterative deflation PLS:

1. Build signal from binary label + optional continuous guidance
2. For each concept k:
   - `w = X^T * signal / n` (covariance direction)
   - Apply artifact penalty: `w = sign(w) * max(0, |w| - eta * artifact * std(w))` (soft threshold) or `w = w / (1 + eta * artifact)` (multiplicative)
   - Orthogonalize against previous concepts (Gram-Schmidt)
   - Normalize
   - Deflate: remove the concept's contribution from X and signal
3. Optional self-distillation: train logistic head on first-pass concepts, use predicted probabilities as signal for a second pass

### Signal sources

| Variant | Signal | Type |
|---------|--------|------|
| `CBM_label_only_plain` | `zscore(y)` only | Label only |
| `RACE_MCBM_optimized` | `zscore(y)` + self-distill | Label + self-distill |
| `RACE_MCBM_local_density` | `zscore(y)` + PCA k-NN local KSI rate + self-distill | Geometric prior |
| `RACE_MCBM_reporting_aware` | `zscore(y)` + LightGBM teacher | Black-box teacher |

### Classification head

Logistic regression on K concepts, with Platt calibration and F1-optimal threshold selection on validation set.

## Results

All results: 5-year STATS19 data (2020-2024), 3 seeds (42/43/44), mean ± std.

### Main experiment: random split

| Method | F1 | ROC-AUC | PR-AUC | Artifact |
|--------|:--:|:-------:|:------:|:--------:|
| LightGBM_raw | 0.4666 ±.0101 | 0.7585 ±.0072 | 0.4473 ±.0221 | - |
| RACE_MCBM_local_density | 0.4585 ±.0120 | 0.7516 ±.0068 | 0.4405 ±.0181 | **0.0519** |
| RACE_MCBM_optimized | 0.4604 ±.0116 | 0.7493 ±.0095 | 0.4357 ±.0168 | **0.0394** |
| RACE_CBM_plain | 0.4587 ±.0128 | 0.7496 ±.0086 | 0.4340 ±.0198 | 0.2066 |
| RACE_MCBM_reporting_aware | 0.4583 ±.0104 | 0.7514 ±.0064 | 0.4394 ±.0183 | 0.1152 |
| SHAP_top48_LR | 0.4583 ±.0098 | 0.7474 ±.0054 | 0.4313 ±.0180 | 0.2003 |
| LR_raw | 0.4602 ±.0137 | 0.7524 ±.0087 | 0.4367 ±.0201 | 0.1569 |
| LR_actionable_only | 0.4600 ±.0107 | 0.7504 ±.0100 | 0.4323 ±.0186 | 0.0000 |
| LR_artifact_only | 0.3577 ±.0056 | 0.6001 ±.0130 | 0.2653 ±.0138 | 1.0000 |

### Main experiment: group split (OOD)

| Method | F1 | ROC-AUC | PR-AUC | Artifact |
|--------|:--:|:-------:|:------:|:--------:|
| LightGBM_raw | 0.4665 ±.0075 | 0.7537 ±.0051 | 0.4490 ±.0246 | - |
| RACE_MCBM_local_density | 0.4584 ±.0154 | 0.7463 ±.0169 | 0.4410 ±.0281 | **0.0514** |
| RACE_CBM_plain | 0.4587 ±.0186 | 0.7448 ±.0182 | 0.4388 ±.0335 | 0.2087 |
| RACE_MCBM_reporting_aware | 0.4578 ±.0156 | 0.7468 ±.0156 | 0.4423 ±.0320 | 0.1094 |
| LR_raw | 0.4586 ±.0185 | 0.7485 ±.0162 | 0.4442 ±.0312 | 0.1569 |
| SHAP_top48_LR | 0.4526 ±.0149 | 0.7426 ±.0172 | 0.4423 ±.0348 | 0.2042 |

### Main experiment: temporal split (hardest generalization)

| Method | F1 | ROC-AUC | PR-AUC | Artifact |
|--------|:--:|:-------:|:------:|:--------:|
| LightGBM_raw | 0.4748 ±.0127 | 0.7455 ±.0054 | 0.4845 ±.0101 | - |
| **RACE_MCBM_local_density** | **0.4740** ±.0103 | 0.7414 ±.0069 | **0.4816** ±.0098 | **0.0526** |
| RACE_MCBM_reporting_aware | 0.4695 ±.0117 | **0.7410** ±.0081 | 0.4784 ±.0103 | 0.1023 |
| SHAP_top48_LR | 0.4700 ±.0125 | 0.7371 ±.0070 | 0.4793 ±.0099 | 0.1997 |
| RACE_CBM_plain | 0.2777 ±.0549 | 0.7141 ±.0124 | 0.3826 ±.0466 | 0.2002 |
| RACE_MCBM_optimized | 0.2984 ±.0258 | 0.7152 ±.0100 | 0.3822 ±.0267 | **0.0407** |
| LR_raw | 0.1418 ±.0601 | 0.6906 ±.0228 | 0.3121 ±.0672 | 0.1574 |
| LR_actionable_only | 0.4343 ±.0174 | 0.7006 ±.0112 | 0.4064 ±.0151 | 0.0000 |

### Teacher ablation

| Method | Random F1 | Group F1 | Temporal F1 | Artifact |
|--------|:---------:|:--------:|:-----------:|:--------:|
| CBM_label_only_plain | 0.4545 | 0.4562 | 0.3055 | 0.2064 |
| RACE_label_only_artifact | 0.4563 | 0.4593 | 0.4624 | **0.0462** |
| RACE_LinearProbe | 0.4573 | 0.4596 | 0.4681 | 0.1096 |
| RACE_LightGBM | 0.4579 | 0.4551 | 0.4650 | 0.1115 |

Note: Artifact penalty alone (label_only_artifact vs plain) recovers temporal F1 from 0.31→0.46.

### Neural CBM baselines (3 seeds × 3 splits, all complete)

| Method | AUC | F1 | Artifact | 概念多样性 |
|--------|:---:|:---:|:--------:|:----------:|
| MLP_CBM (latent bottleneck) | 0.7478 ±.006 | 0.4639 ±.018 | 0.0675 ±.023 | ❌ 1 组 |
| MLP_CBM_RACE (pseudo-supervised) | 0.7478 ±.007 | 0.4653 ±.015 | 0.0523 ±.005 | ⚠️ 3 组 |
| SAE_MCBM | 0.7500 ±.007 | 0.4652 ±.016 | 0.1571 ±.001 | ❌ 1 组 |
| FT_Transformer_CBM | 0.5025 ±.041 | 0.3390 ±.018 | 0.1541 ±.004 | ❌ 未分析 |

### Key findings

1. **RACE_MCBM_local_density matches LightGBM on temporal split** (F1 0.474 vs 0.475) while keeping artifact score at 0.053 — 75% lower than plain CBM (0.207)
2. **Temporal split reveals artifact leakage**: CBM_plain F1 drops from 0.46 (random) to 0.28 (temporal); LR_raw collapses to 0.14. Artifact penalty is essential for cross-year generalization.
3. **Neural concept bottleneck collapses**: MLP_CBM and SAE_MCBM both collapse to 1 semantic group (all 16 concepts encode the same casualty-type direction). RACE pseudo-supervision partially recovers (3 groups), but only Gram-Schmidt explicit orthogonalization achieves full diversity (6 groups).
4. **Nonlinear concepts don't improve performance**: MLP_CBM AUC 0.748 vs RACE AUC 0.746 — near-identical predictive power, but RACE concepts are deterministic, auditable, and semantically diverse.
5. **FT-Transformer fails on 367 features**: AUC 0.50 (random level). Self-attention on 367 feature tokens with 147k samples requires far larger models or pretraining.
6. **Self-distillation (optimized) achieves lowest artifact** (0.039) but loses temporal generalization — overfitting risk on small data.
7. **Artifact-only features** produce severely degraded performance (F1 0.36, AUC 0.60), confirming that reporting shortcuts alone cannot explain crash severity.

## Directory structure

```text
race_mcbm_code/
  README.md
  requirements.txt
  configs/
    artifact_taxonomy.yaml              # artifact scoring rules
  docs/
    analysis.md                         # detailed interpretability analysis
  data/processed/
    stats19_multi_year_raw.csv          # 147k × 368, preprocessed feature matrix
    stats19_multi_year_clean.csv        # 147k × 293, artifact columns removed
  src/race_mcbm/
    __init__.py
    models.py                           # RaceConceptExtractor + _local_density
    metrics.py                          # metrics, Platt calibration, NCC
    taxonomy.py                         # FeatureTaxonomy
    splits.py                           # random + group split strategies
    temporal.py                         # multi-year temporal split
    neural_baselines.py                 # MLP-CBM, SAE-MCBM, FT-Transformer-CBM
    run_experiments.py                  # main experiment pipeline
    run_teacher_ablation.py             # teacher/no-teacher comparison
    run_neural_baselines.py             # neural CBM baseline pipeline
    merge_runs.py                       # merge per-seed result folders
  scripts/
    download_data.sh                    # download STATS19 CSV from DfT
    preprocess_multi_year.py            # join + feature-engineer raw CSVs
    run_all.sh                          # main experiments
    run_teacher_ablation.sh             # teacher ablation
    run_neural_baselines.sh             # neural CBM baselines
    reproduce_all.sh                    # run all suites
  results_final_pca_ld/                 # main experiment outputs
  results_teacher_ablation_v2/          # teacher ablation outputs
  results_neural_baselines/             # neural CBM baseline outputs
```

## Comparison

### RACE-CBM vs Neural CBM

| | RACE-CBM | MLP-CBM |
|---|---|---|
| Concept mapping | Linear `C = XW` | Nonlinear MLP encoder |
| Learning | Deterministic deflation PLS | Gradient descent (AdamW) |
| Concept discovery | Covariance decomposition | Emergent via bottleneck |
| Interpretability | Weight matrix W directly readable | Post-hoc input-gradient attribution |
| Artifact control | Direct penalty on concept weights | Indirect penalty on first-layer weights |
| Training time | CPU, instant (NumPy) | GPU, ~2 min per combo |
| Determinism | Fully deterministic | Seed-dependent initialization |

### All compared methods

| Category | Method | Description |
|----------|--------|-------------|
| Traditional | Logistic Regression | LR on all 367 features |
| Black-box | LightGBM | Gradient boosting (100 trees) |
| Post-hoc | SHAP top48 LR | Top-48 SHAP features + LR |
| Baseline CBM | CBM plain | Label-only PLS, no artifact penalty |
| **RACE (main)** | RACE_MCBM_local_density | PCA k-NN geometric signal + artifact penalty |
| **RACE (main)** | RACE_MCBM_reporting_aware | LightGBM teacher + artifact penalty |
| **RACE (main)** | RACE_MCBM_optimized | Self-distillation + artifact penalty |
| Neural CBM | MLP_CBM | Latent bottleneck MLP |
| Neural CBM | MLP_CBM_RACE | MLP bottleneck + RACE concept supervision |
| Neural CBM | SAE_MCBM | MLP → SAE on hidden → sparse head |
| Neural CBM | FT-Transformer-CBM | Feature-token Transformer + concept layer |
| Audit | LR_artifact_only | Only artifact features (upper-bound leakage) |
| Audit | LR_actionable_only | Only safety features (lower-bound leakage) |
| Audit | RACE_MCBM_artifact_shuffled | Test-time artifact permutation |

## Detailed Analysis

See **[docs/analysis.md](docs/analysis.md)** for:
- Full concept interpretability comparison (7 methods, 16 concepts each)
- Per-concept top features and semantic labeling
- Concept collapse analysis (MLP_CBM 1 group, SAE_MCBM 1 group, MLP_CBM_RACE 3 groups, RACE 6 groups)
- Quantitative diversity, independence, and artifact distribution tables
- Full neural baselines per-split, per-seed results
- Teacher ablation and temporal generalization analysis
- Recommendations for practice

## Limitations

The experiments use STATS19 2020-2024 data (147k records, 5 years) with a 19.8% KSI rate. Temporal split uses train=2020-2022, val=2023, test=2024. Year 2022 has notably fewer records (10k) due to DfT data availability. FT-Transformer-CBM underperforms due to the large feature count (367 tokens) relative to model capacity — tabular Transformers may benefit from feature selection or pretraining for this dataset size. The DfT download server can be unreliable; retrying with `wget` or a different network may be necessary.
