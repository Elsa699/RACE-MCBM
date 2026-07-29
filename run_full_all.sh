#!/usr/bin/env bash
# Serial run of all 9 (seed, split) combos on the full 640k STATS19 dataset.
# Estimated total time: ~6 hours (40 min per combo on this workstation).
set -uo pipefail
cd "$(dirname "$0")"

# PYTHONPATH — use absolute Windows path because Anaconda python on Windows
# reads PYTHONPATH from the process environment, and bash's `export` does
# propagate, but some setups strip Unix-style paths.  We use both styles.
export PYTHONPATH="E:/daima/M-CBM-main-2/src;$PWD/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

RUNS_DIR="results_full_runs"
FINAL_DIR="results_full_final"
mkdir -p "$RUNS_DIR"

# Reuse the smoke-test result for (42, random) if it exists.
if [ -f "results_full_test/seed42_random/all_metrics.csv" ] && [ ! -f "$RUNS_DIR/seed42_random/all_metrics.csv" ]; then
    mkdir -p "$RUNS_DIR/seed42_random"
    cp -r results_full_test/seed42_random/. "$RUNS_DIR/seed42_random/"
    echo "Reused smoke-test result for seed=42 split=random"
fi

for seed in 42 43 44; do
  for split in random group temporal; do
    out="$RUNS_DIR/seed${seed}_${split}"
    if [ -f "$out/all_metrics.csv" ]; then
        echo "Skipping seed=${seed} split=${split} (already done)"
        continue
    fi
    echo "=== seed=${seed} split=${split} $(date '+%H:%M:%S') ==="
    python -u -W ignore::FutureWarning -m race_mcbm.run_experiments \
      --raw_csv data/processed/stats19_multi_year_raw.csv \
      --clean_csv data/processed/stats19_multi_year_clean.csv \
      --config configs/artifact_taxonomy.yaml \
      --outdir "$out" \
      --seeds "$seed" --splits "$split" \
      --n_concepts 8 --teacher_beta 1.0 --artifact_eta 2.0 \
      --local_density_k 30 --local_density_beta 2.0 --pca_components 20 \
      --opt_n_concepts 16 --opt_eta 3.0 \
      --penalty_mode soft_threshold --self_distill_alpha 0.5
    if [ $? -ne 0 ]; then
        echo "!!! seed=${seed} split=${split} FAILED (exit code $?)"
    fi
  done
done

echo "=== Merging ==="
python -u -m race_mcbm.merge_runs --runs_dir "$RUNS_DIR" --outdir "$FINAL_DIR"
echo "=== Done: $FINAL_DIR ==="
