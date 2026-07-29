from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

matplotlib_stub = types.ModuleType("matplotlib")
pyplot_stub = types.ModuleType("matplotlib.pyplot")
for name in ["figure", "close", "savefig", "tight_layout", "ylabel", "xlabel", "xticks", "scatter", "annotate"]:
    setattr(pyplot_stub, name, lambda *args, **kwargs: None)
matplotlib_stub.pyplot = pyplot_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_mcbm.run_experiments import run_raw_setting


def load_same_temporal_sample(sample_per_year: int = 10_000) -> pd.DataFrame:
    raw_csv = ROOT / "data" / "processed" / "stats19_2020_2024_full.csv"
    print(f"Loading shared full data from {raw_csv}", flush=True)
    df = pd.read_csv(raw_csv)
    sampled = []
    for year, year_df in df.groupby("collision_year", sort=True):
        if len(year_df) > sample_per_year:
            part = (
                year_df.groupby("target", group_keys=False)
                .sample(frac=sample_per_year / len(year_df), random_state=42 + int(year))
                .sort_index()
            )
        else:
            part = year_df
        sampled.append(part)
    df = pd.concat(sampled, ignore_index=True)
    print(f"Using identical temporal stratified sample: {df.shape}; max_per_year={sample_per_year}", flush=True)
    print(df["collision_year"].value_counts().sort_index(), flush=True)
    return df


def main() -> None:
    outdir = ROOT / "results" / "new_version_same_data_temporal_sampled10k"
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_same_temporal_sample(sample_per_year=10_000)

    rows, _ = run_raw_setting(
        df=df,
        split_name="temporal",
        seed=42,
        config_path=ROOT / "configs" / "artifact_taxonomy.yaml",
        outdir=outdir,
        n_concepts=8,
        teacher_beta=1.0,
        artifact_eta=4.5,
        opt_n_concepts=8,
        opt_eta=4.5,
        penalty_mode="soft_threshold",
        top_k=None,
        self_distill_alpha=0.5,
        local_density_k=10,
        local_density_beta=1.0,
        pca_components=20,
    )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "new_version_same_data_metrics.csv", index=False)
    print("Saved", outdir / "new_version_same_data_metrics.csv", flush=True)
    print(result[result["method"].isin([
        "LightGBM_raw",
        "LR_raw",
        "RACE_CBM_plain",
        "RACE_MCBM_reporting_aware",
        "RACE_MCBM_optimized",
        "RACE_MCBM_local_density",
    ])][["method", "f1", "roc_auc", "pr_auc", "artifact_score"]], flush=True)


if __name__ == "__main__":
    main()
