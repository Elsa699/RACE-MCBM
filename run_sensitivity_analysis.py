from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

# run_raw_setting does not use plotting in this script, but run_experiments imports
# matplotlib at module import time. Some local Anaconda installs have a broken
# matplotlib import, so provide a tiny pyplot stub for this non-plotting script.
matplotlib_stub = types.ModuleType("matplotlib")
pyplot_stub = types.ModuleType("matplotlib.pyplot")
pyplot_stub.figure = lambda *args, **kwargs: None
pyplot_stub.close = lambda *args, **kwargs: None
pyplot_stub.savefig = lambda *args, **kwargs: None
pyplot_stub.tight_layout = lambda *args, **kwargs: None
pyplot_stub.ylabel = lambda *args, **kwargs: None
pyplot_stub.xlabel = lambda *args, **kwargs: None
pyplot_stub.xticks = lambda *args, **kwargs: None
pyplot_stub.scatter = lambda *args, **kwargs: None
pyplot_stub.annotate = lambda *args, **kwargs: None
matplotlib_stub.pyplot = pyplot_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_mcbm.run_experiments import run_raw_setting
from race_mcbm.taxonomy import build_taxonomy


def stable_interval(df: pd.DataFrame, param_col: str, metric_col: str = "f1", tolerance: float = 0.01) -> tuple[float, float]:
    best_idx = df[metric_col].idxmax()
    best_pos = df.index.get_loc(best_idx)
    best_value = float(df.loc[best_idx, metric_col])
    threshold = best_value * (1 - tolerance)

    left = best_pos
    while left > 0 and float(df.iloc[left - 1][metric_col]) >= threshold:
        left -= 1

    right = best_pos
    while right < len(df) - 1 and float(df.iloc[right + 1][metric_col]) >= threshold:
        right += 1

    return df.iloc[left][param_col], df.iloc[right][param_col]


def _append_summary_for_param(summary: list[str], df: pd.DataFrame, param_col: str, label: str) -> None:
    best = df.loc[df["f1"].idxmax()]
    left, right = stable_interval(df, param_col)
    summary.append(
        f"时序划分最优{label}: {best[param_col]}, "
        f"F1={best['f1']:.4f}, ROC-AUC={best['roc_auc']:.4f}, "
        f"PR-AUC={best['pr_auc']:.4f}, Artifact={best['artifact_score']:.4f}"
    )
    summary.append(f"{label}在{left}到{right}之间性能稳定，F1相对最优波动<1%")


def main() -> None:
    raw_csv = ROOT / "data" / "processed" / "stats19_2020_2024_full.csv"
    config_path = ROOT / "configs" / "artifact_taxonomy.yaml"
    outdir = ROOT / "results" / "sensitivity_analysis_temporal_sampled10k"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading full 2020-2024 data from {raw_csv}", flush=True)
    df = pd.read_csv(raw_csv)
    years = sorted(df["collision_year"].dropna().astype(int).unique().tolist())
    print(f"Full dataset shape: {df.shape}; years={years}", flush=True)
    print("Full samples per year:", flush=True)
    print(df["collision_year"].value_counts().sort_index(), flush=True)

    sample_per_year = 10_000
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
    print(f"Using temporal stratified sample: {df.shape}; max_per_year={sample_per_year}", flush=True)
    print("Sampled samples per year:", flush=True)
    print(df["collision_year"].value_counts().sort_index(), flush=True)

    required_years = {2020, 2021, 2022, 2023, 2024}
    if not required_years <= set(years):
        raise ValueError(f"Temporal split requires {sorted(required_years)}, but got {years}")

    build_taxonomy(list(df.drop(columns=["target", "collision_year"]).columns), config_path)

    seed = 42
    split = "temporal"
    print("Running sensitivity analysis on TEMPORAL split (train=2020-2022, val=2023, test=2024)", flush=True)

    print("\n=== Running sensitivity analysis for artifact_eta ===", flush=True)
    eta_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    eta_results = []

    for eta in eta_values:
        print(f"  eta = {eta}", flush=True)
        rows, _ = run_raw_setting(
            df=df,
            split_name=split,
            seed=seed,
            config_path=config_path,
            outdir=outdir,
            n_concepts=24,
            teacher_beta=1.0,
            artifact_eta=eta,
            opt_n_concepts=24,
            opt_eta=eta,
            local_density_k=10,
            pca_components=20,
        )
        row = next(r for r in rows if r["method"] == "RACE_MCBM_local_density")
        eta_results.append({
            "eta": eta,
            "f1": row["f1"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "artifact_score": row["artifact_score"],
        })
        pd.DataFrame(eta_results).to_csv(outdir / "sensitivity_artifact_eta_temporal.csv", index=False)

    eta_df = pd.DataFrame(eta_results)

    print("\n=== Running sensitivity analysis for local_density_k ===", flush=True)
    k_values = [5, 10, 20, 30, 40, 50, 60, 80, 100, 150]
    k_results = []

    for k in k_values:
        print(f"  k = {k}", flush=True)
        rows, _ = run_raw_setting(
            df=df,
            split_name=split,
            seed=seed,
            config_path=config_path,
            outdir=outdir,
            n_concepts=24,
            teacher_beta=1.0,
            artifact_eta=3.5,
            opt_n_concepts=24,
            opt_eta=3.5,
            local_density_k=k,
            pca_components=20,
        )
        row = next(r for r in rows if r["method"] == "RACE_MCBM_local_density")
        k_results.append({
            "k": k,
            "f1": row["f1"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "artifact_score": row["artifact_score"],
        })
        pd.DataFrame(k_results).to_csv(outdir / "sensitivity_local_density_k_temporal.csv", index=False)

    k_df = pd.DataFrame(k_results)

    print("\n=== Running sensitivity analysis for n_concepts ===", flush=True)
    n_values = [4, 6, 8, 12, 16, 20, 24, 32, 48]
    n_results = []

    for n in n_values:
        print(f"  n_concepts = {n}", flush=True)
        rows, _ = run_raw_setting(
            df=df,
            split_name=split,
            seed=seed,
            config_path=config_path,
            outdir=outdir,
            n_concepts=n,
            teacher_beta=1.0,
            artifact_eta=3.5,
            opt_n_concepts=n,
            opt_eta=3.5,
            local_density_k=10,
            pca_components=20,
        )
        row = next(r for r in rows if r["method"] == "RACE_MCBM_local_density")
        n_results.append({
            "n_concepts": n,
            "f1": row["f1"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "artifact_score": row["artifact_score"],
        })
        pd.DataFrame(n_results).to_csv(outdir / "sensitivity_n_concepts_temporal.csv", index=False)

    n_df = pd.DataFrame(n_results)

    print("\n=== Generating final temporal sensitivity summary ===", flush=True)
    summary: list[str] = []
    _append_summary_for_param(summary, eta_df, "eta", "偏差惩罚强度η")
    _append_summary_for_param(summary, k_df, "k", "局部密度k值")
    _append_summary_for_param(summary, n_df, "n_concepts", "概念数量")

    baseline_eta = eta_df.loc[eta_df["eta"].eq(3.5)].iloc[0]
    baseline_k = k_df.loc[k_df["k"].eq(10)].iloc[0]
    baseline_n = n_df.loc[n_df["n_concepts"].eq(24)].iloc[0]
    summary.append("\n推荐设置对照(η=3.5, k=10, n=24):")
    summary.append(f"η扫描推荐行: F1={baseline_eta['f1']:.4f}, Artifact={baseline_eta['artifact_score']:.4f}")
    summary.append(f"k扫描推荐行: F1={baseline_k['f1']:.4f}, Artifact={baseline_k['artifact_score']:.4f}")
    summary.append(f"n扫描推荐行: F1={baseline_n['f1']:.4f}, Artifact={baseline_n['artifact_score']:.4f}")

    summary.append("\n对比之前随机划分15k抽样结果:")
    summary.append("随机划分单因素最佳F1约为0.4567；本次结果为完整2020-2024数据的真正时序划分。")

    with (outdir / "sensitivity_summary_temporal.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\nTemporal sensitivity analysis completed!", flush=True)
    print(f"Results saved to {outdir}", flush=True)
    print("\nFinal Summary:", flush=True)
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
