from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

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
    df = df.sort_values(param_col).reset_index(drop=True)
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


def load_temporal_sample() -> pd.DataFrame:
    raw_csv = ROOT / "data" / "processed" / "stats19_2020_2024_full.csv"
    print(f"Loading full 2020-2024 data from {raw_csv}", flush=True)
    df = pd.read_csv(raw_csv)

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
    print(df["collision_year"].value_counts().sort_index(), flush=True)
    return df


def main() -> None:
    config_path = ROOT / "configs" / "artifact_taxonomy.yaml"
    outdir = ROOT / "results" / "sensitivity_analysis_temporal_sampled10k"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_temporal_sample()
    build_taxonomy(list(df.drop(columns=["target", "collision_year"]).columns), config_path)

    n_path = outdir / "sensitivity_n_concepts_temporal.csv"
    if n_path.exists():
        n_df = pd.read_csv(n_path)
    else:
        n_df = pd.DataFrame(columns=["n_concepts", "f1", "roc_auc", "pr_auc", "artifact_score"])

    done = set(pd.to_numeric(n_df.get("n_concepts", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    n_values = [4, 6, 8, 12, 16, 20, 24, 32, 48]
    missing = [n for n in n_values if n not in done]
    print(f"Completed n_concepts: {sorted(done)}", flush=True)
    print(f"Missing n_concepts: {missing}", flush=True)

    for n in missing:
        print(f"  n_concepts = {n}", flush=True)
        rows, _ = run_raw_setting(
            df=df,
            split_name="temporal",
            seed=42,
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
        n_df = pd.concat([
            n_df,
            pd.DataFrame([{
                "n_concepts": n,
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "artifact_score": row["artifact_score"],
            }]),
        ], ignore_index=True)
        n_df = n_df.drop_duplicates(subset=["n_concepts"], keep="last").sort_values("n_concepts")
        n_df.to_csv(n_path, index=False)

    eta_df = pd.read_csv(outdir / "sensitivity_artifact_eta_temporal.csv").sort_values("eta")
    k_df = pd.read_csv(outdir / "sensitivity_local_density_k_temporal.csv").sort_values("k")
    n_df = pd.read_csv(n_path).sort_values("n_concepts")

    random_dir = ROOT / "results" / "sensitivity_analysis"
    random_eta = pd.read_csv(random_dir / "sensitivity_artifact_eta.csv") if (random_dir / "sensitivity_artifact_eta.csv").exists() else None
    random_k = pd.read_csv(random_dir / "sensitivity_local_density_k.csv") if (random_dir / "sensitivity_local_density_k.csv").exists() else None
    random_n = pd.read_csv(random_dir / "sensitivity_n_concepts.csv") if (random_dir / "sensitivity_n_concepts.csv").exists() else None

    summary = []
    for df_part, col, label in [
        (eta_df, "eta", "偏差惩罚强度η"),
        (k_df, "k", "局部密度k值"),
        (n_df, "n_concepts", "概念数量"),
    ]:
        best = df_part.loc[df_part["f1"].idxmax()]
        left, right = stable_interval(df_part, col)
        summary.append(
            f"时序划分最优{label}: {best[col]}, F1={best['f1']:.4f}, "
            f"ROC-AUC={best['roc_auc']:.4f}, PR-AUC={best['pr_auc']:.4f}, "
            f"Artifact={best['artifact_score']:.4f}"
        )
        summary.append(f"{label}稳定区间: {left} 到 {right} (F1相对最优波动<1%)")

    temporal_best = max(
        eta_df.loc[eta_df["f1"].idxmax()].to_dict(),
        k_df.loc[k_df["f1"].idxmax()].to_dict(),
        n_df.loc[n_df["f1"].idxmax()].to_dict(),
        key=lambda r: r["f1"],
    )
    summary.append("\n与最开始随机划分15k抽样版本对比:")
    if random_eta is not None and random_k is not None and random_n is not None:
        random_best = max(
            random_eta.loc[random_eta["f1"].idxmax()].to_dict(),
            random_k.loc[random_k["f1"].idxmax()].to_dict(),
            random_n.loc[random_n["f1"].idxmax()].to_dict(),
            key=lambda r: r["f1"],
        )
        summary.append(
            f"随机划分最佳F1={random_best['f1']:.4f}, Artifact={random_best['artifact_score']:.4f}; "
            f"时序划分最佳F1={temporal_best['f1']:.4f}, Artifact={temporal_best['artifact_score']:.4f}; "
            f"F1变化={temporal_best['f1'] - random_best['f1']:+.4f}, "
            f"Artifact变化={temporal_best['artifact_score'] - random_best['artifact_score']:+.4f}"
        )
    else:
        summary.append("未找到最开始随机划分CSV，无法自动计算对比。")

    recommended = eta_df.loc[eta_df["eta"].eq(3.5)].iloc[0]
    summary.append("\n推荐设置(η=3.5, k=10, n=24的单因素基线行以η扫描为代表):")
    summary.append(
        f"F1={recommended['f1']:.4f}, ROC-AUC={recommended['roc_auc']:.4f}, "
        f"PR-AUC={recommended['pr_auc']:.4f}, Artifact={recommended['artifact_score']:.4f}"
    )

    summary_path = outdir / "sensitivity_summary_temporal.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print("\nTemporal sensitivity resume completed!", flush=True)
    print(f"Results saved to {outdir}", flush=True)
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
