from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "paper_overleaf" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure for Section 4.4 (Hyperparameter sensitivity)
# ---------------------------------------------------------------------------

def plot_sensitivity() -> None:
    """Generate sensitivity panels for eta, k, and n_concepts on temporal split.

    Replaces the "Figure ?? (to be added)" placeholder in Section 4.4.
    """
    base = ROOT / "results" / "sensitivity_analysis_temporal_sampled10k"
    eta = pd.read_csv(base / "sensitivity_artifact_eta_temporal.csv")
    k = pd.read_csv(base / "sensitivity_local_density_k_temporal.csv")
    n = pd.read_csv(base / "sensitivity_n_concepts_temporal.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    panels = [
        (axes[0], eta, "eta", r"Artifact penalty $\eta$"),
        (axes[1], k, "k", r"Local density neighbourhood $k$"),
        (axes[2], n, "n_concepts", "Number of concepts"),
    ]
    for ax, df, xcol, xlabel in panels:
        ax2 = ax.twinx()
        ax.plot(df[xcol], df["f1"], marker="o", color=COLORS["blue"], label="F1")
        ax2.plot(df[xcol], df["artifact_score"], marker="s", color=COLORS["orange"], label="Artifact")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F1", color=COLORS["blue"])
        ax2.set_ylabel("Artifact score", color=COLORS["orange"])
        ax.tick_params(axis="y", labelcolor=COLORS["blue"])
        ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_title("(a) Penalty sensitivity")
    axes[1].set_title("(b) Neighbourhood sensitivity")
    axes[2].set_title("(c) Concept count sensitivity")
    savefig(FIGDIR / "sensitivity_panels.png")


# ---------------------------------------------------------------------------
# Figure for Section 3.4 (Dataset descriptive statistics)
# ---------------------------------------------------------------------------

def plot_dataset_stats() -> None:
    """Descriptive figure to replace the "Figure ?? (to be added)" in Section 3.4.

    Shows (a) records-and-KSI-rate per year and (b) artifact-score distribution.

    Numbers match Table 1 in the paper, which reports the full 2020--2024 DfT
    STATS19 release (640,522 casualty-level records).
    """

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

    # Panel (a): per-year records and KSI rate (full 640k DfT release, matches Table 1).
    years = [2020, 2021, 2022, 2023, 2024]
    counts = [115584, 128209, 135480, 132977, 128272]
    ksi_rate = [0.1865, 0.1944, 0.2032, 0.2080, 0.2155]

    ax = axes[0]
    bars = ax.bar(years, counts, color=COLORS["blue"], width=0.6, label="Records")
    ax.set_xlabel("Year")
    ax.set_ylabel("Records", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax.set_xticks(years)
    ax2 = ax.twinx()
    ax2.plot(years, ksi_rate, marker="o", color=COLORS["orange"], linewidth=2, label="KSI rate")
    ax2.set_ylabel("KSI rate", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.set_ylim(0.10, 0.28)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01, f"{c:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) Records and KSI rate by year")
    ax.grid(True, axis="y", alpha=0.25)

    # Panel (b): artifact score histogram from taxonomy.
    import sys
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from race_mcbm.taxonomy import build_taxonomy
    raw_csv = ROOT / "data" / "processed" / "stats19_multi_year_raw.csv"
    df = pd.read_csv(raw_csv, nrows=1)  # only need column names
    feature_cols = [c for c in df.columns if c not in ("target", "collision_year")]
    tax = build_taxonomy(feature_cols, ROOT / "configs" / "artifact_taxonomy.yaml")
    a = tax.artifact_score.reindex(feature_cols).fillna(0.0).to_numpy(dtype=float)

    ax = axes[1]
    bins = [0.0, 0.05, 0.35, 0.55, 0.9, 1.05]
    labels = ["safety\n(0.0)", "low\n(0.0 to 0.35)", "structural\n(0.35 to 0.55)",
              "unknown\n(0.55 to 0.9)", "reporting\n(0.9 to 1.0)"]
    hist, _ = np.histogram(a, bins=bins)
    ax.bar(range(len(hist)), hist,
           color=[COLORS["green"], COLORS["blue"], COLORS["purple"],
                  COLORS["orange"], COLORS["red"]])
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Feature count")
    ax.set_title("(b) Artifact score distribution over {} features".format(len(a)))
    ax.grid(True, axis="y", alpha=0.25)
    for i, h in enumerate(hist):
        ax.text(i, h + max(hist) * 0.01, str(int(h)), ha="center", va="bottom", fontsize=9)

    savefig(FIGDIR / "dataset_stats.png")


# ---------------------------------------------------------------------------
# Figure for Section 4.2 (Temporal performance drop)
# ---------------------------------------------------------------------------

def plot_temporal_drop() -> None:
    """Bar figure to replace the "Figure ?? (to be added)" in Section 4.2.

    Compares random-split F1 with temporal-split F1 for each method.
    Values match Table 2 in the paper, computed on the full 640k STATS19 release.
    """
    data = [
        ("LightGBM_raw",              0.4680, 0.4752, "#666666"),
        ("RACE_MCBM_local_density",   0.4592, 0.4671, COLORS["blue"]),
        ("RACE_MCBM_reporting_aware", 0.4591, 0.4655, COLORS["green"]),
        ("SHAP_top48_LR",             0.4535, 0.4614, COLORS["purple"]),
        ("RACE_CBM_plain",            0.4587, 0.4650, COLORS["orange"]),
        ("RACE_MCBM_optimized",       0.4582, 0.4677, COLORS["red"]),
        ("LR_raw",                    0.4605, 0.4649, "#8B4513"),
    ]
    methods = [d[0] for d in data]
    rnd = np.array([d[1] for d in data])
    tmp = np.array([d[2] for d in data])
    colours = [d[3] for d in data]

    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(x - width / 2, rnd, width, color=colours, alpha=0.55, label="Random F1")
    ax.bar(x + width / 2, tmp, width, color=colours, alpha=0.95, label="Temporal F1")
    for xi, r, t in zip(x, rnd, tmp):
        drop = r - t
        y_top = max(r, t) + 0.005
        ax.text(xi, y_top, f"$\\Delta$={drop:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.40, 0.50)
    ax.set_title("Temporal generalisation on full 640k STATS19: random F1 vs 2024 temporal F1")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(FIGDIR / "temporal_drop.png")


def main() -> None:
    plot_sensitivity()
    plot_dataset_stats()
    plot_temporal_drop()
    print(f"Figures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
