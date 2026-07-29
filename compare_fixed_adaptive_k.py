from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_mcbm.models import RaceConceptExtractor, fit_logistic_head, logit, _zscore, eval_head
from race_mcbm.splits import make_random_split
from race_mcbm.taxonomy import build_taxonomy
from race_mcbm.temporal import make_temporal_split


def load_temporal_sample(sample_per_year: int = 10_000) -> pd.DataFrame:
    raw_csv = ROOT / "data" / "processed" / "stats19_2020_2024_full.csv"
    print(f"Loading full 2020-2024 data from {raw_csv}", flush=True)
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
    print(f"Using temporal stratified sample: {df.shape}; max_per_year={sample_per_year}", flush=True)
    print(df["collision_year"].value_counts().sort_index(), flush=True)
    return df


def fixed_local_density(X: np.ndarray, y: np.ndarray, k: int = 30) -> np.ndarray:
    n_neighbors = min(k + 1, len(y))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", n_jobs=1)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    local = y[indices[:, 1:]].mean(axis=1)
    return np.clip(local, 1e-4, 1 - 1e-4)


def adaptive_local_density(X: np.ndarray, y: np.ndarray, k_min: int = 10, k_max: int = 100) -> np.ndarray:
    n_neighbors = min(k_max, len(y) - 1)
    if n_neighbors < 1:
        return np.full(len(y), np.clip(y.mean(), 1e-4, 1 - 1e-4))

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination="auto",
        algorithm="ball_tree",
        n_jobs=1,
    )
    lof.fit(X)
    lof_scores = lof.negative_outlier_factor_
    denom = lof_scores.max() - lof_scores.min()
    if denom < 1e-12:
        lof_norm = np.ones_like(lof_scores)
    else:
        lof_norm = (lof_scores - lof_scores.min()) / denom

    k_low = min(k_min, n_neighbors)
    k_per_sample = k_low + (n_neighbors - k_low) * lof_norm
    k_per_sample = np.clip(k_per_sample.astype(int), 1, n_neighbors)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree", n_jobs=1)
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)

    local = np.zeros(len(X), dtype=float)
    for i, k_i in enumerate(k_per_sample):
        local[i] = y[indices[i, 1:k_i + 1]].mean()
    return np.clip(local, 1e-4, 1 - 1e-4)


class LocalDensityExtractor(RaceConceptExtractor):
    def __init__(self, *, mode: str, fixed_k: int = 30, adaptive_k_min: int = 10,
                 adaptive_k_max: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.fixed_k = int(fixed_k)
        self.adaptive_k_min = int(adaptive_k_min)
        self.adaptive_k_max = int(adaptive_k_max)

    def fit(self, X: pd.DataFrame, y: np.ndarray, teacher_prob, artifact_scores: pd.Series):
        self.feature_names_ = list(X.columns)
        a = artifact_scores.reindex(self.feature_names_).fillna(0.0).to_numpy(dtype=float)
        self.artifact_scores_ = a
        Xz = self.scaler.fit_transform(X.to_numpy(dtype=float))

        safe_mask = a < 0.5
        Xd = Xz[:, safe_mask]
        if self.pca_components > 0 and self.pca_components < Xd.shape[1]:
            pca = PCA(n_components=self.pca_components, random_state=42)
            Xd = pca.fit_transform(Xd)

        if self.mode == "fixed":
            local_p = fixed_local_density(Xd, y, k=self.fixed_k)
        elif self.mode == "adaptive":
            local_p = adaptive_local_density(
                Xd, y,
                k_min=self.adaptive_k_min,
                k_max=self.adaptive_k_max,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        signal = _zscore(y) + self.teacher_beta * _zscore(logit(local_p))
        signal = _zscore(signal)
        self.components_ = self._extract_concepts(Xz, signal, a)

        if self.self_distill_alpha > 0:
            from sklearn.linear_model import LogisticRegression
            C_train = Xz @ self.components_.T
            lr = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000)
            lr.fit(C_train, y)
            self_pred = lr.predict_proba(C_train)[:, 1]
            signal2 = _zscore(y) + self.self_distill_alpha * _zscore(logit(self_pred))
            signal2 = _zscore(signal2)
            self.components_ = self._extract_concepts(Xz, signal2, a)

        return self


def evaluate_variant(name: str, extractor: LocalDensityExtractor, Xtr, Xva, Xte, ytr, yva, yte, tax) -> dict[str, float]:
    print(f"Running {name}...", flush=True)
    extractor.fit(Xtr, ytr, None, tax.artifact_score)
    Ctr, Cva, Cte = extractor.transform(Xtr), extractor.transform(Xva), extractor.transform(Xte)
    head = fit_logistic_head(Ctr, ytr, Cva, yva)
    report = extractor.concept_report(tax)
    artifact = float(report["artifact_score"].mean()) if len(report) else 0.0
    row = eval_head(name, "temporal", 42, head, Cte, yte,
                    ncc_weights=np.asarray(head.clf.coef_).ravel(), artifact_score=artifact)
    return row


def main() -> None:
    outdir = ROOT / "results" / "fixed_vs_adaptive_k_temporal_sampled10k"
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs" / "artifact_taxonomy.yaml"

    df = load_temporal_sample(sample_per_year=10_000)
    tr, va, te = make_temporal_split(df, year_col="collision_year")
    X = df.drop(columns=["target", "collision_year"])
    y = df["target"].to_numpy(dtype=int)
    tax = build_taxonomy(list(X.columns), config_path)

    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    common = dict(
        n_concepts=8,
        teacher_beta=1.0,
        artifact_eta=4.5,
        penalty_mode="soft_threshold",
        top_k=None,
        self_distill_alpha=0.5,
        local_density_k=1,
        pca_components=20,
    )

    rows = []
    rows.append(evaluate_variant(
        "RACE_MCBM_fixed_k30",
        LocalDensityExtractor(mode="fixed", fixed_k=30, **common),
        Xtr, Xva, Xte, ytr, yva, yte, tax,
    ))
    rows.append(evaluate_variant(
        "RACE_MCBM_adaptive_k10_100",
        LocalDensityExtractor(mode="adaptive", adaptive_k_min=10, adaptive_k_max=100, **common),
        Xtr, Xva, Xte, ytr, yva, yte, tax,
    ))

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "fixed_vs_adaptive_k_temporal.csv", index=False)

    fixed = result[result["method"].eq("RACE_MCBM_fixed_k30")].iloc[0]
    adaptive = result[result["method"].eq("RACE_MCBM_adaptive_k10_100")].iloc[0]
    summary = [
        "Fixed-k vs Adaptive-k temporal comparison (10k/year stratified sample)",
        "Common setting: temporal split train=2020-2022, val=2023, test=2024; n_concepts=8, eta=4.5, pca_components=20.",
        f"Fixed k=30: F1={fixed['f1']:.4f}, ROC-AUC={fixed['roc_auc']:.4f}, PR-AUC={fixed['pr_auc']:.4f}, Artifact={fixed['artifact_score']:.4f}",
        f"Adaptive k=10..100: F1={adaptive['f1']:.4f}, ROC-AUC={adaptive['roc_auc']:.4f}, PR-AUC={adaptive['pr_auc']:.4f}, Artifact={adaptive['artifact_score']:.4f}",
        f"Delta adaptive-fixed: F1={adaptive['f1'] - fixed['f1']:+.4f}, ROC-AUC={adaptive['roc_auc'] - fixed['roc_auc']:+.4f}, PR-AUC={adaptive['pr_auc'] - fixed['pr_auc']:+.4f}, Artifact={adaptive['artifact_score'] - fixed['artifact_score']:+.4f}",
    ]
    (outdir / "fixed_vs_adaptive_k_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
