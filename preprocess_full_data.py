from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from preprocess_multi_year import preprocess


def _read_csv_compat(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, low_memory=False, error_bad_lines=False, warn_bad_lines=True)


def _write_year_files(data_dir: Path, split_dir: Path, years: list[int]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    table_files = {
        "collision": data_dir / "dft-road-casualty-statistics-collision-last-5-years.csv",
        "vehicle": data_dir / "dft-road-casualty-statistics-vehicle-last-5-years.csv",
        "casualty": data_dir / "dft-road-casualty-statistics-casualty-last-5-years.csv",
    }

    print("Loading last-5-years raw tables...", flush=True)
    for table, path in table_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")

        df = _read_csv_compat(path)
        year_col = "collision_year" if "collision_year" in df.columns else "accident_year"
        if year_col not in df.columns:
            raise ValueError(f"{path} has no collision_year/accident_year column")

        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        print(f"  {table}: {len(df)} rows; years={sorted(df[year_col].dropna().astype(int).unique().tolist())}", flush=True)

        for year in years:
            part = df[df[year_col] == year].copy()
            if part.empty:
                print(f"    {table} {year}: 0 rows, skipped", flush=True)
                continue
            out = split_dir / f"dft-road-casualty-statistics-{table}-{year}.csv"
            part.to_csv(out, index=False)
            print(f"    {table} {year}: {len(part)} rows -> {out}", flush=True)


def main() -> None:
    data_dir = Path("E:/CloudCampus")
    out_dir = ROOT / "data" / "processed"
    config_path = ROOT / "configs" / "artifact_taxonomy.yaml"
    split_dir = ROOT / "data" / "raw_last5_split"
    years = [2020, 2021, 2022, 2023, 2024]

    _write_year_files(data_dir, split_dir, years)

    print("\nRunning project-compatible multi-year preprocessing...", flush=True)
    raw_path, clean_path = preprocess(split_dir, out_dir, config_path, years)

    full_path = out_dir / "stats19_2020_2024_full.csv"
    full_clean_path = out_dir / "stats19_2020_2024_full_clean.csv"
    shutil.copy2(raw_path, full_path)
    shutil.copy2(clean_path, full_clean_path)

    full = pd.read_csv(full_path, usecols=["collision_year", "target"])
    print(f"\nDone! Full dataset saved to {full_path}", flush=True)
    print(f"Clean dataset saved to {full_clean_path}", flush=True)
    print(f"Final rows: {len(full)}", flush=True)
    print(f"Years included: {sorted(full['collision_year'].unique().tolist())}", flush=True)
    print("Samples per year:", flush=True)
    print(full["collision_year"].value_counts().sort_index(), flush=True)
    print(f"KSI rate: {full['target'].mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
