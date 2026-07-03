"""Load and preprocess GFP mutant Excel data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


def load_raw(xlsx_path: str, logger) -> pd.DataFrame:
    logger.info(f"Reading {xlsx_path}")
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    required = {"aaMutations", "GFP type", "Brightness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[["aaMutations", "GFP type", "Brightness"]].copy()
    df.dropna(how="all", inplace=True)
    df["aaMutations"] = df["aaMutations"].astype(str).str.strip().str.replace(" ", "", regex=False)
    df["GFP type"] = df["GFP type"].astype(str).str.strip()
    df["Brightness"] = pd.to_numeric(df["Brightness"], errors="coerce")
    before = len(df)
    df.dropna(subset=["Brightness"], inplace=True)
    logger.info(f"Dropped {before - len(df)} rows with non-numeric Brightness")
    logger.info(f"Raw rows: {len(df)}")
    return df


def deduplicate(df: pd.DataFrame, logger) -> pd.DataFrame:
    key = ["GFP type", "aaMutations"]
    grp = df.groupby(key)["Brightness"].agg(
        Brightness_mean="mean",
        Brightness_std="std",
        replicate_count="count",
    ).reset_index()
    grp["Brightness_std"] = grp["Brightness_std"].fillna(0.0)
    before = len(df)
    logger.info(f"Deduplicated {before} → {len(grp)} unique (GFP type, aaMutations) combos")
    return grp


def compute_statistics(df: pd.DataFrame, cfg: dict, out_dir: Path, logger):
    eps = cfg["data"]["eps"]
    lines = []
    lines.append(f"Total samples: {len(df)}")
    lines.append(f"GFP types: {df['GFP type'].nunique()}")
    for t, cnt in df["GFP type"].value_counts().items():
        lines.append(f"  {t}: {cnt}")
    b = df["Brightness_mean"]
    lines.append(f"Brightness min={b.min():.4f} max={b.max():.4f} mean={b.mean():.4f} median={b.median():.4f}")
    wt_rows = df[df["aaMutations"] == "WT"]
    lines.append(f"WT rows: {len(wt_rows)}")
    for _, row in wt_rows.iterrows():
        lines.append(f"  {row['GFP type']} WT Brightness={row['Brightness_mean']:.4f}")
    mc = df["aaMutations"].apply(lambda x: 0 if x == "WT" else len(x.split(":")))
    lines.append(f"Single-point mutations: {(mc == 1).sum()}")
    lines.append(f"Combination mutations (>=2): {(mc >= 2).sum()}")
    lines.append(f"WT entries: {(mc == 0).sum()}")
    lines.append("Mutation count distribution:")
    for k, v in mc.value_counts().sort_index().items():
        lines.append(f"  count={k}: {v}")
    txt = "\n".join(lines)
    (out_dir / "data_summary.txt").write_text(txt)
    logger.info("Data summary:\n" + txt)


def main():
    args = parse_args_config("Load and preprocess GFP data")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("data", log_dir)
    out_dir = get_output_dir(cfg, "outputs", "processed")

    with timer(logger, "data preprocessing"):
        df_raw = load_raw(cfg["paths"]["data_xlsx"], logger)
        df = deduplicate(df_raw, logger)
        compute_statistics(df, cfg, out_dir, logger)
        df.to_csv(out_dir / "gfp_raw_dedup.csv", index=False)
    logger.info(f"Saved deduplicated data → {out_dir}/gfp_raw_dedup.csv")


if __name__ == "__main__":
    main()
