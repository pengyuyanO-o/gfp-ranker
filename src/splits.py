"""Data splitting strategies."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .utils import load_config, setup_logger, get_output_dir, timer, parse_args_config


def random_split(df: pd.DataFrame, seed: int, train_r: float, val_r: float) -> dict:
    """Stratified split by GFP type × brightness quantile."""
    # Create stratification key
    strat = df["GFP_type"].copy()
    for gfp_type in df["GFP_type"].unique():
        mask = df["GFP_type"] == gfp_type
        q = pd.qcut(df.loc[mask, "Brightness"], q=5, labels=False, duplicates="drop")
        strat.loc[mask] = gfp_type + "_q" + q.astype(str)

    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(df))

    # Split off test
    test_r = 1 - train_r - val_r
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_r, random_state=seed)
    try:
        trainval_idx, test_idx = next(sss.split(idx_all, strat))
    except Exception:
        # fallback: random split
        rng.shuffle(idx_all)
        n_test = max(1, int(len(idx_all) * test_r))
        test_idx = idx_all[:n_test]
        trainval_idx = idx_all[n_test:]

    # Split trainval into train/val
    val_frac = val_r / (train_r + val_r)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    try:
        strat_tv = strat.iloc[trainval_idx]
        train_sub, val_sub = next(sss2.split(trainval_idx, strat_tv))
        train_idx = trainval_idx[train_sub]
        val_idx = trainval_idx[val_sub]
    except Exception:
        rng.shuffle(trainval_idx)
        n_val = max(1, int(len(trainval_idx) * val_frac))
        val_idx = trainval_idx[:n_val]
        train_idx = trainval_idx[n_val:]

    return {
        "train": df.iloc[train_idx]["sample_id"].tolist(),
        "val": df.iloc[val_idx]["sample_id"].tolist(),
        "test": df.iloc[test_idx]["sample_id"].tolist(),
    }


def mutation_count_split(df: pd.DataFrame, seed: int) -> dict:
    """Train on low-order mutations, test on high-order."""
    max_count = df["mutation_count"].max()
    best_k = 2
    for k in range(1, max_count):
        train_mask = df["mutation_count"] <= k
        test_mask = df["mutation_count"] > k
        if train_mask.sum() >= 100 and test_mask.sum() >= 20:
            best_k = k
            if train_mask.sum() / len(df) >= 0.3:
                break

    train_mask = df["mutation_count"] <= best_k
    test_mask = df["mutation_count"] > best_k

    train_ids = df[train_mask]["sample_id"].tolist()
    test_ids = df[test_mask]["sample_id"].tolist()

    # Split train into train/val
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    n_val = max(1, int(len(train_ids) * 0.15))
    val_ids = train_ids[:n_val]
    train_ids = train_ids[n_val:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "split_threshold_k": best_k,
    }


def leave_position_split(df: pd.DataFrame, seed: int, held_out_ratio: float = 0.18) -> dict:
    """Hold out samples containing randomly chosen mutation positions."""
    # Collect all unique positions
    all_positions = set()
    for pos_str in df["mut_positions"]:
        try:
            positions = json.loads(pos_str)
            all_positions.update(positions)
        except Exception:
            pass

    all_positions = sorted(all_positions)
    rng = random.Random(seed)
    n_held = max(1, int(len(all_positions) * held_out_ratio))
    held_positions = set(rng.sample(all_positions, n_held))

    def has_held_pos(pos_str):
        try:
            positions = json.loads(pos_str)
            return bool(set(positions) & held_positions)
        except Exception:
            return False

    test_mask = df["mut_positions"].apply(has_held_pos)
    train_mask = ~test_mask

    train_ids = df[train_mask]["sample_id"].tolist()
    test_ids = df[test_mask]["sample_id"].tolist()

    if not test_ids:
        # fallback: random 15% test
        rng2 = random.Random(seed + 100)
        all_ids = df["sample_id"].tolist()
        rng2.shuffle(all_ids)
        n_test = max(1, int(len(all_ids) * 0.15))
        test_ids = all_ids[:n_test]
        train_ids = all_ids[n_test:]

    rng.shuffle(train_ids)
    n_val = max(1, int(len(train_ids) * 0.15))
    val_ids = train_ids[:n_val]
    train_ids = train_ids[n_val:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "held_out_positions": sorted(held_positions),
    }


def main():
    args = parse_args_config("Generate train/val/test splits")
    cfg = load_config(args.config)
    log_dir = get_output_dir(cfg, "outputs", "logs")
    logger = setup_logger("splits", log_dir)
    split_dir = get_output_dir(cfg, "outputs", "processed", "splits")
    proc_dir = Path(cfg["_project_dir"]) / "outputs" / "processed"

    df = pd.read_csv(proc_dir / "gfp_processed.csv")
    logger.info(f"Loaded {len(df)} samples for splitting")
    seeds = cfg["splits"]["seeds"]

    for seed in seeds:
        with timer(logger, f"random split seed={seed}"):
            sp = random_split(df, seed,
                              cfg["splits"]["random_train_ratio"],
                              cfg["splits"]["random_val_ratio"])
            out = split_dir / f"random_seed{seed}.json"
            out.write_text(json.dumps(sp, indent=2))
            logger.info(f"Random split seed={seed}: train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")

        with timer(logger, f"mutation_count split seed={seed}"):
            sp = mutation_count_split(df, seed)
            out = split_dir / f"mutation_count_seed{seed}.json"
            out.write_text(json.dumps(sp, indent=2))
            logger.info(f"Mut-count split seed={seed}: k={sp.get('split_threshold_k')} train={len(sp['train'])} test={len(sp['test'])}")

        with timer(logger, f"leave_position split seed={seed}"):
            sp = leave_position_split(df, seed, cfg["splits"]["leave_position_ratio"])
            out = split_dir / f"leave_position_seed{seed}.json"
            out.write_text(json.dumps(sp, indent=2))
            logger.info(f"Leave-pos split seed={seed}: train={len(sp['train'])} test={len(sp['test'])}")

    logger.info("All splits saved.")


if __name__ == "__main__":
    main()
